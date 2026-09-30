#!/usr/bin/env python3
"""FedKV-KA class-residual local experiment on Office-Caltech-10.

This file is independent from the CIFAR100 runner. It treats the four domains
as four federated clients, uses full participation every round, freezes the ViT
backbone, and trains only the classification head plus FedKV-KA parameters.

This dedicated variant removes the Shared Residual Bank branch and removes the
local router/local slot mechanism entirely. Each client/domain owns one private
local KV residual per patched attention layer.
"""

import argparse
import csv
import json
import pickle
import random
import time
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


DOMAINS = ("amazon", "caltech", "dslr", "webcam")
CLASSES = (
    "back_pack",
    "bike",
    "calculator",
    "headphones",
    "keyboard",
    "laptop_computer",
    "monitor",
    "mouse",
    "mug",
    "projector",
)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def uses_private_local(method):
    return method == "class_reslocal"


def uses_class_router(method):
    return method in {"fedkv_ka", "class_reslocal"}


class OfficeCaltechPickleDataset(Dataset):
    def __init__(self, root, pickle_file, transform=None, max_samples=None):
        self.root = Path(root)
        self.transform = transform
        with Path(pickle_file).open("rb") as f:
            paths, labels = pickle.load(f)
        self.samples = [(str(path), CLASS_TO_INDEX[str(label)]) for path, label in zip(paths, labels)]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel_path, label = self.samples[index]
        image_path = Path(rel_path)
        if not image_path.is_absolute():
            if image_path.parts and image_path.parts[0] == self.root.name:
                image_path = self.root.parent / image_path
            else:
                image_path = self.root / image_path
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def classes(self):
        return sorted({label for _, label in self.samples})


def build_transform(train, image_size):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_domain_datasets(args):
    root = Path(args.data_root)
    train_sets = {}
    test_sets = {}
    for domain in DOMAINS:
        train_sets[domain] = OfficeCaltechPickleDataset(
            root,
            root / f"{domain}_train.pkl",
            transform=build_transform(train=True, image_size=args.image_size),
            max_samples=args.max_train_samples_per_domain,
        )
        test_sets[domain] = OfficeCaltechPickleDataset(
            root,
            root / f"{domain}_test.pkl",
            transform=build_transform(train=False, image_size=args.image_size),
            max_samples=args.max_test_samples_per_domain,
        )
    return train_sets, test_sets


class FedKVAttention(nn.Module):
    def __init__(self, source_attn, num_classes, memory_tokens, ka_alpha, delta_scale, router_tau):
        super().__init__()
        self.qkv = source_attn.qkv
        self.q_norm = getattr(source_attn, "q_norm", nn.Identity())
        self.k_norm = getattr(source_attn, "k_norm", nn.Identity())
        self.attn_drop = source_attn.attn_drop
        self.proj = source_attn.proj
        self.proj_drop = source_attn.proj_drop
        self.num_heads = source_attn.num_heads
        self.head_dim = source_attn.head_dim
        self.scale = getattr(source_attn, "scale", self.head_dim ** -0.5)
        self.num_classes = num_classes
        self.memory_tokens = memory_tokens
        self.ka_alpha = ka_alpha
        self.delta_scale = delta_scale
        self.router_tau = router_tau
        self.active_classes = None
        self.last_router_logits = None
        self.last_subset_logits = None

        self.base_k = nn.Parameter(torch.empty(self.num_heads, memory_tokens, self.head_dim))
        self.base_v = nn.Parameter(torch.empty(self.num_heads, memory_tokens, self.head_dim))
        self.delta_k = nn.Parameter(torch.empty(num_classes, self.num_heads, memory_tokens, self.head_dim))
        self.delta_v = nn.Parameter(torch.empty(num_classes, self.num_heads, memory_tokens, self.head_dim))
        self.router = nn.Linear(self.num_heads * self.head_dim, num_classes)
        self.reset_fedkv_parameters()

    def reset_fedkv_parameters(self):
        nn.init.normal_(self.base_k, std=0.02)
        nn.init.normal_(self.base_v, std=0.02)
        nn.init.zeros_(self.delta_k)
        nn.init.zeros_(self.delta_v)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def set_active_classes(self, active_classes):
        self.active_classes = active_classes

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q_scaled = q * self.scale
        img_attn = (q_scaled @ k.transpose(-2, -1)).softmax(dim=-1)
        img_attn = self.attn_drop(img_attn)
        img_out = img_attn @ v

        active = self.active_classes
        if active is None:
            active = torch.arange(self.num_classes, device=x.device)
        elif not torch.is_tensor(active):
            active = torch.as_tensor(active, dtype=torch.long, device=x.device)
        else:
            active = active.to(device=x.device, dtype=torch.long)

        router_logits = self.router(x[:, 0])
        subset_logits = router_logits.index_select(dim=1, index=active)
        weights = F.softmax(subset_logits / self.router_tau, dim=1)
        self.last_router_logits = router_logits
        self.last_subset_logits = subset_logits

        delta_k = self.delta_k.index_select(dim=0, index=active)
        delta_v = self.delta_v.index_select(dim=0, index=active)
        mixed_delta_k = torch.einsum("bs,shmd->bhmd", weights, delta_k)
        mixed_delta_v = torch.einsum("bs,shmd->bhmd", weights, delta_v)
        mem_k = self.base_k.unsqueeze(0) + self.delta_scale * mixed_delta_k
        mem_v = self.base_v.unsqueeze(0) + self.delta_scale * mixed_delta_v

        mem_attn = torch.einsum("bhnd,bhmd->bhnm", q_scaled, mem_k).softmax(dim=-1)
        mem_attn = self.attn_drop(mem_attn)
        mem_out = torch.einsum("bhnm,bhmd->bhnd", mem_attn, mem_v)

        out = img_out + self.ka_alpha * mem_out
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj_drop(self.proj(out))


class ClassResidualLocalAttention(FedKVAttention):
    """Class-delta FedKV-KA with a private local memory residual."""

    def __init__(
        self,
        source_attn,
        num_classes,
        memory_tokens,
        ka_alpha,
        delta_scale,
        router_tau,
        local_residual_beta,
    ):
        super().__init__(source_attn, num_classes, memory_tokens, ka_alpha, delta_scale, router_tau)
        self.local_residual_beta = local_residual_beta
        self.local_mem_k = nn.Parameter(torch.empty(self.num_heads, memory_tokens, self.head_dim))
        self.local_mem_v = nn.Parameter(torch.empty(self.num_heads, memory_tokens, self.head_dim))
        nn.init.normal_(self.local_mem_k, std=0.02)
        nn.init.normal_(self.local_mem_v, std=0.02)

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q_scaled = q * self.scale
        img_attn = (q_scaled @ k.transpose(-2, -1)).softmax(dim=-1)
        img_attn = self.attn_drop(img_attn)
        img_out = img_attn @ v

        active = self.active_classes
        if active is None:
            active = torch.arange(self.num_classes, device=x.device)
        elif not torch.is_tensor(active):
            active = torch.as_tensor(active, dtype=torch.long, device=x.device)
        else:
            active = active.to(device=x.device, dtype=torch.long)

        router_input = x[:, 0]
        router_logits = self.router(router_input)
        subset_logits = router_logits.index_select(dim=1, index=active)
        weights = F.softmax(subset_logits / self.router_tau, dim=1)
        self.last_router_logits = router_logits
        self.last_subset_logits = subset_logits

        delta_k = self.delta_k.index_select(dim=0, index=active)
        delta_v = self.delta_v.index_select(dim=0, index=active)
        mixed_delta_k = torch.einsum("bs,shmd->bhmd", weights, delta_k)
        mixed_delta_v = torch.einsum("bs,shmd->bhmd", weights, delta_v)
        mem_k = self.base_k.unsqueeze(0) + self.delta_scale * mixed_delta_k
        mem_v = self.base_v.unsqueeze(0) + self.delta_scale * mixed_delta_v
        mem_attn = torch.einsum("bhnd,bhmd->bhnm", q_scaled, mem_k).softmax(dim=-1)
        mem_attn = self.attn_drop(mem_attn)
        mem_out = torch.einsum("bhnm,bhmd->bhnd", mem_attn, mem_v)

        local_attn = torch.einsum("bhnd,hmd->bhnm", q_scaled, self.local_mem_k).softmax(dim=-1)
        local_attn = self.attn_drop(local_attn)
        local_out = torch.einsum("bhnm,hmd->bhnd", local_attn, self.local_mem_v)

        out = img_out + self.ka_alpha * (mem_out + self.local_residual_beta * local_out)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj_drop(self.proj(out))


class FedKVOfficeCaltechModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fedkv_layers = [module for module in self.model.modules() if isinstance(module, FedKVAttention)]

    def set_active_classes(self, classes):
        for module in self.fedkv_layers:
            module.set_active_classes(classes)

    def forward(self, x, active_classes=None):
        self.set_active_classes(active_classes)
        return self.model(x)


def parse_kv_layers(raw, num_blocks):
    layers = []
    for item in raw.replace(",", " ").split():
        idx = int(item)
        if idx < 0:
            idx = num_blocks + idx
        if idx < 0 or idx >= num_blocks:
            raise ValueError(f"KV layer {item} is outside [0, {num_blocks - 1}]")
        layers.append(idx)
    if not layers:
        raise ValueError("--kv-layers must contain at least one layer")
    return sorted(set(layers))


def strip_state_dict_prefix(state):
    for prefix in ("module.", "model."):
        if state and all(k.startswith(prefix) for k in state.keys()):
            return {k[len(prefix) :]: v for k, v in state.items()}
    return state


def create_base_model(args):
    if args.pretrained_path:
        model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes, img_size=args.image_size)
        checkpoint = torch.load(args.pretrained_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        state = strip_state_dict_prefix(state)
        model_state = model.state_dict()
        compatible = {
            k: v
            for k, v in state.items()
            if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
        }
        model.load_state_dict(compatible, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_path}: {len(compatible)} tensors", flush=True)
        return model
    if args.allow_random_init:
        print("WARNING: using random initialization because --allow-random-init was set.", flush=True)
        return timm.create_model(args.model, pretrained=False, num_classes=args.num_classes, img_size=args.image_size)
    try:
        return timm.create_model(args.model, pretrained=True, num_classes=args.num_classes, img_size=args.image_size)
    except Exception as exc:
        raise RuntimeError("Failed to load timm pretrained model. Provide --pretrained-path if needed.") from exc


def create_model(args):
    base_model = create_base_model(args)
    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.head.parameters():
        param.requires_grad = True
    for idx in parse_kv_layers(args.kv_layers, len(base_model.blocks)):
        base_model.blocks[idx].attn = ClassResidualLocalAttention(
            base_model.blocks[idx].attn,
            num_classes=args.num_classes,
            memory_tokens=args.memory_tokens,
            ka_alpha=args.ka_alpha,
            delta_scale=args.delta_scale,
            router_tau=args.router_tau,
            local_residual_beta=args.local_residual_beta,
        )
    embed_dim = getattr(base_model, "num_features", None) or base_model.head.in_features
    base_model.local_head = nn.Linear(embed_dim, args.num_classes)
    base_model.local_head_gamma = args.local_head_gamma
    base_model.forward = types.MethodType(forward_with_local_head, base_model)
    wrapped = FedKVOfficeCaltechModel(base_model)
    for module in wrapped.fedkv_layers:
        for name, param in module.named_parameters():
            if name.startswith(("base_", "delta_", "router.", "local_mem_")):
                param.requires_grad = True
    if hasattr(wrapped.model, "local_head"):
        for param in wrapped.model.local_head.parameters():
            param.requires_grad = True
    return wrapped


def forward_with_local_head(self, x):
    features = self.forward_features(x)
    pooled = self.forward_head(features, pre_logits=True)
    return self.head(pooled) + self.local_head_gamma * self.local_head(pooled)


def is_local_param(name):
    return ".local_mem_" in name or name.startswith("model.local_head.")


def trainable_state_dict(model, local=None):
    state = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if local is not None and is_local_param(name) != local:
            continue
        state[name] = param.detach().cpu().clone()
    return state


def load_trainable_state_dict(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(params[name].device))


def is_delta_param(name):
    return name.endswith(".delta_k") or name.endswith(".delta_v")


def is_class_row_param(name, tensor, num_classes):
    return (
        tensor.ndim >= 1
        and tensor.shape[0] == num_classes
        and (".router." in name or name.endswith("model.head.weight") or name.endswith("model.head.bias"))
    )


def aggregate_updates(server_state, client_updates, client_classes, client_sizes, num_classes, kv_momentum):
    total = float(sum(client_sizes))
    new_state = {name: value.clone() for name, value in server_state.items()}
    client_class_sets = [set(int(c) for c in classes) for classes in client_classes]
    for name, server_tensor in server_state.items():
        if is_delta_param(name):
            for c in range(num_classes):
                owners = [i for i, classes in enumerate(client_class_sets) if c in classes]
                if not owners:
                    continue
                denom = float(sum(client_sizes[i] for i in owners))
                agg = sum(client_updates[i][name][c] * (client_sizes[i] / denom) for i in owners)
                new_state[name][c] = kv_momentum * server_tensor[c] + (1.0 - kv_momentum) * agg
        elif is_class_row_param(name, server_tensor, num_classes):
            for c in range(num_classes):
                owners = [i for i, classes in enumerate(client_class_sets) if c in classes]
                if not owners:
                    continue
                denom = float(sum(client_sizes[i] for i in owners))
                new_state[name][c] = sum(client_updates[i][name][c] * (client_sizes[i] / denom) for i in owners)
        else:
            new_state[name] = sum(client_updates[i][name] * (client_sizes[i] / total) for i in range(len(client_updates)))
    return new_state


def optimizer_param_groups(model, args):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = args.lr
        if ".local_mem_" in name:
            lr = args.local_memory_lr
        elif name.startswith("model.local_head."):
            lr = args.local_head_lr
        elif ".mem_k" in name or ".mem_v" in name:
            lr = args.memory_lr
        elif ".router." in name:
            lr = args.router_lr
        elif name.endswith("head.weight") or name.endswith("head.bias"):
            lr = args.head_lr
        groups.setdefault(float(lr), []).append(param)
    return [{"params": params, "lr": lr} for lr, params in groups.items()]


def private_regularizer(model, args, device):
    loss = torch.zeros((), device=device)
    if args.local_norm_weight > 0:
        for name, param in model.named_parameters():
            if param.requires_grad and (".local_mem_k" in name or ".local_mem_v" in name):
                loss = loss + args.local_norm_weight * param.pow(2).mean()
    if args.local_head_norm_weight > 0:
        for name, param in model.named_parameters():
            if param.requires_grad and name.startswith("model.local_head."):
                loss = loss + args.local_head_norm_weight * param.pow(2).mean()
    return loss


def prox_regularizer(model, server_state, args, device):
    if args.prox_mu <= 0 or server_state is None:
        return torch.zeros((), device=device)
    loss = torch.zeros((), device=device)
    for name, param in model.named_parameters():
        if not param.requires_grad or is_local_param(name) or name not in server_state:
            continue
        if ".mem_k" in name or ".mem_v" in name or ".router." in name:
            loss = loss + (param - server_state[name].to(device)).pow(2).mean()
    return args.prox_mu * loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_train(model, dataset, active_classes, args, device, server_state=None):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(optimizer_param_groups(model, args), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    active = torch.as_tensor(active_classes, dtype=torch.long, device=device)
    label_to_pos = torch.full((args.num_classes,), -1, dtype=torch.long, device=device)
    label_to_pos[active] = torch.arange(active.numel(), device=device)
    total_loss = 0.0
    total_seen = 0

    for _ in range(args.local_epochs):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, active_classes=active)
            loss = F.cross_entropy(logits, labels)
            if uses_class_router(args.method) and args.router_loss_weight > 0:
                router_targets = label_to_pos[labels]
                if (router_targets < 0).any():
                    raise ValueError("Encountered labels outside the client's active class subset")
                router_losses = []
                for module in model.fedkv_layers:
                    router_losses.append(F.cross_entropy(module.last_subset_logits, router_targets))
                loss = loss + args.router_loss_weight * torch.stack(router_losses).mean()
            if uses_private_local(args.method):
                loss = loss + private_regularizer(model, args, device)
                loss = loss + prox_regularizer(model, server_state, args, device)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(labels.numel())
            total_seen += int(labels.numel())
    return total_loss / max(1, total_seen)


@torch.no_grad()
def evaluate_domain(model, dataset, active_classes, args, device):
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    active = torch.as_tensor(active_classes, dtype=torch.long, device=device)
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images, active_classes=active)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    if total == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "balanced_acc": 0.0, "samples": 0}
    return {
        "acc": correct / total,
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", labels=list(range(args.num_classes)), zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(all_labels, all_preds)),
        "samples": total,
    }


def write_metrics_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, round_idx, model, server_state, metrics, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "round": round_idx,
            "model": model.state_dict(),
            "server_trainable_state": server_state,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not visible. Full Office-Caltech-10 training is expected to be slow on CPU.", flush=True)
    print(f"device={device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    train_sets, test_sets = load_domain_datasets(args)
    client_classes = {domain: train_sets[domain].classes for domain in DOMAINS}
    domain_stats = {
        "domains": list(DOMAINS),
        "train_sizes": {domain: len(train_sets[domain]) for domain in DOMAINS},
        "test_sizes": {domain: len(test_sets[domain]) for domain in DOMAINS},
        "client_classes": client_classes,
    }
    with (output_dir / "domain_stats.json").open("w", encoding="utf-8") as f:
        json.dump(domain_stats, f, indent=2, sort_keys=True)

    model = create_model(args).to(device)
    server_state = trainable_state_dict(model, local=False if uses_private_local(args.method) else None)
    domain_local_states = None
    if uses_private_local(args.method):
        initial_local_state = trainable_state_dict(model, local=True)
        domain_local_states = {domain: deepcopy(initial_local_state) for domain in DOMAINS}
    rows = []
    best_acc = -1.0
    writer = SummaryWriter(log_dir=str(output_dir / "tb")) if args.tensorboard else None
    metrics_jsonl = (output_dir / "metrics.jsonl").open("w", encoding="utf-8")
    try:
        for round_idx in range(1, args.rounds + 1):
            started = time.time()
            client_updates = []
            client_sizes = []
            client_class_rows = []
            losses = []

            for domain in DOMAINS:
                load_trainable_state_dict(model, server_state)
                if domain_local_states is not None:
                    load_trainable_state_dict(model, domain_local_states[domain])
                loss = local_train(model, train_sets[domain], client_classes[domain], args, device, server_state)
                losses.append(loss)
                client_updates.append(trainable_state_dict(model, local=False if uses_private_local(args.method) else None))
                if domain_local_states is not None:
                    domain_local_states[domain] = trainable_state_dict(model, local=True)
                client_sizes.append(len(train_sets[domain]))
                client_class_rows.append(client_classes[domain])

            server_state = aggregate_updates(
                server_state,
                client_updates,
                client_class_rows,
                client_sizes,
                args.num_classes,
                args.kv_momentum,
            )
            load_trainable_state_dict(model, server_state)

            row = {
                "round": round_idx,
                "seed": args.seed,
                "selected_client_ids": list(range(len(DOMAINS))),
                "selected_client_domains": list(DOMAINS),
                "local_loss": float(np.mean(losses)),
                "seconds": time.time() - started,
            }
            if args.eval_every > 0 and (round_idx % args.eval_every == 0 or round_idx == args.rounds):
                domain_metrics = {}
                for domain in DOMAINS:
                    if domain_local_states is not None:
                        load_trainable_state_dict(model, domain_local_states[domain])
                    domain_metrics[domain] = evaluate_domain(model, test_sets[domain], client_classes[domain], args, device)
                accs = np.asarray([domain_metrics[d]["acc"] for d in DOMAINS], dtype=np.float64)
                row.update(
                    {
                        "mean_domain_acc": float(accs.mean()),
                        "worst_domain_acc": float(accs.min()),
                        "mean_domain_macro_f1": float(np.mean([domain_metrics[d]["macro_f1"] for d in DOMAINS])),
                        "mean_domain_balanced_acc": float(np.mean([domain_metrics[d]["balanced_acc"] for d in DOMAINS])),
                    }
                )
                for domain in DOMAINS:
                    row[f"{domain}_acc"] = domain_metrics[domain]["acc"]
                    row[f"{domain}_macro_f1"] = domain_metrics[domain]["macro_f1"]
                    row[f"{domain}_balanced_acc"] = domain_metrics[domain]["balanced_acc"]
                if row["mean_domain_acc"] > best_acc:
                    best_acc = row["mean_domain_acc"]
                    save_checkpoint(output_dir / "best.pt", round_idx, model, server_state, row, args)
                    with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                        json.dump(row, f, indent=2, sort_keys=True)
                    if domain_local_states is not None:
                        torch.save(domain_local_states, output_dir / "best_domain_local_states.pt")

            rows.append(deepcopy(row))
            metrics_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_jsonl.flush()
            write_metrics_csv(output_dir / "metrics.csv", rows)
            if writer is not None:
                for key, value in row.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, float(value), round_idx)
                writer.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            save_checkpoint(output_dir / "latest.pt", round_idx, model, server_state, row, args)
            if domain_local_states is not None:
                torch.save(domain_local_states, output_dir / "latest_domain_local_states.pt")
        save_checkpoint(output_dir / "final.pt", args.rounds, model, server_state, rows[-1] if rows else {}, args)
        if domain_local_states is not None:
            torch.save(domain_local_states, output_dir / "final_domain_local_states.pt")
    finally:
        metrics_jsonl.close()
        if writer is not None:
            writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="FedKV-KA Office-Caltech-10 class-residual local experiment without local router"
    )
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", default="runs/fedkv_office_caltech10_class_reslocal_lslot1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)#128
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)#0.005  0.05
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model", default="vit_base_patch16_224")
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--kv-layers", default="0,3,6,9")#"0,3,6,9"
    parser.add_argument("--memory-tokens", type=int, default=8)
    parser.add_argument("--ka-alpha", type=float, default=0.7)
    parser.add_argument("--delta-scale", type=float, default=0.5)
    parser.add_argument("--local-residual-beta", type=float, default=0.2)
    parser.add_argument("--local-head-gamma", type=float, default=0.25)#0.25
    parser.add_argument("--kv-momentum", type=float, default=0.8)
    parser.add_argument("--router-tau", type=float, default=1.0)
    parser.add_argument("--router-loss-weight", type=float, default=0.02)
    parser.add_argument("--head-lr", type=float, default=5e-3)#1e-3    5e-3
    parser.add_argument("--router-lr", type=float, default=0.05)#0.005   0.05
    parser.add_argument("--memory-lr", type=float, default=0.05)#0.005   0.05
    parser.add_argument("--local-memory-lr", type=float, default=5e-2)
    parser.add_argument("--local-head-lr", type=float, default=5e-3)#1e-3  5e-3
    parser.add_argument("--local-norm-weight", type=float, default=1e-5)
    parser.add_argument("--local-head-norm-weight", type=float, default=1e-5)
    parser.add_argument("--prox-mu", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples-per-domain", type=int, default=None)
    parser.add_argument("--max-test-samples-per-domain", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.method = "class_reslocal"
    if args.smoke:
        args.model = "vit_tiny_patch16_224"
        args.allow_random_init = True
        args.rounds = 1
        args.local_epochs = 1
        args.batch_size = 4
        args.eval_batch_size = 8
        args.kv_layers = "0"
        args.memory_tokens = 2
        args.num_workers = 0
        args.max_train_samples_per_domain = 8
        args.max_test_samples_per_domain = 8
    return args


if __name__ == "__main__":
    run(parse_args())
