#!/usr/bin/env python3
"""FedKV class-residual local experiment on CIFAR100.

This script is intentionally self-contained: it reads the CIFAR100 python
pickle files, builds client partitions, patches timm ViT attention layers with
the class-conditioned local residual branch, runs federated training, and
writes metrics/logs.

This dedicated variant removes the Shared Residual Bank branch and removes the
local router/local slot mechanism entirely. Each client owns one private local
KV residual per patched attention layer.
"""

import argparse
import json
import pickle
import random
import time
import types
from collections import defaultdict
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


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class CIFAR100Pickle(Dataset):
    def __init__(self, root, split, indices=None, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        payload_path = self.root / split
        with payload_path.open("rb") as f:
            payload = pickle.load(f, encoding="latin1")

        data = payload["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.data = data
        self.targets = np.asarray(payload["fine_labels"], dtype=np.int64)
        self.indices = np.asarray(indices if indices is not None else np.arange(len(self.targets)), dtype=np.int64)

    def __len__(self):
        return int(len(self.indices))

    def __getitem__(self, item):
        idx = int(self.indices[item])
        image = Image.fromarray(self.data[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[idx])


def build_transforms(train=True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomCrop(224, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def read_targets(root, split):
    with (Path(root) / split).open("rb") as f:
        payload = pickle.load(f, encoding="latin1")
    return np.asarray(payload["fine_labels"], dtype=np.int64)


def make_client_classes(num_clients, num_classes, classes_per_client, seed):
    if num_clients * classes_per_client < num_classes:
        raise ValueError("num_clients * classes_per_client must be >= num_classes")

    rng = np.random.default_rng(seed)
    client_classes = [set() for _ in range(num_clients)]
    capacities = [classes_per_client for _ in range(num_clients)]

    client_order = np.arange(num_clients)
    rng.shuffle(client_order)
    for offset, c in enumerate(rng.permutation(num_classes).tolist()):
        client_id = int(client_order[offset % num_clients])
        while capacities[client_id] == 0:
            client_id = int(rng.integers(0, num_clients))
        client_classes[client_id].add(int(c))
        capacities[client_id] -= 1

    all_classes = np.arange(num_classes)
    for client_id in range(num_clients):
        while capacities[client_id] > 0:
            available = np.setdiff1d(all_classes, np.asarray(list(client_classes[client_id]), dtype=np.int64))
            c = int(rng.choice(available))
            client_classes[client_id].add(c)
            capacities[client_id] -= 1

    return [np.asarray(sorted(s), dtype=np.int64) for s in client_classes]


def partition_by_class_subset(targets, client_classes, alpha, seed):
    rng = np.random.default_rng(seed)
    num_clients = len(client_classes)
    by_class = defaultdict(list)
    for idx, y in enumerate(targets):
        by_class[int(y)].append(idx)

    class_to_clients = defaultdict(list)
    for client_id, classes in enumerate(client_classes):
        for c in classes:
            class_to_clients[int(c)].append(client_id)

    client_indices = [[] for _ in range(num_clients)]
    for c, indices in by_class.items():
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        owners = np.asarray(class_to_clients[c], dtype=np.int64)
        if len(owners) == 0:
            owners = np.asarray([int(rng.integers(0, num_clients))], dtype=np.int64)
        probs = rng.dirichlet(np.full(len(owners), alpha, dtype=np.float64))
        counts = rng.multinomial(len(indices), probs)
        offset = 0
        for owner, count in zip(owners, counts):
            if count:
                client_indices[int(owner)].extend(indices[offset : offset + count].tolist())
            offset += count

    for indices in client_indices:
        rng.shuffle(indices)
    return [np.asarray(indices, dtype=np.int64) for indices in client_indices]


class ClassResidualLocalAttention(nn.Module):
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
        super().__init__()
        self.qkv = source_attn.qkv
        self.q_norm = source_attn.q_norm
        self.k_norm = source_attn.k_norm
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
        self.local_residual_beta = local_residual_beta
        self.router_tau = router_tau
        self.active_classes = None
        self.last_router_logits = None
        self.last_subset_logits = None

        self.base_k = nn.Parameter(torch.empty(self.num_heads, memory_tokens, self.head_dim))
        self.base_v = nn.Parameter(torch.empty(self.num_heads, memory_tokens, self.head_dim))
        self.delta_k = nn.Parameter(torch.empty(num_classes, self.num_heads, memory_tokens, self.head_dim))
        self.delta_v = nn.Parameter(torch.empty(num_classes, self.num_heads, memory_tokens, self.head_dim))
        self.router = nn.Linear(self.num_heads * self.head_dim, num_classes)
        self.local_mem_k = nn.Parameter(torch.empty(self.num_heads, memory_tokens, self.head_dim))
        self.local_mem_v = nn.Parameter(torch.empty(self.num_heads, memory_tokens, self.head_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.base_k, std=0.02)
        nn.init.normal_(self.base_v, std=0.02)
        nn.init.zeros_(self.delta_k)
        nn.init.zeros_(self.delta_v)
        nn.init.normal_(self.local_mem_k, std=0.02)
        nn.init.normal_(self.local_mem_v, std=0.02)
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
        img_attn = q_scaled @ k.transpose(-2, -1)
        img_attn = img_attn.softmax(dim=-1)
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

        mem_attn = torch.einsum("bhnd,bhmd->bhnm", q_scaled, mem_k)
        mem_attn = mem_attn.softmax(dim=-1)
        mem_attn = self.attn_drop(mem_attn)
        mem_out = torch.einsum("bhnm,bhmd->bhnd", mem_attn, mem_v)

        local_attn = torch.einsum("bhnd,hmd->bhnm", q_scaled, self.local_mem_k)
        local_attn = local_attn.softmax(dim=-1)
        local_attn = self.attn_drop(local_attn)
        local_out = torch.einsum("bhnm,hmd->bhnd", local_attn, self.local_mem_v)

        out = img_out + self.ka_alpha * (mem_out + self.local_residual_beta * local_out)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FedKVViT(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fedkv_layers = [
            module for module in self.model.modules() if isinstance(module, ClassResidualLocalAttention)
        ]

    def set_active_classes(self, classes):
        for module in self.fedkv_layers:
            module.set_active_classes(classes)

    def forward(self, x, active_classes=None):
        self.set_active_classes(active_classes)
        return self.model(x)


def forward_with_local_head(self, x):
    features = self.forward_features(x)
    pooled = self.forward_head(features, pre_logits=True)
    return self.head(pooled) + self.local_head_gamma * self.local_head(pooled)


def patch_vit_with_fedkv(model, kv_layers, num_classes, memory_tokens, alpha, delta_scale, router_tau, args):
    for layer_idx in kv_layers:
        source_attn = model.blocks[layer_idx].attn
        model.blocks[layer_idx].attn = ClassResidualLocalAttention(
            source_attn=source_attn,
            num_classes=num_classes,
            memory_tokens=memory_tokens,
            ka_alpha=alpha,
            delta_scale=delta_scale,
            router_tau=router_tau,
            local_residual_beta=args.local_residual_beta,
        )
    embed_dim = getattr(model, "num_features", None) or model.head.in_features
    model.local_head = nn.Linear(embed_dim, num_classes)
    model.local_head_gamma = args.local_head_gamma
    model.forward = types.MethodType(forward_with_local_head, model)
    return FedKVViT(model)


def strip_state_dict_prefix(state):
    for prefix in ("module.", "model."):
        if all(k.startswith(prefix) for k in state.keys()):
            return {k[len(prefix) :]: v for k, v in state.items()}
    return state


def load_pretrained_or_fail(model_name, num_classes, pretrained_path, allow_random_init):
    if pretrained_path:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        state = strip_state_dict_prefix(state)
        model_state = model.state_dict()
        compatible = {
            k: v
            for k, v in state.items()
            if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}: {len(compatible)} tensors")
        print(f"Skipped missing={len(missing)} unexpected={len(unexpected)} incompatible={len(state) - len(compatible)}")
        return model

    if allow_random_init:
        print("WARNING: using random initialization because --allow-random-init was set.")
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    try:
        return timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    except Exception as exc:
        raise RuntimeError(
            "Failed to create pretrained timm model. Provide --pretrained-path when network/cache is unavailable."
        ) from exc


def freeze_for_fedkv(model):
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model.model, "head"):
        for param in model.model.head.parameters():
            param.requires_grad = True
    for module in model.fedkv_layers:
        for name, param in module.named_parameters():
            if name.startswith(("base_", "delta_", "router.", "local_mem_")):
                param.requires_grad = True
    if hasattr(model.model, "local_head"):
        for param in model.model.local_head.parameters():
            param.requires_grad = True


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

    for name, server_tensor in server_state.items():
        if is_delta_param(name):
            for c in range(num_classes):
                owners = [i for i, classes in enumerate(client_classes) if c in classes]
                if not owners:
                    continue
                denom = float(sum(client_sizes[i] for i in owners))
                agg = sum(client_updates[i][name][c] * (client_sizes[i] / denom) for i in owners)
                new_state[name][c] = kv_momentum * server_tensor[c] + (1.0 - kv_momentum) * agg
        elif is_class_row_param(name, server_tensor, num_classes):
            for c in range(num_classes):
                owners = [i for i, classes in enumerate(client_classes) if c in classes]
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
        elif ".base_" in name or ".delta_" in name:
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
    if args.prox_mu <= 0:
        return torch.zeros((), device=device)
    loss = torch.zeros((), device=device)
    for name, param in model.named_parameters():
        if not param.requires_grad or is_local_param(name) or name not in server_state:
            continue
        if (
            ".base_" in name
            or ".delta_" in name
            or ".router." in name
        ):
            loss = loss + (param - server_state[name].to(device)).pow(2).mean()
    return args.prox_mu * loss


def local_train(model, dataset, indices, active_classes, args, device, server_state):
    if len(indices) == 0:
        return
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices.tolist()),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(optimizer_param_groups(model, args), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    active = torch.as_tensor(active_classes, dtype=torch.long, device=device)
    label_to_pos = torch.full((args.num_classes,), -1, dtype=torch.long, device=device)
    label_to_pos[active] = torch.arange(active.numel(), device=device)

    for _ in range(args.local_epochs):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, active_classes=active)
            loss = F.cross_entropy(logits, labels)
            if args.router_loss_weight > 0:
                router_targets = label_to_pos[labels]
                if (router_targets < 0).any():
                    raise ValueError("Encountered labels outside the client's active class subset")
                router_losses = [F.cross_entropy(module.last_subset_logits, router_targets) for module in model.fedkv_layers]
                loss = loss + args.router_loss_weight * torch.stack(router_losses).mean()
            loss = loss + private_regularizer(model, args, device)
            loss = loss + prox_regularizer(model, server_state, args, device)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate(model, dataset, client_indices, client_classes, args, device, server_state, client_local_states):
    model.eval()
    per_client_acc = []
    all_preds = []
    all_labels = []
    for client_id, indices in enumerate(client_indices):
        if len(indices) == 0:
            continue
        load_trainable_state_dict(model, server_state)
        load_trainable_state_dict(model, client_local_states[client_id])
        loader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            sampler=indices.tolist(),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        active = torch.as_tensor(client_classes[client_id], dtype=torch.long, device=device)
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
        if total:
            per_client_acc.append(correct / total)

    if len(all_labels) == 0:
        return {
            "mean_client_acc": 0.0,
            "worst_client_acc": 0.0,
            "macro_f1": 0.0,
            "balanced_acc": 0.0,
            "evaluated_clients": 0,
        }
    return {
        "mean_client_acc": float(np.mean(per_client_acc)),
        "worst_client_acc": float(np.min(per_client_acc)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", labels=list(range(args.num_classes)), zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(all_labels, all_preds)),
        "evaluated_clients": int(len(per_client_acc)),
    }


def save_checkpoint(path, round_idx, model, server_state, metrics, args, client_local_states=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "round": round_idx,
            "model": model.state_dict(),
            "server_trainable_state": server_state,
            "client_local_states": client_local_states,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="FedKV-KA CIFAR100 class-residual local experiment without local router")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-dir", default="runs/fedkv_cifar100_class_reslocal_lslot1_seed1")
    parser.add_argument("--model", default="vit_base_patch16_224")
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--clients-per-round", type=int, default=8)
    parser.add_argument("--classes-per-client", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--kv-layers", nargs="+", type=int, default=[0, 3, 6, 9])
    parser.add_argument("--memory-tokens", type=int, default=8)
    parser.add_argument("--ka-alpha", type=float, default=0.7)
    parser.add_argument("--delta-scale", type=float, default=0.5)
    parser.add_argument("--local-residual-beta", type=float, default=0.2)
    parser.add_argument("--local-head-gamma", type=float, default=0.25)
    parser.add_argument("--kv-momentum", type=float, default=0.8)
    parser.add_argument("--router-tau", type=float, default=1.0)
    parser.add_argument("--router-loss-weight", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--head-lr", type=float, default=0.005)
    parser.add_argument("--memory-lr", type=float, default=0.005)
    parser.add_argument("--router-lr", type=float, default=0.005)
    parser.add_argument("--local-memory-lr", type=float, default=0.005)
    parser.add_argument("--local-head-lr", type=float, default=0.005)
    parser.add_argument("--local-norm-weight", type=float, default=1e-5)
    parser.add_argument("--local-head-norm-weight", type=float, default=1e-5)
    parser.add_argument("--prox-mu", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-test-samples", type=int, default=0)
    return parser.parse_args()


def prepare_indices(args, alpha):
    if args.clients * args.classes_per_client < args.num_classes:
        raise ValueError("--clients * --classes-per-client must cover --num-classes")
    train_targets = read_targets(args.data_dir, "train")
    test_targets = read_targets(args.data_dir, "test")
    if args.limit_train_samples > 0:
        train_targets = train_targets[: args.limit_train_samples]
    if args.limit_test_samples > 0:
        test_targets = test_targets[: args.limit_test_samples]
    client_classes = make_client_classes(args.clients, args.num_classes, args.classes_per_client, args.seed)
    train_indices = partition_by_class_subset(train_targets, client_classes, alpha, args.seed + 1000)
    test_indices = partition_by_class_subset(test_targets, client_classes, alpha, args.seed + 2000)
    return client_classes, train_indices, test_indices


def run_alpha(args, alpha, device):
    run_dir = Path(args.output_dir) / f"alpha_{alpha:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    metrics_path = run_dir / "metrics.jsonl"

    set_seed(args.seed)
    client_classes, train_indices, test_indices = prepare_indices(args, alpha)
    train_dataset = CIFAR100Pickle(
        args.data_dir,
        "train",
        indices=np.arange(args.limit_train_samples) if args.limit_train_samples > 0 else None,
        transform=build_transforms(train=True),
    )
    test_dataset = CIFAR100Pickle(
        args.data_dir,
        "test",
        indices=np.arange(args.limit_test_samples) if args.limit_test_samples > 0 else None,
        transform=build_transforms(train=False),
    )

    base_model = load_pretrained_or_fail(args.model, args.num_classes, args.pretrained_path, args.allow_random_init)
    model = patch_vit_with_fedkv(
        base_model,
        kv_layers=args.kv_layers,
        num_classes=args.num_classes,
        memory_tokens=args.memory_tokens,
        alpha=args.ka_alpha,
        delta_scale=args.delta_scale,
        router_tau=args.router_tau,
        args=args,
    ).to(device)
    freeze_for_fedkv(model)
    server_state = trainable_state_dict(model, local=False)
    initial_local_state = trainable_state_dict(model, local=True)
    client_local_states = [deepcopy(initial_local_state) for _ in range(args.clients)]

    rng = np.random.default_rng(args.seed + int(alpha * 10000))
    best_acc = -1.0
    best_metrics = None
    metrics_file = metrics_path.open("w", encoding="utf-8")
    try:
        for round_idx in range(1, args.rounds + 1):
            started = time.time()
            sampled = rng.choice(args.clients, size=args.clients_per_round, replace=False).tolist()
            client_updates = []
            sampled_classes = []
            sampled_sizes = []

            for client_id in sampled:
                if len(train_indices[client_id]) == 0:
                    continue
                load_trainable_state_dict(model, server_state)
                load_trainable_state_dict(model, client_local_states[client_id])
                local_train(model, train_dataset, train_indices[client_id], client_classes[client_id], args, device, server_state)
                client_updates.append(trainable_state_dict(model, local=False))
                client_local_states[client_id] = trainable_state_dict(model, local=True)
                sampled_classes.append(set(int(c) for c in client_classes[client_id]))
                sampled_sizes.append(max(1, int(len(train_indices[client_id]))))

            if client_updates:
                server_state = aggregate_updates(
                    server_state,
                    client_updates,
                    sampled_classes,
                    sampled_sizes,
                    args.num_classes,
                    args.kv_momentum,
                )
                load_trainable_state_dict(model, server_state)

            metrics = {
                "round": round_idx,
                "alpha": alpha,
                "selected_client_ids": sampled,
                "seconds": time.time() - started,
            }
            if round_idx % args.eval_every == 0 or round_idx == args.rounds:
                eval_metrics = evaluate(
                    model,
                    test_dataset,
                    test_indices,
                    client_classes,
                    args,
                    device,
                    server_state,
                    client_local_states,
                )
                metrics.update(eval_metrics)
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, round_idx)
                if eval_metrics["mean_client_acc"] > best_acc:
                    best_acc = eval_metrics["mean_client_acc"]
                    best_metrics = deepcopy(metrics)
                    save_checkpoint(
                        run_dir / "best.pt",
                        round_idx,
                        model,
                        server_state,
                        metrics,
                        args,
                        client_local_states=client_local_states,
                    )
                    torch.save(client_local_states, run_dir / "best_client_local_states.pt")

            metrics_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            metrics_file.flush()
            print(json.dumps(metrics, ensure_ascii=False))

        save_checkpoint(
            run_dir / "final.pt",
            args.rounds,
            model,
            server_state,
            best_metrics or {},
            args,
            client_local_states=client_local_states,
        )
        torch.save(client_local_states, run_dir / "final_client_local_states.pt")
    finally:
        metrics_file.close()
        writer.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not visible. Full CIFAR100 training is expected to be slow on CPU.")
    print(f"device={device}")
    for alpha in args.alphas:
        run_alpha(args, alpha, device)


if __name__ == "__main__":
    main()
