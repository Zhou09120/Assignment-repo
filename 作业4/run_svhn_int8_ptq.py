# -*- coding: utf-8 -*-
"""
SVHN CNN INT8 static post-training quantization (PTQ) experiment.

This script is designed for Homework 4: SVHN model INT8 static quantization.
It trains or loads an FP32 CNN, applies eager-mode static quantization with
QuantStub/DeQuantStub, performs module fusion, calibrates on SVHN training
samples, converts to INT8, and reports accuracy, model size, latency, layer MSE,
and visualization figures.

Recommended command:
python run_svhn_int8_ptq.py --data-root ./data --train-fp32 --epochs 40 \
  --fp32-checkpoint ./checkpoints/svhn_fp32_best.pt --output-dir ./outputs

If you already have a Homework-2 FP32 checkpoint:
python run_svhn_int8_ptq.py --data-root ./data \
  --fp32-checkpoint ./checkpoints/svhn_fp32_best.pt --output-dir ./outputs
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.ao.quantization import (
    DeQuantStub,
    MinMaxObserver,
    QConfig,
    QuantStub,
    convert,
    fuse_modules,
    prepare,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------
# 0. Reproducibility
# -----------------------------

def seed_everything(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -----------------------------
# 1. Manual linear quantization
# -----------------------------

def linear_quantize(x: torch.Tensor, num_bits: int = 8, signed: bool = False) -> Tuple[torch.Tensor, float, int]:
    """Per-tensor asymmetric linear quantization implemented manually.

    Args:
        x: floating-point tensor.
        num_bits: quantization bit width, default 8.
        signed: if False, use uint range [0, 2^bits-1] for activations;
                if True, use int range [-2^(bits-1), 2^(bits-1)-1] for weights.

    Returns:
        q: integer tensor.
        scale: floating scale.
        zero_point: integer zero point.
    """
    if not torch.is_floating_point(x):
        raise TypeError("linear_quantize expects a floating-point tensor")

    if signed:
        qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
        dtype = torch.int8 if num_bits == 8 else torch.int32
    else:
        qmin, qmax = 0, 2 ** num_bits - 1
        dtype = torch.uint8 if num_bits == 8 else torch.int32

    x_min = float(x.min().item())
    x_max = float(x.max().item())

    # Degenerate tensor: all values are equal. Keep it numerically safe.
    if math.isclose(x_max, x_min, rel_tol=0.0, abs_tol=1e-12):
        scale = 1.0
        zero_point = qmin
        q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax).to(dtype)
        return q, scale, int(zero_point)

    scale = (x_max - x_min) / float(qmax - qmin)
    zero_point = int(round(qmin - x_min / scale))
    zero_point = int(max(qmin, min(qmax, zero_point)))

    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax).to(dtype)
    return q, float(scale), int(zero_point)


def linear_dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """Manual dequantization: x_hat = scale * (q - zero_point)."""
    return scale * (q.float() - float(zero_point))


def manual_quantization_self_test() -> Dict[str, float]:
    """Small numerical check for the manually implemented functions."""
    x = torch.linspace(-2.0, 3.0, steps=257)
    q, s, z = linear_quantize(x, num_bits=8, signed=False)
    x_hat = linear_dequantize(q, s, z)
    mse = F.mse_loss(x_hat, x).item()
    return {"manual_quant_mse": mse, "scale": s, "zero_point": z}


# -----------------------------
# 2. Quantizable SVHN CNN
# -----------------------------

class SVHNQuantCNN(nn.Module):
    """A compact but accurate quantization-friendly CNN for 32x32 SVHN images.

    Design choices for better INT8 PTQ:
      - Conv-BN-ReLU blocks are fully fuseable.
      - ReLU is non-inplace to avoid fusion/hook side effects.
      - Fixed AvgPool2d avoids dynamic shape operations.
      - QuantStub/DeQuantStub define the quantized region.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, num_classes),
        )
        self.dequant = DeQuantStub()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def fuse_model(self) -> None:
        # Conv + BN + ReLU fusion is more stable than Conv + ReLU only, and it is
        # fully compatible with the assignment requirement to fuse Conv/ReLU.
        fuse_modules(
            self,
            [
                ["features.0", "features.1", "features.2"],
                ["features.3", "features.4", "features.5"],
                ["features.7", "features.8", "features.9"],
                ["features.10", "features.11", "features.12"],
                ["features.14", "features.15", "features.16"],
                ["features.17", "features.18", "features.19"],
                ["classifier.1", "classifier.2"],
            ],
            inplace=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


# -----------------------------
# 3. Dataset
# -----------------------------

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


def build_dataloaders(
    data_root: str,
    batch_size: int,
    eval_batch_size: int,
    calib_size: int,
    num_workers: int,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])

    train_set = datasets.SVHN(root=data_root, split="train", transform=train_transform, download=download)
    train_eval_set = datasets.SVHN(root=data_root, split="train", transform=eval_transform, download=download)
    test_set = datasets.SVHN(root=data_root, split="test", transform=eval_transform, download=download)

    calib_size = min(calib_size, len(train_eval_set))
    # Deterministic calibration subset for reproducibility.
    calib_indices = list(range(calib_size))
    calib_set = Subset(train_eval_set, calib_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    calib_loader = DataLoader(
        calib_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader, calib_loader


# -----------------------------
# 4. Training and evaluation
# -----------------------------

@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_fp32(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: Path,
) -> float:
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)

        scheduler.step()
        acc = evaluate_accuracy(model, test_loader, device)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:03d}/{epochs} | loss={avg_loss:.4f} | test_acc={acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "best_acc": best_acc}, checkpoint_path)
            print(f"  Saved best checkpoint to {checkpoint_path} ({best_acc * 100:.2f}%)")

    return best_acc


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


# -----------------------------
# 5. Static PTQ
# -----------------------------

def choose_quant_backend(preferred: str) -> str:
    supported = list(torch.backends.quantized.supported_engines)
    if preferred in supported:
        return preferred
    for candidate in ["x86", "fbgemm", "qnnpack", "onednn"]:
        if candidate in supported:
            return candidate
    raise RuntimeError(f"No supported quantized engine found. supported_engines={supported}")


def build_asymmetric_per_tensor_qconfig() -> QConfig:
    """Activation and weight observers: per-tensor affine/asymmetric INT8.

    Activations use uint8 [0,255]. Weights use int8 [-128,127] with affine
    qscheme, which is the closest practical signed-weight analogue of the
    assignment's per-tensor asymmetric linear quantization.
    """
    activation_observer = MinMaxObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
        reduce_range=False,
    )
    weight_observer = MinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        quant_min=-128,
        quant_max=127,
        reduce_range=False,
    )
    return QConfig(activation=activation_observer, weight=weight_observer)


@torch.inference_mode()
def calibrate_model(prepared_model: nn.Module, calib_loader: DataLoader, num_batches: int) -> None:
    prepared_model.eval()
    for i, (x, _) in enumerate(calib_loader):
        if i >= num_batches:
            break
        prepared_model(x.cpu())


def quantize_static_ptq(
    fp32_model: SVHNQuantCNN,
    calib_loader: DataLoader,
    backend: str,
    calib_batches: int,
) -> Tuple[nn.Module, nn.Module]:
    torch.backends.quantized.engine = backend

    fp32_fused = copy.deepcopy(fp32_model).cpu().eval()
    fp32_fused.fuse_model()

    prepared = copy.deepcopy(fp32_fused).eval()
    prepared.qconfig = build_asymmetric_per_tensor_qconfig()
    prepare(prepared, inplace=True)
    calibrate_model(prepared, calib_loader, num_batches=calib_batches)
    int8_model = convert(prepared, inplace=False).eval()
    return fp32_fused, int8_model


# -----------------------------
# 6. Metrics: size, latency, MSE
# -----------------------------

def get_model_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(model.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return size_mb


@torch.inference_mode()
def benchmark_latency_ms(
    model: nn.Module,
    sample: torch.Tensor,
    warmup: int = 100,
    iters: int = 1000,
) -> float:
    model.eval().cpu()
    x = sample[:1].contiguous().cpu()

    # Avoid timing thread start-up and cache cold-start effects.
    for _ in range(warmup):
        _ = model(x)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Module {name} not found. Available keys include: {list(modules)[:20]}")
    return modules[name]


def _make_activation_hook(name: str, store: Dict[str, torch.Tensor]):
    def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
        if isinstance(output, torch.Tensor):
            out = output.dequantize() if getattr(output, "is_quantized", False) else output.detach().float()
            store[name] = out.cpu()
    return hook


@torch.inference_mode()
def compute_layer_mse(
    fp32_fused: nn.Module,
    int8_model: nn.Module,
    loader: DataLoader,
    layer_names: List[str],
    max_batches: int,
) -> Dict[str, float]:
    fp32_fused.eval().cpu()
    int8_model.eval().cpu()

    fp32_sse = {name: 0.0 for name in layer_names}
    fp32_count = {name: 0 for name in layer_names}

    fp_store: Dict[str, torch.Tensor] = {}
    q_store: Dict[str, torch.Tensor] = {}

    handles = []
    for name in layer_names:
        handles.append(_get_module_by_name(fp32_fused, name).register_forward_hook(_make_activation_hook(name, fp_store)))
        handles.append(_get_module_by_name(int8_model, name).register_forward_hook(_make_activation_hook(name, q_store)))

    try:
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            fp_store.clear()
            q_store.clear()
            x = x.cpu()
            _ = fp32_fused(x)
            _ = int8_model(x)
            for name in layer_names:
                if name not in fp_store or name not in q_store:
                    continue
                a = fp_store[name]
                b = q_store[name]
                if a.shape != b.shape:
                    continue
                diff = (a - b).float()
                fp32_sse[name] += float(torch.sum(diff * diff).item())
                fp32_count[name] += diff.numel()
    finally:
        for h in handles:
            h.remove()

    return {name: fp32_sse[name] / max(fp32_count[name], 1) for name in layer_names}


def save_bar_plot(labels: List[str], values: List[float], ylabel: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(5.0, 4.0), dpi=160)
    bars = plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_layer_mse_csv(layer_mse: Dict[str, float], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "mse"])
        for k, v in layer_mse.items():
            writer.writerow([k, f"{v:.10e}"])


# -----------------------------
# 7. Main
# -----------------------------

@dataclass
class ExperimentResults:
    fp32_accuracy: float
    int8_accuracy: float
    accuracy_drop: float
    fp32_size_mb: float
    int8_size_mb: float
    compression_ratio: float
    fp32_latency_ms: float
    int8_latency_ms: float
    speedup: float
    layer_mse: Dict[str, float]
    backend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SVHN INT8 static quantization homework solution")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--fp32-checkpoint", type=str, default="./checkpoints/svhn_fp32_best.pt")
    parser.add_argument("--train-fp32", action="store_true", help="Train FP32 model before quantization")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--calib-size", type=int, default=4096)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--mse-batches", type=int, default=16)
    parser.add_argument("--latency-iters", type=int, default=1000)
    parser.add_argument("--latency-warmup", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backend", type=str, default="fbgemm", help="Preferred quant backend: fbgemm/x86/qnnpack/onednn")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num-threads", type=int, default=1, help="CPU threads for latency benchmark")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.fp32_checkpoint)

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    backend = choose_quant_backend(args.backend)
    print(f"Using quantized backend: {backend}")
    print("Manual quantization self-test:", manual_quantization_self_test())

    train_loader, test_loader, calib_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        calib_size=args.calib_size,
        num_workers=args.num_workers,
        download=not args.no_download,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp32_model = SVHNQuantCNN(num_classes=10)

    if args.train_fp32 or not ckpt_path.exists():
        print("Training FP32 baseline...")
        train_fp32(
            fp32_model,
            train_loader,
            test_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_path=ckpt_path,
        )

    print(f"Loading FP32 checkpoint: {ckpt_path}")
    load_checkpoint(fp32_model, ckpt_path, device=torch.device("cpu"))
    fp32_model.eval().cpu()

    print("Building INT8 static PTQ model...")
    fp32_fused, int8_model = quantize_static_ptq(fp32_model, calib_loader, backend, args.calib_batches)

    print("Evaluating accuracy...")
    fp32_acc = evaluate_accuracy(fp32_fused, test_loader, torch.device("cpu"))
    int8_acc = evaluate_accuracy(int8_model, test_loader, torch.device("cpu"))

    print("Measuring model size...")
    fp32_size = get_model_size_mb(fp32_fused)
    int8_size = get_model_size_mb(int8_model)

    sample_x, _ = next(iter(test_loader))
    print("Benchmarking latency on CPU single-image input...")
    fp32_latency = benchmark_latency_ms(fp32_fused, sample_x, warmup=args.latency_warmup, iters=args.latency_iters)
    int8_latency = benchmark_latency_ms(int8_model, sample_x, warmup=args.latency_warmup, iters=args.latency_iters)

    print("Computing layer output MSE...")
    layer_names = [
        "features.0",
        "features.3",
        "features.7",
        "features.10",
        "features.14",
        "features.17",
        "classifier.1",
        "classifier.3",
    ]
    layer_mse = compute_layer_mse(fp32_fused, int8_model, test_loader, layer_names, args.mse_batches)

    results = ExperimentResults(
        fp32_accuracy=fp32_acc,
        int8_accuracy=int8_acc,
        accuracy_drop=fp32_acc - int8_acc,
        fp32_size_mb=fp32_size,
        int8_size_mb=int8_size,
        compression_ratio=fp32_size / max(int8_size, 1e-12),
        fp32_latency_ms=fp32_latency,
        int8_latency_ms=int8_latency,
        speedup=fp32_latency / max(int8_latency, 1e-12),
        layer_mse=layer_mse,
        backend=backend,
    )

    metrics = {
        "backend": results.backend,
        "fp32_accuracy_percent": results.fp32_accuracy * 100.0,
        "int8_accuracy_percent": results.int8_accuracy * 100.0,
        "accuracy_drop_percent_point": results.accuracy_drop * 100.0,
        "fp32_size_mb": results.fp32_size_mb,
        "int8_size_mb": results.int8_size_mb,
        "compression_ratio": results.compression_ratio,
        "fp32_latency_ms": results.fp32_latency_ms,
        "int8_latency_ms": results.int8_latency_ms,
        "speedup": results.speedup,
        "layer_mse": results.layer_mse,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_layer_mse_csv(results.layer_mse, out_dir / "layer_mse.csv")

    save_bar_plot(
        labels=["FP32", "INT8"],
        values=[results.fp32_accuracy * 100.0, results.int8_accuracy * 100.0],
        ylabel="Accuracy (%)",
        title="SVHN accuracy before/after INT8 PTQ",
        out_path=out_dir / "accuracy_comparison.png",
    )
    save_bar_plot(
        labels=["FP32", "INT8"],
        values=[results.fp32_latency_ms, results.int8_latency_ms],
        ylabel="Latency (ms / image, CPU)",
        title="SVHN single-image CPU latency before/after INT8 PTQ",
        out_path=out_dir / "latency_comparison.png",
    )

    # Save the converted quantized model state for reproducible submission.
    torch.save(int8_model.state_dict(), out_dir / "svhn_int8_ptq_state_dict.pt")

    print("\n==== Final Results ====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nAll outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
