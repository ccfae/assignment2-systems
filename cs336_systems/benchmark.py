"""End-to-end benchmark and Nsight Systems profiling for the Assignment 1 Transformer.

Edit CONFIG below, then run from the Assignment 2 repository with:

    uv run python -m cs336_systems.benchmark

This script follows Assignment 2, Section 2.1.3.  It measures GPU time only
after synchronizing the selected device, so asynchronous GPU work is included
in each measurement.

For Section 2.1.4, set ``device="cuda"`` and a non-``"none"``
``profile_mode`` in CONFIG.  The profile run performs warm-up first, then
executes only the selected mode inside NVTX ranges.  This lets Nsight capture
the measured step while ignoring warm-up work.
"""

from __future__ import annotations

import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import torch

from cs336_basics.model import BasicsTransformerLM as TransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW


@dataclass(frozen=True)
class ModelSize:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


# Table 1 in Assignment 2, Section 2.1.2.
MODEL_SIZES: dict[str, ModelSize] = {
    "small": ModelSize(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSize(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSize(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSize(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelSize(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


@dataclass
class BenchmarkConfig:
    # Change this block before running.  Start with small on your M5 Mac.
    model_size: str = "small"
    device: str = "mps"  # Change to "cuda" on an NVIDIA machine.
    vocab_size: int = 10_000
    batch_size: int = 4
    context_length: int = 512
    rope_theta: float = 10_000.0

    learning_rate: float = 1e-3
    warmup_steps: int = 5
    measurement_steps: int = 10
    seed: int = 42

    # Section 2.1.4: change this to profile one mode with Nsight Systems.
    # Use "none" for the ordinary Section 2.1.3 benchmark.
    profile_mode: Literal["none", "forward", "forward_backward", "full"] = "none"
    profile_steps: int = 1


CONFIG = BenchmarkConfig()


def synchronize(device: str) -> None:
    """Wait until previously queued work on the selected accelerator has finished."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device == "mps":
        torch.mps.synchronize()


def validate_config(config: BenchmarkConfig) -> ModelSize:
    if config.model_size not in MODEL_SIZES:
        choices = ", ".join(MODEL_SIZES)
        raise ValueError(f"Unknown model_size={config.model_size!r}; choose one of: {choices}.")
    if config.warmup_steps < 0 or config.measurement_steps <= 0 or config.profile_steps <= 0:
        raise ValueError("warmup_steps must be non-negative; measurement_steps and profile_steps must be positive.")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CONFIG asks for CUDA, but CUDA is unavailable.")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("CONFIG asks for MPS, but MPS is unavailable.")
    if config.profile_mode != "none" and not config.device.startswith("cuda"):
        raise RuntimeError("Nsight profiling requires CONFIG.device to be a CUDA device, such as 'cuda'.")
    return MODEL_SIZES[config.model_size]


def make_model(config: BenchmarkConfig, size: ModelSize) -> TransformerLM:
    return TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=size.d_model,
        num_layers=size.num_layers,
        num_heads=size.num_heads,
        d_ff=size.d_ff,
        rope_theta=config.rope_theta,
    ).to(config.device)


def make_random_batch(config: BenchmarkConfig) -> tuple[torch.Tensor, torch.Tensor]:
    # Inputs and targets have shape (batch_size, context_length), just like get_batch.
    inputs = torch.randint(
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=config.device,
        dtype=torch.long,
    )
    targets = torch.randint_like(inputs, high=config.vocab_size)
    return inputs, targets


def milliseconds_since(start: float, device: str) -> float:
    synchronize(device)
    return (time.perf_counter() - start) * 1_000


def summarize(name: str, measurements_ms: list[float]) -> str:
    mean = statistics.mean(measurements_ms)
    std = statistics.stdev(measurements_ms) if len(measurements_ms) > 1 else 0.0
    return f"{name:<30} mean={mean:9.2f} ms | std={std:7.2f} ms"


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Create an Nsight-visible range on CUDA, and do nothing elsewhere."""
    if torch.cuda.is_available():
        with torch.cuda.nvtx.range(name):
            yield
    else:
        yield


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The usual attention calculation, with NVTX labels for Section 2.1.4."""
    d_k = K.shape[-1]
    with nvtx_range("attention: QK^T matmul"):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        with nvtx_range("attention: causal mask"):
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx_range("attention: softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx_range("attention: AV matmul"):
        return torch.matmul(attention_weights, V)


def install_attention_profile_annotations() -> None:
    """Swap in an equivalent attention function whose inner operations are labeled."""
    import cs336_basics.model as model_module

    model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def run_nsys_profile(
    config: BenchmarkConfig,
    model: TransformerLM,
    optimizer: AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Run only the requested profiled operation after warm-up has completed."""
    print(
        f"Nsight profile: mode={config.profile_mode} | model={config.model_size} | "
        f"context={config.context_length} | steps={config.profile_steps}"
    )

    for _ in range(config.profile_steps):
        # This outer range is the capture target passed to nsys with
        # --capture-range=nvtx --nvtx-capture=profiled_step.
        with nvtx_range("profiled_step"):
            if config.profile_mode == "forward":
                model.eval()
                with torch.no_grad():
                    with nvtx_range("forward"):
                        _ = model(inputs)
            else:
                model.train()
                with nvtx_range("zero_grad"):
                    optimizer.zero_grad(set_to_none=True)
                with nvtx_range("forward"):
                    logits = model(inputs)
                with nvtx_range("loss"):
                    loss = cross_entropy(logits, targets)
                with nvtx_range("backward"):
                    loss.backward()
                if config.profile_mode == "full":
                    with nvtx_range("optimizer_step"):
                        optimizer.step()
        synchronize(config.device)


def run_full_training_step(
    model: TransformerLM,
    optimizer: AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()


def benchmark(config: BenchmarkConfig = CONFIG) -> None:
    size = validate_config(config)
    torch.manual_seed(config.seed)

    model = make_model(config, size)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    inputs, targets = make_random_batch(config)

    # Warm-up matters: it initializes optimizer state and lets the accelerator/runtime settle.
    model.train()
    for _ in range(config.warmup_steps):
        run_full_training_step(model, optimizer, inputs, targets)
    synchronize(config.device)

    # For Nsight, annotate only after warm-up so --nvtx-capture can exclude it.
    if config.profile_mode != "none":
        install_attention_profile_annotations()
        run_nsys_profile(config, model, optimizer, inputs, targets)
        return

    forward_only_ms: list[float] = []
    forward_ms: list[float] = []
    backward_ms: list[float] = []
    optimizer_ms: list[float] = []
    forward_backward_ms: list[float] = []
    full_step_ms: list[float] = []

    # 1. Inference-style forward-only timing: no autograd graph is created.
    model.eval()
    with torch.no_grad():
        for _ in range(config.measurement_steps):
            synchronize(config.device)
            start = time.perf_counter()
            _ = model(inputs)
            forward_only_ms.append(milliseconds_since(start, config.device))

    # 2. Training timings.  Time each component separately and also the combined modes.
    model.train()
    for _ in range(config.measurement_steps):
        optimizer.zero_grad(set_to_none=True)

        synchronize(config.device)
        start = time.perf_counter()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        forward_time = milliseconds_since(start, config.device)
        forward_ms.append(forward_time)

        synchronize(config.device)
        start = time.perf_counter()
        loss.backward()
        backward_time = milliseconds_since(start, config.device)
        backward_ms.append(backward_time)

        synchronize(config.device)
        start = time.perf_counter()
        optimizer.step()
        optimizer_time = milliseconds_since(start, config.device)
        optimizer_ms.append(optimizer_time)

    # The handout also asks for these as independent end-to-end measurements.
    # Do not derive them by adding the component times above: direct timing most
    # closely matches how a real training loop queues its work.
    for _ in range(config.measurement_steps):
        optimizer.zero_grad(set_to_none=True)
        synchronize(config.device)
        start = time.perf_counter()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        forward_backward_ms.append(milliseconds_since(start, config.device))

    for _ in range(config.measurement_steps):
        synchronize(config.device)
        start = time.perf_counter()
        run_full_training_step(model, optimizer, inputs, targets)
        full_step_ms.append(milliseconds_since(start, config.device))

    print("\nAssignment 2, Section 2.1.3 - End-to-End Benchmark")
    print(
        f"model={config.model_size} | device={config.device} | "
        f"batch={config.batch_size} | context={config.context_length} | "
        f"warmup={config.warmup_steps} | measurements={config.measurement_steps}"
    )
    print("-" * 72)
    for name, values in (
        ("forward only (no_grad)", forward_only_ms),
        ("forward (training graph)", forward_ms),
        ("backward", backward_ms),
        ("optimizer step", optimizer_ms),
        ("forward + backward", forward_backward_ms),
        ("full training step", full_step_ms),
    ):
        print(summarize(name, values))


if __name__ == "__main__":
    benchmark()
