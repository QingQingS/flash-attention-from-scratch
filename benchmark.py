
"""
FlashAttention Benchmark
========================

Benchmarks three attention implementations:

1. Naive Attention (PyTorch)
2. PyTorch SDPA
3. FlashAttention (Triton)

Metrics:
    - Latency (ms)
    - Throughput (tokens/s)
    - Peak GPU memory (MB)

Usage:
    python benchmark.py --seq_len 512 1024 2048 4096 16384
"""

import torch
import argparse
import time
import json
from math import sqrt
from pathlib import Path
from datetime import datetime

import torch.nn.functional as F
import timeit
from torch.cuda import nvtx

# ─────────────────────────────────────────────
# Kernel imports
# ─────────────────────────────────────────────

try:
    from attention.flash_attention_triton import flash_attention
except ImportError:
    raise RuntimeError("flash_attention_triton not found")

# ─────────────────────────────────────────────
# Attention Implementations
# ─────────────────────────────────────────────

def naive_attention(q, k, v):
    scale = 1 / sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def run_naive(q, k, v):
    out = naive_attention(q, k, v)
    b, h, n, d = q.shape
    return out.transpose(1, 2).reshape(b, n, h * d)


def run_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q, k, v)
    b, h, n, d = q.shape
    return out.transpose(1, 2).reshape(b, n, h * d)


def run_flash(q, k, v):
    return flash_attention(q, k, v)

# ─────────────────────────────────────────────
# Correctness Check
# ─────────────────────────────────────────────

def check_correctness(q, k, v):

    ref = run_sdpa(q, k, v)
    out = run_flash(q, k, v)
    err = (ref - out).abs().max()
    print(f"Max error: {err.item():.3e}")
    if err > 1e-2:
        raise RuntimeError("FlashAttention correctness check failed")

# ─────────────────────────────────────────────
# Benchmark Utility
# ─────────────────────────────────────────────

def measure(fn, q, k, v, warmup=10, runs=50):
    # ── 延迟测量 ──────────────────────────────────────────
    # warmup
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)  
    end = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(runs):
        start.record()
        # torch.cuda.synchronize()
        # start = timeit.default_timer()
        out = fn(q, k, v)
        end.record()     
        torch.cuda.synchronize()
        # end = timeit.default_timer()
        # latencies.append(end - start)
        latency_ms = start.elapsed_time(end)
        latencies.append(latency_ms)

    latency = sum(latencies) / len(latencies)

    # ── 显存测量（增量）─────────────────────
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_mem = torch.cuda.memory_allocated()
    out = fn(q, k, v)
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated()
    extra_mem_mb = (peak_mem - baseline_mem) / (1024 ** 2)

    return latency, peak_mem, extra_mem_mb

def throuhput(tokens, latentcy):
    return  tokens / (latentcy / 1000)
# ─────────────────────────────────────────────
# Benchmark Loop
# ─────────────────────────────────────────────

def benchmark(seq_lens, dim, heads, batch, dtype, warmup, runs):

    device = "cuda"
    head_dim = dim // heads

    results = []

    for seq in seq_lens:

        print(f"\nseq_len = {seq}")

        torch.manual_seed(0)

        q = torch.randn(batch, heads, seq, head_dim,
                        device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # correctness once
        if seq == seq_lens[0]:
            check_correctness(q, k, v)

        # naive
        torch.cuda.empty_cache()
        naive_lat, naive_peak, naive_mem = measure(run_naive, q, k, v, warmup, runs)
        naive_throuhput = throuhput(batch * seq, naive_lat)
        # sdpa
        torch.cuda.empty_cache()
        sdpa_lat, sdpa_peak, sdpa_mem = measure(run_sdpa, q, k, v, warmup, runs)
        sdpa_throuhput = throuhput(batch * seq, sdpa_lat)

        # flash
        torch.cuda.empty_cache()
        flash_lat, flash_peak, flash_mem = measure(run_flash, q, k, v, warmup, runs)
        flash_throuhput = throuhput(batch * seq, flash_lat)

        
        results.append({
            "seq_len": seq,

            "naive_latency_ms": naive_lat,
            "sdpa_latency_ms": sdpa_lat,
            "flash_latency_ms": flash_lat,

            "naive_mem_extra": naive_mem,
            "sdpa_mem_extra": sdpa_mem,
            "flash_mem_extra": flash_mem,

            "naive_mem_peak": naive_peak,
            "sdpa_mem_peak": sdpa_peak,
            "flash_mem_peak": flash_peak,

            "flash_speedup_vs_naive": naive_lat / flash_lat,
            "flash_speedup_vs_sdpa": sdpa_lat / flash_lat,

            "naive_throughput_tokens_s": naive_throuhput,
            "sdpa_throughput_tokens_s": sdpa_throuhput,
            "flash_throughput_tokens_s": flash_throuhput

        })

    return results

# ─────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────

def gpu_info():

    idx = torch.cuda.current_device()

    prop = torch.cuda.get_device_properties(idx)

    return {
        "name": prop.name,
        "memory_gb": round(prop.total_memory / 1e9, 2),
        "cuda": torch.version.cuda,
        "torch": torch.__version__
    }


def main():

    dtype_map = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    }
    # 1. 命令行参数设置
    parser = argparse.ArgumentParser(description="Flash-Attention Benchmarking")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096, 16384])
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up steps")
    parser.add_argument("--runs", type=int, default=20, help="Measurement steps")
    parser.add_argument("--dtype", choices=list(dtype_map.keys()), default="fp16")
    parser.add_argument("--out", default="results")

    args = parser.parse_args()

    dtype = dtype_map[args.dtype]

    results = benchmark(
        seq_lens=args.seq_len,
        dim=args.dim,
        heads=args.num_heads,
        batch=args.batch_size,
        dtype=dtype,
        warmup=args.warmup,
        runs=args.runs
    )

    meta = {
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_info(),
        "config": vars(args)
    }

    payload = {
        "meta": meta,
        "results": results
    }

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    path = out / f"flash_benchmark_{int(time.time())}.json"

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved:", path)

if __name__ == "__main__":
    main()