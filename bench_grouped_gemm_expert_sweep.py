#!/usr/bin/env python3
"""Standalone grouped GEMM benchmark sweeping expert count (B).

Uses Primus-Turbo model benchmark config to derive GEMM shape:
- GateUP: N=2*moe_intermediate_size, K=hidden_size
- Down:   N=hidden_size,            K=moe_intermediate_size
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch


def _import_turbo_and_bench_cfg():
    root = Path(__file__).resolve().parent
    ops_cfg_dir = root / "Primus-Turbo" / "benchmark" / "ops"
    if str(ops_cfg_dir) not in sys.path:
        sys.path.insert(0, str(ops_cfg_dir))

    import config as bench_cfg  # type: ignore
    import primus_turbo.pytorch as turbo  # type: ignore

    return bench_cfg, turbo


def _parse_expert_list(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        val = int(token)
        if val <= 0:
            raise ValueError(f"Expert count must be > 0, got {val}")
        values.append(val)
    if not values:
        raise ValueError("Empty --experts list")
    return sorted(set(values))


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def bench_one(
    turbo,
    *,
    b: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: int,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    torch.cuda.set_device(device)
    x = torch.randn((b * m, k), dtype=dtype, device=device)
    w = torch.randn((b, n, k), dtype=dtype, device=device)
    group_lens = torch.full((b,), m, dtype=torch.int64, device=device)

    def fn():
        _ = turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)

    avg_ms = start.elapsed_time(end) / iters
    flops = 2.0 * b * m * n * k
    tflops = flops / (avg_ms * 1e-3) / 1e12
    return avg_ms, tflops


def main():
    parser = argparse.ArgumentParser(description="Standalone grouped GEMM expert-count sweep")
    parser.add_argument(
        "--model",
        type=str,
        default="DeepSeek-V3",
        help="Model key in Primus benchmark config, or 'all' for every model.",
    )
    parser.add_argument(
        "--op",
        choices=["gateup", "down", "both"],
        default="gateup",
        help="MoE GEMM projection. Use 'both' to run GateUp and Down.",
    )
    parser.add_argument("--m", type=int, default=512, help="Tokens per expert (M).")
    parser.add_argument("--experts", type=str, default="1,2,4,8,16,32", help="Comma-separated B list.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--csv", type=str, default="", help="Optional output CSV path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is required.")

    bench_cfg, turbo = _import_turbo_and_bench_cfg()
    model_names = (
        list(bench_cfg.MoEModelConfigs.keys())
        if args.model.lower() == "all"
        else [args.model]
    )
    for model_name in model_names:
        if model_name not in bench_cfg.MoEModelConfigs:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(bench_cfg.MoEModelConfigs.keys())}")

    op_names = ["gateup", "down"] if args.op == "both" else [args.op]

    dtype = _dtype_from_name(args.dtype)
    experts = _parse_expert_list(args.experts)

    rows = []
    for model_name in model_names:
        model_cfg = bench_cfg.MoEModelConfigs[model_name]
        hidden = int(model_cfg["hidden_size"])
        ffn = int(model_cfg["moe_intermediate_size"])
        n_routed_experts = int(model_cfg["n_routed_experts"])
        for op_name in op_names:
            if op_name == "gateup":
                n, k = 2 * ffn, hidden
            else:
                n, k = hidden, ffn

            print(
                f"\nModel={model_name}, op={op_name}, M={args.m}, N={n}, K={k}, "
                f"dtype={args.dtype}, experts={experts}, routed_experts={n_routed_experts}"
            )
            for b in experts:
                avg_ms, tflops = bench_one(
                    turbo,
                    b=b,
                    m=args.m,
                    n=n,
                    k=k,
                    dtype=dtype,
                    device=args.device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                print(f"B={b:>3} | avg_ms={avg_ms:.4f} | tflops={tflops:.3f}")
                rows.append(
                    {
                        "model": model_name,
                        "op": op_name,
                        "B": b,
                        "M": args.m,
                        "N": n,
                        "K": k,
                        "dtype": args.dtype,
                        "avg_ms": avg_ms,
                        "tflops": tflops,
                    }
                )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["model", "op", "B", "M", "N", "K", "dtype", "avg_ms", "tflops"],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
