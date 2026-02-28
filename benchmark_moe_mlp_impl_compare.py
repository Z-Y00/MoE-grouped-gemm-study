#!/usr/bin/env python3
"""Compare MoE MLP execution styles for cache/locality studies.

Compares:
1) Normal grouped MoE MLP:
   grouped_gemm -> activation -> grouped_gemm
2) Expert-loop MoE MLP:
   for each expert: gemm -> activation -> gemm

The script can run forward-only or training step (forward + backward).
It auto-scans model-derived cases from Primus-Turbo benchmark config.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile


@dataclass
class CaseConfig:
    case: str
    model: str
    b_local: int
    m_tokens_per_expert: int
    hidden_size: int
    ffn_size: int


def _maybe_compile_fwd(fn, enabled: bool):
    if not enabled or not hasattr(torch, "compile"):
        return fn
    try:
        # Primus custom ops may fail graph capture in some cases; keep benchmark robust.
        compiled = torch.compile(fn, fullgraph=False, dynamic=False)
    except Exception as e:
        print(f"[WARN] torch.compile setup failed; using eager. reason={type(e).__name__}: {e}")
        return fn

    failed_once = False

    def compiled_or_eager():
        nonlocal failed_once
        if failed_once:
            return fn()
        try:
            return compiled()
        except Exception as e:
            failed_once = True
            print(f"[WARN] torch.compile runtime failed; falling back to eager. reason={type(e).__name__}: {e}")
            return fn()

    return compiled_or_eager


def _import_turbo_modules():
    root = Path(__file__).resolve().parent
    turbo_root = root / "Primus-Turbo"
    if str(turbo_root) not in sys.path:
        sys.path.insert(0, str(turbo_root))

    ops_cfg_dir = turbo_root / "benchmark" / "ops"
    if str(ops_cfg_dir) not in sys.path:
        sys.path.insert(0, str(ops_cfg_dir))

    import config as bench_cfg  # type: ignore
    import primus_turbo.pytorch as turbo  # type: ignore

    return bench_cfg, turbo


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def build_cases(bench_cfg, experts_mode: str) -> list[CaseConfig]:
    cases: list[CaseConfig] = []
    for model_name, model_cfg in bench_cfg.MoEModelConfigs.items():
        n_routed_experts = int(model_cfg["n_routed_experts"])
        h = int(model_cfg["hidden_size"])
        f = int(model_cfg["moe_intermediate_size"])
        for m in bench_cfg.GROUPED_GEMM_M_SIZE_LIST:
            if experts_mode == "full":
                b_local = n_routed_experts
            elif experts_mode == "div8":
                b_local = max(1, n_routed_experts // 8)
            else:
                raise ValueError(f"Unsupported experts_mode: {experts_mode}")
            cases.append(
                CaseConfig(
                    case=f"{model_name}-B{b_local}-M{m}",
                    model=model_name,
                    b_local=b_local,
                    m_tokens_per_expert=int(m),
                    hidden_size=h,
                    ffn_size=f,
                )
            )
    return cases


def alloc_inputs(case: CaseConfig, dtype: torch.dtype, device: int, train: bool):
    torch.cuda.set_device(device)
    b, m, h, f = case.b_local, case.m_tokens_per_expert, case.hidden_size, case.ffn_size
    x = torch.randn((b * m, h), device=device, dtype=dtype, requires_grad=train)
    w1 = torch.randn((b, 2 * f, h), device=device, dtype=dtype, requires_grad=train)
    w2 = torch.randn((b, h, f), device=device, dtype=dtype, requires_grad=train)
    group_lens = torch.full((b,), m, device=device, dtype=torch.int64)
    return x, w1, w2, group_lens


def moe_mlp_grouped(turbo, x, w1, w2, group_lens):
    up = turbo.ops.grouped_gemm(x, w1, group_lens, trans_b=True)
    u, v = up.chunk(2, dim=-1)
    inter = F.silu(u) * v
    out = turbo.ops.grouped_gemm(inter, w2, group_lens, trans_b=True)
    return out


def moe_mlp_expert_loop(x, w1, w2, group_lens):
    b = group_lens.numel()
    m = int(group_lens[0].item())
    h = x.shape[-1]
    x3 = x.view(b, m, h)
    out_chunks = []
    for i in range(b):
        xi = x3[i]  # [M, H]
        up_i = xi @ w1[i].transpose(0, 1)  # [M, 2F]
        u, v = up_i.chunk(2, dim=-1)
        inter = F.silu(u) * v
        out_i = inter @ w2[i].transpose(0, 1)  # [M, H]
        out_chunks.append(out_i)
    return torch.cat(out_chunks, dim=0)


def _estimate_flops(case: CaseConfig, train: bool, swiglu_fwd_flops: float, swiglu_bwd_flops: float) -> float:
    b, m, h, f = case.b_local, case.m_tokens_per_expert, case.hidden_size, case.ffn_size
    gateup_fwd = 2.0 * b * m * h * (2 * f)
    down_fwd = 2.0 * b * m * f * h
    act_fwd = swiglu_fwd_flops * b * m * f
    if not train:
        return gateup_fwd + down_fwd + act_fwd

    act_bwd = swiglu_bwd_flops * b * m * f
    return (3.0 * gateup_fwd) + (3.0 * down_fwd) + act_fwd + act_bwd


def run_one(
    turbo,
    case: CaseConfig,
    impl_name: str,
    device: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    train: bool,
    torch_compile_fwd: bool,
    swiglu_fwd_flops: float,
    swiglu_bwd_flops: float,
):
    x, w1, w2, group_lens = alloc_inputs(case, dtype, device, train=train)

    if impl_name == "grouped":
        def impl_fn():
            return moe_mlp_grouped(turbo, x, w1, w2, group_lens)
    elif impl_name == "expert_loop":
        def impl_fn():
            return moe_mlp_expert_loop(x, w1, w2, group_lens)
    else:
        raise ValueError(f"Unsupported impl_name: {impl_name}")

    fn = impl_fn
    # Compile forward path only; backward benchmarking keeps eager behavior.
    if not train:
        fn = _maybe_compile_fwd(impl_fn, enabled=torch_compile_fwd)

    if train:
        out_init = fn()
        grad_out = torch.randn_like(out_init)
        out_init.backward(grad_out)
        x.grad = None
        w1.grad = None
        w2.grad = None

        def step():
            out = fn()
            out.backward(grad_out)
            x.grad = None
            w1.grad = None
            w2.grad = None
    else:

        def step():
            _ = fn()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.cuda.synchronize(device)

    avg_ms = start.elapsed_time(end) / iters
    flops = _estimate_flops(case, train, swiglu_fwd_flops, swiglu_bwd_flops)
    tflops = flops / (avg_ms * 1e-3) / 1e12
    return avg_ms, tflops


def profile_one(
    turbo,
    case: CaseConfig,
    impl_name: str,
    device: int,
    dtype: torch.dtype,
    profile_steps: int,
    torch_compile_fwd: bool,
    row_limit: int,
) -> str:
    x, w1, w2, group_lens = alloc_inputs(case, dtype, device, train=False)

    if impl_name == "grouped":
        def impl_fn():
            return moe_mlp_grouped(turbo, x, w1, w2, group_lens)
    elif impl_name == "expert_loop":
        def impl_fn():
            return moe_mlp_expert_loop(x, w1, w2, group_lens)
    else:
        raise ValueError(f"Unsupported impl_name: {impl_name}")

    fn = _maybe_compile_fwd(impl_fn, enabled=torch_compile_fwd)
    # Warm once before profile to avoid one-time setup noise.
    _ = fn()
    torch.cuda.synchronize(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(profile_steps):
            _ = fn()
            torch.cuda.synchronize(device)
            prof.step()

    return prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=row_limit,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark grouped vs expert-loop MoE MLP implementations")
    parser.add_argument("--device", type=int, default=0, help="CUDA/HIP device id.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=["fwd", "train"],
        default="fwd",
        help="fwd: forward only, train: forward+backward.",
    )
    parser.add_argument(
        "--experts-mode",
        choices=["full", "div8"],
        default="div8",
        help="full: b_local=n_routed_experts, div8: b_local=n_routed_experts//8",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases (0 means all).")
    parser.add_argument("--csv", type=str, default="", help="Optional output CSV.")
    parser.add_argument("--swiglu-fwd-flops-per-element", type=float, default=5.0)
    parser.add_argument("--swiglu-bwd-flops-per-element", type=float, default=8.0)
    parser.add_argument(
        "--torch-compile-fwd",
        dest="torch_compile_fwd",
        action="store_true",
        default=True,
        help="Use torch.compile for forward benchmarks (default: enabled).",
    )
    parser.add_argument(
        "--no-torch-compile-fwd",
        dest="torch_compile_fwd",
        action="store_false",
        help="Disable torch.compile for forward benchmarks.",
    )
    parser.add_argument(
        "--check-close",
        dest="check_close",
        action="store_true",
        default=True,
        help="Check grouped and loop outputs are close (default: enabled).",
    )
    parser.add_argument(
        "--no-check-close",
        dest="check_close",
        action="store_false",
        help="Disable grouped-vs-loop numerical closeness check.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Collect one-case torch profiler report for grouped and expert_loop.",
    )
    parser.add_argument(
        "--profile-case-index",
        type=int,
        default=1,
        help="1-based case index to profile when --profile is enabled.",
    )
    parser.add_argument("--profile-steps", type=int, default=20, help="Profiler iterations per impl.")
    parser.add_argument("--profile-row-limit", type=int, default=20, help="Rows in profiler table output.")
    parser.add_argument(
        "--profile-output-prefix",
        type=str,
        default="",
        help="Optional output prefix for profiler text files.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is required.")

    bench_cfg, turbo = _import_turbo_modules()
    dtype = _dtype_from_name(args.dtype)
    train = args.mode == "train"
    cases = build_cases(bench_cfg, experts_mode=args.experts_mode)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    rows = []
    ts = time.time()
    for i, case in enumerate(cases, start=1):
        grouped_ms, grouped_tflops = run_one(
            turbo=turbo,
            case=case,
            impl_name="grouped",
            device=args.device,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            train=train,
            torch_compile_fwd=args.torch_compile_fwd,
            swiglu_fwd_flops=args.swiglu_fwd_flops_per_element,
            swiglu_bwd_flops=args.swiglu_bwd_flops_per_element,
        )
        loop_ms, loop_tflops = run_one(
            turbo=turbo,
            case=case,
            impl_name="expert_loop",
            device=args.device,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            train=train,
            torch_compile_fwd=args.torch_compile_fwd,
            swiglu_fwd_flops=args.swiglu_fwd_flops_per_element,
            swiglu_bwd_flops=args.swiglu_bwd_flops_per_element,
        )

        if args.check_close:
            x, w1, w2, group_lens = alloc_inputs(case, dtype, args.device, train=False)
            y_grouped = moe_mlp_grouped(turbo, x, w1, w2, group_lens)
            y_loop = moe_mlp_expert_loop(x, w1, w2, group_lens)
            close = torch.allclose(y_grouped, y_loop, rtol=1e-2, atol=1e-2)
        else:
            close = True

        speedup = loop_tflops / grouped_tflops if grouped_tflops > 0 else float("nan")
        print(
            f"[{i:>3}/{len(cases)}] {case.case:<26} "
            f"grouped={grouped_tflops:.3f} TF/s, loop={loop_tflops:.3f} TF/s, "
            f"loop_over_grouped={speedup:.3f}, close={close}"
        )

        rows.append(
            {
                "timestamp_s": ts,
                "mode": args.mode,
                "case": case.case,
                "model": case.model,
                "B_local": case.b_local,
                "M": case.m_tokens_per_expert,
                "H": case.hidden_size,
                "F": case.ffn_size,
                "dtype": args.dtype,
                "grouped_avg_ms": grouped_ms,
                "grouped_tflops": grouped_tflops,
                "loop_avg_ms": loop_ms,
                "loop_tflops": loop_tflops,
                "speedup_loop_over_grouped": speedup,
                "allclose": close,
            }
        )

        if args.profile and i == args.profile_case_index:
            grouped_prof = profile_one(
                turbo=turbo,
                case=case,
                impl_name="grouped",
                device=args.device,
                dtype=dtype,
                profile_steps=args.profile_steps,
                torch_compile_fwd=args.torch_compile_fwd,
                row_limit=args.profile_row_limit,
            )
            loop_prof = profile_one(
                turbo=turbo,
                case=case,
                impl_name="expert_loop",
                device=args.device,
                dtype=dtype,
                profile_steps=args.profile_steps,
                torch_compile_fwd=args.torch_compile_fwd,
                row_limit=args.profile_row_limit,
            )
            print(f"\n[PROFILE] case={case.case} impl=grouped\n{grouped_prof}")
            print(f"\n[PROFILE] case={case.case} impl=expert_loop\n{loop_prof}")

            if args.profile_output_prefix:
                grouped_path = f"{args.profile_output_prefix}.{case.case}.grouped.txt"
                loop_path = f"{args.profile_output_prefix}.{case.case}.expert_loop.txt"
                with open(grouped_path, "w", encoding="utf-8") as f:
                    f.write(grouped_prof)
                with open(loop_path, "w", encoding="utf-8") as f:
                    f.write(loop_prof)
                print(f"[PROFILE] Saved: {grouped_path}")
                print(f"[PROFILE] Saved: {loop_path}")

    if args.csv:
        keys = [
            "timestamp_s",
            "mode",
            "case",
            "model",
            "B_local",
            "M",
            "H",
            "F",
            "dtype",
            "grouped_avg_ms",
            "grouped_tflops",
            "loop_avg_ms",
            "loop_tflops",
            "speedup_loop_over_grouped",
            "allclose",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
