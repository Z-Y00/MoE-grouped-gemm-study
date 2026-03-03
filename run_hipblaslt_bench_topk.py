#!/usr/bin/env python3
"""Run hipblaslt-bench for top-K Primus grouped GEMM cases.

This script uses Primus benchmark configs to build grouped GEMM cases, then runs
hipblaslt-bench in strided-batched matmul mode with explicit leading dimensions
and strides for A/B/C/D.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import torch


def load_cases():
    ops_dir = Path(__file__).resolve().parent / "Primus-Turbo" / "benchmark" / "ops"
    if str(ops_dir) not in sys.path:
        sys.path.insert(0, str(ops_dir))
    import config as bench_cfg  # type: ignore

    cases = bench_cfg.gen_grouped_gemm_test_cases()
    return sorted(cases, key=lambda x: (x["Case"], x["B"], x["M"], x["N"], x["K"]))


def load_turbo():
    import primus_turbo.pytorch as turbo  # type: ignore

    return turbo


def parse_perf_line(output_text: str):
    for line in output_text.splitlines():
        s = line.strip()
        if s.startswith("N,T,0,"):
            cols = [x.strip() for x in s.split(",")]
            if len(cols) < 3:
                return None
            try:
                gflops = float(cols[-3])
                gbps = float(cols[-2])
                us = float(cols[-1])
                return gflops, gbps, us, s
            except ValueError:
                return None
    return None


def run_hipblaslt_case(case: dict, device: int, iters: int, cold_iters: int, explicit_stride: bool):
    b = int(case["B"])
    m = int(case["M"])
    n = int(case["N"])
    k = int(case["K"])
    case_name = str(case["Case"])
    model, op = case_name.rsplit("-", 1)

    # Explicit leading dimensions and strided-batched strides.
    # For transA=N, transB=T in hipblaslt-bench:
    # A: [M, K], lda=M, stride_a=M*K
    # B: [N, K] (because transB=T), ldb=N, stride_b=N*K
    # C/D: [M, N], ldc=ldd=M, stride_c/d=M*N
    lda = m
    ldb = n
    ldc = m
    ldd = m
    stride_a = m * k
    stride_b = n * k
    stride_c = m * n
    stride_d = m * n

    cmd = [
        "hipblaslt-bench",
        "--function",
        "matmul",
        "--precision",
        "bf16_r",
        "--compute_type",
        "f32_r",
        "--transA",
        "N",
        "--transB",
        "T",
        "-m",
        str(m),
        "-n",
        str(n),
        "-k",
        str(k),
        "--batch_count",
        str(b),
        "--iters",
        str(iters),
        "--cold_iters",
        str(cold_iters),
        "--algo_method",
        "heuristic",
        "--requested_solution",
        "1",
        "--use_gpu_timer",
        "--device",
        str(device),
    ]
    if explicit_stride:
        cmd.extend(
            [
                "--lda",
                str(lda),
                "--ldb",
                str(ldb),
                "--ldc",
                str(ldc),
                "--ldd",
                str(ldd),
                "--stride_a",
                str(stride_a),
                "--stride_b",
                str(stride_b),
                "--stride_c",
                str(stride_c),
                "--stride_d",
                str(stride_d),
            ]
        )

    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    perf = parse_perf_line(out)

    row = {
        "model": model,
        "op": op,
        "B": b,
        "M": m,
        "N": n,
        "K": k,
        "lda": lda,
        "ldb": ldb,
        "ldc": ldc,
        "ldd": ldd,
        "stride_a": stride_a,
        "stride_b": stride_b,
        "stride_c": stride_c,
        "stride_d": stride_d,
        "status": "ok",
        "hipblaslt_gflops": "",
        "hipblaslt_tflops": "",
        "hipblaslt_gbps": "",
        "kernel_us": "",
        "error": "",
    }

    if perf is None:
        row["status"] = "fail"
        row["error"] = "NO solution found" if "NO solution found" in out else f"no_perf_line_exit_{p.returncode}"
    else:
        gflops, gbps, us, _ = perf
        row["hipblaslt_gflops"] = f"{gflops:.6f}"
        row["hipblaslt_tflops"] = f"{gflops / 1000.0:.6f}"
        row["hipblaslt_gbps"] = f"{gbps:.6f}"
        row["kernel_us"] = f"{us:.6f}"

    return row


def run_case(case: dict, device: int, iters: int, cold_iters: int):
    strided = run_hipblaslt_case(
        case, device=device, iters=iters, cold_iters=cold_iters, explicit_stride=True
    )
    non_strided = run_hipblaslt_case(
        case, device=device, iters=iters, cold_iters=cold_iters, explicit_stride=False
    )

    # Keep existing columns for explicit-stride path for backward compatibility.
    row = dict(strided)
    row["hipblaslt_non_strided_status"] = non_strided["status"]
    row["hipblaslt_non_strided_tflops"] = non_strided["hipblaslt_tflops"]
    row["hipblaslt_non_strided_kernel_us"] = non_strided["kernel_us"]
    row["hipblaslt_non_strided_error"] = non_strided["error"]

    if strided["hipblaslt_tflops"] and non_strided["hipblaslt_tflops"]:
        s = float(strided["hipblaslt_tflops"])
        ns = float(non_strided["hipblaslt_tflops"])
        row["hipblaslt_non_strided_vs_strided"] = f"{ns / s:.6f}" if s > 0 else ""
    else:
        row["hipblaslt_non_strided_vs_strided"] = ""

    return row


def run_primus_case(turbo, case: dict, device: int, iters: int, warmup: int):
    b = int(case["B"])
    m = int(case["M"])
    n = int(case["N"])
    k = int(case["K"])

    torch.cuda.set_device(device)
    x = torch.randn((b * m, k), dtype=torch.bfloat16, device=device)
    w = torch.randn((b, n, k), dtype=torch.bfloat16, device=device)
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
    ap = argparse.ArgumentParser(description="Run top-K Primus grouped GEMM cases with hipblaslt-bench.")
    ap.add_argument("--topk", type=int, default=10, help="Number of cases to run (default: 10).")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cold-iters", type=int, default=5)
    ap.add_argument(
        "--compare-primus",
        action="store_true",
        default=True,
        help="Also run Primus-Turbo grouped_gemm for side-by-side comparison (default: on).",
    )
    ap.add_argument(
        "--no-compare-primus",
        dest="compare_primus",
        action="store_false",
        help="Disable Primus-Turbo side-by-side run.",
    )
    ap.add_argument(
        "--primus-warmup",
        type=int,
        default=-1,
        help="Warmup iterations for Primus benchmark; default uses --cold-iters.",
    )
    ap.add_argument(
        "--primus-iters",
        type=int,
        default=-1,
        help="Timing iterations for Primus benchmark; default uses --iters.",
    )
    ap.add_argument("--output", type=str, default="hipblaslt_bench_top10.csv")
    args = ap.parse_args()

    all_cases = load_cases()
    cases = all_cases[: args.topk]
    turbo = load_turbo() if args.compare_primus else None

    primus_warmup = args.cold_iters if args.primus_warmup < 0 else args.primus_warmup
    primus_iters = args.iters if args.primus_iters < 0 else args.primus_iters

    rows = []
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        row = run_case(case, device=args.device, iters=args.iters, cold_iters=args.cold_iters)
        if turbo is not None:
            primus_ms, primus_tflops = run_primus_case(
                turbo, case, device=args.device, iters=primus_iters, warmup=primus_warmup
            )
            row["primus_avg_ms"] = f"{primus_ms:.6f}"
            row["primus_tflops"] = f"{primus_tflops:.6f}"
        else:
            row["primus_avg_ms"] = ""
            row["primus_tflops"] = ""

        hb_tflops = (
            float(row["hipblaslt_non_strided_tflops"]) if row["hipblaslt_non_strided_tflops"] else None
        )
        pr_tflops = float(row["primus_tflops"]) if row["primus_tflops"] else None
        if hb_tflops is not None and pr_tflops is not None and hb_tflops > 0:
            row["primus_vs_hipblaslt"] = f"{pr_tflops / hb_tflops:.6f}"
        else:
            row["primus_vs_hipblaslt"] = ""

        rows.append(row)
        elapsed_s = time.time() - t0
        if row["status"] == "ok":
            msg = (
                f"[{i}/{len(cases)}] {row['model']}-{row['op']} "
                f"B={row['B']} M={row['M']} N={row['N']} K={row['K']} "
                f"hipblaslt_tflops={row['hipblaslt_tflops']} "
                f"hipblaslt_non_strided_tflops={row['hipblaslt_non_strided_tflops'] or 'NA'} "
                f"non_strided_vs_strided={row['hipblaslt_non_strided_vs_strided'] or 'NA'} "
                f"us={row['kernel_us']} "
                f"primus_tflops={row['primus_tflops'] or 'NA'} "
                f"primus_vs_hipblaslt={row['primus_vs_hipblaslt'] or 'NA'} "
                f"elapsed_s={elapsed_s:.2f} "
                f"strides(a,b,c,d)=({row['stride_a']},{row['stride_b']},{row['stride_c']},{row['stride_d']})"
            )
        else:
            msg = (
                f"[{i}/{len(cases)}] {row['model']}-{row['op']} "
                f"B={row['B']} M={row['M']} N={row['N']} K={row['K']} "
                f"status=fail error={row['error']} "
                f"primus_tflops={row['primus_tflops'] or 'NA'} "
                f"elapsed_s={elapsed_s:.2f} "
                f"strides(a,b,c,d)=({row['stride_a']},{row['stride_b']},{row['stride_c']},{row['stride_d']})"
            )
        print(msg, flush=True)

    out_path = Path(args.output).resolve()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "op",
                "B",
                "M",
                "N",
                "K",
                "lda",
                "ldb",
                "ldc",
                "ldd",
                "stride_a",
                "stride_b",
                "stride_c",
                "stride_d",
                "status",
                "hipblaslt_gflops",
                "hipblaslt_tflops",
                "hipblaslt_gbps",
                "kernel_us",
                "primus_avg_ms",
                "primus_tflops",
                "primus_vs_hipblaslt",
                "hipblaslt_non_strided_status",
                "hipblaslt_non_strided_tflops",
                "hipblaslt_non_strided_kernel_us",
                "hipblaslt_non_strided_error",
                "hipblaslt_non_strided_vs_strided",
                "error",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nSaved CSV: {out_path}", flush=True)
    print(f"Completed: {len(rows)} cases, ok={ok}, fail={len(rows) - ok}", flush=True)


if __name__ == "__main__":
    main()
