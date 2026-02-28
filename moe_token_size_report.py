#!/usr/bin/env python3
"""Report per-token, per-expert MoE training sizes and GEMM MNK dims.

Reads model configs from benchmark/ops/config.py and prints a table for:
  - GateUP input/output
  - Down input/output

Input size includes expert weights:
  GateUP input = token(h) + weight(2f x h)
  Down   input = token(f) + weight(h x f)
"""

import argparse
import sys
from pathlib import Path


def _import_benchmark_config():
    # Allow running this script from workspace root.
    ops_dir = Path(__file__).resolve().parent / "Primus-Turbo" / "benchmark" / "ops"
    sys.path.insert(0, str(ops_dir))
    from config import GROUPED_GEMM_M_SIZE_LIST, MoEModelConfigs  # type: ignore

    return MoEModelConfigs, GROUPED_GEMM_M_SIZE_LIST


def to_mb(num_elements: int, bytes_per_element: int) -> float:
    # Decimal MB to match previous benchmark summaries.
    return (num_elements * bytes_per_element) / 1e6


def compute_size_range(hidden_size: int, moe_intermediate_size: int, m_values: list[int]):
    h = hidden_size
    f = moe_intermediate_size
    min_m = min(m_values)
    max_m = max(m_values)

    def _sizes_for_m(m: int):
        # Per-expert batch activations (M tokens assigned to one expert).
        gate_in_elems = m * h
        gate_out_elems = m * (2 * f)
        down_in_elems = m * f
        down_out_elems = m * h

        # Input MB always includes expert weights.
        gate_in_elems += 2 * f * h  # [2f, h]
        down_in_elems += h * f  # [h, f]
        return gate_in_elems, gate_out_elems, down_in_elems, down_out_elems

    min_sizes = _sizes_for_m(min_m)
    max_sizes = _sizes_for_m(max_m)
    return min_sizes, max_sizes


def compute_training_state_sizes(hidden_size: int, moe_intermediate_size: int):
    # Per-expert parameter tensors:
    # GateUP weight [2f, h], Down weight [h, f]
    h = hidden_size
    f = moe_intermediate_size
    params = (2 * f * h) + (h * f)
    grads = params
    adam_states = 2 * params
    total = params + grads + adam_states
    return params, grads, adam_states, total


def format_mb_range(min_elems: int, max_elems: int, bytes_per_element: int) -> str:
    min_mb = to_mb(min_elems, bytes_per_element)
    max_mb = to_mb(max_elems, bytes_per_element)
    return f"{min_mb:.6f}-{max_mb:.6f}"


def format_gemm_dims(hidden_size: int, moe_intermediate_size: int, m_values: list[int]):
    # Primus-Turbo grouped benchmark M sweep convention:
    #   GateUP: [M, K] x [N, K]^T with K=h, N=2f
    #   Down:   [M, K] x [N, K]^T with K=f, N=h
    h = hidden_size
    f = moe_intermediate_size
    m_str = ",".join(str(m) for m in m_values)
    gateup_mnk = f"M={{{m_str}}},N={2 * f},K={h}"
    down_mnk = f"M={{{m_str}}},N={h},K={f}"
    return gateup_mnk, down_mnk


def print_size_report(model_configs, m_values: list[int], bytes_per_element: int):
    header = (
        "Model | GateUP GEMM(M,N,K) | Down GEMM(M,N,K) | "
        "GateUP Input MB [M_min-M_max] | GateUP Output MB [M_min-M_max] | "
        "Down Input MB [M_min-M_max] | Down Output MB [M_min-M_max] | "
        "TrainState MB (params+grads+adam)"
    )
    print(header)
    print("-" * len(header))

    for model_name, cfg in model_configs.items():
        h = cfg["hidden_size"]
        f = cfg["moe_intermediate_size"]
        (g_in_min, g_out_min, d_in_min, d_out_min), (g_in_max, g_out_max, d_in_max, d_out_max) = (
            compute_size_range(h, f, m_values)
        )
        _, _, _, train_total = compute_training_state_sizes(h, f)
        gateup_mnk, down_mnk = format_gemm_dims(h, f, m_values)

        print(
            f"{model_name} | "
            f"{gateup_mnk} | "
            f"{down_mnk} | "
            f"{format_mb_range(g_in_min, g_in_max, bytes_per_element)} | "
            f"{format_mb_range(g_out_min, g_out_max, bytes_per_element)} | "
            f"{format_mb_range(d_in_min, d_in_max, bytes_per_element)} | "
            f"{format_mb_range(d_out_min, d_out_max, bytes_per_element)} | "
            f"{to_mb(train_total, bytes_per_element):.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Per-token per-expert MoE size report")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp8"],
        default="bf16",
        help="Element dtype used for byte-size conversion (default: bf16)",
    )
    args = parser.parse_args()

    model_configs, m_values = _import_benchmark_config()
    bytes_per_element = {"bf16": 2, "fp16": 2, "fp8": 1}[args.dtype]
    print_size_report(model_configs, m_values, bytes_per_element)


if __name__ == "__main__":
    main()
