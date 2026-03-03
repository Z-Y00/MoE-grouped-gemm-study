# ROCprofv3 Thread-Trace Skill (Compute Viewer)

Use this workflow to collect `GroupedGemmKernel` thread-trace data and open it in ROCprof Compute Viewer.

## 1) Prerequisites

- ROCm with `rocprofv3` installed.
- Target benchmark/script runnable on GPU.
- ROCprof Trace Decoder library available.

Check `rocprofv3`:

```bash
rocprofv3 --help
```

## 2) Install ROCprof Trace Decoder (if missing)

If `rocprofv3 --att` fails with a decoder library error, install decoder artifacts locally:

```bash
mkdir -p ".local/rocprof-trace-decoder"
curl -L "https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.tar.gz" -o ".local/rocprof-trace-decoder/decoder.tar.gz"
tar -xzf ".local/rocprof-trace-decoder/decoder.tar.gz" -C ".local/rocprof-trace-decoder"
```

Decoder library path used below:

```bash
/workspace/MoE-grouped-gemm-study/.local/rocprof-trace-decoder/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib
```

## 3) Run thread trace for one kernel dispatch

Example for DeepSeek-V3 GateUp (`B=32`, `M=512`):

```bash
rocprofv3 \
  --att \
  --att-library-path "/workspace/MoE-grouped-gemm-study/.local/rocprof-trace-decoder/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib" \
  --att-gpu-index 0 \
  --kernel-include-regex GroupedGemmKernel \
  --att-consecutive-kernels 1 \
  -d "att_profile_dsv3_b32_gateup" \
  -- \
  python bench_grouped_gemm_expert_sweep.py \
    --model DeepSeek-V3 \
    --op gateup \
    --m 512 \
    --experts 32 \
    --dtype bf16 \
    --warmup 5 \
    --iters 20
```

For Down op, only switch:

```bash
--op down
```

## 4) Output files you should see

In the output directory (`-d`), thread trace typically produces:

- `stats_ui_output_agent_*_dispatch_*.csv`
- `ui_output_agent_*_dispatch_*/*.json` (Compute Viewer input)
- `*_shader_engine_*.att` (raw trace)
- `*_code_object_id_*.out`
- `*_results.db`

## 5) Open in ROCprof Compute Viewer

- Open the generated `ui_output_agent_*_dispatch_*` folder in ROCprof Compute Viewer.
- Use `stats_ui_output_agent_*_dispatch_*.csv` for per-instruction latency/stall summaries.

## 6) Troubleshooting

- **Missing decoder library**
  - Provide `--att-library-path` (or set `ROCPROF_ATT_LIBRARY_PATH`) to where `librocprof-trace-decoder.so` exists.
- **Wave incomplete / cutoff warning**
  - Increase `--att-buffer-size`.
  - Reduce workload (`--iters`, narrower kernel filter, or shorter run).
- **Too many kernels traced**
  - Keep `--kernel-include-regex` tight and use `--att-consecutive-kernels 1`.

## 7) Useful references

- ROCprofiler-SDK thread trace guide:
  - https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#rocprofv3-output-files
- ROCprof Trace Decoder releases:
  - https://github.com/ROCm/rocprof-trace-decoder/releases
