git clone --recurse-submodules https://github.com/Z-Y00/MoE-grouped-gemm-study.git

```
cd MoE-grouped-gemm-study
cd Primus-Turbo

pip3 install -r requirements.txt
pip3 install --no-build-isolation .

# (Optional) Set GPU_ARCHS environment variable to specify target AMD GPU architectures.
GPU_ARCHS="gfx942;gfx950" pip3 install --no-build-isolation .
cd -
python benchmark_moe_mlp_impl_compare.py
```


```
rocprofv3 \
  --kernel-trace \
  --pmc TCC_HIT,TCC_MISS,TCP_TOTAL_CACHE_ACCESSES,TCP_CACHE_MISS \
  --output-format csv json \
  --output-directory cache_profile_dsv3_b32 \
  --output-file grouped_gemm_cache \
  -- python bench_grouped_gemm_expert_sweep.py \
    --model DeepSeek-V3 --op gateup --m 512 --experts 32 \
    --dtype bf16 --warmup 10 --iters 50

rocprof-compute profile \
  --name grouped_gemm_cache_dsv3_b32 \
  --path "/workspace/MoE-grouped-gemm-study/rocprof_compute" \
  --kernel GroupedGemmKernel \
  --block 16 17 \
  --no-roof \
  --format-rocprof-output csv \
  -- python bench_grouped_gemm_expert_sweep.py \
    --model DeepSeek-V3 --op gateup --m 512 --experts 32 \
    --dtype bf16 --warmup 10 --iters 50
```


python run_hipblaslt_bench_topk.py --topk 30 --output "hipblaslt_vs_primus_smoke.csv"