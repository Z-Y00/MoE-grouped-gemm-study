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
