# Fingerprint checklist

Every field in the result's `machine` block, where it comes from, and how to
read it back by hand. Run the dry-run at the bottom on a Colab A100 (or any
CUDA host) to confirm the recorded fingerprint matches the machine.

The field set is defined once in `crates/caliper-core/src/schema.rs` (`Machine`
/ `Toolkit`) and its completeness rules in
`crates/caliper-core/src/fingerprint.rs`.

## Hardware / driver fields

| Field | Tier | Source of truth | How to read it by hand |
|---|---|---|---|
| `gpu_name` | required | `nvmlDeviceGetName` | `nvidia-smi --query-gpu=name --format=csv,noheader` |
| `sm_arch` | required | `nvmlDeviceGetCudaComputeCapability` -> `sm_<major><minor>` | `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` (e.g. `8.0` -> `sm_80`) |
| `vram_mib` | required | `nvmlDeviceGetMemoryInfo.total` / 2^20 | `nvidia-smi --query-gpu=memory.total --format=csv,noheader` |
| `sm_count` | required | `cudaDeviceProp.multiProcessorCount` | `nvidia-smi -q` -> not listed; `deviceQuery` "Multiprocessors" |
| `l2_bytes` | required | `cudaDeviceProp.l2CacheSize` | `deviceQuery` "L2 Cache Size" |
| `bar1_mib` | required | `nvmlDeviceGetBAR1MemoryInfo.bar1Total` / 2^20 | `nvidia-smi -q` -> `BAR1 Memory Usage` -> `Total` |
| `driver` | required | `nvmlSystemGetDriverVersion` | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| `cuda_driver` | required | `nvmlSystemGetCudaDriverVersion_v2` -> `maj.min` | `nvidia-smi -q` -> `CUDA Version` (top-right of `nvidia-smi`) |
| `cuda_runtime` | required | `cudaRuntimeGetVersion` -> `maj.min` (or PyTorch's bundled CUDA) | `nvcc --version` -> `release 12.x` |
| `nvml_version` | required | `nvmlSystemGetNVMLVersion` | `nvidia-smi -q` -> not listed; `python -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlSystemGetNVMLVersion())"` |
| `ecc` | required | `nvmlDeviceGetEccMode.current` (bool) | `nvidia-smi -q` -> `Ecc Mode` -> `Current` |
| `mig` | required | `nvmlDeviceGetMigMode` -> `"disabled"` or the instance geometry | `nvidia-smi -q` -> `MIG Mode` -> `Current` |
| `persistence_mode` | required | `nvmlDeviceGetPersistenceMode` (bool) | `nvidia-smi --query-gpu=persistence_mode --format=csv,noheader` |
| `pcie_gen` | required | `nvmlDeviceGetCurrPcieLinkGeneration` | `nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader` |
| `pcie_width` | required | `nvmlDeviceGetCurrPcieLinkWidth` | `nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader` |

## Toolchain fields (`machine.toolkit`)

| Field | Tier | Source of truth | How to read it by hand |
|---|---|---|---|
| `toolkit.nvcc` | required | `nvcc --version`, parsed by `fingerprint::parse_nvcc_version` | `nvcc --version` -> `V12.4.131` |
| `toolkit.ptxas` | required | `ptxas --version`, parsed by `fingerprint::parse_ptxas_version` | `ptxas --version` -> `V12.4.131` |
| `toolkit.torch` | recommended | `importlib.metadata.version("torch")` | `pip show torch` -> `Version` |
| `toolkit.triton` | recommended | `importlib.metadata.version("triton")` | `pip show triton` -> `Version` |

A **required** gap makes `fingerprint::check` report `complete: false` and
`caliper fingerprint` exit non-zero. A **recommended** gap is reported but not an
error: a pure CUDA-C host has no Triton or PyTorch.

## Dry-run (Colab A100)

1. Record a session and print the fingerprint:

   ```bash
   CALIPER_GPU_PORTS=record CALIPER_GPU_FIXTURE=/tmp/fp.jsonl \
     python -m caliper fingerprint --json | tee /tmp/fp.json
   ```

2. Confirm it is complete (exit 0; exit 1 lists any missing required field):

   ```bash
   python -m caliper fingerprint --check
   # or, from Python, against the same recording:
   python -c "from caliper import api; print(api.fingerprint_check(recording=open('/tmp/fp.jsonl').read()))"
   # -> {'complete': True, 'missing_required': [], 'missing_recommended': [...]}
   ```

3. Cross-check every **required** row above against its `nvidia-smi` /
   `deviceQuery` / `nvcc --version` reading. In particular:
   - `sm_arch` — `compute_cap` `8.0` must render as `sm_80`.
   - `cuda_runtime` vs `cuda_driver` — the runtime (from `nvcc` / torch) may lag
     the driver; both are recorded, neither is derived from the other.
   - `toolkit.nvcc` / `toolkit.ptxas` — the `V<maj.min.build>` triple, not the
     `release <maj.min>` pair, when the build number is present.

4. Compare `caliper.api.toolchain()` to `pip show triton torch` and
   `nvcc --version`; the strings must match exactly.
