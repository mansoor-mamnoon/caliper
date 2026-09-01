# The reference kernel corpus

`python/caliper/corpus/kernels/` pairs a Triton implementation with a vendor
baseline for a handful of workloads every kernel author recognizes: `gemm`,
`rmsnorm`, `softmax` today (`attention_fwd` / `attention_bwd` land later). Each
one is a `corpus:` target on its own — `corpus:gemm`, `corpus:rmsnorm`,
`corpus:softmax`.

`corpus:gemm` also plugs into `sweep()` (its `{M, N, K}` shape is one the sweep
spec grammar already understands, and its autotune configs feed the config
cache). `rmsnorm` and `softmax` are `{ROWS, COLS}` workloads, which the spec
grammar doesn't model yet, so today they run only through a direct
`kernel.run(cell, config)` call; wiring them into `sweep()` waits on an
elementwise shape kind in `crates/caliper-core/src/spec.rs`.

## How a corpus kernel actually runs

The oracle kernels (`corpus:o1`-`corpus:o6`, CUDA C++) go through the Rust
device layer: `CudaLauncher`, `NvmlClock`, and the rest of
`crates/caliper-gpu`. That layer's on-device ports are still stubs (they run
from recorded fixtures today; the real CUDA/NVML bindings are Colab work).

The corpus kernels don't wait on that. They're Triton, which means Python, a
CUDA device, and nothing else — so `caliper.corpus.kernels.gemm.run()` (and
`rmsnorm`, `softmax`) time themselves directly with
[`caliper.live_timing_ms`](../python/caliper/api.py), the same CUDA-event loop
`do_bench()` uses: a handful of warm-up launches, an L2-flushing buffer zeroed
between reps, and a `torch.cuda.Event` pair around every timed launch. So the
corpus runs without waiting on the still-unfinished launcher; on-device
verification is on Colab (`docs/plan.md` §0.5), same as the rest of the GPU
tiers.

The trade a caller should know about: this path skips the Rust reduction
pipeline entirely — no clock lock, no L2-flush *accounting* (the buffer zero
still happens; caliper just doesn't verify or report on it), no throttle
detection, no steady-state trim. Every record it produces is flagged
`corpus-live-timing` (and `clocks-unlocked`) in `flags` so a reader can tell
the difference from a `bench()`-produced record at a glance.

The machine fingerprint is built the same way, straight from PyTorch device
introspection (`torch.cuda.get_device_properties` / `get_device_capability` /
`torch.version.cuda`) plus `caliper.toolchain()` — no NVML, so it doesn't need
the launcher either.

## The Triton pin

Every kernel module carries `caliper.corpus._common.TRITON_PIN`: the
`triton-lang/triton` commit its kernel-authoring API (`@triton.jit`,
`tl.load`/`tl.store` masking, `@triton.autotune`) was written and last checked
against. This is **not** a claim that any kernel body here is copied from that
commit — every kernel in this corpus is caliper's own implementation. It's an
API-compatibility marker: "this is the Triton this was known to work with,"
so a future API change has somewhere to point back to.

## The source hash

`kernel.source_hash` (`sha256:<hex>`) is the SHA-256 of the kernel module's
own `.py` file, computed once at import (`content_hash(__file__)`). It changes
the moment the kernel's source changes, so two records with the same hash ran
byte-identical code — the whole point of pinning a kernel for comparison
across machines or over time.

## `gemm` — `corpus:gemm`

`(M, K) x (K, N) -> (M, N)`, block-tiled with grouped column ordering for L2
reuse across the swept `N` tiles (the standard shape for a matmul kernel: see
e.g. the Triton project's own matmul tutorial, which this kernel's tiling
follows in spirit, not in source). `CONFIGS` is 5 `(BLOCK_M, BLOCK_N,
BLOCK_K, GROUP_M)` tilings; `gemm.kernel` is `@triton.autotune`-wrapped over
them and exposes the standard `.configs` (a list of `triton.Config`, each with
`.kwargs`) that `sweep(autotune="from_kernel")` reads.

Baseline: `torch.matmul` (cuBLAS).

Roofline: `flops = 2*M*N*K`; `bytes_hbm = (M*K + K*N + M*N) * dtype_bytes` —
one read of each operand tile plus the output write, the standard dense-GEMM
count (`crates/caliper-core/src/roofline.rs::corpus_spec`, the `gemm` arm).

## `rmsnorm` — `corpus:rmsnorm`

`y = x / sqrt(mean(x**2, axis=-1) + eps) * weight`, one Triton program per
row. Baseline: a plain-torch reference (square, mean, `rsqrt`, multiply) —
deliberately not `F.rms_norm`, so it runs on any PyTorch version rather than
needing whatever release added that op.

Roofline (`corpus_spec`'s `rmsnorm` arm): per element, square + reduce-sum +
normalize-multiply + weight-multiply is 4 FLOPs (the row-wise mean/rsqrt is
`O(rows)`, negligible at any real `cols`) — `flops = 4*ROWS*COLS`. HBM traffic
is dominated by reading `x` and writing `y`: `bytes_hbm = 2*ROWS*COLS *
dtype_bytes` (the `COLS`-sized weight vector is negligible at any real `ROWS`
and isn't counted). Memory-bound at any realistic shape.

## `softmax` — `corpus:softmax`

`y = exp(x - max(x, axis=-1)) / sum(exp(x - max(x, axis=-1)), axis=-1)`, one
Triton program per row. Baseline: `torch.softmax`.

Roofline (`corpus_spec`'s `softmax` arm): per element, row-max + subtract +
`exp` + reduce-sum + divide is 5 FLOPs (`exp` counted as one op, matching
common practice) — `flops = 5*ROWS*COLS`; `bytes_hbm = 2*ROWS*COLS *
dtype_bytes`, the same read-`x`-write-`y` accounting as `rmsnorm`. Memory-bound
at any realistic shape.

## Running the corpus

Needs `pip install 'caliper-gpu[triton]'` (installs `torch` + `triton`) and a
CUDA device — none of this runs on the Mac this library is developed on;
verify on a Colab GPU runtime:

```python
from caliper.corpus.kernels import gemm

result = gemm.run({"shape": {"m": 4096, "n": 4096, "k": 4096}, "dtype": "bf16", "layout": "row"})
print(result.validate())          # [] if the record is well-formed
print(result.roofline.roofline_pct, result.roofline.baseline_pct)
```

Or through a sweep, which is what exercises `gemm`'s autotune cache end to
end — see `tests/l6_e2e/test_autotune_cache.py`.
