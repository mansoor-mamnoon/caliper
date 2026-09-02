# Your Triton benchmark is probably lying to you

`triton.testing.do_bench` is the number almost every Triton kernel is judged by.
It is a good tool, used -- very often -- in a way that produces a number well
off what the kernel does in a real pipeline. This page shows four ways that
happens, why, and what `caliper` does instead.

> **On the numbers.** The experiment tables below are blank in the repo: this
> box has no GPU. [`data/misleads.csv`](data/misleads.csv) is a committed
> *template* -- the row schema plus a slot for the `nsys` column -- and
> `make writeup-data` fills it on a CUDA host. See
> [Reproduce this](#reproduce-this).

> **On "caliper" in the tables.** `caliper bench` -- the full pipeline that
> batches launches, locks the clock, and records the flush state on the
> `Result` -- is not wired to a live launcher yet. The comparisons here use
> caliper's *shipped* primitives (`caliper.do_bench`, `caliper.live_timing_ms`,
> `caliper._core.steady_state_index`) plus the small helpers in
> [`examples/misleads/_common.py`](../examples/misleads/_common.py). Where a row
> stands in for behaviour that `caliper bench` will add on-device, the text
> says so.

## TL;DR

| Trap | Naive `do_bench` reads… | Because | The fix |
|---|---|---|---|
| **Short kernel** (< 20 µs) | too slow | a CUDA-event pair per single launch; `cudaEventRecord` costs a fixed ~1-3 µs each | time a *batch* of launches between one event pair |
| **Cold clocks** | too fast | a 25 ms warmup lands inside the SM-clock ramp / shared-box settle | generous warmup budget, then trim to the steady-state index |
| **L2-resident data** | too fast | a tight loop with no cache management re-reads warm L2 every rep | clear an L2-sized buffer between reps |
| **Unlocked clocks** | noisy, unreproducible | nothing locks or even records the SM clock | lock the clock, or at least record its spread and label the run |

The first three are measurement bugs you can fix in a loop. The fourth is why a
"3% speedup" from last week doesn't reproduce this week.

---

## 1. Short kernels: the per-launch event tax

`do_bench` (and `caliper.do_bench`, and Triton's own) wraps **each** kernel
launch in a `torch.cuda.Event` pair. `cudaEventRecord` is not free -- it costs a
fixed amount, independent of the kernel, on the order of 1-3 µs per call. For a
kernel that runs in 200 µs that is rounding error. For a 10 µs elementwise
kernel it is a real fraction, and a hand-rolled `fn(); torch.cuda.synchronize()`
loop -- still the most common thing people write -- adds the *full*
host↔device sync latency (tens of µs) to every sample.

The fix is to amortize: launch `N` kernels back-to-back, put one event pair
around the batch, divide by `N`. That is the shape `caliper bench(batch=N)`
measures on-device (a live launcher for it is pending), and it is what `nsys`
sees -- nsys times the kernel on the device with no per-launch host
instrumentation.

**Experiment (a)** -- [`examples/misleads/fast_kernel.py`](../examples/misleads/fast_kernel.py),
an elementwise add on 64Ki fp32 elements:

| method | value (µs) |
|---|---|
| `naive_per_iter_sync` | see [`data/misleads.csv`](data/misleads.csv) |
| `do_bench_default` | see `data/misleads.csv` |
| `caliper_batched` | see `data/misleads.csv` |
| `nsys` (ground truth) | see `data/misleads.csv` |

See also: Triton
[#2306](https://github.com/triton-lang/triton/issues/2306),
[#404](https://github.com/triton-lang/triton/issues/404).

---

## 2. Cold clocks: a fixed warmup that misses steady state

`do_bench`'s `warmup` argument is a **millisecond budget** (default 25 ms), not
"until the number stops changing". On a device that has been idle -- or on a
shared Colab box that just handed you the GPU -- the first ~100 ms of kernels
run while the SM clock is still ramping toward its sustained frequency. 25 ms of
warmup can leave you measuring inside that window, and the kernel reads faster
than it will ever run again once the clock settles (or, on a thermally limited
card, once it *drops*). The effect needs a genuinely cold device -- run this
experiment first in a fresh runtime.

`caliper.live_timing_ms` takes a real warmup budget (hundreds of ms); feeding
its per-launch samples to `caliper._core.steady_state_index` -- a
trailing-window median walked down to within a tolerance of the converged value
-- and taking the median of what's left is what the example does. `caliper
bench` will do the same trim automatically.

**Experiment (b)** -- [`examples/misleads/cold_warmup.py`](../examples/misleads/cold_warmup.py),
an fp16 4096×4096 matmul run from a cold device:

| method | value (µs) |
|---|---|
| `do_bench_warmup_25ms` | see [`data/misleads.csv`](data/misleads.csv) |
| `caliper_steady_state` | see `data/misleads.csv` |
| `nsys` | see `data/misleads.csv` |

See also: Triton
[#1252](https://github.com/triton-lang/triton/issues/1252),
[#2832](https://github.com/triton-lang/triton/issues/2832).

---

## 3. L2-resident data: a benchmark that never leaves cache

Recent GPUs have a large L2 -- 40 MB on A100, ~48 MB on L4, 50 MB on H100. A
kernel whose inputs fit in that budget, timed in a tight loop, hits warm cache
on every iteration after the first. The reported bandwidth can be 2-4× the DRAM
number the kernel actually sustains when its inputs were last written by some
*other* kernel.

`do_bench` defends against this by zeroing a fixed 256 MB buffer between reps
(`fast_flush`) -- enough to evict any current GPU's L2. A hand-rolled loop
usually skips it. `caliper bench` will size the flush buffer from the device's
real L2 (`caliper._core.flush_buffer_bytes`) and record the flush state on the
`Result`, so a reader knows the number is a cold-cache number; the example here
compares a no-flush loop against `do_bench`'s flush.

**Experiment (c)** -- [`examples/misleads/l2_resident.py`](../examples/misleads/l2_resident.py),
an 8 MiB input that fits in L2:

| method | value (µs) |
|---|---|
| `no_flush` | see [`data/misleads.csv`](data/misleads.csv) |
| `do_bench_flushed` | see `data/misleads.csv` |
| `nsys` | see `data/misleads.csv` |

See also: flashinfer-bench
[#195](https://github.com/flashinfer-ai/flashinfer-bench/issues/195).

---

## 4. Unlocked clocks: why the speedup didn't reproduce

None of the above explains run-to-run drift. That is the clock. Boost frequency
depends on temperature, power headroom, and what the box did five seconds ago.
`do_bench` does not lock the clock, does not record it, and does not notice when
the card throttles mid-measurement -- so a p50 taken now and a p50 taken after
lunch can differ by more than the "improvement" you are trying to land.

`caliper bench` will lock the SM and memory clocks through NVML where it has
permission (the golden-box / bare-metal case) and, where it does not (Colab),
record the observed SM-clock spread, drop samples taken while `nvidia-smi`
reports a throttle reason, and stamp the record `clocks-unlocked` so nobody
compares it to a locked-tier number by accident. What is shipped today:
`caliper doctor` tells you which tier you are on before you start, and the
`Result` schema carries the `clocks` block and the `clocks-unlocked` flag.

---

## Reproduce this

On a GPU (a Colab runtime is fine):

```bash
pip install caliper-gpu
git clone https://github.com/mansoor-mamnoon/caliper && cd caliper
make writeup-data          # runs all three experiments -> docs/data/misleads.csv
```

For the `nsys` rows, profile each script's `--nsys` spin mode and read the
kernel duration from `nsys stats`:

```bash
nsys profile --stats=true -o /tmp/fast_kernel python examples/misleads/fast_kernel.py --nsys
# then take the average kernel duration from the "CUDA GPU Kernel Summary" table
```

The reproducibility check is **generated vs generated**: run `make
writeup-data` in two fresh Colab runtimes; the two CSVs should agree within a
few percent. That agreement -- across a cold start, a different physical GPU,
and a different clock history -- is the thing `do_bench` alone does not give
you.

## What this is not

This is not "`do_bench` is bad". It is a fine primitive, and `caliper.do_bench`
matches it argument-for-argument so a script can swap the import. The claim is
narrower: the *default* way it is called drops context that changes the answer,
and the fix is to measure batches, warm to steady state, flush the cache, and
lock or at least record the clock -- which is what `caliper bench` is being
built to do.
