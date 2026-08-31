# caliper-gpu kernels

CUDA C++ device code for `caliper-gpu`. Two groups:

- **`oracles.cu`** — the self-check kernels O1–O4/O6 (see `docs/plan.md`
  Appendix A and `caliper-core/src/oracles.rs` for the analytic expectations).
- reference/benchmark kernels — added later.

## Build contract

These are compiled **only** by the `caliper-gpu` `cuda` feature, on a host with
the CUDA toolkit, via a `build.rs` that invokes `nvcc` (through the `cc` crate).
The `cuda` feature is off by default, so the default build — and CI on a machine
without CUDA — never touches `nvcc`.

Each kernel exposes an `extern "C"` `launch_*` wrapper; that is the ABI the Rust
launcher (`caliper_gpu::real`) calls. Signatures are stable once a launcher
consumes them.

## Why they're checked in without a builder yet

The Rust launcher that would call these is a feature-gated stub until it is
implemented and validated on a CUDA host (Colab / a rented box), per the
resource-adjusted plan. Keeping the kernel sources here now fixes their
interface and lets the analytic checks in `caliper-core` be written and tested
against the exact math the kernels implement.
