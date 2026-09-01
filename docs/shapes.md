# Shape libraries

A `sweep` spec's `shapes:` field can name one of these libraries instead of an
inline list. Each is a fixed, small set of problem shapes chosen to make a sweep
say something. The lists live in `crates/caliper-core/src/shapes.rs`; this file
is the rationale and the source of every number.

A GEMM shape is `{M, N, K}` for `(M, K) x (K, N) -> (M, N)`. An attention shape
is `{B, H, S, D}` (batch, heads, sequence, head dim) — reserved; no library
emits one yet.

## `square-pow2` — 5 shapes

`M = N = K ∈ {512, 1024, 2048, 4096, 8192}`.

Square, power-of-two GEMMs: the best case for tiling and alignment, and the
shapes most kernels are tuned for. This is the baseline a sweep compares
everything else against.

## `prime-odd` — 6 shapes

`M = N = K ∈ {257, 383, 509, 1021, 2039, 4093}` — every value odd and prime,
most just off a power of two (257 = 2⁸+1, 509 = 2⁹−3, 1021 = 2¹⁰−3,
4093 = 2¹²−3).

Odd, prime dimensions force a kernel through its remainder / predicated-tail
path on every axis and defeat any assumption of 128- or 256-element alignment.
A kernel that is fast on `square-pow2` but collapses here has a tail-handling
problem worth surfacing.

## `llm-7b` — 6 shapes

The three distinct GEMMs in one **Llama-2-7B** decoder layer (multi-head
attention), at prefill sequence lengths **512** and **2048** (batch 1, so
`M = S`):

| GEMM | M | N | K | role |
|---|---|---|---|---|
| projection | S | 4096 | 4096 | q / k / v and output projection |
| MLP up/gate | S | 11008 | 4096 | `hidden -> ffn` |
| MLP down | S | 4096 | 11008 | `ffn -> hidden` |

Config source: Llama-2-7B — `hidden_size = 4096`, `intermediate_size = 11008`,
32 attention heads (no GQA).

## `llm-70b` — 8 shapes

**Llama-2-70B** uses grouped-query attention: 64 query heads and **8** KV heads
of dim 128, so the K/V projection (`hidden -> 8·128 = 1024`) is a distinct,
much smaller GEMM than the Q / output projection.

| GEMM | M | N | K | role |
|---|---|---|---|---|
| Q / output projection | S | 8192 | 8192 | `hidden -> hidden` |
| grouped K / V projection | S | 1024 | 8192 | `hidden -> kv_dim` |
| MLP up/gate | S | 28672 | 8192 | `hidden -> ffn` |
| MLP down | S | 8192 | 28672 | `ffn -> hidden` |

Config source: Llama-2-70B — `hidden_size = 8192`, `intermediate_size = 28672`,
64 query heads, 8 KV heads, `head_dim = 128`. Sequence lengths 512 and 2048.
