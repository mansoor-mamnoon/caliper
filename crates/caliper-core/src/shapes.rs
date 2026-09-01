//! Named shape libraries for `sweep`.
//!
//! A `sweep` spec can point `shapes:` at a named library instead of an inline
//! list. Each library is a fixed, documented set of problem shapes; see
//! `docs/shapes.md` for the rationale and the source of every number.
//!
//! Two shape kinds:
//!
//! * [`Shape::Gemm`] `{m, n, k}` -- a dense matmul `(m, k) x (k, n)`.
//! * [`Shape::Attn`] `{b, h, s, d}` -- attention (batch, heads, sequence, head
//!   dim). No library emits these yet; the variant exists so the spec parser
//!   and Parquet schema are ready for it.

use serde::{Deserialize, Serialize};

/// One problem shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "kind")]
pub enum Shape {
    /// Dense GEMM: `(m, k) x (k, n) -> (m, n)`.
    Gemm {
        /// Rows of the output.
        m: u64,
        /// Columns of the output.
        n: u64,
        /// Contraction dimension.
        k: u64,
    },
    /// Attention: batch, heads, sequence length, head dimension.
    Attn {
        /// Batch size.
        b: u64,
        /// Number of heads.
        h: u64,
        /// Sequence length.
        s: u64,
        /// Per-head dimension.
        d: u64,
    },
}

impl Shape {
    /// A stable `key=value` string, for cell keys and Parquet rows.
    #[must_use]
    pub fn label(&self) -> String {
        match *self {
            Shape::Gemm { m, n, k } => format!("gemm(m={m},n={n},k={k})"),
            Shape::Attn { b, h, s, d } => format!("attn(b={b},h={h},s={s},d={d})"),
        }
    }
}

fn gemm(m: u64, n: u64, k: u64) -> Shape {
    Shape::Gemm { m, n, k }
}

/// The names every caller can pass to `shapes:`.
pub const LIBRARY_NAMES: &[&str] = &["square-pow2", "prime-odd", "llm-7b", "llm-70b"];

/// Resolve a named shape library to its concrete shape list. `None` if the name
/// is not one of [`LIBRARY_NAMES`].
#[must_use]
pub fn resolve(name: &str) -> Option<Vec<Shape>> {
    let shapes = match name.trim() {
        // Square, power-of-two GEMMs -- the well-behaved baseline.
        "square-pow2" => vec![
            gemm(512, 512, 512),
            gemm(1024, 1024, 1024),
            gemm(2048, 2048, 2048),
            gemm(4096, 4096, 4096),
            gemm(8192, 8192, 8192),
        ],
        // Odd / prime square GEMMs -- exercise remainder loops and unaligned
        // tails that power-of-two shapes never hit. Every dim is prime.
        "prime-odd" => vec![
            gemm(257, 257, 257),
            gemm(383, 383, 383),
            gemm(509, 509, 509),
            gemm(1021, 1021, 1021),
            gemm(2039, 2039, 2039),
            gemm(4093, 4093, 4093),
        ],
        // The GEMMs in one Llama-2-7B decoder layer (hidden 4096, MLP 11008),
        // multi-head attention, at prefill sequence lengths 512 and 2048,
        // batch 1. See docs/shapes.md.
        "llm-7b" => llm_layer(4096, 11008, 4096, &[512, 2048]),
        // Llama-2-70B decoder layer (hidden 8192, MLP 28672). Grouped-query
        // attention: 64 query heads, 8 KV heads of dim 128 -> the K/V
        // projection is `hidden -> 1024`, distinct from the `hidden -> hidden`
        // Q/output projection.
        "llm-70b" => llm_layer(8192, 28672, 8 * 128, &[512, 2048]),
        _ => return None,
    };
    Some(shapes)
}

/// The distinct GEMMs in a transformer decoder layer, per sequence length:
/// Q / output projection `(s, hidden) x (hidden, hidden)`; the K/V projection
/// `(s, hidden) x (hidden, kv_dim)` when grouped-query attention makes it
/// smaller than `hidden`; MLP up/gate `(s, hidden) x (hidden, ffn)`; MLP down
/// `(s, ffn) x (ffn, hidden)`.
fn llm_layer(hidden: u64, ffn: u64, kv_dim: u64, seq_lens: &[u64]) -> Vec<Shape> {
    let mut out = Vec::new();
    for &s in seq_lens {
        out.push(gemm(s, hidden, hidden)); // Q / output projection
        if kv_dim != hidden {
            out.push(gemm(s, kv_dim, hidden)); // grouped K / V projection
        }
        out.push(gemm(s, ffn, hidden)); // MLP up / gate
        out.push(gemm(s, hidden, ffn)); // MLP down
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_library_resolves_to_a_non_empty_list() {
        for name in LIBRARY_NAMES {
            let shapes = resolve(name).unwrap_or_else(|| panic!("{name} did not resolve"));
            assert!(!shapes.is_empty(), "{name} is empty");
        }
        assert!(resolve("no-such-library").is_none());
        assert!(resolve("  square-pow2  ").is_some()); // trimmed
    }

    #[test]
    fn square_pow2_is_five_square_powers_of_two() {
        let s = resolve("square-pow2").unwrap();
        assert_eq!(s.len(), 5);
        for shape in s {
            let Shape::Gemm { m, n, k } = shape else {
                panic!("not a gemm")
            };
            assert_eq!(m, n);
            assert_eq!(n, k);
            assert!(m.is_power_of_two());
        }
    }

    #[test]
    fn prime_odd_dims_are_all_odd_and_prime() {
        let is_prime = |x: u64| (2..x).all(|d| x % d != 0);
        for shape in resolve("prime-odd").unwrap() {
            let Shape::Gemm { m, .. } = shape else {
                panic!("not a gemm")
            };
            assert_eq!(m % 2, 1, "{m} is even");
            assert!(is_prime(m), "{m} is not prime");
        }
    }

    #[test]
    fn llm_libraries_cover_the_layer_gemms_per_sequence_length() {
        // llm-7b is MHA: 3 distinct gemms per seq length.
        let s7 = resolve("llm-7b").unwrap();
        assert_eq!(s7.len(), 6);
        assert!(s7.contains(&gemm(2048, 11008, 4096))); // mlp up, hidden 4096
        assert!(s7.contains(&gemm(2048, 4096, 11008))); // mlp down

        // llm-70b is GQA: the K/V projection is a 4th, smaller gemm.
        let s70 = resolve("llm-70b").unwrap();
        assert_eq!(s70.len(), 8);
        assert!(s70.contains(&gemm(512, 8192, 8192))); // Q / output projection
        assert!(s70.contains(&gemm(512, 1024, 8192))); // grouped K/V projection (8*128)
        assert!(s70.contains(&gemm(512, 8192, 28672))); // mlp down
    }

    #[test]
    fn shape_labels_are_stable() {
        assert_eq!(gemm(2048, 4096, 4096).label(), "gemm(m=2048,n=4096,k=4096)");
        assert_eq!(
            (Shape::Attn {
                b: 1,
                h: 32,
                s: 2048,
                d: 128
            })
            .label(),
            "attn(b=1,h=32,s=2048,d=128)"
        );
    }
}
