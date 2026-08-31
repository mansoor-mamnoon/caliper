//! The roofline model: achieved throughput against the hardware ceiling.
//!
//! Given a workload's FLOP count and HBM byte traffic, the wall time it took,
//! and the device architecture and dtype, this reports:
//!
//! * `achieved_tflops` and `achieved_gbps` -- the same analytic numbers the O2
//!   and O3 oracles compute, from `flops / time` and `bytes / time`;
//! * `arithmetic_intensity` -- FLOPs per byte of HBM traffic;
//! * `ridge_point` -- the arithmetic intensity where the roofline turns from
//!   memory-bound to compute-bound, `peak_compute / peak_bandwidth`;
//! * `bound` -- `compute`, `memory`, `latency`, or `unknown`.
//!
//! ## Peaks table
//!
//! [`peak_compute_tflops`] and [`peak_hbm_gbps`] hold per-architecture,
//! dtype-aware ceilings. Every cell carries a `source:` comment naming where the
//! number comes from: an NVIDIA/AMD architecture whitepaper or datasheet, or (to
//! be substituted as they are measured) a sustained value from `caliper
//! selftest` on that SKU. Tensor-core figures are **dense** (no 2:4 sparsity).
//! HBM figures are datasheet peaks; the O2 oracle's sustained bandwidth is the
//! value to prefer once recorded, with the datasheet kept in the comment.

use serde::{Deserialize, Serialize};

/// A workload's roofline inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RooflineSpec {
    /// Element dtype the math runs in: `"fp32"`, `"tf32"`, `"fp16"`, `"bf16"`,
    /// `"fp8"` (and `"fp8_e4m3"` / `"fp8_e5m2"`).
    pub dtype: String,
    /// Total floating-point operations performed (count 2 per FMA).
    pub flops: f64,
    /// Bytes moved through HBM (reads + writes that miss cache).
    pub bytes_hbm: f64,
}

/// Which side of the roofline the measurement sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Bound {
    /// Arithmetic intensity is past the ridge and throughput is near the FLOP
    /// ceiling.
    Compute,
    /// Arithmetic intensity is below the ridge and throughput is near the
    /// bandwidth ceiling.
    Memory,
    /// Well below both ceilings -- launch overhead, occupancy, or dependency
    /// latency is the limit, not the roofline.
    Latency,
    /// Peaks for this architecture/dtype are not in the table, or the inputs
    /// were degenerate.
    Unknown,
}

impl Bound {
    /// Lowercase token used in serialised records.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Memory => "memory",
            Self::Latency => "latency",
            Self::Unknown => "unknown",
        }
    }
}

/// The result of [`analyze`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RooflineResult {
    /// Achieved compute throughput (TFLOP/s).
    pub achieved_tflops: f64,
    /// Achieved HBM throughput (GB/s).
    pub achieved_gbps: f64,
    /// FLOPs per byte of HBM traffic.
    pub arithmetic_intensity: f64,
    /// Arithmetic intensity at the roofline knee, if peaks are known.
    pub ridge_point: Option<f64>,
    /// Fraction of the governing ceiling attained (compute peak when
    /// compute-bound, bandwidth peak when memory-bound), if peaks are known.
    pub roofline_pct: Option<f64>,
    /// Which regime the measurement is in.
    pub bound: Bound,
    /// The compute ceiling used (TFLOP/s), if known.
    pub peak_compute_tflops: Option<f64>,
    /// The bandwidth ceiling used (GB/s), if known.
    pub peak_hbm_gbps: Option<f64>,
}

/// Normalise an architecture tag: `"SM_90a"` -> `"sm_90"`; AMD aliases
/// (`"gfx942"`, `"mi300x"`, `"cdna3"`) -> `"cdna3"`.
fn arch_key(arch: &str) -> String {
    let t = arch.trim().to_ascii_lowercase();
    match t.as_str() {
        "cdna3" | "gfx942" | "gfx940" | "gfx941" | "mi300" | "mi300x" | "mi300a" => {
            return "cdna3".to_string()
        }
        _ => {}
    }
    let digits: String = t
        .strip_prefix("sm_")
        .unwrap_or(&t)
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    if digits.is_empty() {
        t
    } else {
        format!("sm_{digits}")
    }
}

fn dtype_key(dtype: &str) -> &'static str {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "fp32" | "float32" | "f32" | "tf32x3" => "fp32",
        "tf32" => "tf32",
        "fp16" | "float16" | "f16" | "half" => "fp16",
        "bf16" | "bfloat16" => "bf16",
        "fp8" | "fp8_e4m3" | "fp8_e5m2" | "e4m3" | "e5m2" | "float8" => "fp8",
        _ => "other",
    }
}

/// Peak FP32 FMA throughput (TFLOP/s), dense, at boost clock.
#[must_use]
pub fn peak_fp32_fma_tflops(arch: &str) -> Option<f64> {
    let v = match arch_key(arch).as_str() {
        "sm_70" => 15.7, // source: NVIDIA Tesla V100 architecture whitepaper WP-08608-001_v1.1 (15.7 TFLOP/s FP32)
        "sm_75" => 8.1, // source: NVIDIA Turing architecture whitepaper / Tesla T4 datasheet (8.1 TFLOP/s FP32)
        "sm_80" => 19.5, // source: NVIDIA A100 Tensor Core GPU datasheet (19.5 TFLOP/s FP32)
        "sm_86" => 35.6, // source: NVIDIA GA102 (Ampere) whitepaper, RTX 3090 (35.58 TFLOP/s FP32)
        "sm_89" => 82.6, // source: NVIDIA Ada GPU architecture whitepaper, AD102 / RTX 4090 (82.6 TFLOP/s FP32)
        "sm_90" => 66.9, // source: NVIDIA H100 Tensor Core GPU datasheet, SXM5 (67 TFLOP/s FP32)
        "sm_120" => 104.8, // source: NVIDIA GeForce RTX 5090 spec sheet (~105 TFLOP/s FP32) -- provisional, confirm on-device
        "cdna3" => 163.4,  // source: AMD Instinct MI300X datasheet (163.4 TFLOP/s FP32 vector)
        _ => return None,
    };
    Some(v)
}

/// Peak dense tensor-core throughput (TFLOP/s) for `dtype` on `arch`, without
/// structural sparsity. `None` if that architecture has no tensor path for the
/// dtype.
#[must_use]
pub fn peak_tensor_tflops(arch: &str, dtype: &str) -> Option<f64> {
    let v = match (arch_key(arch).as_str(), dtype_key(dtype)) {
        // fp16 tensor (fp16/fp32 accumulate), dense
        ("sm_70", "fp16") => 125.0, // source: NVIDIA Tesla V100 whitepaper (125 Tensor TFLOP/s)
        ("sm_75", "fp16") => 65.0, // source: NVIDIA Turing whitepaper / T4 datasheet (65 FP16 Tensor TFLOP/s)
        ("sm_80", "fp16" | "bf16") => 312.0, // source: NVIDIA A100 datasheet (312 TFLOP/s FP16/BF16 Tensor, dense)
        ("sm_86", "fp16" | "bf16") => 71.0, // source: NVIDIA GA102 whitepaper, RTX 3090 (71 TFLOP/s FP16 Tensor, dense; 142 w/ sparsity)
        ("sm_89", "fp16" | "bf16") => 165.2, // source: NVIDIA Ada whitepaper, AD102 / RTX 4090 (165.2 TFLOP/s FP16 Tensor, dense)
        ("sm_90", "fp16" | "bf16") => 989.4, // source: NVIDIA H100 datasheet, SXM5 (989.4 TFLOP/s FP16/BF16 Tensor, dense)
        ("sm_120", "fp16" | "bf16") => 419.0, // source: NVIDIA GeForce RTX 5090 spec (~419 TFLOP/s FP16 Tensor, dense) -- provisional
        ("cdna3", "fp16" | "bf16") => 1307.4, // source: AMD CDNA3 whitepaper / MI300X datasheet (1307.4 TFLOP/s FP16|BF16 matrix, no sparsity)

        // tf32 tensor, dense
        ("sm_80", "tf32") => 156.0, // source: NVIDIA A100 datasheet (156 TFLOP/s TF32 Tensor, dense)
        ("sm_89", "tf32") => 82.6, // source: NVIDIA Ada whitepaper, AD102 (82.6 TFLOP/s TF32 Tensor, dense)
        ("sm_90", "tf32") => 494.7, // source: NVIDIA H100 datasheet, SXM5 (494.7 TFLOP/s TF32 Tensor, dense)
        ("sm_120", "tf32") => 209.5, // source: NVIDIA RTX 5090 spec (~210 TFLOP/s TF32 Tensor, dense) -- provisional

        // fp8 tensor, dense
        ("sm_89", "fp8") => 330.3, // source: NVIDIA Ada whitepaper, AD102 / RTX 4090 (330.3 TFLOP/s FP8 Tensor, dense)
        ("sm_90", "fp8") => 1978.9, // source: NVIDIA H100 datasheet, SXM5 (1978.9 TFLOP/s FP8 Tensor, dense)
        ("sm_120", "fp8") => 838.0, // source: NVIDIA RTX 5090 spec (~838 TFLOP/s FP8 Tensor, dense) -- provisional
        ("cdna3", "fp8") => 2614.9, // source: AMD MI300X datasheet (2614.9 TFLOP/s FP8 matrix, no sparsity)

        _ => return None,
    };
    Some(v)
}

/// Peak HBM bandwidth (GB/s). Datasheet figure; prefer the O2 oracle's measured
/// sustained value once it exists on the SKU (the datasheet stays in the source
/// comment).
#[must_use]
pub fn peak_hbm_gbps(arch: &str) -> Option<f64> {
    let v = match arch_key(arch).as_str() {
        "sm_70" => 900.0, // source: NVIDIA Tesla V100 SXM2 datasheet (900 GB/s HBM2); measured O2 value pending
        "sm_75" => 320.0, // source: NVIDIA Tesla T4 datasheet (320 GB/s GDDR6); measured O2 value pending
        "sm_80" => 2039.0, // source: NVIDIA A100 80GB SXM datasheet (2039 GB/s HBM2e); measured O2 value pending
        "sm_86" => 936.0, // source: NVIDIA RTX 3090 datasheet (936.2 GB/s GDDR6X); measured O2 value pending
        "sm_89" => 1008.0, // source: NVIDIA RTX 4090 datasheet (1008 GB/s GDDR6X); measured O2 value pending
        "sm_90" => 3350.0, // source: NVIDIA H100 SXM5 datasheet (3.35 TB/s HBM3); measured O2 value pending
        "sm_120" => 1792.0, // source: NVIDIA GeForce RTX 5090 datasheet (1792 GB/s GDDR7); measured O2 value pending
        "cdna3" => 5300.0, // source: AMD Instinct MI300X datasheet (5.3 TB/s HBM3); measured O2 value pending
        _ => return None,
    };
    Some(v)
}

/// The compute ceiling for a dtype: the tensor-core peak where there is one,
/// otherwise the FP32 FMA peak (for `"fp32"`).
#[must_use]
pub fn peak_compute_tflops(arch: &str, dtype: &str) -> Option<f64> {
    match dtype_key(dtype) {
        "fp32" => peak_fp32_fma_tflops(arch),
        _ => peak_tensor_tflops(arch, dtype),
    }
}

/// Run the roofline model. `seconds` is the per-launch wall time the workload
/// took.
///
/// `achieved_tflops` / `achieved_gbps` are always filled from the raw inputs.
/// `ridge_point`, `roofline_pct`, and a non-`unknown` `bound` require the
/// architecture/dtype to be in the peaks table and the inputs to be positive.
#[must_use]
pub fn analyze(arch: &str, spec: &RooflineSpec, seconds: f64) -> RooflineResult {
    let achieved_tflops = if seconds > 0.0 {
        spec.flops / seconds / 1.0e12
    } else {
        0.0
    };
    let achieved_gbps = if seconds > 0.0 {
        spec.bytes_hbm / seconds / 1.0e9
    } else {
        0.0
    };
    let arithmetic_intensity = if spec.bytes_hbm > 0.0 {
        spec.flops / spec.bytes_hbm
    } else {
        f64::INFINITY
    };

    let peak_compute_tflops = peak_compute_tflops(arch, &spec.dtype);
    let peak_hbm_gbps = peak_hbm_gbps(arch);

    let positive_inputs = seconds > 0.0 && spec.flops > 0.0 && spec.bytes_hbm > 0.0;
    let degenerate =
        !positive_inputs || !achieved_tflops.is_finite() || !arithmetic_intensity.is_finite();

    let (ridge_point, roofline_pct, bound) = match (peak_compute_tflops, peak_hbm_gbps) {
        (Some(pc), Some(pb)) if !degenerate => {
            // ridge point in FLOP/byte: (pc * 1e12 flop/s) / (pb * 1e9 byte/s)
            let ridge = pc * 1.0e12 / (pb * 1.0e9);
            // ceiling at this arithmetic intensity, in TFLOP/s
            let mem_ceiling_tflops = arithmetic_intensity * pb / 1.0e3;
            let ceiling = pc.min(mem_ceiling_tflops);
            let attained = if ceiling > 0.0 {
                achieved_tflops / ceiling
            } else {
                0.0
            };
            let bound = if attained < 0.25 {
                Bound::Latency
            } else if arithmetic_intensity >= ridge {
                Bound::Compute
            } else {
                Bound::Memory
            };
            let pct = match bound {
                Bound::Compute => achieved_tflops / pc,
                Bound::Memory => achieved_gbps / pb,
                _ => attained,
            };
            (Some(ridge), Some(pct), bound)
        }
        _ => (
            peak_compute_tflops
                .zip(peak_hbm_gbps)
                .map(|(pc, pb)| pc * 1.0e12 / (pb * 1.0e9)),
            None,
            Bound::Unknown,
        ),
    };

    RooflineResult {
        achieved_tflops,
        achieved_gbps,
        arithmetic_intensity,
        ridge_point,
        roofline_pct,
        bound,
        peak_compute_tflops,
        peak_hbm_gbps,
    }
}

/// Build a [`schema::Roofline`] section from an [`analyze`] result.
///
/// [`schema::Roofline`]: crate::schema::Roofline
#[must_use]
pub fn roofline_section(r: &RooflineResult) -> crate::schema::Roofline {
    crate::schema::Roofline {
        achieved_tflops: Some(r.achieved_tflops),
        roofline_pct: r.roofline_pct,
        achieved_gbps: Some(r.achieved_gbps),
        arithmetic_intensity: r
            .arithmetic_intensity
            .is_finite()
            .then_some(r.arithmetic_intensity),
        ridge_point: r.ridge_point,
        bound: Some(r.bound.as_str().to_string()),
        baseline_pct: None,
        baseline: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracles::{o2_achieved_gbps, o3_achieved_tflops};

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} vs {b}");
    }

    fn spec(dtype: &str, flops: f64, bytes: f64) -> RooflineSpec {
        RooflineSpec {
            dtype: dtype.to_string(),
            flops,
            bytes_hbm: bytes,
        }
    }

    #[test]
    fn every_arch_has_a_cited_hbm_and_fp32_peak() {
        for arch in [
            "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120", "cdna3",
        ] {
            assert!(peak_hbm_gbps(arch).is_some(), "{arch} HBM");
            assert!(peak_fp32_fma_tflops(arch).is_some(), "{arch} FP32");
        }
        // The source citations live next to the numbers; spot-check that the
        // table module actually carries a citation per peaks cell.
        let src = include_str!("roofline.rs");
        let cells = src.matches("source:").count();
        assert!(
            cells >= 24,
            "expected a source comment per peaks cell, found {cells}"
        );
    }

    #[test]
    fn achieved_bandwidth_matches_the_o2_oracle() {
        // O2 triad: 1 GiB per array, 3 GiB of HBM traffic, 3.3 ms/launch.
        let bytes_per_array = 1024.0 * 1024.0 * 1024.0;
        let p50_us = 3300.0;
        let r = analyze(
            "sm_89",
            &spec("fp32", 1.0, 3.0 * bytes_per_array),
            p50_us * 1.0e-6,
        );
        approx(r.achieved_gbps, o2_achieved_gbps(bytes_per_array, p50_us));
    }

    #[test]
    fn achieved_tflops_matches_the_o3_oracle() {
        let threads = 128.0 * 128.0 * 256.0;
        let iters = 100_000.0;
        let ilp = 4.0;
        let p50_us = 1000.0;
        let flops = 2.0 * threads * iters * ilp;
        let r = analyze("sm_89", &spec("fp32", flops, 4096.0), p50_us * 1.0e-6);
        approx(
            r.achieved_tflops,
            o3_achieved_tflops(threads, iters, ilp, p50_us),
        );
    }

    #[test]
    fn ridge_point_is_peak_compute_over_peak_bandwidth() {
        // A100 bf16: 312 TFLOP/s / 2039 GB/s = ~153 FLOP/byte.
        let r = analyze("sm_80", &spec("bf16", 1.0e12, 1.0e9), 1.0e-3);
        approx(r.ridge_point.unwrap(), 312.0e12 / 2039.0e9);
    }

    #[test]
    fn a_dense_matmul_past_the_ridge_is_compute_bound() {
        // 4096^3 bf16 GEMM on A100: 2*N^3 flops, 3*N^2*2 bytes (bf16 I/O).
        let n = 4096.0;
        let flops = 2.0 * n * n * n;
        let bytes = 3.0 * n * n * 2.0;
        // ~0.44 ms would be ~312 TFLOP/s (the roofline); take 0.50 ms (~88%).
        let r = analyze("sm_80", &spec("bf16", flops, bytes), 0.50e-3);
        assert_eq!(r.bound, Bound::Compute);
        assert!(r.arithmetic_intensity > r.ridge_point.unwrap());
        assert!(r.roofline_pct.unwrap() > 0.80 && r.roofline_pct.unwrap() < 1.0);
    }

    #[test]
    fn a_streaming_triad_below_the_ridge_is_memory_bound() {
        // AI = 2 flop / 24 byte on the O2 triad -- far left of any ridge.
        let bytes = 3.0 * 1024.0 * 1024.0 * 1024.0;
        let flops = bytes / 12.0; // AI ~= 0.083
                                  // ~1.6 ms -> ~2 TB/s on A100 (near its 2039 GB/s roof).
        let r = analyze("sm_80", &spec("fp32", flops, bytes), 1.60e-3);
        assert_eq!(r.bound, Bound::Memory);
        assert!(r.arithmetic_intensity < r.ridge_point.unwrap());
        assert!(r.roofline_pct.unwrap() > 0.80);
    }

    #[test]
    fn far_below_both_ceilings_is_latency_bound() {
        // A tiny kernel: real FLOPs and bytes but 100x too slow for either roof.
        let r = analyze("sm_90", &spec("fp16", 1.0e9, 1.0e6), 1.0e-3);
        assert_eq!(r.bound, Bound::Latency);
    }

    #[test]
    fn unknown_arch_or_dtype_leaves_bound_unknown_but_keeps_achieved() {
        let r = analyze("sm_42", &spec("fp16", 1.0e12, 1.0e9), 1.0e-3);
        assert_eq!(r.bound, Bound::Unknown);
        assert!(r.achieved_tflops > 0.0);
        assert!(r.roofline_pct.is_none());

        let r = analyze("sm_70", &spec("fp8", 1.0e12, 1.0e9), 1.0e-3);
        assert_eq!(r.bound, Bound::Unknown); // Volta has no FP8 tensor path
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let r = analyze("sm_90", &spec("bf16", 0.0, 0.0), 0.0);
        assert_eq!(r.bound, Bound::Unknown);
        assert_eq!(r.achieved_tflops, 0.0);
    }

    #[test]
    fn section_conversion_is_populated() {
        let n = 4096.0;
        let r = analyze(
            "sm_90",
            &spec("bf16", 2.0 * n * n * n, 3.0 * n * n * 2.0),
            0.10e-3,
        );
        let s = roofline_section(&r);
        assert!(s.achieved_tflops.is_some());
        assert!(s.achieved_gbps.is_some());
        assert!(s.arithmetic_intensity.is_some());
        assert!(s.ridge_point.is_some());
        assert_eq!(s.bound.as_deref(), Some("compute"));
    }
}
