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
//! value to prefer once recorded, with the datasheet kept in the comment. Every
//! HBM cell is still a datasheet number today (`; measured O2 value pending`) --
//! the sustained measurements come from `caliper selftest` on each SKU.

use serde::{Deserialize, Serialize};

/// Below this fraction of the roofline ceiling at the workload's arithmetic
/// intensity, the run is called `latency`-bound rather than compute- or
/// memory-bound: it is not close enough to either roof for the roofline to be
/// the explanation (launch overhead, low occupancy, or dependency stalls are).
/// A quarter of the ceiling is the usual rule-of-thumb cut in roofline write-ups.
const LATENCY_CEILING_FRACTION: f64 = 0.25;

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
    /// FLOPs per byte of HBM traffic. `f64::INFINITY` when `bytes_hbm` is 0
    /// (`roofline_section` maps that to `None`).
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

// PEAKS-TABLE-START (every `=> <number>,` arm below carries a `source:` comment;
// enforced by `every_arch_has_a_cited_hbm_and_fp32_peak`)

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

// PEAKS-TABLE-END

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
            let bound = if attained < LATENCY_CEILING_FRACTION {
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

/// The largest `roofline_pct` a record may carry (`schema::validate` rejects
/// anything above this). Boost-clock spread over the datasheet peak lands a
/// healthy measurement a little over 1.0; a value past 1.5 means the caller's
/// FLOP/byte counts are wrong, and the section clamps rather than emit a record
/// that fails validation.
const MAX_RECORDED_ROOFLINE_PCT: f64 = 1.5;

/// Build a [`schema::Roofline`] section from an [`analyze`] result. Non-finite
/// numbers are dropped and `roofline_pct` is clamped to the range
/// `schema::validate` accepts, so the section can always go into a valid record.
///
/// [`schema::Roofline`]: crate::schema::Roofline
#[must_use]
pub fn roofline_section(r: &RooflineResult) -> crate::schema::Roofline {
    fn finite(x: f64) -> Option<f64> {
        x.is_finite().then_some(x)
    }
    crate::schema::Roofline {
        achieved_tflops: finite(r.achieved_tflops),
        roofline_pct: r
            .roofline_pct
            .and_then(finite)
            .map(|p| p.clamp(0.0, MAX_RECORDED_ROOFLINE_PCT)),
        achieved_gbps: finite(r.achieved_gbps),
        arithmetic_intensity: finite(r.arithmetic_intensity),
        ridge_point: r.ridge_point.and_then(finite),
        bound: Some(r.bound.as_str().to_string()),
        baseline_pct: None,
        baseline: None,
    }
}

// --- corpus RooflineSpec inference ----------------------------------------

/// Bytes per element for a dtype token; defaults to 4 (fp32) when unknown.
fn dtype_bytes(dtype: &str) -> f64 {
    if matches!(
        dtype.trim().to_ascii_lowercase().as_str(),
        "fp64" | "float64" | "f64" | "double"
    ) {
        return 8.0;
    }
    match dtype_key(dtype) {
        "fp16" | "bf16" => 2.0,
        "fp8" => 1.0,
        _ => 4.0, // fp32, tf32, and anything unrecognised
    }
}

fn shape_num(shape: &crate::schema::JsonMap, key: &str) -> Option<f64> {
    shape
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .filter(|v| v.is_finite() && *v > 0.0)
}

/// Guard against a shape big enough to overflow the FLOP / byte arithmetic to
/// infinity -- `serde_json` cannot serialise a non-finite float.
fn finite_spec(spec: RooflineSpec) -> Option<RooflineSpec> {
    (spec.flops.is_finite() && spec.bytes_hbm.is_finite()).then_some(spec)
}

/// The FLOP and HBM-byte counts for a built-in corpus kernel at a given shape,
/// so `bench(corpus:*)` can fill in the roofline section without a hand-written
/// spec.
///
/// Recognises `corpus:gemm*` (needs `M` / `N` / `K` in `shape`),
/// `corpus:rmsnorm*` / `corpus:softmax*` (needs `ROWS` / `COLS`),
/// `oracle:triad` (needs `n`), and `oracle:fma_peak` (needs `threads` /
/// `iters` / `ilp`). Returns `None` for a kernel with no meaningful roofline
/// (a pure spin) or a shape that is missing a dimension.
#[must_use]
pub fn corpus_spec(
    kernel_key: &str,
    shape: &crate::schema::JsonMap,
    dtype: Option<&str>,
) -> Option<RooflineSpec> {
    let key = kernel_key.to_ascii_lowercase();

    if key.contains("gemm") {
        let (m, n, k) = (
            shape_num(shape, "M")?,
            shape_num(shape, "N")?,
            shape_num(shape, "K")?,
        );
        let dt = dtype.unwrap_or("bf16");
        let elem = dtype_bytes(dt);
        return finite_spec(RooflineSpec {
            dtype: dt.to_string(),
            flops: 2.0 * m * n * k,
            bytes_hbm: (m * k + k * n + m * n) * elem,
        });
    }

    if key.ends_with("triad") {
        let n = shape_num(shape, "n")?;
        let dt = dtype.unwrap_or("fp32");
        return finite_spec(RooflineSpec {
            dtype: dt.to_string(),
            flops: 2.0 * n,                       // one add + one multiply per element
            bytes_hbm: 3.0 * n * dtype_bytes(dt), // read b, read c, write a
        });
    }

    // RMSNorm forward: per element, square + reduce-sum + normalize-multiply +
    // weight-multiply ~= 4 FLOPs (the O(rows) mean/rsqrt is negligible at any
    // real `cols`); HBM traffic is dominated by reading `x` and writing `y`,
    // each `rows*cols` elements (the `cols`-sized weight vector is negligible
    // at any real `rows` and is not counted).
    if key.contains("rmsnorm") {
        let (rows, cols) = (shape_num(shape, "ROWS")?, shape_num(shape, "COLS")?);
        let dt = dtype.unwrap_or("bf16");
        return finite_spec(RooflineSpec {
            dtype: dt.to_string(),
            flops: 4.0 * rows * cols,
            bytes_hbm: 2.0 * rows * cols * dtype_bytes(dt),
        });
    }

    // Softmax forward: per element, row-max + subtract + exp + reduce-sum +
    // divide ~= 5 FLOPs (`exp` counted as one op, matching common practice);
    // HBM traffic is read `x` + write `y`, same accounting as rmsnorm above.
    if key.contains("softmax") {
        let (rows, cols) = (shape_num(shape, "ROWS")?, shape_num(shape, "COLS")?);
        let dt = dtype.unwrap_or("bf16");
        return finite_spec(RooflineSpec {
            dtype: dt.to_string(),
            flops: 5.0 * rows * cols,
            bytes_hbm: 2.0 * rows * cols * dtype_bytes(dt),
        });
    }

    if key.ends_with("fma_peak") {
        let (threads, iters, ilp) = (
            shape_num(shape, "threads")?,
            shape_num(shape, "iters")?,
            shape_num(shape, "ilp")?,
        );
        let dt = dtype.unwrap_or("fp32");
        return finite_spec(RooflineSpec {
            dtype: dt.to_string(),
            flops: 2.0 * threads * iters * ilp,
            // register-resident: essentially no HBM traffic. A single nominal
            // write keeps arithmetic intensity finite and far past any ridge.
            bytes_hbm: threads * 4.0,
        });
    }

    None
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
        // FR-9: "every peaks-table cell has a cited source in code." Check it
        // per cell -- inside the PEAKS-TABLE-START/END region, every
        // `=> <number>,` match arm must carry a `source:` comment on that line.
        let src = include_str!("roofline.rs");
        let table = src
            .split_once("PEAKS-TABLE-START")
            .and_then(|(_, rest)| rest.split_once("PEAKS-TABLE-END"))
            .map(|(body, _)| body)
            .expect("peaks-table sentinels present");
        let mut cells = 0;
        let mut uncited = Vec::new();
        for line in table.lines() {
            let Some((_, rhs)) = line.trim().split_once("=> ") else {
                continue;
            };
            if !rhs.starts_with(|c: char| c.is_ascii_digit()) {
                continue; // `_ => return None`, `=> "fp32"`, etc.
            }
            cells += 1;
            if !line.contains("source:") {
                uncited.push(line.trim().to_string());
            }
        }
        assert!(cells >= 30, "peaks table shrank to {cells} cells");
        assert!(
            uncited.is_empty(),
            "peaks-table cells with no source: {uncited:#?}"
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

    fn shape(pairs: &[(&str, f64)]) -> crate::schema::JsonMap {
        pairs
            .iter()
            .map(|&(k, v)| (k.to_string(), serde_json::json!(v)))
            .collect()
    }

    #[test]
    fn corpus_spec_for_gemm_counts_flops_and_bytes() {
        let s = corpus_spec(
            "corpus:gemm_bf16",
            &shape(&[("M", 4096.0), ("N", 4096.0), ("K", 4096.0)]),
            Some("bf16"),
        )
        .unwrap();
        approx(s.flops, 2.0 * 4096.0 * 4096.0 * 4096.0);
        approx(s.bytes_hbm, 3.0 * 4096.0 * 4096.0 * 2.0); // 3 square bf16 tiles
        assert_eq!(s.dtype, "bf16");
        // and it lands compute-bound on an A100.
        let r = analyze("sm_80", &s, 0.5e-3);
        assert_eq!(r.bound, Bound::Compute);
    }

    #[test]
    fn corpus_spec_for_rmsnorm_is_memory_bound() {
        let s = corpus_spec(
            "corpus:rmsnorm",
            &shape(&[("ROWS", 4096.0), ("COLS", 8192.0)]),
            Some("bf16"),
        )
        .unwrap();
        approx(s.flops, 4.0 * 4096.0 * 8192.0);
        approx(s.bytes_hbm, 2.0 * 4096.0 * 8192.0 * 2.0); // read x + write y, bf16
        assert_eq!(s.dtype, "bf16");
        let r = analyze("sm_80", &s, 50.0e-6);
        assert_eq!(r.bound, Bound::Memory);
    }

    #[test]
    fn corpus_spec_for_softmax_is_memory_bound() {
        let s = corpus_spec(
            "corpus:softmax",
            &shape(&[("ROWS", 4096.0), ("COLS", 8192.0)]),
            Some("bf16"),
        )
        .unwrap();
        approx(s.flops, 5.0 * 4096.0 * 8192.0);
        approx(s.bytes_hbm, 2.0 * 4096.0 * 8192.0 * 2.0);
        let r = analyze("sm_80", &s, 50.0e-6);
        assert_eq!(r.bound, Bound::Memory);
    }

    #[test]
    fn corpus_spec_for_triad_is_memory_bound() {
        let s = corpus_spec("oracle:triad", &shape(&[("n", 1.0e8)]), None).unwrap();
        approx(s.flops, 2.0e8);
        approx(s.bytes_hbm, 3.0 * 1.0e8 * 4.0);
        assert_eq!(s.dtype, "fp32");
        let r = analyze("sm_80", &s, 0.6e-3);
        assert_eq!(r.bound, Bound::Memory);
    }

    #[test]
    fn corpus_spec_for_fma_peak_is_compute_bound() {
        let s = corpus_spec(
            "oracle:fma_peak",
            &shape(&[("threads", 4.0e6), ("iters", 1.0e5), ("ilp", 4.0)]),
            None,
        )
        .unwrap();
        approx(s.flops, 2.0 * 4.0e6 * 1.0e5 * 4.0);
        // arithmetic intensity is far past any ridge point (max ~300 FLOP/byte)
        assert!(s.flops / s.bytes_hbm > 1.0e4);
    }

    #[test]
    fn corpus_spec_is_none_for_a_spin_or_a_missing_dimension() {
        assert!(corpus_spec("oracle:busy", &shape(&[]), None).is_none());
        assert!(corpus_spec("corpus:gemm_bf16", &shape(&[("M", 4096.0)]), None).is_none());
        assert!(corpus_spec("corpus:rmsnorm", &shape(&[("ROWS", 4096.0)]), None).is_none());
        assert!(corpus_spec("corpus:softmax", &shape(&[]), None).is_none());
    }
}
