//! The regression threshold model behind `caliper compare`.
//!
//! Given a baseline and a candidate dataset (each a list of [`Record`]s), this
//! aligns rows by *facet* -- the identity of a comparable measurement: kernel
//! name, impl, dtype, shape, layout, and architecture -- and for every facet
//! decides whether the candidate's median moved outside a variance-aware noise
//! band, plus surfaces the `ptxas` / occupancy deltas and any autotune configs
//! that stopped being timed.
//!
//! The band comes from the baseline's own median absolute deviation:
//! `MAD -> sigma` via the 1.4826 normal-consistency constant, then
//! `sigma_mult` sigmas, with a relative floor so a suspiciously tight baseline
//! cannot make every wobble a regression, and a 50% cap so a suspiciously
//! *noisy* one cannot mask a real one. An explicit `threshold` overrides the
//! derived band for the timing verdict -- a register-spill regression still
//! fails the run. All bands are fractions (`0.10` == 10%).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::schema::{Occupancy, Ptxas, Record};

/// MAD -> standard-deviation scale for a normal distribution.
const MAD_TO_SIGMA: f64 = 1.4826;
/// Default number of sigmas for the derived noise band.
pub const DEFAULT_SIGMA_MULT: f64 = 3.0;
/// Default relative floor for the derived noise band (2%).
pub const DEFAULT_FLOOR_PCT: f64 = 0.02;
/// Cap on the *derived* band: past a 50% swing something is wrong regardless of
/// how noisy the baseline was, so a pathological MAD can't mask a real
/// regression. An explicit `threshold` is never capped.
pub const MAX_DERIVED_BAND: f64 = 0.5;

/// Options for a comparison run. All bands are fractions (`0.10` == 10%).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CompareOpts {
    /// Only compare rows on this architecture (`machine.sm_arch`); when unset,
    /// architecture is part of the facet key so cross-arch rows never align.
    pub arch: Option<String>,
    /// Explicit noise band as a fraction; overrides the MAD-derived band for
    /// every facet when set. Register-spill regressions still fail the run.
    pub threshold: Option<f64>,
    /// Sigmas for the derived band.
    pub sigma_mult: f64,
    /// Relative floor for the derived band.
    pub floor_pct: f64,
}

impl Default for CompareOpts {
    fn default() -> Self {
        Self {
            arch: None,
            threshold: None,
            sigma_mult: DEFAULT_SIGMA_MULT,
            floor_pct: DEFAULT_FLOOR_PCT,
        }
    }
}

/// Whether `x` is a finite, strictly-positive number (a usable median / MAD).
fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// The relative noise band (a fraction of the baseline median) for one facet:
/// `sigma_mult` sigmas of the baseline's own MAD, but never below `floor_pct`
/// and never above [`MAX_DERIVED_BAND`] (unless the floor itself is higher).
#[must_use]
pub fn noise_band(mad_us: f64, p50_us: f64, sigma_mult: f64, floor_pct: f64) -> f64 {
    let floor = floor_pct.max(0.0);
    if !positive(p50_us) || !mad_us.is_finite() {
        return floor;
    }
    let derived = sigma_mult.max(0.0) * MAD_TO_SIGMA * (mad_us.max(0.0) / p50_us);
    derived.clamp(floor, MAX_DERIVED_BAND.max(floor))
}

/// The direction the candidate moved relative to the baseline, given a band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Candidate is slower than the baseline by more than the band.
    Regression,
    /// Candidate is faster than the baseline by more than the band.
    Improvement,
    /// The difference is inside the band.
    WithinNoise,
    /// One side has no usable median for this facet.
    Incomparable,
}

/// Classify a candidate median against a baseline median and a band.
#[must_use]
pub fn verdict(base_p50: f64, cand_p50: f64, band: f64) -> Verdict {
    if !positive(base_p50) || !positive(cand_p50) {
        return Verdict::Incomparable;
    }
    let delta = (cand_p50 - base_p50) / base_p50;
    if delta > band {
        Verdict::Regression
    } else if delta < -band {
        Verdict::Improvement
    } else {
        Verdict::WithinNoise
    }
}

/// The identity of a comparable measurement across two datasets.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct FacetKey {
    /// `kernel.name`.
    pub kernel: String,
    /// `kernel.impl`.
    pub r#impl: String,
    /// `kernel.dtype`.
    pub dtype: String,
    /// Canonical JSON of `kernel.shape` (keys sorted).
    pub shape: String,
    /// `kernel.layout`.
    pub layout: String,
    /// `machine.sm_arch`.
    pub arch: String,
}

impl FacetKey {
    fn of(rec: &Record) -> Self {
        Self {
            kernel: rec.kernel.name.clone().unwrap_or_default(),
            r#impl: rec.kernel.r#impl.clone().unwrap_or_default(),
            dtype: rec.kernel.dtype.clone().unwrap_or_default(),
            shape: serde_json::to_string(&rec.kernel.shape).unwrap_or_default(),
            layout: rec.kernel.layout.clone().unwrap_or_default(),
            arch: rec.machine.sm_arch.clone().unwrap_or_default(),
        }
    }
}

/// Per-field `ptxas` deltas (`candidate - baseline`); `None` when neither side
/// reports the field.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct PtxasDelta {
    pub regs_per_thread: Option<i64>,
    pub smem_static_bytes: Option<i64>,
    pub smem_dynamic_bytes: Option<i64>,
    pub spill_loads_bytes: Option<i64>,
    pub spill_stores_bytes: Option<i64>,
    pub local_bytes: Option<i64>,
    pub stack_bytes: Option<i64>,
}

fn delta_u(base: Option<u64>, cand: Option<u64>) -> Option<i64> {
    match (base, cand) {
        (None, None) => None,
        (b, c) => Some(c.unwrap_or(0) as i64 - b.unwrap_or(0) as i64),
    }
}

fn delta_u32(base: Option<u32>, cand: Option<u32>) -> Option<i64> {
    delta_u(base.map(u64::from), cand.map(u64::from))
}

fn delta_f(base: Option<f64>, cand: Option<f64>) -> Option<f64> {
    match (base, cand) {
        (None, None) => None,
        (b, c) => Some(c.unwrap_or(0.0) - b.unwrap_or(0.0)),
    }
}

impl PtxasDelta {
    fn between(base: &Ptxas, cand: &Ptxas) -> Self {
        Self {
            regs_per_thread: delta_u32(base.regs_per_thread, cand.regs_per_thread),
            smem_static_bytes: delta_u(base.smem_static_bytes, cand.smem_static_bytes),
            smem_dynamic_bytes: delta_u(base.smem_dynamic_bytes, cand.smem_dynamic_bytes),
            spill_loads_bytes: delta_u(base.spill_loads_bytes, cand.spill_loads_bytes),
            spill_stores_bytes: delta_u(base.spill_stores_bytes, cand.spill_stores_bytes),
            local_bytes: delta_u(base.local_bytes, cand.local_bytes),
            stack_bytes: delta_u(base.stack_bytes, cand.stack_bytes),
        }
    }
}

/// Occupancy deltas (`candidate - baseline`).
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct OccupancyDelta {
    pub theoretical: Option<f64>,
    pub achieved: Option<f64>,
    pub active_warps_per_sm: Option<i64>,
    pub waves: Option<f64>,
}

impl OccupancyDelta {
    fn between(base: &Occupancy, cand: &Occupancy) -> Self {
        Self {
            theoretical: delta_f(base.theoretical, cand.theoretical),
            achieved: delta_f(base.achieved, cand.achieved),
            active_warps_per_sm: delta_u32(base.active_warps_per_sm, cand.active_warps_per_sm),
            waves: delta_f(base.waves, cand.waves),
        }
    }
}

/// The comparison outcome for one facet.
#[derive(Debug, Clone, Serialize)]
pub struct FacetDelta {
    pub key: FacetKey,
    pub baseline_p50_us: Option<f64>,
    pub candidate_p50_us: Option<f64>,
    /// `(candidate - baseline) / baseline` as a fraction, when both medians are
    /// usable (`0.1` == the candidate is 10% slower).
    pub delta: Option<f64>,
    /// The noise band this facet was judged against, as a fraction.
    pub band: f64,
    pub verdict: Verdict,
    pub ptxas_delta: PtxasDelta,
    pub occupancy_delta: OccupancyDelta,
    /// The candidate spills more local memory than the baseline.
    pub spill_regression: bool,
    /// Autotune configs present in the baseline for this facet that the
    /// candidate no longer carries (canonical JSON).
    pub autotune_configs_dropped: Vec<String>,
}

/// Counts across every facet.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Summary {
    pub facets: usize,
    pub regressions: usize,
    pub improvements: usize,
    pub within_noise: usize,
    pub incomparable: usize,
    pub spill_regressions: usize,
    pub configs_dropped: usize,
    pub only_in_baseline: usize,
    pub only_in_candidate: usize,
}

/// The full comparison report.
#[derive(Debug, Clone, Serialize)]
pub struct CompareReport {
    pub arch: Option<String>,
    pub facets: Vec<FacetDelta>,
    pub summary: Summary,
    /// True when any facet is a timing regression or a spill regression --
    /// what `--fail-on-regression` keys off.
    pub any_regression: bool,
}

/// One side of a facet: the fastest row plus every autotune config seen.
struct FacetRows<'a> {
    best: Option<&'a Record>,
    configs: BTreeSet<String>,
}

impl<'a> FacetRows<'a> {
    fn new() -> Self {
        Self {
            best: None,
            configs: BTreeSet::new(),
        }
    }

    fn add(&mut self, rec: &'a Record) {
        self.configs
            .insert(serde_json::to_string(&rec.kernel.autotune_config).unwrap_or_default());
        let p50 = rec.timing.p50_us;
        match (self.best, p50) {
            (_, None) => {}
            (None, Some(_)) => self.best = Some(rec),
            (Some(cur), Some(new)) => {
                if new < cur.timing.p50_us.unwrap_or(f64::INFINITY) {
                    self.best = Some(rec);
                }
            }
        }
    }
}

fn group<'a>(records: &'a [Record], arch: Option<&str>) -> BTreeMap<FacetKey, FacetRows<'a>> {
    let mut out: BTreeMap<FacetKey, FacetRows<'a>> = BTreeMap::new();
    for rec in records {
        if let Some(a) = arch {
            if rec.machine.sm_arch.as_deref() != Some(a) {
                continue;
            }
        }
        out.entry(FacetKey::of(rec))
            .or_insert_with(FacetRows::new)
            .add(rec);
    }
    out
}

fn spill_total(p: &Ptxas) -> u64 {
    p.spill_loads_bytes.unwrap_or(0) + p.spill_stores_bytes.unwrap_or(0)
}

/// Compare a baseline and a candidate dataset.
#[must_use]
pub fn compare(baseline: &[Record], candidate: &[Record], opts: &CompareOpts) -> CompareReport {
    let arch = opts.arch.as_deref();
    let base = group(baseline, arch);
    let cand = group(candidate, arch);

    let mut keys: BTreeSet<&FacetKey> = BTreeSet::new();
    keys.extend(base.keys());
    keys.extend(cand.keys());

    let empty_ptxas = Ptxas::default();
    let empty_occ = Occupancy::default();

    let mut facets = Vec::new();
    let mut summary = Summary::default();

    for key in keys {
        let b = base.get(key);
        let c = cand.get(key);
        summary.facets += 1;

        let b_best = b.and_then(|f| f.best);
        let c_best = c.and_then(|f| f.best);
        let b_p50 = b_best.and_then(|r| r.timing.p50_us);
        let c_p50 = c_best.and_then(|r| r.timing.p50_us);

        let band = match opts.threshold {
            Some(t) => t.max(0.0),
            None => {
                let mad = b_best.and_then(|r| r.timing.mad_us).unwrap_or(0.0);
                noise_band(mad, b_p50.unwrap_or(0.0), opts.sigma_mult, opts.floor_pct)
            }
        };

        let v = match (b_p50, c_p50) {
            (Some(bp), Some(cp)) => verdict(bp, cp, band),
            _ => Verdict::Incomparable,
        };
        let delta = match (b_p50, c_p50) {
            (Some(bp), Some(cp)) if bp > 0.0 => Some((cp - bp) / bp),
            _ => None,
        };

        let b_ptx = b_best.map_or(&empty_ptxas, |r| &r.ptxas);
        let c_ptx = c_best.map_or(&empty_ptxas, |r| &r.ptxas);
        let b_occ = b_best.map_or(&empty_occ, |r| &r.occupancy);
        let c_occ = c_best.map_or(&empty_occ, |r| &r.occupancy);

        let spill_regression =
            c_best.is_some() && b_best.is_some() && spill_total(c_ptx) > spill_total(b_ptx);

        let dropped: Vec<String> = match (b, c) {
            (Some(bf), Some(cf)) => bf.configs.difference(&cf.configs).cloned().collect(),
            _ => Vec::new(),
        };

        match v {
            Verdict::Regression => summary.regressions += 1,
            Verdict::Improvement => summary.improvements += 1,
            Verdict::WithinNoise => summary.within_noise += 1,
            Verdict::Incomparable => summary.incomparable += 1,
        }
        if spill_regression {
            summary.spill_regressions += 1;
        }
        if !dropped.is_empty() {
            summary.configs_dropped += 1;
        }
        if b.is_some() && c.is_none() {
            summary.only_in_baseline += 1;
        }
        if b.is_none() && c.is_some() {
            summary.only_in_candidate += 1;
        }

        facets.push(FacetDelta {
            key: key.clone(),
            baseline_p50_us: b_p50,
            candidate_p50_us: c_p50,
            delta,
            band,
            verdict: v,
            ptxas_delta: PtxasDelta::between(b_ptx, c_ptx),
            occupancy_delta: OccupancyDelta::between(b_occ, c_occ),
            spill_regression,
            autotune_configs_dropped: dropped,
        });
    }

    let any_regression = summary.regressions > 0 || summary.spill_regressions > 0;
    CompareReport {
        arch: opts.arch.clone(),
        facets,
        summary,
        any_regression,
    }
}

/// Parse two JSON arrays of records and an options object, run [`compare`], and
/// serialise the report.
///
/// # Errors
/// A `serde_json` error if any input is not the expected JSON shape.
pub fn compare_json(
    baseline_json: &str,
    candidate_json: &str,
    opts_json: &str,
) -> Result<String, serde_json::Error> {
    let baseline: Vec<Record> = serde_json::from_str(baseline_json)?;
    let candidate: Vec<Record> = serde_json::from_str(candidate_json)?;
    let opts: CompareOpts = serde_json::from_str(opts_json)?;
    serde_json::to_string(&compare(&baseline, &candidate, &opts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(kernel: &str, p50: f64, mad: f64) -> Record {
        let mut r = Record::default();
        r.kernel.name = Some(kernel.to_string());
        r.kernel.r#impl = Some("triton".to_string());
        r.kernel.dtype = Some("bf16".to_string());
        r.kernel.layout = Some("row".to_string());
        r.machine.sm_arch = Some("sm_80".to_string());
        r.timing.p50_us = Some(p50);
        r.timing.mad_us = Some(mad);
        r
    }

    #[test]
    fn noise_band_is_the_larger_of_the_mad_band_and_the_floor() {
        // 3 * 1.4826 * (2/100) = 0.08896 -> above the 2% floor.
        assert!((noise_band(2.0, 100.0, 3.0, 0.02) - 0.088_956).abs() < 1e-6);
        // a tiny MAD is floored.
        assert_eq!(noise_band(0.01, 100.0, 3.0, 0.02), 0.02);
        // a degenerate p50 falls back to the floor.
        assert_eq!(noise_band(2.0, 0.0, 3.0, 0.05), 0.05);
    }

    #[test]
    fn verdict_classifies_relative_to_the_band() {
        assert_eq!(verdict(100.0, 112.0, 0.05), Verdict::Regression);
        assert_eq!(verdict(100.0, 88.0, 0.05), Verdict::Improvement);
        assert_eq!(verdict(100.0, 103.0, 0.05), Verdict::WithinNoise);
        assert_eq!(verdict(0.0, 100.0, 0.05), Verdict::Incomparable);
    }

    fn with_config(mut r: Record, block_m: i64) -> Record {
        r.kernel
            .autotune_config
            .insert("BLOCK_M".into(), serde_json::json!(block_m));
        r
    }

    #[test]
    fn an_injected_slowdown_past_the_band_is_a_regression() {
        let report = compare(
            &[rec("gemm", 243.2, 1.4)],
            &[rec("gemm", 272.0, 1.5)],
            &CompareOpts::default(),
        );
        assert_eq!(report.summary.regressions, 1);
        assert!(report.any_regression);
        let f = &report.facets[0];
        assert!(f.delta.unwrap() > 0.11);
        assert!(f.delta.unwrap() > f.band);
    }

    #[test]
    fn a_within_noise_difference_stays_silent() {
        // +0.33%, inside the ~2.6% band
        let report = compare(
            &[rec("gemm", 243.2, 1.4)],
            &[rec("gemm", 244.0, 1.4)],
            &CompareOpts::default(),
        );
        assert_eq!(report.summary.regressions, 0);
        assert!(!report.any_regression);
        assert_eq!(report.facets[0].verdict, Verdict::WithinNoise);
    }

    #[test]
    fn a_register_spill_increase_fires_with_the_delta_shown() {
        let mut b0 = rec("gemm", 243.2, 1.4);
        b0.ptxas.regs_per_thread = Some(168);
        let mut c = rec("gemm", 243.5, 1.4); // timing unchanged
        c.ptxas.spill_stores_bytes = Some(256);
        c.ptxas.spill_loads_bytes = Some(64);
        c.ptxas.regs_per_thread = Some(180);
        let report = compare(&[b0], &[c], &CompareOpts::default());
        let f = &report.facets[0];
        assert!(f.spill_regression);
        assert_eq!(f.ptxas_delta.spill_stores_bytes, Some(256));
        assert_eq!(f.ptxas_delta.spill_loads_bytes, Some(64));
        assert_eq!(f.ptxas_delta.regs_per_thread, Some(12));
        assert_eq!(report.summary.within_noise, 1); // not a *timing* regression
        assert!(report.any_regression); // but it still fails the run
    }

    #[test]
    fn an_explicit_threshold_overrides_the_mad_band() {
        // default band would be ~2% -> regression; a 20% threshold -> silent.
        let opts = CompareOpts {
            threshold: Some(0.20),
            ..CompareOpts::default()
        };
        let report = compare(
            &[rec("gemm", 100.0, 0.1)],
            &[rec("gemm", 108.0, 0.1)], // +8%
            &opts,
        );
        assert_eq!(report.facets[0].verdict, Verdict::WithinNoise);
        assert!((report.facets[0].band - 0.20).abs() < 1e-9);
    }

    #[test]
    fn the_derived_band_is_capped_so_a_noisy_baseline_cannot_mask_a_2x() {
        // MAD is 30% of p50 -> 3-sigma derived band would be ~1.33; capped.
        assert_eq!(noise_band(30.0, 100.0, 3.0, 0.02), MAX_DERIVED_BAND);
        // a 2x-slower candidate is still a regression despite the noise.
        let report = compare(
            &[rec("gemm", 100.0, 30.0)],
            &[rec("gemm", 210.0, 30.0)],
            &CompareOpts::default(),
        );
        assert_eq!(report.facets[0].verdict, Verdict::Regression);
        // an explicit threshold is never capped.
        let opts = CompareOpts {
            threshold: Some(3.0),
            ..CompareOpts::default()
        };
        assert!(
            (compare(
                &[rec("gemm", 100.0, 1.0)],
                &[rec("gemm", 210.0, 1.0)],
                &opts
            )
            .facets[0]
                .band
                - 3.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn a_dropped_autotune_config_is_flagged_per_facet() {
        let baseline = [
            with_config(rec("gemm", 250.0, 2.0), 128),
            with_config(rec("gemm", 243.0, 2.0), 256),
        ];
        let candidate = [with_config(rec("gemm", 244.0, 2.0), 128)];
        let report = compare(&baseline, &candidate, &CompareOpts::default());
        let f = &report.facets[0];
        assert_eq!(f.autotune_configs_dropped, [r#"{"BLOCK_M":256}"#]);
        assert_eq!(report.summary.configs_dropped, 1);
        // best baseline row (p50 243, BLOCK_M 256) vs best candidate (244) -> within noise
        assert_eq!(f.verdict, Verdict::WithinNoise);
    }

    #[test]
    fn rows_only_on_one_side_are_incomparable_not_regressions() {
        let report = compare(
            &[rec("gemm", 243.0, 1.4)],
            &[rec("rmsnorm", 60.0, 0.5)],
            &CompareOpts::default(),
        );
        assert_eq!(report.summary.facets, 2);
        assert_eq!(report.summary.incomparable, 2);
        assert_eq!(report.summary.only_in_baseline, 1);
        assert_eq!(report.summary.only_in_candidate, 1);
        assert!(!report.any_regression);
    }

    #[test]
    fn the_arch_filter_drops_other_architectures() {
        let mut other = rec("gemm", 200.0, 1.0);
        other.machine.sm_arch = Some("sm_90".to_string());
        let opts = CompareOpts {
            arch: Some("sm_90".to_string()),
            ..CompareOpts::default()
        };
        // baseline has no sm_90 row -> the sm_90 candidate is only-in-candidate.
        let report = compare(&[rec("gemm", 100.0, 1.0)], &[other], &opts);
        assert_eq!(report.summary.facets, 1);
        assert_eq!(report.summary.only_in_candidate, 1);
        assert_eq!(report.arch.as_deref(), Some("sm_90"));
    }

    #[test]
    fn compare_json_round_trips_through_the_wire_shape() {
        let base = serde_json::to_string(&[rec("gemm", 100.0, 1.0)]).unwrap();
        let cand = serde_json::to_string(&[rec("gemm", 130.0, 1.0)]).unwrap();
        let out = compare_json(&base, &cand, "{}").unwrap();
        let report: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(report["any_regression"], true);
        assert_eq!(report["summary"]["regressions"], 1);
    }
}
