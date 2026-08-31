//! Theoretical occupancy from the CUDA occupancy model.
//!
//! Given an architecture and a kernel's static resource footprint (registers per
//! thread, shared memory per block, threads per block), this reproduces the
//! number the CUDA Occupancy Calculator / `cudaOccMaxActiveBlocksPerMultiprocessor`
//! reports: the largest number of resident blocks per SM, the resulting active
//! warps, and the occupancy fraction.
//!
//! It is a pure model. On a CUDA host the [`ModuleProbe`] port also asks the
//! driver directly (`cuOccupancyMaxActiveBlocksPerMultiprocessor`); the two are
//! cross-checked there. The model exists so occupancy is reported even with no
//! device, and so the driver's answer has something to be validated against.
//!
//! [`ModuleProbe`]: ../../caliper_gpu/ports/trait.ModuleProbe.html
//!
//! ## Model
//!
//! Per-arch constants come from the CUDA C Programming Guide "Compute
//! Capabilities" technical-specification table. The allocation arithmetic
//! follows `cuda_occupancy.h`:
//!
//! * Registers are allocated per warp, the whole-warp total rounded up to 256.
//!   `blocks_by_regs = (regs_per_sm / regs_per_warp) / warps_per_block`.
//! * Shared memory per block is `(requested + reserved)` rounded up to the arch
//!   allocation unit (128 B on cc >= 7.0). `blocks_by_smem = smem_per_sm / that`.
//!   Reserved shared memory per block is 1024 B on cc >= 8.0, else 0. A kernel
//!   that requests no shared memory is not limited by it.
//! * `blocks_by_warps = max_warps_per_sm / warps_per_block`.
//! * `blocks_by_block_limit = max_blocks_per_sm`.
//!
//! Active blocks is the minimum of the four. The reported limiter is the first
//! of registers, shared memory, warps, blocks to hit that minimum -- a tunable
//! resource is named ahead of a hard architectural cap.
//!
//! Warp-count rounding for non-multiple-of-4 block sizes is not modelled; it
//! only perturbs odd launch geometries and never the register/shared-memory
//! math above.

use serde::{Deserialize, Serialize};

const WARP_SIZE: u32 = 32;
const REG_ALLOC_GRANULARITY: u32 = 256;

/// Fixed per-SM limits for one compute capability.
#[derive(Debug, Clone, Copy)]
struct ArchLimits {
    max_warps_per_sm: u32,
    max_blocks_per_sm: u32,
    regs_per_sm: u32,
    smem_per_sm: u32,
    reserved_smem_per_block: u32,
    smem_alloc_unit: u32,
}

/// Look up the per-SM limits for an `sm_XX` tag. Trailing feature letters
/// (`sm_90a`) are ignored. Returns `None` for an architecture the table does
/// not cover.
fn limits(arch: &str) -> Option<ArchLimits> {
    // source: CUDA C Programming Guide, "Compute Capabilities" -> Table
    // "Technical Specifications per Compute Capability" (max warps/SM, max
    // thread blocks/SM, 32-bit registers/SM, max shared memory/SM). Reserved
    // shared memory per block and the 128 B allocation unit are from
    // cuda_occupancy.h for cc >= 7.0 / >= 8.0.
    let a = normalize(arch);
    let l = match a.as_str() {
        // Volta
        "sm_70" | "sm_72" => ArchLimits {
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            regs_per_sm: 65536,
            smem_per_sm: 96 * 1024,
            reserved_smem_per_block: 0,
            smem_alloc_unit: 256,
        },
        // Turing
        "sm_75" => ArchLimits {
            max_warps_per_sm: 32,
            max_blocks_per_sm: 16,
            regs_per_sm: 65536,
            smem_per_sm: 64 * 1024,
            reserved_smem_per_block: 0,
            smem_alloc_unit: 256,
        },
        // Ampere GA100
        "sm_80" => ArchLimits {
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            regs_per_sm: 65536,
            smem_per_sm: 164 * 1024,
            reserved_smem_per_block: 1024,
            smem_alloc_unit: 128,
        },
        // Ampere GA10x
        "sm_86" | "sm_87" => ArchLimits {
            max_warps_per_sm: 48,
            max_blocks_per_sm: 16,
            regs_per_sm: 65536,
            smem_per_sm: 100 * 1024,
            reserved_smem_per_block: 1024,
            smem_alloc_unit: 128,
        },
        // Ada Lovelace
        "sm_89" => ArchLimits {
            max_warps_per_sm: 48,
            max_blocks_per_sm: 24,
            regs_per_sm: 65536,
            smem_per_sm: 100 * 1024,
            reserved_smem_per_block: 1024,
            smem_alloc_unit: 128,
        },
        // Hopper
        "sm_90" => ArchLimits {
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            regs_per_sm: 65536,
            smem_per_sm: 228 * 1024,
            reserved_smem_per_block: 1024,
            smem_alloc_unit: 128,
        },
        // Blackwell (consumer, GB20x). source: CUDA C Programming Guide
        // compute-capability table (12.x, sm_120 class); confirm on-device.
        "sm_120" | "sm_121" => ArchLimits {
            max_warps_per_sm: 48,
            max_blocks_per_sm: 24,
            regs_per_sm: 65536,
            smem_per_sm: 100 * 1024,
            reserved_smem_per_block: 1024,
            smem_alloc_unit: 128,
        },
        _ => return None,
    };
    Some(l)
}

/// Strip a `sm_` architecture tag down to `sm_<digits>`.
fn normalize(arch: &str) -> String {
    let t = arch.trim().to_ascii_lowercase();
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

/// Which resource bounds the number of resident blocks per SM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Limiter {
    /// Registers per thread.
    Registers,
    /// Shared memory per block.
    SharedMemory,
    /// The `max_warps_per_sm` cap.
    Warps,
    /// The `max_blocks_per_sm` cap.
    Blocks,
}

impl Limiter {
    /// The lowercase token used in serialised records (`"registers"`,
    /// `"sharedmemory"`, `"warps"`, `"blocks"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Registers => "registers",
            Self::SharedMemory => "sharedmemory",
            Self::Warps => "warps",
            Self::Blocks => "blocks",
        }
    }
}

/// The result of the occupancy model for one launch configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct OccupancyEstimate {
    /// Occupancy fraction, `active_warps_per_sm / max_warps_per_sm` (0..=1).
    pub theoretical: f64,
    /// Warps resident per SM at this configuration.
    pub active_warps_per_sm: u32,
    /// Thread blocks resident per SM.
    pub active_blocks_per_sm: u32,
    /// The resource that bounds `active_blocks_per_sm`.
    pub limiter: Limiter,
}

fn round_up(x: u32, m: u32) -> u32 {
    debug_assert!(m > 0);
    x.div_ceil(m) * m
}

/// Theoretical occupancy for a kernel on `arch`.
///
/// `smem_bytes_per_block` is the kernel's total static + dynamic shared memory
/// request; pass 0 if it uses none. Returns `None` if `arch` is not in the
/// table, `threads_per_block` is 0 or above 1024, or `regs_per_thread` exceeds
/// the 255-per-thread hardware maximum.
#[must_use]
pub fn theoretical_occupancy(
    arch: &str,
    regs_per_thread: u32,
    smem_bytes_per_block: u32,
    threads_per_block: u32,
) -> Option<OccupancyEstimate> {
    let l = limits(arch)?;
    if threads_per_block == 0 || threads_per_block > 1024 || regs_per_thread > 255 {
        return None;
    }

    let warps_per_block = threads_per_block.div_ceil(WARP_SIZE).max(1);

    let regs_per_warp = round_up(regs_per_thread.max(1) * WARP_SIZE, REG_ALLOC_GRANULARITY);
    let by_regs = (l.regs_per_sm / regs_per_warp) / warps_per_block;

    let by_smem = if smem_bytes_per_block == 0 {
        u32::MAX
    } else {
        let alloc = round_up(
            smem_bytes_per_block.saturating_add(l.reserved_smem_per_block),
            l.smem_alloc_unit,
        );
        l.smem_per_sm / alloc
    };

    let by_warps = l.max_warps_per_sm / warps_per_block;
    let by_blocks = l.max_blocks_per_sm;

    let active_blocks = by_regs.min(by_smem).min(by_warps).min(by_blocks);

    // Priority: name a tunable resource (registers, shared memory) before a
    // hard architectural cap (warps, blocks).
    let limiter = if by_regs == active_blocks {
        Limiter::Registers
    } else if by_smem == active_blocks {
        Limiter::SharedMemory
    } else if by_warps == active_blocks {
        Limiter::Warps
    } else {
        Limiter::Blocks
    };

    let active_warps = (active_blocks * warps_per_block).min(l.max_warps_per_sm);
    let theoretical = f64::from(active_warps) / f64::from(l.max_warps_per_sm);

    Some(OccupancyEstimate {
        theoretical,
        active_warps_per_sm: active_warps,
        active_blocks_per_sm: active_blocks,
        limiter,
    })
}

/// Scheduling waves: how many rounds of resident blocks the launch grid takes
/// to drain. `1.0` or less means the whole grid fits in one wave (no tail).
///
/// Returns `None` if any input is 0.
#[must_use]
pub fn waves(grid_blocks: u32, active_blocks_per_sm: u32, sm_count: u32) -> Option<f64> {
    if grid_blocks == 0 || active_blocks_per_sm == 0 || sm_count == 0 {
        return None;
    }
    let per_wave = f64::from(active_blocks_per_sm) * f64::from(sm_count);
    Some(f64::from(grid_blocks) / per_wave)
}

/// Build a [`schema::Occupancy`] section from the model, for the pipeline to
/// drop into a record. `grid_blocks` and `sm_count` are optional; supply both
/// to also fill in [`schema::Occupancy::waves`].
///
/// Returns `None` under the same conditions as [`theoretical_occupancy`].
///
/// [`schema::Occupancy`]: crate::schema::Occupancy
#[must_use]
pub fn occupancy_section(
    arch: &str,
    regs_per_thread: u32,
    smem_bytes_per_block: u32,
    threads_per_block: u32,
    grid_blocks: Option<u32>,
    sm_count: Option<u32>,
) -> Option<crate::schema::Occupancy> {
    let est = theoretical_occupancy(
        arch,
        regs_per_thread,
        smem_bytes_per_block,
        threads_per_block,
    )?;
    let waves = match (grid_blocks, sm_count) {
        (Some(g), Some(n)) => waves(g, est.active_blocks_per_sm, n),
        _ => None,
    };
    Some(crate::schema::Occupancy {
        theoretical: Some(est.theoretical),
        achieved: None,
        active_warps_per_sm: Some(est.active_warps_per_sm),
        waves,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row of the checked-in CUDA Occupancy Calculator reference table
    /// must be reproduced exactly (FR-8: theoretical occupancy matches a
    /// checked-in reference table for >= 10 tuples).
    #[test]
    fn matches_the_cuda_occupancy_calculator_reference_table() {
        let table = include_str!("../tests/occupancy/reference.csv");
        let mut checked = 0;
        for line in table.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f.len(), 7, "malformed reference row: {line}");
            let arch = f[0];
            let regs: u32 = f[1].parse().unwrap();
            let smem: u32 = f[2].parse().unwrap();
            let block: u32 = f[3].parse().unwrap();
            let want_occ: f64 = f[4].parse().unwrap();
            let want_warps: u32 = f[5].parse().unwrap();
            let want_limiter = f[6];

            let got = theoretical_occupancy(arch, regs, smem, block)
                .unwrap_or_else(|| panic!("no estimate for {line}"));
            assert!(
                (got.theoretical - want_occ).abs() < 1e-9,
                "{line}: occupancy {} != {want_occ}",
                got.theoretical
            );
            assert_eq!(got.active_warps_per_sm, want_warps, "{line}: active warps");
            assert_eq!(got.limiter.as_str(), want_limiter, "{line}: limiter");
            checked += 1;
        }
        assert!(checked >= 10, "reference table shrank to {checked} rows");
    }

    #[test]
    fn full_occupancy_is_one_and_partial_scales_linearly() {
        // 32 regs, no smem, 256 threads on Ampere: register file is exactly
        // spent at 8 blocks x 64 warps -> 100%.
        let e = theoretical_occupancy("sm_80", 32, 0, 256).unwrap();
        assert_eq!(e.active_blocks_per_sm, 8);
        assert!((e.theoretical - 1.0).abs() < 1e-12);

        // Double the registers -> half the blocks -> half the occupancy.
        let e = theoretical_occupancy("sm_80", 64, 0, 256).unwrap();
        assert_eq!(e.active_blocks_per_sm, 4);
        assert!((e.theoretical - 0.5).abs() < 1e-12);
    }

    #[test]
    fn shared_memory_can_be_the_limiter() {
        // ~99 KiB/block on Ampere leaves room for a single resident block even
        // though registers would allow eight.
        let e = theoretical_occupancy("sm_80", 32, 101_376, 256).unwrap();
        assert_eq!(e.active_blocks_per_sm, 1);
        assert_eq!(e.limiter, Limiter::SharedMemory);
    }

    #[test]
    fn warp_and_block_caps_apply() {
        // Turing caps at 32 warps/SM: 256-thread blocks -> at most 4 resident
        // even though the register file would hold more.
        let e = theoretical_occupancy("sm_75", 32, 0, 256).unwrap();
        assert_eq!(e.active_blocks_per_sm, 4);
        assert_eq!(e.limiter, Limiter::Warps);
    }

    #[test]
    fn rejects_out_of_range_configs() {
        assert!(theoretical_occupancy("sm_80", 32, 0, 0).is_none());
        assert!(theoretical_occupancy("sm_80", 32, 0, 2048).is_none());
        assert!(theoretical_occupancy("sm_80", 300, 0, 256).is_none());
        assert!(theoretical_occupancy("sm_42", 32, 0, 256).is_none());
    }

    #[test]
    fn architecture_tag_is_normalised() {
        assert_eq!(
            theoretical_occupancy("SM_90a", 32, 0, 256),
            theoretical_occupancy("sm_90", 32, 0, 256),
        );
    }

    #[test]
    fn waves_counts_grid_rounds() {
        // 8 resident blocks/SM x 128 SMs = 1024 slots; a 1024-block grid is
        // exactly one wave, a 1536-block grid is one and a half.
        assert!((waves(1024, 8, 128).unwrap() - 1.0).abs() < 1e-12);
        assert!((waves(1536, 8, 128).unwrap() - 1.5).abs() < 1e-12);
        assert!(waves(0, 8, 128).is_none());
    }

    #[test]
    fn section_carries_the_model_into_the_schema() {
        let s = occupancy_section("sm_89", 168, 99_328, 256, Some(4096), Some(128)).unwrap();
        assert!((s.theoretical.unwrap() - 8.0 / 48.0).abs() < 1e-12);
        assert_eq!(s.active_warps_per_sm, Some(8));
        assert!(s.waves.unwrap() > 0.0);
        assert!(s.achieved.is_none());
    }
}
