//! Parsing compiler resource-usage reports.
//!
//! Three formats, one output shape:
//!
//! * **`ptxas -v`** (a.k.a. `nvcc --ptxas-options=-v`, and Triton's dumped
//!   `.ptx` compile log) -- `Compiling entry function` + `Function properties`
//!   + `Used N registers` lines, one block per kernel.
//! * **`cuobjdump -res-usage`** -- a `Resource usage:` section with
//!   `REG: / STACK: / SHARED: / LOCAL:` per `Function`.
//! * **HIP / `amdgpu`** -- the `; NumVgprs:` / `; ScratchSize:` /
//!   `; LDSByteSize:` comment block emitted with `-v` / `--save-temps`.
//!
//! Every parser returns `Vec<ParsedKernel>` (a module can hold several kernels).
//! `ptxas` cannot know a kernel's *dynamic* shared memory (that is a launch
//! parameter), so `ptxas.smem_dynamic_bytes` is always left `None` here.

use serde::{Deserialize, Serialize};

use crate::schema::Ptxas;

/// One kernel's resource usage as parsed from a compiler report.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParsedKernel {
    /// Mangled or Triton kernel name, if the report names it.
    pub name: Option<String>,
    /// Target, e.g. `"sm_90a"` or `"gfx942"`, if reported.
    pub target: Option<String>,
    /// The fields that map onto [`crate::schema::Record::ptxas`].
    pub ptxas: Ptxas,
    /// AMD only: scalar registers per wave.
    pub sgprs: Option<u32>,
    /// AMD only: the compiler's occupancy hint (waves/SIMD).
    pub occupancy_hint: Option<u32>,
}

impl ParsedKernel {
    fn is_populated(&self) -> bool {
        self.ptxas.regs_per_thread.is_some()
    }
}

/// What can go wrong parsing a report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtxasParseError {
    /// The input was empty or whitespace.
    Empty,
    /// The input did not look like any known compiler report.
    UnrecognisedFormat,
    /// The format was recognised but no kernel usage was found.
    NoKernels,
    /// A number could not be parsed.
    BadNumber {
        /// Which field.
        field: String,
        /// The offending text.
        value: String,
    },
}

impl std::fmt::Display for PtxasParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("empty compiler report"),
            Self::UnrecognisedFormat => {
                f.write_str("input is not a ptxas / cuobjdump / amdgpu report")
            }
            Self::NoKernels => f.write_str("no kernel resource usage in the report"),
            Self::BadNumber { field, value } => {
                write!(f, "could not parse {field} from {value:?}")
            }
        }
    }
}

impl std::error::Error for PtxasParseError {}

/// Parse a report, sniffing the format.
///
/// # Errors
/// [`PtxasParseError::Empty`] / [`PtxasParseError::UnrecognisedFormat`] /
/// [`PtxasParseError::NoKernels`] / [`PtxasParseError::BadNumber`].
pub fn parse_any(text: &str) -> Result<Vec<ParsedKernel>, PtxasParseError> {
    if text.trim().is_empty() {
        return Err(PtxasParseError::Empty);
    }
    if text.contains("ptxas info") || text.contains("Compiling entry function") {
        return parse_ptxas_verbose(text);
    }
    if text.contains("Resource usage:") {
        return parse_cuobjdump_res_usage(text);
    }
    if text.contains("NumVgprs") || text.contains("amdgpu") || text.contains(".amdgpu_metadata") {
        return parse_hip_verbose(text);
    }
    Err(PtxasParseError::UnrecognisedFormat)
}

/// Parse `ptxas -v` output.
///
/// # Errors
/// As [`parse_any`].
#[allow(clippy::field_reassign_with_default)] // built up line by line
pub fn parse_ptxas_verbose(text: &str) -> Result<Vec<ParsedKernel>, PtxasParseError> {
    if text.trim().is_empty() {
        return Err(PtxasParseError::Empty);
    }
    let mut kernels: Vec<ParsedKernel> = Vec::new();
    let mut cur: Option<ParsedKernel> = None;

    for raw in text.lines() {
        let line = raw.trim();

        if let Some(rest) = line.split_once("Compiling entry function ").map(|x| x.1) {
            if let Some(k) = cur.take() {
                if k.is_populated() {
                    kernels.push(k);
                }
            }
            let mut k = ParsedKernel::default();
            k.name = quoted(rest).map(str::to_string);
            k.target = rest
                .split_once("for ")
                .and_then(|(_, t)| quoted(t))
                .map(str::to_string);
            cur = Some(k);
            continue;
        }

        let Some(k) = cur.as_mut() else { continue };

        if line.contains("bytes stack frame") {
            k.ptxas.stack_bytes = Some(num_before_phrase(line, "bytes stack frame", "stack")?);
            k.ptxas.spill_stores_bytes =
                Some(opt_num_before_phrase(line, "bytes spill stores").unwrap_or(0));
            k.ptxas.spill_loads_bytes =
                Some(opt_num_before_phrase(line, "bytes spill loads").unwrap_or(0));
        }

        if line.contains("registers") && (line.contains("Used ") || line.starts_with("Used ")) {
            k.ptxas.regs_per_thread =
                Some(num_before_phrase(line, "registers", "registers")? as u32);
            k.ptxas.smem_static_bytes =
                Some(opt_num_before_phrase(line, "bytes smem").unwrap_or(0));
            if let Some(lmem) = opt_num_before_phrase(line, "bytes lmem") {
                k.ptxas.local_bytes = Some(lmem);
            }
        }
    }

    if let Some(k) = cur.take() {
        if k.is_populated() {
            kernels.push(k);
        }
    }

    if kernels.is_empty() {
        return Err(PtxasParseError::NoKernels);
    }
    Ok(kernels)
}

/// Parse `cuobjdump -res-usage` output.
///
/// # Errors
/// As [`parse_any`].
#[allow(clippy::field_reassign_with_default)]
pub fn parse_cuobjdump_res_usage(text: &str) -> Result<Vec<ParsedKernel>, PtxasParseError> {
    let mut kernels: Vec<ParsedKernel> = Vec::new();
    let mut pending_name: Option<String> = None;

    for raw in text.lines() {
        let line = raw.trim();
        if let Some(name) = line
            .strip_prefix("Function ")
            .and_then(|s| s.strip_suffix(':'))
        {
            pending_name = Some(name.to_string());
            continue;
        }
        if line.starts_with("REG:") || line.contains(" REG:") {
            let mut k = ParsedKernel::default();
            k.name = pending_name.take();
            k.ptxas.regs_per_thread = Some(int_after(line, "REG:").ok_or_else(bad("REG"))? as u32);
            k.ptxas.stack_bytes = int_after(line, "STACK:");
            k.ptxas.smem_static_bytes = int_after(line, "SHARED:");
            k.ptxas.local_bytes = int_after(line, "LOCAL:");
            kernels.push(k);
        }
    }

    if kernels.is_empty() {
        return Err(PtxasParseError::NoKernels);
    }
    Ok(kernels)
}

/// Parse the HIP / `amdgpu` `; NumVgprs:` comment block.
///
/// Maps: `NumVgprs` -> `regs_per_thread`, `ScratchSize` -> `local_bytes`,
/// `LDSByteSize` -> `smem_static_bytes`. NVIDIA-style spill / stack numbers do
/// not exist on this path and stay `None`.
///
/// # Errors
/// As [`parse_any`].
#[allow(clippy::field_reassign_with_default)]
pub fn parse_hip_verbose(text: &str) -> Result<Vec<ParsedKernel>, PtxasParseError> {
    let mut kernels: Vec<ParsedKernel> = Vec::new();
    let mut name: Option<String> = None;
    let mut cur: Option<ParsedKernel> = None;

    for raw in text.lines() {
        let line = raw.trim();

        if let Some(rest) = line.strip_prefix(".type ") {
            name = rest.split(',').next().map(|s| s.trim().to_string());
        } else if let Some(rest) = line
            .strip_prefix(".globl\t")
            .or_else(|| line.strip_prefix(".globl "))
        {
            name = Some(rest.trim().to_string());
        } else if line.contains("-- Begin function ") {
            name = line
                .rsplit("-- Begin function ")
                .next()
                .map(|s| s.trim().to_string());
        }

        if line.starts_with("; Kernel info") || line.starts_with("; NumSgprs") {
            let mut k = cur.take().unwrap_or_default();
            if k.name.is_none() {
                k.name = name.clone();
            }
            cur = Some(k);
        }
        let Some(k) = cur.as_mut() else { continue };

        if let Some(v) = int_after(line, "NumVgprs:") {
            k.ptxas.regs_per_thread = Some(v as u32);
        }
        if let Some(v) = int_after(line, "NumSgprs:") {
            k.sgprs = Some(v as u32);
        }
        if let Some(v) = int_after(line, "ScratchSize:") {
            k.ptxas.local_bytes = Some(v);
        }
        if let Some(v) = int_after(line, "LDSByteSize:") {
            k.ptxas.smem_static_bytes = Some(v);
        }
        if let Some(v) = int_after(line, "Occupancy:") {
            k.occupancy_hint = Some(v as u32);
        }
    }

    if let Some(k) = cur.take() {
        if k.is_populated() {
            kernels.push(k);
        }
    }
    if kernels.is_empty() {
        return Err(PtxasParseError::NoKernels);
    }
    Ok(kernels)
}

// --- helpers ---------------------------------------------------------------

fn quoted(s: &str) -> Option<&str> {
    let start = s.find('\'')? + 1;
    let end = s[start..].find('\'')? + start;
    Some(&s[start..end])
}

fn bad(field: &'static str) -> impl Fn() -> PtxasParseError {
    move || PtxasParseError::BadNumber {
        field: field.to_string(),
        value: String::new(),
    }
}

/// The integer immediately before `phrase`, or `None` if the phrase is absent
/// or the preceding token is not a number.
fn opt_num_before_phrase(line: &str, phrase: &str) -> Option<u64> {
    let idx = line.find(phrase)?;
    line[..idx]
        .split_whitespace()
        .next_back()?
        .trim_end_matches(|c: char| !c.is_ascii_digit())
        .parse()
        .ok()
}

/// The integer immediately before `phrase` in `line`, e.g.
/// `num_before_phrase("Used 168 registers", "registers", "registers")` -> 168.
fn num_before_phrase(line: &str, phrase: &str, field: &str) -> Result<u64, PtxasParseError> {
    let idx = line
        .find(phrase)
        .ok_or_else(|| PtxasParseError::BadNumber {
            field: field.to_string(),
            value: line.to_string(),
        })?;
    let token =
        line[..idx]
            .split_whitespace()
            .next_back()
            .ok_or_else(|| PtxasParseError::BadNumber {
                field: field.to_string(),
                value: line.to_string(),
            })?;
    token
        .trim_end_matches(|c: char| !c.is_ascii_digit())
        .parse()
        .map_err(|_| PtxasParseError::BadNumber {
            field: field.to_string(),
            value: token.to_string(),
        })
}

/// The first run of digits after `marker`, e.g.
/// `int_after("REG:10 STACK:0", "REG:")` -> `Some(10)`.
fn int_after(line: &str, marker: &str) -> Option<u64> {
    let after = &line[line.find(marker)? + marker.len()..];
    let digits: String = after
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE: &str = include_str!("../tests/ptxas/simple.txt");
    const SPILLS: &str = include_str!("../tests/ptxas/spills_smem.txt");
    const MULTI: &str = include_str!("../tests/ptxas/multi.txt");
    const CUOBJDUMP: &str = include_str!("../tests/ptxas/cuobjdump.txt");
    const HIP: &str = include_str!("../tests/ptxas/hip.txt");
    const MALFORMED: &str = include_str!("../tests/ptxas/malformed.txt");

    #[test]
    fn simple_no_spill() {
        let ks = parse_any(SIMPLE).unwrap();
        assert_eq!(ks.len(), 1);
        let k = &ks[0];
        assert_eq!(k.name.as_deref(), Some("_Z10add_kernelPfPKfS1_i"));
        assert_eq!(k.target.as_deref(), Some("sm_80"));
        assert_eq!(k.ptxas.regs_per_thread, Some(16));
        assert_eq!(k.ptxas.stack_bytes, Some(0));
        assert_eq!(k.ptxas.spill_stores_bytes, Some(0));
        assert_eq!(k.ptxas.spill_loads_bytes, Some(0));
        assert_eq!(k.ptxas.smem_static_bytes, Some(0));
        assert_eq!(k.ptxas.smem_dynamic_bytes, None); // ptxas cannot know this
    }

    #[test]
    fn spills_and_shared_memory() {
        let k = &parse_any(SPILLS).unwrap()[0];
        assert_eq!(k.name.as_deref(), Some("matmul_kernel_0d1d2d3de4de5de"));
        assert_eq!(k.target.as_deref(), Some("sm_90a"));
        assert_eq!(k.ptxas.regs_per_thread, Some(168));
        assert_eq!(k.ptxas.stack_bytes, Some(1024));
        assert_eq!(k.ptxas.spill_stores_bytes, Some(48));
        assert_eq!(k.ptxas.spill_loads_bytes, Some(32));
        assert_eq!(k.ptxas.smem_static_bytes, Some(99328));
    }

    #[test]
    fn multi_kernel_module() {
        let ks = parse_any(MULTI).unwrap();
        assert_eq!(ks.len(), 2);
        assert_eq!(ks[0].name.as_deref(), Some("fwd_kernel"));
        assert_eq!(ks[0].ptxas.regs_per_thread, Some(64));
        assert_eq!(ks[0].ptxas.smem_static_bytes, Some(16384));
        assert_eq!(ks[1].name.as_deref(), Some("bwd_kernel"));
        assert_eq!(ks[1].ptxas.regs_per_thread, Some(255));
        assert_eq!(ks[1].ptxas.spill_stores_bytes, Some(128));
        assert_eq!(ks[1].ptxas.spill_loads_bytes, Some(96));
    }

    #[test]
    fn cuobjdump_res_usage() {
        let ks = parse_any(CUOBJDUMP).unwrap();
        assert_eq!(ks.len(), 2);
        assert_eq!(ks[0].name.as_deref(), Some("_Z3addPfPKfS1_"));
        assert_eq!(ks[0].ptxas.regs_per_thread, Some(10));
        assert_eq!(ks[0].ptxas.smem_static_bytes, Some(0));
        assert_eq!(ks[1].name.as_deref(), Some("_Z6reducePfPKfi"));
        assert_eq!(ks[1].ptxas.regs_per_thread, Some(24));
        assert_eq!(ks[1].ptxas.stack_bytes, Some(8));
        assert_eq!(ks[1].ptxas.smem_static_bytes, Some(2048));
        assert_eq!(ks[1].ptxas.local_bytes, Some(16));
        assert_eq!(ks[1].ptxas.spill_stores_bytes, None); // not in -res-usage
    }

    #[test]
    fn hip_amdgpu_comment_block() {
        let k = &parse_any(HIP).unwrap()[0];
        assert_eq!(k.name.as_deref(), Some("fused_kernel"));
        assert_eq!(k.ptxas.regs_per_thread, Some(40)); // NumVgprs
        assert_eq!(k.sgprs, Some(36));
        assert_eq!(k.ptxas.local_bytes, Some(128)); // ScratchSize
        assert_eq!(k.ptxas.smem_static_bytes, Some(8192)); // LDSByteSize
        assert_eq!(k.occupancy_hint, Some(8));
        assert_eq!(k.ptxas.spill_stores_bytes, None);
    }

    #[test]
    fn empty_and_malformed() {
        assert_eq!(parse_any(""), Err(PtxasParseError::Empty));
        assert_eq!(parse_any("   \n  \n"), Err(PtxasParseError::Empty));
        assert_eq!(
            parse_any(MALFORMED),
            Err(PtxasParseError::UnrecognisedFormat)
        );
        // recognised header, no kernels
        assert_eq!(
            parse_ptxas_verbose("ptxas info    : 0 bytes gmem\n"),
            Err(PtxasParseError::NoKernels)
        );
    }

    #[test]
    fn parsed_kernel_json_round_trips() {
        let k = &parse_any(SPILLS).unwrap()[0];
        let back: ParsedKernel = serde_json::from_str(&serde_json::to_string(k).unwrap()).unwrap();
        assert_eq!(&back, k);
    }
}
