# Schema

The record schema is defined once, in Rust, and is the single source of truth
for both repos:

- **Record / row** -- `caliper-core::schema` (Appendix B JSON, Appendix C flat
  Parquet row). `caliper._core.default_record_json()` prints an empty record;
  `caliper validate <file>` checks any file against it.
- **Bundle manifest** -- `caliper-core::submit::Manifest`.
- **Bundle gate** -- `caliper-core::submit::validate_bundle`, run by
  `caliper validate <dir>` and by `.github/workflows/validate.yml`.

There is no separate JSON-Schema file to drift out of sync: the validator *is*
the schema. Pin a `caliper-gpu` version in CI to pin the rules.
