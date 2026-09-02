# examples

Runnable scripts for the docs. They need a CUDA host (a Colab runtime is fine);
nothing here imports `torch` at module load, so they stay lint-clean off-GPU.

## `misleads/`

The three experiments behind [`docs/why-do_bench-misleads.md`](../docs/why-do_bench-misleads.md):

| script | shows |
|---|---|
| `fast_kernel.py` | a < 20 µs kernel: per-launch event / sync overhead vs a batched measurement |
| `cold_warmup.py` | a 25 ms warmup landing inside the clock ramp vs a steady-state trim |
| `l2_resident.py` | an L2-resident working set timed with vs without a cache flush |

```bash
make writeup-data          # runs all three -> docs/data/misleads.csv
python examples/misleads/fast_kernel.py          # one experiment, prints a table
python examples/misleads/fast_kernel.py --nsys   # spin mode, for `nsys profile`
```

Each script also prints the `nsys` command to fill in the ground-truth column.
