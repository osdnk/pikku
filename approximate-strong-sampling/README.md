# Approximate strong sampling experiment

This experiment estimates the `epsilon_C` column for
`tab:selected-almost-splitting-primes` in `easy_sampler.tex`.

It depends on Rokoko's `incomplete-rexl` crate for modular field
arithmetic, pulled from GitHub with the exact commit pinned in `Cargo.lock`. The challenge sampler is copied from Rokoko's
`src/common/short_challenge.rs` structure and adjusted to the table's
fixed weights and operator-norm bounds.

The sampler is AVX-512-only and the local Cargo config builds with
`target-cpu=native`.

Run:

```sh
cargo run --release -- --samples 16777216
```

The default sample count is `2^22`. The program prints CSV and LaTeX rows.
The committed `results-2pow24.csv` file is the output used for the table.
