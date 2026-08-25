# PikkuFold artefacts

Code and computations accompanying *PikkuFold: Efficient Folding in a Few Kilobytes*.
Every table and every finite calculation in the paper is reproduced by something here.

```
pikku-fold-schemelet/        Rust implementation of one fold (benchmarks)
approximate-strong-sampling/ Rust sampler experiment (non-unit differences)
estimates.ipynb              Sage notebook: commitment ranks, knowledge error, proof sizes
ternary_jl_parameters.ipynb  Sage notebook: ternary-JL certificate and parameter generation
scripts/                     Sage scripts for the remaining tables
lattice-estimator/           submodule, pinned; used by estimates.ipynb
```

Clone with the estimator submodule:

```sh
git clone --recurse-submodules git@github.com:osdnk/pikku.git
```

## Rust

Both crates depend on the `rokoko` and `incomplete-rexl` crates from
`lattice-arguments/rokoko` over SSH, so building needs read access to that
repository. The ring arithmetic is AVX-512-only and both `.cargo/config.toml`
files build with `target-cpu=native`.

- `pikku-fold-schemelet/` — folding, the layered random projection, and the
  batched ring sumcheck. Produces the runtimes and communication of
  `tab:performance-runtimes`. `cargo run --release -- --log-m 10` is a smoke
  test; `cargo run --release` runs the configured size.
- `approximate-strong-sampling/` — samples pairs from the fixed-weight
  challenge set over almost-splitting primes and counts how often their
  difference is a non-unit, giving the observed `epsilon_C` of
  `tab:selected-almost-splitting-primes`. The committed `results-2pow24.csv`
  is the output used in the paper.

Each crate has its own README with the parameters it defaults to.

## Notebooks

Both notebooks run on a Sage kernel and are meant to be executed from this
directory.

- `estimates.ipynb` — for each ring degree and witness size, the smallest
  commitment rank reaching 129-bit classical SIS security together with the
  knowledge error and communication of one fold
  (`tab:estimated-proof-sizes`). It calls the lattice estimator from the
  pinned `lattice-estimator/` submodule and records the commit it used.
- `ternary_jl_parameters.ipynb` — the branch-and-bound certificate for the
  ternary Johnson–Lindenstrauss envelope, and the parameter generator for
  arbitrary security levels and row counts. Its final cell re-checks the five
  rounded parameter sets printed in the paper, and the modular inequalities
  they have to satisfy, at 200 bits of precision.

## Scripts

Run these with `sage scripts/<name>.sage`.

- `almost_splitting_table.sage` — the degree-128 almost-splitting sampler
  table: primality and order checks, weight selection, and the operator-norm
  distribution.
- `operator_norm_rejection.sage` — the operator-norm rejection cutoffs of
  `tab:operator-norm-rejection` in degrees 64, 128 and 256.
- `sampler_parameters.sage` — the degree-128, weight-32 invertibility error,
  in interval arithmetic.
