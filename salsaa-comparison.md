# Comparing with SALSAA

How the SALSAA rows of `tab:performance-runtimes` (`evaluation.tex` of the paper) are produced
and how the input sizes are matched. Both implementations run single-threaded
on the same machine (one core of an Intel i7-11850H, AVX-512).

## SALSAA runs

Checkout:
```sh
git clone git@github.com:lattice-arguments/salsaa-v2.git
```

Run tests:

```sh
cd ~/salsaa-v2
cargo +nightly run --release -- -m folding-scheme -r 11 -w 27   # matched to (k, m, phi) = (2, 2^18, 128)
cargo +nightly run --release -- -m folding-scheme -r 12 -w 29   # matched to (2, 2^20, 128)
cargo +nightly run --release -- -m folding-scheme -r 12 -w 31   # matched to (2, 2^22, 128): out of memory on 62 GB reference machiine
```

The ranks were first checked with `--features debug-hardness` (SIS hardness
at least 128 bits for the chosen rank), then timed without it. The program
prints `TOTAL Commit time` (ns), `TOTAL Prove time` (ms), `Total proof size`
(KB) and `TOTAL Verify time` (ns).

