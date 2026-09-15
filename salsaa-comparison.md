# Comparing with SALSAA

How the SALSAA rows of `tab:performance-runtimes` (`evaluation.tex` of the paper) are produced
and how the input sizes are matched. Both implementations run single-threaded
on the same machine (one core of an Intel i7-11850H, AVX-512).

## SALSAA runs

Checkout: `~/salsaa-v2` (the v2 code, not `~/projects/salsaa`). Default
features (`p-26`, `incomplete-rexl`), nightly toolchain from its
`rust-toolchain.toml`:

```sh
cd ~/salsaa-v2
cargo +nightly run --release -- -m folding-scheme -r 11 -w 27   # matched to (k, m, phi) = (2, 2^18, 128)
cargo +nightly run --release -- -m folding-scheme -r 12 -w 29   # matched to (2, 2^20, 128)
cargo +nightly run --release -- -m folding-scheme -r 12 -w 31   # matched to (2, 2^22, 128): out of memory on 62 GB
```

The ranks were first checked with `--features debug-hardness` (SIS hardness
at least 128 bits for the chosen rank), then timed without it. The program
prints `TOTAL Commit time` (ns), `TOTAL Prove time` (ms), `Total proof size`
(KB) and `TOTAL Verify time` (ns).

## Matching the sizes

`-w L` gives a witness of `2^L` Z_q-elements split over 8 columns of equal
height: 4 accumulator columns and 4 fresh inputs. Our fold takes `k = 2`
fresh inputs of `m * phi` Z_q-elements each.

| ours `(k, m, phi)` | fresh Z_q-elements | SALSAA | per column | fresh Z_q-elements |
|---|---|---|---|---|
| (2, 2^18, 128) | 2 x 2^25 = 2^26 | `-w 27` | 2^24 | 4 x 2^24 = 2^26 |
| (2, 2^20, 128) | 2 x 2^27 = 2^28 | `-w 29` | 2^26 | 4 x 2^26 = 2^28 |

So the total fresh input matches, and one of our inputs holds as many
Z_q-elements as two SALSAA columns. SALSAA commits to all 8 columns in one
call; its commitment time per matched input is therefore a quarter of the
printed total.

## Numbers (2026-09-15)

| | (2, 2^18, 128) / `-r 11 -w 27` | (2, 2^20, 128) / `-r 12 -w 29` |
|---|---|---|
| SALSAA commit, 8 columns | 1653 ms | 7223 ms |
| SALSAA commit, per matched input (2 columns) | 413 ms | 1806 ms |
| ours, commitment per input | 191 ms | 803 ms |
| SALSAA prover | 7815 ms (paper: 7662) | 31950 ms (paper: 31668) |
| ours, prover | 455 ms | 1699 ms |
| SALSAA verifier | 0.41 ms | 0.42 ms |
| ours, verifier | 3.9 ms | 5.4 ms |
| SALSAA communication | 62.25 KB | 66.06 KB |
| ours, communication | 5.62 KB | 5.72 KB |

Prover times move by a few percent between runs; the paper keeps the values
of the run it was written from. Our figures come from
`pikku-fold-schemelet` (`cargo run --release -- --log-m 18|20`, sampled
key; `--features derived-key` at `--log-m 22`), rows `commit_input0_ms`,
`prover_total_ms`, `verify_total_ms`, `total_communication_kb`.
