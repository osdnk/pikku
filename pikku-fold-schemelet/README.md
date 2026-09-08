# Pikku fold schemelet

Draft: folding, the layered random projection, and one batched ring
sumcheck proving the projection claims together with the evaluation
claims.

Defaults:

- `m = 2^20`
- `rank = 12`
- Rokoko prime `q = 1125899906839937`
- two fresh inputs plus one accumulator
- witness coefficients sampled uniformly from `[-2^10, 2^10]`
- commitment ranks account for 32 sequential folding rounds
- unstructured dense commitment key sampled uniformly
- fold challenges use the degree-128 fixed-weight sampler row
  `s = 23, gamma <= 8.357` from `easy_sampler.tex`, sampled with Rokoko's
  parametrized `sample_fixed_weight_challenge_into`
  (lattice-arguments/rokoko#90)

The fresh witness columns are projected through three transcript-sampled
biased-ternary JL layers of 256 rows each: two coarse ring-level layers with
roughly balanced shrink ratios, then the fine coefficient-level layer with
`256 * degree` columns. The prover sends the coefficient projection
`v_tr` (checked against the JL upper-tail norm bound), the verifier samples
tensor-structured batching points, and the prover answers with the batched
fine projections, checked through the trace-duality identity
`const_coeff(<j_batched, v_2>) = <c_tensor, v_tr>`.

The batched projection values are then proven by the same ring sumcheck
that uniformizes the evaluation claims: the batched claim runs over all
layer boundaries (output 8 + middle + witness variables), the evaluation
claims are padded by the prefix `(1 - X_j)` factors and handled
analytically until the witness boundary, and the prover processes one
boundary at a time, contracting each bound layer into the next using the
already-computed intermediate projections. The verifier evaluates the
layered matrices at the terminal point from their unexpanded bit-plane
descriptions in `O(S_JL)` and never materializes a table of witness size.

Each fresh input and the accumulator carries its own random evaluation
point (sampled from the diagonally embedded `F_{q^2}`) with the claimed MLE
value in the statement. The prover batches the `k + 1` claims with subfield
batching challenges (accumulator coefficient fixed to 1), runs the ring
sumcheck over `log2(k m)` variables with `F_{q^2}` round challenges and
NTT-slot batching, and reveals the per-column terminal evaluations. The
verifier checks every round polynomial and the de-batched terminal value,
then both parties fold. The prover runs the witness rounds over `F_{q^2}`:
the evaluation points, batching challenges, layer eq tables and round
challenges are all diagonal ring elements and commute with slot batching, so
the witness columns are slot-batched once (the ring factor `t_1(r_2)` of the
projection term is folded into the batching vector) and the per-column ring
terminal values come from one direct pass over the witness. Then both
parties fold: the folded instance carries the terminal sumcheck
point and the fold-challenge combination of the terminal evaluations. The
final verification of the folded relation (`output.rs`) recomputes the
folded commitment, checks the folded witness norm against
`beta_acc + k * gamma * beta_in`, and re-evaluates the folded MLE claim; it
is reported separately from verifier timing.

Run a smoke test:

```sh
cargo run --release -- --log-m 10
```

Run the configured size:

```sh
cargo run --release
```
