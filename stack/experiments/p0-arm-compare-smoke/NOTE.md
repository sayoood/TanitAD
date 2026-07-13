# Sample three-arm comparison — SMOKE (untrained tiny checkpoints)

This is the tested-harness output artifact for `scripts/compare_arms.py`, produced
end-to-end from **tiny smoke checkpoints** of all three arms on matched toy val
data (raw frames for flagship/REF-B; synthetic DINO features for REF-A, derived
from the SAME toy episodes/poses/ids). It demonstrates the harness produces a
well-formed, honest comparison table — **the numbers are meaningless** (flagship
smoke trained 12 steps, REF-A/REF-B 6 steps; features are synthetic).

What it demonstrates (the point of the smoke):
- ONE shared eval grid: identical trivial baselines + GT waypoints across arms.
- The rigorous parity metric (`D1 decode ade_0_2s`, frozen-probe) computed by the
  IDENTICAL code path on every arm's compact state.
- Per-arch trajectory decode, same metric/episodes: flagship & REF-A grounded
  operative rollout; REF-B direct BC waypoint head.
- Per-arch capability gating: D2/D3 imagination gates for flagship & REF-A;
  REF-B correctly N/A (no world model).
- Instrument doctrine live: D1 **FAIL** (admissible, ADE misses the 1.0 m camera
  bar), D2 **PASS**, D3 **BLOCKED** (untrained imagination fails the I4
  persistence instrument — the number is not a claim). Exactly the honest
  behaviour a real checkpoint must earn its way out of.

Reproduce (dev-box venv):
1. `python <scratch>/build_smoke.py <dir>`   (matched raw + feature val caches)
2. train tiny arms: `train_flagship4b.py --data toy --config smoke ...`;
   `refb_train.py --smoke ...`; `refa_train.py --adapter grid --smoke ...`
3. `python scripts/compare_arms.py --flagship-config smoke --refa-smoke --refb-smoke ...`

The real run uses `--flagship-config flagship4b`, real `--frame-cache-dirs` +
`--refa-feat-dir`, and the provisioned val subset (see
`stack/docs/phase0_go_criteria.md`).
