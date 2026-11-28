# MM reading: STRAT1 CKPT120 read (2026-11-18 ~00:15Z)

Evidence class MEASURED (ours, zero training). Tiers: STRATEGIC is T0 at the window's first tick; driving cells are T1-kin. Estimator: paired episode-cluster bootstrap B 2,000 seed 0 ("another draw of EPISODES"). The training-run floor F120 is |H1_rep - H1_s0| on the same 120 m cell. Card `PREREG_V8F_CKPT120.md`, frozen before training. Result `out/RESULT_CKPT120.{json,md}`, State FINAL, problems 0, all 7 arms READ with K-REACH-e PASS.

⚠️ **Wording rule (card and CLAUDE.md):**
* Every STRATEGIC number is "with the route supplied along the driven path".
* On PhysicalAI the only route supplier is the ego's own future path, so a supplied route is optimistic by construction.
* These numbers measure route FOLLOWING given a route. They do not measure route choice.

## Outcomes as committed

| rung | route following 120 m | route setting 120 m | route cost 30/50/80 m | driving |
|---|---|---|---|---|
| H1 | **S120-PARTIAL** | FAR-SETS | none | KEEPS |
| H2 | **S120-FOLLOWS** | FAR-SETS | none | SEEDS-DISAGREE |

**H2 follows the supplied route at 120 m at both seeds** (341 windows / 27 episodes):
* real 0.927 / 0.894 against donor 0.182 / 0.199;
* real - donor +0.745 [+0.647, +0.824] / +0.695 [+0.599, +0.778];
* balanced - 0.5 +0.435 / +0.392, separated;
* raw (real minus the best constant) +0.158 [+0.010, +0.344] at s0, separated, and +0.126 [-0.020, +0.306] at s1, not separated. The card's rule needs the raw POINT above 0 and above F120 (0.0147) at both seeds, and it is. **STRONG** (raw separated at both seeds) is not reached.
* It holds where H1 fails: in the stratum where the route token does not reach S, H2 real - donor is +0.730 / +0.689, separated, while H1 is +0.107 / +0.098, not separated. In the far stratum (60-100 m) H2 is +0.711 / +0.680, separated.

**Route setting at 120 m** (position error relative to the extrapolation baseline, negative = closer to the driven route):
* H2 -0.533 [-0.969, -0.113] / -0.569 [-0.975, -0.223], separated at both seeds;
* H1 -0.403 / -0.460, separated.

**Driving (24 T1-kin cells):**
* H1 KEEPS.
* H2 SEEDS-DISAGREE: one cell (tactical 2-6 s accel MAE) costs at one seed only, and nothing costs at both.
* The C0 replicate read measured 5 of 48 single cells separating between two runs with identical flags, so one cell at one seed is within that rig noise. It is reported as the label reads.

**Controls:**
* K1 0.0; K7 and K8 pass; K-PERFECT matches the card to 3 decimals; K-NAVINF matches the card; K-LABEL is 341 / 27 (79 left / 262 right), matching the card.
* R1 (route cut) reads S120-IGNORES, with far position error +12.1 m worse than H1, separated.
* R2 (token off) loses following (H2 - R2 +0.557, separated).
* The regression arms fail as they must.

**Instrument fix this run (MM):**
* The first reader run stopped at the stack-shadowing guard. The frozen otband `read_lib`, imported lazily, put the STRAT1 tree's stack first on `sys.path`.
* `c120_read.py` `load_tree_modules` now imports `tanitad.refs.refc_tactical` from the CKPT120 tree while that tree leads the path, and refuses otherwise. This cannot move a number: `refc_tactical.py` and `tanitad/__init__.py` are byte-identical in both trees, which differ only in 3 model and data files.
* Reader tests: 29 passed. The original is kept beside as `c120_read.before_guardfix.py` (banked here).

## Next (card row S120-FOLLOWS; Rule Zero)

1. **The T-reads-far card:** S's 120 m point becomes a fourth checkpoint token of T. The card itself names it PI go. The MM pre-registers it now, with cost, so it is ready for the PI's decision.
2. **The derived-goals card.** It is pre-registered alongside (PI horizon ruling 2026-11-08: derived goals).
3. H1's shortfall routes to H2 by the card, and H2 is read here, so H1 has no separate open lever.
