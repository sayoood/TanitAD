# RESULT: PREREG_V8F_KIND (which kind of T's constraint cells is worth ADE6 to O), M2b s0 and s1

**OUTCOME, as committed: INTERACTIVE.**
* No kind clears 2F at both seeds. Speed clears it at s0 only, which Amendment 2 says is not a carrier.
* Additivity is A = −0.30 / −0.55, against the 0.8 bar.
* ⇒ The committed next step is O-side, with a stronger prior: **`PREREG_V8F_MIXO`** (train O on codes mixed per kind). T-side single-kind arms are deprioritised.

* **Evidence class:** MEASURED.
* **Tier: T0.** GT-conditioned oracles: a diagnostic of how O consumes T's code, never a driving claim.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, over 141 episodes (2,399 windows). It answers another draw of EPISODES. The seed replicate is the second banked checkpoint.
* **Raw:** `raw/2026-10-25-kind/run/kind_read.json`, 30 evals in `run/eval/`.
* **Run:** box 2, 07:17–07:32 on 2026-10-25, eval only, tip `009d1cba`. Checkpoint sha256s were asserted before and after the evals. The identity controls read exactly +0 [0, 0] at both seeds: own code against the banked eval, and the full oracle against the banked tacorc, both per window.

## 1. The committed reads (m of ADE6; paired against T's own code at the same seed)

| kind | gain_k s0 | gain_k s1 | necessity_k s0 | necessity_k s1 |
|---|---|---|---|---|
| speed | **+0.270 [+0.220, +0.323]** | +0.185 [+0.145, +0.226] | +1.985 [+1.772, +2.193] | +1.765 [+1.582, +1.962] |
| angle | +0.075 [+0.033, +0.130] | +0.043 [+0.011, +0.081] | +0.124 [+0.079, +0.180] | +0.154 [+0.101, +0.211] |
| tte | **−0.237 [−0.340, −0.140]** | **−0.315 [−0.434, −0.210]** | +0.457 [+0.385, +0.534] | +0.460 [+0.397, +0.526] |
| timegap | +0.001 ns | −0.000 ns | +0.010 [+0.002, +0.018] | +0.006 [+0.002, +0.011] |
| geom | **−0.434 [−0.580, −0.305]** | **−0.421 [−0.577, −0.286]** | +0.444 [+0.388, +0.500] | +0.611 [+0.535, +0.687] |
| leadgap | +0.005 ns | +0.002 ns | +0.008 [+0.001, +0.016] | +0.017 [+0.010, +0.025] |
| **union** (all 26 cells GT) | **+1.076 [+0.966, +1.195]** | **+0.923 [+0.811, +1.035]** | | |

* ADE6, s0 / s1:
  * T's own code: 2.175 / 2.055;
  * GT constraints with T's decisions (the union): 1.100 / 1.131;
  * the full GT code: 0.971 / 1.004.
* gain_k = ADE6(own) − ADE6(GT in kind k only). necessity_k = ADE6(GT everywhere except k) − ADE6(full GT).
* **The SPEED side-read, named in advance, decides nothing: NEITHER** (gain_speed 0.270 / 0.185, between the 0.05 and 0.20 bars at s1).

## 2. Amendment 1: by window stratum (beside, deciding nothing)

| stratum (windows / episodes) | speed s0 / s1 | angle s0 / s1 | tte s0 / s1 | geom s0 / s1 |
|---|---|---|---|---|
| JUNCTION (298 / 34) | +0.401 / +0.162 | +0.085 / +0.088 | −0.447 / −0.277 | +0.016 / +0.004 ns |
| CURVE (140 / 17) | +0.167 / +0.111 ns | **+0.770 [+0.118, +1.398] / +0.552 [+0.203, +0.887]** | −0.141 ns / −0.246 | +0.055 ns / −0.124 ns |
| NO-TURN (1,961 / 140) | +0.258 / +0.194 | +0.023 / −0.000 ns | −0.213 / −0.325 | **−0.538 / −0.506** |

All entries are separated unless marked ns (`kind_facts.out`). The per-stratum contributions to the pooled gain are in `kind_read.json` → `strata.contribution_to_pooled_gain_m`.

## 3. The four families (beside)

* **Tactical and strategic are identical across the oracle arms by construction.** The oracle replaces only the code O reads; T's decisions and S are the same arm's. Lateral macro-F1 is 0.816 and longitudinal 0.499; position error at 30 m is 7.02 m.
* **Lateral, the union against own at s0:** heading MAE −0.84° [−1.21, −0.50], separated.
* **Longitudinal, distance keeping (K = 10):** headway and time gap are in `kind_read.json` → `beside.*.longitudinal`.
* **Λ:** the union against the full oracle reads −0.036 [−0.075, +0.004], ns.

## 4. What it shows (EXPLORATORY; the decision is the table above)

1. **O cannot use a partly right code. This is the strongest evidence yet for the whole-code confound the PREREG named in advance.**
   * GT in the time-to-event cells alone makes O WORSE by 0.24–0.32 m. GT geometry alone makes it worse by 0.42–0.43 m, almost all of it in no-turn windows (−0.54 / −0.51).
   * Leaving only speed out of the full GT code costs about 1.8–2.0 m, worse than T's own code throughout (2.96 against 2.18 m at s0).
   * O reads the cells JOINTLY. A code with some cells true and some T's is a distribution it never trained on (`stack/tanitad/models/v8f_mini.py:620-622`).
2. **So the single-kind gains are lower bounds, and they cannot rank T-side levers.** That is exactly the case the PREREG routes to the O-side arm.
3. **One vision lead, for later:** GT angle cells are worth +0.77 / +0.55 m inside the 140 bend windows (17 episodes, wide intervals). Bends are where T's angle prediction costs O most. If MIXO makes O consume partial codes, a T-side angle/perception arm becomes readable.

## 5. Provenance

| artifact | where |
|---|---|
| the committed read | `raw/2026-10-25-kind/run/kind_read.json`; formatted by `run/kind_facts.{py,out}` |
| check-inputs, checkpoint sha256s | `run/check_inputs.json`, `run/ckpt_sha256.json` |
| the 30 evals (json, npz, log) | `run/eval/eval_M2b_s{0,1}-kind-*` |
| box-2 logs, guard, telemetry | `run/box2_logs/`, `run/telemetry_v8fkind.csv` |
