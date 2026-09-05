# The inference-seed replicate reads **SEPARATED on four rows** — with zero levers moved

**MEASURED 2026-09-05, zero GPU**, `raw/pd_fact.md` / `raw/pd_fact.json`
(tool `raw/gm_paired_delta.py`). Paired episode-cluster bootstrap, n = 40 windows /
8 clusters, `n_boot` 2000. Known-value control (an arm against itself) reads
**+0.0000 [+0.0000, +0.0000]** on every metric — the estimator is behaving.

## The measurement

`ccos_seed1` is `ccos_argmax` with **one token changed**: `--plan-seed 0 -> 1`.
Same checkpoint, same windows, same cost, same vocabulary, **no lever moved**. Its
paired deltas against `ccos_argmax`:

| metric | delta [CI] | separated? |
|---|---|---|
| `ade_m` | **+0.0607 [+0.0088, +0.1155]** | **YES** |
| `fde_m` | **+0.3439 [+0.0230, +0.7906]** | **YES** |
| `LAT_cross_mae_m` | **+0.0710 [+0.0101, +0.1412]** | **YES** |
| `LAT_heading_mae_deg` | **+1.1180 [+0.3304, +2.0778]** (n=33) | **YES** |
| `LON_speed_mae_mps` | +0.0038 [−0.0003, +0.0074] | no |
| `LON_accel_mae_mps2` | +0.0061 [−0.0001, +0.0118] | no |
| `LAT_yaw_rate_mae_radps` | +0.0081 [−0.0061, +0.0251] | no |
| `TAC_traj_lat_correct` | −0.0750 [−0.1750, +0.0000] | no |

⛔ **Four of eight rows are "separated" for a difference that has no cause.** This is
`CLAUDE.md`'s third-variance rule — *a separated CI from a one-seed arm is necessary,
not sufficient* — **measured on refav1 itself**, not inherited from the v7-tiny rig
where `H-ESTIM-SEED-1` was established. The mechanism is the documented one: the
episode-cluster bootstrap resamples **episodes with the models held fixed**, so it
answers *"would another draw of EPISODES say this?"* and is structurally blind to the
question a stochastic planner forces — *"would another INFERENCE RUN say this?"*
iCEM samples, so that second question has a non-zero answer, and here it is large
enough to clear the first question's interval on half the rows tested.

## Why this is sharper than the general rule

**Separation and magnitude disagree here, in BOTH directions, on the same panel:**

| contrast | `ade_m` | separated? | vs the 0.0607 seed floor |
|---|---|---|---|
| `ccos_seed1` − `ccos_argmax` (**no lever**) | +0.0607 | **YES** | 1.0x — it *is* the floor |
| `kamm07` − `ccos_argmax` (the cap) | −0.3344 [−0.9536, +0.0625] | **no** | **5.5x the floor** |
| `wk15` − `ccos_argmax` (W_KAPPA) | −0.4338 [−1.0393, −0.0808] | **YES** | 7.1x the floor |
| `combined` − `ccos_argmax` | −0.2768 [−0.8473, +0.0942] | **no** | 4.6x the floor |

⇒ A rule of *"report what is separated"* would have **banked the seed difference and
discarded the cap's 5.5x-floor effect.** Neither test alone is sufficient:
**separation without magnitude admits noise; magnitude without an interval admits a
single lucky draw.** ⭐ **The admissible form on this rig is BOTH — a separated
interval AND a delta that exceeds that metric's own measured seed floor** — which is
exactly what `raw/SPEC_BEST_AND_SEED.md` gate **A** committed to before any of these
numbers existed.

⚠️ **And the floor is per-metric, from the artifact.** `LAT_heading` moves **1.1180 deg**
under a seed change alone; `LON_speed` moves **0.0038 m/s**. A single remembered
scalar cannot serve both — the failure this package already retracted once.

## The substantive result it enables

With the floor applied per metric, the **longitudinal** column separates the two levers
cleanly — and this is interval-backed, not a point estimate:

| contrast | `LON_speed_mae` | `LON_accel_mae` |
|---|---|---|
| `wk15` − `ccos_argmax` (W_KAPPA) | **+0.0764 [+0.0135, +0.1453]** separated WORSE | **+0.0874 [+0.0122, +0.1763]** separated WORSE |
| `kamm07` − `ccos_argmax` (the cap) | −0.0017 [−0.0078, +0.0034] | +0.0013 [−0.0035, +0.0064] |

⇒ **The cap costs the longitudinal family nothing, with tight intervals that exclude
any meaningful cost; `W_KAPPA` is separated-worse on both of its metrics.** Since
88.7 % of the programme's oracle gap is longitudinal, that is not a footnote — it is
the reason the cap outranks `W_KAPPA` as the lever to carry forward.

⚠️ **`l3ladder` − `ccos_argmax` reads `ade_m` −0.0036 [−0.0569, +0.0647]** — the ladder
alone does essentially nothing, independently confirming `D-REFAV1-CG-L3-NULL` on the
paired estimator rather than on per-arm point values.
