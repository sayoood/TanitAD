# ⭐ The SPEED goal channel is 88.7 % of the oracle advantage. OUTCOME A.

**Pre-registered `PREREG_speed_ceiling.md`, committed `0aba041` BEFORE this arm existed.**
MEASURED 2026-07-29 on **pod3** (v5 on pod2 untouched). **881 windows / 40 episodes**, paired
episode-cluster bootstrap (B=2000, seed 0), `overlapping_holdout_se` nowhere.

## 1. The result

| metric | mixed (oracle speed) | produced | **Δ** | CI95 | |
|---|---|---|---|---|---|
| **`ade_0_2s`** | 0.6664 | 0.8563 | **−0.1899** | **[−0.2499, −0.1371]** | **SEPARATED** |
| `fde@2s` | 1.3810 | 1.7378 | −0.3568 | [−0.4700, −0.2557] | SEPARATED |
| **`long_abs_2s`** | 1.0803 | 1.4649 | **−0.3846** | [−0.4965, −0.2861] | SEPARATED |
| `lat_abs_2s` | 0.5503 | 0.5515 | −0.0012 | [−0.0222, +0.0179] | **overlaps 0** |

The arm takes **`vt_band` + `vt_speed` from the oracle** and leaves every other channel produced.

## 2. ⭐ The gap is now almost fully attributed

| component | share of the 0.2140 oracle-vs-produced gap |
|---|---|
| **speed channels** | **0.1899 = 88.7 %** ✅ separated |
| route channel | **≤ 2.6 %** (measured separately: +0.0022 [−0.0008, +0.0055], not separated) |
| everything else (interactions, `route_graded` continuous, unmodelled) | ~9 % |

⇒ **The oracle's advantage over the produced goal is, to first order, ONE THING: it knows the
target speed.**

## 3. ⭐ And it is purely longitudinal — the mechanism is visible, not inferred

`long_abs_2s` improves **−0.3846 (separated)** while `lat_abs_2s` moves **−0.0012 and overlaps
zero**. Speed MAE 0.7507 → **0.5629** (hold-v0 floor 0.4818). The speed goal repairs the
longitudinal axis and **provably touches nothing lateral** — exactly what the Frenet decomposition
predicted, now confirmed by an intervention rather than an association.

## 4. ✅ A free validation that strengthens everything above

`produced40` on pod3's **40-episode subset cache** reproduces **`ade = 0.8563`** — *identical* to the
full-600-episode-cache run on **pod2**. A cross-host, cross-cache reproduction of the baseline, which
means the subset shortcut introduced no bias and the two arms really are on the same windows (the
paired script additionally refuses unless `eid` and ground truth match exactly — they did).

## 5. What this licenses, and what it does not

✅ **`tspeed_5s` is confirmed as the binding constraint on the goal side, and the prize is
quantified: up to 0.1899 m of ADE.** That is a large, well-bounded target for a head of ~8.4 M
parameters currently at R² 0.7635 / RMSE 4.4545 m/s (~16 km/h).

⛔ **This is a CEILING, not an achievement.** The arm is fed a future-derived quantity by
construction and is **not deployable**; the harness records `oracle_channels_substituted` in the
artifact so it can never be mistaken for one. A real `tspeed_5s` improvement captures only the
fraction of this that better estimation actually reaches.

⚠️ **It does not say the head can be fixed** — only that fixing it is worth up to 0.1899. §1's
sibling result stands: the failure is *estimation*, not discretisation (RMSE spans ~4.1 bands), so
there is no cheap trick here.

⚠️ **Open-loop 2 s only.** Nothing here speaks to closed-loop or corridor behaviour.

## 6. Provenance

`code/run_ceiling.sh` · `code/paired_ceiling.py` · `raw_paired_ceiling.json` ·
`raw_v4fs-{mixed40,produced40}.json` · pre-registration `PREREG_speed_ceiling.md` (`0aba041`).
Capability added for this: `--oracle-channels` (per-channel oracle substitution), documented in-code
as diagnostic-only.
