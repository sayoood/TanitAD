# RESULT — the route fix is REAL and INERT. **OUTCOME B**, as pre-registered.

**Pre-registration: `PREREG_thr035_ade.md`, committed `3faab6c` BEFORE this arm was run.**
MEASURED 2026-07-29, pod2. Same checkpoint (step 29,999), same parity val, same
**881 windows / 40 episodes** — `eid` sequence and ground truth verified identical before any
comparison. **Estimator: paired episode-cluster bootstrap** (`taniteval.ci`, B=2000, seed 0).
`overlapping_holdout_se` used nowhere.

## 1. The pre-registered primary

| metric | thr 0.35 | default 0.7616 | **Δ (thr035 − default)** | CI95 | |
|---|---|---|---|---|---|
| **`ade_0_2s`** | 0.8585 | 0.8563 | **+0.0022** | **[−0.0008, +0.0055]** | **overlaps 0** |
| `fde@2s` | 1.7429 | 1.7378 | +0.0050 | [−0.0008, +0.0115] | overlaps 0 |
| `long_abs_2s` | 1.4717 | 1.4649 | +0.0068 | [−0.0002, +0.0151] | overlaps 0 |
| `lat_abs_2s` | 0.5508 | 0.5515 | −0.0007 | [−0.0027, +0.0009] | overlaps 0 |

**OUTCOME B.** Not separated on any metric, and the sign is nominally *worse*.

## 2. ⭐ This is a PRECISE null, not an underpowered one — which is what makes it useful

The paired design leaves the ADE interval at **±0.0055 m**. So the experiment does not merely fail to
find an effect; it **bounds** it:

| quantity | value |
|---|---|
| oracle-vs-produced `ade_0_2s` gap (the thing to explain) | **0.2140** |
| largest ADE improvement compatible with this CI | **0.0055** |
| ⇒ share of the gap that route recalibration can explain | **≤ 2.6 %** |

⇒ **THE HEAD IS ESSENTIALLY INSENSITIVE TO THE ROUTE CHANNEL.**

## 3. The threshold change definitely worked — this is not a null run

Guarding against the obvious alternative explanation (that the override silently did nothing):

| | default | thr 0.35 |
|---|---|---|
| `route_exact_agreement` | 0.5085 | **0.5664** |
| balanced route accuracy (sweep) | 0.4242 | **0.5493** |
| right-turn recall | 0.041 | **0.289** |

The eval also printed `⚠ ROUTE THRESHOLD OVERRIDDEN to 0.35`. **The route got substantially better
and the trajectory did not move.** Both halves are measured.

## 4. What this retires, and what it promotes

⛔ **RETIRED: "fix the route head" as the program's headline goal-side work item.** It was promoted
on the strength of a route-metric collapse; the collapse is real, the repair is real and free, and
**it buys nothing downstream**. Pre-registered Outcome B said exactly this would retire it, so it is
retired rather than argued around.

⭐ **PROMOTED: the SPEED channel.** The oracle-vs-produced gap is longitudinal — paired
`long_abs_2s` +0.4260 vs `lat_abs_2s` +0.0274 — and the surviving suspects are the speed-side goal
channels, not route:

| channel | quality |
|---|---|
| `vt_band` exact agreement | **0.1725** (within-1 0.3837, 23 bands) |
| `tspeed_5s` | R² 0.7635 but **RMSE 4.4545 m/s (~16 km/h)** |

That matches the program's standing longitudinal-blindness finding and the harness's own
`tracks speed > CV: False`.

## 5. Honest limits of this result

1. **This tests the route channel at ONE alternative threshold (0.35, the sweep optimum).** It does
   not prove the head ignores route under *every* manipulation — a differently-shaped route signal
   (e.g. graded rather than 3-way, or a route the head was *trained* to rely on more) is untested.
2. **It says nothing about closed-loop.** All four metrics here are open-loop at 2 s. The corridor
   co-primary was never run on this arm.
3. **The 0.35 value remains eval-split-selected.** That limitation is now moot for deployment — there
   is nothing to deploy — but it would return if route is revisited.
4. **`--route-thr` stays OFF by default**, so no published number moves.

## 6. Provenance

`code/paired_thr.py` · `raw_paired_thr035.json` · `raw_modeB_thr035_v4fs-30k-produced-thr035.json` ·
pre-registration `PREREG_thr035_ade.md` (`3faab6c`). Arm: `--route-thr 0.35`, everything else
identical to the published produced-goal arm.
