# RESULT — the programme's FIRST parity T1, both arms, and what it may not be sold as

**Measured** 2026-08-27 (Thor, overnight) · **Tier** T1 (action-closed loop — the
doctrine's PRIMARY tier) · **Corpus** the registered parity val cache
(`physicalai-val-0c5f7dac3b11-w120-256x640cyl`, `--require-parity` passed) · 40
episodes × 6,924 windows per arm · **Estimator** episode-cluster bootstrap
(n_boot 2000) throughout · **Evidence class** MEASURED (raw:
`raw/t1_postrain30k.json`, `raw/t1_postrain30k_freeze.json`).

## The headline, stated the only admissible way

⛔ **Neither arm drives, and the numbers are the T1 face of the dissociation
blocker.** `_longitudinal_claim_admissible: False` for BOTH arms — holdv0 verdict
**LOSES_TO_HOLDV0** (worse than holding the current speed by ~9 m/s), copy detector
**CLEAN** both. Per the 2026-08-16 rule every longitudinal number below is a
**fidelity diagnostic, not a capability result.**

| metric (T1, pooled, CI95) | `postrain30k` (trained) | `postrain30k_freeze` |
|---|---|---|
| ADE dense (m) | **14.52** [11.55, 17.58] | **14.21** [11.23, 17.35] |
| speed bias (m/s) | **−9.91** | −9.67 |
| jerk RMS (m/s³) | **185.2** | **5.6** |
| accel MAE (m/s²) | 8.38 | 0.67 |
| heading MAE (°) | 103.7 | 98.1 |
| yaw-rate MAE (°/s) | **333.7** | 86.3 |
| cross-track MAE (m) | 1.40 | 1.70 |
| **S-curve masked** (§1.12 def, 57 S-windows) | **0.2632** [0.1404, 0.4182] | **0.0175** [0.0000, 0.0556] |
| decel / accel response ratio | 0.032 / −0.040 | 0.035 / 0.004 |

Reference points: the CV floor is **0.5352 m** and v1.x's closed-loop ADE was
**0.4714 m** — these arms sit **~27×** above the floor.

## The three readings that survive scrutiny

1. ⭐ **The rollout sheds speed to zero — exactly what a drift predictor must do.**
   Speed bias ≈ −(mean speed), ego-progress ratio ≈ −0.09/−0.04, response ratios
   ≈ 0.03. E-DEC-63 measured the predictor as ≈ drift beyond noise; **T1 is that
   measurement wearing its consequence.** The two tiers agree through entirely
   different instruments, which is the strongest internal consistency this
   campaign has produced.
2. ⭐ **The trained-vs-frozen S-contrast is real and separated** (CIs
   non-overlapping) — but its texture matters: the trained arm produces violent
   lateral variance (jerk 185, yaw-rate MAE 334°/s) that sometimes matches S-signs;
   the frozen arm is smooth and inert (jerk 5.6). **"Oscillates near the truth"
   vs "does nothing" — neither is driving**, and 0.2632 must not be quoted as
   closed-loop skill without this sentence attached.
3. ⚠️ **Cross-stack comparisons are architecture-confounded.** v1.x's 0.4714 m came
   from a decoder-conditioned stack purpose-built to output trajectories; these are
   19 M diagnostic WM arms rolling a grounding readout, never trained to drive
   (S-W stage, no planner). The right conclusion is **"the v7-tiny WM line does not
   yet drive at T1"** — not "v7 is worse than v1".

## Instrument state (what this run proves about the harness)

✅ End-to-end parity T1 works: strict load, cylindrical geometry, 40-episode dumps,
four families with CIs, anti-echo gate discharged with a MEASURED verdict, tactical
trajectory-derived block OK.
⛔ Named gaps, each a work item, none silently absorbed: **strategic** UNAVAILABLE
(needs map-derived option sets) · **distance-keeping** absent (`obstacle.offline`
not staged) · **sel_gap** UNAVAILABLE (the arm commits to one action per step — no
fan to gap; `selgap.py` names the work item) · **`paired_decision_grade` came back
EMPTY** — the tool computes pairing only when both arms are in one invocation;
the trained-vs-frozen S-contrast above rests on non-overlapping single-arm CIs,
and a proper paired bootstrap over the banked dumps is queued.

## Consequence

The L-ladder position (`V7_RECIPE_AND_SCALEUP.md` §8): the v7-tiny line stands at
**L3 unresolved / L4 measured-and-failed** — which is the honest baseline the
scaled run must beat, now with the pre-registered instrument that will detect it.
The lever remains as E-DEC-63 sharpened it: representation first; a planner/decoder
stage before any T1 capability claim is expected of the WM line.
