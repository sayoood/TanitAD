# R0d (SAFEOBJ, constrained EPO on g5) -- pre-registered outcome O4, read 2026-10-18

Package target: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-r0d-outcome/` (this file + `raw/`).
Pre-registration: `PREREG_SAFEOBJ.md` (sha256 `b404c2fb7b2f724775a1dda5a9de06c9778505a3374e6467060ae7c7a9e04bcb`) + `PREREG_SAFEOBJ.addendum1.md` (sha256 `3c6a25c14d4266230281b03d92d8201d21dbaa0ab27d9699f934af69a96761c4`), both verified by the tool before it ran.

## 0. Headline

* **ZZSAFEOBJ_OUTCOME R0d O4** ("BREACHED, DUAL HELD IN-SAMPLE"), flags: **none** (not SLACK). Rule applied: addendum R2 stage O4/O5; `ZZSAFEOBJ_OUTCOME R0d O4 precedence O6 > O7 > O4/O5 > O1/O2/O3: stage O45 matches`.
* **Bar S: FAIL.** **Bar G: NOT_READ** (the tool declines a G read for a killed arm; see 3). **L: PENDING** (no R1o). **E: PENDING** (no navtest read; arm killed).
* The breach is **g5, TTC only**. NC never breached. **g3 held at all four checks.** The K3 text says "g3/g5 bound breached"; it is generic. The breaching quantity is TTC.
* The dual held **in-sample**: signed window mean of the logged per-update violation g = **-0.000343 (TTC), -0.000287 (NC)**, bar <= 0.001. On the held-out RL-val the arm's TTC is **-0.767 [-1.487, -0.065]** against its own arm A. That is the O4 pattern (held on train, did not transfer), as pre-registered.
* **Stage 2 (R1d, R1o) does NOT run** (addendum A-O10: only if R0d reads O1). The lever claim L and the endpoint E are therefore not reachable under this protocol.
* **Next lever committed by the prereg for O4** (section 6): dual fed from a **log-disjoint carve of RL-train (never RL-val)**. First step, **0 A40**: measure the train-vs-carve gap on banked R0 states with the A2 harness. Then one arm; that arm is a PI decision.

## 1. Evidence class, tier, estimator

* Evidence class: **MEASURED** (ours, artifacts below). Rows marked INHERITED are quoted from the prereg, not re-read.
* **Tier: RL-val selection tier.** It is a lookup of the true NAVSIM v1.1 sub-scores of the scorer's argmax pick on held-out RL validation sets (3,000 sets / 75 logs), an open-loop pseudo-simulation. It is the arm's own instrument and, by `rl_val_eval.py`'s own tier text, "selection of hyper-parameters / early stops only, never an endpoint". It is **not T1**. No navtest (endpoint) read exists for this arm.
* **Estimator:** paired log-cluster bootstrap (cluster unit = log), 10,000 resamples, 95 % percentile interval, seed 20260927 (`proxy_eval.boot`, as in PREREG section 4). It answers "another draw of LOGS". It does **not** answer "another training run" (one seed per arm, no replicate exists) or "another inference run" (argmax scorer, deterministic up to kernels; the I-rep was not run). The kill-check booleans in `kill_checks.jsonl` are point deltas with no interval.
* Replicate status: **none.** R1o never runs, so no run-to-run floor was measured for this lever. Nothing below is a lever-effect claim (H-ESTIM-SEED-1).

## 2. Which bounds breached, per check (B - own A, RL-val, x100)

Point deltas and 95 % intervals from `rl_val_eval.py --base A` on the arm's own dumps at each tag (u250/u500/u750 added by me as descriptive reads; `kill_checks.jsonl` carries only the booleans, and they match).

| check | PDMS | TTC (g5 bound >= -0.5) | NC (g5 bound >= -0.5) | g3 pick/fan-mean (>= 0.894) | g3 shortest-bin (<= 0.55) | g3 | g5 | kill |
|---|---|---|---|---|---|---|---|---|
| u250 | +0.683 [+0.212, +1.153] | **-1.167** [-1.724, -0.568] | -0.183 [-0.460, +0.082] | 0.9864 | 0.3013 | ok | **TTC breach** | none (single) |
| u500 | +0.984 [+0.473, +1.509] | -0.500 [-1.189, +0.168] | -0.183 [-0.466, +0.101] | 0.9824 | 0.3180 | ok | ok | none |
| u750 | +1.402 [+0.843, +1.987] | **-0.767** [-1.514, -0.063] | -0.200 [-0.509, +0.068] | 0.9854 | 0.3070 | ok | **TTC breach** | none (first of the pair) |
| u1000 | +1.232 [+0.644, +1.822] | **-0.767** [-1.487, -0.065] | -0.233 [-0.532, +0.033] | 0.9847 | 0.3103 | ok | **TTC breach** | **K3** |

* u750 and u1000 are the two consecutive breaches that K3 kills on. u250 is a non-consecutive breach: it fails bar S and does not kill.
* ⚠ **u500 is a knife-edge pass.** TTC delta is exactly -0.500000 (15 sets net of 3,000; one set = 0.0333 points). The bound is "may not fall by > 0.5", so exactly 0.5 passes. A single set the other way would have made it a third breach.
* The "first breached check" the addendum uses for the O4/O5 window is **u750** (the first of the two consecutive breaches). `first_check_with_any_breach` is u250 and is not used by R3.

## 3. Outcome mechanics (all from `safeobj_outcome.json`; numbers carry their source in that file)

| quantity (addendum R3) | value | bar |
|---|---|---|
| kill | K3, why "g3/g5 bound breached at update750 and update1000", at 2026-10-18T01:59:02 | K3 -> stage O45 |
| O6 (lambda_TTC = lambda_NC = 0 at every logged update) | **false**; lambda > 0 on 941 of 1000 updates at u1000 | |
| O7 (non-K3 kill, or g3 alone) | **false**; kill rule K3; breach kind = g5 | |
| in-sample violation, signed mean of `con_g_ttc` / `con_g_nc`, window = updates 651..750 (four 25-update logged means) | **-0.000343 / -0.000287** | held iff both <= 0.001 -> **held** |
| post-ramp updates with lambda > 0, through u750 | **528 / 550 = 0.96** | >= 0.50 -> true |
| lambda_max reached | **no**: max lambda_TTC 0.5927, max lambda_NC 0.2448 (cap 10.0); cap hits 0 | O5 clause false |
| window overlaps the lambda_pg ramp (200 updates) | no | |
| unlogged tail | none (last logged update 1000) | |

* O4 matches, O5 does not. Neither "both match" (O4+O5) nor "neither match" (SLACK) applies.
* ⚠ **How thin the in-sample read is.** Each of the four window rows is itself a 25-update mean; the TTC rows across the whole run range about -0.0040 to +0.0026. The window mean is -0.0003 against a 0.001 bar, so "held" has margin on this window, but it rests on four logged means. Say so before quoting it as a strong in-sample hold.
* ⚠ **Bar G, read by hand, not the tool's verdict.** The tool prints `G[R0d] NOT_READ -- the arm was killed before u1000`. The u1000 dumps do exist (1,000 updates finished; K3 fired at the u1000 check), and I produced the registered input `rlval_B_vs_ownA.json`. By hand: t = -0.7667 is beyond dose.json's last vertex (-0.4667), so F is flat at +0.840554 and G(t) = +0.8406; the PDMS point +1.2316 is above it and the lower bound +0.6444 is > 0. G would read PASS **if it were read**. It decides nothing: R2 puts O4/O5 ahead of O1/O2/O3, and S is FAIL. It is disclosed so nobody rediscovers it as news.
* Disclosure A-G2 (own A vs C0-A, the "same run again" floor): **NOT COMPUTED.** C0's u1000 dumps are not on the pod. (For orientation only: A vs the base model reads PDMS -0.603 [-1.332, +0.082], TTC +0.733 [+0.034, +1.476]; B vs base reads PDMS +0.628 [+0.212, +1.027], TTC -0.033 [-0.581, +0.527]. Source `rlval_vs_A0.json`. The g5 breach is against an arm A that itself gained TTC over the base.)

## 4. The four metric families (as far as this arm's own reads carry them)

| family | measured here | not measured, and why |
|---|---|---|
| LONGITUDINAL | RL-val u1000 B - A: TTC -0.767 [-1.487, -0.065]; EP +3.412 [+2.681, +4.184]; g3 path ratio 0.9847 and shortest-bin share 0.3103 (A: 0.9248, 0.4087). Estimator above. | target-speed accuracy, headway / time-gap / TTC-to-lead: **NOT COMPUTED** (`rl_val_eval` lists "g4 longitudinal bias rows (need plan-vs-reference traces)" as not computed). Navtest RR3 component table: not run (arm killed, no endpoint read exists). |
| LATERAL | NAVSIM DAC +0.267 [-0.243, +0.744]; DDC **-0.917 [-1.455, -0.484]** (flagged adverse by the instrument; DDC is not a g5 term and does not enter bar S). These are NAVSIM sub-scores, not lateral-family metrics. | heading, curvature, yaw-rate, cross-track: **NOT COMPUTED** (PREREG section 8: `rl_val_eval` has no lateral instrument, n = 0). |
| TACTICAL | lane-change rate at u1000 0.0127 (38 of 3,000) against the label base rate 0.0213; precision 0.842, recall 0.500; K5 not breached at any check (`kill_checks.jsonl`). Counts, no interval. | lateral-decision kappa: navtest only, not run. |
| STRATEGIC | -- | **PAUSED** by PI ruling 2026-09-19 (PI_DECISION_QUEUE item 25). Reported as paused: not a pass, not omitted. |

* Per family, never pooled. No composite is reported.
* Context, INHERITED (PREREG section 1.1, citing `PK/raw/2026-10-15-rls1-adam-dose/ab_vs_base.json`, not re-read here): unconstrained R0 at u1000 reads PDMS +1.15 [+0.54, +1.75], TTC -1.00 [-1.73, -0.30], NC -0.33 [-0.69, -0.03]. R0d is +1.23 / -0.77 / -0.23. The TTC difference (+0.23) is far inside the overlapping intervals, comes from two different runs with no replicate, and is **not an effect** under the section 7 rule.

## 5. Evidence table (pod-relative paths; dev-box copies in `raw/`)

| what | path | sha256 (first 16) |
|---|---|---|
| kill record | `/workspace/data/rls1/arms/R0d/KILLED.json` | `6183c3e76f4277a4` |
| per-check booleans, g3/g5, lane-change | `/workspace/data/rls1/arms/R0d/kill_checks.jsonl` | `31865f51e5f8ad5030599c562b4b396c` |
| dual trajectory (lambda, g, cap hits) | `/workspace/data/rls1/arms/R0d/sft.jsonl` | `5380d8d1f61126b1` |
| bar S record | `/workspace/data/rls1/arms/R0d/safeobj_S.json` | `af56bfe3a08609da` |
| bar G input, u1000 B - own A | `/workspace/data/rls1/arms/R0d/rlval_B_vs_ownA.json` | `484aeb5d3eae7f39` |
| per-check B - own A (u250 / u500 / u750) | `raw/rlval_B_vs_ownA_u{250,500,750}.json` (pod: `/workspace/ops/r0d_read/raw/`) | -- |
| chain's own vs-base read (pre-empted by the kill) | `/workspace/data/rls1/arms/R0d/rlval_vs_A0.json` | `62a6b8f01816bf12` |
| outcome record | `/workspace/data/rls1/arms/R0d/safeobj_outcome.json` | `c0ce23950b73371b` (record_sha256 `9c15ac45856f53dd...`) |
| score dumps read (u1000) | `/workspace/data/rls1/arms/R0d/scores/scores_update1000_{base,A,B}.npz` | base `6332fcc6175805349d...`, A `e6bc8eb584b5d5d7...`, B `557ed17226e0eb42...` |
| tool + inputs, staged | `/workspace/ops/r0d_read/` | tool `e534e691d69f202b`; pod reader `b1b17341d2562ffd`; dose.json `735a6d909bc0ef41` |

## 6. Provenance notes, in plain terms

* **The outcome tool was not at the path the brief gave.** `/workspace/refe-plan-rls1/eval/safeobj_outcome.py` does not exist on the pod (the tree has `safeobj_read.py` only, and no `raw/` prereg or dose copies). I copied the tool from the dev-box package (`sobj_pkg/raw/2026-10-17-safety-objective/outcome/`, sha256 `e534e691d69f202b...`) into the scratch dir `/workspace/ops/r0d_read/`, together with the pod tree's **own** `safeobj_read.py` (sha256 `b1b17341...`, the "landed reader" the tool's test log names). Nothing was written into the launch-gate-bound tree, so its tree sha is untouched.
* **dose.json:** the working-tree copy under `D:\Projects\TanitAD` and `E:\Projects\TanitAD` hashes `6da7d64c...` because it has CRLF line endings from the Windows checkout (11,152 bytes). With CR stripped it is byte-identical to the registered file (10,500 bytes, sha256 `735a6d909bc0ef41...`). I used the LF copy. Anyone re-running from the Windows checkout will hit the tool's sha pin; the blob in git is the registered one.
* The S reader was run twice (first before `rlval_vs_A0.json` existed). The outcome tool requires `safeobj_S.json` to equal a fresh re-read, so the second run is the one on record. Its only remaining problem line is "kill_replay.json is missing" (a K3-killed arm never gets one, by design).
* Pod discipline: all reads CPU-only (`CUDA_VISIBLE_DEVICES` empty), `nice -n 10`, 2 threads each, about 20 s per read, detached with `setsid`. Nothing killed. `/workspace/ops/bank3c`, `/workspace/refe-plan-p6`, `/workspace/data/refe_p6` untouched.
* New files in the arm dir on the pod (`safeobj_S.json`, `rlval_B_vs_ownA.{json,log}`, `rlval_vs_A0.json`, `rlval.log`, `safeobj_outcome.json`) postdate the chain's HF egress (`ZZEGRESS_OK arm_R0d 28 files`), so they are **not on HF**. They are small and copied to the dev box here.
* The six `prereg_ambiguities` the tool lists (A-E3b, A-G2b, A-O9b, A-G3b, A-E1b, A-L1b) are the addendum's open readings. None bears on this outcome: A-O9b is moot (kill text is the K3 text, kill code K3), A-G2b is NOT COMPUTED above, the rest concern E / L.

## 7. What the pre-registration commits next

* O4 -> "dual fed from a log-disjoint carve of RL-train (never RL-val). First, 0 A40: measure the train-vs-carve gap on banked R0 states with the A2 harness; then one arm (PI)."
* Stage 2 (R1d, then R1o) **does not run.** L and E are unreachable under this protocol.
* Rule Zero: this refutation is a waypoint. The 0-A40 carve-gap measurement is unblocked and cheap. The arm that follows needs a PI decision (A40 spend).
* One cheap observation for whoever designs that arm, not a result: lambda peaked at 0.59 (TTC) against a cap of 10.0 and stayed below about 0.35 post-ramp (it fell back to 0.0 at some rows) while the in-sample g averaged below zero, so the dual was idle-to-light rather than saturated. That is what O4 (not O5) says; whether a carve-fed dual would push harder is the A2-harness question.

## 8. DRAFT register row for `GOALS_AND_CLAIMS.md` (id `D-REFE-SAFEOBJ-R0D`)

> **D-REFE-SAFEOBJ-R0D** -- SAFEOBJ arm R0d (constrained EPO, `--epo-con g5`, seed 0, vs its own arm A; RL-val selection tier). **Pre-registered outcome O4** (breached, dual held in-sample), no flags; K3-killed at the u1000 check (u750 + u1000 consecutive TTC breaches of g5). Bar S FAIL; G NOT_READ by the tool (killed arm; hand read would pass, decides nothing); L, E PENDING and unreachable (Stage 2 does not run, A-O10).
> MEASURED: u1000 B - own A (paired log-cluster bootstrap, 10,000 resamples, seed 20260927; answers "another draw of logs" only, one seed, no replicate): PDMS +1.23 [+0.64, +1.82], TTC -0.77 [-1.49, -0.06], NC -0.23 [-0.53, +0.03], EP +3.41 [+2.68, +4.18], DDC -0.92 [-1.45, -0.48]. g3 held at all four checks (pick/fan-mean path 0.982 to 0.986; shortest-bin share 0.301 to 0.318). Breach is g5-TTC only, at u250 (-1.17), u750 and u1000 (-0.77); NC never; u500 TTC is exactly -0.500 (knife-edge pass). In-sample dual: signed window mean g -0.00034 (TTC), -0.00029 (NC) over updates 651..750; lambda > 0 on 96 % of post-ramp updates; lambda max 0.59 of cap 10. Longitudinal/lateral/tactical partly measured; lateral kinematics and distance-keeping NOT COMPUTED; STRATEGIC PAUSED (PI 2026-09-19).
> Reading: the constraint held on the training data and not on RL-val; no lever effect is claimed (single seed, no R1o). Next lever (prereg O4): dual fed from a log-disjoint carve of RL-train (never RL-val); first the 0-A40 train-vs-carve gap on banked R0 states with the A2 harness; the arm after it is a PI decision.
> Artifacts: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-r0d-outcome/`; pod `/workspace/data/rls1/arms/R0d/`.

## 9. Deliverable manifest

| artifact | repo target | where it lives now |
|---|---|---|
| this file | `.../raw/2026-10-18-r0d-outcome/RESULT.md` | dev box `C:/Users/Admin/r0d_out/RESULT.md` (**only copy**) |
| raw JSON / logs / scripts | `.../raw/2026-10-18-r0d-outcome/raw/*` | dev box `C:/Users/Admin/r0d_out/raw/` and (arm-dir files) the pod arm dir; the u250/u500/u750 reads and scripts also at pod `/workspace/ops/r0d_read/` |
| `paths.json` | -- | dev box `C:/Users/Admin/r0d_out/paths.json` |
| staged tool + registered inputs | not for the repo (the tool is already in the dev-box package) | pod `/workspace/ops/r0d_read/` only |

Integration escalation: **the Master Mind lands this package** (stage via `paths.json`, scoped commit, push, then verify by `git ls-remote`). The R0d row of `GOALS_AND_CLAIMS.md` is a draft above; the A40 is idle with respect to this stream.
