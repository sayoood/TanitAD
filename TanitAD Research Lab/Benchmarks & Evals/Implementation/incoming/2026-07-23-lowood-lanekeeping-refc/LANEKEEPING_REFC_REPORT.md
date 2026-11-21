# Low-OOD closed-loop LANE-KEEPING instrument + REF-C 2nd arm — the first absolute (confound-free) closed-loop comparison of the two main arms

**Date:** 2026-07-23 (Berlin) · **Host:** `tanitad-eval` (RTX A6000 46 GB, GPU free, `gpu_lock=lowood-lanekeep`) ·
**Author:** lowood-lanekeeping-refc subagent · **Status:** P1 + P2 + P3 all MEASURED and banked; pod released, GPU idle.

**Evidence-class discipline (CLAUDE.md).** Every number below is `MEASURED` with its artifact path
(`lowood_lanekeep_40ep.json`), unless tagged `INHERITED` (registry / another agent / the OOD envelope, not
re-verified here). Decision-grade intervals are the **episode-cluster bootstrap** over the 40 val episodes
(`taniteval/ci.py`, `n_boot=2000`); the two arms are compared with the **paired** version on the identical
windows. Read `RETRACTION_LOG.md` classes C5 (n=1 headlines) and C6 (confounded comparison) before quoting.

---

## Executive summary

The Gate-1 clean run correctly HELD on two MEASURED bounds: the real-footage low-OOD source is
**map-free / agent-free**, so it can only measure drift, not off-road/collision; and there are too few real
junction episodes for a memorization-free fine-tune. This work builds the two cheap, independently-valuable
unblocks that hold flagged — **without re-running the fine-tune**:

1. **P1 — a low-OOD LANE-KEEPING metric.** `corridor_departure_rate` = the fraction of on-policy steps whose
   cross-track error |XTE| exceeds a lane half-width (primary **1.75 m** = half a 3.5 m lane). Plus peak XTE,
   time-to-departure, and a per-window departure rate, over a 1.0 / 1.75 / 2.5 m threshold sweep. This is the
   lane-keeping proxy for the junction failure that IS measurable at low OOD (unlike off-road/collision).
   **tick-0 self-check holds for BOTH arms** (on-path → |XTE| = 0 → zero departures; asserted in-harness).
2. **P2 — the REF-C arm (finally unblocked).** REF-C base 104.2 M (ckpt md5 `8f10d6f934f4199e11ddc7352e074939`,
   **matches Gate-1 `a9147f0e`**) wired into the low-OOD harness via its **deployed** anchored-diffusion decode
   `model(fw, nav_cmd=None, v0, steps=2)["traj"][:,0]`. It sees the **exact same** warped real-footage window
   the flagship sees — the val cache is the canonical phase-0 f-theta stack both arms trained on, so the
   f-theta canonical input contract is met by construction (no re-canonicalization, cf. `refc_driver.py`).
3. **P3 — the comparison.** Both arms on the **40-ep** clean val (`physicalai-val-0c5f7dac3b11`, **881 windows**),
   identical windows, episode-cluster-bootstrap CIs + a paired flagship-vs-REF-C test.

**Result — REF-C base decisively out-drives flagship v1 at low OOD, on BOTH metrics, paired CI excludes 0:**

| overall (881 win / 40 ep) | flagship v1 | REF-C base | paired Δ (flag − refc) | separated |
|---|---|---|---|---|
| closed-loop ADE@2s (m) | **1.488** [1.329, 1.647] | **0.564** [0.452, 0.676] | **+0.924** [+0.781, +1.065] | **yes** |
| corridor_departure_rate @1.75 m | **0.0318** [0.0152, 0.0531] | **0.0134** [0.0059, 0.0223] | **+0.0184** [+0.0077, +0.0328] | **yes** |
| peak XTE (m) | 0.764 [0.530, 1.060] | 0.442 [0.314, 0.585] | +0.321 [+0.193, +0.495] | yes |
| window_departure_rate @1.75 m | 0.135 | 0.078 | — | — |
| on-policy OOD peak ratio | 1.054 [1.037, 1.073] | 1.035 [1.022, 1.050] | — | — |

**Two things make this decision-grade rather than another n=1 headline (RETRACTION_LOG C5):**

- **It removes the confound the AlpaSim result carried and REPRODUCES its direction.** The 07-23 AlpaSim n=12
  paired suite found REF-C base > flagship v1 closed-loop, but flagged it confounded by **reconstruction-OOD
  (~3.2×)** (RETRACTION_LOG 07-22/07-23). This instrument runs on **real footage** — both arms stay at
  **1.02–1.20× OOD on-policy** (vs NuRec's flat **3.75×**) — and reproduces the **same ordering** at **n=40**.
  Measured through a *completely different* instrument, the ordering is not an artifact of reconstruction fidelity.
- **The metric separates the two failure modes, mechanistically.** See below.

**HONEST FRAME (do not overclaim).** This measures **lane-keeping / on-policy drift at low OOD**, explicitly
**NOT off-road departure or collision** — the map-free / agent-free source is structurally unable to emit those.
A real off-road/collision rate needs AlpaSim (map + reactive agents + low-OOD renderer) = the ~3.2×-OOD
instrument this source exists to escape. The "corridor" is the recorded ego path ± a lane half-width, not a
mapped lane. This is a within-source **relative** planner comparison, not a real-world safety rate.

---

## 1. P1 — the corridor / lane-keeping metric

**Definition.** XTE (cross-track error) is the harness's signed lateral offset of the on-policy ego from the
recorded path (`dlat`, left-positive). `corridor_departure_rate` := the fraction of the K=20 on-policy steps
(0–1.9 s) whose |XTE| exceeds a lane **half-width** θ, averaged per window and bootstrapped over episodes.
Companions: peak XTE (m), time-to-departure (s to first |XTE|>θ), and window_departure_rate (fraction of
windows that *ever* leave the corridor).

**Threshold, cited.** θ = **1.75 m** = half of a **3.5 m lane** — the common design width for US arterials /
German Autobahn lanes (US Interstate 3.6 m; German RAA Autobahn 3.5–3.75 m; urban 3.0–3.5 m). |XTE| > 1.75 m
means the ego reference point has crossed from lane centre to lane edge — a genuine lane departure. Companion
thresholds bound the reading: **1.0 m** (with a ~1.8 m-wide vehicle centred, ~0.85 m of edge clearance, so
|XTE| > 1.0 m already puts a wheel over the line) and **2.5 m** (well into the adjacent lane / off a narrow road).

**tick-0 self-check (both arms).** At k=0 the ego is on-path → |XTE| = 0 → no threshold exceeded → corridor
departure is **exactly 0** for each arm. Asserted in the harness (`max_lat = 0.0`, `n_dep = 0`); both arms passed.

**Threshold sweep (overall).**

| corridor_departure_rate | @1.0 m | @1.75 m | @2.5 m |
|---|---|---|---|
| flagship v1 | 0.0661 | 0.0318 | 0.0163 |
| REF-C base | 0.0366 | 0.0134 | 0.0039 |

REF-C is lower at every threshold; the flagship's departure rate is ~1.8–4× REF-C's across the sweep.

---

## 2. P2 — the REF-C arm (C6-clean parity)

Both arms are driven by their **deployed** planner head, fed the **identical** warped real-footage window each
tick, through the **shared** controller (0.5 s pure-pursuit → kinematic bicycle, verbatim `closedloop.py`):

| | flagship v1 | REF-C base |
|---|---|---|
| ckpt | `flagship-30k` step 29999 | `refc-base-30k` step 29999 (md5 `8f10d…4939`) |
| planner head | `strategic_policy(nav=follow)` → `tactical_policy` → `waypoints[5]` (0.5 s) | `model(fw, nav_cmd=None, v0=ev, steps=2)["traj"][:,0]` (0.5 s) |
| nav intent | follow (idx 0) | `nav_cmd=None` → follow (idx 0) — **identical** |
| input | canonical phase-0 f-theta stack [W=8, 9, 256, 256] ∈ [0,1] — **identical warped frame** | same |

**Named differences (C6).** (i) The planner *heads* differ — that is the object of the comparison. (ii) REF-C
consumes the current speed `v0` explicitly (its measurement encoder), the flagship tactical head does not; each
arm receives its natural inputs. (iii) REF-C decodes at `steps=2` (its deployed truncated-diffusion mode).
Everything else — frames, warp, nav, controller, windows, val set — is byte-identical between arms.

---

## 3. P3 — the comparison, by scene

Strata: **junction** = |net heading change@2s| ≥ 10°; **longitudinal** = non-junction ∧ speed ≥ median.
(Not a partition — the remaining non-junction low-speed windows are in neither.)

### 3.1 Longitudinal (374 win / 24 ep) — the flagship's dominant 89 %-failure mode

| | flagship v1 | REF-C base | paired Δ (flag − refc) | sep |
|---|---|---|---|---|
| closed ADE@2s (m) | 1.455 [1.271, 1.642] | 0.354 [0.262, 0.473] | **+1.101** [+0.906, +1.284] | yes |
| corridor_departure @1.75 m | 0.0040 [0.0002, 0.0094] | 0.0004 [0.0, 0.0014] | +0.0036 [+0.0002, +0.0087] | yes |
| peak XTE (m) | 0.460 | 0.241 | +0.219 [+0.116, +0.347] | yes |

**The crux.** In longitudinal scenes **both arms keep the lane nearly perfectly** — corridor departure is
**0.4 % (flagship) / 0.04 % (REF-C)** — yet the flagship's ADE is **4× REF-C's** (1.455 vs 0.354). The
lane-keeping metric therefore *isolates* the flagship's deficit as **purely longitudinal** (speed / spacing),
NOT lateral — exactly its registered 89 %-longitudinal signature (MODEL_REGISTRY §1.2), and REF-C tracks that
axis far better. This is the value of adding corridor_departure_rate: it decomposes the ADE gap into "not a
lane-keeping problem, a longitudinal one".

### 3.2 Junction (182 win / 22 ep) — where lane-keeping is actually stressed

| | flagship v1 | REF-C base | paired Δ | sep |
|---|---|---|---|---|
| closed ADE@2s (m) | 1.879 [1.570, 2.195] | 1.072 [0.871, 1.248] | +0.807 [+0.504, +1.104] | yes |
| corridor_departure @1.75 m | 0.146 [0.092, 0.207] | 0.064 [0.036, 0.090] | +0.0816 [+0.0396, +0.1304] | yes |
| peak XTE (m) | 2.372 [1.747, 3.126] | 1.458 [1.173, 1.720] | +0.914 [+0.446, +1.511] | yes |
| window_departure_rate @1.75 m | 0.588 | 0.368 | — | — |

In junctions both arms depart the corridor much more, and the flagship departs **~2.3× more often** with a
peak XTE (2.37 m) that clears a full lane, vs REF-C's 1.46 m. REF-C is decisively the better lane-keeper where
lane-keeping bites. This independently **confirms the RETRACTION_LOG 07-23 characterization** of flagship v1's
tactical head as a **high-deviation planner** (there: `plan_dev` 1.12 vs 0.34) whose failure mode is off-road,
not collision — here measured as higher on-policy XTE and higher corridor departure.

### 3.3 The instrument stays low-OOD for BOTH arms (validity)

On-policy OOD peak ratio (P1 flagship envelope, applied to each arm's own deviations): flagship
**1.054** / REF-C **1.035** overall; longitudinal **1.018 / 1.004** (≈ OOD-free); junction **1.196 / 1.152**.
Every stratum for both arms is **≪ NuRec's flat 3.75×**. The source is an **absolute low-OOD** closed-loop
source for both arms, not only the flagship — so the comparison is confound-reduced, not merely relative to
one arm's tolerance.

---

## 4. Honest limits

- **Lane-keeping, not safety.** Map-free / agent-free ⇒ no off-road/collision/PDM. A real off-road/collision
  rate needs AlpaSim (with its reconstruction-OOD caveat) or the design's hybrid. This is the instrument gap the
  whole thread exposed and it remains open; this work fills the *measurable* half (drift) exactly.
- **Absolute departure rates are low overall** (both arms mostly stay in-lane on this cruising-heavy corpus).
  The discriminating signal is the **relative** arm gap (separated), the **junction** stratum (where departures
  are frequent), and the **longitudinal ADE** gap (which the lane metric shows is *not* lateral).
- **Within-source RELATIVE at low OOD**, deployed-decoder vs deployed-decoder through a **shared** pure-pursuit
  + bicycle controller. A different controller changes absolute deviations; the direction is robust across all
  three strata and both metrics, with p(Δ>0)=1.0 overall.
- **Flagship arm = the deployed strategic→tactical head**, not a CEM/imagination planner. The comparison is
  "deployed planner vs deployed planner", consistent with how the program measures flagship closed-loop and
  with the AlpaSim suite it corroborates.
- **OOD ratio is INHERITED-from-12ep**: it uses the flagship-measured 12-ep source envelope
  (`lowood_flagship_ci.json`) as the deviation→OOD map for both arms. The corridor / XTE / ADE metrics — every
  headline number — need **no** envelope and are self-contained MEASURED on the 40-ep set.
- **Resolution** 256²/f_eff=266 phase-0 cache (native is 1080×1920). The real-footage source sidesteps
  reconstruction entirely, so the 07-23 native-res residual does not apply here; a native-res P1 baseline would
  still fully close the resolution question (cheap, deferred).

---

## 5. Deliverable manifest

| artifact | where | what | evidence |
|---|---|---|---|
| `LANEKEEPING_REFC_REPORT.md` | repo (staged), this dir | this report | — |
| `lowood_lanekeep.py` | repo (staged) · pod `/root/lanekeep/` | the extended harness (P1 corridor metric + P2 REF-C arm + P3 paired comparison); self-contained (inlines geometry; needs only the on-pod stack) | MEASURED |
| `lowood_lanekeep_40ep.json` | repo (staged) · pod `/root/lanekeep/` | ⭐ raw: both arms × 3 strata × 3 thresholds, TTD, OOD, + paired CIs | MEASURED |
| `lowood_lanekeep_40ep.log` | repo (staged) | run stdout (terminal marker `LOWOOD_LANEKEEP_DONE`) | MEASURED |
| `lowood_lanekeep_smoke.json` | repo (staged) | 2-ep both-arms smoke (tick-0 self-check pass) | MEASURED |
| `lowood_flagship_ci.json` | repo (this dir, copied) · pod `/root/lanekeep/` | the 12-ep flagship OOD envelope reused for the OOD-ratio map (INHERITED) | INHERITED |
| `provenance.json` | repo (staged), this dir | full provenance + honest bounds | — |

**Inputs (pre-existing, pod-side, not modified):** `tanitad-eval:/root/models/flagship-30k/ckpt.pt` (step 29999);
`tanitad-eval:/root/models/refc-base-30k/ckpt.pt` (step 29999, md5 verified); clean val
`tanitad-eval:/root/valdata/physicalai-val-0c5f7dac3b11` (40 ep → 881 win). **No `stack/` code touched** (the
harness lives in `incoming/`), so `pytest` is unaffected. **Pod state:** `gpu_lock` released, GPU idle, no
deletions. Per the Agent Operating Standard these files are `git add`-ed (staged), **not committed / not
pushed**; the index carries other agents' concurrent work — commit with an explicit pathspec, no `--amend`.

**Escalate integration (do not leave in a doc).** The leaderboard §5.5 and MODEL_REGISTRY closed-loop rows now
have an **absolute low-OOD** comparison of the two main arms to cite (REF-C base > flagship v1 closed-loop,
paired-separated at n=40, reconstruction-OOD removed) — this **corroborates** the AlpaSim n=12 finding through
an independent instrument. Integration is the orchestrator's call; the raw JSON is the quotable source.

**Reproduce (eval pod, GPU free):**
```
PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts python3 lowood_lanekeep.py \
  --flagship-ckpt /root/models/flagship-30k/ckpt.pt \
  --refc-ckpt /root/models/refc-base-30k/ckpt.pt --refc-preset base \
  --val-dir /root/valdata/physicalai-val-0c5f7dac3b11 --episodes 40 \
  --p1-json lowood_flagship_ci.json --out lowood_lanekeep_40ep.json
```
