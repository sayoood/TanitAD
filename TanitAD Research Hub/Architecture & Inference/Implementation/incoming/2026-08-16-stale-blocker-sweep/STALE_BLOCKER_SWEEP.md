# STALE BLOCKER SWEEP — a root-cause-class elimination, plus the fourth rot of one prose number

- **Date:** 2026-08-16 · **Discipline:** Architecture & Inference · **Status:** PENDING orchestrator triage
- **Evidence class:** MEASURED (ours, from source + raw artifacts in this repo) unless a row says otherwise.
- **Compute:** CPU-only, dev box, $0. No GPU touched (Thor is training a 336M model on the only GPU).

---

## 0. The class being eliminated

> **"A document that states a blocker is never revisited when the blocker clears."**

This is not a hypothetical. It has now cost the programme three times:

| # | incident | cost |
|---|---|---|
| 1 | An orthogonality instrument whose merge request lived in a README nobody re-read | **10 days** unmerged |
| 2 | `…/incoming/2026-08-03-longitudinal-distance-keeping/INTAKE.md:73-75` — *"Until that lands, arm evals will still report the family UNAVAILABLE"* | **a whole agent commissioned 2026-08-16 to rebuild an instrument that already existed** |
| 3 | `CLAUDE.md`'s *"our ingest reads N of 36 features"* | **rotted 4×**, propagating a wrong number into **17 documents + 1 code docstring** |

⭐ **The finding that ties the sweep together: incidents 2 and 3 have the SAME TRIGGER.**
`obstacle.offline` became a real read on **2026-08-03**. That single landing invalidated *both* the
INTAKE's blocker line *and* the prose feature count. **Neither was revisited**, because nothing in the
programme points *forward* from a blocker to the thing that clears it.

⚠️ **The asymmetry is the mechanism, and it is worth stating precisely.** The closing work *did* cite
the blocker: `taniteval/taniteval/lead_source.py`'s own docstring names this INTAKE's open work item
as its reason for existing. **The reference is one-directional.** The successor knows about the
predecessor; the predecessor never learns it has been superseded. Every instance in the table above
has this shape. ⇒ *A "blocked on X" line is a claim with an expiry date and no alarm attached.*

---

## 1. VERDICT TABLE — the flagship item (lead agent's own probes)

| doc:line | claim | VERDICT | clearing evidence |
|---|---|---|---|
| `…/incoming/2026-08-03-longitudinal-distance-keeping/INTAKE.md:73-76` | *"Wiring it into the eval path (val40 windows → `win["lead"]`) needs the `obstacle.offline` chunks for the 40 val episodes… Until that lands, arm evals will still report the family UNAVAILABLE"* | ⏹ **CLEARED 2026-08-03**, measured through 2026-08-14 | **3 independent probes**, below |
| same file `:3` | *"**Status:** PENDING orchestrator triage"* | ⏹ **CLEARED — landed & in production use**; the ORCHESTRATOR VERDICT block was simply never filled in | an empty verdict block is **not** evidence of rejection |

**Probes for the flagship row (all MEASURED):**

1. **The wiring exists.** `taniteval/taniteval/lead_source.py` (2026-08-03, 21 KB) — *"Turn
   `obstacle.offline` into the `win["lead"]` block `four_families` consumes"*; plus
   `taniteval/tools/build_lead_block.py` for the corpus build.
2. **It was fed, on exactly the 40 val episodes the INTAKE named.**
   `…/Data Engineering/…/incoming/2026-08-04-instrument-durability/raw/val40_lead_report.json`:
   **`n_episodes: 40`, `canonical_881: true`, registration `n_ok: 40 / n_failed: 0`, counts
   `LEAD 270 / NO_LEAD 551 / NO_LABEL 60`**, integrity `poses_sha256_match_all: true`.
3. **The registry closed it.** `Project Steering/MODEL_REGISTRY.md:1187-1190` — ~~`distance_keeping`
   UNAVAILABLE~~ struck, raw `pod4:/workspace/evalout/v1arch_oodval_q90_4fam_LEAD.json`,
   `families_unavailable=[]`.

⚠️ **This blocker generated TWO stale statements, not one.** The registry itself records that the
**2026-08-06 claim "lead block not on pod4" was ITSELF a stale absence-claim** — the block already
existed on pod4. So the same underlying fact was got wrong in both directions within three days.
*(A stale "it does not exist" and a stale "it is blocked" are the same defect.)*

**Closed in place** (annotated, history preserved — nothing deleted).

---

## 2. THE FOURTH ROT — `"our ingest reads N of 36 features"`

### 2.1 The finding, stated loudly as instructed

⛔ **The number was wrong again. `CLAUDE.md` said 5; the correct program-wide count is 6.**

Sequence: **`"2 of 36"` → 4 (2026-07-26) → 5 (2026-08-16) → 6 (2026-08-16, this sweep).**
Four rots, all inside *the very rule that warns about stale absence-claims*.

### 2.2 The real root cause is not carelessness — the SUBJECT was never defined

There is no single right answer, which is exactly why a single number kept rotting. **"Our ingest"
names three different things:**

| layer | count | features | source |
|---|---|---|---|
| `physicalai_r0.py` — r0 clip selection | **2** | `egomotion`, `camera_front_wide_120fov` | `stack/scripts/physicalai_r0.py:36-38` |
| `physicalai.py` — the **episode build** | **5** | + `camera_intrinsics`, `sensor_extrinsics`, `vehicle_dimensions` | `stack/tanitad/data/physicalai.py:232-235`, consumed at `:354` / `:407` / `:456` |
| **program-wide** (incl. the pod-side join) | **6** | + `obstacle.offline` | `stack/scripts/build_obstacle_join.py:148`, `stack/scripts/lead_state_gate.py:308-338`, `stack/tanitad/data/bev_raster.py` |

⚠️ **`obstacle.offline` is deliberately OUTSIDE the episode build** — the join is a pod-side step, which
is why `grep obstacle stack/tanitad/data/physicalai.py` still returns **zero matches** (verified).
So *"the episode build reads 5"* is **true**; *"our ingest reads 5"* is **false as of 2026-08-03**.

⇒ **The durable fix is not a better number, it is naming the layer.** Every future statement must read
*"the episode build reads 5 of 36"*, never the bare *"our ingest"*.

### 2.3 Verification that each of the 5 is a REAL read

A defined-but-unused constant would inflate the count, so this was checked rather than assumed:
all three calibration constants are passed to `_calib_chunk_path(...)` — `_CALIB_INTR` at `:354`,
`_CALIB_VEH` at `:407`, `_CALIB_EXTR` at `:456` — and `labels/egomotion` is read at `:471-472`.
The test below asserts this, so a constant going unused breaks the build.

### 2.4 The pin

**`stack/tests/test_physicalai_feature_readset.py` — 8 tests, green.** It asserts:

1. the three counts **and the exact feature names**, read from live source;
2. that each calibration constant is genuinely *consumed*, not merely declared;
3. that `obstacle.offline` is read at **≥2 of 3** known sites (absence at one location is not absence);
4. that `obstacle.offline` stays **out** of the episode build — the 5-vs-6 split is load-bearing;
5. a **drift detector** that fails when a *new* HF feature path appears in `physicalai.py`. This is the
   case all four rots actually were.

⭐ **The guard is proved able to fail.** `test_drift_detector_actually_fires` runs the detector against
a synthetic source containing `labels/lidar_front/…` and asserts it trips — because this programme has
repeatedly been burned by guards structurally unable to report the answer they are cited for (the C13
class). It also pins the two false-positive classes actually hit while writing it (trailing sentence
punctuation, and the local `calibration/physicalai_*.csv` sidecars, which are not dataset features).

**Every failure message names the documents to update**, so the next change is mechanical rather than a
re-derivation.

### 2.5 Blast radius of the stale count

**14 documents (17 sites) + 1 code docstring**, MEASURED by ripgrep over `*.md` + `*.py`.
*(An earlier draft of this very section said "17 documents" — 17 is the SITE count, 14 the FILE
count. Being imprecise about a count, in the write-up about an imprecise count, is the joke writing
itself; it is corrected here rather than quietly.)* Highest-value, in order:

| site | why it matters |
|---|---|
| `CLAUDE.md:266` | the canonical statement — ✅ **FIXED by this sweep** |
| `Project Steering/EVAL_PROTOCOL_OODVAL_2026-08-05.md:143` | ⛔ a **PROTOCOL** — steers every future eval, and its *"UNAVAILABLE… a WORK ITEM, not a pass"* is **doubly stale** (count wrong AND blocker cleared). Handed to the Project Steering stream with evidence. |
| `Project Steering/V6F_PLANNER_DESIGN.md:536` | live design doc |
| `Project Steering/Gates/flagship-v5-retrain.PREP.md:58` | carries the complement, *"32-of-36"* → should be **30-of-36** |
| ⚠️ `stack/tanitad/data/bev_raster.py:12` | **the count rotted INTO CODE.** See §4 — escalated, not edited (another agent owns this file). |
| ~11 dated write-ups under `**/incoming/**` | history, low blast radius |

---

## 3. VERDICT TABLE — the wider sweep

*Three parallel streams swept the corpus (48 `INTAKE.md`, 121 `Project Steering/*.md`, ~507
non-INTAKE `incoming/**` docs). Their verdict tables are consolidated below.*

### 3.A — the 48 `INTAKE.md` files

**22 claims across 17 INTAKEs annotated in place.** Verdicts: **9 CLEARED · 9 STILL TRUE · 2
SUPERSEDED · 2 PARTIAL/UNVERIFIABLE.**

⭐ **A FOURTH independent confirmation of §1 arrived from inside the corpus.** A sibling package filed
the same day, `…/incoming/2026-08-16-obstacle-offline-sidecar/OBSTACLE_OFFLINE_SIDECAR.md` §0, opens
with **"The briefed task was already built."** ⇒ **The stale distance-keeping blocker mis-commissioned
work at least twice on 2026-08-16 alone**, in two different streams that did not know about each
other. That is the cost of C70 measured directly, on one day.

⛔ **A CITED BAR THAT NO COMMITTED INSTRUMENT COMPUTES.** `MODEL_REGISTRY.md:100,219` quote a
**"best-of-3 kinematic floor 0.5005"** as a live bar. `baselines.py` and `skill_score` **never
landed** (0 hits), and the originating package's own risk note says its denominators are
**comma-hwy-corpus-specific — "do not treat as universal."** ⇒ Same family as the CTRV-0.523
inheritance already caught, and it is **in the registry**, the one quotable source. **Escalated, not
edited** (Project-Steering-owned).

⚠️ **A dangling logging key survived its instrument.** `H15Meter`, `h15_fire_frac`, `h15_fired` →
**0 hits repo-wide**, yet `stack/scripts/train_flagship4b.py:658` still executes
`log["h15"] = float(loss_h15.item())`. Blast radius is now v1-line only (v4/v16 use different
imagination surfaces), which is *why* nobody noticed.

**CLEARED (9):** *"video-only **until an IDM head lands**"* → the head landed (`stack/scripts/idm_head.py`,
`models/dynamics_encoder.py:44,299`, weights v1/v3/v4 banked; ⚠️ carry its caveat — comma yaw R²
**−0.746**, PhysicalAI +0.9035) · CTRV *"the driving block is missing"* → `driving.py:313`
`FLOORS=("cv","holdv0","ctrv")` · the atomic-milestone ops bug → `stack/tanitad/train/ckpt_io.py:23`,
swapped into **5** trainers (⚠️ deviation: the `archive_milestones()` wrapper was **not** adopted) ·
AlpaSim adapter **5 of 6** TODOs → `stack/experiments/alpasim-driver/tanitad_model.py` has **0
TODOs** · pandaset *"blocked until D-016 R1"* → R1 landed 2026-07-17 (466.97 → **266.0**) ·
instrument-durability triage → consumed by `670f614` (27 arms, 270 LEAD windows) · Cosmos stub
latency/params → **0.2628 B / 14.331 ms** MEASURED (⚠️ RTX 4060 — must **not** be swapped into A40
rows) · D2's *"BLOCKED on I4"* → **SUPERSEDED**, `run_d2` rows are `[I1,I2,I3(,I7)]`, I4 is
diagnostic (D-017) · ZOD-as-P0 → **SUPERSEDED**, AV2 landed instead (`6016736`; ZOD has no lane
graph, 3 probes).

⚠️ **STILL TRUE, and these are the ones that should become work items:**

| item | evidence | age |
|---|---|---|
| ⛔ **`rrd` dual-sink guard — 2 lines, guarding a SILENT 3,314× data loss** | 0 hits for `allow_stub_rrd`/`check_sinks`; `replay_app.py:277,279` still accepts `--rrd`+`--serve` unguarded | **26 d** |
| ⛔ **`train_strat.jsonl` is STRANDED — documented in an INTAKE's file table, never banked** | 3 probes: `find` 0 hits, `git ls-files` has only `train_strat_windows.json`, **not gitignored**. Inputs survive; the records do not | — |
| **pandaset loader: unblocked-and-unintegrated** | `stack/tanitad/data/pandaset.py` absent ~30 d after its blocker cleared | **~30 d** |
| **3 opponent scenarios un-adjudicated** (SC-04, SC-13, SC-06) | `SCENARIO_REGISTRY` has 3 entries; both siblings' verdict templates unfilled. Fix = one verdict pass + 3 `ScenarioEntry` rows | 9–23 d |
| `baselines.py` / `skill_score` never landed | see the cited-bar finding above | — |
| H15 logging fidelity; `worldmodel_synth.py` never committed; ZOD dataset access (human application); LAL/OKRI/LOPS renderer-gated | per stream manifest | — |

⚠️ **`ROADMAP.md:79` still names CARLA-on-pod as the closed-loop substrate** while marking it *"not
fired yet"* — and the live line is AlpaSim/NuRec + T1. **Two INTAKEs remain gated on a harness nobody
is building.** *(A gate pointing at an abandoned substrate is a stale blocker that can never clear.)*

### 3.B — `Project Steering/` (the highest blast radius, because these steer work)

**21 claims across 9 steering docs: 13 CLEARED, 3 PARTIAL, 2 UNVERIFIABLE, 3 STILL TRUE.**
Annotations are dated and append-only; no registry row was rewritten and no number restated.

⭐⭐ **THE SMOKING GUN FOR TODAY'S FAILURE — `MODEL_REGISTRY.md:1583,1644`.** The registry cites the
distance-keeping instrument as **`tools/build_lead_block.py`**. That path **does not exist** (three
probes). The file is at **`taniteval/tools/build_lead_block.py`**.

> ⛔ **A WRONG PATH MAKES A BUILT INSTRUMENT LOOK UNBUILT — and that is precisely how an agent gets
> commissioned to rebuild it.** This is a *different mechanism* from the stale blocker in §1, with the
> *same* outcome, pointing at the *same* instrument. Two independent defects aimed at one target.

⛔ **The protocol was telling every future eval to report a family it could already measure.**
`EVAL_PROTOCOL_OODVAL_2026-08-05.md:143` marked distance-keeping **UNAVAILABLE** — while a complete
measurement of *exactly that family* on *exactly that corpus* sat in the repo:
`…/incoming/2026-08-05-v1arch-oodval-four-families/raw/v1arch_oodval_q90_4fam_LEAD.json` —
**`status "OK"`, n 2846/6382, `_families_unavailable []`, headway 25.5263 m.**
⇒ **The binding four-family rule was being silently violated by its own protocol document**, on the
axis carrying **88.7 % of the oracle gap**. **CLEARED.**

⛔ **`V6F_PLANNER_DESIGN.md:673` — the doc's own self-labelled *"highest-priority item in this
document"* was already shipped.** The S-S consumer-invalidation gate exists as
`stack/scripts/train_v6_staged.py:250-259` (`STAGE_INVALIDATES = {"S-S": ("S-T",)}`) with
`stack/tests/test_v6_stage_revalidation.py` (9 tests). **Anyone opening the v6 design doc to pick the
top item would have rebuilt a shipped, tested gate.** **CLEARED 2026-08-16.**

⚠️ **Two blockers were stale THE DAY THEY WERE WRITTEN** — the shortest possible expiry:

| doc | claim | cleared |
|---|---|---|
| `ROADMAP.md:68,112,297` | TanitScena `stack/tanitad/scena/` *"does not exist"* (**3 places**) | **same day, 2026-07-12** — `parse.py`, `vector.py`, `static/`, `scena_app.py`, `test_scena.py`, commit `9ebfb09` |
| `REPO_TRIAGE_2026-07-20.md:67,287` | *"`tools/` does not exist in HEAD at all"* | **same day** — 16 files + 10 tests, commits `c4d8451`, `1e13e3a` |

**Other CLEARED:** `ROADMAP.md:66` REF-B "blocked on pod3 comma-extraction" (registry §3.5,
`refb-refbpatch-v2-30k` 0.5921 ± 0.0685 @ 29999, ✅ FINAL) · `ROADMAP.md:69` nuScenes loader ·
`REPO_TRIAGE` stranded rows 1,2,4,6,8 (**5 of 10**) · `Gates/…PREP.md:17` the 600-clip 120° val split
(2026-08-09) · `BACKLOG` C3 v5 gate · `BOOST_PROGRAM.md:134,269` closed-loop measurability
(superseded by the binding T0/T1/T2 doctrine + `taniteval/tools/t1_eval.py`) · `V6F:536,588` ·
`Gates/…PREP.md:58` ("32-of-36" → **30-of-36**, per §2).

⚠️ **STILL TRUE / PARTIAL, and each for an instructive reason:**

| claim | verdict | why it is instructive |
|---|---|---|
| `PROGRAM_OVERVIEW.md:528` — no `alpasim_runtime` ⇒ no collision/offroad/scene score | **TRUE for Thor, FALSE program-wide** | a live instance of **C2**: a single-host absence written as a programme fact. `…/2026-07-22-alpasim-closedloop-evalpod/M2_results-summary.json` has `scene_score_enabled: true`, `collision_at_fault 0.0`, `offroad 0.0` |
| `ROADMAP.md:104` CARLA "blocked on a graphics-capable pod" | **PARTIAL — premise refuted, conclusion unverifiable** | the *Vulkan* premise died with C2 (`RETRACTION_LOG.md:35`) and AlpaSim ran bare on an A40; but no CARLA run exists, so the blocker is **not** cleared. **A refuted premise does not clear the claim it supported** |
| `MODEL_REGISTRY.md:2207` `refc-small-30k.json` "DOES NOT EXIST" | **STILL TRUE — already correctly resolved in place** | left unedited: the registry already names the real source |
| ⚠️ `BACKLOG` C2 / `BOOST` S-3 / registry §1.7 — v2corpus **"🟢 RUNNING"**, ETA 2026-07-29 | **UNVERIFIABLE — stale on its face (18 days past ETA)** | ⭐ **the INVERSE failure: a stale POSITIVE status.** A stale blocker makes you rebuild; **a stale "running" makes you WAIT FOREVER.** Not asserted dead — flagged for a fleet probe |

### 3.C — non-INTAKE `incoming/**` docs (READMEs, MANIFESTs, DESIGNs, STATUS docs)

**28 claims verified; 12 docs annotated in place.** ⭐ **The headline is a refutation of a
prior conclusion**, not a list:

> ⛔ **The 2026-07-26 program harvest concluded *"most of today's stranding is same-day."* THAT IS
> REFUTED. Of its 12 open items, 9 are still open 21 days later.**
> ⇒ **What clears, clears fast; what does not clear that day tends never to clear** — because
> nobody re-reads the doc. This is C70's survival curve, and it is bimodal, not decaying.

**CLEARED (8):** comma2k19 yaw-at-v≈0 guard (`comma2k19.py:120,339,385-404`, commit `8ab5327`) ·
geometry-audit wheelbase (`GEOMETRY_INTEGRITY_AUDIT.md:40,77-82`) · `closedloop.py` WHEELBASE skew ·
`e2a_localize.py` pod3-only (now tracked in-repo) · **distance-keeping/TTC "has no instrument"**
(commit `49e2229`, `taniteval/taniteval/lead_metrics.py`) · `distance_keeping` hardcoded UNAVAILABLE
(now conditional, `four_families.py:245,315`) · **"our ingest doesn't read `obstacle.offline`"**
(`stack/scripts/lead_state_gate.py`) · v4 eval harness "never built"
(`stack/scripts/eval_flagship_v4.py`) · "0.930 → −2.465" pairing (`MODEL_REGISTRY.md:2718-2724`).

⭐ **Independent corroboration of §1 and §2:** three of those eight are the *same* stale
`obstacle.offline` blocker, found by a stream that was not told about it. **It had propagated into at
least three further docs and sat there for two weeks after the instrument landed.**

**STILL TRUE (17), highest-consequence first:**

| finding | evidence | age |
|---|---|---|
| ⛔ **`--nav-known-channel` HALF-LANDED — the flag parses, then raises** | `stack/scripts/refc_train.py:1221,726` accept it; `:402` never passes `nav_known=` ⇒ `stack/tanitad/refs/refc.py:2005` fails loud on the first forward | ~13 d |
| ⛔ **A false safety-relevant verdict is live in a committed artifact** | `…/2026-07-26-v4-30k-gate/coprimary/corridor_v4_30k_K185.json` → `paired_common_start.185.ood` still reads "within the measured envelope" | 21 d |
| ⛔ **`planner_p2.py` is the last un-migrated estimator — and it decides `G1_pass`/`G4_pass`** | `taniteval/taniteval/planner_p2.py:389,397,415,458,586` still `_jack_*`; **0** `episode_cluster_bootstrap` call sites; siblings migrated 21 d ago | 21 d |
| ⚠️ **`flagship-v16-ab-ft` weights may be UNRECOVERABLE** | `MODEL_REGISTRY.md:599-600` has no `Location` row; its stated mitigation ("push when pod2 frees") is **void — pod2 is out of the fleet** | 21 d |
| `clhorizon.run_v4` raises on first step | `clhorizon.py:928`, `:765-766`; `RawEp` (`data.py:220-228`) has only `.feats` | 21 d |
| 3 CUDA-graph prereq sites | `metric_dynamics.py:241-242,285-286,693`; `predictor.py:165,172` | 27 d |
| `long_accel` still in `SCALAR_NAMES` | `stack/scripts/idm_head.py:37` — card caveated, code not | — |
| AV2 has no ingest driver · Overture entry unapplied (PI/legal) · `driving.tier0()` lacks lateral/corridor · un-retracted "different egress" echo · 3 sitclf items · sitclf `--lead` | see stream manifest | — |

**HALF-CLEARED (1):** `--nav-known-channel`, above. ⚠️ **A half-merge is worse than no merge**: the
"is it done?" probe (does the flag exist?) returns **yes**, and the failure surfaces only at runtime.

---

## 3.X ⛔ STILL-OPEN INTEGRATION REQUESTS — the 10-day class, still running

**Every one of these is tested code that exists and is not wired in.** This is the exact pattern
`AGENT_OPERATING_STANDARD.md` rule 3 exists to prevent, and it is currently happening **eight times
at once**. Ordered by consequence:

| # | item | where the code is | why it matters | age |
|---|---|---|---|---|
| 1 | ⛔ **`--nav-known-channel` half-merge** | `stack/scripts/refc_train.py:402` (missing `nav_known=`) | **~2 lines.** The flag parses and then raises at `refc.py:2005`. A half-merge defeats the "does it exist?" probe. ⚠️ file touched by a live stream — re-probe first | ~13 d |
| 2 | ⛔ **`planner_p2.py` estimator migration** | `taniteval/taniteval/planner_p2.py:389,397,415,458,586` | **it decides `G1_pass`/`G4_pass`** on the deprecated `_jack_*` estimator that CLAUDE.md documents as biasing the POINT ESTIMATE with sign flips | 21 d |
| 3 | **`grad_surgery.py`** | `…/incoming/2026-07-23-planner-wm-gradient-coupling/grad_surgery.py` (+9 green tests) → `stack/tanitad/train/grad_surgery.py` | queued for "the next v4.x launch" — **the programme is on v6, so its gating condition is void**. The 10-day orthogonality case, at 24 | **24 d** |
| 4 | **`latent_screen` not adopted** | `stack/tanitad/eval/latent_screen.py` (+tests) | **0** mentions in `GATE_PROTOCOL.md`, 0 call sites — **requested independently by two packages** | 13 d |
| 5 | **`goal_admissibility` has zero call sites** | `stack/tanitad/eval/goal_admissibility.py` (+tests) | the guard that exists to stop **another 1.0000 nav echo** is imported by nothing | 12 d |
| 6 | **R1a — 4 decision keys** | `taniteval/taniteval/refc_eval.py::collect` return at `:172-194` | flagged *"0 GPU, EXECUTABLE NOW, pure plumbing"*; **TACTICAL/STRATEGIC stay UNAVAILABLE for REF-C until it lands** | 14 d |
| 7 | ⚠️ **`flagship-v16-ab-ft` may be LOST** | `MODEL_REGISTRY.md:599-600` | no `Location` row; mitigation void (pod2 gone). Needs a fleet probe → `Location` row **or a LOST record**. ⚠️ do not confuse with `flagship-v16-unicycle` (§1.10), which *was* banked | 21 d |
| 8 | ⛔ **false safety verdict in a committed artifact** | `…/2026-07-26-v4-30k-gate/coprimary/corridor_v4_30k_K185.json` | `paired_common_start.185.ood` still asserts "within the measured envelope" | 21 d |

⚠️ **These are reported, not fixed** — items 1, 2, 6 touch files owned by live streams this session,
and 3–5 are integration decisions above a sweep's authority. **Per rule 3, they are escalated here
AND in the agent report, not left as a "please merge" line in a doc — which is precisely the failure
mode that produced this list.**

---

## 4. ⚠️ ESCALATION — things this sweep could NOT fix itself

0. ⛔ **REGISTRY INTEGRITY — three items, all Project-Steering-owned, none edited by this sweep:**
   - **`MODEL_REGISTRY.md:100,219`** quote a *"best-of-3 kinematic floor 0.5005"* that **no committed
     instrument computes** (`baselines.py` / `skill_score` never landed), and whose denominators the
     source package calls corpus-specific — *"do not treat as universal."*
   - **`MODEL_REGISTRY.md:1583,1644`** cite `tools/build_lead_block.py`; the real path is
     `taniteval/tools/build_lead_block.py`. **A built instrument reads as unbuilt.**
   - **`MODEL_REGISTRY.md` §1.7 / `BACKLOG` C2 / `BOOST` S-3** carry v2corpus as **"🟢 RUNNING"** with
     an ETA **18 days expired** and no completion row. Needs a fleet probe, then a completion row or a
     LOST record — as does `flagship-v16-ab-ft` (§3.C, no `Location` row, pod2 gone).
1. ⛔ **`stack/tanitad/data/bev_raster.py:12` carries the stale "4 of 36" inside a module docstring.**
   That file is owned by a live agent this session, so it was **not edited**. It needs the one-word
   fix (`4` → the layer-qualified count) by whoever owns it next. **A stale count in code outlives a
   stale count in prose, because nobody greps docstrings.**
2. **The one-directional-reference problem is structural and unsolved.** Annotating today's stale lines
   fixes the instances, not the class. The cheap durable mechanism would be: *a doc that states a
   blocker names the artifact whose existence clears it*, so the check is a `test -f`, not a re-reading.
   That is a real work item, not a rhetorical flourish — it is the same fix shape as §2.4.
3. **Empty ORCHESTRATOR VERDICT blocks read as "rejected" but mean "nobody filled it in."** At least
   one landed-and-in-production package (`2026-08-03-longitudinal-distance-keeping`) has an empty
   verdict block. Worth a sweep of its own.

### 4.1 Two instrument caveats found in passing (both in the "a probe that reports the wrong scope" family)

1. ⚠️ **`git diff --cached --stat` UNDER-REPORTS a new file's size and is not a staging verifier.**
   MEASURED: it reported **44 insertions** for `stack/tests/test_physicalai_feature_readset.py`, a
   **333-line** new file, with rename detection explicitly off (`--numstat --no-renames`) and **no
   `.gitattributes` / diff driver** in this repo (`git check-attr -a` returns nothing). The index was
   in fact correct — `git show :<path>` returned all 333 lines and all 8 test functions. ⇒ **Verify
   staged CONTENT with `git show :<path>`, never with `--stat`'s line count.** Same family as the
   documented `git add` exit-code trap: a git output read as evidence of something it does not report.
2. ✅ **`git ls-files --cached` is insufficient for a MODIFIED TRACKED file** — it answers *"is this
   path in the index?"*, which is `yes` even when the index blob is the pre-edit version. This was
   sharpened into `CLAUDE.md` by the orchestrator mid-sweep and **applied to this package**: all five
   deliverables were re-verified by **blob comparison** (`git ls-files --stage` vs
   `git hash-object`), and again at end of turn, since concurrent commits move the index underneath a
   staged file.

---

## 5. Deliverables

| artifact | repo path |
|---|---|
| This writeup | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-stale-blocker-sweep/STALE_BLOCKER_SWEEP.md` |
| The pin (8 tests, green) | `stack/tests/test_physicalai_feature_readset.py` |
| Corrected canonical count + layer table | `CLAUDE.md` (§"Absence found at ONE location is not absence") |
| Flagship blocker closed in place | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-longitudinal-distance-keeping/INTAKE.md` |

**Suite:** `PYTHONUTF8=1 python -m pytest -q -p no:cacheprovider` from `stack/`.
Baseline at HEAD `8e215b3`: **2919 passed / 0 failed / 17 skipped / 2 xfailed**.
✅ **After this package: 2996 passed / 0 failed / 17 skipped / 2 xfailed** (473.7 s). GREEN.
*(The +77 spans this package's 9 tests and four sibling streams committing concurrently; 0 failed is
the load-bearing number.)*

⚠️ **Staging note.** All deliverables were verified by **blob comparison** (`git ls-files --stage`
vs `git hash-object`), not by `git ls-files --cached` — the orchestrator sharpened that rule into
`CLAUDE.md` mid-sweep and it applies here. Four orchestrator commits (`296139f`, `5725d95`, `7f5dd6b`)
swept this package's files while it was being written; all content verified present in HEAD
afterwards, nothing lost, **and this agent made no commits and no pushes.**

## 6. Sweep totals

| | |
|---|---|
| streams | 3 parallel + lead |
| corpus | 48 `INTAKE.md` · 121 `Project Steering/*.md` · ~507 `incoming/**` docs |
| claims adjudicated | **67+** |
| CLEARED and annotated in place | **30+** |
| docs annotated | **41** (17 INTAKEs · 9 steering · 12 incoming · CLAUDE.md · this package's 2) |
| still-open integration requests surfaced | **8**, oldest **24 days** |
| new root-cause class | **C70** + five sub-classes (`RETRACTION_LOG.md`) |
| GPU used | **none** (Thor training untouched) |

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Reason:**
- **Landed at:**
