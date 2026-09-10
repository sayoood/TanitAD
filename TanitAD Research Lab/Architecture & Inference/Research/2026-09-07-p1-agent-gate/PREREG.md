# PREREG — P1 agent conditioning: the tiny-rig gate that converts "gate open" into IN or OUT

**Written BEFORE any outcome number existed.** · **Date** 2026-09-07 (Europe/Berlin) ·
**Agent** TanitAD Architecture & Inference · **Branch** `agent/arch-inf-20260803`
**Deliverable** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-p1-agent-gate/`
**Register row to append to `Project Steering/GOALS_AND_CLAIMS.md`:** **`D-P1-AGENTCOND-1`**
(no existing row covers agent/environment conditioning as a *lever*; the nearest, `E-DEC-29` /
`E-DEC-8`, are frozen-trunk *probe* rows about what a latent CONTAINS, not about whether wiring
agent tokens into the planner CHANGES DRIVING. Different estimand ⇒ new row.)

---

## 0. What is being decided, and by whom it is binding

`Project Steering/REFCV5_MISSING_PIECES_PLAN.md` §4 and §8.2 bind Phase 2: **the composed arm
`refcv5-cap-b1-v72-40k` does not launch until P1 is either IN or explicitly declared OUT.**
§8 records the gate as OPEN (the B1 joins exist, `--agents head` needs no `agent_gt`) but
explicitly **not validated** — "Gate open ≠ piece validated — it still owes a tiny-rig arm."
**This document pre-registers that arm.**

⚠️ **What this arm can and cannot decide.** It is a ~17 M-param rig on 104 training clips.
A PASS here is **entry to the composed arm**, never a published driving result
(REFCV5_MISSING_PIECES_PLAN §2.1). A FAIL is a WAYPOINT, not the end of P1: the FAIL clause in
§5 below names what enters instead, in the same turn.

---

## 1. Hypothesis

**`D-P1-AGENTCOND-1`** — Wiring DiffusionDrive's agent/environment conditioning into REF-C
(`--agents head`: a learned detection head supervised by the `obstacle.offline` join, whose
agent tokens the decoder cross-attends) **improves the DISTANCE-KEEPING (longitudinal) and
LATERAL families** against the identical recipe with the seam absent (`--agents off`).

**Why those two families are the headline and not ADE:** agent conditioning is an
*environment* lever. DD prices its removal at 88.1 → 55.1 PDMS — a metric dominated by
collision/drivable-area/TTC terms, i.e. distance-keeping — not by ego-only displacement.
An ADE-only read would be the wrong instrument even if it moved.

---

## 2. The ONE variable

| | value |
|---|---|
| **moved** | `--agents off` → `--agents head --w-agent 1.0 --agent-join <B1 EVAL join> --agent-join-allow-legacy-ids` |
| **held constant** | the **60-token argv prefix** — arm `hier`, `--size tiny`, corpus, labels, eval corpus, eval cadence, `--image-hw 256 640`, `--steps`, `--batch 8`, `--workers 0`, `--v2-lru 8`, `--lr 1e-4`, `--warmup`, `--seed`, `--log-every`, `--save-every`, `--nav-from-v7`, `--u8-batches`, `--anchors` (sha256 `51f930dc6f3564ff`), `--n-anchors 117`, `--anchor-v0-conditioned`, `--anchor-control-units alat`, `--sel-accel-max 2.0`, `--goal-str`, `--ego-state-inject --ego-dropout 0.5`, `--sampler ddim`, `--w-u0 0.5` |

⛔ **The preflight is mechanical, not an intention.** `launch_arms.py::preflight` diffs the
**actual argv lists** handed to `refc_v3_train.py`, token by token, and **returns exit 3 without
launching** if any arm differs from the control anywhere in the held-constant prefix. Its report
is banked at `runs/preflight_s<seed>.json`. *(MEASURED precedent: a row-bank arm silently
multiplied effective λ by ~n/24 and invalidated two sweeps — the intent was one variable there
too.)*

⚠️ **Two knobs that are part of the seam and are DELIBERATELY left at their defaults**, so the
arm tests DD's cross-attention and not a monocular geometry term:
`--agent-w-project 0.0`, `--agent-w-ground 0.0` (both stamp `OFF_BY_DEFAULT` in the effective-
weight table). A future arm may move them; this one does not.

---

## 3. The rig, stated so it can be refused

| | |
|---|---|
| **trainer** | `stack/scripts/refc_v3_train.py` — the REAL trainer, `--size tiny` (`V3_RIG_SIZES`, 16,989,725 params), NOT a bespoke script |
| **compute** | dev-box **RTX 4060, 8.0 GiB** · ⛔ the A40 is NOT touched — it is reserved for the run this gate releases · ⛔ Thor is NOT touched |
| **corpus** | `C:/Users/Admin/tanitad-data/refav1-eval141/eps` — 141 v2ep clips, 256×640 cylindrical, 3-stack, UUID clip ids |
| **split** | **episode-disjoint**, by `blake2b("split|"+clip_id)` ascending, first 35 → eval. **104 train / 35 eval, intersection 0.** The rule is a hash of the clip id: it is computed BEFORE any metric and cannot select on outcome. Banked at `split.json`. |
| **labels** | `s2_labels_v7.2_eval.jsonl.gz` — joins 104/104 train episodes (100.0 %) |
| **join** | `b1eval_agents.jsonl.xz`, md5(compressed) `3ddb42ecbd3926066795a94587af2aed` — **139/141 clips**, 104/104 train episodes on the stable-63-bit id space, **16,845/17,787 train windows labelled (94.7 %)**, 563,180 target boxes |

⛔ **THE CORPUS IS THE B1 *EVAL* SET, SPLIT IN TWO — AND THAT IS A REAL LIMITATION, STATED
BEFORE THE NUMBERS.** The B1 *TRAIN* join (md5 `1c985e6d6ad34e605c4ebd30cb353558`, verified
present) covers **0** of the 141 local clips, and no local episode cache exists for the 4,427-clip
B1 train corpus. The only agent-labelled corpus that exists on this box is the eval set. My
train/eval split inside it is episode-disjoint, so no eval episode is trained on — but these 35
eval episodes **are the same episodes the programme's landing reads use**, so:
⇒ **any number from this rig is a GATE number and is inadmissible as a held-out result.**
The correct held-out arm runs on the pod against the B1 train join; it is named as the follow-up.

---

## 4. Arms

| arm | delta | role |
|---|---|---|
| **`off`** | `--agents off` | **control** — the recipe every banked refcv4b/refcv5 arm trained under |
| **`head`** | `--agents head --w-agent 1.0 --agent-join b1eval_agents.jsonl.xz` | **treatment** |
| **`shuf`** | identical to `head`, join replaced by `b1eval_agents_SHUFFLED.jsonl.xz` | ⛔ **DELIBERATE REGRESSION** |

⛔ **The regression arm, and why it is this one.** The two regressions the brief names —
"agents wired, weight 0.0" and "`--agents off` with a weight set" — are **already refused by the
trainer's own preflight** (`refc_v3_train.py:505`, `:524-556`). Those guards are good, and they
mean a config-level regression never reaches a metric. So the regression arm must attack the
**information**, not the config: `shuf` permutes `clip_id` across the join by a **derangement
within equal-`n_frames` groups** (seed 20260907), so the record count, frame indices, box counts
and labelled-window fraction are preserved **exactly**, and only the clip→agents association is
destroyed. **MEASURED: 0 of 26,394 rows remain on their own clip.**
⇒ The graph is built, the DETR set loss is computed on real boxes, and the tokens say nothing
about the scene the planner is driving through.
⛔ **If the gate does not FAIL `shuf`, a PASS on `head` means nothing** and this document's
verdict is INCONCLUSIVE, not IN.

---

## 5. The bar — committed in both directions BEFORE the numbers

Primary contrast **`head − off`**, paired **episode-cluster bootstrap** over the 35 eval
episodes (`taniteval/ci.py::episode_cluster_bootstrap`), 10,000 resamples, 95 % interval.
⛔ **`overlapping_holdout_se` is NOT used** — it biases the point estimate bidirectionally, up
to a sign flip.

**PASS (P1 IS IN)** requires ALL of:
1. **`head − off` separated in the FAVOURABLE direction on the DISTANCE-KEEPING metric** (interval excludes 0), AND
2. **the LATERAL family does not regress separated** (curvature MAE interval does not exclude 0 in the worse direction), AND
3. **`head − shuf` separated in the favourable direction on the same distance-keeping metric** — the lever must beat its own uninformative twin, not merely the arm without the seam, AND
4. every control in §6 reads its known value.

**FAIL (P1 IS OUT of the composed arm)** if the distance-keeping contrast is **separated in the
unfavourable direction**, or if `head − shuf` is **not** separated (⇒ any `head − off` movement
is explained by capacity/optimisation, not by agent information).

**INCONCLUSIVE** — neither separated, or a control off its known value. ⛔ **INCONCLUSIVE is
reported as INCONCLUSIVE and is NOT rounded to IN.** Under INCONCLUSIVE the recommendation to
the PI is **P1 IS OUT of this composed arm** — the gate's job is to release a 40k run, and an
unvalidated piece does not enter one; it re-enters behind the pod-side held-out arm.

⭐ **FAIL clause — what enters instead, named now (RULE ZERO).** If P1 fails or is
inconclusive, the composed arm launches **without** the agent seam carrying **P14 (sampler ranks
the fan, +0.2813 [+0.2127, +0.3543] separated, already PASSED)** and **P4 restricted**, and P1's
next lever is the **pod-side arm on the B1 TRAIN join** — a corpus 42× larger, on which the
tiny rig's `n` objection does not apply. P1 is not dropped; it is re-queued behind data it does
not have on this box.

---

## 6. Controls that must read a KNOWN value

| control | known value it must read |
|---|---|
| **constant-only** | a plan that never moves, at exactly the no-information value for each metric |
| **straight-line floor** | a plan that never steers — ⛔ **quoted BESIDE every curvature number**, because REF-C is currently ~84× worse on curvature than this floor (0.02737 vs 2.30973) |
| **join attachment** | `agent_join_stats.frac_windows_labelled` must be **non-zero and equal** for `head` and `shuf` (they carry the same rows); `off` must build **no** `agent_slots` |
| **effective-weight stamp** | `--w-agent` must stamp `default 0 → layer 1 → effective 1 → builds-a-graph yes → TRAINS` for `head`/`shuf`, and be absent for `off` |
| **n and d** | printed for every interval. ⚠️ `n ≪ d` is **underpowered BY CONSTRUCTION, not a negative** |
| **grep control** | any absence claim carries a same-breath count that must read non-zero (grep under-reports ~5× on the G: mount while exiting 0 — MEASURED again this session: a repo-wide `--agents` grep returned **0** hits where the same pattern under `stack/` returned **40**) |

---

## 7. Reporting rules binding on the RESULT

* ⛔ **Four families, never pooled, never ADE alone**: LONGITUDINAL (target-speed AND
  distance-keeping/TTC), LATERAL (heading, **curvature**, yaw-rate, cross-track), TACTICAL,
  STRATEGIC. A family that cannot be computed is reported **with its reason and its n**, never
  silently dropped.
* ⛔ **T-tier on every number.** T0 = WM diagnostic, never "driving performance"; T1 = self-action
  open loop. ⚠️ Per the binding ruling, **a planner feeding its own predictor is STILL open loop** —
  nothing here is closed loop.
* ⛔⛔ **A separated CI from a one-seed arm is NECESSARY, NOT SUFFICIENT.** A pure replicate
  produced "separated" differences on **6 of 42** cells (**14.3 %** false-positive). Two seeds are
  run if the rig fits the budget; **if only one seed is run, the RESULT must say which variance
  question the interval answered (episode draw) and that the lever claim is NOT established.**
* **Evidence class on every claim.**

---

## 8. Pre-committed record of what was already run before this file existed

Honesty item, so the record cannot be read as post-hoc: **a 3-step MECHANICAL smoke** of `off`
and `head` (seed 99) ran before this file was written. It established only that the trainer
starts, that the join attaches, and that a missing `digest_scope` in the join sidecar was the
sole blocker (fixed by the trainer's own documented `join_meta --write` migration). **No outcome
metric was read from it, and seed 99 is not a scoring seed.** Its logs are banked alongside the
scoring runs.

---

## 9. AMENDMENT 1 — the second-seed rule, committed BEFORE any outcome number

**Written 2026-09-07, while the seed-0 arms were still training and before any
`dump_eval.py` / `analyze.py` output existed for them.** Logged as an amendment rather than edited
into §7, so the record shows what was decided when.

§7 requires two seeds "if the rig fits the budget". At ~10 min/arm the three-arm seed is ~30 min,
so a second seed is affordable but not free. ⛔ **The rule is committed here, before the numbers,
so it cannot be chosen to suit them:**

* **If ANY primary contrast (`head − off` or `head − shuf`, on distance-keeping or curvature) is
  SEPARATED at seed 0 ⇒ seed 1 is MANDATORY.** A one-seed separation sits above a MEASURED 14.3 %
  replicate false-positive floor and cannot on its own release a 40k run.
* **If NO primary contrast is separated at seed 0 ⇒ seed 1 is NOT run**, and the verdict is written
  from seed 0 with the power statement attached explicitly: the interval answered the
  **episode-draw** question only, `n ≪ d`, and **the lever claim is NOT established in either
  direction**. Adding a second seed cannot convert a null into a lever; it can only widen the
  variance the interval answers, and the honest report of a null is the null plus its power.
