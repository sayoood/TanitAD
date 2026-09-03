# TanitAD programme report — 2026-09-03 10:05 Europe/Berlin

*Fixed-clock slot. All wall-clock times Europe/Berlin; **Thor and pod logs are UTC** and are labelled Z. Every
number carries its evidence class and tier. `MODEL_REGISTRY.md` and raw eval JSON are the only quotable sources
for model facts.*

⚠️ **Two corrections to this slot's own arguments, before anything else.** (1) They cite the refav1 run
`refav1-b1-v72-1ep-21109`; that run was **RETIRED on 2026-09-02** and has been replaced twice since. The live run
is `refav1-b1-v72-ep3-speed`. The old directory is still on Thor's disk, which is exactly how a stale path keeps
getting quoted. (2) They ask for the open PI decision **C-NAV-SOURCE-DIVERGENCE**; that decision was **CLOSED on
2026-09-02** — the PI directed the switch, refcv3 moved to the v7.2 nav token at step 16,500, and it was verified
reaching all three layers (`D-REFCV3-NAV-SWITCHED`). The decisions actually open are in §5.

---

## 1. Fresh measurements

### refav1 — `/home/nvidia/experiments/refav1-b1-v72-ep3-speed` (Thor)
MEASURED 2026-09-03 08:02Z, tier T0 (train-side instruments; never driving performance).

| quantity | value |
|---|---|
| step | **3,750 / 21,109 (17.8 %)** |
| marginal pace | **3.49 s/step** (two adjacent log rows) |
| ETA | **≈ 16.8 h** ⇒ ends ≈ 2026-09-04 00:50Z |
| log freshness | 135 s (log-every 50 at 3.5 s/step ⇒ healthy) |
| `DRIFT_ALARM` | **NONE** (one fired 06:00Z on a rising operative loss and self-cleared by 08:12Z) |
| `grad_norm` | 0.95, finite throughout; **1 step SKIPPED** by the new guard |
| `adapter_std` | 0.480 → **0.823** — RISING, the anti-collapse read |
| `tgt_std_op` | **0.997** — the known-value control, ≈ 1.0 as required |
| `tgt_std_tac` | 0.196; **max/min 4.06** since row 100 ⚠️ |
| participation | 16.9 (min 11.1, median 17.6) — a within-run trend, ⛔ **NOT a gate** (§6) |
| `loss_feat_op` | 1.382 → **0.460**; best 10-row window 0.430, last 0.463 |

**Reading.** Healthy, and the anti-collapse instrument reads the right direction. Two watch items: the tactical
target scale keeps growing (the EMA teacher SLOWS it, it does not pin it — `D-REFAV1-TAC-SCALE-SLOWED`), and the
run carries one bf16 gradient overflow that the new `--skip-nonfinite` guard caught. **That guard has already
paid for itself**: the same event in the previous arm drove the loss 0.52 → 2.26 and cost ≈ 150 steps.

### refcv3 — `/workspace/experiments/refcv3-b1-v72-30k` (A40 pod)
MEASURED 2026-09-03 08:02Z, tier T0.

| quantity | value |
|---|---|
| step | **27,300 / 40,284 (67.8 %)** |
| pace | **3.94 s/step** (uint8; 4.70 before, −13.4 % MEASURED over 143 rows) |
| ETA | ≈ 14.2 h ⇒ ends ≈ 2026-09-03 22:15Z |
| supervisor relaunches | **8** (6 kernel-OOM deaths + 3 planned switches; **0 deaths since 17,500**) |
| `oom_kill` | **24, unchanged since the fixes** |
| `eval_error` rows | **0 in 649** |
| uptime | 35,787 s uninterrupted |

**Reading, era-segmented.** The eras may never be pooled: the nav source changed at 16,500 and the eval leaked
held-out label marginals into the model until 17,500, so the first clean eval is 18,000. Movement is in units of
each column's own step-to-step scatter (median absolute consecutive difference); < 1.0 is flat.

| column | era A ≤16,500 | era C ≥18,000 (clean) |
|---|---|---|
| trajectory (oracle-selected L1) | 23.3 | **6.3** |
| strategic goal gate | 45.9 | **21.0** |
| route CE | 9.4 | 3.8 |
| longitudinal manoeuvre CE | 10.2 | 2.4 |
| lateral manoeuvre CE | 10.4 | 1.5 |
| total loss | 17.4 | 1.4 |
| anchor accuracy | 7.3 | **0.9** |
| 2 s goal error | 21.1 | **0.4** |

The clean era still learns, roughly 4× more slowly, and **unevenly**: the strategic gate is opening while the 2 s
goal error and the anchor accuracy have stopped. ⛔ **Three limits travel with every number above.** `eval_traj`
scores the anchor NEAREST THE GROUND TRUTH — the model's own choice agrees on 57 % — and is a mean L1 per
coordinate, **not an ADE**. **No confidence interval is computable from this file by any estimator**: each value
is already a pooled mean over 160 windows, with no per-window rows and no episode ids, and that is not
recoverable later. And this reading **DISAGREES** with the banked package's, which found era C flat in every
column at n = 15 through 25,650, against this n = 18 through 27,300 with a different scatter estimator. Both are
recorded; reconciliation is BACKLOG R33, not a coin toss.

**Recommendation (unchanged): read the epoch-end checkpoint tonight; do not extend the run.**

---

## 2. What landed since the last report — 42 commits, 2026-09-02 21:00 → 2026-09-03 10:00

| stream | produced | state |
|---|---|---|
| refav1 epoch | speed-channel relaunch on PI directive; EMA-resume path; bf16 overflow guard live and verified | ✅ integrated |
| Benchmarks (T1) | the first real T1 read; the `ha0` constant-velocity arm; the trivial-profile instrument | ✅ integrated |
| ArchInf (diagnostics) | the anchored action diagnostic; the cost-surface panel; the tactical-decoder panel | ✅ integrated |
| Data Engineering | the steer/curvature interface resolved; the encoding wheelbase settled at 2.9 exactly | ✅ integrated |
| v7 line | v7.2 label wiring, B1 parity key, truncated BPTT; eval-clip exclusion; the DINOv3 seed converter | ✅ integrated |
| Research Lab | four SOTA themes, 14 primaries banked; the P2 probe-leak audit | ✅ 3 of 4 themes; 🔶 decodability incomplete |
| refcv3 T1 adapter | a 1,746-line draft, never tested or documented | 🔶 **rescued** from a single disk, filed UNVERIFIED (R20) |

---

## 3. The scientific position, in one paragraph

**refav1 trains cleanly and does not yet drive.** Its representation is healthy — collapse solved and validated by
a deliberate-regression arm — and its predictor **hears the lateral action**: the anchored displacement response
is 2,298× / 11.8× its own permutation null, perfectly linear and antisymmetric (`D-ACTDIV-ANCHORED-REFAV1`). Yet
every closed-loop plan is the constant-velocity straight line on 140/140 eval windows, and the cause is now
localised to **the cost and the goal, not the world model**: the goal IS the constant-velocity rollout on 79–92 %
of windows, the decoder emits `LANE_KEEP` on 140/140 and asks for a turn on 0 of 27 human-turn windows, the
`0.05·κ²` penalty is 99.5 % of all cost variation, and `1−cos` spans **2 float32 ULPs** along curvature against
~19,804 along acceleration. The zero-GPU experiment did NOT refute that line; its other pre-registered branch
fired instead — **the defect is the cost**.

---

## 4. The four edges — evidence grade each

| edge | grade | the evidence |
|---|---|---|
| **Planning** | ⛔ **NEGATIVE, and now explained** | No admissible T1 number for any arm. refav1's deployed plan is the CV baseline; the paired read across two checkpoints was VOID (bit-identical driving); and the tactical decoder's apparent skill is **oracle nav** — under nav-zero its ranking collapses to 0.520 and a nav-only predictor beats the model (0.684 > 0.650). Nav will not exist at deployment. |
| **Efficiency** | ✅ **STRONG** | refav1 20.6 → 3.49 s/step (6.1×, bf16 + TF32); refcv3 −13.4 % on uint8 with `/dev/shm` 19 → 4.7 GB; the overflow guard converts a ~150-step loss into a dropped batch; 0 deaths in 9.9 h against 6 in the preceding era. |
| **Safety / self-knowledge** | ✅ **STRONG — the edge that carried the day** | **Nine RETRACTION_LOG entries dated 2026-09-03**, every one self-caught by the measurement it commissioned, none reaching the register as fact. They include a retired gate still commanded by our own validation skill, and a pre-registration whose launch line would have frozen the trunk it was written to unfreeze. |
| **Data efficiency** | 🔶 **MODERATE** | Corpus parity holds (4,572 clips, both arms); the eval-clip exclusion is closed and ON by default; the encoding wheelbase is settled exactly. But **5 of 16 tactical classes have ZERO support** and 8.13 % of batch-8 steps see no labelled row — a class-balanced loss cannot conjure support that is absent (R40). |

---

## 5. Decisions required from the PI — with defaults

1. **DINOv3 ViT-B/16 pull** (R24). Only ViT-L/16 is on the box; the converter never downloads; HF quota is a hard
   ceiling. *Default if unanswered: seed from the on-box ViT-L/16 and accept the predictor budget cost.*
2. **`plan_horizon_s` 2.0 → 6.0** (R34/R38) — now load-bearing. The planner searches 2 s and is scored against a
   6 s goal, so its own canonical seed cannot reproduce its own goal: 62 of 64 token pairs differ and a `TURN`
   over-rotates by 82.5°. The implemented repair buys **identity, not horizon**. *Default: do NOT change it before
   the tiny-ladder arm — it moves the search dimensionality, the proposal head's output shape and every banked
   comparison.*
3. **Does the doctrine admit as T1 a model that consumes NO actions?** (R30) — blocks register decision 9. refcv3
   is one-shot with no rollout, so the closed loop cannot be ported. *Default: score refcv3 at its own tier,
   report each arm's margin over the shared `ha0` floor per family, paired, and refuse any head-to-head of levels.*
4. **When `--action-units steer` becomes the default** (R21). Banked lateral numbers are not comparable across the
   flip. *Default: flip at the next epoch boundary with a stated cut-over.*
5. **Readiness decision G.1 must NOT be adopted on the drift metric** — an endpoint-shuffled control reproduces
   100–102 % of "drift". *Default: re-measure with the control subtracted before adopting.*

---

## 6. Incidents, honestly

- ⛔ **A retired gate was still commanded by our own validation skill.** The `participation ≥ 8.56` floor is not
  reproducible — the corpus it is sourced to reads 5.756 through the same function — and the code and the registry
  retired it on 2026-08-23. The skill every session loads still listed it, so this session quoted it back as a
  cleared gate all night. Corrected; the PASS language is withdrawn; the readings stand as a within-run trend.
- ⛔ **A pre-registration argued for a trainable trunk and instructed a frozen one.** `PREREG_V7F.md` §9's launch
  line specified a stage that trains only the strategic layer. Caught by the agent implementing it, two hours after
  it was committed. Corrected in place, with two further defects in the same line.
- ⛔ **Three of my own framings in the planner area were wrong in one day**: the three-factor ordering of the flat
  plan, the "starved tactical head" premise, and the chord cost's sufficiency. Each was refuted by the measurement
  it commissioned. None reached the register as fact.
- ⚠️ **Work was stranded on a single disk twice** — a completed diagnostic package and a 1,746-line adapter, both
  rescued by later agents. The "only one place?" column in every manifest is now checked before a stream closes.
- ⚠️ **Two rate-limit events killed five agents** mid-task; all were resumed or respawned on a different model.
- ⚠️ **Tooling traps banked**: a background sync reverts mirror edits; `Select-String -Path` silently returned 0
  hits for markers that were present (`-LiteralPath` found them); a pipeline's exit status is its last stage's.

---

## 7. Ordered next steps

1. **Tonight, ≈ 22:15Z** — refcv3's epoch ends. Read the end-of-epoch checkpoint at T1 per the banked checklist
   (`2026-09-03-refcv3-epoch-conclusions/T1_CHECKLIST.md`); the tier ruling (§5.3) gates what may be claimed.
2. **≈ 2026-09-04 00:50Z** — refav1's speed epoch ends; the first admissible T1 read of a full-epoch refav1.
3. **Now, 0 GPU** — the cost repair (chord + L4 jointly; the chord alone flips 1 of 25 windows and silently
   reweights by 5,793×), then the tactical-decoder arms, whose preflight is discharged.
4. **Queued** — wire `--w-trunk-anchor` (blocks v7f rung R3); triage the refcv3 T1 adapter (R20); read the banked
   GS-9 transition-probe JSON (R12); finish the decodability SOTA theme.

---

*Compiled by the Master Mind. Commit route: the scratch-index plumbing pattern, because `scoped_commit.py`'s
`read-tree` has been dying with in-page errors on this mount all night. ⛔ Not pushed — the standing rule is
commit only, never push.*
