# Paper v1.1 — what landed, where each number came from, and the gate-bar re-audit

**Date** 2026-08-18 · **Branch** `agent/arch-inf-20260803` · **Agent** paper-update
**GPU** ⛔ **none.** Writing and verification only. Thor was not touched; no trainer, pod, checkpoint
or cache was read. Every number below was taken from a **raw eval JSON opened in this turn**, never
from a summary, a retraction's restatement, or a program report.
**Cite this document by SECTION HEADING, never by line number** — two of its sources were rewritten
in place last night and every line-number citation into them is void.

---

## 1. ⭐ THE ONE DECISION-GRADE ITEM, AND IT IS A SAVING

⛔ **`LF0` was still standing in the paper as future work, and it has already been executed —
negatively.**

* `Paper/TANITAD_PAPER.md` §10 item 10 read: *"LF0 first — probe pre-pool spatial tokens and the P8
  attempt-2 decoded-BEV read-off (~0.5 GPU-h, decides whether the defect is routing or learning)"*,
  justified by the PUBLISHED DINO-WM line that pooling is where geometry goes to die.
* **`E-R1-0`'s 1:1 arm IS that pre-pool probe.** MEASURED: removing the pool entirely moves r² by
  **|Δ| ≤ 0.0002 with the paired episode-cluster-bootstrap CI containing zero on all five seeds** on
  the four rungs the hypothesis was built to explain.

⇒ **The defect is LEARNING, not ROUTING. The survey's RC1 is measured FALSE on our stack. LF0 as a
diagnostic is SPENT and its ~0.5 GPU-h moves to the encoder arms.**

⚠️ **AND THE HALF THAT MUST TRAVEL WITH IT:** `RC2` (rare-event variable under a fixed-rank latent),
`RC3` (no spatial masking pressure) and `RC4` (horizon) are **untouched by this measurement and stay
open**. ⛔ *"RC1 is dead"* must not be read as *"the RC list is dead"* — the paper now says so
explicitly in §10 item 10, because that is exactly the kind of over-correction a one-line edit
invites.

⏳ **NOT EDITED, deliberately:** `Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md`
presents RC1 as one of four **hypotheses**, quoting DINO-WM rather than our measurement. It is a
**dated report**; rewriting its hypothesis list would destroy the record of what was believed when.
The paper is the living document and carries the correction.

---

## 2. WHAT CHANGED IN `Paper/TANITAD_PAPER.md`

| section | change |
|---|---|
| status line + abstract | **v1.0 → v1.1 (2026-08-18)**; new seventh-round abstract paragraph carrying all five results with tier stamps and estimators |
| ⭐ **new §7.19** | the round itself: (a) pooling refuted + DINOv2 discriminator + the `PC-2OBJ` correction; (b) the encoder falsifier battery and its adversarial re-read; (c) `planner_beats_cv` answered at **T1** with the **full four-family block**; (d) the 3-seed ladder; (e) the 4.3 % Alpamayo contamination |
| ⭐ **new §5.8** | five instrument rules **I11–I15**, continuing the I1–I10 numbering |
| §9.4 | gate-bar **re-audit** block (§3 below) |
| §10 | item 10 **retires LF0**; item 9 + item 11 carry the exclusion-list constraint; **new item 13** promotes the encoder arms; **new item 14** binds a step-30,000 re-measurement |
| §10.1 | **two new open PI decisions** — the 336.5 M vs sub-300 M envelope contradiction, and the DINOv3 licence |
| §3.10, §7.14 | LF0's executed-and-negative status recorded in place |
| Changelog | full **v1.1** entry |

---

## 3. ⭐ THE GATE-BAR RE-AUDIT — reported in both directions

**Why it was asked for:** a wrong number in a results table misreports the past; a wrong number in a
**gate** misdirects the future. The paper had already carried banned-estimator bars *inside* its
pre-registered gate list (G4 **1.69 → 1.7318**, G5 **0.452 → 0.4271**, fixed 2026-08-17).

| bar | status | evidence |
|---|---|---|
| **G4** closed-loop head baseline **1.7318** | ✅ clean | independently reproduced from banked per-window dumps in a run with no other contact with the 08-16 audit, matching to 4 dp (`…/2026-08-18-planner-beats-cv-redrive/raw/planner_beats_cv_banked_analysis.json` → `4b_every_other_verdict.G4_pass.threshold_corrected`) |
| **G5** open-loop **0.4271** | ✅ clean | same artifact, `3_openloop…banked_arms_decision_grade.operative_rollout_trueA` = **0.4271 [0.3675, 0.4871]**, `episode_cluster_bootstrap` |
| tactical-head **3.3839** | ✅ clean | same artifact, **3.3839 [2.8336, 3.9722]** |
| three trivial open-loop floors (best-of-3 **0.5005**, CTRV **0.523**, ego-status **0.5735**) | ✅ clean | `full_set` means **by construction** (`bench.py:511`, `:558`) — never touched by the estimator correction; already stated in §7.2 |
| deciding call sites on `overlapping_holdout_se` anywhere in the repo | ✅ **0** | unbounded walk, 4,696 files; 37 call sites remain, all REPORTS-only; guard is an **AST** walk (`taniteval/tests/test_no_jack_in_gates.py`) because a regex raised ≥ 176 false positives, every one on documentation *declaring* the estimator banned |
| §7.6 primary bar **`ade_0_2s ≤ 0.60`** | ✅ **clean** | derivation recorded in `V4_FLAGSHIP_DESIGN.md` **§17**: *"strictly below hold-v0 0.7876 and CV 0.8377 … and above v1's 30 k **full-set 0.4271**"* — anchored on the honest statistic |
| ⚠️ §7.6 KILL bar **`wm_canary_ade_2s ≤ 0.55`** | ⚠️ **INHERITS THE DEFECT — flagged, not silently restated** | same pre-registration, adjacent line: *"anchored on MEASURED values: v1's canary **0.452** … 0.55 is +0.10 on the baseline"*, and **0.452 is the legacy `heldout` split-mean whose `full_set` value is 0.4271**. Applying the bar's *own* rule to the honest baseline gives **0.53** ⇒ the shipped bar is ~3.8 % **looser** than intended. ⛔ **No verdict moves:** v4.1's 0.4599 passes either way; v4.2 0.7222, v4.2b 0.697 and the v4 30 k gate 1.1409 fail by 1.3–2.2× |
| §7.16 W7 gate **0.4505** | ⚠️ **UNVERIFIED provenance, no verdict at risk** | it is a pre-registered threshold; the arms measured against it fail by **7.4×** (3.3348) and **8.0×** (3.614), ~30× any measured estimator error, so no plausible correction moves the verdict |

⇒ ⭐ **THE TRANSFERABLE POINT IS THAT THIS WAS AUDITABLE AT ALL.** The clean bar and the defective one
are **adjacent lines in the same pre-registration**, and the only reason either could be checked is
that both recorded *what they were computed from*. A threshold whose provenance is unwritten cannot be
audited for a defect it may have inherited — and this programme has now found the defect in a gate
**three times** (G4, G5, and the v4 canary). The paper states that every bar minted from here carries
its source artifact **and** its estimator.

⚠️ **A PROCESS NOTE ON THIS ROW, because it nearly shipped wrong.** My first probe — content grep for
derivation language near the numbers — returned nothing, and I drafted the row as **UNVERIFIED
provenance**. The **second** probe (enumerate `Project Steering/Gates/*.json`, read the card, follow
its `note` field to the pre-registration it was materialised from) found the derivation **stated in
full**. ⇒ *Two probes, and the weaker conclusion was the one the first probe supported.* The same
sequence caught the registry error in §4.

---

## 4. PROVENANCE OF EVERY NUMBER THE PAPER GAINED

⚠️ **A CLAIM I DREW AND THEN CAUGHT WITH MY OWN SECOND PROBE, recorded because the class is the point.**
My first draft of this section read *"`MODEL_REGISTRY.md` does not yet carry any of this round."*
**That is FALSE.** Probe 2 (artifact filenames, arm names, distinctive digits: `er10_main`,
`falsifier_summary`, `0.44997`, `336,542,025`, …) returned **0 hits** and would have shipped the wrong
claim; **probe 1** (retraction ids + concept names) returned **1** — `MODEL_REGISTRY.md` **§5** carries
a full, current **C101** block whose T1 numbers and four-family rows **match this paper's §7.19c to
every digit**. ⇒ *Absence found by one search pattern is not absence — and the pattern that failed was
the more "specific" one.*

**Corrected state, at two probes:**

| round item | in the registry? |
|---|---|
| **C101** — `planner_beats_cv` at T1, four families | ✅ **YES**, `MODEL_REGISTRY.md` **§5**, dated 2026-08-18. The paper cites it *and* the raw JSON |
| **C104** (pooling refuted / DINOv2 discriminator) | ⛔ **no** |
| **C106 + C109** (encoder falsifier battery + adversarial re-read) | ⛔ **no** |
| **C107** (3-seed ladder) | ⛔ **no** |
| **C112** (201-clip overlap) | ⛔ **no** |

For the four absent items the quotable source is therefore the **raw eval JSON**, opened directly.
Registry rows are owed for them and are escalated in §6.

| paper claim | raw artifact | key read |
|---|---|---|
| pooling ladder Δr², 4 rungs, 5 seeds | `…/incoming/2026-08-18-pooling-ladder-ER10/raw/er10_main.json` | `deltas_vs_p40.<rung>.p1.delta_r2c_mean` and `…_per_seed.*.separated` |
| DINOv2 vs ours through the deployed pool | `…/2026-08-18-pooling-ladder-ER10/raw/er10_dino.json`, `…/er10_main.json` | `arms.p40.targets.<rung>.r2_ceiling_mean` |
| 4-arm falsifier table | `…/Research/2026-08-18-encoder-experiments/raw/falsifier_summary.json` | `arms.<arm>.targets.<rung>.r2_mean`, `K1_PASSES_n` |
| Δr²c **+0.150 [+0.055, +0.226]**, 27/27 | `…/incoming/2026-08-18-c106-adversarial/raw/delta_randenc_s*_base_vs_ours_base.json` | paired episode-cluster bootstrap, 70 clusters |
| rank collapse **1.223 vs 67.1–68.2**, design **6.73 vs 16.0–16.9**, top dir **0.97642** | `…/2026-08-18-c106-adversarial/raw/rank.json` | `arms.{trained,random_s*}` |
| **T1** planner/CV/operative + paired deltas + four families | `…/incoming/2026-08-18-planner-beats-cv-redrive/raw/planner_beats_cv_banked_analysis.json` | `4_closedloop_planner_vs_cv_NEW`, `5_four_metric_families_closedloop` |
| 3-seed ladder inventory, α-freeze mechanism | `…/incoming/2026-08-18-ladder-3seed/` (via `…/2026-08-18-citation-sweep/raw/canonical_requote_table.json`) | re-derived from per-seed JSON by the sweep, not copied from a retraction |
| 201-clip overlap + exclusion list | `…/incoming/2026-08-17-thor-concurrency-pilot/alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt` | overlap computed from ids against `/proc/<pid>/cmdline`'s own `--v2-cache` |

### 4.1 ⚠️ Three numbers corrected while transcribing them — the reason the artifact is opened

1. *"pretraining collapses the teacher 27× / 24×"* (as stated in the retraction) recomputes from
   `falsifier_summary.json` as **26× / 23×** (0.70593/0.02660 = 26.5; 0.42743/0.01843 = 23.2). The
   paper carries the **r² pairs** and the recomputed multipliers. *(`lead_closing` 46× is exact.)*
2. The C106 bracket `[0.1736, 0.2011]` is described in one place as a **projection**-seed spread; the
   artifact shows it is the min–max over the three **initialisation** seeds, each of which is itself
   already a mean over three projection seeds. **Both descriptions agree it is a dispersion and not an
   estimator**, which is the load-bearing part — the paper states the axis as the JSON has it.
3. The falsifier battery runs on **1,120 / 1,362** windows, not E-R1-0's **1,302 / 1,507** — an arm
   must be buildable on all four encoders. The paper says so rather than implying one window set.

---

## 5. THE FOUR METRIC FAMILIES

Per the binding rule, each family with its reason and `n` where it does not apply.
⛔ **This document produces no new measurement**; the families below describe what the paper's new
§7.19 reports, which is where the contract binds.

| family | what §7.19 now carries | gap |
|---|---|---|
| **LONGITUDINAL** | ✅ **complete at T1** for the planner/CV comparison — along-track RMSE per horizon, speed error and **speed bias** (`+0.2737` vs `−0.0995` m/s), which is where the planner's loss actually is. In §7.19a/b the longitudinal *state* rungs (`ego_v0`, `lead_gap`, `lead_closing`, `lead_inv_ttc`) are the measurement | ⚠️ **distance-keeping n/a, n = 221** — no lead-agent track in the banked dump, and the P2 cost omits the gap term for the same reason. `obstacle.offline` is a pod-side join not present in this eval |
| **LATERAL** | ✅ **complete at T1** — cross-track, heading, **curvature and yaw-rate**, the last two on moving-only windows (n = 205, 16 excluded) with the exclusion published | ⚠️ in §7.19a/b `ego_yawrate` and `ego_curv` still **lack a positive control**; an unverified negative, carried forward |
| **TACTICAL** | ⚠️ **n/a, n = 221** at T1 — the CEM **emits no manoeuvre class** (continuous action search), so there is no discrete decision to score. ⚠️ **n/a, n = 0** in §7.19a/b/d — the ladder is regression-only and no manoeuvre label is banked in those caches | **a work item, not an excuse**: scoring it needs a manoeuvre labeller applied to the planned *and* the GT path |
| **STRATEGIC** | ⚠️ **n/a, n = 221** at T1 — the P2 cost carries **no route or goal term at all**. ⚠️ **n/a, n = 0** in §7.19a/b/d | genuinely uncomputable on PhysicalAI-AV (no map, lane graph, junction annotation or route signal; §5.5), and blocked behind the PH0→PH2 label stream |

---

## 6. ⛔ ESCALATIONS — decisions, not README notes

1. ⛔ **`MODEL_REGISTRY.md` OWES ROWS FOR FOUR OF THE FIVE ITEMS.** Verified at two probes (§4): it is
   **current on C101** (§5, the T1 planner-vs-CV block, matching digit-for-digit) and carries **nothing**
   on **C104**, **C106/C109**, **C107** or **C112**. The paper quotes raw JSON directly for those four,
   which is admissible — but it means the registry is **not the most current source on the S-W arm**,
   which is the exact inversion the source-of-truth rule exists to prevent. **Owner: the registry
   stream.** Minimum rows owed: E-R1-0's pooling ladder + the DINOv2 discriminator; the encoder
   falsifier battery **with C109's corrections folded in, not as C106 originally stated it**; the
   3-seed ladder inventory; and the Alpamayo 201-clip overlap with its exclusion list.
   ⚠️ **The C106 row in particular must not be written from C106** — the ratio is withdrawn and the
   `lead_gap` half is dead (§7.19b); a registry row copying the retraction would re-publish both.
2. ⛔ **PI DECISION — the "sub-300 M" envelope.** The live successor run is **336,542,025** parameters,
   **12.2 % over** the invariant that §10's dominance definition is stated against. Not a silent
   breach (`param_budget: 350000000` by design), but the paper cannot resolve it: either the envelope
   is restated or the model is rescoped, and whichever is chosen must be applied to §10 item 11's
   definition of dominance **in the same change**. Filed in §10.1 as open decision 6.
3. ⏳ **PI ACTION — the DINOv3 licence.** Gated `manual`; our token receives 403 at three probes. It
   blocks only the *stronger* positive arm; the ungated `dinov2-base` substitute already carries the
   negative result. ⛔ **No mirror or re-upload is an acceptable route around it.** Filed as open
   decision 7.
4. ⚠️ **The exclusion list is a per-definition obligation, not a one-off.** The 201 ids must be applied
   **wherever an Alpamayo eval split is defined**. Written into §10 items 9 and 11; it belongs in the
   split-construction code, which this stream does not own.
5. ⚠️ **`POOLING_BOTTLENECK_R1R2.md` is still statused `DESIGN + PRE-REGISTRATION` for an experiment
   that has been refuted.** A banner was added by the citation sweep, but retiring or rescoping another
   stream's pre-registration is the PI's call, and R2 was *promoted* rather than dropped — so the file
   is not simply obsolete. Unchanged by this stream; recorded so it is not lost.

---

## 7. VERIFICATION OF THIS TURN'S OWN WORK

* **Numbers**: every figure the paper gained was read from a raw JSON in this turn (§4), and three
  were **corrected against the retraction that reported them** (§4.1).
* **Citations**: the paper's new text cites **section headings and artifact paths only** — no line
  numbers into any document, per the sweep's rule.
* ⭐ **Absence claims — and BOTH of the ones I drafted were WRONG until the second probe ran.**
  (i) *"the registry carries none of this round"* → it carries **C101**, in full and current (§4).
  (ii) *"the 0.55/0.60 bar derivations are unrecorded"* → they are recorded **in full**, on adjacent
  lines of `V4_FLAGSHIP_DESIGN.md` §17, and reading them turned a vague flag into a precise finding
  (§3). ⇒ **the two-probe rule earned its place twice in one turn, and in both cases the FIRST probe
  supported the weaker, wronger conclusion.**
* ⛔ **Exit codes are not evidence.** Staging was verified by **blob comparison** — `git ls-files
  --stage` against `git hash-object` for the modified tracked file, and `git ls-files --cached` for
  the new ones — and **re-verified at the end of the turn**, because with several agents live the
  index moves underneath a check.
* ⚠️ **Not run: `pytest`.** This stream modified **no** file under `stack/` or `taniteval/`, so the
  suites are untouched by construction. Running one as a gate while sibling CPU jobs are live returns
  failures that are about the concurrency, not the code.

---

## 8. DELIVERABLE MANIFEST

⛔ **Nothing here lives in only one place** — both artifacts are in the repo and staged.

| artifact | path | what it is |
|---|---|---|
| the paper | `repo:Paper/TANITAD_PAPER.md` | **modified**: v1.1 — new §5.8, §7.19, §9.4 audit block, §10 items 10/13/14, §10.1 decisions 6–7, §3.10 + §7.14 in-place status, abstract + changelog |
| this record | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-18-paper-update/PAPER_UPDATE_v1_1.md` | provenance table, gate-bar audit, four families, escalations |

**Inputs read and NOT copied** (all already banked in-repo by their producing streams):
`…/incoming/2026-08-18-pooling-ladder-ER10/raw/**` · `…/Research/2026-08-18-encoder-experiments/raw/**` ·
`…/incoming/2026-08-18-c106-adversarial/raw/**` · `…/incoming/2026-08-18-planner-beats-cv-redrive/raw/**` ·
`…/incoming/2026-08-18-citation-sweep/**` · `…/incoming/2026-08-17-thor-concurrency-pilot/**` ·
`Project Steering/RETRACTION_LOG.md` (read only; **append-only and serialised — not written to**).

⛔ **STAGED, NEVER PUSHED.** No `git commit` and no `git push` was run by this agent.

### 8.1 ⚠️ THE SHARED INDEX IS NOT MINE, AND WHOEVER COMMITS NEXT MUST KNOW

`MEASURED` at the **end** of this turn (`git diff --cached --name-only`). Beyond my two deliverables,
**six paths are staged by sibling streams**, and ⛔ **I did not write, edit or stage any of them**:

```
CLAUDE.md
Project Steering/MODEL_REGISTRY.md
stack/scripts/scoped_commit.py
stack/scripts/step_time_guard.py
stack/scripts/train_v6_staged.py
stack/tests/test_step_time_guard.py
```

Recorded because a **pathspec-free `git commit` takes the whole index**, so the next committer must
either name these in the message or use `stack/scripts/scoped_commit.py`. ⭐ **`MODEL_REGISTRY.md`
being live in the index is good news for escalation 1** — a sibling appears to be writing the rows
this round owes — but it also means **my §4 registry-currency table is a snapshot of HEAD and may
already be stale in the working tree.** Re-check before quoting it.
