# E4 RESOLVED — the S-T gate is readable, and the route to a tactical selector was DEAD and is now open

**Date:** 2026-08-17 · **Branch:** `agent/arch-inf-20260803` (started at HEAD `8bc4b69`; HEAD advanced to `96614b8` mid-session — see §8) · **Author:** arch-inf subagent (E4 stream)
**PI decision this serves, verbatim:** *"solve the s-t gate contradiction, eventually we need a tactical selector..."*
**Nothing was launched. Thor's GPU was never touched** — every Thor call was one read-only `ssh -n`. All verification ran CPU-only on the dev box.

---

> ## ⛔ ESCALATION — THREE THINGS THE ORCHESTRATOR MUST CARRY
>
> **1. ⭐ THE ROUTE TO A TACTICAL SELECTOR WAS DEAD, AND THAT IS THE REAL FINDING.** SEL-1's *pre-registered* reopening path — the only thing that lifts the refusal — **could not return FUNDED for any measurement whatsoever**. `read_sw_admission` looked for a top-level `sigma_2s_m`; `scripts/e_wc2_sigma_star.py` writes the per-axis 2 s σ at `references_and_ratios.sigma_perax_2s_m`. **Name AND level both differed.** MEASURED by executing the real estimator on a dump with a **planted σ of 0.30 m** — deep inside the FUNDED band — which it recovered as **0.3026** (0.9 % of planted) and which the reader still turned into `verdict: null`, and `assert_selector_admissible` still **REFUSED the launch**. Repaired; the path now returns FUNDED / INCONCLUSIVE / REFUSED correctly, proven on all three. `raw/sel1_reopen_path_{BEFORE,AFTER}.json`.
>
> **2. ⛔ ONE WORK ITEM BLOCKS THE PI'S "EVENTUALLY", AND IT IS NOT A DECISION — IT IS A SCRIPT.** Step 1 of the four-step admission recipe **does not exist**: nothing in `stack/scripts/` dumps v6 S-W latents in E-WC2's dump contract (MEASURED at three probes; `refc_dump_latents.build_model` builds a `RefCModel` and cannot load a v6 checkpoint). Steps 2–4 are built and runnable today. The recipe is now **emitted** by `v6_chain.py admission` with step 1 flagged `⛔ NOT BUILT`, rather than left in a docstring. ⚠️ **The measurement's only cheap window is the S-W→S-T boundary, ~5 days out.**
>
> **3. ⚠️ I SHIPPED A FIRST VERSION THAT ERASED A FAIL, AND THE INCUMBENT SUITE CAUGHT IT — SAY SO WHEN YOU COMMIT.** Excluding a not-applicable criterion *unconditionally* meant a planted, FAILING `sel_gap` supplied through `--gate-probes` was silently discarded and the gate read **PASS on a rung that had FAILED**. I set out to fix a gate that could not report PASS and briefly built one that could not report FAIL. Fixed (a supplied verdict always beats the predicate) and pinned; §2.1.
>
> **4. ⚠️ RETRACTION AGAINST BOTH PRIOR E4 REPORTS — THE MECHANISM WAS MISATTRIBUTED, AND IT FALSIFIES OPTION (a).** `ST_LAUNCH_READINESS.md` §5.2 and `ST_LAUNCH_FIXES.md` §6 both name `train_v6_staged.py`'s `if w.w_select:` as why the gate cannot read `sel_gap`. That line is real and is **not the emitter the gate consumes**: `run_stage_gate` populates probes from `--gate-probes`, `X3_isolation`, `spectrum` and `x4_spectra` and from nothing else — **no training-loop log key ever becomes a gate probe.** ⇒ **turning on `--selector goal --w-select 1.0` would have made the LOG key appear and left the gate exactly as INCONCLUSIVE.** The *verdict* they measured was right; the *mechanism* was not. **Root-cause class: a probe read at the wrong SCOPE** — the `df`-on-a-pod / Thor-`free` / cgroup-`usage_in_bytes` family, here as two identically-named quantities at two eval tiers, the T0 one read as the T1 one.

---

## 1. The decision, and what each alternative costs

The brief offered (a) run S-T with a selector, (b) make the requirement arm-conditional, (c) both. **Chosen: (c), in the only order the evidence permits** — arm-conditional *and* a real route to the selector, where "real" means the pre-registered admission path is repaired and emitted rather than overridden.

| | what it is | what it costs |
|---|---|---|
| **(a)** enable a selector by default | delete or bypass `assert_selector_admissible`, launch S-T with `--selector goal --w-select 1.0` | ⛔ **and it does not even work.** Per escalation 3 it makes the T0 log key appear and leaves the gate INCONCLUSIVE. Beyond that it requires **overriding a FIRED pre-registration** (both outcomes committed 2026-08-16 *before* the measurement), spends **3.15 GPU-days** on an arm whose admission criterion is unmeasured, and trains a scorer on a goal input measured at σ/ADE **9.9915 [7.4492, 13.5119]** against a refusal line of 3.0 — the CI's *lower* bound is 2.48× the threshold. CLAUDE.md rule 5 settles this with a pre-registered experiment, not with deference — **including deference to a steer we agree with.** |
| **(b)** conditional requirement alone | `sel_gap` required only when a selector is configured | ⛔ leaves tactical selection unmeasured **with no route back** — the brief's own stated cost, and it is worse than it looks, because the route back was already broken (escalation 1) and nobody would have found out. |
| **(c)** ⭐ **both** | (c1) arm-conditional requirement · (c2) repair + emit the admission path | the honest cost: **the tactical-selection question stays open for this stage**, recorded as open in the certificate itself, with a named 4-step route and one script to write. |

### Why (a) is not what the PI's steer actually asks for

The steer is *"**eventually** we need a tactical selector"*. That word is doing work, and it is consistent with what the sources say. **SEL-1 did not refuse "a selector".** Read from source:

* **W7's roll-consistency re-ranker is what was refuted, and it stays refuted.** `GoalDistanceScorer`'s docstring records the discriminating measurement (2026-08-15, banked REF-C-XL fan, 881 windows / 40 episodes / 256 candidates, paired episode-cluster bootstrap): `cons_score` selects at **6.4501 m** against a fan oracle of **0.1639 m**, **+5.9787 [+5.3217, +6.7625] WORSE** than the shipped supervised selector, with normalised error-rank **RISING** with the candidate count (0.241 at N=4 → 0.286 at N=256) and lower-tail hit rate **collapsing** (0.57 → 0.28). That is the winner's curse.
* **The goal-distance rule behaves the opposite way** — error-rank **FALLS** with N (0.006 → 0.001), lower-tail hit rate **1.00**, *because a candidate-independent reference has no degenerate minimiser: inaction cannot minimise it.*
* **And its requirement is measured, not assumed.** At σ **0.5 m** the goal rule is **−0.1591 [−0.2300, −0.0894] BETTER** than the trained selector (separated); at σ **1.0 m** it is **+0.0943 [+0.0241, +0.1650] WORSE** (separated). ⇒ *"the goal head must reach ≈0.8 m 1-σ endpoint accuracy to be worth having, and that is a gate, not a hope."*

⇒ **SEL-1 refused a goal-distance selector whose goal input had not been shown predictable enough on the available surface, and it named the measurement that would settle it.** `SW_LATENT_ADMISSION` pre-registers exactly that 0.8 m line (FUNDED ≤ 0.80 · INCONCLUSIVE ≤ 1.41 · REFUSED above), at a cost of **~10–25 GPU-min at the S-W→S-T boundary**. **"Eventually" is satisfied by taking the measurement — which is now possible, and was not.**

⚠️ Stated plainly, because it is the honest cost of (c): **if the measurement comes back INCONCLUSIVE or REFUSED, S-T ships without a certified tactical selector.** That is a real gap, it is recorded in the certificate rather than in a report, and it is not closed by this stream.

---

## 2. (c1) — the arm-conditional requirement, and why it is a STRENGTHENING

**The structural fact, MEASURED on the production stack rather than read from source** (`raw/st_relaunch_verified.json`): built at the real S-T geometry from the real S-W checkpoint, one forward pass emits **`sel_*` keys: `[]`**. On a `--selector none` arm `V6Stack` emits **no selection key at all** — the whole block is under `if self.cand_score is not None`. So there is no `sel_idx`, `tactical.sel_gap_tac` has no argument, `taniteval.selgap` has nothing to score. **`sel_gap` there is not "not yet run" — it is UNCOMPUTABLE, and no battery run can produce it.**

The fix is `GATE_APPLICABILITY` + `probe_applies` + `arm_record` in `train_v6_staged.py`:

| | before | after |
|---|---|---|
| criterion in `STAGE_GATE_SPEC["S-T"]["required"]` | `sel_gap` | ⭐ **unchanged — `sel_gap` is not deleted and not demoted** |
| on an arm **with** a scorer | required → INCONCLUSIVE (nothing could ever supply it in-loop) | **required, and it BINDS** |
| on an arm **without** a scorer | required → INCONCLUSIVE **forever** | **NOT APPLICABLE**, recorded with its reason and the pre-registered measurement that changes it |
| both arms' verdicts | ⛔ **identical** — the gate was equally uninformative about both | distinguishable; only the selector arm can be certified, and it cannot escape certification |

**Four properties that stop it becoming a loophole, each pinned by a test:**

1. ⛔ **Applicability is resolved from the BUILT STACK** (`stack.cand_score is not None`), never from the `--selector` flag. A flag is an intention; the object is what happened. Adjudicating on the flag would certify a selector that failed to build — the `intent_proj` defect (a path present in the declaration, absent from the optimisation) wearing a gate's costume.
2. ⛔ **`arm=None` adjudicates everything as applicable.** Forgetting to describe the arm can only make the gate *harder* to pass. Today's behaviour is preserved byte-for-byte for every existing caller.
3. ⛔ **A gate with nothing left to check is REFUSED.** If every required criterion became not-applicable the gate would "pass" while measuring nothing; `vacuous_gate` forces INCONCLUSIVE instead.
4. ⛔ **A SUPPLIED verdict always beats the predicate** — see §2.1, which is a defect I shipped and the incumbent suite caught.

### 2.1 ⚠️ The defect I shipped, and what caught it

**My first version of `stage_gate_dict` excluded a not-applicable criterion unconditionally.** `test_the_whole_ladder_hands_off_through_the_WRITTEN_files` — an *incumbent* test, not one of mine — plants a **failing** `sel_gap` (`{"pass": false, "value": 0.91}`) through `--gate-probes` on a fixture stack with no scorer, and asserts the ladder stops there. Under my change the planted FAIL was silently discarded and **the gate read PASS on a rung that had FAILED.**

⛔ **That is the erasure of a FAIL — the worst thing a gate can do — arriving through the very mechanism built to stop vacuous verdicts.** It is the same shape as the defect being fixed, one reflection over: I set out to stop a gate that could not report PASS and briefly built one that could not report FAIL.

**The rule that was missing, now explicit:** applicability answers *"can this arm produce the quantity?"* — it **never** licenses discarding a quantity somebody supplied. A probe present with a non-`None` `pass` is adjudicated regardless of the predicate, and the contradiction (*"this arm cannot produce it"* vs *"here is a value for it"*) is surfaced in a new `applicability_conflicts` block rather than resolved silently in either direction, because **one of the two is wrong** and the gate cannot tell which.

⚠️ **The lesson generalises past this change:** a predicate that gates *whether evidence counts* must never be able to outrank *evidence that exists*. Root-cause class: **a filter applied to the wrong side of the join** — it belonged on "which criteria do we go looking for", not on "which measurements do we believe".

And **"not applicable" is never "pass"**: `UNMEASURED_BY_CONSTRUCTION` is stamped into the certificate — the question, why this arm cannot answer it, why this arm was chosen, and the exact four-step route that would change it. `assert_stage_precondition` then **prints it at the next stage's launch** and returns `prev_pass_scope`, so an operator learns that a PASS did not cover tactical selection at the one moment they can still act. That is the four-families rule verbatim: *a family that cannot be computed is declared per family with the reason and the n, never silently dropped.*

---

## 3. ⛔ (c2) — the reopening path was DEAD. Executed, not read.

`assert_selector_admissible` refuses every selector step unless `<sw_dir>/ewc2_sw_latents.json` reports σ ≤ 0.80 m. That is the *only* thing that lifts SEL-1.

**MEASURED 2026-08-17** (`code/sel1_reopen_probe.py`, `raw/sel1_reopen_path_BEFORE.json`) — the real estimator, the real reader, the real chain refusal, on a synthetic dump with a planted σ:

| | |
|---|---|
| planted σ (per-axis, m) | **0.30** — deep inside the FUNDED band (≤ 0.80) |
| estimator recovered | **0.3026** (0.9 % of planted) |
| reader expected | `sigma_2s_m`, **top level** |
| instrument emits | `references_and_ratios.sigma_perax_2s_m` |
| present at top level / anywhere | **false** / **false** |
| `read_sw_admission` | ⛔ `verdict: null` — *"the probe did not report the quantity the threshold is defined on"* |
| `assert_selector_admissible` | ⛔ **REFUSED** |

⇒ **SEL-1 could not be reopened by any measurement.** It is the **mirror of E4**: E4 is a gate that cannot read PASS; this is an admission gate that cannot read FUNDED. Together they made the PI's "eventually" unreachable by construction on both sides of the same question — *and a gate that cannot report PASS and a gate that cannot report FAIL are the same defect.*

⚠️ **ROOT-CAUSE CLASS: A FIXTURE THAT MODELS THE CONSUMER'S EXPECTATION INSTEAD OF THE PRODUCER'S OUTPUT.** `test_v6_chain.py`'s `write_admission` wrote `{"sigma_2s_m": sigma}` — the shape the *reader* wanted — so the join between instrument and reader was never exercised. Same family as `touch_ancestor` writing a `stage_gate.json` with no `config.json` (`ST_LAUNCH_FIXES.md` §9), and as `assert_geometry_carry` comparing 2 of the 76 fields it had in hand. **The fixture now writes the instrument's real nesting by default**, and the replacement pin **runs the real estimator**.

**The repair** (`resolve_sw_sigma`), and the two things it refuses:

* the search locations are **data** (`SW_SIGMA_LOCATIONS`), tried in order, legacy flat form kept last so a hand-written artifact still reads;
* ⛔ **the radial unit is refused by name.** `sigma_radial_rms_m` is **√2×** the per-axis σ the 0.80/1.41 thresholds are defined on (the estimator pins the unit itself: 0.8 / 0.4714 = 1.70 reproduces §3.1's published ratio). Reading it would **flip FUNDED → INCONCLUSIVE on arithmetic alone** — a true 0.56 m surface reads 0.79 and squeaks in; a true 0.79 reads 1.12 and is refused. So the resolver refuses it explicitly rather than merely never looking at it;
* ⛔ **`sigma_6s_m` is refused too** — there is no 6 s threshold, because σ(6 s)/σ(2 s) = 3.7481 on REF-C is past the 3× line and the ratio form does not transfer.

**AFTER** (`raw/sel1_reopen_path_AFTER.json`, and pinned across all three verdicts): planted 0.30 → **FUNDED** and the launch is admitted; 1.10 → **INCONCLUSIVE**; 2.00 → **REFUSED**. Read at `references_and_ratios.sigma_perax_2s_m`, σ recovered within 10 % of planted in every case.

### The recipe is now EMITTED, with its missing step named

`v6_chain.py admission` prints `sw_admission_recipe()` — four steps, each with its status:

| step | status | what |
|---|---|---|
| **1** | ⛔ **NOT BUILT — the work item** | dump the **frozen S-W latents** on the canonical val40 grid (40 episodes, WINDOW=8, STRIDE=8, 881 windows) in E-WC2's dump contract |
| 2 | ✅ built (CPU, no GPU, no model) | `e_wc2_sigma_star.py --dump … --out <sw_dir>/ewc2_sw_latents.json --features pooled,ctx` |
| 3 | ✅ built | `v6_chain.py admission` — adjudicate against the pre-registered thresholds |
| 4 | ✅ built, reachable only if 3 says FUNDED | `v6_chain.py commands --step S-T:goal --st-arms goal` |

⚠️ **Step 1's absence survived TWO probes plus a content search** (Rule 2): no dump script emits `gt_endpoint` for a v6 checkpoint; `refc_dump_latents.build_model` builds a `RefCModel`; `probe_latent_state.py` reads v6 checkpoints but emits P1/P2 retention, not an E-WC2 dump. `e_wc2_sigma_star.py`'s own header agrees: *"S-W latents have never been dumped."* **Emitting a command for a script that does not exist would be a fabricated recipe**, so it is emitted as a build item carrying its contract.

---

## 4. What the gate will ACTUALLY read at S-T — and S-S is solved, not moved

MEASURED end-to-end on the real stack built from the real launch line (`raw/st_relaunch_verified.json`), `arm_record` = `{has_scorer: false, scorer_class: null, selector: "none"}`:

| | before this change | after |
|---|---|---|
| nothing folded in | `INCONCLUSIVE` · missing `["TACTICAL_family", "sel_gap"]` — one of which **no battery can produce** | `INCONCLUSIVE` · `required_effective ["TACTICAL_family"]` · missing `["TACTICAL_family"]` · not-applicable `["sel_gap"]` **with its route attached** |
| P1/P3/P6-equivalent battery folded in via `--gate-probes` | ⛔ **`INCONCLUSIVE` — unconditionally, forever** | ⭐ **`PASS`** |

⇒ At S-T the gate now reads **the four-families TACTICAL evaluation** (`taniteval/tools/eval_four_families.py`, T1 tier, both 0–2 s and 0–6 s), plus the reported set (`P7`, LATERAL / LONGITUDINAL / STRATEGIC families, `X2_seam`). The instruction it gives an operator is *"run the four-families eval"* — actionable — instead of naming a probe that cannot exist.

**S-S — `sel_gap_revalidated`, both branches satisfiable:**

* **no-selector lineage** — the same predicate applies, so `sel_gap_revalidated` is NOT APPLICABLE and S-S adjudicates on `STRATEGIC_family` + `TACTICAL_revalidated`. **Satisfiable.**
* **selector lineage** — the chain already carries `--selector` into S-S **for the geometry**, so the frozen `cand_score` is in the stack and emits `sel_idx` at eval time. `sel_gap_revalidated` is **REQUIRED and computable** — which is exactly what `STAGE_INVALIDATES["S-S"] = ("S-T",)` demands (S-S retrains `layer_str` → moves `e_g_tac`, the selector's only input; S-T's certificate does not survive it). **Satisfiable.**

⚠️ `w_select` is 0 at S-S and the trainer refuses the flag there — **irrelevant to this**, and precisely the misattribution escalation 3 corrects: the gate probe is the T1 eval instrument, not the train-time log key. The frozen scorer still runs in the forward pass.

⇒ **The problem is solved at both stages, not moved to one.** Neither S-T nor S-S now needs a blanket `--allow-inconclusive-gate` on account of `sel_gap`.

---

## 5. ⛔ The trilemma — proved, not asserted

*"Whatever you build, prove your gate can return each of its verdicts on a constructed input, or it is decoration."* Six cells, all executed (`test_E4_the_S_T_gate_can_return_EVERY_verdict_on_both_arms`):

| arm | PASS | FAIL | INCONCLUSIVE |
|---|---|---|---|
| `has_scorer: false` (`--selector none`) | ✅ | ✅ | ✅ |
| `has_scorer: true` (`--selector goal`) | ✅ | ✅ | ✅ |

Plus the two ways it could have gone wrong, each pinned:

* ⛔ **a FAIL is never softened into not-applicable** — applicability is resolved from the *arm*, never from the probe's own verdict, so a failing probe on an applicable arm stays a FAIL (and X5 gives a FAIL no override anywhere downstream);
* ⛔ **a vacuous gate is refused** — every-criterion-skipped forces INCONCLUSIVE.

And the admission gate's own trilemma: **FUNDED / INCONCLUSIVE / REFUSED all produced** from real estimator output at planted σ 0.30 / 1.10 / 2.00.

---

## 6. The S-T launch line — RE-DERIVED and EXECUTED

⭐ **No selector was enabled by default, so the claim is that the line is UNCHANGED — and a claim of "unchanged" is worth nothing unless it is diffed.** It was:

| | |
|---|---|
| emitted line vs banked `st_launch_line_fixed.txt` | ⭐ **byte-identical** — md5 **`5381f2ea28deb0770626b0a173207365`** both sides, empty diff |
| `--v2-lru` | **64** ✅ (chain default is 6; carried via config, as required) |
| `preflight(a)` with both PYTHONPATH roots | **`[]`** |
| params / state-dict keys | **336 575 049 / 575** |
| `encoder.pos` | `[1, 640, 768]` |
| checkpoint | `v6F_sw_step010000.fp16.pt`, **673 312 891 B**, md5 **`a4e2c0e1eb0ca455448472853ccc46d7`** (re-computed here) |
| `load_stage_init` vs the REAL checkpoint | ⭐ `missing []` · `unexpected []` · `introduced [cond_tac_dyn.bias, cond_tac_dyn.weight]` · `init_step 10000` · `prev_stage "S-W"` |
| **one real forward step** | ✅ `z_op [1, 2048]`, `plan` present, **`sel_*` keys `[]`** — the no-scorer fact, measured on the production stack |

⚠️ **A scope note, because I hit it myself.** Run without the `taniteval` root on `sys.path`, `preflight` returns E5's `--dump-seam-plan` refusal. That is the guard working: the *emitted* line sets `PYTHONPATH=<stack>:<taniteval>`, and evaluating a correct line under the wrong environment is exactly the scope error `ST_LAUNCH_FIXES.md` §7 named. With both roots, `preflight` is `[]`.

**Default-build invariant, before AND after:** **87 893 449 params / 405 keys** — unchanged. No v6 vocabulary tuple was touched; the live run's tensor-strict resume is unaffected.

---

## 7. Live-run health at hand-off (read-only)

| | |
|---|---|
| | at stream start | at hand-off |
|---|---|---|
| PID **25477** | ✅ `Ssl`, elapsed 1-22:40:31, RSS 8 883 960 kB | ✅ `Ssl`, elapsed **1-23:20:39** |
| step | 12 550 / 30 000 | ⭐ **12 650 / 30 000 — ADVANCING** |
| `step_s` | 26.4749 | **26.4731** |
| loss | — | 2.363 |
| supervisor | none | ⚠️ still **none** — a crash ends the run with no relaunch |

Nothing was started, stopped or loaded on Thor: two `ssh -n` reads of `ps` plus the tail of `train_log.jsonl`. ⚠️ The second read emits an **opaque `ZZ…ZZ` marker computed pod-side** rather than grepping the raw stream, per the PTY-echo trap — a client-side filter containing the token it searches for matches the echoed command and invents a failure.

---

## 8. Suite

Run with the interpreter named: `cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q`.

| suite | result | baseline | delta |
|---|---|---|---|
| `stack` | **3816 passed · 0 failed · 7 skipped · 2 xfailed** (584 s) | 3803 / 0 / 7 / 2 | ⭐ **+13 — exactly this stream** (16 tests added, the 3 E4 pins replaced) |
| `taniteval` | **1101 passed · 0 failed** (205 s) | 1092 / 0 | ⚠️ **+9 is NOT mine** — see below |
| ladder + chain + runbook subset | **275 passed** | — | — |

⚠️ **THE REPO ADVANCED MID-SESSION AND I CHECKED RATHER THAN ASSUMED.** HEAD moved `8bc4b69 → 96614b8` (three commits by other streams) while this work was in flight. `git diff --name-only 8bc4b69..96614b8` shows the `taniteval` +9 comes from `taniteval/tests/test_cem_is_seeded.py` and `test_ridge_intercept_penalty.py`, **not** from this stream, and that **none of those commits touched any of my four files** — so the `stack` +13 is fully attributable and the `taniteval` delta is fully explained. Reporting `taniteval` as "+9, mine" would have been a provenance error of exactly the kind the operating standard exists to stop.

---

## 9. Evidence classes

| claim | class |
|---|---|
| every param/key count, md5, `missing/unexpected/introduced_keys`, forward output, gate verdict, launch-line diff, planted-vs-recovered σ, step/`step_s` | **MEASURED (ours)** — producers in `code/`, outputs in `raw/` |
| the "reopening path was dead" finding, both directions | **MEASURED (ours)** — real estimator → real reader → real chain refusal, before and after |
| step 1's absence (no v6 S-W latent dumper) | **MEASURED (ours)** — three probes (script listing, `gt_endpoint` content search, `V6Stack` consumer search), plus the instrument's own header agreeing |
| SEL-1's σ/ADE 9.9915 [7.4492, 13.5119] vs the 3.0 line | **INHERITED** — quoted from `v6_chain.SEL1_ADMISSION` (source), not re-measured |
| the winner's-curse and requirement-curve numbers (6.4501 / 0.1639 / +5.9787; σ 0.5 −0.1591, σ 1.0 +0.0943) | **INHERITED** — quoted from `GoalDistanceScorer`'s docstring (source), not re-measured |
| "S-S's `sel_gap_revalidated` is computable on a selector lineage" | **MEASURED for the gate logic** (both branches adjudicated in tests) · ⚠️ **UNVERIFIED that a frozen `cand_score` emits `sel_idx` under a real S-S eval** — no S-S run exists |
| "the E-WC2-SW measurement will admit or refuse a selector" | **PRE-REGISTERED** (2026-08-16, before the dump was taken) — the dump has not been taken |

---

## 10. Deliverable manifest

| artifact | where it lives | staged |
|---|---|---|
| `E4_SELECTOR_RESOLUTION.md` (this file) | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-17-e4-selector-resolution/` | yes |
| `code/sel1_reopen_probe.py` — executes SEL-1's reopening path end-to-end | same, `code/` | yes |
| `code/st_relaunch_verify.py` — re-derives the S-T line, diffs it, loads the REAL checkpoint, takes one forward step, reads the gate | same, `code/` | yes |
| `code/default_build_invariant.py` — the 87,893,449 / 405 check | same, `code/` | yes |
| `raw/sel1_reopen_path_BEFORE.json` · `raw/sel1_reopen_path_AFTER.json` | same, `raw/` | yes |
| `raw/st_relaunch_verified.json` — every number in §6 and §4 | same, `raw/` | yes |
| `stack/scripts/train_v6_staged.py` — `GATE_APPLICABILITY`, `probe_applies`, `arm_record`, `UNMEASURED_BY_CONSTRUCTION`, `SEL_GAP_TIER_NOTE`; arm-aware `stage_gate_dict` / `run_stage_gate` / `assert_stage_precondition` | repo | yes |
| `stack/scripts/v6_chain.py` — `resolve_sw_sigma`, `SW_SIGMA_LOCATIONS`, `SW_SIGMA_FORBIDDEN`, `sw_admission_recipe`, repaired `read_sw_admission`, corrected `SW_LATENT_ADMISSION["field"]` | repo | yes |
| `stack/tests/test_v6_st_launch_fixes.py` — the 3 E4 pins **rewritten to pin the resolution**, + the trilemma, + the executed reopening-path pins | repo | yes |
| `stack/tests/test_v6_chain.py` — `write_admission` fixture writes the INSTRUMENT's shape | repo | yes |

**Nothing is stranded.** No pod file was written; no Thor artifact was created or modified.

⛔ **Not in this stream's ownership and NOT touched:** `Project Steering/MODEL_REGISTRY.md`, `V6_GO_PACKAGE.md`, any v6 vocabulary tuple.
