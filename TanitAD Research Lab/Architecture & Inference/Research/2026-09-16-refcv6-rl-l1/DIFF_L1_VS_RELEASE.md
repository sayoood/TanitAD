# DIFF — lever **L1 / D9** vs the RELEASE, element by element

**Package:** `…/2026-09-16-refcv6-rl-l1/` · **Date:** 2026-09-16 · **Worktree:** `C:/Users/Admin/tanitad-wt-rl-l1` at `9a782fa` (tip of `agent/arch-inf-20260803`) plus the files staged with this package.
**Reference for "the release":** `hustvl/DiffusionDriveV2@1cd12a1` as specified in `…/2026-09-15-ddv2-rl-prep/SPEC_DDV2_RL_PAPER.md` (§ and A-numbers below are its) and as **ported and validated** at commit `424f9e0`.
**PI instruction this file answers:** *"Keep **every** release element except the two named changes … and your DIFF file must show that."*

**Classes:** `UNCHANGED` — the same code object, not merely the same value; `UNCHANGED-VALUE` — the same arithmetic reached by a call that now has a dispatch in front of it; `CHANGED` — one of the two declared changes; `ADDED-TELEMETRY` — a read-only diagnostic that enters no loss and no gradient.

**Evidence tags:** `[T]` pinned by `stack/tests/test_ddv2_il.py`; `[T0]` pinned by the 2026-09-15 suite `stack/tests/test_ddv2_{rl,refc_chain}.py` + `test_pdm_proxy.py`, re-run green on this worktree (**110 passed**: 73 pre-existing + 37 new); `[M]` MEASURED 2026-09-15, `…/2026-09-15-ddv2-rl-prep/RESULT.md`.

---

## 0. The headline

| | |
|---|---|
| files the lever ADDS | `stack/tanitad/rl/ddv2_il.py`, `stack/tests/test_ddv2_il.py`, `…/2026-09-16-refcv6-rl-l1/code/check_arm_l1.py` |
| files the lever EDITS | `stack/scripts/ddv2_rl_refcv5.py` (the driver) — **only** |
| files the lever does NOT touch | `stack/tanitad/rl/ddv2_rl.py` (the released arithmetic), `stack/tanitad/rl/ddv2_refc_chain.py` (the binding), `stack/tanitad/rl/pdm_proxy.py` (the reward), and every other `tanitad.rl` module the banked arms depend on |
| how "not touched" is enforced | the two changes are applied as **post-hoc transforms** of what `ddv2_rl.py` returns (`apply_lambda_scale`) and as a **dispatch in front of** the IL scalar it consumes (`imitation_term`). ⛔ `git diff` on `stack/tanitad/rl/ddv2_rl.py` is **empty**. |
| how "the release is still reachable" is enforced | `--il-form release --grad-clip 0` reproduces the 2026-09-15 recipe exactly; `IlSettings.is_release` is written into `run.json` on every run, so no record can be ambiguous about which ran [T] |

---

## 1. The TWO changes

| # | element | the release | **L1 / D9** | why |
|---|---|---|---|---|
| **C1** | the imitation term's FORM | `traj_l1.mean()` over **all** `G·N = 468` chains at all 10 rollout steps, against the one logged trajectory (`rl.py:1104-1112`) — kept, named and tested as `ddv2_il.all_modes_il` | `ddv2_il.matched_anchor_il`: the L1 lands on the **G chains of the anchor nearest the GT**; the other 464 receive **no imitation gradient at all**. The match is `ddv2_il.matched_anchor_index`, which is our own trainer's expression (`stack/scripts/refc_v3_train.py:2383-2386`) re-typed and pinned against it [T] | **[M]** the release's form collapsed refcv5-v2's 117-anchor fan by **93 %** (held-out endpoint spread **37.5 m [32.8, 42.0] → 2.45 m [1.95, 2.99]**) and produced the whole T1 regression (NORL−BASE **+0.081 m [0.051, 0.116]** of RL−BASE's +0.083 m). Intra-anchor GRPO needs within-group variation to rank; after the collapse there is almost none |
| **C1b** | the IL WEIGHT (the cheaper alternative arm) | `1.0` on rows with no positive advantage, `0.1` otherwise (`rl.py:1113-1117`) | `ddv2_il.apply_lambda_scale` multiplies BOTH by `lambda_scale`. `--il-form lambda` sets **0.1**, giving 0.01 / 0.1 — and **[M]** D6 says essentially every cold-start row carries a positive advantage, so the weight that binds is **λ ≈ 0.01**. ⛔ The release's *form* is unchanged in this arm; only the coefficient moves. `coef_rl` is returned **by identity** [T] | RESULT.md §9 LEVER 1 names it as the alternative to C1; the PI asked for it exposed "so the two can be compared". It is also the **confound control** for C1: it answers whether a surviving fan is due to mode preservation or merely to less imitation pressure |
| **C2** | gradient clipping | **none** — the release clips nothing | `ddv2_il.clip_gradients`: `torch.nn.utils.clip_grad_norm_(params, 100.0)`, **once**, after all 10 per-step `backward()` calls have accumulated and immediately before `opt.step()`. The logged `grad_norm` stays the **pre-clip** norm so it is still comparable with the 2026-09-15 logs [T] | **[M]** RL-s1 diverged: isolated spikes at steps 240-244, sustained from ≈ 297, **max grad norm 15,712.3** against RL-s0's 147.3. The release runs batch 512 (≈ 128× lower per-step variance than our batch 4); this dev box cannot buy that with scale |

⭐ **AMENDMENT A-1, 2026-09-16 — the max-norm is 100, not 1.0, and it was changed BEFORE any arm ran.** C2 was implemented at the instructed 1.0 and the problem was **escalated, not absorbed**. **[M]** on the banked logs (`raw/metrics_arm-*.jsonl`, 600 steps × 3 arms), **independently verified by the coordinator on the same files**: `> 1.0` binds on **600/600** steps of all three arms (min norm 1.87 / 2.22 / 3.45, medians 16.67 / 17.93 / 61.96) — a 17-62× rescale on *every* step; `> 100` binds on **1/600** for each stable arm and **276/600** for the diverged seed. 100 is the spike guard C2 exists to be. `--grad-clip 1.0` and `0` stay available as declared arms. **A-1 is a single scalar**: the F1 gate, `Δfan`, the arms, split, seeds, I1-I10 and the budget are untouched and not re-derived. `RESULT.md` §1 and `PREREG …` §11 carry it in full; `test_AMENDMENT_A1_the_default_clip_is_a_spike_guard_not_an_every_step_rescale` pins the value so it cannot drift back silently.

**What C1 deliberately does NOT do**

* **Not a winner-take-all over the predictions.** The match is over the fixed decoded bank, so a fixed mode is supervised and the choice is attributable to an anchor. A WTA over `path` is a moving target; pinned as a regression [T].
* **Not matched against `decoder.anchors`.** `inp.bank` IS `out["anchor_bank"]` (`refc.py:2615`), this window's **v0-rolled** vocabulary — the trainer's own refusal (`refc_v3_train.py:2376-2382`), because the fixed vocabulary would supervise a geometry the model never emitted and would read plausibly. Pinned as a regression [T].
* **Not a λ change in disguise.** The normalisation is a `mean` over the matched entries, the trainer's convention — so the **batch's total imitation budget is the same as the release's**, concentrated instead of smeared. The rejected alternative (`sum(matched)/(M·S·2)`) would have shrunk the budget 117× and confounded C1 with C1b. Stated in `ddv2_il.matched_anchor_il`'s docstring and separated from C1b by construction.

---

## 2. Every OTHER release element — UNCHANGED

The PI's list, item by item. "Code object unchanged" means the call is the same function in the same module with the same arguments; `git diff stack/tanitad/rl/ddv2_rl.py` is empty, so every row below is unchanged **by construction** and not by re-derivation.

| # | element | the release / SPEC | where it lives now | class | evidence |
|---|---|---|---|---|---|
| 1 | truncated start, **t = 8** anchored Gaussian, group-major tiling | `rl.py:813-820` | `ddv2_rl.truncated_start` / `tile_groups`; called by `ddv2_rl_refcv5.rl_step` with `trunc_t=consts.trunc_t` | UNCHANGED | [T0] bitwise vs diffusers `add_noise`; anchor-major mutant caught |
| 2 | the **10-label chain** `[18 … 0]`, one-unit `t → t−1`, `ᾱ(−1) = 1.0` | `rl.py:801-806`, `:587` | `ddv2_rl.rollout_labels` / `rollout_chain` / `abar_at` | UNCHANGED | [T0] bitwise vs the released rollout loop through the vendored scheduler |
| 3 | **two-scalar exploration**, floor **0.04**, additive pair drawn and × 0 | `rl.py:638-666` | `ddv2_rl.ddim_logprob_step` | UNCHANGED | [T0] bitwise; the two-scalar invariant and the floor each have a regression |
| 4 | **likelihood floor 0.10**, 16 summed terms, at the detached sample | `rl.py:668` | `ddv2_rl.ddim_logprob_step` | UNCHANGED | [T0] analytic literal + bitwise |
| 5 | **G = 4** | `cfg.py:37` | `DDV2.group_size`, `ChainSettings.groups` default | UNCHANGED | [T] `test_the_lever_changes_nothing_in_the_ported_release_constants` |
| 6 | **intra-anchor normalisation** `(r − mean_G)/(std_G + 1e-4)` | `rl.py:886-889` | `ddv2_rl.intra_anchor_advantage` | UNCHANGED | [T0] hand literals; the library's composite is pinned as a *different* object |
| 7 | **per-sample truncation with the ≥GT mask**, then the −1 constraint branch LAST | `rl.py:892`, `:900-902` | `ddv2_rl.intra_anchor_advantage` | UNCHANGED | [T0] literals + two regressions (no bar; veto before the mask) |
| 8 | **γ = 0.8**, weight 1.0 on the FINAL step | `rl.py:926-931` | `ddv2_rl.discount_weights` | UNCHANGED | [T0] literals |
| 9 | **REINFORCE over non-zero samples** (`c_b = max(1, #{A ≠ 0})`) | `rl.py:1096-1102` | `ddv2_rl.rl_loss_per_row` / `step_loss_weights` | UNCHANGED | [T0] all-sample-denominator mutant changes the gradient |
| 10 | the IL weights' **derivation from the advantage** (0.1 / 1.0 per row) | `rl.py:1113-1117` | `ddv2_rl.il_row_weights`, called by `step_loss_weights` | UNCHANGED (C1b scales the RESULT of it; the derivation is untouched) | [T] `apply_lambda_scale` preserves the 1.0-row / 0.1-row structure |
| 11 | IL added as a **batch-global 0-dim scalar** | `rl.py:1104-1119` (A14) | `ddv2_rl.per_step_loss` still refuses a non-scalar; every L1 form returns 0-dim | UNCHANGED-VALUE | [T] `test_imitation_term_dispatches_and_stays_a_zero_dim_scalar` |
| 12 | **AdamW lr 2e-4, wd 1e-4**, 10 % linear warmup + cosine to 1e-6 | `agent.yaml:17`, `cfg.py:120`, P suppl. §7 | `ddv2_rl_refcv5.cmd_train` — CLI defaults `D.DDV2.lr` / `D.DDV2.weight_decay`, `lr_at()` untouched | UNCHANGED | [T] constants test |
| 13 | **both release clamps ON** (decoder-input clamp; `clip_sample` inside the step) | `rl.py:826-827`, `:609-614` (A18) | `ChainSettings(input_clamp=True)`, `DDV2Constants(clip_sample=True)` — the same `--no-input-clamp` / `--no-clip-sample` flags, same defaults | UNCHANGED | [T0] a port without the clamp is pinned as *not* the release |
| 14 | **η = 1** in the RL rollout | `agent.py:118` | `DDV2.eta` | UNCHANGED | [T] constants test |
| 15 | the **NORL control's** definition (zero the policy-gradient coefficient, keep the release's IL weights) | our own control, `RESULT.md` §7 D-1 | `ddv2_rl_refcv5.rl_step`'s `arm.kind == "norl"` branch, unedited | UNCHANGED | [T0] `test_zeroed_policy_coefficient_gives_exactly_the_il_only_gradient` |
| 16 | the **reward proxy**, the corpus, the split, the window filter, the deployed held-out read | `pdm_proxy.py`, `cmd_heldout`, `_select` | untouched | UNCHANGED | `git diff` empty on `pdm_proxy.py`; `cmd_heldout` and `_select` unedited |

---

## 3. What the driver's diff actually contains

`stack/scripts/ddv2_rl_refcv5.py`, the only edited file. Every hunk:

| hunk | what | class |
|---|---|---|
| module docstring | names the lever and the two flags | doc |
| `import ddv2_il as L`, `IL_FORM_FLAGS` | flag → `(form, lambda_scale)` | wiring [T] |
| `ChainArm.__init__` | takes `il`; **`il=None` defaults to the RELEASE** (`grad_clip=None`), so `cmd_diagnose`'s existing `ChainArm("rl", …)` call and its banked D4 numbers are unaffected | wiring [T] |
| `rl_step`: `w = L.apply_lambda_scale(w, arm.il)` | **C1b** | CHANGED |
| `rl_step`: `a_star = L.matched_anchor_index(inp.bank, gt)` | **C1**'s match — computed on every arm, *used* by one, so all four arms can log both IL statistics | CHANGED + ADDED-TELEMETRY |
| `rl_step`: `il_i = L.imitation_term(path, gt, arm.il, …)` | **C1** (dispatch; the release forms return `all_modes_il`, bit-identical to the line it replaces) | CHANGED |
| `rl_step`: `il_all` / `il_match` under `no_grad` | both IL statistics on every arm, because `il_mean_m` is **not comparable across forms** (the matched anchor is the nearest by construction) | ADDED-TELEMETRY |
| `rl_step`: `spread = L.fan_endpoint_spread(path, n)` at the last step | the collapse is visible at the step that produces it, not only 600 steps later | ADDED-TELEMETRY |
| `rl_step`: `gstats = L.clip_gradients(params, arm.il)` replacing the inline norm | **C2**. ⛔ It also replaces the old `if not sq: raise RuntimeError(...)` guard with a louder `Ddv2ConfigError` carrying the same meaning | CHANGED |
| `rl_step` return | `il_form`, `il_lambda_scale`, `il_all_modes_m`, `il_matched_anchor_m`, `grad_norm_clipped`, `grad_clipped`, `grad_clip`, `chain_endpoint_spread_m`, `match_*` | ADDED-TELEMETRY |
| `cmd_train` | builds `IlSettings` from the flags; `run.json` gains `il` + `il_form_flag` and `grad_clip` is now the real value instead of the hard-coded `None`; two log lines | wiring [T] |
| `main` | `--il-form` (default `matched`) and `--grad-clip` (default `1.0`) | wiring [T] |

⚠️ **The CLI defaults are the LEVER, not the release** — the release arms are already banked and this script's remaining purpose is running L1. `--il-form release --grad-clip 0` reproduces them, and `run.json`'s `il.is_release` says which ran either way. That is a deliberate, recorded choice, pinned by `test_the_runner_wires_each_flag_to_the_settings_it_names`.

---

## 4. The mutation proof

⛔ *An assertion that only ever sees the fixed code proves nothing* (MEMORY: an AST census read 0 suspects on BOTH the fixed and the broken trainer). So the central test re-introduces the defect and requires it to be **reachable**:

`test_MUTATION_a_synthetic_fan_survives_the_matched_term_and_does_NOT_survive_the_release` builds a wide synthetic fan (9 anchors × 4 groups, initial endpoint spread **23.70 m**) whose GT is anchor 0's own path, optimises it under each IL term alone for 400 Adam steps, and requires all three of:

| | requirement | MEASURED on CPU, this worktree |
|---|---|---|
| (a) | the RELEASE's term must **collapse** it (`< 10 %` of the initial spread) — the defect is reachable | **23.70 → 0.060 m**, 0.25 % retained |
| (b) | the MATCHED term must **preserve** it (`> 95 %`) | **23.70 → 23.7037 m**, 100.0 % retained |
| (c) | the two must be separated by `> 8×` | 393× |

`test_MUTATION_only_the_matched_anchors_chains_moved_at_all` adds the mechanism, not just the statistic: every **unmatched** anchor's chains are **bit-unchanged** (max displacement exactly `0.0`), while the same run under the release's term moves every one of them.

`test_MUTATION_each_L1_integrity_check_fires_on_its_own_defect` does the same for `check_arm_l1.py`: each of I1/I3/I7/I8/I9/I10 is fed the specific defect it exists to catch — including *an arm that says `matched` and ran the release* — and is required to name it. A checker that cannot fail has never passed anything.

---

## 5. What is NOT claimed by this file

* **No result.** Nothing has been trained under L1. This file certifies what the code *is*, not what it *does* on the model. `RESULT.md` in this package states what will run and what will decide it.
* **No GPU run.** The RTX 4060 belongs to another agent's proof package; the L1 arms wait for the Master Mind's slot.
* The scope limits of the 2026-09-15 package **all carry forward** and are restated in `RESULT.md` §0.
