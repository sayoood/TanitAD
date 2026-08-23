# THE g_str → P_T CONDITIONING PORT — F-1 closed before S-T launches

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` (worktree on `18866b0`) · **Owner:** gstr-port subagent
**Mandate:** `DIAGRAM_CONFORMANCE.md` F-1 (⛔ P0 — BEFORE S-T LAUNCHES): *"the `g_str → P_T` conditioning port DOES NOT EXIST, and S-T must not launch without it."*
**Tier stamp:** this document describes MECHANISM + CPU-measured structural properties. No driving-capability number appears here; capability claims are T1 only.

---

## 1. The defect, restated from the audit (MEASURED there, verified here by read + test)

The binding diagram, `HIERARCHY_VOCABULARY` §5 (*"z_tac_{t+k} = P_T(z_tac_t, a_tac_t | g_str)"*), `V6_TRAINER_DESIGN` §1.2's ASCII, **and `V6Stack`'s own class docstring** all spec the strategic goal conditioning the tactical **dynamics**. The code did not build it: `predictor_tac = FTac(d_tac, d_goal=2*d_goal_embed)` has ONE conditioning input, fully consumed by the concatenated LAT×LON action embeddings, and `e_g_str` reached only `goal_head_tac(z_tac_p, cond=e_g_str)`. Consequence if S-T launched as-is: **the strategic→tactical dynamics downlink never trains in its own stage** — the `intent_proj` defect one level up (a conditioning path present in the diagram, absent from the optimisation), i.e. the fake-hierarchy failure class.

The defect is now **executable, not just described**: `tests/test_v6_gstr_port.py::test_NEGATIVE_CONTROL_the_default_build_cannot_see_g_str` perturbs the strategic goal path on the default build, shows `e_g_str` moves (and `g_tac`'s logits with it — the perturbation is live), and shows `zhat_tac` **bit-identical** — no path from `g_str` into `P_T` exists in the incumbent architecture. MEASURED (ours; this file's suite run).

## 2. What was built

| piece | where | what |
|---|---|---|
| `V6Config.tac_goal_cond` | `stack/tanitad/models/v6.py` | ⛔ **default OFF** — the flag that builds the port. Docstring carries the F-1 provenance and the shape rationale. |
| `V6Stack.cond_tac_dyn` | `stack/tanitad/models/v6.py` (built at the very END of `__init__`, only under the flag) | **zero-init `nn.Linear(d_goal_embed → 2·d_goal_embed)`** (+33,024 params at the production `d_goal_embed=128`, MEASURED; keys `cond_tac_dyn.weight` / `cond_tac_dyn.bias`). |
| the wiring | `V6Stack.forward` | `g_cond_tac = e_a_tac + cond_tac_dyn(self._cut(e_g_str, cut))`; `zh_tac = predictor_tac(z_tac, g_cond_tac)`. Flag off ⇒ `g_cond_tac` **is** `e_a_tac` (same tensor object) — the pre-F-1 forward, bit-for-bit. |
| grouping | `V6Stack._GROUP_PREFIXES` | `("cond_tac_dyn.", "layer_tac")` — trains in S-T/S-J, frozen S-W/S-S (MEASURED via `apply_stage_freeze`). |
| introduction | `train_v6_staged.STAGE_MAY_INTRODUCE["S-T"]` | now `("cand_score.", "cond_tac_dyn.")` — S-T may introduce the port over an S-W checkpoint; partial presence stays fatal. |
| trainer CLI | `train_v6_staged.py` | `--tac-goal-cond` → `build_stack_from_args`; two preflight refusals (§5). |
| chain | `v6_chain.py` | `ChainConfig.tac_goal_cond = True` (**the ladder default — F-1 says S-T must not launch without it**); emitted on every stage after S-W; `assert_geometry_carry` extended; `--no-tac-goal-cond` = the declared way to reproduce a pre-F-1 lineage. |
| runbook | `V6_GO_PACKAGE.md` §2.2 | the three post-S-W launch lines re-rendered from the chain (the runbook is a rendering, `test_runbook_commands.py` enforces argv-identity) + a table row for the flag. |
| tests | `stack/tests/test_v6_gstr_port.py` (48 tests) + updates to `test_v6_stage_init_introduction.py`, `test_v6_chain.py` fixture | §§3–6 below. |

**Why this shape and not a widened `FTac.in_proj` (the constraint the brief and F-1 both set):** `STAGE_MAY_INTRODUCE` adjudicates **keys**; `load_state_dict(strict=False)` still **raises on shapes** (measured — the `--n-candidates` ladder defect went exactly that way), so a widening would bypass the designed introduction path entirely and break every existing checkpoint. A new zero-init module rides the designed path. The additive-into-the-conditioning-input idiom is the **accepted pattern one level down**: `OperativePredictor` does `cond = act_emb(actions) + intent_proj(intent)` for the g_tac→P_O seam.

**Why `layer_tac` and not `planner` (the asymmetry with `intent_proj`, stated so it cannot read as drift):** `intent_proj` lives inside `predictor_op`, whose group trains in S-W while `intent=None` — dead weight at random init — so it was regrouped to `planner` to train exactly when g_tac first flows. `cond_tac_dyn` conditions `predictor_tac`, which is **already** `layer_tac` and **already** trains in S-T, the stage whose t1 loss flows through the conditioned prediction — the train-when-live property holds with no regrouping. It must NOT be `planner`: its output feeds `zh_tac` (WM-side, declared `uplink_side`), not the plan, and a planner-group parameter unreachable from the declared planner surface fails `test_planner_surface_is_total` by construction.

## 3. Byte-identity of the default build — the live resume is untouchable (MEASURED)

All per-tensor comparisons use `torch.equal` against a **CONTENT-anchored** pre-change reference — `v6.py`'s own git history walked for the newest revision **without** the marker `tac_goal_cond` (the C75 discipline; never `HEAD`) — plus the **RNG-stream** check (a default path consuming one extra draw would desynchronise everything initialised after the model).

| build | params | state_dict keys | proof |
|---|---|---|---|
| default, small fixture | 611,293 | 223 | per-tensor `torch.equal` + RNG stream + **forward** bit-identity (`zhat_tac`, `e_g_str`, output-key set) vs the pre-change module |
| **default, FULL `V6Config()`** (the live S-W resume's class) | **87,893,449** | **405** | counts pinned + per-tensor `torch.equal` vs the pre-change module (`@slow`) |
| **config E** (the REAL live geometry, rebuilt from the banked `/proc` argv in `…/2026-08-15-v6-thor-resume/code/RESTART_v6F_SW.sh`: enc 768×12×12 ViT-5+registers, pred-modern 1024×12×16, d_tac 768 / d_str 512, f_hidden 1024×6, budget 350 M) | **336,542,025** | **573** | counts pinned + `assert_param_budget` at 350 M. The literal matching the registry/handover 336,542,025 **is** the cross-check that this reconstruction is config E. |
| port ON (any geometry) | +33,024 | +2 (`cond_tac_dyn.weight/bias`) | ON ⊇ OFF with **every shared tensor `torch.equal`** (built at the END of `__init__`; the flag perturbs no pre-existing init) |

## 4. The port's contract — zero-init, alive, detached (all MEASURED by test)

1. **Exact no-op at init.** Port ON at zero-init: perturbing the strategic goal path moves `e_g_str` and leaves `zhat_tac` **bit-identical**; and the ON-at-init stack predicts `torch.equal` to the OFF stack on the same batch ⇒ **t1 is continuous at introduction** (`test_t1_is_continuous_at_introduction`).
2. **Alive when awake.** With `cond_tac_dyn.weight` off zero, the same g_str perturbation **moves `zhat_tac`** — through the real `V6Stack.forward`, not a unit harness — while `zhat_str` stays bit-identical (the movement is through the port, not some other seam).
3. **The negative control.** The identical detector on the **default** build reports no movement (§1) — the aliveness test can fail, and its failure case is F-1's defect itself.
4. **The detach discipline** (the downward-port rule, same as `intent=self._cut(e_g_tac, cut)` one level down): a t1-style loss on `zhat_tac` reaches **exactly** `{cond_tac_dyn.weight, cond_tac_dyn.bias}` of the port (gradient flows to the port's own parameters even at zero init), **zero** `layer_str` parameters (`goal_head_str`/`vocab_str` — the goal's source is protected by `self._cut(e_g_str, cut)`), **zero** encoder/readout/predictor_op parameters; live groups = `{layer_tac}` only.
5. **X3 unchanged.** `assert_isolation` = `{planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0}` with the port ON, alone and combined with `goal_factored`/`goal_cat_args`/`selector∈{goal,mlp}` — `strategic_to_below` being exactly the edge the port adds a **legal, declared** version of: the goal flows down by design, the probe certifies gradient does not flow back. The planner surface stays total.

## 5. Launch surface (trainer + chain), and the refusals that guard it

* **Trainer:** `--tac-goal-cond`. Preflight refuses:
  * `--stage S-W --tac-goal-cond` — layer_tac frozen in S-W **and** the new keys would break the live run's strict resume (the `--selector`/`--anchor-goal` S-W family);
  * `--tac-goal-cond --w-t1 0` in a stage that **trains** layer_tac (S-T/S-J) without `--i-know-this-is-the-control-arm` — t1 is the port's only gradient source, so this would build a port that never trains (the inert-scorer family). S-S with the flag is deliberately **not** refused: there it is geometry carry, like `--selector`.
* **Chain:** `ChainConfig.tac_goal_cond=True` ⇒ `--tac-goal-cond` on S-T/S-S/S-J and **never** on S-W; `assert_geometry_carry` now also checks the port from the predecessor's `config.json` **before anything is built**: S-T introducing it over a portless ancestor is the design (`introduces_port: "cond_tac_dyn."`); dropping it above S-T is refused (unexpected-keys death, pre-empted); introducing it above S-T is refused (S-S/S-J may introduce nothing — a mis-ordered ladder, not an introduction). `--no-tac-goal-cond` exists only to reproduce a pre-F-1 lineage as a **declared** decision.
* **Runbook `V6_GO_PACKAGE.md` §2.2** re-rendered; `test_runbook_commands.py::test_runbook_launch_lines_are_exactly_what_v6_chain_emits` holds again.

## 6. The whole seam, EXECUTED (not read) — the dry ladder

`v6_chain.run_chain(dry)` on CPU, this box, 2026-08-16 (MEASURED; also exercised by `test_the_WHOLE_LADDER_executes_end_to_end_on_cpu`):

```
S-W rc=0  tac_goal_cond=False  (untouched — the live run's shape)
S-T rc=0  tac_goal_cond=True   introduced=['cond_tac_dyn.bias','cond_tac_dyn.weight']
                               allowance=['cand_score.','cond_tac_dyn.']
S-S rc=0  tac_goal_cond=True   introduced=[]   (geometry carried, strict-clean)
S-J rc=0  tac_goal_cond=True   introduced=[]
```

X3 `{0,0,0}` at every stage. The S-W→S-T edge — the one that killed both selector arms the last time an introduction was designed on paper — was **run**, and the introduction is named in the stage's own `config.json`.

## 7. Test accounting

* Baseline at task start (pristine worktree): **3346 passed, 17 skipped, 2 xfailed** (401.9 s).
* After all edits (full `pytest -q -p no:cacheprovider` from `stack/`): see the final-verification line in the deliverable manifest of the agent report; target is baseline **+48 new** (`test_v6_gstr_port.py`) with zero regressions. Files updated to the new surface: `test_v6_stage_init_introduction.py` (the S-T allowance now names both modules), `test_v6_chain.py` (one fixture made faithful to post-F-1 configs: the selector-drift scenario now holds the port geometry fixed so exactly one thing drifts).

## 8. What this deliberately does NOT do

* **No new loss term.** The port trains through the existing t1 — in S-T the g_str-conditioned prediction *is* the loss surface, so no analogue of `seam_op` is needed (that loss existed because in S-T nothing flowed through `zh_op`; everything flows through `zh_tac`).
* **No default flip anywhere in the model/trainer.** Only the **chain** (the production launch surface) turns the port on, which is where a launch decision belongs.
* **No S-S revalidation change.** S-S already invalidates S-T's certificates (`S_S_REVALIDATIONS`); the port makes the *mechanism* stronger (S-S moves `e_g_str`, which now also conditions P_T directly) but the required re-measurements are the same two, at the same gate.
* **No claim that the port helps driving.** That is S-T's T1 eval to measure, four families, both horizons. This work makes the claim *testable* by making the downlink exist.

## Deliverable manifest

| artifact | where | state |
|---|---|---|
| the port (config flag, module, wiring, grouping, docstring) | `stack/tanitad/models/v6.py` | repo, staged |
| introduction allowance + CLI + preflight refusals | `stack/scripts/train_v6_staged.py` | repo, staged (on top of the audit's staged `STAGE_GATE_SPEC` edits — not reverted) |
| chain default + geometry carry + `--no-tac-goal-cond` | `stack/scripts/v6_chain.py` | repo, staged |
| 48-test suite (byte-identity C75, zero-init/aliveness/negative control, detach, X3, ladder seam, preflight, chain) | `stack/tests/test_v6_gstr_port.py` | repo, staged |
| updated pins | `stack/tests/test_v6_stage_init_introduction.py`, `stack/tests/test_v6_chain.py` | repo, staged |
| runbook §2.2 re-render + flag row | `…/2026-08-07-hierarchical-wm-redesign/V6_GO_PACKAGE.md` | repo, staged |
| this writeup | `…/2026-08-16-gstr-port/GSTR_PORT.md` | repo, staged |

*Nothing lives on a pod or a worktree; Thor untouched; everything CPU. Escalation for the orchestrator: the S-T production launch line now carries `--tac-goal-cond` — anyone regenerating launch commands from an old copy of the runbook would silently launch a portless S-T; regenerate from `v6_chain.py commands`, which is what the runbook test enforces.*
