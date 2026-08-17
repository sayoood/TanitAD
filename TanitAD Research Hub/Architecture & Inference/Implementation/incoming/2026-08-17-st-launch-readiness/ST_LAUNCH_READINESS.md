# S-T LAUNCH READINESS — the transition is SOUND, the launch path is NOT

**Date:** 2026-08-17 · **Branch:** `agent/arch-inf-20260803` (HEAD `72e0c20`) · **Author:** arch-inf subagent (S-T readiness stream)
**Deadline this serves:** `v6F-SW-30k` reaches 30 000 in **5.47 days** (MEASURED: step 12 150, 26.4833 s/step, `/home/nvidia/experiments/v6F-SW-30k/train_log.jsonl`, 2026-08-17 18:35 UTC).
**Nothing was launched. Thor's GPU was not touched.** Every Thor call was a read or an import.

---

> ## ⛔ ESCALATION — INTEGRATION NEEDED, FIVE ITEMS, ALL BEFORE 30k
>
> **E1. `v6_chain.py commands --step S-T` emits an S-T launch that CANNOT LOAD THE S-W CHECKPOINT.** It carries **no model-geometry flag at all**, so the stack is built at the trainer's defaults (enc 384×8, pred 768×6, d_tac 512, d_str 256, no ViT-5, no modern predictor) against a 768×12 / 1024×12 / 768 / 512 checkpoint. MEASURED: `RuntimeError … size mismatch for encoder.pos: [1, 640, 768] vs [1, 640, 384]`. `ChainConfig.extra_common` is the only escape hatch and **no CLI flag reaches it**. A validated replacement command is in §2 and `raw/st_launch_line.txt`.
>
> **E2. The chain also emits `--v2-subframe 176x624`, which the live S-W run does not have.** In `train_v6_staged` the sub-frame moves the **DATA and not the MODEL** (§3.2). `--init-from` then *succeeds*, the corpus mounts, and the run dies on the first forward with `ValueError: encoder input is (176, 624) but the config declares (256, 640)`. MEASURED.
>
> **E3. Thor's checkout is a MIXED-VINTAGE TREE and is missing the entire 2026-08-16 ladder work.** MEASURED by real import in `/home/nvidia/venvs/tanitad-train`: `STAGE_MAY_INTRODUCE` **absent**, `RESUME_CONTRACT` **absent**, `LADDER_UNTRAINED_GROUPS` **absent**, `--selector` / `--tac-goal-cond` / `--dump-seam-plan` **not in the CLI**, `v6_chain` and `taniteval.seam*` **ModuleNotFoundError**. 10/10 launch-path files drift. §4.
>
> **E4. The S-T gate can never read PASS on the arm the chain plans to run.** `STAGE_GATE_SPEC["S-T"]["required"] = ("TACTICAL_family", "sel_gap")`; `sel_gap` is emitted **only inside `if w.w_select:`** (`train_v6_staged.py:1390`), and the default S-T step is `--selector none, w_select 0.0` because SEL-1 is REFUSED. Verdict is INCONCLUSIVE **by construction**, so every S-S launch needs `--allow-inconclusive-gate`. §5.
>
> **E5. `--dump-seam-plan` is wired in the trainer but its import FAILS under the launch's own `PYTHONPATH`.** MEASURED on Thor **and** the dev box: with `PYTHONPATH=<stack>`, `import taniteval` → `ModuleNotFoundError`. The trainer catches it broadly and prints *"training continues"*, so F-16's instrument produces **zero numbers for the third time** — silently, at every checkpoint boundary. §6.

---

## 1. What was proven to WORK — the transition itself, on the real artifact

⭐ **The `S-W → S-T` introduction transition is SOUND, and this is not a test passing at default geometry.** `stack/tests/test_v6_stage_init_introduction.py` builds `V6Config()` at its **defaults**; the live run is a different model. Both were executed here at the **live production geometry**, and one of them against the **real checkpoint**.

**Artifact.** `tanitad-thor-wifi:~/ckpt_snaps/v6F_sw_step010000.fp16.pt`, 673 312 891 B, md5 **`a4e2c0e1eb0ca455448472853ccc46d7`** — identical on Thor and after the pull (both sides computed here, not inherited).

| arm | `missing_keys` | `unexpected_keys` | `introduced_keys` | `trunk_md5_after_load` |
|---|---|---|---|---|
| **REAL fp16 snapshot → S-T** (`--tac-goal-cond`, `--selector none`) | `[]` | `[]` | `cond_tac_dyn.bias`, `cond_tac_dyn.weight` | `5dcc441c96c6b1b8553025912f512f9b` |
| **REAL fp16 snapshot → S-T** (`+ --selector goal`) | `[]` | `[]` | the 2 above **+** `cand_score.cand_bias`, `cand_score.goal_point.{weight,bias}`, `cand_score.log_tau` | `5dcc441c96c6b1b8553025912f512f9b` |
| synthetic S-W at live geometry → S-T (port on) | `[]` | `[]` | the 2 `cond_tac_dyn.*` | `94e89431f99374f7cbe568bb35446149` |
| synthetic S-W at live geometry → S-T (port off) | `[]` | `[]` | `[]` | `94e89431f99374f7cbe568bb35446149` |

`init_step` **10000**, `prev_stage` **"S-W"**, `init_source` **`fp16_weights_only_snapshot`**, `init_precision` **`fp16->fp32 (lossy)`**.
Evidence: `raw/st_transition_probe.json` · producer `code/st_transition_probe.py`.

⭐ **The strongest single number here is `missing_keys == unexpected_keys == []` against the REAL checkpoint.** It proves the real 30k artifact's key set is **exactly** the key set HEAD's `v6.py` builds — S-W-shaped, 573 keys — and that S-T's 575-key stack differs from it by **precisely the two zero-init keys the allowlist admits**. Nothing is silently random-initialised, and no S-W key is orphaned.

⚠️ **WHAT THE fp16 SNAPSHOT DOES NOT PROVE, stated plainly.** It is `{"model", "_meta", "_fp16_weights_only"}` — `ops/ckpt_fp16_snapshot.py` drops `opt` by design (2/3 of the bytes). So this proves the **key set, the shapes, and the fp16-unwrap path** — which is the entire surface `load_stage_init` adjudicates — and it proves **nothing about optimiser state**. That is not a gap for this transition: `--init-from` starts a **new run at step 0** and inherits no moments. It *would* be a gap for a `--resume`, and `RESUME_CONTRACT["has_optimiser"]` correctly refuses this artifact for that. The remaining unproven step is that the **fp32 `ckpt.pt` at 30k** carries the same key set as the fp16 snapshot at 10k — which the same code writes, from the same stack, 250 steps apart in the same process.

⭐ **A second thing was proven in passing, and it de-risks E3's fix: HEAD is RESUME-SAFE for the live run.** The live argv (read verbatim off PID 25477's `/proc` cmdline) **parses under HEAD's `build_parser`**, and HEAD's `v6.py` builds a stack whose key set matches the live checkpoint exactly (the `[]`/`[]` above). So file-shipping HEAD's trainer to Thor cannot break a relaunch on the state-dict side. ⚠️ Key-and-shape identity is **not** behavioural identity — it is a necessary condition, not a sufficient one — and the live run is additionally protected by having **no supervisor** (below).

---

## 2. The command that actually starts S-T

⛔ **Do NOT use `v6_chain.py commands --step S-T` as it stands** (E1, E2). The validated line is `raw/st_launch_line.txt`, composed by `code/st_launch_line.py` **from the live run's own `config.json["args"]`** and then executed through the real parser, the real `build_stack_from_args` and the real `load_stage_init` against the real checkpoint:

* parses to `stage=S-T · selector=none · tac_goal_cond=True · v2_subframe=None · dump_seam_plan=<out>/seam`
* builds **336 575 049 params / 575 keys**
* loads the real S-W snapshot with `missing=[] unexpected=[] introduced=[cond_tac_dyn.bias, cond_tac_dyn.weight]`

Evidence: `raw/st_launch_line.json`.

**What each stage-specific flag does**

| flag | what it does at S-T | why it is not optional |
|---|---|---|
| `--stage S-T` | selects `STAGE_GROUPS["S-T"] = ("layer_tac", "planner")`; `apply_stage_freeze` freezes everything else; `V6LossWeights.for_stage("S-T")` zeroes `o1_*`, `o2`, `o3`, `o5`, `o6`, `s1_latent`, `w_s2_goal` | the frozen S-W trunk **is** the Drive-JEPA claim; a stage that also trained the encoder would not be a ladder |
| `--init-from <SW>/ckpt.pt` | `load_stage_init(..., stage="S-T")` — the adjudicated load in §1 | without it the gate saying "S-W passed" certifies nothing; the ladder becomes four unrelated models |
| `--prev-gate <SW>/stage_gate.json` | `assert_stage_precondition("S-T", …)`: FAIL has no override, INCONCLUSIVE needs `--allow-inconclusive-gate` + `--gate-off-reason` | X5's ordering rule, and see §5 — the S-W gate **will** read INCONCLUSIVE at 30k unless the battery is folded in |
| `--tac-goal-cond` | builds F-1's `g_str→P_T` port `cond_tac_dyn.*` (zero-init, +33 024 params at `d_goal_embed=128`); S-T is where it is **introduced** and **trained** | the diagram, `HIERARCHY_VOCABULARY` §5 and `V6Stack`'s docstring all spec `P_T(z_tac, a_tac | g_str)`; an S-T without it never trains the strategic→tactical dynamics downlink in its own stage. ⚠️ **CARRY RULE:** every later stage must pass it too or the keys become UNEXPECTED |
| `--max-horizon 60` | the window carries `plan_steps` future steps | at the default the 6 s horizon is structurally untrainable |
| `--plan-wta-eps 0.05` | ε-relaxed WTA: bounds the N−1 losing candidates, which under pure WTA receive exactly zero gradient | the banked REF-C-XL fan measured oracle 0.1639 m against fan mean 13.9564 m (85×) — the regime where an argmin is a coin flip |
| `--w-t1 1.0` | turns on the tactical latent-prediction term (`t1_latent`), zeroed in S-W | S-T's own layer objective |
| `--selector none` (default) | builds **no** `cand_score.*` | SEL-1 is REFUSED (E-WC2: σ/ADE 9.9915 [7.4492, 13.5119] vs a pre-registered refusal line of 3.0). ⚠️ this is also what makes the S-T gate unpassable — §5 |
| `--dump-seam-plan <out>/seam` | banks the emitted 60-step plan at each `--save-every` boundary for F-16's probe; **zero extra GPU** | without it `X2_seam` reads `not-run` for the third stage running. ⚠️ **requires the PYTHONPATH fix in §6 or it silently banks nothing** |
| the ~34 geometry flags | reproduce the live S-W stack exactly | E1/E2 |

⚠️ **`--v2-lru`:** the line carries **64**, the live run's value (stable 44 h). `v6_chain`'s Thor default is **6**. This is a host-RAM knob, not geometry — but 6 is not the measured-safe setting on this box, and per `CLAUDE.md` the "~8.6 GB per dataloader worker" figure does **not** apply to this trainer (it spawns zero workers).

---

## 3. The two chain defects, measured

### 3.1 E1 — no geometry in the emitted line

`v6_chain.trainer_argv` (`v6_chain.py:1045-1077`) emits, on a **non-dry** step, exactly: `--stage --out --steps --batch --lr --n-candidates --init-from --prev-gate --max-horizon [--w-select] [--tac-goal-cond] <step.extra> --v2-cache --v2-val-cache --v2-lru --frame-h --frame-w --frame-hfov --projection --v2-subframe --require-parity`. `TINY_GEOMETRY` is added **only** under `cfg.dry and cfg.tiny`.

MEASURED — the first eight lines of the resulting failure:

```
RuntimeError: Error(s) in loading state_dict for V6Stack:
  size mismatch for encoder.pos: copying a param with shape torch.Size([1, 640, 768])
      from checkpoint, the shape in current model is torch.Size([1, 640, 384]).
  size mismatch for encoder.patch.weight: [768, 9, 16, 16] vs [384, 9, 16, 16]
  …
```

The chain-built stack is **87.93 M params**; the checkpoint's is **336.54 M**.

⛔ **And the chain's own geometry guard cannot catch it.** `assert_geometry_carry` (`v6_chain.py:724`) reads the predecessor's `config.json["args"]` — which **contains every geometry field** — and compares exactly **two**: `selector` and `tac_goal_cond`. It has the data in hand and does not look. That is the same shape as the three vacuous gates found this week: *a guard whose name promises a check its body cannot perform.*

⚠️ Both of the fields it does compare are **ABSENT** from the live `config.json["args"]` (Thor's trainer predates them), so it falls back to `"none"` / `False` — which is correct-by-accident here, and is also independent evidence for E3.

**The fix has two shapes and they are not equivalent.** Adding `--extra-common` to the chain CLI makes the geometry *possible to pass*; extending `assert_geometry_carry` to diff the predecessor's `args` against the emitted argv makes it *impossible to omit*. Only the second closes the class.

### 3.2 E2 — `--v2-subframe` moves the data, not the model

`resolve_v2_frames`'s docstring says *"The frame is applied to `cfg` too, so the ENCODER is sized for what it will be fed."* **That is true for `train_flagship_v4`, whose `cfg` IS the model config, and false for `train_v6_staged`**: the call is `resolve_eval_frames(a, cfg_eval)` where `cfg_eval = eval_flagship_v4._eval_cfg()` (a flagship-v4 config used for the plan and the eval seam), and `build_stack_from_args(a)` already built the encoder from `a.frame_h`/`a.frame_w` **120 lines earlier** (`train_v6_staged.py:2323` vs `:2337`).

MEASURED on the built production encoder (`raw/subframe_desync.json`):

```
encoder_pos_shape        [1, 640, 768]
live_256x640             ok,   out [1, 640, 768]
subframe_176x624         ValueError: encoder input is (176, 624) but the config
                         declares (256, 640). RoPE would tolerate another grid,
                         but the joint APE is sized for this one — the guard stays.
```

⚠️ **The dangerous part is the ORDER.** `--init-from` succeeds (checkpoint and stack are both 256×640), `assert_v2_geometry_matches` **passes** (it compares the providers against `model_frame`, which *is* 176×624), and the refusal arrives at the **first forward, after the corpus has mounted** — the "refusal after the compute is paid for" family. A guard exists and it is in the wrong place.

---

## 4. Thor's drift — MEASURED, not read from `git log`

⛔ `git fetch` was **not** run on Thor. Thor's `git rev-parse HEAD` reports `30d6d60` (2026-08-15) and is **not the truth about its working tree** — every file this campaign arrived by file-ship.

**md5, repo (HEAD `72e0c20`) vs `tanitad-thor-wifi:/home/nvidia/TanitAD/`:**

| file | repo | Thor | |
|---|---|---|---|
| `stack/scripts/train_v6_staged.py` | `a37ddba0…` | `97daef49…` | ⛔ **= commit `be3b89b`, 2026-08-14 — 15 commits behind** |
| `stack/tanitad/models/v6.py` | `1e91b074…` (LF: `f36b476d…`) | `24160085…` | ⛔ **= commit `30d6d60`, 2026-08-15** |
| `stack/scripts/v6_chain.py` | `ee638491…` | **ABSENT** | ⛔ `find /home/nvidia -maxdepth 5 -name v6_chain.py` → nothing |
| `stack/tanitad/models/tactical.py` | `c45375b8…` | `6e8c80c4…` | ⛔ |
| `stack/scripts/train_flagship_v4.py` | `74adc6f7…` | `f8c71b38…` | ⛔ |
| `stack/scripts/eval_flagship_v4.py` | `21c1e0b5…` | `bd021afa…` | ⛔ |
| `stack/tanitad/data/v2_dataset.py` | `da5577de…` | `30bd5fd8…` | ⛔ |
| `stack/tanitad/data/parity.py` | `7b429918…` | `eb8c2dad…` | ⛔ |
| `stack/tanitad/data/calib.py` | `041c600e…` | `2f042d63…` | ⛔ |
| `stack/tanitad/geometry.py` | `b7798dfe…` | `18969b47…` | ⛔ |

**10 of 10 drift, and the two most important are at DIFFERENT vintages** (`train_v6_staged.py` 08-14, `v6.py` 08-15) — Thor is not "at a commit", it is a patchwork. The line-ending confound was excluded: `v6.py`'s repo file is CRLF, and Thor's md5 matches neither the CRLF nor the LF hash.

**The real import** — `PYTHONPATH=/home/nvidia/TanitAD/stack`, `/home/nvidia/venvs/tanitad-train/bin/python`, `CUDA_VISIBLE_DEVICES=""` (no GPU touched):

| symbol / flag | on Thor |
|---|---|
| `torch` | **2.13.0+cu130** (imports; the live run is at 97 % GPU, so CUDA works here) |
| `train_v6_staged.STAGE_MAY_INTRODUCE` | ⛔ **absent** |
| `train_v6_staged.RESUME_CONTRACT` | ⛔ **absent** |
| `train_v6_staged.STAGE_GATE_SPEC` | present; `S-W → (P1, P3, P6)`, `S-T → (TACTICAL_family, sel_gap)` |
| `--selector` in the CLI | ⛔ **absent** |
| `--tac-goal-cond` in the CLI | ⛔ **absent** |
| `--dump-seam-plan` in the CLI | ⛔ **absent** |
| `v6.LADDER_UNTRAINED_GROUPS` | ⛔ **absent**; `STAGE_GROUPS["S-J"]` is the full 7-group list |
| `import v6_chain` | ⛔ `ModuleNotFoundError` |
| `import taniteval.seam` / `.seam_dump` | ⛔ `ModuleNotFoundError` |
| `import taniteval.ci` | ✅ `/home/nvidia/TanitAD/taniteval/taniteval/ci.py` |
| `tanitad.eval.v6_probe_trunk.is_v6_checkpoint` | ✅ present (P1/P3/P6 can read a v6 ckpt) |
| `taniteval/tools/eval_four_families.py` | ✅ present |
| `taniteval/tools/seam_probe.py` | ⛔ **absent** |

⚠️ **Rule 2 applied and it paid:** `taniteval` is **not** under `stack/` on Thor (a first probe said "absent"); it is at `/home/nvidia/TanitAD/taniteval`. The absence claims above each survived a second path probe.

**Consequence, stated precisely.** An S-T launched from Thor's tree *today* would **not crash on `--init-from`** — with no `--selector` and no `--tac-goal-cond`, the S-T stack is key-identical to S-W's and the old strict load succeeds. It would instead **silently be the pre-F-1, pre-selector, pre-seam-dump ladder**: no `g_str→P_T` downlink, no introduction allowlist, no resume-lineage guard, no seam dump. That is worse than a crash and it is exactly `CLAUDE.md`'s *"a launch from a drifted pod resurrects fixed bugs"* — here, it un-lands five days of fixes without an error message.

⭐ **File-shipping is SAFE right now, and the window is open.** `ps -eo pid,cmd | grep supervise` → **no supervisor is running**, and `/home/nvidia/experiments/v6F-SW-30k/summary.json` does not exist. So nothing will relaunch PID 25477 with new code, and the state-dict compatibility is proven (§1). ⚠️ The flip side: **the live run has no auto-restart for its remaining 5.5 days.**

**Ship list** (md5-verified per file, `xz+b64` PTY push or the HF relay — never `git fetch`, which hangs on Thor):
`stack/scripts/train_v6_staged.py` · `stack/scripts/v6_chain.py` · `stack/tanitad/models/v6.py` · `stack/tanitad/models/tactical.py` · `stack/scripts/train_flagship_v4.py` · `stack/scripts/eval_flagship_v4.py` · `stack/tanitad/data/{v2_dataset,parity,calib}.py` · `stack/tanitad/geometry.py` · `taniteval/taniteval/{seam,seam_dump}.py` · `taniteval/tools/seam_probe.py`
then re-run the import probe (`code/thor_import_probe.sh`) and require every row above to flip.

---

## 5. The gate battery — every criterion, its owner, and whether it can fire

### 5.1 The S-W gate that must PASS before S-T

`STAGE_GATE_SPEC["S-W"]`: **required** `P1, P3, P6` · **reported** `P2, P5, P8, O6_spectrum, X4_spectrum_layers`.

| criterion | owner | emitter exists? | runs at 30k? |
|---|---|---|---|
| `P1_retention` ≥ 0.85× R²(z) at k=10 per driving target | `scripts/probe_latent_state.py` | ✅ repo **and** Thor; reads v6 ckpts via `tanitad.eval.v6_probe_trunk` | ⛔ **NO — separate GPU run, folded in with `--gate-probes`** |
| `P3_sign` ≥ 0.95 per channel, both lat and lon | `scripts/stage_a_probes.py` | ✅ repo and Thor, v6-aware | ⛔ same |
| `P3_gain` median in [0.5, 2.0] without post-training | `scripts/stage_a_probes.py` | ✅ | ⛔ same |
| `P6_dims` action-subspace dims (80 % var) ≤ 32 | `scripts/stage_a_probes.py` | ✅ | ⛔ same |
| `O6_rank_retention` (`o6_rank_verdict`, 3 clauses) | `tanitad.models.v6` | ✅ in-loop | reported only; **INCONCLUSIVE unless `--spectrum-accum` ≥ the ceiling** |
| `X4_rank_retention` per layer {tac, str} | `tanitad.models.v6.LayerSpectrumMonitor` | ✅ in-loop | reported only; **`--spectrum-accum 33`** (32 leaves tac one row short) — the live run runs **neither**, it has only `--spectrum-every 200` |

⛔ **THE OPERATIONAL FACT: at 30k the S-W gate will be written INCONCLUSIVE.** `run_stage_gate` marks every probe it cannot run as `{"pass": None, "status": "not-run"}` and `stage_gate_dict` turns any absent/None **required** probe into `verdict: INCONCLUSIVE`. MEASURED on a real trainer-written gate (§5.2). This is *by design* (the battery is a separate frozen instrument) and it is **documented** in `V6_GO_PACKAGE.md` §2.9 — but it means **S-T cannot start on the gate the run itself writes**. Either the P1/P3/P6 battery runs against `ckpt.pt` and is folded in via `--gate-probes`, or S-T launches with `--allow-inconclusive-gate --gate-off-reason "…"`. **That battery run is a GPU job that must be scheduled inside the next 5.5 days**, and it needs Thor's stack shipped first (E3).

### 5.2 The S-T gate — ⛔ it can never read PASS on the planned arm

MEASURED by executing a real dry S-T (`raw/st_dry_stage_gate.json`):

```
verdict                INCONCLUSIVE
required               ['TACTICAL_family', 'sel_gap']
inconclusive_required  ['TACTICAL_family', 'sel_gap']
```

`TACTICAL_family` is legitimately "run the eval". **`sel_gap` is not.** It is emitted at `train_v6_staged.py:1415`, inside:

```python
if w.w_select:                     # <- 0.0 on the default S-T step
    …
    sel_idx = score.argmax(dim=-1)
    log |= {"sel_gap": float((err[ar, sel_idx] - err.min(dim=1).values).mean())}
```

and `w_select > 0` **requires** `cfg.selector != "none"` (the trainer raises otherwise). The chain's default S-T step is `selector="none", w_select=0.0` because SEL-1 fired REFUSED on 2026-08-16. With no scorer, `V6Stack` emits **no `sel_*` key at all** (`v6.py:3968` — the whole block is under `if self.cand_score is not None`), so `sel_gap_tac` has no `sel_idx` to consume and `taniteval.selgap` has nothing to score.

⇒ **The gate spec and the default plan contradict each other, and the contradiction resolves as "INCONCLUSIVE, always".** It is the mirror of the three vacuous gates found this week (K3 pinned at 0.5; the pre-S2 goal-provenance audit; `_grad_census`'s zero-parameter group): **a criterion whose verdict is decided by construction rather than by the model.** Here it cannot *pass* instead of cannot *fail* — same class, same cause, and the operational cost is that S-S will be launched under a blanket `--allow-inconclusive-gate`, which erodes the one mechanism that stops a bad stage propagating.

⚠️ **It propagates upward too:** `STAGE_GATE_SPEC["S-S"]["required"]` contains `sel_gap_revalidated`, with the identical dependency.

**Both outcomes are legitimate and the choice is the PI's, not a subagent's:**
* **(a)** move `sel_gap` from `required` to `reported` **when `selector == "none"`**, and record that the S-T certificate then rests on `TACTICAL_family` alone; or
* **(b)** keep it required and make the refusal *explicit and early* — `assert_may_launch` should refuse an S-T step whose `selector == "none"` while `sel_gap` is required, naming the contradiction, instead of letting it surface as an unexplained INCONCLUSIVE 3.15 days later.

⚠️ **No test pins this.** `grep sel_gap stack/tests/` returns seven files, all v4/REF-C; none asserts the S-T gate against the default plan.

### 5.3 The other S-T rows

| criterion | owner | state |
|---|---|---|
| `TACTICAL_family` (+ `LATERAL`, `LONGITUDINAL`, `STRATEGIC`) | `taniteval/tools/eval_four_families.py` | ✅ exists, repo **and** Thor |
| `four_families_horizons` (both 0–2 s and 0–6 s) | same | ✅ instrument exists |
| `P7_rho` ≥ 0.3, CI excluding 0, per stratum | `scripts/w7_roll_rerank.py` | ✅ exists in repo |
| `X2_seam` | `taniteval/tools/seam_probe.py` + `taniteval.seam` | ✅ in repo, ⛔ **absent on Thor**, and ⛔ **fed by nothing** — §6 |

---

## 6. What S-T unblocks — and the one line that stops it

**The §8 blocker in `SEAM_INSTRUMENT.md` is CLOSED in the trainer, and that doc is now stale.** `--dump-seam-plan` exists (`train_v6_staged.py:3554`) and is wired into the training loop at the checkpoint boundary (`:2741-2780`), delegating to `taniteval.seam_dump.{seam_dump_from_plan, save_seam_dump}` — both landed in `6a7c006`. It re-uses `L["out"]["plan"]`, already computed for the step's loss (**zero extra GPU**), passes `eids=b["episode_id"]` (verified present on every window item, `data/_contract.py:138` — so the episode-cluster bootstrap gets real clusters, not one-window-per-cluster), `tier="T1"`, and `gt=batch.get("plan_target")`.

⛔ **But it will bank nothing, for two independent reasons, and neither is loud.**

**(i) The chain never turns it on.** `--dump-seam-plan` appears in no `ChainStep.extra` and in no `trainer_argv` branch. Default is `None` = the module is never imported and nothing is banked.

**(ii) Even when turned on, the import fails under the launch's own PYTHONPATH.** MEASURED on **both** machines with `PYTHONPATH=<stack>` (exactly what `launch_line` sets):

```
taniteval             ModuleNotFoundError: No module named 'taniteval'
taniteval.ci          ModuleNotFoundError: No module named 'taniteval'
taniteval.seam_dump   ModuleNotFoundError: No module named 'taniteval'
```

`taniteval` is a **sibling of `stack/`** (`TanitAD/taniteval/taniteval/`), not a member of it. The trainer's `except Exception` prints `[v6 seam] dump FAILED at <step> (ModuleNotFoundError…) — training continues` and carries on — correct for a diagnostic, and it means the failure lives in a log line at every save boundary while the artifact directory stays empty.

⚠️ It is also a **late** import: the first attempt is at `step % save_every == 0`, i.e. ~1.8 h into the run at `--save-every 250`. That is the `t1_eval.py` analysis-time-import family in a milder costume; the durable fix is a preflight probe at startup when `--dump-seam-plan` is set.

**Exactly what must change, in order:**

1. **The launch line** — `PYTHONPATH=/home/nvidia/TanitAD/stack:/home/nvidia/TanitAD/taniteval` (and the same in `v6_chain.launch_line` / `manifest_text`, which both hardcode `PYTHONPATH={cfg.workdir}`). ⚠️ `ChainConfig` has no field for a second path; this is a code change, not a flag.
2. **The chain** — add `--dump-seam-plan <out>/seam` to the S-T step's `extra` (and S-S/S-J).
3. **Ship `taniteval/taniteval/seam_dump.py`, `taniteval/taniteval/seam.py` and `taniteval/tools/seam_probe.py` to Thor** — all three absent (§4).
4. **A startup preflight** in `train_v6_staged`: when `--dump-seam-plan` is set, import `taniteval.seam_dump` at parse time and refuse in 2 seconds rather than logging a failure 1.8 h later.
5. Then `python taniteval/tools/seam_probe.py --dump <out>/seam/seam_010000.pt --arm v6F-ST@10k --tier T1 --out st_x2_seam.json`.

⚠️ **The S-W run cannot help here and should not be asked to.** `--dump-seam-plan` on S-W banks nothing: the emission head is at its zero-init, every control is exactly (0, 0), and `seam_dump_from_plan` correctly refuses with `SeamDumpError` unless `--dump-seam-plan-degenerate`. A degenerate dump **cannot answer the seam question** and must not be produced to make the row non-empty.
⚠️ The **dry ladder cannot exercise this path either** — MEASURED: a full dry S-T with `--dump-seam-plan` created no seam directory, because `--dry-run` returns before the training loop's save boundary. So this wiring has **never been executed**, by anything, and step 4 above is what makes that visible before 3.15 GPU-days are spent.

---

## 7. Live-run health at hand-off

| | |
|---|---|
| PID **25477** | ✅ alive, `Ssl`, elapsed **1-19:35:38**, RSS 9 516 508 kB |
| step | **12 150 / 30 000** |
| `step_s` | **26.4833 s** (the trainer's own per-process figure, not the `--log-every`-accumulated one) |
| ETA to 30k | **5.47 days** |
| GPU | **97 %** |
| loss | 1.9908 |
| supervisor | ⚠️ **none running** — a crash ends the run with no relaunch |
| done-marker | none (`summary.json` absent) — correct for a live run |
| disk | 524G used / 366G free on `/` |

Nothing was started, stopped or loaded on Thor's GPU. All Thor work was `ssh -n` reads, md5s, and one CPU-only import with `CUDA_VISIBLE_DEVICES=""`.

---

## 7b. Suite

⛔ **No repo code was modified by this stream**, so the suite is a baseline confirmation, not a regression check. Run with the interpreter named (a bare `pytest` hits the wrong venv here and reports *"191 errors during collection"* while exiting 0):

| suite | result | baseline |
|---|---|---|
| `stack` | **3782 passed · 0 failed · 7 skipped · 2 xfailed** (444 s) | matches |
| `taniteval` | **1092 passed · 0 failed** (127 s) | matches |
| ladder subset (`test_v6_stage_init_introduction`, `test_v6_chain`, `test_v6_staged`, `test_v6_ladder_edges`, `test_v6_stage_revalidation`) | **187 passed** | — |

⚠️ **The ladder subset passing while E1, E2, E4 and E5 are all real is the point.** Every one of those defects lives in a seam no test reaches: the chain's emitted geometry is never diffed against a predecessor's `config.json`; `--v2-subframe` is never forwarded through a real encoder; the S-T gate is never adjudicated against the default (`selector="none"`) plan; and the seam dump's import is never exercised under the launch's own `PYTHONPATH`. **A green suite is not a launch rehearsal.**

---

## 8. Evidence classes

| claim | class |
|---|---|
| every md5, key set, param count, `missing/unexpected/introduced_keys`, trunk md5, encoder `ValueError`, gate verdict, import result, step/`step_s`/GPU | **MEASURED (ours)** — producers in `code/`, outputs in `raw/` |
| Thor blob ↔ commit mapping (`be3b89b`, `30d6d60`) | **MEASURED (ours)** — `git show <c>:<path> \| md5sum` over the last 20/25 commits touching each file |
| SEL-1 σ/ADE 9.9915 [7.4492, 13.5119] vs the 3.0 refusal line | **INHERITED** — quoted from `v6_chain.SEL1_ADMISSION` (source), not re-measured here |
| REF-C-XL fan oracle 0.1639 m vs mean 13.9564 m | **INHERITED** — quoted from `train_v6_staged.py`'s comment at the ε-WTA term |
| "the fp32 `ckpt.pt` at 30k will carry the same key set as the fp16 snapshot at 10k" | **UNVERIFIED** — same writer, same stack, but not executed against the 30k artifact (it does not exist yet) |
| "HEAD is behaviourally identical to Thor's vintage for the live run" | **UNVERIFIED** — only key-and-shape identity and argv-parseability were proven |

---

## 9. Deliverable manifest

| artifact | where it lives | staged |
|---|---|---|
| `ST_LAUNCH_READINESS.md` (this file) | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-17-st-launch-readiness/` | yes |
| `code/st_transition_probe.py` — executes the S-W→S-T transition at live geometry, synthetic **and** real artifact | same, `code/` | yes |
| `code/st_launch_line.py` — composes **and validates** the S-T launch line | same, `code/` | yes |
| `code/subframe_desync_probe.py` — the `--v2-subframe` forward test | same, `code/` | yes |
| `code/thor_import_probe.sh` — the read-only Thor md5 + real-import probe (re-run after any file-ship) | same, `code/` | yes |
| `raw/st_transition_probe.json` — the four synthetic cases + both real-artifact arms | same, `raw/` | yes |
| `raw/st_launch_line.json` / `raw/st_launch_line.txt` — the validated command | same, `raw/` | yes |
| `raw/subframe_desync.json` | same, `raw/` | yes |
| `raw/thor_drift.json` — md5 table + import results | same, `raw/` | yes |
| `raw/st_dry_stage_gate.json` — a real trainer-written S-T gate (INCONCLUSIVE) | same, `raw/` | yes |
| `raw/live_run_health.json` — PID 25477 at hand-off | same, `raw/` | yes |
| the pulled fp16 snapshot (673 MB, md5 `a4e2c0e1…`) | **scratchpad only** — `…/scratchpad/v6F_sw_step010000.fp16.pt`; source of truth stays `tanitad-thor-wifi:~/ckpt_snaps/` | no (deliberate: 673 MB, and it is not this stream's artifact) |
| ⛔ **no repo code was modified** — every defect above is reported, none patched | — | — |

**Nothing is stranded on a pod.** The only Thor-resident artifact this stream touched is the checkpoint snapshot it read, which another stream owns.
