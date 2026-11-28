# T-EGOCUT pre-registration: drafting notes (2026-10-26)

Draft only. Nothing was staged, committed or pushed. No GPU, no box 2, no pytest was run, and no Python was executed except small
`json` reads of banked run records. Every repository file was read from the mirror with `git -c safe.directory=* show 2a78f059:<path>`.
The mirror was not modified.

## 1. What was read

**Pre-registrations and results** (tip `2a78f059`):
* `Project Steering/PREREG_V8F_DAGGER.md`: §1–§8 and Amendment 1. Source of the committed arm text (§7 NO EFFECT / HARMS rows), p = 0.1 (A1 §0), the
  RF definition and bar form (§5), the r′ matcher (§2), the control families (§6) and the outcome-table shape (§7).
* `Project Steering/PREREG_V8F_ORECOVER.md`, with Amendments 1–4. Source of the fine-tune mechanics (A1), K4 on `--limit logged` (A2), the
  completeness wording of NO EFFECT (A2), the pair rule (A3) and K4's fresh-age comparator (A4).
* `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/RESULT_V8F_ORECOVER.md`: the HARMS read, the four rows, the
  control-arm levels, §3's four families (T1-kin LATERAL better, T0 LONGITUDINAL worse) and §5 item 5 (the lon floor scale).
* `…/RESULT_V8F_T1_HEADS.md`: the T DEGRADES numbers, per-class turns and the zero-row-offset probe (`PROBE_LAG`).
* `…/RESULT_V8F_T1KIN_CAUSALBUCKET.md` (head only): the causal bucket's definition and that HACC's T1-kin margin holds on it.
* `Project Steering/GOALS_AND_CLAIMS.md`, rows `H-V8F-ORECOVER-1` (:17055) and `H-V8F-DAGGER-1` (:17056), for the row format.

**Code** (tip `2a78f059`; exported to the scratchpad, not edited):
* `stack/tanitad/models/v8f_mini.py` (1,012 lines)
* `stack/scripts/v8f_mini_train.py` (1,345 lines)
* `stack/tanitad/data/v8f_mini_data.py` (1,046 lines)
* `stack/scripts/v8f_t1_eval.py` (599 lines)
* `stack/scripts/v8f_mini_eval.py` (the loader only)
* `taniteval/taniteval/floors_kin.py` (`fit_p10`'s standardisation only)

**Banked run records** (read only):
* `E:/Projects/TanitAD-artifacts/v8f-orecover/M2b_s{0,1}_{orc,orc0}/summary.json` (`seconds`, `steps`, `argv_sha`)
* `…/M2b_s0_orc0/argv.json`, `selfcheck.json` and `gate/gate_M2b-orc0_s0.log` (init record, `source_step` 19164)
* `E:/Projects/TanitAD-artifacts/v8f-orecover-t1/done_*.txt` (`wall_s`), `PINS.txt` (the tool tree and digest) and `runs.json` (the dump paths)

## 2. Every code claim, with its file:line (tip `2a78f059`)

**`stack/tanitad/models/v8f_mini.py`**

| claim | line |
|---|---|
| `INFERENCE_INPUT_KEYS` (the admissible inference keys) | :137 |
| `V8FMiniConfig` is a dataclass with `__post_init__` validation, e.g. `o_interface` | :143, :204 |
| `transform_ckpt` | :221 |
| `P10Prior.standardise` = (f8 − xm) / xs | :264–:265 |
| `Reader`: dropout 0.0, `norm_first` | :544 |
| `Reader.forward`: the query offset `q + cond.mean(1)` | :555 |
| `hacc_w`: a None buffer, absent from the state dict (the pattern the new buffer copies) | :611 |
| `e_ego`, `e_nav`, `e_lim`: one `nn.Linear(8, d)` for ego, shared | :617 |
| `_ego(f8) = e_ego(prior.standardise(f8))` | :637–:638 |
| S's ego token at `last_s` | :675 |
| **T's ego token at `last_t` (the cut)** | **:686** |
| T's nav and limit tokens | :686–:687 |
| T's checkpoint token | :688 |
| T's scene | :689 |
| `trunk_t` → `h_goal` / `h_lat` / `h_lon` / `h_cons` (one shared trunk) | :690–:693 |
| O's ego token | :702 |
| The P10 prior reads `p10f[last_now]` and `base` | :710 |
| `_forward`: `last_t` / `last_s` from the ages; S then T then O | :726–:771 |
| S teacher-forcing draw `torch.rand(B)` | :741 |
| `detach_upper` branch | :743 |
| `ck_for_t = transform_ckpt(ck_src, pose_s, pose_t)` (the odometry back door) | :745 |
| T code teacher-forcing draw (`code_mix whole`) | :754 |
| `forward_infer` (`use_labels` False) | :773 |
| `_load_from_state_dict` (creates `hacc_w` when present) | :818 |
| `trainable_groups` | :907 |
| `built_levers` | :915 |
| `_KeyGuard` | :939 |

**`stack/scripts/v8f_mini_train.py`**

| claim | line |
|---|---|
| `FLAG_CHECKS` | :64 |
| `make_ego_perturb` | :288 |
| `bad_ego_perturb` | :294 |
| `bad_init_ckpt` | :315 |
| `INIT_CKPT_MATCH` (the architecture fields; training levers may differ) | :338 |
| `load_init_ckpt` (strict load, read back) | :342 |
| `declared_vs_built` | :402 |
| the gate refuses a flag without a `FLAG_CHECKS` entry | :407 |
| `live_check` | :502 |
| `code_mix_probe` | :557 |
| `o_interface_probe` (the template for `t_ego_probe`) | :621 |
| `acc_probe` | :695 |
| `ego_perturb_probe` | :765 |
| `main`: `torch.manual_seed(a.seed)`, `np.random.seed(a.seed)` | :1139–:1140 |
| model built | :1163 |
| `--init-ckpt` applied unless a resume (resume wins: init not applied, live weights random at gate time) | :1164–:1168 |
| optimiser parameter groups and AdamW | :1173–:1178 |
| `ChunkedSampler(wins, batch, seed=a.seed, chunk=f(lru))` | :1181 |
| G-LIVE probe batch | :1191 |
| admissibility check | :1213 |
| resume `load_state_dict` | :1246 |
| checkpoint carries `cfg.__dict__` | :1256 |
| `b = ego.apply(b, step)` | :1279 |
| **per-step reseed `torch.manual_seed(a.seed * 1_000_003 + step)`** | **:1289** |
| no `--dagger-*` flag exists at this tip (grep -c `dagger`: 0; same-breath control grep -c `ego_perturb`: 28) | (no line) |

**`stack/tanitad/data/v8f_mini_data.py`**

| claim | line |
|---|---|
| `V8FMiniWindows.item` | :494 |
| `limit` is per clip: `J["limit"][ci]` | :511 |
| `EGO_PERTURB_REBUILT` / `_KEPT` / `_RECORD` | :584–:590 |
| `EgoPerturb.apply` at p = 0 returns the batch, no draw | :669–:678 |
| `ChunkedSampler`; ages a_t ∈ 0..2, a_s = a_t + 0..4 | :854 |
| `epoch` on `np.random.default_rng((seed, ep))` | :863–:877 |

**`stack/scripts/v8f_t1_eval.py`**

| claim | line |
|---|---|
| `SCENE_MODES = ("logged", "frozen")` | :64 |
| `CAUSAL_STEPS = 20` (4 s) | :69 |
| `WindowState.__init__`: logged rows after t0 NaN; deadreckon nav | :156–:178 |
| own-arc nav progress | :216–:220 |
| `--scene frozen` holds t0's frames | :242 |
| `--limit causal` = `causal_bucket(own v, j, 20)` | :250, :261 |
| the `rollout` hook runs after the forward, before the commit (so CHANNEL-SWAP needs a new pre-forward seam) | :400, :417 |
| loader `V8FMiniConfig(**ck["cfg"])` and strict `load_state_dict` | :506, :511 |

**Other files**

| claim | file:line |
|---|---|
| T0 loader, the same two lines | `stack/scripts/v8f_mini_eval.py:237`, :263 |
| `xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8` | `taniteval/taniteval/floors_kin.py:274` |

**Banked numbers used for cost** (MEASURED, read here):
* ORC fine-tunes: `seconds` 2,257.8 (ORC s0), 2,249.1 (ORC-0 s0), 2,253.8 (ORC s1), 2,256.8 (ORC-0 s1), each `steps` 3000.
* T1-kin `wall_s`: ORC-0 s0 causal 284, logged 284; s1 causal 270, logged 255; ORC s0 507 / 507; ORC s1 275 / 262.
* Fresh-age T0 `wall_s`: 36, 34, 46, 34.
* ORC-0 `n_params_trainable` 7,214,495 (`selfcheck.json`). The M2b init `source_step` is 19164 (gate log).

## 3. Design choices, one line each

1. **The cut:** T's ego slot only (`:686`), as committed. S, O, the prior and `e_ego` are untouched. Reason: S ADDS ROUTE at T1 and O needs ego. The
   question is T's.
2. **The constant:** c = the init file's `e_ego.bias`, a frozen buffer. It equals `_ego(xm)` exactly: M2b's own token at the mean kinematic state.
   It is in distribution at init, adds no Parameter, and no RNG draw.
3. **The control:** the existing ORC-0, because the sampler is seeded by (seed, epoch), the forward is reseeded per step, the flag adds no draw and
   no Parameter, and the init is the same. Proven by K-BUILD, K-STREAM, K-DATA and K-TOOL. Fallback fixed now: EGOCUT-0 if K-DATA fails.
4. **RF:** |DAG-0 − ORC-0|, DAgger's definition. DAG-0 exists whenever this arm runs.
5. **Rows and bars:** ORC's and DAgger's, verbatim. New beside rows: R-FROZEN (the rig caveat), the own-state gap and tick-0 F1.
6. **Nexts:** ROBUST → adopt (plus CHANNEL-SWAP if a gap remains); PARTIAL → T-EGO-LATE; NO EFFECT → CHANNEL-SWAP → T-EGOCUT-2 (selection rule
   fixed now); HARMS → T-EGODROP (q = 0.5); REFUSED → fix and re-run.

## 4. Open questions for the Master Mind

1. **Run CHANNEL-SWAP on M2b now?** It is zero training, about 1 CPU hour, but needs a new pre-forward seam in `v8f_t1_eval` (the existing hook is
   post-forward, :417). It decides nothing for EGOCUT, but it could predict EGOCUT's outcome before box-2 time is spent. ⚠ If it is run first,
   EGOCUT's PREREG must land BEFORE its numbers are read, or the arm stops being pre-registered.
2. **R-FROZEN: beside or decisive?** I kept it beside, to reuse ORC's and DAgger's rows unchanged, and attached a committed consequence to
   ROBUST. If you want it as a decisive qualifier on ROBUST (e.g. "ROBUST requires R-FROZEN same-signed"), that must be decided before landing,
   not after.
3. **The decisive control: ORC-0 or DAG-0?** I chose ORC-0: it exists, has been read with 104 controls passing, and its argv is the base of both.
   DAG-0 is built on the newer trainer, closer to EGOCUT's commit. RF is symmetric either way. Swap them only before landing.
4. **T-EGO-LATE (the PARTIAL next) is an architecture change of T's heads.** Per Rule Zero it needs an explicit go. EGOCUT itself adds only a
   buffer and a branch, and is committed by DAgger's PREREG.
5. **Build order:** the trainer at `2a78f059` has no `--dagger-*` flag. EGOCUT's exclusivity clause with `--dagger-p` presupposes DAgger's trainer
   wiring has landed. K-BUILD then compares against `853f2fc3` (ORC-0's commit), not DAgger's.
6. **GPU determinism on box 2 is unmeasured.** No same-argv replicate exists yet; DAG-0 against ORC-0 will be the first. If RF comes out large, the
   bar rises, as designed, and the arm may be underpowered. That is a property of the rig, not a reason to re-choose the bar.
7. **The box-2 gate and T0-eval minutes are ESTIMATED.** The file timestamps under `v8f-orecover/` look like copy times, not run times. If the
   chain logs carry real timings, the ≤ 0.5 h figure can be replaced.
8. **Housekeeping:** I exported `v8f_mini.py`, `v8f_mini_train.py` and `v8f_mini_data.py` into the session scratchpad root, then moved them to
   `scratchpad/egocut_src/`. If files with those names already sat in the scratchpad root, they were overwritten by the tip versions.
