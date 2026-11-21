# FALLBACK_HB: HB arms `--batch 4 --accum 2`, plus the HC G12 memory bound (VDAC A5)

Base: tip `924901b7` of `agent/arch-inf-20260803` (exported with `git archive`, autocrlf off). PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
Nothing was staged, committed or pushed. CPU only (`CUDA_VISIBLE_DEVICES=-1`), venv `C:/Users/Admin/venvs/tanitad`.

## Headline

**LANDABLE as a combined change**: 0 regressions, accumulation proven (section 1), HC G12 bound proven (section 2). One caveat: `test_launch_gate.py` (full) cannot run beyond `static()` on this box (needs `D:\Projects\TanitAD\data`) and stops at the same line in the tip and the patched export. Evidence class: MEASURED (this box, logs in `logs/`).

## 1. Part A: HB fallback (card line 31 / section 9.2)

MEASURED on the pod (MM, 2026-11-12): HB-s0 preflight at batch 8 raised `torch.OutOfMemoryError` (37.78 GiB allocated, 44.36 GiB in use of 44.39). No HB step trained.
Only HB-s0 and HB-SHUF change. Effective batch 8 is unchanged (4 x 2).

Every place the HB batch/accum is set (grep `HB-s0|HB-SHUF|--batch 8|accum`):

| file | place | change |
|---|---|---|
| `refe/launch_gate.py` | `VDAC_ARMS` HB-s0, HB-SHUF (:907-908) | add `"batch": 4, "accum": 2` (other arms carry no key, default 8 / 1) |
| `refe/launch_gate.py` | `vdac_argv` (:933) | `"--batch", str(sp.get("batch", 8)), "--accum", str(sp.get("accum", 1))` |
| `refe/launch_gate.py` | `VDAC_ARM_RECIPE` (:916) | per-class `batch` / `accum` read from the arm table, so `auxs1_refusals` (:1475-1478) requires 4 / 2 for the HB classes and 8 / 1 for the rest |
| `refe/vdac_pod.sh` | line 40 and the case table | `BATCH=8; ACCUM=1` defaults, `BATCH=4; ACCUM=2` on HB-s0 and HB-SHUF, `--batch $BATCH --accum $ACCUM` |
| `refe/vdac_chain.sh:324` | `want = min(upd, ceil(n / 8 * 2 / 1))` | NOT changed: `ceil(n / 4 * 2 / 2)` is the same number for every n, so the check still reads the schedule the trainer builds (proved in `test_vdac_hb_accum.py`) |
| `refe/vdac_pause_p7.sh`, `refe/vdac_stage0*` | none set batch for HB | nothing to change |

Exact diffs: `launch_gate.py.diff`, `vdac_pod.sh.diff`.

### Argv before and after (`MODE=argv ARM=<arm> bash refe/vdac_pod.sh`, both exports; files in `argv/`)

| arm | before | after | result |
|---|---|---|---|
| HA-s0 | `--batch 8 --accum 1` | same | byte-identical (cmp) |
| HA-s1 | `--batch 8 --accum 1` | same | byte-identical |
| HA-SHUF | `--batch 8 --accum 1` | same | byte-identical |
| HB-s0 | `--batch 8 --accum 1` | `--batch 4 --accum 2` | only those two tokens differ (diff of the token lists: lines 37 and 39) |
| HB-SHUF | `--batch 8 --accum 1` | `--batch 4 --accum 2` | only those two tokens differ |
| HC-s0 | `--batch 8 --accum 1` | same | byte-identical |
| HC-SHUF | `--batch 8 --accum 1` | same | byte-identical |

`test_vdac_gate.py` (unchanged, green in the patched export) additionally diffs `vdac_pod.sh` MODE=argv against `launch_gate.vdac_argv` token by token for the arms it covers.

### Accumulation evidence (`refe/test_vdac_hb_accum.py`, NEW, ZZTEST_PASS 40, 20 s)

Reachability, hop by hop:
1. argv: `vdac_argv("HB-s0")` -> trainer parser namespace `ns.batch, ns.accum == (4, 2)`; every other arm `(8, 1)`; `batch x accum == 8` for all seven arms; the profile recipe carries the same pair.
2. schedule: `scorer_finetune.py:2972` `total = ceil(len(tr_idx) / a.batch * a.epochs / a.accum)` is 15,000 for both 8 x 1 and 4 x 2 at 60,000 sets (also at 59,999), so `--max-updates 10000` still binds.
3. loop (`scorer_finetune.py`): `aux_backward(dl.tail, a.accum)` at :2995 divides the head loss by accum; `if micro % a.accum: continue` at :3011-3012 skips the optimiser until the second micro-batch; `HB.lora_step(hb, optL, ...)` (:3027) runs after the `continue`, once per update. The test pins that order in the real trainer text (each deleted line goes RED).
4. numbers (real tiny REFe, the real `HBTrunk` LoRA copy, the real head, 8 rows with varied labels): the copy's LoRA gradients and the head's gradients from 2 micro-batches of 4 at divisor 2 equal the 1 x 8 gradients (rtol 1e-4); ONE `lora_step` per two micro-batches; after the step the LoRA tensors agree with the 1 x 8 update.
   The aux loss is a per-frame mean over labelled frames (`aux_drivable_head.py:232-236`), which is why the equality is exact for equal micro-batches of labelled frames.

Mutations (all RED, in the test's printed table): divisor dropped (grad 2x); optimiser stepped on every micro-batch (2 steps, params differ); accum 1 at batch 4 (a 4-row update); trainer text with the divisor, the `continue` or `HB.lora_step` deleted.
The existing `test_vdac_hb.py` caller layer already runs the real `scorer_finetune.main()` with `--accum 2` (harness default) and asserts the copy's LoRA moved, `update == 25`; it was green in both trees.

Limits (UNVERIFIED): no real-trainer run compares batch 8 x 1 against batch 4 x 2 on the same data order (the dataloader order differs by batch, so the runs are not bit-comparable by design); the memory saving itself is only MEASURABLE on the pod (the card's preflight rerun).

## 2. Part B: HC G12 memory bound (MM scope addition)

MEASURED on the pod 2026-11-12 (MM report): HC-s0 preflight failed on one condition, G12 `head_retained_gb` **1.231** > the hard-coded 1.0 (cell 1.0, 5,120 queries); HA-s0 / HA-SHUF read **0.463** at cell 2.0 (1,280 queries). Every other G12 field and gate was OK.

New bound = `1.0 * (2.0 / cell) ** 2` GB: 2 m -> 1.0 (HA / HB unchanged), 1 m -> 4.0.

| file:line (tip) | before | after |
|---|---|---|
| `scorer_finetune.py:2549` | `g["head_retained_gb"] = None` | `g["head_retained_gb"], g["head_retained_bound_gb"] = None, 1.0 * (2.0 / float(aux.cfg.cell)) ** 2` (the BUILT head's cell, not argv) |
| `scorer_finetune.py:2570` | `(... g["head_retained_gb"] <= 1.0)` | `(... g["head_retained_gb"] <= g["head_retained_bound_gb"])` |
| `gate_checks.py:985-987` | `if mem is not None and not mem <= 1.0:` / `"... (> 1 GB)"` | bound from the parsed `ns.aux_drv_cell`; refuses if the trainer's recorded `head_retained_bound_gb` differs from it; message `the head retains {mem} GB for backward (> {bound:g} GB, the bound at the declared {cell:g} m cell)` |
| `gate_checks.py` (docstring of `judge_g12`) | `<= 1 GB where measured` | the scaled bound |
| `vdac_a5_layer.py` | the A5 reversal table undid the trainer back to blob `4df07e51...` | the changed `<= 1.0` text updated in the existing reversal, one new reversal for the `g["head_retained_gb"], g[...bound...]` line; `peel(trainer)` still hashes to `4df07e51278303abd9854f4f61f9bc4960b5a98d` (checked, and `test_vdac_hb.py` layer_check green) |

The refusal message names the value and the bound; at cell 2.0 it still contains `> 1 GB`, so `test_launch_gate.py:1602` (`head_retained_gb` 1.5, fragment `"> 1 GB"`) stays RED without being edited.
A trainer bound that disagrees with the declared cell (a hard-coded 1.0 under cell 1.0, or a loosened 4.0 under cell 2.0) is refused, so the gate reads the trainer's own number and cross-checks it. An older G12 dict without the key (all of `test_launch_gate.py`'s fixtures) is judged against the gate's own bound.
The 1.0 GB bound is an implementation sanity check, not a card bar; the HB batch fallback is not applied to HC.

Tests: `refe/test_vdac_g12_bound.py` (NEW, ZZTEST_PASS 15, 4 s):
1.231 @ cell 1.0 PASS; 4.5 @ 1.0 RED (names 4.5 and "> 4 GB"); 1.5 @ 2.0 RED (the 1602 case); 0.463 @ 2.0 PASS; trainer bound 1.0 @ cell 1.0 RED; trainer bound 4.0 @ cell 2.0 RED;
MUTATION `gate_checks.py` with the bound hard-coded to 1.0 -> (1.231, cell 1.0) RED, with the unmutated gate as the green control; trainer text mutations (`<= 1.0` again, bound from argv) RED.
Why not inside `test_launch_gate.py`: it cannot run beyond `static()` here (needs `D:\Projects\TanitAD\data`), so cases added there could not be shown GREEN or RED. The cases are in the new file, runnable here; adding them to `VDAC_SUITE` is an MM decision (it changes the suite list and its pins).

Pins and fingerprints moved: none edited. `test_vdac_hb.py` (blob pin of the peeled trainer) and `test_launch_gate.py --static` (1184 asserts) are green on the patched files without edits; no profile fingerprint pinned in a test moved (the profile digest tests `test_div_gate`, `test_dose_gate`, `test_rep_gate`, `test_s1zsm_gate`, `test_scorer1p6_gate` read the older profiles, which are untouched).
The new test files are NOT in any suite list; the run-time pinned tree manifest will change because `scorer_finetune.py`, `gate_checks.py`, `launch_gate.py`, `vdac_a5_layer.py` and `vdac_pod.sh` change (any closure land voids an OVF record bound before it, per the memory note on closure lands).

## 3. Files (hold `vdac_a5_hbfallback/`, repo-relative paths under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/`)

Patched full files: `launch_gate.py`, `vdac_pod.sh`, `scorer_finetune.py`, `gate_checks.py`, `vdac_a5_layer.py`, plus NEW `test_vdac_hb_accum.py`, `test_vdac_g12_bound.py`.
Diffs vs tip `924901b7`: `<name>.diff` next to `FALLBACK_HB.md`. LF endings, verified (0 CR). `argv/argv_{tip,patched}_<ARM>.txt` hold the 14 argv captures.

## 4. Test table (SAME / FIXED / REGRESSION)

Tip export vs patched export (the combined change), same box, same env (`CUDA_VISIBLE_DEVICES=-1`, `S1_READ_V1` = the tip blob of `s1_read.py`, at most 3 processes).

| test | tip | patched (A + B) | class |
|---|---|---|---|
| `test_vdac_hb.py` | PASS 133 | PASS 133 | SAME |
| `test_vdac_gate.py` | ZZTEST_PASS | ZZTEST_PASS | SAME |
| `test_dump_scores.py` | PASS 16 | PASS 16 | SAME |
| `test_aux_s1_wiring.py` | PASS 20 | PASS 20 | SAME |
| `test_vdac_stage0.py` | rc 0 (`OK`) | rc 0 (`OK`) | SAME |
| `test_vdac_stage0_forward.py` | rc 0 | rc 0 | SAME |
| `test_div_gate.py`, `test_dose_gate.py`, `test_rep_gate.py`, `test_s1zsm_gate.py`, `test_scorer1p6_gate.py`, `test_scorer1_gate.py`, `test_sft4r_gate.py` (sft4r 153), `test_s1z_gate.py`, `test_suite_nesting.py` (250) | rc 0 | rc 0, same counts | SAME |
| `test_s1c.py` | PASS 220 | PASS 220 | SAME |
| `test_hn_content_fp.py` | PASS 81 | PASS 81 | SAME |
| `test_hn_ctx_anchor.py` | PASS 72 | PASS 72 (first run died on `OpenBLAS: Memory allocation failed`, box RAM pressure from other jobs; rerun alone green) | SAME |
| `test_hnscreen_ctx_chain.py` | PASS 22 | PASS 22 | SAME |
| `test_gate_suite_prerun.py` | first two runs failed (WinError 32 on `valid.json.tmp`, then "the world's own pre-run must pass"; load flakes), third run PASS 144 | PASS 144 | SAME (flaky under load, green in both) |
| `test_launch_gate.py --static` | ZZLG_STATIC_PASS 1184 (two earlier runs failed at :1210 `ZZGATE-CLEANZZ`, load) | ZZLG_STATIC_PASS 1184 | SAME |
| `test_launch_gate.py` (full) | rc 1: static passes, then `full()` (`:2313` -> `diag_onpolicy_train.py:66` carve) `FileNotFoundError D:\Projects\TanitAD\data\refe_consistent\targets_rank1.jsonl` | the same traceback, same line, same file | SAME (stops identically) |
| `test_vdac_hb_accum.py` (NEW) | n/a (its argv layer is RED on the tip by construction) | PASS 40 | NEW |
| `test_vdac_g12_bound.py` (NEW) | n/a | PASS 15 | NEW |

REGRESSION: **0**. FIXED: **0** (nothing was red that the change turns green). Test files were not edited.
Tests that read profile fingerprints and the ones named by the brief: all above; `test_launch_gate.py` beyond `static()` could not run on this box in either tree. Logs: `logs/` (tip and patched `logs_*`).


## 5. Escalations for the MM

1. Decide whether `test_vdac_hb_accum.py` and `test_vdac_g12_bound.py` join `VDAC_SUITE` (that moves the suite list and its pins; not done here).
2. Land order: both parts touch `launch_gate.py` / trainer / `gate_checks.py`, so a single land; the VDAC card sha and the frozen card are not touched.
3. Real confirmation of both parts is the pod preflight: HB-s0 at 4 x 2 (memory) and HC-s0 G12 at bound 4.0 (1.231 measured).
