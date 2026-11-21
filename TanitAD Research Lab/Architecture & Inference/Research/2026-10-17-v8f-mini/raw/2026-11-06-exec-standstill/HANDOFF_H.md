# HANDOFF_H (AGENT H, 2026-11-06 ~13:55). Frozen read H-EXEC-STANDSTILL-1. CPU only. Nothing committed, staged or deleted.

Package `E:/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill/` (`PK`). Scratch (rollouts, tool tree) `C:/Users/Admin/exec_standstill_scratch/` (`out/`, `gate/`, `gate_red/`, `tree/`, `subset/`).
Card `PREREG_EXEC_STANDSTILL.md` sha c782928e... UNTOUCHED (matches FROZEN_SHA.txt). My interpretation choices are in `DEFINITIONS_H.md` (written before any contrast was read).

## DONE (MEASURED)
* Tool copy `code/v8f_t1_eval_exec.py` sha e042a1bd... = LF6 tool 949c4b7a + `code/apply_exec_patch.py` (regenerates byte for byte; 2 flags: `--exec-eps`, `--dump-exec-ch`). Scratch tree copy has it at `tree/stack/scripts/`.
* ALL FOUR GATES PASS on the 123-window subset (`raw/GATES.json`, rerun: `python -I code/gates_exec.py --out raw/GATES.json --p10-red-dir C:/Users/Admin/exec_standstill_scratch/gate_red`), each with its RED arm:
  G-ID (unpatched vs patched bit-identical; RED eps 0.1 differs on 52/61 model, 44/61 floor launch windows), G-MOV (44 all-moving windows identical; RED eps 2.0 differs on 31/29), G-PHYS (0 violations on 3832+1368 sub-eps frames; RED hold-disabled 1579+1313), G-P10 (tick-0 identical; RED mutated floor breaks it on 123/123).
* G-ID on the FULL grid, seed 0 (probe): XC0_s0_base is BIT-IDENTICAL to the banked C0_s0 rollout (cl, fl, ol, ticks plan/P/tau/pose/v0). The seed-1 check and the PASS file come from the post-chain.
* Reader pure layer + banked known answer: `tests/test_exec_read.py` (10 PASS), `tests/test_banked_ka.py` (banked C0 M1 excess +0.0532 / +0.0594 reproduced; identical-input dE = 0, CI [0, 0]).
* Disclosure: the extractor's own-floor arrays equal the banked `a_*` rows exactly and differ from banked `b_*` (P10) by <= 0.07 (yaw-rate row max; mean 3e-5 m on the floor), because the banked reads used DAGOACC's floor.

## RUNNING (detached via WMI; no model turn needed)
1. Rollout chain `code/exec_chain.cmd` (8 rollouts, 16 threads, ~3 min each, started 13:45): `logs/chain.out`; marker `raw/CHAIN.done` or `raw/CHAIN.fail`. Order XC0_s0 base/hold, XC0_s1 base/hold, XDAG_s0 .., XDAG_s1 .. (tags `XC0_s0_base` ..). Outputs `scratch/out/<tag>/`.
2. Post-chain `code/post_chain.cmd` waits for CHAIN.done, then extract (8 array sets -> `raw/arrays/`), `gate-full` (-> `raw/GATE_ID_FULL.json`; if it fails NO contrast is read), `read` (-> `raw/EXEC_STANDSTILL_READ.json`, refuses unless every gate passed), `make_tables.py` (-> `raw/RESULT_TABLES.md`). Log `logs/post.out`; markers `raw/POST.done` / `raw/POST.fail`.
RESUME if a chain died: launch `code/exec_chain.cmd` (finished rollouts are skipped) and `code/post_chain.cmd` again (WMI `Win32_Process.Create` of `cmd /c <path>`).

## EXACT NEXT STEP
When `raw/POST.done` exists: read `raw/RESULT_TABLES.md` (all numbers are code-copied) and the `outcome_card_section5` block of `raw/EXEC_STANDSTILL_READ.json` (C0 is the verdict; DAG is beside). Write `RESULT_EXEC_STANDSTILL.md`:
outcome per card section 5 exactly as the card commits it (1 ARTIFACT-CONTRIBUTES / 2 PARTIAL / 3 NO-EFFECT; a separated-positive dE or E_base <= 0 is UNCOMMITTED, say so), dE with CI at BOTH seeds, share explained, |dE_s1 - dE_s0| as the training-seed spread, the 19 rows, lateral/heading per start class, the yaw-rate diagnostic, gates with RED arms, tier (T1-kin), evidence class MEASURED + artifact paths, estimator (paired episode-cluster bootstrap B 2000 seed 0; answers another draw of EPISODES only; the planner is deterministic).
If outcome 1 or 2: add the blast-radius list: EVERY banked T1-kin rollout used the base executor (`EPS_M = 1e-3` is in the tool lineage: smooth_t1/code/v8f_t1_eval.py, framefreeze/code/v8f_t1_eval_framefreeze.py, histswap_lf/code/v8f_t1_eval_histswap_lf.py, ri/, lf*/lfo/lf5/lf6, llr, cl, ...); name the launch rows (M1, lateral/heading on L/S/O classes) of those readers. Per Rule Zero, if outcome 3, the next lever is T2 stage 2 under the frozen card (the card says so).
Then write `MANIFEST.txt` (repo-path / scratch-path per artifact; nothing is in git; scratch rollouts exist only on C:), run the three tests again, and hand back in <= 300 words.
Do NOT edit the card; do NOT write in `motion_c/t2v8f/`; no staging by me (the MM stages `exec_standstill/`).
