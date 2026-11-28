# HANDOFF — P7-WIRE-HEAD (≤ 2k tokens; written 2026-10-22 21:28 because the remaining job is ~35 min of CPU)

**State.** Code is COMPLETE and validated piecewise; only the long confirmation chain is still running (no model waits on it).
Package root `C:/Users/Admin/par18/p7wire_head/` (single location; nothing staged/committed/pushed anywhere — local override).

| evidence | result (all CPU, `CUDA_VISIBLE_DEVICES=-1`) |
|---|---|
| `test_p7_head_wiring.py` (static, what G-SUITE runs) | `ZZP7HEAD_PASS 179` (`--mutations` adds 8 real-gate arms: 187) |
| `... --full` (F1 default-off bit-identity vs frozen tip trainer, F2 real gate with the head, F3 instrument arms) | `ZZP7HEAD_PASS 207` / `ZZP7HEAD_FULL 207`, rc 0 (`raw/p7_head_wiring_full.out`) |
| `... --mutations` (`raw/mutation_check.out`) | control PASS; M1 (delete combine line), M2/M2b (default weight 0, flag omitted), M3 (head dropped from optimiser), M4, M5, M6: all RED with the named reason |
| `test_launch_gate.py --static --no-banked` | `ZZLG_STATIC_PASS 1168` before = after |
| `test_p6_levers.py --fast` | `ZZP6_PASS 165` before = after |
| refe_p6 profile dict digest | `80fa4d7e…` unchanged |

**Running now (started 21:19):** `run_chain.sh` → `raw/after/` (whole `refe/` suite, 51 scripts, parallel 3) → `patches/stage_code.sh` (writes `code/`: 4 changed + 2 new files at repo paths, `P7_WIRE_HEAD.diff`, `MANIFEST.txt`) → marker `raw/_CHAIN_DONE`. `run_closer.sh` then runs `python patches/finish.py`, which fills RESULT.md's three placeholders (evidence + before/after table, mutation table, manifest) from the raw files and writes `raw/_FINISHED`.

**Exact next step (fresh agent, Sonnet is enough):**
1. Poll with a script, not turns: `until [ -f raw/_FINISHED ]; do sleep 30; done`. Then `cat raw/finish.out raw/SUITE_BEFORE_AFTER.md | head -80`.
2. Read RESULT.md "Evidence": the line "Identical rc AND final token on N; differing: …" must say `differing: NONE` (new: `test_p7_head_wiring`). Baseline reds are environmental (shapely/nuplan missing, `raw/` artifacts absent from the minimal export, CRLF-rewritten non-`.py` fixtures, a dev-box lock file touched in parallel) — they must be the SAME set before and after. A DIFF: compare `raw/baseline/<t>.out` with `raw/after/<t>.out`; `test_hnscreen_chain` is known to flake on the shared lock files.
3. Check `code/MANIFEST.txt` says `CR-lines=0` for every file; if not, re-run `bash patches/stage_code.sh`.
4. `SubagentHandback` (≤ 300 words): headline = RESULT.md's "Headline" items 1-6, manifest table, "single location".

**Do not:** touch any git tree; edit `refe/p6_pod.sh`, `p6_prep.py`, `session_reserve.py`; edit the `refe_p6` profile; register the three UNKNOWN pins (`P7_AUX_STORE_PINS`, `P7_BANK_TUPLES`; the filter's report must supply them) — `refe_p7` refusing is the design.
**Edit RESULT.md only via `patches/RESULT.template.md`** then `python patches/finish.py` (the template is the master).
Edit tree: `work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/` (pristine tip export: `base_full/…`). Authoritative diff: `code/P7_WIRE_HEAD.diff`.
