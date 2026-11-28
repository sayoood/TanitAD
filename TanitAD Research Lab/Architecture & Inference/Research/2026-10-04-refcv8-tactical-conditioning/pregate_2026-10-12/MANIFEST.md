# Deliverable manifest -- refcv8 A2 pre-gate (2026-10-12)

Staged, NOT committed, NOT pushed. Two copies exist: the staging tree below (dev box, local C: disk) and the live Thor dir.

| artifact | where it lives |
|---|---|
| RESULT.md (headline + per-check table) | `C:/Users/Admin/qland/hold_pregate/stage/TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/pregate_2026-10-12/RESULT.md` |
| HANDOFF.md, MANIFEST.md | same dir |
| commit message | `C:/Users/Admin/qland/hold_pregate/msg.txt` |
| provisional argv (PROVISIONAL, NOT a launch argv) | `<stage>/thor_dir/work/argv_PROVISIONAL.json` (+ `.marked.json`, `.stubgs.json`); Thor `/home/nvidia/pregate_d014ff2/work/` |
| assembler / audit logs | `<stage>/thor_dir/raw/assemble.log`, `audit.log` |
| in-process rules + data manifest evidence | `<stage>/thor_dir/raw/step2_rules_manifest.json`, `step2.log` |
| gate child evidence (JSON) | `<stage>/thor_dir/gate_out_{clock,model,suite,pinned,record}/evidence/G-*.json` |
| diagnostic what-if evidence d1-d7 | `<stage>/thor_dir/gate_out_diag_d1 .. d7/evidence/G-*.json`; digest traces `raw/digest_trace_d6.json`, `d7_omp1.json` (d5 is superseded: it counted recursive calls) |
| logs | `<stage>/thor_dir/raw/gate_*.log` (every one ends in `GATE_CHILD_RC=`) |
| hand ctxs | `<stage>/thor_dir/work/ctx_{model,clock,suite,pinned,record}.json`, `ctx_diag_*.json` |
| recipe / gs / overrides / stamps | `<stage>/thor_dir/work/recipe_provisional.json`, `gs_measure_REAL_smoke.json`, `gs_measure_stub.json`, `remap_overrides_A2_DIAG.json`, `stamps/config_*.json` |
| scripts | `<stage>/thor_dir/code/{mk_ctx_pregate.py, run_check.sh, run_diag.sh, gate_patched.py, mk_stamps.py, step2.py}` |
| source copies read from d014ff2 | `C:/Users/Admin/qland/hold_pregate/work/{launch_gate.py, assemble.py, mk_ctx.py, run_gate_clock.sh}` (scratch, not staged) |
| live Thor dir | `tanitad-thor:/home/nvidia/pregate_d014ff2/` (about 1 MB; nothing outside it was written) |

Not produced, by design: any gate token, any signing key, any file in the repo working tree, any commit.
