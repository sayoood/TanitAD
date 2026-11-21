# HANDOFF: P5-C read instrument (B5-B9). State: built, tested, dry-run done on 500 tokens. Next: the MM lands it, then schedules the full-token run.

## Integration (the MM)
Land `files/` under PK (`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`): `eval/p5c/` (8 files) and `raw/2026-11-13-p5c-read/dryrun/` (11). New files only; max repo path 135 chars. Tests from the landed tree:
`CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest eval/p5c -q` (96 passed, 2 skipped without the dry-run dir).
The scripts find `PK/refe` and `PK/eval/instruments` relative to themselves; `transp_score.py` must run under `C:/Users/Admin/navsim-crun/venv/Scripts/python.exe` (the `--python` flag).

## Full 12,146-token B6 (model-arm identity) = the full B5 pass. NOT run (the dev box CPU is shared). ~80 min ESTIMATED on 2 processes for 3 arms + human (measured 3.3 min per 500 tokens).
Rehearsal on the banked seams (P6, u400, deployed; no GPU, no P5-C claim). Scratch OFF F:, ~250 MB. Do not put anything under `E:/Archive`.
```
PK="<landed repo>/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; B="$PK/raw/p2_r2_instrument/baseline"
W=C:/Users/Admin/p5c_full        # any dir on C: or E:, not F:
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; NPY=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
$PY "$PK/eval/p5c/p5c_read_prep.py" convert --arm C1="$B/p6_navtest_full_seam.npz" C0="$B/u400_navtest_full_seam.npz" base="$B/deployed_navtest_full_seam.npz" --outdir $W/plans
$PY "$PK/eval/p5c/p5c_read_prep.py" score --plans-dir $W/plans --out-dir $W/score --procs 2 --python $NPY      # resumable: rerun the same line after a kill
for p in "C1 p6_navtest_full.csv" "C0 u400_navtest_full.csv" "base deployed_navtest_full.csv"; do set -- $p
  $PY "$PK/eval/p5c/p5c_read_prep.py" identity --rows "$W/score/rows_s*.jsonl" --arm $1 --csv "$B/$2" --plans $W/plans/plans_$1.npz --out $W/identity_$1.json; done
$PY "$PK/eval/p5c/p5c_families.py" --rows "$W/score/rows_s*.jsonl" --arms C1 C0 base --pairs C1:C0 C0:base C1:base --out-dir $W/fam
$PY "$PK/eval/p5c/p5c_read.py" --rows "$W/score/rows_s*.jsonl" --c1 C1 --c0 C0 --base base --human human --fam-ci $W/fam/famci_C1_vs_C0.json --controls $W/identity_C1.json $W/identity_C0.json $W/identity_base.json --rehearsal "REHEARSAL: C1=P6 C0=u400 base=deployed" --out $W/read.json
$PY "$PK/eval/p5c/p5c_render.py" --read $W/read.json --out $W/report.md
```
Expected: identity PASS (0 tokens over 1e-6) on all 3 arms at 12,146 / 12,146; any FAIL = the routing or a seam is wrong, do not read. `--rehearsal` stays on until real P5-C arms exist.
RED arm at scale is already measured at 60 tokens (26 / 60 over); a 12,146-token RED is not needed (the teacher T's DB route is banked at 3,346 / 12,146, `SCALE.md`).

## For the real P5-C read (card section 5, 11)
1. Dump each arm with `refe_navtest_seam.py --dump-proposals` so `pick20` exists; `convert --arm C1=<seam>,<dump> C0=<seam>,<dump> base=<P7 seam>,<P7 dump>` (all three should print form `pick20+plans8`; a mixed-forms warning means a contrast is not like-for-like).
2. Same score / identity / families lines, with `--controls` = the three identity JSONs + the B10 determinism result + the W-TRANSP (ii)-(iv) results (each a JSON with boolean `pass`), and no `--rehearsal`. Add `--c0r C0r` only if the PI approved the replicate (card 6.3).
3. B10 (determinism subset), B11 (SEALED.json, K ledger), B12 are not part of this build.

## Deletions (CLAUDE.md auto-delete rule; the only copy of everything useful is the hold)
Scratch `C:/Users/Admin/p5cr/` (tip export 25 MB, run500 11 MB, pytest basetemps, rows and plans; 61 MB measured before delete) is re-creatable from git `7f7ef7c7` and banked as above. Deleted by literal path at the end of the job; result logged in the final report.
deleted C:/Users/Admin/p5cr (61,943 KB), 2026-11-13T17:14:50Z
