#!/usr/bin/env bash
# PREREG_V8F_SPEEDSUB section 8 (dev-box CPU, <= 4 threads). DRAFT: nothing here has been run against real data.
# Launch DETACHED (a run longer than ~4 min is handed off by file): WMI `cmd /c bash run_speedsub.sh < NUL`. Markers in $W/run.out: ZZSPEEDSUB-RUN-DONE <outcome> | ZZSPEEDSUB-RUN-STOP <why>.
# <TREE> = a git archive of the commit that LANDS stage/ (stack/ + taniteval/ + the SPEEDREAD run tree's inputs); set INSTR_SHA to the landed instrument's sha256 (the PREREG pins it).
set -u
export CUDA_VISIBLE_DEVICES=-1 SPEEDREAD_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
W=${W:-F:/TanitAD-scratch/speedsub}; SR=F:/TanitAD-scratch/speedread; SF=E:/Projects/TanitAD-artifacts/sfresh_inputs
TREE=${TREE:?set TREE to the exported landing tree}; INSTR_SHA=${INSTR_SHA:?set INSTR_SHA to the PREREG-pinned instrument sha256}
T=$TREE/stack/scripts/v8f_mini_eval_speedread.py; WRAP=$SR/../speedsub_tmp/run_with_taniteval.py
RO=$SR/fit/readout.pt; FIT=$SR/fit/fit.json; RO_SHA=a205609ab47325a0ad9c0141be8ae3a0fec653d59c6db17ec1cf0d460bf2316d
X0=C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt;        H0=c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d
X1=C:/Users/Admin/box2_ship/logs/v8f_kindx_s1/runs/M2b_s1_mixo/ckpt_last.pt;    H1=68a0f3356c37b25ce15b5f860f8dc919e23efc8b6bf200a2997f9ad6b521e915
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
stop() { log "ZZSPEEDSUB-RUN-STOP $*"; exit 3; }
mkdir -p "$W"; log "run start (threads $SPEEDREAD_THREADS)"
cat > "$W/run_with_taniteval.py" <<'PYEOF'
import os, runpy, sys
tree, script = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(tree, "taniteval"))
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
PYEOF
WRAP=$W/run_with_taniteval.py
[ "$(sha256sum < "$T" | cut -c1-64)" = "$INSTR_SHA" ] || stop "instrument sha256 differs from the PREREG's pin"
[ "$(sha256sum < $RO | cut -c1-64)" = $RO_SHA ] || stop "readout.pt differs from the PREREG's pin"
[ "$(sha256sum < $X0 | cut -c1-64)" = $H0 ] && [ "$(sha256sum < $X1 | cut -c1-64)" = $H1 ] || stop "an X_s checkpoint differs from its sha256"
for f in eval_s0/eval_M2b-mixo_s0-speedread-own.npz eval_s0/eval_M2b-mixo_s0-speedread-gt.npz eval_s1/eval_M2b-mixo_s1-speedread-own.npz eval_s1/eval_M2b-mixo_s1-speedread-gt.npz \
         eval_s0/speedread_cells_M2b-mixo_s0-speedread-identity.npz eval_s1/speedread_cells_M2b-mixo_s1-speedread-identity.npz read/READ.json; do
  [ -f $SR/$f ] || stop "SPEEDREAD's banked $f is missing (controls S5/S6/S7b need it)"; done
free=$(df -m /f | tail -1 | awk '{print $4}'); [ "$free" -gt 5000 ] || stop "F: has $free MB free"
stage() { local n=$1; shift; [ -f $W/.done_$n ] && { log "$n: done earlier, skipped"; return 0; }; log "$n: start"; "$@" > $W/$n.log 2>&1 || stop "$n rc=$? (log $W/$n.log)"; touch $W/.done_$n; log "$n: done ($(tail -1 $W/$n.log | cut -c1-160))"; }
EV() { # EV <seed> <ckpt> <sha>
  "$PY" -I $WRAP $TREE $T subevals --seed $1 --x $2 --expect-sha $3 --readout $RO --fit-json $FIT --expect-readout-sha $RO_SHA --eval-dir $W/eval_s$1 --eval-cache $SF/cache \
    --tracks $SF/inputs/native_tracks.pt --v9-eval $SF/inputs/v9/v9_labels_eval139.npz --lead-eval $SF/inputs/v9/lead_eval139.npz --speed-eval $SF/inputs/v9/refcv6_speed_max_v8_eval.jsonl \
    --p10 $SF/inputs/p10/p10_prior.pt; }
PRE() { # preflight after seed 0: what O read in `ego` equals SPEEDREAD's banked identity pred_mlp_ego (S7b) and the substitution moved the cells; stop before seed 1 if not
  "$PY" -I - <<'PYEOF'
import numpy as np, sys
W, SR = "F:/TanitAD-scratch/speedsub", "F:/TanitAD-scratch/speedread"
a = np.load(f"{W}/eval_s0/speedread_cells_M2b-mixo_s0-speedsub-identity.npz"); b = np.load(f"{SR}/eval_s0/speedread_cells_M2b-mixo_s0-speedread-identity.npz")
e = np.load(f"{W}/eval_s0/speedread_cells_M2b-mixo_s0-speedsub-ego.npz")
d_bank = float(np.max(np.abs(a["pred_mlp_ego"] - b["pred_mlp_ego"])))
want = ((a["pred_mlp_ego"] - a["cons_mu"]) / a["cons_sd"]).astype(np.float32)
d_path = float(np.max(np.abs(e["used_std"] - want))); moved = float(np.mean(np.any(e["used_std"] != e["own_std"], axis=1)))
print(f"S7b dev {d_bank:.3g} (<=1e-5)  S7 dev {d_path:.3g} (<=1e-5)  moved {moved:.4f} (>=0.99)")
sys.exit(0 if d_bank <= 1e-5 and d_path <= 1e-5 and moved >= 0.99 else 1)
PYEOF
}
stage evals_s0 EV 0 $X0 $H0
stage preflight_s0 PRE
stage evals_s1 EV 1 $X1 $H1
stage subread "$PY" -I $WRAP $TREE $T subread --fit-json $FIT --eval-dir-s0 $W/eval_s0 --eval-dir-s1 $W/eval_s1 --speedread-dir-s0 $SR/eval_s0 --speedread-dir-s1 $SR/eval_s1 \
  --speedread-read-json $SR/read/READ.json --expect-readout-sha $RO_SHA --out $W/subread
OUTC=$(head -1 $W/subread/OUTCOME.txt 2>/dev/null)
# the four families (beside; they decide nothing). pairfam = TACTICAL + STRATEGIC (v8f_paired_families.py, unmodified); longlat = LONGITUDINAL + LATERAL (families_longlat_sub.py, a parametrised copy)
PFT=$TREE/stack/scripts/v8f_paired_families.py; FO=$W/families; mkdir -p $FO
fam() { local n=$1; shift; [ -f $FO/$n.json ] && return 0; "$PY" -B -I "$PFT" "$@" --out $FO/$n.json --md $FO/$n.md > $FO/$n.out 2>&1; log "$n rc=$?"; }
for s in 0 1; do
  E=$W/eval_s$s/eval_M2b-mixo_s$s-speedsub
  fam pairfam_ego_s$s --a $E-ego.npz --b $E-own.npz --label-a ego --label-b own
  fam pairfam_gt_s$s --a $E-gt.npz --b $E-own.npz --label-a gt --label-b own
  [ -f $FO/longlat_s$s.json ] || SUB_DIR=$W SPEEDREAD_TREE=$TREE "$PY" -B -I $TREE/stack/scripts/families_longlat_sub.py $s $FO/longlat_s$s.json > $FO/longlat_s$s.out 2>&1 || log "longlat_s$s rc=$? (a control failed: the JSON is still written)"
done
log "ZZSPEEDSUB-RUN-DONE $OUTC"
