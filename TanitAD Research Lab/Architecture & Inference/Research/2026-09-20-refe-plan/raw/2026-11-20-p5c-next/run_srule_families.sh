#!/bin/bash
# MM 2026-11-20: PREREG_P5C_NEXT §3 four families on the S-RULE picks (rule (2,5), NOT ADOPTED), EVAL logs, like-for-like 8-pose form.
cd /e/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next || exit 1
TIP="F:/TanitAD_p5cread_tip"
EV="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"
PYT=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PYN=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
F=srule_fam; SP=$F/seam_C1pick_eval.npz; SR=$F/seam_C1rule_eval.npz
step() { echo "ZZFAM_STEP $1 start $(date -u +%H:%MZ)"; shift; "$@" > "$F/log_$(date +%s).txt" 2>&1; local rc=$?; echo "ZZFAM_STEP rc=$rc $(date -u +%H:%MZ)"; return $rc; }
step convert "$PYT" "$EV/p5c/p5c_read_prep.py" convert --arm "C1pick=$SP" "C1rule=$SR" --outdir $F/plans || { echo ZZFAM_FAIL convert; exit 0; }
step score "$PYT" "$EV/p5c/p5c_read_prep.py" score --plans-dir $F/plans --out-dir $F/pass --procs 2 --python "$PYN" --no-human || { echo ZZFAM_FAIL score; exit 0; }
step families "$PYT" "$EV/p5c/p5c_families.py" --rows "$F/pass/rows_s*.jsonl" --arms C1rule C1pick --pairs C1rule:C1pick --out-dir $F/families --python "$PYN" --repo "$TIP" || { echo ZZFAM_FAIL families; exit 0; }
step tactical "$PYT" "$EV/p5c/p5c_tactical.py" --seam "C1rule=$SR" "C1pick=$SP" --subset-ok --out $F/tactical.json || echo ZZFAM_WARN tactical
step strategic "$PYT" "$EV/instruments/strategic_family.py" read --arm "C1rule=$SR" --base "C1pick=$SP" --out $F/strategic.json || echo ZZFAM_WARN strategic
echo "ZZFAM_DONE $(date -u +%FT%H:%MZ)"
