#!/bin/bash
# C1r read, steps after the harness CSV is complete (12,148 lines incl. header+average). CPU only, <=2 procs. Run detached; markers in markers/.
set -u
export CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; NAV=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
TIP=F:/TanitAD_p5cread_tip; PK="$TIP/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; EV="$PK/eval"
DATA=E:/Projects/TanitAD/data/refe_navtest; R=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/c1r/read; W=E:/Projects/TanitAD-artifacts/p5c_read
export PYTHONPATH="$TIP/stack;$TIP/taniteval;$EV;$EV/p5c"
O=$R/out; mkdir -p $O/families; L=$O/logs; mkdir -p $L
S1=$DATA/seams/stage2_p5c_c1r; S0=$DATA/seams/stage2_p5c_c1; SC=$DATA/seams/stage2_p5c_c0
step(){ n=$1; shift; [ -f $O/$n.DONE ] && { echo skip $n; return 0; }; echo "RUN $n"; "$@" > $L/$n.log 2>&1 && touch $O/$n.DONE || { echo "FAIL $n"; touch $O/$n.FAILED; exit 1; }; }
G=$($PY E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain/tools/commit_guard.py --min 6) || { echo "commit guard $G < 6"; exit 3; }
step convert $PY "$EV/p5c/p5c_read_prep.py" convert --arm C1r=$S1.npz,$S1.proposals.npz C1=$S0.npz,$S0.proposals.npz --outdir $R/plans
step score $PY "$EV/p5c/p5c_read_prep.py" score --plans-dir $R/plans --out-dir $O/pass1 --procs 2 --python $NAV
step idC1r $PY "$EV/p5c/p5c_read_prep.py" identity --rows "$O/pass1/rows_s*.jsonl" --arm C1r --csv $DATA/score/refe_stage2_p5c_c1r/refe_stage2_p5c_c1r.csv --plans $R/plans/plans_C1r.npz --out $O/identity_C1r.json
step idC1 $PY "$EV/p5c/p5c_read_prep.py" identity --rows "$O/pass1/rows_s*.jsonl" --arm C1 --csv $DATA/score/refe_stage2_p5c_c1/refe_stage2_p5c_c1.csv --plans $R/plans/plans_C1.npz --out $O/identity_C1.json
step families $PY "$EV/p5c/p5c_families.py" --rows "$O/pass1/rows_s*.jsonl" --arms C1r C1 --pairs C1r:C1 --out-dir $O/families --python $NAV --repo $TIP
# tactical roles: C1 := C1r, C0 := C1 (comparator), base := C0
step tactical $PY "$EV/p5c/p5c_tactical.py" --seam C1=$S1.npz C0=$S0.npz base=$SC.npz --dump C1=$S1.proposals.npz --out $O/tactical.json
step strat $PY "$EV/instruments/strategic_family.py" read --arm C1r=$S1.npz --base C1=$S0.npz --out $O/strategic_C1r_vs_C1.json
step strat0 $PY "$EV/instruments/strategic_family.py" read --arm C1r=$S1.npz --base C0=$SC.npz --out $O/strategic_C1r_vs_C0.json
step read $PY $R/tool/c1r_read.py --p5c-dir "$EV/p5c" --rows-new "$O/pass1/rows_s*.jsonl" --rows-c0 "$W/out/pass1/rows_s*.jsonl" --new C1r --ref C1 --c0 C0 \
  --fam-ci $O/families/famci_C1r_vs_C1.json --tactical $O/tactical.json --strategic $O/strategic_C1r_vs_C1.json --strategic-c0 $O/strategic_C1r_vs_C0.json \
  --identity $O/identity_C1r.json $O/identity_C1.json --seam-reports C1r=$S1.report.json C1=$S0.report.json --seams C1r=$S1.npz C1=$S0.npz \
  --models C1r=$R/arms/C1r/model_final.pt C1=$R/arms/C1/model_final.pt --cfg-new $R/pod/config_c1r.json --cfg-ref $R/pod/config_c1.json \
  --code-ctrl $R/pod/ctrl_code.json --out $R/read_c1r.json
cat $R/VERDICT_c1r.txt; touch $R/markers/post.DONE
