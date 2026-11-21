#!/bin/bash
# chain.sh -- the compute chain of trigger probe 3 (PREREG_FROZEN.md), resumable: every stage banks per arm and skips finished arms.  Scripts wake the next stage; no model turn polls.
export PATH=/usr/bin:/bin:$PATH
# usage:  bash chain.sh [stage ...]    (default: all)   log: raw/chain.log, per-stage raw/<name>.log ; completion marker raw/CHAIN_<stage>.done
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
cd /e/Projects/TanitAD-artifacts/dagland/trigger_probe3 || exit 1
run() { name=$1; shift; echo "$(date -Iseconds) START $name" >> raw/chain.log; "$PY" -I "$@" >> raw/$name.log 2>&1; rc=$?; echo "$(date -Iseconds) END $name rc=$rc" >> raw/chain.log; return $rc; }
stage_linear() {
  { [ -f raw/pcaG_pstop.json ] || run pca_pstop pca_g.py --pop pstop; } && run lin_pstop run_g.py --pop pstop --family linear
  { [ -f raw/pcaG_pmove.json ] || run pca_pmove pca_g.py --pop pmove; } && run lin_pmove run_g.py --pop pmove --family linear
  run strata strata_g.py; run tables tables_g.py; touch raw/CHAIN_linear.done
}
stage_eval() {
  run eval_extract eval_g.py --stage extract && run eval_linear eval_g.py --stage linear
  touch raw/CHAIN_eval.done
}
stage_mlp() {
  run mlp_pstop run_g.py --pop pstop --family mlp --only "G4+E,G8+E,G4,G8"
  run mlp_pmove run_g.py --pop pmove --family mlp --only "G4+E,G8+E,G4,G8"
  run tables tables_g.py; touch raw/CHAIN_mlp.done
}
stage_evalmlp() { run eval_mlp eval_g.py --stage mlp; run tables tables_g.py; touch raw/CHAIN_evalmlp.done; }
stage_mutate() {
  { [ -f raw/pcaG_mut_pstop.json ] || run pca_mut_pstop pca_g.py --pop pstop --mutate-upsample; } && run lin_mut_pstop run_g.py --pop pstop --family linear --mutate
  run tables tables_g.py; touch raw/CHAIN_mutate.done
}
stage_mlpctl() {
  run mlpctl_pstop run_g.py --pop pstop --family mlp --only "C1:shuffle:G8+E,C1:shuffle:G4+E"
  run mlpctl_pmove run_g.py --pop pmove --family mlp --only "C1:shuffle:G8+E,C1:shuffle:G4+E"
  run tables tables_g.py; touch raw/CHAIN_mlpctl.done
}
stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=(linear eval mlp evalmlp mutate mlpctl)
for s in "${stages[@]}"; do "stage_$s"; done
echo "$(date -Iseconds) CHAIN-END ${stages[*]}" >> raw/chain.log
