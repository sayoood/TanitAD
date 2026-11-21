#!/usr/bin/env bash
# Dry run of optb_readchain.sh on fake banks (copies of R35 dressed as optb_<tag> in a temp dir). Usage: dry_run.sh A|B
#  A: tags FAKEa_s0 FAKEc_s0; FAKEa_s0 = genuine R35_s0 copy (full stages, planner reduced to --n 8 windows, ONE inference seed), FAKEc_s0 = never banked, box-2 STOP.txt present -> chain must END (not wait).
#  B: tag FAKEb_s0 = R35_s0 copy with ONE flipped byte after the manifest -> chain must refuse at the evidence stage and write ZZ_STOP.txt.
export PATH="/usr/bin:/bin:$PATH"
RC="F:/TanitAD-scratch/actrsp/optb/readchain"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
W="$RC/tmp/dry$1"
rm -rf "${W:?}"; mkdir -p "$W/pulls" "$W/state"
R35="C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/ckpt_last.pt"; FX="$RC/fixture/FT"
export OPTB_READ_DRY=1 OPTB_PULLS="$W/pulls" OPTB_STATE="$W/state" OPTB_RC_OUT="$W/out" OPTB_PLANS="$W/plans" OPTB_READ_POLL_S=2 OPTB_READ_MAX_POLLS=5
export OPTB_EVIDENCE_ARGS="--lever null --init 0 --dry --expect-adam 145"
if [ "$1" = A ]; then
  "$PY" "$RC/mk_fakebank.py" "$W/pulls/optb_FAKEa_s0" "$R35" "$FX"
  echo "box-2 STOP (dry run)" > "$W/state/STOP.txt"
  export OPTB_READ_TAGS="FAKEa_s0 FAKEc_s0" OPTB_PLAN_EXTRA="--n 8" OPTB_PLAN_SEEDS="0"
else
  "$PY" "$RC/mk_fakebank.py" "$W/pulls/optb_FAKEb_s0" "$R35" "$FX" --tamper
  export OPTB_READ_TAGS="FAKEb_s0"
fi
bash "$RC/optb_readchain.sh"
echo "chain rc=$?" > "$W/chain.rc"
