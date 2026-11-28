#!/bin/bash
# rh_env.sh -- sourced by every ROLLHIST (RH) pod script. The readout of the training history is ONE parameter: RH_READOUT=causal_g1x|causal_g1 (PI decision still open when this was written).
#   causal_g1x = C*      : bank flags `--readout causal_g --acc-gain 1 --path-exec`, trainer token `--dagger-bank-readout causal_g1x`   (recommended)
#   causal_g1  = ADD63   : bank flags `--readout causal_g --acc-gain 1` (no --path-exec),  trainer token `--dagger-bank-readout causal_g1`
# Both: heading bound `--kappa-max 0.2` / `--dagger-bank-kappa 0.2`. No default: an unset RH_READOUT is refused. The choice is written once to $R/RH_READOUT.chosen; a different value later is refused.
R=${RH_ROOT:-/root/rollhist}; TREE=$R/tree; PY=${RH_PY:-/root/venv/bin/python}; DS=/root/dscale10k; S1=/root/strat1
COMMIT=2bda9e1a382986cfdce3bb455822678b42a816e2
rh_fail() { echo "ZZ ROLLHIST FAIL ZZ $1"; exit 1; }
rh_readout_flags() {   # sets BANK_RO_FLAGS (bank generator flags) and DECL_TOKEN (the trainer's --dagger-bank-readout value)
  case "${RH_READOUT:-}" in
    causal_g1x) BANK_RO_FLAGS="--readout causal_g --acc-gain 1 --path-exec --kappa-max 0.2"; DECL_TOKEN=causal_g1x;;
    causal_g1)  BANK_RO_FLAGS="--readout causal_g --acc-gain 1 --kappa-max 0.2"; DECL_TOKEN=causal_g1;;
    *) echo "ZZ ROLLHIST FAIL ZZ RH_READOUT must be causal_g1x or causal_g1 (got '${RH_READOUT:-}'): the PI's readout decision is not defaulted"; exit 1;;
  esac
  mkdir -p $R
  if [ -e $R/RH_READOUT.chosen ] && [ "$(cat $R/RH_READOUT.chosen)" != "$RH_READOUT" ]; then
    echo "ZZ ROLLHIST FAIL ZZ RH_READOUT=$RH_READOUT but $R/RH_READOUT.chosen says $(cat $R/RH_READOUT.chosen): a relaunch may not change the readout"; exit 1
  fi
  [ -e $R/RH_READOUT.chosen ] || echo "$RH_READOUT" > $R/RH_READOUT.chosen
  export BANK_RO_FLAGS DECL_TOKEN
}
export_py() { export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; }
