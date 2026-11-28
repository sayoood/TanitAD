#!/bin/bash
# rhk_env.sh -- sourced by every RHK pod script (Amendment 8 DRAFT). Sources the UNCHANGED rh_env.sh (R, TREE, PY, DS, S1, COMMIT, export_py) and REPLACES rh_readout_flags for readout K:
#   causal_g0x = K  : bank flags `--readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2` (the eval's R0KHb20), trainer token `--dagger-bank-readout causal_g0x`.
# RH_READOUT must be exactly causal_g0x (no default). The choice is written to $R/RHK_READOUT.chosen, NOT to RH_READOUT.chosen: the RH chain's file is never touched.
. "$(dirname "${BASH_SOURCE[0]}")/rh_env.sh"
rh_readout_flags() {
  [ "${RH_READOUT:-}" = causal_g0x ] || { echo "ZZ ROLLHIST-K FAIL ZZ RH_READOUT must be causal_g0x (got '${RH_READOUT:-}'): this chain generates the readout-K bank"; exit 1; }
  BANK_RO_FLAGS="--readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2"; DECL_TOKEN=causal_g0x
  mkdir -p $R
  [ -e $R/RHK_READOUT.chosen ] && [ "$(cat $R/RHK_READOUT.chosen)" != "$RH_READOUT" ] && { echo "ZZ ROLLHIST-K FAIL ZZ RHK_READOUT.chosen says $(cat $R/RHK_READOUT.chosen)"; exit 1; }
  [ -e $R/RHK_READOUT.chosen ] || echo "$RH_READOUT" > $R/RHK_READOUT.chosen
  export BANK_RO_FLAGS DECL_TOKEN
}
