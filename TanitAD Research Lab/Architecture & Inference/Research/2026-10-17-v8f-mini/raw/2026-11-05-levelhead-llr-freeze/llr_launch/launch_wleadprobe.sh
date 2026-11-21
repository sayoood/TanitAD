#!/usr/bin/env bash
# LEVELHEAD arm LLR, STEP 1: the W_lead probe (LLR_0's argv with --selfcheck-only --w-lead 1 and the FROZEN wiring, its own --out ...\M2b_s0_wleadprobe_LLR). Trains nothing; writes selfcheck.json on box 2. REFUSES (locally) until a2_prereg/FROZEN_LLR.txt carries the wiring.
# RUN ONLY AFTER THE PI'S BOX-2 GO: `LLR_PI_GO="<HH:MM>" bash .../llr_launch/launch_wleadprobe.sh`. Prints ZZLLR-LAUNCHED wprobe on success. All logic is in llr_lib.sh (sourced).
set -u
: "${LLR_PI_GO:?set LLR_PI_GO to the PI go time (HH:MM): this launcher refuses to run without the go of the PI}"
MODE=wprobe
. "$(dirname "${BASH_SOURCE[0]}")/llr_lib.sh" || exit 64
llr_launch
