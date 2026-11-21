#!/usr/bin/env bash
# LEVELHEAD arm LLR: the two-run chain LLR_0 then LLR_1 (3,000 steps each; CLR's argv + the lead head at the frozen W_lead and wiring). REFUSES (locally, before the lock and any remote call) while a placeholder stands in the argv, a2_prereg/FROZEN_LLR.txt or WLEAD_PROBE.json is absent or disagrees, the lead-head files are not recorded as landed, or the A2 card / ruling on disk are not the frozen ones.
# RUN ONLY AFTER THE PI'S BOX-2 GO: `LLR_PI_GO="<HH:MM>" bash .../llr_launch/launch_llr.sh`. Prints ZZLLR-LAUNCHED chain on success. All logic is in llr_lib.sh (sourced).
set -u
: "${LLR_PI_GO:?set LLR_PI_GO to the PI go time (HH:MM): this launcher refuses to run without the go of the PI}"
MODE=chain
. "$(dirname "${BASH_SOURCE[0]}")/llr_lib.sh" || exit 64
llr_launch
