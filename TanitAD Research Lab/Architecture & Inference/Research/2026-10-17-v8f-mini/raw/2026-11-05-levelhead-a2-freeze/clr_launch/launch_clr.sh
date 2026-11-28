#!/usr/bin/env bash
# LEVELHEAD arm CLR: the two-run chain CLR_0 then CLR_1 (3,000 steps each; RAPR-P01's argv + the level head at CL's frozen W). REFUSES (locally, before the lock and any remote call) when cl_prereg/FROZEN_W.txt or W_PROBE.json is absent or disagrees with the argv, or the A2 card on disk is not the frozen one.
# RUN ONLY AFTER THE PI'S BOX-2 GO: `CLR_PI_GO="<HH:MM>" bash .../clr_launch/launch_clr.sh`. Prints ZZCLR-LAUNCHED chain on success. All logic is in clr_lib.sh (sourced).
set -u
: "${CLR_PI_GO:?set CLR_PI_GO to the PI go time (HH:MM): this launcher refuses to run without the go of the PI}"
MODE=chain
. "$(dirname "${BASH_SOURCE[0]}")/clr_lib.sh" || exit 64
clr_launch
