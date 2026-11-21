#!/usr/bin/env bash
# LEVELHEAD arm CL, STEP 1: the two-run chain CL_0 then CL_1 (3,000 steps each). REFUSES (locally, before the lock and any remote call) while W is the placeholder __W__, FROZEN_W.txt is absent, or it disagrees with the argv.
# RUN ONLY AFTER THE PI'S BOX-2 GO: `CL_PI_GO="<HH:MM>" bash .../cl_launch/launch_cl.sh`. Prints ZZCL-LAUNCHED chain on success. All logic is in cl_lib.sh (sourced).
set -u
: "${CL_PI_GO:?set CL_PI_GO to the PI go time (HH:MM): this launcher refuses to run without the go of the PI}"
MODE=chain
. "$(dirname "${BASH_SOURCE[0]}")/cl_lib.sh" || exit 64
cl_launch
