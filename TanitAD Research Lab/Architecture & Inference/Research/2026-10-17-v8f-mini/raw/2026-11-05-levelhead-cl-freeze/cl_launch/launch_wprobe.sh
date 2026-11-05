#!/usr/bin/env bash
# LEVELHEAD arm CL, STEP 0: the W probe (CL_0's argv with --selfcheck-only --level-head on --w-level 1, its own --out ...\M2b_s0_wprobe_CL). Trains nothing; writes selfcheck.json on box 2.
# RUN ONLY AFTER THE PI'S BOX-2 GO: `CL_PI_GO="<HH:MM>" bash .../cl_launch/launch_wprobe.sh`. Prints ZZCL-LAUNCHED wprobe on success. All logic is in cl_lib.sh (sourced).
set -u
: "${CL_PI_GO:?set CL_PI_GO to the PI go time (HH:MM): this launcher refuses to run without the go of the PI}"
MODE=wprobe
. "$(dirname "${BASH_SOURCE[0]}")/cl_lib.sh" || exit 64
cl_launch
