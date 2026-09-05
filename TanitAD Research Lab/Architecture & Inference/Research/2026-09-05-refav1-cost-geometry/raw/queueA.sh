#!/usr/bin/env bash
# LADDER lane A: W_KAPPA = 15.11245 (10 % of the measured goal decision at the
# realised curvature) then 151.1245 (100 %). One variable vs the banked
# `ccos_argmax` (W_KAPPA = 0): the curvature penalty.
set -u
. "$(dirname "$0")/wk_lib.sh"
wait_pids "23596,4456"
run_wk wk15  15.11245
run_wk wk151 151.1245
echo "ZZQUEUEA-DONE-$(date -u +%FT%TZ)ZZ"
