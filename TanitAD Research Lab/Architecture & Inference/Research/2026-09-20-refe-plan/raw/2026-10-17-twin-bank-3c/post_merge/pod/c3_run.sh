#!/bin/bash
# POD-SIDE, launched DETACHED by post_merge_to_p6gate.sh.   (c'3) of PREREG_3C.md = HANDOFF.md step 7, command for command:
#   D7 `diag_d7_twin_route.py verify` (unchanged instrument; --limit 1000 is the build agent's choice, not pre-registered) once per rank, in parallel,
#   then `reroll_twins.py check` (must print ZZTWIN_HOLD_OK).
# Env in: ENV3C F OUT MC MC_TWIN KS EXPECT_SHA.   Writes ONLY under $F (verify_k*.json/.log, twin_hold_check.*, c3_merged_sha.txt, PID, C3_FINISHED).
set -u
export PYTHONDONTWRITEBYTECODE=1                  # no __pycache__ written into the 3-C or P6 code trees
source "$ENV3C"                                   # PKG CODE3C PY REFE_NUPLAN_DB_ROOT NUPLAN_MAPS_ROOT ...
echo $$ > "$F/c3.PID"
rm -f "$F/C3_FINISHED" "$F/twin_hold_check.out" "$F/twin_hold_check.rc"
cur=$(sha256sum "$OUT/targets_rank1.jsonl" | cut -d' ' -f1)
if [ "$cur" != "$EXPECT_SHA" ]; then echo "BAD merged bank changed under the script" > "$F/twin_hold_check.out"; touch "$F/C3_FINISHED"; exit 3; fi
echo "$cur" > "$F/c3_merged_sha.txt"
cd "$PKG/refe" || exit 5
export PYTHONPATH=$PKG/refe:$PKG/code
pids=()
for k in $KS; do
  env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK -u REFE_D7_MUTATE OMP_NUM_THREADS=1 nice -n 10 $PY diag_d7_twin_route.py verify --bank "$OUT" --k $k \
      --twin-cache "$MC_TWIN" --logged-cache "$MC" --db-root "$REFE_NUPLAN_DB_ROOT" --map-root "$NUPLAN_MAPS_ROOT" --limit 1000 \
      --out "$F/verify_k$k.json" > "$F/verify_k$k.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done                  # verify exits 1 whenever (c) fails; the verdict is read from its ZZD7V_SUMMARY line, never from this rc
python3 "$CODE3C/reroll_twins.py" check --bank "$OUT" > "$F/twin_hold_check.out" 2>&1
echo $? > "$F/twin_hold_check.rc"
touch "$F/C3_FINISHED"
