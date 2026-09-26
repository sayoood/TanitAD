#!/bin/bash
# DZ-11 preflight guard -- MUTATION TEST. Scratch fixture only; no real data is touched.
#
# ⛔ WHY THIS IS RE-RUN RATHER THAN RE-READ. The split guard was changed on 2026-09-20 to require
# only the splits the REQUESTED TASKS read, instead of both unconditionally. A guard edited without
# a mutation test is a guard that may be green forever -- and this package has already produced four
# checks that could not go red. Every arm below must land on its stated exit code, and the PASS
# controls matter as much as the refusals: a guard that refuses EVERYTHING passes all the refusal
# arms and is wrong in the other direction.
#
# ⛔ Exit codes are read from $? DIRECTLY, never through a pipe -- `cmd | tail` reports tail's
# status and has twice reported a clean exit for a failing command in this programme.
set -u
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
S="$PKG/code/run_headline_suite.sh"
FIX="${TMPDIR:-/tmp}/dz11fix.$$"
TV="$FIX/nuplan-v1.1/splits/trainval"; TE="$FIX/nuplan-v1.1/splits/test"
mkdir -p "$TV" "$TE"
mk() { local d="$1" n="$2" bytes="$3"; local i; for ((i=0;i<n;i++)); do
         if [[ "$bytes" -gt 0 ]]; then head -c "$bytes" /dev/zero > "$d/f$i.db"; else : > "$d/f$i.db"; fi; done; }

pass=0; fail=0
run() {  # run <label> <expected_exit|PROCEED> <env-assignments...>
  local label="$1" exp="$2"; shift 2
  local out rc
  out=$(env "$@" DZ11_DATA_ROOT="$FIX" bash "$S" "$TASKS_ARG" 2>&1); rc=$?
  local verdict
  if [[ "$exp" == "PROCEED" ]]; then
    if grep -q "HEADLINE_RUN" <<<"$out"; then verdict=PASS; else verdict=FAIL; fi
  else
    if [[ "$rc" == "$exp" ]]; then verdict=PASS; else verdict=FAIL; fi
  fi
  [[ "$verdict" == PASS ]] && pass=$((pass+1)) || fail=$((fail+1))
  printf '%-58s expect %-8s rc %-3s %s\n' "$label" "$exp" "$rc" "$verdict"
  grep -oE "GUARD_FAIL [a-z]+: [^\"]{0,58}|REFUSING an inherited|NOT required: [^\"]{0,40}" <<<"$out" | head -2 | sed 's/^/      /'
}

echo "== DZ-11 guard mutation test =="
echo "fixture: $FIX"

# ---- arms that must REFUSE ------------------------------------------------------------------
mk "$TE" 64 1048576            # mini-sized test split
TASKS_ARG="test14hard_nr"
run "1 test split is MINI-sized (64 DBs)" 3
rm -f "$TE"/*.db; mk "$TE" 300 0     # right count, zero bytes
run "2 test split has 300 DBs but 0 GB (partial extract)" 3
rm -f "$TE"/*.db; mk "$TE" 300 1048576
run "3 inherited DRIVERL_EVAL_DB_LINK_ROOT" 4 \
    DZ11_MIN_GB_TEST=0 DRIVERL_EVAL_DB_LINK_ROOT="$FIX/dblinks"
TASKS_ARG="val14_nr"
rm -rf "$TV"; run "4 val14 asked for while trainval is ABSENT" 3 DZ11_MIN_GB_TEST=0
TASKS_ARG="not_a_task"
run "5 task name names neither split" 3 DZ11_MIN_GB_TEST=0 DZ11_MIN_GB_TRAINVAL=0

# ---- controls that must PROCEED -------------------------------------------------------------
TASKS_ARG="test14hard_nr,test14random_nr"
run "6 CONTROL test14 only, trainval ABSENT -> must PROCEED" PROCEED DZ11_MIN_GB_TEST=0
mkdir -p "$TV"; mk "$TV" 300 1048576
TASKS_ARG="val14_nr,test14hard_nr"
run "7 CONTROL both asked for, both present -> must PROCEED" PROCEED \
    DZ11_MIN_GB_TEST=0 DZ11_MIN_GB_TRAINVAL=0

rm -rf "$FIX"
echo
echo "PASS $pass   FAIL $fail"
[[ "$fail" -eq 0 ]] && echo "GUARD_MUTATION_OK" || echo "GUARD_MUTATION_FAILED"
