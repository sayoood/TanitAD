#!/bin/bash
# restart_e3.sh [DRY|REAL] <REC SHA40> <CAP s/step> -- MM 4a3d05, 2026-11-23 (derived from restart_e2dc02cl.sh). E3 = the refcv9 FULL run (73,280 steps) of arm K2DC02, study e3 (root /home/nvidia/r8ns/r9e3,
# profile refcv9-full): `R9_STUDY=e3 bash r9gate3_restart.sh <tip> "K2DC02" <REC12>`. REAL needs the PI's go recorded in stack/ops/runs.d/r9e3_K2DC02.pi_cost_approval.json with max_s_per_step == CAP (10.5).
# REC = the commit queue 28 bound E3's (and D1's) records at; the launch tree is the remote TIP and its record inputs must equal REC's.
# Guards: FDISK / THORDISK >= 15 GB, THORSHM, NORECORDS / WRONGV (queue 28 ended DONE or DONE-E3ONLY, and bound K2DC02), RECINPUT (model/harness closure unchanged REC..tip), NOAPPROVAL (the cap, both arms),
#   NOARM (the E3 argv: steps 73280, seed 0, save-at 30000, banded map), NOZGENV (r9gate3_zg_env.sh at the tip AND the e3 root's zerograd_controls.env on Thor), NOPAUSE (the early-D1 stamp), NOKIT (the e3 root's kit),
#   CHAINLIVE (no gate3 chain whose cwd is the e3 root), GPULOCK (REAL only: the Thor GPU lock is free: K2DC02cl's chain must have ended).
# DRY lists every failing guard and arms / restarts nothing (it only READS Thor: ssh df / cat / ps / readlink).
# THE EARLY-D1 PAUSE (PI 2026-11-11 "read it early"): the chain writes PAUSE_AT_STEP into the arm's manifest from $R/gate3/PAUSE_AT_STEP (r9gate3_chain.sh PAUSE-MANIFEST-BEGIN..END; never for a *_d1 arm); the stamp is
# written by prep_q28.sh. The runbook after the pause is in HANDOFF.md; resume_e3.sh is the E3 restart after D1's chain ended.
export PATH="/usr/bin:/bin:$PATH"
trap '' INT
V=K2DC02; MODE=${1:?usage: restart_e3.sh DRY|REAL <REC SHA40> <CAP>}; [ "$MODE" = REAL ] && MODE=""
S=${R9K_LOG:-C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad}
OUT=$S/AUTO_E3.txt; GD=C:/Users/Admin/tanitad-push/.git; BR=agent/arch-inf-20260803
REC=${2:?REC SHA40}; CAP=${3:?CAP s/step}; [ ${#REC} = 40 ] || { echo "REC must be 40 chars"; exit 2; }; RC12=${REC:0:12}; R=/home/nvidia/r8ns/r9e3
RD=stack/ops/runs.d; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT"; }
th() { timeout 90 ssh -n -o ConnectTimeout=20 tanitad-thor "$1" 2>/dev/null | tr -d '\r'; }
FAILS=0
bad() { say "ZZAUTOE3-$1ZZ $2"; if [ "$MODE" = DRY ]; then FAILS=$((FAILS+1)); else exit "${3:-5}"; fi; }
say "START (pid $$) V $V mode ${MODE:-REAL} REC $RC12 CAP $CAP"
FG=$(df -BG /f | awk 'NR==2 {gsub("G","",$4); print $4}')
[ "$FG" -ge 15 ] 2>/dev/null || bad FDISK "F: free ${FG} GB < 15"
TG=$(th "df -BG --output=avail / | tail -1" | tr -d ' G')
[ "$TG" -ge 15 ] 2>/dev/null || bad THORDISK "Thor / free ${TG} GB < 15 (E3 holds ckpt.pt ~1.6 GB + ckpt_full_30000.pt ~1.6 GB + snapshots; sweep MY dead gate3 smoke checkpoints and finished-arm scratch first, never another session's)"
SH=$(th "df -BG --output=used /dev/shm | tail -1" | tr -d ' G')
[ "$SH" -le 5 ] 2>/dev/null || bad THORSHM "Thor /dev/shm holds ${SH} GB (> 5): orphan torch_* segments starve the gate smoke; attribute and sweep first"
QS=$R/logs/recq28_$RC12.status
# E3 needs only ITS OWN two records: the queue writes "<V>-RECORDS-READY at <c12>" right after both E3 binds pass, before any D1 step, so a later D1
# failure (MEASURED 2026-11-24: a marker-regex miss on K2DC02_d1) cannot void them. D1's records are d1_restart.sh's guard, not this one.
[ "$(th "grep -c ' $V-RECORDS-READY at $RC12 ' $QS")" -ge 1 ] 2>/dev/null || bad NORECORDS "queue 28 has no ' $V-RECORDS-READY at $RC12 ' line in $QS (E3's own map-banded + box records; last line: $(th "tail -1 $QS" | cut -c1-120))"
[ "$(th "grep -c 'QUEUE28-START $RC12 code .* k2 $V ' $QS")" -ge 1 ] 2>/dev/null || bad WRONGV "queue 28 did not bind $V ($QS)"
[ "$(th "grep -c -E '(mapb|box)_$V OK ' $QS")" = 2 ] 2>/dev/null || bad NORECORDS "$QS lacks the two OK lines of map-banded + box $V"
[ -n "$(th "test -s $R/kit/r9w_gsuite_precheck.py && echo y")" ] || bad NOKIT "no $R/kit/r9w_gsuite_precheck.py on Thor (prep_q28.sh ships the e3 kit)"
[ -n "$(th "test -s $R/gate3/zerograd_controls.env && echo y")" ] || bad NOZGENV "no $R/gate3/zerograd_controls.env on Thor (prep_q28.sh installs it)"
PS=$(th "cat $R/gate3/PAUSE_AT_STEP 2>/dev/null")
case "$PS" in "") bad NOPAUSE "no $R/gate3/PAUSE_AT_STEP: E3 would run past step 30,000 without releasing the GPU for the early D1 (prep_q28.sh writes it; R9E3_PAUSE_AT=none is the explicit opt-out)";; *[!0-9]*|0*) bad NOPAUSE "PAUSE_AT_STEP '$PS' is not a positive integer";; esac
[ "$(th "cat $R/gate3/REC_C12 2>/dev/null")" = "$RC12" ] || bad NORECSTAMP "$R/gate3/REC_C12 is not $RC12"
for try in 1 2 3; do
  git --git-dir=$GD fetch -q origin $BR || { say "fetch failed (try $try)"; sleep 60; continue; }
  T=$(git --git-dir=$GD ls-remote origin $BR | cut -c1-40)
  git --git-dir=$GD cat-file -e "$T^{commit}" || { say "tip $T not in mirror (try $try)"; sleep 60; continue; }
  git --git-dir=$GD diff --quiet $REC $T -- $RD/r9e3_$V.argv.json $RD/r9e3_${V}_d1.argv.json stack/scripts/map_hires_overfit.py \
      stack/scripts/g_box_overfit.py stack/scripts/refc_v3_train.py stack/tanitad taniteval || bad RECINPUT "a record input changed $REC..$T: MM decision (the records no longer match the launch tree's closure)" 6
  NC=$(git --git-dir=$GD diff $REC $T -- stack/ops/r9e1_bind.sh | grep -E '^[+-][^+-]' | grep -v -E '^[+-][[:space:]]*#' | wc -l)
  [ "$NC" = 0 ] || bad RECINPUT "r9e1_bind.sh has $NC non-comment changed lines $REC..$T" 6
  for A in $V ${V}_d1; do
    git --git-dir=$GD show "$T:$RD/r9e3_$A.pi_cost_approval.json" 2>/dev/null | grep -q "\"max_s_per_step\": $CAP," || bad NOAPPROVAL "no $RD/r9e3_$A.pi_cost_approval.json with the $CAP cap at $T" 14
  done
  OKA=$(git --git-dir=$GD show "$T:$RD/r9e3_$V.argv.json" | C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -c "import json,sys; a=json.load(sys.stdin)['argv']; v=lambda k: a[a.index(k) + 1]; print(1 if a.count('--steps') == 1 and v('--steps') == '73280' and a.count('--seed') == 1 and v('--seed') == '0' and a.count('--save-at') == 1 and v('--save-at') == '30000' and a.count('--r9-map-banded') == 1 and a.count('--r9-hist-supervise-geom') == 1 and '--wsd-branch-from' not in a else 0)" | tr -d '\r')
  [ "$OKA" = 1 ] || bad NOARM "r9e3_$V.argv.json at $T is not the E3 argv (--steps 73280, --seed 0, --save-at 30000, banded map, no --wsd-branch-from)" 14
  git --git-dir=$GD cat-file -e "$T:stack/ops/r9e1_gate3/r9gate3_zg_env.sh" || bad NOZGENV "no r9gate3_zg_env.sh at $T" 14
  NCH=$(th "n=0; for p in \$(ps -eo pid,args | awk '/[k]it\\/r9gate3_chain.sh/ {print \$1}'); do [ \"\$(readlink /proc/\$p/cwd)\" = $R ] && n=\$((n+1)); done; echo \$n")
  [ "$NCH" = 0 ] || bad CHAINLIVE "$NCH gate3 chain process(es) with cwd $R (a live e3 chain: never a second one)" 7
  if [ -z "$MODE" ]; then
    [ -n "$(th "flock -n $LOCKF true && echo FREE")" ] || bad GPULOCK "the Thor GPU lock is held (the training arm's chain / supervisor): E3 restarts only after it released the lock" 7
  fi
  if [ "$MODE" = DRY ]; then say "ZZAUTOE3-DRY-ENDZZ tip $T: $FAILS failing guard(s), nothing armed or restarted"; exit 0; fi
  D=/c/Users/Admin/r9restart_e3_${T:0:7}
  [ -e "$D" ] && { say "ZZAUTOE3-KITEXISTSZZ $D already exists"; exit 9; }
  mkdir -p "$D" && git --git-dir=$GD -c core.autocrlf=false archive $T stack/ops | tar -x -C "$D" || { say "ZZAUTOE3-KITZZ archive"; exit 9; }
  BAD=0
  while read -r mode typ blob path; do
    [ "$(git --git-dir=$GD hash-object --no-filters "$D/$path")" = "$blob" ] || { say "blob mismatch $path"; BAD=1; }
  done < <(git --git-dir=$GD ls-tree -r $T stack/ops/r9e1_gate3)
  [ "$BAD" = 0 ] || { say "ZZAUTOE3-KITZZ blob mismatch"; exit 9; }
  say "kit $D verified; restarting [$V] REC12 $RC12 study e3 (PAUSE_AT_STEP stamp $PS)"
  R9_STUDY=e3 R9_PRECHECK_ARMS="$V" bash "$D/stack/ops/r9e1_gate3/r9gate3_restart.sh" "$T" "$V" $RC12 > "$D/restart.out" 2>&1
  tail -3 "$D/restart.out" | cut -c1-240 | tee -a "$OUT"
  if grep -q "ZZR9GATE3-RESTART-OKZZ ${T:0:12}" "$D/restart.out"; then
    C12=${T:0:12}; t0=$(date +%s)
    while :; do
      ST=$(th "awk '/ START commit $C12 /{f=1} f' $R/gate3/CHAIN_STATUS.txt | grep -E ' (SUITE-STARTED|STOP-[A-Z0-9-]+) ' | tail -1")
      case "$ST" in
        *" STOP-"*) say "ZZAUTOE3-CHAINSTOPZZ $(echo "$ST" | cut -c1-200)"; exit 11;;
        *SUITE-STARTED*) say "suite started: $(echo "$ST" | cut -c1-120)"; break;;
      esac
      [ $(( $(date +%s) - t0 )) -gt 5400 ] && { say "ZZAUTOE3-NOSUITEZZ no SUITE-STARTED in 90 min"; exit 12; }
      sleep 60
    done
    say "ZZAUTOE3-OKZZ $T [$V]: E3 chain armed (the gate and the 73,280-step run follow detached; pause at ${PS:-none}); freeze closure-touching lands until D1's records and D1 itself are done"
    exit 0
  fi
  grep -q "is not the remote tip" "$D/restart.out" && { say "tip moved during the restart (try $try)"; rm -rf -- "${D:?}"; continue; }
  say "ZZAUTOE3-RESTARTREFUSEDZZ (see $D/restart.out)"; exit 8
done
say "ZZAUTOE3-TRIESZZ"; exit 10
