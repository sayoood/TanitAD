#!/bin/bash
# restart_e2dc02cl.sh [DRY|REAL] <REC SHA40> <CAP s/step> -- MM 4a3d05, 2026-11-20 (derived from restart_e2sg.sh md5 15313d1b). The E2 chain with arms [K2DC02cl] ONLY (K0 trained at
# 269f43973a46 and is not re-run). K2DC02cl = K2DC02's flags with the CLAMP (--w-map-hires 3.0, --map-hires-w-bounds 2.5,3.5) (PREREG_K2DC02CL_2026-11-22.md; trains only if the K2DC02s1 read is DOSE-VOID). REAL needs the PI's go recorded in
# stack/ops/runs.d/r9e2_K2DC02cl.pi_cost_approval.json with max_s_per_step == CAP (conditional approval: the K2DC02s1 read must be DOSE-VOID, which the MM checks before REAL). REC = the commit queue 27 bound K2DC02cl's records at
# (then K0's, queue 19). Guards: FDISK / THORDISK >= 15 GB, THORSHM, NORECORDS / WRONGV / NOK0REC, RECINPUT / K0CLOSURE, NOAPPROVAL, NOREDOSE, NODC / NOK2DC02, NOARM, NOZGENV, CHAINLIVE.
# DRY lists every failing guard and arms / restarts nothing.
export PATH="/usr/bin:/bin:$PATH"
trap '' INT
V=K2DC02cl; MODE=${1:?usage: restart_e2dc02cl.sh DRY|REAL <REC SHA40> <CAP>}; [ "$MODE" = REAL ] && MODE=""
S=${R9K_LOG:-C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad}
OUT=$S/AUTO_E2CL.txt; GD=C:/Users/Admin/tanitad-push/.git; BR=agent/arch-inf-20260803
REC=${2:?REC SHA40}; CAP=${3:?CAP s/step}; [ ${#REC} = 40 ] || { echo "REC must be 40 chars"; exit 2; }; RC12=${REC:0:12}; R1=/home/nvidia/r8ns/r9e1; R=/home/nvidia/r8ns/r9e2
RD=stack/ops/runs.d
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT"; }
th() { timeout 90 ssh -n -o ConnectTimeout=20 tanitad-thor "$1" 2>/dev/null | tr -d '\r'; }
FAILS=0
bad() { say "ZZAUTOE2CL-$1ZZ $2"; if [ "$MODE" = DRY ]; then FAILS=$((FAILS+1)); else exit "${3:-5}"; fi; }
say "START (pid $$) V $V mode ${MODE:-REAL}"
FG=$(df -BG /f | awk 'NR==2 {gsub("G","",$4); print $4}')
[ "$FG" -ge 15 ] 2>/dev/null || bad FDISK "F: free ${FG} GB < 15"
TG=$(th "df -BG --output=avail / | tail -1" | tr -d ' G')
[ "$TG" -ge 15 ] 2>/dev/null || bad THORDISK "Thor / free ${TG} GB < 15 (one arm: sweep MY dead gate3/ and gate3_explore/ smoke checkpoints first)"
SH=$(th "df -BG --output=used /dev/shm | tail -1" | tr -d ' G')
[ "$SH" -le 5 ] 2>/dev/null || bad THORSHM "Thor /dev/shm holds ${SH} GB (> 5): orphan torch_* segments starve the gate smoke; attribute and sweep first"
QS=$R1/logs/recq27_$RC12.status
[ "$(th "tail -1 $QS")" = ZZRECQ27-DONEZZ ] || bad NORECORDS "queue 27 has not ended ZZRECQ27-DONEZZ ($QS: $(th "tail -1 $QS" | cut -c1-120))"
[ "$(th "grep -c 'QUEUE27-START $RC12 code .* k2 $V ' $QS")" -ge 1 ] 2>/dev/null || bad WRONGV "queue 27 did not bind $V ($QS)"
# 11-11 02:09:44Z the explore gate (arm K0) REFUSED: one REC_C12 serves every arm incl. the explore arm, so K0 needs records at $RC12 too (queue 19)
Q19=$R1/logs/recq19_$RC12.status
[ "$(th "tail -1 $Q19")" = ZZRECQ19-DONEZZ ] || bad NOK0REC "queue 19 (K0's explore-gate records at $RC12) has not ended ZZRECQ19-DONEZZ ($Q19)"
for try in 1 2 3; do
  git --git-dir=$GD fetch -q origin $BR || { say "fetch failed (try $try)"; sleep 60; continue; }
  T=$(git --git-dir=$GD ls-remote origin $BR | cut -c1-40)
  git --git-dir=$GD cat-file -e "$T^{commit}" || { say "tip $T not in mirror (try $try)"; sleep 60; continue; }
  git --git-dir=$GD diff --quiet $REC $T -- $RD/r9e2_K0.argv.json $RD/r9e2_$V.argv.json stack/scripts/map_hires_overfit.py \
      stack/scripts/g_box_overfit.py stack/scripts/refc_v3_train.py stack/tanitad || bad RECINPUT "a record input changed $REC..$T: MM decision" 6
  AUD="stack/scripts/refc_v3_train.py stack/tanitad/eval/refcv7_loader.py stack/tanitad/models/map_head_hires.py stack/tanitad/refs/refcv9_perc_rig.py stack/tanitad/train/map_own_enc_flag.py stack/tanitad/train/map_own_enc_lift64_flag.py stack/tanitad/train/refcv8_train.py stack/tanitad/train/map_dose_ctl.py stack/tanitad/train/map_trunk_gs.py"
  UNAUD=$(git --git-dir=$GD diff --name-only 269f43973a46a3a2ad86b3f6d31f9d01c77e8751 $T -- stack/scripts/refc_v3_train.py stack/tanitad | grep -vxF -f <(tr ' ' '\n' <<< "$AUD") | tr '\n' ' ')
  [ -z "$UNAUD" ] || bad K0CLOSURE "model code changed since K0's 269f43973a46 OUTSIDE the audited set: $UNAUD. MM decision" 6
  git --git-dir=$GD diff --quiet $REC $T -- stack/scripts/refc_v3_train.py stack/tanitad || bad K0CLOSURE "model code changed since the K2DC02cl records commit $REC: re-audit. MM decision" 6
  NC=$(git --git-dir=$GD diff $REC $T -- stack/ops/r9e1_bind.sh | grep -E '^[+-][^+-]' | grep -v -E '^[+-][[:space:]]*#' | wc -l)
  [ "$NC" = 0 ] || bad RECINPUT "r9e1_bind.sh has $NC non-comment changed lines $REC..$T" 6
  for AC in K0:10.5 $V:$CAP; do
    A=${AC%%:*}; CAP=${AC#*:}
    git --git-dir=$GD show "$T:$RD/r9e2_$A.pi_cost_approval.json" 2>/dev/null | grep -q "\"max_s_per_step\": $CAP," || bad NOAPPROVAL "no $RD/r9e2_$A.pi_cost_approval.json with the $CAP cap at $T" 14
  done
  git --git-dir=$GD show "$T:$RD/r9e2_$V.argv.json" | grep -q '"3.0"' || bad NOREDOSE "r9e2_$V.argv.json at $T carries no 3.0 (the initial weight inside the clamp)" 14
  git --git-dir=$GD show "$T:$RD/r9e2_$V.argv.json" | grep -q '"25.0"' && bad NOSTALE "r9e2_$V.argv.json at $T still carries K2DC02's 25.0" 14
  git --git-dir=$GD show "$T:$RD/r9e2_$V.argv.json" | grep -q '"--map-hires-share-target"' || bad NODC "r9e2_$V.argv.json at $T carries no controller" 14
  git --git-dir=$GD show "$T:$RD/r9e2_$V.argv.json" | grep -q '"2.5,3.5"' || bad NOK2DC02 "r9e2_$V.argv.json at $T lacks the clamp bounds 2.5,3.5" 14
  git --git-dir=$GD show "$T:$RD/r9e2_$V.argv.json" | grep -q '"0.1,25"' && bad NOSTALE "r9e2_$V.argv.json at $T still carries bounds 0.1,25" 14
  OKA=$(git --git-dir=$GD show "$T:$RD/r9e2_$V.argv.json" | C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -c "import json,sys; a=json.load(sys.stdin)['argv']; print(1 if a.count('--seed') == 1 and a[a.index('--seed') + 1] == '0' and a.count('--r9-hist-supervise-geom') == 1 else 0)" | tr -d '\r')
  [ "$OKA" = 1 ] || bad NOARM "r9e2_$V.argv.json at $T is not the K2DC02cl argv (--seed 0 exactly once and --r9-hist-supervise-geom exactly once)" 14
  git --git-dir=$GD cat-file -e "$T:stack/ops/r9e1_gate3/r9gate3_zg_env.sh" || bad NOZGENV "no r9gate3_zg_env.sh at $T" 14
  NCH=$(th "ps -eo args | grep -c -E '[k]it/r9gate3_chain.sh|[k]it/r9gate3_thor_gate.sh|[r]9_recqueue'")
  [ "$NCH" = 0 ] || bad CHAINLIVE "$NCH chain / gate / record-queue processes on Thor" 7
  if [ "$MODE" = DRY ]; then say "ZZAUTOE2CL-DRY-ENDZZ tip $T: $FAILS failing guard(s), nothing armed or restarted"; exit 0; fi
  D=/c/Users/Admin/r9restart_e2_${T:0:7}
  [ -e "$D" ] && { say "ZZAUTOE2CL-KITEXISTSZZ $D already exists"; exit 9; }
  mkdir -p "$D" && git --git-dir=$GD -c core.autocrlf=false archive $T stack/ops | tar -x -C "$D" || { say "ZZAUTOE2CL-KITZZ archive"; exit 9; }
  BAD=0
  while read -r mode typ blob path; do
    [ "$(git --git-dir=$GD hash-object --no-filters "$D/$path")" = "$blob" ] || { say "blob mismatch $path"; BAD=1; }
  done < <(git --git-dir=$GD ls-tree -r $T stack/ops/r9e1_gate3)
  [ "$BAD" = 0 ] || { say "ZZAUTOE2CL-KITZZ blob mismatch"; exit 9; }
  if [ "$(th "test -s $R/gate3/zerograd_controls.env && echo EXISTS")" = EXISTS ]; then   # a THOR test (11-11 02:05Z the local [ -s ] read the dev box and skipped the move)
    # 11-11 07:17Z: the aside name was FIXED (the 11-10 install's), so the 2nd move (fab2cbd8's zg_auto env) collided -> name it by the env's own mtime
    MT=$(th "date -u -r $R/gate3/zerograd_controls.env +%Y%m%dT%H%M%SZ")
    [ ${#MT} = 16 ] || { say "ZZAUTOE2CL-ASIDEZZ could not read the env's mtime ($MT)"; exit 15; }
    AS=$R/gate3/zgfix_20261110/zerograd_controls.env.aside_mtime_$MT
    th "test -e $AS && echo EXISTS" | grep -q EXISTS && { say "ZZAUTOE2CL-ASIDEEXISTSZZ $AS"; exit 15; }
    th "mv $R/gate3/zerograd_controls.env $AS && echo MOVED" | grep -qx MOVED || { say "ZZAUTOE2CL-ASIDEZZ could not move the env aside"; exit 15; }
    say "moved the previous segment's zerograd_controls.env aside to $AS"
  fi
  ZE=$(th "test -s $R/gate3/zerograd_controls.env && echo EXISTS")
  [ -z "$ZE" ] || { say "ZZAUTOE2CL-STALEENVZZ $R/gate3/zerograd_controls.env still exists. MM decision"; exit 15; }
  th "cd $R && setsid nohup bash kit/r9e2_zg_auto.sh $T > $R/logs/zg_auto_${T:0:12}.out 2>&1 < /dev/null & sleep 2; pgrep -f '[r]9e2_zg_auto.sh $T' | wc -l" | tail -1 | grep -qx '[1-9]' || { say "ZZAUTOE2CL-ZGARMZZ zg_auto did not start"; exit 16; }
  say "zg_auto armed for $T; kit $D verified; restarting [$V] REC12 $RC12 study e2"
  R9_STUDY=e2 R9_PRECHECK_ARMS="$V" bash "$D/stack/ops/r9e1_gate3/r9gate3_restart.sh" "$T" "$V" $RC12 > "$D/restart.out" 2>&1
  tail -3 "$D/restart.out" | cut -c1-240 | tee -a "$OUT"
  if grep -q "ZZR9GATE3-RESTART-OKZZ ${T:0:12}" "$D/restart.out"; then
    C12=${T:0:12}; t0=$(date +%s)
    while :; do
      ST=$(th "awk '/ START commit $C12 /{f=1} f' $R/gate3/CHAIN_STATUS.txt | grep -E ' (SUITE-STARTED|STOP-[A-Z0-9-]+) ' | tail -1")
      case "$ST" in
        *" STOP-"*) say "ZZAUTOE2CL-CHAINSTOPZZ $(echo "$ST" | cut -c1-200)"; exit 11;;
        *SUITE-STARTED*) say "suite started: $(echo "$ST" | cut -c1-120)"; break;;
      esac
      [ $(( $(date +%s) - t0 )) -gt 5400 ] && { say "ZZAUTOE2CL-NOSUITEZZ no SUITE-STARTED in 90 min"; exit 12; }
      sleep 60
    done
    say "ZZAUTOE2CL-OKZZ $T [$V]: arm the launch waiter; closure freeze holds until $V TRAINS (then message peer v8F local_7718c5d9-da1d-4e01-bfe3-fb70778185a6)"
    exit 0
  fi
  grep -q "is not the remote tip" "$D/restart.out" && { say "tip moved during the restart (try $try)"; th "pkill -f '[r]9e2_zg_auto.sh $T'"; rm -rf -- "${D:?}"; continue; }
  say "ZZAUTOE2CL-RESTARTREFUSEDZZ (see $D/restart.out)"; exit 8
done
say "ZZAUTOE2CL-TRIESZZ"; exit 10
