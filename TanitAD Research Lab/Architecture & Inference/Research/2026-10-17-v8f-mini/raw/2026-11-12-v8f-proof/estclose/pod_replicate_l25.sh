#!/usr/bin/env bash
# pod_replicate_l25.sh -- beside card ADD79 (not a control of the card): one rollout of the card repeated with the identical command line into another folder, to measure whether the
# T1-kin/hb0.2 rollout with the rule is the same at a second inference run (asked by the other session for row 141). Run: I10RQ50eL25KF5P30Hb20_s0 (AGE, estimator L25, seed 0).
# The command is the line that the queue printed in its dry run (/tmp/dry_estclose.out), with the output folder t1/<run> replaced by rep_test/<run>. Writes only into $W/rep_test.
# No process listing or stopping, no file removal or move. Refuses to run twice.
set -u
W=/workspace/v8proof/integ10k
R=$W/rep_test
RUN=I10RQ50eL25KF5P30Hb20_s0
[ -e "$R/$RUN" ] && { echo "REFUSED: $R/$RUN exists"; exit 2; }
[ -f /tmp/dry_estclose.out ] || { echo "STOP: no dry-run listing"; exit 3; }
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
free_gb=$(( (mm - an) / 1073741824 )); [ "$free_gb" -ge 28 ] || { echo "STOP: memory free $free_gb GB"; exit 3; }
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n" -lt 3 ] || { echo "STOP: GPU jobs $n"; exit 3; }
cmd=$(grep '^CMD: ' /tmp/dry_estclose.out | grep -- "--tag ${RUN}_m10_causal_heads " | sed "s/^CMD: //; s#integ10k/t1/${RUN}#integ10k/rep_test/${RUN}#g")
[ -n "$cmd" ] || { echo "STOP: command line not found"; exit 3; }
[ "$(printf '%s\n' "$cmd" | wc -l)" = 1 ] || { echo "STOP: more than one command line"; exit 3; }
case "$cmd" in *"integ10k/t1/"*) echo "STOP: the command still writes under t1"; exit 3;; esac
mkdir -p "$R/$RUN/ticks"
printf '%s\n' "$cmd" > "$R/cmd_$RUN.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$R/START_$RUN.txt"
bash -c "$cmd" > "$R/$RUN/t1.log" 2>&1
echo "rc $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$R/DONE_$RUN.txt"
