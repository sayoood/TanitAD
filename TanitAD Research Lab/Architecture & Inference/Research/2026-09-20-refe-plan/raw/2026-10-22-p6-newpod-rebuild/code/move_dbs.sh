#!/bin/bash
# Move the 1,192 nuPlan DBs (public, re-fetchable) from the network volume (quota ~16 GiB free) to the local container disk; symlink in place.
set -u
L=/root/tad; V=/workspace/data/navtrain_dbs; O=/workspace/ops/rebuild; mkdir -p $L $O
log() { echo "$(date -u +%FT%TZ) $*" >> $O/move_dbs.log; }
[ -L $V ] && { log "ZZMOVE_OK already a symlink -> $(readlink $V)"; exit 0; }
n=0; while [ $n -lt 20 ]; do dd if=/dev/zero of=$L/zz_q_$n bs=1M count=1024 conv=fsync status=none 2>/dev/null || break; n=$((n+1)); done; rm -f $L/zz_q_*
[ $n = 20 ] || { log "ZZMOVE_FAIL local write test: only $n GiB"; exit 1; }
log "local write test 20 GiB ok; copying"
rm -rf $L/navtrain_dbs.tmp; cp -a $V $L/navtrain_dbs.tmp || { log "ZZMOVE_FAIL cp"; exit 1; }
a=$(cd $V && find . -type f -printf '%P %s\n' | sort | sha256sum | cut -c1-16)
b=$(cd $L/navtrain_dbs.tmp && find . -type f -printf '%P %s\n' | sort | sha256sum | cut -c1-16)
c=$(ls $L/navtrain_dbs.tmp/*.db | wc -l)
[ "$a" = "$b" ] && [ "$c" -ge 1192 ] || { log "ZZMOVE_FAIL listing differs vol $a local $b dbs $c"; exit 1; }
mv $L/navtrain_dbs.tmp $L/navtrain_dbs && mv $V $V.vol_old && ln -s $L/navtrain_dbs $V || { log "ZZMOVE_FAIL swap"; exit 1; }
rm -rf $V.vol_old
log "ZZMOVE_OK $c dbs, name+size listing $a identical; $V -> $L/navtrain_dbs"
