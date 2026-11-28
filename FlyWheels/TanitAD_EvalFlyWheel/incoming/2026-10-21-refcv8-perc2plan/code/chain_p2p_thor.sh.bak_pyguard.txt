#!/bin/bash
# chain_p2p_thor.sh -- PERC2PLAN-R8 (PREREG_PERC2PLAN.md sha256 d1a2eb53...): 12 os-only rolls, s1 first then s0,
# order BASE, AGT_ZERO, BEV_ZERO, BOTH_ZERO, AGT_SHUF, BEV_SHUF; each roll under the Thor GPU lock (blocking flock, 2 h).
# Refuses to start while a sibling agent (fanceil / perc8) has ANY process, or while any python runs.  Resumable: a roll
# whose meta says C4 pass + rc 0 is skipped.  Markers ZZC<NAME> in out/chain.log (opaque; never grep the command text).
P=/home/nvidia/r8ns/perc2plan; L=$P/out/chain.log; mkdir -p $P/out
echo "ZZCSTART $(date -Is) pid $$" >> $L
sib=$(ps -eo args | grep -cE "[r]8ns/(fanceil|perc8)/|[w]ait_and_run|[c]hain_fanceil|[c]hain_perc8")
pys=$(ps -u nvidia -o pid=,comm= | awk '$2 ~ /^python/' | wc -l)
if [ "$sib" != "0" ] || [ "$pys" != "0" ]; then echo "ZZCBUSY sib=$sib py=$pys $(date -Is)" >> $L; exit 3; fi
bad=0
while read m p n; do got=$(md5sum "$p" 2>/dev/null | cut -c1-32); if [ ${#got} -ne 32 ] || [ "$got" != "$m" ]; then echo "ZZCMD5BAD $n got=$got" >> $L; bad=1; fi; done < $P/code/expected_md5_thor_paths.txt
(cd $P/ref && sha256sum -c --quiet $P/ref_sha256.txt) >> $L 2>&1 || { echo "ZZCREFSHABAD" >> $L; bad=1; }
[ $bad = 0 ] || { echo "ZZCINPUTSBAD $(date -Is)" >> $L; exit 2; }
echo "ZZCINPUTSOK $(wc -l < $P/code/expected_md5_thor_paths.txt) files md5 + ref tree sha256 $(date -Is)" >> $L
for S in 1 0; do
  for A in BASE AGT_ZERO BEV_ZERO BOTH_ZERO AGT_SHUF BEV_SHUF; do
    flock -w 7200 /home/nvidia/refcv7_post/thor_gpu.lock bash $P/code/run_p2p_thor.sh $A $S 200>&-
    echo "ZZCROLL $A s$S rc=$? $(date -Is)" >> $L
  done
done
echo "ZZCEND $(date -Is)" >> $L
