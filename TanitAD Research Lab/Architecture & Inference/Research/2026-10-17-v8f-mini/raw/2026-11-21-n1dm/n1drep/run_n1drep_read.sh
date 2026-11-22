#!/bin/bash
# run_n1drep_read.sh -- MM 2026-11-22, card PREREG_V8F_N1DREP (0c641197). Dev box, CPU only (CUDA_VISIBLE_DEVICES=-1), one python at a time, each gated on commit >= 5.5 GB.
# 1 wait for the pod chain's last line (ZZ N1DREP DONE ZZ / FAIL), polling by ssh every 60 s, cap 4 h
# 2 pull N1DREP_s0's checkpoint and the pod's sha256 line; check it
# 3 eval-141 T1 hb0.2 follow dump of N1DREP_s0 (n1dm_dumps.py N1DM-N1D_s0 with N1DM_CKPTS_N1D -> this folder's ckpts)   -> dumps_rep
# 4 K-REDUMP: the same dump of N1D_s0 from its banked checkpoint, same session / tree / env                               -> dumps_redump
# 5 n1drep_read.py -> RESULT_N1DREP.json
# Last line of the log: ZZ N1DREP-READ DONE ZZ <outcome> or ZZ N1DREP-READ FAIL ZZ <why>
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
N=F:/TanitAD-scratch/n1dc/n1drep
EV=F:/TanitAD-scratch/n1dc/read_n1dm/eval141
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$N/n1drep_read_chain.log
export TMP=$N/tmp TEMP=$N/tmp
mkdir -p $N/tmp $N/ckpts/N1D_s0 $N/out
fail() { echo "ZZ N1DREP-READ FAIL ZZ $1" >> $LOG; exit 1; }
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
gate() { $PY F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 >> $LOG 2>&1 || fail "commit headroom below 5.5 GB for 60 min before $1"; }
source C:/Users/Admin/lorapod/podenv.sh
say "start"
last=""
for i in $(seq 1 240); do
  last=$(pssh 'tail -n 1 /root/strat1/n1drep/n1drep_chain.log' 2>/dev/null | tr -d '\r')
  case "$last" in "ZZ N1DREP DONE ZZ"*) break;; "ZZ N1DREP FAIL ZZ"*) fail "pod chain: $last";; esac
  sleep 60
done
case "$last" in "ZZ N1DREP DONE ZZ"*) say "pod chain done";; *) fail "pod chain not done after 4 h (last line: $last)";; esac
pscp -q "$POD_TARGET:/root/strat1/n1drep/runs/N1D_s0/train/ckpt_last.pt" $N/ckpts/N1D_s0/ckpt_last.pt || fail "pull ckpt"
pscp -q "$POD_TARGET:/root/strat1/n1drep/SHA256_N1DREP_CKPT.txt" $N/ckpts/SHA256_N1DREP_CKPT_pod.txt || fail "pull sha line"
sha=$(cut -d' ' -f1 $N/ckpts/SHA256_N1DREP_CKPT_pod.txt)
[ ${#sha} = 64 ] || fail "pod sha line malformed"
echo "$sha  N1D_s0/ckpt_last.pt" > $N/ckpts/SHA256_PODS.txt
(cd $N/ckpts && sha256sum -c SHA256_PODS.txt) >> $LOG 2>&1 || fail "pulled ckpt sha256 differs from the pod's"
say "ckpt ${sha:0:16} pulled and checked"
cd $EV || fail "cd $EV"
gate dump_rep
env N1DM_TREE=F:/TanitAD-scratch/n1dc/read_n1dm/tree N1DM_CKPTS=$EV/ckpts N1DM_CKPTS_N1D=$N/ckpts N1DM_D_141=$N/dumps_rep N1DM_KAPPA=0.2 $PY -u n1dm_dumps.py N1DM-N1D_s0 >> $LOG 2>&1 || fail "dump N1DREP_s0"
grep -q "$sha" $N/dumps_rep/argv_N1DM-N1D_s0_m10_causal_heads.txt 2>/dev/null || grep -rq "${sha:0:16}" $N/dumps_rep/DUMPS_N1DM_*.json 2>/dev/null || say "note: the dump record does not quote the ckpt sha (checked by path instead)"
grep -q "$N/ckpts/N1D_s0/ckpt_last.pt" $N/dumps_rep/argv_N1DM-N1D_s0_m10_causal_heads.txt || fail "the replicate dump did not read the replicate checkpoint"
gate dump_redump
env N1DM_TREE=F:/TanitAD-scratch/n1dc/read_n1dm/tree N1DM_CKPTS=$EV/ckpts N1DM_CKPTS_N1D=F:/TanitAD-scratch/n1dc/read/eval141/ckpts N1DM_D_141=$N/dumps_redump N1DM_KAPPA=0.2 $PY -u n1dm_dumps.py N1DM-N1D_s0 >> $LOG 2>&1 || fail "K-REDUMP dump N1D_s0"
grep -q "F:/TanitAD-scratch/n1dc/read/eval141/ckpts/N1D_s0/ckpt_last.pt" $N/dumps_redump/argv_N1DM-N1D_s0_m10_causal_heads.txt || fail "the re-dump did not read the banked N1D_s0 checkpoint"
cd $N || fail "cd $N"
gate read
$PY n1drep_read.py --rep $N/dumps_rep/dump_N1DM-N1D_s0_m10_causal_heads --redump $N/dumps_redump/dump_N1DM-N1D_s0_m10_causal_heads --out $N/out/RESULT_N1DREP.json >> $LOG 2>&1
rc=$?
oc=$(grep "ZZ N1DREP READ" $LOG | tail -n 1 | sed 's/ZZ N1DREP READ //')
[ $rc -eq 0 ] || fail "read rc=$rc ($oc)"
echo "ZZ N1DREP-READ DONE ZZ $oc" >> $LOG
