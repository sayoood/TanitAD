#!/bin/bash
# rbdpod.sh (RBDPOD, 2026-11-18): build tree_fix (5d474c91), G-SUITE it, train N1Dx_s0, dump it. Serial, one heavy job at a time.
# Launch (detached):  cd /root/strat1/navrbd; nohup setsid bash /root/strat1/navrbd/rbdpod/rbdpod.sh < /dev/null > /root/strat1/logs/rbdpod.log 2>&1 &
# DRY=1 : build tree_fix_dry and shard-compose the G-SUITE (no wait, no pytest, no training); ends with ZZ RBDPOD DRY DONE ZZ.
# Last line of /root/strat1/logs/rbdpod.log: ZZ RBDPOD DONE ZZ or ZZ RBDPOD FAIL ZZ <step and why>
R=/root/strat1; N=$R/navrbd; P=$N/rbdpod; PY=/root/venv/bin/python
SFX=${DRY:+_dry}
NT=$N/tree_fix$SFX
C1=5d474c9135298a40b3d50b56d611477d9fe97ca3
fail() { echo "ZZ RBDPOD FAIL ZZ $1"; exit 1; }
echo "$(date -u +%FT%TZ) rbdpod start DRY=${DRY:-0}"

# ---- step 1: wait for the held-out dumps of N1D_s0 / N1D_s1
if [ -z "$DRY" ]; then
  for i in $(seq 1 360); do tail -n1 $R/logs/navrbd_dumps2.log 2>/dev/null | grep -q "ZZ NAVRBD DUMPS2 END ZZ" && break; sleep 30; done
  tail -n1 $R/logs/navrbd_dumps2.log | grep -q "ZZ NAVRBD DUMPS2 END ZZ" || fail "step 1: dumps2 sentinel not seen in 3 h"
fi
[ -s $N/out/t1/DONE_N1D_s1 ] || [ -n "$DRY" ] || fail "step 1: out/t1/DONE_N1D_s1 is missing (the N1D_s1 dump did not finish)"
[ -e $N/out/t1/FAIL_N1D_s1 ] && [ -z "$DRY" ] && fail "step 1: out/t1/FAIL_N1D_s1 exists"
echo "$(date -u +%T) step 1 ok"

# ---- step 2: tree_fix = tree + the 5 rbdfix files
[ -d $N/tree ] || fail "step 2: no $N/tree"
[ ! -e "$NT" ] || fail "step 2: $NT already exists (refusing to overwrite)"
cp -a $N/tree "$NT" || fail "step 2: cp -a tree"
for f in $(awk '{print $2}' $P/TREE_FILES_SHA256.txt | tr -d '*'); do
  [ -s "$P/files/$f" ] || fail "step 2: staged file missing $P/files/$f"
  mkdir -p "$NT/$(dirname $f)"; cp "$P/files/$f" "$NT/$f" || fail "step 2: overlay $f"
done
(cd "$NT" && sha256sum -c $P/TREE_FILES_SHA256.txt) > $R/logs/rbdpod_tree_sha$SFX.log 2>&1 || fail "step 2: the 5 files do not match TREE_FILES_SHA256.txt (logs/rbdpod_tree_sha$SFX.log)"
n=$(grep -c ': OK$' $R/logs/rbdpod_tree_sha$SFX.log); [ "$n" = 5 ] || fail "step 2: $n of 5 files verified"
(cd "$NT" && sha256sum -c $N/navrbd_tree_fix.sha256) > $R/logs/rbdpod_tree_manifest$SFX.log 2>&1 || fail "step 2: navrbd_tree_fix.sha256 mismatch (logs/rbdpod_tree_manifest$SFX.log)"
D=$(diff -rq -x __pycache__ $N/tree "$NT")
nd=$(printf '%s\n' "$D" | grep -v '^$' | wc -l)
[ "$nd" = 5 ] || fail "step 2: tree_fix differs from tree in $nd entries, expected 5: $(printf '%s' "$D" | tr '\n' ';' | cut -c1-400)"
for f in $(awk '{print $2}' $P/TREE_FILES_SHA256.txt | tr -d '*'); do printf '%s\n' "$D" | grep -qF "$(basename $f)" || fail "step 2: $f not among the differing entries"; done
G=$(find "$NT" -name eval_grid_windows.json | head -5); [ "$(printf '%s\n' "$G" | grep -c .)" = 1 ] || fail "step 2: expected exactly one eval_grid_windows.json in tree_fix, found: $G"
gb=$($PY -c "import hashlib,sys; b=open(sys.argv[1],'rb').read(); print(hashlib.sha1(b'blob %d\0'%len(b)+b).hexdigest())" "$G")
case "$gb" in c489b553*) ;; *) fail "step 2: eval_grid_windows.json blob $gb is not c489b553";; esac
[ "$(sha256sum $N/tree/stack/scripts/v8f_t1_eval_rbd.py | cut -c1-64)" = "$(sha256sum $NT/stack/scripts/v8f_t1_eval_rbd.py | cut -c1-64)" ] || fail "step 2: the dump wrapper v8f_t1_eval_rbd.py differs between tree and tree_fix"
echo "$(date -u +%T) step 2 ok: 5 files by sha256, 5 differing entries, grid blob $gb, wrapper identical"

# ---- step 3: G-SUITE for 5d474c91
DRY=$DRY bash $N/navrbd_gsuite_fix.sh > $R/logs/navrbd_gsuite_fix.driver$SFX.log 2>&1
tail -n1 $R/logs/navrbd_gsuite_fix$SFX.log
if [ -n "$DRY" ]; then
  tail -n1 $R/logs/navrbd_gsuite_fix$SFX.log | grep -q "ZZ DRY OK ZZ" || fail "step 3 (dry): $(tail -n1 $R/logs/navrbd_gsuite_fix$SFX.log)"
  echo "ZZ RBDPOD DRY DONE ZZ"; exit 0
fi
tail -n1 $R/logs/navrbd_gsuite_fix.log | grep -q "ZZ NAVRBD GSUITE DONE ZZ" || fail "step 3: G-SUITE did not pass: $(tail -n1 $R/logs/navrbd_gsuite_fix.log | cut -c1-300)"
$PY -c "
import json,sys
r=json.load(open('$R/gsuite_result_navrbd_fix.json'))
ok = r['n_regressions']==0 and not r['new_test_failures'] and not r['missing_in_new'] and r['new_commit']=='$C1'
print('gsuite n_regressions',r['n_regressions'],'new_test_failures',len(r['new_test_failures']),'missing',len(r['missing_in_new']),'new_commit',r['new_commit'][:8])
sys.exit(0 if ok else 1)" || fail "step 3: gsuite_result_navrbd_fix.json is not a clean pass for $C1"
echo "$(date -u +%T) step 3 ok"

# ---- step 4: train N1Dx_s0 at 5d474c91
[ ! -e $N/runs/N1Dx_s0.attempt_refused ] || fail "step 4: runs/N1Dx_s0.attempt_refused already exists"
[ -d $N/runs/N1Dx_s0 ] && { mv $N/runs/N1Dx_s0 $N/runs/N1Dx_s0.attempt_refused || fail "step 4: mv of the refused attempt"; }
for f in mint_N1Dx_s0.log pre_argv_N1Dx_s0.json pre_argv_N1Dx_s0.log pre_pair_N1Dx_s0.log; do [ -e $N/runs/$f ] && cp -p $N/runs/$f $N/runs/N1Dx_s0.attempt_refused/; done
[ "$(grep -c $'\r' $N/navrbd_chain_fix.sh)" = 0 ] || fail "step 4: navrbd_chain_fix.sh has CR characters"
bash -n $N/navrbd_chain_fix.sh || fail "step 4: bash -n navrbd_chain_fix.sh"
cd $N || fail "cd $N"
flock /root/gpu_queue.lock bash $N/navrbd_chain_fix.sh < /dev/null > $R/logs/navrbd_fix.log 2>&1
tail -n1 $R/logs/navrbd_fix.log | grep -q "ZZ STRAT1 NAVRBD DONE ZZ" || fail "step 4: the chain did not end DONE: $(tail -n1 $R/logs/navrbd_fix.log | cut -c1-300)"
[ -e $N/runs/N1Dx_s0/ARM_DONE ] && [ -s $N/runs/N1Dx_s0/train/ckpt_last.pt ] || fail "step 4: no ARM_DONE or checkpoint for N1Dx_s0"
echo "$(date -u +%T) step 4 ok"

# ---- step 5: held-out dump of N1Dx_s0
bash $N/dump_rbd_fix.sh N1Dx_s0 follow 0
[ -s $N/out/t1/DONE_N1Dx_s0 ] || fail "step 5: dump N1Dx_s0: $(cat $N/out/t1/FAIL_N1Dx_s0 2>/dev/null | cut -c1-300)"
echo "$(date -u +%T) step 5 ok"
echo "ZZ RBDPOD DONE ZZ"
