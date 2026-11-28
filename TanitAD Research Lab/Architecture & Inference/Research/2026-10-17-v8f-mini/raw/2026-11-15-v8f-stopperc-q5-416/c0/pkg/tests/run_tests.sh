#!/usr/bin/env bash
# run_tests.sh -- dry run of q5c0_chain.sh against a FAKE box 2 (stubs under this folder; nothing real is reached).  Scratch: $T (F:), removed at the end if the shell allows.
export PATH=/usr/bin:/mingw64/bin:/bin:$PATH
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; CH="$(cygpath -m "$HERE/..")"
T="${Q5C0_TEST_ROOT:-F:/TanitAD-scratch/q5c0_test}"
PYV=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ok=0; bad=0
chk() { if [ "$2" = "$3" ]; then ok=$((ok+1)); echo "PASS $1"; else bad=$((bad+1)); echo "FAIL $1: got '$2' want '$3'"; fi; }
fresh() {   # fresh <name>: a clean fake world, exports the env of a dry run
  W="$T/$1"; rm -rf "${W:?}"; mkdir -p "$W/box/ctl" "$W/pred" "$W/aside" "$W/s0p_state" "$W/ref"
  echo "ZZ ACTSENS-DONE" > "$W/pred/DONE.txt"
  export Q5C0_DRY=1 Q5C0_STATE="$W/state" Q5C0_BASE="$W/base" Q5C0_PRED_STATE="$W/pred" Q5C0_BUSY_DEV="$W/BOX2_BUSY.txt" Q5C0_ASIDE_DEV="$W/aside" Q5C0_EVAL_NPZ="${Q5C0_REAL_EVAL:-F:/TanitAD-scratch/q5_416/run/c0/s0p_eval.npz}"
  export Q5C0_N_TOTAL=160 Q5C0_N_POS=40 Q5C0_SHARD_BYTES=1000 Q5C0_F_FLOOR_MB=1000 Q5C0_WAIT_POLL_S=1 Q5C0_WAIT_POLLS=3 S0P_POLL_S=0 Q5C0_S0P_STATE="$W/s0p_state" Q5C0_REF="$W/ref"
  export S0P_FAKE_BOX="$W/box" S0P_SSH="$(cygpath -m "$HERE/fake_ssh.sh")" S0P_SFTP="$(cygpath -m "$HERE/fake_sftp.sh")" S0P_TEST_PY=$PYV
  unset Q5C0_FAKE_GPU Q5C0_FAKE_PYCOUNT; echo 40 > "$W/box/ctl/npos"; echo 160 > "$W/box/ctl/ntotal"
  mkdir -p "$W/state" "$W/box/D/tanitad/repo/stack/scripts" "$W/box/D/tanitad/venv/Scripts" "$W/box/D/tanitad/runs/v8f_mini_full" "$W/box/D/tanitad/cache_full/train/clips" "$W/box/D/tanitad/cache_full/eval"
  for f in repo/stack/scripts/v8f_t1_eval.py venv/Scripts/python.exe runs/v8f_mini_full/join_train.npz cache_full/native_tracks.pt cache_full/eval/manifest.jsonl launch.ps1; do echo x > "$W/box/D/tanitad/$f"; done
}
runc() { bash "$CH/q5c0_chain.sh" > "$W/console.log" 2>&1; echo $?; }
lastline() { tail -n 1 "$W/state/chain.log"; }
mkrefs() {   # reference = what a banked S0-P run of THIS fake world would have left (shard meta with a different elapsed_s, verify.json from its manifest)
  for i in 0 1 2 3; do
    p="$W/base/_pull/S1/shard_$i"
    "$PYV" -I -c "
import sys,json,hashlib; sys.path.insert(0,'$CH'); import q5c0_tool as Q
m=json.load(open('$p/meta.json',encoding='utf-8')); m['elapsed_s']=161.4
open('$W/ref/sh$i.log','w',encoding='utf-8').write('===== x\npool {\"a\": 1}\n'+json.dumps(m,indent=1)+'\n===== end\n')
json.dump({'n':40,'manifest_sha256':Q.manifest_with_meta('$p',m)},open('$W/s0p_state/shard_$i.verify.json','w'))
"
  done
}

echo "== T6 test hook without Q5C0_DRY: refused rc 8"
fresh t6; unset Q5C0_DRY; rc=$(runc); chk "T6 rc" "$rc" 8; chk "T6 final line" "$(lastline | cut -c1-22)" "ZZ Q5C0 FAIL ZZ test-h"

echo "== T2 predecessor STOP: report and exit rc 12"
fresh t2; rm -f "$W/pred/DONE.txt"; echo "ZZ ACTSENS-STOP NULL_s3 box2-busy rc=12" > "$W/pred/STOP.txt"; rc=$(runc); chk "T2 rc" "$rc" 12; chk "T2 final" "$(lastline | cut -c1-22)" "ZZ Q5C0 FAIL ZZ actsen"; chk "T2 nothing launched" "$(ls "$W/box/launches.log" 2>/dev/null | wc -l)" 0

echo "== T2b no DONE yet: waits, times out rc 12, box untouched"
fresh t2b; rm -f "$W/pred/DONE.txt"; rc=$(runc); chk "T2b rc" "$rc" 12; chk "T2b no ssh" "$(ls "$W/box/ssh.log" 2>/dev/null | wc -l)" 0

echo "== T3 dev busy marker: waits, rc 12, marker not taken over"
fresh t3; echo "actsens chain 1 x" > "$W/BOX2_BUSY.txt"; rc=$(runc); chk "T3 rc" "$rc" 12; chk "T3 marker intact" "$(cat "$W/BOX2_BUSY.txt")" "actsens chain 1 x"

echo "== T3b box 2 own busy file / GPU busy / python running: rc 12"
fresh t3b; mkdir -p "$W/box/D/tanitad"; echo x > "$W/box/D/tanitad/BOX2_BUSY.txt"; rc=$(runc); chk "T3b busyfile rc" "$rc" 12; chk "T3b no launch" "$(ls "$W/box/launches.log" 2>/dev/null | wc -l)" 0
fresh t3c; export Q5C0_FAKE_GPU=6780; rc=$(runc); chk "T3c gpu rc" "$rc" 12; chk "T3c no launch" "$(ls "$W/box/launches.log" 2>/dev/null | wc -l)" 0
fresh t3d; export Q5C0_FAKE_PYCOUNT=1; rc=$(runc); chk "T3d python rc" "$rc" 12; chk "T3d no launch" "$(ls "$W/box/launches.log" 2>/dev/null | wc -l)" 0

echo "== T5 dry-run sample sha wrong: rc 4"
fresh t5; touch "$W/box/ctl/badsha_S2"; rc=$(runc); chk "T5 rc" "$rc" 4; chk "T5 busy released" "$(ls "$W/BOX2_BUSY.txt" 2>/dev/null | wc -l)" 0

echo "== T7 a shard job fails: rc 5, busy released, resume finishes without a second launch of finished shards"
fresh t7; touch "$W/box/ctl/fail_S1SH2"; rc=$(runc); chk "T7 rc" "$rc" 5; chk "T7 busy released" "$(ls "$W/BOX2_BUSY.txt" 2>/dev/null | wc -l)" 0
n1=$(grep -c "shard" "$W/box/launches.log"); export Q5C0_RESUME=1; rc=$(runc); n2=$(grep -c "shard" "$W/box/launches.log")
chk "T7 resume reaches C0 check (fails: no banked refs, fail closed) rc" "$rc" 10; chk "T7 shards launched on resume: S1 2,3 + S2 0..3 = 6" "$((n2 - n1))" 6
unset Q5C0_RESUME

echo "== T1 full chain: C0 fails closed without refs (rc 10), then refs from the same run, resume -> DONE"
fresh t1; rc=$(runc); chk "T1 no refs rc" "$rc" 10; chk "T1 final FAIL" "$(lastline | cut -c1-32)" "ZZ Q5C0 FAIL ZZ c0check c0-check"
chk "T1 result written FAIL" "$("$PYV" -I -c "import json;print(json.load(open('$W/base/C0_RESULT.json'))['overall'])")" "FAIL"
chk "T1 pulls kept for diagnosis" "$(ls -d "$W/base/_pull/S1/shard_3" 2>/dev/null | wc -l)" 1
chk "T1 box-2 shard copies deleted" "$(ls -d "$W/box/D/tanitad/stopperc/q5c0/out/S1/shard_0" 2>/dev/null | wc -l)" 0
mkrefs; export Q5C0_RESUME=1; rc=$(runc); chk "T1 resume rc" "$rc" 0; chk "T1 final DONE" "$(lastline)" "ZZ Q5C0 DONE ZZ"
chk "T1 C0 overall" "$("$PYV" -I -c "import json;d=json.load(open('$W/base/C0_RESULT.json'));print(d['overall'],d['literal_manifest_equal_all_shards'])")" "PASS False"
chk "T1 pulls deleted after PASS" "$(ls -d "$W/base/_pull" 2>/dev/null | wc -l)" 0
chk "T1 assembled y == sample y (S2)" "$("$PYV" -I -c "
import numpy as np,json; y=np.load('$W/base/S2/y.npy'); s=json.load(open('$W/base/S2/sample.json')); print(len(y), y.tolist()==s['y'], np.load('$W/base/S2/v40.npy',mmap_mode='r').shape)")" "160 True (160, 40, 1024)"
chk "T1 busy marker gone" "$(ls "$W/BOX2_BUSY.txt" 2>/dev/null | wc -l)" 0
chk "T1 box-2 files only under q5c0" "$(grep -c 'D/tanitad/stopperc/s0p' "$W/box/sftp.log")" 0
unset Q5C0_RESUME

echo "== T8 regression arm: one flipped byte in a pulled S1 shard array makes c0check FAIL"
fresh t8; rc=$(runc); mkrefs
"$PYV" -I -c "
import numpy as np; p='$W/base/_pull/S1/shard_1/E.npy'; a=np.load(p); a[0,0]+=np.float32(1e-6); np.save(p,a)"
"$PYV" -I "$CH/q5c0_tool.py" c0check --state "$W/state" --base "$W/base" --out "$W/c0_mut.json" > "$W/c0_mut.out" 2>&1; chk "T8 mutated rc 3" "$?" 3
chk "T8 names shard 1" "$("$PYV" -I -c "import json;d=json.load(open('$W/c0_mut.json'));print([k for k,v in d['checks'].items() if not v['pass']])")" "['S1_shard1_manifest']"

unset Q5C0_REF Q5C0_S0P_STATE Q5C0_N_TOTAL Q5C0_N_POS
echo "== T9 python helpers: ref_meta parses the four real S0-P shard logs; mkjob has no BS mangling"
chk "T9 ref_meta" "$("$PYV" -I -c "
import sys; sys.path.insert(0,'$CH'); import q5c0_tool as Q; print([round(Q.ref_meta(i)['elapsed_s'],1) for i in range(4)], Q.ref_meta(0)['seed'])")" "[161.4, 156.6, 160.7, 159.4] 20261111"
chk "T9 job file clean" "$("$PYV" -I -c "
import sys; sys.path.insert(0,'$CH'); import q5c0_tool as Q; j=Q.mkjob('shard','S2','g',3,4).encode(); print(b'\x08' not in j and b'\t' not in j and b'--seed 20261115' in j and b'--shard 3/4' in j and j.count(b'\r\n')==j.count(b'\n'))")" "True"
echo "RESULT ok=$ok failed=$bad"
