#!/bin/bash
# n1drep_chain.sh -- MM 2026-11-22, card PREREG_V8F_N1DREP (FROZEN, sha256 0c641197). PI go "Run both" (about 07:12Z).
# ONE arm: N1DREP_s0 = kind N1D, seed s0, in N1D's own tree /root/strat1/navrbd/tree at 0c3b374f, N1D's own G-SUITE result, runs dir /root/strat1/n1drep/runs
# (so its argv differs from N1D_s0's in --out only). Adapted from navrbd_chain.sh (which trained N1D_s0 in 672 s).
# Takes /root/gpu_queue.lock on fd 9 and HOLDS it to the end. Order: lock -> G-SUITE / tree / inputs checks -> render -> K-REP-CODE argv -> pair -> mint (GPU selfcheck + gate token)
# -> K-REP-CODE argv + token -> train -> post -> zero-grad gate (e_nav_rb.weight control) -> ckpt sha256 -> ZZ N1DREP DONE ZZ. Any failed check stops before training.
# Log: /root/strat1/n1drep/n1drep_chain.log (last line ZZ N1DREP DONE ZZ or ZZ N1DREP FAIL ZZ <why>).
# LAUNCH: nohup setsid bash /root/strat1/n1drep/n1drep_chain.sh < /dev/null > /dev/null 2>&1 &
R=/root/strat1; N0=$R/navrbd; N=$R/n1drep; TREE=$N0/tree; INP=$R/inputs; RUNS=$N/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"; ZG="$TREE/stack/scripts/zerograd_gate.py"
COMMIT=0c3b374fb1036b0ca4ca7620f98137dd6c59cca4
SUITE=$R/gsuite_result_navrbd.json
REF=$N0/runs/N1D_s0
kind=N1D; seed=s0; tag=${kind}_${seed}
LOG=$N/n1drep_chain.log
mkdir -p $N $RUNS $R/logs
exec >> $LOG 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
fail() { echo "ZZ N1DREP FAIL ZZ $1"; exit 1; }
say "chain start commit=$COMMIT card 0c641197"
[ -e $RUNS/$tag/ARM_DONE ] && [ -s $RUNS/$tag/train/ckpt_last.pt ] && { say "ARM_DONE and ckpt present: nothing to do"; echo "ZZ N1DREP DONE ZZ"; exit 0; }
[ -d $RUNS/$tag/train ] && fail "$RUNS/$tag/train exists from an earlier attempt: move it aside by hand"
exec 9>/root/gpu_queue.lock
say "waiting for the GPU queue lock"
flock 9 || fail "flock"
say "GPU queue lock held to the end of the chain"
[ -e $R/chains/STOP ] && fail "STOP file at $R/chains/STOP"
tail -c 600 $R/logs/navrbd_gsuite.log 2>/dev/null | grep -q "ZZ NAVRBD GSUITE DONE ZZ" || fail "the NAVRBD G-SUITE log does not end DONE"
[ "$(sha256sum $SUITE | cut -d' ' -f1)" = "20f0279e93597f09e1dc51ed6b861166519b71c62c49850f8e367c1d19b2294f" ] || fail "G-SUITE result is not the one N1D_s0's token binds"
(cd $TREE && sha256sum -c $N0/navrbd_tree.sha256) > $R/logs/n1drep_tree_verify.log 2>&1 || fail "navrbd tree sha256 (logs/n1drep_tree_verify.log)"
$PY $N0/blobcheck.py $TREE $N0/all_blobs.txt > $R/logs/n1drep_blobcheck.log 2>&1 || fail "navrbd tree blob check (logs/n1drep_blobcheck.log)"
# Amendment 1: the tree holds exactly one file outside all_blobs.txt, the eval-141 grid (placed 2026-11-18 21:02:08Z, before N1D_s0's mint at 21:05:37Z), at blob c489b553
GRID="$TREE/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json"
grep -q "eval_grid_windows" $N0/all_blobs.txt && fail "all_blobs.txt lists the grid: the +1 count below would be wrong"
[ "$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l)" = "$(( $(wc -l < $N0/all_blobs.txt) + 1 ))" ] || fail "navrbd tree file count (want all_blobs.txt + the eval-141 grid)"
[ "$(find "$TREE/TanitAD Research Lab" -type f | wc -l)" = 1 ] || fail "navrbd tree: more than the one Research Lab file"
[ "$($PY -c 'import hashlib,sys; b=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest())' "$GRID")" = "c489b553fe9355cf50e6007bdf395bd4a07a6e9a" ] || fail "eval-141 grid blob is not c489b553"
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify_n1drep.log 2>&1 || fail "inputs sha256"
(cd $INP/strat1 && sha256sum -c $R/navrb2/navrb2_inputs.sha256 && sha256sum -c $N0/navrbd_inputs.sha256) >> $R/logs/inputs_verify_n1drep.log 2>&1 || fail "nav inputs sha256"
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle (>= 3500 MiB used)"; sleep 10; done
mkdir -p $RUNS/$tag
$PY $N0/render_argv.py $kind $seed $RUNS/pre_argv_$tag.json $RUNS/$tag/train > $RUNS/pre_argv_$tag.log 2>&1 || fail "argv render: $(tail -c 300 $RUNS/pre_argv_$tag.log)"
$PY $N/n1drep_rep_check.py argv $RUNS/pre_argv_$tag.json $REF/argv.json $RUNS/$tag/train $REF/train > $RUNS/rep_argv_pre.log 2>&1 || fail "K-REP-CODE pre-mint argv: $(tail -c 300 $RUNS/rep_argv_pre.log)"
$PY $CHK pair $kind --argv-json $RUNS/pre_argv_$tag.json --ref-argv-json $R/navrb2/runs/N1R_$seed/argv.json > $RUNS/pre_pair_$tag.log 2>&1 || fail "pre-mint pairing: $(tail -c 300 $RUNS/pre_pair_$tag.log)"
if [ ! -f $RUNS/$tag/gate/gate_token.json ]; then
  say "mint $tag"
  $PY $CHK mint $kind $seed $COMMON --steps 3000 > $RUNS/mint_$tag.log 2>&1 || fail "mint refused: $(tail -c 300 $RUNS/mint_$tag.log)"
fi
$PY $N/n1drep_rep_check.py argv $RUNS/$tag/argv.json $REF/argv.json $RUNS/$tag/train $REF/train > $RUNS/rep_argv.log 2>&1 || fail "K-REP-CODE argv: $(tail -c 300 $RUNS/rep_argv.log)"
$PY $N/n1drep_rep_check.py token $RUNS/$tag/gate/gate_token.json $REF/gate/gate_token.json > $RUNS/rep_token.log 2>&1 || fail "K-REP-CODE token: $(tail -c 300 $RUNS/rep_token.log)"
$PY $CHK pair $kind --argv-json $RUNS/$tag/argv.json --ref-argv-json $R/navrb2/runs/N1R_$seed/argv.json > $RUNS/pair_$tag.log 2>&1 || fail "pairing: $(tail -c 300 $RUNS/pair_$tag.log)"
[ -e $R/chains/STOP ] && fail "STOP file before training"
say "run $tag"
$PY $CHK run $kind $seed $COMMON --steps 3000 > $RUNS/run_$tag.log 2>&1; rc=$?
[ $rc -eq 0 ] || fail "run rc=$rc: $(tail -c 300 $RUNS/run_$tag.log)"
$PY $CHK post $kind --init $INP/M2b_s0/ckpt_last.pt --final $RUNS/$tag/train/ckpt_last.pt > $RUNS/post_$tag.log 2>&1 || fail "post check: $(tail -c 300 $RUNS/post_$tag.log)"
PYTHONPATH=$TREE/stack:$TREE $PY $ZG --ckpt $RUNS/$tag/train/ckpt_last.pt --allow '.*' --control 'e_nav_rb\.weight' --json-out $RUNS/$tag/zerograd_navrbd.json > $RUNS/zerograd_$tag.log 2>&1 || fail "zerograd gate: $(tail -c 300 $RUNS/zerograd_$tag.log)"
(cd $RUNS && sha256sum $tag/train/ckpt_last.pt > $N/SHA256_N1DREP_CKPT.txt)
echo "ZZ ARM N1DREP_s0 DONE ZZ" > $RUNS/$tag/ARM_DONE
say "ckpt $(cut -c1-16 $N/SHA256_N1DREP_CKPT.txt)"
echo "ZZ N1DREP DONE ZZ"
