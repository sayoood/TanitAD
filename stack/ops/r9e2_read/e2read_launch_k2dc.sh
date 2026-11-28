#!/bin/bash
# e2read_launch_k2dc.sh <train_commit12> <env_blob_sha1> -- the ONE command that starts the refcv9 E2 read of K2DC on Thor the moment K2DC training is DONE.
#   Ship to /home/nvidia/r8ns/r9e2/inputs/ (the MM ships; the guards below read, and the launch writes only under $R/runs/E2_READ_K2DC, the new read tree and $R/logs).
#   <train_commit12>  the 12-char commit the K2DC chain segment trained from (CHAIN_STATUS.txt: `START commit <c> ... arms [K2DC]`, then ` K2DC-DONE `)
#   <env_blob_sha1>   git blob sha1 of the landed env file already on Thor as inputs/r9e2_read_env.thor.sh_<first 8>  (760d28d7... = stack/ops/r9e2_read/r9e2_read_env.thor.sh at the tip)
#   LAUNCH_DRY=1      run every guard, then `R9E2_DRY=1 r9e2_roll.sh` in the foreground (prints the commands, touches nothing); no tree build, no seeding, no launch.
# = e2read_launch_v4.sh's discipline with the K2DC differences (PLAN.md in the prep folder lists each, with file:line evidence):
#   (1) the arm is K2DC: run dir runs/r9e2-K2DC, OUT runs/E2_READ_K2DC (env: `source <env> K2DC`); the K2DC-DONE marker is read from the segment of <train_commit12>;
#   (2) the READ TREE is the K2DC launch tree itself: every fix the K2M1N20 read needed (e77c39d4 loader + dump, 55f14ef2 Thor GPU gate, 3624402e families dt, 2ab7f995 pose-sync lead block,
#       b25de778 the K2DC controller in the eval loader) is an ANCESTOR of the launch commit, so NO overlay is applied; built by r9e2_make_read_tree.sh, never rewritten;
#   (3) K0 is NOT re-rolled: K0's roll, perception dump, MAPSEL traces and EVAL are HARD-LINKED (cp -al, no disk) from the banked K2M1N20 read (E2_READ_K2M1N20), because nothing the K0
#       roll reads changed between that read's tree and this one (every diff on the K0 path is default-OFF: PLAN.md section 4); the roll then SKIPS those stages by itself;
#   (4) a fresh OUT is REQUIRED (this launcher only STARTS; a read is never overwritten).
set -u
C=${1:?train commit (12 chars)}; ENV_BLOB=${2:?landed env blob sha1}
R=/home/nvidia/r8ns/r9e2
ARM=K2DC; OLD=$R/runs/E2_READ_K2M1N20; OUT=$R/runs/E2_READ_$ARM; K0=$R/runs/r9e2-K0; K2=$R/runs/r9e2-$ARM
TREE=$R/read_tree_${C}_k2dc; ENVF=$R/inputs/r9e2_read_env.thor.sh_${ENV_BLOB:0:8}
BLK=$R/inputs/b1_eval_lead_block_k4_dt0p5_psync.npz
WRAP_BLOB=fcf2e5c342309c3ce76d2782619d5145b11ff29a; ROLL_BLOB=b4d141b9a540901fb44e3c3f644d11ff8e1535e5
die() { echo "ZZE2READ-REFUSED $*"; exit 3; }
b() { python3 -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$1" 2>/dev/null; }
[ ${#C} = 12 ] || die "train commit must be 12 chars, got '$C'"
SEG=$(awk -v c="$C" 'index($0," START commit " c " ") && /arms \[K2DC\]/ {f=1} f' $R/gate3/CHAIN_STATUS.txt)
[ -n "$SEG" ] || die "no 'START commit $C ... arms [K2DC]' in CHAIN_STATUS.txt"
echo "$SEG" | grep -q ' K2DC-DONE ' || die "no K2DC-DONE in the segment of commit $C"
for d in $K0 $K2; do [ -s $d/model_ema.pt ] && [ -s $d/config.json ] || die "$d lacks model_ema.pt or config.json"; done
[ ! -e $K2/ZEROGRAD_FAIL.json ] || die "K2DC has ZEROGRAD_FAIL.json"
grep -q -- '"--map-hires-share-target"' $K2/config.json || die "K2DC config.json does not carry --map-hires-share-target: not the K2DC argv"
p=$(cat $K2/train.pid 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && die "K2DC trainer pid $p is still alive"
[ "$(b $ENVF)" = "$ENV_BLOB" ] || die "env $ENVF blob != $ENV_BLOB"
grep -q '^export R9E2_LEAD_BLOCK="$R/inputs/b1_eval_lead_block_k4_dt0p5_psync.npz"' $ENVF || die "env does not name the pose-synced lead block"
[ -s $BLK ] && [ -s $BLK.sha256 ] && [ "$(sha256sum $BLK | cut -c1-64)" = "$(cut -c1-64 $BLK.sha256)" ] || die "$BLK missing or its sha256 != $BLK.sha256"
# ---- the K0 that is reused: the banked K2M1N20 read must hold K0's four finished stages, and K0's checkpoint must not be newer than its roll ----
for m in 'ROLL-DONE K0' 'DUMP-DONE K0' 'MAPSEL-DONE K0 seed 1' 'EVAL-DONE K0'; do grep -q "$m" $OLD/ROLL_STATUS.txt || die "$OLD/ROLL_STATUS.txt has no '$m': K0 is not reusable"; done
for f in $OLD/E2_K0_pf/r9e2_dev_filter.json $OLD/E2_K0_pf/dump_s0/manifest.json $OLD/E2_K0_pf/dump_s1/manifest.json $OLD/read/dump_K0/dump.json $OLD/read/EVAL_K0.json \
         $OLD/E2_K0_mapsel/trace_s0.npz $OLD/E2_K0_mapsel/trace_s1.npz; do [ -s $f ] || die "reusable K0 file missing: $f"; done
[ ! $K0/model_ema.pt -nt $OLD/E2_K0_pf/r9e2_dev_filter.json ] || die "K0's model_ema.pt is NEWER than the banked K0 roll: K0 changed, re-roll it"
[ ! -e $OUT ] || die "$OUT already exists (a read is never overwritten)"
# ---- the read tree: the K2DC launch tree, no overlay ----
if [ ! -d $TREE/stack ]; then
  LC=$(tr -d '[:space:]' < $R/launch_tree.commit); case "$LC" in $C*) ;; *) die "launch_tree.commit is $LC, not $C: extract $R/launch_tree_$C.tar first (never read K2DC with another commit's tree)";; esac
  [ "${LAUNCH_DRY:-0}" = 1 ] && echo "[dry] would build $TREE from $R/launch_tree + $R/inputs/kit_pkgs (r9e2_make_read_tree.sh)" || {
    bash $R/launch_tree/stack/ops/r9e2_read/r9e2_make_read_tree.sh $R/launch_tree $R/inputs/kit_pkgs $TREE || die "read tree build failed"
    echo "launch_tree $LC (K2DC trained from $C; no overlay: e77c39d4 55f14ef2 3624402e 2ab7f995 b25de778 are ancestors) -- built $(date -u +%Y-%m-%dT%H:%M:%SZ) by e2read_launch_k2dc.sh" > $TREE.provenance; }
fi
if [ -d $TREE/stack ]; then
  [ "$(b $TREE/stack/ops/r9e2_read/run_battery_r9_e2.py)" = $WRAP_BLOB ] || die "tree wrapper blob != $WRAP_BLOB"
  [ "$(b $TREE/stack/ops/r9e2_read/r9e2_roll.sh)" = $ROLL_BLOB ] || die "tree r9e2_roll.sh blob != $ROLL_BLOB"
  grep -q 'controller_for' $TREE/stack/tanitad/eval/refcv7_loader.py || die "tree eval loader has no K2DC controller (b25de778): G-DVB would refuse every K2DC argv"
fi
if [ "${LAUNCH_DRY:-0}" = 1 ]; then
  [ -d $TREE/stack ] || { echo "ZZE2READ-DRY-GUARDS-OK (tree not built yet: the roll dry run needs it)"; exit 0; }
  R9E2_DRY=1 bash -c "export R9E2_TREE=$TREE; source $ENVF $ARM && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" | tail -n 5; exit 0
fi
# ---- seed K0 (hard links: same filesystem, no disk; the roll never writes into a stage it skips) ----
mkdir -p $OUT/read $R/logs || die "cannot create $OUT"
cp -al $OLD/E2_K0_pf $OUT/E2_K0_pf && cp -al $OLD/E2_K0_mapsel $OUT/E2_K0_mapsel && cp -al $OLD/read/dump_K0 $OUT/read/dump_K0 && cp -p $OLD/read/EVAL_K0.json $OUT/read/EVAL_K0.json \
  || die "seeding K0 from $OLD failed (cp -al needs one filesystem); $OUT left for inspection"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) K0-SEEDED from $OLD (E2_K0_pf, E2_K0_mapsel, read/dump_K0 hard-linked; read/EVAL_K0.json copied) by e2read_launch_k2dc.sh, K2DC commit $C" >> $OUT/ROLL_STATUS.txt
LOG=$R/logs/e2read_${ARM}_$(date -u +%Y%m%dT%H%M%SZ).log
setsid nohup bash -c "export R9E2_TREE=$TREE; source $ENVF $ARM && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" > $LOG 2>&1 < /dev/null &
sleep 5
echo "ZZE2READ-STARTED pid $! log $LOG"
head -n 5 $LOG | cut -c1-200
