#!/bin/bash
# e2read_launch_k2sg.sh <train_commit12> <env_blob_sha1> -- the ONE command that starts the refcv9 E2 read of K2SG on Thor the moment K2SG training is DONE.
#   Ship to /home/nvidia/r8ns/r9e2/inputs/ (the MM ships; the guards below read, and the launch writes only under $R/runs/E2_READ_K2SG, the new read tree and $R/logs).
#   <train_commit12>  the 12-char commit the K2SG chain segment trained from (CHAIN_STATUS.txt: `START commit <c> ... arms [K2SG]`, then ` K2SG-DONE `)
#   <env_blob_sha1>   git blob sha1 of the landed env file already on Thor as inputs/r9e2_read_env.thor.sh_<first 8>  (760d28d7... = stack/ops/r9e2_read/r9e2_read_env.thor.sh at the tip)
#   LAUNCH_DRY=1      run every guard, then `R9E2_DRY=1 r9e2_roll.sh` in the foreground (prints the commands, touches nothing); no tree build, no seeding, no launch.
# = e2read_launch_v4.sh's discipline with the K2SG differences (PLAN.md in the prep folder lists each, with file:line evidence):
#   (1) the arm is K2SG: run dir runs/r9e2-K2SG, OUT runs/E2_READ_K2SG (env: `source <env> K2SG`); the K2SG-DONE marker is read from the segment of <train_commit12>;
#   (2) the READ TREE is the K2SG launch tree itself: every fix the K2M1N20 read needed (e77c39d4 loader + dump, 55f14ef2 Thor GPU gate, 3624402e families dt, 2ab7f995 pose-sync lead block,
#       b25de778 the K2DC controller in the eval loader) is an ANCESTOR of the launch commit, so NO overlay is applied; built by r9e2_make_read_tree.sh, never rewritten;
#   (3) K0 is NOT re-rolled: K0's roll, perception dump, MAPSEL traces and EVAL are HARD-LINKED (cp -al, no disk) from the banked K2M1N20 read (E2_READ_K2M1N20), because nothing the K0
#       roll reads changed between that read's tree and this one (every diff on the K0 path is default-OFF: PLAN.md section 4); the roll then SKIPS those stages by itself;
#   (4) a fresh OUT is REQUIRED (this launcher only STARTS; a read is never overwritten).
# = e2read_launch_k2dc02.sh (tip 6828dd62) with EXACTLY these changes for K2SG (stack/tests/test_r9e2_read_k2sg.py pins that nothing else differs; K2SG = K2M1N20 + --map-hires-trunk-grad-scale 0.0, PREREG_K2SG_2026-11-19.md section 5):
#   (a) ARM=K2SG throughout: e2read_launch_k2dc02.sh:26 ARM, :33-35 the `arms \[K2DC02\]` segment match and ` K2DC02-DONE ` marker, :37 and :42 messages, run dir runs/r9e2-K2SG, OUT runs/E2_READ_K2SG; the K0 seed source default stays $R/runs/E2_READ_K2DC (:26, unchanged);
#   (b) the config.json guards of e2read_launch_k2dc02.sh:38-41 (carries --map-hires-share-target; target 0.02; bounds 0.1,25) become K2SG's: REFUSE if "--map-hires-share-target" appears (K2SG has no controller),
#       and REQUIRE the argv token "--map-hires-trunk-grad-scale" exactly once AND the pair "--map-hires-trunk-grad-scale","0.0" exactly once (counted on the whitespace-stripped config.json, the form the :39-41 case guards match);
#       literal source: stack/ops/runs.d/r9e2_K2SG.argv.json:230-231 ("--map-hires-trunk-grad-scale", "0.0"; the file has no --map-hires-share-target token);
#   (c) the read-tree guard e2read_launch_k2dc02.sh:63 (`controller_for` in the eval loader) becomes the K2SG replay: `_mgs.attach` in stack/tanitad/eval/refcv7_loader.py (tip :433 `tr._mgs.attach(model, args)`) and stack/tanitad/train/map_trunk_gs.py present;
#       without them G-DVB would refuse every K2SG argv. The "no overlay" ancestor list of :58 is unchanged (the K2DC controller commit b25de778 is still an ancestor); the tree name stays read_tree_${C}_k2dc (:27, a path, unchanged).
#   (d) MM 4a3d05, 2026-11-22: the trunk-grad-scale guard counts INSIDE the `argv` array (and inside `argv_effective` when that key exists), not over the whole
#       file: the 1f14f8d8 trainer writes the argv twice, and the same whole-file form REFUSED the valid K2DC02s1 read that day (fixed there in 8b581767).
#   Every other line is unchanged. (MM 2026-11-19: the roll re-pin to 38161299 and the R9E2_CKPT_NAME guard of c69b51a9 are carried over from e2read_launch_k2dc02.sh unchanged.)
set -u
C=${1:?train commit (12 chars)}; ENV_BLOB=${2:?landed env blob sha1}
R=/home/nvidia/r8ns/r9e2
ARM=K2SG; OLD=${K0_SEED_SRC:-$R/runs/E2_READ_K2DC}; OUT=$R/runs/E2_READ_$ARM; K0=$R/runs/r9e2-K0; K2=$R/runs/r9e2-$ARM
TREE=$R/read_tree_${C}_k2dc; ENVF=$R/inputs/r9e2_read_env.thor.sh_${ENV_BLOB:0:8}
BLK=$R/inputs/b1_eval_lead_block_k4_dt0p5_psync.npz
WRAP_BLOB=fcf2e5c342309c3ce76d2782619d5145b11ff29a; ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb
die() { echo "ZZE2READ-REFUSED $*"; exit 3; }
case "${R9E2_CKPT_NAME:-model_ema.pt}" in model_ema.pt) ;; *) die "R9E2_CKPT_NAME='${R9E2_CKPT_NAME}' is set: this read must take model_ema.pt like the K0 roll it reuses";; esac
b() { python3 -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$1" 2>/dev/null; }
[ ${#C} = 12 ] || die "train commit must be 12 chars, got '$C'"
SEG=$(awk -v c="$C" 'index($0," START commit " c " ") && /arms \[K2SG\]/ {f=1} f' $R/gate3/CHAIN_STATUS.txt)
[ -n "$SEG" ] || die "no 'START commit $C ... arms [K2SG]' in CHAIN_STATUS.txt"
echo "$SEG" | grep -q ' K2SG-DONE ' || die "no K2SG-DONE in the segment of commit $C"
for d in $K0 $K2; do [ -s $d/model_ema.pt ] && [ -s $d/config.json ] || die "$d lacks model_ema.pt or config.json"; done
[ ! -e $K2/ZEROGRAD_FAIL.json ] || die "K2SG has ZEROGRAD_FAIL.json"
grep -q -- '"--map-hires-share-target"' $K2/config.json && die "K2SG config.json carries --map-hires-share-target: that is a controller-arm argv, K2SG has no controller"
CFG1=$(tr -d '[:space:]' < $K2/config.json)
NF=$(( $(printf '%s' "$CFG1" | grep -o '"argv":\[[^]]*\]' | grep -o '"--map-hires-trunk-grad-scale"' | wc -l) )); NP=$(( $(printf '%s' "$CFG1" | grep -o '"argv":\[[^]]*\]' | grep -o '"--map-hires-trunk-grad-scale","0.0"' | wc -l) )); NEK=$(( $(printf '%s' "$CFG1" | grep -o '"argv_effective":' | wc -l) )); NEF=$(( $(printf '%s' "$CFG1" | grep -o '"argv_effective":\[[^]]*\]' | grep -o '"--map-hires-trunk-grad-scale"' | wc -l) )); NEP=$(( $(printf '%s' "$CFG1" | grep -o '"argv_effective":\[[^]]*\]' | grep -o '"--map-hires-trunk-grad-scale","0.0"' | wc -l) )); [ "$NF" = 1 ] && [ "$NP" = 1 ] && [ "$NEK" -le 1 ] && [ "$NEF" = "$NEK" ] && [ "$NEP" = "$NEK" ] || die "K2SG config.json must carry --map-hires-trunk-grad-scale 0.0 exactly once in argv, and once in argv_effective when that key exists (argv: flag x$NF, pair x$NP; argv_effective: key x$NEK, flag x$NEF, pair x$NEP)"
p=$(cat $K2/train.pid 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && die "K2SG trainer pid $p is still alive"
[ "$(b $ENVF)" = "$ENV_BLOB" ] || die "env $ENVF blob != $ENV_BLOB"
grep -q '^export R9E2_LEAD_BLOCK="$R/inputs/b1_eval_lead_block_k4_dt0p5_psync.npz"' $ENVF || die "env does not name the pose-synced lead block"
[ -s $BLK ] && [ -s $BLK.sha256 ] && [ "$(sha256sum $BLK | cut -c1-64)" = "$(cut -c1-64 $BLK.sha256)" ] || die "$BLK missing or its sha256 != $BLK.sha256"
# ---- the K0 that is reused: the banked K2M1N20 read must hold K0's four finished stages, and K0's checkpoint must not be newer than its roll ----
for d in $OLD $OLD/E2_K0_pf $OLD/E2_K0_mapsel $OLD/read $OLD/read/dump_K0; do [ -d $d ] || die "K0 seed source directory missing: $d (K0_SEED_SRC=$OLD)"; done
for m in 'ROLL-(DONE|SKIP) K0' 'DUMP-(DONE|SKIP) K0' 'MAPSEL-(DONE|SKIP) K0 seed 1' 'EVAL-(DONE|SKIP) K0'; do grep -qE "$m" $OLD/ROLL_STATUS.txt || die "$OLD/ROLL_STATUS.txt has no '$m': K0 is not reusable"; done
for f in $OLD/E2_K0_pf/r9e2_dev_filter.json $OLD/E2_K0_pf/dump_s0/manifest.json $OLD/E2_K0_pf/dump_s1/manifest.json $OLD/read/dump_K0/dump.json $OLD/read/EVAL_K0.json \
         $OLD/E2_K0_mapsel/trace_s0.npz $OLD/E2_K0_mapsel/trace_s1.npz; do [ -s $f ] || die "reusable K0 file missing: $f"; done
[ ! $K0/model_ema.pt -nt $OLD/E2_K0_pf/r9e2_dev_filter.json ] || die "K0's model_ema.pt is NEWER than the banked K0 roll: K0 changed, re-roll it"
[ ! -e $OUT ] || die "$OUT already exists (a read is never overwritten)"
# ---- the read tree: the K2SG launch tree, no overlay ----
if [ ! -d $TREE/stack ]; then
  LC=$(tr -d '[:space:]' < $R/launch_tree.commit); case "$LC" in $C*) ;; *) die "launch_tree.commit is $LC, not $C: extract $R/launch_tree_$C.tar first (never read K2SG with another commit's tree)";; esac
  [ "${LAUNCH_DRY:-0}" = 1 ] && echo "[dry] would build $TREE from $R/launch_tree + $R/inputs/kit_pkgs (r9e2_make_read_tree.sh)" || {
    bash $R/launch_tree/stack/ops/r9e2_read/r9e2_make_read_tree.sh $R/launch_tree $R/inputs/kit_pkgs $TREE || die "read tree build failed"
    echo "launch_tree $LC (K2SG trained from $C; no overlay: e77c39d4 55f14ef2 3624402e 2ab7f995 b25de778 are ancestors) -- built $(date -u +%Y-%m-%dT%H:%M:%SZ) by e2read_launch_k2sg.sh" > $TREE.provenance; }
fi
if [ -d $TREE/stack ]; then
  [ "$(b $TREE/stack/ops/r9e2_read/run_battery_r9_e2.py)" = $WRAP_BLOB ] || die "tree wrapper blob != $WRAP_BLOB"
  [ "$(b $TREE/stack/ops/r9e2_read/r9e2_roll.sh)" = $ROLL_BLOB ] || die "tree r9e2_roll.sh blob != $ROLL_BLOB"
  grep -q '_mgs.attach' $TREE/stack/tanitad/eval/refcv7_loader.py && [ -f $TREE/stack/tanitad/train/map_trunk_gs.py ] || die "tree lacks the K2SG map->trunk stop-gradient replay (_mgs.attach in the eval loader, train/map_trunk_gs.py): G-DVB would refuse every K2SG argv"
fi
if [ "${LAUNCH_DRY:-0}" = 1 ]; then
  [ -d $TREE/stack ] || { echo "ZZE2READ-DRY-GUARDS-OK (tree not built yet: the roll dry run needs it)"; exit 0; }
  R9E2_DRY=1 bash -c "export R9E2_TREE=$TREE; source $ENVF $ARM && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" | tail -n 5; exit 0
fi
# ---- seed K0 (hard links: same filesystem, no disk; the roll never writes into a stage it skips) ----
mkdir -p $OUT/read $R/logs || die "cannot create $OUT"
cp -al $OLD/E2_K0_pf $OUT/E2_K0_pf && cp -al $OLD/E2_K0_mapsel $OUT/E2_K0_mapsel && cp -al $OLD/read/dump_K0 $OUT/read/dump_K0 && cp -p $OLD/read/EVAL_K0.json $OUT/read/EVAL_K0.json \
  || die "seeding K0 from $OLD failed (cp -al needs one filesystem); $OUT left for inspection"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) K0-SEEDED from $OLD (E2_K0_pf, E2_K0_mapsel, read/dump_K0 hard-linked; read/EVAL_K0.json copied) by e2read_launch_k2sg.sh, K2SG commit $C" >> $OUT/ROLL_STATUS.txt
LOG=$R/logs/e2read_${ARM}_$(date -u +%Y%m%dT%H%M%SZ).log
setsid nohup bash -c "export R9E2_TREE=$TREE; source $ENVF $ARM && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" > $LOG 2>&1 < /dev/null &
sleep 5
echo "ZZE2READ-STARTED pid $! log $LOG"
head -n 5 $LOG | cut -c1-200
