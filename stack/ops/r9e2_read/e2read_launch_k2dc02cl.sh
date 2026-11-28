#!/bin/bash
# e2read_launch_k2dc02cl.sh <train_commit12> <env_blob_sha1> -- the ONE command that starts the refcv9 E2 read of K2DC02cl on Thor the moment K2DC02cl training is DONE.
#   K2DC02cl = the CLAMP arm: K2DC02's flags (seed 0) with --w-map-hires 3.0 and --map-hires-w-bounds 2.5,3.5 (PREREG_K2DC02CL_2026-11-22.md; stack/ops/runs.d/r9e2_K2DC02cl.argv.json).
#   Ship to /home/nvidia/r8ns/r9e2/inputs/ (the MM ships; the guards below read, and the launch writes only under $R/runs/E2_READ_K2DC02cl, the new read tree and $R/logs).
#   <train_commit12>  the 12-char commit the K2DC02cl chain segment trained from (CHAIN_STATUS.txt: `START commit <c> ... arms [K2DC02cl]`, then ` K2DC02cl-DONE `)
#   <env_blob_sha1>   git blob sha1 of the landed env file already on Thor as inputs/r9e2_read_env.thor.sh_<first 8>  (d4bdcf07... = stack/ops/r9e2_read/r9e2_read_env.thor.sh at the tip ea503356 and at 67b5802a)
#   LAUNCH_DRY=1      run every guard, then `R9E2_DRY=1 r9e2_roll.sh` in the foreground (prints the commands, touches nothing); no tree build, no seeding, no launch.
# = e2read_launch_k2dc02.sh (tip ea503356) with EXACTLY these changes (stack/tests/test_r9e2_read_k2dc02cl.py pins that nothing else differs, compared after K2DC02cl -> K2DC02):
#   (a) ARM=K2DC02cl throughout: the `arms \[K2DC02cl\]` segment match and ` K2DC02cl-DONE ` marker, the messages, run dir runs/r9e2-K2DC02cl, OUT runs/E2_READ_K2DC02cl; the K0 seed source default stays $R/runs/E2_READ_K2DC;
#   (b) every blob pin is RE-CHECKED against the CURRENT tip, not copied: run_battery_r9_e2.py fcf2e5c3 and r9e2_roll.sh 38161299 are the same blobs at ea503356 and at 67b5802a (the test recomputes them
#       from the files); the R9E2_CKPT_NAME guard (model_ema.pt only, c69b51a9) is carried unchanged;
#   (c) the read tree is the K2DC02cl launch tree itself, no overlay (the K2DC02 argument: e77c39d4 55f14ef2 3624402e 2ab7f995 b25de778 are ancestors of every later commit);
#   (d') the bounds guard names the CLAMP (2.5,3.5) instead of K2DC02's 0.1,25 (K2DC02's own config would pass the old one); (g) THREE NEW config.json guards after it: the initial weight pair
#       `--w-map-hires`,`3.0`; `--seed` exactly once AND as the pair `--seed`,`0` (counted on the whitespace-stripped config.json); and `--r9-hist-supervise-geom` IS present (K2DC02's levers;
#       a config without it is K2DC02nh's). Source: r9e2_K2DC02cl.argv.json.
#   (h) MM 4a3d05, 2026-11-22: the seed guard counts INSIDE the `argv` array (and inside `argv_effective` when that key exists), not over the whole file.
#       The 1f14f8d8 trainer writes the argv twice, so the whole-file form REFUSED the valid K2DC02s1 read on 2026-11-22 (fixed there in 8b581767); K2DC02's
#       own config has the same two-key layout, so this arm's read would have been refused the same way.
#   Every other guard is unchanged. PI approval (2026-11-22): train K2DC02cl only on a DOSE-VOID K2DC02s1 read (r9e2_K2DC02cl.pi_cost_approval.json); this launcher starts nothing until that training exists.
set -u
C=${1:?train commit (12 chars)}; ENV_BLOB=${2:?landed env blob sha1}
R=/home/nvidia/r8ns/r9e2
ARM=K2DC02cl; OLD=${K0_SEED_SRC:-$R/runs/E2_READ_K2DC}; OUT=$R/runs/E2_READ_$ARM; K0=$R/runs/r9e2-K0; K2=$R/runs/r9e2-$ARM
TREE=$R/read_tree_${C}_k2dc; ENVF=$R/inputs/r9e2_read_env.thor.sh_${ENV_BLOB:0:8}
BLK=$R/inputs/b1_eval_lead_block_k4_dt0p5_psync.npz
WRAP_BLOB=fcf2e5c342309c3ce76d2782619d5145b11ff29a; ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb
die() { echo "ZZE2READ-REFUSED $*"; exit 3; }
case "${R9E2_CKPT_NAME:-model_ema.pt}" in model_ema.pt) ;; *) die "R9E2_CKPT_NAME='${R9E2_CKPT_NAME}' is set: this read must take model_ema.pt like the K0 roll it reuses";; esac
b() { python3 -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$1" 2>/dev/null; }
[ ${#C} = 12 ] || die "train commit must be 12 chars, got '$C'"
SEG=$(awk -v c="$C" 'index($0," START commit " c " ") && /arms \[K2DC02cl\]/ {f=1} f' $R/gate3/CHAIN_STATUS.txt)
[ -n "$SEG" ] || die "no 'START commit $C ... arms [K2DC02cl]' in CHAIN_STATUS.txt"
echo "$SEG" | grep -q ' K2DC02cl-DONE ' || die "no K2DC02cl-DONE in the segment of commit $C"
for d in $K0 $K2; do [ -s $d/model_ema.pt ] && [ -s $d/config.json ] || die "$d lacks model_ema.pt or config.json"; done
[ ! -e $K2/ZEROGRAD_FAIL.json ] || die "K2DC02cl has ZEROGRAD_FAIL.json"
grep -q -- '"--map-hires-share-target"' $K2/config.json || die "K2DC02cl config.json does not carry --map-hires-share-target: not the K2DC02cl argv"
CFG1=$(tr -d '[:space:]' < $K2/config.json)
case "$CFG1" in *'"--map-hires-share-target","0.02"'*) ;; *) die "K2DC02cl config.json does not carry --map-hires-share-target 0.02";; esac
case "$CFG1" in *'"--map-hires-w-bounds","2.5,3.5"'*) ;; *) die "K2DC02cl config.json does not carry --map-hires-w-bounds 2.5,3.5 (the clamp)";; esac
case "$CFG1" in *'"--w-map-hires","3.0"'*) ;; *) die "K2DC02cl config.json does not carry --w-map-hires 3.0 (the initial weight inside the clamp)";; esac
NS=$(printf '%s' "$CFG1" | grep -o '"argv":\[[^]]*\]' | grep -o '"--seed"' | wc -l); NV=$(printf '%s' "$CFG1" | grep -o '"argv":\[[^]]*\]' | grep -o '"--seed","0"' | wc -l); NEK=$(printf '%s' "$CFG1" | grep -o '"argv_effective":' | wc -l); NES=$(printf '%s' "$CFG1" | grep -o '"argv_effective":\[[^]]*\]' | grep -o '"--seed"' | wc -l); NEV=$(printf '%s' "$CFG1" | grep -o '"argv_effective":\[[^]]*\]' | grep -o '"--seed","0"' | wc -l); [ "$NS" = 1 ] && [ "$NV" = 1 ] && [ "$NEK" -le 1 ] && [ "$NES" = "$NEK" ] && [ "$NEV" = "$NEK" ] || die "K2DC02cl config.json must carry --seed 0 exactly once in argv, and once in argv_effective when that key exists (argv: flag x$NS, pair x$NV; argv_effective: key x$NEK, flag x$NES, pair x$NEV): not the K2DC02cl argv"
case "$CFG1" in *'"--r9-hist-supervise-geom"'*) ;; *) die "K2DC02cl config.json lacks --r9-hist-supervise-geom: K2DC02cl keeps K2DC02's levers (a config without it is K2DC02nh)";; esac
p=$(cat $K2/train.pid 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && die "K2DC02cl trainer pid $p is still alive"
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
# ---- the read tree: the K2DC02cl launch tree, no overlay ----
if [ ! -d $TREE/stack ]; then
  LC=$(tr -d '[:space:]' < $R/launch_tree.commit); case "$LC" in $C*) ;; *) die "launch_tree.commit is $LC, not $C: extract $R/launch_tree_$C.tar first (never read K2DC02cl with another commit's tree)";; esac
  [ "${LAUNCH_DRY:-0}" = 1 ] && echo "[dry] would build $TREE from $R/launch_tree + $R/inputs/kit_pkgs (r9e2_make_read_tree.sh)" || {
    bash $R/launch_tree/stack/ops/r9e2_read/r9e2_make_read_tree.sh $R/launch_tree $R/inputs/kit_pkgs $TREE || die "read tree build failed"
    echo "launch_tree $LC (K2DC02cl trained from $C; no overlay: e77c39d4 55f14ef2 3624402e 2ab7f995 b25de778 are ancestors) -- built $(date -u +%Y-%m-%dT%H:%M:%SZ) by e2read_launch_k2dc02cl.sh" > $TREE.provenance; }
fi
if [ -d $TREE/stack ]; then
  [ "$(b $TREE/stack/ops/r9e2_read/run_battery_r9_e2.py)" = $WRAP_BLOB ] || die "tree wrapper blob != $WRAP_BLOB"
  [ "$(b $TREE/stack/ops/r9e2_read/r9e2_roll.sh)" = $ROLL_BLOB ] || die "tree r9e2_roll.sh blob != $ROLL_BLOB"
  grep -q 'controller_for' $TREE/stack/tanitad/eval/refcv7_loader.py || die "tree eval loader has no K2DC02cl controller (b25de778): G-DVB would refuse every K2DC02cl argv"
fi
if [ "${LAUNCH_DRY:-0}" = 1 ]; then
  [ -d $TREE/stack ] || { echo "ZZE2READ-DRY-GUARDS-OK (tree not built yet: the roll dry run needs it)"; exit 0; }
  R9E2_DRY=1 bash -c "export R9E2_TREE=$TREE; source $ENVF $ARM && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" | tail -n 5; exit 0
fi
# ---- seed K0 (hard links: same filesystem, no disk; the roll never writes into a stage it skips) ----
mkdir -p $OUT/read $R/logs || die "cannot create $OUT"
cp -al $OLD/E2_K0_pf $OUT/E2_K0_pf && cp -al $OLD/E2_K0_mapsel $OUT/E2_K0_mapsel && cp -al $OLD/read/dump_K0 $OUT/read/dump_K0 && cp -p $OLD/read/EVAL_K0.json $OUT/read/EVAL_K0.json \
  || die "seeding K0 from $OLD failed (cp -al needs one filesystem); $OUT left for inspection"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) K0-SEEDED from $OLD (E2_K0_pf, E2_K0_mapsel, read/dump_K0 hard-linked; read/EVAL_K0.json copied) by e2read_launch_k2dc02cl.sh, K2DC02cl commit $C" >> $OUT/ROLL_STATUS.txt
LOG=$R/logs/e2read_${ARM}_$(date -u +%Y%m%dT%H%M%SZ).log
setsid nohup bash -c "export R9E2_TREE=$TREE; source $ENVF $ARM && exec bash $TREE/stack/ops/r9e2_read/r9e2_roll.sh" > $LOG 2>&1 < /dev/null &
sleep 5
echo "ZZE2READ-STARTED pid $! log $LOG"
head -n 5 $LOG | cut -c1-200
