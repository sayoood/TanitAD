#!/usr/bin/env bash
# k2dc02s1tail_build.sh -- Thor, CPU only. K2DC02s1-TAIL step 1 (PREREG_K2DC02S1_2026-11-20.md sec. 1 and 2.2). Zero training.
#   (1) IDENTITY CONTROL (sec. 2.2, exact): rebuild K2DC's tail from runs/r9e2-K2DC/{model_ema.pt, ckpt_ema_5000.pt} into a SCRATCH path with the same tool
#       and require BOTH (a) sha256 == sha256_out in E2-TAIL's banked K2DC.model_ema_tail5k6k.pt.provenance.json and (b) md5 == md5 of the live K2DC tail file
#       (the md5 every E2-TAIL dump recorded as ckpt_md5). Mismatch => ZZK2DC02S1TAILBUILD-ABORT-IDCTL-*ZZ and NOTHING is built. The scratch rebuild is deleted after the check.
#   (2) BUILD K2DC02s1's tail: W_tail = (EMA_6000 - 0.81872 * EMA_5000) / 0.18128 from runs/r9e2-K2DC02s1/{model_ema.pt, ckpt_ema_5000.pt} (the same ema_tail_weights.py,
#       the same coefficients, which the tool derives from decay 0.9998 and n = 1000). The output goes to a STAGING run dir
#       runs/E2TAIL_K2DC02s1/run_K2DC02s1/ (model_ema_tail5k6k.pt + provenance + a symlink to K2DC02s1's config.json): no file is written into any runs/r9e2-*/ dir.
# Markers: ZZK2DC02S1TAILBUILD-<EVENT>ZZ on stdout and in $W/BUILD_STATUS.txt. Exit 0 = built, 3 = abort.
# DRY (K2S_DRY=1 or first arg DRY): every guard runs, NOTHING is written (no status file, no mkdir, no scratch, no build); the commands are printed. Ends ...-DRY-OKZZ (rc 0) or the first ABORT (rc 3).
# Differences from k2dc02tail_build.sh (PREREG_K2DC02S1_2026-11-20.md sec. 2 items 1 and 3): the arm is K2DC02s1; the FIRST guard is that runs/r9e2-K2DC02s1 exists (it is not trained yet when this is written);
# the K2DC02s1 chain segment must carry ' K2DC02s1-DONE ' (as e2read_launch_k2dc02s1.sh checks); config.json carries --seed 1 exactly once and --r9-hist-supervise-geom (validity item 1);
# the read tree is read_tree_<K2S_TRAIN_COMMIT>_k2dc, the SAME tree the EMA read of K2DC02s1 uses (e2read_launch_k2dc02s1.sh builds it); the banked K2DC provenance for the identity control is READ from the
# K2DC02 tail's inputs/ (W0, never written). The identity control (A1, tensor identity) is kept as it was.
export PATH=/usr/bin:/bin:$PATH; set -u
DRY=0; if [ "${K2S_DRY:-0}" = 1 ] || [ "${1:-}" = DRY ]; then DRY=1; fi
R=/home/nvidia/r8ns/r9e2; PY=/home/nvidia/venvs/tanitad-train/bin/python
W0=$R/runs/E2TAIL_K2DC02      # the K2DC02 tail read (its inputs/ are read here, never written)
W=$R/runs/E2TAIL_K2DC02s1; SC=$W/scratch_idctl; ST=$W/BUILD_STATUS.txt; OUTD=$W/run_K2DC02s1
K2DC=$R/runs/r9e2-K2DC; K2DC02s1=$R/runs/r9e2-K2DC02s1
BANKED_PROV=$W0/inputs/K2DC.model_ema_tail5k6k.pt.provenance.json     # shipped from the repo: e2/e2raw/e2tail/stageA/tail_weights/ (sha256 pinned below)
BANKED_PROV_SHA=$(printf '%s' "${BANKED_PROV_SHA_PIN:-}")
ev() { [ $DRY = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$ST"; echo "ZZK2DC02S1TAILBUILD-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
b() { python3 -c 'import hashlib,sys; d=open(sys.argv[1],"rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "$1" 2>/dev/null; }
# ---- guard 1 (the FIRST): the trained arm exists. K2DC02s1 has not trained when this kit is written; nothing below may run without it ----
[ -s $K2DC02s1/model_ema.pt ] && [ -s $K2DC02s1/ckpt_ema_5000.pt ] && [ -s $K2DC02s1/config.json ] || die NO-RUN "$K2DC02s1 lacks model_ema.pt / ckpt_ema_5000.pt / config.json: K2DC02s1 has not trained"
TC=${K2S_TRAIN_COMMIT:-}; [ ${#TC} = 12 ] || die NO-TRAIN-COMMIT "K2S_TRAIN_COMMIT (the 12-char commit K2DC02s1 trained from) is required, got '$TC'"
SEG=$(awk -v c="$TC" 'index($0," START commit " c " ") && /arms \[K2DC02s1\]/ {f=1} f' $R/gate3/CHAIN_STATUS.txt)
echo "$SEG" | grep -q ' K2DC02s1-DONE ' || die NO-DONE "no 'START commit $TC ... arms [K2DC02s1]' + ' K2DC02s1-DONE ' in $R/gate3/CHAIN_STATUS.txt"
CFG1=$(tr -d '[:space:]' < $K2DC02s1/config.json)
NS=$(printf '%s' "$CFG1" | grep -o '"argv":\[[^]]*\]' | grep -o '"--seed"' | wc -l); NV=$(printf '%s' "$CFG1" | grep -o '"argv":\[[^]]*\]' | grep -o '"--seed","1"' | wc -l); NEK=$(printf '%s' "$CFG1" | grep -o '"argv_effective":' | wc -l); NES=$(printf '%s' "$CFG1" | grep -o '"argv_effective":\[[^]]*\]' | grep -o '"--seed"' | wc -l); NEV=$(printf '%s' "$CFG1" | grep -o '"argv_effective":\[[^]]*\]' | grep -o '"--seed","1"' | wc -l); [ "$NS" = 1 ] && [ "$NV" = 1 ] && [ "$NEK" -le 1 ] && [ "$NES" = "$NEK" ] && [ "$NEV" = "$NEK" ] || die SEED "config.json must carry --seed 1 exactly once in argv, and once in argv_effective when that key exists (argv: flag x$NS, pair x$NV; argv_effective: key x$NEK, flag x$NES, pair x$NEV): not the replicate argv"
case "$CFG1" in *'"--r9-hist-supervise-geom"'*) ;; *) die NO-HSG "config.json lacks --r9-hist-supervise-geom: the replicate keeps K2DC02's levers";; esac
p=$(cat $K2DC02s1/train.pid 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && die TRAINING "K2DC02s1 trainer pid $p is still alive"
RT=$R/read_tree_${TC}_k2dc    # the EMA read's tree (built by e2read_launch_k2dc02s1.sh): the tail read must use the same code
TOOL=$RT/stack/scripts/ema_tail_weights.py; TOOL_BLOB=e456ff67be132e9090f688dde4b8e9043ea43d5d    # the repo tip's blob of stack/scripts/ema_tail_weights.py
[ $DRY = 1 ] || mkdir -p $W
ev START "tool $TOOL tree $RT out $OUTD dry $DRY"
[ "$(b $TOOL)" = $TOOL_BLOB ] || die TOOL-BLOB "$TOOL blob $(b $TOOL) != $TOOL_BLOB"
[ -d $RT/stack ] || die NO-TREE "$RT: build the EMA read tree first (e2read_launch_k2dc02s1.sh)"
for f in $K2DC/model_ema.pt $K2DC/ckpt_ema_5000.pt $K2DC/model_ema_tail5k6k.pt $K2DC02s1/model_ema.pt $K2DC02s1/ckpt_ema_5000.pt $K2DC02s1/config.json $BANKED_PROV; do [ -s "$f" ] || die NO-INPUT "$f"; done
[ -z "${BANKED_PROV_SHA_PIN:-}" ] || [ "$(sha256sum $BANKED_PROV | cut -d' ' -f1)" = "$BANKED_PROV_SHA_PIN" ] || die PROV-SHA "banked provenance sha256 != pin"
FREE=$(df -BM --output=avail /home/nvidia | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -gt $((15360 + 1300)) ] || die DISK "free ${FREE} MB < 15360 floor + 1300 MB for the scratch rebuild and the K2DC02s1 tail"
[ ! -e $OUTD/model_ema_tail5k6k.pt ] || die EXISTS "$OUTD/model_ema_tail5k6k.pt already built (never overwritten)"
if [ $DRY = 1 ]; then
  echo "[dry] would: IDENTITY CONTROL $PY -X utf8 $TOOL --ema-a $K2DC/ckpt_ema_5000.pt --ema-b $K2DC/model_ema.pt --out $SC/K2DC_rebuild.pt  (then torch.equal on every tensor vs $K2DC/model_ema_tail5k6k.pt when IDCTL_TENSOR_EQUAL=1)"
  echo "[dry] would: BUILD $PY -X utf8 $TOOL --ema-a $K2DC02s1/ckpt_ema_5000.pt --ema-b $K2DC02s1/model_ema.pt --out $OUTD/model_ema_tail5k6k.pt ; provenance check n=1000 decay 0.9998 step_b 6000"
  ev DRY-OK "all guards pass; nothing built"; exit 0
fi
# ---- (1) identity control ----
rm -rf "${SC:?}" 2>/dev/null; mkdir -p $SC || die MKDIR "$SC"
$PY -X utf8 $TOOL --ema-a $K2DC/ckpt_ema_5000.pt --ema-b $K2DC/model_ema.pt --out $SC/K2DC_rebuild.pt > $SC/rebuild.log 2>&1 || die IDCTL-TOOL "rebuild failed (read $SC/rebuild.log)"
RB_SHA=$(sha256sum $SC/K2DC_rebuild.pt | cut -d' ' -f1); RB_MD5=$(md5sum $SC/K2DC_rebuild.pt | cut -d' ' -f1)
BK_SHA=$($PY -I -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["sha256_out"])' $BANKED_PROV)
LV_MD5=$(md5sum $K2DC/model_ema_tail5k6k.pt | cut -d' ' -f1)
ev IDCTL "rebuilt sha256 $RB_SHA md5 $RB_MD5 | banked provenance sha256_out $BK_SHA | live K2DC tail md5 $LV_MD5"
# IDCTL_TENSOR_EQUAL=1 (OFF by default; NOT part of the prereg -- it exists only if the MM amends PREREG sec. 2.2): file bytes are not reproducible (torch.save is not byte-deterministic here:
# MEASURED 2026-11-20, two rebuilds gave sha256 b8092eb0.. and 20328b81.., neither == the banked ff066691..), so the amended control is TENSOR identity: every tensor torch.equal, same keys/step/num_updates.
TENS="not-run"
if [ "${IDCTL_TENSOR_EQUAL:-0}" = 1 ]; then
  TENS=$($PY -I -X utf8 -c 'import torch,sys
a=torch.load(sys.argv[1],map_location="cpu",weights_only=False); b=torch.load(sys.argv[2],map_location="cpu",weights_only=False)
ma,mb=a["model"],b["model"]; ne=[k for k in ma if k not in mb or not torch.equal(ma[k],mb[k])]
ok=(sorted(a)==sorted(b) and list(ma)==list(mb) and not ne and all(a[k]==b[k] for k in a if k!="model"))
print(("TENSOR-EQUAL" if ok else "TENSOR-DIFF"), len(ma), "tensors", len(ne), "differ"); sys.exit(0 if ok else 1)' $SC/K2DC_rebuild.pt $K2DC/model_ema_tail5k6k.pt) || die IDCTL-TENSOR-DIFF "$TENS"
fi
SZ=$(du -sm $SC/K2DC_rebuild.pt | cut -f1); rm -f $SC/K2DC_rebuild.pt $SC/K2DC_rebuild.pt.provenance.json; ev DELETED "$SC/K2DC_rebuild.pt (${SZ} MB) + its provenance sidecar: scratch identity-control rebuild"
if [ "${IDCTL_TENSOR_EQUAL:-0}" = 1 ]; then ev IDCTL-PASS-TENSOR "AMENDED control (tensor identity, file bytes differ: rebuilt sha256 $RB_SHA md5 $RB_MD5 vs banked $BK_SHA / $LV_MD5): $TENS"; else
[ "$RB_SHA" = "$BK_SHA" ] || die IDCTL-SHA256-MISMATCH "rebuilt $RB_SHA != banked $BK_SHA"
[ "$RB_MD5" = "$LV_MD5" ] || die IDCTL-MD5-MISMATCH "rebuilt $RB_MD5 != live K2DC tail $LV_MD5"
ev IDCTL-PASS "K2DC tail rebuilt bit-for-bit: md5 $RB_MD5 sha256 $RB_SHA"; fi
# ---- (2) K2DC02s1's tail ----
mkdir -p $OUTD || die MKDIR "$OUTD"
$PY -X utf8 $TOOL --ema-a $K2DC02s1/ckpt_ema_5000.pt --ema-b $K2DC02s1/model_ema.pt --out $OUTD/model_ema_tail5k6k.pt > $OUTD/build.log 2>&1 || die BUILD "K2DC02s1 tail failed (read $OUTD/build.log)"
ln -sf $K2DC02s1/config.json $OUTD/config.json
$PY -I -c 'import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
ok=(p["num_updates_a"]==5000 and p["num_updates_b"]==6000 and p["n"]==1000 and abs(p["decay"]-0.9998)<1e-12 and p["step_b"]==6000 and abs(p["d_pow_n"]-0.8187143764431175)<1e-12)
print(("PROV-OK" if ok else "PROV-BAD"), "n_computed", p["n_computed"], "n_copied", p["n_copied"], "d_pow_n", p["d_pow_n"]); sys.exit(0 if ok else 1)' $OUTD/model_ema_tail5k6k.pt.provenance.json > $OUTD/prov_check.txt || die PROV "provenance not the E2-TAIL coefficients: $(cat $OUTD/prov_check.txt)"
ev BUILT "K2DC02s1 tail $OUTD/model_ema_tail5k6k.pt md5 $(md5sum $OUTD/model_ema_tail5k6k.pt | cut -d' ' -f1) sha256 $(sha256sum $OUTD/model_ema_tail5k6k.pt | cut -d' ' -f1) ($(cat $OUTD/prov_check.txt))"
ev DONE "build complete; stage A next"
