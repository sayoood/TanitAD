#!/bin/bash
# MM 2026-11-13 (PREREG_VDAC_P6_A5 amendment A2, DRAFT): complete HC-s0 by hand after the chain's post-training gate
# failed on the 2 m parameter default (vdac_chain.sh:350 passes no --n-params; vdac_postgate.py N_PARAMS_CARD 2,440,450;
# HC's 1 m head has 3,423,490). Mirrors vdac_chain.sh:350-353 and :370-392 (the arm's egress) with the ONE change
# --n-params 3423490. Writes NO chain marker, no ZZSFT_DONE, no stage file. CPU + network only (P7 trains on the GPU).
set -u
source /workspace/teacher_env.sh
PY_C="${DRIVERL_EVAL_PYTHON:?}"; HPY=${HF_PY:-$PY_C}
RUN=/workspace/refe-plan-vdac/refe
ROOT=/workspace/data/refe_vdac
OUT=$ROOT/hc-s0
L=$ROOT/a2_hc_s0.log
REPO=Sayood/tanitad-refe-scorer1p6
EXPORT=$ROOT/export/h48_proposals_subscores.npz
TF=$(grep -o "/root/\.[A-Za-z0-9_.]*" /workspace/data/refe_scorer1p6/after_arms.sh | head -1)
log() { echo "$(date -u +%FT%T) $*" >> $L; }
[ -f "$TF" ] || { log "A2_REFUSE no token file"; exit 1; }
# preconditions: the chain stopped AT the post-training gate, after its run check passed
grep -q "ZZVDAC_RUN_OK" $OUT/run_check.log || { log "A2_REFUSE run_check.log has no ZZVDAC_RUN_OK"; exit 1; }
grep -q "ZZVDAC_CHAIN_FAIL HC-s0 (post-training gate)" $ROOT/chain_hc-s0.log || { log "A2_REFUSE the chain did not fail at the post-training gate"; exit 1; }
grep -q "^ZZVDAC_POSTGATE_FAIL" $OUT/postgate.log || { log "A2_REFUSE postgate.log is not a FAIL"; exit 1; }
[ -f $OUT/postgate_a2.log ] && { log "A2_REFUSE postgate_a2.log exists (one run only)"; exit 1; }
log "A2 start: chain post-gate FAIL line: $(tail -n 1 $OUT/postgate.log | cut -c1-300)"
# 1. the post-training gate with HC's built head count (gate_checks.AUX_HEAD_PARAMS_BY_CELL[1.0])
env -u HF_TOKEN "$PY_C" $RUN/vdac_postgate.py "$OUT" --updates 10000 --n-params 3423490 > "$OUT/postgate_a2.log" 2>&1
PGRC=$?
log "A2 post-training gate rc $PGRC: $(tail -n 1 $OUT/postgate_a2.log | cut -c1-300)"
[ $PGRC -eq 0 ] && grep -q "^ZZVDAC_POSTGATE_OK" $OUT/postgate_a2.log || { log "ZZA2_FAIL post-training gate with --n-params 3423490"; exit 1; }
# 2. pod sha256 of every file banked (before the push)
( cd $OUT && sha256sum aux_drv.pt head_pred_h48x.npz scores/head_pred_update10000.npz sft.log sft.jsonl preflight.log mutate_exclude.log run_check.log postgate.log postgate_a2.log model_sft_A.pt model_sft_B.pt scorer_states_last.pt ) > $OUT/a2_local_sha256.txt 2>> $L
sha256sum $EXPORT >> $OUT/a2_local_sha256.txt
# 3. the egress of the chain's file list (vdac_chain.sh:374) + the A2 gate log, the chain log and the sha list
N0=$(wc -l < $L)
"$HPY" $RUN/hf_egress.py push --repo $REPO --prefix "vdac/hc-s0" --max-push-bytes 6e9 --tag "arm_vdac_hc-s0" --chain-log $L --egress-log $ROOT/egress.jsonl --token-file "$TF" \
  --files $OUT/aux_drv.pt $OUT/head_pred_h48x.npz $OUT/scores/head_pred_update10000.npz $OUT/sft.log $OUT/sft.jsonl $OUT/preflight.log $OUT/mutate_exclude.log $ROOT/launch_gate_HC-s0.log $OUT/run_check.log $OUT/postgate.log \
          $OUT/postgate_a2.log $ROOT/chain_hc-s0.log $OUT/a2_local_sha256.txt >> $ROOT/a2_egress_hc_s0.log 2>&1
tail -n +$((N0 + 1)) $L | grep -q " ZZEGRESS_OK arm_vdac_hc-s0 " || { log "ZZA2_FAIL the egress push did not verify: HC-s0 lives on the pod only"; exit 1; }
# 4. hub LFS sha256 of the head and of the export (the export was banked by an earlier arm on this pod) vs the pod files
LOCSHA=$(sha256sum $OUT/aux_drv.pt | cut -c1-64); EXSHA=$(sha256sum $EXPORT | cut -c1-64)
env -u HF_TOKEN "$HPY" - "$REPO" "$TF" "$LOCSHA" "$EXSHA" "$OUT/a2_lfs_sha.json" >> $L 2>&1 <<'PYEOF'
import json, re, sys
from huggingface_hub import HfApi
repo, tokfile, loc, exloc, outp = sys.argv[1:6]
tok = re.search(r"hf_[A-Za-z0-9]+", open(tokfile, encoding="utf-8").read()).group(0)
paths = ["vdac/hc-s0/aux_drv.pt", "vdac/export/h48_proposals_subscores.npz"]
info = {p.path: p for p in HfApi(token=tok).get_paths_info(repo, paths, expand=True)}
def lsha(p):
    lfs = getattr(info.get(p), "lfs", None)
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
r = {"repo": repo, "head": {"hub": lsha(paths[0]), "pod": loc}, "export": {"hub": lsha(paths[1]), "pod": exloc}}
r["head"]["equal"] = r["head"]["hub"] == loc; r["export"]["equal"] = r["export"]["hub"] == exloc
json.dump(r, open(outp, "w"), indent=1)
print("ZZA2_LFS head equal %s export equal %s" % (r["head"]["equal"], r["export"]["equal"]))
sys.exit(0 if r["head"]["equal"] and r["export"]["equal"] else 1)
PYEOF
LRC=$?
[ $LRC -eq 0 ] && log "ZZA2_DONE HC-s0 complete (post-gate OK at 3,423,490, banked vdac/hc-s0, LFS sha equal)" || log "ZZA2_LFS_UNVERIFIED rc $LRC (see a2_lfs_sha.json)"
