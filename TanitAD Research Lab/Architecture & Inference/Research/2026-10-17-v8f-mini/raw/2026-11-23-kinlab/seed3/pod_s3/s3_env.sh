#!/bin/bash
# s3_env.sh -- sourced by the SEED3 pod scripts (rh_s3_chain.sh, s3_arm.sh, chk_arm_s3.sh, s3_dry_mint.sh). Card PREREG_V8F_SEED3 (frozen, sha256 75f3b191).
# It sources the pod's rh_env.sh UNCHANGED (R, PY, DS, S1, readout functions, export_py), then defines the TWO code trees of the three arms:
#   B0_s2, RH_s2  -> $R/tree_cdb , commit 97fe0a56 (the commit B0_s0 / RH_s0 trained at, read from their gate records; the card's "cdbff971" is the G-SUITE BASE of that commit), G-SUITE $R/gsuite_result_rh.json
#   KI_s2         -> $R/tree_ki  , commit 4097a89e (as KI_s0 / KI_s1), G-SUITE $R/gsuite_result_ki.json
# s3_arm_env TAG sets TREE / COMMIT / BLOBS / GSUITE / GR (the gate-record script) / TREEDONE for the arm's kind; call it BEFORE export_py (PYTHONPATH reads $TREE).
_S3HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. ${RH_ROOT:-/root/rollhist}/rh_env.sh
RH_TREE_DIR=$R/tree_cdb;  RH_COMMIT=97fe0a56767320dbc5f4b823d67fb07a3209a745; RH_BLOBS=$R/all_blobs_97fe0a56.txt; RH_GSUITE=$R/gsuite_result_rh.json
KI_TREE_DIR=$R/tree_ki;   KI_COMMIT=4097a89e4b4a6c9bf309cbae9dc6a8dd1e574fcb; KI_BLOBS=$_S3HERE/all_blobs_ki.txt; KI_GSUITE=$R/gsuite_result_ki.json
for _c in $RH_COMMIT $KI_COMMIT; do [[ "$_c" =~ ^[0-9a-f]{40}$ ]] || { echo "s3_env.sh: commit '$_c' is not 40 lowercase hex"; exit 1; }; done
UE2=$DS/runs/UE_s2/A/ckpt_last.pt
export S3_DS=$DS
UE2_SHA=6d4345258e07dee15b1c5171268182967189df1cdc13b89b81d26aae71c3f327
UE2_SIZE=86970901
s3_arm_env() {
  case "$1" in
    B0_s2|RH_s2) TREE=$RH_TREE_DIR; COMMIT=$RH_COMMIT; BLOBS=$RH_BLOBS; GSUITE=$RH_GSUITE; GR=$_S3HERE/rh_gate_record_s3.py; TREEDONE=$R/done_s3/tree_rh.done;;
    KI_s2)       TREE=$KI_TREE_DIR; COMMIT=$KI_COMMIT; BLOBS=$KI_BLOBS; GSUITE=$KI_GSUITE; GR=$_S3HERE/ki_gate_record_s3.py; TREEDONE=$R/done_s3/tree_ki.done;;
    *) echo "s3_env.sh: unknown tag '$1' (B0_s2 | RH_s2 | KI_s2 only)"; return 1;;
  esac
  export TREE COMMIT BLOBS GSUITE GR TREEDONE
}
