#!/bin/bash
# D-REFAV1-CCOS-EVAL — dev-box (RTX 4060) chain: chord @ shipped (deliberate regression) then
# ccos @ compensated weights (arm2_weights.txt), FULL 141-clip / 282-window grid, SAME flags as the
# banked Thor read except --cost-metric / --cost-weights. MEASURED 2026-09-05: first plan() 21.48 s
# here vs 44.76 s on Thor (which was sharing its GPU with the panel probe at the time).
# Cross-box note: the checkpoint bytes are identical (md5 1189bc020018c2c67ce03d566c390285, size
# 2,122,997,633); torch differs (2.11.0+cu128 here vs 2.13.0+cu130 on Thor), so bit-exactness across
# boxes is a measured quantity (the Thor chain's own chord/compensated arms), never assumed.
set -u
B=/c/Users/Admin/ccos_eval/devbox
WT="C:/Users/Admin/tanitad-wt"
export PYTHONPATH="C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/taniteval" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
COMMON=(--ckpt "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt" --config "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json"
  --cache "C:/Users/Admin/refav1_eval_full/fp8" --episodes "C:/Users/Admin/refav1_eval_full/eps"
  --labels "C:/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz"
  --nav "C:/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz"
  --lead-block "C:/Users/Admin/tanitad-wt/TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
  --device cuda --window-stride 40 --episodes-n 0 --no-navshuf)
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
md5=$(md5sum "C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt" | cut -d' ' -f1)
[ "$md5" = "1189bc020018c2c67ce03d566c390285" ] || { stamp "CKPT MD5 MISMATCH $md5"; exit 5; }
cd "$WT" || exit 1
run_arm() {
  local name=$1 metric=$2 w=$3; local extra=()
  [ -n "$w" ] && extra=(--cost-weights "$w")
  stamp "ARM $name start metric=$metric weights=${w:-shipped}"
  "$PY" taniteval/tools/refav1_arm.py "${COMMON[@]}" --cost-metric "$metric" "${extra[@]}" \
     --dump-dir "$B/dump_$name" --out "$B/rec_$name.json" --arm "refav1-21109-$name-devbox" > "$B/$name.log" 2>&1
  local rc=$?; stamp "ARM $name exit=$rc"; echo "$rc" > "$B/$name.EXIT"; return $rc
}
rm -f "$B/chain.DONE"; stamp "devbox chain start (pid $$)"
run_arm chord_shipped chord ""
W=$(tr -d ' \r\n' < /c/Users/Admin/ccos_eval/arm2_weights.txt)
run_arm ccos_comp ccos "$W"
stamp "devbox chain end"; touch "$B/chain.DONE"
