#!/bin/bash
# Everything that needs the finished refcv4b stride-1 roll, in priority order:
# a killed run still yields the earlier steps.
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f407bc82-7969-457c-a947-6be2014fee89/scratchpad/sel"
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=/c/Users/Admin/tanitad-selq-20260906
DUMP=$W/out/refcv4b_t1_s1_dump
ARM=$W/out/refcv4b_t1_s1.json
export OMP_NUM_THREADS=6
cd "$SP"

echo "########## 1. P1e  LC EMISSION + RECALL + DETERMINISM ##########"
"$PY" -X utf8 p1e_emission.py --dump "$DUMP" --arm-json "$ARM" \
  --replicate-dump "$W/out/smoke2_dump" --out out_p1e_emission.json \
  2>&1 | tee "$W/pkg/raw/p1e_emission.txt"

echo; echo "########## 2. P1g  LANE-CHANGE DEMAND (model-free) ##########"
"$PY" -X utf8 p1g_lc_demand.py --dump "$DUMP" --arm-json "$ARM" \
  --out out_p1g_lc_demand.json 2>&1 | tee "$W/pkg/raw/p1g_lc_demand.txt"

echo; echo "########## 3. P2 + P3  STABILITY AND REGRET ##########"
"$PY" -X utf8 p2p3_run.py --dump "$DUMP" --arm-json "$ARM" \
  --out out_p2p3.json --n-boot 2000 2>&1 | tee "$W/pkg/raw/p2p3.txt"
echo "AFTER_ROLL_EXIT=$?"

echo; echo "########## 4. P1f  LOGIT RANK of the lane-change classes (GPU, ~2 min) ##########"
"$PY" -X utf8 p1f_logit_rank.py \
  --ckpt /c/Users/Admin/refcv4b_final/ckpt_40284_FINAL.pt \
  --config /c/Users/Admin/refcv4b_final/config.json \
  --episodes /c/Users/Admin/refav1_eval_full/eps \
  --labels /c/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz \
  --grid 2s --arm refcv4b-logitrank --episodes-n 5 --window-stride 1 \
  --out "$W/out/logitrank.json" --dump-dir "$W/out/logitrank_dump" \
  --device cuda --lru 6 --n-boot 50 --seed 0 \
  --no-navshuf --no-navzero --no-lead-block \
  2>&1 | tee "$W/pkg/raw/p1f_logit_rank.txt"
echo "P1F_EXIT=$?"
