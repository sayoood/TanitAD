#!/usr/bin/env bash
# Analyse whatever P4 arms have landed. Safe to re-run; skips missing dumps.
#
# ⭐ REUSES the predecessor's `paired_delta_refav1.py` rather than rebuilding a
# bootstrap: it already takes N dumps at once (so every pair is drawn from ONE
# set of components), pairs against ha / ha0 / ha0_ext, reports every family
# separately with `taniteval.ci.paired_episode_cluster_bootstrap`, emits
# STRATEGIC as UNAVAILABLE with its reason and n rather than omitting it, and
# prints a known-value control (an arm against ITSELF) that must read exactly
# 0.0000 with a zero-width interval.
#
# The pairs are chosen so each isolates ONE thing:
#   ccos - cos          the COST METRIC   (gate 2)
#   ccos_s1 - ccos      the SEED REPLICATE — H-ESTIM-SEED-1's noise floor, and
#                       the only thing against which the other two are readable
#   ccos_k002 - ccos    the VOCABULARY    (--goal-kappa-turn 0.02)
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_margin/p4out"
AN="C:/Users/Admin/refav1_margin/analysis"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f407bc82-7969-457c-a947-6be2014fee89/scratchpad"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$AN"

echo "############ GATE: executed curvature where the ROAD turns ############"
for name in cos_argmax ccos_argmax ccos_seed1 ccos_k002; do
  d="$OUT/dump_$name"
  if [ -d "$d" ] && [ -n "$(ls -A "$d"/ep*.npz 2>/dev/null)" ]; then
    "$PY" "$SP/gm_p4_turn_gate2.py" --dump "$d" --label "$name" \
      --episodes-json C:/Users/Admin/refav1_margin/p4_episodes.json \
      --gt-bank C:/Users/Admin/refav1_margin/intent_stride2.npz \
      --cache-dir C:/Users/Admin/refav1_margin/p4/fp8 \
      --out "$AN/gate_$name.json"
    echo
  else
    echo "[skip] $name — no dump yet"
  fi
done

echo "############ P5: four families, paired episode-cluster bootstrap ############"
DUMPS=()
PAIRS=()
[ -d "$OUT/dump_cos_argmax" ]  && DUMPS+=(--dump "cos=$OUT/dump_cos_argmax")
[ -d "$OUT/dump_ccos_argmax" ] && DUMPS+=(--dump "ccos=$OUT/dump_ccos_argmax")
[ -d "$OUT/dump_ccos_seed1" ]  && DUMPS+=(--dump "ccos_s1=$OUT/dump_ccos_seed1")
[ -d "$OUT/dump_ccos_k002" ]   && DUMPS+=(--dump "ccos_k002=$OUT/dump_ccos_k002")
[ -d "$OUT/dump_ccos_argmax" ] && [ -d "$OUT/dump_cos_argmax" ]  && PAIRS+=(--pair "ccos-cos")
[ -d "$OUT/dump_ccos_seed1" ]  && [ -d "$OUT/dump_ccos_argmax" ] && PAIRS+=(--pair "ccos_s1-ccos")
[ -d "$OUT/dump_ccos_k002" ]   && [ -d "$OUT/dump_ccos_argmax" ] && PAIRS+=(--pair "ccos_k002-ccos")
if [ ${#DUMPS[@]} -ge 4 ] && [ ${#PAIRS[@]} -ge 1 ]; then
  "$PY" "$SP/gm_paired_delta.py" "${DUMPS[@]}" "${PAIRS[@]}" \
    --arm cl --stack "$M/stack" --taniteval "$M/taniteval" \
    --out "$AN/paired_families.json" --md "$AN/PAIRED_FAMILIES.md"
else
  echo "[skip] fewer than two dumps present — nothing to pair"
fi
echo "ANALYSIS_DONE $(date -u +%FT%TZ)"
