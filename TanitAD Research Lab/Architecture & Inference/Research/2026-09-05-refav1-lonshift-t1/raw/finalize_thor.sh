#!/usr/bin/env bash
# Land the Thor LON arms. Idempotent, zero GPU.
# It NAMES the arms whose record is absent rather than silently thinning the panel.
set -u
R=/home/nvidia/refav1_lon
P=$R/out
PY=/home/nvidia/venvs/tanitad-edge/bin/python
export PYTHONPATH=$R/code/stack:$R/code/taniteval
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4
export LON_OUT=$P
export LON_TOOLS=$R/code/taniteval/tools

BASE=T_wk15
ALL="T_wk15 T_lonshift T_lonshift_s1 T_lonvocab T_wk15_s1 T_lonseam T_loncomb"

# ---- 0. say what landed and what did not --------------------------------- #
{
  echo "# arm records present at $(date -u +%FT%TZ)"
  for a in $ALL; do
    if [ -s "$P/rec_$a.json" ]; then
      echo "  LANDED  $a  $(wc -c < "$P/rec_$a.json") bytes"
    else
      echo "  ABSENT  $a  (not run, still running, or FAILED -- check $P/$a.log)"
    fi
  done
} | tee "$P/lon_arms_present.txt"

if [ ! -s "$P/rec_$BASE.json" ]; then
  echo "ZZFATAL-BASELINE-ABSENT-${BASE}ZZ"; exit 1
fi

# ---- 1. the emitted longitudinal control, per arm ------------------------ #
# "beats the floor" must be conjoined with "acts": an arm that ties ha0_ext by
# emitting nothing has not driven.
PRESENT=""
for a in $ALL; do [ -s "$P/rec_$a.json" ] && PRESENT="$PRESENT $a"; done
"$PY" "$R/lon_emitted.py" $PRESENT > "$P/lon_emitted_all.txt" 2>&1
echo "-- emitted table: $(wc -l < "$P/lon_emitted_all.txt") lines"

# ---- 2. the four-family PAIRED deltas ------------------------------------ #
# Floors (ha / ha0 / ha0_ext) and the known-value control are added by the tool
# automatically, and it asserts the floors are bit-identical across dumps AND
# refuses outright if two arms are not on the same windows.
DUMPS=(--dump "$BASE=$P/dump_$BASE")
PAIRS=()
add_pair () {   # add "B-A" only if BOTH arms landed
  local b="${1%%-*}" x="${1##*-}"
  if [ -s "$P/rec_$b.json" ] && [ -s "$P/rec_$x.json" ]; then
    PAIRS+=(--pair "$1")
  else
    echo "SKIPPING pair $1 -- an arm is absent"
  fi
}
for a in $ALL; do
  [ "$a" = "$BASE" ] && continue
  if [ -d "$P/dump_$a/decisions" ] && [ -s "$P/rec_$a.json" ]; then
    DUMPS+=(--dump "$a=$P/dump_$a")
  else
    echo "SKIPPING $a in the paired panel -- record or dump absent"
  fi
done
add_pair "T_lonshift-T_wk15"          # D2 vs baseline -- THE HEADLINE
add_pair "T_lonvocab-T_wk15"          # D1 vs baseline -- attribution
add_pair "T_lonshift_s1-T_lonshift"   # SEED FLOOR on the lever arm (MANDATORY)
add_pair "T_wk15_s1-T_wk15"           # SEED FLOOR on the baseline
add_pair "T_lonshift-T_lonvocab"      # D2 vs D1 directly
add_pair "T_lonseam-T_wk15"           # the cost lever alone
add_pair "T_loncomb-T_wk15"           # D2 + seam

# 2a. WITHOUT the lead block -- directly comparable to the banked pd_lonbase panel
"$PY" "$R/code/taniteval/tools/refav1_paired_delta.py" \
  --stack "$R/code/stack" --taniteval "$R/code/taniteval" \
  "${DUMPS[@]}" "${PAIRS[@]}" \
  --out "$P/pd_thor.json" --md "$P/pd_thor.md" 2>&1 | tail -6

# 2b. WITH the lead block -- completes the LONGITUDINAL family (headway / time-gap
# / TTC). Zero GPU: distance-keeping is attached at ANALYSIS time from the dumps.
# If coverage is 0 the tool REFUSES per family with its reason and n, which is the
# admissible form -- never a silent drop.
if [ -s "$R/b1_eval_lead_block.npz" ]; then
  "$PY" "$R/code/taniteval/tools/refav1_paired_delta.py" \
    --stack "$R/code/stack" --taniteval "$R/code/taniteval" \
    --lead-block "$R/b1_eval_lead_block.npz" \
    "${DUMPS[@]}" "${PAIRS[@]}" \
    --out "$P/pd_thor_lead.json" --md "$P/pd_thor_lead.md" 2>&1 | tail -6
else
  echo "LEAD BLOCK ABSENT -- longitudinal distance-keeping not computed"
fi

# ---- 3. the vocabulary-expressivity table, re-read on each arm's dump ----- #
for a in $PRESENT; do
  if [ -d "$P/dump_$a/decisions" ]; then
    echo "== $a =="
    "$PY" "$R/lon_oracle.py" "$P/dump_$a" 2>&1 | grep -v Deprecation
  fi
done > "$P/lon_oracle_all.txt" 2>&1

# ---- 4. the attribution table (where the LON error lives) ---------------- #
for a in $PRESENT; do
  if [ -d "$P/dump_$a/decisions" ]; then
    echo "== $a =="
    "$PY" "$R/lon_attribution.py" "$P/dump_$a" 2>&1 | grep -v Deprecation
  fi
done > "$P/lon_attribution_all.txt" 2>&1

echo "ZZFINALIZE-THOR-DONE-$(date -u +%FT%TZ)ZZ"
