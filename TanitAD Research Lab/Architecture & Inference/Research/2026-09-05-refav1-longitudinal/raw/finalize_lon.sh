#!/usr/bin/env bash
# Land the longitudinal arms without re-deriving anything. Idempotent, zero GPU.
# It NAMES the arms whose record is absent rather than silently thinning the
# panel -- the cost-geometry package's finalize.sh rule, applied here.
#
# Run:  bash <scratchpad>/finalize_lon.sh
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f407bc82-7969-457c-a947-6be2014fee89/scratchpad"
P="C:/Users/Admin/refav1_margin/p4out"
M="C:/Users/Admin/tanitad-wt"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PKG="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-longitudinal"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4

BASE="wk15"
NEW="lonvocab lonseam lonvocab_s1 loncomb"

# ---- 0. say what landed and what did not --------------------------------- #
{
  echo "# arm records present at $(date -u +%FT%TZ)"
  for a in $BASE wk151 $NEW; do
    if [ -s "$P/rec_$a.json" ]; then
      echo "  LANDED  $a  $(wc -c < "$P/rec_$a.json") bytes"
    else
      echo "  ABSENT  $a  (not run, still running, or FAILED -- check $P/$a.log)"
    fi
  done
} | tee "$SP/lon_arms_present.txt"

# ---- 1. the emitted longitudinal control, per arm ------------------------ #
"$PY" "$SP/lon_emitted.py" $BASE wk151 $NEW > "$SP/lon_emitted_all.txt" 2>&1

# ---- 2. the four-family PAIRED deltas ------------------------------------ #
# every new arm against the SAME named baseline `wk15`, plus both floors, plus
# the arm-against-itself known-value control that MUST read +0.0000 [0, 0].
DUMPS=(--dump "wk15=$P/dump_wk15" --dump "wk151=$P/dump_wk151")
PAIRS=(--pair "wk15-wk15" --pair "wk151-wk15")
for a in $NEW; do
  if [ -d "$P/dump_$a/decisions" ] && [ -s "$P/rec_$a.json" ]; then
    DUMPS+=(--dump "$a=$P/dump_$a")
    PAIRS+=(--pair "$a-wk15")
  else
    echo "SKIPPING $a in the paired panel -- record or dump absent"
  fi
done
"$PY" "$M/taniteval/tools/refav1_paired_delta.py" \
  --stack "$M/stack" --taniteval "$M/taniteval" \
  "${DUMPS[@]}" "${PAIRS[@]}" \
  --out "$SP/pd_lon.json" --md "$SP/pd_lon.md" 2>&1 | tail -5

# ---- 3. the vocabulary-expressivity table, re-read on each new arm's dump - #
for a in $BASE $NEW; do
  if [ -d "$P/dump_$a/decisions" ]; then
    echo "== $a =="
    "$PY" "$SP/lon_oracle.py" "$P/dump_$a" 2>&1 | grep -v Deprecation
  fi
done > "$SP/lon_oracle_all.txt" 2>&1

# ---- 4. bank into the repo package --------------------------------------- #
mkdir -p "$PKG/raw/arms" 2>/dev/null
for f in lon_arms_present.txt lon_emitted_all.txt pd_lon.md pd_lon.json \
         lon_oracle_all.txt finalize_lon.sh; do
  for i in 1 2 3 4 5 6 7 8; do
    cp "$SP/$f" "$PKG/raw/$f" 2>/dev/null && break
    sleep 4
  done
  echo "banked $f -> $(wc -c < "$PKG/raw/$f" 2>/dev/null) bytes"
done
# the arm records + decision sidecars are single-copy off-repo until this runs
for a in $NEW; do
  [ -s "$P/rec_$a.json" ] || continue
  for i in 1 2 3 4 5 6; do cp "$P/rec_$a.json" "$PKG/raw/arms/rec_$a.json" 2>/dev/null && break; sleep 4; done
  mkdir -p "$PKG/raw/arms/dump_$a/decisions" 2>/dev/null
  for f in "$P/dump_$a"/*.npz "$P/dump_$a"/manifest.json "$P/dump_$a"/decisions/*.npz; do
    [ -e "$f" ] || continue
    d="$PKG/raw/arms/dump_$a/${f#$P/dump_$a/}"
    for i in 1 2 3 4 5 6; do cp "$f" "$d" 2>/dev/null && break; sleep 3; done
  done
  echo "banked arm $a"
done
echo "ZZFINALIZE-LON-DONE-$(date -u +%FT%TZ)ZZ"
