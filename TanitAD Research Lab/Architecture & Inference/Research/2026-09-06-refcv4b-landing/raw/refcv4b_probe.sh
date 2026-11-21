#!/bin/bash
# refcv4b landing probe. Computes POD-SIDE and emits ONE opaque marker.
# The emitted token is DISJOINT from every searched token: the marker is
# ZZ<int>-<int>-<int>-<int>-<int>-<int>ZZ and this file's own text can never
# match that regex (it contains ${VAR} where the digits must be).
# Error patterns use character classes so the argv never contains the literal
# word it searches for.
set -uo pipefail
OUT=/workspace/experiments/refcv4b-b1-v72-40k

STEP=$( { cat "$OUT/metrics.jsonl" 2>/dev/null || true; } | tail -n 400 | python3 -c '
import sys, json
s = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        s = max(s, int(json.loads(line).get("step", 0)))
    except Exception:
        pass
print(s)
' 2>/dev/null | tail -n 1 )
case "$STEP" in ""|*[!0-9]*) STEP=0 ;; esac

SUM=0
[ -s "$OUT/summary.json" ] && SUM=1

# failure scan: stderr in full, train.log tail. Character classes keep the
# searched literals out of this process's own argv/command line.
PAT='T[r]aceback|CUDA out of m[e]mory|out of m[e]mory|K[i]lled|E[r]ror:|F[A]ILED|A[s]sertionError|Segmentation f[a]ult'
E1=$( { cat "$OUT/train.stderr.log" 2>/dev/null || true; } | grep -cE "$PAT" )
E2=$( { tail -n 4000 "$OUT/train.log" 2>/dev/null || true; } | grep -cE "$PAT" )
ERR=$(( ${E1:-0} + ${E2:-0} ))

# process census, pattern-broken so grep cannot match its own argv
SUP=$( ps -eo args= | grep -c 'sup[_]refcv4b' )
TRN=$( ps -eo args= | grep -c 'refc[_]v3[_]train' )
SUP=$(( SUP > 0 ? SUP : 0 ))
TRN=$(( TRN > 0 ? TRN : 0 ))

CKPT=$( ls -1 "$OUT"/ckpt_*.pt 2>/dev/null | wc -l )

printf 'ZZ%d-%d-%d-%d-%d-%dZZ\n' "$STEP" "$SUM" "$ERR" "$SUP" "$TRN" "$CKPT"
