#!/usr/bin/env bash
# MM 2026-11-15: C6a (Amendment 7a) over every eval clip. Per clip: pull the pod's 416 eval v2ep to a temp file, compare with E:'s copy (c6a_full.py one), delete the temp file.
# Resumable: a clip with a line in the jsonl is skipped. Ends with c6a_full.py summarise. Read-only on the pod.
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
source /c/Users/Admin/lorapod/podenv.sh
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
B=/f/TanitAD-scratch/q5_416/build
R=/f/TanitAD-scratch/q5_416/run/c6a
ED=E:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl
mkdir -p "$R/tmp"
OUT="$R/c6a_full.jsonl"; touch "$OUT"
export OMP_NUM_THREADS=2
n=0
for f in $(ls "$ED" | grep '\.v2ep\.pt$'); do
  c="${f%.v2ep.pt}"
  grep -q "\"clip\": \"$c\"" "$OUT" && continue
  t="$R/tmp/$c.v2ep.pt"
  pscp "$POD_TARGET:/root/q5_416/v2ep416/eval/$c.v2ep.pt" "$t" 2>&1 | grep -v -i -E 'post-quantum|store now|upgraded'
  if [ ! -s "$t" ]; then echo "C6A PULL-FAILED $c"; continue; fi
  "$PY" "$B/c6a_full.py" one --clip "$c" --pod-file "$t" --e-dir "$ED" --out "$OUT" || echo "C6A COMPARE-FAILED $c"
  rm -f "$t"
  n=$((n+1))
done
echo "compared $n clips this run"
"$PY" "$B/c6a_full.py" summarise --out "$OUT" --json "$R/RESULT_C6A.json"
echo "C6A END rc=$? $(date '+%F %T')"
