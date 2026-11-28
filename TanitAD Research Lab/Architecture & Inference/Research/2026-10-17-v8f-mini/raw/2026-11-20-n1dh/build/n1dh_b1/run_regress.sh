#!/bin/bash
# usage: run_regress.sh <tree> <logdir> <tmpdir> file1 file2 ...  (one pytest at a time, CPU only, whole files; gated on free commit >= 5.5 GB, 60 s polls, 45 min cap -> SKIPPED-MEMORY)
TREE="$1"; LOGDIR="$2"; TMPD="$3"; shift 3
mkdir -p "$LOGDIR" "$TMPD"
export CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 PYTHONPATH="$(cygpath -w "$TREE/stack")"
cd "$TREE/stack" || exit 2
freegb() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" | tr -d '\r' | tr ',' '.'; }
for f in "$@"; do
  n=$(basename "$f" .py); ok=0
  for i in $(seq 1 45); do g=$(freegb); if awk "BEGIN{exit !($g >= 5.5)}"; then ok=1; break; fi; sleep 60; done
  if [ $ok = 0 ]; then echo "$n SKIPPED-MEMORY" >> "$LOGDIR/_summary.txt"; continue; fi
  rt="$TMPD/$n"; mkdir -p "$rt"; export TMP="$rt" TEMP="$rt"
  /c/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest "tests/$f" -q -p no:cacheprovider --basetemp="$rt/bt" -rfE --junitxml="$LOGDIR/$n.junit.xml" > "$LOGDIR/$n.log" 2>&1
  echo "$n rc=$? free_before=${g}GB" >> "$LOGDIR/_summary.txt"
  rm -rf "$rt"; echo "deleted $rt (pytest tmp of $n)" >> "$LOGDIR/_deleted.log"
done
echo ALLDONE >> "$LOGDIR/_summary.txt"
