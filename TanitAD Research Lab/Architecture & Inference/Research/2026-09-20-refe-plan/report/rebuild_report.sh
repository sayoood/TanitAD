#!/usr/bin/env bash
# Rebuild the REFe report page from fresh pod data (heartbeat step 3). Prints ZZREPORT_READY or fails loudly.
# Banked copy of the dev-box tool (the session scratchpad's rebuild_report.sh, 2026-09-27); S is that session's
# working directory for the page build, R this package's report/. Nothing is written on the pod: the logs are
# copied off it and code/op_status.py runs from stdin.
set -u
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/bd7d00af-b98e-42f1-a53c-cb4113059b0f/scratchpad/report"
R="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/report"
RD=/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=25 -i $HOME/.ssh/tanitad_pod -p <port> root@<redacted-ip>"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=25 -i $HOME/.ssh/tanitad_pod -P <port>"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cp "$R/template.html" "$R/build_data.py" "$S/" || exit 1
for f in metrics.jsonl train.log config.json; do
  $SCP "root@<redacted-ip>:$RD/$f" "$S/$f.new" && mv "$S/$f.new" "$S/$f" || { echo "fetch $f FAILED"; exit 1; }
done
$SCP root@<redacted-ip>:/workspace/data/refe_navtrain/train_grow/assembler.log "$S/assembler.log.new" && mv "$S/assembler.log.new" "$S/assembler.log"
# the status script is streamed from the package (code/op_status.py) so a change to it needs no pod-side write;
# no -n here: the script arrives on stdin
ssh -o BatchMode=yes -o ConnectTimeout=25 -i $HOME/.ssh/tanitad_pod -p <port> root@<redacted-ip> 'nice -n 19 python3 -' < "$R/../code/op_status.py" > "$S/onpolicy.json.new" && [ -s "$S/onpolicy.json.new" ] && mv "$S/onpolicy.json.new" "$S/onpolicy.json"
cd "$S" || exit 1
$PY build_data.py > build.out 2>&1 || { tail -20 build.out; echo "build_data FAILED"; exit 1; }
$PY inject.py > inject.out 2>&1 || { cat inject.out; echo "inject FAILED"; exit 1; }
E="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
MSYS_NO_PATHCONV=1 timeout 60 "$E" --headless=new --disable-gpu --no-first-run --virtual-time-budget=10000 \
  --dump-dom "file:///$S/refe-training-run.html" > "$S/dom_check.html" 2>/dev/null
PYTHONIOENCODING=utf-8 $PY - "$S/dom_check.html" <<'EOF'
import re, sys
d = open(sys.argv[1], encoding="utf-8", errors="replace").read()
bad = [c for c in ("c-selpdms", "c-selauc", "c-pdms", "c-ade", "c-speed")
       if not re.search(r'id="%s"[^>]*>\s*<svg' % c, d) and not re.search(r'id="%s"(.*?)<svg' % c, d[:len(d)], re.S)]
n_find = len(re.findall(r'<article class="finding">', d))
print("DOM:", len(d), "bytes; charts missing:", bad, "; findings:", n_find)
sys.exit(1 if bad or n_find < 5 else 0)
EOF
[ $? = 0 ] && echo ZZREPORT_READY || echo ZZREPORT_DOM_FAIL
