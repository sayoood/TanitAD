#!/bin/bash
# after_ckpt120.sh -- STRAT1 CKPT120 arms H1_s0 / H1_s1 / H2_s0 / H2_s1 / H1_rep / R1_s0 / R2_s0 (pod commit 213250ce, branch agent/v8f-ckpt120-20261117).  Polls the pod log for the chain sentinel (script polling, sleep 300, no model turns), then:
# (1) bank the seven arms PRIVATE on HF from the pod (strat1_bank_pod_ckpt120.py, hub LFS sha256 checked against the manifest), (2) pull ckpt_last.pt + argv.json + BANK_MANIFEST.json to read/ckpts/<tag>/ and check the sha256 and the commit locally,
# (3) resolve K-REACH-e off the GPU (resolve_reach.py --controls first, then every arm; HEAD-DEAD arms are INVALID and are skipped by the read, INCONCLUSIVE stops the script), (4) launch the CKPT120 read detached (launch_wmi.ps1) unless SKIP_LAUNCH=1.
# Dev box, Git bash; the MM starts it.  Never touches the pod GPU jobs or /root/gpu_queue.lock (it only reads the log and files).
# Last line: "ZZ AFTER CKPT120 DONE ZZ" or "ZZ AFTER CKPT120 FAIL ZZ <why>".
D=F:/TanitAD-scratch/strat1/read/ckpt120
CK=$D/ckpts
CKU=/f/TanitAD-scratch/strat1/read/ckpt120/ckpts          # scp reads "F:/..." as host F:, so local scp paths use the /f/ form
DU=/f/TanitAD-scratch/strat1/read/ckpt120
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$D/out/after_ckpt120.log
ARMS="H1_s0 H1_s1 H2_s0 H2_s1 H1_rep R1_s0 R2_s0"
COMMIT=213250cee3c0a695c22f7ce823f4a7aabee6e311
mkdir -p "$D/out" "$D/reach"
FLT='post-quantum|store now|upgraded'
fail() { echo "ZZ AFTER CKPT120 FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
podq() { timeout "${2:-60}" bash -c "source C:/Users/Admin/lorapod/podenv.sh; pssh \"$1\"" 2>&1 | grep -vE "$FLT"; }
echo "[after_ckpt120] start $(date -u +%FT%TZ)" >> "$LOG"
for i in $(seq 1 576); do
  t=$(podq "tail -n 3 /root/strat1/logs/ckpt120.log")
  printf '%s\n' "$t" | grep -qE '^ZZ STRAT1 CKPT120 FAIL' && fail "the pod chain failed: $(printf '%s' "$t" | tail -n 1)"
  printf '%s\n' "$t" | grep -qE '^ZZ STRAT1 CKPT120 DONE' && break
  [ "$i" -eq 576 ] && fail "no CKPT120 sentinel after 48 h"
  sleep 300
done
echo "$(date -u +%T) CKPT120 DONE seen" >> "$LOG"
timeout 120 bash -c "source C:/Users/Admin/lorapod/podenv.sh; pscp '$DU/strat1_bank_pod_ckpt120.py' '$DU/bank_ckpt120.sh' \"\$POD_TARGET:/root/strat1/tmp/\"" >> "$LOG" 2>&1 || fail "copy of the bank scripts to the pod"
b=$(podq "bash /root/strat1/tmp/bank_ckpt120.sh" 2400)
echo "$b" >> "$LOG"
printf '%s\n' "$b" | grep -q '^rc=0' || fail "pod bank step (see /root/strat1/tmp/bank_ckpt120.log)"
for tag in $ARMS; do
  mkdir -p "$CK/$tag"
  for pair in "train/ckpt_last.pt:ckpt_last.pt" "argv.json:argv.json"; do
    src=${pair%%:*}; dst=${pair##*:}
    timeout 900 bash -c "source C:/Users/Admin/lorapod/podenv.sh; pscp \"\$POD_TARGET:/root/strat1/runs/$tag/$src\" '$CKU/$tag/$dst'" >> "$LOG" 2>&1 || fail "pull $tag/$src"
  done
  timeout 120 bash -c "source C:/Users/Admin/lorapod/podenv.sh; pscp \"\$POD_TARGET:/root/strat1/tmp/BANK_MANIFEST_$tag.json\" '$CKU/$tag/BANK_MANIFEST.json'" >> "$LOG" 2>&1 || fail "pull $tag manifest"
  $PY - "$CK/$tag" "$tag" "$COMMIT" >> "$LOG" 2>&1 <<'PYEOF' || fail "local sha256 / commit check $tag"
import hashlib, json, sys
d, tag, commit = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(d + "/BANK_MANIFEST.json", encoding="utf-8"))
assert m["tag"] == tag and m["commit"] == commit, (m["tag"], m["commit"])
for f in ("ckpt_last.pt", "argv.json"):
    h = hashlib.sha256(open(d + "/" + f, "rb").read()).hexdigest()
    assert h == m["files"][f]["sha256"], (f, h)
print(tag, "local sha256 == manifest", m["files"]["ckpt_last.pt"]["sha256"][:12])
PYEOF
done
# K-REACH-e off the GPU (CPU, torch): the controls on a banked checkpoint first, then every arm
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
$PY -u F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
(cd "$D" && $PY -u resolve_reach.py --controls) >> "$LOG" 2>&1 || fail "K-REACH-e controls (a trained head must read PASS, a planted dead head HEAD-DEAD): see reach/CONTROLS.json"
(cd "$D" && $PY -u resolve_reach.py $ARMS) >> "$LOG" 2>&1 || fail "K-REACH-e INCONCLUSIVE on an arm (see reach/<arm>.json)"
for tag in $ARMS; do
  v=$($PY -c "import json,sys; print(json.load(open('$D/reach/$tag.json',encoding='utf-8')).get('verdict'))")
  echo "K-REACH-e $tag $v" >> "$LOG"
done
if [ "${SKIP_LAUNCH:-0}" = "1" ]; then echo "ZZ AFTER CKPT120 DONE ZZ (launch skipped)" | tee -a "$LOG"; exit 0; fi
r=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\TanitAD-scratch\strat1\read\ckpt120\launch_wmi.ps1" 2>&1 | tr -d '\r')
echo "launch: $r" >> "$LOG"
printf '%s\n' "$r" | grep -q 'rv=0' || fail "WMI launch of the CKPT120 read: $r"
echo "ZZ AFTER CKPT120 DONE ZZ" | tee -a "$LOG"
