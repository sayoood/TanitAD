#!/bin/bash
# after_s1d.sh -- STRAT1 S1d at seeds 0 / 1 / 10 / 11 (S1 with --ctx-detach on; pod tree new9 = 4dedac45).  Polls the pod log for the chain sentinel (script polling, sleep 300, no model turns), then:
# (1) bank the four arms PRIVATE on HF from the pod (strat1_bank_pod_s1d.py, hub LFS sha256 checked against the manifest), (2) pull ckpt_last.pt + argv.json + BANK_MANIFEST.json to read/ckpts/<tag>/
# and check the sha256 locally, (3) launch the S1d read detached (launch_wmi.ps1) unless SKIP_LAUNCH=1.  Dev box, Git bash; the MM starts it.  Never touches the pod GPU jobs or its lock.
# Last line: "ZZ AFTER S1D DONE ZZ" or "ZZ AFTER S1D FAIL ZZ <why>".
D=F:/TanitAD-scratch/strat1/read/s1d
CK=F:/TanitAD-scratch/strat1/read/ckpts
CKU=/f/TanitAD-scratch/strat1/read/ckpts          # scp reads "F:/..." as host F:, so local scp paths use the /f/ form
DU=/f/TanitAD-scratch/strat1/read/s1d
LOG=$D/out/after_s1d.log
mkdir -p "$D/out"
FLT='post-quantum|store now|upgraded'
fail() { echo "ZZ AFTER S1D FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
podq() { timeout "${2:-60}" bash -c "source C:/Users/Admin/lorapod/podenv.sh; pssh \"$1\"" 2>&1 | grep -vE "$FLT"; }
echo "[after_s1d] start $(date -u +%FT%TZ)" >> "$LOG"
for i in $(seq 1 576); do
  t=$(podq "tail -n 3 /root/strat1/logs/s1d.log")
  printf '%s\n' "$t" | grep -qE '^ZZ STRAT1 S1D FAIL' && fail "the pod chain failed: $(printf '%s' "$t" | tail -n 1)"
  printf '%s\n' "$t" | grep -qE '^ZZ STRAT1 S1D DONE' && break
  [ "$i" -eq 576 ] && fail "no S1D sentinel after 48 h"
  sleep 300
done
echo "$(date -u +%T) S1D DONE seen" >> "$LOG"
timeout 120 bash -c "source C:/Users/Admin/lorapod/podenv.sh; pscp '$DU/strat1_bank_pod_s1d.py' '$DU/bank_s1d.sh' \"\$POD_TARGET:/root/strat1/tmp/\"" >> "$LOG" 2>&1 || fail "copy of the bank scripts to the pod"
b=$(podq "bash /root/strat1/tmp/bank_s1d.sh" 2400)
echo "$b" >> "$LOG"
printf '%s\n' "$b" | grep -q '^rc=0' || fail "pod bank step (see /root/strat1/tmp/bank_s1d.log)"
for tag in S1d_s0 S1d_s1 S1d_s10 S1d_s11; do
  mkdir -p "$CK/$tag"
  for pair in "train/ckpt_last.pt:ckpt_last.pt" "argv.json:argv.json"; do
    src=${pair%%:*}; dst=${pair##*:}
    timeout 900 bash -c "source C:/Users/Admin/lorapod/podenv.sh; pscp \"\$POD_TARGET:/root/strat1/runs/$tag/$src\" '$CKU/$tag/$dst'" >> "$LOG" 2>&1 || fail "pull $tag/$src"
  done
  timeout 120 bash -c "source C:/Users/Admin/lorapod/podenv.sh; pscp \"\$POD_TARGET:/root/strat1/tmp/BANK_MANIFEST_$tag.json\" '$CKU/$tag/BANK_MANIFEST.json'" >> "$LOG" 2>&1 || fail "pull $tag manifest"
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe - "$CK/$tag" "$tag" >> "$LOG" 2>&1 <<'PYEOF' || fail "local sha256 check $tag"
import hashlib, json, sys
d, tag = sys.argv[1], sys.argv[2]
m = json.load(open(d + "/BANK_MANIFEST.json", encoding="utf-8"))
assert m["tag"] == tag and m["commit"] == "4dedac45dc8d4f6922711ed9af63e0ab54e435be", (m["tag"], m["commit"])
for f in ("ckpt_last.pt", "argv.json"):
    h = hashlib.sha256(open(d + "/" + f, "rb").read()).hexdigest()
    assert h == m["files"][f]["sha256"], (f, h)
print(tag, "local sha256 == manifest", m["files"]["ckpt_last.pt"]["sha256"][:12])
PYEOF
done
if [ "${SKIP_LAUNCH:-0}" = "1" ]; then echo "ZZ AFTER S1D DONE ZZ (launch skipped)" | tee -a "$LOG"; exit 0; fi
r=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\TanitAD-scratch\strat1\read\s1d\launch_wmi.ps1" 2>&1 | tr -d '\r')
echo "launch: $r" >> "$LOG"
printf '%s\n' "$r" | grep -q 'rv=0' || fail "WMI launch of the S1d read: $r"
echo "ZZ AFTER S1D DONE ZZ" | tee -a "$LOG"
