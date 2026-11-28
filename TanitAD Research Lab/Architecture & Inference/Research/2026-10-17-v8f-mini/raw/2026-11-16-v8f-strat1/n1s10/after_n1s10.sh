#!/bin/bash
# after_n1s10.sh -- N1 at seeds 10 / 11 (addendum PREREG_ADDENDUM_N1S10.md).  When the pod's N1S10 chain ends DONE: (1) bank N1_s10 / N1_s11 PRIVATE on HF from the pod
# (strat1_bank_pod_n1s10.py, hub LFS sha256 checked against the manifest), (2) pull ckpt_last.pt + argv.json + BANK_MANIFEST.json to read/ckpts/<tag>/ and check the sha256 locally,
# (3) launch the N1 read detached (launch_wmi.ps1) unless SKIP_LAUNCH=1.  Dev box, Git bash.  Never touches the pod's GPU jobs.
# Last line: "ZZ AFTER N1S10 DONE ZZ" or "ZZ AFTER N1S10 FAIL ZZ <why>".
D=F:/TanitAD-scratch/strat1/read/n1s10
CK=F:/TanitAD-scratch/strat1/read/ckpts
CKU=/f/TanitAD-scratch/strat1/read/ckpts          # scp reads "F:/..." as host F:, so local scp paths use the /f/ form
DU=/f/TanitAD-scratch/strat1/read/n1s10
LOG=$D/out/after_n1s10.log
mkdir -p "$D/out"
FLT='post-quantum|store now|upgraded'
fail() { echo "ZZ AFTER N1S10 FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
podq() { timeout "${2:-60}" bash -c "source C:/Users/Admin/lorapod/podenv.sh; pssh \"$1\"" 2>&1 | grep -vE "$FLT"; }
echo "[after_n1s10] start $(date -u +%FT%TZ)" >> "$LOG"
for i in $(seq 1 120); do
  t=$(podq "tail -n 3 /root/strat1/logs/n1s10.log")
  printf '%s\n' "$t" | grep -qE '^ZZ STRAT1 N1S10 FAIL' && fail "the pod chain failed: $(printf '%s' "$t" | tail -n 1)"
  printf '%s\n' "$t" | grep -qE '^ZZ STRAT1 N1S10 DONE' && break
  [ "$i" -eq 120 ] && fail "no N1S10 sentinel after 10 h"
  sleep 300
done
echo "$(date -u +%T) N1S10 DONE seen" >> "$LOG"
timeout 120 bash -c "source C:/Users/Admin/lorapod/podenv.sh; pscp '$DU/strat1_bank_pod_n1s10.py' '$DU/bank_n1s10.sh' \"\$POD_TARGET:/root/strat1/tmp/\"" >> "$LOG" 2>&1 || fail "copy of the bank scripts to the pod"
b=$(podq "bash /root/strat1/tmp/bank_n1s10.sh" 2400)
echo "$b" >> "$LOG"
printf '%s\n' "$b" | grep -q '^rc=0' || fail "pod bank step (see /root/strat1/tmp/bank_n1s10.log)"
for tag in N1_s10 N1_s11; do
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
assert m["tag"] == tag and m["commit"] == "a4357e7986f3aa30dab2ed83fde2351fda8d5662", (m["tag"], m["commit"])
for f in ("ckpt_last.pt", "argv.json"):
    h = hashlib.sha256(open(d + "/" + f, "rb").read()).hexdigest()
    assert h == m["files"][f]["sha256"], (f, h)
print(tag, "local sha256 == manifest", m["files"]["ckpt_last.pt"]["sha256"][:12])
PYEOF
done
if [ "${SKIP_LAUNCH:-0}" = "1" ]; then echo "ZZ AFTER N1S10 DONE ZZ (launch skipped)" | tee -a "$LOG"; exit 0; fi
r=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\TanitAD-scratch\strat1\read\n1s10\launch_wmi.ps1" 2>&1 | tr -d '\r')
echo "launch: $r" >> "$LOG"
printf '%s\n' "$r" | grep -q 'rv=0' || fail "WMI launch of the N1 read: $r"
echo "ZZ AFTER N1S10 DONE ZZ" | tee -a "$LOG"
