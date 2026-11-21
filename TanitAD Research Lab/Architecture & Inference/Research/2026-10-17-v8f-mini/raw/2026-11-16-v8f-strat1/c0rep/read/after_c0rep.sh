#!/bin/bash
# after_c0rep.sh -- C0 same-seed replicates (addendum PREREG_ADDENDUM_C0REP.md).  The pod chain has ended DONE.  (1) bank C0_s10r / C0_s11r PRIVATE on HF from the pod under DISTINCT tags
# (strat1_bank_pod_c0rep.py: refuses if the tag prefix is not empty, checks the originals are unchanged, hub LFS sha256 vs the manifest), (2) pull ckpt_last.pt + argv.json + BANK_MANIFEST.json to
# read/ckpts/<tag>/ and check the sha256 locally, (3) launch the read detached (launch_wmi.ps1) unless SKIP_LAUNCH=1.  Dev box, Git bash.  Never touches the pod's GPU.
# Last line: "ZZ AFTER C0REP DONE ZZ" or "ZZ AFTER C0REP FAIL ZZ <why>".
D=F:/TanitAD-scratch/strat1/read/c0rep
CK=F:/TanitAD-scratch/strat1/read/ckpts
CKU=/f/TanitAD-scratch/strat1/read/ckpts          # scp reads "F:/..." as host F:, so local scp paths use the /f/ form
DU=/f/TanitAD-scratch/strat1/read/c0rep
LOG=$D/out/after_c0rep.log
mkdir -p "$D/out"
FLT='post-quantum|store now|upgraded'
fail() { echo "ZZ AFTER C0REP FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
podq() { timeout "${2:-60}" bash -c "source C:/Users/Admin/lorapod/podenv.sh; pssh \"$1\"" 2>&1 | grep -vE "$FLT"; }
echo "[after_c0rep] start $(date -u +%FT%TZ)" >> "$LOG"
t=$(podq "tail -n 3 /root/strat1/logs/c0rep.log")
printf '%s\n' "$t" | grep -qE '^ZZ STRAT1 C0REP DONE' || fail "no C0REP DONE sentinel on the pod"
for tag in C0_s10r C0_s11r; do [ -e "$CK/$tag" ] && fail "$CK/$tag already exists locally: refuse to overwrite"; done
timeout 120 bash -c "source C:/Users/Admin/lorapod/podenv.sh; pscp '$DU/strat1_bank_pod_c0rep.py' '$DU/bank_c0rep.sh' \"\$POD_TARGET:/root/strat1/tmp/\"" >> "$LOG" 2>&1 || fail "copy of the bank scripts to the pod"
b=$(podq "bash /root/strat1/tmp/bank_c0rep.sh" 2400)
echo "$b" >> "$LOG"
printf '%s\n' "$b" | grep -q '^rc=0' || fail "pod bank step (see /root/strat1/tmp/bank_c0rep.log)"
for pair in C0_s10:C0_s10r C0_s11:C0_s11r; do
  src=${pair%%:*}; tag=${pair##*:}
  mkdir -p "$CK/$tag"
  for fp in "train/ckpt_last.pt:ckpt_last.pt" "argv.json:argv.json"; do
    s=${fp%%:*}; dst=${fp##*:}
    timeout 900 bash -c "source C:/Users/Admin/lorapod/podenv.sh; pscp \"\$POD_TARGET:/root/strat1/runs_c0rep/$src/$s\" '$CKU/$tag/$dst'" >> "$LOG" 2>&1 || fail "pull $tag/$s"
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
if [ "${SKIP_LAUNCH:-0}" = "1" ]; then echo "ZZ AFTER C0REP DONE ZZ (launch skipped)" | tee -a "$LOG"; exit 0; fi
r=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\TanitAD-scratch\strat1\read\c0rep\launch_wmi.ps1" 2>&1 | tr -d '\r')
echo "launch: $r" >> "$LOG"
printf '%s\n' "$r" | grep -q 'rv=0' || fail "WMI launch of the C0REP read: $r"
echo "ZZ AFTER C0REP DONE ZZ" | tee -a "$LOG"
