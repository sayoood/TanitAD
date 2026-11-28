#!/usr/bin/env bash
# E-ARCH-TSC-2 — final banking: compute reads.json, copy A' raw + reads + RESULT + tools into the
# research package on G:, verify every copy by md5 against its dev-box source.
set -u
cd /c/Users/Admin/tsc2 || exit 9
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
PKG="/g/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-02-refav1-ema-inflation"
# A' was killed by the harness at ~step 226 (no row past 220): read R4 at 220, stated in reads.json
"$PY" tools/reads.py --bp Bp --ap Ap --ap-final 220 --r7 R7_resume --r7-log R7_resume.log --final 250 --out reads.json || exit 8
mkdir -p "$PKG/raw/A_prime" "$PKG/raw/tools"
cpv() {  # copy with retry + md5 read-back (G: flaps)
  local src="$1" dst="$2" i
  for i in 1 2 3 4 5; do
    cp "$src" "$dst" 2>/dev/null
    if [ -f "$dst" ] && [ "$(md5sum < "$src")" = "$(md5sum < "$dst")" ]; then echo "ok   $dst"; return 0; fi
    sleep 3
  done
  echo "FAIL $dst"; return 1
}
rc=0
cpv Ap/train_log.jsonl "$PKG/raw/A_prime/train_log.jsonl" || rc=1
cpv Ap/config.json "$PKG/raw/A_prime/config.json" || rc=1
cpv Ap.log "$PKG/raw/A_prime.log" || rc=1
cpv reads.json "$PKG/raw/reads.json" || rc=1
cpv RESULT.md "$PKG/RESULT.md" || rc=1
for t in run_arm.py inspect_and_build_cache.py reads.py run_r7.sh wait_step.py finalize_copy.sh; do
  cpv tools/$t "$PKG/raw/tools/$t" || rc=1
done
# re-verify the earlier copies too (B', R7, fit probe, cache index)
for p in B_prime/train_log.jsonl B_prime/config.json; do cpv "Bp/$(basename $p)" "$PKG/raw/$p" || rc=1; done
cpv Bp.log "$PKG/raw/B_prime.log" || rc=1
cpv R7_resume.log "$PKG/raw/R7_resume.log" || rc=1
cpv R7_resume/config.json "$PKG/raw/R7_resume/config.json" || rc=1
cpv cache/index.json "$PKG/raw/cache_index.json" || rc=1
echo "finalize_copy rc=$rc"
exit $rc
