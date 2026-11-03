#!/usr/bin/env bash
# r9gate3_devbox_pinned.sh -- DEV-BOX side of the refcv9-e1rig launch gate: G-SUITE-PINNED for C0 then C1 (argv-bound, ~25 min each),
# then md5-verified ship of each evidence dir to Thor.  Adapted from r9w_gsuite.sh `devbox-pinned` (profile refcv9-e1rig, per-arm dirs).
# Launch DETACHED through WMI (CreateFlags 0x08000200) -- see memory devbox-drive-letter-and-detached-jobs.
# Markers (status file F:/refcv9_e1/gate/pinned_status.txt, one line per event):  ZZGS3-PINNED-<ARM>-OKZZ | ZZGS3-PINNED-<ARM>-FAILZZ | ZZGS3-PINNED-ALLDONEZZ
export PATH="/usr/bin:/bin:$PATH"
export MSYS_NO_PATHCONV=1
set -u
CAND="${1:?usage: r9gate3_devbox_pinned.sh <SHA40 of the commit shipped to Thor>}"; C12=${CAND:0:12}
REPO=C:/Users/Admin/tanitad-push
STAGE=F:/refcv9_e1/gate
DEVPY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
HOST=tanitad-thor; ROOT=/home/nvidia/r8ns/r9e1
ST="$STAGE/pinned_status.txt"
say() { echo "[r9gate3-pinned] $(date +%H:%M:%S) $*" | tee -a "$STAGE/pinned_driver.log"; }
mark() { echo "$1" | tee -a "$ST"; }
mkdir -p "$STAGE" || exit 1
T="$STAGE/ntlt_$C12"
if [ ! -s "$T/stack/scripts/launch_gate.py" ]; then
  rm -rf "$T"; mkdir -p "$T" && git --git-dir="$REPO/.git" -c core.autocrlf=false archive "$CAND" stack taniteval | tar --force-local -x -C "$T" || { say "git archive failed"; mark ZZGS3-PINNED-ARCHIVE-FAILZZ; exit 2; }
fi
[ -s "$T/stack/scripts/launch_gate.py" ] || { say "extracted tree has no launch_gate.py"; mark ZZGS3-PINNED-ARCHIVE-FAILZZ; exit 2; }
# the NT env (final_pinned_f7.sh): the CUDA-gated tests, the lidar-BEV GT dir, the calibration dir
export TANITAD_REFAV1_CUDA_TEST=1 TANITAD_LIDAR_BEV_GT_DIR=E:/Projects/TanitAD-artifacts/bev-lidar-gt-b1eval-20260913 TANITAD_CALIB_DIR=E:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915/stage/calibration
for ARM in C0 C1; do
  ARGVF="$T/stack/ops/runs.d/r9e1_${ARM}.argv.json"; OUT="$STAGE/devbox_evidence_${C12}_${ARM}"; SW="$STAGE/gsw_${ARM}"; LOG="$STAGE/devbox_pinned_${C12}_${ARM}.log"
  [ -s "$ARGVF" ] || { say "no argv $ARGVF"; mark ZZGS3-PINNED-${ARM}-FAILZZ; continue; }
  t0=$(date +%s)
  until [ "$(powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" 2>/dev/null | tr -d '\r')" -ge 6 ] 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt 3600 ] && break; sleep 30; done
  rm -rf "$OUT" "$SW" 2>/dev/null; mkdir -p "$SW"
  say "$ARM: launch_gate run --stage devbox (argv $ARGVF)"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$T/stack;$T/stack/scripts" timeout 5400 "$DEVPY" "$T/stack/scripts/launch_gate.py" run \
    --profile refcv9-e1rig --stage devbox --tree "$T" --commit "$CAND" --git-dir "$REPO/.git" \
    --argv-file "$ARGVF" --suite-work "$SW" --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null
  EV="$OUT/evidence/G-SUITE-PINNED.json"; [ -s "$EV" ] || EV="$OUT/G-SUITE-PINNED.json"
  PST="$([ -s "$EV" ] && "$DEVPY" -I -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(d.get('status'), d.get('started_utc'), d.get('finished_utc'))" "$EV" 2>/dev/null | tr -d '\r')"
  say "$ARM: G-SUITE-PINNED evidence: ${PST:-MISSING} ($EV) wall $(( $(date +%s) - t0 )) s"
  case "$PST" in PASS*) ;; *) mark "ZZGS3-PINNED-${ARM}-FAILZZ ${PST:-MISSING}"; continue;; esac
  TAR="$STAGE/ev_${C12}_${ARM}.tar"; tar --force-local -cf "$TAR" -C "$OUT" . || { say "tar failed"; mark ZZGS3-PINNED-${ARM}-FAILZZ; continue; }
  LM="$(md5sum "$TAR" | cut -c1-32)"; D="$ROOT/devbox_evidence_${C12}_${ARM}"
  scp -q -o BatchMode=yes "$TAR" "$HOST:$ROOT/" < /dev/null || { say "scp failed"; mark ZZGS3-PINNED-${ARM}-FAILZZ; continue; }
  RM="$(ssh -n -o BatchMode=yes $HOST "md5sum $ROOT/ev_${C12}_${ARM}.tar | cut -c1-32" | tr -d '\r')"
  [ "$RM" = "$LM" ] || { say "evidence md5 mismatch local $LM remote $RM"; mark ZZGS3-PINNED-${ARM}-FAILZZ; continue; }
  ssh -n -o BatchMode=yes $HOST "rm -rf $D && mkdir -p $D && tar -xf $ROOT/ev_${C12}_${ARM}.tar -C $D && ls $D/evidence/G-SUITE-PINNED.json $D/G-SUITE-PINNED.json 2>/dev/null | head -1" | tr -d '\r' | grep -q G-SUITE-PINNED \
    || { say "evidence unreadable on Thor under $D"; mark ZZGS3-PINNED-${ARM}-FAILZZ; continue; }
  say "$ARM: evidence on Thor: $D"; mark "ZZGS3-PINNED-${ARM}-OKZZ $D"
done
mark ZZGS3-PINNED-ALLDONEZZ
