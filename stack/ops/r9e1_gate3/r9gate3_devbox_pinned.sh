#!/usr/bin/env bash
# r9gate3_devbox_pinned.sh -- DEV-BOX side of the refcv9-e1rig launch gate: G-SUITE-PINNED for C0 then C1 (argv-bound, ~25 min each),
# then md5-verified ship of each evidence dir to Thor.  Adapted from r9w_gsuite.sh `devbox-pinned` (profile refcv9-e1rig, per-arm dirs).
# Launch DETACHED through WMI (CreateFlags 0x08000200) -- see memory devbox-drive-letter-and-detached-jobs.
# STUDY (R9_STUDY, default e1; set by r9gate3_launch_pinned.ps1 -Study from r9gate3_restart.sh): root / argv prefix / profile / default arms / stage dir come from
#   r9gate3_study.sh NEXT TO THIS SCRIPT (e1: F:/refcv9_e1/gate, refcv9-e1rig, r9e1; e2: F:/refcv9_e2/gate, refcv9-e2, r9e2).
# Markers (status file <stage>/pinned_status.txt, e1: F:/refcv9_e1/gate/pinned_status.txt, one line per event):  ZZGS3-PINNED-<ARM>-OKZZ | ZZGS3-PINNED-<ARM>-FAILZZ [DISK-LOW] | ZZGS3-PINNED-ALLDONEZZ
export PATH="/usr/bin:/bin:$PATH"
export MSYS_NO_PATHCONV=1
set -u
CAND="${1:?usage: r9gate3_devbox_pinned.sh <SHA40 of the commit shipped to Thor> [ARMS, quoted]}"; C12=${CAND:0:12}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/r9gate3_study.sh" || { echo "ZZGS3-PINNED-BADSTUDYZZ"; exit 2; }    # R9_ROOT R9_PREFIX R9_PROFILE R9_DEVSTAGE ... (one definition; refuses an unknown study)
ARMS="${2:-$R9_DEFAULT_ARMS}"
REPO=C:/Users/Admin/tanitad-push
STAGE=$R9_DEVSTAGE
DEVPY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
HOST=tanitad-thor; ROOT=$R9_ROOT
ST="$STAGE/pinned_status.txt"
say() { echo "[r9gate3-pinned] $(date +%H:%M:%S) $*" | tee -a "$STAGE/pinned_driver.log"; }
mark() { echo "$1" | tee -a "$ST"; }
free_gb() { df -BG "$1" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}'; }   # free GB on the drive holding $1 (an integer; empty if df fails)
# suite-work dirs of SHIPPED arms not yet fully deleted. One rm is not enough (MEASURED 2026-11-03: gsw_B2 took 3 passes over ~7 min to go, the first
# ~4.5 min long), so a residue is retried before every later arm and at the end. Paths are <stage>/sw<N> (N = the arm's place in this run; see MAX_PATH below): no spaces.
PENDING=""
purge() { local d left=""; for d in $PENDING; do rm -rf "${d:?}" 2>/dev/null; [ -e "$d" ] && left="$left $d"; done; PENDING="$left"; }
mkdir -p "$STAGE" || exit 1
T="$STAGE/ntlt_$C12"
if [ ! -s "$T/stack/scripts/launch_gate.py" ]; then
  rm -rf "$T"; mkdir -p "$T" && git --git-dir="$REPO/.git" -c core.autocrlf=false archive "$CAND" stack taniteval | tar --force-local -x -C "$T" || { say "git archive failed"; mark ZZGS3-PINNED-ARCHIVE-FAILZZ; exit 2; }
fi
[ -s "$T/stack/scripts/launch_gate.py" ] || { say "extracted tree has no launch_gate.py"; mark ZZGS3-PINNED-ARCHIVE-FAILZZ; exit 2; }
# MAX_PATH (MM 2026-11-09, MEASURED): G-SUITE-PINNED extracts the FULL commit into <suite-work>/f<commit[:10]>/ (launch_gate.py job_suite_pinned) and
# Windows refuses a path of 260+ characters. The longest repo path at 269f43973a46 is 219 characters, so <stage>/gsw_K2M1N20 (29) made the deepest
# file 261 and the extraction died 165 s in with FileNotFoundError, while K0 (256) passed. The work dir is therefore <stage>/sw<N> (21 characters on
# both stages), and an arm whose deepest extracted path would still not fit is refused BY NAME (FAILZZ PATH-LONG) before its gate runs.
ML="$(git --git-dir="$REPO/.git" -c core.quotepath=false ls-tree -r --name-only "$CAND" | awk '{ if (length > m) m = length } END { print m + 0 }')"
[ "$ML" -gt 0 ] 2>/dev/null || { say "cannot read the longest path of $CAND"; mark ZZGS3-PINNED-ARCHIVE-FAILZZ; exit 2; }
path_need() { echo $(( ${#1} + 13 + $2 )); }   # <suite-work> "/f" <10 hex> "/" <longest repo path>: the deepest file the extraction writes
# a FAILed arm keeps its sw<N> for diagnosis until the NEXT driver run, which frees every earlier sw<N> before its first arm (the archive is a git
# export; the FAIL record itself is devbox_evidence_<C12>_<ARM>, which is kept)
for d in "$STAGE"/sw[0-9]*; do [ -d "$d" ] && PENDING="$PENDING $d"; done; purge
NA=0
# the NT env (final_pinned_f7.sh): the CUDA-gated tests, the lidar-BEV GT dir, the calibration dir
# CPU ONLY (MM 2026-11-03, MEASURED): with the dev-box GPU visible, tests/test_v8f_ego_dagger_compose.py fails 3/36 (device mismatch "cuda:0 and
# cpu", golden bank, resume bit-identity) at bf2e7d22 AND 9be3211; with CUDA_VISIBLE_DEVICES=-1 the same archive reads 36/36 PASS. The one CUDA test
# in the pinned files (test_refa_v1_precision::test_cuda_bf16_autocast...) then SKIPS "no CUDA device" -- a REGISTERED skip (launch_gate.py:515).
export CUDA_VISIBLE_DEVICES=-1
export TANITAD_REFAV1_CUDA_TEST=1 TANITAD_LIDAR_BEV_GT_DIR=E:/Projects/TanitAD-artifacts/bev-lidar-gt-b1eval-20260913 TANITAD_CALIB_DIR=E:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915/stage/calibration
for ARM in $ARMS; do
  R9_PROFILE="$(r9_profile_of_arm "$ARM")"   # e3: a `<V>_d1` arm runs under the D1 profile (its argv carries --wsd-branch-from, which only that profile admits)
  # a D1 arm's `--wsd-branch-from` names a Thor path whose config.json the gate reads: <stage>/pathmap_<ARM>.txt (one line FROM=TO, optional, absent for every other arm) maps it to a local copy
  PMAP="$(cat "$STAGE/pathmap_${ARM}.txt" 2>/dev/null | tr -d '\r' | head -1)"
  ARGVF="$T/stack/ops/runs.d/${R9_PREFIX}_${ARM}.argv.json"; OUT="$STAGE/devbox_evidence_${C12}_${ARM}"; NA=$((NA + 1)); SW="$STAGE/sw$NA"; LOG="$STAGE/devbox_pinned_${C12}_${ARM}.log"
  [ -s "$ARGVF" ] || { say "no argv $ARGVF"; mark ZZGS3-PINNED-${ARM}-FAILZZ; continue; }
  PN=$(path_need "$SW" "$ML"); [ "$PN" -le 259 ] || { say "$ARM: the deepest extracted path would be $PN characters (> 259) under $SW"; mark "ZZGS3-PINNED-${ARM}-FAILZZ PATH-LONG $PN"; continue; }
  t0=$(date +%s)
  until [ "$(powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" 2>/dev/null | tr -d '\r')" -ge 6 ] 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt 3600 ] && break; sleep 30; done
  # a re-run keeps the previous evidence (a FAIL is a record; ~1 MB) and frees the previous suite-work archive before the disk check below
  [ -d "$OUT" ] && mv "$OUT" "$OUT.prev_$(date +%Y%m%dT%H%M%S)"
  purge; rm -rf "$SW" 2>/dev/null
  # G-SUITE-PINNED extracts a ~12.5 GB full-commit archive into $SW: wait (<= 1 h) for room on the stage drive, else refuse the arm by name
  # rather than run a suite into a full disk, whose failures would read as test failures.
  t1=$(date +%s); until [ "$(free_gb "$STAGE")" -ge "${R9_PINNED_MIN_FREE_GB:-20}" ] 2>/dev/null; do [ $(( $(date +%s) - t1 )) -gt 3600 ] && break; sleep 60; done
  [ "$(free_gb "$STAGE")" -ge "${R9_PINNED_MIN_FREE_GB:-20}" ] 2>/dev/null || { say "$ARM: only $(free_gb "$STAGE") GB free under $STAGE"; mark "ZZGS3-PINNED-${ARM}-FAILZZ DISK-LOW"; continue; }
  mkdir -p "$SW"
  say "$ARM: launch_gate run --stage devbox (argv $ARGVF)"
  # MSYS_NO_PATHCONV is for THIS script's ssh / scp of Thor paths only. Exported into the gate it reaches pytest, and test_r9_e1_floor's
  # r9e1_read.sh dry runs then hand native python an unconverted /f/... path: 6 / 28 FAIL (MM 2026-11-03, MEASURED: the same test in the same
  # archive passes without it and fails with it). The pinned suite runs in the ordinary Git-bash environment its tests were written for.
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$T/stack;$T/stack/scripts" env -u MSYS_NO_PATHCONV timeout "${R9_PINNED_TIMEOUT_S:-10800}" "$DEVPY" "$T/stack/scripts/launch_gate.py" run \
    --profile "$R9_PROFILE" --stage devbox --tree "$T" --commit "$CAND" --git-dir "$REPO/.git" \
    --argv-file "$ARGVF" ${PMAP:+--path-map "$PMAP"} --suite-work "$SW" --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null
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
  # the evidence is on Thor: the suite-work archive (~12.5 GB per arm) is dead weight -- F: fell 120 -> 43 GB in ~2.5 h on 2026-11-03 while every
  # arm's stayed. A FAILed arm (the `continue`s above) keeps its own for diagnosis; that arm's next run frees it before its disk check.
  PENDING="$PENDING $SW"; purge; [ -n "$PENDING" ] && say "$ARM: suite-work not fully deleted yet (retried before the next arm):$PENDING"
done
for _i in $(seq 1 20); do purge; [ -z "$PENDING" ] && break; sleep 30; done
[ -n "$PENDING" ] && say "suite-work NOT deleted after 10 min of retries:$PENDING"
mark ZZGS3-PINNED-ALLDONEZZ
