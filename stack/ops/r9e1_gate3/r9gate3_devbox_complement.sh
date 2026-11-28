#!/usr/bin/env bash
# r9gate3_devbox_complement.sh <C40> <ARM> -- DEV-BOX half of G-SUITE-PINNED SPLIT MODE (PI ruling 2026-11-28, "Split: Thor + dev box"), refcv9-full-d1 ONLY.
# Pulls the Thor half's G-SUITE-PINNED-THOR-HALF.json (written by r9gate3_thor_pinned_half.sh on Thor, sha256-verified), runs `launch_gate.py run
# --pinned-half devbox-complement` (the Thor half is refused unless bound to this commit / tree / argv / profile and run on the FULL archive; the SAME commit is
# archived REDUCED -- Research Lab / FlyWheels members >= 1 MB dropped, the list in the evidence -- and ONLY the files Thor cannot pass are re-run), writes the
# merged G-SUITE-PINNED evidence to F:/refcv9_e3/gate/devbox_evidence_<c12>_<ARM>/ and ships it to Thor $R/devbox_evidence_<c12>_<ARM>/ (the chain waits there for
# evidence/G-SUITE-PINNED.json), verifying the copy by sha256.  Launch DETACHED through WMI (CreateFlags 0x08000200).
# Markers (status file <stage>/complement_status.txt):  ZZR9DC-<ARM>-OKZZ <Thor dir>  |  ZZR9DC-<ARM>-FAILZZ <why>
export PATH="/usr/bin:/bin:$PATH"
export MSYS_NO_PATHCONV=1
set -u
C40="${1:?usage: r9gate3_devbox_complement.sh <SHA40> <ARM>}"; ARM="${2:?usage: r9gate3_devbox_complement.sh <SHA40> <ARM>}"; C12=${C40:0:12}
export R9_STUDY=e3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/r9gate3_study.sh" || { echo "ZZR9DC-${ARM}-FAILZZ BADSTUDY"; exit 2; }
REPO=C:/Users/Admin/tanitad-push
DEVPY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
HOST=tanitad-thor; ROOT=$R9_ROOT
STAGE=$R9_DEVSTAGE                                   # F:/refcv9_e3/gate
W=${R9_COMPLEMENT_W:-C:/Users/Admin/pinsplit_c_${C12}_${ARM}}   # work dir (tree, Thor-half copy); default C:
SW=${R9_COMPLEMENT_SW:-C:/q/e3pc}                    # suite-work: short, two levels deep (MAX_PATH: repo paths up to 236 characters fit); default C:
# (MM 4a3d05, 2026-11-28: both overridable -- C: fell to 4.5 GB free, so the D1 run puts them on F: with R9_COMPLEMENT_W / R9_COMPLEMENT_SW)
OUT="$STAGE/devbox_evidence_${C12}_${ARM}"; D="$ROOT/devbox_evidence_${C12}_${ARM}"
THALF="$ROOT/pinned_half_${C12}_${ARM}/out/evidence/G-SUITE-PINNED-THOR-HALF.json"
ST="$STAGE/complement_status.txt"; LOG="$STAGE/devbox_complement_${C12}_${ARM}.log"
mkdir -p "$STAGE" || exit 1
say() { echo "[r9gate3-complement:$ARM] $(date +%H:%M:%S) $*" | tee -a "$STAGE/complement_driver.log"; }
mark() { echo "$1" | tee -a "$ST"; }
free_gb() { df -BG "$1" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}'; }
sha() { sha256sum "$1" | cut -c1-64; }
fail() { say "FAIL: $*"; mark "ZZR9DC-${ARM}-FAILZZ $*"; exit 1; }
# deletes only what THIS script made: the reduced archive and the work dir (re-creatable from git / from Thor), by literal path
cleanup() { local sz; sz="$(du -sm "$SW" "$W" 2>/dev/null | awk '{s+=$1} END {print s+0}')"; rm -rf "${SW:?}" "${W:?}"; say "deleted $SW and $W (${sz:-?} MB; re-creatable from git / Thor)"; }
[ "${#C40}" = 40 ] || fail "C40 is not a 40-char sha"
case "$ARM" in ""|*/*|*" "*) fail "bad arm '$ARM'";; esac
[ "$(r9_profile_of_arm "$ARM")" = "refcv9-full-d1" ] || fail "split mode is refcv9-full-d1 only (arm $ARM is gated under $(r9_profile_of_arm "$ARM"))"
git --git-dir="$REPO/.git" cat-file -e "${C40}^{commit}" 2>/dev/null || fail "$C40 is not a commit in $REPO (git fetch the mirror first)"
for P in "${W:0:2}/" "${SW:0:2}/"; do FG="$(free_gb "$P")"; [ -n "$FG" ] && [ "$FG" -ge 6 ] 2>/dev/null || fail "only ${FG:-?} GB free on $P (floor 6)"; done
ML="$(git --git-dir="$REPO/.git" -c core.quotepath=false ls-tree -r --name-only "$C40" | awk '{ if (length > m) m = length } END { print m + 0 }')"
[ "$ML" -gt 0 ] 2>/dev/null || fail "cannot read the longest path of $C40"
PN=$(( ${#SW} + 13 + ML )); [ "$PN" -le 259 ] || fail "PATH-LONG: the deepest extracted path would be $PN characters under $SW"
rm -rf "${SW:?}" "${W:?}"; mkdir -p "$W/tree" "$W/thor_half" || fail "cannot create $W"
# 1. the code tree (the gate runs from it; its stack/+taniteval/ digest is what the Thor half is bound to)
git --git-dir="$REPO/.git" -c core.autocrlf=false archive "$C40" stack taniteval | tar --force-local -x -C "$W/tree" || { cleanup; fail "git archive failed"; }
ARGVF="$W/tree/stack/ops/runs.d/${R9_PREFIX}_${ARM}.argv.json"; [ -s "$ARGVF" ] || { cleanup; fail "no argv $ARGVF"; }
# 2. the Thor half, sha256-verified across the copy
RS="$(ssh -n -o BatchMode=yes $HOST "sha256sum $THALF | cut -c1-64" < /dev/null | tr -d '\r')"
[ "${#RS}" = 64 ] || { cleanup; fail "no Thor-half evidence on Thor ($THALF)"; }
scp -q -o BatchMode=yes "$HOST:$THALF" "$W/thor_half/" < /dev/null || { cleanup; fail "scp of the Thor half failed"; }
LS="$(sha "$W/thor_half/G-SUITE-PINNED-THOR-HALF.json")"; [ "$LS" = "$RS" ] || { cleanup; fail "Thor-half sha256 mismatch: local $LS remote $RS"; }
say "Thor half pulled: sha256 $LS"
PMAP="$(cat "$STAGE/pathmap_${ARM}.txt" 2>/dev/null | tr -d '\r' | head -1)"
# the dev-box environment of r9gate3_devbox_pinned.sh: CPU only, the CUDA-gated tests, the lidar-BEV GT dir, the calibration dir
export CUDA_VISIBLE_DEVICES=-1
export TANITAD_REFAV1_CUDA_TEST=1 TANITAD_LIDAR_BEV_GT_DIR=E:/Projects/TanitAD-artifacts/bev-lidar-gt-b1eval-20260913 TANITAD_CALIB_DIR=E:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915/stage/calibration
[ -d "$OUT" ] && mv "$OUT" "$OUT.prev_$(date +%Y%m%dT%H%M%S)"
mkdir -p "${SW%/*}" "$OUT"
say "launch_gate run --pinned-half devbox-complement (argv $ARGVF)"
t0=$(date +%s)
# MSYS_NO_PATHCONV=1 STAYS SET for the gate's exec, so MSYS hands every argument to native python LITERALLY: each one is already a
# Windows path, and a D1 arm's --path-map FROM is a THOR path. `env -u MSYS_NO_PATHCONV` re-enabled the conversion and MSYS rewrote it
# to C:\Program Files\Git\home\nvidia\... (MEASURED 2026-11-28 11:28Z, D1 af1ce565: ZZGATE-ERRORZZ in 74 s). NOPC_SHIM then drops the
# variable from the gate's environment before the gate starts, so it still never reaches pytest (MEASURED 2026-11-03: 6 / 28 FAIL with it).
NOPC_SHIM='import os, subprocess, sys; os.environ.pop("MSYS_NO_PATHCONV", None); sys.exit(subprocess.call([sys.executable] + sys.argv[1:]))'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$W/tree/stack;$W/tree/stack/scripts" timeout "${R9_PINNED_TIMEOUT_S:-10800}" "$DEVPY" -I -c "$NOPC_SHIM" "$W/tree/stack/scripts/launch_gate.py" run \
  --profile refcv9-full-d1 --stage devbox --tree "$W/tree" --commit "$C40" --git-dir "$REPO/.git" --argv-file "$ARGVF" ${PMAP:+--path-map "$PMAP"} \
  --suite-work "$SW" --out-dir "$OUT" --cpu-only --pinned-half devbox-complement --pinned-thor-half "$W/thor_half/G-SUITE-PINNED-THOR-HALF.json" > "$LOG" 2>&1 < /dev/null
EV="$OUT/evidence/G-SUITE-PINNED.json"
PST="$([ -s "$EV" ] && "$DEVPY" -I -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(d.get('status'), d.get('started_utc'), d.get('finished_utc'))" "$EV" 2>/dev/null | tr -d '\r')"
say "merged G-SUITE-PINNED evidence: ${PST:-MISSING} ($EV) wall $(( $(date +%s) - t0 )) s"
cp "$W/thor_half/G-SUITE-PINNED-THOR-HALF.json" "$OUT/" 2>/dev/null     # beside (not inside) evidence/: the chain imports evidence/G-*.json only
cleanup                                                                # the reduced archive is dead weight now (a FAIL keeps $OUT and $LOG, not the archive)
case "$PST" in PASS*) ;; *) fail "${PST:-MISSING}";; esac
# 3. ship, verify by sha256
TAR="$STAGE/ev_${C12}_${ARM}.tar"; tar --force-local -cf "$TAR" -C "$OUT" . || fail "tar failed"
LT="$(sha "$TAR")"
scp -q -o BatchMode=yes "$TAR" "$HOST:$ROOT/" < /dev/null || fail "scp failed"
RT="$(ssh -n -o BatchMode=yes $HOST "sha256sum $ROOT/ev_${C12}_${ARM}.tar | cut -c1-64" | tr -d '\r')"
[ "$RT" = "$LT" ] || fail "evidence sha256 mismatch: local $LT remote $RT"
ssh -n -o BatchMode=yes $HOST "rm -rf $D && mkdir -p $D && tar -xf $ROOT/ev_${C12}_${ARM}.tar -C $D" < /dev/null || fail "extract on Thor failed"
LE="$(sha "$EV")"; RE="$(ssh -n -o BatchMode=yes $HOST "sha256sum $D/evidence/G-SUITE-PINNED.json | cut -c1-64" | tr -d '\r')"
[ "$RE" = "$LE" ] || fail "G-SUITE-PINNED.json on Thor differs from the local one (local $LE remote $RE)"
say "evidence on Thor: $D (G-SUITE-PINNED.json sha256 $LE)"
mark "ZZR9DC-${ARM}-OKZZ $D"
