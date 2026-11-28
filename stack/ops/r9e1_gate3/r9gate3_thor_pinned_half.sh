#!/usr/bin/env bash
# r9gate3_thor_pinned_half.sh <C40> <ARM> -- runs ON THOR: the THOR HALF of G-SUITE-PINNED SPLIT MODE (PI ruling 2026-11-28, "Split: Thor + dev box"),
# profile refcv9-full-d1 ONLY (launch_gate.PINNED_SPLIT_PROFILES).  Every pinned test file on the FULL commit archive (~14.1 GB), CPU only, no GPU lock taken;
# per-test outcomes are written to <work>/out/evidence/G-SUITE-PINNED-THOR-HALF.json (a status that is NOT PASS on purpose: only the dev-box complement's merge
# is ever G-SUITE-PINNED).  The dev box (r9gate3_devbox_complement.sh) pulls that file, re-runs only the files Thor cannot pass, and ships the merged evidence.
#   R9_THOR_GIT_DIR  REQUIRED: a git dir on Thor holding <C40> (a bare mirror or a .git); refused if unset or if <C40>^{commit} is absent.
# Study e3 (root /home/nvidia/r8ns/r9e3); the launch tree is $R/launch_tree, its argv $R/launch_tree/stack/ops/runs.d/r9e3_<ARM>.argv.json.
# Markers (stdout and <work>/status.txt):  ZZR9PH-<ARM>-OKZZ <evidence path>  |  ZZR9PH-<ARM>-FAILZZ <why>
export PATH="/usr/bin:/bin:$PATH"
set -u
C40="${1:?usage: r9gate3_thor_pinned_half.sh <SHA40> <ARM>}"; ARM="${2:?usage: r9gate3_thor_pinned_half.sh <SHA40> <ARM>}"; C12=${C40:0:12}
export R9_STUDY=e3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/r9gate3_study.sh" || { echo "ZZR9PH-${ARM}-FAILZZ BADSTUDY"; exit 2; }
R=$R9_ROOT; T=$R/launch_tree
# the launch venv the chain uses (r9gate3_chain.sh / r9gate3_thor_gate.sh: PY=...)
PY=/home/nvidia/venvs/tanitad-train/bin/python
W=$R/pinned_half_${C12}_${ARM}; OUT=$W/out; SW=$W/sw; ST=$W/status.txt
ARGV=$T/stack/ops/runs.d/${R9_PREFIX}_${ARM}.argv.json
mkdir -p "$W" || { echo "ZZR9PH-${ARM}-FAILZZ NOWORKDIR"; exit 2; }
say() { echo "[r9gate3-thor-half:$ARM] $(date -u +%H:%M:%SZ) $*" | tee -a "$W/driver.log"; }
mark() { echo "$1" | tee -a "$ST"; }
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
refuse() { say "REFUSED: $*"; mark "ZZR9PH-${ARM}-FAILZZ $*"; exit 10; }
[ "${#C40}" = 40 ] || refuse "C40 is not a 40-char sha"
case "$ARM" in ""|*/*|*" "*) refuse "bad arm '$ARM'";; esac
[ "$(r9_profile_of_arm "$ARM")" = "refcv9-full-d1" ] || refuse "split mode is refcv9-full-d1 only (arm $ARM is gated under $(r9_profile_of_arm "$ARM"))"
GD="${R9_THOR_GIT_DIR:-}"; [ -n "$GD" ] || refuse "R9_THOR_GIT_DIR is unset (a git dir on Thor that holds $C40)"
git --git-dir="$GD" cat-file -e "${C40}^{commit}" 2>/dev/null || refuse "$C40 is not a commit in $GD"
[ "$(cat "$R/launch_tree.commit" 2>/dev/null)" = "$C40" ] || refuse "$R/launch_tree.commit is not $C40 (the launch tree is another commit)"
[ -s "$ARGV" ] || refuse "no argv $ARGV"
[ -s "$T/stack/scripts/launch_gate.py" ] || refuse "no launch tree under $T"
# a ~14.1 GB archive: keep 18 GB free before it starts (a full disk would read as test failures)
FG="$(free_gb "$R")"; [ -n "$FG" ] && [ "$FG" -ge 18 ] || refuse "only ${FG:-?} GB free under $R (floor 18)"
# Thor's launch venv has NO pytest: the gate's child pytest finds it only through this side folder (the suite runner's own,
# thor_gate_runner2.py), kept after the archive's tree by launch_gate.pytest_side_entries. MEASURED 2026-11-28: without it every
# child ended "No module named pytest".
PYT="${R9_THOR_PYTEST_PATH:-/home/nvidia/refcv8_audit/pyt}"
[ -f "$PYT/pytest/__init__.py" ] || refuse "no pytest under $PYT (R9_THOR_PYTEST_PATH): every pinned child would end 'No module named pytest'"
# a re-run keeps the previous evidence (a FAIL is a record) and removes the previous archive (re-creatable from git)
[ -d "$OUT" ] && mv "$OUT" "$OUT.prev_$(date -u +%Y%m%dT%H%M%S)"
[ -d "$SW" ] && { SZ="$(du -sm "$SW" 2>/dev/null | cut -f1)"; rm -rf "${SW:?}"; say "deleted stale archive $SW (${SZ:-?} MB; re-creatable from git)"; }
mkdir -p "$SW"
say "launch_gate run --pinned-half thor (profile refcv9-full-d1, commit $C12, argv $ARGV)"
t0=$(date +%s)
# TANITAD_PINNED_PYTEST_PATH is what reaches the pinned children: run_job's _child_env REPLACES PYTHONPATH (MEASURED 2026-11-28 08:36Z)
TANITAD_PINNED_PYTEST_PATH="$PYT" PYTHONPATH="$PYT" PYTHONDONTWRITEBYTECODE=1 timeout "${R9_PINNED_TIMEOUT_S:-14400}" "$PY" "$T/stack/scripts/launch_gate.py" run \
  --profile refcv9-full-d1 --stage devbox --tree "$T" --commit "$C40" --git-dir "$GD" --argv-file "$ARGV" \
  --suite-work "$SW" --out-dir "$OUT" --cpu-only --pinned-half thor > "$W/gate.log" 2>&1 < /dev/null
EV="$OUT/evidence/G-SUITE-PINNED-THOR-HALF.json"
PST="$([ -s "$EV" ] && "$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(d.get('status'), len((d.get('details') or {}).get('complement_files') or []))" "$EV" 2>/dev/null)"
say "Thor half evidence: ${PST:-MISSING} ($EV) wall $(( $(date +%s) - t0 )) s"
# the archive is re-creatable from git and ~14 GB: delete it either way, keep the evidence and the logs
SZ="$(du -sm "$SW" 2>/dev/null | cut -f1)"; rm -rf "${SW:?}"; say "deleted $SW (${SZ:-?} MB; re-creatable from git)"
case "$PST" in HALF-COMPLETE*) mark "ZZR9PH-${ARM}-OKZZ $EV";; *) mark "ZZR9PH-${ARM}-FAILZZ ${PST:-MISSING}"; exit 1;; esac
