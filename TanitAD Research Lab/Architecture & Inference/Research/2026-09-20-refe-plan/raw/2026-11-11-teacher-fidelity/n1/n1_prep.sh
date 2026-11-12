#!/usr/bin/env bash
# TF-N1 prep (PREREG_TF_N1.md section 12): export the pinned code and banked inputs into a FRESH run dir from the git
# tip, copy the N1 scripts + card + pins beside them, and verify every pinned sha256. No scorer, no navtest score.
#   bash n1_prep.sh <run-dir> [<git ref>]      e.g.  bash n1_prep.sh C:/Users/Admin/n1_run origin/agent/arch-inf-20260803
# Run it from the folder that holds the N1 scripts (the deliverables folder, or the landed PK/raw/.../n1/ export).
set -euo pipefail
RUN="${1:?run dir}"
REF="${2:-origin/agent/arch-inf-20260803}"
MIR=C:/Users/Admin/tanitad-push
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
N1SRC="$(cd "$(dirname "$0")" && pwd)"
PY=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
G() { git -C "$MIR" -c core.autocrlf=false "$@"; }
[ -e "$RUN" ] && { echo "REFUSING: $RUN already exists (a resume uses the existing dir; prep only makes a fresh one)"; exit 1; }
echo "ref $REF = $(G rev-parse "$REF")   remote: $(G ls-remote origin "${REF#origin/}" | cut -c1-40)"
mkdir -p "$RUN/w/pk/eval" "$RUN/repo" "$RUN/n1" "$RUN/inputs/rollouts" "$RUN/x"
# ---- code: refe/, eval/*.py + eval/instruments/, TF-2's tf2_score.py
G ls-tree --name-only "$REF" "$PK/eval/" | grep -E '\.py$' > "$RUN/x/evalpy.txt"
{ echo "$PK/refe"; echo "$PK/eval/instruments"; echo "$PK/raw/2026-11-11-teacher-fidelity/tf2/tf2_score.py"; cat "$RUN/x/evalpy.txt"; } \
  | tr '\n' '\0' | xargs -0 git -C "$MIR" -c core.autocrlf=false archive --format=tar "$REF" | tar -x -C "$RUN/x"
mv "$RUN/x/$PK/refe" "$RUN/w/pk/refe"
mv "$RUN/x/$PK/eval/"* "$RUN/w/pk/eval/"
mv "$RUN/x/$PK/raw/2026-11-11-teacher-fidelity/tf2/tf2_score.py" "$RUN/w/tf2_score.py"
# ---- W-FAM's repo code: taniteval (code only) + stack/tanitad
G ls-tree -r --name-only "$REF" taniteval stack/tanitad | grep -E '\.(py|json|yaml|yml)$' | grep -v -E '^taniteval/(results|tests)/' \
  | tr '\n' '\0' | xargs -0 git -C "$MIR" -c core.autocrlf=false archive --format=tar "$REF" | tar -x -C "$RUN/repo"
# ---- banked inputs (all from the tip)
for i in 0 1 2 3; do
  G show "$REF:$PK/raw/2026-10-05-teacher-navtest/full/rows_s${i}of4.jsonl.xz" > "$RUN/inputs/rollouts/rows_s${i}of4.jsonl.xz"
done
G show "$REF:$PK/raw/2026-10-05-teacher-navtest/refe_p4_teacher_navtest.npz" > "$RUN/inputs/refe_p4_teacher_navtest.npz"
G show "$REF:$PK/raw/2026-10-05-teacher-navtest/csv/refe_p4_teacher_navtest.csv" > "$RUN/inputs/refe_p4_teacher_navtest.csv"
G show "$REF:$PK/raw/2026-10-05-teacher-navtest/csv/refe_p4_human_navtest.csv" > "$RUN/inputs/refe_p4_human_navtest.csv"
G show "$REF:$PK/raw/2026-10-04-navtest-final/frame_control_navtest_full.json" > "$RUN/inputs/frame_control_navtest_full.json"
G show "$REF:$PK/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz" > "$RUN/inputs/p5s_cubic.npz"
# ---- the N1 scripts, card and pins
cp "$N1SRC/tf_n1_build.py" "$N1SRC/tf_n1_analyze.py" "$N1SRC/tf_n1_chain.py" "$N1SRC/PREREG_TF_N1.md" "$RUN/n1/"
rm -rf "${RUN:?}/x"
if [ -f "$N1SRC/N1_PINS.json" ]; then cp "$N1SRC/N1_PINS.json" "$RUN/n1/"; else echo "NO N1_PINS.json: pins not verified (make them with n1_make_pins.py)"; exit 0; fi
# ---- verify every pin
"$PY" - "$RUN" <<'EOF'
import hashlib, json, os, sys
run = sys.argv[1]
pins = json.load(open(os.path.join(run, "n1", "N1_PINS.json"), encoding="utf-8"))["files"]
bad = []
for k, v in pins.items():
    p = os.path.join(run, k)
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    if got != v:
        bad.append((k, got))
print("ZZN1_PREP", "PASS" if not bad else "FAIL", len(pins), "pins", bad)
sys.exit(1 if bad else 0)
EOF
echo "card sha256: $(sha256sum "$RUN/n1/PREREG_TF_N1.md" | cut -c1-64)"
