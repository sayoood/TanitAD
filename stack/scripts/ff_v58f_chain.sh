#!/usr/bin/env bash
# ============================================================================ #
# ff_v58f_chain.sh — POD-SIDE four-family rescore of the banked v5.8f T1 dumps  #
#                                                                              #
# Closes MODEL_REGISTRY §1.14, which is ADE-dominated and therefore INCOMPLETE  #
# under the binding rule (Sayed 2026-08-02): every eval reports LONGITUDINAL +  #
# LATERAL + TACTICAL + STRATEGIC in ADDITION to ADE, per-family, never pooled,  #
# each with the paired episode-cluster bootstrap. The row says the gap itself:  #
# "NOT yet a release row: four families + episode-cluster CIs on the banked     #
# windows, then T1 (E1.4), complete it."                                       #
#                                                                              #
# ⛔ FOUR RULES FROM CLAUDE.md THAT SHAPE THIS FILE. Each has cost hours.        #
#                                                                              #
# 1. NO `git fetch` / `git checkout` ANYWHERE IN THIS CHAIN.                    #
#    PODS HAVE NO GIT CREDENTIALS. `git fetch` HANGS (it does not fail), and if #
#    a failed fetch is followed by a successful `checkout -B <b> origin/<b>`    #
#    the tree RESETS to an ancient commit — MEASURED 2026-08-11, pod5's HEAD    #
#    sat at 6d714ad (weeks old) while its working tree was fully current. That  #
#    would DESTROY the very files this chain was shipped to run. Files arrive   #
#    by md5-verified FILE-SHIP; this script only VERIFIES them.                 #
#                                                                              #
# 2. VERIFY THE SHIPPED CODE BEFORE LAUNCH — by grep AND by a real `import`.    #
#    A pod's checkout drifts silently and a launch from it resurrects fixed     #
#    bugs (MEASURED 2026-07-27: pod2 sat at 0f93b98 while the v5 gate fix was   #
#    at HEAD). `git log` on the pod is NOT proof. Three chains this campaign    #
#    refused to run stale code rather than running it — that is the design.     #
#                                                                              #
# 3. PYTHONPATH MUST CARRY BOTH ROOTS. `PYTHONPATH=/workspace/TanitAD/stack` is #
#    required or trainers/tools die with `ModuleNotFound: tanitad`; the         #
#    taniteval package root is needed too, and `cd` alone is not enough.        #
#                                                                              #
# 4. EVERY BRANCH EMITS AN EXPLICIT `FF_EXIT=`. A chain that ends silently is   #
#    indistinguishable from one that was killed.                               #
#                                                                              #
# Also applied: OMP_NUM_THREADS is set BEFORE any torch import. MEASURED        #
# 2026-07-27 — 7 concurrent arms sat at GPU sm 0-6 % for 50 MINUTES with zero   #
# progress (torch spawns ~113 threads per process); the same arm finished in    #
# 232 s with OMP_NUM_THREADS=6. It looks exactly like a hang.                   #
#                                                                              #
# ⓘ CPU-ONLY. This is a RESCORE of already-banked per-window dumps: no model,   #
#   no forward pass, no GPU. It is therefore safe to run beside a training job  #
#   ("never add GPU/RAM load to a pod that is training" is respected).          #
# ============================================================================ #
set -uo pipefail

REPO="${REPO:-/workspace/TanitAD}"
STACK="$REPO/stack"
TE="$REPO/taniteval"
TOOL="$TE/tools/ff_rescore.py"

# The dumps a T1 run is producing on pod5 (E1.4). Override to rescore others.
DUMP_A="${DUMP_A:-/workspace/experiments/t1-v58f/dump_v5f_30k}"
DUMP_B="${DUMP_B:-/workspace/experiments/t1-v58f/dump_stage_a_repaired}"
LABEL_A="${LABEL_A:-v5f30k}"
LABEL_B="${LABEL_B:-stageA}"
OUT="${OUT:-/workspace/experiments/t1-v58f/four_families}"
LEAD="${LEAD:-}"          # optional lead block; without it distance-keeping is
                          # UNAVAILABLE — a WORK ITEM, not a pass
NBOOT="${NBOOT:-2000}"
SEED="${SEED:-0}"
DT="${DT:-0.1}"

export PYTHONPATH="$STACK:$TE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
PY="${PY:-/workspace/venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

say() { echo "[ff-chain] $*"; }
fail() { echo "[ff-chain] ⛔ $*"; echo "FF_EXIT=$1"; exit "$1"; }

say "repo=$REPO  py=$PY  PYTHONPATH=$PYTHONPATH  OMP=$OMP_NUM_THREADS"
# ⚠️ `ps -C python3` returns EMPTY for a healthy job because pods run
# /workspace/venv/bin/python — do not use it to conclude anything here.

# ---------------------------------------------------------------------------- #
# GATE 1 — the shipped files EXIST                                              #
# ---------------------------------------------------------------------------- #
for f in "$TOOL" "$TE/taniteval/four_families.py" "$TE/taniteval/ci.py" \
         "$STACK/tanitad/refs/refc_tactical.py"; do
  [ -f "$f" ] || fail 10 "MISSING $f — ship it (xz+b64 PTY push, per-file md5) and re-run. ⛔ Do NOT git-fetch on a pod."
done
say "gate1 OK — all four source files present"
md5sum "$TOOL" "$TE/taniteval/four_families.py" 2>/dev/null | sed 's/^/[ff-chain] md5 /'

# ---------------------------------------------------------------------------- #
# GATE 2 — the shipped files are the RIGHT VERSION (grep for the specific fix)   #
# A stale four_families.py has no trajectory-derived TACTICAL, so the rescore    #
# would run, succeed, and emit the SAME incomplete row this chain exists to fix. #
# ---------------------------------------------------------------------------- #
grep -q "def tactical_from_trajectory" "$TE/taniteval/four_families.py" \
  || fail 11 "four_families.py is STALE (no tactical_from_trajectory) — it would emit the same ADE-dominated row. Re-ship it."
grep -q "def strategic_unavailable" "$TE/taniteval/four_families.py" \
  || fail 11 "four_families.py is STALE (no strategic_unavailable) — re-ship it."
grep -q "TARGET_SPEED_BANDS_MPS" "$TE/taniteval/four_families.py" \
  || fail 11 "four_families.py is STALE (no target-speed accuracy) — re-ship it."
grep -q "CROSS-GRID JOIN REFUSED" "$TOOL" \
  || fail 11 "ff_rescore.py is STALE (no cross-grid refusal) — re-ship it."
say "gate2 OK — greps confirm the shipped versions carry the fixes"

# ---------------------------------------------------------------------------- #
# GATE 3 — a REAL import. grep proves the text is there; only an import proves   #
# the module actually loads on THIS box with THIS PYTHONPATH.                    #
# ---------------------------------------------------------------------------- #
"$PY" - <<'PYEOF' || fail 12 "import gate FAILED — the modules do not load on this pod. Fix PYTHONPATH or re-ship; do NOT launch."
import sys
from taniteval import four_families as ff, ci
from tanitad.refs.refc_tactical import factor_from_kinematics, LAT_CLASSES, LON_CLASSES
for name in ("tactical_from_trajectory", "strategic_unavailable",
             "maneuver_kinematics", "TARGET_SPEED_BANDS_MPS"):
    assert hasattr(ff, name), f"four_families lacks {name}"
assert hasattr(ci, "paired_episode_cluster_bootstrap")
assert hasattr(ci, "episode_cluster_bootstrap")
# a 3-line smoke of the new maths, so a broken build cannot pass a mere import
import torch
g = torch.zeros(4, 20, 2); g[:, :, 0] = torch.arange(1, 21).float()
o = ff.tactical_from_trajectory(g.clone(), g, 0.1, ["e0"] * 4, n_boot=5, tier="T1")
assert o["status"] == "OK" and o["lateral_decision"]["accuracy"] == 1.0, o
print(f"[ff-chain] import gate OK — python {sys.version.split()[0]}, "
      f"lat={LAT_CLASSES} lon={LON_CLASSES}")
PYEOF

# ---------------------------------------------------------------------------- #
# GATE 4 — the dumps exist and are non-empty                                    #
# ---------------------------------------------------------------------------- #
ARGS=()
N_DUMPS=0
for pair in "$LABEL_A=$DUMP_A" "$LABEL_B=$DUMP_B"; do
  lbl="${pair%%=*}"; dir="${pair#*=}"
  if [ -z "$dir" ]; then continue; fi
  if [ ! -d "$dir" ]; then
    say "⚠️ dump $lbl not present at $dir — SKIPPING it (a T1 roll may still be writing)"
    continue
  fi
  n=$(ls "$dir"/ep*.npz 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then
    say "⚠️ dump $lbl at $dir has NO ep*.npz — SKIPPING (roll not finished?)"
    continue
  fi
  say "dump $lbl: $n episode files at $dir"
  ARGS+=(--dump "$lbl=$dir")
  N_DUMPS=$((N_DUMPS + 1))
done
[ "$N_DUMPS" -gt 0 ] || fail 13 "no usable dump directory — nothing to rescore. (Not a failure of this chain: wait for the T1 roll, then re-run.)"

mkdir -p "$OUT" || fail 14 "cannot create $OUT"
# ⛔ never judge pod disk with `df` — it reports the 965 TB cluster and hides the
# per-pod MooseFS quota. A real write test is the only admissible probe.
if ! dd if=/dev/zero of="$OUT/.ffprobe" bs=1M count=16 status=none 2>/dev/null; then
  rm -f "$OUT/.ffprobe"
  fail 15 "16 MB write test FAILED at $OUT — the per-pod quota is full. A full quota has already killed a flagship mid-checkpoint."
fi
rm -f "$OUT/.ffprobe"
say "gate4 OK — $N_DUMPS dump(s), $OUT writable (dd test, not df)"

[ -n "$LEAD" ] && { [ -f "$LEAD" ] || fail 16 "--lead given but $LEAD does not exist"; ARGS+=(--lead "$LEAD"); }
[ -z "$LEAD" ] && say "⚠️ NO LEAD BLOCK — the distance-keeping half of LONGITUDINAL will report UNAVAILABLE with its reason and n. That is a WORK ITEM, not a pass: build one with taniteval/tools/build_lead_block.py on THIS window grid."

# ---------------------------------------------------------------------------- #
# THE RESCORE                                                                    #
# --strategic-no-label: PhysicalAI-AV carries no map, no lane graph, no junction  #
# label and no route signal ("we do not include open maps data"), and egomotion   #
# has no lat/lon — so STRATEGIC is n/a WITH ITS REASON AND ITS n, per clause 5.   #
# It is declared explicitly and never inferred, because the fact is a property of #
# the CORPUS and guessing it would let a real strategic gap hide behind it.       #
# ---------------------------------------------------------------------------- #
say "rescoring -> $OUT"
"$PY" "$TOOL" "${ARGS[@]}" \
  --out-dir "$OUT" \
  --strategic-no-label \
  --n-boot "$NBOOT" --seed "$SEED" --dt "$DT" \
  2>&1 | tee "$OUT/ff_rescore.log"
rc=${PIPESTATUS[0]}

case "$rc" in
  0) : ;;
  3) say "⛔ CROSS-GRID REFUSAL — the arms are not on the same windows. The"
     say "   per-arm records are still valid on their OWN grids and the refusal"
     say "   is BANKED in $OUT/ff_comparison.json. Rebuild the dumps on a common"
     say "   grid before comparing; do NOT truncate."
     echo "FF_EXIT=3"; exit 3 ;;
  2) fail 2 "the tool REFUSED (tier stamp / estimator / lead grid). Read $OUT/ff_rescore.log — the refusal states which." ;;
  *) fail "$rc" "rescore failed with exit $rc; log at $OUT/ff_rescore.log" ;;
esac

# ---------------------------------------------------------------------------- #
# POST-GATE — the output must actually satisfy the binding rule                  #
# ---------------------------------------------------------------------------- #
"$PY" - "$OUT" <<'PYEOF' || fail 20 "post-gate FAILED — the emitted JSON does not satisfy the binding rule"
import glob, json, os, sys
out = sys.argv[1]
files = [f for f in glob.glob(os.path.join(out, "ff_*.json"))
         if not f.endswith("ff_comparison.json")]
assert files, f"no per-arm JSON under {out}"
for f in files:
    d = json.load(open(f))
    fam = d["four_families"]
    for k in ("longitudinal", "lateral", "tactical", "strategic"):
        assert k in fam, f"{f}: family {k} MISSING — that is the silent drop the rule forbids"
        blk = fam[k]
        assert blk.get("tier"), f"{f}: {k} has no tier stamp"
        if blk.get("status") == "UNAVAILABLE":
            assert blk.get("reason") and blk.get("n") is not None, \
                f"{f}: {k} is UNAVAILABLE without a reason and an n (clause 5)"
    assert fam["_rule_satisfied"] is True, f"{f}: _rule_satisfied is False"
    est = d["intervals"]["estimator"]
    assert est == "episode_cluster_bootstrap", f"{f}: estimator is {est!r}"
    assert d["intervals"]["point_estimate"].startswith("full_set"), f
    print(f"[ff-chain] post-gate OK  {os.path.basename(f)}  tier={d['tier']}  "
          f"n={d['n_windows']}  ade={d['intervals']['metrics']['ade_dense_m']['mean']}  "
          f"unavailable={fam['_families_unavailable']}")
PYEOF

say "artifacts:"
ls -la "$OUT" | sed 's/^/[ff-chain] /'
say "⭐ BANK THESE: copy $OUT/*.json back to the repo under"
say "   'TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-07-hierarchical-wm-redesign/'"
say "   and git add them. An artifact on ONE disk is NOT done."
echo "FF_EXIT=0"
exit 0
