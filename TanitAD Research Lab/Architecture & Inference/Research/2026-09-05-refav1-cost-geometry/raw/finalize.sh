#!/usr/bin/env bash
# Produce every §7 artifact from whatever arms have landed. Idempotent; skips
# arms whose record is absent and SAYS so, rather than silently thinning the
# panel. Zero GPU.
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f407bc82-7969-457c-a947-6be2014fee89/scratchpad"
P="C:/Users/Admin/refav1_margin/p4out"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4

ARMS="ccos_argmax ccos_seed1 cos_argmax cos_wk wk15 wk151 ccosh_w000 l3ladder kamm07 combined"

# ---- 1. four families, per arm ------------------------------------------- #
{
  echo "# Four-family tables, MEASURED. Every arm carries its (metric, W_JERK, W_KAPPA, W_VEND)."
  echo "# ADE alone is INCOMPLETE (binding). STRATEGIC and distance-keeping are declared"
  echo "# UNAVAILABLE with their reason and n, per clause 5 - never silently dropped."
  echo
  for a in $ARMS; do
    if [ -s "$P/rec_$a.json" ]; then "$PY" "$SP/four_family_table.py" "$P/rec_$a.json"
    else echo "ARM $a  -- RECORD ABSENT (not run, or still running)"; echo; fi
  done
} > "$SP/four_family_all.txt" 2>&1

# ---- 2. realised curvature by decoded goal + GT --------------------------- #
{
  echo "# Realised planner curvature by DECODED GOAL TOKEN and by GROUND TRUTH."
  echo "# Vocabulary IMPORTED from tanitad.models.vocab_v7. dir_correct is meaningful"
  echo "# only on the GT-turn row."
  echo
  for a in $ARMS; do
    if [ -d "$P/dump_$a/decisions" ]; then "$PY" "$SP/kappa_by_goal.py" "$P/dump_$a" "$a"
    else echo "== $a -- DUMP ABSENT"; echo; fi
  done
} > "$SP/kappa_by_goal_all.txt" 2>&1

# ---- 3. the cost scale + the quantisation --------------------------------- #
{
  echo "# Goal-term scale per arm, read off plan_cost / basecost_cv in the sidecars."
  echo
  for a in $ARMS; do
    if [ -d "$P/dump_$a/decisions" ]; then "$PY" "$SP/cost_scale.py" "$P/dump_$a" "$a"; echo
    else echo "== $a -- DUMP ABSENT"; echo; fi
  done
} > "$SP/cost_scale_all.txt" 2>&1

"$PY" - <<'PYEOF' > "$SP/kappa_quantisation_all.txt" 2>&1
import glob, os
import numpy as np
P = "C:/Users/Admin/refav1_margin/p4out"
print("# Is the winning plan just the decoded token's canonical profile?")
print("# EXACTLY-constant curvature series over the 2 s horizon, and the distinct values.")
print()
for a in ("ccos_argmax", "ccos_seed1", "cos_argmax", "cos_wk", "wk15", "wk151",
          "ccosh_w000", "l3ladder", "kamm07"):
    fs = sorted(glob.glob(os.path.join(P, "dump_" + a, "decisions", "*.npz")))
    if not fs:
        print("== %s -- DUMP ABSENT\n" % a); continue
    K = [np.load(f, allow_pickle=True)["cl_controls"][..., 1] for f in fs]
    kap = np.concatenate(K, 0)
    absk = np.abs(kap).max(-1)
    rng = kap.max(-1) - kap.min(-1)
    u, c = np.unique(np.round(absk, 6), return_counts=True)
    print("== %s  n=%d  distinct max|k| = %d  EXACTLY-constant frac = %.4f"
          % (a, len(absk), len(u), float((rng == 0).mean())))
    for v, n in zip(u, c):
        print("    %.6f  x%d" % (v, n))
    print()
PYEOF

# ---- 4. feasibility / Kamm audit ------------------------------------------ #
{
  echo "# L4 audit via tanitad.refs.feasible_decode.assert_feasible (the SCORER's own flags)."
  echo "# CONTROL: the ground-truth path 'g' must read envelope 0.0000 / kamm_over 0.0000."
  echo "# At vmin=0 it does NOT (the kappa = a_lat/v^2 singularity) -> that block is INADMISSIBLE."
  echo
  for v in 0 2 5; do
    args=""
    for a in $ARMS; do [ -d "$P/dump_$a" ] && args="$args $P/dump_$a $a"; done
    FEAS_VMIN=$v "$PY" "$SP/feas_audit.py" $args
  done
} > "$SP/feas_audit_all.txt" 2>&1

# ---- 5. the PAIRED cross-arm deltas, four families ------------------------ #
# ⛔ ONLY COMPLETE ARMS MAY BE PAIRED. A dump with fewer than 8 episode files is a
# DIFFERENT PANEL, and pairing it against a full arm compares two window grids.
dumps=""
pairs=""
for a in $ARMS; do
  n=$(ls "$P/dump_$a"/ep*.npz 2>/dev/null | wc -l)
  if [ "${n:-0}" -eq 8 ]; then dumps="$dumps --dump $a=$P/dump_$a"
  else echo "PAIRING SKIPS $a (n_ep=${n:-0}, not 8)"; fi
done
for a in cos_wk wk15 wk151 ccosh_w000 l3ladder kamm07 combined ccos_seed1 cos_argmax; do
  n=$(ls "$P/dump_$a"/ep*.npz 2>/dev/null | wc -l)
  [ "${n:-0}" -eq 8 ] && pairs="$pairs --pair $a-ccos_argmax"
done
# shellcheck disable=SC2086
"$PY" "$SP/gm_paired_delta.py" $dumps $pairs \
  --stack C:/Users/Admin/tanitad-wt/stack --taniteval C:/Users/Admin/tanitad-wt/taniteval \
  --out "$SP/pd_all.json" --md "$SP/pd_all.md" > "$SP/pd_all.log" 2>&1
echo "FINALIZE-DONE $(date -u +%FT%TZ)"
