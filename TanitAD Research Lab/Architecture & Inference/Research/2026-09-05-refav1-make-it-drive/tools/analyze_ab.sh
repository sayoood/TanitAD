#!/usr/bin/env bash
# Analyse the decision-rule A/B with the DECISION-GRADE estimator.
#
# ⛔ RUN THIS, NOT A PER-ARM COMPARISON. `taniteval.ci`'s PAIRED
# episode-cluster bootstrap is the decision-grade form for two arms on the same
# windows (CLAUDE.md); per-arm intervals are not, and combining them in
# quadrature is wrong. `paired_delta_refav1.py` is the instrument the surface
# sweep already used, so this A/B is reported on the same footing as
# `D-REFAV1-SURFACE-PAIRED`.
#
# ⭐ THE GPU IS ALREADY PAID. If the rollout was killed, the dumps under
# $OUT/dump_* are complete for the episodes they reached — analyse them with
# `refav1_arm.py --analyze-only <dump>` rather than re-rolling anything. That
# rule exists because a 3.0 h rollout once died in analyze() and read as a
# total failure.
#
# ⚠️ H-ESTIM-SEED-1: a separated CI from a ONE-SEED arm is NECESSARY, NOT
# SUFFICIENT. These two arms differ in exactly one flag on identical windows,
# which removes episode-draw noise but NOT training/run-to-run noise. Before
# any "the decision rule moved it" claim is quotable, add a REPLICATE arm
# (`--plan-seed 1` on the argmax arm) and read the lever against the rig's own
# noise floor. The iCEM search is stochastic, so a replicate here is cheap and
# is NOT optional.
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_drive/ab"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PKG="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-make-it-drive"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8

"$PY" "$PKG/tools/paired_delta_refav1.py" \
  --dump "argmax=$OUT/dump_argmax" \
  --dump "prior050=$OUT/dump_prior050" \
  --pair "prior050-argmax" \
  --pair "argmax-argmax" \
  --stack "$M/stack" --taniteval "$M/taniteval" \
  --n-boot 2000 --seed 0 \
  --out "$PKG/raw/ab_paired_delta.json" \
  --md  "$PKG/raw/ab_paired_delta.md"
#            ^^^^^^^^^^^^^^^^^^^^^^ `argmax-argmax` is the KNOWN-VALUE control:
#            it must read EXACTLY 0.0000 with a zero-width interval on every
#            metric. If it does not, the estimator or the join is broken and no
#            other row in the table is admissible.
echo "ANALYZE_DONE $(date -u +%FT%TZ)"
