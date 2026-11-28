#!/usr/bin/env python3
"""Writes MANIFEST.txt: every deliverable in the package with sha256, bytes, role and the suggested repo path (raw/2026-10-27-dagger-families/ under the v8f-mini package).
ref/ (exports of tip files used as inputs) and logs are listed as NOT-FOR-LANDING / optional."""
import hashlib
import os

D = "E:/Projects/TanitAD-artifacts/dagfam"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = PK + "/raw/2026-10-27-dagger-families"
ROLES = [
    ("FAMILIES_DAG.md", "the document: four families x two tiers x two seeds, controls, not-computed table (generated; no hand-typed numbers)", True),
    ("DAG_FAMILIES.json", "raw: LONGITUDINAL / LATERAL / TACTICAL(plan-derived) / DISTANCE KEEPING, DAG-DAG0 at T0 + T1-kin, DAG-M2b at T0 (dagfam_families.py)", True),
    ("DAG_T1CELLS.json", "raw: R-TLAT/R-TLON reproduction, DAG-ORC, per-class lat F1, paired S-POS at T1-kin (dagfam_t1cells.py)", True),
    ("PAIRFAM_T0_DAG-DAG0_s0.json", "raw: unchanged CLI v8f_paired_families.py, TACTICAL heads + goal + STRATEGIC at T0, DAG vs DAG-0 s0", True),
    ("PAIRFAM_T0_DAG-DAG0_s1.json", "raw: same, s1", True),
    ("PAIRFAM_T0_DAG-M2b_s0.json", "raw: same, DAG vs M2b s0", True),
    ("PAIRFAM_T0_DAG-M2b_s1.json", "raw: same, DAG vs M2b s1", True),
    ("DAG_FAMILIES_ROWS.json", "every table cell as a record (contrast, family, metric, tier, seed, delta, lo, hi, verdict, levels, n, source key)", True),
    ("dagfam_families.py", "driver 1 (orc_families.py re-parametrised)", True),
    ("dagfam_t1cells.py", "driver 2 (T1-HEADS reducer: head F1, per-class F1, paired S-POS)", True),
    ("run_pairfam.sh", "driver 3 (the unchanged paired-families CLI, 4 runs)", True),
    ("build_families_md.py", "generator of FAMILIES_DAG.md and DAG_FAMILIES_ROWS.json", True),
    ("verify_families.py", "independent checks (document vs rows, 30 cells recomputed directly from the raw npz, semantics, hygiene)", True),
    ("verify_families.out", "output of verify_families.py", True),
    ("make_manifest.py", "this script", True),
    ("out/PAIRFAM_T0_DAG-DAG0_s0.md", "the CLI's own markdown (optional)", False),
    ("out/PAIRFAM_T0_DAG-DAG0_s1.md", "the CLI's own markdown (optional)", False),
    ("out/PAIRFAM_T0_DAG-M2b_s0.md", "the CLI's own markdown (optional)", False),
    ("out/PAIRFAM_T0_DAG-M2b_s1.md", "the CLI's own markdown (optional)", False),
]
lines = ["# MANIFEST of E:/Projects/TanitAD-artifacts/dagfam/ (the DAG 'beside' families package). sha256  bytes  file  =>  suggested repo path (LAND=yes) | optional | NOT-FOR-LANDING",
         "# NOT-FOR-LANDING: ref/ (exports of mirror-tip 7d29172b files used as inputs: M2b banked evals, ORC_FAMILIES.json, verify_numbers.py, MODEL_REGISTRY.md; already on the tip),",
         "#                  logs/ (driver stdout/stderr), _unused.json. No checkpoint, nothing above 20 MB, nothing on C:.", ""]
tot = 0
for f, role, land in ROLES:
    p = f"{D}/{f}"
    if not os.path.isfile(p):
        raise SystemExit("missing " + p)
    raw = open(p, "rb").read()
    assert len(raw) > 0 and b"\r" not in raw, f
    tot += len(raw)
    lines.append(f"{hashlib.sha256(raw).hexdigest()}  {len(raw):>9}  {f}  =>  {RAW + '/' + os.path.basename(f) if land else 'optional: ' + RAW + '/' + f}   # {role}")
lines.append("")
lines.append(f"# {len(ROLES)} files, {tot} bytes. Inputs consumed (not copied): v8f-dagger-t1/runs.json, v8f-dagger-t1/read/DAG_READ.json, the DAG/DAG-0 T0 npz and T1 dumps/ticks listed in runs.json, "
             "v8f-orecover-t1 ORC ticks (for DAG-ORC), sfresh_inputs cache + native_tracks.pt (T1-HEADS Ctx), dagt1_tree_b43c7a9a (tool tree), ro/tree lead block (sha 7a879baa...).")
open(D + "/MANIFEST.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(len(ROLES), tot)
