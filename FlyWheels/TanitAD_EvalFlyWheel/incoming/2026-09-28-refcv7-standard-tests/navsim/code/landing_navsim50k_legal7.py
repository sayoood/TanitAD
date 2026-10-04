#!/usr/bin/env python3
"""Write ``LANDING_READY_NAVSIM50K_LEGAL.txt`` for the Master Mind (single committer). Any python. Never runs git add.

    python code/landing_navsim50k_legal7.py          # idempotent: rewrite after every arm that completes

Line format (the brief):  ``<md5>  <repo-relative path>  -- <what>``  (repo root = D:/Projects/TanitAD).
Scope: (1) the code/test deliverables of the 2026-10-04 RAM-governor + LEGAL-row work package; (2) the step-50,400
evidence of every scorer whose counts read PASS (counts, csv, wrapper score frames, manifests, driver / governor
logs) plus the runner / queue logs; (3) the LEGAL-row (R7_VMAXOFF) bridge manifests, scores and summaries.
REFUSED and listed (never landed): a file over 20 MB, and a file whose text carries a canonical UUID
(8-4-4-4-12 hex): the devkit scorer logs print hydra thread ids of that shape, so their evidence is carried by
the counts.json beside them (clip-id rule, CLAUDE.md).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = "D:/Projects/TanitAD"
MS = os.path.join(PKG, "raw", "milestones", "step50400")
MAX_BYTES = 20 * 2 ** 20
UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
EXCLUDE = ("*/__pycache__/*", "*.pyc", "*.tmp", "*/_gov_smoke_scratch/*", "*/_parse_stage_navhard/*",
           "*.nohup.txt", "*/queue.stdout.txt")
CODE = {
    "code/ram_governor7.py": "NEW: RAM governor (pause-instead-of-abort) + scoring-slot lock for the scorer drivers",
    "code/score_queue_gov7.py": "NEW: sequential governed scoring queue over existing seams",
    "code/run_vmaxoff_legal7.py": "NEW: refcv7-50,400 LEGAL-input row = R7_VMAXOFF on full navtest + navhard (bridge + score)",
    "code/vmaxoff_legal_summary7.py": "NEW: summary of the LEGAL row vs STOP/CV/HUMAN/ECHO and vs R7_A1 (map speed)",
    "code/landing_navsim50k_legal7.py": "NEW: writes this landing list",
    "code/status50k7.py": "NEW: one-screen status of the step-50,400 scoring (artifacts only)",
    "code/landing_refresher7.py": "NEW: re-runs this landing list every 10 min until both finishers are done",
    "code/finish_step50400_7.py": "NEW: event-driven finisher (re-score leftovers, warmup + all splits post-processed, MILESTONE_SUMMARY, BARS, compares) and LEGAL-row summary trigger",
    "COMMS.md": "MODIFIED: appended the 2026-10-04 evening section (RAM governor, warmup gap, LEGAL row, restart recipe)",
    "raw/milestones/step50400/runner_arms_queue.sh": "launch script of the governed queue over the runner's still-unscored navtest / navhard arms",
    "code/score_arm7.py": "MODIFIED: scorer driver wrapped in ram_governor7 (+ governor summary in counts.json); abs wrapper out-dir fix",
    "code/score_navtest7.py": "MODIFIED: scorer driver wrapped in ram_governor7 (+ governor sidecar json)",
    "tests/test_ram_governor7.py": "NEW: 26 tests incl. real-process pause/resume, PASS-skip guard, per-job lock; 6 mutation-checked properties",
    "raw/milestones/driver_backups_pre_governor/score_arm7.py.pre_governor": "provenance: score_arm7.py as it ran until 2026-10-04 22:00 Berlin",
    "raw/milestones/driver_backups_pre_governor/score_navtest7.py.pre_governor": "provenance: score_navtest7.py as it ran until 2026-10-04 22:00 Berlin",
}
#: evidence file patterns inside step50400 (relative to MS) and what they are
EVIDENCE = (
    ("runner.log", "step-50,400 runner log (UTC)"),
    ("CLAIMED_ARMS.txt", "dev-box / Thor arm claims (coordination file)"),
    ("scores_*/queue.log", "governed scoring-queue log (UTC)"),
    ("scores_*/*.driver.txt", "scorer driver stdout/stderr"),
    ("scores_*/*.counts.json", "count guard + governor summary of one scorer run"),
    ("scores_*/*.csv", "official scorer CSV (kept only beside a PASS count guard)"),
    ("scores_*/*.governor.json*", "governor log / summary of one scorer run"),
    ("scores_*/*_wrapper/*final_scores_frame*.csv", "wrapper score frame (per-scene sub-scores)"),
    ("scores_*/*_wrapper/*_manifest.json", "wrapper manifest (resources, RAM guard, patches)"),
    ("scores_*/*_wrapper/*_hooks.json", "wrapper observation hooks (per-token agent poses / timings)"),
    ("scores_*/*/*.counts.json", "count guard + governor summary of one navtest scorer run"),
    ("scores_*/*/*.csv", "official navtest scorer CSV (kept only beside a PASS count guard)"),
    ("scores_*/*/*_manifest.json", "navtest wrapper manifest"),
    ("scores_*/*/*_hooks.json", "navtest wrapper observation hooks"),
    ("scores_*/*/*.governor.json*", "governor log / summary of one navtest scorer run"),
    ("vmaxoff_legal/*.log", "LEGAL-row orchestrator / bridge logs"),
    ("vmaxoff_legal/*.json", "LEGAL-row manifest / summaries"),
    ("vmaxoff_legal/bridge_*/seam_*.manifest.json", "LEGAL-row seam manifest"),
    ("vmaxoff_legal/bridge_*/seam_*.npz", "LEGAL-row seam (the plan the scorer consumed)"),
    ("vmaxoff_legal/scores_*", None),
    ("bridge_warmup/seam_*.manifest.json", "warmup seam manifest"),
)


def md5_of(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def counts_pass_for(path: str) -> bool | None:
    """For a scorer artifact, does the PASS count guard beside it read PASS? None = not a scorer artifact /
    no guard found."""
    d, n = os.path.split(path)
    if os.path.basename(d).endswith("_wrapper"):
        root = os.path.dirname(d)
        c = os.path.join(root, os.path.basename(d)[:-len("_wrapper")] + ".counts.json")
    else:
        base = re.sub(r"(_hooks\.json|_manifest\.json|_rows\.jsonl|\.calls\.jsonl|\.governor\.jsonl|"
                      r"\.governor\.json|\.log|\.csv)$", "", n)
        c = os.path.join(d, base + ".counts.json")
    if not os.path.exists(c):
        return None
    try:
        return json.load(open(c, encoding="utf-8")).get("status") == "PASS"
    except (OSError, ValueError):
        return False


def main() -> int:
    rows, refused, seen = [], [], set()

    def add(full: str, what: str) -> None:
        full = os.path.abspath(full).replace("\\", "/")
        if full in seen or not os.path.isfile(full):
            return
        rel_pkg = os.path.relpath(full, PKG).replace("\\", "/")
        if any(fnmatch.fnmatch(rel_pkg, g) or fnmatch.fnmatch("/" + rel_pkg, "*/" + g) for g in EXCLUDE):
            return
        seen.add(full)
        size = os.path.getsize(full)
        if size > MAX_BYTES:
            refused.append((rel_pkg, f"{size / 2 ** 20:.1f} MB > 20 MB"))
            return
        raw = open(full, "rb").read()
        if UUID.search(raw):
            refused.append((rel_pkg, "carries a canonical UUID (devkit thread id / clip id)"))
            return
        repo_rel = os.path.relpath(full, ROOT).replace("\\", "/")
        rows.append(f"{hashlib.md5(raw).hexdigest()}  {repo_rel}  -- {what}")

    for rel, what in CODE.items():
        add(os.path.join(PKG, rel), what)
    import glob
    for pat, what in EVIDENCE:
        if what is None:
            continue
        for full in sorted(glob.glob(os.path.join(MS, pat))):
            full = full.replace("\\", "/")
            ok = counts_pass_for(full)
            if ok is False and (full.endswith((".csv", "_hooks.json")) or "final_scores_frame" in full):
                refused.append((os.path.relpath(full, PKG).replace("\\", "/"),
                                "beside a count guard that does not read PASS (partial work)"))
                continue
            add(full, what)
    lines = ["## 2026-10-04 REFCV7-NAVSIM-50K-LEGAL",
             "# EvalFlyWheel NavSim operator: RAM governor for the step-50,400 scorers + the LEGAL-input row",
             "# (R7_VMAXOFF = bare command, unknown speed row) for SPEC_REFCV8 BAR-R8-N4.",
             f"# Generated {time.strftime('%Y-%m-%d %H:%M:%S')} Berlin by code/landing_navsim50k_legal7.py; re-run after each arm.",
             "# Format: <md5>  <repo path relative to D:/Projects/TanitAD>  -- <what>"] + rows
    if refused:
        lines.append("# REFUSED (not landed):")
        lines += [f"#   {p}: {why}" for p, why in refused]
    out = os.path.join(PKG, "LANDING_READY_NAVSIM50K_LEGAL.txt")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(rows)} files listed, {len(refused)} refused -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
