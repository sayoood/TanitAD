"""Deliverable manifest with BLOB verification: for every deliverable, the git blob id STAGED in
the index must equal ``git hash-object`` of the working file. Prints a markdown table.

Run from anywhere:  python verify_manifest.py [--json out.json]
"""
import argparse
import json
import os
import subprocess

WT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
PKG = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-15-ddv2-rl-prep"
DELIVERABLES = [
    f"{PKG}/SPEC_DDV2_RL_PAPER.md", f"{PKG}/DIFF_OURS_VS_SPEC.md", f"{PKG}/RESULT.md",
    "Project Steering/PREREG_DDV2_RL_VALIDATION.md",
    "stack/tanitad/rl/ddv2_rl.py", "stack/tanitad/rl/ddv2_refc_chain.py", "stack/tanitad/rl/pdm_proxy.py",
    "stack/scripts/ddv2_rl_refcv5.py",
    "stack/tests/test_ddv2_rl.py", "stack/tests/test_ddv2_refc_chain.py", "stack/tests/test_pdm_proxy.py",
    "stack/tests/fixtures/ddv2_released_scheduler.py",
    f"{PKG}/code/ddv2_step_sensitivity.py", f"{PKG}/code/check_arm.py", f"{PKG}/code/run_validation.sh",
    f"{PKG}/code/run_validation_resume.sh",
    f"{PKG}/code/mutation_sweep.py", f"{PKG}/code/analyse_validation.py", f"{PKG}/code/verify_manifest.py",
    f"{PKG}/raw/ddv2_step_sensitivity.txt", f"{PKG}/raw/SPLIT_eval141_sha12.json", f"{PKG}/raw/ddv2_diagnose.json",
    f"{PKG}/raw/MUTATION_SWEEP.json", f"{PKG}/raw/VALIDATION_ANALYSIS.json",
    f"{PKG}/raw/CODE_MANIFEST_at_launch.md5",
    f"{PKG}/raw/metrics_arm-rl-s0.jsonl", f"{PKG}/raw/metrics_arm-norl-s0.jsonl", f"{PKG}/raw/metrics_arm-rl-s1.jsonl",
    f"{PKG}/raw/heldout_base.json", f"{PKG}/raw/heldout_base_repeat.json", f"{PKG}/raw/heldout_arm-rl-s0.json",
    f"{PKG}/raw/heldout_arm-norl-s0.json", f"{PKG}/raw/heldout_arm-rl-s1.json",
]


def git(*args):
    env = dict(os.environ)
    env.pop("GIT_DIR", None)
    p = subprocess.run(["git", "-C", WT, *args], capture_output=True, text=True, env=env)
    return p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = []
    for rel in DELIVERABLES:
        full = os.path.join(WT, rel)
        exists = os.path.exists(full)
        staged = git("ls-files", "-s", "--", rel)
        staged_blob = staged.split()[1] if staged else None
        work_blob = git("hash-object", "--", rel) if exists else None
        rows.append({"path": rel, "exists": exists, "staged_blob": staged_blob, "worktree_blob": work_blob,
                     "blob_match": bool(staged_blob and staged_blob == work_blob),
                     "bytes": os.path.getsize(full) if exists else None})
    print("| repo path | bytes | staged blob | worktree blob | match |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| `{r['path']}` | {r['bytes']} | {(r['staged_blob'] or '-')[:12]} | "
              f"{(r['worktree_blob'] or '-')[:12]} | {'YES' if r['blob_match'] else 'NO'} |")
    print(f"\nall match: {all(r['blob_match'] for r in rows)} ({sum(r['blob_match'] for r in rows)}/{len(rows)})")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
