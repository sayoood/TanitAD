"""Stage the PICKRULE-AUDIT-NC package in C:/Users/Admin/par18/pickrule/out/<repo path>/ with paths.json ([repo path, abs source, "raw"]), MANIFEST.md and MANIFEST.sha256.json.
Requires RESULT.md (hand-written) and run/results/result.json.  Re-runs the synthetic suite and the synthetic false-fire script unless --no-tests.  Stage, never push: no git add / commit.
    python build_package.py [--no-tests]
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = "C:/Users/Admin/par18/pickrule"
RUN = f"{ROOT}/run"
CODE = f"{ROOT}/code"
OUT = f"{ROOT}/out"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-pickrule-audit-nc"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    no_tests = "--no-tests" in sys.argv
    for need in (f"{ROOT}/RESULT.md", f"{RUN}/results/result.json"):
        if not os.path.exists(need):
            raise SystemExit(f"{need} missing")
    logs = f"{RUN}/test_logs"
    os.makedirs(logs, exist_ok=True)
    if not no_tests:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="8")
        with open(f"{logs}/test_pickrule.log", "w", encoding="utf-8", newline="\n") as f:
            r = subprocess.run([PY, "-W", "ignore", "-m", "unittest", "-v", "test_pickrule"], cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise SystemExit("test_pickrule FAILED: not staging")
        with open(f"{logs}/null_false_red_synthetic.log", "w", encoding="utf-8", newline="\n") as f:
            subprocess.run([PY, "-I", "-W", "ignore", f"{CODE}/null_false_red_synthetic.py"], cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    # EP columns (the four, float32 [15808, 64]) from the prepare arrays
    z = np.load(f"{RUN}/prepare/arrays.npz", allow_pickle=False)
    cols = {k: z[k] for k in z.files if k.startswith("ep_")}
    os.makedirs(f"{RUN}/stage", exist_ok=True)
    np.savez(f"{RUN}/stage/ep_columns.npz", log_idx=z["log_idx"], **cols)
    sealed = []
    for n in ("PREREG_PICKRULE_AUDIT_NC", "PREREG_PICKRULE_AUDIT_NC_AMENDMENT1", "PREREG_PICKRULE_AUDIT_NC_AMENDMENT2"):
        sealed.append(f"{n}.md sealed_at {open(f'{ROOT}/{n}.sealed_at.txt', encoding='utf-8').read().strip() if os.path.exists(f'{ROOT}/{n}.sealed_at.txt') else open(f'{ROOT}/PREREG_PICKRULE_AUDIT_NC.sealed_at.txt', encoding='utf-8').read().strip()}  sha256 {sha(f'{ROOT}/{n}.md')}")
    with open(f"{RUN}/stage/SEALED_AT.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sealed) + "\n")
    # the pinned planner lines
    sys.path.insert(0, CODE)
    import pickrule_lib as P
    pin = P.check_planner_pin(f"{P.REF}/planner.py")
    with open(f"{RUN}/stage/planner_lines_558_561.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(pin["rule_lines_text"])
    refs = {"tip_commit": "993a78cfc3a809b91748fcdd7fb8cf213286893d", "branch": "agent/arch-inf-20260803", "planner_py_git_blob": "6fb6598a2ecf230153a4d7adfb790afc30728070", "planner_py_sha256": P.PLANNER_SHA256,
            "planner_lines_558_561_sha256": P.PLANNER_RULE_LINES_SHA256, "epr1_py_sha256": P.EPR1_SHA256, "fastneg_lib_py_sha256": P.FASTNEG_LIB_SHA256,
            "epr1_result_json_sha256": sha(f"{P.REF}/epr1/epr1_result.json"), "export_method": "git show / git cat-file -p FETCH_HEAD:<path> (byte-exact)"}
    with open(f"{RUN}/stage/REFS.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(refs, indent=1, sort_keys=True))
    code_files = ["pickrule_lib.py", "pickrule_run.py", "test_pickrule.py", "gen_tables.py", "posthoc_diag.py", "v1_applicability_check.py", "build_package.py", "null_false_red_synthetic.py"]
    with open(f"{RUN}/stage/CODE_SHA256.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({n: sha(f"{CODE}/{n}") for n in code_files}, indent=1, sort_keys=True))
    plan = []                                                   # (repo-relative path, source)
    for n in ("PREREG_PICKRULE_AUDIT_NC.md", "PREREG_PICKRULE_AUDIT_NC.sha256", "PREREG_PICKRULE_AUDIT_NC_AMENDMENT1.md", "PREREG_PICKRULE_AUDIT_NC_AMENDMENT1.sha256",
              "PREREG_PICKRULE_AUDIT_NC_AMENDMENT2.md", "PREREG_PICKRULE_AUDIT_NC_AMENDMENT2.sha256", "RESULT.md"):
        plan.append((n, f"{ROOT}/{n}"))
    plan.append(("SEALED_AT.txt", f"{RUN}/stage/SEALED_AT.txt"))
    for n in code_files:
        plan.append((f"code/{n}", f"{CODE}/{n}"))
    plan.append(("code/CODE_SHA256.json", f"{RUN}/stage/CODE_SHA256.json"))
    plan += [("raw/inputs/prepare_report.json", f"{RUN}/prepare/prepare_report.json"), ("raw/inputs/oof_rows_seed0.npy", f"{RUN}/prepare/oof_rows_seed0.npy"),
             ("raw/inputs/oof_rows_seed1.npy", f"{RUN}/prepare/oof_rows_seed1.npy"), ("raw/inputs/ep_columns.npz", f"{RUN}/stage/ep_columns.npz"),
             ("raw/refs/REFS.json", f"{RUN}/stage/REFS.json"), ("raw/refs/planner_lines_558_561.txt", f"{RUN}/stage/planner_lines_558_561.txt")]
    for n in ("result.json", "gate_report.json", "gate_report_attempt1_C-ID3_FAIL.json", "RESULT_tables.md", "posthoc_diag.json", "v1_applicability.json"):
        plan.append((f"raw/results/{n}", f"{RUN}/results/{n}"))
    for src, dst in (("prepare.log", "prepare.log"), ("analyze.log", "analyze.log"), ("analyze_attempt1.log", "analyze_attempt1_gate_stop.log"), ("test_logs/test_pickrule.log", "test_pickrule.log"),
                     ("test_logs/null_false_red_synthetic.log", "null_false_red_synthetic.log")):
        if os.path.exists(f"{RUN}/{src}"):
            plan.append((f"raw/logs/{dst}", f"{RUN}/{src}"))
    base = f"{OUT}/{REPO}"
    shutil.rmtree(OUT, ignore_errors=True)
    rows, paths = [], []
    for rel, src in plan:
        dst = f"{base}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if rel.endswith((".md", ".json", ".txt", ".log", ".py", ".sha256")):
            raw = open(src, "rb").read().replace(b"\r\n", b"\n")                  # LF files: Windows stdout redirection writes CRLF into logs and tables
            with open(dst, "wb") as f:
                f.write(raw)
        else:
            shutil.copyfile(src, dst)
        rows.append((rel, sha(dst), os.path.getsize(dst), src))
        paths.append([f"{REPO}/{rel}", dst.replace("\\", "/"), "raw"])
    tab = ["# MANIFEST: PICKRULE-AUDIT-NC (staged %d files + this manifest)" % len(rows), "",
           f"Repo target: `{REPO}/`. Staged under `{OUT}/`; `paths.json` is the landing list `[repo path, abs source, \"raw\"]`. Nothing is committed or pushed. Full sha256 of every file: `MANIFEST.sha256.json`.", "",
           "| repo path | sha256 | bytes | lives ALSO at |", "|---|---|---|---|"]
    for rel, h, n, src in rows:
        tab.append(f"| `{rel}` | `{h[:16]}...` | {n:,} | `{src}` |")
    tab += ["", "**Held only on the dev box (NOT staged; regenerable by `python pickrule_run.py prepare`, deterministic):** `C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz` (66 MB: the bank's stored logits, `labels6`, top8, the four EP columns; sha256 in `raw/inputs/prepare_report.json`). "
            "`ref/` (byte-exact exports of the tip's `epr1.py`, `epr1_result.json`, the census code and `planner.py`, all pinned by sha256 in `raw/refs/REFS.json`).", "",
            "**Name mapping:** the prereg's `scratch/fp_rate.py` is staged as `code/null_false_red_synthetic.py` (its log: `raw/logs/null_false_red_synthetic.log`).", "",
            "**Not staged by design:** the S0 bank (`C:/Users/Admin/srttc_s0/`, read only) and the 518 MB feature tensor (never copied)."]
    with open(f"{base}/MANIFEST.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(tab) + "\n")
    paths.append([f"{REPO}/MANIFEST.md", f"{base}/MANIFEST.md".replace("\\", "/"), "raw"])
    allsha = {rel: h for rel, h, n, s in rows}
    allsha["MANIFEST.md"] = sha(f"{base}/MANIFEST.md")
    with open(f"{base}/MANIFEST.sha256.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(allsha, indent=1, sort_keys=True))
    paths.append([f"{REPO}/MANIFEST.sha256.json", f"{base}/MANIFEST.sha256.json".replace("\\", "/"), "raw"])
    with open(f"{OUT}/paths.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(paths, indent=1))
    print(f"staged {len(paths)} files under {base}")


if __name__ == "__main__":
    main()
