"""Stage the FASTNEG-CENSUS package in C:/Users/Admin/par18/fastneg/out/<repo path>/ with paths.json ([repo path, abs source, "raw"]) and MANIFEST.md.
Requires C:/Users/Admin/par18/fastneg/RESULT.md (hand-written from run/results/RESULT_tables.md) to exist.  Runs the three synthetic test files and saves their logs unless --no-tests.
    python build_package.py [--no-tests]
Stage, never push: no git add / commit.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = "C:/Users/Admin/par18/fastneg"
RUN = f"{ROOT}/run"
CODE = f"{ROOT}/code"
OUT = f"{ROOT}/out"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-fastneg-census"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    no_tests = "--no-tests" in sys.argv
    if not os.path.exists(f"{ROOT}/RESULT.md"):
        raise SystemExit("RESULT.md missing: write it first")
    logs = f"{RUN}/test_logs"
    os.makedirs(logs, exist_ok=True)
    if not no_tests:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="8")
        for t in ("test_fastneg.py", "test_fastneg_more.py", "test_fastneg_census.py"):
            r = subprocess.run([PY, "-W", "ignore", "-I", f"{CODE}/{t}"], capture_output=True, text=True, env=env, cwd=CODE)
            with open(f"{logs}/{t[:-3]}.log", "w", encoding="utf-8", newline="\n") as f:
                f.write(r.stderr[-6000:])
            tail = [ln for ln in r.stderr.strip().splitlines() if ln.startswith(("Ran ", "OK", "FAILED"))]
            print(t, r.returncode, tail)
            if r.returncode != 0:
                raise SystemExit(f"{t} FAILED")
    items = []                                   # (repo-relative path, source)
    add = lambda rel, src: items.append((f"{REPO}/{rel}", src.replace("\\", "/")))               # noqa: E731
    add("PREREG_FASTNEG_CENSUS_AMENDMENT1.md", f"{ROOT}/PREREG_FASTNEG_CENSUS_AMENDMENT1.md")
    add("PREREG_FASTNEG_CENSUS_AMENDMENT1.sha256", f"{ROOT}/PREREG_FASTNEG_CENSUS_AMENDMENT1.sha256")
    add("RESULT.md", f"{ROOT}/RESULT.md")
    code_files = sorted(glob.glob(f"{CODE}/*.py") + glob.glob(f"{CODE}/*.sh"))
    cs = {os.path.basename(p): sha(p) for p in code_files}
    cs_path = f"{ROOT}/CODE_SHA256.json"
    with open(cs_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cs, f, indent=1, sort_keys=True)
    add("code/CODE_SHA256.json", cs_path)
    for p in code_files:
        add(f"code/{os.path.basename(p)}", p)
    for fn in ("strata.json", "strata.json.sha256", "speeds.npz", "strata_label_free_proof.json"):
        add(f"raw/strata/{fn}", f"{RUN}/strata/{fn}")
    add("raw/inputs/inputs_report.json", f"{RUN}/inputs/inputs_report.json")
    add("raw/gate_token.json", f"{RUN}/gate_token.json")
    for fn in ("census.json", "RESULT_tables.md"):
        add(f"raw/results/{fn}", f"{RUN}/results/{fn}")
    for p in sorted(glob.glob(f"{RUN}/fits/*.json")):
        add(f"raw/fits/{os.path.basename(p)}", p)
    for tid in ("L_f100_s0", "L_f100_s1", "L_f25_s0", "L_f25_s1"):
        add(f"raw/fits/{tid}.npy", f"{RUN}/fits/{tid}.npy")
    for fn in ("fit_main.log", "fit_null.log", "analyze.log", "finish.log"):
        if os.path.exists(f"{RUN}/{fn}"):
            add(f"raw/logs/{fn}", f"{RUN}/{fn}")
    for p in sorted(glob.glob(f"{logs}/*.log")):
        add(f"raw/logs/{os.path.basename(p)}", p)
    # copy under out/<repo path>/ (byte-exact)
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    rows = []
    for rel, src in items:
        if not os.path.exists(src):
            raise SystemExit(f"missing source {src}")
        dst = f"{OUT}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        rows.append((rel, src, sha(dst), os.path.getsize(dst)))
    big = [f for f in glob.glob(f"{RUN}/fits/*.npy") if os.path.basename(f)[:-4] not in ("L_f100_s0", "L_f100_s1", "L_f25_s0", "L_f25_s1")]
    man = [f"# MANIFEST: FASTNEG-CENSUS (staged {len(rows)} files)\n", "Repo target: `" + REPO + "/`. Staged under `C:/Users/Admin/par18/fastneg/out/`; `paths.json` is the landing list `[repo path, abs source, \"raw\"]`. Nothing is committed or pushed.\n",
           "| repo path | sha256 | bytes | lives ALSO at |\n|---|---|---|---|"]
    for rel, src, h, n in rows:
        man.append(f"| `{rel[len(REPO) + 1:]}` | `{h[:16]}...` | {n:,} | `{src}` |")
    man.append(f"\n**Held only on the dev box (NOT staged; deterministic CPU refits, regenerable by `fastneg_census.py fit`, each table's sha256 is in its `raw/fits/<id>.json` marker):** {len(big)} OOF tables of 8 MB each in `{RUN}/fits/` "
               f"and the 518 MB fp16 feature copy `{RUN}/inputs/feat_f10.npy` (sha256 in `raw/inputs/inputs_report.json`), `{RUN}/inputs/meta.npz`.\n")
    mp = f"{OUT}/{REPO}/MANIFEST.md"
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(man))
    rows.append((f"{REPO}/MANIFEST.md", mp, sha(mp), os.path.getsize(mp)))
    paths = [[rel, (src if rel.endswith("MANIFEST.md") is False else mp), "raw"] for rel, src, _, _ in rows]
    with open(f"{OUT}/paths.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sorted(paths), f, indent=1)
    print(len(paths), "files staged;", len(big), "tables dev-box only")


if __name__ == "__main__":
    main()
