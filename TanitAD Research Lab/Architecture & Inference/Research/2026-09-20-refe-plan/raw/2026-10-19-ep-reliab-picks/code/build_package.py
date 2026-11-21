"""EP-RELIAB-PICKS: stage the package under out/<repo path>, write CODE_SHA256.json, REFS.json, MANIFEST.md and paths.json.  Copies are verified by sha256 against their sources.  LF everywhere."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

ROOT = "C:/Users/Admin/par18/epreliab"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-ep-reliab-picks"
OUT = f"{ROOT}/out/{REPO}"
CODE = ["epreliab_lib.py", "epreliab_run.py", "test_epreliab.py", "null_false_fire_synthetic.py", "independent_check.py", "gen_tables.py", "build_package.py"]
PREREG = ["PREREG_EP_RELIAB_PICKS.md", "PREREG_EP_RELIAB_PICKS.sha256", "PREREG_EP_RELIAB_PICKS.sealed_at.txt", "PREREG_EP_RELIAB_PICKS_AMENDMENT1.md", "PREREG_EP_RELIAB_PICKS_AMENDMENT1.sha256",
          "PREREG_EP_RELIAB_PICKS_AMENDMENT1.sealed_at.txt"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    code_sha = {n: sha(f"{ROOT}/code/{n}") for n in CODE if n != "build_package.py"}
    res = json.load(open(f"{ROOT}/run/results/result.json", "r", encoding="utf-8"))
    for n in ("epreliab_lib.py", "epreliab_run.py"):
        if res["code_sha256"][n] != code_sha[n]:
            raise RuntimeError(f"{n} changed after the real run: result.json recorded {res['code_sha256'][n]}, now {code_sha[n]}")
    with open(f"{ROOT}/code/CODE_SHA256.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(code_sha, indent=1, sort_keys=True))
    refs = {"prereg_sha256": sha(f"{ROOT}/PREREG_EP_RELIAB_PICKS.md"), "amendment1_sha256": sha(f"{ROOT}/PREREG_EP_RELIAB_PICKS_AMENDMENT1.md"), "inputs_read_only": {
        "pickrule arrays.npz (C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz)": sha("C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz"),
        "pickrule prepare_report.json": sha("C:/Users/Admin/par18/pickrule/run/prepare/prepare_report.json"),
        "pickrule result.json (landed audit, C-ID-LANDED)": sha("C:/Users/Admin/par18/pickrule/run/results/result.json"),
        "ncfastcal posthoc_diag.json (C-ID-R53)": sha("C:/Users/Admin/par18/ncfastcal/run/results/posthoc_diag.json")},
        "pinned_landed_code": {"pickrule_lib.py": sha(f"{ROOT}/ref/pickrule_lib.py"), "epr1.py": sha(f"{ROOT}/ref/epr1.py"), "fastneg_lib.py": sha(f"{ROOT}/ref/fastneg_lib.py"), "planner.py": sha(f"{ROOT}/ref/planner.py")},
        "tip": "agent/arch-inf-20260803 FETCH_HEAD 54815c08014f2edf49ced80209fc15154724a4e4"}
    os.makedirs(f"{ROOT}/run/refs", exist_ok=True)
    with open(f"{ROOT}/run/refs/REFS.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(refs, indent=1, sort_keys=True))
    plan = [(n, f"{ROOT}/{n}") for n in PREREG] + [("RESULT.md", f"{ROOT}/RESULT.md")]
    plan += [(f"code/{n}", f"{ROOT}/code/{n}") for n in CODE + ["CODE_SHA256.json"]]
    plan += [(f"raw/results/{n}", f"{ROOT}/run/results/{n}") for n in ("result.json", "gate_report.json", "RESULT_tables.md", "independent_check.json")]
    plan += [("raw/logs/analyze.log", f"{ROOT}/run/analyze.log"), ("raw/logs/test_epreliab.log", f"{ROOT}/run/test_logs/test_epreliab.log"), ("raw/logs/independent_check.log", f"{ROOT}/run/independent_check.log"),
             ("raw/synthetic/null_false_fire.json", f"{ROOT}/run/synthetic/null_false_fire.json"), ("raw/refs/REFS.json", f"{ROOT}/run/refs/REFS.json")]
    rows = []
    for rel, src in plan:
        dst = f"{OUT}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        if sha(dst) != sha(src) or os.path.getsize(dst) != os.path.getsize(src):
            raise RuntimeError(f"copy of {src} is not complete")
        if b"\r" in open(dst, "rb").read():
            raise RuntimeError(f"{rel} carries CR")
        rows.append((rel, os.path.getsize(dst), sha(dst), src))
    lines = ["# MANIFEST: EP-RELIAB-PICKS (Research Lab agent, 2026-10-19)\n",
             f"Repo path: `{REPO}/`. Staged (NOT committed, NOT pushed) under `{ROOT}/out/`; `paths.json` is the landing list `[repo path, abs source, \"raw\"]`. Every copy was sha256-verified against its source.\n",
             "| file (relative to the package) | bytes | sha256 | lives in |", "|---|---|---|---|"]
    for rel, n, h, src in rows:
        lines.append(f"| `{rel}` | {n} | `{h}` | repo (staged) + dev box `{src}` |")
    lines += ["", "Inputs read, never copied or modified (dev box only, regenerable; hashes in `raw/refs/REFS.json`): the landed audit's `arrays.npz` / `prepare_report.json` / `result.json` under `C:/Users/Admin/par18/pickrule/run/`, "
              "NC-FASTCAL's `posthoc_diag.json`, and the pinned landed code `pickrule_lib.py`, `epr1.py`, `fastneg_lib.py`, `planner.py` (already in the repo). Single-copy items: none.",
              "Landing note for the Master Mind: this package has no integration step (no code to wire in); the GAC / BACKLOG text is in `RESULT.md` section 4."]
    with open(f"{OUT}/MANIFEST.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    paths = [[f"{REPO}/{rel}", f"{OUT}/{rel}", "raw"] for rel, *_ in rows] + [[f"{REPO}/MANIFEST.md", f"{OUT}/MANIFEST.md", "raw"]]
    with open(f"{ROOT}/out/paths.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(paths, indent=1))
    print(f"staged {len(paths)} files under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
