"""NC-FASTCAL: stage the package under out/<repo path>, write CODE_SHA256.json, REFS.json, MANIFEST.md and paths.json.  Copies are verified by sha256 against their sources (a truncated copy is an error).  LF everywhere."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

ROOT = "C:/Users/Admin/par18/ncfastcal"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-nc-fastcal"
OUT = f"{ROOT}/out/{REPO}"
CODE = ["ncfastcal_lib.py", "ncfastcal_run.py", "test_ncfastcal.py", "posthoc_diag.py", "planted_power_report.py", "gen_tables.py", "build_package.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    code_sha = {n: sha(f"{ROOT}/code/{n}") for n in CODE}
    with open(f"{ROOT}/code/CODE_SHA256.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(code_sha, indent=1, sort_keys=True))
    refs = {"prereg_sha256": sha(f"{ROOT}/PREREG_NC_FASTCAL.md"), "inputs_read_only": {
        "pickrule arrays.npz (C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz)": sha("C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz"),
        "pickrule prepare_report.json": sha("C:/Users/Admin/par18/pickrule/run/prepare/prepare_report.json"),
        "pickrule result.json (landed audit, C-ID c)": sha("C:/Users/Admin/par18/pickrule/run/results/result.json"),
        "census speeds.npz (exported byte-exact from the tip, raw/strata/speeds.npz)": sha(f"{ROOT}/ref/speeds.npz"),
        "census strata.json (run/strata/strata.json)": sha(f"{ROOT}/ref/strata.json")},
        "pinned_landed_code": {"pickrule_lib.py": sha(f"{ROOT}/ref/pickrule_lib.py"), "epr1.py": sha(f"{ROOT}/ref/epr1.py"), "fastneg_lib.py": sha(f"{ROOT}/ref/fastneg_lib.py")},
        "tip": "agent/arch-inf-20260803 FETCH_HEAD 51099763b122d4a421d00d369f6822a359d89974"}
    os.makedirs(f"{ROOT}/run/refs", exist_ok=True)
    with open(f"{ROOT}/run/refs/REFS.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(refs, indent=1, sort_keys=True))
    plan = [("PREREG_NC_FASTCAL.md", f"{ROOT}/PREREG_NC_FASTCAL.md"), ("PREREG_NC_FASTCAL.sha256", f"{ROOT}/PREREG_NC_FASTCAL.sha256"),
            ("PREREG_NC_FASTCAL.sealed_at.txt", f"{ROOT}/PREREG_NC_FASTCAL.sealed_at.txt"), ("RESULT.md", f"{ROOT}/RESULT.md")]
    plan += [(f"code/{n}", f"{ROOT}/code/{n}") for n in CODE + ["CODE_SHA256.json"]]
    plan += [(f"raw/results/{n}", f"{ROOT}/run/results/{n}") for n in ("gate_report.json", "result.json", "posthoc_diag.json", "RESULT_tables.md", "nc_cal_oof.npz")]
    plan += [("raw/logs/analyze.log", f"{ROOT}/run/analyze.log"), ("raw/logs/analyze_repro.log", f"{ROOT}/run/analyze_repro.log"), ("raw/logs/posthoc_diag.log", f"{ROOT}/run/posthoc_diag.log"),
             ("raw/logs/planted_power_report.log", f"{ROOT}/run/planted_power_report.log"), ("raw/logs/test_ncfastcal.log", f"{ROOT}/run/test_logs/test_ncfastcal.log"),
             ("raw/logs/test_ncfastcal_attempt1.log", f"{ROOT}/run/test_logs/test_ncfastcal_attempt1.log"), ("raw/refs/REFS.json", f"{ROOT}/run/refs/REFS.json")]
    rows = []
    for rel, src in plan:
        dst = f"{OUT}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        if sha(dst) != sha(src) or os.path.getsize(dst) != os.path.getsize(src):
            raise RuntimeError(f"copy of {src} is not complete")
        if not rel.endswith(".npz") and b"\r" in open(dst, "rb").read():
            raise RuntimeError(f"{rel} carries CR")
        rows.append((rel, os.path.getsize(dst), sha(dst), src))
    lines = ["# MANIFEST: NC-FASTCAL (Research Lab agent, 2026-10-19)\n",
             f"Repo path: `{REPO}/`. Staged (NOT committed, NOT pushed) under `{ROOT}/out/`; `paths.json` is the landing list `[repo path, abs source, \"raw\"]`. Every copy was sha256-verified against its source.\n",
             "| file (relative to the package) | bytes | sha256 | lives in |", "|---|---|---|---|"]
    for rel, n, h, src in rows:
        lines.append(f"| `{rel}` | {n} | `{h}` | repo (staged) + dev box `{src}` |")
    lines += ["", "Inputs read, never copied or modified (dev box only, regenerable): `C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz` (the audit's prepare output), `.../prepare_report.json`, `.../run/results/result.json`;",
              "landed census `speeds.npz` / `strata.json` and landed code `pickrule_lib.py`, `epr1.py`, `fastneg_lib.py` (already in the repo; hashes in `raw/refs/REFS.json`). Single-copy items: none; `nc_cal_oof.npz` is regenerable by `ncfastcal_run.py analyze` (deterministic, re-run identical).", ""]
    with open(f"{OUT}/MANIFEST.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    paths = [[f"{REPO}/{rel}", f"{OUT}/{rel}", "raw"] for rel, *_ in rows] + [[f"{REPO}/MANIFEST.md", f"{OUT}/MANIFEST.md", "raw"]]
    with open(f"{ROOT}/out/paths.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(paths, indent=1))
    print(f"staged {len(paths)} files under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
