"""Stage the SH5-DAC-PROX package: copy what exists into out/<repo path>/, write out/paths.json ([repo path, abs source, "raw"]) and MANIFEST.md.  Re-runnable.  No git add / commit / push.

SMALL files (code, prereg, READY, jsons, logs) are staged under out/ for the repo.  LARGE payloads (sh5_inputs.npz, sh5_labels.npz and, later, ctx / OOF arrays) live ONLY on D: and are recorded in the package
as raw/ARTIFACTS_D.json with sha256 (re-hashed here), NOT staged for the repo.

    python build_package.py
"""
import glob
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..")).replace("\\", "/")           # Amendment 2: derived from this file's own location (it was hardcoded to par18/sh5, and `main()` removes ROOT/out: running a copy would have deleted the ORIGINAL staging)
BASE = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-sh5-dac-prox"
OUT = f"{ROOT}/out/{BASE}"
ARTIFACTS = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19"
REFDIR = "C:/Users/Admin/par18/privgeom/ref"
ARTIFACTS_A2 = "D:/Projects/TanitAD-artifacts/sh5_a2_2026-10-19"                      # Amendment 2: the pre-amendment (Amendment-1-state) copies of the two files it re-pins
ARTIFACTS_A3 = "D:/Projects/TanitAD-artifacts/sh5_a3_2026-10-19"                      # Amendment 3: the pre-amendment (Amendment-2-state) copies of the two files it re-pins
SRC = [  # (glob under ROOT, destination folder inside the package)
    ("PREREG_SH5_DAC_PROX.md", ""), ("PREREG_SH5_DAC_PROX.sha256", ""), ("SEALED_AT.txt", ""), ("CODE_PINS.json", ""), ("READY.md", ""),
    ("PREREG_SH5_DAC_PROX_AMENDMENT*.md", ""), ("PREREG_SH5_DAC_PROX_AMENDMENT*.sha256", ""), ("CODE_PINS_AMENDMENT*.json", ""),
    ("code/*.py", "code"), ("code/launch.ps1", "code"),
    ("work/amend3_power.json", "raw"), ("work/amend3_guard_proof.json", "raw"), ("work/tests_summary_amend3.json", "raw"), ("work/run_tests_amend3.log", "raw/logs"),
    ("work/amend2_power.json", "raw"), ("work/amend2_guard_proof.json", "raw"), ("work/tests_summary_amend2.json", "raw"), ("work/run_tests_amend2.log", "raw/logs"),
    ("work/atest.json", "raw"), ("work/atest_candidate.json", "raw"), ("work/plant_calibration.json", "raw"), ("work/mde.json", "raw"), ("work/cost.json", "raw"),
    ("work/falsefire_sim.json", "raw"), ("work/falsefire_sim.log", "raw/logs"),
    ("work/tests_summary.json", "raw"), ("work/tests_summary_amend1.json", "raw"), ("work/tests_summary_sealed_state.json", "raw"), ("work/run_tests_amend1.log", "raw/logs"), ("work/dry_real_cpu.log", "raw/logs"), ("work/test_*.log", "raw/logs"),
    ("work/prep_report.json", "raw"), ("work/prep_labels_report.json", "raw"), ("work/postseal_alignment_check.txt", "raw"), ("work/run_tests.log", "raw/logs"), ("work/prep_labels_run.log", "raw/logs"),
    ("work/plan/*.json", "raw/plan"), ("work/audit/*.json", "raw/audit"),
]
BIG = [  # (absolute glob, folder label)
    (f"{ARTIFACTS}/data/*.npz", "data"),
    (f"{ARTIFACTS}/sealed_state/*.py", "sealed_state"),
    (f"{ARTIFACTS_A2}/amend1_state/*.py", "sealed_state_amend1"),                       # sh5_lib.py / test_sh5_lib.py as they stood after Amendment 1 (the files Amendment 2 re-pins)
    (f"{ARTIFACTS_A3}/amend2_state/*.py", "sealed_state_amend2"),                       # sh5_lib.py / test_sh5_lib.py as they stood after Amendment 2 (the files Amendment 3 re-pins)
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.isdir(f"{ROOT}/out"):
        shutil.rmtree(f"{ROOT}/out")
    big = []
    for pat, lab in BIG:
        for p in sorted(glob.glob(pat)):
            big.append({"file": f"{lab}/{os.path.basename(p)}", "path": p, "bytes": os.path.getsize(p), "sha256": sha(p)})
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    with open(f"{OUT}/raw/ARTIFACTS_D.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"note": "LARGE payloads, NOT in the repo; they exist ONLY on D: (single location, exFAT external drive: copy them before relying on them).  sh5_inputs.npz is label-free (keys, folds, strata, z6, key-time points, "
                           "sign-clean targets); sh5_labels.npz holds the labels + the privileged signed key clearances (planted controls only) and was written only AFTER the seal.  The ctx (3.0 / 10.2 GiB) and the OOF "
                           "arrays are produced by the queue in READY.md and are not yet built.  sealed_state/ holds the SEALED-state copies (hash-verified against CODE_PINS.json / SEALED_AT.txt) of the five files and the guard that Amendment 1 changed; "
                           "the sealed state of code/test_sh5_ctx.py was not preserved (its Amendment-1 changes are additive tests only).  sealed_state_amend1/ holds code/sh5_lib.py and code/test_sh5_lib.py exactly as they stood after Amendment 1 "
                           "(the pins Amendment 2 moves FROM; hash-verified against CODE_PINS.json), the pre-amendment analysis code the guard must refuse.  sealed_state_amend2/ holds the same two files as they stood after Amendment 2 "
                           "(the pins Amendment 3 moves FROM, hash-verified against CODE_PINS_AMENDMENT2.json): the C-NULL-LABEL rule `lower CI > +0.02` that the guard must refuse.",
                   "root": ARTIFACTS, "total_bytes": sum(r["bytes"] for r in big), "n_files": len(big), "files": big}, f, indent=1)
    entries = []
    for pat, dst in SRC:
        for p in sorted(glob.glob(f"{ROOT}/{pat}")):
            if ".tmp." in os.path.basename(p) or p.endswith(".lock") or os.path.isdir(p) or "__pycache__" in p:
                continue
            name = os.path.basename(p)
            d = f"{OUT}/{dst}" if dst else OUT
            os.makedirs(d, exist_ok=True)
            q = f"{d}/{name}"
            shutil.copyfile(p, q)
            if name.endswith((".py", ".md", ".sh", ".ps1", ".json", ".txt", ".sha256", ".log")):
                b = open(q, "rb").read()
                if b"\r\n" in b:
                    open(q, "wb").write(b.replace(b"\r\n", b"\n"))
            entries.append(((BASE + "/" + (dst + "/" if dst else "") + name), q, sha(q), os.path.getsize(q)))
    refs = {n: sha(f"{REFDIR}/{n}") for n in sorted(os.listdir(REFDIR)) if n.endswith(".py")}
    os.makedirs(f"{OUT}/code", exist_ok=True)
    with open(f"{OUT}/code/REFS.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"note": "byte-exact copies of LANDED files loaded by sha256 at run time (sh5_lib.refs(); not re-staged): fastneg_lib.py = the census package, pickrule_lib.py = the pickrule package, epr1.py = the EP-R1 package",
                   "source_dir": REFDIR, "sha256": refs}, f, indent=1)
    for rel in ("code/REFS.json", "raw/ARTIFACTS_D.json"):
        q = f"{OUT}/{rel}"
        entries.append((BASE + "/" + rel, q, sha(q), os.path.getsize(q)))
    lines = ["# MANIFEST: SH5-DAC-PROX (staged by build_package.py; never committed or pushed)", "",
             f"Files staged for the repo: {len(entries)}. Sealed prereg present: {os.path.exists(OUT + '/PREREG_SH5_DAC_PROX.md')}.  No ctx built, no branch trained on real data, no GPU job run.",
             f"Large payloads NOT staged: {len(big)} files, {sum(r['bytes'] for r in big) / 1e6:.1f} MB, on `{ARTIFACTS}` (sha256 in `raw/ARTIFACTS_D.json`; D: only).", "",
             "| repo path (under `" + BASE + "/`) | bytes | sha256 |", "|---|---|---|"]
    for rp, q, h, n in entries:
        lines.append(f"| `{rp[len(BASE) + 1:]}` | {n} | `{h}` |")
    mp = f"{OUT}/MANIFEST.md"
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    entries.append((BASE + "/MANIFEST.md", mp, sha(mp), os.path.getsize(mp)))
    with open(f"{ROOT}/out/paths.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump([[rp, q, "raw"] for rp, q, h, n in entries], f, indent=1)
    print(f"staged {len(entries)} files under {OUT}; {len(big)} large files ({sum(r['bytes'] for r in big) / 1e6:.1f} MB) recorded on {ARTIFACTS}; paths.json written")


if __name__ == "__main__":
    main()
