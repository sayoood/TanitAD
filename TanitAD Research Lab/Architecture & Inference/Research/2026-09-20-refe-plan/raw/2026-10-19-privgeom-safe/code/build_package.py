"""Stage the PRIVGEOM-SAFE package: copy what exists into out/<repo path>/, write out/paths.json ([repo path, abs source, "raw"]) and MANIFEST.md.  Re-runnable.  No git.

SMALL files (code, prereg, results, jsons, logs) are staged under out/ for the repo.  LARGE binary payloads (> ~20 MB in total: the 41 per-log scalar stores, the OOF logit arrays, the join arrays)
are copied to ARTIFACTS on D: and recorded in the package as raw/ARTIFACTS_D.json with sha256 (verified by re-hashing the D: copy), NOT staged for the repo.

    python build_package.py
"""
import glob
import hashlib
import json
import os
import shutil

ROOT = "C:/Users/Admin/par18/privgeom"
BASE = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-privgeom-safe"
OUT = f"{ROOT}/out/{BASE}"
ARTIFACTS = "D:/Projects/TanitAD-artifacts/privgeom_2026-10-19"
SRC = [  # (glob under ROOT, destination folder inside the package)
    ("PREREG_PRIVGEOM_SAFE*.md", ""), ("PREREG_PRIVGEOM_SAFE*.sha256", ""), ("SEALED_AT.txt", ""), ("HANDOFF.md", ""), ("RESULT.md", ""),
    ("code/*.py", "code"), ("code/finish.sh", "code"), ("code/launch.ps1", "code"),
    ("run/cost/cost.json", "raw/cost"), ("run/subset.json", "raw"),
    ("run/scalars/*.timing.json", "raw/scalars"),
    ("run/join/join_report.json", "raw/join"), ("run/gate_report.json", "raw"), ("run/gate_token.json", "raw"),
    ("run/fits/*.json", "raw/fits"), ("run/results/*", "raw/results"),
    ("run/descr/*.json", "raw/descr"), ("run/descr/*.md", "raw/descr"), ("run/descr/fits/*.json", "raw/descr/fits"),
    ("run/deploy_tests/deploy_tests.json", "raw"),
    ("run/*.log", "raw/logs"),
]
BIG = [  # (glob under ROOT, folder under ARTIFACTS)
    ("run/scalars/*.npz", "scalars"), ("run/fits/*.npy", "fits"), ("run/descr/fits/*.npy", "descr_fits"), ("run/join/arrays.npz", "join"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def bank_big():
    rows = []
    for pat, dst in BIG:
        for p in sorted(glob.glob(f"{ROOT}/{pat}")):
            if ".tmp." in os.path.basename(p):
                continue
            d = f"{ARTIFACTS}/{dst}"
            os.makedirs(d, exist_ok=True)
            q = f"{d}/{os.path.basename(p)}"
            hs = sha(p)
            if not (os.path.exists(q) and sha(q) == hs):
                shutil.copyfile(p, q)
            if sha(q) != hs:
                raise SystemExit(f"D: copy does not re-hash: {q}")
            rows.append({"file": f"{dst}/{os.path.basename(p)}", "source": p, "bytes": os.path.getsize(q), "sha256": hs})
    return rows


def main():
    if os.path.isdir(f"{ROOT}/out"):
        shutil.rmtree(f"{ROOT}/out")
    big = bank_big()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    with open(f"{OUT}/raw/ARTIFACTS_D.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"note": "LARGE binary payloads, NOT in the repo; copied to D: and re-hashed after the copy.  The OOF logit arrays and join arrays are also named by their sha256 inside raw/fits/*.json, raw/join/join_report.json and raw/gate_token.json.",
                   "root": ARTIFACTS, "total_bytes": sum(r["bytes"] for r in big), "n_files": len(big), "files": big}, f, indent=1)
    entries = []
    for pat, dst in SRC:
        for p in sorted(glob.glob(f"{ROOT}/{pat}")):
            if ".tmp." in os.path.basename(p) or p.endswith(".lock") or os.path.isdir(p):
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
    refs = {n: sha(f"{ROOT}/ref/{n}") for n in sorted(os.listdir(f"{ROOT}/ref")) if n.endswith(".py")}
    os.makedirs(f"{OUT}/code", exist_ok=True)
    with open(f"{OUT}/code/REFS.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"note": "byte-exact copies of LANDED files used from ref/ (not re-staged): fastneg_lib.py = the census package, pickrule_lib.py = the pickrule package, epr1.py = the EP-R1 package", "sha256": refs}, f, indent=1)
    for rel in ("code/REFS.json", "raw/ARTIFACTS_D.json"):
        q = f"{OUT}/{rel}"
        entries.append((BASE + "/" + rel, q, sha(q), os.path.getsize(q)))
    lines = ["# MANIFEST: PRIVGEOM-SAFE (staged by build_package.py; never committed or pushed)", "", f"Files staged for the repo: {len(entries)}. Result present: {os.path.exists(OUT + '/RESULT.md')}.",
             f"Large payloads NOT staged: {len(big)} files, {sum(r['bytes'] for r in big) / 1e6:.1f} MB, on `{ARTIFACTS}` (sha256 in `raw/ARTIFACTS_D.json`).", "",
             "| repo path (under `" + BASE + "/`) | bytes | sha256 |", "|---|---|---|"]
    for rp, q, h, n in entries:
        lines.append(f"| `{rp[len(BASE) + 1:]}` | {n} | `{h}` |")
    mp = f"{OUT}/MANIFEST.md"
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    entries.append((BASE + "/MANIFEST.md", mp, sha(mp), os.path.getsize(mp)))
    with open(f"{ROOT}/out/paths.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump([[rp, q, "raw"] for rp, q, h, n in entries], f, indent=1)
    print(f"staged {len(entries)} files under {OUT}; banked {len(big)} large files ({sum(r['bytes'] for r in big) / 1e6:.1f} MB) on {ARTIFACTS}; paths.json written")


if __name__ == "__main__":
    main()
