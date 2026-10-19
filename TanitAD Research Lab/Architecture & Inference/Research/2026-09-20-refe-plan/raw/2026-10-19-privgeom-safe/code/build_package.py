"""Stage the PRIVGEOM-SAFE package: copy what exists into out/<repo path>/, write out/paths.json ([repo path, abs source, "raw"]) and MANIFEST.md.  Re-runnable (interim and final).  No git.

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
SRC = [  # (glob under ROOT, destination folder inside the package)
    ("PREREG_PRIVGEOM_SAFE*.md", ""), ("PREREG_PRIVGEOM_SAFE*.sha256", ""), ("SEALED_AT.txt", ""), ("HANDOFF.md", ""), ("RESULT.md", ""),
    ("code/*.py", "code"), ("code/finish.sh", "code"), ("code/launch.ps1", "code"),
    ("run/cost/cost.json", "raw/cost"), ("run/subset.json", "raw"),
    ("run/scalars/*.npz", "raw/scalars"), ("run/scalars/*.timing.json", "raw/scalars"),
    ("run/join/join_report.json", "raw/join"), ("run/gate_report.json", "raw"), ("run/gate_token.json", "raw"),
    ("run/fits/*.npy", "raw/fits"), ("run/fits/*.json", "raw/fits"), ("run/results/*", "raw/results"),
    ("run/*.log", "raw/logs"),
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
    entries = []
    for pat, dst in SRC:
        for p in sorted(glob.glob(f"{ROOT}/{pat}")):
            if ".tmp." in os.path.basename(p) or p.endswith(".lock"):
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
    q = f"{OUT}/code/REFS.json"
    entries.append((BASE + "/code/REFS.json", q, sha(q), os.path.getsize(q)))
    lines = ["# MANIFEST: PRIVGEOM-SAFE (staged by build_package.py; never committed or pushed)", "", f"Files: {len(entries)}. Result present: {os.path.exists(OUT + '/RESULT.md')}.", "",
             "| repo path (under `" + BASE + "/`) | bytes | sha256 |", "|---|---|---|"]
    for rp, q, h, n in entries:
        lines.append(f"| `{rp[len(BASE) + 1:]}` | {n} | `{h}` |")
    mp = f"{OUT}/MANIFEST.md"
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    entries.append((BASE + "/MANIFEST.md", mp, sha(mp), os.path.getsize(mp)))
    with open(f"{ROOT}/out/paths.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump([[rp, q, "raw"] for rp, q, h, n in entries], f, indent=1)
    print(f"staged {len(entries)} files under {OUT}; paths.json written")


if __name__ == "__main__":
    main()
