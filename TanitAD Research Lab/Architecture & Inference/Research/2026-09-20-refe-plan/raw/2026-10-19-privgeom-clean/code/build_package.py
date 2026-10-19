"""Stage the PRIVGEOM-CLEAN package: copy what exists into out/<repo path>/, write out/paths.json ([repo path, abs source, "raw"]) and MANIFEST.md.  Re-runnable.  No git.  Nothing large is staged or copied to D:.

    python build_package.py
"""
import glob
import hashlib
import json
import os
import shutil

ROOT = "C:/Users/Admin/par18/pgclean"
BASE = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-privgeom-clean"
OUT = f"{ROOT}/out/{BASE}"
SRC = [  # (glob under ROOT, destination folder inside the package)
    ("PREREG_PRIVGEOM_CLEAN.md", ""), ("PREREG_PRIVGEOM_CLEAN.sha256", ""), ("SEALED_AT.txt", ""), ("MM_RULING_PGCLEAN.md", ""), ("HANDOFF.md", ""), ("RESULT.md", ""), ("draft/*.md", "draft"),
    ("code/*.py", "code"), ("code/finish_clean.sh", "code"), ("code/rehearsal_chain.sh", "code"), ("code/launch.ps1", "code"), ("code/PINS.json", "code"),
    ("run/coverage.json", "raw"), ("run/design.json", "raw"), ("run/plant_calibration.json", "raw"), ("run/admit_seen.json", "raw"), ("run/cost.json", "raw"),
    ("run/identity/*.json", "raw/identity"), ("run/glive/*.json", "raw/glive"),
    ("run_rehearsal/gate_report.json", "raw/rehearsal"), ("run_rehearsal/gate_token.json", "raw/rehearsal"), ("run_rehearsal/results/*", "raw/rehearsal/results"),
    ("run_rehearsal/fits/*.json", "raw/rehearsal/fits"), ("run_rehearsal/rehearsal_vs_prior.json", "raw/rehearsal"), ("run_rehearsal/*.log", "raw/rehearsal/logs"),
    ("logs/glive*.log", "raw/logs"),
    # ---- the READ (run_clean/): seal proofs, the chain, the gate, the fits' JSONs, the results.  The big arrays (scalars/*.npz, fits/*.npy, join/arrays.npz) are NOT staged here (they stay on C: and are listed in RESULT.md)
    ("run_clean/seal_proofs/seal_proofs.json", "raw/seal_proofs"), ("run_clean/*.log", "raw/read/logs"), ("run_clean/CHAIN_*.txt", "raw/read"), ("run_clean/subset_clean.json", "raw/read"),
    ("run_clean/join/join_report.json", "raw/read/join"), ("run_clean/gate_report.json", "raw/read"), ("run_clean/gate_token.json", "raw/read"), ("run_clean/fits/*.json", "raw/read/fits"),
    ("run_clean/results/*", "raw/read/results"), ("run_clean/scalars/*.timing.json", "raw/read/scalars"),
]
RAW_DST = ("", "code", "draft")                      # sealed / pinned bytes are staged byte-exact (never line-ending normalised: the seal hashes raw bytes)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.isdir(f"{ROOT}/out"):
        shutil.rmtree(f"{ROOT}/out")
    os.makedirs(OUT, exist_ok=True)
    entries = []
    for pat, dst in SRC:
        for p in sorted(glob.glob(f"{ROOT}/{pat}")):
            if ".tmp." in os.path.basename(p) or p.endswith(".lock") or os.path.isdir(p) or p.endswith(".npy"):
                continue
            name = os.path.basename(p)
            d = f"{OUT}/{dst}" if dst else OUT
            os.makedirs(d, exist_ok=True)
            q = f"{d}/{name}"
            shutil.copyfile(p, q)
            if dst not in RAW_DST and name.endswith((".py", ".md", ".sh", ".ps1", ".json", ".txt", ".sha256", ".log")):
                b = open(q, "rb").read()
                if b"\r\n" in b:
                    open(q, "wb").write(b.replace(b"\r\n", b"\n"))
            entries.append(((BASE + "/" + (dst + "/" if dst else "") + name), q, sha(q), os.path.getsize(q)))
    refs = {n: sha(f"{ROOT}/ref/{n}") for n in sorted(os.listdir(f"{ROOT}/ref")) if n.endswith(".py")}
    with open(f"{OUT}/code/REFS.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"note": "byte-exact copy of the LANDED census library used from ref/ (not re-staged): fastneg_lib.py = the census package (the same sha256 PRIVGEOM-SAFE pinned)", "sha256": refs}, f, indent=1)
    q = f"{OUT}/code/REFS.json"
    entries.append((BASE + "/code/REFS.json", q, sha(q), os.path.getsize(q)))
    lines = ["# MANIFEST: PRIVGEOM-CLEAN (SEALED prereg + tested code path + read outputs when they exist; staged by build_package.py; never committed or pushed)", "",
             f"Files staged for the repo: {len(entries)}. Sealed: {os.path.exists(OUT + '/PREREG_PRIVGEOM_CLEAN.sha256')} (see SEALED_AT.txt). Whether a read result exists: see RESULT.md (absent until the read chain finishes).", "",
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
