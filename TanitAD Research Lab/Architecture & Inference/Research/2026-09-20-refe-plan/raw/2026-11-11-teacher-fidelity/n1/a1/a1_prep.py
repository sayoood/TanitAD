#!/usr/bin/env python3
"""TF-N1 A1 prep (run BEFORE the amendment is frozen; it scores nothing and opens no N1 value).

  python a1_prep.py --run-dir C:/Users/Admin/n1_run --mirror C:/Users/Admin/tanitad-push/.git --tip <sha-or-ref> [--hold <this dir>]

1. exports eval/harness/ at --tip from the mirror (git archive; autocrlf is irrelevant to archive) into <run>/w/pk/eval/harness/
   and checks it with the tree's own harness_pins.py (every file sha256-pinned, LF-normalised);
2. copies the hold's score_p4_a1.py -> <run>/w/pk/eval/, and tf_n1_a1_chain.py + tf_n1_analyze_a1.py -> <run>/n1/a1/;
3. refuses unless the E: copy of score_navtest_refe.py (which harness/score_e.py loads by an absolute path) equals <run>/w/pk/eval's;
4. writes <run>/n1/a1/A1_PINS.json (sha256 of every file the A1 chain depends on) and prints its sha256: the amendment quotes it.
Hashing fam/*.json and the build plans reads BYTES only, never values.
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile

PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(p):
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mirror", required=True, help="the .git dir of the repo mirror (read only)")
    ap.add_argument("--tip", required=True)
    ap.add_argument("--hold", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--no-fam-pins", action="store_true", help="test only: skip the fam/ and score pins")
    a = ap.parse_args()
    run = a.run_dir
    dst = os.path.join(run, "w", "pk", "eval", "harness")
    if os.path.isdir(dst):
        shutil.rmtree(dst)                                  # our own previous export of the same tree
    blob = subprocess.run(["git", "--git-dir", a.mirror, "-c", "core.autocrlf=false", "archive", a.tip, f"{PK}/eval/harness"],
                          check=True, capture_output=True).stdout
    pre = f"{PK}/eval/harness/"
    n = 0
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.startswith(pre):
                continue
            rel = m.name[len(pre):]
            if rel.startswith("/") or ".." in rel.split("/"):
                sys.exit(f"REFUSING: odd tar member {m.name}")
            out = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "wb").write(tf.extractfile(m).read())
            n += 1
    print(f"exported {n} harness files at {a.tip}")
    sys.path.insert(0, os.path.join(run, "w", "pk", "eval"))
    import harness_pins
    v = harness_pins.verify(root=harness_pins.Path(dst), caller=harness_pins.Path(run) / "w" / "pk" / "eval" / "score_navtest_refe.py")
    if not v.get("ok"):
        sys.exit("REFUSING: exported harness does not match HARNESS_PINS: " + json.dumps(v, default=str)[:600])
    print("harness_pins.verify OK:", v["n_files"], "files, caller variant", v["caller"]["variant"])
    shutil.copyfile(os.path.join(a.hold, "score_p4_a1.py"), os.path.join(run, "w", "pk", "eval", "score_p4_a1.py"))
    os.makedirs(os.path.join(run, "n1", "a1"), exist_ok=True)
    for f in ("tf_n1_a1_chain.py", "tf_n1_analyze_a1.py"):
        shutil.copyfile(os.path.join(a.hold, f), os.path.join(run, "n1", "a1", f))
    src = open(os.path.join(dst, "score_e.py"), encoding="utf-8").read()
    m = re.search(r'^PKG = "([^"]+)"', src, re.M)
    ec = m.group(1) + "/eval/score_navtest_refe.py"
    mine = os.path.join(run, "w", "pk", "eval", "score_navtest_refe.py")
    if not os.path.isfile(ec) or sha_lf(ec) != sha_lf(mine):
        sys.exit(f"REFUSING: score_e.py loads {ec}, which is missing or differs from {mine}")
    print("score_e.py caller identical to the run tree's:", ec)
    files = {}
    for d, _, fs in os.walk(dst):
        for f in fs:
            if "__pycache__" not in d:
                files[os.path.relpath(os.path.join(d, f), run).replace(os.sep, "/")] = sha(os.path.join(d, f))
    rel = ["w/pk/eval/score_p4_a1.py", "w/pk/eval/score_navtest_refe.py", "w/pk/eval/harness_pins.py",
           "n1/a1/tf_n1_a1_chain.py", "n1/a1/tf_n1_analyze_a1.py", "build/out/refe_tfn1_tc20s.npz", "build/out/refe_tfn1_tw.npz",
           "build/out/refe_tfn1_m.npz", "build/out/n1_build_report.json", "inputs/refe_p4_teacher_navtest.npz"]
    if not a.no_fam_pins:
        rel += [f"fam/fam_{x}.json" for x in ("N1C", "N1T", "N1W", "CERT", "human")]
    for r in rel:
        files[r] = sha(os.path.join(run, r))
    out = os.path.join(run, "n1", "a1", "A1_PINS.json")
    json.dump({"card": "PREREG_TF_N1_A1.md C-PIN-A1", "tip": a.tip, "files": dict(sorted(files.items()))},
              open(out, "w", encoding="utf-8"), indent=1)
    print(f"ZZN1A1_PINS {len(files)} files -> {out}\nA1_PINS.json sha256 {sha(out)}")


if __name__ == "__main__":
    main()
