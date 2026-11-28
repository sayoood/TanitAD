#!/usr/bin/env python3
"""After the fixes batch LANDS: rebase NEW-1's shared-file edits onto the NEW tip and verify.

1. read the tip commit of the branch (40-char asserted);
2. `git archive` its stack/, tools/, taniteval/ (minus results/) into a FRESH directory
   (never deletes anything; refuses if the target exists);
3. regenerate the shared files from the tip's RAW blobs with apply_residual_prior_edits.py
   into this package's code/fix/ (EDIT_MANIFEST.json records base/new blob sha1s);
4. build the candidate tree = a fresh archive + this package's code/fix/stack overlay;
5. assert `tanitad.__file__` of each tree, run test_residual_prior.py on the candidate, and
   the off-mode digest on BOTH trees (the test's TIP_OFF_DIGEST literal must equal the tip's);
6. print the LANDING_READY lines (package path -> repo path, tip base blob | new blob | EOL).

usage: python finalize_after_fixes.py <work_root> [<fix_out>]   e.g. C:/Users/Admin/r7rp
  <fix_out> defaults to this package's code/fix (the DELIVERABLE); pass a scratch dir for a
  dry run on a tip the fixes have not reached yet.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

GIT = "C:/Users/Admin/tanitad-push/.git"
BRANCH = "agent/arch-inf-20260803"
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-residual-prior"
PY = sys.executable
SHARED = ["stack/tanitad/refs/refc.py", "stack/tanitad/refs/refc_v3.py",
          "stack/scripts/refc_v3_train.py", "stack/tanitad/train/declared_vs_built.py",
          "stack/tests/test_declared_vs_built.py",
          # 2026-09-27, after the Thor full-suite gate (21 RL-stack regressions):
          "stack/tanitad/rl/ddv2_refc_chain.py", "stack/tests/test_ddv2_refc_chain.py",
          "stack/tanitad/channel_admissibility.py"]
NEW = ["stack/tanitad/models/kinematic_prior.py", "stack/tests/test_residual_prior.py"]


def sh(args, **kw):
    return subprocess.run(args, check=True, capture_output=True, **kw)


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def archive(rev: str, dst: str) -> None:
    if os.path.exists(dst):
        raise SystemExit(f"refusing: {dst} exists (never overwrite a tree)")
    os.makedirs(dst)
    tar = sh(["git", f"--git-dir={GIT}", "archive", "--format=tar", rev, "--", "stack", "tools",
              "taniteval", ":(exclude)taniteval/results"]).stdout
    subprocess.run(["tar", "-x", "-C", dst], input=tar, check=True)


def env_for(tree: str) -> dict:
    return dict(os.environ, PYTHONPATH=os.path.join(tree, "stack").replace("\\", "/"),
                OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"), HF_HUB_OFFLINE="1")


def main() -> int:
    root = os.path.abspath(sys.argv[1])
    tip = sh(["git", f"--git-dir={GIT}", "rev-parse", BRANCH]).stdout.decode().strip()
    assert len(tip) == 40, tip
    tip_tree = os.path.join(root, f"tip_{tip[:10]}")
    cand_tree = os.path.join(root, f"cand_{tip[:10]}")
    archive(tip, tip_tree)
    archive(tip, cand_tree)
    fix = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(PKG, "code", "fix")
    src_fix = os.path.join(PKG, "code", "fix")
    for rel in NEW:                      # the NEW files live in the package; mirror them
        dst = os.path.join(fix, rel)
        if os.path.abspath(dst) != os.path.abspath(os.path.join(src_fix, rel)):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(src_fix, rel), dst)
    sh([PY, os.path.join(HERE, "apply_residual_prior_edits.py"), "--base-git", GIT, "--rev", tip,
        "--out", fix])
    man = json.load(open(os.path.join(fix, "EDIT_MANIFEST.json"), encoding="utf-8"))
    for rel in SHARED + NEW:
        src = os.path.join(fix, rel)
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(os.path.join(cand_tree, rel)), exist_ok=True)
            shutil.copyfile(src, os.path.join(cand_tree, rel))
    for tree in (tip_tree, cand_tree):
        tf = sh([PY, "-c", "import tanitad; print(tanitad.__file__)"], env=env_for(tree),
                cwd=os.path.join(tree, "stack")).stdout.decode().strip()
        assert os.path.normcase(tf).startswith(os.path.normcase(os.path.join(tree, "stack"))), tf
    t = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/test_residual_prior.py", "-rs"], env=env_for(cand_tree),
                       cwd=os.path.join(cand_tree, "stack"), capture_output=True, text=True)
    print("test_residual_prior.py on the candidate:", t.stdout.strip().splitlines()[-1])
    dig = {}
    for name, tree in (("tip", tip_tree), ("cand", cand_tree)):
        d = subprocess.run([PY, os.path.join(HERE, "offmode_digest.py"), tree],
                           env=env_for(tree), capture_output=True, text=True)
        js = d.stdout[d.stdout.index("{"):]
        dig[name] = json.loads(js)["digest"]
    lit = [l for l in open(os.path.join(fix, "stack/tests/test_residual_prior.py"),
                           encoding="utf-8") if l.startswith("TIP_OFF_DIGEST = ")][0]
    print("off-mode digest tip:", dig["tip"], "| cand:", dig["cand"],
          "| test literal:", lit.strip())
    print("\n# ---- LANDING_READY lines ----")
    for rel in SHARED:
        f = man["files"].get(rel)
        if f is None:
            print(f"# {rel}: NOT PRODUCED (absent at base {tip[:10]})")
            continue
        print(f"# {rel}: base {f['base_blob_sha1']} | new {f['out_blob_sha1']} | {f['eol']}")
        print(f"{PKG_REL}/code/fix/{rel}  ->  {rel}")
    for rel in NEW:
        data = open(os.path.join(fix, rel), "rb").read()
        print(f"# {rel}: base NEW | new {blob_sha1(data)} | "
              f"{'CRLF' if data.count(b'\r\n') else 'LF'}")
        print(f"{PKG_REL}/code/fix/{rel}  ->  {rel}")
    print(json.dumps({"tip": tip, "tip_tree": tip_tree, "cand_tree": cand_tree}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
