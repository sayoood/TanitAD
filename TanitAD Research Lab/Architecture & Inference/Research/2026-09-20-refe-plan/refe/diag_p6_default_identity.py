#!/usr/bin/env python3
"""Evidence that the P6 edits leave P2 / SFT behaviour BIT-IDENTICAL when the new flags are off (task constraint: "a default-argv run must produce identical losses on a fixed
tiny batch before and after your change").

BEFORE = the files as STAGED in the git index (git show :<path>) for the four files this work edits (train.py, model.py, ckpt_io.py, planner.py); everything else is the
working tree. AFTER = the working tree. Both run the SAME argv on the dev box's tiny P2 rehearsal bank (ViT-S, 1 camera, synthetic images, CPU):
  (a) P2's own rehearsal argv (on-policy + PDM-only + scorer-sees-trunk + warm start)       (b) the fixed-bank default (scorer targets, no flags)
and compare every logged traj_L1 / score and the SHA-256 of every tensor of model_final.pt. Writes raw/2026-10-11-p6-impl/default_identity.json.
    python diag_p6_default_identity.py --git-dir <repo .git> --repo-prefix "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_fixture as F  # noqa: E402

EDITED = ("train.py", "model.py", "ckpt_io.py", "planner.py")


def digest(path) -> str:
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(pkg: Path, argv: list, out: Path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, str(pkg / "train.py")] + argv + ["--out", str(out)], cwd=str(pkg), env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()] if (out / "metrics.jsonl").exists() else []
    return r.returncode, [(x["traj_L1"], x["score"]) for x in rows if "traj_L1" in x], digest(out / "model_final.pt") if (out / "model_final.pt").exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--repo-prefix", required=True)
    ap.add_argument("--work-tree", default=None, help="the repo working tree root (git --work-tree)")
    a = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix="p6_ident_"))
    pre = root / "pre"
    shutil.copytree(HERE, pre, ignore=shutil.ignore_patterns("__pycache__"))
    shas = {}
    for f in EDITED:
        blob = subprocess.run(["git", "--git-dir", a.git_dir, "show", f":{a.repo_prefix}{f}"], capture_output=True)
        if blob.returncode != 0:
            raise SystemExit(f"cannot read the staged {f}: {blob.stderr.decode()[:200]}")
        (pre / f).write_bytes(blob.stdout)
        shas[f] = {"before_sha256": hashlib.sha256(blob.stdout).hexdigest(), "after_sha256": hashlib.sha256((HERE / f).read_bytes()).hexdigest()}
    fx = F.tiny_p2_fixture(str(root / "fx"))
    argv_a = F.rehearsal_argv(fx, str(root / "unused"), steps=4)[:-2]                 # drop the fixture's own --out; run() appends one
    argv_b = ["--backbone", "vits16", "--n-cameras", "1", "--synthetic", "--cpu", "--targets", fx["bank"], "--scorer-targets", fx["bank"], "--batch", "1", "--accum", "2",
              "--steps", "4", "--lr", "5e-5", "--log-every", "1", "--ckpt-every-min", "999"]
    res = {"files": shas, "arms": {}}
    for name, argv in (("p2_rehearsal_onpolicy_pdm_only", argv_a), ("fixed_bank_default", argv_b)):
        rb = run(pre, argv, root / f"{name}_before")
        ra = run(HERE, argv, root / f"{name}_after")
        res["arms"][name] = {"rc_before": rb[0], "rc_after": ra[0], "n_logged_steps": len(rb[1]), "losses_before": rb[1], "losses_after": ra[1],
                             "losses_identical": rb[1] == ra[1] and len(rb[1]) > 0, "model_sha256_before": rb[2], "model_sha256_after": ra[2],
                             "model_identical": rb[2] is not None and rb[2] == ra[2]}
        print(name, "losses identical:", res["arms"][name]["losses_identical"], "model identical:", res["arms"][name]["model_identical"], flush=True)
    res["identical"] = all(v["losses_identical"] and v["model_identical"] and v["rc_before"] == 0 and v["rc_after"] == 0 for v in res["arms"].values())
    out = HERE.parent / "raw/2026-10-11-p6-impl/default_identity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    shutil.rmtree(root, ignore_errors=True)
    print("ZZDEFAULT_IDENTICAL" if res["identical"] else "ZZDEFAULT_DIFFERS")
    return 0 if res["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
