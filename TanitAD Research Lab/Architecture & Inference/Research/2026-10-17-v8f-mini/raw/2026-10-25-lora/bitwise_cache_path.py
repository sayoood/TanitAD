"""PREREG_V8F_LORA proof: ``--encoder cache`` IS today's trainer, bit for bit.

Runs the SAME synthetic argv (M1 and M2b, i.e. with and without the WM auxiliary + SIGReg) through
  * PRE : a pristine tree = the tip export (190be80a) with v8f_mini_train.py / v8f_mini_data.py / v8f_mini.py as they were BEFORE the LoRA work, and
  * NEW : the tree with the LoRA work, once WITHOUT ``--encoder`` and once with ``--encoder cache``,
each in its own process (single thread, CPU, fixed seeds), and compares every tensor of the final checkpoint's ``model`` and ``opt`` state and every number of
``train_log.jsonl`` (except wall-clock ``sec``) bitwise. Also prints the sha256 of the final model state (the golden the pytest pins).

Usage: python bitwise_cache_path.py --pre <pre-tree root> --new <new-tree root> --work <scratch dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

PY = sys.executable


def run(tree, out, extra):
    argv = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--steps", "6", "--log-every", "1", "--warmup", "2", "--ckpt-every", "3",
            "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--out", out, *extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=os.pathsep.join([os.path.join(tree, "stack"), os.path.join(tree, "taniteval")]))
    r = subprocess.run([PY, os.path.join(tree, "stack", "scripts", "v8f_mini_train.py"), *argv], env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"{tree} {extra}: rc {r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-1500:]}")


def state_hash(sd) -> str:
    import torch
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        if torch.is_tensor(v):
            h.update(k.encode())
            h.update(v.detach().contiguous().view(torch.uint8).numpy().tobytes() if v.numel() else b"")
    return h.hexdigest()


def flat(x, pre=""):
    import torch
    if torch.is_tensor(x):
        yield pre, x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{pre}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from flat(v, f"{pre}[{i}]")


def compare(a_dir, b_dir):
    import torch
    A = torch.load(os.path.join(a_dir, "ckpt_last.pt"), weights_only=False)
    B = torch.load(os.path.join(b_dir, "ckpt_last.pt"), weights_only=False)
    fa, fb = dict(flat(A["model"], "model")), dict(flat(B["model"], "model"))
    fa.update(dict(flat(A["opt"], "opt")))
    fb.update(dict(flat(B["opt"], "opt")))
    bad = []
    if set(fa) != set(fb):
        bad.append(f"key sets differ: {sorted(set(fa) ^ set(fb))[:6]}")
    for k in sorted(set(fa) & set(fb)):
        if fa[k].dtype != fb[k].dtype or fa[k].shape != fb[k].shape or not torch.equal(fa[k], fb[k]):
            bad.append(k)
    la = [json.loads(l) for l in open(os.path.join(a_dir, "train_log.jsonl"), encoding="utf-8")]
    lb = [json.loads(l) for l in open(os.path.join(b_dir, "train_log.jsonl"), encoding="utf-8")]
    for x, y in zip(la, lb):
        for k in x:
            if k != "sec" and x[k] != y.get(k):
                bad.append(f"train_log step {x['step']} {k}: {x[k]} != {y.get(k)}")
    if len(la) != len(lb):
        bad.append("train_log length")
    return {"n_tensors": len(fa), "n_log_rows": len(la), "mismatches": bad, "model_sha256": state_hash(A["model"]), "keys_extra_in_new": sorted(set(B) - set(A))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    res = {}
    import torch
    res["torch"] = torch.__version__
    for arm in ("M1", "M2b"):
        d = {k: os.path.join(a.work, f"{arm}_{k}") for k in ("pre", "new_noflag", "new_cache")}
        run(a.pre, d["pre"], ["--arm", arm])
        run(a.new, d["new_noflag"], ["--arm", arm])
        run(a.new, d["new_cache"], ["--arm", arm, "--encoder", "cache"])
        res[arm] = {"pre_vs_new_noflag": compare(d["pre"], d["new_noflag"]), "pre_vs_new_encoder_cache": compare(d["pre"], d["new_cache"])}
        for k in ("pre_vs_new_noflag", "pre_vs_new_encoder_cache"):
            r = res[arm][k]
            print(f"{arm} {k}: {r['n_tensors']} tensors, {r['n_log_rows']} log rows, mismatches {len(r['mismatches'])}, model sha256 {r['model_sha256'][:16]}")
            for m in r["mismatches"][:5]:
                print("   MISMATCH", m)
    ok = all(not res[arm][k]["mismatches"] for arm in ("M1", "M2b") for k in ("pre_vs_new_noflag", "pre_vs_new_encoder_cache"))
    print("VERDICT", "BIT-FOR-BIT IDENTICAL" if ok else "DIFFERENT")
    json.dump(res, open(os.path.join(a.work, "bitwise_cache_path.json"), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
