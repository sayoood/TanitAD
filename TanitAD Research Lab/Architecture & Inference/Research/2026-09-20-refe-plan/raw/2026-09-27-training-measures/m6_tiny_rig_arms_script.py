"""Scratch: causal test of the heading-branch trap on the tiny rig (real trainer, real DEV10 targets)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REFE = Path("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
PY = sys.executable
root = Path(tempfile.gettempdir()) / "yaw19_repro"
bank = root / "bank"                                  # 3,000 DEV10 rank-0 rows (made by repro_yaw19.py)
empty = root / "empty"
code = {"orig": root / "code_orig", "pat": root / "code_pat"}
for k, d in code.items():
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    for f in REFE.glob("*.py"):
        shutil.copy2(f, d / f.name)
    if k == "pat":
        for pch in ("measures_hooks.patch", "measures_consumers.patch"):
            subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-p1", str(REFE / pch)], cwd=str(d), check=True)


def run(tag, which, extra, steps):
    out = root / f"x_{tag}"
    if out.exists() and not any(a == "--resume" for a in extra):
        shutil.rmtree(out)
    argv = ["--backbone", "tiny", "--weights", "none", "--synthetic", "--cpu", "--targets", str(bank),
            "--scorer-targets", str(empty), "--batch", "16", "--accum", "1", "--lr", "1e-3", "--steps", str(steps),
            "--log-every", "500", "--out", str(out), "--ckpt-every-min", "1e9"] + extra
    spec = root / f"xspec_{tag}.json"
    spec.write_text(json.dumps({"root": str(code[which]), "out": str(root / f"xchild_{tag}.out"), "argv": argv}))
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTHONIOENCODING="utf-8")
    with open(root / f"xtrain_{tag}.log", "w", encoding="utf-8") as fh:
        p = subprocess.run([PY, str(REFE / "selftest_measures.py"), "--child", "trainer", "--spec", str(spec)],
                           stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(code[which]))
    rc = torch.load(root / f"xchild_{tag}.out", weights_only=False)["rc"]
    return tag, p.returncode, rc


def evaluate(tag, which="pat"):
    sys.path.insert(0, str(code[which]))
    import model as M
    import train as T
    cfg = M.REFeConfig(backbone="tiny", width=64, depth=2, heads=4, img_h=32, img_w=64, lora_rank=4, n_registers=2,
                       dec_width=32, dec_depth=2, dec_heads=4, reg_heads=4)
    m = M.REFe(cfg).eval()
    m.load_state_dict(torch.load(root / f"x_{tag}" / "model_final.pt", map_location="cpu", weights_only=False)["model"])
    ds = T.TargetBank(str(bank), None, cfg, synthetic=True)
    idx = list(range(0, len(ds), 25))[:120]
    it = [ds[i] for i in idx]
    with torch.no_grad():
        tr = m(torch.stack([x[0] for x in it]), torch.stack([x[1] for x in it]), torch.stack([x[2] for x in it]))[0]
    tr = tr.double().numpy()
    tg = torch.stack([x[3] for x in it]).double().numpy()
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi                          # noqa: E731
    d = np.abs(tr[..., :2] - tg[:, None, :, :2]).sum(-1).mean(-1)
    w = d.argmin(1)
    ar = np.arange(len(idx))
    frac = [float(np.mean(np.abs(tr[:, :, k, 2]) > np.pi)) for k in range(20)]
    werr = [float(np.median(np.abs(wrap(tr[ar, w, k, 2] - tg[:, k, 2])))) for k in range(20)]
    trapped = [k for k in range(20) if frac[k] > 0.10 and werr[k] > 0.5]
    return {"tag": tag, "trapped_steps": trapped, "max_frac_beyond_pi": max(frac),
            "winner_wrap_err_median_by_step": [round(v, 3) for v in werr],
            "winner_pos_L1_m": float(d[ar, w].mean())}


if __name__ == "__main__":
    arms = [("off_s0", "orig", ["--seed", "0"], 1500), ("off_s1", "orig", ["--seed", "1"], 1500),
            ("plain_s0", "pat", ["--seed", "0", "--yaw-loss", "plain"], 1500),
            ("plain_s1", "pat", ["--seed", "1", "--yaw-loss", "plain"], 1500)]
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(lambda a: run(*a), arms):
            print("run", r, flush=True)
    # live-fix rehearsal: trap under wrapped for 1500 steps, then (a) continue wrapped, (b) declared switch to plain
    print("run", run("rec_base", "pat", ["--seed", "0", "--ckpt-every-steps", "1500", "--halt-after-steps", "1500"], 3000), flush=True)
    for t in ("rec_wrapped", "rec_plain"):
        if (root / f"x_{t}").exists():
            shutil.rmtree(root / f"x_{t}")
        shutil.copytree(root / "x_rec_base", root / f"x_{t}")
    with ThreadPoolExecutor(2) as ex:
        for r in ex.map(lambda a: run(*a), [("rec_wrapped", "pat", ["--seed", "0", "--resume"], 3000),
                                             ("rec_plain", "pat", ["--seed", "0", "--resume", "--yaw-loss", "plain",
                                                                   "--declare-change", "yaw_loss"], 3000)]):
            print("run", r, flush=True)
    res = [evaluate(t) for t in ("off_s0", "off_s1", "plain_s0", "plain_s1", "rec_wrapped", "rec_plain")]
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k != "winner_wrap_err_median_by_step"}),
              "| winner wrap err by step:", " ".join(f"{v:.2f}" for v in r["winner_wrap_err_median_by_step"]))
    json.dump(res, open(root / "yaw_experiments.json", "w"), indent=1)
