"""wh_run.py -- the T0 and T1 passes of PREREG_V8F_WEATHERHEAD form (a): the argv of both evaluators for a head checkpoint (``--ctx-pool`` NAMED in every argv, on for an H arm and off for the S1 re-dump) and the
runner (CPU, 2 threads, ONE environment and one thread count for every dump of a T-ISO (d) pair).  Gates new CPU jobs on the commit headroom (>= 5.5 GB free commit; exit 75 otherwise).

  python wh_run.py t0 --ckpt CK --arm H_s0 --out DIR [--ctx-pool on|off]          -> DIR/eval_WH-<arm>_real.npz
  python wh_run.py t1 --ckpt CK --arm H_s0 --out DIR [--ctx-pool on|off] [--max-windows N]  -> DIR/ticks/ticks_WH-<arm>_m10_causal_heads.npz
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wh_lib as W  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ART = "E:/Projects/TanitAD-artifacts"
SF = ART + "/sfresh_inputs"
RAWT = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
NAVTEXT_SHA = "03bd75bc3a3de17928681bedb8e46a48a891304de45d395c49cc876acf6e6160"


def nav_args(ckpt):
    import torch
    src = torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"].get("nav_source", "v9")
    if src == "text":
        return ["--nav-text-npz", W.INP + "/navtext_nav.npz", "--nav-text-sha256", NAVTEXT_SHA]
    return []


def t0_argv(ckpt, tag, ctx_pool, out_dir, grid=None, device="cpu"):
    return [PY, W.TREE + "/stack/scripts/v8f_mini_eval.py", "run", "--ckpt", ckpt, "--tag", tag, "--ages", "fresh", "--eval-grid", grid or W.INP + "/eval_grid_windows.json",
            "--eval-kin", RAWT + "/eval141_kin.npz", "--eval-join", W.INP + "/join_eval141.npz", "--v9-eval", SF + "/inputs/v9/v9_labels_eval139.npz", "--lead-eval", SF + "/inputs/v9/lead_eval139.npz",
            "--speed-eval", SF + "/inputs/v9/refcv6_speed_max_v8_eval.jsonl", "--kin-mode", "yaw", "--p10", SF + "/inputs/p10/p10_prior.pt", "--eval-cache", SF + "/cache",
            "--tracks", SF + "/inputs/native_tracks.pt", "--out-dir", out_dir, "--device", device, "--ctx-pool", ctx_pool] + nav_args(ckpt)


def t1_argv(ckpt, tag, ctx_pool, out_dir, max_windows=0, device="cpu"):
    return ([PY, W.TREE + "/stack/scripts/v8f_t1_eval_pool.py", "run", "--ckpt", ckpt, "--tag", tag, "--tracks", SF + "/inputs/native_tracks.pt", "--eval-cache", SF + "/cache",
             "--eval-grid", W.INP + "/eval_grid_windows.json", "--eval-join", W.INP + "/join_eval141.npz", "--out-dir", out_dir, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged",
             "--schedule", "periodic", "--limit", "causal", "--dump-ticks", out_dir + "/ticks", "--dump-heads", "--device", device, "--ctx-pool", ctx_pool, "--dump-nav"]
            + nav_args(ckpt) + (["--max-windows", str(max_windows)] if max_windows else []))


def child_env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                PYTHONPATH=";".join(os.path.normpath(p) for p in (W.TREE + "/stack", W.TREE + "/taniteval")), STRAT1_TREE=W.TREE)


def run_argv(argv, log_path, wait_s=0):
    """-> rc.  75 = refused by the commit gate (nothing started); ``wait_s`` > 0 polls every 30 s for that long first (a script waits, no model turn does)."""
    import time
    t0 = time.time()
    while True:
        free = W.commit_headroom()
        if free >= W.COMMIT_MARGIN:
            break
        print(f"COMMIT-GATE: free commit {free / 1e9:.1f} GB < 5.5 GB: not started", flush=True)
        if time.time() - t0 >= wait_s:
            return 75
        time.sleep(30)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "ab") as lf:
        lf.write((" ".join(argv) + "\n").encode("utf-8"))
        return subprocess.run(argv, env=child_env(), stdout=lf, stderr=subprocess.STDOUT).returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("t0", "t1"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx-pool", choices=("on", "off"), default="on")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--grid", default="")
    ap.add_argument("--wait-s", type=int, default=0, help="poll the commit gate for this long before giving up (exit 75)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    tag = f"WH-{a.arm}_real" if a.kind == "t0" else f"WH-{a.arm}_m10_causal_heads"
    av = t0_argv(a.ckpt, tag, a.ctx_pool, a.out, a.grid or None) if a.kind == "t0" else t1_argv(a.ckpt, tag, a.ctx_pool, a.out, a.max_windows)
    json.dump(av, open(f"{a.out}/argv_{tag}.json", "w", encoding="utf-8"))
    rc = run_argv(av, f"{a.out}/log_{tag}.txt", a.wait_s)
    print(a.kind, a.arm, "rc", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
