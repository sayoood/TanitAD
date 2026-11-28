"""run_t0_strat1.py -- the T0 STRATEGIC passes of the STRAT1 read (READ_PLAN 2.3; CPU, 2 threads, zero training).
Usage: python run_t0_strat1.py [--arms C0_s0 ...] [--variants real donor zero] [--k4 S1_s0 S1_s1] [--grid GRID.json] [--out DIR]
For every arm and variant one `v8f_mini_eval.py run --ages fresh` (real = neither flag, donor = --nav-shuffle (donor_indices seed 0, episode-disjoint), zero = --nav-withheld), writing
eval_STRAT1-<arm>_<variant>.npz / .json into DIR.  The nav-text arguments come from the checkpoint's cfg.nav_source exactly as run_dumps_strat1 (text arm: the pinned npz + its sha256; C0: none).
K4 (C-RAND): strat1_k4_eval.py, the same argv for the S1 checkpoint with its ctx_head re-initialised in memory (seed 0): eval_STRAT1-<arm>_k4.npz.  A pass whose npz exists is skipped.
Every checkpoint is sha256-checked against its BANK_MANIFEST.json first (the same refusal as run_dumps_strat1.refusal)."""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strat1_lib as L  # noqa: E402
import run_dumps_strat1 as RDS  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ART = "E:/Projects/TanitAD-artifacts"
SF = ART + "/sfresh_inputs"
RAWT = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
VARIANT_FLAGS = {"real": [], "donor": ["--nav-shuffle"], "zero": ["--nav-withheld"], "k4": []}


def argv_for(arm, variant, out, grid=None, nav_source=None):
    tag = f"STRAT1-{arm}"
    ck = f"{L.CK}/{arm}/ckpt_last.pt"
    src = nav_source or RDS.nav_source_of(ck)
    if (src == "text") != (arm in L.TEXT_ARMS):
        raise ValueError(f"{arm}: nav_source {src!r} does not match the card's arm")
    nav = RDS.nav_args(arm, src, L.pinned("navtext_nav.npz") if src == "text" else "", L.PINS["navtext_nav.npz"] if src == "text" else "")
    script = HERE + "/strat1_k4_eval.py" if variant == "k4" else L.TREE + "/stack/scripts/v8f_mini_eval.py"
    a = [PY, script, "run", "--ckpt", ck, "--tag", f"{tag}_{variant}", "--ages", "fresh",
         "--eval-grid", grid or L.INP + "/eval_grid_windows.json", "--eval-kin", RAWT + "/eval141_kin.npz", "--eval-join", L.INP + "/join_eval141.npz",
         "--v9-eval", SF + "/inputs/v9/v9_labels_eval139.npz", "--lead-eval", SF + "/inputs/v9/lead_eval139.npz", "--speed-eval", SF + "/inputs/v9/refcv6_speed_max_v8_eval.jsonl",
         "--kin-mode", "yaw", "--p10", SF + "/inputs/p10/p10_prior.pt", "--eval-cache", SF + "/cache", "--tracks", SF + "/inputs/native_tracks.pt",
         "--out-dir", out, "--device", "cpu"] + nav + VARIANT_FLAGS[variant]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(L.ARMS))
    ap.add_argument("--variants", nargs="*", default=["real", "donor", "zero"])
    ap.add_argument("--k4", nargs="*", default=list(L.CTX_ARMS))
    ap.add_argument("--grid", default="")
    ap.add_argument("--out", default=L.OUTD + "/t0")
    a = ap.parse_args()
    tp = RDS.tree_problem() or RDS.import_check()
    if tp:
        raise SystemExit("REFUSED: " + tp)
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=";".join(os.path.normpath(p) for p in (L.TREE + "/stack", L.TREE + "/taniteval")), STRAT1_TREE=L.TREE)
    jobs = [(arm, v) for arm in a.arms for v in a.variants] + [(arm, "k4") for arm in a.k4]
    rc = 0
    for arm, v in jobs:
        why = RDS.refusal(f"STRAT1-{arm}")
        if why:
            print("REFUSED", arm, v, why, flush=True)
            rc = 1
            continue
        npz = f"{a.out}/eval_STRAT1-{arm}_{v}.npz"
        if os.path.exists(npz):
            print("skip (exists)", npz, flush=True)
            continue
        t0 = time.time()
        with open(f"{a.out}/log_STRAT1-{arm}_{v}.txt", "w", encoding="utf-8") as lf:
            argv = argv_for(arm, v, a.out, a.grid or None)
            open(f"{a.out}/argv_STRAT1-{arm}_{v}.txt", "w", encoding="utf-8").write("\n".join(argv) + "\n")
            r = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8")
        ok = r.returncode == 0 and os.path.exists(npz)
        print(time.strftime("%H:%M:%S"), arm, v, "rc", r.returncode, "ok" if ok else "FAILED", round(time.time() - t0), "s", flush=True)
        rc |= (not ok)
    sys.exit(1 if rc else 0)


if __name__ == "__main__":
    main()
