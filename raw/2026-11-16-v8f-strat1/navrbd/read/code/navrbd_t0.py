"""navrbd_t0.py -- the T0 STRATEGIC passes of the NAVRBD read for the two N1D arms (CPU, 2 threads, zero training).  Adapted from navrb2_t0.py (same recipe, the RBD channel flags).
Usage: python navrbd_t0.py [--arms N1D_s0 N1D_s1] [--variants real donor zero] [--grid GRID.json] [--out DIR]
Per arm and variant one `v8f_mini_eval_rbd.py run --ages fresh` (the NAVRBD tree's study wrapper over the pinned v8f_mini_eval.py), writing eval_NAVRBD-<arm>_<variant>.npz / .json into DIR:
  real  = --nav-rb on --nav-rb-feat exitdist --nav-rb-variant follow;
  zero  = --nav-withheld (the channel follows: all 0.0);
  donor = --nav-shuffle with --nav-rb-variant zero.  DONOR MEANING (as navrb2_t0.py, READING r4): the evaluator swaps ONLY the base nav array for the donor clip's, and the RB channel is held at ZERO beside it
          (the wrapper REFUSES --nav-shuffle with variant follow; the T1 donor rule, which would take the donor's channel rows, is not available at T0).  This is the variant NAVRB2 used; nothing new is invented.
There is no K4 pass here (no S1D arm exists).  Every arm is checked first (navrbd_dumps.refusal_ck: checkpoint exists, bank manifest sha256 when present).  A pass whose npz exists is skipped.
Everything else in the argv is the pinned STRAT1 T0 recipe (as navrb2_t0.argv_for)."""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrbd_dumps as ND  # noqa: E402
import navrb2_lib as NL  # noqa: E402

sys.path.insert(0, NL.SR)
import strat1_lib as L  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ART = "E:/Projects/TanitAD-artifacts"
SF = ART + "/sfresh_inputs"
RAWT = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
T0_OUT = os.environ.get("NAVRBD_T0", ND.NR.HERE + "/out/t0")
VARIANT_FLAGS = {"real": ["--nav-rb-variant", "follow"], "donor": ["--nav-shuffle", "--nav-rb-variant", "zero"], "zero": ["--nav-withheld"]}
D_ARMS = ND.D_ARMS


def argv_for(arm, variant, out, grid=None, tree=None):
    ck = f"{ND.CK}/{arm}/ckpt_last.pt"
    script = (tree or ND.TREE) + "/stack/scripts/v8f_mini_eval_rbd.py"
    nav = ["--nav-text-npz", L.pinned("navtext_nav.npz", NL.SR + "/inputs"), "--nav-text-sha256", L.PINS["navtext_nav.npz"]]
    rb = ["--nav-rb", "on", "--nav-rb-feat", "exitdist", "--nav-rb-npz", ND.rb_npz_checked(), "--nav-rb-sha256", ND.RB_SHA256]
    return [PY, script, "run", "--ckpt", ck, "--tag", f"NAVRBD-{arm}_{variant}", "--ages", "fresh",
            "--eval-grid", grid or NL.SR + "/inputs/eval_grid_windows.json", "--eval-kin", RAWT + "/eval141_kin.npz", "--eval-join", NL.SR + "/inputs/join_eval141.npz",
            "--v9-eval", SF + "/inputs/v9/v9_labels_eval139.npz", "--lead-eval", SF + "/inputs/v9/lead_eval139.npz", "--speed-eval", SF + "/inputs/v9/refcv6_speed_max_v8_eval.jsonl",
            "--kin-mode", "yaw", "--p10", SF + "/inputs/p10/p10_prior.pt", "--eval-cache", SF + "/cache", "--tracks", SF + "/inputs/native_tracks.pt",
            "--out-dir", out, "--device", "cpu"] + nav + rb + VARIANT_FLAGS[variant]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(D_ARMS))
    ap.add_argument("--variants", nargs="*", default=["real", "donor", "zero"])
    ap.add_argument("--grid", default="")
    ap.add_argument("--out", default=T0_OUT)
    a = ap.parse_args()
    tp = ND.tree_problem() or NL.import_check(PY, ND.TREE)
    if tp:
        raise SystemExit("REFUSED: " + tp)
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=";".join(os.path.normpath(p) for p in (ND.TREE + "/stack", ND.TREE + "/taniteval")))
    rc = 0
    for arm in a.arms:
        for v in a.variants:
            why = ND.refusal_ck(arm)
            if why:
                print("REFUSED", arm, v, why, flush=True)
                rc = 1
                continue
            npz = f"{a.out}/eval_NAVRBD-{arm}_{v}.npz"
            if os.path.exists(npz):
                print("skip (exists)", npz, flush=True)
                continue
            t0 = time.time()
            argv = argv_for(arm, v, a.out, a.grid or None)
            open(f"{a.out}/argv_NAVRBD-{arm}_{v}.txt", "w", encoding="utf-8").write("\n".join(argv) + "\n")
            with open(f"{a.out}/log_NAVRBD-{arm}_{v}.txt", "w", encoding="utf-8") as lf:
                r = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8")
            ok = r.returncode == 0 and os.path.exists(npz)
            print(time.strftime("%H:%M:%S"), arm, v, "rc", r.returncode, "ok" if ok else "FAILED", round(time.time() - t0), "s", flush=True)
            rc |= (not ok)
    sys.exit(1 if rc else 0)


if __name__ == "__main__":
    main()
