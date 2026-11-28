"""navrb2_t0.py -- the T0 STRATEGIC passes of the NAVRB2 read for the four R arms (CPU, 2 threads, zero training).
Usage: python navrb2_t0.py [--arms N1R_s0 ...] [--variants real donor zero] [--k4 S1R_s0 S1R_s1] [--grid GRID.json] [--out DIR]
Per arm and variant one `v8f_mini_eval_rb.py run --ages fresh` (the e71782a3 tree's study wrapper over the pinned v8f_mini_eval.py), writing eval_NAVRB2-<arm>_<variant>.npz / .json into DIR:
  real  = --nav-rb on --nav-rb-variant follow;   zero = --nav-withheld (the channel follows: all 0.0);
  donor = --nav-shuffle with --nav-rb-variant zero (READING r4: the evaluator swaps only the nav array, so the channel is held at zero beside the donor nav; the T1 donor rule is not available at T0).
K4 = navrb2_k4_eval.py (the S1R checkpoint's ctx_head re-initialised in memory).  Every arm is checked against its bank manifest first (navrb2_lib.refusal).  A pass whose npz exists is skipped.
Everything else in the argv is run_t0_strat1.argv_for's (the pinned STRAT1 T0 recipe)."""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrb2_lib as NL  # noqa: E402

sys.path.insert(0, NL.SR)
import strat1_lib as L  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ART = "E:/Projects/TanitAD-artifacts"
SF = ART + "/sfresh_inputs"
RAWT = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
VARIANT_FLAGS = {"real": ["--nav-rb-variant", "follow"], "donor": ["--nav-shuffle", "--nav-rb-variant", "zero"], "zero": ["--nav-withheld"], "k4": ["--nav-rb-variant", "follow"]}


def argv_for(arm, variant, out, grid=None):
    ck = f"{NL.CK}/{arm}/ckpt_last.pt"
    script = HERE + "/navrb2_k4_eval.py" if variant == "k4" else NL.TREE + "/stack/scripts/v8f_mini_eval_rb.py"
    nav = ["--nav-text-npz", L.pinned("navtext_nav.npz", NL.SR + "/inputs"), "--nav-text-sha256", L.PINS["navtext_nav.npz"]]
    rb = ["--nav-rb", "on", "--nav-rb-npz", NL.rb_npz_checked(), "--nav-rb-sha256", NL.RB_SHA256]
    return [PY, script, "run", "--ckpt", ck, "--tag", f"NAVRB2-{arm}_{variant}", "--ages", "fresh",
            "--eval-grid", grid or NL.SR + "/inputs/eval_grid_windows.json", "--eval-kin", RAWT + "/eval141_kin.npz", "--eval-join", NL.SR + "/inputs/join_eval141.npz",
            "--v9-eval", SF + "/inputs/v9/v9_labels_eval139.npz", "--lead-eval", SF + "/inputs/v9/lead_eval139.npz", "--speed-eval", SF + "/inputs/v9/refcv6_speed_max_v8_eval.jsonl",
            "--kin-mode", "yaw", "--p10", SF + "/inputs/p10/p10_prior.pt", "--eval-cache", SF + "/cache", "--tracks", SF + "/inputs/native_tracks.pt",
            "--out-dir", out, "--device", "cpu"] + nav + rb + VARIANT_FLAGS[variant]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(NL.R_ARMS))
    ap.add_argument("--variants", nargs="*", default=["real", "donor", "zero"])
    ap.add_argument("--k4", nargs="*", default=[a for a in NL.R_ARMS if a.startswith("S1R")])
    ap.add_argument("--grid", default="")
    ap.add_argument("--out", default=NL.T0)
    a = ap.parse_args()
    tp = NL.tree_problem() or NL.import_check(PY)
    if tp:
        raise SystemExit("REFUSED: " + tp)
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=";".join(os.path.normpath(p) for p in (NL.TREE + "/stack", NL.TREE + "/taniteval")), NAVRB2_TREE=NL.TREE)
    jobs = [(arm, v) for arm in a.arms for v in a.variants] + [(arm, "k4") for arm in a.k4]
    rc = 0
    for arm, v in jobs:
        why = NL.refusal(arm)
        if why:
            print("REFUSED", arm, v, why, flush=True)
            rc = 1
            continue
        npz = f"{a.out}/eval_NAVRB2-{arm}_{v}.npz"
        if os.path.exists(npz):
            print("skip (exists)", npz, flush=True)
            continue
        t0 = time.time()
        argv = argv_for(arm, v, a.out, a.grid or None)
        open(f"{a.out}/argv_NAVRB2-{arm}_{v}.txt", "w", encoding="utf-8").write("\n".join(argv) + "\n")
        with open(f"{a.out}/log_NAVRB2-{arm}_{v}.txt", "w", encoding="utf-8") as lf:
            r = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8")
        ok = r.returncode == 0 and os.path.exists(npz)
        print(time.strftime("%H:%M:%S"), arm, v, "rc", r.returncode, "ok" if ok else "FAILED", round(time.time() - t0), "s", flush=True)
        rc |= (not ok)
    sys.exit(1 if rc else 0)


if __name__ == "__main__":
    main()
