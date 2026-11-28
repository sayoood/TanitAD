"""C7 diagnosis: plan ONE unit (arm, inference seed, batch) of R35_s<k> exactly as run_hwm.py plan does with --workers 1
(main process: run_hwm._init once, then run_hwm._work(unit)), and compare it, array by array, with
  (1) the banked phase-1 plans (HF rev aa2d8014) rows of that batch, and
  (2) the C7 driver's plans rows of that batch (optional).
Read-only on every input; writes only <out_npz> and <out_npz>.json.
usage: one_unit.py <label> <chassis 0|1> <arm> <seed> <batch> <out_npz> [c7_plans_dir]
env: the caller sets OMP/MKL threads, WMACT_CODE_ROOT and PYTHONPATH (see one_unit.sh)."""
import json, os, sys, time, argparse
sys.dont_write_bytecode = True
H = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan"
sys.path.insert(0, H)
import numpy as np
import torch
import run_hwm as R
import plantest_lib as P
import tanitad

label, k, arm, seed, b = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
out_npz = sys.argv[6]
c7dir = sys.argv[7] if len(sys.argv) > 7 else None
REF = f"F:/TanitAD-scratch/actsens/read/hf_phase1/R35/plan_runs/s{k}"
tag = f"R35_s{k}"
argv = ["plan", "--tag", tag, "--ckpt", f"C:/Users/Admin/t2v8f_pulls/ladder_{tag}/ckpt_last.pt",
        "--train-latents", f"F:/TanitAD-scratch/hwm_plan/latents_train_{tag}.pt", "--train-tag", tag,
        "--seed-mode", "prior", "--lam", "0", "--out", REF]
ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
p = sp.add_parser("plan"); R.add_common(p)
p.add_argument("--n", type=int, default=0)
p.add_argument("--arms", nargs="+", default=list(R.E.ARMS)); p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
p.add_argument("--workers", type=int, default=1); p.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args(argv)
info = dict(label=label, chassis=k, unit=[arm, seed, b, b + 1], torch_threads=torch.get_num_threads(),
            OMP=os.environ.get("OMP_NUM_THREADS"), MKL=os.environ.get("MKL_NUM_THREADS"),
            WMACT_CODE_ROOT=os.environ.get("WMACT_CODE_ROOT"), tanitad_file=os.path.abspath(tanitad.__file__),
            torch=torch.__version__)
print(json.dumps(info), flush=True)
t0 = time.time()
R._init({kk: v for kk, v in vars(a).items() if kk != "fn"})
if os.environ.get("PRIOR_NPZ"):                      # E5/E6: overwrite the fitted action priors with those dumped by prior_probe.py under another thread count
    pz = np.load(os.environ["PRIOR_NPZ"])
    st_ = R._CTX["st"]
    n_set = 0
    for lv in (1, 2, 3):
        for nm in ("mean", "std", "lo", "hi"):
            buf = getattr(st_, f"prior_{nm}_{lv}")
            buf.copy_(torch.from_numpy(pz[f"buf_prior_{nm}_{lv}"]).to(buf.dtype))
            n_set += 1
    assert R._CTX["planner"].st is st_
    info["prior_npz"] = os.environ["PRIOR_NPZ"]; info["prior_buffers_set"] = n_set
t1 = time.time()
u, r = R._work((arm, seed, b, b + 1))
t2 = time.time()
info.update(init_s=round(t1 - t0, 1), unit_s=round(t2 - t1, 1))
np.savez(out_npz, **r)
lo, hi = b * P.BATCH, b * P.BATCH + len(r["idx"])


def cmp(other_path):
    o = np.load(other_path)
    res = {}
    for kk in r:
        if kk == "wall_s":
            continue
        x = np.asarray(r[kk]); y = o[kk][lo:hi]
        same = bool(x.shape == y.shape and np.array_equal(x, y, equal_nan=True))
        mad = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.shape == y.shape else None
        res[kk] = dict(bit_equal=same, max_abs_diff=mad)
    res["ALL_BIT_EQUAL"] = all(v["bit_equal"] for v in res.values())
    return res


info["vs_banked_phase1"] = cmp(f"{REF}/plans_{arm}_s{seed}.npz")
if c7dir:
    info["vs_c7_driver"] = cmp(f"{c7dir}/plans_{arm}_s{seed}.npz")
json.dump(info, open(out_npz + ".json", "w"), indent=1)
print(json.dumps(info, indent=1), flush=True)
