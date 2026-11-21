#!/usr/bin/env python3
"""Prerequisite 1 of PREREG_V8F_SPEEDSUB s6: attribution refits of the banked ego-only read-out `mlp_ego` (and its ridge twin `lin_ego`).

CPU only, <= 4 threads. Uses the SAME instrument functions that produced fit.json (v8f_mini_eval_speedread.py, sha fd5ce0be..., the SPEEDREAD A1 file whose fit code is
byte-identical to the SPEEDSUB staged instrument's), on the SAME train arrays, folds (sha256(clip) mod 5), holdout fold 0, lambdas, mlp_cfg and seed.
No eval window is read. No substitution is run. Reads only train_arrays.npz, fit.json, readout.pt (control) and READ.json (T_own's eval-141 numbers, beside only).

Designs (columns of the 22-dim ego block: p10f 0:8 | nav 8:13 | limit 13:17 | limit_ok 17:18 | frame_ok 18:22):
  full        all 22                         (CONTROL: must reproduce fit.json mlp_ego / lin_ego)
  a_p10f_fok  p10f + frame_ok                (12)  requested (a)
  b_nonav     p10f + limit + limit_ok + frame_ok (17)  requested (b): the full block minus nav
  c_nolimit   p10f + nav + frame_ok          (17)  EXTRA (completes the 2x2: a+nav)
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

INSTR = "F:/TanitAD-scratch/speedread/tree_f3971165/stack/scripts/v8f_mini_eval_speedread.py"
INSTR_SHA = "fd5ce0bea37c3e0e608144e041f54a337fd4378b38fb312df65787bc864be4fe"
TRAIN = "F:/TanitAD-scratch/speedread/train/train_arrays.npz"
FITJSON = "F:/TanitAD-scratch/speedread/fit/fit.json"
READOUT = "F:/TanitAD-scratch/speedread/fit/readout.pt"
READ = "F:/TanitAD-scratch/speedread/read/READ.json"
OUT = "F:/TanitAD-scratch/speedsub_t1_prep"

COLS = {"p10f": range(0, 8), "nav": range(8, 13), "limit": range(13, 17), "limit_ok": range(17, 18), "frame_ok": range(18, 22)}
DESIGNS = {
    "full": ["p10f", "nav", "limit", "limit_ok", "frame_ok"],
    "a_p10f_fok": ["p10f", "frame_ok"],
    "b_nonav": ["p10f", "limit", "limit_ok", "frame_ok"],
    "c_nolimit": ["p10f", "nav", "frame_ok"],
}
CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_instr():
    assert sha_file(INSTR) == INSTR_SHA, "instrument sha differs"
    spec = importlib.util.spec_from_file_location("sr_instr", INSTR)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sr_instr"] = m
    spec.loader.exec_module(m)
    return m


def cols_of(groups):
    return [c for g in groups for c in COLS[g]]


def main(only=None):
    t0 = time.time()
    sr = load_instr()
    fit = json.load(open(FITJSON, encoding="utf-8"))
    assert sha_file(READOUT) == fit["readout_sha256"], "readout.pt != fit.json readout_sha256"
    z = np.load(TRAIN)
    ego, Y, M, clips = z["ego"], z["y"], z["m"], z["clip"]
    print("[arrays]", {k: (z[k].shape, str(z[k].dtype)) for k in z.files}, flush=True)
    assert ego.shape[1] == 22 and Y.shape[1] == 5 and M.shape == Y.shape
    # ---- layout control: the column blocks must look like what the source says they are
    lay = {
        "limit_rowsum_in_0_1": bool(np.isin(ego[:, 13:17].sum(1), [0.0, 1.0]).all()),
        "limit_ok_binary": bool(np.isin(ego[:, 17], [0.0, 1.0]).all()),
        "frame_ok_binary": bool(np.isin(ego[:, 18:22], [0.0, 1.0]).all()),
        "nav_onehot_first3_rowsum_in_0_1": bool(np.isin(ego[:, 8:11].sum(1), [0.0, 1.0]).all()),
        "p10f_v_mean_ms": float(ego[:, 0].mean()),
        "limit_ok_iff_onehot": bool(((ego[:, 13:17].sum(1) == 1.0) == (ego[:, 17] == 1.0)).all()),
        "nav_col3_unique_head": [float(v) for v in np.unique(ego[:, 11])[:6]],
        "nav_col4_min_max": [float(ego[:, 12].min()), float(ego[:, 12].max())],
    }
    print("[layout]", lay, flush=True)

    E = ego.astype(np.float32)
    em, es = sr.standardiser(E)
    Ez = (E - em) / es
    folds = sr.fold_of(clips)
    cvf = [f for f in range(sr.N_FOLDS) if f != sr.HOLDOUT_FOLD]
    tr, ho = np.flatnonzero(folds != sr.HOLDOUT_FOLD), np.flatnonzero(folds == sr.HOLDOUT_FOLD)
    sd = np.array([np.std(Y[M[:, c], c]) for c in range(Y.shape[1])])
    const = np.array([float(np.mean(Y[M[:, c], c])) for c in range(Y.shape[1])])
    chk = {
        "train_sd_vs_fit_maxabs": float(np.abs(sd - np.array(fit["train_sd"])).max()),
        "const_vs_fit_maxabs": float(np.abs(const - np.array(fit["const"])).max()),
        "n_present_vs_fit": [int(v) for v in M.sum(0)] == fit["n_present_per_cell"],
        "n_windows": int(len(E)), "n_clips": int(len(set(map(str, clips)))), "n_tr": int(len(tr)), "n_ho": int(len(ho)),
        "ho_clips": int(len(set(map(str, clips[ho])))),
        "mlp_cfg_equal_instrument": dict(sr.MLP_CFG) == fit["mlp_cfg"], "lambdas_equal": list(sr.LAMBDAS) == fit["lambdas"],
    }
    print("[inputs-vs-fit.json]", chk, flush=True)
    assert chk["n_present_vs_fit"] and chk["mlp_cfg_equal_instrument"] and chk["lambdas_equal"]
    const_ho_rmse = sr.masked_rmse(np.broadcast_to(const, Y[ho].shape), Y[ho], M[ho])
    n_ho = [int(v) for v in M[ho].sum(0)]

    res = {"schema": "speedsub-t1-prep-refit-v1", "instrument": INSTR, "instrument_sha256": INSTR_SHA, "train": TRAIN, "train_sha256": sha_file(TRAIN),
           "fit_json_sha256": sha_file(FITJSON), "layout_check": lay, "input_checks": chk, "threads": 4, "torch": torch.__version__, "numpy": np.__version__,
           "holdout_n_per_cell": n_ho, "const_holdout_rmse_ms": [float(v) for v in const_ho_rmse], "train_sd": [float(v) for v in sd], "designs": {}}
    banked = sr.load_readouts(READOUT)

    def metrics(P, name):
        rm = sr.masked_rmse(P, Y[ho], M[ho])
        return {"holdout_rmse_sd": [float(v) for v in rm / sd], "holdout_rmse_ms": [float(v) for v in rm],
                "holdout_skill_vs_trainmean": [float(v) for v in 1.0 - rm ** 2 / const_ho_rmse ** 2]}

    for name, groups in DESIGNS.items():
        if only and name not in only:
            continue
        cols = cols_of(groups)
        D_ = np.ascontiguousarray(Ez[:, cols])
        ts = time.time()
        rec = {"groups": groups, "cols": cols, "d": len(cols)}
        # ---------------- MLP (== mlp_ego when groups == full)
        m = sr.mlp_fit(D_, Y, M, tr, ho, sr.MLP_CFG)
        Pm = sr.mlp_predict(m, D_[ho])
        rec["mlp"] = {"best_epoch": int(m["best_epoch"]), **metrics(Pm, "mlp")}
        print(f"[{name}] mlp best_epoch {m['best_epoch']} ho_sd {np.round(rec['mlp']['holdout_rmse_sd'], 5).tolist()}  ({time.time() - ts:.0f}s)", flush=True)
        # ---------------- ridge (== lin_ego when groups == full)
        cv = sr.ridge_cv(D_, Y, M, folds, cvf)
        Mtr = M & (folds != sr.HOLDOUT_FOLD)[:, None]
        Wh, bh = sr.ridge_fit(D_, Y, Mtr, cv["lam"])
        Pr = D_[ho] @ Wh + bh
        rec["ridge"] = {"lam": cv["lam"], **metrics(Pr, "ridge")}
        Wf, bf = sr.ridge_fit(D_, Y, M, cv["lam"])
        print(f"[{name}] ridge lam {cv['lam']} ho_sd {np.round(rec['ridge']['holdout_rmse_sd'], 5).tolist()}  ({time.time() - ts:.0f}s)", flush=True)
        # ---------------- bank the read-outs (plain tensors; same container as readout.pt). Reduced designs read `cols` of the 22-dim ego block.
        xm, xs = em[cols], es[cols]
        ros = {"mlp_ego_" + name: sr.ReadOut("mlp_ego", mlp=m, ego_only=True, pca_mu=None, pca_V=None, xm=xm, xs=xs),
               "lin_ego_" + name: sr.ReadOut("lin_ego", lin=(Wf, bf), ego_only=True, pca_mu=None, pca_V=None, xm=xm, xs=xs)}
        path = f"{OUT}/readout_refit_{name}.pt"
        sr.save_readouts(ros, path)
        rec["readout_file"] = path
        rec["readout_sha256"] = sha_file(path)
        if name == "full":
            ctl = {}
            b = banked["mlp_ego"]
            # (1) the refit MLP vs the banked mlp_ego: weights and holdout predictions
            diffs = {k: float((m["state"][k].float() - b.mlp["state"][k].float()).abs().max()) for k in m["state"]}
            Pb = sr.mlp_predict(b.mlp, D_[ho])
            ctl["mlp_state_max_abs_diff_per_tensor"] = diffs
            ctl["mlp_state_max_abs_diff"] = max(diffs.values())
            ctl["mlp_ho_pred_max_abs_diff_ms"] = float(np.abs(Pm - Pb).max())
            ctl["mlp_best_epoch_equal"] = int(m["best_epoch"]) == int(fit["mlp_ego"]["best_epoch"])
            ctl["mlp_ho_rmse_sd_max_abs_diff_vs_fit_json"] = float(np.abs(np.array(rec["mlp"]["holdout_rmse_sd"]) - np.array(fit["mlp_ego"]["holdout_rmse_sd"])).max())
            ctl["mlp_ym_ys_max_abs_diff"] = float(max(np.abs(m["ym"] - b.mlp["ym"]).max(), np.abs(m["ys"] - b.mlp["ys"]).max()))
            ctl["xm_xs_max_abs_diff_vs_banked"] = float(max(np.abs(xm - b.xm).max(), np.abs(xs - b.xs).max()))
            # (2) the ridge twin
            lb = banked["lin_ego"]
            ctl["ridge_lam_equal"] = [float(v) for v in cv["lam"]] == [float(v) for v in fit["lin_ego"]["lam"]]
            ctl["ridge_cv_rmse_max_abs_diff_vs_fit_json"] = float(np.abs(np.array(cv["cv_rmse"]) - np.array(fit["lin_ego"]["cv_rmse"])).max())
            ctl["ridge_ho_rmse_sd_max_abs_diff_vs_fit_json"] = float(np.abs(np.array(rec["ridge"]["holdout_rmse_sd"]) - np.array(fit["lin_ego"]["holdout_rmse_sd"])).max())
            ctl["ridge_final_W_max_abs_diff_vs_banked"] = float(np.abs(Wf - lb.lin[0]).max())
            ctl["ridge_final_b_max_abs_diff_vs_banked"] = float(np.abs(bf - lb.lin[1]).max())
            ctl["ridge_final_ho_pred_max_abs_diff_ms"] = float(np.abs(D_[ho] @ Wf + bf - (D_[ho] @ lb.lin[0] + lb.lin[1])).max())
            rec["control"] = ctl
            print("[CONTROL]", json.dumps({k: v for k, v in ctl.items() if k != "mlp_state_max_abs_diff_per_tensor"}), flush=True)
        res["designs"][name] = rec
        json.dump(res, open(f"{OUT}/refit_{'_'.join(sorted(only)) if only else 'all'}.json", "w", encoding="utf-8", newline="\n"), indent=1)
    # ---- the banked mlp_ego / lin_ego numbers from fit.json, and T_own / banked mlp_ego on eval-141 (beside only; a different population)
    res["fit_json_beside"] = {"mlp_ego": {"best_epoch": fit["mlp_ego"]["best_epoch"], "holdout_rmse_sd": fit["mlp_ego"]["holdout_rmse_sd"]},
                              "lin_ego": {"lam": fit["lin_ego"]["lam"], "holdout_rmse_sd": fit["lin_ego"]["holdout_rmse_sd"]}}
    rd = json.load(open(READ, encoding="utf-8"))
    res["eval141_beside"] = {"source": READ, "read_json_sha256": sha_file(READ),
                             "note": "eval-141 windows, NOT the TRAIN fold-0 holdout; T_own exists only on eval windows. Beside only, never a like-for-like comparison.",
                             "T_own": {s: rd["cells"][s]["T_own"] for s in ("0", "1")}, "mlp_ego": {s: rd["cells"][s]["mlp_ego"] for s in ("0", "1")},
                             "lin_ego": {s: rd["cells"][s]["lin_ego"] for s in ("0", "1")}, "const": {s: rd["cells"][s]["const"] for s in ("0", "1")}}
    res["seconds"] = round(time.time() - t0, 1)
    out = f"{OUT}/refit_{'_'.join(sorted(only)) if only else 'all'}.json"
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("[done]", out, f"{res['seconds']}s", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
