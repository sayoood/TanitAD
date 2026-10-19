"""progress_scalar.py -- a MODEL-FREE progress regressor from the admissible t0 scalars, fit on TRAIN clips only, as the floor a
learned progress head must beat.  (refcv9 L1 design annex, item 4, arm E0-c.  CPU only, banked dumps, no model call.)

Target per S2 instant k: h_k = s_H(t_k) - s_P(t_k)  (the human's along-track residual over the causal prior ha0_ext_pose).
Features (all admissible at t0: measured ego state + the fed speed-ceiling label the model already receives; NO tactical / situation
output, NO future):  [1, v0, a0, omega0, v0*omega0, v0^2, a0*v0, min(v_lim,40), min(v_lim,40)-v0, v_lim_valid]
  v0, a0, omega0 from kinematic_prior.prior_controls('ha0_ext_pose') on the observed pose window (the prior's own inputs).
Ridge per instant; lambda chosen by 5-fold EPISODE cross-validation ON THE TRAIN CLIPS ONLY (never on eval).
Arms scored on eval139 compliant S2 (paired episode-cluster bootstrap, pair_read_r8.paired_cell):
    PROG_SCALAR_OS     refcv8's own PATH, progress s' = s_P + h_hat           (a progress head with ego-scalar inputs only)
    PROG_SCALAR_PRIOR  the PRIOR's path, progress s' = s_P + h_hat            (fully model-free: a better prior, NOT the model)
    RETIME_TRAIN       the shipped stage (S2 train gains)                     (reference)
Controls: RETIME_TRAIN reproduces FORWARD_TRAINFIT (-0.0318 / -0.0313); h_hat := 0 reproduces the prior exactly (PROG_SCALAR_PRIOR with
zero correction == ha0_ext_pose, ADE delta 0.0000).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import along_track_localise as AL  # noqa: E402
import progress_headroom as PH  # noqa: E402

RT = AL.RT
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def feats(win: np.ndarray, lim: np.ndarray) -> np.ndarray:
    import torch
    from tanitad.models import kinematic_prior as KP
    from tanitad.models.ego_history import ego_channels_from_poses
    pw = torch.from_numpy(np.ascontiguousarray(win, dtype=np.float32))
    a0, _k = KP.prior_controls("ha0_ext_pose", pw)
    ch = ego_channels_from_poses(pw, AL.RT.W_OBS, dt=AL.RT.DT)
    v0, om = ch[:, -1, 0].numpy().astype(np.float64), ch[:, -1, 2].numpy().astype(np.float64)
    a0 = a0.numpy().astype(np.float64)
    lim = np.asarray(lim, np.float64)
    valid = np.isfinite(lim)
    vl = np.where(valid, np.minimum(lim, 40.0), 40.0)
    return np.stack([np.ones_like(v0), v0, a0, om, v0 * om, v0 ** 2, a0 * v0, vl, vl - v0, valid.astype(np.float64)], 1)


def ridge_fit(X, Y, lam):
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
    Z = np.concatenate([X[:, :1], (X[:, 1:] - mu) / sd], 1)
    R = lam * np.eye(Z.shape[1])
    R[0, 0] = 0.0
    W = np.linalg.solve(Z.T @ Z + R, Z.T @ Y)
    return {"W": W, "mu": mu, "sd": sd}


def ridge_pred(m, X):
    Z = np.concatenate([X[:, :1], (X[:, 1:] - m["mu"]) / m["sd"]], 1)
    return Z @ m["W"]


def cv_lambda(X, Y, eid, k=5, seed=0) -> tuple[float, dict]:
    u = np.unique(eid)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(u, rng.permutation(len(u)) % k))
    f = np.asarray([fold_of[e] for e in eid])
    score = {}
    for lam in LAMBDAS:
        sse = 0.0
        for j in range(k):
            tr, te = f != j, f == j
            m = ridge_fit(X[tr], Y[tr], lam)
            sse += float(((ridge_pred(m, X[te]) - Y[te]) ** 2).sum())
        score[lam] = sse
    best = min(score, key=score.get)
    return best, {str(k_): v for k_, v in score.items()}


def load(tools, CRmod, tree, seed, first, train: bool):
    ctx = CRmod.prepare_seed(tree, seed, tools, tools["rc"])
    if train:
        import retime_trainfit_r8 as TF
        S = TF.train_surface_inputs(tools, ctx, "S2", first, tree)
    else:
        S = RT.surface_inputs(tools, ctx, "S2", first, tree)
    rows = ctx["panels"]["S2"]["rows"]
    lim = ctx["lim"][rows]
    sel = np.asarray(S["mask"], bool)
    d = {"O": np.asarray(S["X"]["O"])[sel], "HP": np.asarray(S["X"]["HP"])[sel], "G": np.asarray(S["X"]["G"])[sel],
         "X": feats(S["dec"]["win"][sel], lim[sel]), "eid": np.asarray([str(e) for e in S["eid"]])[sel], "dt": S["dt"]}
    del ctx
    return d


def main() -> int:
    t0 = time.time()
    import compliant_reread_r8 as CR
    tools = RT.tools_load()
    PR, ra = tools["PR"], tools["ra"]
    ev = {s: load(tools, CR, "step30000ema_unc", s, {}, False) for s in AL.SEEDS}
    import retime_trainfit_r8 as TF
    _cp, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    ftr = {}
    tr = {s: load(tools, CR2, TF.TRAIN_TREE, s, ftr, True) for s in AL.SEEDS}
    # the target and features are MODEL-FREE (prior, human, ego state): identical across inference seeds -> fit once on seed 0's rows
    for k in ("HP", "G", "X"):
        assert np.array_equal(tr[0][k], tr[1][k]) and np.array_equal(ev[0][k], ev[1][k]), k
    Xtr, eidtr = tr[0]["X"], tr[0]["eid"]
    Htr = AL.arc(tr[0]["G"]) - AL.arc(tr[0]["HP"])
    lam, cvs = cv_lambda(Xtr, Htr, eidtr)
    m = ridge_fit(Xtr, Htr, lam)
    out = {"tool": "progress_scalar.py", "generated": time.strftime("%FT%T"), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "fit": {"set": "train139 EMA-30k tree, ceiling-compliant S2 windows (model-free target; inference seeds identical)", "n_windows": int(len(Xtr)),
                   "n_episodes": int(len(np.unique(eidtr))), "lambda": lam, "cv_sse": cvs,
                   "train_R2_per_instant": [float(1 - ((ridge_pred(m, Xtr)[:, k] - Htr[:, k]) ** 2).sum() / ((Htr[:, k] - Htr[:, k].mean()) ** 2).sum()) for k in range(4)]},
           "tier": "T1 open loop on logged eval139 frames; PROG_SCALAR_PRIOR is model-free", "seeds": {}, "controls": {}}
    banked = json.loads(PH.BANKED.read_text(encoding="utf-8"))
    for s in AL.SEEDS:
        E = ev[s]
        sp = AL.arc(E["HP"])
        Heval = AL.arc(E["G"]) - sp
        hh = ridge_pred(m, E["X"])
        arms = {"os": E["O"], "RETIME_TRAIN": RT.retime_apply(E["O"], E["HP"], np.asarray(PH.TRAIN_S2, np.float64)),
                "PROG_SCALAR_OS": PH.at_progress(E["O"], sp + hh), "PROG_SCALAR_PRIOR": PH.at_progress(E["HP"], sp + hh),
                "ZERO_CORR_PRIOR": PH.at_progress(E["HP"], sp)}
        ade = {k: PR.components(ra, v, E["G"], E["dt"])["ade_m"] for k, v in arms.items()}
        ade_hp = PR.components(ra, E["HP"], E["G"], E["dt"])["ade_m"]
        eid = list(E["eid"])
        cells = {k: PR.paired_cell(v, ade_hp, eid) for k, v in ade.items()}
        vs_rt = {k: PR.paired_cell(ade[k], ade["RETIME_TRAIN"], eid) for k in ("PROG_SCALAR_OS", "PROG_SCALAR_PRIOR")}
        b = banked["seeds"][str(s)]["surfaces"]["S2"]["sets"]["compliant"]["arms"]["RETIME_TRAINFIT"]["minus_ha0_ext_pose"]["delta"]
        out["controls"][f"s{s}"] = {"RETIME_reproduces": [cells["RETIME_TRAIN"]["delta"], b], "zero_corr_prior_delta": cells["ZERO_CORR_PRIOR"]["delta"],
                                    "zero_corr_prior_maxabs_m": float(np.abs(arms["ZERO_CORR_PRIOR"].astype(np.float64) - E["HP"].astype(np.float64)).max()),
                                    "PASS": bool(abs(cells["RETIME_TRAIN"]["delta"] - b) < 1e-9 and cells["ZERO_CORR_PRIOR"]["delta"] == 0.0)}
        out["seeds"][str(s)] = {"n_windows": int(len(eid)), "minus_ha0_ext_pose": cells, "minus_RETIME_TRAIN": vs_rt,
                                "eval_R2_per_instant": [float(1 - ((hh[:, k] - Heval[:, k]) ** 2).sum() / ((Heval[:, k] - Heval[:, k].mean()) ** 2).sum()) for k in range(4)]}
    out["controls"]["ALL_PASS"] = bool(all(v["PASS"] for k, v in out["controls"].items() if k.startswith("s")))
    (HERE / "PROGRESS_SCALAR.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"[{time.time() - t0:.1f}s] WROTE PROGRESS_SCALAR.json controls ALL_PASS={out['controls']['ALL_PASS']} lambda={lam}")
    print("ZZSCALARDONEZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
