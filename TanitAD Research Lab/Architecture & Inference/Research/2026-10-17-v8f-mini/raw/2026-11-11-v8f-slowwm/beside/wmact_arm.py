"""SLOWWM beside B2 (WMACT action-sensitivity audit), protocol fixed in ``../BESIDE_RULES_SLOWWM.md``. Tier T0, CPU. Never deciding.

B2 as written: the ``PREREG_V8F_WMACT`` statistics and controls (``wmact/wmact_lib.py``, ``wmact/run_wmact.py``) per arm, unchanged; only the latent collection is routed through the basis path for a basis arm.
  * ``run_wmact.py`` is a script (it reads latents.pt and writes RESULT_WMACT.json at import), so its per-model body is carried here as ``stats_for``, expression for expression, calling ``wmact_lib`` for every
    statistic. A control reproduces the banked ``RESULT_WMACT.json`` rows of M2b_s0 from the banked latents with it (``wmact_controls.py``).
  * Collection. An ADAPTER arm: ``wmact_lib.collect_latents`` itself (the window route: ``wm.z1(adapter(feats) + pos)`` on the 16 frame window). A BASIS arm: the same windows, the same batches, ``wm.z1_from_basis`` of
    ``slowwm_hook.ubasis_of_feats`` of the window's live features (the window route of the hook's own u: the file route of ``prepare_arm_read`` reads whole clips, a different history for K = 3). A basis arm's
    ``selfcheck.json`` readback, kind and sha, and the argv / BUILT strides are checked by ``lp_slowwm.load_checked`` before any window is read; the live u is cross-checked against the arm's ``ubasis`` file.
  * Window set = ``wmact/latents.pt`` (eval-141, N = 2,399 windows, 1,976 with all 16 frames real, the set ``run_wmact.py`` uses); the mean action constant is the eval-141 one, as the read fixes it."""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decod_lib as DL  # noqa: E402  (code root, paths, asserts)
import numpy as np  # noqa: E402
import torch  # noqa: E402

W, S, LR, LP, SH = DL.W, DL.S, DL.LR, DL.LP, DL.SH
BANKED_EVAL = f"{DL.THEORY}/wmact/latents.pt"
BANKED_RESULT = f"{DL.THEORY}/wmact/RESULT_WMACT.json"


@torch.no_grad()
def collect_basis_z1(model, B, ds, n, batch=64):
    """``wmact_lib.collect_latents`` for a basis model: the same windows, order and batches; z1 = ``wm.z1_from_basis`` of the live u of each window. Returns Z [n, 16, D], A_raw, ok, t (as collect_latents)."""
    Z, A, OK, T = [], [], [], []
    for b0 in range(0, n, batch):
        items = [ds.item(ds.windows[i][0], ds.windows[i][1]) for i in range(b0, min(b0 + batch, n))]
        u = torch.from_numpy(np.stack([SH.ubasis_of_feats(B, it["feats"]) for it in items]))
        A.append(torch.from_numpy(np.stack([it["act"] for it in items])))
        OK.append(torch.from_numpy(np.stack([it["frame_ok"] for it in items])))
        T.append(np.array([it["t"] for it in items]))
        Z.append(model.wm.z1_from_basis(u))
    return torch.cat(Z), torch.cat(A), torch.cat(OK), np.concatenate(T)


def collect_arm_z1(model, kind, B, ds, n, batch=64):
    """The new collection: adapter arms through ``wmact_lib.collect_latents`` (unchanged), basis arms through ``collect_basis_z1``."""
    if kind == "adapter":
        Z, A, OK, T = W.collect_latents({"arm": model}, ds, n, batch)
        return Z["arm"], A, OK, T
    return collect_basis_z1(model, B, ds, n, batch)


def stats_for(model, Z1, A_raw, eid, mean_sc):
    """The per-model body of ``run_wmact.py`` (rows, sens, controls 1 and 2), every statistic from ``wmact_lib``. Z1 / A_raw / eid are the windows with all 16 frames real."""
    ks_full = dict(W.KS)
    ks = feasible_ks(model.wm, Z1[:2], A_raw[:2])
    W.KS = ks                                                    # wmact_lib reads its KS global: only the cells the 16-frame window can hold are computed (restored below)
    try:
        res = _stats_for(model, Z1, A_raw, eid, mean_sc)
    finally:
        W.KS = ks_full
    res["ks_used"] = {str(l): list(v) for l, v in ks.items()}
    res["ks_not_computable"] = {f"L{l}/k{k}": f"the stack's level-{l} has {n_states[l]} states on a 16-frame window" for l, v in ks_full.items() for k in v if k not in ks.get(l, ()) for n_states in [window_states(model.wm, Z1[:2], A_raw[:2])]}
    return res


@torch.no_grad()
def window_states(wm, Z1, A_raw):
    """{level: number of states} of ``stack.encode`` on a 16-frame window of this arm (R35H: strides 3, 5, so level 2 has few states)."""
    enc = wm.stack.encode(Z1, wm.scale_actions(A_raw)[:, :-1])
    return {l + 1: int(t.shape[1]) for l, t in enumerate(enc.z)}


def feasible_ks(wm, Z1, A_raw):
    """The WMACT cells (level, k) of ``wmact_lib.KS`` whose k-th state exists on a 16-frame window of this arm: k <= states - 1 (the oracle state is the encoded state k steps on). M2b (strides 2, 2) holds all of
    them; a ladder arm with strides (3, 5) does not hold level 2 / 3 at k = 3 (MEASURED in the smoke run). A level with no feasible k is dropped."""
    ns = window_states(wm, Z1, A_raw)
    out = {l: tuple(k for k in ks if k <= ns[l] - 1) for l, ks in W.KS.items()}
    return {l: v for l, v in out.items() if v}


def _stats_for(model, Z1, A_raw, eid, mean_sc):
    wm = model.wm
    uniq, Wt = W.boot_weights(eid)
    W.check_draws_equal_ci(eid, uniq, Wt)
    n, n_ep = len(eid), len(uniq)
    R = lambda es, C, sign: W.ratio_row(es, C, sign, Wt, n, n_ep)
    A_sc = wm.scale_actions(A_raw)[:, :-1]
    cols = W.model_columns(wm, Z1, A_sc, eid, mean_sc)
    perm = cols.pop("perm")
    es = W.ep_sums(cols, eid, uniq)
    rows = {}
    for l, ks in W.KS.items():
        for k in ks:
            C, S_ = es[f"C/{l}/{k}"], f"/{l}/{k}"
            row = {"SKILL": R(es, C, {"C" + S_: 1.0, "err_true" + S_: -1.0}),
                   "ACTION-USE": R(es, C, {"err_donor" + S_: 1.0, "err_true" + S_: -1.0}),
                   "ACTION-USE-MEAN": R(es, C, {"err_mean" + S_: 1.0, "err_true" + S_: -1.0}),
                   "CTRL3_SKILL_startdonor": R(es, C, {"C" + S_: 1.0, "err_startdonor" + S_: -1.0}),
                   "CTRL1_identity_ACTION-USE": R(es, C, {"err_self" + S_: 1.0, "err_true" + S_: -1.0})}
            if l == 1:
                for nm in ("acc", "yaw"):
                    row[f"ACTION-USE[{nm} swapped to donor]"] = R(es, C, {f"err_donor_{nm}" + S_: 1.0, "err_true" + S_: -1.0})
                    row[f"ACTION-USE-MEAN[{nm} set to mean]"] = R(es, C, {f"err_mean_{nm}" + S_: 1.0, "err_true" + S_: -1.0})
            rows[f"L{l}/k{k}"] = row
    sens = W.sens_block(wm, Z1, A_sc, eid, perm)
    sid = W.sens_block(wm, Z1, A_sc, eid, torch.arange(n))                                  # control 1: donor = the window itself
    ctl = {"1_identity_sens_max_abs": max(abs(v) for v in sid.values()), "1_identity_actionuse_max_abs": max(abs(r["CTRL1_identity_ACTION-USE"]["point"]) for r in rows.values())}
    m2 = W.zero_act_embed(model)                                                            # control 2: level-1 action embedding zeroed
    c2 = W.model_columns(m2.wm, Z1, A_sc, eid, mean_sc)
    c2.pop("perm")
    e2 = W.ep_sums(c2, eid, uniq)
    ctl["2_zeroed_act_embed"] = {f"k{k}": {"SKILL": R(e2, e2[f"C/1/{k}"], {f"C/1/{k}": 1.0, f"err_true/1/{k}": -1.0}),
                                          "ACTION-USE_point": float(R(e2, e2[f"C/1/{k}"], {f"err_donor/1/{k}": 1.0, f"err_true/1/{k}": -1.0})["point"])} for k in W.KS[1]}
    ctl["2_zeroed_act_embed"]["sens_L1_max_abs"] = max(abs(v) for kk, v in W.sens_block(m2.wm, Z1, A_sc, eid, perm).items() if kk.startswith("L1/"))
    return {"rows": rows, "sens": sens, "controls": ctl, "windows_used": int(n), "episodes_used": int(n_ep)}


@torch.no_grad()
def control4(model, kind, B, ds, n=64):
    """Control 4 as run_wmact.py runs it: the stack's loss on z1 / scaled actions equals the WM's own forward on the first ``n`` windows, same RNG state. A basis arm's forward is ``forward_basis`` of its u."""
    items = [ds.item(ds.windows[i][0], ds.windows[i][1]) for i in range(n)]
    act = torch.from_numpy(np.stack([it["act"] for it in items]))
    if kind == "adapter":
        feats = torch.from_numpy(np.stack([it["feats"] for it in items])).float()
        adapt = model.adapter(feats) + model.pos[None, None]
        torch.manual_seed(0)
        direct = model.wm(adapt, act)
        torch.manual_seed(0)
        via = model.wm.stack.loss(model.wm.z1(adapt), model.wm.scale_actions(act)[:, :-1])
    else:
        u = torch.from_numpy(np.stack([SH.ubasis_of_feats(B, it["feats"]) for it in items]))
        torch.manual_seed(0)
        direct = model.wm.forward_basis(u, act)
        torch.manual_seed(0)
        via = model.wm.stack.loss(model.wm.z1_from_basis(u), model.wm.scale_actions(act)[:, :-1])
    return {"max_abs_diff": max(abs(float(direct[k]) - float(via[k])) for k in direct), "terms": {k: float(v) for k, v in direct.items()}, "n_windows": int(n)}


def ubasis_file_vs_live(L, ds, eid_ds, n=64):
    """A basis arm: the live u of the first ``n`` windows against the arm's ``ubasis`` file (whole-clip rows g .. g+15). Max abs difference."""
    fr, B = L["fr"], L["B"]
    d = 0.0
    for i in range(n):
        ci, t = ds.windows[i][0], ds.windows[i][1]
        it = ds.item(ci, t)
        live = SH.ubasis_of_feats(B, it["feats"])
        j = fr["clips"].index(str(eid_ds[i]))
        g = int(t) - 15
        d = max(d, float(np.abs(live - fr["ubasis"][j][g:g + 16]).max()))
    return d


def read_arm_wmact(name, run_dir, basis=None, ubasis=None, limit=None, crosscheck=True):
    t0 = time.time()
    DL.wait_ram()
    L = S.load_checked(name, run_dir, basis, ubasis)
    model, kind, B = L["model"], L["kind"], L["B"]
    banked = torch.load(BANKED_EVAL, weights_only=False)
    ds, wins, eid_ds = W.open_windows(16)
    n = int(limit or banked["N"])
    assert [str(e) for e in eid_ds[:n]] == [str(e) for e in banked["eid"][:n]], "the eval windows are not the banked ones"
    route = {"kind": kind, "collection": "wmact_lib.collect_latents (adapter window route)" if kind == "adapter" else "wm.z1_from_basis of the live u of each window (slowwm_hook.ubasis_of_feats)",
             "basis_sha256": B["sha"] if B else None}
    if kind != "adapter" and crosscheck:
        route["ubasis_file_vs_live_max_abs"] = d = ubasis_file_vs_live(L, ds, eid_ds, min(64, n))
        if d > 1e-4:
            raise S.SH.BasisMismatch(f"{name}: the live u differs from the ubasis file by {d:.3e} (> 1e-4)")
    Z, A, OK, T = collect_arm_z1(model, kind, B, ds, n)
    ok_all = OK.all(1).numpy()
    sel = np.flatnonzero(ok_all)
    eid = np.array([str(e) for e in banked["eid"][:n]])[sel]
    mean_sc, mean_raw = W.mean_action_scaled(model.wm, banked["A_raw"], banked["ok"])
    res = stats_for(model, Z[sel], A[sel], eid, mean_sc)
    res["controls"]["4_reproduction"] = control4(model, kind, B, ds, 64 if n >= 64 else n)
    res.update({"protocol": "B2 WMACT audit (BESIDE_RULES_SLOWWM.md)", "tier": "T0", "role": "BESIDE, no bar", "name": name, "route": route, "windows_total": int(n), "limit": limit,
                "action_mean_raw_eval141_a_yaw": [float(x) for x in mean_raw], "argv_checked": L["checked"], "free_gb_at_start": round(LR.free_gb(), 2), "wall_s": DL.stamp(t0)})
    return res
