#!/usr/bin/env python3
"""extract_znn.py -- ARM 2 (PREREG_FROZEN.md 5, 8): h_O of the DAG-OACC s0 checkpoint on the SAME sampled TRAIN windows with the nav input replaced by a DONOR nav.
The model has no null-nav path (e_nav is applied unconditionally); the donor is a random other TRAIN window of a DIFFERENT clip (np.random.default_rng(0), drawn in the sorted item order); the donor window's whole nav history [hmax, 5]
replaces batch["nav"].  Two tensors are written per population:   Znn  = nav swapped (the PRE-REGISTERED primary Z_nn);   Znl = nav AND limit/limit_ok swapped (POST-FREEZE, see raw/POSTFREEZE_LOG.md; never decisive).
    CUDA_VISIBLE_DEVICES=-1  python -I extract_znn.py --stage run [--budget-s 200]      resumable, time-boxed
    CUDA_VISIBLE_DEVICES=-1  python -I extract_znn.py --stage ctl                         controls N1 (identity) and N2 (reachability)
Imports (unmodified, from the previous package): extract_z.load_model (checkpoint sha256-checked)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np

import common2 as C  # noqa: E402

sys.path.insert(0, C.OLD)
import extract_z as EZ  # noqa: E402,F401  (previous package, imported unmodified: load_model + paths)
import torch  # noqa: E402

torch.set_num_threads(4)
SEED_TRAIN_DONOR = 0


draw_donors, nav_hist = C.draw_donors, C.nav_hist                              # pure-numpy copies live in common2 (unit-tested)


def build(budget_ctx=None):
    M, model, cfg = EZ.load_model()
    from tanitad.data import v8f_mini_data as D
    A = C.load_windows()
    clips = [str(c) for c in A["clips"]]
    J = D.load_join(f"{C.OLDSCR}/join_train_rebuilt.npz")
    jix = {c: i for i, c in enumerate(J["clip"].tolist())}
    tracks = D.load_native_tracks(C.TRACKS)
    _load = torch.load
    torch.load = lambda *aa, **kk: _load(*aa, **{**kk, "mmap": True, "weights_only": True})
    store = D.ClipStore(C.CACHE_ROOT, "train", lru=6, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    ds = D.V8FMiniWindows(store, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    items, n_pop = [], {}
    for pop in C.POPS:
        idx = np.load(f"{C.OLDRAW}/sample_{pop}.npz")["idx"]
        n_pop[pop] = len(idx)
        for row, w in enumerate(idx):
            items.append((int(A["clip_i"][w]), int(A["t"][w]), pop, row))
    items.sort(key=lambda x: (x[0], x[1], x[2]))                                   # the SAME order as extract_z.py
    return M, model, cfg, D, A, clips, J, jix, ds, items, n_pop


def swap(feed, J, jix, clips, A, cfg, donor_w):
    """feed (dict of tensors) -> (feed_nn, feed_nl, same_nav, same_lim).  donor_w: donor window indices (into the canonical list), one per row of the batch."""
    dn, dl, dok = [], [], []
    for w in donor_w:
        jc = jix[clips[int(A["clip_i"][w])]]
        dn.append(nav_hist(J, jc, int(A["t"][w]), cfg.hmax))
        dl.append(J["limit"][jc])
        dok.append(bool(J["limit_ok"][jc]))
    dn = torch.as_tensor(np.stack(dn), dtype=torch.float32)
    dl = torch.as_tensor(np.stack(dl), dtype=torch.float32)
    dok = torch.as_tensor(dok, dtype=torch.bool)
    f1 = dict(feed)
    f1["nav"] = dn
    f2 = dict(f1)
    f2["limit"], f2["limit_ok"] = dl, dok
    same_nav = (dn[:, -1] == feed["nav"][:, -1]).all(-1)
    same_lim = (dl == feed["limit"]).all(-1) & (dok == feed["limit_ok"])
    return f1, f2, same_nav, same_lim


def stage_run(budget_s):
    t0 = time.time()
    C.require_ram(3000, "extract_znn")
    M, model, cfg, D, A, clips, J, jix, ds, items, n_pop = build()
    clip_all = A["clip_i"]
    own = np.array([it[0] for it in items])
    don = draw_donors(own, clip_all, SEED_TRAIN_DONOR)
    assert (clip_all[don] != own).all()
    prog = f"{C.SCR}/znn_progress.json"
    n_done = json.load(open(prog))["n_done"] if os.path.exists(prog) else 0
    mode = "r+" if n_done > 0 else "w+"
    mm = {(k, p): np.lib.format.open_memmap(f"{C.SCR}/{k}_{p}.npy", mode=mode, dtype=np.float32, shape=(n_pop[p], cfg.n_q * cfg.d)) for k in ("Znn", "Znl") for p in n_pop}
    flp = f"{C.SCR}/znn_flags.npz"
    fl = dict(np.load(flp)) if (os.path.exists(flp) and n_done > 0) else {f"{k}__{p}": np.zeros(n_pop[p], bool) for k in ("same_nav", "same_lim") for p in n_pop}
    cap = {}
    model.read_o.register_forward_hook(lambda m, i, o: cap.__setitem__("hO", o.flatten(1).detach().float()))
    pos, bs = n_done, 32
    print(f"start pos {pos}/{len(items)} model+join ready {time.time() - t0:.0f}s", flush=True)
    while pos < len(items):
        if time.time() - t0 > budget_s:
            break
        chunk = items[pos:pos + bs]
        its = [ds.item(ds.clip_index[clips[ci]], t, 0, 0) for ci, t, _, _ in chunk]
        batch = D.collate(its)
        feed = {k: batch[k] for k in M.INFERENCE_INPUT_KEYS}
        f1, f2, sn, sl = swap(feed, J, jix, clips, A, cfg, don[pos:pos + len(chunk)])
        with torch.no_grad():
            model.forward_infer(f1)
            h1 = cap["hO"].clone()
            model.forward_infer(f2)
            h2 = cap["hO"].clone()
        for j, (ci, t, p, row) in enumerate(chunk):
            mm[("Znn", p)][row] = h1[j].numpy()
            mm[("Znl", p)][row] = h2[j].numpy()
            fl[f"same_nav__{p}"][row], fl[f"same_lim__{p}"][row] = bool(sn[j]), bool(sl[j])
        pos += len(chunk)
        if (pos // bs) % 20 == 0:
            print(f"{pos}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    for m_ in mm.values():
        m_.flush()
    np.savez(flp, **fl)
    json.dump({"n_done": pos}, open(prog, "w"))
    done = pos >= len(items)
    rep = {"n_done": pos, "n_total": len(items), "complete": done, "seconds_last_call": time.time() - t0, "ckpt_sha256": C.CKPT_SHA, "donor_seed": SEED_TRAIN_DONOR,
           "donor_clip_differs_from_own": bool((clip_all[don] != own).all())}
    if done:
        rep["share_donor_nav_equals_own"] = {p: float(fl[f"same_nav__{p}"].mean()) for p in n_pop}
        rep["share_donor_limit_equals_own"] = {p: float(fl[f"same_lim__{p}"].mean()) for p in n_pop}
    json.dump(rep, open(f"{C.RAW}/extract_znn_status.json", "w"), indent=1)
    print(json.dumps(rep), flush=True)


def stage_ctl(n=128):
    """N1: (a) a fresh forward with the TRUE nav reproduces the banked Z; (b) the swap harness with donor := self reproduces it too (the swap is a no-op when it should be).
    N2: the donor nav / donor nav+limit change h_O and the emitted plan (reachability)."""
    t0 = time.time()
    C.require_ram(3000, "extract_znn ctl")
    M, model, cfg, D, A, clips, J, jix, ds, items, n_pop = build()
    sub = [it for it in items if it[2] == "pstop"][::max(1, n_pop["pstop"] // n)][:n]
    Zb = np.load(f"{C.OLDSCR}/Z_pstop.npy", mmap_mode="r")
    clip_all = A["clip_i"]
    idx_p = np.load(f"{C.OLDRAW}/sample_pstop.npz")["idx"]
    own_w = np.array([idx_p[row] for _, _, _, row in sub])
    don = draw_donors(np.array([it[0] for it in sub]), clip_all, 123)
    cap = {}
    model.read_o.register_forward_hook(lambda m, i, o: cap.__setitem__("hO", o.flatten(1).detach().float()))
    res = {"n": len(sub), "a_true_vs_banked_max_abs": 0.0, "b_selfdonor_vs_banked_max_abs": 0.0, "nav_differs": [], "lim_differs": [], "rel_hO_nn": [], "rel_hO_nl": [], "dtraj_nn": [], "dtraj_nl": [], "dtraj_true_ref": []}
    for a in range(0, len(sub), 32):
        ch = sub[a:a + 32]
        its = [ds.item(ds.clip_index[clips[ci]], t, 0, 0) for ci, t, _, _ in ch]
        batch = D.collate(its)
        feed = {k: batch[k] for k in M.INFERENCE_INPUT_KEYS}
        zb = torch.as_tensor(np.stack([Zb[row] for _, _, _, row in ch]))
        with torch.no_grad():
            o0 = model.forward_infer(feed)
            h0 = cap["hO"].clone()
            fs, _, _, _ = swap(feed, J, jix, clips, A, cfg, own_w[a:a + len(ch)])          # donor := self
            o_s = model.forward_infer(fs)
            hs = cap["hO"].clone()
            f1, f2, sn, sl = swap(feed, J, jix, clips, A, cfg, don[a:a + len(ch)])
            o1 = model.forward_infer(f1)
            h1 = cap["hO"].clone()
            o2 = model.forward_infer(f2)
            h2 = cap["hO"].clone()
        res["a_true_vs_banked_max_abs"] = max(res["a_true_vs_banked_max_abs"], float((h0 - zb).abs().max()))
        res["b_selfdonor_vs_banked_max_abs"] = max(res["b_selfdonor_vs_banked_max_abs"], float((hs - zb).abs().max()))
        rel = lambda h: (torch.linalg.norm(h - h0, dim=1) / torch.linalg.norm(h0, dim=1).clamp_min(1e-9)).numpy()
        dt = lambda o: (o["traj60"] - o0["traj60"]).abs().flatten(1).max(1).values.numpy()
        res["nav_differs"] += (~sn).tolist()
        res["lim_differs"] += (~sl).tolist()
        res["rel_hO_nn"] += rel(h1).tolist()
        res["rel_hO_nl"] += rel(h2).tolist()
        res["dtraj_nn"] += dt(o1).tolist()
        res["dtraj_nl"] += dt(o2).tolist()
        res["dtraj_true_ref"] += (o_s["traj60"] - o0["traj60"]).abs().flatten(1).max(1).values.numpy().tolist()
    nd, ld = np.array(res["nav_differs"]), np.array(res["lim_differs"])
    out = {"n": res["n"], "N1a_true_forward_vs_banked_Z_max_abs": res["a_true_vs_banked_max_abs"], "N1b_selfdonor_swap_vs_banked_Z_max_abs": res["b_selfdonor_vs_banked_max_abs"],
           "N1b_selfdonor_traj60_max_abs_change": float(np.max(res["dtraj_true_ref"])),
           "N2_n_nav_differs": int(nd.sum()), "N2_n_limit_differs": int(ld.sum()),
           "N2_nn_windows_nav_differs": {"frac_relL2_gt_0": float((np.array(res["rel_hO_nn"])[nd] > 0).mean()) if nd.any() else None,
                                         "median_relL2": float(np.median(np.array(res["rel_hO_nn"])[nd])) if nd.any() else None,
                                         "median_max_abs_dtraj60_m": float(np.median(np.array(res["dtraj_nn"])[nd])) if nd.any() else None},
           "N2_nl_windows_nav_or_limit_differs": {"frac_relL2_gt_0": float((np.array(res["rel_hO_nl"])[nd | ld] > 0).mean()) if (nd | ld).any() else None,
                                                  "median_relL2": float(np.median(np.array(res["rel_hO_nl"])[nd | ld])) if (nd | ld).any() else None,
                                                  "median_max_abs_dtraj60_m": float(np.median(np.array(res["dtraj_nl"])[nd | ld])) if (nd | ld).any() else None},
           "N2_windows_nav_equal_unchanged_max_relL2": float(np.max(np.array(res["rel_hO_nn"])[~nd])) if (~nd).any() else None,
           "seconds": time.time() - t0, "ckpt_sha256": C.CKPT_SHA}
    json.dump(out, open(f"{C.RAW}/control_znn.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("run", "ctl"))
    ap.add_argument("--budget-s", type=float, default=200.0)
    a = ap.parse_args()
    {"run": lambda: stage_run(a.budget_s), "ctl": stage_ctl}[a.stage]()
