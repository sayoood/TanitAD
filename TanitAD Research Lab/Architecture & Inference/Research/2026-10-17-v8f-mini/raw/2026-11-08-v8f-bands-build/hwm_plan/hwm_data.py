"""HWM-PLAN data side. A WindowSet is the planner's whole input surface: level-1 latents z (N, F, D) with NOW = frame 0, the logged raw actions (N, F, 2) (read only as a0 by the planner and in full by the metric),
the frame-ok mask, the episode id, the NOW cache step, the speed at NOW and the donor map. Providers:
  eval_windows_from_cache(model, pairs, F)  live, from the local feature cache of the 141 eval clips (the adapter + proj of the GIVEN model: a ladder checkpoint re-projects its own latents);
  trainset_from_latents_pt(path, tag)       the banked 16-frame TRAIN latents of M2b (prior fit only; a ladder checkpoint needs its own TRAIN latents, see IMPL_CHOICES_HWMPLAN.md section 7).
Code root: WMACT_CODE_ROOT (the read-only ladder tree). Nothing here writes into the tree."""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from types import SimpleNamespace
import numpy as np
import torch
import hwm_plan as H
import plantest_lib as P
import wmact_lib as W

ART = "E:/Projects/TanitAD-artifacts"
CK = {"M2b_s0": f"{ART}/sfresh_inputs/inputs/M2b_s0/ckpt_last.pt", "M2b_s1": f"{ART}/sfresh_inputs/inputs/M2b_s1/ckpt_last.pt"}
EVAL16 = f"{HERE}/../wmact/latents.pt"          # PLANTEST's window list: (eid, t) of the 1,976 all-real 16-frame windows; NOW = t - 15
TRAIN16 = f"{HERE}/../wmact/latents_train.pt"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
LEAD_BLOCK = f"{ART}/refcv5cmp/data/b1_eval_lead_block.npz"      # taniteval/tools/build_lead_block_b1.py: one row per (clip, RAW 10 Hz frame), K = 10 knots at 0.2 s (a 2 s horizon), window-origin ego frame


def load_model(path):
    m, cfg, ck = W.load_model(path)
    return m, cfg, ck


def fit_priors(st, wm, Z1, A_raw):
    """As plantest_real.fit_priors: level 1 from the scaled logged TRAIN actions, levels 2 and 3 from the macro embeddings of the same TRAIN windows. Eval windows are not touched."""
    A_sc = wm.scale_actions(A_raw)[:, :-1]
    with torch.no_grad():
        enc = st.encode(Z1, A_sc)
    P.apply_prior(st, 1, A_sc)
    P.apply_prior(st, 2, enc.act[1])
    P.apply_prior(st, 3, enc.act[2])


def trainset_from_latents_pt(path, tag):
    L = P.load_latents(path, tag)
    return L["Z1"], L["A"]


def plantest_pairs():
    """(eid, NOW cache step) of PLANTEST's 1,976 windows (NOW = t - 15 of the 16-frame window ending at t)."""
    L = P.load_latents(EVAL16, "M2b_s0")
    return L["eid"], np.asarray(L["t"]).astype(int) - 15


def v0_of(eids, nows, tracks_path=TRACKS):
    TRK = torch.load(tracks_path, weights_only=False)["tracks"]
    v0 = np.full(len(eids), np.nan)
    for i, (e, g) in enumerate(zip(eids, nows)):
        r = TRK.get(str(e))
        if r is not None and g >= 0 and 2 * g < len(r):
            v0[i] = float(np.asarray(r)[2 * g, 3])
    return v0


def eval_windows_from_cache(model, eids, nows, F, batch=16, store=None):
    """Live windows: frames NOW .. NOW+F-1 of each clip from the local eval cache, through model.adapter + model.pos + model.wm.z1. A window whose F frames do not all lie inside the clip is DROPPED (no clamping:
    D.V8FMiniWindows.item would repeat the last frame silently). Returns (WindowSet fields dict, kept mask over the input pairs)."""
    if store is None:
        ds, _, _ = W.open_windows(16)
        store = ds.store
    keep, acts, oks, Z = [], [], [], []
    buf = []

    def flush():
        if buf:
            f = torch.from_numpy(np.stack(buf)).float()
            with torch.no_grad():
                Z.append(model.wm.z1(model.adapter(f) + model.pos[None, None]))
            buf.clear()

    for i, (e, g) in enumerate(zip(eids, nows)):
        d = store.get(str(e))
        T = len(d["pose"])
        if g < 0 or g + F > T:
            keep.append(False); continue
        idx = np.arange(g, g + F)
        keep.append(True)
        oks.append(np.isfinite(d["p10f"][idx]).all(1) | (idx < 1))
        acts.append(np.stack([np.nan_to_num(d["a"][idx]), d["ch1"][idx]], -1).astype(np.float32))
        buf.append(np.asarray(d["feats"][idx]))
        if len(buf) >= batch:
            flush()
    flush()
    return dict(z=torch.cat(Z), A=torch.from_numpy(np.stack(acts)), ok=np.stack(oks)), np.asarray(keep)


def make_windowset(model, eids, nows, F, v0_all=None, n=None):
    """Build the WindowSet for the pairs, keep windows with all F frames real (ok) and a finite v0, add donors (a different episode)."""
    if n:
        eids, nows = eids[:n], nows[:n]
    d, keep = eval_windows_from_cache(model, eids, nows, F)
    e = np.asarray(eids)[keep]; g = np.asarray(nows)[keep]
    v0 = v0_of(e, g) if v0_all is None else v0_all[keep]
    good = d["ok"].all(1) & np.isfinite(v0)
    d["z"], d["A"] = d["z"][torch.from_numpy(good)], d["A"][torch.from_numpy(good)]
    e, g, v0 = e[good], g[good], v0[good]
    codes = {x: i for i, x in enumerate(sorted(set(e)))}
    from tanitad.models.hjepa import donor_indices
    donor = donor_indices(len(e), episode_ids=torch.tensor([codes[x] for x in e]), seed=0).numpy()
    return SimpleNamespace(z=d["z"], A=d["A"].numpy().astype(np.float64), eid=e.astype("U36"), now=g, v0=v0, donor=donor, n=len(e), n_input=len(eids), n_dropped=int(len(eids) - len(e)),
                           n_dropped_length=int((~keep).sum()), n_dropped_notok=int((~good).sum()))


LEAD_BLOCK6 = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan/data/b1_eval_lead_block_k30.npz"      # built 2026-11-08 with --k 30 from the same ego tar, timestamps tar and obstacle parquets as LEAD_BLOCK


def lead_for(eids, nows, path=LEAD_BLOCK):
    """Lead-agent block for windows (eid, NOW cache step): the b1 eval lead block row (clip, RAW frame 2 * NOW) (the builder's rule, cache step t <-> frame 2t). Returns a dict of leads (n, K, 2), lens (n,), state (n,) in
    {LEAD, NO_LEAD, NOT_STRAIGHT, NO_LABEL, NO_ROW}, speeds (n,), ts_rel (K,), dt, row (n,). A window with no row gets NO_ROW (counted, never read as free flow). The lead's future positions are ground truth, a scoring input, never a planner input."""
    z = np.load(path, allow_pickle=False)
    key = {(str(c), int(f)): i for i, (c, f) in enumerate(zip(z["clip_id"], z["frame"]))}
    row = np.array([key.get((str(e), 2 * int(g)), -1) for e, g in zip(eids, nows)], int)
    has = row >= 0
    r = np.where(has, row, 0)
    leads = np.where(has[:, None, None], z["leads"][r], np.nan)
    lens = np.where(has, z["lead_lens"][r], np.nan)
    state = np.where(has, z["state"][r], "NO_ROW").astype("U12")
    speeds = np.where(has, z["speeds"][r], np.nan)
    return dict(leads=leads, lens=lens, state=state, speeds=speeds, ts_rel=np.asarray(z["ts_rel_s"], float), dt=float(z["dt_s"][0]), row=row)


def lead_join_check(lead, v0):
    """Plumbing guard, not a result: the lead block's own ego speed at the window origin against the harness's v0 (native_tracks) on the windows that joined. Returns (median |dv|, max |dv|, n joined)."""
    ok = np.isfinite(lead["speeds"]) & np.isfinite(v0)
    d = np.abs(lead["speeds"][ok] - np.asarray(v0)[ok])
    return (float(np.median(d)) if d.size else float("nan"), float(d.max()) if d.size else float("nan"), int(ok.sum()))


def goal_frame_counts(eids, nows, frames, store=None):
    """G2: per goal frame (level-1 steps ahead of NOW), the number of windows whose true-future frame exists: NOW + frame lies inside the clip and the frame is real (finite p10f) and every earlier frame of the window is too,
    the harness's own keep rule at F = frame + 1 (the oracle arms need it; the predicted-goal arms need only n_traj + 1 frames). Returns {frame: n}, plus n_pairs."""
    if store is None:
        ds, _, _ = W.open_windows(16)
        store = ds.store
    out = {int(f): 0 for f in frames}
    for e, g in zip(eids, nows):
        d = store.get(str(e))
        T = len(d["pose"])
        for f in out:
            if g < 0 or g + f + 1 > T:
                continue
            idx = np.arange(g, g + f + 1)
            if (np.isfinite(d["p10f"][idx]).all(1) | (idx < 1)).all():
                out[f] += 1
    return dict(counts=out, n_pairs=int(len(eids)))
