"""PREREG_V8F_PLANTEST: the real-data side. prepare() loads a model, the eval-141 deciding windows and the TRAIN windows (prior only); unit() makes one saved plan set.
No logged action reaches run_planner / plan_all: A_raw is used (a) on TRAIN windows to fit the prior, (b) on eval windows only inside window_metrics / hold_plan (the metric and the HOLD reference)."""
import json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from types import SimpleNamespace
import numpy as np
import torch
import plantest_lib as P
import wmact_lib as W

assert "plantest_tree" in os.path.abspath(W.tanitad.__file__).replace("\\", "/"), W.tanitad.__file__
ART = "E:/Projects/TanitAD-artifacts"
CK = {"M2b_s0": f"{ART}/sfresh_inputs/inputs/M2b_s0/ckpt_last.pt", "M2b_s1": f"{ART}/sfresh_inputs/inputs/M2b_s1/ckpt_last.pt"}
EVAL = f"{HERE}/../wmact/latents.pt"
TRAIN = f"{HERE}/../wmact/latents_train.pt"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
RUNS = f"{HERE}/runs"
ACC_SCALE = 2.0                                       # tanitad.models.v8f_mini.WM_ACC_SCALE (the loader: act[...,0] / WM_ACC_SCALE)


def fit_priors(st, wm, tr):
    """Level 1 from the scaled logged TRAIN actions (15 per window), levels 2 and 3 from the macro-action embeddings of the same TRAIN windows. Eval windows are not touched."""
    A_sc = wm.scale_actions(tr["A"])[:, :-1]
    with torch.no_grad():
        enc = st.encode(tr["Z1"], A_sc)
    P.apply_prior(st, 1, A_sc)
    P.apply_prior(st, 2, enc.act[1])
    P.apply_prior(st, 3, enc.act[2])


def prepare(tag, n_eval=None, random_r=None):
    """random_r: build the stack with untrained weights (torch.manual_seed(5000 + r)) on M2b_s0's level-1 latents (beside rows)."""
    m, cfg, ck = W.load_model(CK[tag])
    wm = m.wm
    if random_r is not None:
        import imag1_model as IM
        wm = IM.make_random_model(random_r, cfg).wm.eval()
    st = wm.stack.eval()
    ev, tr = P.load_latents(EVAL, tag), P.load_latents(TRAIN, tag)
    if n_eval:
        for k in ("Z1", "A", "eid", "t"):
            ev[k] = ev[k][:n_eval]
    fit_priors(st, wm, tr)
    TRK = torch.load(TRACKS, weights_only=False)["tracks"]
    v0 = np.full(len(ev["eid"]), np.nan)
    for i, (e, t) in enumerate(zip(ev["eid"], ev["t"])):
        g = int(t) - 15
        r = TRK.get(str(e))
        if r is not None and g >= 0 and 2 * g < len(r):
            v0[i] = float(np.asarray(r)[2 * g, 3])
    codes = {e: i for i, e in enumerate(sorted(set(ev["eid"])))}
    ep_int = torch.tensor([codes[e] for e in ev["eid"]])
    from tanitad.models.hjepa import donor_indices
    donor = donor_indices(len(ev["eid"]), episode_ids=ep_int, seed=0).numpy()
    return SimpleNamespace(tag=tag, wm=wm, st=st, ch1=float(wm.ch1_scale), z_now=ev["Z1"][:, 0], z_fut=ev["Z1"][:, 12], A=ev["A"].numpy().astype(np.float64), eid=ev["eid"],
                           v0=v0, donor=donor, n=len(ev["eid"]), n_total=ev["n_total"], v0_nan=int(np.isnan(v0).sum()), ep_int=ep_int)


def to_raw(ctx, a):
    out = np.array(a, np.float64)
    out[..., 0] *= ACC_SCALE
    out[..., 1] *= ctx.ch1
    return out


def unit(ctx, kind, gname, idx=None, seed=P.PLAN_SEED):
    """One plan set. kind: td3 | flat | flat1. Returns raw plans [n,12,2] and the per-level cost diagnostics."""
    t = time.time()
    k = "flat" if kind == "flat1" else kind
    r = P.plan_all(ctx.st, k, gname, ctx.z_now, ctx.z_fut, ctx.donor, idx=idx, seed=seed, last_k=1 if kind == "flat1" else None)
    r["plan_raw"] = to_raw(ctx, r.pop("actions"))
    r["wall_s"] = np.array(time.time() - t)
    return r


def save_unit(ctx, kind, gname, r, seed):
    os.makedirs(RUNS, exist_ok=True)
    np.savez(f"{RUNS}/{ctx.tag}_{kind}_{gname}_s{seed}.npz", **r)
