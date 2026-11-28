"""PREREG_V8F_WMACT: per-level prediction skill and action use of v8F-mini's world model (T0, CPU only).

Library: model / window loading, WM latents, the statistics, the episode-cluster bootstrap and the four controls.
Run: run_wmact.py.  Controls as tests: test_wmact_controls.py.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys

import numpy as np
import torch

CODE_ROOT = os.environ.get("WMACT_CODE_ROOT", "")          # a pinned export of the tip: <root>/stack, <root>/taniteval
for _p in (os.path.join(CODE_ROOT, "stack"), os.path.join(CODE_ROOT, "taniteval")):
    if CODE_ROOT and _p not in sys.path:
        sys.path.insert(0, _p)

import tanitad  # noqa: E402
if CODE_ROOT:
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith(os.path.abspath(CODE_ROOT).replace("\\", "/").lower()), tanitad.__file__

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from tanitad.models.hjepa import donor_indices  # noqa: E402
from taniteval import ci as CI  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
SF = f"{ART}/sfresh_inputs"
RAW = f"{SF}/code/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CKPT = {"M2b_s0": f"{ART}/v8f-mini-hacc/M2b_s0_hacc.pt", "M2b_s1": f"{ART}/v8f-mini-hacc/M2b_s1_hacc.pt"}
CKPT_C0 = {"C0_s0": f"{ART}/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt", "C0_s1": f"{ART}/v8f-recover/M2b_s1_recft_C0/ckpt_last.pt"}
KS = {1: (1, 5, 10), 2: (1, 3), 3: (1, 3)}
B_BOOT, SEED = 2000, 0


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_model(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    r = model.load_state_dict(ck["model"], strict=False)
    assert not r.missing_keys and set(r.unexpected_keys) <= {"hacc_w"}, (r.missing_keys, r.unexpected_keys)
    return model.eval(), cfg, ck


def wm_state(ck) -> dict:
    return {k: v for k, v in ck["model"].items() if k.startswith("wm.")}


def adapter_state(ck) -> dict:
    return {k: v for k, v in ck["model"].items() if k.startswith("adapter.") or k == "pos"}


# --------------------------------------------------------------------------------------------- windows
def open_windows(hmax: int):
    ej = D.load_join(f"{RAW}/join_eval141.npz")
    tracks = D.load_native_tracks(f"{SF}/inputs/native_tracks.pt")
    store = D.ClipStore(f"{SF}/cache", "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(f"{RAW}/eval_grid_windows.json", ej)
    ds = D.V8FMiniWindows(store, ej, wins, hist=4, hmax=hmax, max_age_t=2, max_age_s=6)
    eid = np.array([str(ej["clip"][ci]) for ci, _, _, _ in wins])
    return ds, wins, eid


@torch.no_grad()
def collect_latents(models: dict, ds, n: int, batch: int = 64):
    """One pass over the windows: for every model its level-1 latents ``z1`` [N, F, D] (``WMAux.z1`` of the adapter tokens) and the
    RAW cached action channels [N, F, 2] and frame_ok [N, F]."""
    Z = {k: [] for k in models}
    A, OK, T = [], [], []
    for b0 in range(0, n, batch):
        items = [ds.item(ds.windows[i][0], ds.windows[i][1]) for i in range(b0, min(b0 + batch, n))]
        feats = torch.from_numpy(np.stack([it["feats"] for it in items]))
        A.append(torch.from_numpy(np.stack([it["act"] for it in items])))
        OK.append(torch.from_numpy(np.stack([it["frame_ok"] for it in items])))
        T.append(np.array([it["t"] for it in items]))
        for k, m in models.items():
            adapt = m.adapter(feats.float()) + m.pos[None, None]
            Z[k].append(m.wm.z1(adapt))
    return {k: torch.cat(v) for k, v in Z.items()}, torch.cat(A), torch.cat(OK), np.concatenate(T)


# --------------------------------------------------------------------------------------------- the WM, level by level
@torch.no_grad()
def level_pack(wm, Z1: torch.Tensor, A: torch.Tensor, chunk: int = 512) -> dict:
    """For ``Z1`` [N, F, D1] and the WM-scaled actions ``A`` [N, F-1, 2]: per level l, the TRUE states ``z[l]`` [N, T_l, D_l]
    (``HJEPAStack.encode`` = the lifted real frames) and the rollout ``pred[(l, k)]`` [N, D_l] of ``HJEPAStack.rollout`` (context 1)
    from the start state under the level's actions (raw at level 1, macro embeddings of the SAME actions above)."""
    st = wm.stack
    true, pred = {}, {}
    for s0 in range(0, Z1.shape[0], chunk):
        sl = slice(s0, s0 + chunk)
        enc = st.encode(Z1[sl], A[sl])
        for l, ks in KS.items():
            H = max(ks)
            acts = A[sl][:, :H] if l == 1 else enc.act[l - 1][:, :H]
            roll = st.rollout(l, enc.z[l - 1][:, 0], acts)
            true.setdefault(l, []).append(enc.z[l - 1])
            for k in ks:
                pred.setdefault((l, k), []).append(roll[:, k - 1])
    return {"true": {l: torch.cat(v) for l, v in true.items()}, "pred": {k: torch.cat(v) for k, v in pred.items()}}


def sq(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return ((a.double() - b.double()) ** 2).sum(-1).numpy()


def mean_action_scaled(wm, A_raw: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """The mean RAW (a, yaw rate) over real cached frames, through ``WMAux.scale_actions`` -> [2]."""
    raw = A_raw[ok].double().mean(0).float()
    return wm.scale_actions(raw), raw


# --------------------------------------------------------------------------------------------- bootstrap
def boot_weights(eid, n_boot: int = B_BOOT, seed: int = SEED) -> tuple:
    """Episode-resampling counts [n_boot, n_ep]: the SAME rng calls as ``taniteval.ci._draws`` (seed 0), so every row shares its draws."""
    uniq, _ = CI.episode_index(eid)
    rng = np.random.default_rng(seed)
    n_ep = len(uniq)
    W = np.zeros((n_boot, n_ep))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=n_ep, replace=True)
        W[b] = np.bincount(np.searchsorted(uniq, pick), minlength=n_ep)
    return uniq, W


def ep_sums(cols: dict, eid, uniq) -> dict:
    pos = np.searchsorted(uniq, np.asarray([str(e) for e in eid]))
    out = {}
    for k, v in cols.items():
        s = np.zeros(len(uniq))
        np.add.at(s, pos, np.asarray(v, np.float64))
        out[k] = s
    return out


def ratio_row(num: dict, den: np.ndarray, sign: dict, W: np.ndarray, n: int, n_ep: int) -> dict:
    """stat = sum_i sign_i * sum num_i / sum den, over per-episode sums; point = full set, interval = percentile of the W-weighted draws."""
    p = float(sum(s * num[k].sum() for k, s in sign.items()) / den.sum())
    b = sum(s * (W @ num[k]) for k, s in sign.items()) / (W @ den)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"point": p, "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0), "n_windows": int(n), "n_episodes": int(n_ep)}


def check_draws_equal_ci(eid, uniq, W, n_check: int = 5):
    """The weights reproduce ``taniteval.ci._draws``: the first draws select the same windows."""
    u, idx = CI.episode_index(eid)
    pos = np.searchsorted(u, np.asarray([str(e) for e in eid]))
    n_win = np.bincount(pos, minlength=len(u))                     # windows per episode: a draw of an episode selects all of them
    for b, sel in zip(range(n_check), CI._draws(u, idx, n_check, SEED)):
        cnt = np.bincount(pos[sel], minlength=len(u))              # windows selected per episode
        assert (cnt == W[b] * n_win).all()
    return True


# --------------------------------------------------------------------------------------------- the statistics of one model
@torch.no_grad()
def model_columns(wm, Z1, A_sc, eid, A_mean_sc) -> dict:
    """All per-window squared errors / copy baselines for one model (arrays [N] keyed by name), plus sens of ``action_sensitivity``."""
    N = Z1.shape[0]
    perm = donor_indices(N, episode_ids=torch.as_tensor(np.unique(eid, return_inverse=True)[1]), seed=0)
    base = level_pack(wm, Z1, A_sc)
    cols = {"perm": perm}
    variants = {"true": A_sc, "donor": A_sc[perm], "mean": A_mean_sc.expand_as(A_sc).contiguous(), "self": A_sc[torch.arange(N)]}
    for name, ch in (("acc", [0]), ("yaw", [1])):                     # one channel from the donor / from the mean, the other true
        d = A_sc.clone(); d[..., ch] = A_sc[perm][..., ch]; variants[f"donor_{name}"] = d
        m = A_sc.clone(); m[..., ch] = A_mean_sc.expand_as(A_sc)[..., ch]; variants[f"mean_{name}"] = m
    packs = {"true": base}
    for v, Av in variants.items():
        if v != "true":
            packs[v] = level_pack(wm, Z1, Av)
    for l, ks in KS.items():
        zt = base["true"][l]
        for k in ks:
            tgt = zt[:, k]
            cols[f"C/{l}/{k}"] = sq(zt[:, 0], tgt)
            for v, p in packs.items():
                cols[f"err_{v}/{l}/{k}"] = sq(p["pred"][(l, k)], tgt)
            cols[f"err_startdonor/{l}/{k}"] = None            # filled below (scorer control)
    # scorer control: rollout from another window's z_start under THIS window's actions
    Zp = Z1.clone()
    st = wm.stack
    enc_all = [st.encode(Z1[s:s + 512], A_sc[s:s + 512]) for s in range(0, N, 512)]
    for l, ks in KS.items():
        z0 = torch.cat([e.z[l - 1][:, 0] for e in enc_all])
        acts = A_sc[:, :max(ks)] if l == 1 else torch.cat([e.act[l - 1][:, :max(ks)] for e in enc_all])
        roll = torch.cat([st.rollout(l, z0[perm][s:s + 512], acts[s:s + 512]) for s in range(0, N, 512)])
        for k in ks:
            cols[f"err_startdonor/{l}/{k}"] = sq(roll[:, k - 1], base["true"][l][:, k])
    return cols


def sens_block(wm, Z1, A_sc, eid, perm) -> dict:
    """``HJEPAStack.action_sensitivity`` per level (beside, deciding nothing)."""
    st = wm.stack
    enc = [st.encode(Z1[s:s + 512], A_sc[s:s + 512]) for s in range(0, Z1.shape[0], 512)]
    out = {}
    for l, ks in KS.items():
        z0 = torch.cat([e.z[l - 1][:, 0] for e in enc])
        acts = A_sc[:, :max(ks)] if l == 1 else torch.cat([e.act[l - 1][:, :max(ks)] for e in enc])
        r = st.action_sensitivity(z0, acts, level=l, ks=ks, perm=perm, channels={"all": list(range(acts.shape[-1]))} if l > 1 else
                                  {"all": [0, 1], "acc": [0], "yaw": [1]})
        out.update({f"L{l}/{k}": v for k, v in r.items()})
    return out


def zero_act_embed(model):
    """The deliberate-regression copy: level 1's action embedding zeroed (weights AND bias)."""
    m2 = copy.deepcopy(model)
    for p in m2.wm.stack.act_embed.parameters():
        p.data.zero_()
    return m2
