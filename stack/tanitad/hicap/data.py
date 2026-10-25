"""Cached-feature dataset for HiCAP over the v9 label release (paper sec:training-data).

Per-clip CACHE FILE (`<cache_dir>/<sha12>.npz`, written by `scripts/build_hicap_cache.py` on a pod, or by `make_synth_clip` in tests),
all arrays indexed by EPISODE ROW j (= v9 row j = the trainer's `ep.poses[j]`; raw egomotion row k = k0 + j, k0 = 2):
  sid int64 | poses [T,4] (x, y, yaw, v) | frames [T,3,64,160] uint8 (the LATEST RGB frame of row j's stack, downsized; motion input
  stacks rows j-2, j-1, j) | tick_k [n] (episode rows of the 2 Hz backbone refreshes) |
  vlm [n,3,N,d_v] float16 (taps pooled to N tokens) | last [n,d_v] float16.
v9 label rows come from `v9_labels_<split>.npz` (V9_SCHEMA.md): clip__* tables and row__* arrays, row j of a clip = raw row k0+j.
Window at raw row k: NOW=k; the past stack is rows k-2,k-1,k (100 ms apart); the future is poses[k+1..k+60].
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from . import vocab as V

FUT = V.T_STEPS
G = len(V.S_GRID)
SIG_ALONG, SIG_LAT, RC_DROPOUT = 2.0, 0.75, 0.3          # RC-A50 training noise (v9 README)


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def to_ego(poses_k: np.ndarray, k: int) -> np.ndarray:
    """Future positions [FUT,2] in the ego frame at row k."""
    x0, y0, th = poses_k[k, 0], poses_k[k, 1], poses_k[k, 2]
    d = poses_k[k + 1:k + 1 + FUT, :2] - np.array([x0, y0])
    c, s = np.cos(-th), np.sin(-th)
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1).astype(np.float32)


def ego_state(poses: np.ndarray, k: int):
    v = poses[k, 3]; v3 = poses[k - 3, 3]
    return float(v), float((v - v3) / 0.3), float(wrap(poses[k, 2] - poses[k - 3, 2]) / 0.3)    # v, a_lon, yaw rate (NO curvature)


def factor_distances(path_log, dv_log, v0, paths, profiles):
    """Distances (m) of every vocabulary path / profile to the logged ones: d_P mean point distance along the logged arc length
    (<= 60 m), d_Q mean |arc-length difference| over 6 s. Inputs numpy; returns (d_P [K_P], d_Q [K_V])."""
    s_tot = float(np.sum(np.linalg.norm(np.diff(path_log[:61], axis=0), axis=1)))
    n = int(np.clip(min(s_tot, 60.0), 5, 60)) + 1
    dP = np.linalg.norm(paths[:, :n] - path_log[None, :n], axis=-1).mean(1)
    s_log = np.cumsum(np.maximum(v0 + dv_log, 0) * V.DT)
    s_q = np.cumsum(np.maximum(v0 + profiles, 0) * V.DT, axis=1)
    dQ = np.abs(s_q - s_log[None]).mean(1)
    return dP.astype(np.float32), dQ.astype(np.float32)


class V9Rows:
    """Thin reader of a v9 label npz (keeps arrays in memory; 254 MB for train)."""
    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=True)
        self.z = {k: z[k] for k in z.files}
        self.sha12 = [str(s) for s in self.z["clip__sha12"]]
        self.index = {s: i for i, s in enumerate(self.sha12)}

    def clip(self, sha12: str):
        i = self.index[sha12]; r0, n = int(self.z["clip__row0"][i]), int(self.z["clip__n_rows"][i])
        return dict(sid=int(self.z["clip__sid"][i]), k0=int(self.z["clip__k0"][i]), rows=slice(r0, r0 + n))

    def __getitem__(self, name): return self.z["row__" + name]


class HiCAPWindows(Dataset):
    def __init__(self, cache_dir: str, rows: V9Rows, vocab: V.FactoredVocab, sha12s: list[str] | None = None,
                 train: bool = True, stride: int = 1, require_label: bool = False, rc_variant: str = "A50", seed: int = 0, cfg=None,
                 strat: np.ndarray | None = None, country: dict | None = None):
        self.dir, self.rows, self.vocab, self.train, self.rc = cache_dir, rows, vocab, train, rc_variant
        self.rng = np.random.default_rng(seed); self.cfg = cfg; self.strat = strat       # strat_y per v9 row (strategic.py), None -> IGNORE
        self.country = country or {}                            # sha12 -> country_id (v10 country component; load_country), missing -> IGNORE
        self.clips: dict[str, dict] = {}
        self.index: list[tuple[str, int]] = []                 # (sha12, j)
        for s in (sha12s or rows.sha12):
            f = os.path.join(cache_dir, s + ".npz")
            if not os.path.exists(f): continue
            c = np.load(f, allow_pickle=False)
            meta = rows.clip(s); T = len(c["poses"]); k0 = meta["k0"]
            tick = c["tick_k"]
            self.clips[s] = dict(meta=meta, tick=tick, T=T, path=f)
            n_rows = meta["rows"].stop - meta["rows"].start
            for j in range(0, n_rows, stride):
                k = j                                       # episode row j == v9 row j (trainer clock; raw k = k0 + j)
                if k < 3 or k + FUT >= T or tick.size == 0 or k < tick[0]: continue
                if require_label and rows["lat_cls_a"][meta["rows"].start + j] < 0: continue
                self.index.append((s, j))
        self._open: dict[str, dict] = {}

    def __len__(self): return len(self.index)

    def _clip(self, s):
        if s not in self._open:
            if len(self._open) > 8: self._open.pop(next(iter(self._open)))
            z = np.load(self.clips[s]["path"], allow_pickle=False); self._open[s] = {k: z[k] for k in z.files}
        return self._open[s]

    def episode_ids(self):
        return np.array([self.clips[s]["meta"]["sid"] for s, _ in self.index], np.int64)

    def __getitem__(self, i):
        s, j = self.index[i]; cm = self.clips[s]["meta"]; c = self._clip(s); R = self.rows
        k = j; r = cm["rows"].start + j
        P = c["poses"]; v0, a0, yaw = ego_state(P, k)
        fut = to_ego(P, k)
        path_log, dv_log = V.extract_path_profile(fut, v0)
        dP, dQ = factor_distances(path_log, dv_log, v0, self.vocab.paths, self.vocab.profiles)
        t_idx = int(np.searchsorted(c["tick_k"], k, side="right") - 1)
        age = (k - int(c["tick_k"][t_idx])) * V.DT
        fr = c["frames"][[k - 2, k - 1, k]].astype(np.float32) / 255.0                # [3,3,64,160]
        gy, gw = int(R["goal_y"][r]), int(R["goal_w"][r])
        goal_y = np.array([(gy >> b) & 1 for b in range(22)], np.float32); goal_w = np.array([(gw >> b) & 1 for b in range(22)], np.float32)
        rc = np.array([R[f"rc_{self.rc}_{n}"][r] for n in ("x", "y", "psi", "valid")], np.float32)
        route_present = np.float32(1.0)
        if self.train:                                                                  # noise + dropout >= 0.3 (v9 README)
            th = np.deg2rad(rc[2]); rc[0] += self.rng.normal(0, SIG_ALONG) * np.cos(th) - self.rng.normal(0, SIG_LAT) * np.sin(th)
            rc[1] += self.rng.normal(0, SIG_ALONG) * np.sin(th) + self.rng.normal(0, SIG_LAT) * np.cos(th)
            route_present = np.float32(self.rng.random() >= RC_DROPOUT)
        route_present = np.float32(route_present * (rc[3] > 0.5))
        rc = np.nan_to_num(rc); rc[2] = np.deg2rad(rc[2])
        nav_args = np.nan_to_num(np.array([R["nav_d_next_m"][r] / 100.0, R["nav_d_end_m"][r] / 100.0, R["nav_dyaw_next_deg"][r] / 90.0, R["nav_args_valid"][r]], np.float32))
        idx5 = [int(round(t / V.DT)) - 1 for t in (2.0, 4.0, 6.0)]
        gp = np.stack([np.concatenate([fut[q], [np.linalg.norm(fut[q] - fut[max(q - 1, 0)]) / V.DT]]) for q in idx5]).astype(np.float32)
        lat = int(R["lat_cls_a"][r]); lon = int(R["lon_cls"][r])
        return dict(
            vlm=torch.from_numpy(c["vlm"][t_idx].astype(np.float32)), last=torch.from_numpy(c["last"][t_idx].astype(np.float32)),
            age_s=torch.tensor(age, dtype=torch.float32), frames=torch.from_numpy(fr),
            ego=torch.tensor([v0, a0, yaw], dtype=torch.float32) * torch.tensor([0.1, 0.2, 1.0]),
            nav=torch.tensor(int(np.nan_to_num(R["nav_token"][r])), dtype=torch.long), nav_args=torch.from_numpy(nav_args),
            route=torch.from_numpy(np.array([rc[0] / 50.0, rc[1] / 50.0, rc[2], 1.0], np.float32)), route_present=torch.tensor(route_present),
            strat_y=torch.tensor(int(self.strat[r]) if self.strat is not None else -100),
            country_y=torch.tensor(int(self.country.get(s, -100))), lat_y=torch.tensor(lat if lat >= 0 else -100), lon_y=torch.tensor(lon if lon >= 0 else -100),
            dP=torch.from_numpy(dP), dQ=torch.from_numpy(dQ), goal_y=torch.from_numpy(goal_y), goal_w=torch.from_numpy(goal_w),
            goal_pt=torch.from_numpy(gp / np.array([50.0, 50.0, 10.0], np.float32)), goal_pt_w=torch.ones(3),
            fut=torch.from_numpy(fut), v0=torch.tensor(v0, dtype=torch.float32), dv_log=torch.from_numpy(dv_log),
            path_log=torch.from_numpy(path_log), eid=torch.tensor(cm["sid"]), k=torch.tensor(k))


def load_country(parquet: str) -> dict:
    """v10 country component (`country/v10_country.parquet`): sha12 -> country_id (rules.COUNTRIES order; a LABEL, never an input)."""
    import pandas as pd
    d = pd.read_parquet(parquet, columns=["sha12", "country_id"])
    return {str(a): int(b) for a, b in zip(d.sha12, d.country_id) if int(b) >= 0}


def collate(items: list[dict]) -> dict:
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def targets_from_distances(b: dict, sigma: float = 0.5, hard_lo=0.25, hard_hi=1.0, n_cand: int = 5, vocab: V.FactoredVocab | None = None):
    """Adds yP, yQ (soft targets), hardP/hardQ (near-miss masks for stage 2), cand_p/cand_q + refine_t (open-candidate residuals)."""
    from .losses import soft_target
    b["yP"] = soft_target(b["dP"], sigma); b["yQ"] = soft_target(b["dQ"], sigma)
    b["hardP"] = ((b["dP"] > hard_lo) & (b["dP"] < hard_hi)).float(); b["hardQ"] = ((b["dQ"] > hard_lo) & (b["dQ"] < hard_hi)).float()
    if vocab is not None:
        cp = b["dP"].topk(n_cand, largest=False).indices; cq = b["dQ"].topk(n_cand, largest=False).indices
        b["cand_p"], b["cand_q"] = cp, cq
        ref = []
        for i in range(len(cp)):
            logged = V.slots(b["fut"][i].numpy())
            ref.append(np.stack([(logged - V.slots(V.compose(vocab.paths[int(p)], vocab.profiles[int(q)], float(b["v0"][i])))).reshape(-1) for p, q in zip(cp[i], cq[i])]))
        b["refine_t"] = torch.from_numpy(np.stack(ref).astype(np.float32))
    return b


def compute_priors(ds: HiCAPWindows, kappa: float = 0.5, max_n: int = 20000) -> dict:
    """Train-split marginal log-priors with additive (Dirichlet) smoothing: s2lat, s2lon from labels; s3P/s3V from nearest-node hard assignment."""
    n = min(len(ds), max_n); sel = np.linspace(0, len(ds) - 1, n).astype(int)
    lat = np.zeros(V.N_LAT); lon = np.zeros(V.N_LON); cp = np.zeros(ds.vocab.K_P); cq = np.zeros(ds.vocab.K_V)
    for i in sel:
        it = ds[int(i)]
        if it["lat_y"] >= 0: lat[int(it["lat_y"])] += 1
        if it["lon_y"] >= 0: lon[int(it["lon_y"])] += 1
        cp[int(it["dP"].argmin())] += 1; cq[int(it["dQ"].argmin())] += 1
    lg = lambda c: np.log((c + kappa) / (c.sum() + kappa * len(c)))
    return dict(s2lat=lg(lat), s2lon=lg(lon), s3P=lg(cp), s3V=lg(cq))


# ---------------------------------------------------------------- synthetic clips (tests / smoke)
def make_synth_clip(path_dir: str, sha12: str, sid: int, rng: np.random.Generator, T: int = 160, d_v: int = 48, n_tok: int = 8, k0: int = 2):
    """A physically feasible synthetic clip + cache file; returns (cache_file, v9-like row dict for the clip)."""
    v = np.clip(8 + np.cumsum(rng.normal(0, 0.05, T)), 0, 20); curv = rng.choice([0.0, 0.02, -0.02]) * np.ones(T)
    th = np.cumsum(v * curv * 0.1); x = np.cumsum(v * np.cos(th) * 0.1); y = np.cumsum(v * np.sin(th) * 0.1)
    poses = np.stack([x, y, th, v], 1).astype(np.float32)
    tick = np.arange(0, T, 5)
    os.makedirs(path_dir, exist_ok=True)
    np.savez(os.path.join(path_dir, sha12 + ".npz"), sid=np.int64(sid), poses=poses, frames=rng.integers(0, 255, (T, 3, 64, 160), dtype=np.uint8),
             tick_k=tick, vlm=rng.normal(size=(len(tick), 3, n_tok, d_v)).astype(np.float16), last=rng.normal(size=(len(tick), d_v)).astype(np.float16))
    n = T
    lat = np.full(n, 0 if curv[0] == 0 else (1 if curv[0] > 0 else 2), np.int16)
    a = np.gradient(v) / 0.1; lon = np.where(a > 0.3, 5, np.where(a < -0.3, 4, 6)).astype(np.int16)
    return dict(sid=sid, n=n, lat=lat, lon=lon)


def make_synth_v9(rows_by_clip: list[dict], sha12s: list[str], k0: int = 2) -> dict:
    """Assemble a V9Rows-compatible dict (field names of V9_SCHEMA.md) for synthetic clips."""
    n_rows = [c["n"] for c in rows_by_clip]; tot = sum(n_rows); row0 = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    Z = dict(clip__sha12=np.array(sha12s), clip__sid=np.array([c["sid"] for c in rows_by_clip], np.int64), clip__row0=row0.astype(np.int64),
             clip__n_rows=np.array(n_rows, np.int64), clip__k0=np.full(len(n_rows), k0, np.int64))
    Z["row__lat_cls_a"] = np.concatenate([c["lat"] for c in rows_by_clip]); Z["row__lon_cls"] = np.concatenate([c["lon"] for c in rows_by_clip])
    Z["row__goal_y"] = np.zeros(tot, np.uint32); Z["row__goal_w"] = np.full(tot, (1 << 22) - 1 - (1 << 7), np.uint32)
    for n in ("x", "y", "psi", "valid"): Z[f"row__rc_A50_{n}"] = np.full(tot, 1.0 if n == "valid" else 20.0, np.float64)
    Z["row__nav_token"] = np.zeros(tot); Z["row__nav_d_next_m"] = np.full(tot, 50.0); Z["row__nav_d_end_m"] = np.full(tot, 60.0)
    Z["row__nav_dyaw_next_deg"] = np.zeros(tot); Z["row__nav_args_valid"] = np.ones(tot)
    return Z


def v9rows_from_dict(Z: dict) -> V9Rows:
    o = V9Rows.__new__(V9Rows); o.z = Z; o.sha12 = [str(s) for s in Z["clip__sha12"]]; o.index = {s: i for i, s in enumerate(o.sha12)}; return o
