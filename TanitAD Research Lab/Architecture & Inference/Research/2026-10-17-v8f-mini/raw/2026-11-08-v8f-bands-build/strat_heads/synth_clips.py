"""Synthetic clips of the real schemas, for local tests and smoke runs: zcache npz (zmean f32 [T,256], ok, t_s), release frames parquet, stream A strategic_geom npz, sidecar npz (act, v, nav).
The signal is planted so every head can learn: zmean carries the country / context as additive vectors, the checkpoints integrate the logged actions, nav and context decide the goal token."""
import os

import numpy as np

import strat_heads as S


def make(root: str, n_train: int = 12, n_eval: int = 6, T: int = 101, seed: int = 0, d: int = 256, nav_dim: int = 8) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    emb = {n: rng.normal(0, 1, (k, d)).astype(np.float32) for n, k in zip(("country", "illumination", "weather", "surface", "visibility"), S.CTX_DIMS)}
    paths = {}
    for split, n in (("train", n_train), ("eval", n_eval)):
        zdir, sdir = os.path.join(root, "zcache", split), os.path.join(root, "side", split)
        os.makedirs(zdir, exist_ok=True)
        os.makedirs(sdir, exist_ok=True)
        rows, ids, geo_xyh, geo_valid, offs = [], [], [], [], []
        for i in range(n):
            cid = f"{split}-clip-{i:04d}"
            ids.append(cid)
            country = int(rng.integers(0, 4)) if split == "train" else int(rng.integers(0, 4))
            ctx = [int(rng.integers(0, k)) for k in S.CTX_DIMS[1:]]
            z = rng.normal(0, 0.3, (T, d)).astype(np.float32) + emb["country"][country]
            for (name, _), c in zip(S.CTX_ATTRS, ctx):
                z += emb[name][c]
            ok = np.ones(T, bool)
            ok[0] = False if i % 5 == 0 else True
            np.savez(os.path.join(zdir, cid + ".npz"), zmean=z, ok=ok, t_s=(np.arange(T) * 0.2).astype(np.float32))
            Te = T + 160
            v0 = float(rng.uniform(3, 25))
            yaw = float(rng.normal(0, 0.03))
            act = np.stack([np.zeros(Te, np.float32) + float(rng.normal(0, 0.2)), np.full(Te, yaw, np.float32)], 1)
            v = np.full(Te, v0, np.float32)
            nav = np.zeros((T, nav_dim), np.float32)
            cmd = int(rng.integers(0, 3))
            nav[:, cmd] = 1.0
            np.savez(os.path.join(sdir, cid + ".npz"), act=act, v=v, nav=nav)
            tok = [0, 1, 2][cmd]
            xyh = np.zeros((T, S.N_HZ, 3), np.float32)
            for h, hz in enumerate(S.HORIZONS_S):
                xyh[:, h] = [v0 * hz, 0.5 * yaw * v0 * hz * hz, yaw * hz]
            geo_xyh.append(xyh)
            valid = np.ones((T, S.N_HZ), bool)
            valid[:, -1] = (np.arange(T) < T - 20)
            geo_valid.append(valid)
            offs.append(i * T)
            for t in range(T):
                rows.append({"clip_id": cid, "frame_idx": t, "country": S.COUNTRIES[country], "weather": S.WEATHER[ctx[1]], "surface": S.SURFACE[ctx[2]], "visibility": S.VISIB[ctx[3]],
                             "illumination": S.ILLUM[ctx[0]], "g_str_token": S.STR_TOKENS[tok], "g_str_by_time_s": float("nan") if tok == 0 else 12.0 + cmd, "g_str_within_m": float("nan") if tok == 0 else 90.0 + 10 * cmd,
                             "g_str_valid": True})
        fdir = os.path.join(root, "release_" + split, "frames")
        os.makedirs(fdir, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), os.path.join(fdir, "frames_0000.parquet"))
        geo = os.path.join(root, f"strategic_geom_{split}.npz")
        np.savez(geo, clip_ids=np.array(ids), horizons_s=np.array(S.HORIZONS_S, np.float32), ck_xyh=np.concatenate(geo_xyh), ck_valid=np.concatenate(geo_valid),
                 clip__off=np.array(offs), clip__n=np.full(n, T))
        paths[split] = {"zcache": zdir, "side": sdir, "release": os.path.join(root, "release_" + split), "geom": geo}
    return paths
