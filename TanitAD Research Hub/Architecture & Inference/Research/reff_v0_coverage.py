#!/usr/bin/env python3
"""REF-F V0 preview: how well does a fixed, train-derived trajectory set cover val futures?

Torch-free (numpy only): reads the committed REF-C anchor pool (farthest-point samples
over parity-TRAIN ego-frame waypoint targets; smaller sets are nested FPS prefixes) and
the committed val-40 window dump, then reports oracle-in-set ADE/FDE at 0.5/1/1.5/2 s
for nested prefix sizes N: full-set mean over the 881 windows + episode-cluster
bootstrap CI95 (B = 2000, resampling the 40 val episodes).

This previews the ABSOLUTE parameterisation only. REF-F's default residual-over-CTRA
vocabulary needs train trajectories (pod-side) and is measured in M1.
"""
import collections
import json
import pickle
import sys
import zipfile

import numpy as np

ANCHORS = ("TanitAD Research Hub/Data Engineering/Implementation/incoming/"
           "2026-08-04-instrument-durability/refc_anchors_full_REBUILD.pt")
WINDOWS = "taniteval/results/windows_refc-xl-30k.pt"

_DT = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
       "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_,
       "ByteStorage": np.uint8}


def load_pt(path):
    """Minimal reader for torch zip checkpoints holding plain tensors/dicts (no torch)."""
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]

    def rebuild(storage, offset, size, stride, *_):
        item = storage.itemsize
        view = np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                               strides=tuple(s * item for s in stride))
        return np.array(view)

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "torch" and name in _DT:
                return _DT[name]
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, dtype, key, _loc, numel = pid
            return np.frombuffer(z.read(prefix + "data/" + key), dtype=dtype)[:numel]

    return Unpickler(z.open(pkl)).load()


def episode_bootstrap(values, eid, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    eps = np.unique(eid)
    idx = {e: np.flatnonzero(eid == e) for e in eps}
    means = []
    for _ in range(B):
        pick = rng.choice(eps, size=len(eps), replace=True)
        means.append(np.concatenate([values[idx[e]] for e in pick]).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def main():
    anc = load_pt(ANCHORS)
    win = load_pt(WINDOWS)
    A = np.asarray(anc["anchors"], dtype=np.float64)          # [N,4,2], FPS order
    gt = np.asarray(win["gt"], dtype=np.float64)              # [W,4,2]
    eid = np.asarray(win["eid"]).reshape(-1)
    out = {"anchors_file": ANCHORS, "windows_file": WINDOWS,
           "anchor_meta": {k: (v if isinstance(v, (int, float, str)) else str(type(v).__name__))
                           for k, v in anc.items() if k != "anchors"},
           "n_pool": int(A.shape[0]), "n_windows": int(gt.shape[0]),
           "n_episodes": int(len(np.unique(eid))),
           "estimator": "full_set mean + episode-cluster bootstrap CI95, B=2000",
           "rows": []}
    d = np.linalg.norm(gt[:, None, :, :] - A[None, :, :, :], axis=-1)   # [W,N,4]
    ade_all = d.mean(-1)                                                 # [W,N]
    fde_all = d[..., -1]
    Ns = [n for n in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192) if n <= A.shape[0]]
    if A.shape[0] not in Ns:
        Ns.append(int(A.shape[0]))
    for n in Ns:
        best = ade_all[:, :n].argmin(1)
        ade = ade_all[np.arange(len(gt)), best]
        fde = fde_all[np.arange(len(gt)), best]
        m, lo, hi = episode_bootstrap(ade, eid)
        fm, flo, fhi = episode_bootstrap(fde, eid)
        out["rows"].append({"N": n, "oracle_ade_2s": [round(m, 4), round(lo, 4), round(hi, 4)],
                            "oracle_fde_2s": [round(fm, 4), round(flo, 4), round(fhi, 4)]})
    # Crude preview of the residual/prior idea, using only inference-time inputs:
    # rescale each anchor uniformly so its implied initial speed (|wp@0.5s| / 0.5 s)
    # equals the window's v0.  Scale clamped to [0, 3]; stopped anchors are kept as is.
    v0 = np.asarray(win["speed"], dtype=np.float64).reshape(-1)          # [W] m/s
    s_anchor = np.linalg.norm(A[:, 0, :], axis=-1) / 0.5                 # [N]
    scale = np.where(s_anchor[None, :] > 0.5,
                     np.clip(v0[:, None] / np.maximum(s_anchor[None, :], 1e-6), 0.0, 3.0), 1.0)
    As = A[None, :, :, :] * scale[:, :, None, None]                      # [W,N,4,2]
    ds = np.linalg.norm(gt[:, None, :, :] - As, axis=-1)
    ade_s = ds.mean(-1)
    out["speed_normalised_rows"] = []
    for n in Ns:
        best = ade_s[:, :n].argmin(1)
        ade = ade_s[np.arange(len(gt)), best]
        m, lo, hi = episode_bootstrap(ade, eid)
        out["speed_normalised_rows"].append({"N": n, "oracle_ade_2s": [round(m, 4), round(lo, 4), round(hi, 4)]})
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
