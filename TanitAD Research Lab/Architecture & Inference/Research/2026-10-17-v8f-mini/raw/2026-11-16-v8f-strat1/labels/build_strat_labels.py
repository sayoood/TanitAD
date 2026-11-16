"""Per-step strategic context label table aligned row for row to navtok_table.npz (train4572 + eval141).

Time rule (stated):  label frame f has video time t_f = frame_idx*0.2 s (release `t_s`; video_frame = 6*frame_idx, nominal 30 fps).
Cache step j of a clip sits at video time  tv(j) = j * dt_c,  dt_c = median spacing of the join's `now_s` over that clip's steps
(median 0.20133 s = 2 x 0.100667 s cache grid; step 0 = first camera row, MEASURED: implied now_s(0) median 0.1131 s = sidecar
grid_start median 0.1135 s).  ASSUMED: cache row 0 = video frame 0 and video time is frame/30 s.
Alignment: nearest PRESENT label frame in |tv - t_f|; ties go to the earlier frame; valid only if |tv - t_f| <= TOL_S (0.25 s).
Past the last label frame (up to TOL_S) the last label is HELD.  A class of `unknown` is invalid for that variable.
flag: 0 ok | 1 clip absent from the release | 2 step further than TOL_S from every present label frame
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os
import numpy as np

TOL_S = 0.25
LABEL_DT = 0.2
VARS = ["country", "weather", "surface", "visibility", "illumination", "period_of_day"]
JOIN_DIR = "F:/TanitAD-scratch/datascale/navtok/raw/"
REL = "F:/TanitAD-scratch/strat1/labels/rel/"


def step_times(now_s: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(now_s)
    d = d[np.isfinite(d)]
    dt = float(np.median(d)) if len(d) else 0.2013333
    return np.arange(n, dtype=np.float64) * dt


def align(tv: np.ndarray, t_lab: np.ndarray, tol: float = TOL_S):
    """tv: step times; t_lab: sorted label-frame times. Returns (index into t_lab or -1, signed offset tv - t_lab)."""
    p = np.searchsorted(t_lab, tv, "left")
    lo = np.clip(p - 1, 0, len(t_lab) - 1)
    hi = np.clip(p, 0, len(t_lab) - 1)
    use_hi = np.abs(t_lab[hi] - tv) < np.abs(tv - t_lab[lo])
    k = np.where(use_hi, hi, lo)
    off = tv - t_lab[k]
    return np.where(np.abs(off) <= tol, k, -1), off


def build(J: dict, lab: dict, vocab: dict, tol: float = TOL_S, t_shift: float = 0.0) -> dict:
    """lab: clip -> dict(t_s, present-mask-applied arrays per var as int ids (-1 = unknown))."""
    N = int(J["n"].sum())
    out = {"frame_idx": np.full(N, -1, np.int16), "off_s": np.full(N, np.nan, np.float32), "flag": np.zeros(N, np.uint8)}
    for v in VARS:
        out[v + "_id"] = np.full(N, -1, np.int8)
        out[v + "_valid"] = np.zeros(N, bool)
    for ci, cid in enumerate(J["clip"]):
        o, n = int(J["off"][ci]), int(J["n"][ci])
        sl = slice(o, o + n)
        L = lab.get(str(cid))
        if L is None:
            out["flag"][sl] = 1
            continue
        tv = step_times(J["now_s"][sl], n) + t_shift
        k, off = align(tv, L["t_s"], tol)
        ok = k >= 0
        out["flag"][sl] = np.where(ok, 0, 2)
        out["frame_idx"][sl] = np.where(ok, L["frame_idx"][np.clip(k, 0, None)], -1)
        out["off_s"][sl] = np.where(ok, off, np.nan)
        for v in VARS:
            i = L[v][np.clip(k, 0, None)]
            i = np.where(ok, i, -1)
            out[v + "_id"][sl] = i
            out[v + "_valid"][sl] = i >= 0
    return out


def load_release():
    import pandas as pd
    cols = ["clip_id", "frame_idx", "t_s", "frame_present"] + VARS
    F = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(REL + "frames/*.parquet"))], ignore_index=True)
    vocab = {}
    for v in VARS:
        vs = sorted(x for x in F[v].dropna().unique() if x != "unknown")
        vocab[v] = vs
    F = F[F.frame_present].sort_values(["clip_id", "frame_idx"], kind="stable")
    ids = {v: F[v].map({c: i for i, c in enumerate(vocab[v])}).fillna(-1).astype(np.int8).to_numpy() for v in VARS}
    cl = F.clip_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(cl[1:] != cl[:-1]) + 1, len(cl)]
    lab = {}
    ts = F.t_s.to_numpy(np.float64); fi = F.frame_idx.to_numpy(np.int16)
    for a, b in zip(starts[:-1], starts[1:]):
        lab[str(cl[a])] = {"t_s": ts[a:b], "frame_idx": fi[a:b], **{v: ids[v][a:b] for v in VARS}}
    return lab, vocab


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="F:/TanitAD-scratch/strat1/labels/strat_labels_table.npz")
    a = ap.parse_args()
    lab, vocab = load_release()
    nt = np.load("F:/TanitAD-scratch/datascale/navtok/navtok_table.npz")
    pack, rep = {}, {"tol_s": TOL_S, "release": "Sayood/tanitad-context-labels-v1@100f03da5c068e841b6eb7b6fc768f0f39714b27",
                     "flag_legend": {"0": "ok", "1": "clip absent from the release", "2": "step further than tol from every present label frame"}}
    for split, jf in (("train", "join_train.npz"), ("eval", "join_eval141.npz")):
        J = dict(np.load(JOIN_DIR + jf))
        for k in ("clip", "off", "n"):   # row-for-row identity with navtok_table
            assert np.array_equal(nt[f"{split}_{k}"], J[k]), (split, k)
        T = build(J, lab, vocab)
        assert len(T["flag"]) == len(nt[f"{split}_flag"]) == int(J["n"].sum())
        for k in ("clip", "off", "n"):
            pack[f"{split}_{k}"] = J[k]
        for k, v in T.items():
            pack[f"{split}_{k}"] = v
        rep[split] = {"clips": int(len(J["n"])), "steps": int(len(T["flag"])),
                      "flags": {str(i): int((T["flag"] == i).sum()) for i in range(3)},
                      "valid_steps": {v: int(T[v + "_valid"].sum()) for v in VARS}}
    json.dump(vocab, open(os.path.join(os.path.dirname(a.out), "strat_labels_vocab.json"), "w"), indent=1)
    for v in VARS:
        pack["vocab_" + v] = np.array(vocab[v])
    np.savez_compressed(a.out, **pack)
    rep["file"] = os.path.basename(a.out); rep["bytes"] = os.path.getsize(a.out); rep["sha256"] = sha256(a.out)
    json.dump(rep, open(a.out.replace(".npz", "_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
