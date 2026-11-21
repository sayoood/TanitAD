"""Load the banked refcv4b/refcv3 step-40284 T1 dump (141 ep / 4823 windows)."""
import json
import os

import numpy as np

DUMP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
        r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
        r"\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad\dump40284"
        r"\refcv3_40284_dump")

TRAJ_KEYS = ("g", "os", "ha", "ha0", "os_navshuf", "os_navzero", "oracle_sel")
SCALARS = ("v0", "ws")


def episodes():
    fs = sorted(f for f in os.listdir(DUMP) if f.startswith("ep")
                and f.endswith(".npz"))
    return fs


def load_all():
    """-> dict of concatenated arrays + `epi` (episode row index per window)."""
    out = {}
    dec = {}
    epi, eid, clip = [], [], []
    per_ep = []
    for i, f in enumerate(episodes()):
        z = np.load(os.path.join(DUMP, f), allow_pickle=True)
        d = np.load(os.path.join(DUMP, "decisions", f), allow_pickle=True)
        n = int(z["ws"].shape[0])
        assert int(d["ws"].shape[0]) == n, (f, n, d["ws"].shape)
        assert np.array_equal(z["ws"], d["ws"]), f
        for k in z.files:
            if k in ("eid", "clip_index"):
                continue
            out.setdefault(k, []).append(z[k])
        for k in d.files:
            if k == "ws":
                continue
            dec.setdefault(k, []).append(d[k])
        epi.append(np.full(n, i, dtype=np.int64))
        eid.append(np.full(n, int(z["eid"][0]), dtype=np.int64))
        clip.append(np.full(n, int(z["clip_index"][0]), dtype=np.int64))
        per_ep.append(n)
    res = {k: np.concatenate(v, 0) for k, v in out.items()}
    res.update({k: np.concatenate(v, 0) for k, v in dec.items()})
    res["epi"] = np.concatenate(epi)
    res["eid"] = np.concatenate(eid)
    res["clip_index"] = np.concatenate(clip)
    res["_n_per_ep"] = np.array(per_ep, dtype=np.int64)
    res["_files"] = np.array(episodes())
    return res


def manifest():
    with open(os.path.join(DUMP, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)
