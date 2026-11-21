"""bitid_digest.py -- digest every forward output + every state tensor of the refcv8 CPU rig, for a pristine-vs-modified
bit-identity read.  Run once per tree with PYTHONPATH=<tree>/stack; compare the two JSONs (`--compare a.json b.json`).

  CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<tree>/stack python -X utf8 bitid_digest.py --tree <tree> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _digest(t) -> str:
    import torch
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def _walk(obj, pre, acc):
    import torch
    if torch.is_tensor(obj):
        acc[pre] = _digest(obj)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _walk(obj[k], f"{pre}.{k}", acc)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(v, f"{pre}[{i}]", acc)
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        acc[pre] = repr(obj)


def run(tree: Path) -> dict:
    sys.path.insert(0, str(tree / "stack" / "tests"))
    import torch
    import tanitad
    import _refcv8_rig as R
    torch.set_num_threads(1)
    res = {"tanitad_file": tanitad.__file__, "torch": torch.__version__, "arms": {}}
    T = R.trainer()
    for name, r8, vmax in (("refcv7_rig", False, False), ("refcv8_rig", True, False), ("refcv8_rig_vmax", True, True)):
        cfg, m = R.build(T, r8, seed=0, vmax=vmax)
        bt = R.batch(cfg, b=3, seed=1)
        kw = {}
        if vmax:
            kw = {"v_max_ms": torch.tensor([8.33, 13.89, 22.22]), "v_max_valid": torch.tensor([1.0, 1.0, 0.0])}
        acc = {}
        for tag, train in (("eval", False), ("train", True)):
            out = R.forward(cfg, m, bt, seed=2, train=train, **kw)
            _walk(out, f"{tag}.out", acc)
        sd = {k: _digest(v) for k, v in m.state_dict().items()}
        res["arms"][name] = {"out": acc, "state": sd, "n_out": len(acc), "n_state": len(sd)}
    return res


def compare(a: dict, b: dict) -> dict:
    rep = {}
    for arm in a["arms"]:
        A, Bm = a["arms"][arm], b["arms"][arm]
        d_out = sorted(k for k in A["out"] if A["out"][k] != Bm["out"].get(k))
        x_out = sorted(set(Bm["out"]) - set(A["out"]))
        d_st = sorted(k for k in A["state"] if A["state"][k] != Bm["state"].get(k))
        x_st = sorted(set(Bm["state"]) - set(A["state"]))
        rep[arm] = {"n_out": A["n_out"], "out_differ": d_out, "out_extra_in_b": x_out,
                    "n_state": A["n_state"], "state_differ": d_st, "state_extra_in_b": x_st,
                    "BIT_IDENTICAL": not (d_out or x_out or d_st or x_st)}
    rep["ALL_BIT_IDENTICAL"] = all(v["BIT_IDENTICAL"] for v in rep.values() if isinstance(v, dict))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.compare:
        r = compare(*(json.loads(Path(p).read_text(encoding="utf-8")) for p in a.compare))
        Path(a.report).write_text(json.dumps(r, indent=1), encoding="utf-8")
        print("ALL_BIT_IDENTICAL", r["ALL_BIT_IDENTICAL"])
        sys.exit(0 if r["ALL_BIT_IDENTICAL"] else 3)
    r = run(Path(a.tree))
    Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print("tanitad from", r["tanitad_file"], {k: (v["n_out"], v["n_state"]) for k, v in r["arms"].items()})
