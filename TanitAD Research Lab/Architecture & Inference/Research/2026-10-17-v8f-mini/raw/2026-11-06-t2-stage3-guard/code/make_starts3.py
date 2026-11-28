"""Stage 3, item 1 (AGENT K): the guard-scene starts of PREREG_T2_CRAWL section 3 + 6b, with their start classes.

    python -I make_starts3.py [--out stage3/GUARD_STARTS.json]

The definitions are in `DEFINITIONS_K.md` (written before the first draw). The draw is `t2v8f_kit.all_starts()` (the kit's, unchanged); this module RE-DERIVES it by hand (`derive`) from the screening
table and refuses to write if the two differ. Per start it records the logged speed at f0 and the class (`make_stage2.start_class`, read from the scene's own `rig_trajectories.json`).
Writes GUARD_STARTS.json: per guard scene (slot, order_index, scene, idx 42..61, n_ticks, lo, hi, the raw draws, the starts after the collapse, v0, class, 6 s extent) + a sha256 over the
starts (the identity every later artifact is checked against). Dev box, CPU. Nothing is committed or staged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
T2 = os.path.dirname(HERE)
RAW = f"{os.path.dirname(T2)}/t2_scenes/raw"
IDX0 = 42                                   # stage 2's launch scenes are idx 0..41 (DEFINITIONS_K.md row 8)
F_MIN, TAIL, N_DRAWS = 18, 61, 2            # lo = 18 (card 6b); hi = n_ticks - 61 (leave 60 ticks = 6 s); two draws per scene


def guard_entries(table):
    """The selected guard scenes in `order_index` order (= guard_1 .. guard_20)."""
    g = sorted((e for e in table if e.get("selected") and str(e.get("slot", "")).startswith("guard")), key=lambda e: int(e["order_index"]))
    assert [e["slot"] for e in g] == [f"guard_{k}" for k in range(1, len(g) + 1)], "the guard slots are not guard_1..guard_N in order_index order"
    return g


def derive(table, seed=1):
    """Hand-rolled draw, independent of the kit: ONE default_rng(seed), per guard scene in order two `integers(lo, hi + 1)` in sequence, equal draws collapsed (order kept).
    -> [{scene, slot, n_ticks, lo, hi, raw, starts}]."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for e in guard_entries(table):
        lo, hi = F_MIN, int(e["n_ticks"]) - TAIL
        raw = [int(rng.integers(lo, hi + 1)) for _ in range(N_DRAWS)]
        starts = []
        for f in raw:
            if f not in starts:
                starts.append(f)
        out.append({"scene": e["scene"], "slot": e["slot"], "n_ticks": int(e["n_ticks"]), "lo": lo, "hi": hi, "raw": raw, "starts": starts})
    return out


def starts_sha256(starts_by_scene):
    """sha256 over `scene|f0,f1` lines in guard order: the identity of a draw."""
    h = hashlib.sha256()
    for sid, s in starts_by_scene:
        h.update(f"{sid}|{','.join(str(int(f)) for f in s)}\n".encode())
    return h.hexdigest()


def build(table=None, plan=None, rig_root=None, seed=1):
    import make_stage2 as MK
    import t2v8f_kit as K
    import t2v8f_stage2 as D
    if table is None:
        table = json.load(open(K.SCREEN_TABLE, encoding="utf-8"))["scenes"]
    if plan is None:
        plan = json.load(open(f"{RAW}/DOWNLOAD_PLAN.json", encoding="utf-8"))["scenes"]
    root_of = {p["scene"]: p["root"] for p in plan}
    mine = derive(table, seed)
    kit = K.all_starts(table) if seed == K.GUARD_SEED else None
    rows = []
    for k, d in enumerate(mine):
        if kit is not None:
            assert kit[d["scene"]] == d["starts"], f"{d['slot']}: the kit draws {kit[d['scene']]}, the hand derivation {d['starts']}"
        rig = f"{rig_root or root_of[d['scene']]}/{d['scene']}/rig_trajectories.json"
        gt, ts = D.load_rig(rig)
        assert len(ts) == d["n_ticks"], f"{d['slot']}: the rig has {len(ts)} ticks, the screening table {d['n_ticks']}"
        assert all(d["lo"] <= f <= d["hi"] and f + 60 < len(ts) for f in d["starts"]), f"{d['slot']}: a start leaves less than 6 s or less than 18 frames of history"
        cls = [MK.start_class(gt, f) for f in d["starts"]]
        rows.append(dict(d, order_index=int(next(e["order_index"] for e in table if e["scene"] == d["scene"])), idx=IDX0 + k, start_class=[c[0] for c in cls],
                         start_v0_mps=[c[1] for c in cls], start_extent_6s_m=[c[2] for c in cls]))
    return {"rule": "ONE numpy default_rng(1); per guard scene (order_index order) two integers(18, n_ticks - 61 inclusive); equal draws collapse (DEFINITIONS_K.md)", "seed": seed,
            "numpy": __import__("numpy").__version__, "n_scenes": len(rows), "n_starts": sum(len(r["starts"]) for r in rows),
            "start_classes": {c: sum(r["start_class"].count(c) for r in rows) for c in ("launch", "moving")},
            "starts_sha256": starts_sha256([(r["scene"], r["starts"]) for r in rows]), "scenes": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{T2}/stage3/GUARD_STARTS.json")
    a = ap.parse_args(argv)
    res = build()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "scenes"}, indent=1))
    for r in res["scenes"]:
        print(r["slot"], r["scene"][:8], "n", r["n_ticks"], "raw", r["raw"], "starts", r["starts"], r["start_class"], r["start_v0_mps"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
