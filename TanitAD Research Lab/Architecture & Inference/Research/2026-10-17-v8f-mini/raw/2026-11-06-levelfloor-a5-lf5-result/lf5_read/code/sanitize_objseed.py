#!/usr/bin/env python3
"""sanitize_objseed.py -- make the four RILFO100 tick dumps loadable by the pinned tick loader, changing nothing the read uses.

Defect (found when the LF5 chain's reader failed, 2026-11-06 06:21): with sigma = 0 the LF5 tool writes the header value
``floor_vhat_noise_seed = None`` into the ticks npz, which numpy stores as a 0-d OBJECT array. The pinned loader
(t1hd_read.load_ticks, called by extract_smooth.py) opens ticks with allow_pickle=False and refuses the whole file.
The noise arms carry an int64 seed there and extracted fine; the A4 tool wrote no such key and its ticks extracted fine.

Fix (after the rollouts, before any rule is read): for each RILFO100 rollout, the original file is MOVED ASIDE to
``ticks_<tag>.npz.objseed_orig`` (kept, never deleted) and a new ``ticks_<tag>.npz`` is written with every member copied
BYTE FOR BYTE except ``floor_vhat_noise_seed.npy``, which is dropped (absent = no draw, as in the A4 tool's ticks).
The seed stays recorded in the run header, which is where the reader's controls read it (lf5_precheck.py:72).
Checks, all recorded in raw/SANITIZE_OBJSEED.json: the dropped member is a 0-d object array holding None; every kept
member's bytes are identical; the new file loads with allow_pickle=False; no other member is an object array.
usage: python -I sanitize_objseed.py [--out-root D]"""
import argparse
import hashlib
import json
import os
import sys
import zipfile

import numpy as np
from numpy.lib import format as F

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
TAGS = ("RILFO100_s0", "RILFO100_s1", "RILFO100C_s0", "RILFO100C_s1")
KEY = "floor_vhat_noise_seed.npy"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def member_dtype(raw):
    import io
    f = io.BytesIO(raw)
    v = F.read_magic(f)
    sh, _fo, dt = F.read_array_header_1_0(f) if v == (1, 0) else F.read_array_header_2_0(f)
    return sh, dt


def one(out_root, tag):
    p = f"{out_root}/{tag}/ticks/ticks_{tag}.npz"
    orig = p + ".objseed_orig"
    if os.path.exists(orig):
        src = orig
    else:
        src = p
    with open(src, "rb") as f:
        src_bytes = f.read()
    r = {"tag": tag, "orig_path": orig, "orig_sha256": sha(src_bytes)}
    with zipfile.ZipFile(src) as z:
        members = [(i, z.read(i.filename)) for i in z.infolist()]
    drop = [m for m in members if m[0].filename == KEY]
    if len(drop) != 1:
        raise SystemExit(f"REFUSED {tag}: expected exactly one {KEY}, found {len(drop)}")
    sh_, dt_ = member_dtype(drop[0][1])
    val = np.load(__import__("io").BytesIO(drop[0][1]), allow_pickle=True)
    r["dropped"] = {"member": KEY, "dtype": str(dt_), "shape": list(sh_), "value_is_None": bool(val.shape == () and val.item() is None)}
    if not (dt_.hasobject and sh_ == () and r["dropped"]["value_is_None"]):
        raise SystemExit(f"REFUSED {tag}: {KEY} is not a 0-d object array holding None: {r['dropped']}")
    others_obj = [i.filename for i, b in members if i.filename != KEY and member_dtype(b)[1].hasobject]
    if others_obj:
        raise SystemExit(f"REFUSED {tag}: other object members {others_obj}")
    if src == p:
        os.replace(p, orig)                                                    # moved aside, kept
    tmp = p + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zo:
        for i, b in members:
            if i.filename == KEY:
                continue
            zo.writestr(i.filename, b, compress_type=i.compress_type)
    os.replace(tmp, p)
    with zipfile.ZipFile(p) as zn:
        new = {i.filename: zn.read(i.filename) for i in zn.infolist()}
    kept = {i.filename: b for i, b in members if i.filename != KEY}
    r["n_members_orig"], r["n_members_new"] = len(members), len(new)
    r["kept_members_byte_identical"] = bool(set(new) == set(kept) and all(new[k] == kept[k] for k in kept))
    z2 = np.load(p, allow_pickle=False)
    r["new_loads_without_pickle"] = bool(len({k: z2[k] for k in z2.files}) == len(kept))
    with open(p, "rb") as f:
        r["new_sha256"] = sha(f.read())
    r["PASS"] = bool(r["kept_members_byte_identical"] and r["new_loads_without_pickle"] and r["n_members_new"] == r["n_members_orig"] - 1)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="F:/TanitAD-scratch/motion_c/lf5_read/out")
    ap.add_argument("--raw", default=f"{PK}/raw")
    a = ap.parse_args(argv)
    res = {"why": __doc__.split("\n\n")[1].strip(), "results": [one(a.out_root, t) for t in TAGS]}
    res["PASS"] = all(r["PASS"] for r in res["results"])
    with open(f"{a.raw}/SANITIZE_OBJSEED.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    for r in res["results"]:
        print(r["tag"], "PASS" if r["PASS"] else "FAIL", r["orig_sha256"][:12], "->", r["new_sha256"][:12])
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
