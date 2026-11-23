"""KINCACHE on REAL records (CPU, one python process): the first N ok records of shard 0 of the RH bank s0 (pulled read-only from the pod; the same inputs as census 2/3), tree = tree2.
For speed_v1 and kin_v1: fresh compute -> write cache -> read cache; the labels, status_counts, is_kin, n_labelled and census must be equal BIT FOR BIT; times and file size are recorded.
usage: python kincache_real.py <shard.npz> <tracks.pt> <out.json> <tree_root> <scratch_dir> [N]"""
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

shard, tpath, outp, tree, scratch = sys.argv[1:6]
N = int(sys.argv[6]) if len(sys.argv) > 6 else 4000
sys.path.insert(0, tree + "/stack")
sys.path.insert(0, tree + "/taniteval")
import torch  # noqa: E402
import tanitad  # noqa: E402
from tanitad.data import v8f_dagger as G  # noqa: E402
from tanitad.data import v8f_recovery as RV  # noqa: E402

assert tanitad.__file__.replace("\\", "/").startswith(tree + "/"), tanitad.__file__


def read_member(zf, name, last_row_only=False, chunk=65536):
    with zf.open(name + ".npy") as fh:
        ver = np.lib.format.read_magic(fh)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(fh)
        n = shape[0]
        rest = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        if not last_row_only:
            return np.frombuffer(fh.read(n * rest * dtype.itemsize), dtype).reshape(shape).copy()
        out = np.empty((n,) + tuple(shape[2:]), dtype)
        per = rest * dtype.itemsize
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            out[a:b] = np.frombuffer(fh.read((b - a) * per), dtype).reshape((b - a,) + tuple(shape[1:]))[:, -1]
        return out


def main():
    h = hashlib.sha256()
    with open(shard, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    zf = zipfile.ZipFile(shard)
    hdr = json.loads(bytes(read_member(zf, "header_json")).decode("utf-8"))
    ci, rr, reason, tick = (read_member(zf, k) for k in ("ci", "r", "reason", "tick"))
    p10_last = read_member(zf, "p10f", last_row_only=True)
    names = {int(k): v for k, v in hdr["clips"].items()}
    clip_names = np.array([names.get(i, "") for i in range(max(names) + 1)], dtype=object)
    tracks = {c: t.double().numpy() for c, t in torch.load(tpath, map_location="cpu", weights_only=True)["tracks"].items()}
    read_clip = lambda name: {"pose": tracks[name][0::2]}                      # noqa: E731  the logged 0.2 s grid
    sel = np.flatnonzero(reason == 0)[:N]
    # N ok records PLUS the first 500 non-ok records, so the eligible index is a proper subset of the records
    extra = np.flatnonzero(reason != 0)[:500]
    sel = np.sort(np.concatenate([sel, extra]))
    arrays = {"ci": ci[sel], "reason": reason[sel], "r": rr[sel], "tick": tick[sel], "p10f": p10_last[sel][:, None, :]}
    bank = G.DaggerBank(arrays, {}, sha256=h.hexdigest(), path=shard)        # the shard's own sha256 stands for bank.json's (this is a shard, not a merged bank)
    out = {"inputs": {"shard_sha256": h.hexdigest(), "n_records": int(bank.n), "n_ok": int((bank.reason == 0).sum()), "n_clips": int(len(np.unique(bank.ci))), "tree": tree}, "kinds": {}}
    scr = Path(scratch)
    scr.mkdir(parents=True, exist_ok=True)
    for kind, fn in (("speed_v1", RV.label_bank), ("kin_v1", RV.kin_label_bank)):
        t0 = time.time()
        res = fn(bank.a, bank.ci, clip_names, read_clip)
        t_compute = time.time() - t0
        t0 = time.time()
        ident = G.label_cache_identity(bank, clip_names, read_clip, label_kind=kind, labeller_path=RV.__file__, dagger_path=G.__file__)
        t_ident = time.time() - t0
        path = scr / f"real_{kind}.npz"
        t0 = time.time()
        sha_w = G.write_label_cache(path, ident, res)
        t_write = time.time() - t0
        t0 = time.time()
        back, sha_r = G.read_label_cache(path, ident)
        t_read = time.time() - t0
        eq = {"slot": back["slot"].dtype == res["slot"].dtype and back["slot"].tobytes() == res["slot"].tobytes(),
              "label": back["label"].dtype == res["label"].dtype and back["label"].shape == res["label"].shape and back["label"].tobytes() == res["label"].tobytes(),
              "status_counts": list(back["status_counts"].items()) == list(res["status_counts"].items()),
              "n_labelled": back["n_labelled"] == res["n_labelled"],
              "census": json.dumps(back["census"]) == json.dumps(res["census"]) and G.canon(back["census"]) == G.canon(res["census"])}
        if kind == "kin_v1":
            eq["is_kin"] = back["is_kin"].tobytes() == res["is_kin"].tobytes()
        out["kinds"][kind] = {"bit_for_bit": eq, "all_equal": all(eq.values()), "sha_write_equals_read": sha_w == sha_r, "file_bytes": path.stat().st_size, "n_labelled": res["n_labelled"], "n_label_rows": int(len(res["label"])),
                              "n_kin": int(res["is_kin"].sum()) if kind == "kin_v1" else None, "status_counts": res["status_counts"],
                              "seconds": {"compute": round(t_compute, 2), "identity": round(t_ident, 3), "write": round(t_write, 3), "read": round(t_read, 3)}, "sha256": sha_r}
        print(kind, json.dumps(out["kinds"][kind]["bit_for_bit"]), out["kinds"][kind]["seconds"], flush=True)
        path.unlink()                                                          # delete our own scratch copy at once (the CLAUDE.md rule)
    json.dump(out, open(outp, "w"), indent=1)
    assert all(v["all_equal"] for v in out["kinds"].values()), "ROUND TRIP NOT BIT FOR BIT"


main()
