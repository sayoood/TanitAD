"""KINLAB census 2 (CPU, one python process): ``--dagger-label kin_v1`` on shard 0 of the RH bank s0 (n ok = 372,791), tree = the KINLAB build (stack of df250a00 + the kin_v1 change).

Reads shard members ci, r, reason, tick, j, kin3 and the LAST row of p10f (row chunks: low RAM), the human's native 10 Hz tracks (cache grid = every 2nd frame, as ``label_bank`` is read in RECLAB), then
  1. GOLD ON REAL RECORDS: the first GOLD_N ok records' speed_v1 labels (``label_bank``) and human labels equal the df250a00 code (``tests/golden_v8f_recovery_df250a00.py``) bit for bit;
  2. ``kin_label_bank`` over every ok record (the constants are the MM's design; NOTHING is tuned on this census): relabelled / identity / refused by reason, per tick and pooled, of ok records; the refused-infeasible checker terms;
     merge time, max lateral deviation from the human polyline, |P_kin[30] - P_speed[30]| and |P_kin[5] - P_speed[5]| (point distance, m) and the arc difference at knot 30, over the relabelled records.
usage: python census_kinlab2.py <shard.npz> <tracks.pt> <out.json> <tree_root>
"""
import hashlib
import importlib.util
import json
import sys
import time
import zipfile

import numpy as np

shard, tpath, outp, tree = sys.argv[1:5]
sys.path.insert(0, tree + "/stack")
sys.path.insert(0, tree + "/taniteval")
import torch  # noqa: E402
import tanitad  # noqa: E402
from tanitad.data import v8f_recovery as RV  # noqa: E402

assert tanitad.__file__.replace("\\", "/").startswith(tree + "/"), tanitad.__file__
GOLD_N = 4000


def read_member(zf, name, last_row_only=False, chunk=65536):
    with zf.open(name + ".npy") as fh:
        ver = np.lib.format.read_magic(fh)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(fh)
        assert not fortran
        n = shape[0]
        rest = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        if not last_row_only:
            return np.frombuffer(fh.read(n * rest * dtype.itemsize), dtype).reshape(shape).copy()
        out = np.empty((n,) + tuple(shape[2:]), dtype)
        per = rest * dtype.itemsize
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            blk = np.frombuffer(fh.read((b - a) * per), dtype).reshape((b - a,) + tuple(shape[1:]))
            out[a:b] = blk[:, -1]
        return out


def main():
    t0 = time.time()
    h = hashlib.sha256()
    with open(shard, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    zf = zipfile.ZipFile(shard)
    hdr = json.loads(bytes(read_member(zf, "header_json")).decode("utf-8"))
    ci, rr, reason, tick = (read_member(zf, k) for k in ("ci", "r", "reason", "tick"))
    p10_last = read_member(zf, "p10f", last_row_only=True)                  # [N, 8]
    names = {int(k): v for k, v in hdr["clips"].items()}
    clip_names = np.array([names.get(i, "") for i in range(max(names) + 1)], dtype=object)
    tracks = {c: t.double().numpy() for c, t in torch.load(tpath, map_location="cpu", weights_only=True)["tracks"].items()}
    read_clip = lambda name: {"pose": tracks[name][0::2]}                    # noqa: E731  the logged 0.2 s grid (x, y, yaw, v)
    arrays = {"reason": reason, "r": rr, "tick": tick, "p10f": p10_last[:, None, :]}
    ok = reason == 0
    n_ok = int(ok.sum())
    print("loaded", len(ci), "records, ok", n_ok, round(time.time() - t0, 1), "s", flush=True)
    log = lambda m: print(m, round(time.time() - t0, 1), "s", flush=True)    # noqa: E731
    res = {"inputs": {"shard_sha256": h.hexdigest(), "tracks": tpath, "n_records_shard": int(len(ci)), "n_ok": n_ok, "tree": tree,
                      "header_keys": {k: hdr[k] for k in hdr if k in ("readout", "acc_gain", "path_exec", "kappa_max", "schema", "split")}}}
    # ---- 1. gold on real records
    spec = importlib.util.spec_from_file_location("golden_rv", tree + "/stack/tests/golden_v8f_recovery_df250a00.py")
    gr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr)
    sel = np.flatnonzero(ok)[:GOLD_N]
    sub = {k: v[sel] for k, v in arrays.items()}
    sub["reason"] = np.zeros(len(sel), reason.dtype)
    sci = ci[sel]
    new, old = RV.label_bank(sub, sci, clip_names, read_clip), gr.label_bank(sub, sci, clip_names, read_clip)
    gold = {"n_records": int(len(sel)), "speed_v1_slot_equal": bool(np.array_equal(new["slot"], old["slot"])), "speed_v1_label_equal": bool(np.array_equal(new["label"], old["label"])),
            "speed_v1_status_counts_equal": new["status_counts"] == old["status_counts"], "speed_v1_census_equal": new["census"] == old["census"], "n_speed_labels": int(len(new["label"]))}
    eq_h, n_h = 0, 0
    for i in sel[:1000]:
        g = np.asarray(read_clip(str(clip_names[int(ci[i])]))["pose"], np.float64)[:, :3]
        eq_h += int(np.array_equal(RV.human_label(g, int(rr[i])), gr.human_label(g, int(rr[i]))))
        n_h += 1
    gold["human_label_equal_of"] = [eq_h, n_h]
    res["gold_real"] = gold
    print("GOLD", json.dumps(gold), flush=True)
    assert all(v is True for k, v in gold.items() if k.endswith("_equal")) and eq_h == n_h, "the gold check on real records FAILED"
    # ---- 2. the kinematic label over every ok record
    t1 = time.time()
    kb = RV.kin_label_bank(arrays, ci, clip_names, read_clip, log=log)
    res["kin_v1"] = {"status_counts": kb["status_counts"], "n_labelled": kb["n_labelled"], "n_slots": int((kb["slot"] >= 0).sum()), "n_kin": int(kb["is_kin"].sum()), "census": kb["census"], "wall_s": round(time.time() - t1, 1)}
    # reachability on real records: the kin labels differ from the speed_v1 label of the same record
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps({"counts": kb["status_counts"], "census": kb["census"]}, indent=1))


if __name__ == "__main__":
    main()
