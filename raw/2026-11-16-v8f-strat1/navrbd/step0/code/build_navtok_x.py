"""NAVRBD step 0, input 1: the navtok table of a join (the 10k, or eval141 as the proof), by the UNEDITED navtok/build_table.build, exactly as probe/count_rb10k.py does in memory.
usage: python build_navtok_x.py JOIN.npz PARQUET OUT.npz [--compare PINNED_navtok_table.npz]   (writes eval_* keys + vocab; with --compare also asserts the arrays equal the pinned table's eval split bit for bit)
CPU only; reads inputs, writes OUT and OUT.json."""
import sys, json, hashlib
import numpy as np
import pyarrow.parquet as pq
S1 = "F:/TanitAD-scratch"
sys.path.insert(0, S1 + "/datascale/navtok")
import navtok as N, build_table as B


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    jp, pp, out = sys.argv[1:4]
    cmp = sys.argv[sys.argv.index("--compare") + 1] if "--compare" in sys.argv else None
    J = dict(np.load(jp, allow_pickle=False))
    clips = [str(c) for c in J["clip"]]
    want = set(clips)
    t = pq.read_table(pp).to_pandas()
    t = t[t.clip_id.isin(want)].sort_values(["clip_id", "timestamp"], kind="stable")
    texts = sorted(set(t.nav_text))
    tix = {s: i for i, s in enumerate(texts)}
    parsed = [N.parse_full(s) for s in texts]
    cls_id = np.array([N.CLS_ID[p.cls] for p in parsed], np.int16)
    dist = np.array([np.nan if p.dist_m is None else p.dist_m for p in parsed], np.float32)
    cid = t.clip_id.to_numpy(); ts = t.timestamp.to_numpy().astype(np.int64); ti = t.nav_text.map(tix).to_numpy().astype(np.int32)
    d = {}
    starts = np.r_[0, np.flatnonzero(cid[1:] != cid[:-1]) + 1, len(cid)]
    for a, b in zip(starts[:-1], starts[1:]):
        d[str(cid[a])] = (int(a), int(b))
    T = B.build(J, {"clips": d, "ts": ts, "ti": ti, "cls_id": cls_id, "dist": dist, "texts": texts})
    pack = {"eval_clip": J["clip"], "eval_off": J["off"], "eval_n": J["n"]}
    for k, v in T.items():
        pack["eval_" + k] = v
    pack["vocab"] = np.array(N.VOCAB)
    rep = {"join": jp, "join_sha256": sha(jp), "parquet_sha256": sha(pp), "clips": len(clips), "clips_with_nav_rows": len(d), "steps": int(len(T["flag"]))}
    if cmp:
        P = np.load(cmp, allow_pickle=False)
        eq = {k: bool(np.array_equal(pack["eval_" + k], P["eval_" + k], equal_nan=(k in ("dist_m", "age_s")))) for k in ("cls_id", "dist_m", "flag", "age_s")}
        eq["clip_off_n"] = all(np.array_equal(np.asarray(pack["eval_" + k]).astype(str) if k == "clip" else pack["eval_" + k], np.asarray(P["eval_" + k]).astype(str) if k == "clip" else P["eval_" + k]) for k in ("clip", "off", "n"))
        eq["vocab"] = bool(np.array_equal(pack["vocab"].astype(str), P["vocab"].astype(str)))
        rep["compare_pinned"] = {"pinned_sha256": sha(cmp), "bit_equal": eq, "all": all(eq.values())}
        print(json.dumps(rep["compare_pinned"]))
    np.savez_compressed(out, **pack)
    rep["out"] = out; rep["out_sha256"] = sha(out)
    json.dump(rep, open(out + ".json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
