"""GEO10K step 3b: the lead pass (build_v9_labels.cmd_lead_pass logic, same functions) over the per-shard obstacle joins of
keep10k, whose schema is {clip_id, frame_idx (EPISODE index, = manifest row), t_s, agents}. The old lead pass keys frames by the
RAW v2ep frame k (= row + 2); LEAD_OFF = 2 maps one to the other. The offset is MEASURED here, not assumed: for every clip the
join's own t_s is compared with g0 + (frame_idx + OFF) * dt of the clip clock; the offset with the smaller residual is reported."""
import glob, gzip, json, os, sys, time
import numpy as np
import torch
from multiprocessing import Pool
sys.path.insert(0, "/root/geo10k/code")
import geo_common as G
G.point_ego_lookup()
G.ES._index()
import build_v9_labels as B

OFF = int(os.environ.get("LEAD_OFF", "2"))
SHARDS = sorted(glob.glob("/root/datascale/keep10k/*/join.jsonl.gz"))
ONLY = os.environ.get("LEAD_SHARDS")            # comma list of shard names for a test run
if ONLY:
    SHARDS = [s for s in SHARDS if os.path.basename(os.path.dirname(s)) in ONLY.split(",")]
MAN = torch.load(f"{G.OUT}/new10k_v2manifest.pt", weights_only=False)
IDX = {c: i for i, c in enumerate(MAN["clip_id"])}
SIDE = B._read_sidecar(f"{G.OUT}/clock_new10k.jsonl")


def groups(path):
    cur, fr = None, []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["clip_id"] != cur:
                if cur is not None:
                    yield cur, fr
                cur, fr = r["clip_id"], []
            fr.append(r)
    if cur is not None:
        yield cur, fr


def do_shard(path):
    out = {"sha12": [], "k": [], "has": [], "lead": [], "gap": [], "tg": [], "s": [], "v": []}
    resid = []
    n = 0
    for cid, fr in groups(path):
        if cid not in IDX:
            continue
        i = IDX[cid]
        sid = B.stable_episode_id(cid)
        g0, dt, src = B.clip_clock(sid, SIDE, MAN["poses"][i].double().numpy())
        ts = np.array([r["t_s"] for r in fr]); fi = np.array([r["frame_idx"] for r in fr])
        resid.append({"clip": cid, "src": src, "r0": float(np.median(np.abs(ts - (g0 + fi * dt)))), "r2": float(np.median(np.abs(ts - (g0 + (fi + 2) * dt))))})
        log = B.load_log(f"{G.FLAT}/{cid}.parquet")
        tr = B.track10(log)
        p0 = B.smooth_path(tr, 0.0)
        s12 = B.sha12(cid)
        for r in sorted(fr, key=lambda r: r["frame_idx"]):
            kk = int(r["frame_idx"]) + OFF
            t = g0 + kk * dt
            if t < 0 or t > log.t_end:
                continue
            L, gap, tg, s, v = B.lead_for_frame(log, tr, p0, t, r["agents"])
            for key, val in (("sha12", s12), ("k", kk), ("has", 1), ("lead", int(L)), ("gap", gap), ("tg", tg), ("s", s), ("v", v)):
                out[key].append(val)
        n += 1
    return os.path.basename(os.path.dirname(path)), out, resid, n


if __name__ == "__main__":
    t0 = time.time()
    print(f"[g3b] shards {len(SHARDS)} OFF={OFF}", flush=True)
    with Pool(min(24, len(SHARDS))) as pool:
        res = pool.map(do_shard, SHARDS, chunksize=1)
    cat = {k: [] for k in ("sha12", "k", "has", "lead", "gap", "tg", "s", "v")}
    allres, ncl = [], 0
    for sh, out, resid, n in res:
        for k in cat:
            cat[k] += out[k]
        allres += resid
        ncl += n
    r0 = np.array([x["r0"] for x in allres]); r2 = np.array([x["r2"] for x in allres])
    rep = {"off_used": OFF, "n_clips": ncl, "median_abs_resid_s_off0": float(np.median(r0)), "median_abs_resid_s_off2": float(np.median(r2)),
           "frac_clips_off2_better": float((r2 < r0).mean()), "n_frames": len(cat["k"]), "src_counts": {s: sum(1 for x in allres if x["src"] == s) for s in ("sidecar", "pose_dt", "nominal_dt")}}
    print("[g3b]", json.dumps(rep), flush=True)
    name = "lead_new10k.npz" if not ONLY else "lead_test.npz"
    np.savez_compressed(f"{G.OUT}/{name}", **{k: np.asarray(v) for k, v in cat.items()},
                        meta=np.asarray(json.dumps({"n_clips": ncl, "builder": "build_v9_labels lead functions, cdbff971; join keep10k per shard; frame = frame_idx + %d" % OFF})))
    json.dump(rep, open(f"{G.OUT}/lead_report{'' if not ONLY else '_test'}.json", "w"), indent=1)
    print(f"[g3b] done {time.time()-t0:.0f}s")
    print("ZZ GEO10K-G3B DONE ZZ")
