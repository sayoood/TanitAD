"""NAVRBD feasibility probe: RB windows / episodes of the 10k data-scale clips by rbdist's rule. CPU, reads inputs, writes only out/."""
import sys, json, hashlib, numpy as np, pyarrow.parquet as pq
S1 = "F:/TanitAD-scratch"
sys.path.insert(0, S1 + "/datascale/navtok")
import navtok as N, build_table as B
IN = S1 + "/strat1/navrbd/in"
JP = sys.argv[1] if len(sys.argv) > 1 else IN + "/join_X.npz"; TAG = sys.argv[2] if len(sys.argv) > 2 else "RB10K"
J = dict(np.load(JP, allow_pickle=False))
clips = [str(c) for c in J["clip"]]
want = set(clips)
t = pq.read_table(IN + "/navigation.parquet").to_pandas()
t = t[t.clip_id.isin(want)].sort_values(["clip_id", "timestamp"], kind="stable")
texts = sorted(set(t.nav_text))
tix = {s: i for i, s in enumerate(texts)}
parsed = [N.parse_full(s) for s in texts]
cls_id = np.array([N.CLS_ID[p.cls] for p in parsed], np.int16)
dist = np.array([np.nan if p.dist_m is None else p.dist_m for p in parsed], np.float32)
exn = np.array([-1 if p.exit_n is None else p.exit_n for p in parsed], np.int32)
cid = t.clip_id.to_numpy(); ts = t.timestamp.to_numpy().astype(np.int64); ti = t.nav_text.map(tix).to_numpy().astype(np.int32)
d = {}
starts = np.r_[0, np.flatnonzero(cid[1:] != cid[:-1]) + 1, len(cid)]
for a, b in zip(starts[:-1], starts[1:]): d[str(cid[a])] = (int(a), int(b))
nav = {"clips": d, "ts": ts, "ti": ti, "cls_id": cls_id, "dist": dist, "texts": texts}
T = B.build(J, nav)
vocab = [str(s) for s in N.VOCAB]
isrb = np.array([c.startswith(("RB_", "ROTARY")) for c in vocab])
c = T["cls_id"]; fl = T["flag"]
rb = (fl == 0) & np.where(c >= 0, isrb[np.clip(c, 0, None)], False)
lab = J["has"].astype(bool) & (J["lat_cls"] >= 0)
off = np.repeat(J["off"], J["n"]); idx = np.arange(len(c)) - off
epi = np.repeat(np.arange(len(J["n"])), J["n"])
res = {"n_clips": len(clips), "clips_with_nav_rows": len(d), "clips_without_nav": len(clips) - len(d),
       "steps": int(len(c)), "flags": {str(i): int((fl == i).sum()) for i in range(4)}}
for name, m in (("rb_steps_all", rb), ("rb_labelled", rb & lab), ("rb_labelled_t3", rb & lab & (idx >= 3)), ("rb_labelled_t3_stride2", rb & lab & (idx >= 3) & (idx % 2 == 0)), ("rb_labelled_grid_stride4_t3", rb & lab & (idx % 4 == 3))):
    res[name] = {"windows": int(m.sum()), "episodes": int(len(set(epi[m])))}
m = rb & lab & (idx >= 3)
ex = np.where(T["text_i"] >= 0, exn[np.clip(T["text_i"], 0, None)], -1)
res["exit_cur_counts_labelled_t3"] = {str(k): int(((ex == k) & m).sum()) for k in range(-1, 10)}
res["label_counts_labelled_t3"] = np.bincount(J["lat_cls"][m].astype(int), minlength=5).tolist()
res["dist_valid_share_labelled_t3"] = float(np.isfinite(T["dist_m"][m]).mean())
tm = m & np.isin(J["lat_cls"], (1, 2)); res["turn_windows"] = {"windows": int(tm.sum()), "episodes": int(len(set(epi[tm])))}
res["has_labelled_steps_total"] = int(lab.sum())
import os; os.makedirs(S1 + "/strat1/navrbd/out", exist_ok=True)
json.dump(res, open(S1 + "/strat1/navrbd/out/" + TAG + "_COUNTS.json", "w"), indent=1)
print(json.dumps(res, indent=1))
