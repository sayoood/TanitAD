"""D3 check 1d -- independent IMAGE channel for the pairs whose 20 s windows share wall-clock time (0.5 <= tau < 19 s, score < 0.1).
For pair (B window inside A's log at tau): B's raw frame 0 must reappear in A's video.  We do NOT trust the row formula: sweep ALL 201 raw frames of A and
report the best 128x52 grey-thumbnail NCC (sky rows 0-11 and the bottom 8 rows dropped) and the row where it peaks; the predicted row from the clock sidecar is
reported next to it.  NULL: the same sweep of B's frame 0 against a random other clip's 201 frames (max over frames).  Ids sha12."""
import hashlib, io, json, sys
import numpy as np, torch
from PIL import Image
OUT = "/home/nvidia/refcv8_audit/D3/run2"
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
train = [l.strip().replace(".v2ep.pt", "") for l in open("/home/nvidia/data/refcv6_train_clips.txt") if l.strip()]
ev = [l.strip().replace(".v2ep.pt", "") for l in open("/home/nvidia/data/eval139_ids.txt") if l.strip()]
ids = train + ev; evs = set(ev)
P = np.load(f"{OUT}/c1_pairs_lt6.npz")
b, a, tau, sc = P["b"], P["a"], P["tau"] / 10.0, P["score"]
sel = np.where((sc < 0.1) & (tau >= 0.5) & (tau < 19.0))[0]
side = {}
for l in open("/home/nvidia/data/refcv6_clip_clock_sidecar.jsonl"):
    r = json.loads(l); side[int(r["sid"])] = r
mt = torch.load("/home/nvidia/data/refcv6-b1-416x1024-train/_v2manifest.pt", map_location="cpu", weights_only=False)
me = torch.load("/home/nvidia/data/refcv6-b1-416x1024-eval139/_v2manifest.pt", map_location="cpu", weights_only=False)
uid = {c: int(u) for c, u in zip(list(mt["clip_id"]) + list(me["clip_id"]), list(mt["episode_uid"]) + list(me["episode_uid"]))}
def gsdt(c):
    r = side.get(uid[c]); return (float(r["grid_start_s"]), float(r["dt_s"])) if r else (0.113, 0.100667)
def thumbs(cid, rows=None):
    d = torch.load(f"{'/home/nvidia/data/refcv6-b1-416x1024-eval139' if cid in evs else '/home/nvidia/data/refcv6-b1-416x1024-train'}/{cid}.v2ep.pt", map_location="cpu", weights_only=False)
    buf = d["jpeg_buf"].numpy(); ln = d["jpeg_len"].numpy(); off = np.concatenate([[0], np.cumsum(ln)])
    rr = range(len(ln)) if rows is None else rows
    out = {}
    for r in rr:
        im = Image.open(io.BytesIO(buf[off[r]:off[r + 1]].tobytes())).convert("L").resize((128, 52), Image.BILINEAR)
        out[r] = np.asarray(im, np.float32)[12:44]
    return out
def ncc(x, y):
    x = (x - x.mean()) / (x.std() + 1e-6); y = (y - y.mean()) / (y.std() + 1e-6); return float((x * y).mean())
rng = np.random.default_rng(2)
rows, nulls = [], []
for k in sel:
    B, A = ids[b[k]], ids[a[k]]
    tb = thumbs(B, [0])[0]; ta = thumbs(A)
    sw = {r: ncc(tb, v) for r, v in ta.items()}
    top = sorted(sw.items(), key=lambda kv: -kv[1])[:3]
    gB, dB = gsdt(B); gA, dA = gsdt(A)
    pred = (tau[k] + gB - gA) / dA
    rows.append({"B": sha12(B), "A": sha12(A), "tau_s": float(tau[k]), "score": float(sc[k]), "pred_row": round(float(pred), 1), "best_row": int(top[0][0]), "best_ncc": round(top[0][1], 3),
                 "second_ncc": round(top[1][1], 3), "ncc_at_pred_row_pm1": round(max(sw.get(int(round(pred)) + d, -9) for d in (-1, 0, 1)), 3), "peak_to_second": round(top[0][1] - top[1][1], 3)})
    a2 = ids[int(rng.integers(len(train)))]
    t2 = thumbs(a2)
    nulls.append(max(ncc(tb, v) for v in t2.values()))
    print(rows[-1], flush=True)
res = {"n_pairs": len(rows), "rows": rows, "null_max_ncc_random_clip": {"median": float(np.median(nulls)), "max": float(max(nulls)), "n": len(nulls)},
       "n_best_ncc_gt_null_max": int(sum(r["best_ncc"] > max(nulls) for r in rows)), "n_best_row_within_3_of_pred": int(sum(abs(r["best_row"] - r["pred_row"]) <= 3 for r in rows))}
json.dump(res, open(f"{OUT}/c1d_image_sweep.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "rows"})
