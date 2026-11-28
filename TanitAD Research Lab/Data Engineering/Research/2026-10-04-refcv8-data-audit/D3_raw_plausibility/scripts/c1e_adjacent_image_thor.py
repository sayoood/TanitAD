"""D3 check 1e -- image channel for ADJACENT pairs (19.9 <= tau <= 20.4 s, score < 0.1): B starts when A's 20 s clip ends, so A's LAST frames and B's FIRST frame are
~0.1-0.3 s apart in wall-clock and must show the same scene.  NCC of 128x32 grey thumbnails (sky and hood rows cropped), best over A's last 4 rows, versus a NULL of the same
statistic for B's first frame against the last 4 frames of a random other clip.  Ids sha12."""
import hashlib, io, json, numpy as np, torch
from PIL import Image
OUT = "/home/nvidia/refcv8_audit/D3/run2"
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
train = [l.strip().replace(".v2ep.pt", "") for l in open("/home/nvidia/data/refcv6_train_clips.txt") if l.strip()]
ev = [l.strip().replace(".v2ep.pt", "") for l in open("/home/nvidia/data/eval139_ids.txt") if l.strip()]
ids = train + ev; evs = set(ev)
P = np.load(f"{OUT}/c1_pairs_lt6.npz"); b, a, tau, sc = P["b"], P["a"], P["tau"] / 10.0, P["score"]
sel = np.where((sc < 0.1) & (tau >= 19.9) & (tau <= 20.4))[0]
def rows(cid, which):
    d = torch.load(f"{'/home/nvidia/data/refcv6-b1-416x1024-eval139' if cid in evs else '/home/nvidia/data/refcv6-b1-416x1024-train'}/{cid}.v2ep.pt", map_location="cpu", weights_only=False)
    buf = d["jpeg_buf"].numpy(); ln = d["jpeg_len"].numpy(); off = np.concatenate([[0], np.cumsum(ln)]); n = len(ln)
    rr = [0] if which == "first" else list(range(n - 4, n))
    return [np.asarray(Image.open(io.BytesIO(buf[off[r]:off[r + 1]].tobytes())).convert("L").resize((128, 52), Image.BILINEAR), np.float32)[12:44] for r in rr]
def ncc(x, y):
    x = (x - x.mean()) / (x.std() + 1e-6); y = (y - y.mean()) / (y.std() + 1e-6); return float((x * y).mean())
rng = np.random.default_rng(4)
res, nul = [], []
for k in sel:
    B, A = ids[b[k]], ids[a[k]]
    fb = rows(B, "first")[0]; la = rows(A, "last")
    res.append({"B": sha12(B), "A": sha12(A), "B_split": "eval" if B in evs else "train", "A_split": "eval" if A in evs else "train", "tau_s": float(tau[k]), "score": float(sc[k]), "ncc_best_last4": round(max(ncc(fb, f) for f in la), 3)})
    A2 = ids[int(rng.integers(len(train)))]
    nul.append(max(ncc(fb, f) for f in rows(A2, "last")))
out = {"n_pairs": len(res), "rows": res, "null_ncc": {"n": len(nul), "median": float(np.median(nul)), "max": float(max(nul)), "q90": float(np.quantile(nul, .9))},
       "n_pairs_ncc_gt_null_max": int(sum(r["ncc_best_last4"] > max(nul) for r in res))}
json.dump(out, open(f"{OUT}/c1e_adjacent_image.json", "w"), indent=1); print(json.dumps(out, indent=1))
