"""D3 check 1f -- the start-aligned pairs (tau < 0.5 s, score < 0.1): are they the SAME VIDEO under two clip ids?  NCC of B frame r against A frame r (and r+1), r = 0, 50, 100, 150."""
import hashlib, io, json, numpy as np, torch
from PIL import Image
OUT = "/home/nvidia/refcv8_audit/D3/run2"
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
train = [l.strip().replace(".v2ep.pt", "") for l in open("/home/nvidia/data/refcv6_train_clips.txt") if l.strip()]
ev = [l.strip().replace(".v2ep.pt", "") for l in open("/home/nvidia/data/eval139_ids.txt") if l.strip()]
ids = train + ev; evs = set(ev)
P = np.load(f"{OUT}/c1_pairs_lt6.npz"); b, a, tau, sc = P["b"], P["a"], P["tau"] / 10.0, P["score"]
sel = np.where((sc < 0.1) & (tau < 0.5))[0]
def th(cid, rr):
    d = torch.load(f"{'/home/nvidia/data/refcv6-b1-416x1024-eval139' if cid in evs else '/home/nvidia/data/refcv6-b1-416x1024-train'}/{cid}.v2ep.pt", map_location="cpu", weights_only=False)
    buf = d["jpeg_buf"].numpy(); ln = d["jpeg_len"].numpy(); off = np.concatenate([[0], np.cumsum(ln)])
    return {r: (np.asarray(Image.open(io.BytesIO(buf[off[r]:off[r + 1]].tobytes())).convert("L").resize((128, 52), Image.BILINEAR), np.float32)[12:44], buf[off[r]:off[r + 1]].tobytes()) for r in rr if r < len(ln)}
def ncc(x, y):
    x = (x - x.mean()) / (x.std() + 1e-6); y = (y - y.mean()) / (y.std() + 1e-6); return float((x * y).mean())
out = []
for k in sel:
    B, A = ids[b[k]], ids[a[k]]
    rr = [0, 1, 50, 51, 100, 101, 150, 151]
    tb, ta = th(B, rr), th(A, rr)
    res = {}
    for r in (0, 50, 100, 150):
        if r in tb and r in ta:
            res[r] = {"ncc_same_row": round(ncc(tb[r][0], ta[r][0]), 3), "ncc_best_pm1": round(max(ncc(tb[r][0], ta[q][0]) for q in (r, r + 1) if q in ta), 3), "png_bytes_identical": bool(tb[r][1] == ta[r][1])}
    out.append({"B": sha12(B), "A": sha12(A), "tau_s": float(tau[k]), "score": float(sc[k]), "rows": res})
json.dump(out, open(f"{OUT}/c1f_startaligned_image.json", "w"), indent=1); print(json.dumps(out, indent=1))
