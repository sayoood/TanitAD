# n1dh_zuse_ckpt.py SRC DST -- PREREG_V8F_N1DH section 4 "Z-USE" (beside, no bar): the posterior zeroed at inference. Adapted from n1dc_zuse_ckpt.py (which ran).
# Copies a trained N1DH checkpoint with model['route_emb.weight'] set to 0. route_emb is nn.Linear(3, n_q*d, bias=False), so h + route_emb(p) = h + 0.0: the exact "output x 0" ablation, as N1DC's o_code copy.
# route_head (the decision head) is left as trained. Every other tensor and every other key is kept.
# Refuses if the key is absent, if route_emb has a bias, if route_head.weight is absent (not an N1DH checkpoint), or if the source weight is already all zero (a conditioning that never moved).
import sys, hashlib, torch
src, dst = sys.argv[1], sys.argv[2]
c = torch.load(src, map_location="cpu", weights_only=False)
m = c["model"]
k = "route_emb.weight"
if k not in m:
    sys.exit(f"REFUSED: {k} not in the checkpoint's model state")
if "route_emb.bias" in m:
    sys.exit("REFUSED: route_emb has a bias; zeroing the weight would not zero the output")
if "route_head.weight" not in m:
    sys.exit("REFUSED: route_head.weight not in the checkpoint's model state (not an N1DH checkpoint)")
w = m[k]
l1 = float(w.abs().sum())
if l1 == 0.0:
    sys.exit("REFUSED: route_emb.weight is already all zero in the source")
m[k] = torch.zeros_like(w)
others = [n for n in m if n != k]
torch.save(c, dst)
c2 = torch.load(dst, map_location="cpu", weights_only=False)
assert float(c2["model"][k].abs().sum()) == 0.0
ref = torch.load(src, map_location="cpu", weights_only=False)["model"]
bad = [n for n in others if not torch.equal(ref[n], c2["model"][n])]
if bad:
    sys.exit(f"REFUSED: {len(bad)} other tensors differ, e.g. {bad[:3]}")
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
print(f"ZUSE copy OK src_route_emb_l1={l1:.6f} shape={tuple(w.shape)} other_tensors_equal={len(others)} dst_sha256={h}")
