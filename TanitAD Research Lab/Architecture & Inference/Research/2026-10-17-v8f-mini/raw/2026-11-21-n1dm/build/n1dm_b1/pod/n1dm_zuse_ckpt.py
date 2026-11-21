# n1dm_zuse_ckpt.py MODE SRC DST -- PREREG_V8F_N1DM section 4 controls Z-USE and Z-UNIF (beside, no bar). Adapted from n1dh_zuse_ckpt.py (which ran).
#   MODE zuse : a copy of the trained N1DM checkpoint with every model['route_br.k.weight'] (k = 0, 1, 2) set to 0. The branches have no bias, so coef + sum_k p_k * 0 = coef: the plan is head_o's alone, the exact "branch residual removed"
#               ablation (the N1DH Z-USE analogue: real minus posterior-zeroed).
#   MODE unif : a copy with route_head.weight and route_head.bias set to 0: the logits are 0, softmax is uniform (1/3 on every row), so the plan is head_o + (1/3) sum_k branch_k(h): ONE shared mixture, no per-row selection.
#               real minus unif reads what the per-row selection adds on top of the extra branch capacity.
# Every other tensor and key is kept. Refuses if a key is absent, if a branch has a bias, if route_emb is present (not a modes checkpoint), if the weights to be zeroed are already all zero (a branch / head that never moved).
import sys, hashlib, torch
mode, src, dst = sys.argv[1:4]
if mode not in ("zuse", "unif"):
    sys.exit(f"REFUSED: mode {mode!r} is not zuse or unif")
c = torch.load(src, map_location="cpu", weights_only=False)
m = c["model"]
if any(k.startswith("route_emb.") for k in m):
    sys.exit("REFUSED: route_emb keys present (not an N1DM checkpoint)")
if "route_head.weight" not in m or "route_head.bias" not in m:
    sys.exit("REFUSED: route_head.weight / bias not in the checkpoint's model state")
br = [f"route_br.{k}.weight" for k in range(3)]
if any(k not in m for k in br) or any(k.startswith("route_br.") and k.endswith(".bias") for k in m) or len([k for k in m if k.startswith("route_br.")]) != 3:
    sys.exit("REFUSED: the checkpoint does not hold exactly route_br.0/1/2.weight (no bias)")
zero_keys = br if mode == "zuse" else ["route_head.weight", "route_head.bias"]
l1 = {k: float(m[k].abs().sum()) for k in zero_keys}
if any(v == 0.0 for v in l1.values()):
    sys.exit(f"REFUSED: a tensor to zero is already all zero in the source: {l1}")
for k in zero_keys:
    m[k] = torch.zeros_like(m[k])
others = [n for n in m if n not in zero_keys]
torch.save(c, dst)
c2 = torch.load(dst, map_location="cpu", weights_only=False)
assert all(float(c2["model"][k].abs().sum()) == 0.0 for k in zero_keys)
ref = torch.load(src, map_location="cpu", weights_only=False)["model"]
bad = [n for n in others if not torch.equal(ref[n], c2["model"][n])]
if bad:
    sys.exit(f"REFUSED: {len(bad)} other tensors differ, e.g. {bad[:3]}")
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
print(f"{mode.upper()} copy OK zeroed={zero_keys} src_l1={ {k: round(v, 6) for k, v in l1.items()} } other_tensors_equal={len(others)} dst_sha256={h}")
