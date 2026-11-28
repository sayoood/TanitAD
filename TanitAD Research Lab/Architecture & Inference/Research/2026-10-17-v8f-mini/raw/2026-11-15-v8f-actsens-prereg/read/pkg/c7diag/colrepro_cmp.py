"""COLREPRO compare: new latents file vs the banked phase-1 one, key by key. Exact equality is the bar (max_abs_diff 0.0, same shapes, same clip order)."""
import json, sys
import numpy as np, torch

new_p, ref_p, out_p = sys.argv[1:4]
a = torch.load(new_p, weights_only=False)
b = torch.load(ref_p, weights_only=False)
res = {"new": new_p, "ref": ref_p, "keys": {}}


def cmp(x, y):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        x = torch.as_tensor(np.asarray(x)) if isinstance(x, np.ndarray) else x
        y = torch.as_tensor(np.asarray(y)) if isinstance(y, np.ndarray) else y
        if tuple(x.shape) != tuple(y.shape):
            return {"equal": False, "shape": [list(x.shape), list(y.shape)]}
        if x.dtype == torch.bool or not x.is_floating_point():
            return {"equal": bool(torch.equal(x, y)), "shape": list(x.shape)}
        d = (x.double() - y.double()).abs()
        return {"equal": bool(torch.equal(x, y)), "shape": list(x.shape), "max_abs_diff": float(d.max()) if d.numel() else 0.0,
                "n_diff": int((d > 0).sum())}
    return {"equal": x == y}


for k in ("A_raw", "ok", "t", "eid", "N", "clips", "clips_sha256", "starts"):
    res["keys"][k] = cmp(a[k], b[k]) if k in b else {"missing_in_ref": True}
zk = "R35_s0"
res["keys"]["Z1"] = cmp(a["Z1"][zk], b["Z1"][zk])
res["ctrl4_new"] = {zk: a["ctrl4"][zk]["max_abs_diff"]}
res["tanitad_file_new"] = a.get("tanitad_file"); res["tanitad_file_ref"] = b.get("tanitad_file")
res["COLREPRO_EXACT"] = all(v.get("equal") is True for v in res["keys"].values())
json.dump(res, open(out_p, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
