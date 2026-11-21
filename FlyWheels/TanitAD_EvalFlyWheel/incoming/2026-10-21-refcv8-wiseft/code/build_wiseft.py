#!/usr/bin/env python3
"""R8-WISEFT builder: weight-space interpolation between refcv8's INIT (refcv7-r101-s0 ckpt.pt, step 50,400 -- the exact
--init-from of refcv8-r101-s0) and refcv8's EMA-48,852, theta_lam = (1 - lam) * theta7 + lam * theta8 on every SHARED floating
key of identical shape (computed in fp32, cast back to theta8's dtype).  Keys refcv7 does not have (refcv8's 34 new tensors) and
non-float / shape-mismatched entries are taken from theta8 unchanged.  (1-lam)*a + lam*b is used, never a + lam*(b-a), so that
lam = 1 reproduces theta8 and lam = 0 reproduces theta7 on shared keys BIT-EXACTLY -- both are asserted (controls C1, C0).

    python build_wiseft.py --theta7 <refcv7 ckpt.pt> --theta8 <ckpt_ema_48852.pt> --lams 0,0.25,0.5,0.75 --out-dir <dir>
Writes <out-dir>/ckpt_wiseft_l<LAM>.pt (strict-loadable: same 1,165 keys as theta8) + WISEFT_BUILD.json (counts, md5s, controls).
"""
import argparse, hashlib, json, os, sys, time
import torch


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def mix(a, b, lam):
    out = ((1.0 - lam) * a.float() + lam * b.float()).to(b.dtype)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--theta7", required=True); ap.add_argument("--theta8", required=True)
ap.add_argument("--lams", default="0,0.25,0.5,0.75"); ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
t0 = time.time()
d7 = torch.load(a.theta7, map_location="cpu", weights_only=False); d8 = torch.load(a.theta8, map_location="cpu", weights_only=False)
m7, m8 = d7["model"], d8["model"]
cls = {"interp": [], "new_in_r8": [], "nonfloat": [], "shape_mismatch": []}
for k, v in m8.items():
    if k not in m7: cls["new_in_r8"].append(k)
    elif not (torch.is_tensor(v) and v.is_floating_point()): cls["nonfloat"].append(k)
    elif tuple(m7[k].shape) != tuple(v.shape): cls["shape_mismatch"].append(k)
    else: cls["interp"].append(k)
only7 = sorted(set(m7) - set(m8))
rep = {"id": "R8-WISEFT", "theta7": {"path": a.theta7, "md5": md5(a.theta7), "step": d7.get("step"), "n_keys": len(m7)},
       "theta8": {"path": a.theta8, "md5": md5(a.theta8), "step": d8.get("step"), "kind": d8.get("kind"), "n_keys": len(m8)},
       "counts": {k: len(v) for k, v in cls.items()}, "keys_only_in_theta7": only7,
       "numel": {k: int(sum(m8[x].numel() for x in v if torch.is_tensor(m8[x]))) for k, v in cls.items()},
       "new_in_r8": cls["new_in_r8"], "shape_mismatch": cls["shape_mismatch"], "nonfloat_sample": cls["nonfloat"][:20],
       "outputs": {}, "controls": {}}
assert len(m8) == 1165 and len(cls["new_in_r8"]) == 34, (len(m8), len(cls["new_in_r8"]))
# controls on the ARITHMETIC before anything is written: lam = 1 reproduces theta8 exactly, lam = 0 reproduces theta7 on shared keys
c1 = all(torch.equal(mix(m7[k], m8[k], 1.0), m8[k]) for k in cls["interp"])
c0 = all(torch.equal(mix(m7[k], m8[k], 0.0), m7[k].to(m8[k].dtype)) for k in cls["interp"])
rep["controls"] = {"C1_lam1_equals_theta8_bitexact": c1, "C0_lam0_equals_theta7_on_shared_bitexact": c0}
if not (c1 and c0):
    json.dump(rep, open(os.path.join(a.out_dir, "WISEFT_BUILD.json"), "w"), indent=1); sys.exit("ZZREFUSE_wiseft_controls")
for lam in [float(x) for x in a.lams.split(",")]:
    sd = {}
    for k, v in m8.items():
        sd[k] = mix(m7[k], v, lam) if k in cls["interp"] else v
    assert list(sd.keys()) == list(m8.keys())
    # per-lambda read-back control: relative distance to theta8 over interpolated keys must equal (1 - lam) * |theta7 - theta8|
    num = sum(float((sd[k].float() - m8[k].float()).norm() ** 2) for k in cls["interp"]) ** 0.5
    den = sum(float((m7[k].float() - m8[k].float()).norm() ** 2) for k in cls["interp"]) ** 0.5
    tag = f"{lam:.2f}".replace(".", "p")
    p = os.path.join(a.out_dir, f"ckpt_wiseft_l{tag}.pt")
    out = {"model": sd, "step": d8.get("step"), "ema_decay": d8.get("ema_decay"), "num_updates": d8.get("num_updates"),
           "kind": f"wiseft_lambda_{lam}", "wiseft": {"lambda": lam, "theta7_md5": rep["theta7"]["md5"], "theta8_md5": rep["theta8"]["md5"]}}
    torch.save(out, p)
    back = torch.load(p, map_location="cpu", weights_only=False)["model"]
    rb = all(torch.equal(back[k], sd[k]) for k in sd)
    rep["outputs"][tag] = {"lambda": lam, "path": p, "md5": md5(p), "readback_bitexact": rb,
                           "dist_to_theta8_over_dist78": (num / den) if den else None, "expected": 1.0 - lam}
    if not rb or (den and abs(num / den - (1.0 - lam)) > 1e-3): rep["controls"][f"FAIL_{tag}"] = True
rep["controls"]["all_pass"] = c1 and c0 and not any(k.startswith("FAIL_") for k in rep["controls"])
rep["wall_s"] = round(time.time() - t0, 1)
json.dump(rep, open(os.path.join(a.out_dir, "WISEFT_BUILD.json"), "w"), indent=1)
print("ZZWISEBUILT" if rep["controls"]["all_pass"] else "ZZREFUSE_wiseft_outputs", json.dumps(rep["counts"]), json.dumps({t: o["md5"] for t, o in rep["outputs"].items()}))
