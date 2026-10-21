#!/usr/bin/env python3
"""R8-WISEFT-DECLAM builder (registered before any of its arms is built, rolled or scored): the decoder-only blend at lambdas closer to refcv8.

R8-WISEFT-MOD2 (742fca0) read SUB-LOCUS = DEC: lambda 0.5 on core.decoder only (168 keys, 15.26 M) recovers all of theta_0.5's navtrain gain.
theta_0.5's only SHIP-R8 failure is E1 (short-horizon progress); if theta_DEC fails it too, the next arm moves the decoder lambda toward refcv8.
Arms (theta_lam = (1 - lam) theta7 + lam theta8 on the core.decoder. keys ONLY; everything else refcv8 EMA-48,852):
  DEC625  lambda 0.625      DEC75  lambda 0.75      DEC875  lambda 0.875
Controls: the lambda 0.5 decoder blend rebuilt here equals the banked theta_DEC (md5 dad4cb5b...) key by key; each arm equals mix(theta7, theta8,
lam) on decoder keys and theta8 elsewhere; each arm differs from theta8 on some decoder key; readback bit-exact.

    python build_wiseft_declam.py --theta7 <refcv7 ckpt.pt> --theta8 <ckpt_ema_48852.pt> --theta-dec <ckpt_wiseft_mod2_DEC.pt> --out-dir <dir>
"""
import argparse, hashlib, json, os, sys, time
import torch

PERC = ("core.encoder.", "_perception.", "_map_hires.", "core.agent_head.", "core.agent_embed.", "core.law_head.", "core.ego_hist.",
        "core.measurement.", "tac_latent_proj.")
DEC = ("core.decoder.",)
LAMS = {"DEC625": 0.625, "DEC75": 0.75, "DEC875": 0.875}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def mix(a, b, lam):
    return ((1.0 - lam) * a.float() + lam * b.float()).to(b.dtype)


ap = argparse.ArgumentParser()
ap.add_argument("--theta7", required=True); ap.add_argument("--theta8", required=True); ap.add_argument("--theta-dec", required=True)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
t0 = time.time()
d7 = torch.load(a.theta7, map_location="cpu", weights_only=False); d8 = torch.load(a.theta8, map_location="cpu", weights_only=False)
m7, m8 = d7["model"], d8["model"]
mD = torch.load(a.theta_dec, map_location="cpu", weights_only=False)["model"]
interp = [k for k, v in m8.items() if k in m7 and torch.is_tensor(v) and v.is_floating_point() and tuple(m7[k].shape) == tuple(v.shape)]
assert len(interp) == 1027, len(interp)
dec = [k for k in interp if not k.startswith(PERC) and k.startswith(DEC)]
assert len(dec) == 168, len(dec)
ds = set(dec)
rep = {"id": "R8-WISEFT-DECLAM", "arms": {g: {"lambda": lam, "n_keys": len(dec), "numel": int(sum(m8[k].numel() for k in dec))} for g, lam in LAMS.items()},
       "theta7_md5": md5(a.theta7), "theta8_md5": md5(a.theta8), "theta_dec_md5": md5(a.theta_dec), "outputs": {}, "controls": {}}
c = rep["controls"]
c["rebuilt_theta_dec_equals_banked"] = all(torch.equal(mix(m7[k], m8[k], 0.5), mD[k]) for k in dec) and all(torch.equal(mD[k], m8[k]) for k in m8 if k not in ds)
arms = {g: {k: (mix(m7[k], v, lam) if k in ds else v) for k, v in m8.items()} for g, lam in LAMS.items()}
c["arms_exact"] = all(torch.equal(arms[g][k], mix(m7[k], m8[k], lam)) for g, lam in LAMS.items() for k in dec)
c["outside_decoder_is_theta8"] = all(torch.equal(arms[g][k], m8[k]) for g in arms for k in m8 if k not in ds)
c["each_arm_changed"] = all(any(not torch.equal(arms[g][k], m8[k]) for k in dec) for g in arms)
if not all(c.values()):
    json.dump(rep, open(os.path.join(a.out_dir, "WISEFT_DECLAM_BUILD.json"), "w"), indent=1); sys.exit("ZZREFUSE_declam_controls")
for g, sd in arms.items():
    p = os.path.join(a.out_dir, f"ckpt_wiseft_declam_{g}.pt")
    torch.save({"model": sd, "step": d8.get("step"), "ema_decay": d8.get("ema_decay"), "num_updates": d8.get("num_updates"),
                "kind": f"wiseft_declam_{g}", "wiseft": {"lambda": LAMS[g], "group": "DEC", "theta7_md5": rep["theta7_md5"], "theta8_md5": rep["theta8_md5"]}}, p)
    back = torch.load(p, map_location="cpu", weights_only=False)["model"]
    rep["outputs"][g] = {"path": p, "md5": md5(p), "readback_bitexact": all(torch.equal(back[k], sd[k]) for k in sd)}
c["all_pass"] = all(v for k, v in c.items() if k != "all_pass") and all(o["readback_bitexact"] for o in rep["outputs"].values())
rep["wall_s"] = round(time.time() - t0, 1)
json.dump(rep, open(os.path.join(a.out_dir, "WISEFT_DECLAM_BUILD.json"), "w"), indent=1)
print("ZZDECLAMBUILT" if c["all_pass"] else "ZZREFUSE_declam_outputs", json.dumps({g: o["md5"] for g, o in rep["outputs"].items()}))
