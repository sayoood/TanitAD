#!/usr/bin/env python3
"""R8-WISEFT-MOD builder: the lambda = 0.5 blend applied to ONE module group only, the other group left at refcv8 EMA-48,852.

Groups partition the 1,027 shared float keys by prefix (asserted: every interpolated key falls in exactly one group):
  PERC  perception stack: core.encoder, _perception, _map_hires, core.agent_head, core.agent_embed, core.law_head, core.ego_hist,
        core.measurement, tac_latent_proj (the latent projection feeding the tactical stack from perception)
  PLAN  planning stack:   everything else (core.decoder, tac_decoder_v6, core.tactical_trunk, phi_tac, lat/lon heads, nav_to_tac, ...)
theta_PLAN = blend on PLAN keys, EMA-48,852 elsewhere; theta_PERC = blend on PERC keys, EMA-48,852 elsewhere.  refcv8's 34 new tensors and
non-float entries from theta8 in both.  Controls: per key, exactly one of the two arms equals theta_0.5 and the other equals theta8 (so the
two arms are complements whose union reproduces theta_0.5 bit-exactly); the theta_0.5 used is re-built here and compared to the banked
file's md5 d72f7f57....

    python build_wiseft_groups.py --theta7 <refcv7 ckpt.pt> --theta8 <ckpt_ema_48852.pt> --theta05 <ckpt_wiseft_l0p50.pt> --out-dir <dir>
"""
import argparse, hashlib, json, os, sys, time
import torch

PERC = ("core.encoder.", "_perception.", "_map_hires.", "core.agent_head.", "core.agent_embed.", "core.law_head.", "core.ego_hist.",
        "core.measurement.", "tac_latent_proj.")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def mix(a, b, lam):
    return ((1.0 - lam) * a.float() + lam * b.float()).to(b.dtype)


ap = argparse.ArgumentParser()
ap.add_argument("--theta7", required=True); ap.add_argument("--theta8", required=True); ap.add_argument("--theta05", required=True)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
t0 = time.time()
d7 = torch.load(a.theta7, map_location="cpu", weights_only=False); d8 = torch.load(a.theta8, map_location="cpu", weights_only=False)
m7, m8 = d7["model"], d8["model"]
m05 = torch.load(a.theta05, map_location="cpu", weights_only=False)["model"]
interp = [k for k, v in m8.items() if k in m7 and torch.is_tensor(v) and v.is_floating_point() and tuple(m7[k].shape) == tuple(v.shape)]
grp = {k: ("PERC" if k.startswith(PERC) else "PLAN") for k in interp}
assert len(interp) == 1027, len(interp)
rep = {"id": "R8-WISEFT-MOD", "lambda": 0.5, "perc_prefixes": list(PERC),
       "n_keys": {g: sum(1 for k in interp if grp[k] == g) for g in ("PERC", "PLAN")},
       "numel": {g: int(sum(m8[k].numel() for k in interp if grp[k] == g)) for g in ("PERC", "PLAN")},
       "theta7_md5": md5(a.theta7), "theta8_md5": md5(a.theta8), "theta05_md5": md5(a.theta05), "outputs": {}, "controls": {}}
c05 = all(torch.equal(mix(m7[k], m8[k], 0.5), m05[k]) for k in interp)
rep["controls"]["C05_rebuilt_blend_equals_banked_theta05"] = c05
arms = {}
for g in ("PLAN", "PERC"):
    sd = {k: (mix(m7[k], v, 0.5) if (k in grp and grp[k] == g) else v) for k, v in m8.items()}
    arms[g] = sd
comp = all((torch.equal(arms["PLAN"][k], m05[k]) and torch.equal(arms["PERC"][k], m8[k])) or
           (torch.equal(arms["PERC"][k], m05[k]) and torch.equal(arms["PLAN"][k], m8[k])) for k in interp)
rep["controls"]["C_complement_union_is_theta05"] = comp
rep["controls"]["new_and_nonfloat_from_theta8"] = all(torch.equal(arms[g][k], m8[k]) for g in arms for k in m8 if k not in grp)
if not (c05 and comp and rep["controls"]["new_and_nonfloat_from_theta8"]):
    json.dump(rep, open(os.path.join(a.out_dir, "WISEFT_GROUPS_BUILD.json"), "w"), indent=1); sys.exit("ZZREFUSE_groups_controls")
for g, sd in arms.items():
    p = os.path.join(a.out_dir, f"ckpt_wiseft_l0p50_{g}.pt")
    torch.save({"model": sd, "step": d8.get("step"), "ema_decay": d8.get("ema_decay"), "num_updates": d8.get("num_updates"),
                "kind": f"wiseft_lambda_0.5_{g}_only", "wiseft": {"lambda": 0.5, "group": g, "theta7_md5": rep["theta7_md5"],
                                                                  "theta8_md5": rep["theta8_md5"]}}, p)
    back = torch.load(p, map_location="cpu", weights_only=False)["model"]
    rep["outputs"][g] = {"path": p, "md5": md5(p), "readback_bitexact": all(torch.equal(back[k], sd[k]) for k in sd)}
rep["controls"]["all_pass"] = all(rep["controls"][k] for k in ("C05_rebuilt_blend_equals_banked_theta05", "C_complement_union_is_theta05",
                                                                "new_and_nonfloat_from_theta8")) and all(o["readback_bitexact"] for o in rep["outputs"].values())
rep["wall_s"] = round(time.time() - t0, 1)
json.dump(rep, open(os.path.join(a.out_dir, "WISEFT_GROUPS_BUILD.json"), "w"), indent=1)
print("ZZGROUPSBUILT" if rep["controls"]["all_pass"] else "ZZREFUSE_groups_outputs", json.dumps(rep["n_keys"]), json.dumps(rep["numel"]),
      json.dumps({g: o["md5"] for g, o in rep["outputs"].items()}))
