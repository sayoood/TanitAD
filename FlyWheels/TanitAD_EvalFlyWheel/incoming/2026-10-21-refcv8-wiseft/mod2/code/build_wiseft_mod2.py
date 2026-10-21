#!/usr/bin/env python3
"""R8-WISEFT-MOD2 builder (registered before any of its arms is rolled or scored): splits R8-WISEFT-MOD's locus (PLAN) and moves its lambda.

Arms (each on refcv8 EMA-48,852 with refcv7-50,400 mixed into a SUBSET of the 1,027 shared float keys; theta_lam = (1 - lam) theta7 + lam theta8):
  DEC     lambda 0.5 on the PLAN keys under core.decoder. (168 keys, 15.26 M; drift |theta8 - theta7| / |theta7| = 0.037)
  TAC     lambda 0.5 on every OTHER PLAN key (122 keys, 9.35 M: core.tactical_trunk 0.53, phi_tac 0.26, tac_decoder_v6 0.26, the lat / lon / tac
          heads 0.20-0.83, nav_to_tac 0.42, scorer 0.31, ...)
  PLAN75  lambda 0.75 on all 290 PLAN keys (a quarter refcv7)
PLAN = build_wiseft_groups.py's complement of its PERC prefixes (unchanged).  Controls: DEC and TAC are complements whose union reproduces the
banked theta_PLAN (md5 47c7fdac...) bit-exactly key by key; PLAN75 equals mix(theta7, theta8, 0.75) on PLAN keys and theta8 elsewhere; the
rebuilt theta_PLAN's keys equal the banked file's; new / non-float keys from theta8; readback bit-exact.

    python build_wiseft_mod2.py --theta7 <refcv7 ckpt.pt> --theta8 <ckpt_ema_48852.pt> --theta-plan <ckpt_wiseft_l0p50_PLAN.pt> --out-dir <dir>
"""
import argparse, hashlib, json, os, sys, time
import torch

PERC = ("core.encoder.", "_perception.", "_map_hires.", "core.agent_head.", "core.agent_embed.", "core.law_head.", "core.ego_hist.",
        "core.measurement.", "tac_latent_proj.")
DEC = ("core.decoder.",)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def mix(a, b, lam):
    return ((1.0 - lam) * a.float() + lam * b.float()).to(b.dtype)


ap = argparse.ArgumentParser()
ap.add_argument("--theta7", required=True); ap.add_argument("--theta8", required=True); ap.add_argument("--theta-plan", required=True)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
t0 = time.time()
d7 = torch.load(a.theta7, map_location="cpu", weights_only=False); d8 = torch.load(a.theta8, map_location="cpu", weights_only=False)
m7, m8 = d7["model"], d8["model"]
mP = torch.load(a.theta_plan, map_location="cpu", weights_only=False)["model"]
interp = [k for k, v in m8.items() if k in m7 and torch.is_tensor(v) and v.is_floating_point() and tuple(m7[k].shape) == tuple(v.shape)]
assert len(interp) == 1027, len(interp)
plan = [k for k in interp if not k.startswith(PERC)]
assert len(plan) == 290, len(plan)
sub = {k: ("DEC" if k.startswith(DEC) else "TAC") for k in plan}
spec = {"DEC": ({k for k in plan if sub[k] == "DEC"}, 0.5), "TAC": ({k for k in plan if sub[k] == "TAC"}, 0.5), "PLAN75": (set(plan), 0.75)}
rep = {"id": "R8-WISEFT-MOD2", "arms": {g: {"lambda": lam, "n_keys": len(ks), "numel": int(sum(m8[k].numel() for k in ks))} for g, (ks, lam) in spec.items()},
       "theta7_md5": md5(a.theta7), "theta8_md5": md5(a.theta8), "theta_plan_md5": md5(a.theta_plan), "outputs": {}, "controls": {}}
arms = {g: {k: (mix(m7[k], v, lam) if k in ks else v) for k, v in m8.items()} for g, (ks, lam) in spec.items()}
c = rep["controls"]
c["rebuilt_theta_plan_equals_banked"] = all(torch.equal(mix(m7[k], m8[k], 0.5), mP[k]) for k in plan) and all(torch.equal(mP[k], m8[k]) for k in m8 if k not in set(plan))
c["DEC_TAC_complement_union_is_theta_plan"] = all((torch.equal(arms["DEC"][k], mP[k]) and torch.equal(arms["TAC"][k], m8[k])) or
                                                  (torch.equal(arms["TAC"][k], mP[k]) and torch.equal(arms["DEC"][k], m8[k])) for k in plan)
c["PLAN75_exact"] = all(torch.equal(arms["PLAN75"][k], mix(m7[k], m8[k], 0.75)) for k in plan)
c["outside_set_is_theta8"] = all(torch.equal(arms[g][k], m8[k]) for g, (ks, _) in spec.items() for k in m8 if k not in ks)
c["DEC_TAC_both_nonempty_and_changed"] = all(any(not torch.equal(arms[g][k], m8[k]) for k in spec[g][0]) for g in ("DEC", "TAC"))
if not all(c.values()):
    json.dump(rep, open(os.path.join(a.out_dir, "WISEFT_MOD2_BUILD.json"), "w"), indent=1); sys.exit("ZZREFUSE_mod2_controls")
for g, sd in arms.items():
    p = os.path.join(a.out_dir, f"ckpt_wiseft_mod2_{g}.pt")
    torch.save({"model": sd, "step": d8.get("step"), "ema_decay": d8.get("ema_decay"), "num_updates": d8.get("num_updates"),
                "kind": f"wiseft_mod2_{g}", "wiseft": {"lambda": spec[g][1], "group": g, "theta7_md5": rep["theta7_md5"], "theta8_md5": rep["theta8_md5"]}}, p)
    back = torch.load(p, map_location="cpu", weights_only=False)["model"]
    rep["outputs"][g] = {"path": p, "md5": md5(p), "readback_bitexact": all(torch.equal(back[k], sd[k]) for k in sd)}
c["all_pass"] = all(v for k, v in c.items() if k != "all_pass") and all(o["readback_bitexact"] for o in rep["outputs"].values())
rep["wall_s"] = round(time.time() - t0, 1)
json.dump(rep, open(os.path.join(a.out_dir, "WISEFT_MOD2_BUILD.json"), "w"), indent=1)
print("ZZMOD2BUILT" if c["all_pass"] else "ZZREFUSE_mod2_outputs", json.dumps({g: v["n_keys"] for g, v in rep["arms"].items()}),
      json.dumps({g: o["md5"] for g, o in rep["outputs"].items()}))
