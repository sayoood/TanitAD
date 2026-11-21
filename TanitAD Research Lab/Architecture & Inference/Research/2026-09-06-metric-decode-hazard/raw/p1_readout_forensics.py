"""P1 - is the step_readout_op metric-decode hazard REAL or an ACCOUNTING one?

Zero GPU. Reads the 9 banked v7-tiny checkpoints plus the classes from the clone.

Three questions, each with a control that must read a KNOWN value:

  Q1  Is `step_readout_op` in each banked ckpt bit-identical to a FRESH init?
      CONTROL (must differ per arm): predictor_op.heads.1.weight md5.
      CONTROL (must read a known value): LayerNorm net.0.weight == all ones,
      net.0.bias == all zeros -- an identity of nn.LayerNorm's own init.

  Q2  What does an at-init readout EMIT, in metres, INDEPENDENT of the latents?
      The readout's first layer is a LayerNorm over the concatenated pair, so
      the input scale is normalised away BY CONSTRUCTION: the emitted magnitude
      is a property of the weights, not of the data.
      CONTROL: feed inputs scaled by 1e-3 / 1 / 1e3 -- the emission must not move.

  Q3  Is the emitted trajectory SEED-ARBITRARY? Decode the same latents through
      the banked readout and through a different-seed init and compare.
      CONTROL: the same seed twice must give exactly 0 difference.
"""
import hashlib
import json
import math
import os
import sys

import torch

CKROOT = r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901"
OUT = r"C:\Users\Admin\tanitad-mdhazard\_work\p1_readout_forensics.json"

sys.path.insert(0, r"C:\Users\Admin\tanitad-mdhazard\stack")
sys.path.insert(0, r"C:\Users\Admin\tanitad-mdhazard\stack\scripts")
import tanitad  # noqa: E402
from tanitad.models.metric_dynamics import (  # noqa: E402
    StepDisplacementReadout, accumulate_se2)

print("[import] tanitad.__file__ =", tanitad.__file__, flush=True)
assert "tanitad-mdhazard" in tanitad.__file__, "WRONG TREE IMPORTED"

RES = {"import_origin": tanitad.__file__, "torch": torch.__version__}


def md5t(t):
    return hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sd_of(ck):
    for k in ("stack", "model", "state_dict"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
            return ck[k]
    return ck


# --------------------------------------------------------------------------- #
# Q1 - per-arm fingerprint                                                     #
# --------------------------------------------------------------------------- #
arms = sorted(d for d in os.listdir(CKROOT)
              if d.startswith("v7tiny_") and
              os.path.isfile(os.path.join(CKROOT, d, "ckpt.pt")))
print("[arms]", len(arms), arms, flush=True)

per_arm = {}
for a in arms:
    p = os.path.join(CKROOT, a, "ckpt.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = sd_of(ck)
    ro = {k[len("step_readout_op."):]: v for k, v in sd.items()
          if k.startswith("step_readout_op.")}
    row = {"n_tensors": len(ro),
           "n_params": int(sum(v.numel() for v in ro.values())),
           "md5": {k: md5t(v) for k, v in sorted(ro.items())}}
    # the TRAINED control
    for ctl in ("predictor_op.heads.1.weight", "predictor_op.heads.1.0.weight"):
        if ctl in sd:
            row["control_trained_md5"] = md5t(sd[ctl])
            row["control_trained_key"] = ctl
            break
    # LayerNorm identity check -- a KNOWN value, not an estimate
    if "net.0.weight" in ro:
        row["ln_weight_all_ones"] = bool(torch.equal(
            ro["net.0.weight"], torch.ones_like(ro["net.0.weight"])))
        row["ln_bias_all_zeros"] = bool(torch.equal(
            ro["net.0.bias"], torch.zeros_like(ro["net.0.bias"])))
    # nn.Linear init signature: U(-b, b), b = 1/sqrt(fan_in)
    lin = {}
    for k in ("net.1.weight", "net.1.bias", "net.3.weight", "net.3.bias"):
        if k not in ro:
            continue
        w = ro[k].float()
        fan_in = (ro[k.replace(".bias", ".weight")].shape[1]
                  if k.endswith(".bias") else w.shape[1])
        b = 1.0 / math.sqrt(fan_in)
        lin[k] = {"fan_in": int(fan_in), "init_bound": b,
                  "max_abs": float(w.abs().max()),
                  "max_abs_over_bound": float(w.abs().max()) / b,
                  "std": float(w.std()),
                  "std_over_uniform_std": float(w.std()) / (b / math.sqrt(3.0))}
    row["linear_init_signature"] = lin
    # config recipe (the RECORD-based detection)
    cfg = ck.get("config") if isinstance(ck, dict) else None
    args = (cfg or {}).get("args") if isinstance(cfg, dict) else None
    if isinstance(args, dict):
        row["recipe"] = {k: args.get(k) for k in
                         ("w_o1_ctrl", "w_o1_fact", "w_o1_scene", "w_o3",
                          "w_o5", "steps", "seed", "stage", "d_op")}
    per_arm[a] = row
    del ck, sd
    print(f"  [{a}] tensors={row['n_tensors']} params={row['n_params']} "
          f"ln_ones={row.get('ln_weight_all_ones')} "
          f"recipe_o1={None if 'recipe' not in row else (row['recipe']['w_o1_ctrl'], row['recipe']['w_o1_fact'], row['recipe']['w_o1_scene'])}",
          flush=True)

RES["per_arm"] = per_arm

# distinct-fingerprint census
def distinct(getter):
    vals = {}
    for a, r in per_arm.items():
        v = getter(r)
        vals.setdefault(v, []).append(a)
    return {"n_distinct": len(vals), "groups": {k: v for k, v in vals.items()}}


RES["census"] = {
    "step_readout_op.net.1.weight": distinct(lambda r: r["md5"].get("net.1.weight")),
    "step_readout_op.net.3.weight": distinct(lambda r: r["md5"].get("net.3.weight")),
    "CONTROL_predictor_op.heads.1.weight": distinct(
        lambda r: r.get("control_trained_md5")),
}

# --------------------------------------------------------------------------- #
# Q1b - bit-compare against a FRESH init at seeds 0..3                         #
# --------------------------------------------------------------------------- #
sd_dim = None
for a, r in per_arm.items():
    for k, s in r["linear_init_signature"].items():
        if k == "net.1.weight":
            sd_dim = s["fan_in"] // 2
if sd_dim is None:
    raise SystemExit("could not infer state_dim")
RES["state_dim"] = sd_dim
print("[state_dim]", sd_dim, flush=True)

fresh = {}
for seed in range(4):
    torch.manual_seed(seed)
    m = StepDisplacementReadout(sd_dim)
    fresh[seed] = {k: v.detach().clone() for k, v in m.state_dict().items()}

match = {}
for a, r in per_arm.items():
    hits = []
    for seed, f in fresh.items():
        if all(r["md5"].get(k) == md5t(v) for k, v in f.items()):
            hits.append(seed)
    match[a] = hits
RES["bit_identical_to_fresh_init_seeds"] = match
print("[bit-identical to fresh init]", match, flush=True)

# --------------------------------------------------------------------------- #
# Q2 - what does the banked readout EMIT?  (LayerNorm -> scale-invariant)      #
# --------------------------------------------------------------------------- #
DT = 0.1
K = 20            # t1_eval horizon_steps for the banked T1 read (2.0 s)
B = 4096

def emission_stats(state_dict, scale, seed=1234):
    m = StepDisplacementReadout(sd_dim)
    m.load_state_dict(state_dict)
    m.eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        # per-step latents; the pair is concatenated then LayerNorm'd
        z = torch.randn(B, K + 1, sd_dim, generator=g) * scale
        dp = torch.stack([m(z[:, j], z[:, j + 1]) for j in range(K)], dim=1)
        wp = accumulate_se2(dp)                     # [B, K, 2]
    return {
        "dpose_xy_rms_m_per_step": float(dp[..., :2].pow(2).sum(-1)
                                         .sqrt().mean()),
        "dyaw_rms_rad_per_step": float(dp[..., 2].pow(2).mean().sqrt()),
        "implied_speed_mps": float(dp[..., :2].pow(2).sum(-1).sqrt().mean() / DT),
        "final_displacement_m": float(wp[:, -1].norm(dim=-1).mean()),
        "path_len_m": float(dp[..., :2].norm(dim=-1).sum(-1).mean()),
    }


# use the arm that produced D-T1-V7-READ's headline numbers
T1_ARMS = ["v7tiny_emao14_30k", "v7tiny_o14fut30k", "v7tiny_emao14_30k_tauramp"]
emis = {}
for a in T1_ARMS:
    p = os.path.join(CKROOT, a, "ckpt.pt")
    if not os.path.isfile(p):
        emis[a] = "CKPT_NOT_ON_THIS_BOX"
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = sd_of(ck)
    ro = {k[len("step_readout_op."):]: v for k, v in sd.items()
          if k.startswith("step_readout_op.")}
    row = {}
    for scale in (1e-3, 1.0, 1e3):
        row[f"latent_scale_{scale:g}"] = emission_stats(ro, scale)
    emis[a] = row
    del ck, sd
    print(f"  [emit {a}] {row['latent_scale_1']}", flush=True)
RES["emission_at_init"] = emis

# --------------------------------------------------------------------------- #
# Q3 - seed arbitrariness                                                      #
# --------------------------------------------------------------------------- #
def decode_traj(state_dict, seed=1234):
    m = StepDisplacementReadout(sd_dim)
    m.load_state_dict(state_dict)
    m.eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        z = torch.randn(512, K + 1, sd_dim, generator=g)
        dp = torch.stack([m(z[:, j], z[:, j + 1]) for j in range(K)], dim=1)
        return accumulate_se2(dp)


ck = torch.load(os.path.join(CKROOT, "v7tiny_emao14_30k", "ckpt.pt"),
                map_location="cpu", weights_only=False)
sd = sd_of(ck)
ro = {k[len("step_readout_op."):]: v for k, v in sd.items()
      if k.startswith("step_readout_op.")}
del ck, sd
wp_bank = decode_traj(ro)
wp_bank2 = decode_traj(ro)
alt = {}
for seed in (1, 2, 7):
    torch.manual_seed(seed)
    m = StepDisplacementReadout(sd_dim)
    wp = decode_traj({k: v.detach().clone() for k, v in m.state_dict().items()})
    alt[f"seed_{seed}"] = {
        "mean_endpoint_dist_from_banked_m": float(
            (wp[:, -1] - wp_bank[:, -1]).norm(dim=-1).mean()),
        "mean_endpoint_norm_m": float(wp[:, -1].norm(dim=-1).mean()),
        "cos_endpoint": float(torch.nn.functional.cosine_similarity(
            wp[:, -1], wp_bank[:, -1], dim=-1).mean()),
    }
RES["seed_arbitrariness"] = {
    "CONTROL_same_readout_twice_max_abs_diff": float(
        (wp_bank - wp_bank2).abs().max()),
    "banked_endpoint_norm_m": float(wp_bank[:, -1].norm(dim=-1).mean()),
    "alternative_seeds": alt,
}
print("[seed arbitrariness]", json.dumps(RES["seed_arbitrariness"], indent=1),
      flush=True)

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(RES, fh, indent=1)
print("[wrote]", OUT, flush=True)
