"""C8 (card section 5): ZERO-ACT deliberate regression, dev-box CPU, zero training. R35_s0's WM with the level-1 action embedding (every parameter of stack.act_embed, weights and bias) zeroed.
Reads the level-1 forward response in the planning form (probe_actsens.py Part A, same windows): a +1 sd step of one action channel must move the predicted next latent by EXACTLY 0.0 with the embedding zeroed,
and by the R35 value (> 0, 8.7 % / 10.3 % of the real step at s0, card section 1) with it intact (the control that proves the probe can read a non-zero).
usage: python c8_zeroact.py <tree_root> <out.json>"""
import dataclasses
import json
import os
import sys

tree, outp = sys.argv[1:3]
sys.path[:0] = [os.path.join(tree, "stack"), os.path.join(tree, "taniteval")]
import torch  # noqa: E402
import tanitad  # noqa: E402
assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), tanitad.__file__
from tanitad.models import v8f_mini as M  # noqa: E402

torch.set_num_threads(2)
PULL, LAT = "C:/Users/Admin/t2v8f_pulls", "F:/TanitAD-scratch/hwm_plan"
tag = "R35_s0"
ck = torch.load(f"{PULL}/ladder_{tag}/ckpt_last.pt", map_location="cpu", weights_only=False)
names = {f.name for f in dataclasses.fields(M.V8FMiniConfig)}
cfg = M.V8FMiniConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items() if k in names})
L = torch.load(f"{LAT}/latents_train_{tag}.pt", map_location="cpu", weights_only=False)


def response(zero):
    torch.manual_seed(0)
    wm = M.WMAux(cfg)
    wm.load_state_dict({k[3:]: v for k, v in ck["model"].items() if k.startswith("wm.")}, strict=True)
    wm.eval()
    n_zeroed = 0
    if zero:
        with torch.no_grad():
            for p in wm.stack.act_embed.parameters():
                p.zero_()
                n_zeroed += p.numel()
    z = L["Z1"][tag].float()
    a = wm.scale_actions(L["A_raw"].float())[:, :-1]
    ok = L["ok"]
    v = ok[:, :-1] & ok[:, 1:]
    st = wm.stack
    with torch.no_grad():
        z0, a0, z1t = z[:, :-1][v], a[v], z[:, 1:][v]
        sd = a0.std(0, unbiased=False)
        base = st.rollout(1, z0, a0[:, None, :], context=1)[:, 0]
        real = (z1t - z0).norm(dim=-1).mean()
        out = {"n_zeroed_params": n_zeroed, "n_transitions": int(v.sum())}
        for c, nm in ((0, "acc"), (1, "yaw_rate")):
            ap = a0.clone()
            ap[:, c] += sd[c]
            d = (st.rollout(1, z0, ap[:, None, :], context=1)[:, 0] - base).norm(dim=-1)
            out[nm] = {"dz_per_sd_mean": float(d.mean()), "dz_per_sd_max": float(d.max()), "rel_to_real_step": float(d.mean() / real)}
    return out


res = {"tanitad_file": tanitad.__file__, "control_intact": response(False), "zero_act": response(True)}
za = res["zero_act"]
res["verdict"] = {"zero_act_exactly_0.0": all(za[k]["dz_per_sd_max"] == 0.0 for k in ("acc", "yaw_rate")),
                  "control_nonzero": all(res["control_intact"][k]["dz_per_sd_mean"] > 0.0 for k in ("acc", "yaw_rate"))}
json.dump(res, open(outp, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
