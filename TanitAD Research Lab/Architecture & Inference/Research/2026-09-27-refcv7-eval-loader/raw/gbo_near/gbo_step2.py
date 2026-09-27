"""Step 2 (MEASURED): the FIXED harness replay vs train() on the tiny rig -- state_dict (keys + bytes), every
attribute train() sets on the model (the `_w_*` carriers, G-EVAL's named ones, and every other `_` attribute
both carry), param_breakdown, and the forward on a fixed batch (the gate's `_forward_out`, eval mode, seed 1234).
Writes C:/lgt/r7ldr_scratch/gbo_step2.json."""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, "C:/lgt/r7ldr/stack/tests")
import torch  # noqa: E402

import test_g_box_overfit_near_lift as TM  # noqa: E402

LG = TM.LG
d = Path("C:/lgt/r7ldr_scratch/gbo_step2_rig")
d.mkdir(exist_ok=True)
TM._synth_inputs(d)
argv = TM._argv(d)
T = TM._load("refc_v3_train_gbo_s2", TM.STACK / "scripts" / "refc_v3_train.py")
cap = LG.run_trainer_until(T, argv, "model")
mt = cap["model"]
G = TM._load("g_box_overfit_s2", TM.GBO_PATH)
mh, ad = TM._harness_model(G, T, argv)
res = {"dvb_mismatches": ad.dvb_mismatches}
dt, dh = LG.state_digests(mt.state_dict()), LG.state_digests(mh.state_dict())
res["state"] = {"n_trainer": len(dt), "n_harness": len(dh), "keys_symdiff": sorted(set(dt) ^ set(dh)),
                "n_bytes_differ": sum(1 for k in dt if k in dh and dt[k] != dh[k])}
# every underscore attribute either model carries (the train()-set carriers), compared by value
attrs = {}
for a in sorted(set(vars(mt)) | set(vars(mh))):
    if not a.startswith("_") or a.startswith("__") or a in ("_modules", "_parameters", "_buffers",
                                                              "_non_persistent_buffers_set",
                                                              "_backward_hooks", "_backward_pre_hooks",
                                                              "_forward_hooks", "_forward_pre_hooks",
                                                              "_forward_hooks_with_kwargs",
                                                              "_forward_hooks_always_called",
                                                              "_forward_pre_hooks_with_kwargs",
                                                              "_state_dict_hooks", "_state_dict_pre_hooks",
                                                              "_load_state_dict_pre_hooks",
                                                              "_load_state_dict_post_hooks",
                                                              "_is_full_backward_hook", "training"):
        continue
    va, vb = getattr(mt, a, "<absent>"), getattr(mh, a, "<absent>")
    if torch.is_tensor(va) and torch.is_tensor(vb):
        same = LG.tensor_digest(va) == LG.tensor_digest(vb)
    elif isinstance(va, (int, float, bool, str, type(None))) and isinstance(vb, (int, float, bool, str, type(None))):
        same = va == vb
    elif isinstance(va, dict) and isinstance(vb, dict):
        same = json.dumps(va, sort_keys=True, default=str) == json.dumps(vb, sort_keys=True, default=str)
    else:
        same = f"type {type(va).__name__} vs {type(vb).__name__}"
    attrs[a] = same
res["attributes"] = {"n": len(attrs), "not_equal": {k: v for k, v in attrs.items() if v is not True},
                     "names": sorted(attrs)}
pb_t = {k: int(v) for k, v in T.v3.param_breakdown_v3(mt).items()}
pb_h = {k: int(v) for k, v in T.v3.param_breakdown_v3(mh).items()}
res["param_breakdown_equal"] = pb_t == pb_h
# the forward on a fixed batch (synthetic episodes stamped with the rig's clip ids)
cfg = mt.cfg
eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=list(TM.CLIPS))
ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
ds.u8_frames, ds.ego_history = True, True
b = torch.utils.data.default_collate([ds[5]])
e_i, _t = ds.index[5]
b["map_ep"] = torch.tensor([int(eps[e_i].episode_id)], dtype=torch.long)
b["v_max_ms"], b["v_max_valid"] = torch.tensor([13.8889]), torch.tensor([True])
mt.eval(); mh.eval()
ot = LG.output_digests(LG._forward_out(T, mt, b, cap["args"], 1234))
oh = LG.output_digests(LG._forward_out(T, mh, b, ad.args, 1234))
res["forward"] = {"n_outputs": len(ot), "differ": sorted(k for k in set(ot) | set(oh) if ot.get(k) != oh.get(k))}
json.dump(res, open("C:/lgt/r7ldr_scratch/gbo_step2.json", "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "attributes"}, indent=1, default=str))
print("attributes", res["attributes"]["n"], "not_equal", res["attributes"]["not_equal"])
print("names", res["attributes"]["names"])
