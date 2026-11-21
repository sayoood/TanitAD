"""Diagnose G-EVAL on Thor: build the trainer model exactly as the gate's model job does and the eval
loader's model from the smoke's config.json, then diff every NON-state attribute and non-persistent buffer
module by module (the state_dicts are already identical). CPU only, like the gate's model job."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, math
from pathlib import Path

GD = Path(sys.argv[1])
sys.path.insert(0, str(Path(sys.argv[2]) / "stack" / "scripts"))
import launch_gate as LG  # noqa: E402
import torch  # noqa: E402

ctx = LG.Ctx.load(GD / "ctx.json")
T = LG.load_trainer(ctx)
scratch = GD / "diag_scratch"
scratch.mkdir(exist_ok=True)
argv, arec = LG._sentinel_argv(ctx, ctx.prof["model_input_flags"], scratch)
cap = LG.run_trainer_until(T, argv, "model")
mt = cap["model"].eval()
ck = scratch / "mt.pt"
torch.save({"model": mt.state_dict(), "step": 0}, ck)
ctx.options["eval_loader"] = str(Path(sys.argv[2]) / "stack/tanitad/eval/refcv7_loader.py")
ctx.options["eval_kit"] = "/home/nvidia"
L, _ = LG._load_loader(ctx)
smoke_cfg = LG.latest_smoke_config(ctx)
stamps = LG.read_json(smoke_cfg)
config = {k: v for k, v in stamps.items() if k != "argv"}
config["argv"] = list(ctx.argv)
ml, cfg_l, args_l, rec = L.build_model(config, str(ck), device="cpu", strict=True)
ml.eval()


def scal(v):
    return isinstance(v, (int, float, str, bool, tuple)) and not isinstance(v, torch.Tensor)


diffs = []
mods_t = dict(mt.named_modules())
mods_l = dict(ml.named_modules())
for name in sorted(set(mods_t) & set(mods_l)):
    a, b = mods_t[name], mods_l[name]
    for k in sorted(set(vars(a)) | set(vars(b))):
        if k.startswith("_") and k in ("_parameters", "_buffers", "_modules", "_forward_hooks",
                                       "_forward_pre_hooks", "_backward_hooks", "_state_dict_hooks",
                                       "_load_state_dict_pre_hooks", "_non_persistent_buffers_set"):
            continue
        va, vb = getattr(a, k, "<absent>"), getattr(b, k, "<absent>")
        if scal(va) or scal(vb):
            if va != vb:
                diffs.append((name, k, repr(va)[:80], repr(vb)[:80]))
    for bn in getattr(a, "_non_persistent_buffers_set", set()):
        ta, tb = a._buffers.get(bn), b._buffers.get(bn)
        if ta is None or tb is None or ta.shape != tb.shape or not torch.equal(ta.cpu(), tb.cpu()):
            diffs.append((name, "buffer:" + bn, str(None if ta is None else ta.flatten()[:4].tolist()),
                          str(None if tb is None else tb.flatten()[:4].tolist())))
only_t = sorted(set(mods_t) - set(mods_l))[:10]
only_l = sorted(set(mods_l) - set(mods_t))[:10]
out = {"n_diffs": len(diffs), "diffs": diffs[:80], "only_trainer_modules": only_t, "only_loader_modules": only_l,
       "sentinel_argv_dropped": arec.get("dropped")}
json.dump(out, open(GD / "geval_diff.json", "w"), indent=1)
print("ZZDIFF", len(diffs))
for d in diffs[:40]:
    print(d)


# ---- model-level OBJECT attributes (rig camera, lift banks, seams ...) ----------------------------
def flat(o, depth=0, pre=""):
    out = {}
    if isinstance(o, torch.Tensor):
        out[pre] = (tuple(o.shape), float(o.double().sum()) if o.numel() and o.is_floating_point() else o.numel())
    elif isinstance(o, (int, float, str, bool)) or o is None:
        out[pre] = o
    elif depth < 3 and isinstance(o, dict):
        for k, v in list(o.items())[:50]:
            out.update(flat(v, depth + 1, f"{pre}[{k!r}]"))
    elif depth < 3 and isinstance(o, (list, tuple)):
        for i, v in enumerate(list(o)[:20]):
            out.update(flat(v, depth + 1, f"{pre}[{i}]"))
    elif depth < 3 and hasattr(o, "__dict__") and not isinstance(o, torch.nn.Module):
        for k, v in vars(o).items():
            out.update(flat(v, depth + 1, f"{pre}.{k}"))
    return out


od = []
for k in sorted(set(vars(mt)) | set(vars(ml))):
    if k in ("_parameters", "_buffers", "_modules") or k.startswith("_forward") or k.startswith("_backward"):
        continue
    a, b = getattr(mt, k, "<absent>"), getattr(ml, k, "<absent>")
    if isinstance(a, torch.nn.Module) or isinstance(b, torch.nn.Module):
        continue
    fa, fb = flat(a, 0, k), flat(b, 0, k)
    for p in sorted(set(fa) | set(fb)):
        if fa.get(p, "<absent>") != fb.get(p, "<absent>"):
            od.append((p, str(fa.get(p, "<absent>"))[:90], str(fb.get(p, "<absent>"))[:90]))
print("ZZOBJ", len(od))
for d in od[:40]:
    print("OBJ", d)
print("ARGV_SENTINEL_DIFF", [(f, v) for f, v in LG.flag_pairs(argv) if (f, v) not in LG.flag_pairs(list(ctx.argv))][:20])


# ---- the forward itself: which outputs differ, and does requires_grad matter? --------------------------
e_ds, e_eps, drec = L.build_eval_dataset(ml, cfg_l, args_l, config, with_perception_targets=False)
idx = int(L.inrun_eval_perm(e_ds, 1, 1)[0])
batch = torch.utils.data.default_collate([e_ds[idx]])
e_i, t = e_ds.index[idx]
if getattr(mt, "_perception", None) is not None:
    batch["map_ep"] = torch.tensor([int(e_eps[e_i].episode_id)], dtype=torch.long)
ot = LG.output_digests(LG._forward_out(T, mt, batch, cap.get("args", args_l), 1234))
ol = LG.output_digests(LG._forward_out(T, ml, batch, args_l, 1234))
dk = sorted(k for k in set(ot) | set(ol) if ot.get(k) != ol.get(k))
print("ZZFWD", len(dk), "of", len(ot))
print("FWD_DIFF_KEYS", dk)
for p in mt.parameters():
    p.requires_grad_(False)
ot2 = LG.output_digests(LG._forward_out(T, mt, batch, cap.get("args", args_l), 1234))
dk2 = sorted(k for k in set(ot2) | set(ol) if ot2.get(k) != ol.get(k))
print("ZZFWD_REQGRAD_OFF", len(dk2))
