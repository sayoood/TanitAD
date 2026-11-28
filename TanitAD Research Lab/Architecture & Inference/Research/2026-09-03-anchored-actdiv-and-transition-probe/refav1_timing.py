import importlib.util, time, sys, os, torch, numpy as np
torch.set_num_threads(6)
sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\stack"); sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\taniteval")
spec = importlib.util.spec_from_file_location("refav1_arm", r"C:\Users\Admin\tanitad-wt\taniteval\tools\refav1_arm.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
import argparse
CK = r"C:\Users\Admin\refav1_eval_slice\ckpt_ep2\ckpt.pt"
t0=time.time(); ck = torch.load(CK, map_location="cpu", weights_only=False); print("keys", list(ck.keys())[:12], "load s", round(time.time()-t0,1))
for k,v in ck.items():
    if isinstance(v, dict) and k in ("model","ema","ema_model"): 
        t = next(iter(v.values())); print(k, "n_tensors", len(v), "dtype", t.dtype)
print("step", ck.get("step"), "cfg speed_channel", (ck.get("cfg") or {}).get("speed_channel"), "no_hierarchy?", (ck.get("cfg") or {}).get("nav_inject"))
del ck
t0=time.time(); model, cfg, prov = ra.load_model(CK, None, "cpu"); print("load_model s", round(time.time()-t0,1), "params", prov["trainable_parameters"], "a_dim", cfg.a_dim, "speed_channel", cfg.speed_channel, "W", cfg.op_window, "K", cfg.op_steps, "plan_steps", cfg.plan_steps, "bptt", getattr(cfg,"bptt_truncate",None))
a = argparse.Namespace(cache=r"C:\Users\Admin\refav1_eval_slice\fp8", episodes=r"C:\Users\Admin\refav1_eval_slice\eps", lru=64,
    labels=r"C:\Users\Admin\tanitad-wt\_s2build\release\v72\s2_labels_v7.2_eval.jsonl.gz", nav=r"C:\Users\Admin\tanitad-wt\_s2build\release\v72\s2_labels_v7.2_eval.jsonl.gz")
names = ra.episode_names(a.cache)[:20]
ld = ra.build_loader(a, cfg, 10, names)
sel = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows) if (t - (ld.W - 1)) % 10 == 0]
print("windows total", len(ld), "selected(stride10)", len(sel), "W", ld.W, "episodes", len(ld.names))
ld._order, ld._cursor = [sel[0][0]], 0
b = ld.batch(1)
print({k: (tuple(v.shape), v.dtype) for k,v in b.items() if hasattr(v,"shape")})
feats = b["feats"].float(); act = b["actions"].float(); v0 = b["v0"].float()
print("actions[0,:3]", act[0,:3].tolist(), "v0", v0.tolist())
with torch.no_grad():
    t0=time.time(); field = model.encode(feats); print("encode s", round(time.time()-t0,2), tuple(field.shape))
    last = model._last_state(field)
    pooled_win = field.mean(dim=-2)
    brains = model._run_brains(pooled_win, b.get("nav_cmd"))
    intent = None if brains is None else brains["intent"]
    print("intent", None if intent is None else tuple(intent.shape))
    acts_in = model.augment_actions(act, v0)
    print("acts_in", tuple(acts_in.shape))
    for B in (1, 4):
        L = last.expand(B, -1, -1).contiguous(); A = acts_in[:, 0].expand(B, -1).contiguous()
        I = None if intent is None else intent.expand(B, -1).contiguous()
        t0=time.time(); z1 = model.operative.step(L, A, intent=I); dt=time.time()-t0
        print(f"step B={B}: {dt:.2f} s  ({dt/B:.2f} s/window)  out {tuple(z1.shape)}")
    z1b = model.operative.step(last, acts_in[:,0], intent=intent)
    print("C0 identity max|diff|", float((z1b - model.operative.step(last, acts_in[:,0], intent=intent)).abs().max()))
    zero = torch.zeros_like(acts_in[:, 0]); z0 = model.operative.step(last, model.augment_actions(torch.zeros_like(act), v0)[:,0], intent=intent)
    print("|zhat(a_true)-zhat(0)| mean", float((z1b - z0).abs().mean()), " |field move| mean", float((z1b - last).abs().mean()))
