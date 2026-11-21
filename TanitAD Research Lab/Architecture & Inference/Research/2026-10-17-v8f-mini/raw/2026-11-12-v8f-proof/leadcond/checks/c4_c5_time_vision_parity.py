"""c4 time and vision only, c5 train / infer parity (LEADCOND), tree_patch, tiny world.
c4  u (from the real forward_infer path, recorded at each call of V8FMini._lead_u) is EXACTLY unchanged when
      - every ego input of the batch is perturbed (act, wm_speed, p10f, nav, limit, limit_ok, pose*, base, kin3 and every other float / int key except feats and frame_ok);
      - a frame BEFORE the lead window is perturbed (the window is the last lead_hist frames);
    and u at an earlier frame `last` (direct call) is exactly unchanged when any frame AFTER `last` is perturbed;
    controls: perturbing the last frame changes u; perturbing a frame inside the window changes u.
c5  u from `forward` equals u from `forward_infer` on the same batch (max abs diff, float32), and equals the u built from the head's own training-time output (wm(adapt, act)['lead_out'][:, -1]) (max abs diff).
Usage: c4_c5_time_vision_parity.py"""
import torch

import lc_common as C

C.use_tree("patch")
from tanitad.models import v8f_mini as M

batch, J = C.world_batch("c4")
_, cfg, model = C.build_from_args(["--lead-sup", "on", "--w-lead-sup", "1.0", "--lead-cond", "on"], J, seed=3)
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    for m_ in model.wm.lead_head.mlp:
        if isinstance(m_, torch.nn.Linear):
            m_.weight.copy_(torch.randn(m_.weight.shape, generator=g) * 0.3)
    model.lead_cond.weight.copy_(torch.randn(model.lead_cond.weight.shape, generator=g) * 0.3)
model.eval()
rec = []
orig = model._lead_u


def spy(b, adapt, last):
    u = orig(b, adapt, last)
    rec.append(u.detach().clone())
    return u


model._lead_u = spy
INF = M.INFERENCE_INPUT_KEYS


def infer_u(b):
    rec.clear()
    with torch.no_grad():
        model.forward_infer({k: v for k, v in b.items() if k in INF})
    return [r.clone() for r in rec]


def train_u(b):
    rec.clear()
    with torch.no_grad():
        model.forward(dict(b), p_gt=0.0)
    return [r.clone() for r in rec]


def clone(b):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}


def noise(x, gen):
    if x.dtype.is_floating_point:
        return x + torch.randn(x.shape, generator=gen, dtype=x.dtype) * 3.0
    if x.dtype == torch.bool:
        return x
    return (x + 1).to(x.dtype) if x.numel() else x


F_ = batch["feats"].shape[1]
H = cfg.lead_hist
base_u = infer_u(batch)
n_nonzero = int((base_u[-1].abs().sum(-1) > 0).sum())
res = {"n_rows": int(batch["feats"].shape[0]), "n_rows_u_nonzero": n_nonzero, "frames": F_, "lead_hist": H, "n_calls_per_forward_infer": len(base_u)}
# ego inputs
b1 = clone(batch)
perturbed = []
gen = torch.Generator().manual_seed(9)
for k, v in batch.items():
    if k in ("feats", "frame_ok") or not torch.is_tensor(v):
        continue
    if v.dtype.is_floating_point or v.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        b1[k] = noise(v, gen)
        if not torch.equal(b1[k], v):
            perturbed.append(k)
u_ego = infer_u(b1)
res["ego"] = {"perturbed_keys": perturbed, "max_abs_diff_u": max(float((a - b).abs().max()) for a, b in zip(base_u, u_ego)), "includes": {k: (k in perturbed) for k in ("act", "p10f", "nav", "limit", "pose", "base")}}
# same through the training forward (p_gt 0)
tb, tp = train_u(batch), train_u(b1)
res["ego"]["max_abs_diff_u_training_forward"] = max(float((a - b).abs().max()) for a, b in zip(tb, tp))
# frame before the window
b2 = clone(batch)
b2["feats"][:, : F_ - H] += torch.randn(b2["feats"][:, : F_ - H].shape, generator=gen) * 3.0
u_old = infer_u(b2)
res["frames_before_window"] = {"max_abs_diff_u": max(float((a - b).abs().max()) for a, b in zip(base_u, u_old))}
# frames after `last` (direct call at an earlier frame)
adapt = model._encode(batch)
last = torch.full((adapt.shape[0],), F_ - 3, dtype=torch.long)
with torch.no_grad():
    u_a = orig(batch, adapt, last)
    b3 = clone(batch)
    b3["feats"][:, F_ - 2:] += torch.randn(b3["feats"][:, F_ - 2:].shape, generator=gen) * 3.0
    u_b = orig(b3, model._encode(b3), last)
    u_c = orig(batch, adapt, torch.full_like(last, F_ - 1))
res["frames_after_t"] = {"last_frame": F_ - 3, "max_abs_diff_u": float((u_a - u_b).abs().max()), "u_nonzero_rows": int((u_a.abs().sum(-1) > 0).sum())}
# controls
b4 = clone(batch)
b4["feats"][:, -1] += torch.randn(b4["feats"][:, -1].shape, generator=gen) * 3.0
u_last = infer_u(b4)
b5 = clone(batch)
b5["feats"][:, F_ - H] += torch.randn(b5["feats"][:, F_ - H].shape, generator=gen) * 3.0
u_in = infer_u(b5)
res["controls"] = {"last_frame_perturbed_max_abs_diff_u": max(float((a - b).abs().max()) for a, b in zip(base_u, u_last)),
                   "oldest_window_frame_perturbed_max_abs_diff_u": max(float((a - b).abs().max()) for a, b in zip(base_u, u_in)),
                   "u_at_earlier_frame_vs_current_max_abs_diff": float((u_a - u_c).abs().max())}
# c5
ti, tt = infer_u(batch), train_u(batch)
res["c5"] = {"forward_vs_forward_infer_max_abs_diff": max(float((a - b).abs().max()) for a, b in zip(ti, tt)), "n_calls": [len(ti), len(tt)]}
with torch.no_grad():
    wm = model.wm(adapt, batch["act"])
    o = wm["lead_out"][:, -1]
    p = torch.sigmoid(o[:, 0])
    gp = o[:, 1].clamp(0.0, M.LEAD_COND_GAP_MAX)
    u_head = torch.stack([p, p * gp], -1)
    ok = batch["frame_ok"].unfold(1, H, 1).all(-1)[:, -1]
    u_head = torch.where(ok[:, None], u_head, torch.zeros_like(u_head))
res["c5"]["u_vs_training_time_head_output_max_abs_diff"] = max(float((u_head - r).abs().max()) for r in ti)
res["c5"]["bar"] = 1e-5
ok4 = (res["ego"]["max_abs_diff_u"] == 0.0 and res["ego"]["max_abs_diff_u_training_forward"] == 0.0 and all(res["ego"]["includes"].values()) and res["frames_before_window"]["max_abs_diff_u"] == 0.0
       and res["frames_after_t"]["max_abs_diff_u"] == 0.0 and res["frames_after_t"]["u_nonzero_rows"] > 0 and n_nonzero > 0
       and res["controls"]["last_frame_perturbed_max_abs_diff_u"] > 0.0 and res["controls"]["oldest_window_frame_perturbed_max_abs_diff_u"] > 0.0 and res["controls"]["u_at_earlier_frame_vs_current_max_abs_diff"] > 0.0)
ok5 = res["c5"]["forward_vs_forward_infer_max_abs_diff"] <= 1e-5 and res["c5"]["u_vs_training_time_head_output_max_abs_diff"] <= 1e-5
res["c4_PASS"], res["c5_PASS"], res["PASS"] = bool(ok4), bool(ok5), bool(ok4 and ok5)
C.dump(f"{C.LC}/checks/c4_c5_time_vision_parity.json", res)
print(res)
