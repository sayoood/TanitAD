"""Thor P6 — THE OPTIMISATION AND THE PRECISION GATE, ON REAL TRAINED WEIGHTS.

⛔ WHY THIS RUN EXISTS. Every number in `THOR_DEPLOYMENT_RUNBOOK.md` §1/§2/§3 was measured on a
**randomly-initialised** `WorldModel` fed `torch.randn` — there is no `torch.load` in any of the
five Thor scripts. Latency is a property of shapes and kernels and survives that; **numerics are
not**. Quantisation error is a function of the TRAINED weight and activation distribution, and a
random network has no outlier channels — which are the entire difficulty. Our own paper §7.10
measured the OPPOSITE on real weights (+0.0215 m ADE@2s past a 0.02 m falsifier, degradation
growing 27x from 0.5 s to 2 s).

THIS SCRIPT LOADS REAL CHECKPOINTS AND FEEDS REAL VALIDATION FRAMES.

Pre-registered questions, each with its falsifier, BOTH outcomes committed in advance:

Q1 LATENCY-IS-WEIGHT-INDEPENDENT.  Re-measure every published stage on real weights.
    FALSIFIER: any stage differs from the published random-weight figure by > 10 %
    => the published 5.33x / 51.2 ms was weight-dependent after all and must be restated.
    (Expected: no change. Denormals are the only plausible mechanism and bf16/fp16 have none.)

Q2 THE PRECISION GATE ON REAL ACTIVATIONS.  The runbook reports TRT-fp16 rel-err 1.41e-3 (1 step)
    -> 1.80e-3 (20 steps) on `torch.randn`. Re-measure with REAL encoded states from REAL val
    windows, rolled under the expert's REAL actions.
    FALSIFIER: real-weight rel-err > 10x the random-weight figure, or growth ratio > 2x
    => the published gate is not transferable and fp16 needs an accuracy gate before it ships.

Q3 THE ENCODER'S bf16 AUTOCAST — the single largest lever (6.76x) and never checked for numerics
    at all, at any weight init. Measure fp32-vs-bf16 on real frames: rel-err AND cosine of the
    latent, because a latent that keeps its norm but rotates is exactly the failure a rel-err on
    a random net cannot show.
    FALSIFIER: cosine < 0.999 on real windows => bf16 encoder is not free and O2 (TRT encoder)
    stops being an optimisation and becomes the fallback.

Q4 THE MHA FASTPATH.  The 2026-08-03 annotation says the flag is INERT (0.726 did not reproduce)
    — but that probe was ALSO on random weights. A wrong graph shows itself through the weights it
    multiplies. Re-run the 4 predictor cells (opset 17/18 x fastpath ON/OFF) on REAL weights.
    FALSIFIER: any cell's ORT-vs-eager rel-err > 1e-4 => the mechanism is real and weight-gated,
    and the annotation must be narrowed to "inert on random weights".

⚠️ CONTROL, not decoration: the RANDOM-weight cells are re-run IN THIS PROCESS so the real-vs-random
contrast is measured under one toolchain, not compared across sessions.
"""
import json
import os
import subprocess
import sys
import time
import dataclasses
from types import SimpleNamespace

for _p in ("~/TanitAD/stack", "~/TanitAD/stack/scripts", "~/TanitAD/taniteval"):
    sys.path.insert(0, os.path.expanduser(_p))
sys.path.insert(0, "/usr/lib/python3.12/dist-packages")   # TRT bindings are system-level

import torch  # noqa: E402

DEV = "cuda"
OUT_JSON = os.path.expanduser("~/thor_c2_real_weights.json")
WORK = os.path.expanduser("~/trt_c2")
os.makedirs(WORK, exist_ok=True)

V1_CKPT = os.path.expanduser("~/models/flagship-v1-speedjerk/ckpt.pt")
V5F_CKPT = os.path.expanduser("~/models/v5f/ckpt.pt")
V1_VAL = os.path.expanduser("~/valdata/physicalai-val-0c5f7dac3b11")
K_ROLL, M_FAN, WIN = 20, 9, 8

OUT = {"purpose": "the Thor optimisation + precision gate ON REAL TRAINED WEIGHTS",
       "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
try:
    import tensorrt as trt
    OUT["trt_version"] = trt.__version__
except Exception as e:
    trt = None
    OUT["trt_version"] = f"IMPORT FAILED {e}"
OUT["thor_repo_sha"] = subprocess.run(
    ["git", "-C", os.path.expanduser("~/TanitAD"), "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True).stdout.strip()


def bank(tag=""):
    with open(OUT_JSON, "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    print(f"[bank] {tag} -> {OUT_JSON}", flush=True)


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"p50_ms": round(ts[len(ts) // 2], 3),
            "p99_ms": round(ts[min(len(ts) - 1, int(len(ts) * 0.99))], 3), "n": iters}


def relerr(a, b):
    """||a-b|| / ||b||, float64, on cpu."""
    a = torch.as_tensor(a).double().flatten().cpu()
    b = torch.as_tensor(b).double().flatten().cpu()
    return float(torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-30))


# ===================================================================== MODEL BUILDERS
from tanitad.config import flagship4b_config              # noqa: E402
from tanitad.models.fourbrain import WorldModel           # noqa: E402
from tanitad.models.metric_dynamics import HierarchicalGrounding  # noqa: E402


def build_v1(load_weights=True):
    """flagship-v1-speedjerk — 256x256 SQUARE, step 29999, the DEPLOYED v1."""
    cfg = flagship4b_config()
    object.__setattr__(cfg.predictor, "action_dim", 3)     # --speed-input
    if cfg.tactical_pred is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", 3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    m = WorldModel(cfg)
    g = HierarchicalGrounding(m.state_dim)
    meta = {"loaded": False}
    if load_weights:
        ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])                     # STRICT
        g.load_state_dict(ck["grounding"])                 # STRICT
        meta = {"loaded": True, "step": int(ck.get("step", -1)),
                "ckpt": V1_CKPT,
                "w_absmax": round(float(max(p.abs().max() for p in m.parameters())), 4),
                "w_std": round(float(torch.cat([p.flatten() for p in m.parameters()]).std()), 6)}
    meta["params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 2)
    meta["H"], meta["W"] = cfg.encoder.image_size, cfg.encoder.image_size
    meta["in_channels"] = cfg.encoder.in_channels
    return m.to(DEV).eval(), g.to(DEV).eval(), cfg, meta


def build_v5f(load_weights=True):
    """v5f — 429-token 176x624 / 117 deg cylindrical, the DEPLOYED geometry the runbook timed."""
    from train_flagship_v4 import resolve_v2_frames
    cfg = flagship4b_config()
    ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                         projection="cylindrical", v2_subframe="176x624", f_ref=None)
    resolve_v2_frames(ns, cfg, label="c2")
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    m = WorldModel(cfg)
    meta = {"loaded": False}
    if load_weights:
        ck = torch.load(V5F_CKPT, map_location="cpu", weights_only=False)
        miss, unexp = m.load_state_dict(ck["model"], strict=False)
        meta = {"loaded": True, "step": int(ck.get("step", -1)), "ckpt": V5F_CKPT,
                "sd_missing": len(miss), "sd_unexpected": len(unexp),
                "sd_missing_sample": list(miss)[:5], "sd_unexpected_sample": list(unexp)[:5],
                "w_absmax": round(float(max(p.abs().max() for p in m.parameters())), 4),
                "w_std": round(float(torch.cat([p.flatten() for p in m.parameters()]).std()), 6)}
    meta["params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 2)
    meta["H"], meta["W"] = 176, 624
    meta["in_channels"] = cfg.encoder.in_channels
    return m.to(DEV).eval(), cfg, meta


# ===================================================================== TRT RUNTIME
class TRTPredictor:
    """A drop-in for ``model.predictor``: returns a tuple whose [1] is z_next, so it plugs
    straight into ``rollout_decode`` / ``imagine`` without touching the eval harness.

    ⛔ Binds by NAME ('states','actions','z_next'), never by index — the runbook's own rule."""

    def __init__(self, plan_path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(plan_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"failed to deserialize {plan_path}"
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i)
                      for i in range(self.engine.num_io_tensors)]
        for n in ("states", "actions", "z_next"):
            assert n in self.names, f"engine IO {self.names} lacks {n!r}"
        self.stream = torch.cuda.current_stream().cuda_stream

    def __call__(self, states, actions):
        st = states.contiguous().float()
        ac = actions.contiguous().float()
        self.ctx.set_input_shape("states", tuple(st.shape))
        self.ctx.set_input_shape("actions", tuple(ac.shape))
        oshape = tuple(self.ctx.get_tensor_shape("z_next"))
        out = torch.empty(oshape, device=DEV, dtype=torch.float32)
        self.ctx.set_tensor_address("states", st.data_ptr())
        self.ctx.set_tensor_address("actions", ac.data_ptr())
        self.ctx.set_tensor_address("z_next", out.data_ptr())
        ok = self.ctx.execute_async_v3(self.stream)
        assert ok, "TRT execute_async_v3 returned False"
        torch.cuda.current_stream().synchronize()
        return (None, out)


class PredWrap(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, states, actions):
        return self.p(states, actions)[1]


def export_onnx(pred, S, A, B, path, fastpath=False, opset=17):
    torch.backends.mha.set_fastpath_enabled(fastpath)
    st = torch.randn(B, WIN, S, device=DEV)
    ac = torch.randn(B, WIN, A, device=DEV)
    wrap = PredWrap(pred).eval()
    with torch.no_grad():
        ref = wrap(st, ac)
    torch.onnx.export(wrap, (st, ac), path, input_names=["states", "actions"],
                      output_names=["z_next"], opset_version=opset, dynamo=False)
    torch.backends.mha.set_fastpath_enabled(False)
    return (st, ac, ref)


def build_engine(onnx_path, plan_path, fp16=True):
    t0 = time.perf_counter()
    cmd = ["/usr/src/tensorrt/bin/trtexec", f"--onnx={onnx_path}",
           f"--saveEngine={plan_path}", "--skipInference"]
    if fp16:
        cmd.append("--fp16")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if not os.path.exists(plan_path):
        return {"ok": False, "err": (r.stderr or r.stdout)[-400:]}
    return {"ok": True, "build_s": round(time.perf_counter() - t0, 1),
            "MB": round(os.path.getsize(plan_path) / 1e6, 1)}


# ===================================================================== STAGE 1: models
print("=== STAGE 1: load REAL weights ===", flush=True)
v1, v1g, v1cfg, v1meta = build_v1(True)
OUT["v1_flagship_speedjerk"] = v1meta
print("v1", v1meta, flush=True)
v5f, v5cfg, v5meta = build_v5f(True)
OUT["v5f"] = v5meta
print("v5f", v5meta, flush=True)
OUT["⚠_v5f_caveat"] = ("v5f on Thor is step %s — REAL trained weights but EARLY. Its numerics are "
                       "a second weight distribution (a control), not a deployment verdict. The "
                       "decision-grade accuracy arm is v1 @ step %s."
                       % (v5meta.get("step"), v1meta.get("step")))
bank("stage1")

# ===================================================================== STAGE 2: real activations
print("=== STAGE 2: REAL validation windows ===", flush=True)
from taniteval import data as TD                          # noqa: E402
from taniteval import rollout as RO                       # noqa: E402

# ⛔ VERIFY BY LOADING, NOT BY EXIT CODE / SIZE. One clip (ep_00028.pt, 92.3 MB against the
# cohort's ~117 MB) arrived TRUNCATED from the pod relay and reads as
# `PytorchStreamReader ... failed finding central directory`. It cannot be re-pulled: the HF
# transfer repo `Sayood/tanitad-transfer-2026-08` carries only the 256x640cyl v2ep val, not the
# 256px ep_*.pt cache — runbook §11's corpus-durability rule, live.
import glob as _glob                                       # noqa: E402
_all = sorted(_glob.glob(os.path.join(V1_VAL, "ep_*.pt")))
_bad = []
for _f in _all:
    try:
        _d = torch.load(_f, map_location="cpu", weights_only=True, mmap=True)
        _ = _d["frames_u8"].shape
    except Exception as _e:
        _bad.append({"file": os.path.basename(_f), "bytes": os.path.getsize(_f),
                     "err": type(_e).__name__})
OUT["val_integrity"] = {
    "n_listed": len(_all), "n_loadable": len(_all) - len(_bad), "corrupt": _bad,
    "policy": ("verified by LOADING each clip, not by size or exit code. Corrupt clips are "
               "EXCLUDED and named. The run is stamped decision_grade=False for ABSOLUTE levels; "
               "the precision DELTA is PAIRED on identical windows in both arms and is therefore "
               "unaffected by the missing episode.")}
_badnames = {b["file"] for b in _bad}
files = TD.list_val_episodes(V1_VAL, n=40, allow_partial=True)
OUT["val_parity"] = TD.last_val_parity()
files = [f for f in files if os.path.basename(str(f)) not in _badnames]
eps = TD.load_frames(files)
OUT["val"] = {"dir": V1_VAL, "n_episodes": len(eps),
              "frame_shape": list(eps[0].feats.shape[1:]),
              "decision_grade_absolute": len(_bad) == 0,
              "note": "CLEAN held-out split; v1's TRAINED raster 256x256 — parity respected"}
print("val", OUT["val"], flush=True)

N_PROBE = 64          # real windows used for the numerics probe
probe = {"fw": [], "aw": [], "fa": []}
for ep in eps:
    T = min(ep.feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
    starts = list(range(0, T - WIN - K_ROLL, 64))[:2]
    for t in starts:
        if len(probe["fw"]) >= N_PROBE:
            break
        probe["fw"].append(torch.as_tensor(ep.feats[t:t + WIN]).float().div(255.0))
        last = torch.tensor([t + WIN - 1])
        aw = ep.actions[t:t + WIN][None].to(DEV)
        fa = ep.actions[t + WIN:t + WIN + K_ROLL][None].to(DEV)
        aw, fa = RO.append_ego(aw, fa, ep.poses, last, True, False, False, DEV)
        probe["aw"].append(aw[0].cpu())
        probe["fa"].append(fa[0].cpu())
    if len(probe["fw"]) >= N_PROBE:
        break
FW = torch.stack(probe["fw"]).to(DEV)             # [N, W, 9, 256, 256] real frames
AW = torch.stack(probe["aw"]).to(DEV)             # [N, W, 3] real actions (+v0)
FA = torch.stack(probe["fa"]).to(DEV)             # [N, K, 3]
OUT["probe"] = {"n_windows": int(FW.shape[0]), "frames": list(FW.shape),
                "actions": list(AW.shape), "future_actions": list(FA.shape),
                "source": "REAL val frames + REAL expert actions (NOT torch.randn)"}
print("probe", OUT["probe"], flush=True)
bank("stage2")

# ===================================================================== STAGE 3: Q3 encoder bf16
print("=== STAGE 3 (Q3): encoder fp32 vs bf16 on REAL frames ===", flush=True)
import torch.nn.functional as F                            # noqa: E402


def encoder_numerics(model, fw, tag):
    with torch.no_grad():
        s32 = model.encode_window(fw).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sbf = model.encode_window(fw)
        sbf = sbf.float()
    cos = F.cosine_similarity(s32.flatten(0, 1), sbf.flatten(0, 1), dim=-1)
    rel_per = (s32 - sbf).norm(dim=-1) / (s32.norm(dim=-1) + 1e-30)
    return {"tag": tag, "rel_err_global": round(relerr(sbf, s32), 8),
            "latent_cosine_mean": round(float(cos.mean()), 8),
            "latent_cosine_min": round(float(cos.min()), 8),
            "latent_cosine_p01": round(float(cos.quantile(0.01)), 8),
            "rel_err_per_state_mean": round(float(rel_per.mean()), 8),
            "rel_err_per_state_max": round(float(rel_per.max()), 8),
            "state_absmax_fp32": round(float(s32.abs().max()), 4),
            "state_std_fp32": round(float(s32.std()), 6),
            # outlier-channel census: THE mechanism a random net cannot have
            "n_channels_over_10sigma": int((s32.flatten(0, 1).abs().max(0).values
                                            > 10 * s32.std()).sum()),
            "n_states": int(s32.shape[0] * s32.shape[1])}


enc = {}
enc["v1_REAL_weights_real_frames"] = encoder_numerics(v1, FW, "v1 real/real")
# CONTROL 1: real weights, random input -> isolates the input distribution
enc["v1_REAL_weights_randn_input"] = encoder_numerics(v1, torch.randn_like(FW), "v1 real/randn")
# CONTROL 2: random weights, random input -> exactly what the runbook measured
v1r, _, _, _ = build_v1(False)
enc["v1_RANDOM_weights_randn_input"] = encoder_numerics(v1r, torch.randn_like(FW), "v1 rand/randn")
del v1r
torch.cuda.empty_cache()
OUT["Q3_encoder_bf16_numerics"] = enc
c = enc["v1_REAL_weights_real_frames"]["latent_cosine_min"]
OUT["Q3_VERDICT"] = {
    "falsifier": "latent cosine < 0.999 on real windows",
    "measured_min_cosine": c,
    "fires": bool(c < 0.999),
    "reading": ("⛔ bf16 encoder ROTATES the latent on real data — not free"
                if c < 0.999 else
                "✅ bf16 encoder preserves the latent direction on real data")}
print(json.dumps(OUT["Q3_encoder_bf16_numerics"], indent=1), flush=True)
bank("stage3")

# ===================================================================== STAGE 4: Q2 predictor TRT
print("=== STAGE 4 (Q2): predictor TRT fp16/fp32 vs eager, REAL states + REAL actions ===",
      flush=True)
S1, A1 = v1.state_dim, v1cfg.predictor.action_dim
with torch.no_grad():
    STATES = v1.encode_window(FW).float()                  # REAL encoded states


def roll_ref(pred_fn, states, aw, fa, k):
    """The EXACT roll rollout_decode performs, returning every intermediate latent."""
    win_s, win_a, zs = states, aw, []
    for j in range(k):
        z = pred_fn(win_s, win_a)[1]
        zs.append(z)
        if j < k - 1:
            win_s = torch.cat([win_s[:, 1:], z.unsqueeze(1)], dim=1)
            win_a = torch.cat([win_a[:, 1:], fa[:, j].unsqueeze(1)], dim=1)
    return torch.stack(zs, 1)                              # [N, k, S]


q2 = {}
if trt is not None:
    for prec in ("fp16", "fp32"):
        try:
            onnx_p = f"{WORK}/v1_pred_b1_{prec}.onnx"
            plan_p = f"{WORK}/v1_pred_b1_{prec}.plan"
            export_onnx(v1.predictor, S1, A1, 1, onnx_p, fastpath=False, opset=17)
            binfo = build_engine(onnx_p, plan_p, fp16=(prec == "fp16"))
            q2[prec] = {"engine": binfo}
            if not binfo["ok"]:
                continue
            eng = TRTPredictor(plan_p)
            with torch.no_grad():
                errs1, errs20, growth = [], [], []
                for i in range(int(STATES.shape[0])):
                    st = STATES[i:i + 1]
                    aw1, fa1 = AW[i:i + 1], FA[i:i + 1]
                    z_eager = roll_ref(v1.predictor, st, aw1, fa1, K_ROLL)
                    z_trt = roll_ref(eng, st, aw1, fa1, K_ROLL)
                    e1 = relerr(z_trt[:, 0], z_eager[:, 0])
                    e20 = relerr(z_trt[:, -1], z_eager[:, -1])
                    errs1.append(e1)
                    errs20.append(e20)
                    growth.append(e20 / (e1 + 1e-30))
            t1 = torch.tensor(errs1)
            t20 = torch.tensor(errs20)
            q2[prec].update({
                "rel_err_1step_mean": round(float(t1.mean()), 8),
                "rel_err_1step_p99": round(float(t1.quantile(0.99)), 8),
                "rel_err_1step_max": round(float(t1.max()), 8),
                "rel_err_20step_mean": round(float(t20.mean()), 8),
                "rel_err_20step_p99": round(float(t20.quantile(0.99)), 8),
                "rel_err_20step_max": round(float(t20.max()), 8),
                "growth_mean_x": round(float(torch.tensor(growth).mean()), 3),
                "growth_max_x": round(float(torch.tensor(growth).max()), 3),
                "n_windows": len(errs1)})
            print(prec, q2[prec], flush=True)
            del eng
        except Exception as e:
            q2[prec] = {"FAILED": f"{type(e).__name__}: {str(e)[:300]}"}
            print("Q2", prec, "FAILED", e, flush=True)

    # CONTROL: the runbook's own condition — random weights, torch.randn states
    try:
        v1r, _, _, _ = build_v1(False)
        onnx_p = f"{WORK}/rand_pred_b1_fp16.onnx"
        plan_p = f"{WORK}/rand_pred_b1_fp16.plan"
        export_onnx(v1r.predictor, S1, A1, 1, onnx_p, fastpath=False, opset=17)
        binfo = build_engine(onnx_p, plan_p, fp16=True)
        if binfo["ok"]:
            eng = TRTPredictor(plan_p)
            st = torch.randn(16, WIN, S1, device=DEV)
            aw1 = torch.randn(16, WIN, A1, device=DEV)
            fa1 = torch.randn(16, K_ROLL, A1, device=DEV)
            with torch.no_grad():
                ze = roll_ref(v1r.predictor, st, aw1, fa1, K_ROLL)
                zt = roll_ref(eng, st, aw1, fa1, K_ROLL)
            e1 = relerr(zt[:, 0], ze[:, 0])
            e20 = relerr(zt[:, -1], ze[:, -1])
            q2["CONTROL_random_weights_randn"] = {
                "rel_err_1step": round(e1, 8), "rel_err_20step": round(e20, 8),
                "growth_x": round(e20 / (e1 + 1e-30), 3),
                "runbook_published": {"1step": 1.41e-3, "20step": 1.80e-3, "growth": 1.3}}
            del eng
        del v1r
        torch.cuda.empty_cache()
    except Exception as e:
        q2["CONTROL_random_weights_randn"] = {"FAILED": f"{type(e).__name__}: {str(e)[:250]}"}
OUT["Q2_predictor_trt_numerics_REAL"] = q2

pub1, pub20 = 1.41e-3, 1.80e-3
m1 = q2.get("fp16", {}).get("rel_err_1step_mean")
m20 = q2.get("fp16", {}).get("rel_err_20step_mean")
if m1 and m20:
    OUT["Q2_VERDICT"] = {
        "falsifier": "real-weight rel-err > 10x the published random-weight figure, "
                     "or growth ratio > 2x",
        "published_random_weights": {"1step": pub1, "20step": pub20, "growth": 1.3},
        "measured_real_weights": {"1step": m1, "20step": m20,
                                  "growth": q2["fp16"]["growth_mean_x"]},
        "ratio_1step_x": round(m1 / pub1, 2), "ratio_20step_x": round(m20 / pub20, 2),
        "fires": bool(m1 > 10 * pub1 or m20 > 10 * pub20
                      or q2["fp16"]["growth_mean_x"] > 2.6)}
bank("stage4")

# ===================================================================== STAGE 5: Q1 latency
print("=== STAGE 5 (Q1): latency on REAL weights, both geometries ===", flush=True)
lat = {}
with torch.no_grad():
    for tag, model, meta in (("v1_256x256", v1, v1meta), ("v5f_176x624", v5f, v5meta)):
        H, W = meta["H"], meta["W"]
        C = meta["in_channels"]
        Wn = 8
        fr = torch.randn(1, Wn, C, H, W, device=DEV)
        S = model.state_dim
        A = 3
        st1 = model.encode_window(fr).float()
        r = {"geometry": f"{H}x{W}", "params_M": meta["params_M"], "step": meta.get("step")}
        r["encoder_fp32"] = bench(lambda: model.encode_window(fr), 3, 15)

        def _enc_bf16():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.encode_window(fr)
        r["encoder_bf16"] = bench(_enc_bf16, 3, 15)
        r["encoder_speedup_x"] = round(r["encoder_fp32"]["p50_ms"] /
                                       r["encoder_bf16"]["p50_ms"], 3)
        for B in (1, M_FAN):
            stB = st1.expand(B, -1, -1).contiguous()
            acB = torch.randn(B, Wn, A, device=DEV)
            r[f"predictor_eager_b{B}"] = bench(lambda: model.predictor(stB, acB), 10, 50)
        ac1 = torch.randn(1, Wn, A, device=DEV)
        r["roll_K20_eager_b1"] = bench(
            lambda: roll_ref(model.predictor, st1, ac1.expand(1, Wn, A).contiguous(),
                             torch.randn(1, K_ROLL, A, device=DEV), K_ROLL), 2, 8)
        lat[tag] = r
        print(tag, json.dumps(r), flush=True)
OUT["Q1_latency_REAL_weights"] = lat

# random-weight latency control at the SAME geometry, same process
try:
    v5r, _, _ = build_v5f(False)
    fr = torch.randn(1, 8, v5cfg.encoder.in_channels, 176, 624, device=DEV)
    with torch.no_grad():
        v5r.encode_window(fr)

        def _e32():
            return v5r.encode_window(fr)

        def _ebf():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return v5r.encode_window(fr)
        st1 = v5r.encode_window(fr).float()
        ac1 = torch.randn(1, 8, 3, device=DEV)
        OUT["Q1_latency_RANDOM_weights_control"] = {
            "geometry": "176x624",
            "encoder_fp32": bench(_e32, 3, 15), "encoder_bf16": bench(_ebf, 3, 15),
            "predictor_eager_b1": bench(lambda: v5r.predictor(st1, ac1), 10, 50)}
    del v5r
    torch.cuda.empty_cache()
except Exception as e:
    OUT["Q1_latency_RANDOM_weights_control"] = {"FAILED": f"{type(e).__name__}: {str(e)[:250]}"}
bank("stage5")

# ===================================================================== STAGE 5b: TRT latency real
print("=== STAGE 5b: TRT engine latency from REAL weights (b1 + b9) ===", flush=True)
trtlat = {}
if trt is not None:
    for tag, model, meta in (("v5f_176x624", v5f, v5meta), ("v1_256x256", v1, v1meta)):
        S = model.state_dim
        for B in (1, M_FAN):
            key = f"{tag}_b{B}"
            try:
                op = f"{WORK}/{key}.onnx"
                pp = f"{WORK}/{key}.plan"
                export_onnx(model.predictor, S, 3, B, op, fastpath=False, opset=17)
                info = build_engine(op, pp, fp16=True)
                if info["ok"]:
                    r2 = subprocess.run(["/usr/src/tensorrt/bin/trtexec", f"--loadEngine={pp}",
                                         "--iterations=200", "--warmUp=500", "--avgRuns=100"],
                                        capture_output=True, text=True, timeout=1800)
                    for line in (r2.stdout or "").splitlines():
                        if "GPU Compute Time" in line and "median" in line:
                            for part in line.split(","):
                                if "median" in part:
                                    info["median_ms"] = float(part.split("=")[1].strip().split()[0])
                    if B == M_FAN and info.get("median_ms"):
                        info["per_candidate_ms"] = round(info["median_ms"] / M_FAN, 4)
                trtlat[key] = info
                print(key, info, flush=True)
            except Exception as e:
                trtlat[key] = {"FAILED": f"{type(e).__name__}: {str(e)[:250]}"}
OUT["Q1_trt_latency_REAL_weights"] = trtlat
bank("stage5b")

# ===================================================================== STAGE 6: Q4 fastpath
print("=== STAGE 6 (Q4): MHA fastpath cells on REAL weights ===", flush=True)
q4 = {"mha_module_census": {
    "predictor": sum(1 for m in v1.predictor.modules()
                     if isinstance(m, torch.nn.MultiheadAttention)),
    "encoder": sum(1 for m in v1.encoder.modules()
                   if isinstance(m, torch.nn.MultiheadAttention)),
    "whole_model": sum(1 for m in v1.modules()
                       if isinstance(m, torch.nn.MultiheadAttention))}}
try:
    import onnx
    import onnxruntime as ort
    have_ort = True
except Exception as e:
    have_ort = False
    q4["ort"] = f"unavailable: {e}"

for opset in (17, 18):
    for fastpath in (True, False):
        key = f"REALW_predictor_op{opset}_fastpath{'ON' if fastpath else 'OFF'}"
        path = f"{WORK}/fp_{opset}_{int(fastpath)}.onnx"
        cell = {"opset": opset, "fastpath": fastpath}
        try:
            st, ac, ref = export_onnx(v1.predictor, S1, A1, 1, path,
                                      fastpath=fastpath, opset=opset)
            cell["export_ok"] = True
            if have_ort:
                g = onnx.load(path, load_external_data=False)
                ops = {n.op_type for n in g.graph.node}
                cell["n_nodes"] = len(g.graph.node)
                cell["has_fused_mha_op"] = any("MultiHeadAttention" in o
                                               or "native_multi_head" in o for o in ops)
                sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                y = sess.run(["z_next"], {"states": st.cpu().numpy(),
                                          "actions": ac.cpu().numpy()})[0]
                cell["ort_rel_err_vs_eager_RANDN_input"] = round(relerr(y, ref), 10)
                # ⭐ THE NEW VARIABLE: a REAL encoded state, not torch.randn
                st_r = STATES[:1].contiguous()
                ac_r = AW[:1].contiguous()
                with torch.no_grad():
                    ref_r = PredWrap(v1.predictor)(st_r, ac_r)
                y_r = sess.run(["z_next"], {"states": st_r.cpu().numpy(),
                                            "actions": ac_r.cpu().numpy()})[0]
                cell["ort_rel_err_vs_eager_REAL_input"] = round(relerr(y_r, ref_r), 10)
        except Exception as e:
            cell["export_ok"] = False
            cell["err"] = f"{type(e).__name__}: {str(e)[:220]}"
        q4[key] = cell
        print(key, cell, flush=True)
torch.backends.mha.set_fastpath_enabled(False)

errs = [v.get("ort_rel_err_vs_eager_REAL_input") for k, v in q4.items()
        if isinstance(v, dict) and v.get("ort_rel_err_vs_eager_REAL_input") is not None]
OUT["Q4_fastpath_REAL_weights"] = q4
OUT["Q4_VERDICT"] = {
    "falsifier": "any cell's ORT-vs-eager rel-err > 1e-4 on REAL weights",
    "max_rel_err_real_input": max(errs) if errs else None,
    "fires": bool(errs and max(errs) > 1e-4),
    "reading": (None if not errs else
                ("⛔ the fastpath mechanism IS real and weight-gated — the 2026-08-03 "
                 "'inert' annotation must be narrowed to random weights"
                 if max(errs) > 1e-4 else
                 "✅ the fastpath flag is inert on REAL weights too — the 0.726 was NOT this "
                 "mechanism; a second, independent weight distribution now supports the "
                 "wiring-bug explanation"))}
bank("stage6")

OUT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
bank("DONE")
print("=== DONE ===", flush=True)
