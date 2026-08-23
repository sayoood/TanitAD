"""Thor P6 — the RANDOM-WEIGHT CONTROL that stage 4 of `thor_c2_real_weights.py` dropped.

That cell died on a batch mismatch (engine built at B=1, fed B=16) and the JSON records the
failure. The control is not decoration: it is what makes the real-vs-random contrast a
WITHIN-SESSION measurement instead of a comparison across two sessions and two toolchains.

Reproduces the runbook §2 condition EXACTLY — randomly-initialised `WorldModel`, `torch.randn`
states and actions, TRT-fp16 predictor, rel-err at 1 step and after a 20-step recursive roll —
alongside a real-weight cell run in the SAME process on the SAME engine build path.
"""
import json
import os
import subprocess
import sys
import time

for _p in ("~/TanitAD/stack", "~/TanitAD/stack/scripts", "~/TanitAD/taniteval"):
    sys.path.insert(0, os.path.expanduser(_p))
sys.path.insert(0, "/usr/lib/python3.12/dist-packages")

import torch  # noqa: E402
import tensorrt as trt  # noqa: E402

DEV, WIN, K = "cuda", 8, 20
WORK = os.path.expanduser("~/trt_c2b")
os.makedirs(WORK, exist_ok=True)
OUT_JSON = os.path.expanduser("~/thor_c2b_random_control.json")
V1_CKPT = os.path.expanduser("~/models/flagship-v1-speedjerk/ckpt.pt")

from tanitad.config import flagship4b_config          # noqa: E402
from tanitad.models.fourbrain import WorldModel       # noqa: E402

OUT = {"purpose": "the random-weight control for the real-weight precision gate, in ONE process",
       "runbook_published_random_weights": {"trt_fp16_1step": 1.41e-3,
                                            "trt_fp16_20step": 1.80e-3, "growth_x": 1.3},
       "torch": torch.__version__, "trt": trt.__version__,
       "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def build(load):
    cfg = flagship4b_config()
    object.__setattr__(cfg.predictor, "action_dim", 3)
    if cfg.tactical_pred is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", 3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    m = WorldModel(cfg)
    if load:
        ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
    return m.to(DEV).eval(), cfg


class PredWrap(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, s, a):
        return self.p(s, a)[1]


class TRTPred:
    def __init__(self, plan):
        with open(plan, "rb") as f:
            self.e = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(f.read())
        self.c = self.e.create_execution_context()

    def __call__(self, s, a):
        s = s.contiguous().float()
        a = a.contiguous().float()
        self.c.set_input_shape("states", tuple(s.shape))
        self.c.set_input_shape("actions", tuple(a.shape))
        o = torch.empty(tuple(self.c.get_tensor_shape("z_next")), device=DEV)
        self.c.set_tensor_address("states", s.data_ptr())
        self.c.set_tensor_address("actions", a.data_ptr())
        self.c.set_tensor_address("z_next", o.data_ptr())
        assert self.c.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        return (None, o)


def relerr(a, b):
    a = a.double().flatten().cpu()
    b = b.double().flatten().cpu()
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def roll(fn, st, aw, fa, k):
    ws, wa, zs = st, aw, []
    for j in range(k):
        z = fn(ws, wa)[1]
        zs.append(z)
        if j < k - 1:
            ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], 1)
            wa = torch.cat([wa[:, 1:], fa[:, j].unsqueeze(1)], 1)
    return torch.stack(zs, 1)


def cell(tag, load_weights, B=8):
    m, cfg = build(load_weights)
    S, A = m.state_dim, cfg.predictor.action_dim
    op, pp = f"{WORK}/{tag}.onnx", f"{WORK}/{tag}.plan"
    torch.backends.mha.set_fastpath_enabled(False)
    st = torch.randn(B, WIN, S, device=DEV)
    ac = torch.randn(B, WIN, A, device=DEV)
    torch.onnx.export(PredWrap(m.predictor).eval(), (st, ac), op,
                      input_names=["states", "actions"], output_names=["z_next"],
                      opset_version=17, dynamo=False)
    r = subprocess.run(["/usr/src/tensorrt/bin/trtexec", f"--onnx={op}",
                        f"--saveEngine={pp}", "--fp16", "--skipInference"],
                       capture_output=True, text=True, timeout=3600)
    assert os.path.exists(pp), (r.stderr or r.stdout)[-400:]
    eng = TRTPred(pp)
    fa = torch.randn(B, K, A, device=DEV)
    with torch.no_grad():
        ze = roll(m.predictor, st, ac, fa, K)
        zt = roll(eng, st, ac, fa, K)
        e1 = relerr(zt[:, 0], ze[:, 0])
        e20 = relerr(zt[:, -1], ze[:, -1])
        per = []
        for i in range(B):
            a1 = relerr(zt[i:i + 1, 0], ze[i:i + 1, 0])
            a20 = relerr(zt[i:i + 1, -1], ze[i:i + 1, -1])
            per.append((a1, a20, a20 / a1))
    del m, eng
    torch.cuda.empty_cache()
    return {"weights": "REAL (step 29999)" if load_weights else "RANDOM init",
            "input": "torch.randn (the runbook's condition)",
            "batch": B, "rel_err_1step": round(e1, 8), "rel_err_20step": round(e20, 8),
            "growth_x": round(e20 / e1, 3),
            "per_window_growth_x_mean": round(sum(p[2] for p in per) / len(per), 3),
            "per_window_growth_x_max": round(max(p[2] for p in per), 3)}


OUT["RANDOM_weights_randn_input"] = cell("randw", False)
print("RANDOM", OUT["RANDOM_weights_randn_input"], flush=True)
OUT["REAL_weights_randn_input"] = cell("realw", True)
print("REAL/randn", OUT["REAL_weights_randn_input"], flush=True)

r, q = OUT["RANDOM_weights_randn_input"], OUT["REAL_weights_randn_input"]
OUT["READING"] = {
    "isolates": ("both cells use torch.randn INPUTS, so the ONLY thing that changes is the WEIGHT "
                 "distribution. Any difference is attributable to trained weights, not to the "
                 "input distribution."),
    "growth_random_x": r["growth_x"], "growth_real_x": q["growth_x"],
    "growth_ratio_real_over_random": round(q["growth_x"] / r["growth_x"], 2),
    "1step_ratio_real_over_random": round(q["rel_err_1step"] / r["rel_err_1step"], 3),
    "20step_ratio_real_over_random": round(q["rel_err_20step"] / r["rel_err_20step"], 3)}
OUT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
with open(OUT_JSON, "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT["READING"], indent=1), flush=True)
print("WROTE", OUT_JSON, flush=True)
