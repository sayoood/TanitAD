"""H1 branch of G12 on CUDA: why are two identical head-less 2-step scorer runs (X1, X2) not bit-identical?  (A5, second failure: A40 2026-10-14 23:31Z)

Replicates scorer_finetune.py g12_probe's `run_arm` (deepcopy scorer, AdamW lr 3e-5, two steps of [fwd under autocast bf16 -> BCE -> backward -> clip 1.0 -> step]) with the REAL trained scorer
(model_final.pt) and the REAL head, at pod-like shapes (B=2, K=64, N=7680 visual tokens).  One variant per CLI flag; env vars that must precede CUDA init are set BEFORE `import torch`.

  python probe_d_h1_determinism.py [--cublas CFG] [--fp32] [--sdpa math|efficient|flash|cudnn] [--det warn|strict|off] [--late-cublas] [--tag NAME] [--ctx N]
-> one JSON line on stdout and  probe_d_<tag>.json.
"""
import argparse
import copy
import json
import os
import sys
import warnings

ap = argparse.ArgumentParser()
ap.add_argument("--cublas", default="")
ap.add_argument("--late-cublas", action="store_true", help="the trainer's order: os.environ.setdefault AFTER the CUDA context exists")
ap.add_argument("--fp32", action="store_true")
ap.add_argument("--sdpa", default="")
ap.add_argument("--det", default="warn", choices=("warn", "strict", "off"))
ap.add_argument("--tag", default="x")
ap.add_argument("--ctx", type=int, default=7680)
ap.add_argument("--batch", type=int, default=2)
ns = ap.parse_args()
if ns.cublas:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ns.cublas        # before torch is imported: before any CUDA context

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "refe")))
from model import REFe, REFeConfig, CrossBlock  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402

dev = "cuda"
torch.zeros(1, device=dev)                                # the CUDA context now exists
CKPT = "D:/Projects/TanitAD-artifacts/refe-final-2026-10-01/model_final.pt"


class ScorerCopy(nn.Module):
    def __init__(self, sd):
        super().__init__()
        c = REFeConfig()
        self.score_dec = nn.ModuleList([CrossBlock(c.dec_width, c.dec_heads, c.dec_width) for _ in range(c.dec_depth)])
        self.score_q_mlp = nn.Sequential(nn.Linear(60, 256), nn.GELU(), nn.Linear(256, 256))
        self.score_head = nn.Linear(256, 6)
        self.load_state_dict({k: v for k, v in sd.items() if k.split(".")[0] in ("score_q_mlp", "score_dec", "score_head")}, strict=True)

    def forward(self, trajs, sctx):
        return REFe.score_trajectories(self, trajs, sctx)


sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)["model"]
armB = ScorerCopy(sd).to(dev)
aux = AUX.AuxDrivable(AUX.AuxDrivableConfig(feed="off", w_aux=0.1, seed=0)).to(dev)
g = torch.Generator().manual_seed(0)
B, K, N = ns.batch, 64, ns.ctx
t = torch.arange(1, 21, dtype=torch.float32) * 0.2
v, lat = 6.0 + 8.0 * torch.rand(B, K, 1, generator=g), (torch.rand(B, K, 1, generator=g) - 0.5) * 0.6
cxy_d = torch.stack([(v * t).expand(B, K, 20), (lat * t ** 2 / 2).expand(B, K, 20), torch.atan2(lat * t, v).expand(B, K, 20)], -1).to(dev)
sctx_d = (torch.randn(B, N, 256, generator=g) * 1.0).to(dev)
tg = (torch.rand(B, K, 6, generator=g) > 0.5).float().to(dev)
lab = torch.randint(-80, 80, (B, 160, 128), generator=g, dtype=torch.int8).to(dev)
valid = torch.ones(B, dtype=torch.bool, device=dev)


def run_arm(with_head):
    S = copy.deepcopy(armB)
    for p_ in S.parameters():
        p_.requires_grad_(True)
    oS = torch.optim.AdamW(list(S.parameters()), lr=3e-5, weight_decay=0.01)
    H = oH = None
    if with_head:
        H = copy.deepcopy(aux)
        oH = torch.optim.AdamW(list(H.head.parameters()), lr=2e-4, weight_decay=0.01)
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not ns.fp32):
            sx = S(cxy_d, sctx_d)
            if with_head:
                lg_, sd_ = H.predict(sctx_d)
        torch.nn.functional.binary_cross_entropy_with_logits(sx.float(), tg).backward()
        if with_head:
            H.combine(torch.zeros((), device=dev), H.loss(lg_, sd_, lab, valid)[0]).backward()
            torch.nn.utils.clip_grad_norm_(list(H.head.parameters()), 1.0)
            oH.step()
            oH.zero_grad(set_to_none=True)
        torch.nn.utils.clip_grad_norm_(list(S.parameters()), 1.0)
        oS.step()
        oS.zero_grad(set_to_none=True)
    return S


def diff(P, Q):
    n_bad, tot, mx, bad_names = 0, 0, 0.0, []
    for (nm, p_), q_ in zip(P.named_parameters(), Q.parameters()):
        tot += 1
        if not torch.equal(p_, q_):
            n_bad += 1
            bad_names.append(nm)
            mx = max(mx, float((p_ - q_).abs().max()))
    return {"params": tot, "params_differ": n_bad, "max_abs_diff": mx, "first_differing": bad_names[:4]}


def go():
    if ns.sdpa:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        be = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}[ns.sdpa]
        ctxm = sdpa_kernel(be)
    else:
        import contextlib
        ctxm = contextlib.nullcontext()
    with ctxm:
        X1 = run_arm(False)
        X2 = run_arm(False)
        Y = run_arm(True)
    return X1, X2, Y


out = {"tag": ns.tag, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "flags": vars(ns), "env_CUBLAS_WORKSPACE_CONFIG_at_start": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
was = torch.are_deterministic_algorithms_enabled()
if ns.late_cublas:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if ns.det != "off":
    torch.use_deterministic_algorithms(True, warn_only=(ns.det == "warn"))
ws = []
try:
    with warnings.catch_warnings(record=True) as w_:
        warnings.simplefilter("always")
        X1, X2, Y = go()
    ws = sorted({str(x.message)[:230] for x in w_})
    out["x1_vs_x2"], out["x1_vs_y"] = diff(X1, X2), diff(X1, Y)
    out["x1_eq_x2"], out["x1_eq_y"] = out["x1_vs_x2"]["params_differ"] == 0, out["x1_vs_y"]["params_differ"] == 0
except Exception as e:                                                               # noqa: BLE001
    out["raised"] = f"{type(e).__name__}: {str(e)[:400]}"
finally:
    torch.use_deterministic_algorithms(was)
out["warnings"] = ws
open(os.path.join(HERE, f"probe_d_{ns.tag}.json"), "w").write(json.dumps(out, indent=1))
print(json.dumps(out))
