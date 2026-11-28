"""Cross-version BIT-IDENTITY dump (PI 2026-09-27 item 1: "prove bit-identical-when-off").

Run the SAME script under two stacks -- the unmodified c36b6ddd `stack/` and the R1-R6 copy --
and compare the two dumps with `bitident_compare.py`. Everything here uses ONLY pre-directive
APIs (no new flag, no new kwarg), so both trees run it unchanged:

  A. MODEL: 9 configs x (init state_dict, forward on a fixed batch WITH labels / nav / v0 /
     str-ext targets -> every output, backward -> every parameter gradient, one AdamW step ->
     the new state_dict, and an inference forward without targets);
  B. PLAN: `RefAV1.plan()` on one window of a tiny model (small iCEM), three call variants ->
     controls / cost / source / every baseline + fine cost / the result's attribute set;
  C. LOADER: `RefAV1Windows` on 3 REAL eval episodes (v7.2 labels + nav), 3 option sets x 5
     batches -> every tensor;
  D. TRAINER: `refa_v1_train.py --smoke` 4 steps (+ `--no-hierarchy`, + `--speed-channel
     --ema-targets`) -> every log row (minus wall-clock) and the final checkpoint's model and
     optimizer state.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stack", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workdir", required=True)
a = ap.parse_args()
sys.path.insert(0, a.stack)
import tanitad  # noqa: E402

assert Path(tanitad.__file__).resolve().is_relative_to(Path(a.stack).resolve()), tanitad.__file__
torch.set_num_threads(4)
from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig  # noqa: E402
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config  # noqa: E402
from tanitad.refs.refa_v1_plan import PlanConfig  # noqa: E402

D = {"tanitad_file": tanitad.__file__}


def tiny(**kw):
    base = dict(d_enc=16, d_state=16, n_tokens=8, op_layers=1, op_heads=2, op_window=3,
                tac_queries=4, tac_layers=1, str_dim=8, str_layers=1,
                strategic_cfg=StrategicPolicyConfig(d_model=16, depth=1, n_heads=2, d_ctx=8,
                                                    d_cmd=8),
                tactical_cfg=TacticalPolicyConfig(d_model=16, depth=1, n_heads=2, d_intent=8))
    base.update(kw)
    return RefAV1Config(**base)


CONFIGS = {
    "default": {},
    "no_hierarchy": dict(strategic_cfg=None, tactical_cfg=None),
    "speed_channel": dict(speed_channel=True),
    "ema_targets": dict(ema_targets=True),
    "cf": dict(w_cf=0.5, cf_negs=2),
    "motion": dict(motion_inject=True),
    "adapter_space": dict(target_space="adapter", w_sigreg=0.01, var_floor=0.5),
    "proposal_k2": dict(proposal_k=2, w_aux_head=0.1),
    "tmix1_v6vocab": dict(tmix_groups=1, tac_vocab_version="v6.0"),
}


def batch(c, g, labels=True):
    B = 3
    b = {"feats": torch.randn(B, c.op_window, c.n_tokens, c.d_enc, generator=g),
         "actions": torch.randn(B, c.op_steps, c.a_dim, generator=g) * 0.1,
         "future_feats": torch.randn(B, c.op_steps, c.n_tokens, c.d_enc, generator=g),
         "v0": torch.rand(B, generator=g) * 20.0,
         "str_ext_targets": torch.randn(B, c.str_ext_steps, c.n_tokens, c.d_enc, generator=g),
         "str_ext_actions": torch.randn(B, c.str_ext_steps, c.a_dim, generator=g) * 0.1}
    if labels and c.tactical_cfg is not None:
        b["lat_label"] = torch.tensor([0, 3, -100])
        b["lon_label"] = torch.tensor([1, -100, 2])
        b["route_label"] = torch.tensor([0, 1, 2])
        b["nav_cmd"] = torch.tensor([0, 1, 2])
    return b


def tens(out, prefix):
    r = {}
    for k, v in out.items():
        if torch.is_tensor(v):
            r[f"{prefix}.{k}"] = v.detach().clone()
        elif isinstance(v, (int, float, str, list, bool)) or v is None:
            r[f"{prefix}.{k}"] = v
    return r


# ---------------- A. MODEL ----------------
for name, kw in CONFIGS.items():
    torch.manual_seed(1234)
    m = RefAV1(tiny(**kw))
    D.update({f"A.{name}.init.{k}": v.clone() for k, v in m.state_dict().items()})
    g = torch.Generator().manual_seed(7)
    b = batch(m.cfg, g)
    with torch.no_grad():
        m.std.fit(torch.randn(64, m.cfg.d_enc, generator=g))
    feats, acts, fut = b.pop("feats"), b.pop("actions"), b.pop("future_feats")
    torch.manual_seed(99)          # SigReg / cf draw from the global RNG
    out = m(feats, acts, future_feats=fut, **b)
    D.update(tens(out, f"A.{name}.out"))
    out["loss"].backward()
    for n, p in m.named_parameters():
        D[f"A.{name}.grad.{n}"] = None if p.grad is None else p.grad.clone()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    opt.step()
    D.update({f"A.{name}.step1.{k}": v.clone() for k, v in m.state_dict().items()})
    if not m.cfg.w_aux_head:       # the shipped model refuses w_aux_head without targets
        with torch.no_grad():
            out2 = m(feats, acts, v0=b["v0"], nav_cmd=b.get("nav_cmd"))
        D.update(tens(out2, f"A.{name}.nofut"))

# ---------------- B. PLAN ----------------
torch.manual_seed(5)
m = RefAV1(tiny(speed_channel=True)).eval()
with torch.no_grad():
    m.std.fit(torch.randn(64, 16))
g = torch.Generator().manual_seed(11)
f1 = torch.randn(1, 3, 8, 16, generator=g)


def pc():
    return PlanConfig(horizon=m.cfg.plan_steps, dt=m.cfg.op_dt, n_samples=16, n_iters=3,
                      n_elites=4, min_samples=8, seed=0)


variants = {
    "shipped": dict(),
    "ccosh_seam": dict(cost_metric="ccosh", cost_weights=(0.02, 15.11245, 64.3),
                       a_shift=0.3, jerk_seam_a0=0.3, cost_time_grid="tactical"),
    "steer_goalplan": dict(model_action_units="steer", goal_time_grid="plan",
                           seed_kappa_ladder=(0.01, 0.05), w_kappa_by_goal=(0.05, 0.0)),
}
for vn, kw in variants.items():
    res = m.plan(f1, v0=8.5, nav_cmd=torch.tensor([1]), plan_cfg=pc(), **kw)
    D[f"B.{vn}.controls"] = res.controls.clone()
    D[f"B.{vn}.cost"] = float(res.cost)
    D[f"B.{vn}.source"] = res.source
    D[f"B.{vn}.baseline_costs"] = json.dumps(res.baseline_costs, sort_keys=True)
    D[f"B.{vn}.fine_costs"] = json.dumps(getattr(res, "fine_costs", {}), sort_keys=True)
    D[f"B.{vn}.attrs"] = json.dumps(sorted(vars(res)))
    D[f"B.{vn}.goal_action"] = json.dumps(
        {k: (v.tolist() if torch.is_tensor(v) else v)
         for k, v in (res.goal_action or {}).items()}, sort_keys=True)

# ---------------- C. LOADER (real eval episodes) ----------------
from tanitad.data.refav1_loader import RefAV1Windows  # noqa: E402

C = os.environ.get("TANITAD_REFAV1_EVAL_ROOT",
                   "C:/Users/Admin/tanitad-data/refav1-eval141") + "/refav1-fp8-eval"
E = os.environ.get("TANITAD_REFAV1_EVAL_ROOT",
                   "C:/Users/Admin/tanitad-data/refav1-eval141") + "/eps"
L = os.environ.get("TANITAD_V72_EVAL_LABELS",
                   "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz")
names = sorted(Path(p).stem for p in glob.glob(C + "/*.pt") if Path(p).stem != "index")[:3]
for tag, kw in (("labels_nav_ext", dict(str_ext_steps=2)),
                ("labels_nav_noext", dict(str_ext_steps=0)),
                ("plain", dict(str_ext_steps=2, labels_path=None, nav_path=None))):
    kw2 = dict(labels_path=L, nav_path=L)
    kw2.update(kw)
    ld = RefAV1Windows(C, E, op_window=4, op_steps=30, lru=4, seed=3, episodes=names, **kw2)
    D[f"C.{tag}.join_report"] = json.dumps(ld.join_report, sort_keys=True, default=str)
    D[f"C.{tag}.n_windows"] = len(ld)
    for i in range(5):
        bb = ld.batch(4)
        for k, v in bb.items():
            D[f"C.{tag}.b{i}.{k}"] = v.clone() if torch.is_tensor(v) else v
        D[f"C.{tag}.b{i}.keys"] = json.dumps(list(bb))

# ---------------- D. TRAINER --smoke ----------------
spec = importlib.util.spec_from_file_location(
    "trainer_under_test", str(Path(a.stack) / "scripts" / "refa_v1_train.py"))
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
for tag, extra in (("smoke", []), ("smoke_nohier", ["--no-hierarchy"]),
                   ("smoke_speed_ema", ["--speed-channel", "--ema-targets"])):
    od = Path(a.workdir) / tag
    argv = ["--smoke", "--steps", "4", "--log-every", "1", "--bs", "2", "--device", "cpu",
            "--seed", "0", "--out", str(od), *extra]
    assert T.main(argv) == 0
    rows = [json.loads(x) for x in (od / "train_log.jsonl").read_text("utf-8").splitlines()
            if x.strip()]
    for r in rows:
        r.pop("elapsed_s", None)
    D[f"D.{tag}.rows"] = json.dumps(rows, sort_keys=True)
    ck = torch.load(od / "ckpt.pt", map_location="cpu", weights_only=False)
    for k, v in ck["model"].items():
        D[f"D.{tag}.ckpt.model.{k}"] = v
    for pid, st in ck["opt"]["state"].items():
        for k, v in st.items():
            D[f"D.{tag}.ckpt.opt.{pid}.{k}"] = v
    D[f"D.{tag}.ckpt.step"] = ck["step"]
    cfgj = json.loads((od / "config.json").read_text("utf-8"))
    D[f"D.{tag}.config.cfg_common"] = json.dumps(cfgj["cfg"], sort_keys=True, default=str)
torch.save(D, a.out)
print("dumped", len(D), "entries ->", a.out)
