"""CPU smoke of the REAL `refa_v1_train.py` on 3 REAL eval episodes, TINY architecture.

⛔ A SMOKE, NEVER A METRIC: 3 episodes, 3 steps, random init. It proves the real trainer loop
(argparse -> refusals -> G-DVB -> build -> loader on the real fp8 cache + v7.2 labels/nav ->
forward/backward/step -> log -> checkpoint) runs with the PI 2026-09-27 flags ON and OFF.

The architecture is shrunk by patching the trainer's module-level `RefAV1` name (depth / heads /
brain widths / R5 widths only); the GEOMETRY is untouched -- d_enc 1024, 640 tokens, d_state
1024 -- because the cache contract refuses anything else. After training it reloads the
checkpoint and runs `RefAV1.trajectory()` on one real batch with the R1 cap, recording the cap
record (`n_rows_over_limit` must be 0).

usage: python smoke_real.py --stack <stack> --out <dir> [--on] [-- extra trainer flags]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--stack", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default=os.environ.get("REFAV1_SMOKE_DATA",
                                               "C:/Users/Admin/refav1_r1r6/smoke_data"))
ap.add_argument("--labels", default=os.environ.get(
    "TANITAD_V72_EVAL_LABELS", "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz"))
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("extra", nargs="*")
a = ap.parse_args()
sys.path.insert(0, a.stack)
import torch  # noqa: E402
import tanitad  # noqa: E402

assert Path(tanitad.__file__).resolve().is_relative_to(Path(a.stack).resolve()), tanitad.__file__
torch.set_num_threads(4)
from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig  # noqa: E402

spec = importlib.util.spec_from_file_location("refa_v1_train_smoke",
                                              str(Path(a.stack) / "scripts" / "refa_v1_train.py"))
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
_Real = T.RefAV1


def _tiny(cfg):
    cfg.op_layers, cfg.op_heads, cfg.tac_layers = 1, 2, 1
    cfg.tac_queries, cfg.str_dim, cfg.str_layers = 8, 16, 1
    if cfg.strategic_cfg is not None:
        cfg.strategic_cfg = StrategicPolicyConfig(d_model=32, depth=1, n_heads=2, d_ctx=16, d_cmd=8)
    if cfg.tactical_cfg is not None:
        cfg.tactical_cfg = TacticalPolicyConfig(d_model=32, depth=1, n_heads=2, d_intent=16)
    if hasattr(cfg, "traj_d_region"):
        cfg.traj_d_region, cfg.traj_hidden = 16, 64
    return _Real(cfg)


T.RefAV1 = _tiny
out = Path(a.out)
argv = ["--cache", f"{a.data}/cache3", "--episodes", f"{a.data}/eps3",
        "--labels", a.labels, "--nav", a.labels, "--steps", str(a.steps), "--bs", "2",
        "--log-every", "1", "--save-every", "1000", "--device", "cpu", "--lru", "4",
        "--seed", "0", "--out", str(out), *a.extra]
print("[smoke] argv:", " ".join(argv[:2] + ["..."] + argv[-(8 + len(a.extra)):]), flush=True)
t0 = time.time()
rc = T.main(argv)
wall = time.time() - t0
rows = [json.loads(x) for x in (out / "train_log.jsonl").read_text("utf-8").splitlines() if x.strip()]
cfg = json.loads((out / "config.json").read_text("utf-8"))
rep = {"rc": rc, "wall_s": round(wall, 1), "steps": len(rows),
       "all_losses_finite": all(r["loss"] == r["loss"] and abs(r["loss"]) < 1e9 for r in rows),
       "row1": rows[0], "row_last": rows[-1],
       "config_keys_r1r6": sorted(k for k in cfg if k not in ("args", "cfg", "precision",
                                                              "autocast", "grad_scaler",
                                                              "master_weights", "tf32",
                                                              "device", "torch",
                                                              "written_utc"))}
if "traj_ade_m" in rows[0]:
    rep["known_value_row1_traj_ade_eq_prior_ade"] = rows[0]["traj_ade_m"] == rows[0]["traj_prior_ade_m"]
if "speed_max_derivation_v6" in cfg:
    from tanitad.refs.refcv6_max_speed import assert_speed_max_stamp_v6
    assert_speed_max_stamp_v6(cfg, on=True)
    rep["speed_max_stamp"] = "present, assert_speed_max_stamp_v6 PASS"
# ---- inference: trajectory() with the R1 cap on one real batch, from the saved checkpoint --
ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
if not hasattr(T, "build_parser"):      # the PRE-directive trainer (the OFF cross-check run)
    print("ZZSMOKE " + json.dumps(rep, default=str))
    sys.exit(0)
m = T.build_model(T.build_parser().parse_args(argv))
m.load_state_dict(ck["model"], strict=True)
m.eval()
if m.traj_head is not None:
    from tanitad.data.refav1_loader import RefAV1Windows
    c = m.cfg
    ld = RefAV1Windows(f"{a.data}/cache3", f"{a.data}/eps3", op_window=c.op_window,
                       op_steps=c.op_steps, str_ext_steps=0, lru=4, seed=5,
                       labels_path=a.labels, nav_path=a.labels, speed_max=True,
                       traj_steps=c.traj_steps,
                       traj_kappa_source=c.traj_kappa_source)
    b = ld.batch(6)
    r = m.trajectory(b["feats"], v0=b["v0"], a0=b["a0"], kappa0=b["kappa0"],
                     nav_cmd=b["nav_cmd"], speed_max_ms=b["speed_max_ms"],
                     speed_max_valid=b["speed_max_valid"])
    from tanitad.refs.refav1_traj import path_speeds
    rep["trajectory_inference"] = {
        "vmax_cap": r["vmax_cap"],
        "max_speed_uncapped_mps": [round(float(x), 3) for x in path_speeds(r["traj_uncapped"], c.op_dt).max(1).values],
        "max_speed_capped_mps": [round(float(x), 3) for x in path_speeds(r["traj"], c.op_dt).max(1).values],
        "v0_mps": [round(float(x), 3) for x in b["v0"]],
        "shape": list(r["traj"].shape),
        "ade_vs_gt_m": float(((r["traj"] - b["traj_gt"]).norm(dim=-1)).mean()),
        "prior_ade_vs_gt_m": float(((r["traj_prior"] - b["traj_gt"]).norm(dim=-1)).mean())}
print("ZZSMOKE " + json.dumps(rep, default=str))
