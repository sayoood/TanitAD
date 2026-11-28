#!/usr/bin/env python3
"""EXECUTE the E1/E2/E5 fixes — do not read them.

Every claim in `ST_LAUNCH_FIXES.md` is produced here, by running the real
`v6_chain.trainer_argv`, the real `train_v6_staged.build_parser`, the real
`build_stack_from_args`, the real `load_stage_init` against the REAL S-W
checkpoint, and a real encoder forward.

⚠️ `pyarrow` is imported BEFORE torch: the dev box segfaults (0xC0000005) in the
other order, proven A/B.

Usage:
  python verify_st_launch_fixes.py <repo> <geometry-from.json> <sw.fp16.pt> <out.json>
"""
import pyarrow  # noqa: F401
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(sys.argv[1])
GEOM = os.path.abspath(sys.argv[2])
CKPT = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else ""
OUT = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else ""
STACK = os.path.join(REPO, "stack")
sys.path.insert(0, os.path.join(STACK, "scripts"))
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(REPO, "taniteval"))

import torch                                                    # noqa: E402
import v6_chain as VC                                           # noqa: E402
import train_v6_staged as T                                     # noqa: E402

R: dict = {"_evidence_class": "MEASURED (ours; this script)"}

# ---------------------------------------------------------------------------
# 0. the emitted argv — from the real chain, at the real Thor paths
# ---------------------------------------------------------------------------
cfg = VC.ChainConfig(
    root="/home/nvidia/experiments", workdir="/home/nvidia/TanitAD/stack",
    python="/home/nvidia/venvs/tanitad-train/bin/python",
    train_cache="/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl",
    val_cache="/home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl",
    v2_lru=64, geometry_from=GEOM)
plan = VC.build_plan(cfg)
st = VC.step_by_key(plan, "S-T")
argv = VC.trainer_argv(st, cfg, plan)
R["emitted_argv"] = argv
R["launch_line"] = VC.launch_line(st, cfg, plan)
R["pythonpath"] = cfg.pythonpath
R["n_geometry_dests"] = len(VC.geometry_dests())
R["geometry_dests"] = sorted(VC.geometry_dests())

# ---------------------------------------------------------------------------
# 1. the REAL parser accepts it, and the REAL preflight passes
# ---------------------------------------------------------------------------
ap = T.build_parser()
a = ap.parse_args(argv)
R["parsed"] = {"stage": a.stage, "selector": a.selector,
               "tac_goal_cond": bool(a.tac_goal_cond),
               "v2_subframe": a.v2_subframe,
               "dump_seam_plan": a.dump_seam_plan,
               "enc_dim": a.enc_dim, "enc_depth": a.enc_depth,
               "pred_dim": a.pred_dim, "pred_depth": a.pred_depth,
               "d_tac": a.d_tac, "d_str": a.d_str,
               "frame_h": a.frame_h, "frame_w": a.frame_w,
               "grad_checkpoint": bool(a.grad_checkpoint),
               "vit5_encoder": bool(a.vit5_encoder),
               "pred_modern": bool(a.pred_modern)}
R["preflight"] = T.preflight(a)

# ---------------------------------------------------------------------------
# 2. E1 — the stack this argv BUILDS, and the load against the REAL artifact
# ---------------------------------------------------------------------------
t0 = time.time()
stack = T.build_stack_from_args(a)
R["built"] = {"params": sum(p.numel() for p in stack.parameters()),
              "n_keys": len(stack.state_dict()),
              "encoder_pos": list(stack.encoder.pos.shape),
              "build_s": round(time.time() - t0, 1)}

# the OLD (pre-fix) emitted argv, reconstructed by stripping the carry — the
# negative control for E1, built from THIS argv so it cannot drift
old = ["--stage", "S-T", "--out", st.out, "--steps", str(st.steps),
       "--batch", "8", "--lr", str(st.lr), "--n-candidates", "8",
       "--init-from", st.out + "/x.pt", "--prev-gate", st.out + "/g.json",
       "--max-horizon", "60", "--tac-goal-cond", "--plan-wta-eps", "0.05",
       "--w-t1", "1.0", "--v2-cache", cfg.train_cache,
       "--v2-val-cache", cfg.val_cache, "--v2-lru", "64",
       "--frame-h", "256", "--frame-w", "640", "--frame-hfov", "120",
       "--projection", "cylindrical", "--v2-subframe", "176x624",
       "--require-parity"]
a_old = ap.parse_args(old)
t0 = time.time()
old_stack = T.build_stack_from_args(a_old)
R["e1_old_argv_builds"] = {
    "params": sum(p.numel() for p in old_stack.parameters()),
    "n_keys": len(old_stack.state_dict()),
    "encoder_pos": list(old_stack.encoder.pos.shape),
    "_read": "the geometry-free line the chain USED to emit — this is the "
             "87.93 M stack that met a 336.54 M checkpoint."}
del old_stack

if CKPT and os.path.exists(CKPT):
    rep = T.load_stage_init(stack, CKPT, stage="S-T")
    R["e1_load_stage_init"] = {
        k: rep.get(k) for k in ("missing_keys", "unexpected_keys",
                                "introduced_keys", "init_step", "prev_stage",
                                "init_source", "init_precision")}
else:
    R["e1_load_stage_init"] = {"_read": f"checkpoint {CKPT} absent — NOT RUN"}

# ---------------------------------------------------------------------------
# 3. E2 — the encoder takes a forward at the DECLARED frame; the sub-frame
#    variant is now refused at STARTUP instead of at that forward
# ---------------------------------------------------------------------------
with torch.no_grad():
    x = torch.zeros(1, a.in_channels, a.frame_h, a.frame_w)
    t0 = time.time()
    z = stack.encoder(x)
    R["e2_forward_at_declared_frame"] = {
        "input": list(x.shape), "out": list(z.shape),
        "s": round(time.time() - t0, 1), "ok": True}
    bad = ap.parse_args(argv + ["--v2-subframe", "176x624"])
    R["e2_preflight_refuses_subframe"] = T.preflight(bad)
    R["e2_subframe_desync"] = T.subframe_desync(bad)
    R["e2_subframe_noop_is_ok"] = T.subframe_desync(
        ap.parse_args(argv + ["--v2-subframe", f"{a.frame_h}x{a.frame_w}"]))
    try:
        stack.encoder(torch.zeros(1, a.in_channels, 176, 624))
        R["e2_forward_at_subframe"] = {"ok": True, "_read": "UNEXPECTED"}
    except Exception as e:                                      # noqa: BLE001
        R["e2_forward_at_subframe"] = {"ok": False,
                                       "err": f"{type(e).__name__}: {e}"}
R["e2_chain_emits_subframe"] = "--v2-subframe" in argv

# ---------------------------------------------------------------------------
# 4. E1 guard — the FULL derived diff refuses the old geometry-free argv
# ---------------------------------------------------------------------------
try:
    VC.assert_geometry_carry(st, plan, old, cfg)
    R["e1_guard_on_old_argv"] = {"refused": False, "_read": "UNEXPECTED"}
except VC.ChainRefusal as e:
    R["e1_guard_on_old_argv"] = {"refused": True, "msg": str(e)}
except Exception as e:                                          # noqa: BLE001
    R["e1_guard_on_old_argv"] = {"refused": None,
                                 "err": f"{type(e).__name__}: {e}"}
R["e1_guard_on_new_argv"] = VC.assert_geometry_carry(st, plan, argv, cfg)
# ⚠️ the guard with NO cfg is the pre-fix reachability: on a box where
# <prev.out>/config.json does not exist it evaporates to ok:None
R["e1_guard_without_cfg"] = VC.assert_geometry_carry(st, plan, old)

# ---------------------------------------------------------------------------
# 5. E5 — the seam-dump import, under BOTH PYTHONPATHs, as a subprocess
# ---------------------------------------------------------------------------
probe = ("import json,importlib,sys\n"
         "try:\n"
         "    importlib.import_module('taniteval.seam_dump'); r={'ok':True}\n"
         "except BaseException as e:\n"
         "    r={'ok':False,'err':type(e).__name__+': '+str(e)[:90]}\n"
         "print('ZZ'+json.dumps(r)+'ZZ')\n")
for name, pp in (("stack_only", STACK),
                 ("both_roots", STACK + os.pathsep
                  + os.path.join(REPO, "taniteval"))):
    env = dict(os.environ, PYTHONPATH=pp)
    p = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, env=env, cwd=STACK)
    body = p.stdout.split("ZZ")[1] if "ZZ" in p.stdout else "{}"
    R.setdefault("e5_import", {})[name] = json.loads(body or "{}")
R["e5_preflight_no_taniteval"] = None
env = dict(os.environ, PYTHONPATH=STACK)
p = subprocess.run(
    [sys.executable, os.path.join(STACK, "scripts", "train_v6_staged.py")]
    + argv, capture_output=True, text=True, env=env, cwd=STACK, timeout=900)
R["e5_preflight_no_taniteval"] = {
    "returncode": p.returncode,
    "seam_line": [ln for ln in (p.stdout + p.stderr).splitlines()
                  if "dump-seam-plan" in ln or "taniteval" in ln][:4]}

if OUT:
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
print("ZZJSONZZ" + json.dumps({k: v for k, v in R.items()
                               if k not in ("emitted_argv", "launch_line",
                                            "geometry_dests")}) + "ZZENDZZ")
