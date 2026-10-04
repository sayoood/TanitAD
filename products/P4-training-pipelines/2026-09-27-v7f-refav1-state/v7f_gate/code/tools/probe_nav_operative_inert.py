"""MEASURED probe (gate agent, 2026-09-27): is R2's OPERATIVE nav channel trained anywhere in the v7F
ladder (S-W -> S-T)? Runs the gate's own S-W and S-T rehearsal smokes (tiny config, synthetic corpus,
synthetic nav records through the real NavEmitter), then reads (1) G-LIVE's dead leaves, (2) the trainer's
own grad_reach.json, (3) the saved checkpoints' nav tensors. Arg 1: the output JSON (run BEFORE the
merge's F7 fix -> probe_nav_operative_inert.json; AFTER -> probe_nav_operative_after_F7_fix.json)."""
import glob, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, "C:/Users/Admin/v7f_gate/tree_m/stack")
sys.path.insert(0, "C:/Users/Admin/v7f_gate/tree_m/stack/tests")
sys.path.insert(0, "C:/Users/Admin/v7f_gate/tree_m/stack/scripts")
import torch
import test_launch_gate_v7f as TT
LG = TT.LG
root = Path(tempfile.mkdtemp(prefix="navop_", dir="C:/Users/Admin/v7f_gate/gate_runs"))
res = {"root": str(root)}
NAV = ("nav.embed.weight", "nav.arg_proj.weight", "nav.arg_proj.bias", "nav.layer_proj.operative.weight",
       "nav.layer_proj.operative.bias", "nav.gate.operative", "nav.layer_proj.tactical.weight",
       "nav.layer_proj.tactical.bias", "nav.gate.tactical")
for label, argv in (("S-W", TT.SW_ARGV), ("S-T", TT.ST_ARGV)):
    ctx = TT._ctx(root / label, argv)
    ev = LG.job_smoke_v6(ctx, ["G-LIVE"])["G-LIVE"]
    g = ev["details"].get("grads", {})
    r = {"G-LIVE": ev["status"], "reasons": ev.get("reasons"), "dead_leaves": g.get("dead_leaves"),
         "live_top_modules": g.get("live_top_modules"),
         "control_trainer_grad_reach": ev["details"].get("control_trainer_grad_reach")}
    cks = sorted(glob.glob(str(root / label / "smoke" / "*" / "run" / "ckpt.pt")))
    r["ckpts"] = cks
    if cks:
        sd = torch.load(cks[0], map_location="cpu", weights_only=False)["stack"]
        r["ckpt_nav"] = {k: {"absmax": float(sd[k].float().abs().max()), "numel": int(sd[k].numel())}
                         for k in NAV if k in sd}
    grj = sorted(glob.glob(str(root / label / "smoke" / "*" / "run" / "grad_reach.json")))
    if grj:
        gr = json.loads(Path(grj[0]).read_text(encoding="utf-8"))
        r["trainer_grad_reach_file"] = grj[0]
        r["trainer_grad_reach_nav"] = {k: v for k, v in gr.items() if "nav" in k} if isinstance(gr, dict) else None
        r["trainer_grad_reach_keys"] = list(gr)[:20] if isinstance(gr, dict) else None
    # S-T predecessor ckpt (the tiny S-W that S-T inits from)
    pc = sorted(glob.glob(str(root / label / "rehearsal_predecessor" / "*" / "ckpt.pt")))
    if pc:
        sd = torch.load(pc[0], map_location="cpu", weights_only=False)["stack"]
        r["predecessor_ckpt_nav"] = {k: float(sd[k].float().abs().max()) for k in NAV if k in sd}
    res[label] = r
out = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/v7f_gate/raw/probe_nav_operative_inert.json")
out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, indent=1, default=str)[:6000])
