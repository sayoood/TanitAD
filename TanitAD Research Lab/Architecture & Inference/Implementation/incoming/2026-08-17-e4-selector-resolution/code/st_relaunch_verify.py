"""Item 3: the S-T launch line, RE-DERIVED after the E4 change, and EXECUTED.

No selector was enabled by default, so the claim under test is that the line is
UNCHANGED. A claim of "unchanged" is worth nothing unless it is diffed.
"""
import pyarrow  # noqa: F401
import json, sys, pathlib, hashlib, difflib
REPO = pathlib.Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "stack"))
# the launch line sets BOTH roots; evaluating it without the
# sibling is the scope error ST_LAUNCH_FIXES.md §7 names.
sys.path.insert(0, str(REPO / "taniteval"))
import torch  # noqa
import v6_chain as C
import train_v6_staged as T

INC = REPO / "TanitAD Research Hub" / "Architecture & Inference" / "Implementation" / "incoming"
BANKED = (INC / "2026-08-17-st-launch-fixes" / "raw" / "st_launch_line_fixed.txt")
GEOM = (INC / "2026-08-17-st-launch-fixes" / "raw" / "v6F-SW-30k.config.json")

cfg = C.ChainConfig(
    root="/home/nvidia/experiments", workdir="/home/nvidia/TanitAD/stack",
    python="/home/nvidia/venvs/tanitad-train/bin/python",
    train_cache="/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl",
    val_cache="/home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl",
    v2_lru=64, geometry_from=str(GEOM))
plan = C.build_plan(cfg)
st = C.step_by_key(plan, "S-T")
line = C.launch_line(st, cfg, plan, allow_inconclusive=False, off_reason="")

banked = BANKED.read_text(encoding="utf-8").strip()
same = line.strip() == banked
diff = [] if same else list(difflib.unified_diff(
    banked.split(), line.strip().split(), lineterm="", n=1))

# --- and EXECUTE it: real parser, real build, real load of the REAL ckpt -----
argv = C.trainer_argv(st, cfg, plan)
a = T.build_parser().parse_args(argv)
pre = T.preflight(a)
stack = T.build_stack_from_args(a)
sd = stack.state_dict()

CK = pathlib.Path(r"C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/v6F_sw_step010000.fp16.pt")
ck_md5 = hashlib.md5(CK.read_bytes()).hexdigest()
init = T.load_stage_init(stack, str(CK), stage="S-T")

# --- ONE REAL FORWARD STEP on synthetic tensors (CPU; Thor untouched) -------
torch.manual_seed(0)
B, W = 1, a.window
x = torch.randn(B, W, a.in_channels, a.frame_h, a.frame_w)
with torch.no_grad():
    acts = torch.zeros(B, W, 3)
    out = stack(x, acts, torch.full((B,), 8.0))
fwd = {"ok": True, "keys": sorted(k for k in out if k.startswith("sel_")),
       "z_op": list(out["z_op"].shape) if "z_op" in out else None,
       "has_plan": "plan" in out}

# --- what the GATE will actually read at S-T under this resolution ----------
arm = T.arm_record(stack)
gate = T.stage_gate_dict("S-T", {}, arm=arm)
gate_all_run = T.stage_gate_dict(
    "S-T", {p: {"pass": True} for p in T.STAGE_GATE_SPEC["S-T"]["required"]
            if T.probe_applies("S-T", p, arm) is None}, arm=arm)

print(json.dumps({
    "line_unchanged_vs_banked": same,
    "line_diff": diff[:20],
    "line_md5": hashlib.md5(line.strip().encode()).hexdigest(),
    "banked_md5": hashlib.md5(banked.encode()).hexdigest(),
    "preflight": pre,
    "params": sum(p.numel() for p in stack.parameters()),
    "state_dict_keys": len(sd),
    "encoder_pos": list(sd["encoder.pos"].shape),
    "ckpt": {"path": CK.name, "md5": ck_md5, "bytes": CK.stat().st_size},
    "load_stage_init": {k: init[k] for k in
                        ("missing_keys", "unexpected_keys", "introduced_keys",
                         "init_step", "prev_stage") if k in init},
    "forward": fwd,
    "arm_record": arm,
    "gate_nothing_run": {"verdict": gate["verdict"],
                         "required": gate["required"],
                         "required_effective": gate["required_effective"],
                         "not_applicable": [x["probe"] for x in
                                            gate["not_applicable_required"]],
                         "missing": gate["missing_required"]},
    "gate_battery_folded_in": {"verdict": gate_all_run["verdict"],
                               "required_effective":
                                   gate_all_run["required_effective"]},
}, indent=1, default=str))
