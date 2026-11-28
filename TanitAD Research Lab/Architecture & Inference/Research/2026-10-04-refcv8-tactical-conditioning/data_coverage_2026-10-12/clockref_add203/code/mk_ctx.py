"""Build the three G-CLOCK ctx.json files (the gate's own `check` child job reads them).
C  = CONTROL : refcv7 profile, refcv7's recorded LAUNCH argv (old 4,369 view + old sidecar), refs=[ALLTRAIN]
B  = BEFORE  : refcv8 profile, refcv8 smoke argv with the A2 clock-relevant data flags, refs=[ALLTRAIN]
A  = AFTER   : same argv as B, refs=[ALLTRAIN, ADD203]
Only data flags change; --out is redirected to a new scratch dir (the trainer's main may mkdir it)."""
import hashlib, json, sys
from pathlib import Path
H = "/home/nvidia/clockref203"
TREE = "/home/nvidia/refcv6_run/fa43d9c5ce"
COMMIT = "fa43d9c5cec21c14c6b799045dc2d874124725ba"
sys.path.insert(0, "raw")
import importlib.util
spec = importlib.util.spec_from_file_location("asm", "raw/refcv8_assemble_argv.py"); asm = importlib.util.module_from_spec(spec); spec.loader.exec_module(asm)

def setf(argv, flag, val):
    ps = asm.set_flag(asm.pairs(argv), flag, [val])
    return asm.flat(ps)

c7 = json.load(open("raw/refcv7-r101-s0.argv.json"))["argv"]
c8 = json.load(open("raw/refcv8-wpb-smoke.argv.json"))["argv"]
A2 = asm.DATA_A2
argv_c = setf(c7, "--out", f"{H}/work/gate_out_C/run")
argv_b = setf(c8, "--out", f"{H}/work/gate_out_B/run")
for f in ("--v2-cache", "--agent-rig-extrinsics", "--pose-sync-sidecar", "--clip-clock-sidecar"):
    argv_b = setf(argv_b, f, A2[f])
argv_b = setf(argv_b, "--join-defect-masks", f"{TREE}/{asm.JDM_REL}")   # the A2 mask file ships in the gated tree
argv_b = setf(argv_b, "--steps", str(asm.STEPS))
argv_a = setf(argv_b, "--out", f"{H}/work/gate_out_A/run")
R_ALL = f"{H}/work/q4c_grid_vs_egolog_ALLTRAIN.json"
R_ADD = f"{H}/work/q4c_grid_vs_egolog_ADD203.json"
def ctx(profile, argv, out, refs):
    return {"profile": profile, "tree": TREE, "commit": COMMIT, "argv": argv, "out_dir": out, "path_map": [],
            "tree_sha256": "n/a (clockref203 manual ctx; the tree is the verified fa43d9c checkout)",
            "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
            "options": {"cpu_only": True, "omp": 4, "clock_reference": refs, "clock_manifest": {}}, "arm": None}
Path("work").mkdir(exist_ok=True)
for name, c in (("C", ctx("refcv7", argv_c, f"{H}/work/gate_out_C", [R_ALL])),
                ("B", ctx("refcv8", argv_b, f"{H}/work/gate_out_B", [R_ALL])),
                ("A", ctx("refcv8", argv_a, f"{H}/work/gate_out_A", [R_ALL, R_ADD]))):
    json.dump(c, open(f"work/ctx_{name}.json", "w", encoding="utf-8", newline="\n"), indent=1)
    flags = {f: dict(asm.pairs(c["argv"]))[f] for f in ("--v2-cache", "--clip-clock-sidecar", "--v7-labels", "--eval-cache", "--eval-labels", "--out")}
    print(name, c["profile"], flags)
