"""Read-only probe: run PREREG_V7F sec.9's launch line through the REAL argparse and
preflight() of train_v6_staged.py AT THE SNAPSHOT (b3f7ea6f). CPU only, no GPU, no
training, no writes (PYTHONDONTWRITEBYTECODE=1; --out points into the scratchpad and
preflight() never creates it). Aborts if `tanitad` resolves outside the snapshot.
"""
import io
import json
import os
import sys
import contextlib

SNAP = "C:/Users/Admin/tipsnap/b3f7ea6f"
sys.path.insert(0, SNAP + "/stack/scripts")
sys.path.insert(0, SNAP + "/stack")

import tanitad  # noqa: E402
tf = os.path.normcase(os.path.abspath(tanitad.__file__))
if not tf.startswith(os.path.normcase(os.path.abspath(SNAP))):
    print("ABORT: tanitad resolved OUTSIDE the snapshot:", tanitad.__file__)
    sys.exit(3)
print("tanitad from:", tanitad.__file__)

import train_v6_staged as T  # noqa: E402
tv = os.path.normcase(os.path.abspath(T.__file__))
if not tv.startswith(os.path.normcase(os.path.abspath(SNAP))):
    print("ABORT: train_v6_staged resolved OUTSIDE the snapshot:", T.__file__)
    sys.exit(3)
print("train_v6_staged from:", T.__file__)

OUT = os.environ.get("PROBE_OUT", "C:/nonexistent_probe_out/v7f-b1-1ep")
V2 = "/home/nvidia/data/physicalai-b1-w120-256x640cyl"  # Thor path, absent here
LAB = "/home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz"  # absent here
SEED = "/home/nvidia/seeds/dinov3_vitb16_seed.pt"  # absent here

# PREREG_V7F.md sec.9 verbatim, placeholders filled with the values the prereg names
S9 = [
    "--stage", "S-W", "--out", OUT,
    "--v2-cache", V2, "--require-parity", "--exclude-eval-clips", "auto", "--v2-lru", "64",
    "--s2-labels", LAB, "--w-s2-goal", "1.0", "--nav-labels", LAB, "--nav-cond",
    "--enc-init-from", SEED, "--newest-frame-only", "--in-channels", "3",
    "--enc-dim", "768", "--enc-depth", "12", "--enc-heads", "12",
    "--patch", "16", "--frame-h", "256", "--frame-w", "640", "--projection", "cylindrical",
    "--frame-hfov", "120",
    "--trunk-lr-scale", "0.1", "--trunk-lr-warmup-steps", "2000",
    "--w-trunk-anchor", "1.0", "--trunk-anchor-model", "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "--o5-form", "l1", "--w-o5", "1.0", "--w-o6", "0.1",
    "--sigreg-subspaces", "32", "--sigreg-slices", "512", "--spectrum-accum", "4096",
    "--cond-param", "omega_accel_v",
    "--w-o14", "1.0", "--o14-mode", "fut", "--o14-k", "4",
    "--o5-target", "ema", "--ema-decay", "0.996",
    "--o5-k", "60", "--bptt-truncate", "4", "--rollout-grad-checkpoint", "on",
    "--w-o1-ctrl", "0", "--w-o1-fact", "0", "--w-o1-scene", "0", "--w-o2", "0", "--w-o3", "0",
    "--w-o7-distill", "0", "--w-o8-pixel", "0", "--w-o9-ema", "0", "--w-o10-psg", "0",
    "--w-o11-cf", "0", "--w-o13-ego", "0",
    "--refuse-unreached", "--allow-unreached", "step_readout_op", "masked_cells",
    "--steps", "1000", "--batch", "8", "--lr", "1e-4", "--clip", "1.0",
    "--seed", "0", "--save-every", "1000", "--log-every", "50", "--print-launch",
]
LDAD = ["--w-ldad", "1.0", "--ldad-target", "a_kappa", "--ldad-form", "delta_z"]


def ascii(s):
    return str(s).encode("ascii", "replace").decode("ascii")


def parse(argv):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            a = T.build_parser().parse_args(argv)
        return a, None
    except SystemExit as e:
        return None, ascii(err.getvalue().strip().splitlines()[-1] if err.getvalue() else e)


def run_preflight(a):
    probs = T.preflight(a)
    return [ascii(p) for p in probs]


result = {"snapshot": SNAP, "tanitad_file": tanitad.__file__}

# A. exact sec.9 (with the LDAD triple)
a, perr = parse(S9 + LDAD)
result["A_exact_s9_argparse_error"] = perr

# B. sec.9 minus LDAD (the only argparse-unknown flags) -> preflight
a, perr = parse(S9)
result["B_minusLDAD_argparse_error"] = perr
if a is not None:
    pb = run_preflight(a)
    result["B_minusLDAD_preflight_n"] = len(pb)
    result["B_minusLDAD_preflight_first_lines"] = [p.splitlines()[0][:400] for p in pb]
    result["B_anchor_preflight"] = None
    try:
        T.assert_trunk_anchor_preflight(a)
        result["B_anchor_preflight"] = "PASS"
    except SystemExit as e:
        result["B_anchor_preflight"] = ascii(str(e).splitlines()[0])[:400]

# C. B with the two design fixes the record names (--horizons 1; drop --w-s2-goal and
#    --s2-labels at S-W; add --obs-monitor-every 250) -> preflight
S9C = list(S9)
i = S9C.index("--s2-labels"); del S9C[i:i + 2]
i = S9C.index("--w-s2-goal"); del S9C[i:i + 2]
S9C += ["--horizons", "1", "--obs-monitor-every", "250"]
a, perr = parse(S9C)
result["C_fixed_argparse_error"] = perr
if a is not None:
    pc = run_preflight(a)
    result["C_fixed_preflight_n"] = len(pc)
    result["C_fixed_preflight_first_lines"] = [p.splitlines()[0][:400] for p in pc]
    try:
        T.assert_trunk_anchor_preflight(a)
        result["C_anchor_preflight"] = "PASS"
    except SystemExit as e:
        result["C_anchor_preflight"] = ascii(str(e).splitlines()[0])[:400]

print(json.dumps(result, indent=1))
