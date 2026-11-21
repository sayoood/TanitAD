"""Build the Thor-side inputs of the gate's CPU evidence run (no torch; JSON only).

    python make_thor_inputs.py <thor gate dir> <local staging dir>

* argv_refcv6_thor.json   -- refcv6-r101-s0's OWN argv with `--out` moved to a scratch dir under the
                             gate dir (the real run dir is never written, even by accident);
* argv_refcv7like_thor.json -- the same + SPEC_REFCV7 7's three selection mechanisms and the banked
                             tau (the flags the tip at ab436ee knows; NEW-1/NEW-2 are not landed);
* argv_refcv7_canon_thor.json -- THE canonical refcv7 argv (stack/ops/runs.d/...), `--out` moved;
* argv_tiny_thor.json     -- the CPU rehearsal rig's argv with Thor paths;
* eval_remap_overrides_thor.json -- the eval loader's kit remap, IDENTITY on Thor (every flag it
                             remaps is pointed at the argv's own Thor path).
"""
import json
import sys
from pathlib import Path

G = sys.argv[1].rstrip("/")
# ⛔ MEASURED 2026-09-26: run from Git Bash without MSYS_NO_PATHCONV=1, `/home/nvidia/gate_lg_2317`
# arrived here as `C:/Program Files/Git/home/nvidia/gate_lg_2317` and every generated path was wrong
assert G.startswith("/") and ":" not in G, f"mangled gate dir {G!r}: set MSYS_NO_PATHCONV=1"
S = Path(sys.argv[2])
S.mkdir(parents=True, exist_ok=True)
IN = Path("C:/Users/Admin/lg0926/inputs")
TAU = "0.18063741505146028"          # the banked record (ab436ee), tau field, verbatim


def set_flag(argv, flag, vals):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out + ([flag, *vals] if vals is not None else [])


r6 = json.loads((IN / "argv_refcv6_r101_s0.json").read_text(encoding="utf-8"))
assert r6[r6.index("--out") + 1] == "/home/nvidia/refcv6_run/runs/refcv6-r101-s0"
r6t = set_flag(r6, "--out", [f"{G}/launch_out/refcv6-r101-s0"])
(S / "argv_refcv6_thor.json").write_text(json.dumps(r6t, indent=1), encoding="utf-8")
# refcv7 as far as the tip builds it: NEW-1 (06380de) and the selection set with the tau FILE
# (fixes batch 2); NEW-2's flags are NOT in the trainer until NEW-2 lands, so they are absent here
# and the refcv7 profile names each one as missing (the named blocker)
r7 = [*r6t[: r6t.index("--out")], "--residual-prior", "ha0_ext_pose",
      "--graft-tac8-prior", "--graft-nav-compliance", "--nav-compliance-tau-rad", TAU,
      "--nav-compliance-tau-file", "/home/nvidia/data/refcv7/nav_compliance_tau_train.json",
      "--speed-ceiling-filter", "--out", f"{G}/launch_out/refcv7like"]
(S / "argv_refcv7like_thor.json").write_text(json.dumps(r7, indent=1), encoding="utf-8")
# the CANONICAL launch argv, `--out` moved into the gate dir (the real run dir is never written)
canon = json.loads(Path("C:/Users/Admin/lg0926/work/stack/ops/runs.d/refcv7-r101-s0.argv.json")
                   .read_text(encoding="utf-8"))["argv"]
canon_t = set_flag(canon, "--out", [f"{G}/scratch/refcv7_out"])
assert len(canon_t) == len(canon), "the canonical argv changed length when --out moved"
(S / "argv_refcv7_canon_thor.json").write_text(json.dumps(canon_t, indent=1), encoding="utf-8")
tiny = json.loads(Path("C:/Users/Admin/lg0926/rig/argv_tiny.json").read_text(encoding="utf-8"))
tiny = set_flag(set_flag(tiny, "--anchors", [f"{G}/rig/anchors_v0cond_alat_20.pt"]),
                "--out", [f"{G}/rig/launch_out"])
(S / "argv_tiny_thor.json").write_text(json.dumps(tiny, indent=1), encoding="utf-8")


def val(argv, f):
    return argv[argv.index(f) + 1] if f in argv else None


remap_flags = ("--anchors", "--eval-cache", "--eval-labels", "--speed-max-sidecar-v6-eval",
               "--agent-join", "--agent-rig-extrinsics", "--map-gt-root", "--join3d",
               "--clip-clock-sidecar")
ov = {f: val(r6t, f) for f in remap_flags if val(r6t, f)}
(S / "eval_remap_overrides_thor.json").write_text(json.dumps(ov, indent=1), encoding="utf-8")
print(json.dumps({"argv_refcv6_thor": len(r6t), "argv_refcv7like_thor": len(r7),
                  "argv_tiny_thor": len(tiny), "overrides": ov}, indent=1))
