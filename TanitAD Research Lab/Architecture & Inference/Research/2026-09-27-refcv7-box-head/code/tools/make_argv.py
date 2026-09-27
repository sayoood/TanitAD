"""Build the NON-BINDING G-BOX-OVERFIT candidate argv from the refcv6-r101-s0 run's own recorded argv.

Departures, each declared (and stamped by the harness via the argv sha256):
  * --trunk-compile dropped (torch.compile of the backbone call only: eager numerics, no warm-up);
  * --out -> this run's scratch dir (the harness never writes a checkpoint);
  * + the A9 box-head flags: --slot-presence-loss focal --slot-presence-prior 0.01 --slot-deep-supervision
    --slot-vis1 --vis1-sidecar <the 4,508-clip sidecar>.
The refcv7 planner-side flags (FIX-4 selection mechanisms, NEW-1 residual prior) and NEW-2 (not landed) are NOT
added: they do not touch the box path at this base.
"""
import json
import sys

cfg = json.load(open("/home/nvidia/refcv6_run/runs/refcv6-r101-s0/config.json", encoding="utf-8"))
argv = list(cfg["argv"])
out = []
i = 0
while i < len(argv):
    a = argv[i]
    if a == "--trunk-compile":
        i += 1
        continue
    if a == "--out":
        out += ["--out", sys.argv[1] + "/gbo_out"]
        i += 2
        continue
    out.append(a)
    i += 1
out += ["--slot-presence-loss", "focal", "--slot-presence-prior", "0.01", "--slot-deep-supervision",
        "--slot-vis1", "--vis1-sidecar", "/home/nvidia/bx_0252/full/vis1_sidecar_refcv6b1_train4369_eval139.npz"]
json.dump(out, open(sys.argv[1] + "/launch_argv.json", "w"), indent=1)
print(len(out), "args; dropped --trunk-compile; appended the A9 flags")
