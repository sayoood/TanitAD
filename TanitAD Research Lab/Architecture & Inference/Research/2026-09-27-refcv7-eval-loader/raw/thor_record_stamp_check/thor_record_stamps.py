"""THOR RECORD variant: the loader's stamp_checks against the Thor probe's config.json (verbatim), with the
reference model built by the REAL train() on the dev box. (Adapted from:) Full-size check of the loader's stamp_checks against the stamps the REAL train() builds for the
intended 154-token argv. train() builds `map_hires_stamp` (8043-8069), `perception_stamp`
(8105-8127) and `_navc_tau_stamp` (7932) BEFORE its first data read; the capture reads them out of
train()'s frame at that call (refuse_eval_clips_in_train, 8342). The loader then rebuilds from a
record carrying those stamps and must (a) load strictly 0/0 and (b) find every stamp EQUAL.
CPU only. Writes C:/lgt/r7ldr_scratch/thor_record_stamps.json."""
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["REFCV6_REPO"] = "C:/lgt/r7ldr"
os.environ["REFCV6_KIT"] = "D:/refcv6_eval_kit"
os.environ.pop("REFCV6_REMAP_OVERRIDES", None)
for p in ("C:/lgt/r7ldr/stack", "C:/lgt/r7ldr/stack/scripts", "C:/lgt/r7ldr/taniteval"):
    sys.path.insert(0, p)
import torch  # noqa: E402

import launch_gate as LG  # noqa: E402

OUT = Path("C:/lgt/r7ldr_scratch/thorrec")
OUT.mkdir(parents=True, exist_ok=True)
argv = json.load(open("C:/Users/Admin/qland/work/refcv7/probe_cost/launch_argv_intended.json", encoding="utf-8"))
assert len(argv) == 154
MODEL_INPUTS = ("--anchors", "--agent-rig-extrinsics", "--nav-compliance-tau-file",
                "--map-hires-class-weights")
loc = []
i = 0
while i < len(argv):
    a = argv[i]
    if a == "--trunk-compile":
        i += 1
        continue
    loc.append(a)
    if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
        vals = []
        j = i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            vals.append(argv[j])
            j += 1
        for v in vals:
            if a == "--out":
                loc.append(str(OUT / "out"))
            elif v.startswith("/home/nvidia/data"):
                loc.append(v.replace("/home/nvidia/data", "D:/refcv6_eval_kit/data") if a in MODEL_INPUTS
                           else str(OUT / "__unread__" / a.strip("-")))
            else:
                loc.append(v)
        i = j
    else:
        i += 1

T = LG.load_trainer(type("C", (), {"tree": "C:/lgt/r7ldr",
                                   "prof": {"trainer": "stack/scripts/refc_v3_train.py"}})())
grab = {}
from tanitad.data import parity as _parity  # noqa: E402
orig = _parity.assert_v2_parity_cache


def grab_then_orig(*a, **k):
    # train() calls this at the top of its v2-cache branch, right BEFORE the gate's stop
    # (refuse_eval_clips_in_train); every stamp below is already built in train()'s frame
    f = sys._getframe(1)
    for n in ("map_hires_stamp", "perception_stamp", "_navc_tau_stamp"):
        grab[n] = f.f_locals.get(n)
    grab["_frame_function"] = f.f_code.co_name
    return orig(*a, **k)


_parity.assert_v2_parity_cache = grab_then_orig
t0 = time.time()
try:
    cap = LG.run_trainer_until(T, loc, "model")
finally:
    _parity.assert_v2_parity_cache = orig
assert grab.get("_frame_function") == "train", grab.get("_frame_function")
grab.pop("_frame_function")
model_t = cap["model"]
print("trainer built in", round(time.time() - t0, 1), "s; grabbed", {k: v is not None for k, v in grab.items()})
ck = OUT / "ckpt.pt"
torch.save({"model": model_t.state_dict(), "step": 0}, ck)

# the THOR record, verbatim (md5 0a659ec830b0df19781cdfd1b21075b8), with argv = the launch argv as G-EVAL sets it
import hashlib  # noqa: E402
_raw = open("C:/lgt/r7ldr_gate/thor_probe_config.json", "rb").read()
assert hashlib.md5(_raw).hexdigest() == "0a659ec830b0df19781cdfd1b21075b8"
config = {k: v for k, v in json.loads(_raw.decode("utf-8")).items() if k != "argv"}
config["argv"] = list(argv)
rt = {"map_hires": config["map_hires"]}

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("refcv7_loader_fullsize", "C:/lgt/r7ldr/stack/tanitad/eval/refcv7_loader.py")
L = importlib.util.module_from_spec(spec)
sys.modules["refcv7_loader_fullsize"] = L
spec.loader.exec_module(L)
t1 = time.time()
model_l, cfg_l, args_l, rec = L.build_model(config, str(ck), device="cpu", strict=True)
sc = rec["stamp_checks"]
res = {"build_s": round(time.time() - t1, 1),
       "strict": {"missing": rec["state_dict"]["missing"], "unexpected": rec["state_dict"]["unexpected"],
                  "n_keys": rec["state_dict"]["n_keys"], "n_map_hires_keys": rec["state_dict"]["n_map_hires_keys"]},
       "n_compared": sc.get("n_compared"),
       "not_equal": sorted(k for k, v in sc.items() if isinstance(v, dict) and "equal" in v and not v["equal"]),
       "compared": sorted(k for k, v in sc.items() if isinstance(v, dict) and "equal" in v),
       "absent": {k: v for k, v in sc.items() if isinstance(v, str)},
       "declared_vs_built": rec["declared_vs_built"],
       "stamps_present_in_train": {k: v is not None for k, v in grab.items()},
       "map_hires_stamp_keys": sorted(rt["map_hires"].keys()) if rt["map_hires"] else None}
dt = LG.state_digests(model_t.state_dict())
dl = LG.state_digests(model_l.state_dict())
res["state_n_differ"] = sum(1 for k in dt if dt.get(k) != dl.get(k))
res["state_n"] = [len(dt), len(dl)]
# the RED arm on the SAME real stamps: a record claiming R1 (no near lift) must be REFUSED
bad = json.loads(json.dumps(config))
bad["map_hires"]["near_lift_x_m"] = 0.0
try:
    L.build_model(bad, str(ck), device="cpu", strict=True)
    res["red_R1_record"] = "NOT REFUSED"
except SystemExit as e:
    res["red_R1_record"] = "REFUSED: " + str(e)[:300]
json.dump(res, open("C:/lgt/r7ldr_scratch/thor_record_stamps.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1)[:4000])
try:
    ck.unlink()
except OSError:
    pass
