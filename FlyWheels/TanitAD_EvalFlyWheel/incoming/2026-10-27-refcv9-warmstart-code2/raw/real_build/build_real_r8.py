"""Build the REAL refcv8-r101-s0 model (48,852 config) on CPU through the eval loader (refcv7_loader.build_model,
which replays train()'s build and strict-loads the shipped EMA file), then dump the parameter inventory.
usage: python build_real_r8.py <tree> <config.json> <ckpt> <out.json>"""
import json, os, sys, time
tree, cfgp, ckpt, outp = sys.argv[1:5]
os.environ["REFCV7_REPO"] = tree
sys.path.insert(0, os.path.join(tree, "stack"))
from tanitad.eval import refcv7_loader as L
L.bootstrap()
import tanitad, torch
torch.set_num_threads(4)
config = json.load(open(cfgp, encoding="utf-8"))
t0 = time.time()
model, cfg, args, rec = L.build_model(config, ckpt, device="cpu")
inv = [(n, list(p.shape), int(p.numel())) for n, p in model.named_parameters()]
bufs = [(n, list(b.shape), int(b.numel())) for n, b in model.named_buffers()]
json.dump({"tanitad_file": tanitad.__file__, "torch": torch.__version__, "ckpt": ckpt, "config": cfgp,
           "build_s": round(time.time() - t0, 1), "departures": rec.get("departures"),
           "state_dict": rec.get("state_dict", {}).get("n_keys"),
           "dvb": rec.get("declared_vs_built"), "params": inv, "buffers": bufs},
          open(outp, "w", encoding="utf-8"), indent=0)
print("tanitad from", tanitad.__file__, "params", len(inv), sum(x[2] for x in inv), "buffers", len(bufs),
      "build_s", round(time.time() - t0, 1))
