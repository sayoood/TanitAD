"""Driver: run the UNCHANGED q4c_grid_vs_egolog.main() (md5 4f394339...) on Thor CPU.
Only module-level constants are patched (EGO_DIR, MANIFESTS, N_TRAIN_SAMPLE); no function body
is touched.  Modes:  ctl  = control clips (20 random OLD train + the old clips ALLTRAIN did not
reference) with a K2 carrier;  add = the 203 ADD clips;  all = the full 4,572 view.
Outputs go to <HOLD>/raw via _common.write_json (sha12-only, UUID-scrubbed)."""
import hashlib, json, os, sys, time
from pathlib import Path
HOLD = Path("/home/nvidia/clockref203")
os.environ["AUDIT_TIP"] = "/home/nvidia/refcv6_run/fa43d9c5ce"
os.environ["AUDIT_MIN_FREE_GB"] = "25"
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, str(HOLD / "code"))
import numpy as np
import torch
import q4c_grid_vs_egolog as Q          # runs _common.bootstrap() at import (asserts tanitad from the Thor tree, CPU only)
import _common as C

mode = sys.argv[1]
MAN = "/home/nvidia/data/refcv8-b1-416x1024-train-4572/_v2manifest.pt"
Q.EGO_DIR = Path("/home/nvidia/data/obstacle/egomotion_alpamayo")
m = torch.load(MAN, map_location="cpu", weights_only=False)
add = [l.strip() for l in open(HOLD / "code/ADD_203_clip_ids.txt") if l.strip()]
assert len(add) == 203 and len(set(add)) == 203
pos = {c: i for i, c in enumerate(m["clip_id"])}
assert all(c in pos for c in add), "an ADD clip is missing from the 4,572 manifest"
old_ids = [c for c in m["clip_id"] if c not in set(add)]
assert len(old_ids) == 4369, len(old_ids)

def subset(cids, name):
    idx = [pos[c] for c in cids]
    sm = {"version": m["version"]}
    for k, v in m.items():
        if k != "version":
            sm[k] = [v[i] for i in idx]
    p = HOLD / "work" / f"{name}.pt"
    torch.save(sm, p)
    return p

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

t0 = time.time()
if mode == "ctl":
    ref = json.load(open(HOLD / "work/q4c_grid_vs_egolog_ALLTRAIN.json"))
    ref_tr = {r["clip"]: r for r in ref["splits"]["train"]["per_clip"]}
    rng = np.random.default_rng(20261012)
    rand20 = [old_ids[i] for i in sorted(rng.choice(len(old_ids), 20, replace=False).tolist())]
    nonref = [c for c in old_ids if C.sha12(c) not in ref_tr or ref_tr[C.sha12(c)]["fit_resid_max_ms"] >= 5.0]
    ctl = list(dict.fromkeys(rand20 + nonref))
    # K2 carrier: first clip must be moving -> take a random one that ALLTRAIN fitted with 199 moving rows
    k2 = [c for c in rand20 if C.sha12(c) in ref_tr and ref_tr[C.sha12(c)]["n_moving_rows"] == 199][:1]
    assert k2, "no moving K2 carrier"
    Q.MANIFESTS = {"eval139": subset(k2, "k2carrier"), "train": subset(ctl, "ctl_train")}
    Q.N_TRAIN_SAMPLE = 10 ** 9
    print("CTL clips: random20", len(rand20), "| ALLTRAIN-nonref old", len(nonref), "| union", len(ctl), flush=True)
    os.environ["Q4C_OUT"] = "q4c_CTL.json"
elif mode == "add":
    ref = json.load(open(HOLD / "work/q4c_grid_vs_egolog_ALLTRAIN.json"))
    ref_tr = {r["clip"]: r for r in ref["splits"]["train"]["per_clip"]}
    k2 = [c for c in old_ids if C.sha12(c) in ref_tr and ref_tr[C.sha12(c)]["n_moving_rows"] == 199][:1]
    Q.MANIFESTS = {"eval139": subset(k2, "k2carrier_add"), "train": subset(add, "add203")}
    Q.N_TRAIN_SAMPLE = 10 ** 9
    os.environ["Q4C_OUT"] = "q4c_ADD203_rawmain.json"
elif mode == "all":
    Q.MANIFESTS = {"eval139": subset(old_ids[:1], "k2carrier_all"), "train": subset(list(m["clip_id"]), "all4572")}
    Q.N_TRAIN_SAMPLE = 10 ** 9
    os.environ["Q4C_OUT"] = "q4c_ALL4572_rawmain.json"
else:
    raise SystemExit("mode")
print("MANIFESTS:", {k: (str(v), md5(v)) for k, v in Q.MANIFESTS.items()}, flush=True)
Q.main()
print("DONE", mode, round(time.time() - t0, 1), "s", flush=True)
