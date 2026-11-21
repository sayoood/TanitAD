"""klim_synth.py OUT_DIR -- K-LIM built and run end to end on CPU: a synthetic world (NOT real data: no real batch is on this box), a cond arm trained for 60 steps through the real trainer, then the K-LIM CLI on a collated batch.
A MECHANICS record (the script, the checkpoint and batch hand-off and the record format); the numbers are not a measurement of N1DH."""
import json
import os
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
TREE = "F:/TanitAD-scratch/n1dc/build/n1dh_b1/tree"
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE + "/taniteval", TREE + "/stack/tests"):
    sys.path.insert(0, p)
from tanitad.data import v8f_mini_data as D  # noqa: E402
import n1dh_klim as K  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_n1dw import N1, RBD, _argv, load_J, make_batch, write_nav_npz, write_rbd_npz  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
root = out / "world"
w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
J = D.load_join(root / "join_train.npz")
world = {"root": root, "rbd": write_rbd_npz(root / "rbd.npz", J), "nav": write_nav_npz(root / "nav.npz", J), **w}
assert TR.main(_argv(world, "klim_arm", ["--steps", "60", *N1(world), *RBD(world), "--aux-route", "cond", "--w-aux-route", "0.5", "--detach-upper"])) == 0
b, J, _ = make_batch(world, n=32, seed=1, with_rb=False)
torch.save(b, out / "batch.pt")
rc = K.main(["--ckpt", str(root / "klim_arm" / "ckpt_last.pt"), "--batch-pt", str(out / "batch.pt"), "--out", str(out / "KLIM_synthetic.json")])
rec = json.load(open(out / "KLIM_synthetic.json", encoding="utf-8"))
print(json.dumps({k: rec[k] for k in ("n_labelled_rows", "majority_share", "gap_logged_minus_causal", "gap_logged_minus_off")}), {v: rec[v]["acc"] for v in K.VARIANTS})
sys.exit(rc)
