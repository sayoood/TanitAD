"""mk_tiny_arm.py OUTDIR -- train a TINY real N1DM arm on the synthetic world (CPU, a few steps) with the trainer's own main(), plus the tiny base (init) checkpoint, so the post-training lines of n1dm_arm.sh (post check, zero-grad gate,
K-CLIP) can each be run once on a REAL finished checkpoint (N1DH Amendment 4 lesson). Writes OUTDIR/init/ckpt_last.pt, OUTDIR/arm/ckpt_last.pt (+ train_log.jsonl, summary.json, selfcheck.json), OUTDIR/argv.json."""
import json
import sys
from pathlib import Path

REPO = Path("F:/TanitAD-scratch/n1dc/build/n1dm_b1/tree")
for p in (REPO / "stack", REPO / "stack" / "scripts", REPO / "taniteval", REPO / "stack" / "tests"):
    sys.path.insert(0, str(p))
import torch  # noqa: E402

torch.set_num_threads(1)
from tanitad.data import v8f_mini_data as D  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_n1dw import N1, RBD, _argv, write_nav_npz, write_rbd_npz  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
w = D.make_synthetic_world(out, n_train=6, n_eval=2, T=60, seed=0)
J = D.load_join(Path(out) / "join_train.npz")
world = {"root": out, "rbd": write_rbd_npz(out / "rbd.npz", J), "nav": write_nav_npz(out / "nav.npz", J), **w}
assert TR.main(_argv(world, "init", ["--steps", "3", *N1(world)])) == 0
argv = _argv(world, "arm", ["--steps", "4", "--init-ckpt", str(out / "init" / "ckpt_last.pt"), *N1(world), *RBD(world), "--aux-route", "modes", "--w-aux-route", "0.5", "--detach-upper"])
assert TR.main(argv) == 0
json.dump(argv, open(out / "argv.json", "w"))
print("TINY ARM OK", out / "arm" / "ckpt_last.pt")
