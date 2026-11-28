"""collect_p.py -- the state-collection step of the pixel read alone (latvis_gw.cmd_collect, unchanged), for one arm of CELLPRED.  CPU.
  python collect_p.py P_s0 C:/Users/Admin/v8proof_c/ckpt/P_s0/ckpt_last.pt
Writes latents/<ARM>.pt in this folder (the same file the pixel job writes and then finds cached).  The built level-1 predictor kind is asserted
through LATVIS_EXPECT_PRED=cell (set here)."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["LATVIS_EXPECT_PRED"] = "cell"
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
sys.path.insert(0, HERE)
import latvis_gw as T  # noqa: E402

arm = {"name": sys.argv[1], "ckpt": sys.argv[2].replace(os.sep, "/"), "width": 640, "kind": "grid", "seed": int(sys.argv[1].rsplit("_s", 1)[1]), "type": sys.argv[1].split("_s")[0]}
T.assert_tree()
sha = T.check_sha(arm["ckpt"])
T.cmd_collect([arm], {arm["name"]: sha})
print("ZZ COLLECT DONE", arm["name"], sha[:12])
