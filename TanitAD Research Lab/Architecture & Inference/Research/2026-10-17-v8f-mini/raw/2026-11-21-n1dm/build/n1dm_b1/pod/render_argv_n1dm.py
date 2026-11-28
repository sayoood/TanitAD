"""render_argv_n1dm.py KIND SEED OUT_JSON ARM_OUT -- render an N1DM arm's argv exactly as `strat1_check.py mint` does on the pod (CPU only, no torch). Tree /root/strat1/n1dm/tree; inputs /root/strat1/inputs. Adapted from render_argv_n1dw.py (tree path only)."""
import json
import os
import sys

TREE = "/root/strat1/n1dm/tree"
INP = "/root/strat1/inputs"
sys.path.insert(0, os.path.join(TREE, "stack", "scripts"))
sys.path.insert(0, os.path.join(TREE, "stack"))
import strat1_check as S  # noqa: E402

assert os.path.realpath(S.__file__).startswith(TREE + "/stack/"), S.__file__
kind, seed, out_json, arm_out = sys.argv[1:5]
toks = S.render(kind, seed, INP, arm_out, os.path.join(INP, "strat1", "navtext_nav.npz"), os.path.join(INP, "strat1", "strat_labels_table.npz"),
                base_argvs=os.path.join(INP, "frozen", "argvs_obd.json"), cache_dir="/root/data/cache_pod",
                rb_npz=os.path.join(INP, "strat1", S.rb_names(kind)[0]) if kind in S.RB_KINDS else "")
json.dump(toks, open(out_json, "w", encoding="utf-8"))
print(len(toks), "tokens", out_json)
