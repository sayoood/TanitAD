"""R2 side-finding probe (NOT fixed here -- R2 is declared out of scope): two ways a real
`--nav-cond --nav-labels` launch of train_v6_staged.py dies at the TIP. Read-only, CPU.

(1) BUILD TIME: build_stack_from_args -> V6Stack.assert_isolation -> synthetic_batch, which
    carries no nav token -> NavTokenMissing. MEASURED by executing it on a tiny stack.
(2) STEP 1: in train()'s batch dict literal the NavEmitter's `**{nav_token, nav_args}` splat is
    followed by the literal keys "nav_token": b.get("nav_token") -- a later key WINS, and the
    collated dataset item carries no nav_token, so the emitted command is replaced by None.
    MEASURED statically (AST key order of the one dict literal) + the semantics executed.
Usage: python r2_nav_defects_probe.py <tip_stack_root>
"""
import ast
import json
import sys

root = sys.argv[1]
sys.path.insert(0, root)
sys.path.insert(0, root + "/scripts")
out = {}

# ---- (1) build-time
import torch  # noqa: E402
from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402
cfg = V6Config(
    encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32, patch_size=16,
                          d_model=32, depth=1, n_heads=2),
    readout=ReadoutConfig(grid=4, d_readout=8),
    predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4, horizons=(1,),
                              action_dim=3),
    d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32, f_hidden_tac=32, f_hidden_str=32,
    d_plan_feat=16, emission_hidden=16, n_candidates=3, aux_hidden=16, sigreg_slices=8,
    nav_cond=True)
try:
    V6Stack(cfg).assert_isolation(batch_size=1, strict=True)
    out["build_time_assert_isolation_with_nav_cond"] = "PASSED"
except Exception as e:                                   # noqa: BLE001
    out["build_time_assert_isolation_with_nav_cond"] = f"RAISED {type(e).__name__}: {str(e)[:120]}"

# ---- (2) the batch dict literal
src = open(root + "/scripts/train_v6_staged.py", encoding="utf-8").read()
tree = ast.parse(src)
found = []
for node in ast.walk(tree):
    if isinstance(node, ast.Dict):
        keys = []
        for k, v in zip(node.keys, node.values):
            if k is None:
                seg = ast.get_source_segment(src, v) or ""
                keys.append(("**splat", "nav_token" in seg, v.lineno))
            elif isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append((k.value, None, k.lineno))
        names = [x[0] for x in keys]
        if "nav_token" in names and any(x[0] == "**splat" and x[1] for x in keys):
            splat_i = max(i for i, x in enumerate(keys) if x[0] == "**splat" and x[1])
            lit_i = names.index("nav_token")
            found.append({"dict_line": node.lineno, "splat_carrying_nav_line": keys[splat_i][2],
                          "literal_nav_token_line": keys[lit_i][2],
                          "literal_comes_AFTER_splat": lit_i > splat_i})
out["batch_dict_literal"] = found
d = {**{"nav_token": "EMITTED", "nav_args": "EMITTED"}, "nav_token": None, "nav_args": None}
out["python_semantics_later_key_wins"] = d
print(json.dumps(out, indent=1))
