"""dk_eval_wrap: run the dagt1 tree's `v8f_mini_eval.py run` on a checkpoint whose cfg carries `launch_intent_w` (the recover-C0 family: wsp_common.load_model drops it after asserting 0.0).
The eval tool builds `V8FMiniConfig(**ck["cfg"])` directly and would refuse the key; this wrapper drops it (asserting it is 0.0) and changes nothing else. Produces eval_<tag>.npz / .json (the plans `pred` [2399, 30, 2]) for dk141.py.
usage: dk_eval_wrap.py run --ckpt ... --tag ... [the eval tool's own flags]      (dev-box CPU, zero training)"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
for p in (TREE + "/stack", TREE + "/taniteval", TREE + "/stack/scripts"):
    sys.path.insert(0, p)
import tanitad.models.v8f_mini as M

assert os.path.abspath(M.__file__).replace("\\", "/").startswith(TREE), M.__file__
_orig = M.V8FMiniConfig


def _cfg(**kw):
    assert kw.pop("launch_intent_w", 0.0) == 0.0
    return _orig(**kw)


M.V8FMiniConfig = _cfg
import v8f_mini_eval as EV

if __name__ == "__main__":
    sys.exit(EV.main(sys.argv[1:]))
