"""make_tiny_ckpt.py: builds a tiny V8FMini checkpoint (d 32, 40 x 16 grid state = 640, lead head on, random weights) in the layout the trainer writes
({"model", "cfg", "grid"}) from the patched tree. Used by the plumbing run and the tests; never a trained model."""
import os
import sys

TREE = "C:/Users/Admin/v8proof_c/leadsup/tree_patch"
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)


def build(out_path, seed=0, lead_sup="on", tree=TREE, head_scale=1.0):
    sp = tree + "/stack"
    for p in (sp, sp + "/scripts"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace(os.sep, "/").startswith(tree), tanitad.__file__
    from tanitad.models import v8f_mini as M
    import v8f_mini_train as TR
    extra = ["--lead-sup", "on", "--w-lead-sup", "1.0"] if lead_sup == "on" else []
    a = TR.build_parser().parse_args(["--out", "x", "--d", "32", "--n-q", "2", "--arm", "M2b", "--wm-state", "grid", "--wm-cell-dim", "16", *extra])
    c0 = TR.make_config(a)
    cfg = M.V8FMiniConfig(**{**c0.__dict__, **TINY})
    torch.manual_seed(seed)
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    if lead_sup == "on":
        with torch.no_grad():                                                       # head_scale 1.0 = the default init
            for p in model.wm.lead_head.parameters():
                p.mul_(head_scale)
    torch.save({"model": model.state_dict(), "cfg": dict(cfg.__dict__), "grid": "pool4x10", "step": 0}, out_path)
    return out_path


if __name__ == "__main__":
    print(build(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0))
