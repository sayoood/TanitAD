"""strat1_k4_eval.py -- K4 (C-RAND): the T0 evaluator run on a STRAT1 S1 checkpoint with its ctx_head RE-INITIALISED in memory (torch seed 0).  Eval only: no checkpoint is written, the
file on disk is untouched.  Usage: the same arguments as `v8f_mini_eval.py run`.  The evaluator's own load path runs unchanged; only `V8FMini.load_state_dict` is wrapped so that the
head's parameters are re-drawn after the strict load.  reinit_ctx_head() is the unit under test."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def reinit_ctx_head(model, seed=0):
    """re-initialise every parameterised submodule of model.ctx_head (torch's default init) under a local seed.  -> number of tensors changed"""
    import torch
    if getattr(model, "ctx_head", None) is None:
        raise RuntimeError("K4 needs a model with a ctx_head")
    before = [p.detach().clone() for p in model.ctx_head.parameters()]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for m in model.ctx_head.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    return sum(0 if torch.equal(b, p.detach()) else 1 for b, p in zip(before, model.ctx_head.parameters()))


def main():
    root = os.environ.get("STRAT1_TREE", HERE + "/tree")
    for p in (root + "/stack", root + "/taniteval", root + "/stack/scripts"):
        sys.path.insert(0, p)
    import v8f_mini_eval as E
    from tanitad.models import v8f_mini as M
    orig = M.V8FMini.load_state_dict

    def patched(self, *a, **k):
        r = orig(self, *a, **k)
        n = reinit_ctx_head(self, 0)
        print(f"K4: ctx_head re-initialised in memory (seed 0), {n} tensors changed", flush=True)
        return r
    M.V8FMini.load_state_dict = patched
    E.main(sys.argv[1:])


if __name__ == "__main__":
    main()
