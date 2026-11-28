"""navrb2_k4_eval.py -- K4 (C-RAND) for an R arm: the NAVRB2 T0 evaluator (v8f_mini_eval_rb.py, which pins the tool's sha256) run with the checkpoint's ctx_head RE-INITIALISED in memory (torch seed 0).
Eval only: the file on disk is untouched.  Same unit under test as strat1_k4_eval.reinit_ctx_head (imported UNEDITED).  Usage: the arguments of `v8f_mini_eval_rb.py run`."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    root = os.environ.get("NAVRB2_TREE", HERE + "/tree")
    for p in (root + "/stack", root + "/taniteval", root + "/stack/scripts"):
        sys.path.insert(0, p)
    sys.path.insert(0, os.environ.get("NAVRB2_STRAT1_READ", "F:/TanitAD-scratch/strat1/read"))
    import strat1_k4_eval as K4
    import v8f_mini_eval_rb as ER
    from tanitad.models import v8f_mini as M
    orig = M.V8FMini.load_state_dict

    def patched(self, *a, **k):
        r = orig(self, *a, **k)
        n = K4.reinit_ctx_head(self, 0)
        print(f"K4: ctx_head re-initialised in memory (seed 0), {n} tensors changed", flush=True)
        return r
    M.V8FMini.load_state_dict = patched
    ER.main(sys.argv[1:])


if __name__ == "__main__":
    main()
