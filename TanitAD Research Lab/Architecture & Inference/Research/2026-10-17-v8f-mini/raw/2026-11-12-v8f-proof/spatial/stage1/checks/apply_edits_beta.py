import sys
P = "F:/TanitAD-scratch/v8proof/spatial/stage1/tree_patch/stack/tanitad/models/v8f_beta.py"
s = open(P, encoding="utf-8", newline="").read()
def sub(old, new):
    global s
    if s.count(old) != 1:
        sys.exit("MISMATCH " + old[:60])
    s = s.replace(old, new)
sub('''# ------------------------------------------------------------------------------------------------ the scorer
def _dev(wm)''', '''# ------------------------------------------------------------------------------------------------ the scorer
def refuse_grid_state(wm) -> None:
    """PREREG_V8F_PROOF_ADD12 (SPATIAL): the beta planner was written for the mean level-1 state (width 128, goal latents at level 2). It is NOT ported to a
    grid state: refuse, loudly, rather than score candidates on a state whose width and meaning it was never validated for."""
    if getattr(wm, "state_mode", "mean") != "mean":
        raise ValueError(f"rung beta REFUSES a world model with wm_state {wm.state_mode!r}: the planner is not ported to the grid state "
                         "(PREREG_V8F_PROOF_ADD12 builds the lever for the training and the T0 / T1-kin reads only)")


def _dev(wm)''')
sub('''        raise ValueError("rung beta needs a checkpoint with a world model (arm M2a / M2b); this one was built with wm='off'")
''', '''        raise ValueError("rung beta needs a checkpoint with a world model (arm M2a / M2b); this one was built with wm='off'")
    refuse_grid_state(model.wm)
''')
sub('''    dev, d2 = _dev(wm), wm.stack.cfg[WM_LEVEL - 1].latent_dim
    out = []''', '''    refuse_grid_state(wm)
    dev, d2 = _dev(wm), wm.stack.cfg[WM_LEVEL - 1].latent_dim
    out = []''')
open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
