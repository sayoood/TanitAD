"""Is "one un-no_grad forward away" HYPOTHETICAL, or is there a REAL call site?

`v6_chain.py::write_dry_predecessor` builds a stack, applies the stage freeze, and then does

    tr = [p for p in stack.parameters() if p.requires_grad]
    sum((p * p).sum() for p in tr).backward()
    opt.step()

-- a dummy loss over EVERY trainable parameter. That never calls `_EmaCopy.forward`, so the
`no_grad` that made the un-freeze harmless in the TRAINER does not protect it here.

This reproduces that exact pattern under both freeze rules and asks whether the teacher's
weights MOVE. Control that must read a known value: the encoder STUDENT must move in both arms
(otherwise the rig is dead and a null means nothing).

ASCII-only output. CPU only, zero GPU.
"""
import copy, json, sys
from pathlib import Path

import torch

CLONE = Path(r"C:\Users\Admin\tanitad-emaforensics")
sys.path.insert(0, str(CLONE / "stack"))
sys.path.insert(0, str(CLONE / "stack" / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import (V6Config, V6Stack, _EmaCopy,               # noqa: E402
                               apply_stage_freeze, stage_trainable_groups)


def tiny():
    return V6Config(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1,), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)


def build():
    torch.manual_seed(0)
    st = V6Stack(tiny())
    st.ema_o5_enc = _EmaCopy(st.encoder, 0.996)
    st.ema_o5_ro = _EmaCopy(st.readout, 0.996)
    return st


def group_map_alone(st, stage):
    groups = set(stage_trainable_groups(stage))
    for name, p in st.named_parameters():
        p.requires_grad_(st.group_of(name) in groups)


def run(freeze_fn, label):
    st = build()
    freeze_fn(st, "S-W")
    before = {k: v.detach().clone() for k, v in st.state_dict().items()}
    # ---- v6_chain.py::write_dry_predecessor, verbatim in shape ----
    tr = [p for p in st.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tr, lr=1e-3, weight_decay=0.05)
    sum((p * p).sum() for p in tr).backward()
    opt.step()
    after = st.state_dict()

    def moved(prefix):
        ks = [k for k in before if k.startswith(prefix)]
        return sum(1 for k in ks if not torch.equal(before[k], after[k])), len(ks)

    tm, tn = moved("ema_o5_enc.")
    rm, rn = moved("ema_o5_ro.")
    sm, sn = moved("encoder.")
    return {"arm": label,
            "n_trainable_params": int(sum(p.numel() for p in tr)),
            "TEACHER_enc_tensors_MOVED": "%d of %d" % (tm, tn),
            "TEACHER_ro_tensors_MOVED": "%d of %d" % (rm, rn),
            "CONTROL_encoder_student_MOVED": "%d of %d" % (sm, sn)}


OUT = [
    run(group_map_alone, "PRE-2026-09-06 freeze (group map alone) -- THE DEFECT"),
    run(apply_stage_freeze, "CURRENT freeze (honours declare_grad_unreachable)"),
]
print(json.dumps(OUT, indent=1))
