"""Local check of gbo_diagnose.AssignProbe on the toy decoder: the last-layer assignment is captured inside a
box3d_loss_row-like call for BOTH the refined (3 matchings) and the legacy (1 matching) path, and the wrappers are
removed afterwards."""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gbo_diagnose as D  # noqa: E402
import toy_probe_check as T  # noqa: E402  (runs its own checks on import)
from tanitad.models import agent_slots as AS  # noqa: E402
from tanitad.models import box3d_head as B3  # noqa: E402
from tanitad.models import slot_presence as SP  # noqa: E402

dec, model, tgt, vis = T.toy()


def row_refined(slots, t, **k):
    return SP.refined_box3d_losses(slots, t, presence_loss="focal", vis1=True, vis=vis)


def row_legacy(slots, t, **k):
    return B3.box3d_set_loss(slots, t, visible_filter=False)


for name, fn, want_calls in (("refined", row_refined, 3), ("legacy", row_legacy, 1)):
    PB = types.SimpleNamespace(box3d_loss_row=fn)
    ap = D.AssignProbe(PB, AS)
    ap.install(dec)
    orig_ms = ap._orig["match_slots"]
    try:
        ap.active, ap.calls = True, []
        dec.deep_supervision = name == "refined"
        pred = dec(torch.randn(2, 8, 16))
        PB.box3d_loss_row(pred, tgt)
        a = ap.last_assign()
        assert len(ap.calls) == want_calls, (name, len(ap.calls))
        assert a is not None and len(a) == 2 and all(len(x) >= 2 for x in a), a
        assert ap.pred is pred
        # outside the row call nothing is captured
        n0 = len(ap.calls)
        AS.match_slots(pred, tgt)
        assert len(ap.calls) == n0
    finally:
        ap.remove()
    assert AS.match_slots is orig_ms, "match_slots not restored"
    print(name, "calls", want_calls, "assign", a)
print("ASSIGN CHECK PASS")
