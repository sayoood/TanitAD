"""PI RULING 2026-09-26 (SPEC_REFCV7 §7 / A2, D-REFCV7-E1): the speed-ceiling argmax filter is
INFERENCE-ONLY. CRLF-preserving exact-match edits on code/fix/stack/tanitad/refs/refc.py."""
import sys

P = sys.argv[1]
d = open(P, "rb").read()
assert d.count(b"\r\n") == d.count(b"\n"), "expected an all-CRLF file"
s = d.decode("utf-8").replace("\r\n", "\n")


def edit(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch] {tag}: anchor found {n} times")
    s = s.replace(old, new)


edit('''    forward -> {anchor_logits [B, N], anchor_traj [B, N, S, 2], offset (base)
    [B, N, S, 2], traj [B, S, 2] (selected), sel_idx [B]}.
    """
''', '''    forward -> {anchor_logits [B, N], anchor_traj [B, N, S, 2], offset (base)
    [B, N, S, 2], traj [B, S, 2] (selected), sel_idx [B]}.
    """

    #: ⛔⛔ PI RULING 2026-09-26 (SPEC_REFCV7 §7 / A2, D-REFCV7-E1): the speed-ceiling argmax
    #: filter is INFERENCE-ONLY. It has no parameters, but in TRAINING it changed WHICH candidate
    #: is selected, and the selected `traj` feeds `law_pred = law_head(cat(pooled, traj))`, whose
    #: MSE is a training loss -- so a filter meant for the emitted plan was also shaping the LAW
    #: head's input (and its gradient into the fan) from an EGO-FUTURE channel. False keeps the
    #: training-time selection unmasked. ⚠️ True restores the pre-ruling behaviour and exists ONLY
    #: for the deliberate-regression test (`test_speed_ceiling_inference_only.py`); G-DVB refuses
    #: a built decoder that carries it.
    speed_ceiling_in_training: bool = False
''', "class attribute")

edit('''        if self.speed_ceiling_filter and v_limit_ms is not None:
            _keep, _st = v6sel.SpeedCeilingFilter(
''', '''        if (self.speed_ceiling_filter and v_limit_ms is not None
                and (not self.training or self.speed_ceiling_in_training)):   # PI 2026-09-26 A2
            _keep, _st = v6sel.SpeedCeilingFilter(
''', "the guard")

open(P, "wb").write(s.replace("\n", "\r\n").encode("utf-8"))
print("refc.py: speed-ceiling filter is inference-only (CRLF kept)")
