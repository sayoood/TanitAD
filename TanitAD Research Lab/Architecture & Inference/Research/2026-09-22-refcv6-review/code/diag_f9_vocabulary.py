"""F9 / Q6 — WHICH vocabulary would a refcv6 run load, and do its guards test
the TENSOR or only the DECLARATION?

⛔ The guards that exist:
  * `refc_v3_train.py:909-914`  -- `--sampler ddim` refuses a build whose
    `core.anchors.v0_conditioned` is False, quoting *"a fixed-path bank carries
    anchor_controls of all zeros"*;
  * `refc.py:2751-2761`         -- the same refusal inside the decoder forward;
  * `refcv6_diffusion.assert_f9_vocabulary` -- F9, opt-in.

All three read `self.anchor_v0_cond` / the `v0_conditioned` ARGUMENT -- a
CONFIG FLAG. None of them reads `anchor_controls`. This arm constructs the
state they describe in their own error messages (flag True, tensor all zeros)
and measures whether any of them fires.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from diag_f1_f9_liveness import build, fwd
from tanitad.models import refcv6_diffusion as rv6

def main():
    out = {}
    # the state every guard's message describes: DECLARED v0-conditioned,
    # anchor_controls still the all-zero buffer (i.e. no --anchors was given)
    dec = build(rv6.DiffusionFlags(f9_assert_vocab=True, f9_n_anchors=6))
    with torch.no_grad():
        dec.anchor_controls.zero_()   # the "no --anchors" state: buffer untouched
    out["anchor_v0_cond_flag"] = bool(dec.anchor_v0_cond)
    out["anchor_controls_all_zero"] = bool(float(dec.anchor_controls.abs().sum()) == 0.0)
    try:
        o = fwd(dec, steps=2)
        out["forward_raised"] = False
        bank = o["anchor_bank"]
        out["bank_shape"] = list(bank.shape)
        # DEGENERACY: are all N candidates the same path?
        spread = float((bank - bank[:, :1]).abs().max())
        out["bank_max_spread_across_anchors_m"] = round(spread, 9)
        out["bank_is_degenerate_all_identical"] = bool(spread == 0.0)
        out["bank_lateral_max_abs_m"] = round(float(bank[..., 1].abs().max()), 9)
        out["oracle_in_vocabulary_is_a_single_line"] = out["bank_is_degenerate_all_identical"]
    except Exception as e:
        out["forward_raised"] = f"{type(e).__name__}: {str(e)[:120]}"
    # CONTROL that must read a KNOWN value: with real controls the bank spreads
    dec2 = build(rv6.DiffusionFlags(f9_assert_vocab=True, f9_n_anchors=6))
    with torch.no_grad():
        dec2.anchor_controls.copy_(torch.stack(
            [torch.linspace(-2, 2, dec2.anchors.shape[0]),
             torch.linspace(-1.5, 1.5, dec2.anchors.shape[0])], -1))
    b2 = fwd(dec2, steps=2)["anchor_bank"]
    out["CONTROL_with_real_controls_spread_m"] = round(
        float((b2 - b2[:, :1]).abs().max()), 6)
    # does assert_f9_vocabulary see the tensor at all?
    import inspect
    out["assert_f9_signature"] = str(inspect.signature(rv6.assert_f9_vocabulary))
    out["assert_f9_params"] = list(inspect.signature(rv6.assert_f9_vocabulary).parameters)
    out["assert_f9_can_see_the_tensor"] = any(
        "control" in p or "anchor_t" in p
        for p in out["assert_f9_params"])
    # and the trainer-side gate
    src = Path("D:/Projects/TanitAD/stack/scripts/refc_v3_train.py").read_text(encoding="utf-8")
    out["trainer_ddim_gate_reads"] = "core.anchors.v0_conditioned" in src
    out["trainer_refuses_missing_anchors"] = ("NO --anchors GIVEN" in src)
    out["trainer_missing_anchors_is_WARNING_not_refusal"] = (
        "print(\"[v3] ⛔ anchors: NO --anchors GIVEN" in src)
    txt = json.dumps(out, indent=2, default=str); print(txt)
    (Path(__file__).resolve().parents[1] / "raw" / "f9_vocabulary.json").write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    sys.exit(main())
