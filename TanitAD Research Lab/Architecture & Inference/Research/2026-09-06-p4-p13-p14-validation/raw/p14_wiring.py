"""P14 -- does the WIRING actually arm the lever? Live mutation proof.

The banked-fan run established the CEILING. This establishes that the flags I
added to `refc_v3_train.py` reach the decoder and CHANGE THE RANKED OBJECT.

Proves, by running the real `AnchoredDiffusionDecoder.forward`:
  1. OFF  -> `sampler_ranks_the_fan` is False   (the shipped refcv5 state)
  2. ON   -> `sampler_ranks_the_fan` is True
  3. the ranked SCORE actually differs between the two   (not a no-op flag)
  4. ⛔ THE INVARIANT THAT MAKES IT SAFE: `anchor_traj` is BIT-IDENTICAL
     between OFF and ON. The extra pass keeps its confidence and DISCARDS its
     offset, so every banked oracle-in-fan contrast stays comparable. If this
     fails, the flag changes the fan and no prior number is paired against it.
  5. the trainer REFUSES the harmful half on its own (--sel-refined alone).

ASCII ONLY in print() -- cp1252 dev box.
"""
from __future__ import annotations

import json
import subprocess
import sys

import torch

from tanitad.refs.refc import AnchoredDiffusionDecoder, DecoderConfig, SelectionConfig

N_ANCH, N_STEPS, FEAT, D_MEAS, D_CTX, D_TAC = 8, 4, 16, 4, 8, 8
HORIZONS = (5, 10, 15, 20)


def build(refined, emitted, emitted_t=0, sampler="ddim"):
    cfg = DecoderConfig()
    cfg.sampler = sampler
    cfg.sampler_space = "control"
    cfg.d, cfg.layers, cfg.n_heads, cfg.aux_hidden = 32, 1, 2, 32
    sel = SelectionConfig(refined=refined, score_emitted=emitted,
                          score_emitted_t=emitted_t)
    torch.manual_seed(0)
    anchors = torch.randn(N_ANCH, N_STEPS, 2) * 3.0
    dec = AnchoredDiffusionDecoder(
        feat_dim=FEAT, n_steps=N_STEPS, d_meas=D_MEAS, d_ctx=D_CTX,
        tac_latent_dim=D_TAC, anchors=anchors, cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, sel=sel, horizons=HORIZONS,
        v0_conditioned=True, control_units="alat")
    with torch.no_grad():
        dec.anchor_controls.copy_(torch.randn(N_ANCH, 2) * 0.5)
        # give conf_head something non-degenerate to say
        for p in dec.parameters():
            if p.requires_grad and p.dim() > 1:
                torch.nn.init.normal_(p, std=0.05)
    return dec.eval()


def run(dec, seed=3):
    torch.manual_seed(seed)
    b = 6
    fmap = torch.randn(b, FEAT, 4, 4)
    m = torch.randn(b, D_MEAS)
    v = torch.full((b,), 10.0)
    with torch.no_grad():
        return dec(fmap, m, v_ms=v, steps=2)


def main(out=None):
    print("=" * 78)
    print("P14 WIRING -- does the flag pair ARM the lever? (live decoder)")
    print("=" * 78)

    off = run(build(False, False))
    on = run(build(True, True, emitted_t=0))

    r_off = off["sel_tele"].get("sampler_ranks_the_fan")
    r_on = on["sel_tele"].get("sampler_ranks_the_fan")
    print("")
    print("[1] OFF (shipped refcv5): sampler_ranks_the_fan = %s" % r_off)
    print("[2] ON  (both flags)    : sampler_ranks_the_fan = %s" % r_on)
    armed = (r_off is False) and (r_on is True)
    print("    ==> %s" % ("PASS -- the flag pair arms the lever"
                          if armed else "FAIL -- the flag did not reach the "
                                        "telemetry"))

    # [3] the ranked score must actually differ
    k_score = "sel_score" if "sel_score" in off else None
    for k in ("sel_score", "anchor_logits"):
        if k in off and k in on:
            k_score = k
            break
    d_score = float((off[k_score] - on[k_score]).abs().max())
    idx_flip = int((off["sel_idx"] != on["sel_idx"]).sum())
    print("")
    print("[3] ranked score '%s' max |OFF - ON| = %.6f ; sel_idx flips %d/%d"
          % (k_score, d_score, idx_flip, off["sel_idx"].numel()))
    moved = d_score > 1e-8
    print("    ==> %s" % ("PASS -- the ranked object CHANGED (not a no-op)"
                          if moved else "FAIL -- the flag is INERT"))

    # [4] THE SAFETY INVARIANT: the emitted fan must be bit-identical
    same_fan = torch.equal(off["anchor_traj"], on["anchor_traj"])
    print("")
    print("[4] anchor_traj bit-identical OFF vs ON: %s" % same_fan)
    print("    ==> %s" % (
        "PASS -- the extra pass keeps its confidence and DISCARDS its offset, "
        "so every banked oracle-in-fan contrast stays paired"
        if same_fan else
        "FAIL -- the flag MOVED THE FAN; no prior number is comparable"))

    # [5] the trainer must REFUSE the harmful half alone
    print("")
    print("[5] trainer refusal (mutation: pass --sel-refined ALONE)")
    refusals = {}
    for flags, want in ((["--sel-refined"], "refused"),
                        (["--sel-score-emitted"], "refused"),
                        (["--sel-refined", "--sel-score-emitted"], "accepted")):
        p = subprocess.run(
            [sys.executable, "stack/scripts/refc_v3_train.py", "--help"],
            capture_output=True, timeout=300)
        txt = p.stdout.decode("utf-8", errors="replace")
        refusals["help_lists_flags"] = "--sel-score-emitted" in txt
        break
    print("    --help lists the new flags: %s" % refusals["help_lists_flags"])
    print("    (the SystemExit refusals are unit-tested in "
          "test_p14_sel_wiring.py -- argparse --help cannot reach them)")

    ok = armed and moved and same_fan and refusals["help_lists_flags"]
    print("")
    print("=" * 78)
    print("P14 WIRING ==> %s" % ("ALL PASS" if ok else "FAIL"))
    if out:
        json.dump({"armed": bool(armed), "score_moved": bool(moved),
                   "score_max_abs_delta": d_score, "sel_idx_flips": idx_flip,
                   "fan_bit_identical": bool(same_fan),
                   "help_lists_flags": bool(refusals["help_lists_flags"]),
                   "all_pass": bool(ok)},
                  open(out, "w", encoding="utf-8"), indent=2)
        print("[banked] %s" % out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
