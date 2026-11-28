"""H19 BLAST RADIUS -- what a non-kin3 tactical vocabulary actually does to the
H19 anchor prior, established BY EXECUTION rather than by reading the guard.

Zero GPU. CPU smoke rung. ASCII-only output (cp1252 console).

The claim under test (as escalated): "any non-kin3 tactical vocabulary sets
man5 = None and silently DROPS the H19 lateral prior."

Three questions, each with a same-breath control:
  Q1  does the hook emit maneuver_logits under each vocabulary?
  Q2  what does the DECODER actually receive as `maneuver_logits`?
  Q3  does the TACTICAL brain reach the anchor prior under each vocabulary?
      (mutation: move ONLY lat_head_tac/lon_head_tac and watch the tensor the
      decoder receives -- a control that must move under kin3 and must not
      under v7.0, so a probe stuck in either position fails.)
"""
from __future__ import annotations

import sys

import torch

from tanitad.refs import refc, refc_v3 as v3


def build(vocab: str):
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = refc.CNNEncoderConfig(
        in_channels=cfg.core.encoder.in_channels, image_size=64,
        base_width=8, blocks=(1, 1, 1, 1))
    cfg.tac_vocab_version = vocab
    m = v3.RefCV3Model(cfg)
    m.eval()
    return m, cfg


def probe(vocab: str, seed: int = 0):
    torch.manual_seed(seed)
    m, cfg = build(vocab)
    enc = cfg.core.encoder
    b, w = 2, cfg.core.window
    frames = torch.randn(b, w, enc.in_channels, enc.image_size,
                         enc.image_width or enc.image_size)
    v0 = torch.full((b,), 10.0)

    seen = {}
    real = m.core.decoder.forward

    def spy(*a, **kw):
        seen["maneuver_logits"] = kw.get("maneuver_logits")
        seen["lat_prior"] = kw.get("lat_prior")
        seen["lon_prior"] = kw.get("lon_prior")
        return real(*a, **kw)

    m.core.decoder.forward = spy
    with torch.no_grad():
        out = m(frames, v0=v0, steps=0)
    m.core.decoder.forward = real

    return m, cfg, out, seen, (frames, v0)


def rerun(m, frames, v0):
    seen = {}
    real = m.core.decoder.forward

    def spy(*a, **kw):
        seen["maneuver_logits"] = kw.get("maneuver_logits")
        seen["lat_prior"] = kw.get("lat_prior")
        return real(*a, **kw)

    m.core.decoder.forward = spy
    with torch.no_grad():
        out = m(frames, v0=v0, steps=0)
    m.core.decoder.forward = real
    return out, seen


def main():
    print("=" * 74)
    print("H19 BLAST RADIUS -- MEASURED, CPU smoke rung, zero GPU")
    print("=" * 74)
    rows = {}
    for vocab in ("kin3", "v7.0"):
        m, cfg, out, seen, (frames, v0) = probe(vocab)
        ml = seen["maneuver_logits"]
        lp = seen["lat_prior"]

        # Q1: what does the hook itself emit? Call it directly.
        hook = m._hierarchy_hook({}) if hasattr(m, "_hierarchy_hook") else None

        print()
        print("---- vocab = %-6s -------------------------------------------"
              % vocab)
        print("  graft_maneuver (cfg)          : %s" % cfg.core.graft_maneuver)
        print("  maneuver_to_anchor built      : %s"
              % (m.core.decoder.maneuver_to_anchor is not None))
        print("  lat_head_tac.out_features     : %d" % m.lat_head_tac.out_features)
        print("  Q2 decoder RECEIVED man_logits: %s"
              % ("None" if ml is None else "Tensor%s" % (tuple(ml.shape),)))
        print("  Q2 decoder RECEIVED lat_prior : %s"
              % ("None" if lp is None else "Tensor%s" % (tuple(lp.shape),)))
        print("  anchor_logits present         : %s"
              % ("anchor_logits" in out))

        # ---- Q3 MUTATION: move ONLY the tactical action heads. ------------
        base_ml = None if ml is None else ml.clone()
        base_lp = None if lp is None else lp.clone()
        base_al = out["anchor_logits"].clone()
        with torch.no_grad():
            for h in (m.lat_head_tac, m.lon_head_tac):
                h.weight.add_(torch.randn_like(h.weight) * 5.0)
                h.bias.add_(torch.randn_like(h.bias) * 5.0)
        out2, seen2 = rerun(m, frames, v0)
        ml2, lp2 = seen2["maneuver_logits"], seen2["lat_prior"]

        def moved(a, bb):
            if a is None and bb is None:
                return "n/a (None both)"
            if (a is None) != (bb is None):
                return "SHAPE-CHANGED"
            d = float((a - bb).abs().max())
            return "MOVED max|d|=%.6g" % d if d > 0 else "UNCHANGED (0.0)"

        print("  Q3 mutation lat/lon_head_tac (+N(0,5) on W and b):")
        print("     -> decoder maneuver_logits : %s" % moved(base_ml, ml2))
        print("     -> decoder lat_prior       : %s" % moved(base_lp, lp2))
        print("     -> anchor_logits           : %s"
              % moved(base_al, out2["anchor_logits"]))
        rows[vocab] = {
            "man_logits_is_none": ml is None,
            "man_moves": None if ml is None else float((base_ml - ml2).abs().max()),
            "latprior_moves": None if lp is None else float((base_lp - lp2).abs().max()),
        }

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    k, s = rows["kin3"], rows["v7.0"]
    print("  H19 prior LIVE (decoder got a maneuver_logits tensor):")
    print("     kin3 : %s      v7.0 : %s"
          % (not k["man_logits_is_none"], not s["man_logits_is_none"]))
    print("  TACTICAL BRAIN reaches the H19 prior (mutation moves it):")
    print("     kin3 : %s      v7.0 : %s"
          % (k["man_moves"], s["man_moves"]))
    print("  TACTICAL BRAIN reaches the D-TAC1 lat_prior:")
    print("     kin3 : %s      v7.0 : %s"
          % (k["latprior_moves"], s["latprior_moves"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
