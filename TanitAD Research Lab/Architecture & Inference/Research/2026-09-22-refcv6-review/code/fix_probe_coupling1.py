"""THE PROPOSED FIX, SIMULATED AND MEASURED — without editing the shared tree.

⭐ Rule Zero: *"X is refuted -- Y is the next lever -- here is Y's RESULT."*
§2 of the review shows coupling (1) never runs. This proves the two-line fix
turns the guard GREEN, so whoever lands it is not guessing.

The fix, as it would be written in `refc.py`:
  * `:2997`  `self._sample(kv, cond, bank, v_ms, steps, agent_tokens,
               agent_pad, agent_pos, bev)`      <- add `bev`
  * `:2884`  `self._decode(kv, cond, x0, 0, agent_tokens, agent_pad,
               agent_pos, bev)`                 <- add `bev`

Here it is simulated by wrapping the two bound methods so they receive the
`bev` the forward was called with. ⛔ Nothing in `stack/` is modified.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from diag_decoder_couplings import build_decoder, make_inputs, Recorder
from tanitad.models import refcv6_diffusion as rv6

def count(dec, fn):
    rec = Recorder(); rec.attach(dec)
    try:
        with torch.no_grad():
            fn()
    finally:
        rows = [r for r in rec.rows if r["kind"] == "bev_wp"]; rec.remove()
    return len(rows), rows

def apply_fix(dec, bev):
    """The two missing `bev` arguments, injected at the two call sites."""
    _sample, _decode = dec._sample, dec._decode
    def sample(kv, cond, bank, v_ms, steps, agents=None, agent_pad=None,
               agent_pos=None, bev_=None):
        return _sample(kv, cond, bank, v_ms, steps, agents, agent_pad,
                       agent_pos, bev if bev_ is None else bev_)
    def decode(kv, cond, x_est, t_idx, agents=None, agent_pad=None,
               agent_pos=None, bev_=None):
        return _decode(kv, cond, x_est, t_idx, agents, agent_pad, agent_pos,
                       bev if bev_ is None else bev_)
    dec._sample, dec._decode = sample, decode
    return lambda: (setattr(dec, "_sample", _sample),
                    setattr(dec, "_decode", _decode))

def main():
    flags = rv6.DiffusionFlags(f1_random_t=True, f2_dd_step=True,
                               f3_per_layer=True, f4_adaln=True,
                               f5_emitting_conf=True)
    dec, params = build_decoder(flags=flags); dec.eval()
    inp = make_inputs(dec)
    L, K = len(dec.layers), 2
    kw = dict(steps=2, v_ms=inp["v_ms"], agent_tokens=inp["agent_tokens"],
              agent_pad=inp["agent_pad"], agent_pos=inp["agent_pos"],
              bev=inp["bev"])
    out = {"n_layers": L, "n_denoise_passes": K}

    before, _ = count(dec, lambda: dec(inp["fmap"], inp["m"], **kw))
    # ⛔ THE SAMPLER IS STOCHASTIC (`_sample` draws a fresh eps). A bit-identity
    # claim across two forwards is meaningless without a seed -- measured here
    # the hard way: unseeded, the "identity" read False purely from inference
    # noise. CLAUDE.md's third variance, in miniature.
    torch.manual_seed(4242)
    with torch.no_grad():
        base_traj = dec(inp["fmap"], inp["m"], **kw)["traj"].clone()
    out["BEFORE_fix_sampler_fires"] = before

    undo = apply_fix(dec, inp["bev"])
    after, rows = count(dec, lambda: dec(inp["fmap"], inp["m"], **kw))
    kw0 = dict(kw); kw0["steps"] = 0
    after_cls, _ = count(dec, lambda: dec(inp["fmap"], inp["m"], **kw0))
    out["AFTER_fix_sampler_fires"] = after
    out["AFTER_fix_classifier_fires"] = after_cls
    # a ddim build runs the sampler even at steps=0 (`_sample` falls back to
    # cfg.sampler_steps), so BOTH modes expect 1 classifier + K denoise passes.
    out["AFTER_fix_expected_sampler"] = L * (K + 1)   # 1 classifier + K denoise
    out["AFTER_fix_ctx_shape"] = rows[0]["ctx_shape"] if rows else None
    out["AFTER_fix_wp_shape"] = rows[0]["wp_shape"] if rows else None
    # ⭐ REMOVABILITY still holds: the gate is zero-init, so the emitted
    # trajectory must be BIT-IDENTICAL even though the module now runs.
    torch.manual_seed(4242)
    with torch.no_grad():
        fixed_traj = dec(inp["fmap"], inp["m"], **kw)["traj"]
    out["gate_zero_so_traj_is_BIT_IDENTICAL"] = bool(
        torch.equal(base_traj, fixed_traj))
    # ...and it is GATED, not dead: open the gate and the plan must move.
    # ⛔ `control_head` / `cascade.control_heads` are ZERO-INIT, so at
    # construction `du = 0` and `traj` is the anchored Gaussian REGARDLESS of
    # every decoder weight. Measured the hard way: without un-zeroing, opening
    # the gate moved the plan by EXACTLY 0.0 and looked like a dead coupling.
    from diag_f3_f4_zeroinit import unzero_
    unzero_(dec, scale=0.2)
    torch.manual_seed(4242)
    with torch.no_grad():
        base_traj = dec(inp["fmap"], inp["m"], **kw)["traj"].clone()
    with torch.no_grad():
        for ly in dec.layers:
            ly.bev_wp.gate.fill_(1.0)
            ly.bev_wp.output_proj.weight.normal_(0, 0.1)
        torch.manual_seed(4242)
        moved = float((dec(inp["fmap"], inp["m"], **kw)["traj"]
                       - base_traj).abs().max())
    out["gate_opened_traj_moves_m"] = round(moved, 6)
    undo()
    out["VERDICT"] = ("FIX WORKS: the guard goes GREEN, the plan is bit-identical "
                      "while the gate is zero, and the coupling is gated-not-dead"
                      if (after >= out["AFTER_fix_expected_sampler"]
                          and after_cls >= L
                          and out["gate_zero_so_traj_is_BIT_IDENTICAL"]
                          and moved > 0) else "INCONCLUSIVE - read the arms")
    txt = json.dumps(out, indent=2); print(txt)
    (Path(__file__).resolve().parents[1] / "raw" / "fix_probe_coupling1.json").write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    sys.exit(main())
