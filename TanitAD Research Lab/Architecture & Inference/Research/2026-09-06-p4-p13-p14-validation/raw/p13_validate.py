"""P13 -- DD's `t ~ U[0, t_max)` TRAINING DRAW. Mutation validation.

Executes PREREG.md section 2 exactly:
  B1  the flag must acquire a consumer, PROVABLY -- a live spy on the real
      decoder records the timesteps the training pass actually consumes,
      under t_max = 50 and t_max = 1, before and after the fix.
  B2  the noising must be DD's -- sigma(t) against DDIMSchedule, sigma(8)
      against the published 0.0316.
  R   deliberate regression: t_max = 1 (the constant t = 0 draw) MUST be
      distinguishable from t_max = 50, or the instrument is blind.
  controls at KNOWN values: sigma(0), sigma(8), mean drawn t, current variance.

⛔ THE POINT OF THE SPY. An AST census once read 0 suspects on BOTH the fixed
and the broken trainer (CLAUDE.md). So this does not inspect source: it RUNS
the real `AnchoredDiffusionDecoder._sample`, records what reaches
`_decode_ctrl`, and reintroduces/repairs the defect to prove both branches are
reachable and distinguishable.

ASCII ONLY in print() -- cp1252 dev box.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from tanitad.refs import refc_sampler as rs
from tanitad.refs.refc import AnchoredDiffusionDecoder, DecoderConfig

N_ANCHORS, N_STEPS, FEAT, D_MEAS, D_CTX, D_TAC = 8, 4, 16, 4, 8, 8
HORIZONS = (5, 10, 15, 20)


def build(t_max: int, infer_t: int = 8, steps: int = 2):
    cfg = DecoderConfig()
    cfg.sampler = "ddim"
    cfg.sampler_space = "control"
    cfg.sampler_train_t_max = int(t_max)
    cfg.sampler_infer_t = int(infer_t)
    cfg.sampler_steps = int(steps)
    cfg.d = 32
    cfg.layers = 1
    cfg.n_heads = 2
    cfg.aux_hidden = 32
    anchors = torch.randn(N_ANCHORS, N_STEPS, 2) * 3.0
    dec = AnchoredDiffusionDecoder(
        feat_dim=FEAT, n_steps=N_STEPS, d_meas=D_MEAS, d_ctx=D_CTX,
        tac_latent_dim=D_TAC, anchors=anchors, cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=HORIZONS, v0_conditioned=True,
        control_units="alat")
    with torch.no_grad():
        dec.anchor_controls.copy_(torch.randn(N_ANCHORS, 2) * 0.5)
    return dec


def spy(dec):
    """Record every `t` that reaches `_decode_ctrl`."""
    seen = []
    orig = dec._decode_ctrl

    def wrapped(kv, cond, x_path, t, agents, agent_pad):
        seen.append(t.detach().cpu().numpy().copy())
        return orig(kv, cond, x_path, t, agents, agent_pad)
    dec._decode_ctrl = wrapped
    return seen


# --------------------------------------------------------------- the FIX --- #
def sample_train_p13(dec, kv, cond, bank, v_ms, generator=None):
    """THE P13 REPAIR, as it would live in `_sample`'s training branch.

    DD trains with ONE draw and ONE pass -- not the inference ladder:
        t ~ U[0, t_max) ; x_t = add_noise(x0, eps, t) ; x0_hat = x_t + du(x_t, t)
    The truncated ladder [10, 0] is an INFERENCE object and stays one.
    Returns (fan, u0_hat, conf, telemetry) with the SAME contract as `_sample`.
    """
    cfg = dec.cfg
    b = bank.shape[0]
    dev, dtype = bank.device, bank.dtype
    norm = bank.new_tensor(tuple(cfg.control_norm))
    x0_n = dec.anchor_control_seq(b, dtype) / norm
    v = (bank.new_full((b,), dec.anchor_ref_speed)
         if v_ms is None else v_ms.reshape(-1).to(torch.float32))
    dec.sched.to(dev)
    t = rs.draw_train_timesteps(b, int(cfg.sampler_train_t_max), dev,
                                generator=generator)
    x_n, _eps = rs.train_noise(dec.sched, x0_n, t, generator=generator)
    x_path = dec._state_to_path(x_n * norm, v, False)
    conf, du = dec._decode_ctrl(kv, cond, x_path, t.to(torch.float32),
                                None, None)
    x0_hat_n = x_n + du
    u0_hat = x0_hat_n * norm
    fan = dec._state_to_path(u0_hat, v, False)
    tele = {"sampler": "ddim", "sampler_space": cfg.sampler_space,
            "sampler_train_t_max": int(cfg.sampler_train_t_max),
            "sampler_train_t_mean": float(t.to(torch.float64).mean()),
            "sampler_train_t_draw": "U[0, t_max)"}
    return fan, u0_hat, conf, tele


def realised_t(dec, n_batches, batch, fixed, generator=None):
    """Run the sampler `n_batches` times and return every realised timestep."""
    kv = torch.randn(batch, 6, dec.cfg.d)
    cond = torch.randn(batch, dec.cfg.d)
    v = torch.full((batch,), 10.0)
    bank = dec.roll_bank(v, None, batch, torch.float32)
    seen = spy(dec)
    out = []
    for _ in range(n_batches):
        seen.clear()
        if fixed:
            dec._sample(kv, cond, bank, v, 0, None, None)
        else:
            sample_train_p13(dec, kv, cond, bank, v, generator=generator)
        out.append(np.concatenate([np.atleast_1d(s) for s in seen]))
    return np.concatenate(out)


def main():
    torch.manual_seed(0)
    print("=" * 78)
    print("P13 VALIDATION -- DD's t ~ U[0, t_max) TRAINING DRAW")
    print("=" * 78)

    # ---- B2 + controls: the schedule ------------------------------------- #
    sch = rs.DDIMSchedule()
    s0 = float(sch.sqrt_one_minus_abar(0))
    s8 = float(sch.sqrt_one_minus_abar(8))
    print("")
    print("[CONTROL sigma(0)] %.6e  -- MUST be > 0 (steps_offset=1 means "
          "abar_0 < 1, so t=0 is NOT zero noise): %s"
          % (s0, "PASS" if s0 > 0 else "FAIL"))
    print("[CONTROL sigma(8)] %.6f  -- MUST be the published 0.0316: %s"
          % (s8, "PASS" if abs(s8 - 0.0316) < 5e-5 else "FAIL"))
    ok_sched = rs.assert_matches_diffusers(sch)
    print("[CONTROL schedule] pinned against diffusers: %s"
          % ("PASS" if ok_sched else "SKIPPED (diffusers not importable)"))
    b2 = (s0 > 0) and abs(s8 - 0.0316) < 5e-5
    print("BAR P13-B2 ==> %s" % ("PASS" if b2 else "FAIL"))

    # ---- the draw itself, as a distribution ------------------------------ #
    g = torch.Generator().manual_seed(7)
    d50 = rs.draw_train_timesteps(20000, 50, torch.device("cpu"), generator=g)
    d1 = rs.draw_train_timesteps(20000, 1, torch.device("cpu"), generator=g)
    print("")
    print("[CONTROL draw t_max=50] mean %.3f (expected 24.5)  min %d  max %d "
          " n=%d  -> %s"
          % (float(d50.double().mean()), int(d50.min()), int(d50.max()),
             d50.numel(),
             "PASS" if abs(float(d50.double().mean()) - 24.5) < 0.5
             and int(d50.max()) == 49 and int(d50.min()) == 0 else "FAIL"))
    print("[CONTROL draw t_max=1 ] mean %.3f (expected 0.0)   max %d  "
          "variance %.6f (expected exactly 0)  -> %s"
          % (float(d1.double().mean()), int(d1.max()),
             float(d1.double().var()),
             "PASS" if float(d1.double().var()) == 0.0 else "FAIL"))

    # ---- B1: the LIVE mutation proof on the real decoder ------------------ #
    print("")
    print("-" * 78)
    print("BAR P13-B1 -- DOES THE FLAG HAVE A CONSUMER? (live spy on "
          "AnchoredDiffusionDecoder._sample)")
    print("-" * 78)
    rows = {}
    for label, t_max, fixed in (("SHIPPED  t_max=50", 50, True),
                                ("SHIPPED  t_max=1 ", 1, True),
                                ("REPAIRED t_max=50", 50, False),
                                ("REPAIRED t_max=1 ", 1, False)):
        torch.manual_seed(0)
        dec = build(t_max).train()
        gg = torch.Generator().manual_seed(11)
        t = realised_t(dec, 40, 8, fixed, generator=gg)
        rows[label] = t
        print("%-18s realised t: unique %-22s mean %7.3f  var %9.4f  n=%d"
              % (label, str(sorted(set(t.tolist()))[:6])[:22],
                 float(t.mean()), float(t.var()), t.size))

    same_shipped = np.array_equal(np.sort(rows["SHIPPED  t_max=50"]),
                                  np.sort(rows["SHIPPED  t_max=1 "]))
    diff_repaired = not np.array_equal(np.sort(rows["REPAIRED t_max=50"]),
                                       np.sort(rows["REPAIRED t_max=1 "]))
    ship_const = float(rows["SHIPPED  t_max=50"].var())
    print("")
    print("SHIPPED: changing t_max from 50 to 1 leaves the realised timesteps "
          "IDENTICAL: %s" % same_shipped)
    print("         -> the flag has NO CONSUMER. Realised t are the INFERENCE "
          "ladder %s, in TRAINING mode."
          % sorted(set(rows["SHIPPED  t_max=50"].tolist())))
    print("REPAIRED: changing t_max from 50 to 1 CHANGES the realised "
          "timesteps: %s" % diff_repaired)
    b1 = same_shipped and diff_repaired
    print("")
    print("BAR: PASS if the shipped code is PROVEN inert AND the repair is "
          "PROVEN to consume the flag.")
    print("MEASURED ==> %s" % ("PASS" if b1 else "FAIL"))

    # ---- R: the deliberate-regression arm --------------------------------- #
    print("")
    print("-" * 78)
    print("ARM P13-R -- DELIBERATE REGRESSION (t_max = 1, the no-noise draw)")
    print("-" * 78)
    r50, r1 = rows["REPAIRED t_max=50"], rows["REPAIRED t_max=1 "]
    print("t_max=50 realised t: mean %.3f var %.3f" % (r50.mean(), r50.var()))
    print("t_max=1  realised t: mean %.3f var %.3f (MUST be exactly 0)"
          % (r1.mean(), r1.var()))
    valid = diff_repaired and float(r1.var()) == 0.0
    print("GATE VALIDITY ==> %s"
          % ("VALID -- a knowingly-broken (zero-noise) draw IS distinguishable"
             if valid else
             "VOID -- the instrument cannot separate a no-noise draw from DD's"))

    print("")
    print("=" * 78)
    print("P13 SUMMARY: B1 %s | B2 %s | regression-arm %s"
          % ("PASS" if b1 else "FAIL", "PASS" if b2 else "FAIL",
             "VALID" if valid else "VOID"))
    print("⛔ B3 (BENEFIT) IS NOT ANSWERED HERE -- it needs a tiny-rig arm "
          "with a replicate. P13 is MECHANISM-VALIDATED, "
          "UNVALIDATED-FOR-BENEFIT.".replace("⛔", "[!]"))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as fh:
            json.dump({"b1_pass": bool(b1), "b2_pass": bool(b2),
                       "regression_valid": bool(valid),
                       "shipped_realised_t_unique":
                           sorted(set(rows["SHIPPED  t_max=50"].tolist())),
                       "shipped_t_invariant_to_flag": bool(same_shipped),
                       "repaired_t_responds_to_flag": bool(diff_repaired),
                       "sigma_0": s0, "sigma_8": s8,
                       "draw50_mean": float(d50.double().mean()),
                       "draw1_var": float(d1.double().var()),
                       "shipped_t_var": ship_const}, fh, indent=2)
            print("[banked] %s" % sys.argv[1])


if __name__ == "__main__":
    main()
