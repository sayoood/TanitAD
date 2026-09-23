"""refcv6 §3 F1..F9 — is each flag LIVE, or only declared?

⛔ The programme's own standard: *"this programme has repeatedly shipped flags
that parse and reach nothing."* So NO flag is accepted on the strength of an
`if` statement. Each arm below is one of:

  * **effect**   — flag on vs off changes a measured tensor / telemetry value;
  * **mutation** — the module the flag BUILDS is perturbed and the output must
                   move (proves the module is IN the graph, not merely built);
  * **refusal**  — the flag's own guard is made to fire.

Run:
  PYTHONPATH=…/stack python diag_f1_f9_liveness.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from tanitad.models import refcv6_diffusion as rv6
from tanitad.refs import refc

B, N, FEAT, DMEAS, DCTX, D = 2, 6, 16, 8, 4, 32
HORIZONS = (5, 10, 15, 20)
S = len(HORIZONS)


def build(flags=None, *, sampler="ddim", space="control", seed=0, layers=2,
          agents=False, **cfgkw):
    torch.manual_seed(seed)
    cfg = refc.DecoderConfig(d=D, n_heads=4, layers=layers, ff_mult=2,
                             sampler=sampler, sampler_space=space,
                             refcv6=flags, cross_agent=agents, **cfgkw)
    dec = refc.AnchoredDiffusionDecoder(
        feat_dim=FEAT, n_steps=S, d_meas=DMEAS, d_ctx=DCTX, tac_latent_dim=4,
        anchors=torch.randn(N, S, 2), cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=HORIZONS, v0_conditioned=True,
        control_units="alat")
    with torch.no_grad():
        dec.anchor_controls.copy_(torch.stack(
            [torch.linspace(-2.0, 2.0, N), torch.linspace(-1.5, 1.5, N)], -1))
    return dec


def inputs(seed=11):
    torch.manual_seed(seed)
    return (torch.randn(B, FEAT, 4, 4), torch.randn(B, DMEAS),
            torch.full((B,), 8.0))


def fwd(dec, *, steps=2, train=False, seed=99):
    fmap, m, v = inputs()
    dec.train(train)
    torch.manual_seed(seed)
    with torch.no_grad():
        return dec(fmap, m, steps=steps, v_ms=v)


def tele(out):
    return out.get("sel_tele", {}) or {}


# --------------------------------------------------------------------------
def f1() -> dict:
    """random-t training: ONE decoder call at t ~ U[0, t_max)."""
    r = {}
    on = build(rv6.DiffusionFlags(f1_random_t=True))
    off = build(rv6.DiffusionFlags(f2_dd_step=True))     # any_on, f1 off
    o_tr = fwd(on, train=True)
    o_ev = fwd(on, train=False)
    f_tr = fwd(off, train=True)
    r["ON_train_pairs"] = tele(o_tr).get("refcv6_pairs")
    r["ON_train_t_sample"] = tele(o_tr).get("refcv6_train_t")
    r["ON_eval_pairs"] = tele(o_ev).get("refcv6_pairs")
    r["OFF_train_pairs"] = tele(f_tr).get("refcv6_pairs")
    # a 512-window draw, to show the support is really U[0, 50)
    ts = []
    big = build(rv6.DiffusionFlags(f1_random_t=True))
    big.train(True)
    fmap, m, v = inputs()
    for s in range(24):
        torch.manual_seed(s)
        with torch.no_grad():
            o = big(fmap, m, steps=2, v_ms=v)
        ts += list(tele(o).get("refcv6_train_t") or [])
    r["draw_n"] = len(ts)
    r["draw_min"], r["draw_max"] = (min(ts), max(ts)) if ts else (None, None)
    r["draw_distinct"] = len(set(ts))
    # MUTATION: t_max = 1 must collapse the support to {0}
    m1 = build(rv6.DiffusionFlags(f1_random_t=True, f1_t_max=1))
    m1.train(True)
    ts1 = []
    for s in range(8):
        torch.manual_seed(s)
        with torch.no_grad():
            o = m1(fmap, m, steps=2, v_ms=v)
        ts1 += list(tele(o).get("refcv6_train_t") or [])
    r["mutation_t_max_1_support"] = sorted(set(ts1))
    r["LIVE"] = bool(r["ON_train_pairs"] == [[None, None]]
                     and r["OFF_train_pairs"] != [[None, None]]
                     and (r["draw_max"] or 0) > 10
                     and r["mutation_t_max_1_support"] == [0])
    return r


def f2() -> dict:
    """t -> t-1 step semantics."""
    r = {}
    on = build(rv6.DiffusionFlags(f2_dd_step=True))
    off = build(rv6.DiffusionFlags(f1_random_t=True))   # any_on, f2 off
    r["ON_pairs"] = tele(fwd(on)).get("refcv6_pairs")
    r["OFF_pairs"] = tele(fwd(off)).get("refcv6_pairs")
    sched = on.sched
    r["residual_retained_10_to_9"] = round(
        rv6.residual_retained(sched, 10, 9), 4)
    r["residual_retained_10_to_0"] = round(
        rv6.residual_retained(sched, 10, 0), 4)
    # do the two ladders actually emit different trajectories?
    a = fwd(on, seed=5)["traj"]
    b = fwd(off, seed=5)["traj"]
    r["traj_max_abs_delta_on_vs_off"] = float((a - b).abs().max())
    r["LIVE"] = bool(r["ON_pairs"] == [[10, 9], [0, -1]]
                     and r["OFF_pairs"] != r["ON_pairs"]
                     and r["traj_max_abs_delta_on_vs_off"] > 0)
    return r


def f3() -> dict:
    """per-layer heads + per-layer export + DETACH between layers."""
    r = {}
    on = build(rv6.DiffusionFlags(f3_per_layer=True))
    off = build(rv6.DiffusionFlags(f2_dd_step=True))
    o = fwd(on)
    r["cascade_built"] = on.cascade is not None
    r["layer_u0_hat_len"] = (len(o["layer_u0_hat"])
                             if "layer_u0_hat" in o else 0)
    r["layer_logits_len"] = (len(o["layer_logits"])
                             if "layer_logits" in o else 0)
    r["OFF_has_layer_u0_hat"] = "layer_u0_hat" in fwd(off)
    r["tele_cascade_stages"] = tele(o).get("refcv6_cascade_stages")
    # MUTATION: perturb stage 0's control head; the emitted trajectory must move
    base = fwd(on, seed=3)["traj"].clone()
    with torch.no_grad():
        on.cascade.control_heads[0].bias.add_(1.0)
    moved0 = float((fwd(on, seed=3)["traj"] - base).abs().max())
    with torch.no_grad():
        on.cascade.control_heads[0].bias.sub_(1.0)
        on.cascade.control_heads[-1].bias.add_(1.0)
    movedL = float((fwd(on, seed=3)["traj"] - base).abs().max())
    with torch.no_grad():
        on.cascade.control_heads[-1].bias.sub_(1.0)
    r["mutation_stage0_head_moves_traj"] = moved0
    r["mutation_lastStage_head_moves_traj"] = movedL
    # DETACH: grad from the LAST stage must not reach the FIRST layer's params
    on2 = build(rv6.DiffusionFlags(f3_per_layer=True))
    fmap, m, v = inputs()
    on2.train(False)
    out = on2(fmap, m, steps=2, v_ms=v)
    out["layer_u0_hat"][-1].sum().backward()
    g_first_attn = on2.layers[0].cross.out_proj.weight.grad
    g_last_attn = on2.layers[-1].cross.out_proj.weight.grad
    r["grad_layer0_attn_from_LAST_stage"] = (
        None if g_first_attn is None else float(g_first_attn.abs().sum()))
    r["grad_layerL_attn_from_LAST_stage"] = (
        None if g_last_attn is None else float(g_last_attn.abs().sum()))
    r["LIVE"] = bool(r["cascade_built"] and r["layer_u0_hat_len"] > 0
                     and not r["OFF_has_layer_u0_hat"]
                     and moved0 > 0 and movedL > 0)
    return r


def f4() -> dict:
    """per-layer AdaLN timestep modulation."""
    r = {}
    on = build(rv6.DiffusionFlags(f4_adaln=True))
    r["adaln_built"] = on.adaln is not None
    r["n_adaln"] = 0 if on.adaln is None else len(on.adaln)
    r["zero_init_default"] = bool(rv6.DiffusionFlags().f4_zero_init)
    base = fwd(on, seed=3)["traj"].clone()
    with torch.no_grad():
        on.adaln[0].scale_shift_mlp[-1].bias.add_(0.5)
    r["mutation_adaln0_moves_traj"] = float(
        (fwd(on, seed=3)["traj"] - base).abs().max())
    with torch.no_grad():
        on.adaln[0].scale_shift_mlp[-1].bias.sub_(0.5)
        on.adaln[-1].scale_shift_mlp[-1].bias.add_(0.5)
    r["mutation_adalnLast_moves_traj"] = float(
        (fwd(on, seed=3)["traj"] - base).abs().max())
    with torch.no_grad():
        on.adaln[-1].scale_shift_mlp[-1].bias.sub_(0.5)
    # order check: Mish FIRST (DD `:235-238`)
    mod = on.adaln[0].scale_shift_mlp
    r["module_order"] = [type(x).__name__ for x in mod]
    # zero-init variant must be an exact identity at construction
    z = build(rv6.DiffusionFlags(f4_adaln=True, f4_zero_init=True), seed=0)
    nz = build(None, seed=0)
    r["zeroinit_traj_equals_noflag"] = bool(
        torch.equal(fwd(z, seed=3)["traj"], fwd(nz, seed=3)["traj"]))
    r["LIVE"] = bool(r["adaln_built"]
                     and r["mutation_adaln0_moves_traj"] > 0
                     and r["module_order"][0] == "Mish")
    return r


def f5() -> dict:
    """emitting pass's own confidence + focal loss."""
    r = {}
    on = build(rv6.DiffusionFlags(f5_emitting_conf=True))
    off = build(rv6.DiffusionFlags(f2_dd_step=True))
    o_on, o_off = fwd(on, seed=3), fwd(off, seed=3)
    r["sel_score_differs"] = float(
        (o_on["sel_score"] - o_off["sel_score"]).abs().max())
    # MUTATION: move the SAMPLER's conf head -> ranking must move with F5 on
    base_on = fwd(on, seed=3)["sel_score"].clone()
    base_off = fwd(off, seed=3)["sel_score"].clone()
    with torch.no_grad():
        on.conf_head.bias.add_(0.0)       # no-op; conf_head is shared
        on.control_head.bias.add_(0.7)    # moves the SAMPLE the conf sees
    r["mutation_control_head_moves_F5_score"] = float(
        (fwd(on, seed=3)["sel_score"] - base_on).abs().max())
    with torch.no_grad():
        on.control_head.bias.sub_(0.7)
        off.control_head.bias.add_(0.7)
    r["mutation_control_head_moves_OFF_score"] = float(
        (fwd(off, seed=3)["sel_score"] - base_off).abs().max())
    with torch.no_grad():
        off.control_head.bias.sub_(0.7)
    # focal: a numeric identity against a hand-written reference
    torch.manual_seed(0)
    logits = torch.randn(4, 7)
    tgt = torch.tensor([0, 3, 6, 1])
    got = float(rv6.focal_cls_loss(logits, tgt, 2.0, 0.25))
    oh = torch.zeros_like(logits).scatter_(1, tgt[:, None], 1.0)
    p = logits.sigmoid()
    pt = (1 - p) * oh + p * (1 - oh)
    w = (0.25 * oh + 0.75 * (1 - oh)) * pt.pow(2.0)
    ref = float((F.binary_cross_entropy_with_logits(
        logits, oh, reduction="none") * w).mean())
    r["focal_value"] = round(got, 8)
    r["focal_reference"] = round(ref, 8)
    r["focal_matches_reference"] = bool(abs(got - ref) < 1e-9)
    r["focal_vs_cross_entropy"] = round(
        float(F.cross_entropy(logits, tgt)), 6)
    r["LIVE"] = bool(r["sel_score_differs"] > 0
                     and r["focal_matches_reference"])
    return r


def f6() -> dict:
    """`--w-u0 0` — asserted at the TRAINER, so read the trainer."""
    import argparse as _ap
    sys.path.insert(0, str(Path("D:/Projects/TanitAD/stack/scripts")))
    import refc_v3_train as T
    r = {}
    ns = _ap.Namespace(f6_w_u0_zero=True)
    fl = T.refcv6_flags_from_args(ns)
    r["flags_from_args_f6"] = None if fl is None else bool(fl.f6_w_u0_zero)
    ns_off = _ap.Namespace()
    r["flags_from_args_default_is_None"] = T.refcv6_flags_from_args(
        ns_off) is None
    # the pin: does `_pin_trainer_cfg` force w_u0 = 0 and the ack?
    import ast
    src = Path("D:/Projects/TanitAD/stack/scripts/refc_v3_train.py").read_text(
        encoding="utf-8")
    seg = "\n".join(src.splitlines()[385:393])
    r["pin_source_lines_386_393"] = seg
    r["LIVE"] = bool(r["flags_from_args_f6"]
                     and "args.w_u0 = 0.0" in seg
                     and "ack_ddim_no_u0" in seg)
    return r


def f7() -> dict:
    """G noise samples per anchor."""
    r = {}
    try:
        bad = build(rv6.DiffusionFlags(f7_samples_per_anchor=2))
        fwd(bad)
        r["refuses_without_ack"] = False
    except NotImplementedError as e:
        r["refuses_without_ack"] = True
        r["refusal_head"] = str(e)[:90]
    on = build(rv6.DiffusionFlags(f7_samples_per_anchor=2,
                                  f7_ack_eval_join=True))
    o = fwd(on)
    r["fan_shape"] = list(o["anchor_traj"].shape)
    r["logits_shape"] = list(o["anchor_logits"].shape)
    r["has_sel_anchor_id"] = "sel_anchor_id" in o
    r["sel_anchor_id"] = (None if "sel_anchor_id" not in o
                          else o["sel_anchor_id"].tolist())
    r["sel_idx"] = o["sel_idx"].tolist()
    # the group-major layout identity: candidate g*N + a is anchor a
    bank = o["anchor_bank"]
    r["bank_shape"] = list(bank.shape)
    r["bank_group_major"] = bool(torch.equal(bank[:, :N], bank[:, N:2 * N]))
    r["tile_layout_is_repeat"] = bool(torch.equal(
        rv6.tile_anchor_prior(torch.arange(N).float()[None], 2),
        torch.cat([torch.arange(N).float()[None]] * 2, 1)))
    r["LIVE"] = bool(r["fan_shape"][1] == 2 * N and r["has_sel_anchor_id"]
                     and r["refuses_without_ack"])
    return r


def f8() -> dict:
    """flat waypoint-space noise (DD's `norm_odo` box)."""
    r = {}
    try:
        build(rv6.DiffusionFlags(f8_flat_waypoint_noise=True), space="control")
        r["refuses_on_control_space"] = False
    except ValueError as e:
        r["refuses_on_control_space"] = True
        r["refusal_head"] = str(e)[:80]
    on = build(rv6.DiffusionFlags(f8_flat_waypoint_noise=True), space="metre")
    off = build(None, space="metre")
    o_on, o_off = fwd(on, seed=4), fwd(off, seed=4)
    r["traj_differs"] = float((o_on["traj"] - o_off["traj"]).abs().max())
    # the DD box constants, against the released source
    r["DD_X_OFF/SPAN"] = [rv6.DD_X_OFF, rv6.DD_X_SPAN]
    r["DD_Y_OFF/SPAN"] = [rv6.DD_Y_OFF, rv6.DD_Y_SPAN]
    # round-trip identity
    xy = torch.randn(3, 5, 2) * 10
    r["norm_denorm_roundtrip_max_err"] = float(
        (rv6.dd_denorm_waypoints(rv6.dd_norm_waypoints(xy)) - xy).abs().max())
    # FLATNESS: sigma at the truncation point, per waypoint, in metres
    s = float(on.sched.sqrt_one_minus_abar(int(on.cfg.sampler_infer_t)))
    r["sqrt_one_minus_abar_at_infer_t"] = round(s, 6)
    r["sigma_x_m"] = round(s * rv6.DD_X_SPAN / 2, 4)
    r["sigma_y_m"] = round(s * rv6.DD_Y_SPAN / 2, 4)
    # ⛔ DD clamps INSIDE the loop (`:519`); do we?
    src = Path("D:/Projects/TanitAD/stack/tanitad/refs/refc.py").read_text(
        encoding="utf-8").splitlines()
    clamp_lines = [i + 1 for i, ln in enumerate(src)
                   if "clamp(-1.0, 1.0)" in ln]
    loop_line = next((i + 1 for i, ln in enumerate(src)
                      if "for i, (t, t_prev) in enumerate(pairs)" in ln), None)
    r["clamp_lines"] = clamp_lines
    r["sampler_loop_line"] = loop_line
    r["clamp_is_inside_loop"] = bool(
        any(c > (loop_line or 10 ** 9) for c in clamp_lines))
    r["LIVE"] = bool(r["refuses_on_control_space"] and r["traj_differs"] > 0)
    return r


def f9() -> dict:
    """assert-only: the 117-anchor v0-conditioned vocabulary."""
    r = {}
    try:
        rv6.assert_f9_vocabulary(117, True, 117)
        r["accepts_117_v0"] = True
    except ValueError:
        r["accepts_117_v0"] = False
    for name, args_ in (("wrong_n", (116, True, 117)),
                        ("not_v0", (117, False, 117))):
        try:
            rv6.assert_f9_vocabulary(*args_)
            r[f"rejects_{name}"] = False
        except ValueError as e:
            r[f"rejects_{name}"] = True
            r[f"{name}_msg"] = str(e)[:70]
    # is the assertion REACHED from a forward?
    on = build(rv6.DiffusionFlags(f9_assert_vocab=True))   # N == 6, not 117
    try:
        fwd(on)
        r["forward_enforces"] = False
    except ValueError as e:
        r["forward_enforces"] = True
        r["forward_msg"] = str(e)[:80]
    r["LIVE"] = bool(r["rejects_wrong_n"] and r["rejects_not_v0"]
                     and r["forward_enforces"])
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    for name, fn in (("F1", f1), ("F2", f2), ("F3", f3), ("F4", f4),
                     ("F5", f5), ("F6", f6), ("F7", f7), ("F8", f8),
                     ("F9", f9)):
        try:
            out[name] = fn()
        except Exception as e:                      # noqa: BLE001
            out[name] = {"ERROR": f"{type(e).__name__}: {e}", "LIVE": False}
    out["VERDICT"] = {k: v.get("LIVE") for k, v in out.items()
                      if isinstance(v, dict)}
    txt = json.dumps(out, indent=2, default=str)
    print(txt)
    path = a.json or str(Path(__file__).resolve().parents[1]
                         / "raw" / "f1_f9_liveness.json")
    Path(path).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
