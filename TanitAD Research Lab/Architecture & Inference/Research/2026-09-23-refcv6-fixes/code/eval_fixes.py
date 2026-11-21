#!/usr/bin/env python
"""EVALUATE the five 2026-09-23 refcv6 fixes: what did each one CHANGE?

⭐ The BEFORE arm is materialised from ``git show HEAD:<path>`` — the fixes are
STAGED, NEVER COMMITTED, so ``HEAD`` is exactly the pre-fix source. Both
``refc.py`` AND ``refcv6_diffusion.py`` are materialised together and loaded as
a COHERENT PAIR (the pre-fix ``refc`` is exec'd with the pre-fix
``tanitad.models.refcv6_diffusion`` installed in ``sys.modules``), because
loading one against the other's module would measure a chimera.

⛔ Every number here is MEASURED on this box, CPU, at a PINNED inference seed.
The seed is not decoration: ``_sample`` draws a fresh ``eps`` every forward
(`refc.py:2538`), so an unseeded comparison measures INFERENCE noise — the
review's own first run read `BIT_IDENTICAL: false` for a correct fix that way.

⚠️ `control_head` is ZERO-INIT, so at construction the emitted trajectory is
independent of every decoder weight. EVERY arm below is reported in BOTH
states: gate CLOSED (as built) and gate OPENED (emitting head un-zeroed).

Usage:  python eval_fixes.py --json ../raw/eval_fixes.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

import torch

REPO = pathlib.Path(r"D:/Projects/TanitAD")
REL_REFC = "stack/tanitad/refs/refc.py"
REL_RV6 = "stack/tanitad/models/refcv6_diffusion.py"


def git_blob(rev: str, rel: str) -> bytes:
    p = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=str(REPO),
                       capture_output=True)
    out = p.stdout or b""
    if p.returncode != 0 or not out:
        err = (p.stderr or b"").decode("utf-8", errors="replace").strip()
        raise SystemExit(f"INVALID: could not read {rev}:{rel} — {err}")
    return out


def load_pre_fix(tmpdir: pathlib.Path):
    """-> (pre_refc_module, pre_rv6_module). The pair, not one of them."""
    tmpdir.mkdir(parents=True, exist_ok=True)
    rv6_p = tmpdir / "_prefix_rv6.py"
    refc_p = tmpdir / "_prefix_refc.py"
    rv6_p.write_bytes(git_blob("HEAD", REL_RV6))
    refc_p.write_bytes(git_blob("HEAD", REL_REFC))

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    rv6_pre = _load("_prefix_rv6", rv6_p)
    # ⛔ `sys.modules` ALONE IS NOT ENOUGH, and the failure is silent-ish:
    # `refc.py:149` is `from tanitad.models import refcv6_diffusion as _rv6`,
    # which reads the ATTRIBUTE off the already-imported parent package. With
    # only `sys.modules` swapped the pre-fix `refc` binds the CURRENT flags
    # class and its own isinstance guard rejects the pre-fix one —
    # "DiffusionFlags ... got DiffusionFlags", MEASURED. Swap both.
    import tanitad.models as _pkg
    real_mod = sys.modules.get("tanitad.models.refcv6_diffusion")
    real_attr = getattr(_pkg, "refcv6_diffusion", None)
    sys.modules["tanitad.models.refcv6_diffusion"] = rv6_pre
    _pkg.refcv6_diffusion = rv6_pre
    try:
        refc_pre = _load("_prefix_refc", refc_p)
    finally:
        if real_mod is not None:
            sys.modules["tanitad.models.refcv6_diffusion"] = real_mod
        if real_attr is not None:
            _pkg.refcv6_diffusion = real_attr
    # POSITIVE assertion that the pair is coherent and NOT a chimera.
    assert refc_pre._rv6 is rv6_pre, "INVALID: the pre-fix pair is a chimera"
    assert "f5_refuse_blind_rank" not in {
        f.name for f in __import__("dataclasses").fields(
            rv6_pre.DiffusionFlags)}, (
        "INVALID: the 'before' rv6 already carries the fix")
    return refc_pre, rv6_pre


# --------------------------------------------------------------------------- #
def ctrl_ladder(n):
    return torch.stack([torch.linspace(-2.0, 2.0, n),
                        torch.linspace(-1.5, 1.5, n)], dim=-1)


def build(mod, rv6mod, flags_kw=None, n=5, horizons=(5, 10, 15, 20),
          seed=7, bev_coupling=False, d_bev=6, **cfgkw):
    flags = None if flags_kw is None else rv6mod.DiffusionFlags(**flags_kw)
    torch.manual_seed(seed)
    cfg = mod.DecoderConfig(d=32, n_heads=4, layers=2, ff_mult=2,
                            sampler="ddim", refcv6=flags, **cfgkw)
    torch.manual_seed(seed)
    d = mod.AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=len(horizons), d_meas=8, d_ctx=4,
        tac_latent_dim=4, anchors=torch.zeros(n, len(horizons), 2), cfg=cfg,
        hierarchy=False, graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=horizons, v0_conditioned=True,
        control_units="alat")
    with torch.no_grad():
        d.anchor_controls.copy_(ctrl_ladder(n))
    if bev_coupling:
        import tanitad.models.refc_bev_coupling as bevc
        d.attach_bev_coupling(
            bevc.BEVCouplingConfig(enable=True, d_model=32, d_bev=d_bev,
                                   n_points=len(horizons)), d_bev=d_bev)
    return d.eval()


def inputs(b=64, d_bev=6):
    torch.manual_seed(11)
    return dict(fmap=torch.randn(b, 16, 3, 5), m=torch.randn(b, 8),
                v_ms=torch.rand(b) * 20 + 2,
                bev=torch.randn(b, d_bev, 120, 64))


def fwd(dec, inp, seed=1234, steps=2, bev=None):
    with torch.no_grad():
        torch.manual_seed(seed)
        return dec(inp["fmap"], inp["m"], steps=steps, v_ms=inp["v_ms"],
                   **({} if bev is None else {"bev": bev}))


def cmp_outputs(a, b):
    keys = sorted(set(a) | set(b))
    diffs, maxdiff = {}, 0.0
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if torch.is_tensor(va) and torch.is_tensor(vb):
            if va.shape != vb.shape:
                diffs[k] = "SHAPE"
                continue
            d = float((va.float() - vb.float()).abs().max())
            if d != 0.0:
                diffs[k] = d
                maxdiff = max(maxdiff, d)
    return {"keys_only_in_a": sorted(set(a) - set(b)),
            "keys_only_in_b": sorted(set(b) - set(a)),
            "tensor_keys_that_differ": diffs,
            "max_abs_tensor_diff": maxdiff,
            "BIT_IDENTICAL": (not diffs and set(a) == set(b))}


VALIDATED_ARM = dict(f1_random_t=True, f2_dd_step=True, f3_per_layer=True,
                     f4_adaln=True, f5_focal=True, f6_w_u0_zero=True,
                     f9_assert_vocab=True, f9_n_anchors=5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import tanitad.models.refcv6_diffusion as rv6_post
    from tanitad.refs import refc as refc_post
    tmp = pathlib.Path(a.json).resolve().parent / "_prefix_tmp"
    refc_pre, rv6_pre = load_pre_fix(tmp)

    R: dict = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "before": "git HEAD blob (the fixes are staged, not committed)",
               "after": "worktree", "device": "cpu", "windows": 64,
               "inference_seed": 1234}
    inp = inputs(64)

    # ---------------------------------------------------------------- BINDING
    # Does a TRAINED ARM compute anything different? Three configurations.
    binding = {}
    for name, kw in (("all_flags_off", None),
                     ("validated_arm_2026-09-17_f1f2f3f4_f5focal_f6_f9",
                      VALIDATED_ARM),
                     ("F5_emitting_conf_on",
                      dict(VALIDATED_ARM, f5_emitting_conf=True))):
        pre = build(refc_pre, rv6_pre, kw)
        post = build(refc_post, rv6_post, kw)
        sd_a, sd_b = pre.state_dict(), post.state_dict()
        same_keys = sd_a.keys() == sd_b.keys()
        w_bad = [k for k in sd_a if not torch.equal(sd_a[k], sd_b[k])] \
            if same_keys else ["<keys differ>"]
        oa, ob = fwd(pre, inp), fwd(post, inp)
        c = cmp_outputs(oa, ob)
        c["state_dict_keys_identical"] = bool(same_keys)
        c["weight_mismatches"] = w_bad
        c["sel_tele_identical"] = (oa["sel_tele"] == ob["sel_tele"])
        c["sel_tele_before"] = {k: v for k, v in oa["sel_tele"].items()
                                if not torch.is_tensor(v)}
        c["sel_tele_after"] = {k: v for k, v in ob["sel_tele"].items()
                               if not torch.is_tensor(v)}
        binding[name] = c
    R["BINDING_existing_arm_behaviour"] = binding

    # ------------------------------------------------------------- FINDING 1
    f1: dict = {}
    fires = {"before": {}, "after": {}}
    for tag, mod, rv6m in (("before", refc_pre, rv6_pre),
                           ("after", refc_post, rv6_post)):
        dec = build(mod, rv6m, None, bev_coupling=True)
        for steps in (2, 0):
            n = {"k": 0}
            hs = [ly.bev_wp.register_forward_pre_hook(
                lambda m, args, n=n: n.__setitem__("k", n["k"] + 1))
                for ly in dec.layers]
            fwd(dec, inp, steps=steps, bev=inp["bev"])
            for h in hs:
                h.remove()
            fires[tag][f"steps_{steps}"] = n["k"]
    f1["bev_wp_forward_calls"] = fires
    f1["expected_after"] = "2 layers x (1 classifier + 2 denoise passes) = 6"
    # removability at the zero-init gate, and the gate opened
    dec = build(refc_post, rv6_post, None, bev_coupling=True)
    off = fwd(dec, inp, bev=None)
    on = fwd(dec, inp, bev=inp["bev"])
    f1["zero_init_gate_traj_max_abs_delta_m"] = float(
        (off["traj"] - on["traj"]).abs().max())
    f1["zero_init_gate_u0hat_max_abs_delta"] = float(
        (off["u0_hat"] - on["u0_hat"]).abs().max())
    with torch.no_grad():
        torch.manual_seed(5)
        dec.control_head.weight.normal_(0.0, 0.5)
    base = fwd(dec, inp, bev=inp["bev"])
    with torch.no_grad():
        for ly in dec.layers:
            ly.bev_wp.gate.fill_(1.0)
    opened = fwd(dec, inp, bev=inp["bev"])
    f1["gate_opened_traj_max_abs_delta_m"] = float(
        (base["traj"] - opened["traj"]).abs().max())
    f1["gate_opened_u0hat_max_abs_delta"] = float(
        (base["u0_hat"] - opened["u0_hat"]).abs().max())
    f1["gate_opened_sel_idx_changed_windows"] = int(
        (base["sel_idx"] != opened["sel_idx"]).sum())
    f1["bev_coupling_params"] = int(dec.bev_coupling_params())

    # ---- BREAK B, at the MODEL level (the caller that never supplied it) --- #
    def _smoke(mod, couple=True, d_bev=6):
        cfg = mod.refc_smoke_config()
        m = mod.RefCModel(cfg).eval()
        if couple:
            import tanitad.models.refc_bev_coupling as bevc
            m.decoder.attach_bev_coupling(
                bevc.BEVCouplingConfig(enable=True,
                                       d_model=int(m.decoder.cfg.d),
                                       d_bev=d_bev,
                                       n_points=m.decoder.n_steps),
                d_bev=d_bev)
        enc = m.encoder
        real = enc.forward

        def ff(x, _r=real):                    # a double for the TRUNK only:
            fm, po = _r(x)                     # `fmap_s16` needs `--trunk timm`
            return torch.zeros(x.shape[0], 5, 4, 8), fm, po
        enc.forward_features = ff
        return m, cfg

    bb: dict = {}
    dense = torch.randn(2, 6, 120, 64)
    for tag, mod in (("before", refc_pre), ("after", refc_post)):
        m, cfg = _smoke(mod)
        frames = torch.randn(2, cfg.window, cfg.encoder.in_channels, 64, 64)
        fires = {"k": 0}
        hs = [ly.bev_wp.register_forward_pre_hook(
            lambda _m, _a, f=fires: f.__setitem__("k", f["k"] + 1))
            for ly in m.decoder.layers]
        try:
            with torch.no_grad():
                torch.manual_seed(3)
                m(frames, steps=2,
                  bev_hook=lambda s16: {"bev_feats": dense})
            bb[f"hook_fed__{tag}"] = {"ran": True, "bev_wp_calls": fires["k"]}
        except ValueError as e:
            bb[f"hook_fed__{tag}"] = {"ran": False, "bev_wp_calls": fires["k"],
                                      "refused": str(e)[:80]}
        for h in hs:
            h.remove()
        m2, _ = _smoke(mod)
        try:
            with torch.no_grad():
                m2(frames, steps=2)
            bb[f"built_coupling_NO_map__{tag}"] = "RAN (0 fires, silently)"
        except ValueError as e:
            bb[f"built_coupling_NO_map__{tag}"] = f"REFUSED: {str(e)[:80]}"
        m3, _ = _smoke(mod, couple=False)      # CONTROL: the §4 tactical arm
        with torch.no_grad():
            torch.manual_seed(3)
            t_nohook = m3(frames, steps=2)["traj"]
            torch.manual_seed(3)
            t_hook = m3(frames, steps=2,
                        bev_hook=lambda s16: {"bev_feats": dense})["traj"]
        bb[f"CONTROL_no_coupling_traj_max_abs_delta__{tag}"] = float(
            (t_nohook - t_hook).abs().max())
    f1["BREAK_B_model_level"] = bb
    R["FINDING_1_coupling_reaches_the_decoder"] = f1

    # ------------------------------------------------------------- FINDING 2
    f2: dict = {}
    for tag, kw in (("DEFAULT_no_F5", None),
                    ("F5_emitting_conf", dict(f5_emitting_conf=True))):
        dec = build(refc_post, rv6_post, kw)
        with torch.no_grad():
            torch.manual_seed(5)
            dec.control_head.weight.normal_(0.0, 0.5)
        a0 = fwd(dec, inp, seed=4321)
        with torch.no_grad():                    # perturb ONLY the sampler
            torch.manual_seed(7)
            dec.control_head.weight.add_(
                torch.randn_like(dec.control_head.weight) * 0.5)
        a1 = fwd(dec, inp, seed=4321)
        f2[tag] = {
            "traj_max_abs_delta_m": float(
                (a0["traj"] - a1["traj"]).abs().max()),
            "sel_score_max_abs_delta": float(
                (a0["sel_score"] - a1["sel_score"]).abs().max()),
            "sel_idx_changed_windows": int(
                (a0["sel_idx"] != a1["sel_idx"]).sum()),
            "sampler_ranks_the_fan": a0["sel_tele"]["sampler_ranks_the_fan"]}
    # the telemetry corner table, before vs after
    tele = {}
    for f5 in (False, True):
        for refined in (False, True):
            row = {}
            for tag, mod, rv6m in (("before", refc_pre, rv6_pre),
                                   ("after", refc_post, rv6_post)):
                d = build(mod, rv6m, dict(f5_emitting_conf=f5))
                d.sel.refined = refined
                row[tag] = fwd(d, inp)["sel_tele"]["sampler_ranks_the_fan"]
            tele[f"f5={f5},sel.refined={refined}"] = row
    f2["sampler_ranks_the_fan_stamp"] = tele
    # the opt-in refusal
    try:
        d = build(refc_post, rv6_post, dict(f5_refuse_blind_rank=True))
        fwd(d, inp)
        f2["f5_refuse_blind_rank_on_a_blind_build"] = "DID NOT REFUSE"
    except ValueError as e:
        f2["f5_refuse_blind_rank_on_a_blind_build"] = f"REFUSED: {str(e)[:90]}"
    d = build(refc_post, rv6_post, dict(f5_refuse_blind_rank=True,
                                        f5_emitting_conf=True))
    fwd(d, inp)
    f2["f5_refuse_blind_rank_with_F5"] = "ran (as it must)"
    R["FINDING_2_ranked_surface"] = f2

    # ------------------------------------------------------------- FINDING 3
    f3: dict = {}
    dec = build(refc_post, rv6_post, None)
    with torch.no_grad():
        real = dec.roll_bank(inp["v_ms"][:4], None, 4, torch.float32)
        f3["bank_spread_real_controls_m"] = float(
            (real - real[:, :1]).abs().max())
        dec.anchor_controls.zero_()
        dead = dec.roll_bank(inp["v_ms"][:4], None, 4, torch.float32)
        f3["bank_spread_zero_controls_m"] = float(
            (dead - dead[:, :1]).abs().max())
    for tag, mod, rv6m in (("before", refc_pre, rv6_pre),
                           ("after", refc_post, rv6_post)):
        d = build(mod, rv6m, None)
        with torch.no_grad():
            d.anchor_controls.zero_()
        d._v0_controls_nonzero = False
        try:
            fwd(d, inp)
            f3[f"forward_on_a_degenerate_bank__{tag}"] = "RAN SILENTLY"
        except ValueError as e:
            f3[f"forward_on_a_degenerate_bank__{tag}"] = \
                f"REFUSED: {str(e)[:90]}"
        try:
            rv6m.assert_f9_vocabulary(5, True, 5,
                                      **({} if tag == "before"
                                         else {"anchor_controls":
                                               torch.zeros(5, 2)}))
            f3[f"assert_f9_vocabulary_on_zeros__{tag}"] = "PASSED"
        except ValueError as e:
            f3[f"assert_f9_vocabulary_on_zeros__{tag}"] = \
                f"RAISED: {str(e)[:70]}"
        except TypeError as e:
            f3[f"assert_f9_vocabulary_on_zeros__{tag}"] = \
                f"NO SUCH PARAMETER: {e}"
    R["FINDING_3_v0_guard_reads_the_tensor"] = f3

    # ------------------------------------------------------------- FINDING 4
    f4: dict = {}
    head = git_blob("HEAD", REL_REFC)
    base_blob = git_blob("cbadba5844a2db70a968172ab0b0bbe3a0140af0", REL_REFC)
    f4["refcv6_mentions"] = {
        "HEAD": head.count(b"refcv6"),
        "cbadba5_(8c7d215^)": base_blob.count(b"refcv6")}
    f4["CONTROL_class_AnchoredDiffusionDecoder_mentions"] = {
        "HEAD": head.count(b"class AnchoredDiffusionDecoder"),
        "cbadba5_(8c7d215^)": base_blob.count(b"class AnchoredDiffusionDecoder")}
    f4["baseline_blob_equals_HEAD_blob"] = (base_blob == head)
    R["FINDING_4_bitidentity_baseline"] = f4

    # ------------------------------------------------------------- FINDING 5
    f5r: dict = {}
    for tag, mod, rv6m in (("before", refc_pre, rv6_pre),
                           ("after", refc_post, rv6_post)):
        d = build(mod, rv6m, dict(f8_flat_waypoint_noise=True),
                  sampler_space="metre")
        with torch.no_grad():
            torch.manual_seed(5)
            d.control_head.weight.normal_(0.0, 0.5)
        ev, steps_out = [], []
        orig_d, orig_s, orig_c = (rv6m.dd_denorm_waypoints, d.sched.step,
                                  d._decode_ctrl)

        def dn(z, _o=orig_d):
            ev.append(("denorm", float(z.abs().max())))
            return _o(z)

        def st(*x, _o=orig_s, **k):
            o = _o(*x, **k)
            steps_out.append(float(o.abs().max()))
            return o

        def ct(*x, _o=orig_c, **k):
            ev.append(("decode", None))
            return _o(*x, **k)

        rv6m.dd_denorm_waypoints = dn
        d.sched.step = st
        d._decode_ctrl = ct
        try:
            fwd(d, inp)
        finally:
            rv6m.dd_denorm_waypoints = orig_d
            del d.sched.step
            del d._decode_ctrl
        ladder = [v for i, (k, v) in enumerate(ev)
                  if k == "denorm" and i + 1 < len(ev)
                  and ev[i + 1][0] == "decode"]
        f5r[tag] = {"per_ladder_pass_max_abs_x_n": ladder,
                    "max_sched_step_output_abs": max(steps_out)
                    if steps_out else None,
                    "outside_DD_box": [v for v in ladder if v > 1.0]}
    f5r["DD_source"] = ("transfuser_model_v2.py:519 clamps at the top of every "
                        "iteration; :474 clamps once in training")
    R["FINDING_5_F8_clamp"] = f5r

    pathlib.Path(a.json).write_text(json.dumps(R, indent=2, default=str),
                                    encoding="utf-8")
    # ⚠️ `__pycache__` too, or `rmdir` fails and the scratch dir gets banked.
    import shutil as _sh
    _sh.rmtree(tmp, ignore_errors=True)
    print(json.dumps(R, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
