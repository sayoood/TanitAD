"""⛔⛔ DECLARED vs BUILT — a lever named in argv must be a lever the BUILT model has.

MEASURED 2026-09-26 on refcv6-r101-s0, three levers were declared and not built, each found by
hand weeks late: ``--equalize-bottom-rows 43`` (FIX-3, D-REFCV6-EQUALIZE-DROPPED: an undeclared
attribute on ``CNNEncoderConfig`` dropped by the ``--image-hw`` ``dataclasses.replace`` rebuild,
the trunk built 0), F3's per-stage outputs (D-REFCV6-F3-WHITELIST), and three of the selection
mechanisms its config.json declared (FIX-4, D-REFCV6-CONFIG-BUILD).

This file pins:

1. **FIX-3** through ``_pin_trainer_cfg`` WITH ``--image-hw 416 1024`` — the path that dropped it.
   The trunk must zero the bottom 43 rows in the [0, 1] domain and count the call.
2. **Two mutation arms that must go RED**: the field declaration removed (with G-HYG on, the pin
   raises at assignment; with G-HYG also off -- the historical state -- the trunk does not
   equalise and G-DVB names the lever), and a SYNTHETIC lever set as an undeclared attribute
   just before the rebuild (caught by G-HYG at assignment, and by G-DVB when G-HYG is bypassed).
3. **G-DVB** (``tanitad/train/declared_vs_built.py``) through the REAL ``train()``: a clean build
   passes and records itself; the FIX-3 mechanism re-introduced makes ``train()`` REFUSE before
   the first step; every trainer flag has a registry entry; the refcv6 selection mechanisms are
   read off the decoder and an UNWIRED one is named.

Every expectation is a LITERAL (43, (416, 1024), the flag names, the key names), never an
expression over the code under test. CPU only; the timm trunks are built weightless
(``--no-trunk-pretrained``) so nothing is downloaded.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T  # noqa: E402
from tanitad.refs import refc  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.train import config_hygiene as hyg  # noqa: E402
from tanitad.train import declared_vs_built as dvb  # noqa: E402

N_ROWS = 43                        # refcv6-r101-s0's argv
HW = (416, 1024)                   # refcv6-r101-s0's --image-hw
SMALL_TRUNK = ["--trunk", "timm", "--trunk-name", "resnet18.a1_in1k", "--no-trunk-pretrained",
               "--trunk-in-channels", "9"]


def _args(*extra):
    return T.build_parser().parse_args(["--arm", "hier", "--out", "z", *extra])


def _pin(*extra, base=None):
    a = _args(*extra)
    return a, T._pin_trainer_cfg(base if base is not None else v3.refc_v3_smoke_config(True), a)


def _hw_argv(n=N_ROWS, hw=HW):
    return [*SMALL_TRUNK, "--image-hw", str(hw[0]), str(hw[1]), "--equalize-bottom-rows", str(n)]


def _assert_trunk_equalises(trunk, n=N_ROWS, hw=HW):
    """The bottom ``n`` rows of every frame are 0.0 in the [0, 1] domain BEFORE normalisation,
    i.e. exactly ``(0 - mean) / std`` after it; the row above is untouched; one call counted."""
    k = int(trunk.k)
    x = torch.rand(1, 3 * k, *hw) * 0.5 + 0.25          # strictly inside (0, 1): 0 is distinct
    y = trunk.normalise(x)
    zero = ((torch.zeros(1, 3 * k, 1, 1) - trunk._mean) / trunk._std).expand(1, 3 * k, n, hw[1])
    assert torch.equal(y[..., hw[0] - n:, :], zero), "the bottom rows were NOT zeroed"
    above = (x[..., hw[0] - n - 1, :] - trunk._mean[..., 0]) / trunk._std[..., 0]
    assert torch.equal(y[..., hw[0] - n - 1, :], above), "a row ABOVE the strip was touched"
    assert int(getattr(trunk, "equalize_calls", 0)) == 1, "the equalisation was not counted"


# ============================================================================================ #
# 1. FIX-3 through the path that dropped it                                                     #
# ============================================================================================ #
def test_FIX3_the_field_is_DECLARED_and_survives_the_rebuild():
    names = {f.name for f in dataclasses.fields(refc.CNNEncoderConfig)}
    assert "trunk_equalize_bottom_rows" in names
    assert refc.CNNEncoderConfig().trunk_equalize_bottom_rows == 0          # 0 = OFF
    c = dataclasses.replace(refc.CNNEncoderConfig(trunk_equalize_bottom_rows=N_ROWS),
                            image_size=416, image_width=1024)
    assert c.trunk_equalize_bottom_rows == 43


def test_FIX3_image_hw_416x1024_build_EQUALISES_the_bottom_43_rows():
    _, cfg = _pin(*_hw_argv())
    assert cfg.core.encoder.trunk_equalize_bottom_rows == 43
    assert cfg.core.encoder.image_hw() == (416, 1024)
    trunk = refc.build_encoder(cfg.core.encoder)
    assert type(trunk).__name__ == "TimmResNetTrunk"
    assert trunk.cfg.equalize_bottom_rows == 43
    assert tuple(trunk.cfg.image_hw) == (416, 1024)
    _assert_trunk_equalises(trunk)


def test_FIX3_the_in_repo_trunk_REFUSES_a_lever_it_does_not_implement():
    with pytest.raises(SystemExit, match="implements no equalisation"):
        _pin("--equalize-bottom-rows", "43")                 # --trunk refc (the default)


# ============================================================================================ #
# 2. mutation arms                                                                              #
# ============================================================================================ #
def _encoder_class_without_the_field(strict: bool):
    """CNNEncoderConfig exactly as it was before FIX-3: every field but the one that was never
    declared, the same methods, and -- for ``strict=False`` -- no G-HYG guard either."""
    src = refc.CNNEncoderConfig
    flds = [(f.name, f.type, f) for f in dataclasses.fields(src)
            if f.name != "trunk_equalize_bottom_rows"]
    ns = {k: v for k, v in vars(src).items()
          if k in ("feat_dim", "s16_dim", "grid", "image_hw", "grid_shape")}
    cls = dataclasses.make_dataclass("CNNEncoderConfig", flds, namespace=ns)
    return hyg.strict_fields(cls) if strict else cls


def _mutant_cfg(strict: bool):
    cfg = v3.refc_v3_smoke_config(True)
    old = cfg.core.encoder
    Mut = _encoder_class_without_the_field(strict)
    cfg.core.encoder = Mut(**{f.name: getattr(old, f.name) for f in dataclasses.fields(Mut)})
    return cfg


def test_MUTATION_field_removed_WITH_G_HYG_the_pin_RAISES_at_assignment():
    with pytest.raises(hyg.UndeclaredConfigAttribute, match="trunk_equalize_bottom_rows"):
        _pin(*_hw_argv(), base=_mutant_cfg(strict=True))


def test_MUTATION_field_removed_WITHOUT_G_HYG_the_trunk_does_not_equalise_and_G_DVB_names_it():
    """The EXACT historical state (no field, no guard): the pin succeeds, the rebuild drops the
    value, the trunk builds with 0 -- and both checks must go RED."""
    args, cfg = _pin(*_hw_argv(), base=_mutant_cfg(strict=False))
    assert not hasattr(cfg.core.encoder, "trunk_equalize_bottom_rows"), \
        "the mutant did not reproduce the drop"
    trunk = refc.build_encoder(cfg.core.encoder)
    assert trunk.cfg.equalize_bottom_rows == 0
    with pytest.raises(AssertionError):
        _assert_trunk_equalises(trunk)
    model = SimpleNamespace(core=SimpleNamespace(encoder=trunk), _lift_bank=None)
    got = dvb.REGISTRY["equalize_bottom_rows"].check(model, args)
    assert [(m.lever, m.declared, m.built) for m in got] == [("--equalize-bottom-rows", 43, 0)]


class _DropBeforeRebuild:
    """Stands in for the trainer's ``_dc`` (used ONLY by the ``--image-hw`` rebuild): it writes a
    SYNTHETIC lever onto the encoder config immediately before ``replace`` -- the exact position
    of the pin line that dropped C26 -- then rebuilds as the trainer does."""

    def __init__(self, real, bypass: bool):
        self._real, self._bypass, self.wrote = real, bypass, False

    def __getattr__(self, name):
        return getattr(self._real, name)

    def replace(self, obj, **kw):
        if self._bypass:                        # the undeclared attribute, via __dict__
            object.__setattr__(obj, "trunk_synthetic_lever", 7)
        else:
            obj.trunk_synthetic_lever = 7       # the pattern, as the pin wrote it
        self.wrote = True
        return self._real.replace(obj, **kw)


def test_MUTATION_synthetic_undeclared_lever_is_REFUSED_at_assignment(monkeypatch):
    monkeypatch.setattr(T, "_dc", _DropBeforeRebuild(T._dc, bypass=False))
    with pytest.raises(hyg.UndeclaredConfigAttribute, match="trunk_synthetic_lever"):
        _pin(*_hw_argv())


def test_MUTATION_synthetic_lever_that_BYPASSES_G_HYG_is_refused_by_G_DVB(monkeypatch):
    """G-HYG bypassed (a ``__dict__`` write): the rebuild drops the lever silently, exactly like
    C26. G-DVB catches it twice -- the new flag has no registry entry (refused as unregistered),
    and once registered with its built reader, the built value disagrees with argv."""
    shim = _DropBeforeRebuild(T._dc, bypass=True)
    monkeypatch.setattr(T, "_dc", shim)
    args, cfg = _pin(*_hw_argv())
    assert shim.wrote and not hasattr(cfg.core.encoder, "trunk_synthetic_lever")
    parser = T.build_parser()
    parser.add_argument("--synthetic-lever", type=int, default=0)
    args.synthetic_lever = 7
    assert dvb.coverage(parser) == ["synthetic_lever"]
    trunk = refc.build_encoder(cfg.core.encoder)
    model = SimpleNamespace(core=SimpleNamespace(encoder=trunk, cfg=cfg.core), _lift_bank=None,
                            cfg=cfg)
    unreg = [m for m in dvb.check(model, args, parser) if m.lever == "--synthetic-lever"]
    assert len(unreg) == 1 and unreg[0].built == "none"
    monkeypatch.setitem(dvb.REGISTRY, "synthetic_lever", dvb.Lever(
        "synthetic_lever", "built",
        lambda m, a: dvb._eq("synthetic_lever", int(a.synthetic_lever),
                             int(getattr(m.core.encoder.cfg, "synthetic_lever", 0)),
                             "core.encoder.cfg.synthetic_lever")))
    got = dvb.REGISTRY["synthetic_lever"].check(model, args)
    assert [(m.lever, m.declared, m.built) for m in got] == [("--synthetic-lever", 7, 0)]


# ============================================================================================ #
# 3. G-DVB through the REAL train()                                                             #
# ============================================================================================ #
_E2E = ["--arm", "hier", "--smoke", "--synth-episodes", "2", "--steps", "1", "--batch", "2",
        "--device", "cpu", "--log-every", "1", "--save-every", "100",
        "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k", "--no-trunk-pretrained",
        "--trunk-in-channels", "3", "--image-hw", "64", "128", "--equalize-bottom-rows", "8"]


def test_GDVB_a_clean_build_passes_in_train_and_is_RECORDED(tmp_path, monkeypatch):
    seen = {}
    real = dvb.refuse_on_mismatch

    def spy(model, args, parser=None, where="train"):
        seen["mismatches"] = dvb.check(model, args, parser)
        seen["trunk_rows"] = int(model.core.encoder.cfg.equalize_bottom_rows)
        return real(model, args, parser, where)
    monkeypatch.setattr(T._dvb, "refuse_on_mismatch", spy)
    out = tmp_path / "run"
    T.train(T.build_parser().parse_args(_E2E + ["--out", str(out)]))
    assert seen["mismatches"] == [], [str(m) for m in seen["mismatches"]]
    assert seen["trunk_rows"] == 8
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["seams"]["trunk_equalize_bottom_rows"] == 8
    assert cfg["declared_vs_built"]["mismatches"] == 0
    assert cfg["declared_vs_built"]["registry_entries"] >= 200


def test_GDVB_MUTATION_train_REFUSES_before_step_1_when_C26_is_dropped(tmp_path, monkeypatch):
    """FIX-3's mechanism re-introduced into the REAL train() (the pre-fix class, no G-HYG):
    the pin passes, the rebuild drops the value, and G-DVB refuses naming the lever -- before
    config.json and before a single step."""
    real = v3.refc_v3_smoke_config
    monkeypatch.setattr(T.v3, "refc_v3_smoke_config",
                        lambda hier=True: _swap_encoder(real(hier)))
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match="--equalize-bottom-rows: declared 8 but BUILT 0"):
        T.train(T.build_parser().parse_args(_E2E + ["--out", str(out)]))
    assert not (out / "config.json").exists(), "the refusal came after config.json was written"
    assert not (out / "metrics.jsonl").exists()


def _swap_encoder(cfg):
    old = cfg.core.encoder
    Mut = _encoder_class_without_the_field(strict=False)
    cfg.core.encoder = Mut(**{f.name: getattr(old, f.name) for f in dataclasses.fields(Mut)})
    return cfg


def test_GDVB_every_trainer_flag_has_an_entry_and_every_entry_is_well_formed():
    parser = T.build_parser()
    assert dvb.coverage(parser) == []
    # and the reverse: no entry names a flag the parser does not have (a typo'd key would
    # otherwise sit in the registry while the real flag goes unchecked)
    assert set(dvb.REGISTRY) == {a.dest for a in parser._actions if a.dest != "help"}
    # the tip's 197 dests + the 5 flags of the fixes batch + refcv7 NEW-1's --residual-prior
    # + --nav-compliance-tau-file (batch 2, SPEC_REFCV7 §7)
    # + refcv7 NEW-2's 9 flags (--map-hires, --w-map-hires, --map-hires-class-weights,
    # --map-hires-decision-rule, and A6/A7's --map-hires-x-max-m, --map-hires-y-half-m,
    # --map-hires-grad-ckpt, --bev-source, --bev-planner-crop-m; registered by
    # map_head_hires.register_dvb_levers)
    # + NEW-2 R2's --map-hires-near-lift-m (SPEC_REFCV7 A12, the 0.1 m near-range lift)
    # + NEW-2 R3's --map-hires-near-refine-blocks (SPEC_REFCV7 §20, A15, the decoder lever)
    # refcv8 WP-C I1-I3 (INTEGRATION_WPC, wired by WP-B 2026-10-04): +3 = --det-nms, --det-zh-trust,
    # --join-defect-masks, all "data", registered in declared_vs_built itself (223 -> 226)
    # refcv8 WP-B (2026-10-04): +30 = --refcv8 and its 29 --r8-* / --w-r8-* flags (incl. the v9 wiring's
    # --r8-v9-md5, --r8-v9-eval-md5, --r8-v9-lat-variant, --r8-rc-noise-along-m, --r8-rc-noise-lat-m,
    # --r8-no-rc), registered by `tanitad.train.refcv8_train._register_gdvb` (226 -> 256)
    # + --grad-share-every (the refcv8 X4 in-run gradient-share instrument, "runtime") (256 -> 257)
    # + the ladder's regression-arm flags --r8-derange-feed / --r8-rc-roll / --r8-roll-targets ("runtime") (-> 260)
    # + --init-from (the refcv8 warm start, "data") (-> 261)
    # + X3 (SPEC_REFCV8 8.3): --r8-speed-input ("built": cfg.refcv8.speed_input), --r8-speed-unknown-p and
    #   --r8-roll-speed-input ("runtime") (-> 264)
    # + MM ruling Q2: --w-r8-v9-cons (the v9 constraint heads, "loss") (-> 265)
    # + refcv8 X10 (pose-to-image timing, Data FlyWheel): --pose-sync-sidecar, "data", registered in
    #   declared_vs_built itself (-> 266)
    # + MM ruling Q1: --r8-alloc-emit-start ("built": cfg.refcv8.emit_start) (-> 267)
    # + refcv8 (B): --r8-speed-enc8 ("built": cfg.refcv8.speed_enc8 + the seam) (-> 268)
    # + the drivable critic: --r8-critic-drivable ("built") and --w-r8-drivable ("loss") (-> 270)
    assert len(dvb.REGISTRY) == 270
    assert dvb.REGISTRY["r8_speed_enc8"].kind == "built"
    assert dvb.REGISTRY["r8_alloc_emit_start"].kind == "built"
    assert dvb.REGISTRY["w_r8_v9_cons"].kind == "loss"
    assert dvb.REGISTRY["pose_sync_sidecar"].kind == "data"
    assert dvb.REGISTRY["init_from"].kind == "data"
    assert dvb.REGISTRY["r8_speed_input"].kind == "built"
    for d in ("r8_speed_unknown_p", "r8_roll_speed_input"):
        assert dvb.REGISTRY[d].kind == "runtime", d
    for d in ("r8_derange_feed", "r8_rc_roll", "r8_roll_targets"):
        assert dvb.REGISTRY[d].kind == "runtime", d
    assert dvb.REGISTRY["grad_share_every"].kind == "runtime"
    for d in ("r8_v9_md5", "r8_v9_eval_md5", "r8_v9_lat_variant"):
        assert dvb.REGISTRY[d].kind == "data", d
    for d in ("r8_rc_noise_along_m", "r8_rc_noise_lat_m", "r8_no_rc"):
        assert dvb.REGISTRY[d].kind == "runtime", d
    for d in ("det_nms", "det_zh_trust", "join_defect_masks"):
        assert dvb.REGISTRY[d].kind == "data", d
    for d in ("refcv8", "r8_n_alloc", "r8_prior_free_group", "r8_lat_prior_dropout", "r8_seed"):
        assert dvb.REGISTRY[d].kind == "built", d
    for d in ("w_r8_cons", "w_r8_sat", "w_r8_listwise", "w_r8_subscore"):
        assert dvb.REGISTRY[d].kind == "loss", d
    assert dvb.REGISTRY["r8_rc_dropout"].kind == "runtime"
    assert dvb.REGISTRY["r8_v9_labels"].kind == "data"
    # refcv7 diagnostics F1/F2/F4 (2026-10-04): +2 = --map-hires-class-thresholds ("built",
    # _c_map_hires_class_thresholds) and --det-presence-gates ("data"), both OPT-IN (221 -> 223)
    # refcv7 A14 (HQS): +1 = --slot-query-select (220 -> 221)
    # refcv7 A9 (box head): +5 = --slot-presence-loss, --slot-presence-prior,
    # --slot-deep-supervision, --slot-vis1, --vis1-sidecar (215 -> 220)
    for d in ("map_hires", "w_map_hires", "map_hires_class_weights",
              "map_hires_decision_rule", "map_hires_x_max_m", "map_hires_y_half_m",
              "map_hires_grad_ckpt", "bev_source", "bev_planner_crop_m",
              "map_hires_near_lift_m", "map_hires_near_refine_blocks"):
        assert dvb.REGISTRY[d].kind == ("loss" if d == "w_map_hires" else "built"), d
    assert dvb.REGISTRY["map_hires_class_thresholds"].kind == "built"
    assert dvb.REGISTRY["det_presence_gates"].kind == "data"
    for dest, lever in dvb.REGISTRY.items():
        assert lever.kind in dvb.KINDS, dest
        if lever.kind in ("built", "loss"):
            assert callable(lever.check), dest
        else:
            assert lever.reason, dest


def test_GDVB_a_NEW_flag_without_an_entry_is_REFUSED():
    parser = T.build_parser()
    parser.add_argument("--brand-new-lever", action="store_true")
    assert dvb.coverage(parser) == ["brand_new_lever"]


# ---- FIX-4: the selection mechanisms ------------------------------------------------------ #
def _v6_model(**core_flags):
    """The refcv6 tactical build of `test_refcv6_tactical.py::_e2e_model`, on the smoke rig."""
    from tanitad.refs import refcv6_tactical as v6tac
    from tanitad.refs.refc_agents import AgentSeamConfig
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.tac_vocab_version = "v7.0"
    cfg.tac_decoder_v6 = True
    cfg.max_speed_onehot_v6 = True
    cfg.core.agents = AgentSeamConfig(enable=True)
    cfg.core.decoder.cross_agent = True
    for k, v in core_flags.items():
        setattr(cfg.core, k, v)
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(d_model=64, n_layers=1, n_heads=4,
                                                      d_bev=0)
    return v3.RefCV3Model(cfg)


_ALL_ON = dict(graft_tac8_prior=True, graft_behaviour_sel=True, graft_nav_compliance=True,
               nav_compliance_tau_rad=0.08, speed_ceiling_filter=True)
_SEL_ARGV = argparse.Namespace(graft_tac8_prior=True, graft_behaviour_sel=True,
                               graft_nav_compliance=True, nav_compliance_tau_rad=0.08,
                               speed_ceiling_filter=True)


def _sel_mismatches(model, args):
    out = []
    for d in ("graft_tac8_prior", "graft_behaviour_sel", "graft_nav_compliance",
              "speed_ceiling_filter", "nav_compliance_tau_rad"):
        out += dvb.REGISTRY[d].check(model, args)
    return out + dvb._c_selection_declaration(model, args)


def test_FIX4_the_declaration_is_READ_OFF_THE_DECODER():
    m = _v6_model(**_ALL_ON)
    roles = m.provenance_roles()
    assert roles["selection_mechanisms_built"] == {
        "anchor_confidence": True, "maneuver_prior_5way": False, "image_prior_lat3_lon3": False,
        "tac8_prior": True, "behaviour_set": True, "nav_compliance": True,
        "speed_ceiling": True}
    assert roles["selection_inputs_not_built"] == [
        "image-only 5-way maneuver prior -> anchor prior (H19)",
        "image-only lat3/lon3 prior -> anchor prior (D-TAC1 factored head; in use because the "
        "8x8 tactical posterior does NOT replace it on this build)"]
    assert _sel_mismatches(m, _SEL_ARGV) == []


def test_FIX4_refcv6_r101_s0s_build_DECLARES_only_what_it_built():
    """The live run's selection surface: only the behaviour set of the four refcv6 mechanisms
    (MEASURED on ckpt_30000: 0 `tac8_*` / 0 `navc_*` keys, lat3/lon3 grafts trained)."""
    m = _v6_model(graft_behaviour_sel=True)
    b = m.provenance_roles()["selection_mechanisms_built"]
    assert (b["tac8_prior"], b["behaviour_set"], b["nav_compliance"], b["speed_ceiling"],
            b["image_prior_lat3_lon3"]) == (False, True, False, False, True)
    assert "nav compliance (PARAMETER-FREE geometric predicate, one zero-init gate)" \
        not in m.provenance_roles()["selection_inputs"]


def test_FIX4_MUTATION_unwire_one_selection_term_goes_RED():
    m = _v6_model(**_ALL_ON)
    m.core.decoder.speed_ceiling_filter = False          # the term, unwired after the build
    got = _sel_mismatches(m, _SEL_ARGV)
    assert ("--speed-ceiling-filter", True, False) in [(x.lever, x.declared, x.built) for x in got]


def test_FIX4_MUTATION_the_STATIC_declaration_goes_RED(monkeypatch):
    """The historical defect: a declaration that lists every mechanism whenever v6 is on."""
    m = _v6_model(graft_behaviour_sel=True)
    monkeypatch.setattr(v3, "refcv6_selection_built",
                        lambda model: {k: True for k, _ in v3.REFCV6_SELECTION_MECHANISMS})
    args = argparse.Namespace(graft_behaviour_sel=True)
    levers = sorted(x.lever for x in dvb._c_selection_declaration(m, args))
    assert levers == ["provenance_roles[nav_compliance]", "provenance_roles[speed_ceiling]",
                      "provenance_roles[tac8_prior]"]


def test_FIX4_dead_selection_flags_REFUSE_at_the_pin():
    with pytest.raises(SystemExit, match="--graft-tac8-prior without --tac-decoder-v6"):
        _pin("--graft-tac8-prior")
    with pytest.raises(SystemExit, match="needs --nav-compliance-tau-rad"):
        _pin("--graft-nav-compliance")
    with pytest.raises(SystemExit, match="without --graft-nav-compliance"):
        _pin("--nav-compliance-tau-rad", "0.08")
    with pytest.raises(SystemExit, match="--speed-ceiling-filter without --max-speed-input-v6"):
        _pin("--speed-ceiling-filter")


def test_FIX4_the_flags_default_OFF():
    a = _args()
    assert (a.graft_tac8_prior, a.graft_nav_compliance, a.nav_compliance_tau_rad,
            a.speed_ceiling_filter) == (False, False, 0.0, False)


# ============================================================================================ #
# 4. the record: the ack stamp and the legacy as-trained resolver                               #
# ============================================================================================ #
def test_the_ack_ddim_no_u0_stamp_REACHES_the_seam_stamp():
    args, cfg = _pin("--sampler", "ddim", "--anchor-v0-conditioned", "--f6-w-u0-zero")
    st = T._seam_stamp(cfg, args)
    assert st["u0_absent_under_ddim"] == "pi-acknowledged-2026-09-11-refcv6-arm-D"
    assert T._seam_stamp(*reversed(_pin()))["u0_absent_under_ddim"] is None


@pytest.mark.parametrize("config,rows", [
    ({"argv": ["--image-hw", "416", "1024", "--equalize-bottom-rows", "43"]}, 0),
    ({"argv": ["--equalize-bottom-rows", "43"]}, 43),
    ({"argv": ["--image-hw", "416", "1024"]}, 0),
    ({"argv": ["--image-hw", "416", "1024", "--equalize-bottom-rows", "43"],
      "seams": {"trunk_equalize_bottom_rows": 43}}, 43),
])
def test_a_PRE_FIX_record_is_rebuilt_with_the_rows_its_trunk_really_zeroed(config, rows):
    assert T.trunk_equalize_rows_as_trained(config)[0] == rows


# ============================================================================================ #
# 5. DrivoR-T (SPEC_REFCV7 §6.1): listed, never read, and OFF on a refcv7 launch                #
# ============================================================================================ #
def test_DRIVORT_levers_are_listed_with_their_ARGPARSE_defaults():
    """The literal table must equal what the parser defaults to -- a drifted literal would let a
    DrivoR-T flag at its (new) default read as 'set' or, worse, a set one read as off."""
    parser = T.build_parser()
    defaults = {a.dest: a.default for a in parser._actions}
    assert {d: defaults[d] for d in dvb.DRIVORT_DEFAULTS} == dvb.DRIVORT_DEFAULTS
    assert all(dvb.REGISTRY[d].kind == "drivort" for d in dvb.DRIVORT_DEFAULTS)
    assert len(dvb.DRIVORT_DEFAULTS) == 19


def test_DRIVORT_a_refcv7_launch_REFUSES_any_set_DrivoR_T_flag():
    args = _args("--refcv7", "--w-r7-wta", "0.5")
    assert dvb.drivort_levers_set(args) == ["--refcv7", "--w-r7-wta"]
    model = SimpleNamespace()                  # nothing built: only the argv half is exercised
    got = [m.lever for m in dvb.check(model, args, forbid_kinds=("drivort",))
           if m.read_from == "argv"]
    assert got == ["--refcv7", "--w-r7-wta"]
    assert [m for m in dvb.check(model, _args(), forbid_kinds=("drivort",))
            if m.read_from == "argv"] == []
    # train() does NOT forbid them: without the kind the argv half reports nothing
    assert [m for m in dvb.check(model, args) if m.read_from == "argv"] == []


# ============================================================================================ #
# 6. PI RULING 2026-09-26 (D-REFCV7-E1): all three selection mechanisms ON for refcv7           #
# ============================================================================================ #
def test_REFCV7_REQUIRED_ON_is_the_PI_literal():
    assert dvb.REFCV7_REQUIRED_ON == ("graft_tac8_prior", "graft_nav_compliance",
                                      "speed_ceiling_filter")


#: ⛔ SPEC_REFCV7 §10 (A5): a refcv7 build ALSO carries the residual prior `ha0_ext_pose` on the
#: control-space DDIM sampler (v0-conditioned vocabulary) and the ego history the prior reads.
#: NEW-1's `_arm` recipe (`test_residual_prior.py`) through the trainer's OWN pin, then the
#: `_v6_model` tactical setup and the three selection mechanisms on the config.
_R7_PIN = ["--sampler", "ddim", "--anchor-v0-conditioned", "--anchor-control-units", "alat",
           "--n-anchors", "20", "--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln",
           "--f5-focal", "--f5-emitting-conf", "--f6-w-u0-zero"]
_R7_ARGV = argparse.Namespace(**vars(_SEL_ARGV), residual_prior="ha0_ext_pose", ego_history=True)


def _r7_model(residual_prior="ha0_ext_pose", **core_flags):
    from tanitad.refs import refcv6_tactical as v6tac
    from tanitad.refs.refc_agents import AgentSeamConfig
    argv = list(_R7_PIN) + ["--ego-history"]
    if residual_prior is not None:
        argv += ["--residual-prior", residual_prior]
    _a_, cfg = _pin(*argv)
    cfg.tac_vocab_version = "v7.0"
    cfg.tac_decoder_v6 = True
    cfg.max_speed_onehot_v6 = True
    cfg.core.agents = AgentSeamConfig(enable=True)
    cfg.core.decoder.cross_agent = True
    for k, v in {**_ALL_ON, **core_flags}.items():
        setattr(cfg.core, k, v)
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(d_model=64, n_layers=1, n_heads=4,
                                                      d_bev=0)
    return v3.RefCV3Model(cfg)


@pytest.fixture(scope="module")
def r7():
    return _r7_model()


def test_REFCV7_all_three_on_and_built_PASSES(r7):
    assert dvb.check_refcv7_required(r7, _R7_ARGV) == []


@pytest.mark.parametrize("missing", ["graft_tac8_prior", "graft_nav_compliance",
                                     "speed_ceiling_filter"])
def test_REFCV7_RED_an_argv_missing_one_is_REFUSED(r7, missing):
    args = argparse.Namespace(**{**vars(_R7_ARGV), missing: False})
    got = [(x.lever, x.built) for x in dvb.check_refcv7_required(r7, args)]
    assert got == [("--" + missing.replace("_", "-"), "OFF in argv")]


def test_REFCV7_RED_on_in_argv_but_UNBUILT_is_REFUSED():
    """refcv6-r101-s0's own build (behaviour set only, no residual prior, no ego history) under a
    refcv7 argv: FIVE refusals."""
    got = sorted((x.lever, x.built) for x in
                 dvb.check_refcv7_required(_v6_model(graft_behaviour_sel=True), _R7_ARGV))
    assert got == [("--ego-history", "not built"), ("--graft-nav-compliance", "not built"),
                   ("--graft-tac8-prior", "not built"), ("--residual-prior", "off"),
                   ("--speed-ceiling-filter", "not built")]


def test_REFCV7_tau_must_EQUAL_the_banked_file(r7, tmp_path):
    p = tmp_path / "nav_compliance_tau_train.json"
    p.write_text(json.dumps({"tau": 0.08}), encoding="utf-8")
    assert dvb.check_refcv7_required(r7, _R7_ARGV, tau_file=str(p)) == []
    off = argparse.Namespace(**{**vars(_R7_ARGV), "nav_compliance_tau_rad": 0.09})
    assert [x.lever for x in dvb.check_refcv7_required(r7, off, tau_file=str(p))] == \
        ["--nav-compliance-tau-rad"]


# ---- SPEC_REFCV7 §10 (A5): "G-DVB refuses any other mode for a refcv7 launch" (batch 3 (c)) - #
def test_REFCV7_RESIDUAL_PRIOR_is_the_SPEC_literal():
    assert dvb.REFCV7_RESIDUAL_PRIOR == "ha0_ext_pose"


@pytest.mark.parametrize("mode", ["ha0_ext", "cv_yawrate", "off"])
def test_REFCV7_RED_any_other_residual_prior_in_ARGV_is_REFUSED(r7, mode):
    args = argparse.Namespace(**{**vars(_R7_ARGV), "residual_prior": mode})
    got = [(x.lever, x.declared, x.built, x.read_from)
           for x in dvb.check_refcv7_required(r7, args)]
    assert got == [("--residual-prior", "ha0_ext_pose", mode, "argv")]


@pytest.mark.parametrize("mode", ["ha0_ext", "cv_yawrate", None])
def test_REFCV7_RED_a_decoder_BUILT_with_any_other_prior_is_REFUSED(mode):
    """The argv says `ha0_ext_pose`; the BUILT decoder carries another mode (None = off)."""
    got = [(x.lever, x.declared, x.built, x.read_from)
           for x in dvb.check_refcv7_required(_r7_model(residual_prior=mode), _R7_ARGV)]
    assert got == [("--residual-prior", "ha0_ext_pose", mode or "off",
                    "core.decoder.residual_prior")]


def test_REFCV7_RED_ego_history_OFF_in_argv_or_NOT_BUILT_is_REFUSED(r7):
    args = argparse.Namespace(**{**vars(_R7_ARGV), "ego_history": False})
    assert [(x.lever, x.built) for x in dvb.check_refcv7_required(r7, args)] == \
        [("--ego-history", "OFF in argv")]
    m = _r7_model()
    m.core.ego_hist = None                      # REGRESSION ARM: lost in the build
    got = [(x.lever, x.built, x.read_from) for x in dvb.check_refcv7_required(m, _R7_ARGV)]
    assert got == [("--ego-history", "not built", "core.ego_hist")]


def test_GDVB_REFUSES_a_decoder_carrying_the_in_training_ceiling_switch(monkeypatch):
    m = _v6_model(**_ALL_ON)
    assert dvb.REGISTRY["speed_ceiling_filter"].check(m, _SEL_ARGV) == []
    monkeypatch.setattr(refc.AnchoredDiffusionDecoder, "speed_ceiling_in_training", True)
    got = [(x.lever, x.read_from) for x in dvb.REGISTRY["speed_ceiling_filter"].check(m, _SEL_ARGV)]
    assert got == [("--speed-ceiling-filter", "core.decoder.speed_ceiling_in_training")]
