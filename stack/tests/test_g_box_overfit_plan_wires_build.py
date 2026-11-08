"""G-BOX-OVERFIT builds -- and G-DVB passes -- the refcv9 C8 planner wires the E2 argv DECLARES (arm K2 and the banded K2M* variants).

WHY (MEASURED on Thor 2026-11-07 ~19:00Z, ``r9e1_bind.sh --argv-prefix r9e2 box K2``, ``stack/ops/runs.d/r9e2_K2.argv.json``): the box harness
exited 1 at its own build check, 165 s in:

    [gbo] the replayed build is NOT the declared launch model:
      - r9_plan_box_tokens: declared True but BUILT False (read from model.core.box_tok_r9) -- argv and the build disagree

``train()`` builds the planner wires with the trainer's own ``_pwf.attach(model, args)`` (``refc_v3_train.py``, right after ``_bbf.attach``, before ``_bda`` /
``_p3f`` / ``_bc5``); ``TrainerAdapter._build_model`` replays that build and never called it -- its comment said "no C8 flags in the rig", true of the E1 rig and
false of the E2 FULL model (K2 = K0 + T1h bundle + ``--r9-plan-box-tokens`` + W-MAP-1; K2M* add the banded head + ``--r9-plan-map-channels``).

The REFERENCE is the trainer's own ``train()`` on a tiny K2-shaped rig, captured as the launch gate captures it (``launch_gate.run_trainer_until(..., "model")``,
before any data is read). The tiny rigs are ``test_g_box_overfit_near_lift._argv`` (the K0 blocks at rig size: --arm hier, --agents head, --w-box3d, the tactical
v6 decoder, --bev-source map_hires_pool) plus EXACTLY the build levers the real E2 argv adds (pinned against ``runs.d/r9e2_*.argv.json`` below).

* (a) the harness replay of K2 / K2M1N20 / K2M3 / K2M3N20 IS train()'s model: G-DVB clean, ``model.core.box_tok_r9`` built, a zero-init ``box_gate`` on every
  consuming planner AND tactical layer, ``map_chan_proj`` on the banded variants, same state_dict keys, same fresh-init digests, the trainer's own stamp;
* (b) the OFF path is BIT-IDENTICAL: on K0 and the E1 rig (C0) the harness and a COPY with the new ``_pwf.attach`` line removed (= the tip's replay) build the
  same keys, the same digests AND leave the same global RNG state; no ``box_tok_r9`` / ``box_gate`` / ``map_chan_proj`` name exists; ``_pwf.attach`` itself touches
  nothing when both flags are off;
* (c) RED (mutation): with the new line removed from a COPY, K2 is refused with the MEASURED mismatch line, BY NAME, and the banded variants add the W-MAP-2 pair;
* the replay calls the attach functions in the TRAINER's order (``_bbf`` -> ``_pwf`` -> ``_bc5``);
* the MAP harness (``map_hires_overfit.py``) needed no change: it builds a trunk + ``MapHiresBranch`` by hand, no ``RefCV3Model``, no ``model._perception``, no G-DVB; the
  banded E2 argvs reach it only as build facts (banded / own-enc / near band), and the C8 flags move none of them.
"""
from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest
import torch

STACK = Path(__file__).resolve().parents[1]
for _p in (str(STACK), str(STACK / "scripts"), str(STACK / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG                                           # noqa: E402
import map_hires_overfit as MH                                     # noqa: E402
import test_g_box_overfit_near_lift as NL                          # noqa: E402  (its tiny rig, inputs, adapter helpers)
import test_g_box_overfit_rig as RG                                # noqa: E402  (the E1 rig argv at tiny size)

GBO_PATH = STACK / "scripts" / "g_box_overfit.py"
MAP_PATH = STACK / "scripts" / "map_hires_overfit.py"
TRAINER_PATH = STACK / "scripts" / "refc_v3_train.py"
RUNS = STACK / "ops" / "runs.d"
PWF_LINE = "        tr._pwf.attach(m, a)\n"
BOX_FLAG, MAP_FLAG = "--r9-plan-box-tokens", "--r9-plan-map-channels"
#: the C8 names a build WITHOUT the wires must not contain (parameter / buffer names)
C8_NAMES = ("box_tok_r9", "box_gate", "map_chan_proj")
#: what Thor measured for K2 (verbatim), and the W-MAP-2 pair the banded variants add
BOX_RED = "  - r9_plan_box_tokens: declared True but BUILT False (read from model.core.box_tok_r9) -- argv and the build disagree"
MAP_RED = ["  - r9_plan_map_channels: declared True but BUILT False (read from model._perception.bev_pool.map_chan_proj) -- argv and the build disagree",
           "  - r9_plan_map_channels: declared True but BUILT False (read from model._r9_plan_map_channels) -- argv and the attribute disagree"]
#: variant -> (real E2 argv file, the tiny near-lift m, extra tiny flags beyond tiny K0 [the build levers the real argv adds], has W-MAP-2)
VARIANTS = {
    "k2": ("r9e2_K2", None, [BOX_FLAG], False),
    "k2m1n20": ("r9e2_K2M1N20", 20, ["--r9-map-banded", "--r9-map-banded-near-m", "20", MAP_FLAG, BOX_FLAG], True),
    "k2m3": ("r9e2_K2M3", 10, ["--r9-map-banded", "--r9-map-own-enc", MAP_FLAG, BOX_FLAG], True),
    "k2m3n20": ("r9e2_K2M3N20", 20, ["--r9-map-banded", "--r9-map-own-enc", "--r9-map-banded-near-m", "20", MAP_FLAG, BOX_FLAG], True),
}
OFF = ("k0", "c0")


def _real_argv(stem: str) -> list[str]:
    return list(json.loads((RUNS / f"{stem}.argv.json").read_text(encoding="utf-8"))["argv"])


def _tiny_argv(d: Path, near_lift, extra) -> list[str]:
    """tiny K0 (``NL._argv``) + ``extra``; a banded variant is built over the M arms' extent (100 x +-30 m) and its near lift, as ``test_g_box_overfit_banded`` does."""
    argv = NL._argv(d)
    if near_lift is not None:
        for flag, val in (("--map-hires-x-max-m", "100"), ("--map-hires-y-half-m", "30"), ("--map-hires-near-lift-m", str(near_lift))):
            argv[argv.index(flag) + 1] = val
        argv[argv.index("--map-hires-class-weights") + 1] = str(d / "cw100.json")
    return argv + list(extra)


@pytest.fixture(scope="module")
def rigs(tmp_path_factory):
    d = tmp_path_factory.mktemp("gbo_pwf_build")
    NL._synth_inputs(d)
    from tanitad.data.semantic_map_gt_fine import FINE_CLASSES
    from tanitad.models import map_head_hires as H
    (d / "cw100.json").write_text(json.dumps(          # the class weights AT the banded extent (the reader refuses another)
        {"schema": H.CLASS_WEIGHT_SCHEMA, "classes": list(FINE_CLASSES), "weights": [1.0, 1.5, 3.0, 2.0, 4.0, 1.2, 2.5, 1.1],
         "dry_run": False, "extent": {"x_max_m": 100.0, "y_half_m": 30.0}}), encoding="utf-8")
    argvs = {"k0": NL._argv(d), "c0": RG._rig_argv(d)}
    argvs.update({k: _tiny_argv(d, near, extra) for k, (_stem, near, extra, _m) in VARIANTS.items()})
    T = NL._load("refc_v3_train_gbo_pwf_ref", TRAINER_PATH)
    refs = {k: LG.run_trainer_until(T, v, "model")["model"] for k, v in argvs.items()}      # the REFERENCES: the real train()
    G = NL._load("g_box_overfit_pwf_build_t", GBO_PATH)
    return types.SimpleNamespace(d=d, T=T, G=G, argvs=argvs, ref=refs)


def _mutated(tmp_path: Path, name: str, edits: list[tuple[str, str]]):
    src = GBO_PATH.read_text(encoding="utf-8")
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    p = tmp_path / f"{name}.py"
    p.write_text(src, encoding="utf-8", newline="\n")
    return NL._load(f"{name}_t", p)


def _tip_replay(tmp_path: Path):
    """The harness as it was at the tip: the ``_pwf.attach`` line removed."""
    return _mutated(tmp_path, "gbo_tip_replay_pwf", [(PWF_LINE, "")])


def _digests(m):
    return LG.state_digests(m.state_dict())


def _c8_names(m) -> list[str]:
    return sorted(n for n, _ in list(m.named_parameters()) + list(m.named_buffers()) if any(c in n for c in C8_NAMES))


def _flags(argv) -> set:
    return {str(t).split("=", 1)[0] for t in argv if str(t).startswith("--")}


# ---------------------------------------------------------------------------------------------------------------------- #
# 0. the tiny rigs carry exactly the build levers the REAL E2 argvs add to K0 (and K0 / C0 carry none)                    #
# ---------------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", list(VARIANTS))
def test_the_tiny_rig_carries_the_build_levers_the_real_E2_argv_adds_to_K0(rigs, kind):
    stem, _near, extra, has_map = VARIANTS[kind]
    real, k0 = _real_argv(stem), _real_argv("r9e2_K0")
    T = rigs.T
    pr, p0 = T.build_parser().parse_args(real), T.build_parser().parse_args(k0)
    from tanitad.train import plan_wires_flag as PW
    assert PW.box_on(pr) and PW.map_on(pr) is has_map and not PW.box_on(p0) and not PW.map_on(p0)       # the real argv declares it, K0 does not
    pt = T.build_parser().parse_args(rigs.argvs[kind])
    assert PW.box_on(pt) and PW.map_on(pt) is has_map                                                   # and so does the tiny rig
    # the harness's own build argv (minus --trunk-compile, the aux flags, T1) still carries the C8 flags of the REAL argv: the replay MUST build them
    build, _rec = rigs.G.harness_build_argv(real)
    assert BOX_FLAG in build and (MAP_FLAG in build) is has_map
    # the tiny rig's added flags are exactly the real argv's build levers (the T1h bundle is stripped by the harness; W-MAP-1's critic flags are not a build lever here)
    added_real = (_flags(real) - _flags(k0)) - set(rigs.G.T1_FLAGS) - {"--r8-critic-drivable", "--w-r8-drivable"}
    added_tiny = _flags(rigs.argvs[kind]) - _flags(rigs.argvs["k0"])
    assert added_tiny == added_real == _flags(extra), (sorted(added_tiny), sorted(added_real))


@pytest.mark.parametrize("kind", ["k0", "c0"])
def test_the_OFF_rigs_declare_no_C8_wire(rigs, kind):
    from tanitad.train import plan_wires_flag as PW
    p = rigs.T.build_parser().parse_args(rigs.argvs[kind])
    assert not PW.box_on(p) and not PW.map_on(p)
    assert not (_flags(rigs.argvs[kind]) & {BOX_FLAG, MAP_FLAG})
    stem = {"k0": "r9e2_K0", "c0": "r9e1_C0"}[kind]                      # the real arms they stand for carry none either
    assert not (_flags(_real_argv(stem)) & {BOX_FLAG, MAP_FLAG})


# ---------------------------------------------------------------------------------------------------------------------- #
# (a) the harness replay IS the model train() builds                                                                      #
# ---------------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", list(VARIANTS))
def test_the_harness_replay_IS_the_C8_model_train_builds(rigs, kind):
    has_map = VARIANTS[kind][3]
    ref, T = rigs.ref[kind], rigs.T
    m, ad = NL._harness_model(rigs.G, T, rigs.argvs[kind])
    assert ad.dvb_mismatches == []                                       # zero G-DVB mismatches
    assert m.core.box_tok_r9 is not None and getattr(m, "_r9_plan_box_tokens", False) is True
    n_cross = [ly for ly in m.core.decoder.layers if getattr(ly, "cross_agent", None) is not None]
    assert n_cross and all(ly.box_gate is not None and float(ly.box_gate.detach()) == 0.0 for ly in n_cross)          # a zero-init gate on every consuming planner layer
    assert m.tac_decoder_v6 is not None and all(float(ly.box_gate.detach()) == 0.0 for ly in m.tac_decoder_v6.layers)   # ... and every tactical layer
    assert (m._perception.bev_pool.map_chan_proj is not None) is has_map and getattr(m, "_r9_plan_map_channels", False) is has_map
    kt, kh = set(ref.state_dict()), set(m.state_dict())
    assert sorted(kt ^ kh) == []                                         # the same parameters
    assert any(k.startswith("core.box_tok_r9.") for k in kh) and any(k.endswith(".box_gate") for k in kh)
    assert any(k.endswith("map_chan_proj.proj.weight") or ".map_chan_proj." in k for k in kh) is has_map
    dh, dt = _digests(m), _digests(ref)
    assert [k for k in dh if dh[k] != dt[k]] == []                       # the same fresh-init digests (the wires draw inside fork_rng: the global stream is the OFF build's)
    assert T._pwf.stamp(m, ad.args) == T._pwf.stamp(ref, T.build_parser().parse_args(rigs.argvs[kind]))


# ---------------------------------------------------------------------------------------------------------------------- #
# (b) the OFF path is bit-identical                                                                                       #
# ---------------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", list(OFF))
def test_OFF_path_is_bit_identical_to_the_tip_replay_and_to_train(rigs, tmp_path, kind):
    G0 = _tip_replay(tmp_path)
    m_new, ad_new = NL._harness_model(rigs.G, rigs.T, rigs.argvs[kind])
    rng_new = torch.get_rng_state().clone()
    m_old, ad_old = NL._harness_model(G0, rigs.T, rigs.argvs[kind])
    rng_old = torch.get_rng_state().clone()
    ref = rigs.ref[kind]
    assert ad_new.dvb_mismatches == ad_old.dvb_mismatches == []
    dn, do_, dt = _digests(m_new), _digests(m_old), _digests(ref)
    assert sorted(dn) == sorted(do_) == sorted(dt)                       # parameter NAMES unchanged
    assert [k for k in dn if dn[k] != do_[k]] == [] and [k for k in dn if dn[k] != dt[k]] == []     # and every fresh-init digest
    assert torch.equal(rng_new, rng_old)                                 # and the global RNG stream the build leaves
    assert _c8_names(m_new) == _c8_names(m_old) == _c8_names(ref) == []  # no box_tok_r9 / box_gate / map_chan_proj
    for mm in (m_new, m_old, ref):
        assert not hasattr(mm, "_r9_plan_box_tokens") and not hasattr(mm, "_r9_plan_map_channels")   # ABSENT (not False), as in train()
        assert getattr(mm.core, "box_tok_r9", None) is None
        assert getattr(getattr(mm._perception, "bev_pool", None), "map_chan_proj", None) is None
    assert rigs.T._pwf.stamp(m_new, ad_new.args) is None                  # config.json carries no "plan_wires" block


@pytest.mark.parametrize("kind", list(OFF))
def test_pwf_attach_touches_nothing_when_both_C8_flags_are_off(rigs, kind):
    m, ad = NL._harness_model(rigs.G, rigs.T, rigs.argvs[kind])
    before_d, before_rng = _digests(m), torch.get_rng_state().clone()
    before_mods = [(n, id(x)) for n, x in m.named_modules()]
    before_attrs = sorted(vars(m))
    rigs.T._pwf.attach(m, ad.args)                                       # called a second time: still a no-op
    assert _digests(m) == before_d and torch.equal(torch.get_rng_state(), before_rng)
    assert [(n, id(x)) for n, x in m.named_modules()] == before_mods and sorted(vars(m)) == before_attrs


# ---------------------------------------------------------------------------------------------------------------------- #
# (c) RED: without the new line the replay is refused, by name                                                            #
# ---------------------------------------------------------------------------------------------------------------------- #
def test_RED_without_the_attach_line_K2_is_refused_with_the_MEASURED_thor_mismatch(rigs, tmp_path):
    G0 = _tip_replay(tmp_path)
    with pytest.raises(SystemExit) as e:
        NL._harness_model(G0, rigs.T, rigs.argvs["k2"])
    msg = str(e.value)
    assert msg.startswith("[gbo] the replayed build is NOT the declared launch model:")
    assert [ln for ln in msg.splitlines()[1:]] == [BOX_RED]              # exactly the one line Thor printed, nothing else
    # control: the fixed harness builds the same argv clean (so the RED is the missing line, not the rig)
    assert NL._harness_model(rigs.G, rigs.T, rigs.argvs["k2"])[1].dvb_mismatches == []


@pytest.mark.parametrize("kind", ["k2m1n20", "k2m3", "k2m3n20"])
def test_RED_without_the_attach_line_the_banded_variants_add_the_W_MAP_2_pair(rigs, tmp_path, kind):
    G0 = _tip_replay(tmp_path)
    with pytest.raises(SystemExit) as e:
        NL._harness_model(G0, rigs.T, rigs.argvs[kind])
    assert sorted(str(e.value).splitlines()[1:]) == sorted(MAP_RED + [BOX_RED])


def test_RED_each_wire_is_refused_alone_by_its_own_flag(rigs, tmp_path):
    """W-MAP-2 without W-BOX on a banded argv: only the map pair is refused (the box line must not appear), so each wire is checked by its own flag."""
    argv = [x for x in rigs.argvs["k2m3"] if x != BOX_FLAG]
    G0 = _tip_replay(tmp_path)
    with pytest.raises(SystemExit) as e:
        NL._harness_model(G0, rigs.T, argv)
    assert sorted(str(e.value).splitlines()[1:]) == sorted(MAP_RED)
    m, ad = NL._harness_model(rigs.G, rigs.T, argv)                      # the fixed harness: map wire built, no box tokens
    assert ad.dvb_mismatches == [] and getattr(m.core, "box_tok_r9", None) is None and m._perception.bev_pool.map_chan_proj is not None


# ---------------------------------------------------------------------------------------------------------------------- #
# the replay calls the attach functions in the TRAINER's order                                                            #
# ---------------------------------------------------------------------------------------------------------------------- #
ATTACH_OWNERS = ("_bbf", "_pwf", "_bda", "_p3f", "_bc5")


def _attach_order(src: str, only_in: str | None = None) -> list[str]:
    """The owners of every ``<owner>.attach(...)`` call (``_pwf.attach`` / ``tr._pwf.attach``), in source order, optionally inside the function ``only_in``."""
    tree = ast.parse(src)
    scope = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == only_in] if only_in else [tree]
    assert scope, only_in
    hits = []
    for root in scope:
        for n in ast.walk(root):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "attach":
                v = n.func.value
                owner = v.id if isinstance(v, ast.Name) else (v.attr if isinstance(v, ast.Attribute) else None)
                if owner in ATTACH_OWNERS:
                    hits.append((n.lineno, owner))
    return [o for _l, o in sorted(hits)]


def test_the_replay_attaches_in_the_trainers_order():
    trainer = _attach_order(TRAINER_PATH.read_text(encoding="utf-8"), only_in="train")
    harness = _attach_order(GBO_PATH.read_text(encoding="utf-8"))
    assert trainer == ["_bbf", "_pwf", "_bda", "_p3f", "_bc5"], trainer
    assert harness == ["_bbf", "_pwf", "_bc5"], harness                  # _bda / _p3f are the training-only aux heads, stripped on purpose (strip_aux)
    assert harness == [o for o in trainer if o in harness]               # the replay is a subsequence of the trainer's order


def test_RED_the_order_pin_catches_a_pwf_attach_moved_after_bc5():
    src = GBO_PATH.read_text(encoding="utf-8")
    bc5 = "        tr._bc5.attach(m, a)\n"
    assert src.count(PWF_LINE) == 1 and src.count(bc5) == 1
    moved = src.replace(PWF_LINE, "").replace(bc5, bc5 + PWF_LINE)
    assert _attach_order(moved) == ["_bbf", "_bc5", "_pwf"]
    assert _attach_order(moved) != _attach_order(src)


# ---------------------------------------------------------------------------------------------------------------------- #
# the MAP harness: no change needed -- it builds no model that G-DVB reads and no planner pool the wires extend           #
# ---------------------------------------------------------------------------------------------------------------------- #
#: names that would mean the map harness builds (or replays) the full model / the planner pool the C8 wires attach to
MAP_FORBIDDEN = ("RefCV3Model", "build_perception_branch", "declared_vs_built", "plan_wires_flag", "_pwf", "bev_pool", "map_chan_proj", "tac_decoder_v6")


def _names_used(src: str) -> set:
    out = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update(a.name.split(".")[-1] for a in n.names)
            if isinstance(n, ast.ImportFrom) and n.module:
                out.update(n.module.split("."))
    return out


def test_the_map_harness_builds_a_trunk_and_a_branch_by_hand_and_no_full_model():
    used = _names_used(MAP_PATH.read_text(encoding="utf-8"))
    assert {"TimmResNetTrunk", "MapHiresBranch", "MapHiresConfig"} <= used                      # control: the probe sees what the harness does build
    assert sorted(used & set(MAP_FORBIDDEN)) == []


def test_RED_the_map_pin_goes_red_when_the_map_harness_gains_a_full_model_build():
    src = MAP_PATH.read_text(encoding="utf-8") + "\n_x = RefCV3Model(None)\n"
    assert _names_used(src) & set(MAP_FORBIDDEN) == {"RefCV3Model"}


@pytest.mark.parametrize("kind", ["k2m1n20", "k2m3", "k2m3n20"])
def test_the_banded_E2_argvs_reach_the_map_harness_as_build_facts_only_and_C8_moves_none_of_them(kind):
    stem = VARIANTS[kind][0]
    want_own = kind != "k2m1n20"
    want_near = 10.0 if kind == "k2m3" else 20.0
    facts = MH.launch_argv_facts(RUNS / f"{stem}.argv.json")
    assert (facts["banded"], facts["own_enc"]) == (True, want_own)
    assert MH.argv_banded_near_m(facts["argv"]) == want_near
    MH.check_near_vs_argv(want_near, facts)                              # accepted at the argv's own near band
    with pytest.raises(SystemExit):
        MH.check_near_vs_argv(30.0 - want_near, facts)                   # control: a different near band is refused
    harness = {"other_args": {"banded": True, "arms": None, "decision_rule": None, "near_lift_m": want_near, "near_refine_blocks": 1,
                              "x_max_m": 100.0, "y_half_m": 30.0, "grad_ckpt": "on", "device": "cpu", "launch_commit": ""},
               "own_enc_manual": False, "roots": {}}
    blk = MH.identity_block(facts["argv"], harness)
    assert blk["build_facts"] == {"banded": True, "own_enc": want_own}
    bare = [x for x in facts["argv"] if x not in (BOX_FLAG, MAP_FLAG)]       # the same argv WITHOUT the C8 flags
    assert len(bare) == len(facts["argv"]) - 2
    assert MH.identity_block(bare, harness)["identity_sha256"] == blk["identity_sha256"]
    assert (MH.launch_argv_facts(RUNS / f"{stem}.argv.json")["argv"] != bare)
