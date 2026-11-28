"""K2DC: the two REPLAY builds -- the box-overfit harness and the eval loader -- build the map-dose share controller the argv declares, as train() does.

WHY (MEASURED on Thor 2026-11-13 03:50Z, queue 21 ``box_K2DC`` at fd225446310b, ``r9e2/ovf_K2DC_fd225446310b/box.out``, 157 s in): the box harness exited 1
at its own build check:

    [gbo] the replayed build is NOT the declared launch model:
      - --map-hires-share-target: declared 0.1 but BUILT None (read from model._map_dose_ctl) -- the controller declared in argv was never built (the 11,286-parameter defect class)

``train()`` builds the controller with ``_mdc.controller_for(args)`` BEFORE its G-DVB (``refc_v3_train.py:11198`` -> ``model._map_dose_ctl``). The two replays
build the model by hand and never did, so G-DVB refused the K2DC argv in BOTH: the box record could not be bound, and the eval loader (``refcv7_loader.py``,
G-DVB at ``train():8995``'s replay) would refuse the launch gate's G-EVAL and every later read of a K2DC checkpoint the same way.

* (a) the harness replay of a tiny K2DC rig is G-DVB clean, its controller equals ``controller_for`` of the trainer's own parse field for field, and its
  parameters / fresh-init digests are train()'s (the controller adds no parameter, no buffer and draws no RNG);
* (b) OFF (no ``--map-hires-share-target``): the harness and the loader attach nothing, and the harness build is bit-identical to the tip's replay (digests + RNG);
* (c) RED (mutation): with the new lines removed from a COPY, the harness and the loader refuse a K2DC argv with exactly the MEASURED mismatch line;
* the REAL ``r9e2_K2DC.argv.json`` keeps all four controller flags through ``harness_build_argv`` (the harness builds the declared controller, not a stripped one).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

STACK = Path(__file__).resolve().parents[1]
REPO = STACK.parent
for _p in (str(STACK), str(STACK / "scripts"), str(STACK / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG                                           # noqa: E402
import test_g_box_overfit_near_lift as NL                          # noqa: E402  (tiny K0 rig + harness helpers)
import test_refcv7_eval_loader as EL                               # noqa: E402  (its rig BUILDERS only -- not its fixtures)

GBO_PATH = STACK / "scripts" / "g_box_overfit.py"
TRAINER_PATH = STACK / "scripts" / "refc_v3_train.py"
RUNS = STACK / "ops" / "runs.d"
#: the four controller flags (tiny cadence: every 10 rows on a 10-step rig, a multiple of --log-every 5)
DC_FLAGS = ["--map-hires-share-target", "0.10", "--map-hires-share-every", "10", "--map-hires-share-group", "trunk",
            "--map-hires-w-bounds", "1,25", "--log-every", "5"]
REAL_DC = {"--map-hires-share-target": "0.10", "--map-hires-share-every": "250", "--map-hires-share-group": "trunk",
           "--map-hires-w-bounds": "1,25"}
#: what Thor printed (verbatim, after the harness's "  - " bullet)
MEASURED = ("--map-hires-share-target: declared 0.1 but BUILT None (read from model._map_dose_ctl) -- the controller declared in argv was never built "
            "(the 11,286-parameter defect class)")
GBO_LINES = ("        _dose = tr._mdc.controller_for(a)\n"
             "        if _dose is not None:\n"
             "            m._map_dose_ctl = _dose\n")
LDR_LINES = ("    _dose = tr._mdc.controller_for(args)\n"
             "    if _dose is not None:\n"
             "        model._map_dose_ctl = _dose\n"
             "        rec[\"map_dose_ctl\"] = {\"built\": True, \"ckpt_carries_state\": tr._mdc.STATE_KEY in rec[\"state_dict\"][\"ckpt_keys\"]}\n")


def _ctl_fields(c) -> tuple:
    return (c.target, c.every, c.group, c.lo, c.hi, c.w_eff)


def _copy_without(src_path: Path, lines: str, tmp_path: Path, name: str):
    raw = src_path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert raw.count(lines) == 1, f"{src_path.name}: the fix's lines x{raw.count(lines)}"
    p = tmp_path / f"{name}.py"
    p.write_text(raw.replace(lines, ""), encoding="utf-8", newline="\n")
    return p


# ---------------------------------------------------------------------------------------------------------------------- #
# the box harness                                                                                                         #
# ---------------------------------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    d = tmp_path_factory.mktemp("k2dc_gbo_build")
    NL._synth_inputs(d)
    argvs = {"k0": NL._argv(d), "k2dc": NL._argv(d) + DC_FLAGS}
    T = NL._load("refc_v3_train_k2dc_gbo_ref", TRAINER_PATH)
    refs = {k: LG.run_trainer_until(T, v, "model") for k, v in argvs.items()}          # the REFERENCES: the real train()
    G = NL._load("g_box_overfit_k2dc_build_t", GBO_PATH)
    return types.SimpleNamespace(d=d, T=T, G=G, argvs=argvs, ref=refs)


def test_the_harness_replay_builds_the_K2DC_controller_train_declares(rig):
    m, ad = NL._harness_model(rig.G, rig.T, rig.argvs["k2dc"])
    assert ad.dvb_mismatches == []                                                      # G-DVB clean
    want = rig.T._mdc.controller_for(rig.T.build_parser().parse_args(rig.argvs["k2dc"]))
    assert want is not None and _ctl_fields(m._map_dose_ctl) == _ctl_fields(want) == (0.10, 10, "trunk", 1.0, 25.0, 1.0)
    ref = rig.ref["k2dc"]["model"]
    assert sorted(set(ref.state_dict()) ^ set(m.state_dict())) == []                   # the same parameters (the controller adds none)
    dh, dt = LG.state_digests(m.state_dict()), LG.state_digests(ref.state_dict())
    assert [k for k in dh if dh[k] != dt[k]] == []                                      # the same fresh-init digests
    assert float(m._w_map_hires) == 1.0                                                 # the argv's INITIAL weight (the harness never fires the controller)


def test_OFF_the_harness_attaches_nothing_and_is_bit_identical_to_the_tip_replay(rig, tmp_path):
    G0 = NL._load("gbo_tip_replay_k2dc_t", _copy_without(GBO_PATH, GBO_LINES, tmp_path, "gbo_tip_replay_k2dc"))
    m_new, ad_new = NL._harness_model(rig.G, rig.T, rig.argvs["k0"])
    rng_new = torch.get_rng_state().clone()
    m_old, ad_old = NL._harness_model(G0, rig.T, rig.argvs["k0"])
    rng_old = torch.get_rng_state().clone()
    assert ad_new.dvb_mismatches == ad_old.dvb_mismatches == []
    assert not hasattr(m_new, "_map_dose_ctl") and not hasattr(m_old, "_map_dose_ctl")
    dn, do_ = LG.state_digests(m_new.state_dict()), LG.state_digests(m_old.state_dict())
    assert sorted(dn) == sorted(do_) and [k for k in dn if dn[k] != do_[k]] == []
    assert torch.equal(rng_new, rng_old)


def test_RED_without_the_new_lines_the_harness_refuses_K2DC_with_the_MEASURED_line(rig, tmp_path):
    G0 = NL._load("gbo_tip_replay_k2dc_red_t", _copy_without(GBO_PATH, GBO_LINES, tmp_path, "gbo_tip_replay_k2dc_red"))
    with pytest.raises(SystemExit) as e:
        NL._harness_model(G0, rig.T, rig.argvs["k2dc"])
    msg = str(e.value)
    assert msg.startswith("[gbo] the replayed build is NOT the declared launch model:")
    assert msg.splitlines()[1:] == ["  - " + MEASURED]                                 # exactly the one line Thor printed
    assert NL._harness_model(rig.G, rig.T, rig.argvs["k2dc"])[1].dvb_mismatches == []   # control: the fixed harness builds it clean


def test_the_real_K2DC_argv_keeps_all_four_controller_flags_through_the_harness_build_argv(rig):
    real = list(json.loads((RUNS / "r9e2_K2DC.argv.json").read_text(encoding="utf-8"))["argv"])
    build, _rec = rig.G.harness_build_argv(real)
    for flag, val in REAL_DC.items():
        assert build.count(flag) == 1 and build[build.index(flag) + 1] == val, flag
    k0 = list(json.loads((RUNS / "r9e2_K0.argv.json").read_text(encoding="utf-8"))["argv"])
    b0, _ = rig.G.harness_build_argv(k0)
    assert not (set(b0) & set(REAL_DC))                                                 # and K0's carries none


# ---------------------------------------------------------------------------------------------------------------------- #
# the eval loader                                                                                                         #
# ---------------------------------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def T_ref():
    return EL._load_module("refc_v3_train_k2dc_loader_ref", TRAINER_PATH)


def _loader_arm(T, d: Path, extra: list[str], with_state: bool):
    EL._synth_inputs(d)
    argv = EL._argv(d) + list(extra)
    cap = LG.run_trainer_until(T, LG.set_flag(argv, "--trunk-compile", None), "model")
    mt, at = cap["model"], cap["args"]
    ck = d / "ckpt.pt"
    state = {"model": mt.state_dict(), "step": 0}
    ctl = T._mdc.controller_for(at)
    if with_state and ctl is not None:
        state[T._mdc.STATE_KEY] = ctl.state_dict()                                      # what train()'s checkpoint writers add (ckpt_extra)
    torch.save(state, ck)
    from test_refcv7_loader_aux_stamp import trainer_map_stamp, trainer_perception_stamp
    cfg = EL._record(T, argv, refcv6_perception=trainer_perception_stamp(T, mt, at), map_hires=trainer_map_stamp(mt, at))
    return types.SimpleNamespace(ck=ck, config=cfg, argv=argv, args=at)


@pytest.fixture(scope="module")
def loader_arms(T_ref, tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        for k in ("REFCV7_REPO", "REFCV7_KIT", "REFCV7_REMAP_OVERRIDES", "REFCV6_REMAP_OVERRIDES"):
            mp.delenv(k, raising=False)
        mp.setenv("REFCV6_REPO", str(REPO))
        yield {"c0": _loader_arm(T_ref, tmp_path_factory.mktemp("k2dc_ldr_c0"), [], False),
               "k2dc": _loader_arm(T_ref, tmp_path_factory.mktemp("k2dc_ldr_on"), DC_FLAGS, True)}


def _build(L, arm):
    return L.build_model(arm.config, str(arm.ck), device="cpu", strict=True, remap={})


def test_the_loader_builds_the_K2DC_controller_and_records_the_checkpoint_state(loader_arms, T_ref, monkeypatch):
    monkeypatch.setenv("REFCV6_REPO", str(REPO))
    L = EL._load_module("refcv7_loader_k2dc_under_test", EL.LOADER_PATH)
    m, _c, _a, rec = _build(L, loader_arms["k2dc"])
    assert rec["declared_vs_built"]["mismatches"] == []
    assert rec["map_dose_ctl"] == {"built": True, "ckpt_carries_state": True}
    want = T_ref._mdc.controller_for(loader_arms["k2dc"].args)
    assert _ctl_fields(m._map_dose_ctl) == _ctl_fields(want)


def test_OFF_the_loader_attaches_and_records_nothing(loader_arms, monkeypatch):
    monkeypatch.setenv("REFCV6_REPO", str(REPO))
    L = EL._load_module("refcv7_loader_k2dc_off_under_test", EL.LOADER_PATH)
    m, _c, _a, rec = _build(L, loader_arms["c0"])
    assert rec["declared_vs_built"]["mismatches"] == []
    assert "map_dose_ctl" not in rec and not hasattr(m, "_map_dose_ctl")


def test_RED_without_the_new_lines_the_loader_refuses_K2DC_with_the_MEASURED_line(loader_arms, tmp_path, monkeypatch):
    monkeypatch.setenv("REFCV6_REPO", str(REPO))
    L0 = EL._load_module("refcv7_loader_k2dc_red", _copy_without(EL.LOADER_PATH, LDR_LINES, tmp_path, "refcv7_loader_k2dc_red"))
    with pytest.raises(SystemExit) as e:
        _build(L0, loader_arms["k2dc"])
    assert "G-DVB: the REPLAYED build is not what argv declares (1 mismatch(es))" in str(e.value)
    assert str(e.value).splitlines()[1:] == ["  - " + MEASURED]
    m, _c, _a, rec = _build(L0, loader_arms["c0"])                                       # control: OFF is unaffected by the missing lines
    assert rec["declared_vs_built"]["mismatches"] == []
