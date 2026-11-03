"""G-BOX-OVERFIT builds the banded map head (and the map's own encoder) the argv DECLARES (refcv9 E1 arms M1 / M2a / M2b / M3).

WHY (MEASURED on Thor 2026-11-03 17:21Z, ``ovf_M1_23fbf8f7e2b1/box.out``): ``r9e1_bind.sh box M1`` runs ``g_box_overfit.py`` on
``stack/ops/runs.d/r9e1_M1.argv.json`` (which declares ``--r9-map-banded``) and exited 1 at the harness's own build check:

    [gbo] the replayed build is NOT the declared launch model:
      - r9_map_banded: declared True but BUILT False (read from model._map_hires.banded) -- argv and the build disagree
      - r9_map_banded: declared True but BUILT False (read from model._r9_map_banded) -- argv and the attribute the LOSS reads disagree

``TrainerAdapter._build_model`` replays ``train()``'s 10 cm branch build but never passed ``banded=`` / ``own_enc=`` to ``MapHiresConfig``
(``refc_v3_train.py:9997-9998``) nor set the model attribute the loss reads (``:10001-10002``). M1 / M2a / M2b (and M3) carry the flag, so none of
them could get the ``box`` record their gate needs.

The REFERENCE is the trainer's own ``train()`` on a tiny rig, captured as the launch gate captures it (``launch_gate.run_trainer_until(...,
"model")``, before any data is read); every expectation is that model or a LITERAL.

* the harness replay of a banded / banded+own-enc rig argv IS train()'s model: G-DVB clean, same state_dict keys, same fresh-init digests, the
  attribute the loss reads is set, the band geometry and the own-encoder widths equal the trainer's stamps;
* the OFF path is BIT-IDENTICAL: on a non-banded argv the harness and a COPY with the new lines reverted build the same keys and digests, both
  equal train()'s model, and the attribute is ABSENT (as it is in train());
* RED: drop the new build call (all three lines / each alone) from a COPY of the harness => the mismatch returns BY NAME; a replay that builds the
  banded head over another map extent than the argv declares is refused with its reason;
* ``--identity-only`` (GPU-free, before any build) is the same ``identity_sha256`` for the harness and the copy on the C0 / C1 / B3 / M1 argvs.
"""
from __future__ import annotations

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
import test_g_box_overfit_near_lift as NL                          # noqa: E402  (its tiny rig, inputs, adapter helpers)
import test_g_box_overfit_rig as RG                                # noqa: E402  (the E1 rig argv at tiny size)

GBO_PATH = STACK / "scripts" / "g_box_overfit.py"
RUNS = STACK / "ops" / "runs.d"
AUDIT = STACK / "ops" / "refcv8_box_audit"
#: the three lines of the fix, verbatim. Reverting them from a COPY re-installs the tip's replay (the defect).
CFG_OLD = ('                decision_rule=str(getattr(a, "map_hires_decision_rule", _mhr.DECISION_RULES[0])),\n')
CFG_BANDED = ("                banded=bool(tr._mbf.is_on(a)), own_enc=bool(tr._moe.is_on(a)))\n")
ATTR_LINES = ("            if tr._mbf.is_on(a):", "                setattr(m, tr._mbf.MODEL_ATTR, True)\n")
#: the extent the E1 M arms declare (stack/ops/runs.d/r9e1_M1.argv.json): the banded head tiles 100 m x +-30 m
X_MAX, Y_HALF, NEAR_LIFT = "100", "30", "10"


def _banded_argv(d: Path, *extra) -> list[str]:
    """The E1 rig argv at tiny size (``test_g_box_overfit_rig._rig_argv``) at the M arms' extent (100 x +-30 m, near lift 10 m) + ``extra``."""
    argv = RG._rig_argv(d)
    for flag, val in (("--map-hires-x-max-m", X_MAX), ("--map-hires-y-half-m", Y_HALF), ("--map-hires-near-lift-m", NEAR_LIFT)):
        argv[argv.index(flag) + 1] = val
    return argv + list(extra)


@pytest.fixture(scope="module")
def rigs(tmp_path_factory):
    d = tmp_path_factory.mktemp("gbo_banded")
    NL._synth_inputs(d)
    from tanitad.data.semantic_map_gt_fine import FINE_CLASSES
    from tanitad.models import map_head_hires as H
    (d / "cw.json").write_text(json.dumps(              # the class weights AT the declared extent (the reader refuses another)
        {"schema": H.CLASS_WEIGHT_SCHEMA, "classes": list(FINE_CLASSES), "weights": [1.0, 1.5, 3.0, 2.0, 4.0, 1.2, 2.5, 1.1],
         "dry_run": False, "extent": {"x_max_m": float(X_MAX), "y_half_m": float(Y_HALF)}}), encoding="utf-8")
    argvs = {"banded": _banded_argv(d, "--r9-map-banded"),
             "own_enc": _banded_argv(d, "--r9-map-banded", "--r9-map-own-enc"),
             "off": _banded_argv(d)}
    T = NL._load("refc_v3_train_gbo_banded_ref", STACK / "scripts" / "refc_v3_train.py")
    caps = {k: LG.run_trainer_until(T, v, "model")["model"] for k, v in argvs.items()}     # the REFERENCES: the real train()
    G = NL._load("g_box_overfit_banded_t", GBO_PATH)
    return types.SimpleNamespace(d=d, T=T, G=G, argvs=argvs, ref=caps)


def _mutated(tmp_path: Path, name: str, edits: list[tuple[str, str]]):
    src = GBO_PATH.read_text(encoding="utf-8")
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    p = tmp_path / f"{name}.py"
    p.write_text(src, encoding="utf-8", newline="\n")
    return NL._load(f"{name}_t", p)


def _tip_replay_edits() -> list[tuple[str, str]]:
    """The edits that turn the harness back into the tip's (b: e87a932) replay: the config line restored, the attribute lines removed."""
    src = GBO_PATH.read_text(encoding="utf-8")
    old_cfg = CFG_OLD.rstrip("\n").rstrip(",") + ")\n"
    i = src.index(CFG_OLD)
    j = src.index(CFG_BANDED) + len(CFG_BANDED)
    block = src[i:j]
    assert block.count("\n") == 6 and block.endswith(CFG_BANDED), block         # the call line + 4 comment lines + the 2-keyword line
    a0 = src.index(ATTR_LINES[0])
    a1 = src.index(ATTR_LINES[1]) + len(ATTR_LINES[1])
    attr = src[a0:a1]
    assert attr.count("\n") == 2, attr
    return [(block, old_cfg), (attr, "")]


def _digests(m):
    return LG.state_digests(m.state_dict())


def test_the_argv_levers_are_the_M_arms_levers():
    """The tiny rig carries exactly what M1 / M2a / M2b declare for the map branch (the shipped argvs are the source)."""
    for arm in ("M1", "M2a", "M2b"):
        av = json.loads((RUNS / f"r9e1_{arm}.argv.json").read_text(encoding="utf-8"))["argv"]
        assert "--r9-map-banded" in av and "--r9-map-own-enc" not in av and "--r9-perc-rig" in av, arm
        assert av[av.index("--map-hires-x-max-m") + 1] == X_MAX and av[av.index("--map-hires-y-half-m") + 1] == Y_HALF
        assert av[av.index("--map-hires-near-lift-m") + 1] == NEAR_LIFT


@pytest.mark.parametrize("kind", ["banded", "own_enc"])
def test_the_harness_replay_IS_the_banded_model_train_builds(rigs, kind):
    m, ad = NL._harness_model(rigs.G, rigs.T, rigs.argvs[kind])
    ref = rigs.ref[kind]
    assert ad.dvb_mismatches == []
    assert getattr(m, "_r9_map_banded", False) is True and m._map_hires.banded is not None
    assert m._map_hires.refine.cls is None                             # the 8-way readout is absent under the flag
    assert m._map_hires.cfg == ref._map_hires.cfg
    assert (m._map_hires.cfg.banded, m._map_hires.cfg.own_enc) == (True, kind == "own_enc")
    kt, kh = set(ref.state_dict()), set(m.state_dict())
    assert sorted(kt ^ kh) == []
    assert any(k.startswith("_map_hires.banded.") for k in kh)
    assert any(k.startswith("_map_hires.map_encoder.") for k in kh) == (kind == "own_enc")
    dt, dh = _digests(ref), _digests(m)
    assert [k for k in dt if dt[k] != dh[k]] == []                       # same seed, same build order: the harness's fresh init IS the launch init
    T = rigs.T
    assert T._mbf.stamp(m._map_hires) == T._mbf.stamp(ref._map_hires)    # the band geometry (rows, cols, cells) the stamp reads off the branch
    if kind == "own_enc":
        assert T._moe.stamp(m._map_hires) == T._moe.stamp(ref._map_hires)
        assert m._map_hires.lift.d_out == 128 and m._map_hires.refine.inp[0].in_channels == 128
    assert ad.rig_record["built_roots"] == ["core.encoder", "core.agent_head", "_perception", "_map_hires"]


def test_OFF_path_is_bit_identical_to_the_tip_replay_and_to_train(rigs, tmp_path):
    G0 = _mutated(tmp_path, "gbo_tip_replay", _tip_replay_edits())
    m_new, ad_new = NL._harness_model(rigs.G, rigs.T, rigs.argvs["off"])
    m_old, ad_old = NL._harness_model(G0, rigs.T, rigs.argvs["off"])
    ref = rigs.ref["off"]
    assert ad_new.dvb_mismatches == ad_old.dvb_mismatches == []
    dn, do_, dt = _digests(m_new), _digests(m_old), _digests(ref)
    assert sorted(dn) == sorted(do_) == sorted(dt)
    assert [k for k in dn if dn[k] != do_[k]] == [] and [k for k in dn if dn[k] != dt[k]] == []
    assert m_new._map_hires.cfg == m_old._map_hires.cfg == ref._map_hires.cfg
    assert not hasattr(m_new, "_r9_map_banded") and not hasattr(ref, "_r9_map_banded")      # ABSENT (not False), as in train()
    assert m_new._map_hires.banded is None and getattr(m_new._map_hires, "map_encoder", None) is None
    assert sum(k.startswith("_map_hires.") for k in dn) == 54                                # the refcv8 branch, as the near-lift test pins


def test_RED_without_the_new_build_call_the_mismatch_returns_by_name(rigs, tmp_path):
    """Revert the fix in a COPY (= the tip's replay): the banded argv is refused, with the two MEASURED mismatch lines."""
    G0 = _mutated(tmp_path, "gbo_unfixed", _tip_replay_edits())
    with pytest.raises(SystemExit) as e:
        NL._harness_model(G0, rigs.T, rigs.argvs["banded"])
    msg = str(e.value)
    assert msg.startswith("[gbo] the replayed build is NOT the declared launch model:")
    assert "  - r9_map_banded: declared True but BUILT False (read from model._map_hires.banded) -- argv and the build disagree" in msg
    assert "  - r9_map_banded: declared True but BUILT False (read from model._r9_map_banded) -- argv and the attribute the LOSS reads disagree" in msg
    with pytest.raises(SystemExit) as e2:                                   # M3: own-enc is refused the same way (and the banded head with it)
        NL._harness_model(G0, rigs.T, rigs.argvs["own_enc"])
    assert "r9_map_own_enc: declared True but BUILT False (read from model._map_hires.map_encoder)" in str(e2.value)


def test_RED_each_half_of_the_fix_is_load_bearing(rigs, tmp_path):
    edits = _tip_replay_edits()
    (cfg_block, cfg_old), (attr_block, _none) = edits
    # (1) the config keywords gone, the attribute still set: the HEAD is absent (the attribute alone is not the build)
    G1 = _mutated(tmp_path, "gbo_no_cfg", [(cfg_block, cfg_old)])
    with pytest.raises(SystemExit) as e1:
        NL._harness_model(G1, rigs.T, rigs.argvs["banded"])
    assert "(read from model._map_hires.banded)" in str(e1.value) and "(read from model._r9_map_banded)" not in str(e1.value)
    # (2) the config keywords kept, the attribute gone: the head is built but the attribute the LOSS reads is not (a loss that would hit the
    #     8-way path on a banded branch -- the trainer's `_map_hires_loss` branches on this attribute)
    G2 = _mutated(tmp_path, "gbo_no_attr", [(attr_block, "")])
    with pytest.raises(SystemExit) as e2:
        NL._harness_model(G2, rigs.T, rigs.argvs["banded"])
    assert "(read from model._r9_map_banded)" in str(e2.value) and "(read from model._map_hires.banded)" not in str(e2.value)
    # (3) own_enc alone dropped on the M3 argv: the banded head is built, the own encoder is not
    G3 = _mutated(tmp_path, "gbo_no_own_enc", [(CFG_BANDED, "                banded=bool(tr._mbf.is_on(a)))\n")])
    with pytest.raises(SystemExit) as e3:
        NL._harness_model(G3, rigs.T, rigs.argvs["own_enc"])
    assert "r9_map_own_enc: declared True but BUILT False (read from model._map_hires.map_encoder)" in str(e3.value)
    assert "(read from model._map_hires.banded)" not in str(e3.value)


def test_RED_a_replay_that_builds_the_banded_head_over_another_extent_is_refused_with_its_reason(rigs, tmp_path):
    """The banded head builds (flag honoured) but over 80 m instead of the declared 100 m: the replay is NOT the launch model; G-DVB names the
    extent. (The head's own band geometry follows the config, so the stamp differs from train()'s too -- the difference is real, not cosmetic.)"""
    x_old ="                w_map_hires=float(a.w_map_hires), x_max_m=float(_hext.x_max_m), y_half_m=float(_hext.y_half_m),\n"
    G0 = _mutated(tmp_path, "gbo_other_extent", [(x_old, x_old.replace("x_max_m=float(_hext.x_max_m)", "x_max_m=80.0"))])
    with pytest.raises(SystemExit) as e:
        NL._harness_model(G0, rigs.T, rigs.argvs["banded"])
    msg = str(e.value)
    assert msg.startswith("[gbo] the replayed build is NOT the declared launch model:")
    assert "--map-hires-x-max-m: declared 100.0 but BUILT 80.0 (read from model._map_hires.cfg.x_max_m)" in msg, msg
    assert "--map-hires: declared (1000, 600) but BUILT (800, 600) (read from model._map_hires.out_hw)" in msg, msg
    # G-DVB silenced: the branch is the banded one but its geometry is NOT train()'s -- the reason the check exists
    from tanitad.train import declared_vs_built as dvb
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(dvb, "check", lambda *a, **k: [])
        mp.setattr(rigs.T._r9rig_mod, "assert_rig_built", lambda m: {})
        m0, _ad = NL._harness_model(G0, rigs.T, rigs.argvs["banded"])
    finally:
        mp.undo()
    assert m0._map_hires.banded is not None
    assert rigs.T._mbf.stamp(m0._map_hires)["bands"] != rigs.T._mbf.stamp(rigs.ref["banded"]._map_hires)["bands"]
    assert [(g.rows, g.cols) for g in m0._map_hires.banded.geom] != [(g.rows, g.cols) for g in rigs.ref["banded"]._map_hires.banded.geom]


def test_the_replay_goes_through_the_trainers_own_flag_modules_not_a_reimplementation():
    src = GBO_PATH.read_text(encoding="utf-8")
    for needle in ("banded=bool(tr._mbf.is_on(a))", "own_enc=bool(tr._moe.is_on(a))", "setattr(m, tr._mbf.MODEL_ATTR, True)"):
        assert src.count(needle) == 1, needle
    for banned in ("BandedMapHead", "import map_banded", "_build_own_enc", "_r9_map_banded"):     # the trainer's modules own these
        assert banned not in src, banned
    tsrc = (STACK / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    assert "banded=bool(_mbf.is_on(args))," in tsrc and "own_enc=bool(_moe.is_on(args)))" in tsrc      # the trainer lines the replay mirrors
    assert "setattr(model, _mbf.MODEL_ATTR, True)" in tsrc


@pytest.mark.parametrize("arm", ["C0", "C1", "B3", "M1", "M2a"])
def test_identity_only_sha_is_unchanged_by_the_fix(arm, tmp_path):
    """``--identity-only`` exits BEFORE any build: the change cannot move ``identity_sha256``. Measured against a COPY with the fix reverted."""
    G1 = NL._load("g_box_overfit_banded_id_t", GBO_PATH)
    G0 = _mutated(tmp_path, "gbo_id_tip", _tip_replay_edits())
    shas = []
    for tag, G in (("hold", G1), ("tip", G0)):
        out = tmp_path / f"{arm}.{tag}.json"
        toks = ["--launch-argv", str(RUNS / f"r9e1_{arm}.argv.json"), "--audit-dir", str(AUDIT), "--out", str(tmp_path / "rec.json"),
                "--device", "cuda", "--binding", "--commit", "ab" * 20, "--candidate", "x", "--arms", "main", "--identity-only", str(out)]
        assert G.main(toks) == 0
        shas.append(json.loads(out.read_text(encoding="utf-8"))["identity_sha256"])
    assert shas[0] == shas[1] and len(shas[0]) == 64
