"""refcv8 WP-C I1-I3 -- the TRAINER wiring of the three OPT-IN perception fixes (WP-B owns ``refc_v3_train.py``).

Source: ``TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-fixes/INTEGRATION_WPC.md``.
The modules are pinned on their own side (``test_refcv8_det_nms.py``, ``test_refcv8_det_zh_range.py``,
``test_refcv8_join_label_hygiene.py``); this file pins that the trainer REACHES them, and that a run launched with the
OLD argv is exactly what it was:

1. ``--det-nms`` (I1), ``--det-zh-trust`` (I2), ``--join-defect-masks`` (I3) exist and default to ``None``; a default
   argv reads none of the three files.
2. every refusal, each with a GREEN control: a file read by nothing (I1 / I2 without VIS-1, I2 without a box head or the
   3-D join, I3 without the agent join), a missing file, a malformed file.
3. the eval-row helper ``_wpc_eval_keys``: ``{}`` by default; with the shipped NMS file the 26 ``eval_<head>_nms_*`` keys
   (box3d radius 2.5 m, agent 3.0 m), with the trust file the ``eval_box3d_zh_*`` keys on box3d only; no declared key
   changes and no pack is mutated.
4. SOURCE pins for the call sites that cannot run in a unit test (the in-run eval loop, the TRAIN join reader) -- each
   with a DELIBERATE REGRESSION that must go red.
5. THROUGH THE REAL ``train()`` (the launch gate's capture, ``launch_gate.run_trainer_until``): all three flags build a
   model whose ``state_dict`` is BIT-IDENTICAL to the default argv's and carries ``_det_zh_range``; the EVAL LOADER
   (``refcv7_loader``) rebuilds it strictly, with the same carrier.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from test_map_hires_wiring import T                                                    # noqa: E402  (the trainer)
from test_refcv7_eval_loader import _record, rig                                       # noqa: E402,F401  (the real-train() rig)
from test_refcv8_det_nms import EVAL_FIX, _packs as _fixture_packs                    # noqa: E402
from test_refcv8_det_zh_range import _packs as _zh_scene_packs                         # noqa: E402
import launch_gate as LG                                                              # noqa: E402  (scripts/ is on sys.path by now)
from tanitad.data import join_label_hygiene as JLH                                     # noqa: E402
from tanitad.eval import detection_metrics as det                                      # noqa: E402
from tanitad.eval import detection_nms as N                                            # noqa: E402
from tanitad.eval import detection_zh as Z                                             # noqa: E402

STACK = Path(__file__).resolve().parents[1]
CONFIGS = STACK / "tanitad" / "configs"
NMS = CONFIGS / "refcv7_det_nms_train.json"
TRUST = CONFIGS / "refcv8_box_zh_range_trust.json"
MASKS = CONFIGS / "refcv8_join_label_defects.json"
GATES = CONFIGS / "refcv7_det_presence_gates_train.json"
BASE = ["--arm", "hier", "--out", "X"]
VIS = ["--slot-vis1"]
ZH_OK = ["--slot-vis1", "--w-box3d", "1.0", "--join3d", "J3"]
FLAGS = ("--det-nms", "--det-zh-trust", "--join-defect-masks")


def _ns(argv):
    return T.build_parser().parse_args(argv)


def _action(parser, flag):
    return [a for a in parser._actions if flag in a.option_strings][0]


# =========================================================================== #
# 1. the flags                                                                 #
# =========================================================================== #
def test_the_flags_exist_default_to_None_and_have_clean_help():
    p = T.build_parser()
    for flag in FLAGS:
        a = _action(p, flag)
        assert a.default is None, flag
        assert "%" not in (a.help or "").replace("%%", ""), flag                       # argparse help hygiene
    ns = _ns(BASE)
    assert (ns.det_nms, ns.det_zh_trust, ns.join_defect_masks) == (None, None, None)


def test_a_default_argv_reads_NONE_of_the_three_files(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("a default run read a WP-C file")
    monkeypatch.setattr(N, "load_head_nms", boom)
    monkeypatch.setattr(Z, "load_zh_trust", boom)
    monkeypatch.setattr(JLH.JoinDefectMasks, "load", classmethod(lambda cls, p: boom()))
    T._pin_wpc_fixes(_ns(BASE))
    T._pin_wpc_fixes(_ns(BASE + ZH_OK + ["--agent-join", "J"]))                         # prerequisites alone: nothing read
    assert T._wpc_eval_keys([{"logit": None}], "box3d") == {}                          # no cfg -> no computation at all


# =========================================================================== #
# 2. the pins                                                                  #
# =========================================================================== #
def test_GREEN_each_flag_with_its_prerequisites_passes():
    T._pin_wpc_fixes(_ns(BASE + VIS + ["--det-nms", str(NMS)]))
    T._pin_wpc_fixes(_ns(BASE + ZH_OK + ["--det-zh-trust", str(TRUST)]))
    T._pin_wpc_fixes(_ns(BASE + ["--agent-join", "J", "--join-defect-masks", str(MASKS)]))
    T._pin_wpc_fixes(_ns(BASE + ZH_OK + ["--agent-join", "J", "--det-nms", str(NMS), "--det-zh-trust", str(TRUST),
                                         "--join-defect-masks", str(MASKS)]))


@pytest.mark.parametrize("argv,needle", [
    (BASE + ["--det-nms", str(NMS)], "--det-nms without --slot-vis1"),
    (BASE + VIS + ["--det-nms", "no_such_nms.json"], "does not exist"),
    (BASE + ["--det-zh-trust", str(TRUST)], "without --slot-vis1 / --w-box3d > 0 / --join3d"),
    (BASE + ["--slot-vis1", "--w-box3d", "1.0", "--det-zh-trust", str(TRUST)], "without --join3d"),
    (BASE + ["--slot-vis1", "--join3d", "J3", "--det-zh-trust", str(TRUST)], "without --w-box3d > 0"),
    (BASE + ["--w-box3d", "1.0", "--join3d", "J3", "--det-zh-trust", str(TRUST)], "without --slot-vis1:"),
    (BASE + ZH_OK + ["--det-zh-trust", "no_such_trust.json"], "does not exist"),
    (BASE + ["--join-defect-masks", str(MASKS)], "--join-defect-masks without --agent-join"),
    (BASE + ["--agent-join", "J", "--join-defect-masks", "no_such_masks.json"], "--join-defect-masks"),
])
def test_every_dead_or_unreadable_combination_refuses(argv, needle):
    with pytest.raises(SystemExit) as e:
        T._pin_wpc_fixes(_ns(argv))
    assert needle in str(e.value), str(e.value)[:300]


@pytest.mark.parametrize("flag,src,pre", [
    ("--det-nms", NMS, VIS),
    ("--det-zh-trust", TRUST, ZH_OK),
    ("--join-defect-masks", MASKS, ["--agent-join", "J"]),
])
def test_a_file_of_another_schema_refuses_at_the_pin(tmp_path, flag, src, pre):
    d = json.loads(src.read_text(encoding="utf-8"))
    d["schema"] = "other/1"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        T._pin_wpc_fixes(_ns(BASE + pre + [flag, str(p)]))
    assert flag in str(e.value) and "schema" in str(e.value), str(e.value)[:300]


def test_the_validator_runs_inside_the_pin_chain_before_the_slot_refine_early_return(monkeypatch):
    """``_pin_slot_refine`` returns early when no slot refine is active; the WP-C validator must sit BEFORE that return
    (like the F4 gates), or a run without slot refine would skip the refusals."""
    with pytest.raises(SystemExit) as e:
        T._pin_slot_refine(None, _ns(BASE + ["--det-nms", str(NMS)]))
    assert "--det-nms without --slot-vis1" in str(e.value)
    # GREEN control: with its prerequisite the validator passes and the chain reaches the early return
    monkeypatch.setattr(T, "_slot_refine_active", lambda _a: False)
    assert T._pin_slot_refine(None, _ns(BASE + VIS + ["--det-nms", str(NMS)])) is None
    with pytest.raises(SystemExit) as e:                                               # still refused on that path
        T._pin_slot_refine(None, _ns(BASE + ["--join-defect-masks", str(MASKS)]))
    assert "--join-defect-masks without --agent-join" in str(e.value)


# =========================================================================== #
# 3. the eval-row helper                                                       #
# =========================================================================== #
def _digest_packs(packs) -> str:
    h = hashlib.sha256()
    for pk in packs:
        for k in sorted(pk):
            a = np.asarray(pk[k]) if pk[k] is not None else np.asarray([])
            h.update(repr((k, a.dtype.str, a.shape)).encode()
                     + (b"" if a.dtype == object else np.ascontiguousarray(a).tobytes()))
    return h.hexdigest()


def test_with_the_shipped_nms_file_the_eval_row_gains_EXACTLY_the_26_nms_keys_and_changes_no_declared_one():
    cfg, _st = N.load_head_nms(NMS)
    gates, _gs = det.load_head_gates(GATES)
    new_keys = set()
    for hd in det.HEADS:
        packs = _fixture_packs(EVAL_FIX, hd)
        before = _digest_packs(packs)
        # the eval loop's row for this head, exactly as the tip writes it (declared summary + F4 gated census)
        erow = {}
        for k, v in {**det.summarise(packs, hd), **det.gated_census_keys(packs, hd, gates)}.items():
            erow[k] = None if (isinstance(v, float) and v != v) else round(float(v), 5)
        tip_row = dict(erow)
        got = T._wpc_eval_keys(packs, hd, cfg, None)
        assert set(got) == set(N.nms_key_names(hd)) and len(got) == 13, hd
        assert not set(got) & set(tip_row), hd                                         # NEW keys only
        for k, v in got.items():                                                       # the loop's merge
            erow[k] = None if (isinstance(v, float) and v != v) else round(float(v), 5)
        assert {k: erow[k] for k in tip_row} == tip_row, hd                            # declared keys key-for-key
        assert _digest_packs(packs) == before, hd                                      # packs untouched
        assert got == N.nms_census_keys(packs, hd, cfg), hd                            # the helper IS the module
        new_keys |= set(got)
        if hd == "box3d":
            assert got["eval_box3d_nms_radius_m"] == 2.5 and erow["eval_box3d_nms_radius_m"] == 2.5
        else:
            assert got["eval_agent_nms_radius_m"] == 3.0
        assert got[f"eval_{hd}_nms_gate"] == cfg[hd]["gate"]                           # read at the file's OWN gate
    assert len(new_keys) == 26


def test_the_zh_keys_appear_on_box3d_only_and_equal_the_module():
    trust, _st = Z.load_zh_trust(TRUST)
    packs = _zh_scene_packs(with_zh_range=True)
    got = T._wpc_eval_keys(packs, "box3d", None, trust)
    assert set(got) == set(Z.zh_key_names("box3d"))
    assert got == Z.zh_range_keys(packs, "box3d", trust)
    assert got["eval_box3d_zh_nearfield_range_m"] == 30.0                              # derived by the loader
    assert T._wpc_eval_keys(packs, "agent", None, trust) == {}                         # no cz / h on the agent head
    assert T._wpc_eval_keys(packs, "box3d", None, None) == {}


# =========================================================================== #
# 4. SOURCE pins (the in-run eval loop and the TRAIN join reader cannot run here)  #
# =========================================================================== #
def _src(rel: str) -> str:
    return (STACK / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def _func(src: str, name: str) -> str:
    a = src.index(f"\ndef {name}(")
    b = src.index("\ndef ", a + 1)
    return src[a:b]


def _wpc_pin(src: str, loader: str) -> None:
    # I1 / I2: the eval loop calls the helper right after the F4 gated census, in the same loop
    a = src.find("_det_metrics.gated_census_keys(_pk, _hd, det_gates)")
    b = src.find("_wpc_eval_keys(_pk, _hd, det_nms_cfg, det_zh_cfg)")
    assert a > 0 and b > 0 and 0 < b - a < 900, (a, b)
    assert "det_nms_cfg = None" in src and "det_zh_cfg = None" in src                 # a default run: `{}`
    assert "det_nms_cfg, _dn_stamp = _det_nms.load_head_nms(args.det_nms)" in src
    assert 'calib_stamp["det_nms"] = _dn_stamp' in src
    assert "det_zh_cfg, _dz_stamp = _det_zh.load_zh_trust(args.det_zh_trust)" in src
    assert 'calib_stamp["det_zh_trust"] = _dz_stamp' in src
    # I2: the carrier, and the box3d pack gains the range in EVAL mode only
    assert 'model._det_zh_range = bool(getattr(args, "det_zh_trust", None))' in src
    assert 'model._det_zh_range = bool(getattr(args, "det_zh_trust", None))' in loader
    cl = _func(src, "compute_losses_v3")
    assert re.search(r'with_zh_range=\(bool\(getattr\(model, "_det_zh_range", False\)\)\s+and not model\.training\)',
                     cl), "the box3d window_packs call must carry the range in EVAL mode only"
    # the planner / loss path never reaches the NMS or the z/h report
    # (module ACCESS, `_det_zh.`: the `model._det_zh_range` carrier above is the one allowed mention)
    assert "_det_nms." not in cl and "_det_zh." not in cl and "_wpc_eval_keys" not in cl
    # I3: the TRAIN reader is masked, the EVAL readers are not
    tr = src[src.index("_rd = JoinFileReader("):src.index('print(f"[v3] agent join loaded')]
    assert "defect_masks=(_jlh.JoinDefectMasks.load(args.join_defect_masks)" in tr
    ev = src[src.index("_e_rd = _JFR("):src.index("eval_agent_stats = e_ds.enable_agent_join(")]
    assert ev and "defect_masks" not in ev
    lev = loader[loader.index("_JFR("):]
    assert "defect_masks" not in lev[:lev.index("\n\n")]
    assert 'agent_stats["defect_masks"] = _rd.defect_stats()' in src
    assert "if _rd.defect_masks is not None:" in src                                  # absent from a default stamp
    # the validator is called in the pin chain BEFORE the slot-refine early return
    ps = _func(src, "_pin_slot_refine")
    assert 0 < ps.index("_pin_wpc_fixes(args)") < ps.index("if not _slot_refine_active(args):")


def test_the_call_sites_are_wired_as_INTEGRATION_WPC_specifies():
    _wpc_pin(_src("scripts/refc_v3_train.py"), _src("tanitad/eval/refcv7_loader.py"))


@pytest.mark.parametrize("old,new,where", [
    # the z/h range leaks into TRAINING packs
    ("                                       and not model.training))", "                                       ))", "src"),
    # the EVAL reader gets the mask too (eval comparability with refcv7 silently lost)
    ("_e_rd = _JFR(args.agent_join,", "_e_rd = _JFR(args.agent_join, defect_masks=None,", "src"),
    # the eval row stops calling the new census
    ("_wpc_eval_keys(_pk, _hd, det_nms_cfg, det_zh_cfg)", "{}.get(_pk, {_hd: det_nms_cfg})", "src"),
    # the loader forgets the carrier
    ('model._det_zh_range = bool(getattr(args, "det_zh_trust", None))', "pass", "loader"),
    # the validator moves after the early return
    ("    _pin_wpc_fixes(args)                 # refcv8 WP-C I1-I3", "    pass  # moved", "src"),
])
def test_DELIBERATE_REGRESSION_each_wiring_defect_goes_RED(old, new, where):
    src, loader = _src("scripts/refc_v3_train.py"), _src("tanitad/eval/refcv7_loader.py")
    if where == "src":
        assert old in src, old
        src = src.replace(old, new, 1)
    else:
        assert old in loader, old
        loader = loader.replace(old, new, 1)
    with pytest.raises((AssertionError, ValueError)):
        _wpc_pin(src, loader)


def test_DELIBERATE_REGRESSION_the_nms_reaching_the_loss_path_goes_RED():
    src, loader = _src("scripts/refc_v3_train.py"), _src("tanitad/eval/refcv7_loader.py")
    old = 'extra.update(_det_metrics.train_row_keys(_pk_b3, "box3d"))'
    assert old in src
    with pytest.raises(AssertionError):
        _wpc_pin(src.replace(old, old + "\n                    extra.update(_det_nms.nms_census_keys([_pk_b3], "
                             "\"box3d\", None))", 1), loader)


# =========================================================================== #
# 5. through the REAL train() and the eval loader                              #
# =========================================================================== #
def _all_three(argv):
    a = LG.set_flag(argv, "--det-nms", [str(NMS)])
    a = LG.set_flag(a, "--det-zh-trust", [str(TRUST)])
    return LG.set_flag(a, "--join-defect-masks", [str(MASKS)])


def test_the_default_reference_model_carries_no_WPC_state(rig):
    assert (rig.args_t.det_nms, rig.args_t.det_zh_trust, rig.args_t.join_defect_masks) == (None, None, None)
    assert rig.model_t._det_zh_range is False


def test_through_the_real_train_det_nms_without_vis1_refuses(rig):
    a = LG.set_flag(rig.argv, "--trunk-compile", None)
    no_vis = LG.set_flag(LG.set_flag(a, "--slot-vis1", None), "--vis1-sidecar", None)
    with pytest.raises(SystemExit) as e:
        LG.run_trainer_until(rig.T, LG.set_flag(no_vis, "--det-nms", [str(NMS)]), "config")
    assert "--det-nms without --slot-vis1" in str(e.value)


def test_through_the_real_train_all_three_build_the_SAME_model_plus_the_carrier(rig):
    cap = LG.run_trainer_until(rig.T, _all_three(LG.set_flag(rig.argv, "--trunk-compile", None)), "model")
    m = cap["model"]
    assert m._det_zh_range is True
    assert (cap["args"].det_nms, cap["args"].det_zh_trust, cap["args"].join_defect_masks) == (
        str(NMS), str(TRUST), str(MASKS))
    ref, got = rig.model_t.state_dict(), m.state_dict()
    assert list(ref) == list(got)                                                      # no parameter / buffer added
    bad = [k for k in ref if torch.is_tensor(ref[k]) and not torch.equal(ref[k], got[k])]
    assert not bad, bad[:5]                                                            # and no RNG consumed


def test_the_eval_loader_rebuilds_the_opt_in_model_strictly_with_the_carrier(rig):
    argv = _all_three(list(rig.argv))
    model, _cfg, args, rec = rig.L.build_model(_record(rig.T, argv), str(rig.ck), device="cpu", strict=True,
                                               remap={})
    assert rec["declared_vs_built"]["mismatches"] == []
    assert model._det_zh_range is True and args.det_nms == str(NMS)
    m0, _c, _a, r0 = rig.L.build_model(_record(rig.T, rig.argv), str(rig.ck), device="cpu", strict=True, remap={})
    assert m0._det_zh_range is False and r0["declared_vs_built"]["mismatches"] == []
