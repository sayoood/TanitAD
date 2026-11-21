"""refcv9 GEVAL-AUXSTAMP -- the eval loader's rebuilt `refcv6_perception.branch_params` is taken at the TRAINER's build point.

DEFECT (MEASURED on Thor 2026-11-04T07:55:34Z, arm B1's launch gate, G-EVAL FAIL): `config.json {... 'total': 4797135}` vs rebuilt `{... 'total': 4284463}`
-- 512,672 short = B1's training-only depth head. train() stamps `model._perception.param_breakdown()` AFTER `_bbf/_pwf/_bda/_p3f/_bc5.attach`; the loader's
`build_perception_block` stamped it after only `_pwf.attach`, and `_bda / _bc5 / _p3f` ran later. FIX: `refcv7_loader.refresh_perception_stamp(model, rec)`, called
after the last attach.

REFERENCE = the trainer's own `train()` (`launch_gate.run_trainer_until(T, argv, "model")`: the real main -> train(), captured at its first data-source call, i.e.
AFTER every attach). The stamp the trainer writes is read OFF that model with the trainer's own expressions (train() ~:10145-10170); the loader contributes nothing to
its own expectation. Arms: C0 (no aux), B1 (--r9-box-depth-aux), B2 (--r9-box-persp-aux), B2c (--r9-box-bev2-aux), B4-style (persp aux + --r9-box-layers 6 + s8 + pfq).

RED arms: (1) the refresh turned into a no-op -> every aux arm is REFUSED on `refcv6_perception.branch_params` (by exactly the aux parameters) and C0 still builds;
(2) the refresh call deleted from the loader source / a later attach added after it -> the source-order guard goes RED.
"""
from __future__ import annotations

import inspect
import re
import sys
import types
from pathlib import Path

import pytest
import torch

STACK = Path(__file__).resolve().parents[1]
REPO = STACK.parent
for _p in (str(STACK), str(STACK / "scripts"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG                                           # noqa: E402
import test_refcv7_eval_loader as EL                               # noqa: E402  (its rig BUILDERS only -- not its fixtures)

#: arm -> (extra argv, state_dict prefixes the arm adds under `_perception.`)
ARMS = ("C0", "B1", "B2", "B2c", "B4", "B3")
AUX_PREFIX = {"C0": (), "B1": ("box_depth_aux.",), "B2": ("box_persp_aux.",), "B2c": ("box_bev2_aux.",),
              "B4": ("box_persp_aux.", "box_pfq.", "box_qpos."), "B3": ("box_dec.dn_embed.", "box_dec.dn_cls_embed.")}


def _arm_argv(d: Path, name: str) -> list[str]:
    a = EL._argv(d)
    if name == "B4":                                               # pfq refuses a second anchor supplier: `learned`, not the rig's `learned_ref`
        a = LG.set_flag(a, "--slot-query-select", ["learned"])
    extra = {"C0": [], "B1": ["--r9-box-depth-aux", "--w-box-depth", "1.0"],
             "B2": ["--r9-box-persp-aux", "--w-box-persp", "1.0"],
             "B2c": ["--r9-box-bev2-aux", "--w-box-bev2", "1.0"],
             "B4": ["--r9-box-persp-aux", "--w-box-persp", "1.0", "--r9-box-layers", "6", "--r9-box-s8-horizon",
                    "--r9-box-perframe-queries"],
             "B3": ["--r9-box-quality", "--r9-box-dn", "--r9-box-inloss-nms"]}[name]      # MM: DN is the B3 lever that adds parameters
    return a + extra


def trainer_perception_stamp(T, m, a) -> dict:
    """train() ~:10145-10170, applied to the model train() built: the keys `stamp_checks` can compare."""
    br = m._perception
    fr = T._perc.frame_for_model(m)
    st = {**br.cfg.as_dict(), "equalize_bottom_rows": int(getattr(a, "equalize_bottom_rows", 0) or 0),
          "branch_params": br.param_breakdown(),
          "fmap_s16_channels": int(m.core.encoder.s16_dim), "fmap_s16_hw": list(m.core.encoder.s16_shape),
          "frame": {"height": int(fr.height), "width": int(fr.width), "projection": str(fr.projection), "f_ref": float(fr.f_ref)}}
    pw = T._pwf.stamp(m, a)
    if pw is not None:
        st["plan_wires"] = pw
    return st


def trainer_map_stamp(m, a) -> dict:
    """train() ~:10050-10066 (read BEFORE `_bw.attach`, which is absent in these arms)."""
    return {**m._map_hires.cfg.as_dict(), "branch_params": m._map_hires.param_breakdown(),
            "class_weights": {"sha256": m._map_hires_class_weight_stamp["sha256"]}, "bev_source": str(a.bev_source),
            "planner_crop_m": [float(v) for v in a.bev_planner_crop_m], "lift_valid_mask": bool(m._map_lift_valid_mask)}


def _n(m, prefixes) -> int:
    return sum(p.numel() for k, p in m.state_dict().items() if k.startswith(tuple("_perception." + x for x in prefixes)))


@pytest.fixture(scope="module")
def T():
    return EL._load_module("refc_v3_train_geval_auxstamp_ref", STACK / "scripts" / "refc_v3_train.py")


@pytest.fixture(scope="module")
def L():
    return EL._load_module("refcv7_loader_geval_auxstamp_under_test", EL.LOADER_PATH)


@pytest.fixture(scope="module", params=ARMS)
def arm(request, T, tmp_path_factory):
    name = request.param
    d = tmp_path_factory.mktemp("geval_auxstamp_" + name)
    EL._synth_inputs(d)
    argv = _arm_argv(d, name)
    with pytest.MonkeyPatch.context() as mp:
        for k in ("REFCV7_REPO", "REFCV7_KIT", "REFCV7_REMAP_OVERRIDES", "REFCV6_REMAP_OVERRIDES"):
            mp.delenv(k, raising=False)
        mp.setenv("REFCV6_REPO", str(REPO))
        cap = LG.run_trainer_until(T, LG.set_flag(argv, "--trunk-compile", None), "model")
        mt, at = cap["model"], cap["args"]
        ck = d / "ckpt.pt"
        torch.save({"model": mt.state_dict(), "step": 0}, ck)
        cfg = EL._record(T, argv, refcv6_perception=trainer_perception_stamp(T, mt, at), map_hires=trainer_map_stamp(mt, at))
        c5 = T._bc5.stamp(mt, at)
        if c5 is not None:
            cfg[T._bc5.STAMP_KEY] = c5
        yield types.SimpleNamespace(name=name, mt=mt, at=at, ck=ck, config=cfg, argv=argv)


def _build(L, arm):
    return L.build_model(arm.config, str(arm.ck), device="cpu", strict=True, remap={})


# --------------------------------------------------------------------------------------------- #
# GREEN: the rebuilt stamp IS the trainer's, and the strict load reads every aux key             #
# --------------------------------------------------------------------------------------------- #
def test_the_rebuilt_stamp_equals_the_trainers_and_the_strict_load_reads_the_aux_heads(arm, L, T, monkeypatch):
    spy = {}
    orig = L.refresh_perception_stamp

    def watch(model, rec):                                         # measure the STALE value the pre-fix loader would have compared
        spy["stale_total"] = int(rec["perception"]["branch_params"]["total"])
        orig(model, rec)
        spy["fresh_total"] = int(rec["perception"]["branch_params"]["total"])
    monkeypatch.setattr(L, "refresh_perception_stamp", watch)
    ml, _cfg, al, rec = _build(L, arm)
    mt = arm.mt
    sd = rec["state_dict"]
    assert sd["missing"] == [] and sd["unexpected"] == []                                              # the strict load (deliverable 3)
    dt, dl = LG.state_digests(mt.state_dict()), LG.state_digests(ml.state_dict())
    assert set(dt) == set(dl) and [k for k in dt if dt[k] != dl[k]] == []
    n_aux = _n(mt, AUX_PREFIX[arm.name])
    assert (n_aux > 0) == (arm.name != "C0") and _n(ml, AUX_PREFIX[arm.name]) == n_aux
    sc = rec["stamp_checks"]
    assert sc["refcv6_perception.branch_params"]["equal"] is True
    assert sc["map_hires.branch_params"]["equal"] is True and sc["map_hires.config"]["equal"] is True
    assert sc["n_compared"] > 8
    tot = int(mt._perception.param_breakdown()["total"])
    assert spy["fresh_total"] == tot == int(ml._perception.param_breakdown()["total"])
    if arm.name in ("C0", "B3"):
        assert spy["stale_total"] == tot                                     # nothing to refresh (B3 attaches INSIDE the block, before its stamp, as train())
    else:                                                                                              # the defect, measured: stale by the aux params
        assert spy["stale_total"] < tot
        if arm.name in ("B1", "B2", "B2c"):
            assert tot - spy["stale_total"] == n_aux
    if arm.name == "B4":
        assert len(ml._perception.box_dec.blocks.layers) == 6 == len(mt._perception.box_dec.blocks.layers)
    # CENSUS: every perception field the block stamped equals the same field re-read off the FINAL loader model (no other field is order-stale)
    again = trainer_perception_stamp(T, ml, al)
    assert {k: rec["perception"][k] for k in rec["perception"] if k != "lift_bank_n_clips"} == \
           {k: again[k] for k in rec["perception"] if k != "lift_bank_n_clips"}


# --------------------------------------------------------------------------------------------- #
# RED 1: without the refresh every aux arm is REFUSED on branch_params; C0 still builds          #
# --------------------------------------------------------------------------------------------- #
def test_RED_without_the_refresh_the_aux_arms_CONTRADICT_on_branch_params_and_C0_still_passes(arm, L, monkeypatch):
    monkeypatch.setattr(L, "refresh_perception_stamp", lambda model, rec: None)
    if arm.name in ("C0", "B3"):
        _m, _c, _a, rec = _build(L, arm)
        assert rec["stamp_checks"]["refcv6_perception.branch_params"]["equal"] is True
        return
    with pytest.raises(SystemExit) as e:
        _build(L, arm)
    msg = str(e.value)
    assert "config.json CONTRADICTS the rebuilt model" in msg and "refcv6_perception.branch_params" in msg


# --------------------------------------------------------------------------------------------- #
# source-order guards (trainer stamps AFTER; loader refreshes AFTER the last attach)             #
# --------------------------------------------------------------------------------------------- #
def _loader_order_problems(src: str) -> list[str]:
    """`_build_model`'s source: the refresh must follow EVERY `tr._xxx.attach(` after build_perception_block and precede stamp_checks."""
    head = src.index("build_perception_block(tr, model, args, device)")
    tail = src.index("stamp_checks(model, config, rec, tr)")
    bad = []
    mr = re.search(r"^[ \t]*refresh_perception_stamp\(model, rec\)", src, flags=re.M)
    if mr is None:
        return ["no refresh_perception_stamp(model, rec) call in _build_model"]
    if not head < mr.start() < tail:
        bad.append("the refresh is not between build_perception_block and stamp_checks")
    for m in re.finditer(r"^[ \t]*tr\._\w+\.attach\(", src, flags=re.M):
        if head < m.start() < tail and m.start() > mr.start():
            bad.append("an attach runs AFTER the refresh: " + src[m.start():m.start() + 40].strip())
    return bad


def test_the_loader_refreshes_after_the_last_attach_and_deleting_the_call_or_a_later_attach_goes_RED(L):
    src = inspect.getsource(L._build_model)
    assert _loader_order_problems(src) == []
    assert len(re.findall(r"^[ \t]*tr\._(?:bda|bc5|p3f|bw)\.attach\(", src, flags=re.M)) == 4          # the four attaches the refresh follows
    no_call = re.sub(r"^[ \t]*refresh_perception_stamp\(model, rec\)[^\n]*\n", "", src, flags=re.M)
    assert _loader_order_problems(no_call) == ["no refresh_perception_stamp(model, rec) call in _build_model"]
    late = src.replace("    refresh_perception_stamp(model, rec)\n",
                       "    refresh_perception_stamp(model, rec)\n    tr._zzz.attach(model, args)\n")
    assert any("AFTER the refresh" in p for p in _loader_order_problems(late))


def test_the_trainer_stamps_perception_AFTER_every_attach_and_the_map_branch_BEFORE_bw_attach():
    src = (STACK / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    t0 = src.index("def train(")
    seg_start = src.index("model._perception = _perc.build_perception_branch(model, _pcfg)", t0)
    seg_end = src.index('"branch_params": model._perception.param_breakdown()', seg_start)
    seg = src[seg_start:seg_end]
    order = [seg.find(f"{n}.attach(model, args)") for n in ("_bbf", "_pwf", "_bda", "_p3f", "_bc5")]
    assert all(i >= 0 for i in order) and order == sorted(order)                # all five run, in this order, before the stamp
    mh = src.index('"branch_params": model._map_hires.param_breakdown()', t0)
    assert mh < src.index("_bw.attach(model, args)", t0)                      # map branch: stamped before the fuse attach (loader: see below)


def test_the_loader_map_block_is_taken_before_bw_attach_like_the_trainers(L):
    assert "attach(" not in inspect.getsource(L.build_map_hires_block)
    src = inspect.getsource(L._build_model)
    assert src.index("build_map_hires_block(tr, model, args, device)") < src.index("tr._bw.attach(model, args)")


# --------------------------------------------------------------------------------------------- #
# RED 3 (MM): without the loader's `_bbf.attach` replay a B3-DN checkpoint is REFUSED by the strict load #
# --------------------------------------------------------------------------------------------- #
def test_RED_without_the_bbf_attach_a_B3_DN_checkpoint_is_refused_and_C0_is_unaffected(arm, L, monkeypatch):
    if arm.name not in ("C0", "B3"):
        pytest.skip("the B3 attach is exercised on C0 (inert) and B3 (DN parameters)")
    monkeypatch.setattr(L.trainer()._bbf, "attach", lambda model, args: None)
    if arm.name == "C0":
        _m, _c, _a, rec = _build(L, arm)
        assert rec["state_dict"]["unexpected"] == []
        return
    with pytest.raises((SystemExit, RuntimeError)) as e:
        _build(L, arm)
    assert "dn_embed" in str(e.value) or "dn_cls_embed" in str(e.value), str(e.value)[:400]
