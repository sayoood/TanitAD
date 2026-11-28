"""refcv9 T1-GEVAL -- the launch gate's G-EVAL on a ``--r9-bev-warp`` RIG argv (E1 arms T1 / T1s).  CPU, the tiny real rig.

THE DEFECT (MEASURED 2026-11-03 at tip 9be3211, tiny CPU rig with T1 built): G-EVAL's rig branch builds its probe dataset with
``rig_eval_probe_dataset`` -- which never ``attach_dataset``s the T1 history -- so the probe batch carries no ``bw_*`` and
``_forward_out`` raises ``[bev-warp] the model is built with the temporal BEV warp but the batch carries no ['bw_frames', 'bw_pose_now',
'bw_pose_past', 'bw_has']: the dataset was not attached``.  ⚠ THE BRIEF'S SECOND HALF IS NOT TRUE AT THE TIP: ``compute_losses_v3`` DOES stage for
a rig model -- it DISPATCHES (refc_v3_train.py:5702) to ``compute_losses_perc_rig``, which calls ``_bw.stage`` (:7129) before ``model(...)``.
The ONLY missing piece was the attach.  ``_forward_out`` is deliberately left byte-identical: it must exercise the TRAINER'S own staging
(a ``_forward_out`` that staged for the trainer would hide a trainer that stopped staging -- the very defect G-EVAL certifies the absence of).

  G1  the defect, as measured: a probe dataset built WITHOUT the attach (the tip's behaviour) -> the forward REFUSES, naming the keys
  G2  the fix: the probe dataset carries the history; the trainer-built and the loader-built model give the same bytes, every output, incl.
      the determinism control; the history is READ (a control: other history content moves the output at a non-zero gate, not at gate 0)
  G3  double staging / the one-shot fuse: stage twice before one forward == once, bit for bit; the forward CONSUMES the staged history, so a
      second forward without a stage REFUSES (a stage outside ``_forward_out`` would break the determinism control; there is none)
  G4  RED: the trainer stops staging -> the forward REFUSES (never an un-warped forward); the attach dropped -> REFUSES, named
  G5  the helper's own refusals (no model, no ``_bw``, a model with no fuse) are named reasons
  G6  the two builds must agree on the lags (seconds and rows), the shuffle and the fuse -- the forward comparison CANNOT see it (measured)
  G7  OFF: a non-T1 argv's probe dataset and record are the tip's, the trainer module's ``_bw`` is never read
  G8  reachability pins: the call site passes the loader's model, reads the reasons, ``_forward_out`` stays the trainer's forward
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import refc_v3_train as T  # noqa: E402
import test_bev_warp as TB  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.train import bev_warp_flag as BF  # noqa: E402

IDX = 14                       # a window whose NOW row has both lags (rows 5 and 10), as test_bev_warp's _batch_t1


def _args_for(base, **over):
    """What the helper reads off a T1 rig argv: the eval cache / lru / u8 / pose-sync / labels flags on top of the T1 namespace."""
    a = argparse.Namespace(**vars(base))
    a.eval_cache, a.v2_lru, a.u8_batches, a.pose_sync_sidecar, a.eval_labels = "/x/eval", 2, False, None, None
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _loader(trainer=T):
    return types.SimpleNamespace(trainer=lambda: trainer)


def _providers(monkeypatch, cfg):
    import tanitad.data.v2_dataset as V2
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=[R.CLIP, R.CLIP + "-b"])
    monkeypatch.setattr(V2, "build_v2_providers", lambda caches, lru_size: eps)
    return eps


def _probe(rig, monkeypatch, model=None, args=None, loader=None):
    """What G-EVAL does at the probe site: the helper, then ``default_collate([e_ds[idx]])`` + ``map_ep`` (the rig reads nothing else)."""
    _providers(monkeypatch, rig.cfg)
    args = _args_for(rig.args_l) if args is None else args
    ds, eps, rec = LG.rig_eval_probe_dataset(loader or _loader(), rig.cfg, args, model=rig.ml if model is None else model)
    batch = torch.utils.data.default_collate([ds[IDX]])
    e_i, _t = ds.index[IDX]
    batch["map_ep"] = torch.tensor([int(eps[e_i].episode_id)], dtype=torch.long)
    return ds, rec, batch


def _out(model, batch, args, seed=1234):
    return LG.output_digests(LG._forward_out(T, model, batch, args, seed))


def _gate(model, v):
    with torch.no_grad():
        model._map_hires.bev_warp.gate.fill_(v)


@pytest.fixture(scope="module")
def rig():
    """A trainer-built T1 rig model and a LOADER-built one (a second build + ``attach`` exactly as ``refcv7_loader._build_model`` does, :439, then the
    trainer's state strict-loaded), both at a NON-ZERO gate so the history is numerically read, both in G-EVAL's eval-time flags."""
    with pytest.MonkeyPatch.context() as mp:
        mt, cfg, args_t = TB._rig(mp, on=True)
        ml, _, args_l = TB._rig(mp, on=True)
        _gate(mt, 0.3)
        ml.load_state_dict(mt.state_dict(), strict=True)
        for m in (mt, ml):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        yield types.SimpleNamespace(mt=mt, ml=ml, cfg=cfg, args_t=args_t, args_l=args_l)


@pytest.fixture()
def gate_restored(rig):
    yield
    _gate(rig.mt, 0.3)
    _gate(rig.ml, 0.3)


# =========================================================================== #
# G1  the defect, as measured                                                  #
# =========================================================================== #
def test_G1_a_probe_dataset_without_the_attach_REFUSES_the_T1_forward_naming_the_keys(rig, monkeypatch):
    """The tip's behaviour (the attach is a no-op): no ``bw_*`` in the batch -> the stage REFUSES, for the trainer's model and the loader's."""
    monkeypatch.setattr(BF, "attach_dataset", lambda ds, model, args: {})
    _ds, _rec, batch = _probe(rig, monkeypatch)
    assert not [k for k in batch if k.startswith("bw_")]
    for m in (rig.mt, rig.ml):
        with pytest.raises(SystemExit, match=r"carries no \['bw_frames', 'bw_pose_now', 'bw_pose_past', 'bw_has'\].*dataset was not attached"):
            LG._forward_out(T, m, batch, rig.args_t, 1234)


# =========================================================================== #
# G2  the fix                                                                  #
# =========================================================================== #
def test_G2_the_probe_dataset_carries_the_history_and_both_builds_forward_identically(rig, monkeypatch):
    ds, rec, batch = _probe(rig, monkeypatch)
    st = rig.ml._r9_bev_warp
    assert rec["bev_warp"] == {"lag_rows": list(st.lag_rows), "shuffle": False, "n_episodes": 2}
    assert ds._r9_bw.lag_rows == st.lag_rows and rec["rig"] is True and rec["labels"] is None
    assert set(BF.BATCH_KEYS) <= set(batch) and bool(batch["bw_has"].all()), "the probe window must have both lags (row >= 10)"
    ot = _out(rig.mt, batch, rig.args_t)
    assert _out(rig.mt, batch, rig.args_t) == ot                                    # G-EVAL's determinism control (trainer model, every call stages)
    ol = _out(rig.ml, batch, rig.args_l)
    assert sorted(k for k in set(ot) | set(ol) if ot.get(k) != ol.get(k)) == []     # trainer-built == loader-built, every output
    assert sum(1 for v in ot.values() if not v.startswith(("py:", "type:"))) > 0
    assert any(k.startswith("out.perception.map_hires_logits") for k in ot)


def test_G2_the_history_is_READ_at_a_non_zero_gate_and_only_there(rig, monkeypatch, gate_restored):
    """Control: the comparison can SEE the history.  Other history content moves the output at gate 0.3 and moves NOTHING at gate 0."""
    _ds, _rec, batch = _probe(rig, monkeypatch)
    other = dict(batch)
    other["bw_frames"] = (255 - batch["bw_frames"].to(torch.int32)).to(batch["bw_frames"].dtype) if batch["bw_frames"].dtype == torch.uint8 \
        else 1.0 - batch["bw_frames"]
    base = _out(rig.mt, batch, rig.args_t)
    assert _out(rig.mt, other, rig.args_t) != base
    _gate(rig.mt, 0.0)
    assert _out(rig.mt, other, rig.args_t) == _out(rig.mt, batch, rig.args_t)


# =========================================================================== #
# G3  double staging; the fuse is one-shot                                     #
# =========================================================================== #
def test_G3_staging_twice_before_one_forward_is_the_same_forward_and_the_forward_consumes_the_staged_history(rig, monkeypatch):
    _ds, _rec, batch = _probe(rig, monkeypatch)
    fuse = rig.mt._map_hires.bev_warp
    once = _out(rig.mt, batch, rig.args_t)
    assert fuse._ctx is None                                                        # the lift hook CLEARED it: ONE-SHOT
    BF.stage(rig.mt, batch, "cpu")                                                  # an extra stage OUTSIDE the trainer's own
    first = fuse._ctx
    assert first is not None
    twice = _out(rig.mt, batch, rig.args_t)                                         # _forward_out -> compute_losses_perc_rig stages AGAIN
    assert twice == once and fuse._ctx is None                                      # stage OVERWRITES (never accumulates): idempotent
    BF.stage(rig.mt, batch, "cpu")
    BF.stage(rig.mt, batch, "cpu")
    assert _out(rig.mt, batch, rig.args_t) == once                                  # three stages, one forward
    BF.stage(rig.mt, batch, "cpu")
    _out(rig.mt, batch, rig.args_t)
    with pytest.raises(RuntimeError, match="no history was staged"):                # consumed: a 2nd forward on one stage REFUSES
        TB._mh(rig.mt, batch)


# =========================================================================== #
# G4  RED arms                                                                 #
# =========================================================================== #
def test_G4_RED_a_trainer_that_stops_staging_makes_the_forward_REFUSE_for_both_builds(rig, monkeypatch):
    """`_forward_out` must NOT stage for the trainer: with the trainer's stage gone the forward REFUSES at the lift hook (never an un-warped forward)."""
    _ds, _rec, batch = _probe(rig, monkeypatch)
    monkeypatch.setattr(BF, "stage", lambda model, batch, device: None)
    for m, a in ((rig.mt, rig.args_t), (rig.ml, rig.args_l)):
        with pytest.raises(RuntimeError, match="no history was staged"):
            LG._forward_out(T, m, batch, a, 1234)


def test_G4_RED_a_dropped_attach_inside_the_helper_REFUSES_by_name(rig, monkeypatch):
    """The probe helper with its attach line deleted (source mutation): the helper's own source minus the attach, exec'd in the module's namespace."""
    src = inspect.getsource(LG.rig_eval_probe_dataset)
    assert src.count("rec[\"bev_warp\"] = bw.attach_dataset(e_ds, model, args)") == 1
    mutated = src.replace("rec[\"bev_warp\"] = bw.attach_dataset(e_ds, model, args)", "pass")
    ns = dict(vars(LG))
    exec(compile(mutated, "<mutated rig_eval_probe_dataset>", "exec"), ns)          # noqa: S102 -- a test-only source mutation
    _providers(monkeypatch, rig.cfg)
    ds, eps, rec = ns["rig_eval_probe_dataset"](_loader(), rig.cfg, _args_for(rig.args_l), model=rig.ml)
    batch = torch.utils.data.default_collate([ds[IDX]])
    batch["map_ep"] = torch.tensor([int(eps[ds.index[IDX][0]].episode_id)], dtype=torch.long)
    assert "bev_warp" not in rec
    with pytest.raises(SystemExit, match="dataset was not attached"):
        LG._forward_out(T, rig.mt, batch, rig.args_t, 1234)


# =========================================================================== #
# G5  the helper's own refusals                                                #
# =========================================================================== #
def test_G5_a_T1_argv_with_no_model_or_no_trainer_bw_or_a_model_without_the_fuse_is_refused_by_name(rig, monkeypatch):
    _providers(monkeypatch, rig.cfg)
    a = _args_for(rig.args_l)
    with pytest.raises(LG.GateError, match="--r9-bev-warp but the probe dataset was built without the loader's model"):
        LG.rig_eval_probe_dataset(_loader(), rig.cfg, a)
    with pytest.raises(LG.GateError, match="the trainer module has no `_bw`"):
        LG.rig_eval_probe_dataset(_loader(types.SimpleNamespace(V3Dataset=T.V3Dataset)), rig.cfg, a, model=rig.ml)
    off_model, _, _ = TB._rig(monkeypatch, on=False)                                # T1 declared, the loader built no fuse
    with pytest.raises(SystemExit, match="the model carries no temporal BEV warp"):
        LG.rig_eval_probe_dataset(_loader(), rig.cfg, a, model=off_model)


# =========================================================================== #
# G6  the two builds must agree on the lags                                    #
# =========================================================================== #
def test_G6_identical_builds_give_no_reason_and_other_lags_or_shuffle_or_a_missing_fuse_each_give_a_NAMED_reason(rig, monkeypatch):
    r, d = LG.rig_eval_t1_build_reasons(rig.args_t, rig.mt, rig.args_l, rig.ml)
    assert r == [] and d["on"] is True and d["trainer"] == d["loader"] and d["loader"]["lag_rows"] == [5, 10]
    other, _, args_o = TB._rig(monkeypatch, on=True, r9_bev_warp_lags="0.5,1.5")    # same fuse width (2 lags), other rows
    r, d = LG.rig_eval_t1_build_reasons(rig.args_t, rig.mt, args_o, other)
    assert [x.split(" differs")[0] for x in r] == ["T1 (--r9-bev-warp): lags_s", "T1 (--r9-bev-warp): lag_rows", "T1 (--r9-bev-warp): fuse_lag_rows"]
    assert "[5, 10]" in r[1] and "[5, 15]" in r[1] and d["loader"]["lag_rows"] == [5, 15]
    shuf, _, args_s = TB._rig(monkeypatch, on=True, r9_bev_warp_shuffle=True)
    r, _ = LG.rig_eval_t1_build_reasons(rig.args_t, rig.mt, args_s, shuf)
    assert [x.split(" differs")[0] for x in r] == ["T1 (--r9-bev-warp): shuffle"]
    off, _, args_off = TB._rig(monkeypatch, on=False, r9_bev_warp=False)             # an OFF build AND an OFF argv (`_rig(on=False)` alone keeps the flag True)
    r, _ = LG.rig_eval_t1_build_reasons(rig.args_t, rig.mt, args_off, off)
    assert {x.split(" differs")[0].split(": ")[1] for x in r} >= {"declared", "fuse_built"}
    assert LG.rig_eval_t1_build_reasons(args_off, off, args_off, off) == ([], {"on": False})   # OFF: inert


def test_G6_the_forward_comparison_cannot_see_other_lags_so_the_reason_is_the_check(rig, monkeypatch):
    """MEASURED: a fuse built for other lags gives the SAME bytes on the loader's probe batch at ANY gate -- the lag rows reach the forward only
    through the batch.  The build comparison is therefore not redundant with the forward comparison."""
    other, _, args_o = TB._rig(monkeypatch, on=True, r9_bev_warp_lags="0.5,1.5")
    other.load_state_dict(rig.mt.state_dict(), strict=True)
    other.eval()
    _ds, _rec, batch = _probe(rig, monkeypatch)
    assert _out(other, batch, args_o) == _out(rig.mt, batch, rig.args_t)
    assert LG.rig_eval_t1_build_reasons(rig.args_t, rig.mt, args_o, other)[0] != []


# =========================================================================== #
# G7  OFF                                                                      #
# =========================================================================== #
class _NoBw:
    """a trainer stand-in whose `_bw` must never be read when the argv declares no T1"""
    V3Dataset = None

    def __getattr__(self, name):
        raise AssertionError("OFF: the helper read trainer.%s" % name)


def test_G7_OFF_the_probe_dataset_record_and_trainer_access_are_the_tips(rig, monkeypatch):
    _providers(monkeypatch, rig.cfg)
    off = _args_for(rig.args_l, r9_bev_warp=False)

    class _Tr(_NoBw):
        V3Dataset = staticmethod(lambda eps, **kw: T.V3Dataset(eps, **kw))
    for model in (None, rig.ml):                                                   # the model is not read when OFF
        ds, eps, rec = LG.rig_eval_probe_dataset(_loader(_Tr()), rig.cfg, off, model=model)
        assert set(rec) == {"rig", "labels", "n_episodes", "n_windows", "newest_row_only", "label_clock"}
        assert not hasattr(ds, "_r9_bw") and "bev_warp" not in rec
        assert not [k for k in ds[IDX] if k.startswith("bw_")]
    assert LG.rig_eval_t1_build_reasons(off, rig.ml, off, rig.ml)[1]["on"] is True       # (a fuse-carrying model is still compared when OFF is declared)


# =========================================================================== #
# G8  reachability pins                                                        #
# =========================================================================== #
def test_G8_the_call_site_hands_over_the_loaders_model_and_reads_the_build_reasons_and_forward_out_stays_the_trainers_forward():
    src = inspect.getsource(LG.eval_identity)
    assert "e_ds, e_eps, drec = rig_eval_probe_dataset(L, cfg_l, args_l, model=model_l)" in src
    assert "t1r, t1d = rig_eval_t1_build_reasons(args_t, model_t, args_l, model_l)" in src and "reasons += t1r" in src
    assert src.index("t1r, t1d = rig_eval_t1_build_reasons") < src.index("idx = int(L.inrun_eval_perm")
    assert 'det["bev_warp"] = t1d' in src and 'det["probe"]["bev_warp"] = drec["bev_warp"]' in src
    assert "L.build_eval_dataset(model_l, cfg_l, args_l, config," in src                      # the non-rig call is still the vendored loader's
    fo = inspect.getsource(LG._forward_out)
    assert "T.compute_losses_v3(model, batch, \"cpu\"" in fo and "_bw" not in fo and "stage" not in fo   # the TRAINER stages; G-EVAL certifies that
    ts = inspect.getsource(T.compute_losses_v3)
    assert "compute_losses_perc_rig(model, batch, device" in ts.split("cfg = model.cfg")[0]       # the rig dispatch the forward rides
    assert re.search(r"_bw\.stage\(model, batch, device\)\s+#[^\n]*\n[^\n]*\n(?:\s*_bc5\.stage\(model, batch, device\)[^\n]*\n)?\s*out = model\(frames", inspect.getsource(T.compute_losses_perc_rig))   # C5-PFQ: + its camera stage


def test_G8_the_attribute_names_the_gate_reads_are_the_flags_own():
    src = inspect.getsource(LG.rig_eval_t1_build_reasons)
    assert 'getattr(mh, "%s"' % BF.HEAD_ATTR in src and 'getattr(m, "%s"' % BF.MODEL_ATTR in src and 'getattr(args, "%s"' % BF.DEST in src
    assert 'getattr(args, "r9_bev_warp", False)' in inspect.getsource(LG.rig_eval_probe_dataset) and BF.DEST == "r9_bev_warp"
