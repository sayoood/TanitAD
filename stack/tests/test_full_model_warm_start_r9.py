"""refcv9 FULL-WARM -- the FULL model's ``--init-from`` admits the refcv9 levers' DECLARED new modules, through the rig's ONE exemption list.

THE DEFECT (MEASURED 2026-11-03 on Thor CPU, E2 source ckpt_ema_48852.pt, 1,165 keys): ``refcv8_train.warm_start_from`` knew only ``r8_*`` seams and R9-PF
keys, so K0 + banded was REFUSED (18 new ``_map_hires.banded.*`` keys + 2 unexpected source keys ``_map_hires.refine.cls.*``), K0 + B3 was REFUSED (3 new
``dn_embed`` / ``dn_cls_embed`` keys), and K0 + banded + own-enc was refused by the trainer itself ("--init-from without --r9-perc-rig"). The perception rig
ALREADY solved all of it (``refcv9_perc_rig.warm_start_perc_rig``): NAMED new prefixes per flag, the banded head's replaced ``refine.cls.*`` dropped, own-enc's
widening applied. THE FIX: that list is ONE function, ``refcv9_perc_rig.refcv9_warm_exemptions`` (+ ``apply_refcv9_exemptions``), used by BOTH warm starts.

Proved here on the tiny FULL model (``test_map_own_enc._build``: trunk + s8 tap + 10 cm map + the pooled-BEV perception branch with a box3d head; NO rig):
  ADMIT   per lever (banded, B3, own-enc, B1, B2, B2c, T1 and a combination): the lever-OFF build's state_dict is the refcv8 analogue; the lever-ON build
          warm-starts from it -> ADMITTED; the record names the new prefixes and the dropped source keys; every loaded key equals the source's bytes; the new
          modules keep their own init
  RED     an undeclared extra module, a shape mismatch, a stray source key, an undeclared missing key (lever ON) -> REFUSED; the declared list with one prefix
          deleted (MUTATION, per lever) -> REFUSED; the own-enc widening rule withheld -> REFUSED
  OFF     lever-off: the record and the loaded state are the tip's algorithm's, key for key (the tip's body is re-implemented below, verbatim)
  ONE     the rig and the full model both consult the same function, neither spells the per-lever list out; the rig's list equals the old composition
  STEP 0  per lever, the source's PLANNER outputs vs the warm-started ON build's, on one batch with one RNG state (MEASURED, asserted as measured)
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_aux_heads_full_model as AUXT  # noqa: E402   (_ns / _attach: the B1 / B2 / B2c heads on the tiny full model)
import test_bev_warp as BWT  # noqa: E402                (the T1 argv helper)
import test_map_only_ft as R  # noqa: E402
import test_map_own_enc as MOT  # noqa: E402            (_build(own, banded, seed): the tiny FULL model with the 10 cm map + perception branch)
from tanitad.refs import refcv9_perc_rig as rig  # noqa: E402
from tanitad.train import bev_warp_flag as BF  # noqa: E402
from tanitad.train import box_b3_flag as B3F  # noqa: E402
from tanitad.train import box_depth_aux_flag as BDF  # noqa: E402
from tanitad.train import map_own_enc_flag as MO  # noqa: E402
from tanitad.train import persp3d_aux_flag as PF  # noqa: E402
from tanitad.train import refcv8_train as r8train  # noqa: E402

LEVERS = ("banded", "b3", "own_enc", "b1", "b2", "b2c", "t1")
#: lever -> the key prefixes the record must NAME (own-enc is defined on top of the banded head: it names both)
NEW = {"banded": ("_map_hires.banded.",),
       "b3": ("_perception.box_dec.dn_cls_embed.", "_perception.box_dec.dn_embed."),
       "own_enc": ("_map_hires.banded.",) + tuple(MO.NEW_PREFIXES),
       "b1": (BDF.KEY_PREFIX,), "b2": (PF.KEY_PREFIX_P,), "b2c": (PF.KEY_PREFIX_B,),
       "t1": (BF.KEY_PREFIX,)}
#: lever -> the SOURCE keys with no home in the build (the banded head replaces the 8-way readout; own-enc rebuilds the readout's input)
DROPPED = {"banded": ("_map_hires.refine.cls.",)}


def _b3_args():
    return argparse.Namespace(r9_box_quality=True, r9_box_dn=True, r9_box_class_sqrt=False, r9_box_inloss_nms=False)


def _build_on(lever: str, seed: int = 9):
    """The lever-ON FULL model, a DIFFERENT init from the source's (``MOT._build`` seeds itself): nothing is shared by luck."""
    if lever == "banded":
        return MOT._build(own=False, banded=True, seed=seed)
    if lever == "own_enc":
        return MOT._build(own=True, banded=True, seed=seed)
    model, cfg = MOT._build(own=False, banded=False, seed=seed)
    if lever == "b3":
        B3F.attach(model, _b3_args())
    elif lever in ("b1", "b2", "b2c"):
        AUXT._attach(model, AUXT._ns({"b1": "depth", "b2": "persp", "b2c": "bev2"}[lever]))
    elif lever == "t1":
        BF.attach(model, BWT._args(r9_perc_rig=False))
    return model, cfg


def _build_off(seed: int = 3):
    """The refcv8 analogue: the lever-OFF full model (8-way map readout, no new module)."""
    return MOT._build(own=False, banded=False, seed=seed)


def _save(model, path):
    R._save_source(model, path)


def _snap(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _under(k, prefixes):
    return any(k.startswith(q) for q in prefixes)


_SRC = {}


@pytest.fixture(scope="module", autouse=True)
def _one_source_checkpoint_for_the_module():
    """ONE refcv8-analogue source file for the whole module (a tiny full model's state is ~45 MB: a file per test would fill a small disk); deleted at teardown."""
    import shutil
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="fullwarm_src_"))
    off, _ = _build_off(3)
    _save(off, d / "ema.pt")
    _SRC["path"] = d / "ema.pt"
    yield
    _SRC.clear()
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def _drop_the_checkpoints_a_test_wrote(tmp_path):
    yield
    for f in tmp_path.glob("*.pt"):
        f.unlink()


def _source(tmp_path=None, seed=3):
    """(the lever-OFF build, its config, the path of its saved state): the build is deterministic in ``seed`` and equals what the file holds."""
    off, cfg = _build_off(seed)
    return off, cfg, _SRC["path"]


# ===================================================================================================================== #
# ADMIT                                                                                                                 #
# ===================================================================================================================== #
@pytest.mark.parametrize("lever", LEVERS)
def test_ADMIT_the_lever_on_full_model_warm_starts_from_the_lever_off_state_and_the_record_names_what_was_admitted(lever, tmp_path):
    off, _, p = _source(tmp_path)
    on, _ = _build_on(lever)
    src = _snap(off)
    pre = _snap(on)
    new_pfx = NEW[lever]
    assert set(rig.refcv9_warm_exemptions(on)[0]) == set(new_pfx)              # the shared list reads the declaration off the BUILT model
    rec = r8train.warm_start_from(on, str(p))
    json.dumps(rec)                                                            # goes into config.json["init_from"]
    d = rec["r9_declared_new"]
    assert set(d["new_prefixes"]) == set(new_pfx)
    post = _snap(on)
    # the banded head REPLACES the 8-way readout: the source's `refine.cls.*` has no home (banded and own-enc, which is built on it); own-enc additionally
    # re-sizes the readout's input: the source's `refine.inp.*` is dropped by the widening rule (recorded in rec["own_enc"])
    want_dropped = sorted(k for k in src if _under(k, DROPPED["banded"])) if lever in ("banded", "own_enc") else []
    assert d["dropped_source_keys"] == want_dropped
    if lever in ("banded", "own_enc"):
        assert want_dropped and all(k not in post for k in want_dropped)
    # every key of the build is EITHER loaded (== the source's bytes, or the widened slice) OR a declared new key that kept its own init
    n_new = 0
    widened = set(MO.WIDENED) if lever == "own_enc" else set()
    for k, v in post.items():
        if _under(k, new_pfx):
            n_new += 1
            assert torch.equal(v, pre[k]), k                                   # declared new module: untouched by the warm start
        elif k in widened:
            ax = MO.WIDENED[k]
            d0 = int(on._map_hires.cfg.d_lift)
            assert torch.equal(v.narrow(ax, 0, d0), src[k]), k                 # the source's slice
        else:
            assert k in src and torch.equal(v, src[k]), k                      # everything else is the source's bytes
    assert d["n_new_keys"] == n_new > 0
    assert rec["n_new_refcv8_keys"] == 0 and rec["n_loaded"] + n_new == len(post)
    assert rec["md5"] and rec["source_step"] == 48852 and rec["optimizer"].startswith("fresh")
    if lever == "own_enc":
        assert sorted(rec["own_enc"]["widened_keys"]) == sorted(MO.WIDENED)
        assert rec["own_enc"]["dropped_source_keys"] == sorted(k for k in src if _under(k, ("_map_hires.refine.inp.",)))
        stem = post["_map_hires.encoder.stem.0.weight"]
        d0 = int(on._map_hires.cfg.d_lift)
        assert float(stem[:, d0:].abs().sum()) == 0.0                          # the widening rule: the stem's extra input channels are ZERO
    else:
        assert "own_enc" not in rec


def test_ADMIT_a_combination_of_levers_names_the_union_and_nothing_else(tmp_path):
    """The real E2 / E3 argvs stack levers: banded + own-enc + B3 + B1 on one build."""
    off, _, p = _source(tmp_path)
    on, _ = MOT._build(own=True, banded=True, seed=9)
    B3F.attach(on, _b3_args())
    AUXT._attach(on, AUXT._ns("depth"))
    want = set(NEW["own_enc"]) | set(NEW["b3"]) | set(NEW["b1"])
    assert set(rig.refcv9_warm_exemptions(on)[0]) == want
    rec = r8train.warm_start_from(on, str(p))
    assert set(rec["r9_declared_new"]["new_prefixes"]) == want and rec["n_new_refcv8_keys"] == 0
    post, src = _snap(on), _snap(off)
    stray = [k for k in post if not _under(k, want) and k not in MO.WIDENED and not (k in src and torch.equal(post[k], src[k]))]
    assert stray == []


# ===================================================================================================================== #
# RED -- every silent mismatch is still refused                                                                         #
# ===================================================================================================================== #
@pytest.mark.parametrize("lever", LEVERS)
def test_RED_an_undeclared_extra_module_is_refused_even_with_the_lever_on(lever, tmp_path):
    off, _, p = _source(tmp_path)
    on, _ = _build_on(lever)
    on._perception.mystery_head = torch.nn.Linear(3, 3)                        # a module NO flag declares
    with pytest.raises(SystemExit, match="NOT refcv8 seams"):
        r8train.warm_start_from(on, str(p))


@pytest.mark.parametrize("lever", LEVERS)
def test_RED_an_undeclared_missing_key_is_refused_even_with_the_lever_on(lever, tmp_path):
    off, _, _ = _source(tmp_path)
    sd = _snap(off)
    k = "_map_hires.encoder.blocks.0.conv1.weight"                             # a shared key no lever exempts
    assert k in sd
    del sd[k]
    bad = tmp_path / "missing.pt"
    torch.save({"model": sd, "step": 1}, str(bad))
    on, _ = _build_on(lever)
    with pytest.raises(SystemExit, match="NOT refcv8 seams"):
        r8train.warm_start_from(on, str(bad))


@pytest.mark.parametrize("lever", LEVERS)
def test_RED_a_shape_mismatch_and_a_stray_source_key_are_refused_with_the_lever_on(lever, tmp_path):
    off, _, _ = _source(tmp_path)
    sd = _snap(off)
    k = "_map_hires.encoder.blocks.0.conv1.weight"
    bad = dict(sd, **{k: torch.zeros(3, 3)})
    p1 = tmp_path / "shape.pt"
    torch.save({"model": bad, "step": 1}, str(p1))
    on, _ = _build_on(lever)
    with pytest.raises(SystemExit, match="init-from"):
        r8train.warm_start_from(on, str(p1))
    p2 = tmp_path / "stray.pt"
    torch.save({"model": dict(sd, **{"_map_hires.encoder.mystery.weight": torch.zeros(1)}), "step": 1}, str(p2))
    on2, _ = _build_on(lever)
    with pytest.raises(SystemExit, match="does not have"):
        r8train.warm_start_from(on2, str(p2))


@pytest.mark.parametrize("lever", LEVERS)
def test_RED_MUTATION_the_declared_list_with_one_prefix_deleted_refuses(lever, tmp_path, monkeypatch):
    off, _, p = _source(tmp_path)
    on, _ = _build_on(lever)
    real = rig.refcv9_warm_exemptions
    victim = NEW[lever][-1]

    def mutated(m):
        new, drop, widen = real(m)
        return tuple(q for q in new if q != victim), drop, widen
    monkeypatch.setattr(rig, "refcv9_warm_exemptions", mutated)
    with pytest.raises(SystemExit):
        r8train.warm_start_from(on, str(p))
    monkeypatch.setattr(rig, "refcv9_warm_exemptions", real)                   # the control: un-mutated, the same call is ADMITTED
    on2, _ = _build_on(lever)
    assert r8train.warm_start_from(on2, str(p))["r9_declared_new"]["n_new_keys"] > 0


def test_RED_MUTATION_the_banded_drop_list_deleted_refuses_on_the_unhomed_source_keys(tmp_path, monkeypatch):
    off, _, p = _source(tmp_path)
    on, _ = _build_on("banded")
    real = rig.refcv9_warm_exemptions
    monkeypatch.setattr(rig, "refcv9_warm_exemptions", lambda m: (real(m)[0], (), real(m)[2]))
    with pytest.raises(SystemExit, match="does not have"):
        r8train.warm_start_from(on, str(p))


def test_RED_MUTATION_the_own_enc_widening_withheld_refuses_on_the_shape(tmp_path, monkeypatch):
    off, _, p = _source(tmp_path)
    on, _ = _build_on("own_enc")
    real = rig.refcv9_warm_exemptions
    monkeypatch.setattr(rig, "refcv9_warm_exemptions", lambda m: (real(m)[0], real(m)[1], None))
    with pytest.raises(SystemExit, match="init-from"):
        r8train.warm_start_from(on, str(p))


def test_RED_own_enc_widening_refuses_a_source_that_is_already_widened(tmp_path):
    off, _, _ = _source(tmp_path)
    on, _ = _build_on("own_enc")
    sd = _snap(off)
    sd["_map_hires.lift.proj.weight"] = torch.zeros_like(on.state_dict()["_map_hires.lift.proj.weight"])
    p = tmp_path / "wide.pt"
    torch.save({"model": sd, "step": 1}, str(p))
    with pytest.raises(SystemExit, match="widening rule"):
        r8train.warm_start_from(on, str(p))


# ===================================================================================================================== #
# OFF -- the tip's warm start, key for key                                                                              #
# ===================================================================================================================== #
def _tip_warm_start_from(model, path):
    """The tip's ``refcv8_train.warm_start_from`` body (638d573), re-implemented VERBATIM here: the OFF reference."""
    import hashlib
    p = Path(path)
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    state = torch.load(str(p), map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise SystemExit(f"[refcv8] --init-from {path}: {str(e).splitlines()[0][:400]}") from None
    if unexpected:
        raise SystemExit("unexpected")
    from tanitad.refs.refcv9_progress import is_refcv9_key
    undeclared = [k for k in missing if not r8train.is_refcv8_key(k) and not is_refcv9_key(k)]
    if undeclared:
        raise SystemExit("undeclared")
    return {"path": str(p), "md5": h.hexdigest(), "source_step": (int(state["step"]) if isinstance(state, dict)
                                                                  and "step" in state else None),
            "n_loaded": len(sd), "n_new_refcv8_keys": len(missing), "optimizer": "fresh (not loaded)"}


def test_OFF_the_lever_off_full_model_warm_start_is_the_tips_record_and_state(tmp_path):
    off, _, p = _source(tmp_path)
    a, _ = _build_off(seed=9)
    b, _ = _build_off(seed=9)
    assert rig.refcv9_warm_exemptions(a) == ((), (), None)                     # every lever OFF: nothing declared
    ra = r8train.warm_start_from(a, str(p))
    rb = _tip_warm_start_from(b, str(p))
    assert ra == rb and list(ra) == list(rb)                                   # the SAME dict (no new key when OFF), in the same order
    sa, sb = _snap(a), _snap(b)
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert all(torch.equal(sa[k], _snap(off)[k]) for k in sa)                  # and the whole state IS the source's


def test_OFF_a_refcv8_seam_build_keeps_counting_its_seams_as_new_refcv8_keys():
    """The seams branch of the same function (the tip's test_refcv8_init_from covers the forward identity); here: ``n_new_refcv8_keys`` keeps its meaning
    (r8 / R9-PF keys the source lacks) and the new ``r9_declared_new`` block is ABSENT unless a refcv9 lever is declared."""
    src_txt = inspect.getsource(r8train.warm_start_from)
    assert 'rec["r9_declared_new"]' in src_txt and "if new_pfx:" in src_txt and '"n_new_refcv8_keys": len(missing) - len(declared_new)' in src_txt


# ===================================================================================================================== #
# ONE list                                                                                                              #
# ===================================================================================================================== #
def test_ONE_both_warm_starts_consult_the_same_function_and_neither_spells_the_list_out():
    names = ("banded_exemptions(", "b3_exemptions(", "own_enc_exemptions(", "box_depth_exemptions(", "persp_exemptions(", "bev_warp_exemptions(")
    for fn in (rig.warm_start_perc_rig, r8train.warm_start_from):
        src = inspect.getsource(fn)
        assert "apply_refcv9_exemptions(" in src, fn.__name__
        assert not any(n in src for n in names), fn.__name__
    shared = inspect.getsource(rig.refcv9_warm_exemptions)
    assert all(n in shared for n in names)                                     # the one place the list is written down
    assert "refcv9_warm_exemptions(" in inspect.getsource(rig.apply_refcv9_exemptions)


@pytest.mark.parametrize("lever", LEVERS)
def test_ONE_the_shared_function_equals_the_rigs_old_composition_per_lever(lever):
    m, _ = _build_on(lever)
    new_pfx, drop_pfx = rig.banded_exemptions(m)
    old = tuple(new_pfx) + rig.b3_exemptions(m) + rig.own_enc_exemptions(m) + rig.box_depth_exemptions(m) + rig.persp_exemptions(m) + rig.bev_warp_exemptions(m)
    n, d, w = rig.refcv9_warm_exemptions(m)
    assert n == old and d == tuple(drop_pfx) and (w is not None) == bool(rig.own_enc_exemptions(m))
    assert (w is MO.apply_warm_start) == (lever == "own_enc")


def test_ONE_the_two_warm_starts_agree_on_the_same_build(tmp_path, monkeypatch):
    """The SAME ON build, the SAME source file, through the rig's function and through the full model's: the shared part of the two records is the same
    (the rig's model is not the full model, so this is the exemption bookkeeping only -- the shared function is called with the same model)."""
    seen = []
    real = rig.apply_refcv9_exemptions
    monkeypatch.setattr(rig, "apply_refcv9_exemptions", lambda *a, **k: (seen.append(a[0]), real(*a, **k))[1])
    off, _, p = _source(tmp_path)
    on, _ = _build_on("own_enc")
    r8train.warm_start_from(on, str(p))
    assert seen == [on]


def test_the_trainer_dispatches_on_the_rig_and_the_own_enc_init_from_refusal_is_lifted():
    src = (ROOT / "scripts" / "refc_v3_train.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert src.count("(_r9rig.warm_start_perc_rig if _r9rig_on else r8train.warm_start_from)(model, args.init_from)") == 1
    a = argparse.Namespace(r9_map_own_enc=True, r9_map_banded=True, init_from="x.pt", r9_perc_rig=False)
    MO.pin_args(a)                                                             # the full model's --init-from is admitted
    with pytest.raises(SystemExit, match="--r9-map-banded is absent"):         # every OTHER refusal stands
        MO.pin_args(argparse.Namespace(r9_map_own_enc=True, r9_map_banded=False, init_from="x.pt", r9_perc_rig=False))
    assert "without --r9-perc-rig" not in inspect.getsource(MO.pin_args)


# ===================================================================================================================== #
# STEP 0 -- does the warm-started full model reproduce the source's PLANNER outputs?                                   #
# ===================================================================================================================== #
def _leaves(x, p=""):
    if torch.is_tensor(x):
        yield p, x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, p + "." + str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _leaves(v, p + "[%d]" % i)


def _batch_for(lever, on, cfg_on):
    """ONE batch per lever, built from the ON model (its targets / history fields), with two scenes' cuboids so the box path and B3's DN pack are live."""
    if lever == "t1":
        return BWT._batch_t1(cfg_on, on, BWT._args(r9_perc_rig=False))[0]
    kinds = {"b1": ("depth",), "b2": ("persp",), "b2c": ("bev2",)}.get(lever, ())
    batch = AUXT._batch(cfg_on, on, *kinds)
    if lever == "b3":                                                         # the DN pack reads the GT class too
        batch["agent_cls"] = torch.zeros(batch["agent_valid"].shape, dtype=torch.long)
    return batch


def _outputs(model, batch, seed=5, train=True, dn=True):
    """One forward through ``compute_losses_v3`` on ``batch`` (cloned), one RNG state -> {leaf name: tensor} of the model's raw output.
    ``train=False``: the eval forward (no DN queries, no aux head). ``dn=False``: the training forward with B3's DN pack switched OFF."""
    cap = {}
    h = model.register_forward_hook(lambda mod, a, o: cap.__setitem__("o", o))
    real_prepare = T._bbf.prepare
    try:
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        if BF._fuse(model) is not None:                                       # T1: the full loss does not stage the history (OUT of scope): stage it as the rig does
            BF.stage(model, b, "cpu")
        if not dn:
            T._bbf.prepare = lambda *a, **k: None
        torch.manual_seed(seed)
        model.train(train)
        with torch.set_grad_enabled(train):
            T.compute_losses_v3(model, b, "cpu", mode="diffusion")
    finally:
        T._bbf.prepare = real_prepare
        h.remove()
    return {k: v.detach().clone() for k, v in _leaves(cap["o"])}


MAP_LEAVES = (".perception.map_banded", ".perception.map_hires")


def _step0(lever, tmp_path, **kw):
    """-> (leaf names whose value differs between the SOURCE and the warm-started lever-ON build, leaves only the ON build has, the source's leaves)."""
    off, cfg, p = _source(tmp_path)
    on, cfg_on = _build_on(lever)
    batch = _batch_for(lever, on, cfg_on)
    a = _outputs(off, batch, **kw)
    r8train.warm_start_from(on, str(p))
    b = _outputs(on, batch, **kw)
    assert a, "the source produced no output"
    differ = sorted(k for k in a if k not in b or not torch.equal(a[k], b[k]))
    return differ, sorted(set(b) - set(a)), sorted(a)


@pytest.mark.parametrize("lever", LEVERS)
def test_STEP0_the_planner_outputs_of_the_warm_started_full_model_equal_the_sources(lever, tmp_path):
    """MEASURED (CPU, the tiny full model, one batch with GT cuboids, one RNG state): banded / own-enc move ONLY the map readout (`map_hires_logits`, plus the new
    banded outputs); B1 / B2 / B2c / T1 (zero gate) move NOTHING; B3 with its DN queries LIVE in the training forward moves the box slots AND the planner outputs
    (the DN rows are real queries in the decoder, not a zero-init seam) -- asserted below only for the two claims that hold: the EVAL forward and the training
    forward with the DN pack OFF reproduce the source."""
    if lever == "b3":
        for kw in ({"train": False}, {"dn": False}):
            differ, extra, leaves = _step0(lever, tmp_path, **kw)
            print("STEP0", lever, kw, "n_leaves", len(leaves), "differ", differ, "only-in-ON", extra)
            assert differ == [] and extra == [], kw
        differ, extra, leaves = _step0(lever, tmp_path)
        print("STEP0", lever, "train+DN", "n_leaves", len(leaves), "n_differ", len(differ), "differ", differ, "only-in-ON", extra)
        assert differ, "the DN queries are live in this forward: the box slots must move (else the DN pack was not exercised)"
        return
    differ, extra, leaves = _step0(lever, tmp_path)
    print("STEP0", lever, "n_leaves", len(leaves), "differ", differ, "only-in-ON", extra)
    planner_differ = [k for k in differ if not k.startswith(MAP_LEAVES)]
    assert planner_differ == [], planner_differ                               # the planner path is the source's, bit for bit
    if lever in ("banded", "own_enc"):
        assert differ == [".perception.map_hires_logits"]                     # a replacing map head changes the map readout and only that
    else:
        assert differ == [] and extra == []                                   # a zero-init / training-only / gated seam moves NOTHING at step 0


def test_STEP0_the_check_can_fail_a_build_that_was_not_warm_started_moves_the_planner(tmp_path):
    """The control: an ON build that was NOT warm-started (its own random init) does NOT reproduce the source's planner -> the check above has teeth."""
    off, cfg, _ = _source(tmp_path)
    on, cfg_on = _build_on("b1")
    batch = _batch_for("b1", on, cfg_on)
    a = _outputs(off, batch)
    b = _outputs(on, batch)
    assert [k for k in a if not k.startswith(MAP_LEAVES) and not torch.equal(a[k], b[k])]
