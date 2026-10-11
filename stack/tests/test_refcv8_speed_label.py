"""refcv8 -- the ``label`` speed source (``--r8-speed-input label``), PI RULING 2026-10-11 (Sayed, verbatim):

  "yes use the quantified label as input, it is not a training signal, it is the maximal wished speed. it is ok that it
  is matching the future speed in some cases since it also must be achieved in the future. It is important to assure
  the road type quantization and avoid the exact value of the future speed as target speed."

The source of the 4-way set-speed channel on refcv8 is refcv7's road-type sidecar ({30, 50, 100, 120} km/h one-hot;
``--speed-max-sidecar-v6`` + ``-eval``). What this file pins:

  1. the argv pin: ``label`` ACCEPTS the sidecar and REQUIRES it (and ``--max-speed-input-v6``); n2 / n3 / none still
     REFUSE it (the earlier X3 ruling stands for them); the speed-only arm (no --refcv8) takes it too;
  2. THE PI'S REQUIREMENT AS A REFUSAL: ``read_road_type_sidecar`` refuses any sidecar row that is not exactly a
     road-type bin -- an exact / finer future speed in a quantized field, a missing bin, a one-hot that is not the
     bin's, an "unknown" row with mass -- and the value it returns is the ROAD-TYPE STEP, never the sidecar's
     ``v_hi_ms``. A DELIBERATE-REGRESSION arm removes the check and goes RED (the dirty sidecar loads);
  3. the dataset feeds the step (never ``v_hi_ms``), with the per-step histogram in the census;
  4. the trained unknown row and the V-VSHUF roll apply to this source exactly as to n2;
  5. the route ruling: ``--r8-no-rc`` keeps the nav token and its turn distance, and refuses ``--r8-rc-roll``.

Every expectation is a LITERAL (a check derived from the value it checks is green forever).
"""
from __future__ import annotations

import json
import types

import pytest
import torch

import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refcv6_max_speed as v6ms  # noqa: E402
from tanitad.refs import refcv8_conditioning as C  # noqa: E402
from tanitad.train import refcv8_train as RT  # noqa: E402

T = R.trainer()
BASE = ["--arm", "hier", "--tac-decoder-v6", "--sampler", "ddim", "--out", "X"]
ON = BASE + ["--refcv8", "--w-r8-cons", "0.05"]
LABEL = ["--max-speed-input-v6", "--r8-speed-input", "label", "--speed-max-sidecar-v6", "S.jsonl",
         "--speed-max-sidecar-v6-eval", "SE.jsonl"]
X3 = ["--max-speed-input-v6", "--r8-speed-input", "n2", "--r8-v9-labels", "v9.npz"]
STEPS_KMH = (30, 50, 100, 120)


def _pin(argv):
    args = T.build_parser().parse_args(argv)
    cfg = types.SimpleNamespace(refcv8=C.R8Config())
    T._pin_refcv8(cfg, args)
    return cfg.refcv8


# =========================================================================== #
# 0. the literals                                                              #
# =========================================================================== #
def test_the_road_type_ladder_is_the_PIs_four_values_and_equals_the_models():
    assert RT.R8_ROAD_TYPE_KMH == (30, 50, 100, 120)
    assert tuple(v6ms.SPEED_MAX_STEPS_KMH_V6) == (30, 50, 100, 120)       # the model's one-hot ladder is the same four
    assert RT.R8_SPEED_LABEL == "label" and RT.R8_SPEED_PAST_ONLY == ("n2", "n3")


def test_the_label_stamp_declares_the_ruling_and_the_guard_checks_it():
    st = RT.speed_derivation("label")
    for tok in ("PI 2026-10-11", "maximal wished speed", "road-type quantized", "{30, 50, 100, 120} km/h",
                "ego-future derived per clip", "accepted by the PI", "4-way one-hot", "unknown row",
                "not a training signal"):
        assert tok in st, tok
    RT.assert_r8_speed_stamp({"r8_speed_derivation": st}, "label")
    with pytest.raises(SystemExit, match="does not declare"):
        RT.assert_r8_speed_stamp({"r8_speed_derivation": RT.R8_SPEED_DERIVATION["n2"]}, "label")
    with pytest.raises(SystemExit, match="does not declare"):          # and the label text is not an N2 stamp
        RT.assert_r8_speed_stamp({"r8_speed_derivation": st}, "n2")
    with pytest.raises(SystemExit, match="no `r8_speed_derivation`"):
        RT.assert_r8_speed_stamp({}, "label")
    with pytest.raises(SystemExit, match="must not be stamped"):
        RT.assert_r8_speed_stamp({"r8_speed_derivation": st}, None)


# =========================================================================== #
# 1. the argv pin                                                              #
# =========================================================================== #
def test_GREEN_label_pins_through_with_the_sidecar_on_refcv8_and_on_the_speed_only_arm():
    assert _pin(ON + LABEL).speed_input == "label"
    assert _pin(ON + LABEL + ["--r8-speed-unknown-p", "0.45"]).speed_input == "label"
    _pin(ON + LABEL + ["--r8-roll-speed-input", "--batch", "4"])
    _pin(ON + LABEL + ["--eval-cache", "E"])                          # the eval sidecar is named -> admitted
    r = _pin(BASE + LABEL)                                           # SPEED-ONLY (V0): no --refcv8, label accepted
    assert r.speed_input == "label" and r.enable is False
    _pin(BASE + LABEL + ["--r8-v9-labels", "v9.npz", "--r8-v9-labels-eval", "v9e.npz"])   # the ladder's V0 carries v9


@pytest.mark.parametrize("argv,needle", [
    # label NEEDS its source
    (ON + ["--max-speed-input-v6", "--r8-speed-input", "label"], "without --speed-max-sidecar-v6"),
    (BASE + ["--max-speed-input-v6", "--r8-speed-input", "label"], "without --speed-max-sidecar-v6"),
    (ON + ["--r8-speed-input", "label", "--speed-max-sidecar-v6", "S.jsonl"], "--max-speed-input-v6"),
    (ON + LABEL[:-2] + ["--eval-cache", "E"], "--speed-max-sidecar-v6-eval"),
    # the same knobs as n2
    (ON + LABEL + ["--r8-speed-unknown-p", "0"], "(0, 1)"),
    (ON + LABEL + ["--r8-speed-unknown-p", "1.0"], "(0, 1)"),
    (ON + LABEL + ["--r8-roll-speed-input", "--batch", "1"], "--batch >= 2"),
    # enc8 encodes the N2 ladder; the four road-type steps have nothing finer
    (ON + LABEL + ["--r8-speed-enc8"], "needs --r8-speed-input n2"),
    # the earlier X3 ruling STANDS for the past-only sources and for no source
    (ON + X3 + ["--speed-max-sidecar-v6", "S.jsonl"], "FUTURE-MAX sidecar"),
    (ON + X3 + ["--speed-max-sidecar-v6-eval", "S.jsonl"], "FUTURE-MAX sidecar"),
    (ON + ["--r8-speed-input", "n3", "--max-speed-input-v6", "--r8-v9-labels", "v9.npz",
           "--speed-max-sidecar-v6", "S.jsonl"], "FUTURE-MAX sidecar"),
    (ON + ["--max-speed-input-v6", "--speed-max-sidecar-v6", "S.jsonl"], "FUTURE-MAX sidecar"),
    (BASE + X3 + ["--speed-max-sidecar-v6", "S.jsonl"], "FUTURE-MAX sidecar"),
    (ON + ["--max-speed-input-v6"], "no past-only source"),
])
def test_label_refusals_and_the_n2_n3_refusals_that_stand(argv, needle):
    with pytest.raises(SystemExit) as e:
        _pin(argv)
    assert needle in str(e.value), str(e.value)[:300]


def test_an_unknown_source_value_is_not_a_choice():
    with pytest.raises(SystemExit) as e:                               # argparse: exit code 2
        T.build_parser().parse_args(ON + ["--max-speed-input-v6", "--r8-speed-input", "n4"])
    assert e.value.code == 2
    assert T.build_parser().parse_args(ON + LABEL).r8_speed_input == "label"


def test_DELIBERATE_REGRESSION_without_the_label_pin_a_sidecarless_label_arm_passes(monkeypatch):
    """The refusal above is the helper's: remove it and the sidecar-less label arm is admitted -- a label channel with
    no source (a constant unknown row stamped as the PI's quantized label)."""
    monkeypatch.setattr(T, "_pin_r8_label_source", lambda args: None)
    assert _pin(ON + ["--max-speed-input-v6", "--r8-speed-input", "label"]).speed_input == "label"


def test_the_v6_pin_demands_the_sidecar_for_label_and_not_for_the_past_only_sources():
    """`_pin_refcv6_tactical` (SOURCE pin: a minimal argv cannot pass the rest of `_pin_trainer_cfg`)."""
    import inspect
    src = inspect.getsource(T._pin_refcv6_tactical).replace("\r\n", "\n")
    assert ('if (not getattr(args, "speed_max_sidecar_v6", None)\n'
            '                and getattr(args, "r8_speed_input", None) not in r8train.R8_SPEED_PAST_ONLY):') in src


# =========================================================================== #
# 2. THE PI'S REQUIREMENT: road-type quantization is a REFUSAL at load         #
# =========================================================================== #
def _row(sid, v_hi_ms, *, bin_=None, limit=None, over=False, valid=1, one_hot=None, unknown=False):
    """One sidecar row in the REAL format (refcv6_speed_max_v8_*.jsonl): bin / limit_kmh / one_hot are the containing
    road-type window of ``v_hi_ms``, computed here by hand (km/h: (0,30]->30 (30,50]->50 (50,100]->100 (100,120]->120)."""
    if unknown:
        return {"sid": sid, "clip_id": f"c{sid}", "v_hi_ms": None, "bin": None, "limit_kmh": None,
                "one_hot": [0.0, 0.0, 0.0, 0.0], "over_ceiling": False, "valid": 0}
    kmh = v_hi_ms * 3.6
    b = 0 if kmh <= 30 else 1 if kmh <= 50 else 2 if kmh <= 100 else 3
    b = b if bin_ is None else bin_
    ok = isinstance(b, int) and 0 <= b < 4
    return {"sid": sid, "clip_id": f"c{sid}", "v_hi_ms": v_hi_ms, "bin": b,
            "limit_kmh": (STEPS_KMH[b] if ok else 50) if limit is None else limit,
            "one_hot": [1.0 if i == b else 0.0 for i in range(4)] if one_hot is None else one_hot,
            "over_ceiling": over, "valid": valid}


def _write(tmp_path, rows, name="side.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


GOOD = [_row(1, 3.17), _row(2, 11.19), _row(3, 9.38), _row(4, 20.0), _row(5, 26.0), _row(6, 30.0),
        _row(7, 36.0, over=True), _row(8, 0.0, unknown=True)]
#: v_hi_ms (m/s) -> km/h: 11.4 / 40.3 / 33.8 / 72 / 93.6 / 108 / 129.6 / unknown  =>  bins 30 / 50 / 50 / 100 / 100 / 120 / 120


def test_a_good_sidecar_loads_and_feeds_the_ROAD_TYPE_STEP_never_the_exact_speed(tmp_path):
    by_sid, rep = RT.read_road_type_sidecar(_write(tmp_path, GOOD))
    assert by_sid[1] == (30 / 3.6, 1.0)
    assert by_sid[2] == (50 / 3.6, 1.0) and by_sid[3] == (50 / 3.6, 1.0)
    assert by_sid[4] == (100 / 3.6, 1.0) and by_sid[5] == (100 / 3.6, 1.0)
    assert by_sid[6] == (120 / 3.6, 1.0) and by_sid[7] == (120 / 3.6, 1.0)   # 129.6 km/h clamps to the top step
    assert by_sid[8] == (0.0, 0.0)                                           # the explicit unknown: zeros next to a 0
    exact = {3.17, 11.19, 9.38, 20.0, 26.0, 30.0, 36.0}
    assert not ({v for v, _ok in by_sid.values()} & exact)                   # NO exact future speed reaches the batch
    assert {v for v, ok in by_sid.values() if ok} <= {30 / 3.6, 50 / 3.6, 100 / 3.6, 120 / 3.6}
    assert rep["histogram_kmh"] == {"30": 1, "50": 2, "100": 2, "120": 2}    # the per-step histogram, a literal
    assert rep["n_unknown"] == 1 and rep["n_valid"] == 7 and rep["exact_future_speed_fed"] is False
    assert rep["road_type_kmh"] == [30, 50, 100, 120] and rep["n_over_ceiling"] == 1


@pytest.mark.parametrize("mut,needle", [
    # an exact / finer value in a QUANTIZED field
    (dict(limit=96), "limit_kmh 96 is not one of the road-type values"),
    (dict(limit=100.5), "not one of the road-type values"),
    (dict(limit=100), "not bin 1's road-type value 50"),              # the road-type step of ANOTHER bin
    # no road type at all / a bin outside the four
    (dict(bin_=7), "bin"), (dict(bin_=-1), "bin"), (dict(bin_=1.5), "bin"),
    # a one-hot that is not exactly the bin's
    (dict(one_hot=[0.0, 0.3, 0.7, 0.0]), "not exactly the one-hot of bin 1"),
    (dict(one_hot=[0.0, 1.0, 1.0, 0.0]), "not exactly the one-hot of bin 1"),
])
def test_REFUSED_a_sidecar_row_that_is_not_exactly_a_road_type(tmp_path, mut, needle):
    rows = [dict(r) for r in GOOD]
    rows[1] = _row(2, 11.19, **mut)                                   # 11.19 m/s = 40.3 km/h = bin 1 / 50
    with pytest.raises(SystemExit) as e:
        RT.read_road_type_sidecar(_write(tmp_path, rows))
    assert needle in str(e.value), str(e.value)[:400]


def test_REFUSED_a_sidecar_that_carries_only_the_exact_speed(tmp_path):
    rows = [dict(r) for r in GOOD]
    rows[1] = {"sid": 2, "clip_id": "c2", "v_hi_ms": 11.19, "valid": 1}      # the exact speed and NO road type
    with pytest.raises(SystemExit, match="not a road-type bin"):
        RT.read_road_type_sidecar(_write(tmp_path, rows))
    # the same row with the bin but no limit / one-hot is half a road type: refused too
    rows[1] = {"sid": 2, "clip_id": "c2", "v_hi_ms": 11.19, "valid": 1, "bin": 1}
    with pytest.raises(SystemExit, match="limit_kmh None"):
        RT.read_road_type_sidecar(_write(tmp_path, rows))


def test_REFUSED_an_unknown_row_with_mass_a_duplicate_sid_and_the_sidecar_cross_checks(tmp_path):
    rows = [dict(r) for r in GOOD]
    rows[7] = dict(rows[7], one_hot=[0.0, 1.0, 0.0, 0.0])             # 'unknown' but it says 50 km/h
    with pytest.raises(SystemExit, match="UNKNOWN row"):
        RT.read_road_type_sidecar(_write(tmp_path, rows))
    rows = [dict(r) for r in GOOD] + [_row(2, 9.0)]
    with pytest.raises(SystemExit, match="appears twice"):
        RT.read_road_type_sidecar(_write(tmp_path, rows))
    # the inherited cross-check (the sidecar's bin re-derived from v_hi_ms on the model's ladder) still bites
    rows = [dict(r) for r in GOOD]
    rows[3] = _row(4, 20.0, bin_=1, limit=50)                          # 72 km/h filed under 50
    with pytest.raises(SystemExit, match="Two ladders is two experiments"):
        RT.read_road_type_sidecar(_write(tmp_path, rows))
    with pytest.raises(SystemExit, match="does not exist"):
        RT.read_road_type_sidecar(str(tmp_path / "nope.jsonl"))
    with pytest.raises(SystemExit, match="ZERO rows"):
        RT.read_road_type_sidecar(_write(tmp_path, [], name="empty.jsonl"))


def test_DELIBERATE_REGRESSION_with_the_check_removed_a_finer_speed_sidecar_loads_and_with_the_old_reader_the_exact_speed_is_fed(
        tmp_path, monkeypatch):
    """(a) the guard removed: the dirty sidecar (limit_kmh 96, the exact future speed in the quantized field) LOADS --
    so the refusals above are the guard's, not an accident of the fixture. (b) the OLD reader on a CLEAN sidecar returns
    the EXACT 11.19 m/s: that is what `enable_max_speed_v6` feeds, and why the label source quantizes at load."""
    rows = [dict(r) for r in GOOD]
    rows[1] = _row(2, 11.19, limit=96)
    path = _write(tmp_path, rows)
    with pytest.raises(SystemExit):
        RT.read_road_type_sidecar(path)
    monkeypatch.setattr(RT, "_label_row_refusal", lambda r: None)
    by_sid, _rep = RT.read_road_type_sidecar(path)                      # no refusal: the mutant is caught by the test above
    assert by_sid[2][1] == 1.0
    old, _r = v6ms.read_speed_max_sidecar_v6(_write(tmp_path, GOOD, name="clean.jsonl"))
    assert old[2] == (11.19, 1.0) and old[7] == (36.0, 1.0)             # the v8 path ships the exact value, even 36 m/s


def test_a_value_above_the_top_step_cannot_reach_the_models_over_ceiling_override():
    """`refc_v3._scene_hook` lifts the plan ceiling to the FED value on an over-ceiling row whenever
    `cfg.refcv8.speed_input` is set. Fed the raw 36.0 m/s that would be a 129.6 km/h ceiling (the exact future speed);
    fed the label's step (33.33 m/s = 120 km/h) `over` is False and the ceiling is the road type's."""
    raw_idx, raw_over = v6ms.speed_max_bin_tensor(torch.tensor([36.0]))
    assert raw_idx.tolist() == [3] and raw_over.tolist() == [True]
    fed = torch.tensor([120 / 3.6], dtype=torch.float32)
    idx, over = v6ms.speed_max_bin_tensor(fed)
    assert idx.tolist() == [3] and over.tolist() == [False]
    for k, b in ((30, 0), (50, 1), (100, 2), (120, 3)):                  # every step lands on its own bin, never over
        i, o = v6ms.speed_max_bin_tensor(torch.tensor([k / 3.6], dtype=torch.float32))
        assert i.tolist() == [b] and o.tolist() == [False]


# =========================================================================== #
# 3. the dataset                                                               #
# =========================================================================== #
SID = 515151


def _dataset(row, n_eps=1):
    cfg = T.v3.refc_v3_smoke_config(True)
    eps = T._synth_episodes(n_eps, cfg.core, seed=0, min_frames=140)
    for k, ep in enumerate(eps):
        ep.episode_id = SID + k
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    return ds


def test_the_item_feeds_the_road_type_step_and_the_census_carries_the_histogram(tmp_path):
    ds = _dataset(None)
    path = _write(tmp_path, [_row(SID, 11.19), _row(7, 36.0, over=True)])
    rep = ds.enable_r8_speed_label(path)
    n = len(ds.index)
    assert rep["histogram_kmh"] == {"30": 0, "50": 1, "100": 0, "120": 1}      # per-CLIP rows in the file
    assert rep["window_histogram_kmh"] == {"30": 0, "50": n, "100": 0, "120": 0} and rep["window_n_unknown"] == 0
    assert rep["n_windows"] == n and rep["n_windows_fed"] == n and rep["window_ceiling_frac"] == 1.0
    assert rep["exact_future_speed_fed"] is False
    for i in range(4):
        it = ds[i]
        assert it["v_max_ms"].dtype == torch.float32 and float(it["v_max_valid"]) == 1.0
        assert torch.equal(it["v_max_ms"], torch.tensor(50 / 3.6, dtype=torch.float32))   # the step, to the last bit
        assert not torch.equal(it["v_max_ms"], torch.tensor(11.19, dtype=torch.float32))   # NOT the exact speed


def test_a_window_with_no_sidecar_row_is_the_unknown_row_and_a_foreign_split_refuses(tmp_path):
    ds = _dataset(None, n_eps=2)                                                 # ep 0 = SID (known), ep 1 = SID + 1
    rep = ds.enable_r8_speed_label(_write(tmp_path, [_row(SID, 11.19), _row(SID + 1, 0.0, unknown=True)]))
    n0 = sum(1 for e_i, _t in ds.index if e_i == 0)
    n1 = sum(1 for e_i, _t in ds.index if e_i == 1)
    assert n0 > 0 and n1 > 0 and rep["window_n_unknown"] == n1 and rep["n_windows_fed"] == n0
    assert rep["window_histogram_kmh"] == {"30": 0, "50": n0, "100": 0, "120": 0}
    k1 = next(i for i, (e_i, _t) in enumerate(ds.index) if e_i == 1)
    it = ds[k1]
    assert float(it["v_max_valid"]) == 0.0 and float(it["v_max_ms"]) == 0.0      # zeros next to a 0 (X15)
    ds_none = _dataset(None)                                                     # a split where NOTHING is fed refuses
    with pytest.raises(SystemExit, match="NOT ONE"):
        ds_none.enable_r8_speed_label(_write(tmp_path, [_row(SID, 0.0, unknown=True), _row(9, 5.0)], name="u.jsonl"))
    ds2 = _dataset(None)
    with pytest.raises(SystemExit, match="NOT ONE"):
        ds2.enable_r8_speed_label(_write(tmp_path, [_row(1, 3.0), _row(2, 11.0)], name="other.jsonl"))
    ds3 = _dataset(None)
    ds3.max_speed_enabled = True
    with pytest.raises(SystemExit, match="one channel, one source"):
        ds3.enable_r8_speed_label(_write(tmp_path, GOOD, name="g2.jsonl"))


def test_the_label_dataset_rides_the_same_item_path_as_the_v8_channel_but_with_the_step(tmp_path):
    """Same keys, same dtypes as `enable_max_speed_v6` (so `r8_before_forward` / the model need no second path)."""
    a, b = _dataset(None), _dataset(None)
    a.enable_r8_speed_label(_write(tmp_path, [_row(SID, 11.19)]))
    b.enable_max_speed_v6(_write(tmp_path, [_row(SID, 11.19)], name="v8.jsonl"))
    ia, ib = a[0], b[0]
    assert set(ia) == set(ib)
    assert ia["v_max_ms"].dtype == ib["v_max_ms"].dtype and ia["v_max_valid"].dtype == ib["v_max_valid"].dtype
    assert float(ib["v_max_ms"]) == pytest.approx(11.19, abs=1e-6)               # the v8 path: the exact speed
    assert float(ia["v_max_ms"]) == pytest.approx(50 / 3.6, abs=1e-6)            # the label path: the step


# =========================================================================== #
# 4. the trained unknown row and the roll apply to the label source             #
# =========================================================================== #
def _label_batch(b):
    steps = [30 / 3.6, 50 / 3.6, 100 / 3.6, 120 / 3.6]
    return torch.tensor([steps[i % 4] for i in range(b)], dtype=torch.float32), torch.ones(b)


def _stub(unknown_p=0.45, roll=False, training=True):
    m = types.SimpleNamespace(cfg=types.SimpleNamespace(refcv8=C.R8Config(speed_input="label")), training=training)
    m._r8_speed_unknown_p = unknown_p
    m._r8_roll_speed_train = roll
    return m


def test_the_unknown_row_rate_on_the_label_source_is_p_within_a_binomial_bound():
    n, p = 20_000, 0.45
    vm, vv = _label_batch(n)
    st = torch.get_rng_state()
    m = _stub(p)
    f = RT.speed_only_fwd(m, {"v_max_ms": vm, "v_max_valid": vv}, "cpu")
    assert torch.equal(torch.get_rng_state(), st)                                 # refcv7's global stream untouched
    rate = float((f["v_max_valid"] == 0).float().mean())
    sd = (p * (1 - p) / n) ** 0.5                                                 # 0.003517
    assert abs(rate - p) < 5 * sd, (rate, p, sd)                                  # 5 sigma = 0.0176
    keep = f["v_max_valid"] == 1
    assert torch.equal(f["v_max_ms"][keep], vm[keep])                             # a kept row is its own step, unchanged
    assert float(f["v_max_ms"][~keep].abs().sum()) == 0.0                         # a dropped row is zeros next to a 0
    steps32 = {float(torch.tensor(k / 3.6, dtype=torch.float32)) for k in STEPS_KMH}
    assert {float(x) for x in f["v_max_ms"][keep].unique()} == steps32            # exactly the four road-type steps


def test_the_VSHUF_roll_and_eval_on_the_label_source():
    vm, vv = _label_batch(12)
    plain = RT.speed_only_fwd(_stub(roll=False), {"v_max_ms": vm, "v_max_valid": vv}, "cpu")
    rolled = RT.speed_only_fwd(_stub(roll=True), {"v_max_ms": vm, "v_max_valid": vv}, "cpu")
    assert torch.equal(rolled["v_max_ms"], torch.roll(plain["v_max_ms"], 1, 0))   # another window's step
    assert torch.equal(rolled["v_max_valid"], torch.roll(plain["v_max_valid"], 1, 0))
    ev = RT.speed_only_fwd(_stub(roll=True, training=False), {"v_max_ms": vm, "v_max_valid": vv}, "cpu")
    assert torch.equal(ev["v_max_ms"], vm) and torch.equal(ev["v_max_valid"], vv)  # eval: nothing drawn, own input


# =========================================================================== #
# 5. the route ruling: --r8-no-rc keeps the nav command and the turn distance   #
# =========================================================================== #
V9 = ["--r8-v9-labels", "v9.npz"]


def test_no_rc_keeps_the_nav_token_and_refuses_the_rc_roll():
    r = _pin(ON + V9 + ["--r8-nav-from-v9", "--r8-no-rc"])
    assert r.enable is True
    a = T.build_parser().parse_args(ON + V9 + ["--r8-nav-from-v9", "--r8-no-rc"])
    assert a.r8_no_rc is True and a.r8_nav_from_v9 is True
    assert 0.0 < float(a.r8_nav_args_dropout) < 1.0                    # the turn-distance args stay a trained input
    with pytest.raises(SystemExit, match="and not --r8-no-rc"):
        _pin(ON + V9 + ["--r8-no-rc", "--r8-rc-roll", "--batch", "4"])
    with pytest.raises(SystemExit, match="--r8-nav-from-v9 without --r8-v9-labels"):
        _pin(ON + ["--r8-nav-from-v9", "--r8-no-rc"])


def test_no_rc_gates_ONLY_the_route_checkpoint_in_r8_before_forward():
    """SOURCE pin on the feed: `r8_nav_raw` (d_next / d_end / dyaw / side / lookahead = the nav command's turn distance)
    is fed unconditionally; `r8_rc_raw` is the only branch behind `_r8_no_rc`."""
    import inspect
    src = inspect.getsource(RT.r8_before_forward)
    assert 'if "r8_nav_raw" in batch:' in src and 'fwd["r8_nav"] = scale_nav(nav, known)' in src
    assert 'if "r8_rc_raw" in batch and not bool(getattr(model, "_r8_no_rc", False)):' in src
    assert src.count("_r8_no_rc") == 2                                  # the docstring line and the one guard
    nav_branch = src.split('if "r8_nav_raw" in batch:')[1].split('if "r8_rc_raw" in batch')[0]
    assert "_r8_no_rc" not in nav_branch


def test_the_scaled_nav_input_carries_the_turn_distance_and_the_bit():
    raw = torch.tensor([[100.0, 400.0, 90.0, -1.0, 150.0]])
    out = RT.scale_nav(raw, torch.tensor([True]))
    assert out.shape == (1, 6) and float(out[0, 5]) == 1.0
    assert float(out[0, 0]) == pytest.approx(torch.log1p(torch.tensor(100.0 / 50.0)).item(), abs=1e-6)   # d_next fed
    assert float(out[0, 0]) > 0.0
    off = RT.scale_nav(raw, torch.tensor([False]))
    assert float(off.abs().sum()) == 0.0                                 # unknown distance: exactly zeros next to a 0


# =========================================================================== #
# 6. train() wiring (SOURCE pins: a full train() needs the label blob + manifest the sidecar's md5 is checked against) #
# =========================================================================== #
def _train_src():
    import inspect
    return inspect.getsource(T.train).replace("\r\n", "\n")


def test_train_dispatches_label_to_the_road_type_reader_on_both_splits_and_n2_n3_to_the_v9_column():
    src = _train_src()
    assert "ds.enable_r8_speed_label(str(args.speed_max_sidecar_v6), manifest)" in src
    assert "e_ds.enable_r8_speed_label(_esc, e_man)" in src
    assert src.count('getattr(args, "r8_speed_input", None) in r8train.R8_SPEED_PAST_ONLY:') == 2   # train + eval v9 columns
    assert 'ds.enable_r8_speed(args.r8_speed_input)' in src and 'e_ds.enable_r8_speed(args.r8_speed_input)' in src
    # the n2 / n3 path never reaches the v8 reader and `label` never reaches the v9 column
    assert 'and getattr(args, "r8_speed_input", None) not in r8train.R8_SPEED_PAST_ONLY' in src


def test_train_stamps_both_declarations_for_label_the_oracle_one_and_the_ruling_one():
    src = _train_src()
    # `speed_max_derivation_v6` (the declared ego-future oracle INPUT) rides with `label`, not with n2 / n3
    assert ('if (getattr(args, "max_speed_input_v6", False)\n'
            '                and getattr(args, "r8_speed_input", None) not in r8train.R8_SPEED_PAST_ONLY) else None),') in src
    assert ('and getattr(args, "r8_speed_input", None) not in r8train.R8_SPEED_PAST_ONLY)\n'
            '    r8train.assert_r8_speed_stamp(') in src
    assert '"r8_speed_derivation": r8train.speed_derivation(getattr(args, "r8_speed_input", None),' in src
    import inspect
    assert "PI 2026-10-11 label source" in inspect.getsource(T._seam_stamp)    # the config.json provenance text


def test_label_stamps_satisfy_both_guards_and_n2_stamps_still_refuse_the_v6_oracle_guard():
    v6ms.assert_speed_max_stamp_v6({"speed_max_derivation_v6": v6ms.SPEED_MAX_DERIVATION_V6}, True)   # label: stamped
    RT.assert_r8_speed_stamp({"r8_speed_derivation": RT.speed_derivation("label")}, "label")
    with pytest.raises(v6ms.SpeedMaxStampError):                       # the mirror: a label run that LOST its oracle stamp
        v6ms.assert_speed_max_stamp_v6({}, True)
