"""Tests for the banded map head's decision rule (REFCV9_DESIGN 2.1): the FIT-split threshold fitter (``r9_map_fit_thresholds.py``) and the
thresholds input of ``r9_e1_eval.py`` / ``r9_e1_floor.py`` / ``r9_e1_verdict.py`` / ``r9_e1_dump.py --fit-k-per-ep``.

Literal known answers (the cuts are derived by hand in ``_world``'s comment), an UNSET-identity check, and MUTATION checks: each guard is also run
against a deliberately broken implementation and must go RED (the guard function returns False), so a guard that cannot fail is not trusted."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_e1_dump as D                                   # noqa: E402
import r9_e1_eval as E                                   # noqa: E402
import r9_e1_floor as FL                                 # noqa: E402
import r9_e1_verdict as V                                # noqa: E402
import r9_map_fit_thresholds as F                        # noqa: E402
import refit_perception_thresholds as R8                 # noqa: E402

H_, W_ = 400, 600                                       # 0-40 m at 10 cm, full width: the near-field read's whole domain
BANDS2 = "0-10,10-40"
FIT_CLIPS = ["a1b2c3d4e5f6", "0123456789ab"]
EVAL_SPLIT = {"DEV": {"sha12": ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]}, "HOLD": {"sha12": ["cccccccccccc", "dddddddddddd"]}}


def _put(arr, idx, val):
    arr.reshape(-1)[idx] = val


def _world(path, e_i, ws):
    """One raster, half of the world below on EACH file (so two files pool to the whole). Band 0 = rows 0-99, band 1 = rows 100-399.
    psw band 0: GT-sidewalk(7) 500 cells at level 90 and 500 at 200; GT-other 1500 at level 50, 750 at 90, 50 at 200.
        IoU(t): t<=50 -> 1000/3300 = 0.303; 51..90 -> 1000/1800 = 0.5556; 91..200 -> 500/1050 = 0.476; >200 -> 0.   BEST = level 51 (first of the tie).
    psw band 1: GT-sidewalk 1000 cells at 150; GT-other 250 at 150, 2000 at 30.  IoU: t<=30 -> 1000/3250; 31..150 -> 1000/1250 = 0.8; >150 -> 0.   BEST = 31.
    pedge (both bands): GT-edge(5) 150 cells at 20; GT-other 50 at 20 and 250 at 2.  IoU: t<=2 -> 150/450; 3..20 -> 150/200 = 0.75.   BEST = 3.
    Everything else: GT 0 (nocls, a valid negative) with level 0."""
    psw = np.zeros((H_, W_), np.uint8)
    pedge = np.zeros((H_, W_), np.uint8)
    pdrv = np.zeros((H_, W_), np.uint8)
    gt = np.zeros((H_, W_), np.uint8)
    gt[:, 590:] = 255                                    # an unobserved strip: must never be counted
    b0, b1 = np.arange(0, 100 * W_).reshape(100, W_)[:, :590].reshape(-1), np.arange(100 * W_, 400 * W_).reshape(300, W_)[:, :590].reshape(-1)

    def cells(pool, start, n):
        return pool[start:start + n]
    # psw band 0
    s = 0
    for n, lvl, cls in ((500, 90, 7), (500, 200, 7), (1500, 50, 0), (750, 90, 0), (50, 200, 0)):
        ix = cells(b0, s, n)
        _put(psw, ix, lvl)
        _put(gt, ix, cls)
        s += n
    # psw band 1
    s = 0
    for n, lvl, cls in ((1000, 150, 7), (250, 150, 0), (2000, 30, 0)):
        ix = cells(b1, s, n)
        _put(psw, ix, lvl)
        _put(gt, ix, cls)
        s += n
    # pedge (own cells, away from the psw ones: the last cells of each band's pool)
    for pool in (b0, b1):
        s = len(pool) - 1000
        for n, lvl, cls in ((150, 20, 5), (50, 20, 0), (250, 2, 0)):
            ix = cells(pool, s, n)
            _put(pedge, ix, lvl)
            _put(gt, ix, cls)
            s += n
    np.savez_compressed(path, pdrv=pdrv, pedge=pedge, psw=psw, code=np.zeros_like(psw), gt=gt)


def _dump(tmp_path, n_files=2, set_name="FIT", clips=None, smoke=False, ckpt="ck1"):
    """A FIT-dump directory: <d>/rast/e*_w*.npz + <d>/dump.json (the fields the fitter reads)."""
    d = tmp_path / "dump"
    (d / "rast").mkdir(parents=True)
    names = []
    for k in range(n_files):
        nm = "e%03d_w%04d.npz" % (k, 7)
        _world(d / "rast" / nm, k, 7)
        names.append(nm)
    meta = {"split": {"set": set_name, "n_clips": 2}, "raster_names_sha256": hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest(),
            "ckpt_md5": ckpt, "ckpt_kind": "ema_model_only", "step": 4000, "weights_are_ema": True, "config_md5": "cfg", "order_digest": "od", "SMOKE_RESTRICTED": smoke}
    if set_name == "FIT":
        meta["fit"] = {"k_per_ep": 8, "n_clips": 2, "clips_sha12": clips or FIT_CLIPS, "rule": "r"}
    (d / "dump.json").write_text(json.dumps(meta), encoding="utf-8")
    sp = tmp_path / "split.json"
    sp.write_text(json.dumps(EVAL_SPLIT), encoding="utf-8")
    return d, sp


def _fit(d, sp, out, **kw):
    return F.fit(d / "rast", sp, out, "T", bands_s=kw.pop("bands_s", BANDS2), **kw)


# ============================================================================================================== the fitter
def test_fit_recovers_the_known_best_cut_per_plane_and_band(tmp_path):
    d, sp = _dump(tmp_path)
    obj = _fit(d, sp, tmp_path / "thr.json")
    assert obj["planes"] == {"psw": [51, 31], "pedge": [3, 3]}                       # derived by hand in _world's docstring
    det = {pl: obj["provenance"]["fit_detail"][pl] for pl in ("psw", "pedge")}
    assert det["psw"][0]["iou_at_level"] == pytest.approx(1000 / 1800) and det["psw"][1]["iou_at_level"] == pytest.approx(0.8)
    assert det["pedge"][0]["iou_at_level"] == pytest.approx(0.75)
    assert det["psw"][0]["n_pos"] == 2 * 1000 and det["psw"][1]["n_pos"] == 2 * 1000       # the 255 strip is not counted, two files pooled
    # 128 is the WORSE cut here (band 0: flags the 200-level cells only: 1000 TP of 1000+100 flags, 2000 GT -> 1000/2100 = 0.476)
    assert det["psw"][0]["iou_at_128"] == pytest.approx(1000 / 2100) and det["psw"][0]["iou_at_128"] < det["psw"][0]["iou_at_level"]
    back = E.load_map_thresholds(tmp_path / "thr.json")                               # the evaluator's own loader reads the artifact
    assert back.planes == obj["planes"] and back.level("psw", 0, 5) == 51 and back.level("psw", 10, 20) == 31 and back.level("pdrv", 0, 5) == 128
    assert obj["provenance"]["ckpt_md5"] == "ck1" and obj["provenance"]["fit"]["disjoint_from_dev_and_hold"] is True


def test_the_criterion_is_the_8way_machinerys_not_a_copy():
    assert F.best_threshold is R8.best_threshold                                       # imported and called, not re-implemented
    rng = np.random.default_rng(0)
    neg, pos = rng.integers(0, 50, 256).astype(float), rng.integers(0, 50, 256).astype(float)
    pos[0] = neg[0] = 0.0
    i, iou = R8.best_threshold(neg, pos)
    assert F.level_from_hist(neg, pos) == (max(i, 1), iou if i else iou)
    assert F.level_from_hist(neg, pos)[1] == F.iou_at(neg, pos, F.level_from_hist(neg, pos)[0])


def test_level_zero_is_never_a_cut():
    neg, pos = np.zeros(256), np.zeros(256)
    pos[0], neg[0] = 10, 1                                                              # a useless plane: everything sits at level 0
    with pytest.raises(F.FitRefused, match="level 0"):
        F.level_from_hist(neg, pos)
    with pytest.raises(F.FitRefused, match="not fittable"):
        F.level_from_hist(np.ones(256), np.zeros(256))
    neg, pos = np.zeros(256), np.zeros(256)
    pos[5] = 10                                                                         # no cell at level 0: level 1 and level 0 are the same mask -> 1
    assert F.level_from_hist(neg, pos)[0] == 1


@pytest.mark.parametrize("what", ["not_fit", "clash", "wrong_dir", "smoke", "no_dump", "no_split"])
def test_fit_refuses_anything_that_is_not_a_clean_fit_dump(tmp_path, what):
    clips = FIT_CLIPS + (["bbbbbbbbbbbb"] if what == "clash" else [])
    d, sp = _dump(tmp_path, set_name="DEV" if what == "not_fit" else "FIT", clips=clips, smoke=(what == "smoke"))
    if what == "wrong_dir":
        _world(d / "rast" / "e009_w0009.npz", 9, 9)                                     # a raster the dump never wrote
    if what == "no_dump":
        (d / "dump.json").unlink()
    if what == "no_split":
        sp.unlink()
    with pytest.raises(F.FitRefused):
        _fit(d, sp, tmp_path / "thr.json")
    assert not (tmp_path / "thr.json").exists()                                         # a refusal leaves no artifact
    rc = F.main(["fit", "--rast-dir", str(d / "rast"), "--dev-hold", str(sp), "--arm", "T", "--bands", BANDS2, "--out", str(tmp_path / "thr2.json")])
    assert rc == 2 and not (tmp_path / "thr2.json").exists()


def test_fit_planes_and_population_options(tmp_path):
    d, sp = _dump(tmp_path)
    only = _fit(d, sp, tmp_path / "a.json", planes=("psw",))
    assert list(only["planes"]) == ["psw"] and E.load_map_thresholds(tmp_path / "a.json").level("pedge", 0, 5) == 128    # an unfitted plane keeps 128
    with pytest.raises(F.FitRefused):
        _fit(d, sp, tmp_path / "b.json", planes=("pdrv",))                              # no drivable GT in this world: refuses, never an arbitrary cut
    cor = _fit(d, sp, tmp_path / "c.json", population="corridor")                       # |y| < 3 m only: the world's cells sit in rows, spread over all columns
    assert cor["criterion"]["population"] == "corridor"


# ============================================================================================================== the reader
def _thr_file(tmp_path, planes, bands=((0, 10), (10, 40)), name="t.json", ckpt=None):
    obj = {"schema": E.THR_SCHEMA, "planes": planes, "bands_m": [list(b) for b in bands]}
    if ckpt:
        obj["provenance"] = {"ckpt_md5": ckpt}
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _np_counts(files, t_sw, t_ed, bands=E.NEARFIELD_BANDS):
    """Independent numpy: per band [flags, flags on GT sidewalk] summed over files, flag psw >= t_sw[band], union adds pedge >= t_ed[band]."""
    out = np.zeros((len(bands), 4))
    c0, c1 = E.CORRIDOR_COLS
    for f in files:
        z = np.load(f)
        for i, (lo, hi) in enumerate(bands):
            sl = (slice(lo * 10, hi * 10), slice(c0, c1))
            seen = z["gt"][sl] != 255
            sw = z["gt"][sl] == 7
            p = (z["psw"][sl] >= t_sw[i]) & seen
            u = ((z["psw"][sl] >= t_sw[i]) | (z["pedge"][sl] >= t_ed[i])) & seen
            out[i] += [p.sum(), (p & sw).sum(), u.sum(), (u & (sw | (z["gt"][sl] == 5))).sum()]
    return out


def _files(tmp_path, n=3):
    d = tmp_path / "rasters"
    d.mkdir()
    for k in range(n):
        _world(d / ("e%03d_w%04d.npz" % (k, 7)), k, 7)
    return sorted(str(p) for p in d.glob("*.npz"))


def test_unset_path_is_identical_and_a_128_file_is_the_identity(tmp_path):
    files = _files(tmp_path)
    n0, a0, g0, i0 = E.nearfield_per_file_ex(files)
    assert "thresholds" not in i0                                                       # an unset read carries no stamp: its info is what it was
    ref = _np_counts(files, [128] * 4, [128] * 4)
    assert np.array_equal(a0[:, :, [0, 1]].sum(0), ref[:, :2]) and np.array_equal(a0[:, :, [6, 7]].sum(0), ref[:, 2:])
    ident = E.load_map_thresholds(_thr_file(tmp_path, {"psw": [128, 128], "pedge": [128, 128]}))
    n1, a1, g1, i1 = E.nearfield_per_file_ex(files, thresholds=ident)
    assert n0 == n1 and np.array_equal(a0, a1, equal_nan=True) and np.array_equal(g0, g1) and i1["thresholds"]["sha256"] == ident.sha256


def _flags_follow_the_file(tmp_path, files):
    """The guard under test: a thresholds file must change the flags exactly as an independent numpy read at those levels says."""
    thr = E.load_map_thresholds(_thr_file(tmp_path, {"psw": [200, 60], "pedge": [10, 128]}, name="f.json"))
    _n, a, _g, info = E.nearfield_per_file_ex(files, thresholds=thr)
    want = _np_counts(files, [200, 200, 60, 60], [10, 10, 128, 128])
    got = np.stack([a[:, i, :].sum(0) for i in range(4)])
    return (np.array_equal(got[:, [0, 1]], want[:, :2]) and np.array_equal(got[:, [6, 7]], want[:, 2:])
            and info["thresholds"]["applied"]["0-5"] == {"psw": 200, "pedge": 10} and info["thresholds"]["applied"]["20-40"] == {"psw": 60, "pedge": 128}
            and not np.array_equal(a, E.nearfield_per_file_ex(files)[1]))


def test_a_thresholds_file_changes_the_flags_as_an_independent_read_says(tmp_path):
    assert _flags_follow_the_file(tmp_path, _files(tmp_path)) is True


def test_MUTATION_an_eval_that_ignores_the_file_makes_the_guard_red(tmp_path, monkeypatch):
    files = _files(tmp_path)
    monkeypatch.setattr(E.MapThresholds, "level", lambda self, plane, lo, hi: E.FIXED_LEVEL)          # mutant 1: the cut is always 128
    assert _flags_follow_the_file(tmp_path, files) is False
    monkeypatch.undo()
    orig = E.nearfield_per_file_ex
    monkeypatch.setattr(E, "nearfield_per_file_ex", lambda *a, **k: orig(*a, **dict(k, thresholds=None)))   # mutant 2: the argument is dropped on the way
    assert _flags_follow_the_file(tmp_path, files) is False


def test_cli_stamps_the_thresholds_and_unset_cli_has_no_key(tmp_path):
    files = _files(tmp_path)
    d = Path(files[0]).parent
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_r9_e1_eval as TE
    import pickle
    pk = tmp_path / "perc_det.pkl"
    pk.write_bytes(pickle.dumps([{"sha12": s, "ws": w, "box3d": [p]} for (s, w, p) in TE._records(n_clips=3)]))
    base = ["--packs", str(pk), "--rast-dir", str(d), "--gate", "0.5", "--boot", "20", "--seed", "0"]
    assert E.main(base + ["--out", str(tmp_path / "u.json")]) == 0
    tf = _thr_file(tmp_path, {"psw": [60, 200]}, name="cli.json")
    assert E.main(base + ["--map-thresholds", str(tf), "--out", str(tmp_path / "t.json")]) == 0
    u, t = (json.loads((tmp_path / n).read_text())["nearfield"] for n in ("u.json", "t.json"))
    assert "thresholds" not in u and t["thresholds"]["sha256"] == hashlib.sha256(tf.read_bytes()).hexdigest()
    assert t["thresholds"]["applied"]["0-5"]["psw"] == 60 and "flag_rule" in t["definition"] and "flag_rule" not in u["definition"]
    assert t["table"]["0-5"]["flag_rate"] != u["table"]["0-5"]["flag_rate"]
    with pytest.raises(SystemExit, match="needs --rast-dir"):
        E.main(["--packs", str(pk), "--gate", "0.5", "--map-thresholds", str(tf), "--out", str(tmp_path / "x.json")])


def test_loader_refusals_straddle_and_binding(tmp_path):
    for bad in ({"psw": [0, 5]}, {"psw": [256, 5]}, {"psw": [5]}, {"psw": [True, 5]}, {}):
        with pytest.raises(E.ThresholdsError):
            E.MapThresholds(bad, [[0, 10], [10, 40]], "x", "x")
    with pytest.raises(E.ThresholdsError, match="schema"):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"schema": "nope", "planes": {"psw": [5]}, "bands_m": [[0, 10]]}))
        E.load_map_thresholds(p)
    t = E.MapThresholds({"psw": [10, 200]}, [[0, 7], [7, 40]], "x", "x")             # a read band 5-10 m straddles 0-7 / 7-40 with different levels
    with pytest.raises(E.ThresholdsError, match="straddles"):
        t.level("psw", 5, 10)
    assert E.MapThresholds({"psw": [10, 10]}, [[0, 7], [7, 40]], "x", "x").level("psw", 5, 10) == 10       # ...but agreeing levels are not ambiguous
    with pytest.raises(E.ThresholdsError, match="outside"):
        E.MapThresholds({"psw": [10]}, [[0, 10]], "x", "x").level("psw", 10, 20)
    d, _sp = _dump(tmp_path / "dd" if (tmp_path / "dd").mkdir() is None else tmp_path)
    ok = E.load_map_thresholds(_thr_file(tmp_path, {"psw": [5, 5]}, name="ok.json", ckpt="ck1"))
    assert E.thresholds_binding(ok, d / "rast") == {"checked": True, "ckpt_md5": "ck1"}
    other = E.load_map_thresholds(_thr_file(tmp_path, {"psw": [5, 5]}, name="other.json", ckpt="ck2"))
    with pytest.raises(E.ThresholdsError, match="bound to its own model"):
        E.thresholds_binding(other, d / "rast")
    assert E.thresholds_binding(E.load_map_thresholds(_thr_file(tmp_path, {"psw": [5, 5]}, name="nb.json")), d / "rast")["checked"] is False


# ============================================================================================================== floor and verdict
def _two_arms(tmp_path):
    a = tmp_path / "A"
    b = tmp_path / "B"
    a.mkdir()
    b.mkdir()
    for k in range(3):
        _world(a / ("e%03d_w%04d.npz" % (k, 7)), k, 7)
        _world(b / ("e%03d_w%04d.npz" % (k, 7)), k, 7)
    return a, b


def test_floor_reads_each_side_on_its_own_cut_and_stamps_it(tmp_path):
    a, b = _two_arms(tmp_path)
    t0, t1 = _thr_file(tmp_path, {"psw": [51, 31]}, name="t0.json"), _thr_file(tmp_path, {"psw": [60, 60]}, name="t1.json")
    rows_u, meta_u = FL.nearfield_rows(a, b, boot=20)
    assert "thresholds" not in meta_u and rows_u                                         # unset: map_inputs has no `thresholds`
    rows, meta = FL.nearfield_rows(a, b, boot=20, thr0=t0, thr1=t1)
    assert meta["thresholds"]["c0"]["sha256"] == hashlib.sha256(t0.read_bytes()).hexdigest() and meta["thresholds"]["c1"]["applied"]["10-20"]["psw"] == 60
    assert rows["nf_prec_0_5"]["c0"] != rows_u["nf_prec_0_5"]["c0"] and rows["nf_prec_0_5"]["c1"] != rows_u["nf_prec_0_5"]["c1"]
    probs = []
    r, _m = FL.nearfield_rows(a, b, boot=20, thr0=t0, problems=probs)                   # one side only
    assert r == {} and any("ONE side only" in p for p in probs)


def _prot(tv, tf, e_arm=None, e_ref=None, ref="M9"):
    rm = {"thresholds": tv} if tv else {}
    floor = {"map_inputs": {"thresholds": tf} if tf else {}}
    out, checks, problems = {}, [], []
    V._thresholds_protocol(out, checks, problems, floor, rm, e_arm or {}, e_ref or {}, ref)
    return out, checks, problems


def test_verdict_refuses_a_floor_on_another_threshold_protocol():
    rec = lambda s: {"sha256": s}                                                       # noqa: E731
    both = {"c0": rec("r" * 64), "c1": rec("a" * 64)}
    assert _prot(None, None) == ({}, [], [])                                              # fixed vs fixed: nothing added, nothing checked
    out, checks, problems = _prot(both, both)
    assert out["map_thresholds"]["same_protocol"] is True and problems == []
    _o, _c, problems = _prot(both, None)                                                  # arm on FIT cuts, floor at 128
    assert problems and "map-threshold protocol" in problems[0]
    _o, _c, problems = _prot(None, both)                                                  # arm at 128, floor on FIT cuts
    assert problems and "map-threshold protocol" in problems[0]
    # the single evals must carry the cut the verdict was given
    e_ok = {"nearfield": {"thresholds": {"sha256": "a" * 64}}}
    _o, _c, problems = _prot(both, both, e_arm=e_ok)
    assert problems == []
    _o, _c, problems = _prot(both, both, e_arm={"nearfield": {"thresholds": {"sha256": "f" * 64}}})
    assert problems and "single eval" in problems[0]
    _o, _c, problems = _prot(both, both, e_arm={"nearfield": {}})                         # an eval read at 128 judged against a thresholded arm
    assert problems
    # a C0 reference must be cut by the very file the floor's C0 was
    fl = {"c0": rec("0" * 64), "c1": rec("1" * 64)}
    _o, _c, problems = _prot(both, fl, ref="C0")
    assert problems and "reference is cut by the file" in problems[0]
    assert _prot(both, {"c0": rec("r" * 64), "c1": rec("1" * 64)}, ref="C0")[2] == []


def test_end_to_end_floor_then_verdict_on_one_threshold_protocol_and_the_refusals(tmp_path):
    """The REAL eval -> floor -> verdict chain on the verdict suite's own rig (stub dumps), each arm on its OWN cut."""
    import test_r9_e1_verdict as TV
    r = TV.Rig(tmp_path)
    r.arm("B1", {"miss": [7, 8, 9]})
    thr = {n: _thr_file(tmp_path, {"psw": lv, "pedge": lv}, name="t_%s.json" % n) for n, lv in (("C0", [100, 100]), ("C1", [110, 110]), ("B1", [90, 90]))}
    com = TV._com(r.split)

    def ev(name, d, t, lv=None):
        out = tmp_path / ("EVALT_%s.json" % name)
        assert E.main(["--packs", str(d / "perc_det.pkl"), "--rast-dir", str(d / "rast"), *com, "--map-thresholds", str(thr[t]), "--out", str(out)]) == 0
        return out
    e0t, e1t, ebt = ev("C0", r.dumps["C0"], "C0"), ev("C1", r.dumps["C1"], "C1"), ev("B1", r.dumps["B1"], "B1")
    pair = TV._eval_pair(tmp_path, "C1", "C0", r.dumps["C1"], r.dumps["C0"], r.split)
    fl_args = lambda a, b, d: ["--eval-c0", str(a), "--eval-c1", str(b), "--paired", str(pair), "--rast-c0", str(r.dumps["C0"] / "rast"),   # noqa: E731
                                 "--rast-c1", str(r.dumps["C1"] / "rast"), "--boot", TV.BOOT, "--out-dir", str(d)]
    # the floor: refused when its single evals were read at 128 but the floor is cut; one-sided thresholds refused; both sides cut -> OK
    assert FL.main(fl_args(r.eval0, r.eval1, tmp_path / "f_bad") + ["--map-thresholds-c0", str(thr["C0"]), "--map-thresholds-c1", str(thr["C1"])]) == 2
    assert "map thresholds" in " ".join(json.loads((tmp_path / "f_bad" / "FLOOR.json").read_text())["problems"])
    assert FL.main(fl_args(e0t, e1t, tmp_path / "f_one") + ["--map-thresholds-c0", str(thr["C0"])]) == 2
    assert FL.main(fl_args(e0t, e1t, tmp_path / "f_ok") + ["--map-thresholds-c0", str(thr["C0"]), "--map-thresholds-c1", str(thr["C1"])]) == 0
    floor_t = tmp_path / "f_ok" / "FLOOR.json"
    fobj = json.loads(floor_t.read_text())
    assert fobj["map_inputs"]["thresholds"]["c0"]["sha256"] == hashlib.sha256(thr["C0"].read_bytes()).hexdigest()
    assert "thresholds" not in json.loads(r.floor.read_text())["map_inputs"]                  # the fixed floor is unchanged by all of this
    assert "OWN FIT-split thresholds" in (tmp_path / "f_ok" / "FLOOR.md").read_text()

    def verdict(floor, e_arm, e_ref, e_c0, arm_t, ref_t, out):
        saved = (r.evals["B1"], r.evals["C0"], r.eval0)
        r.evals["B1"], r.evals["C0"], r.eval0 = e_arm, e_ref, e_c0
        try:
            extra = (["--map-thresholds-arm", str(thr[arm_t])] if arm_t else []) + (["--map-thresholds-ref", str(thr[ref_t])] if ref_t else [])
            return r.verdict("B1", "C0", floor=floor, extra=extra, out=out)
        finally:
            r.evals["B1"], r.evals["C0"], r.eval0 = saved
    rc, v, md = verdict(floor_t, ebt, e0t, e0t, "B1", "C0", tmp_path / "v_ok")
    assert rc == 0 and v["status"] == "OK" and v["map_thresholds"]["same_protocol"] is True and "OWN FIT-split thresholds" in md
    assert v["map_thresholds"]["arm"]["sha256"] == hashlib.sha256(thr["B1"].read_bytes()).hexdigest()
    # the REFUSALS: a floor read at 128 judging a thresholded arm; a thresholded floor judging a 128 arm; one-sided; a C0 cut by another file
    rc, v, _ = verdict(r.floor, ebt, e0t, r.eval0, "B1", "C0", tmp_path / "v_r1")
    assert rc == 2 and v["status"] == "REFUSED" and "map-threshold protocol" in " ".join(v["problems"])
    rc, v, _ = verdict(floor_t, r.evals["B1"], r.evals["C0"], e0t, None, None, tmp_path / "v_r2")
    assert rc == 2 and "map-threshold protocol" in " ".join(v["problems"])
    rc, v, _ = verdict(floor_t, ebt, e0t, e0t, "B1", None, tmp_path / "v_r3")
    assert rc == 2 and "go together" in " ".join(v["problems"])
    rc, v, _ = verdict(floor_t, ebt, ev("C0x", r.dumps["C0"], "B1"), e0t, "B1", "B1", tmp_path / "v_r4")      # the reference is cut by B1's file, the floor's C0 by C0's
    assert rc == 2 and "reference is cut by the file" in " ".join(v["problems"])
    rc, v, _ = verdict(floor_t, r.evals["B1"], e0t, e0t, "B1", "C0", tmp_path / "v_r5")                      # the arm's single eval was read at 128
    assert rc == 2 and "single eval" in " ".join(v["problems"])


def test_floor_cli_flags_reach_the_floor():
    ap_src = Path(FL.__file__).read_text(encoding="utf-8")
    assert "thr_c0=a.map_thresholds_c0, thr_c1=a.map_thresholds_c1" in ap_src and "thr0=thr_c0, thr1=thr_c1" in ap_src
    vs = Path(V.__file__).read_text(encoding="utf-8")
    assert "thr_arm=a.map_thresholds_arm, thr_ref=a.map_thresholds_ref" in vs
    ev = Path(E.__file__).read_text(encoding="utf-8")
    assert "thresholds=thr_m" in ev and "ThresholdsError" in ev


# ============================================================================================================== the FIT dump
def _run_diag_pick(index, sha_of_ei, K):
    """VERBATIM arithmetic of run_diag.py (2026-10-21-refcv8-perc8/code/run_diag.py, the `else` branch of the window-set block)."""
    by_ep = {}
    for i, (e_i, t) in enumerate(index):
        by_ep.setdefault(int(e_i), []).append((int(t), i))
    idx = []
    for e_i in sorted(by_ep, key=lambda e: sha_of_ei[int(e)]):
        ts = sorted(by_ep[e_i])
        for j in range(K):
            idx.append(ts[int((j + 0.5) * len(ts) / K)][1])
    return idx


def test_fit_window_order_is_run_diags_picker_restricted_to_the_fit_clips():
    clips = ["clip-%d" % k for k in range(6)]
    sha = lambda c: hashlib.sha256(str(c).encode()).hexdigest()[:12]                    # noqa: E731
    lens = [37, 36, 50, 21, 36, 44]
    index = [(e, t) for e in range(6) for t in range(lens[e])]
    W = 5
    for K in (1, 3, 8):
        full = _run_diag_pick(index, {e: sha(c) for e, c in enumerate(clips)}, K)
        got = D.fit_window_order(index, clips, sha, {sha(c) for c in clips}, K, W)
        assert [i for (_c, _w, i) in got] == full                                        # identical windows, identical order
        assert all(w == index[i][1] + W - 1 and c == clips[index[i][0]] for (c, w, i) in got)
        want = {sha(clips[1]), sha(clips[4])}
        sub = D.fit_window_order(index, clips, sha, want, K, W)
        assert len(sub) == 2 * K and {sha(c) for (c, _w, _i) in sub} == want
        assert [i for (_c, _w, i) in sub] == [i for (c, _w, i) in got if sha(c) in want]
    one = D.fit_window_order(index, clips, sha, {sha(clips[3])}, 8, W)                  # 21 starts, 8 picks: int((j + 0.5) * 21 / 8)
    assert [index[i][1] for (_c, _w, i) in one] == [int((j + 0.5) * 21 / 8) for j in range(8)]


def _fit_stub_run(monkeypatch, tmp_path, windows, split_p_for_fit=None, extra=()):
    import test_r9_e1_dump as TD
    st = TD._install_stub(monkeypatch, tmp_path, True)
    ids = list(reversed(st.clips))
    wl = tmp_path / "fit_clips.json"
    wl.write_text(json.dumps(windows(st)), encoding="utf-8")
    ex = tmp_path / "evalsplit.json"
    ex.write_text(json.dumps(EVAL_SPLIT), encoding="utf-8")
    out = tmp_path / "fitdump"
    argv = ["--ckpt", str(st.ck_p), "--config", str(st.cfg_p), "--windows", str(wl), "--exclude-split", str(split_p_for_fit or ex), "--fit-k-per-ep", "4",
            "--s2-windows", str(tmp_path / "does_not_exist.json"), "--out", str(out), "--device", "cpu", *extra]
    return st, ids, out, ex, argv


def test_fit_dump_end_to_end_on_the_stub_loader_and_the_fitter_accepts_it(monkeypatch, tmp_path):
    import test_r9_e1_dump as TD
    pick = lambda st: sorted(TD._sha12(c) for c in st.clips[:2])                          # noqa: E731
    st, ids, out, ex, argv = _fit_stub_run(monkeypatch, tmp_path, pick)
    assert D.main(argv) == 0                                                              # --s2-windows names a file that does not exist: FIT mode never reads it
    meta = json.loads((out / "dump.json").read_text(encoding="utf-8"))
    want = pick(st)
    assert meta["split"]["set"] == "FIT" and meta["fit"]["clips_sha12"] == want and meta["fit"]["k_per_ep"] == 4 and meta["s2_grid_sha256"] is None
    assert meta["fit"]["disjoint_from_dev_and_hold"] is True and meta["n_windows"] == 8 == meta["n_rasters"] and meta["SMOKE_RESTRICTED"] is False
    names = sorted(p.name for p in (out / "rast").glob("*.npz"))
    assert len(names) == 8 and {n.split("_")[0] for n in names} == {"e%03d" % ids.index(c) for c in st.clips[:2]}
    ts = list(range(36))                                                                  # the stub's dataset: 36 starts per episode (T 60, W 5, horizon 20)
    ws_want = sorted(t + 5 - 1 for t in (int((j + 0.5) * 36 / 4) for j in range(4)))
    assert sorted({int(n.split("_w")[1][:4]) for n in names}) == ws_want and ts
    # the contract between the dump and the fitter: it accepts this dump against a split that does not hold its clips...
    rec = F.verify_fit_dump(meta, out / "rast", ex)
    assert rec["disjoint_from_dev_and_hold"] is True and rec["n_clips"] == 2
    # ...and refuses it against a split that does
    with pytest.raises(F.FitRefused, match="never fitted on eval"):
        F.verify_fit_dump(meta, out / "rast", st.split_p)


def test_fit_dump_refuses_eval_clips_a_missing_exclude_and_a_split_flag(monkeypatch, tmp_path):
    import test_r9_e1_dump as TD
    st, ids, out, ex, argv = _fit_stub_run(monkeypatch, tmp_path, lambda st: sorted(TD._sha12(c) for c in st.clips[:2]))
    argv_clash = list(argv)
    argv_clash[argv_clash.index("--exclude-split") + 1] = str(st.split_p)                 # the real DEV / HOLD of the stub contain these clips
    with pytest.raises(SystemExit, match="never reads eval"):
        D.main(argv_clash)
    with pytest.raises(SystemExit, match="needs --windows"):
        D.main(argv[:argv.index("--exclude-split")] + argv[argv.index("--exclude-split") + 2:])
    with pytest.raises(SystemExit, match="takes a plain sha12 list"):
        D.main(argv + ["--split", "dev"])
    assert not (out / "dump.json").exists()


def test_fit_flag_default_is_off_and_the_argparse_flag_exists():
    a = D.build_parser().parse_args(["--ckpt", "c", "--config", "g", "--out", "o"])
    assert a.fit_k_per_ep == 0 and a.exclude_split is None
    assert D.build_parser().parse_args(["--ckpt", "c", "--config", "g", "--out", "o", "--fit-k-per-ep", "8"]).fit_k_per_ep == 8


# ============================================================================================================== r9e1_arm_read.sh (dry run, the arm-read suite's own rig)
from test_r9_e1_arm_read import dry, floor_template                       # noqa: E402,F401  (fixtures)
import test_r9_e1_arm_read as TA                                          # noqa: E402


def _thr_floor(dry_, tmp_path, thr_c0, thr_c1):
    """A threshold-protocol floor dir (FLOOR.json + EVAL_C0.json) from the dry rig's own C0 / C1 dumps."""
    fd = tmp_path / "thrfloor"
    fd.mkdir()
    com = ["--windows", dry_.split.as_posix(), "--split", "dev", "--gate", "0.5", "--boot", TA.BOOT, "--seed", "0"]
    for name, t in (("C0", thr_c0), ("C1", thr_c1)):
        d = dry_.rd / ("dump_" + name)
        assert E.main(["--packs", str(d / "perc_det.pkl"), "--rast-dir", str(d / "rast"), *com, "--map-thresholds", str(t), "--out", str(fd / ("EVAL_%s.json" % name))]) == 0
    assert E.main(["--packs", str(dry_.rd / "dump_C1" / "perc_det.pkl"), "--packs-b", str(dry_.rd / "dump_C0" / "perc_det.pkl"), *com,
                   "--out", str(fd / "EVAL_C1_vs_C0.json")]) == 0                                    # the paired (box-only) eval, re-made on THIS rig's dump paths
    assert FL.main(["--eval-c0", str(fd / "EVAL_C0.json"), "--eval-c1", str(fd / "EVAL_C1.json"), "--paired", str(fd / "EVAL_C1_vs_C0.json"),
                    "--rast-c0", str(dry_.rd / "dump_C0" / "rast"), "--rast-c1", str(dry_.rd / "dump_C1" / "rast"), "--boot", TA.BOOT,
                    "--map-thresholds-c0", str(thr_c0), "--map-thresholds-c1", str(thr_c1), "--out-dir", str(fd)]) == 0
    return fd


def test_arm_read_thresholds_set_goes_to_both_arm_evals_and_the_verdict_and_never_touches_the_fixed_read(dry, tmp_path):
    t = {n: _thr_file(tmp_path, {"psw": lv, "pedge": lv}, name="t_%s.json" % n) for n, lv in (("C0", [100, 100]), ("C1", [110, 110]), ("B4", [90, 90]), ("B2", [95, 95]))}
    fd = _thr_floor(dry, tmp_path, t["C0"], t["C1"])
    env, log = TA._recorder(dry)
    TA._read_b4(dry, R9E1_MAP_THR_ARM=t["B4"].as_posix(), R9E1_MAP_THR_REF=t["B2"].as_posix(), R9E1_MAP_THR_FLOOR_DIR=fd.as_posix(), **env)
    ev, vd = TA._calls(log)
    flags = lambda toks: [x for x in toks if x.startswith("--")]                                  # noqa: E731
    with_thr = lambda fl: fl[:fl.index("--seed") + 1] + ["--map-thresholds"] + fl[fl.index("--seed") + 1:]            # noqa: E731
    assert [flags(x) for x in ev] == [with_thr(TA.EVAL_FLAGS), with_thr(TA.EVAL_FLAGS), TA.PAIRED_FLAGS], ev          # arm + reference singles cut; the (box-only) paired eval is not
    assert ev[0][ev[0].index("--map-thresholds") + 1].endswith("t_B4.json") and ev[1][ev[1].index("--map-thresholds") + 1].endswith("t_B2.json")
    vfl = flags(vd[0])
    assert "--map-thresholds-arm" in vfl and "--map-thresholds-ref" in vfl and vd[0][vd[0].index("--floor") + 1] == (fd / "FLOOR.json").as_posix()
    xd = [p for p in dry.ad("B4").iterdir() if p.is_dir() and p.name.startswith("thr_")]
    assert len(xd) == 1 and (xd[0] / "VERDICT_B4.json").exists() and (xd[0] / "READ_DONE").exists() and "MAP-THR" in dry.events("B4")
    assert not (dry.ad("B4") / "VERDICT_B4.json").exists() and not (dry.ad("B4") / "READ_DONE").exists()          # the fixed-128 read's files are untouched / absent
    v = TA._strict_json(xd[0] / "VERDICT_B4.json")
    assert v["status"] == "OK" and v["map_thresholds"]["same_protocol"] is True and v["map_thresholds"]["floor_c0"]["sha256"] == hashlib.sha256(t["C0"].read_bytes()).hexdigest()


@pytest.mark.parametrize("env_keys,tag", [(("R9E1_MAP_THR_ARM",), "MAP-THR-ARGS"), (("R9E1_MAP_THR_ARM", "R9E1_MAP_THR_REF"), "MAP-THR-ARGS"),
                                          (("R9E1_MAP_THR_ARM", "R9E1_MAP_THR_REF", "R9E1_MAP_THR_FLOOR_DIR"), "NO-MAP-THR")])
def test_arm_read_thresholds_misuse_aborts_before_the_lock(dry, tmp_path, env_keys, tag):
    b1 = dry.run_dir("B1", {"miss": []})
    dry.chain("START commit bbb", "B1-LAUNCH s; run dir %s" % b1.as_posix(), "B1-DONE ok")
    vals = {"R9E1_MAP_THR_ARM": (tmp_path / "no_such_arm.json").as_posix(), "R9E1_MAP_THR_REF": (tmp_path / "no_such_ref.json").as_posix(),
            "R9E1_MAP_THR_FLOOR_DIR": tmp_path.as_posix()}
    dry.start("B1", **{k: vals[k] for k in env_keys})
    assert dry.wait(lambda: (dry.ad("B1") / "READ_ABORTED").exists(), 90) and dry.proc.wait(timeout=60) == 3
    assert tag in (dry.ad("B1") / "READ_ABORTED").read_text(encoding="utf-8") and "LOCK-WAIT" not in dry.events("B1") and not (dry.ad("B1") / "dump_B1").exists()


def test_arm_read_thresholds_against_a_fixed_128_floor_is_refused_by_the_verdict(dry, tmp_path):
    t = {n: _thr_file(tmp_path, {"psw": [100, 100]}, name="t_%s.json" % n) for n in ("a", "r")}
    fd = tmp_path / "fixedfloor"
    fd.mkdir()
    for f in ("FLOOR.json", "EVAL_C0.json"):                                                  # the dry rig's own fixed-128 floor, offered as the "threshold" floor
        (fd / f).write_bytes((dry.rd / f).read_bytes())
    b1 = dry.run_dir("B1", {"miss": []})
    dry.chain("START commit bbb", "B1-LAUNCH s; run dir %s" % b1.as_posix(), "B1-DONE ok")
    dry.start("B1", R9E1_MAP_THR_ARM=t["a"].as_posix(), R9E1_MAP_THR_REF=t["r"].as_posix(), R9E1_MAP_THR_FLOOR_DIR=fd.as_posix())
    assert dry.wait(lambda: (dry.ad("B1") / "READ_ABORTED").exists() or "DONE" in dry.events("B1"), 420), (dry.events("B1"), dry.stdout()[-2500:])
    assert dry.proc.wait(timeout=60) == 3 and "VERDICT-NOT-OK" in (dry.ad("B1") / "READ_ABORTED").read_text(encoding="utf-8")
    xd = [p for p in dry.ad("B1").iterdir() if p.is_dir() and p.name.startswith("thr_")][0]
    v = TA._strict_json(xd / "VERDICT_B1.json")
    assert v["status"] == "REFUSED" and "map-threshold protocol" in " ".join(v["problems"]) and not (xd / "READ_DONE").exists()
