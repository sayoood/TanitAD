"""M1's two design BESIDE rows -- lane tolerance-IoU 0-20 m (E0d MAP-3a) and drivable-boundary recall @0.3 m 0-20 m (E0d MAP-3b) -- computed by r9_e1_eval
(``d3_beside_per_file`` / ``_d3_stats_keys``) and carried through r9_e1_floor.nearfield_rows.

THE HAND WORLD (three 220 x 40 rasters, one window each; bands 0-5 rows 0-49, 5-10 rows 50-99, 10-20 rows 100-199, 20-40 rows 200-399):
  LANE   file A: GT lane (class 2) row 10 cols 5-14, predicted ``code == 2`` row 10 cols 10-19, GT-unseen (255) strip row 8 cols 8-21.  Both rendered 5 cells wide:
           Gd rows 8-12 cols 3-16 (70), Pd rows 8-12 cols 8-21 (70), Pd & Gd = rows 8-12 cols 8-16 (45); the unseen strip removes row 8 from every count
           (Pd 14, Gd 9, both 9)  ->  A 36, B 56, C 61  -> IoU 36 / 81.      5-10 m: GT lane row 60 cols 5-14, no prediction -> [0, 0, 70].
           10-20 m: prediction row 120 cols 5-9, no GT -> [0, 45, 0].   0-20 m POOLED BY COUNTS [36, 101, 131] -> 36 / 196  (the mean of the band IoUs is not it).
  BOUNDARY  pdrv = 200 inside a drivable block, 0 elsewhere (flag pdrv >= 128; inner boundary = the block's perimeter ring).
           A: block rows 20-39 cols 10-29; GT edge (class 5) row 18 cols 10-29 (20 cells, 2 rows off the top ring: hit 20) and row 70 cols 10-29 (20, far: hit 0).
           B: block rows 125-149 cols 5-34; GT edge row 130 cols 0-29 (30 cells): only cols 2-8 are within 3 cells of the left ring col 5 -> hit 7.
           C: A's block; GT edge row 45 cols 0-18 (19 cells, 6 rows off the ring: hit 0) -- ONE cell below the 20-cell eligibility floor.
         0-5 m:   A [hit 20, nE 20, eligible]   B [0, 0, no]   C [0, 19, NOT eligible]   window mean 1 (A alone); pooled cells 20 / 39
         5-10 m:  A [0, 20, eligible]                                                    window mean 0
         10-20 m: B [7, 30, eligible]                                                     window mean 7/30 = pooled cells
         0-20 m:  A [20, 40, 1/2]  B [7, 30, 7/30]  C [0, 19, NOT eligible]              window mean (1/2 + 7/30) / 2 = 11/30; pooled cells 27 / 89 = 0.3034
         20-40 m: no GT edge anywhere -> NaN.
THE E0D CONTROL: random rasters scored by ``e0d_score.plane_stats`` itself (loaded by file) must give the same A / B / C, hit / nE and the same pooled lane IoU /
window-mean recall to 1e-12.   MUTATIONS (tests/mutate_d3beside.py): each makes the named test RED."""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
import r9_e1_eval as E                                   # noqa: E402
import r9_e1_floor as FL                                 # noqa: E402

NAN = float("nan")
_E0D_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/e0/e0d_score.py"


def _e0d():
    cands = [os.environ.get("D3_E0D_PATH", ""), str(Path(__file__).resolve().parents[2] / _E0D_REL), str(Path(E.__file__).resolve().parents[2] / _E0D_REL),
             "C:/t/near20/tree/" + _E0D_REL, "E:/Projects/TanitAD/" + _E0D_REL]
    for c in cands:
        if c and Path(c).is_file():
            spec = importlib.util.spec_from_file_location("e0d_score_d3", c)
            m = importlib.util.module_from_spec(spec)
            sys.modules["e0d_score_d3"] = m
            spec.loader.exec_module(m)
            return m
    pytest.skip("e0d_score.py not found (set D3_E0D_PATH): the control cannot run")


def _save(path, gt, pdrv, code, with_pdrv=True, with_code=True):
    z = {"gt": gt, "pedge": np.zeros_like(gt), "psw": np.zeros_like(gt)}
    if with_pdrv:
        z["pdrv"] = pdrv
    if with_code:
        z["code"] = code
    np.savez_compressed(path, **z)


def _blank():
    return np.zeros((220, 40), np.uint8), np.zeros((220, 40), np.uint8), np.zeros((220, 40), np.uint8)


def _world(d):
    """files e000_w0007 (A), e001_w0007 (B), e002_w0007 (C) of the module docstring, in directory ``d``."""
    gt, pdrv, code = _blank()                                      # A
    gt[10, 5:15] = 2
    code[10, 10:20] = 2
    gt[8, 8:22] = 255
    gt[60, 5:15] = 2
    code[120, 5:10] = 2
    pdrv[20:40, 10:30] = 200
    gt[18, 10:30] = 5
    gt[70, 10:30] = 5
    _save(d / "e000_w0007.npz", gt, pdrv, code)
    gt, pdrv, code = _blank()                                      # B
    pdrv[125:150, 5:35] = 200
    gt[130, 0:30] = 5
    _save(d / "e001_w0007.npz", gt, pdrv, code)
    gt, pdrv, code = _blank()                                      # C
    pdrv[20:40, 10:30] = 200
    gt[45, 0:19] = 5
    _save(d / "e002_w0007.npz", gt, pdrv, code)


@pytest.fixture()
def world(tmp_path):
    d = tmp_path / "rast"
    d.mkdir()
    _world(d)
    return d, sorted(str(p) for p in d.glob("*.npz"))


def _unit(acc):
    """the full-width [4, D3_OFF + D3_COLS] block ``_d3_stats_keys`` reads, from the D3 columns of a per-file stack summed over files."""
    return np.concatenate([np.zeros((acc.shape[1], E.D3_OFF)), acc.sum(0)], axis=1)


def _same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-12)


def test_hand_lane_counts_and_pooled_iou(world):
    _, files = world
    names, acc, info = E.d3_beside_per_file(files)
    assert names == ["e000_w0007.npz", "e001_w0007.npz", "e002_w0007.npz"]
    assert acc[0, :, 0:3].tolist() == [[36, 56, 61], [0, 0, 70], [0, 45, 0], [0, 0, 0]]               # file A, per read band: A, B, C
    assert acc[1, :, 0:3].sum() == 0 and acc[2, :, 0:3].sum() == 0
    k = E._d3_stats_keys({}, _unit(acc))
    assert k["lane_tiou05_0_5"] == pytest.approx(36 / 81, abs=1e-12) and k["lane_tiou05_5_10"] == 0.0 and k["lane_tiou05_10_20"] == 0.0
    assert math.isnan(k["lane_tiou05_20_40"])
    assert k["lane_tiou05_0_20"] == pytest.approx(36 / 196, abs=1e-12)                                  # pooled by COUNTS
    assert abs(k["lane_tiou05_0_20"] - (36 / 81 + 0 + 0) / 3) > 0.03                                    # not the mean of the band IoUs


def test_hand_boundary_window_mean_and_eligibility(world):
    _, files = world
    _, acc, info = E.d3_beside_per_file(files)
    b = {n: acc[i] for i, n in enumerate("ABC")}
    assert b["A"][:, 3:7].tolist() == [[20, 20, 1.0, 1], [0, 20, 0.0, 1], [0, 0, 0.0, 0], [0, 0, 0.0, 0]]      # hit, nE, rec_w, elig per band
    assert b["B"][2, 3:7].tolist() == [7, 30, 7 / 30, 1] and b["B"][0, 3:7].tolist() == [0, 0, 0.0, 0]
    assert b["C"][0, 3:7].tolist() == [0, 19, 0.0, 0]                                                 # 19 GT edge cells: NOT eligible (floor is 20)
    assert [b[n][0, 7:11].tolist() for n in "ABC"] == [[20, 40, 0.5, 1], [7, 30, 7 / 30, 1], [0, 19, 0.0, 0]]    # the 0-20 m block, band row 0
    assert (b["A"][1:, 7:11] == 0).all()
    k = E._d3_stats_keys({}, _unit(acc))
    assert k["drv_bnd_rec03_0_5"] == 1.0 and k["drv_bnd_rec03_5_10"] == 0.0 and _same(k["drv_bnd_rec03_10_20"], 7 / 30) and math.isnan(k["drv_bnd_rec03_20_40"])
    assert k["drv_bnd_rec03_0_20"] == pytest.approx(11 / 30, abs=1e-12)                               # the WINDOW MEAN
    assert k["drv_bnd_rec03_cells_0_20"] == pytest.approx(27 / 89, abs=1e-12) and k["drv_bnd_rec03_cells_0_5"] == pytest.approx(20 / 39, abs=1e-12)
    assert abs(k["drv_bnd_rec03_0_20"] - k["drv_bnd_rec03_cells_0_20"]) > 0.05                          # the two estimators differ on this world


def test_eligibility_boundary_is_exactly_20_cells(tmp_path):
    gt, pdrv, code = _blank()
    pdrv[20:40, 10:30] = 200
    gt[18, 10:29] = 5                                                                                 # 19 cells
    _save(tmp_path / "e000_w0001.npz", gt, pdrv, code)
    gt[18, 29] = 5                                                                                    # 20 cells
    _save(tmp_path / "e000_w0002.npz", gt, pdrv, code)
    _, acc, _ = E.d3_beside_per_file(sorted(str(p) for p in tmp_path.glob("*.npz")))
    assert [acc[i, 0, 6] for i in (0, 1)] == [0, 1] and [acc[i, 0, 10] for i in (0, 1)] == [0, 1]


def test_e0d_control_reproduces_e0d_to_1e12(tmp_path):
    e0d = _e0d()
    rng = np.random.default_rng(7)
    files = []
    for n in range(14):
        h, w = 400, 90
        gt = np.zeros((h, w), np.uint8)
        pdrv = np.zeros((h, w), np.uint8)
        code = np.zeros((h, w), np.uint8)
        for _ in range(7):                                                                            # random drivable / sidewalk blobs
            r0, c0 = int(rng.integers(0, h - 40)), int(rng.integers(0, w - 30))
            gt[r0:r0 + int(rng.integers(10, 80)), c0:c0 + int(rng.integers(8, 30))] = rng.choice([1, 3, 4, 6, 7, 0])
        for _ in range(5):                                                                            # thin lane / edge lines, a GT-unseen strip
            r0, c0 = int(rng.integers(0, h - 60)), int(rng.integers(0, w - 40))
            gt[r0, c0:c0 + int(rng.integers(5, 40))] = 5
            gt[r0 + 3:r0 + 40, c0] = 2
        gt[int(rng.integers(0, h)), :] = 255
        for _ in range(6):
            r0, c0 = int(rng.integers(0, h - 40)), int(rng.integers(0, w - 30))
            pdrv[r0:r0 + int(rng.integers(10, 90)), c0:c0 + int(rng.integers(8, 30))] = rng.integers(60, 250)
            code[r0:r0 + int(rng.integers(10, 60)), c0 + int(rng.integers(0, 20))] = 2
        p = tmp_path / ("e%03d_w0001.npz" % n)
        np.savez_compressed(p, gt=gt, pdrv=pdrv, pedge=np.zeros_like(gt), psw=np.zeros_like(gt), code=code)
        files.append(str(p))
    _, acc, _ = E.d3_beside_per_file(files)
    lane, bnd = [], []
    for i, f in enumerate(files):
        z = np.load(f)
        ps = e0d.plane_stats(z["pdrv"], z["pedge"], z["code"], e0d.gt_static(z["gt"]), thr_drv=(128,), thr_edge=(128,))
        l, bd = ps["lane"][e0d.LANE_RS.index(2)][0].astype(float), ps["bnd"][0][0].astype(float)           # e0d band "0-20", [A B C ...] / [nE hitE ...]
        lane.append(l)
        bnd.append(bd)
        assert acc[i, :3, 0:3].sum(0).tolist() == l[:3].tolist()
        assert (acc[i, 0, 7], acc[i, 0, 8]) == (bd[1], bd[0])
    k = E._d3_stats_keys({}, _unit(acc))
    assert abs(k["lane_tiou05_0_20"] - e0d.f_ioudil(np.sum(lane, 0)[:3])) < 1e-12
    assert abs(k["drv_bnd_rec03_0_20"] - e0d.f_ratio(e0d.recall_comp(np.stack(bnd)).sum(0))) < 1e-12
    assert abs(k["drv_bnd_rec03_cells_0_20"] - e0d.f_ratio(e0d.pooled_rec_comp(np.stack(bnd)).sum(0))) < 1e-12
    assert 0 < k["lane_tiou05_0_20"] < 1 and 0 < k["drv_bnd_rec03_0_20"] < 1                              # a control that reads 0 or 1 everywhere proves nothing


def test_pdrv_level_comes_from_the_thresholds_file_and_is_stamped(world, tmp_path):
    _, files = world
    nb = [[0, 5], [5, 10], [10, 20], [20, 40]]

    def thr(planes):
        p = tmp_path / "t.json"
        p.write_text(json.dumps({"schema": E.THR_SCHEMA, "bands_m": nb, "planes": planes, "provenance": {}}), encoding="utf-8")
        return E.load_map_thresholds(p)
    _, a128, i128 = E.d3_beside_per_file(files)
    assert i128["pdrv_level_source"].startswith("fixed 128 (no --map-thresholds") and set(i128["pdrv_levels"].values()) == {128}
    _, ap, ip = E.d3_beside_per_file(files, 1, thr({"psw": [128] * 4}))                                   # a file with NO pdrv plane keeps 128, and says so
    assert ip["pdrv_level_source"] == "fixed 128 (the thresholds file carries no pdrv plane)" and (ap == a128).all()
    _, ah, ih = E.d3_beside_per_file(files, 1, thr({"pdrv": [250] * 4}))                                 # 250 > every pdrv cell (200): nothing is drivable
    assert ih["pdrv_level_source"] == "the thresholds file's own pdrv plane" and set(ih["pdrv_levels"].values()) == {250}
    assert ah[:, :, 3].sum() == 0 and ah[:, 0, 7].sum() == 0 and a128[:, :, 3].sum() == 27 + 0 and (ah[:, :, 0:3] == a128[:, :, 0:3]).all()
    with pytest.raises(E.ThresholdsError):                                                                # 0-20 m straddles fit bands with different pdrv cuts
        E.d3_beside_per_file(files, 1, thr({"pdrv": [128, 128, 250, 250]}))


def test_missing_planes_are_nan_not_zero(tmp_path):
    gt, pdrv, code = _blank()
    gt[18, 10:30] = 5
    _save(tmp_path / "e000_w0001.npz", gt, pdrv, code, with_pdrv=False)
    _save(tmp_path / "e001_w0001.npz", gt, pdrv, code, with_code=False)
    _, acc, info = E.d3_beside_per_file(sorted(str(p) for p in tmp_path.glob("*.npz")))
    assert info["n_with_code_plane"] == 1 and info["n_with_pdrv_plane"] == 1
    assert np.isnan(acc[0, :, 3:11]).all() and not np.isnan(acc[0, :, 0:3]).any() and np.isnan(acc[1, :, 0:3]).all()
    assert E._d3_stats_keys({"x": 1}, _unit(acc)) == {"x": 1}                                             # a mixed set measures neither: no key at all


def test_unset_identity_and_additivity(world):
    _, files = world
    n0, a0, c0, i0 = E.nearfield_per_file_ex_edge(files)
    n1, a1, c1, i1 = E.nearfield_per_file_ex(files)
    assert n0 == n1 and a0.shape == a1.shape == (3, 4, E.NF_COLS) and np.array_equal(a0, a1, equal_nan=True) and np.array_equal(c0, c1) and i0 == i1 and "d3_beside" not in i0
    assert not [k for k in E.nearfield_stats(list(a0)) if k.startswith(("lane_tiou", "drv_bnd"))]
    nf, af, _, inf = E.nearfield_per_file_ex_edge(files, beside_d3=True)
    assert af.shape == (3, 4, E.D3_OFF + E.D3_COLS) and "d3_beside" in inf and np.array_equal(af[:, :, :E.NF_COLS], a0, equal_nan=True)
    assert np.isnan(af[:, :, E.NF_COLS:E.D3_OFF]).all()                                                    # edge-class columns not measured: NaN, so no nf_edge03 row
    ks = E.nearfield_stats([af[i] for i in range(3)])
    assert not [k for k in ks if k.startswith("nf_edge03")] and {k for k in ks if k.startswith(("lane_tiou", "drv_bnd"))} == {
        "%s_%s" % (p, r) for p in ("lane_tiou05", "drv_bnd_rec03", "drv_bnd_rec03_cells") for r in E.D3_REGIONS}
    assert {k: v for k, v in ks.items() if not k.startswith(("lane_tiou", "drv_bnd"))}.keys() == E.nearfield_stats([a0[i] for i in range(3)]).keys()


def test_edge_class_and_d3_columns_coexist(world, tmp_path):
    _, files = world
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"schema": E.THR_SCHEMA, "bands_m": [[0, 40]], "planes": {"pedge": [128]}, "provenance": {}}), encoding="utf-8")
    thr = E.load_map_thresholds(p)
    _, ae, _, _ = E.nearfield_per_file_ex_edge(files, thresholds=thr, edge_tol=True)
    _, ab, _, _ = E.nearfield_per_file_ex_edge(files, thresholds=thr, edge_tol=True, beside_d3=True)
    assert ae.shape[2] == E.NF_COLS + E.EDGE_TOL_N and np.array_equal(ab[:, :, :ae.shape[2]], ae, equal_nan=True)
    ks = E.nearfield_stats([ab[i] for i in range(3)])
    assert any(k.startswith("nf_edge03") for k in ks) and "lane_tiou05_0_20" in ks and "drv_bnd_rec03_0_20" in ks


def test_floor_carries_the_counts_into_the_paired_bootstrap(world, tmp_path):
    d, _ = world
    d2 = tmp_path / "rast2"
    d2.mkdir()
    _world(d2)
    gt, pdrv, code = _blank()                                                                              # make arm 2 differ: its code plane has no lane at all
    for f in d2.glob("*.npz"):
        z = dict(np.load(f))
        z["code"] = np.zeros_like(z["code"])
        np.savez_compressed(f, **z)
    rows, meta = FL.nearfield_rows(str(d), str(d2), boot=40, seed=0, beside_d3=True)
    assert meta["beside_d3"]["c0"]["n_files"] == 3 and meta["beside_d3"]["c1"]["pdrv_level_source"].startswith("fixed 128")
    r = rows["lane_tiou05_0_20"]
    assert _same(r["c0"], 36 / 196) and r["c1"] == 0.0 and r["delta"] == pytest.approx(-36 / 196, abs=1e-12)
    assert _same(rows["drv_bnd_rec03_0_20"]["c0"], 11 / 30) and rows["drv_bnd_rec03_0_20"]["delta"] == pytest.approx(0.0, abs=1e-12)
    assert {"lane_tiou05_0_20", "drv_bnd_rec03_0_20", "drv_bnd_rec03_cells_0_20", "lane_tiou05_5_10"} <= set(rows)
    assert FL.higher_is_better("lane_tiou05_0_20") and "BESIDE" in FL.metric_group("drv_bnd_rec03_0_20") and "BESIDE" in FL.metric_group("lane_tiou05_0_5")
    rows0, meta0 = FL.nearfield_rows(str(d), str(d2), boot=40, seed=0)                                    # the flag unset: none of them, no stamp
    assert not [k for k in rows0 if k.startswith(("lane_tiou", "drv_bnd"))] and "beside_d3" not in meta0
    assert all(rows0[k].get("delta") == rows[k].get("delta") for k in rows0)                                      # and every row it had is unchanged


def test_cli_flag_unset_output_has_no_d3_key_and_set_has_stamp(world, tmp_path):
    import tanitad
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(tanitad.__file__).resolve().parents[1] / "tests"))
    import test_r9_e1_eval as T
    import pickle
    recs = T._records(n_clips=3)
    pk = tmp_path / "perc_det.pkl"
    pickle.dump([{"sha12": s, "ws": w, "box3d": [p]} for (s, w, p) in recs], open(pk, "wb"))
    d, _ = world
    out = {}
    for tag, extra in (("unset", []), ("set", ["--map-beside-d3"])):
        o = tmp_path / ("%s.json" % tag)
        assert E.main(["--packs", str(pk), "--rast-dir", str(d), "--gate", "0.5", "--boot", "30", "--seed", "0", "--out", str(o)] + extra) == 0
        out[tag] = json.loads(o.read_text(encoding="utf-8"))["nearfield"]
    assert "d3_beside" not in out["unset"] and not [k for k in out["unset"]["ci"] if k.startswith(("lane_tiou", "drv_bnd"))]
    assert out["set"]["d3_beside"]["pdrv_level_source"].startswith("fixed 128") and out["set"]["d3_beside"]["n_files"] == 3
    assert out["set"]["ci"]["lane_tiou05_0_20"]["point"] == pytest.approx(36 / 196, abs=1e-12) and out["set"]["ci"]["drv_bnd_rec03_0_20"]["point"] == pytest.approx(11 / 30, abs=1e-12)
    assert {k: v for k, v in out["set"]["ci"].items() if not k.startswith(("lane_tiou", "drv_bnd"))} == out["unset"]["ci"]
    with pytest.raises(SystemExit):
        E.main(["--packs", str(pk), "--gate", "0.5", "--out", str(tmp_path / "x.json"), "--map-beside-d3"])        # no --rast-dir: refused, never ignored


def test_key_source_pins_above_l697_are_untouched():
    """The verdict's KEY_SRC cites lines 588 / 589 / 594 / 605 / 606 / 697 of r9_e1_eval.py: all D3 code is added BELOW, and the one edit above is a same-length swap."""
    lines = Path(E.__file__).read_text(encoding="utf-8").splitlines()
    for ln, needle in ((588, '"nf_prec_%d_%d"'), (589, '"nf_prec05_%d_%d"'), (594, '"nf_prec_0_10"'), (605, '"nf_swedge_prec_%s%d_%d"'), (606, '"nf_swedge_prec_%s5_20"'),
                       (697, '"nf_edge03_prec_"')):
        assert needle in lines[ln - 1], (ln, lines[ln - 1])
    assert min(i for i, s in enumerate(lines, 1) if s.startswith(("def d3_", "D3_", "def _d3_"))) > 697
