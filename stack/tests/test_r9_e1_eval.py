"""r9_e1_eval.py known-answer + pin tests (refcv9 C7 / E1). Literal expectations; the two ORIGINALS the evaluator copies
(MAPSEL ``nearfield_prec.py``, PERC8 ``perc8_score.census_one``) are embedded verbatim below and the pinned copies are held
equal to them on fixtures. GT-as-prediction => AP@2m 1.0, precision 1.0."""
import contextlib
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
import r9_e1_eval as E                                   # noqa: E402

from tanitad.eval import detection_metrics as dm        # noqa: E402

# ---- the ORIGINALS, verbatim (sha256 of the file text is pinned so an edit to the copy below is visible) ----
NEARFIELD_ORIG = 'import numpy as np, os\nfrom scipy.ndimage import binary_dilation, distance_transform_edt\nR = "F:/refcv9_mapsel/rast_s0"\nfs = sorted(os.listdir(R))[::6]\nbands = [(0, 5), (5, 10), (10, 20), (20, 40)]\nc0, c1 = 270, 330\nacc = {b: np.zeros(6) for b in bands}\ngtcls = {b: np.zeros(9) for b in bands}\nn = 0\nfor f in fs:\n    z = np.load(os.path.join(R, f))\n    if "gt" not in z.files:\n        continue\n    n += 1\n    gt, psw = z["gt"], z["psw"]\n    sw = gt == 7\n    near_sw = binary_dilation(sw, iterations=5)          # within 0.5 m of a GT sidewalk cell\n    for (a, b) in bands:\n        sl = (slice(a * 10, b * 10), slice(c0, c1))\n        p, g, seen, ns = psw[sl] >= 128, sw[sl], gt[sl] != 255, near_sw[sl]\n        p = p & seen\n        acc[(a, b)] += [p.sum(), (p & g).sum(), (p & ns).sum(), g.sum(), (g & p).sum(), seen.sum()]\n        cls = gt[sl][p]\n        for c in range(8):\n            gtcls[(a, b)][c] += (cls == c).sum()\nprint("windows", n, "corridor |y|<3 m, own psw>=0.5 on GT-seen cells")\nprint("band | own flag rate | precision exact | precision within 0.5 m | recall | GT class of own flags (0 nocls,1 drv,2 lane,3 xwalk,4 arrow,5 edge,6 hatch,7 sw)")\nfor k, v in acc.items():\n    P, TP, TPn, G_, TPr, S_ = v\n    d = gtcls[k]; d = d[:8] / max(d[:8].sum(), 1)\n    print(f"{k[0]:>2}-{k[1]:<3}| {P / S_:.4f} | {TP / max(P, 1):.3f} | {TPn / max(P, 1):.3f} | {TPr / max(G_, 1):.3f} | " + " ".join(f"{x:.2f}" for x in d))\n'
NEARFIELD_ORIG_SHA256 = "0da351d40ad73610618b90185e36f1811ca170f2f60c49f7664675b8e36a97dd"
CENSUS_ORIG = 'def census_one(pk, gate, nms_cfg=None):\n    """[tp, n_conf, n_pos, sum_cnt, n_det, n_ge2, acc_raw, n_acc] + tp_by_class[10] for one window at `gate`."""\n    from tanitad.eval import detection_metrics as dm\n    from tanitad.eval import detection_nms as nms\n    q = pk if nms_cfg is None else nms.nms_pack(pk, nms_cfg["radius_m"], nms_cfg.get("p_floor", nms.P_FLOOR))\n    rows = dm.greedy_rows(q, dm.PR_DIST_M, None, min_conf=float(gate))\n    tp = sum(1 for r in rows if r[1] == 1)\n    n_conf = sum(1 for r in rows if r[1] >= 0)\n    n_pos = int(q["pos"].sum())\n    p = _sig(q["logit"])\n    gx = np.asarray(q["gt_xy"])[np.asarray(q["pos"], bool)].astype(np.float64)\n    sum_cnt = n_det = n_ge2 = 0\n    if len(gx):\n        d = np.sqrt(((np.asarray(q["xy"])[None, :, :].astype(np.float64) - gx[:, None, :]) ** 2).sum(-1))\n        cnt = ((d <= dm.PR_DIST_M) & (p >= float(gate))[None]).sum(1)\n        det1 = cnt[cnt >= 1]\n        sum_cnt, n_det, n_ge2 = int(det1.sum()), int(len(det1)), int((det1 >= 2).sum())\n    tpc = np.zeros(10, np.int64)\n    acc_raw = n_acc = 0\n    for r in rows:\n        if r[1] != 1:\n            continue\n        gc = int(q["gt_cls"][r[4]])\n        if 0 <= gc < 10:\n            tpc[gc] += 1\n            n_acc += 1\n            acc_raw += int(int(q["cls"][r[3]]) == gc)\n    return np.array([tp, n_conf, n_pos, sum_cnt, n_det, n_ge2, acc_raw, n_acc], np.float64), tpc\n'
CENSUS_ORIG_SHA256 = "2014e0306edc2cffa9106118bff5357874fbbbdd8ff5a50e38f412a96335b6d4"


def _pack(gt_xy, gt_cls=None, dx=0.0, dup=0, n_slots=20):
    """Slot i sits on GT i shifted by dx metres (logit +8); dup extra copies per GT (logit +7); the rest far away (logit -8)."""
    g = len(gt_xy)
    gt_xy = np.asarray(gt_xy, np.float32).reshape(-1, 2)
    gc = np.zeros(g, np.int16) if gt_cls is None else np.asarray(gt_cls, np.int16)
    xy, logit, cls = [], [], []
    for i in range(g):
        for k in range(1 + dup):
            xy.append(gt_xy[i] + np.array([dx, 0.0], np.float32)); logit.append(8.0 - k); cls.append(gc[i])
    while len(xy) < n_slots:
        xy.append(np.array([500.0 + len(xy), 500.0], np.float32)); logit.append(-8.0); cls.append(0)
    n = len(xy)
    return {"ep": None, "logit": np.array(logit, np.float32), "xy": np.array(xy, np.float32), "cls": np.array(cls, np.int16),
            "cls_corr": np.array(cls, np.int16), "matched": np.zeros(n, bool), "exempt": np.zeros(n, bool),
            "pair_err": np.zeros(0), "pair_size_err": np.zeros(0), "pair_z_err": np.zeros(0),
            "gt_xy": gt_xy, "gt_cls": gc, "pos": np.ones(g, bool), "ign": np.zeros(g, bool), "hidden": np.zeros(g, bool)}


GT = [[5.0, 1.0], [12.0, -2.0], [25.0, 3.0], [33.0, 0.0], [47.0, 2.0], [55.0, -1.0]]      # bands 0_20 x2, 20_40 x2, 40_60 x2


def _records(n_clips=4, **kw):
    return [("%012x" % (c + 1), 8 * w, _pack(GT, gt_cls=[0, 1, 0, 2, 0, 1], **kw)) for c in range(n_clips) for w in range(2)]


def _stats(records, gate=0.5, nms_cfg=None):
    keys, items = E.build_clip_items(records, (1.0, 2.0), gate, nms_cfg)
    cl, _ids = E.pool_by_clip(keys, items)
    return E.box_stats(cl, (1.0, 2.0))


def test_gt_as_prediction_reads_the_known_answer():
    s = _stats(_records())
    assert s["ap2m_all"] == 1.0 and s["ap1m_all"] == 1.0
    assert all(s["ap2m_" + b] == 1.0 for b in E.BAND_NAMES)
    assert s["prec_gate_noNMS"] == 1.0 and s["rec_gate_noNMS"] == 1.0
    assert s["boxes_per_object_noNMS"] == 1.0 and s["frac_objects_ge2_noNMS"] == 0.0
    assert s["tp_class_acc"] == 1.0 and s["dx_0_40_mean"] == 0.0 and s["n_pos"] == 48.0


def test_boxes_per_object_by_range_known_answers():
    """refcv9 B3: GT-as-prediction reads 1.00 boxes/object in EVERY range band; two boxes per object read 2.00 in every band; the
    per-band pooled count agrees with the all-range figure (sum_cnt / n_det); a band with no GT reads NaN, never 0."""
    s = _stats(_records())
    assert all(s["boxes_per_object_noNMS_" + b] == 1.0 for b in E.BAND_NAMES)
    assert [s["n_det_noNMS_" + b] for b in E.BAND_NAMES] == [16.0, 16.0, 16.0]
    d = _stats(_records(dup=1))
    assert all(d["boxes_per_object_noNMS_" + b] == 2.0 for b in E.BAND_NAMES) and d["boxes_per_object_noNMS"] == 2.0
    far = _stats([("%012x" % 1, 0, _pack([[5.0, 1.0], [12.0, -2.0]]))])
    assert far["boxes_per_object_noNMS_0_20"] == 1.0 and far["boxes_per_object_noNMS_40_60"] != far["boxes_per_object_noNMS_40_60"]
    w = _stats(_records(dx=10.0))                                                       # wrong position: no detected object, in any band
    assert all(w["n_det_noNMS_" + b] == 0.0 for b in E.BAND_NAMES)


def test_wrong_position_control_reads_zero():
    s = _stats(_records(dx=10.0))                         # every box 10 m off: no TP at 2 m
    assert s["ap2m_all"] == 0.0 and s["rec_gate_noNMS"] == 0.0


def test_duplicate_boxes_known_answer_and_nms():
    s = _stats(_records(dup=1))                           # every object carries TWO boxes at the gate
    assert s["boxes_per_object_noNMS"] == 2.0 and s["frac_objects_ge2_noNMS"] == 1.0
    assert s["prec_gate_noNMS"] == 0.5                    # the second box finds its GT taken: a false positive
    sn = _stats(_records(dup=1), nms_cfg={"radius_m": 1.0})
    assert sn["boxes_per_object_NMS"] == 1.0 and sn["prec_gate_NMS"] == 1.0 and sn["ap2m_all_NMS"] == 1.0


def test_dx_known_answer_on_a_shifted_arm():
    s = _stats(_records(dx=0.5))                          # within 2 m: still TPs, |dx| = 0.5 exactly on the 0-40 m GTs
    assert s["ap2m_all"] == 1.0 and s["dx_0_40_mean"] == 0.5 and s["dx_0_40_median"] == 0.5


def _random_pack(rng, n_slots=30, logit_mu=0.0):
    g = int(rng.integers(2, 8))
    pk = _pack(rng.uniform(0, 60, (g, 2)) * np.array([1, 0.3]), n_slots=n_slots)
    n = len(pk["logit"])
    pk["logit"] = rng.normal(logit_mu, 2, n).astype(np.float32)
    pk["xy"] = (pk["gt_xy"][rng.integers(0, g, n)] + rng.normal(0, 1.4, (n, 2))).astype(np.float32)
    return pk


def test_fast_ap_equals_the_programme_ap_on_random_rows():
    rng = np.random.default_rng(0)
    for _ in range(5):
        pks = [_random_pack(rng) for _w in range(6)]
        rows = [r for pk in pks for r in dm.greedy_rows(pk, 2.0)]
        n = sum(int(pk["pos"].sum()) for pk in pks)
        arr = np.array([[c, k, 0] for (c, k, *_r) in rows if k >= 0], np.float64).reshape(-1, 3)
        assert E.fast_ap(arr, n) == pytest.approx(dm._ap(rows, n), abs=1e-12)


def test_census_copy_equals_the_perc8_original_on_random_windows():
    assert hashlib.sha256(CENSUS_ORIG.encode("utf-8")).hexdigest() == CENSUS_ORIG_SHA256
    ns = {"np": np, "_sig": E._sig}
    exec(compile(CENSUS_ORIG, "perc8_score.census_one", "exec"), ns)
    rng = np.random.default_rng(1)
    for _ in range(6):
        pk = _random_pack(rng, 25, -1.0)
        for nms_cfg in (None, {"radius_m": 1.0}):
            orig = ns["census_one"](pk, 0.3, nms_cfg)[0][:6]            # the original returns (vec8, tp_by_class)
            np.testing.assert_array_equal(E.census_one(pk, 0.3, nms_cfg), orig)


def _write_rasters(d, n_files, rng):
    for i in range(n_files):
        gt = rng.choice(np.array([0, 1, 2, 3, 5, 7, 7, 255], np.uint8), size=(500, 600))
        psw = (rng.random((500, 600)) * 255).astype(np.uint8)
        np.savez(Path(d) / ("e%03d_w%04d.npz" % (i // 2, i)), gt=gt, psw=psw)


def test_nearfield_copy_equals_the_original_script_output(tmp_path):
    assert hashlib.sha256(NEARFIELD_ORIG.encode("utf-8")).hexdigest() == NEARFIELD_ORIG_SHA256
    _write_rasters(tmp_path, 13, np.random.default_rng(2))                      # stride 6 -> files 0, 6, 12
    src = NEARFIELD_ORIG.replace('R = "F:/refcv9_mapsel/rast_s0"', "R = %r" % str(tmp_path))
    assert "R = %r" % str(tmp_path) in src
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "nearfield_prec.py", "exec"), {"__name__": "__orig__"})
    orig_lines = buf.getvalue().strip().splitlines()[2:]                         # drop the 'windows N ...' and header lines
    names, acc, gtcls = E.nearfield_per_file(sorted(str(p) for p in tmp_path.glob("*.npz")), stride=6)
    assert len(names) == 3
    tab = E.nearfield_table(acc, gtcls)
    mine = []
    for (lo, hi) in E.NEARFIELD_BANDS:
        r = tab["%d-%d" % (lo, hi)]
        mine.append(f"{lo:>2}-{hi:<3}| {r['flag_rate']:.4f} | {r['precision_exact']:.3f} | {r['precision_within_0p5m']:.3f} | "
                    f"{r['recall']:.3f} | " + " ".join(f"{x:.2f}" for x in r["gt_class_of_flags"]))
    assert mine == orig_lines and len(mine) == 4


def test_window_list_loader(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("# dev\nAAAAAAAAAAAA\nbbbbbbbbbbbb0000ffff\n", encoding="utf-8")
    assert E.load_window_list(p) == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    q = tmp_path / "s.json"
    q.write_text(json.dumps({"splits": {"DEV": ["a" * 12, "c" * 12], "HOLD": ["b" * 12]}}), encoding="utf-8")
    assert E.load_window_list(q, "dev") == ["a" * 12, "c" * 12] and E.load_window_list(q, "hold") == ["b" * 12]
    with pytest.raises(SystemExit):
        E.load_window_list(q)                                                    # a dict needs --split
    bad = tmp_path / "b.txt"
    bad.write_text("not-a-sha\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        E.load_window_list(bad)


def test_cli_end_to_end_single_arm_and_paired(tmp_path):
    def dump(path, **kw):
        path.write_bytes(pickle.dumps([{"sha12": s, "ws": w, "box3d": [pk]} for (s, w, pk) in _records(**kw)]))
    dump(tmp_path / "a.pkl")
    dump(tmp_path / "b.pkl", dx=10.0)
    win = tmp_path / "dev.txt"
    win.write_text("\n".join("%012x" % c for c in (1, 2, 3)) + "\n", encoding="utf-8")
    out = tmp_path / "o.json"
    rc = E.main(["--packs", str(tmp_path / "a.pkl"), "--packs-b", str(tmp_path / "b.pkl"), "--windows", str(win),
                 "--gate", "0.5", "--boot", "50", "--out", str(out)])
    assert rc == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["n_clips"] == 3 and d["n_windows"] == 6 and d["boxes"]["ap2m_all"]["point"] == 1.0
    p = d["paired_vs_B"]["boxes"]["ap2m_all"]
    assert p["a"] == 1.0 and p["b"] == 0.0 and p["delta"] == 1.0 and p["separated"] is True
    with pytest.raises(SystemExit):                                              # no gate => refused, never a silent 0.5
        E.main(["--packs", str(tmp_path / "a.pkl"), "--out", str(out)])
