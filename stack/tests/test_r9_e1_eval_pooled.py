"""r9_e1_eval.py's two POOLED refcv9 E1 primaries (REFCV9_DESIGN 4.3), known answers:
  * ``ap{1,2}m_20_60`` (B2 / B2c): ONE all-point AP over the union of the 20_40 and 40_60 rows with their pooled GT -- 1 detected GT at 25 m and 3
    undetected GTs at 45-55 m read AP@2m 20_40 = 1.0, 40_60 = 0.0, and 20_60 = 0.25, NOT the band mean 0.5;
  * ``nf_prec_0_10`` (T1 / T1s): corridor precision with the 0-5 and 5-10 band COUNTS summed -- 9 / 10 and 6 / 30 flags read 15 / 40 = 0.375, NOT
    the mean of the two precisions 0.55; no flags at all reads NaN, never 0.
Both are ADDITIVE keys: the existing keys keep their values (test_r9_e1_eval.py / test_r9_e1_floor.py are unchanged and must stay green).
"""
import sys
from pathlib import Path

import numpy as np

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_T = Path(__file__).resolve().parent
sys.path.insert(0, str(_T))
import r9_e1_eval as E                                   # noqa: E402
import test_r9_e1_eval as TE                             # noqa: E402  (its pack / stats fixtures, reused unchanged)


def _near_hit_far_miss():
    gt = [[25.0, 1.0], [45.0, 0.0], [50.0, 1.0], [55.0, -1.0]]          # 1 GT in 20_40, 3 GTs in 40_60
    p = TE._pack(gt)
    for i in (1, 2, 3):                                                   # the three far GTs get no detection: their slots go far away
        p["xy"][i] = np.array([500.0 + 10 * i, 500.0], np.float32)
        p["logit"][i] = -8.0
    return [("%012x" % 1, 0, p)]


def test_ap_20_60_is_one_ap_over_the_pooled_bands():
    s = TE._stats(_near_hit_far_miss())
    assert s["ap2m_20_40"] == 1.0 and s["ap2m_40_60"] == 0.0
    assert s["ap2m_20_60"] == 0.25 and s["ap1m_20_60"] == 0.25
    assert s["ap2m_20_60"] != (s["ap2m_20_40"] + s["ap2m_40_60"]) / 2          # the band mean would read 0.5


def test_ap_20_60_known_answer_on_gt_as_prediction():
    s = TE._stats(TE._records())
    assert s["ap2m_20_60"] == 1.0 and s["ap1m_20_60"] == 1.0


def test_ap_20_60_is_nan_without_far_gt():
    s = TE._stats([("%012x" % 1, 0, TE._pack([[5.0, 1.0], [12.0, -2.0]]))])
    assert s["ap2m_20_60"] != s["ap2m_20_60"]                                    # NaN: no GT 20-60 m, never 0


def _unit(p0, tp0, p1, tp1):
    a = np.zeros((len(E.NEARFIELD_BANDS), 6))
    a[0, 0], a[0, 1], a[1, 0], a[1, 1] = p0, tp0, p1, tp1
    a[2, 0], a[2, 1] = 100.0, 100.0                                           # the 10-20 m band must not enter the 0-10 m figure
    return a


def test_nf_prec_0_10_pools_the_counts():
    s = E.nearfield_stats([_unit(4, 4, 10, 2), _unit(6, 5, 20, 4)])          # band 0: 9 / 10, band 1: 6 / 30
    assert s["nf_prec_0_5"] == 0.9 and s["nf_prec_5_10"] == 0.2
    assert s["nf_prec_0_10"] == 0.375                                         # 15 / 40, not (0.9 + 0.2) / 2 = 0.55


def test_nf_prec_0_10_is_nan_without_flags():
    s = E.nearfield_stats([_unit(0, 0, 0, 0)])
    assert s["nf_prec_0_10"] != s["nf_prec_0_10"]
