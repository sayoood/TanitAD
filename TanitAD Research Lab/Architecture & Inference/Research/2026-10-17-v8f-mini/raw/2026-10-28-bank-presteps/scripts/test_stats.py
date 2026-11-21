"""Pins scripts/stats.py to taniteval/ci.py (the copy at the tip in ../src_tip/ci.py). Synthetic data only.
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_stats.py   (cwd = this directory)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stats as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("ci_tip", HERE.parent / "src_tip" / "ci.py")
CI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CI)


def _data(n_clusters=60, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.array([f"clip{c:03d}" for c in range(n_clusters)])
    sizes = rng.integers(3, 30, n_clusters)
    eid = np.repeat(ids, sizes)
    crawl = (rng.random(len(eid)) < 0.07).astype(float)
    use = (rng.random(len(eid)) < 0.7).astype(float)
    crawl *= use
    return ids, eid, crawl, use


def test_ratio_boot_matches_ci_episode_cluster_bootstrap():
    ids, eid, crawl, use = _data()
    # ci.py reduces a per-record vector; encode (crawl, use) as complex-free pair via a callable on an index vector
    idx = np.arange(len(eid), dtype=float)
    ref = CI.episode_cluster_bootstrap(idx, eid, reduce=lambda v: crawl[v.astype(int)].sum() / use[v.astype(int)].sum(), n_boot=300, seed=0, dp=12)
    uniq = np.unique(eid)
    num = np.array([crawl[eid == u].sum() for u in uniq])
    den = np.array([use[eid == u].sum() for u in uniq])
    mine = S.ratio_boot(num, den, B=300, seed=0)
    assert abs(mine["point"] - ref["mean"]) < 1e-12
    assert abs(mine["lo"] - ref["lo"]) < 1e-9 and abs(mine["hi"] - ref["hi"]) < 1e-9, (mine["lo"], ref["lo"], mine["hi"], ref["hi"])


def test_known_share_and_interval_contains_it():
    num = np.array([1.0, 0, 2, 0, 1, 0, 0, 3, 0, 1])
    den = np.array([20.0, 15, 30, 10, 12, 25, 18, 40, 22, 8])
    r = S.ratio_boot(num, den, B=500)
    assert abs(r["point"] - 8 / 200) < 1e-12
    assert r["lo"] <= r["point"] <= r["hi"]


def test_two_sample_ratio_point():
    r = S.two_sample_ratio_boot(np.array([2.0, 4]), np.array([1.0, 1]), np.array([4.0, 4]), np.array([1.0, 1]), B=100)
    assert abs(r["point"] - 3 / 4) < 1e-12
