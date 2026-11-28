"""v8F-mini longitudinal / roughness diagnostic (code/lon_rough_diag.py). CPU only, < 20 s.

Identities, each with the reading that would go wrong without it:
  * the ground truth as a run reads ADE 0 and oracle ADE 0, and the human's own roughness;
  * a path uniformly SLOWED (x 0.8) reads oracle-arc-scale ADE exactly 0 (the oracle restores the length, the shape is right) and a
    negative arc bias; one global scale also restores it, because the slowing is global;
  * a path with per-window speed errors of BOTH signs is not fixed by one global scale, but is by the oracle;
  * a path with added high-frequency jitter reads rougher than the human; a run off the grid, or with a wrong saved ade6, is REFUSED.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "code" / "lon_rough_diag.py"
pytestmark = pytest.mark.skipif(not CODE.exists(), reason="diagnostic not in this checkout")


def _load():
    spec = importlib.util.spec_from_file_location("lon_rough_diag", CODE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _world(tmp, **runs_kw):
    rng = np.random.default_rng(5)
    n, H = 200, 30
    t = np.arange(1, H + 1)[None] * 0.2
    v0 = rng.uniform(2, 15, n)
    gt = np.stack([v0[:, None] * t, 0.05 * (v0[:, None] * t) ** 1.2 * rng.normal(0, 0.1, n)[:, None]], -1)
    eid = np.repeat(np.arange(20), 10)
    np.savez(tmp / "gt.npz", gt=gt, eid=eid, dt=0.2, v0=v0)
    ev = tmp / "eval"
    ev.mkdir()
    sgn = np.where(np.arange(n) % 2 == 0, 0.7, 1.3)[:, None, None]
    jit = gt + rng.normal(0, 0.15, gt.shape)
    for tag, p in (("slow", gt * 0.8), ("mixed", gt * sgn), ("jitter", jit), ("truth", gt)):
        ade = np.linalg.norm(p - gt, axis=-1).mean(1)
        e = eid[::-1].copy() if runs_kw.get("bad_eid") == tag else eid
        np.savez(ev / f"eval_{tag}.npz", pred=p, ade6=ade + (0.01 if runs_kw.get("bad_ade") == tag else 0.0), eid=e)
    return tmp


def _run(tmp, tags=("slow", "mixed", "jitter", "truth")):
    rc = _load().main(["--eval-dir", str(tmp / "eval"), "--tags", *tags, "--out", str(tmp / "o.json"), "--gt-npz", str(tmp / "gt.npz"),
                       "--p10-pred", str(tmp / "absent.npy")])
    return rc, json.loads((tmp / "o.json").read_text(encoding="utf-8"))["runs"]


def test_identities(tmp_path):
    rc, r = _run(_world(tmp_path))
    assert rc == 0 and "P10" not in r
    T, G = r["truth"]["longitudinal"], r["GT"]
    assert T["ade6"] == 0.0 and T["ade6_oracle_arc_scale"] == pytest.approx(0.0, abs=1e-9)
    assert r["truth"]["roughness"] == G["roughness"]
    S = r["slow"]["longitudinal"]
    assert S["ade6"] > 0.5 and S["ade6_oracle_arc_scale"] == pytest.approx(0.0, abs=1e-9) and S["final_arc_bias_m"] < 0
    assert S["ade6_one_global_scale"] == pytest.approx(0.0, abs=1e-6) and S["global_scale"] == pytest.approx(1.25, abs=1e-9)
    M = r["mixed"]["longitudinal"]
    assert M["ade6_oracle_arc_scale"] == pytest.approx(0.0, abs=1e-9) and M["ade6_one_global_scale"] > 0.5 * M["ade6"]
    assert r["jitter"]["roughness"]["accel_mean_mps2"] > 5 * G["roughness"]["accel_mean_mps2"]
    shares = [v["share_of_oracle_gain"] for k, v in S["strata"].items() if k.startswith("dv[") and v["share_of_oracle_gain"] is not None]
    assert sum(shares) == pytest.approx(1.0, abs=1e-9)                     # the dv bins partition the windows


@pytest.mark.parametrize("kw, msg", [(dict(bad_eid="slow"), "is not on the rebuilt eval grid"),
                                     (dict(bad_ade="mixed"), "the ground truth is not its grid")])
def test_refusals(tmp_path, capsys, kw, msg):
    with pytest.raises(SystemExit) as e:
        _run(_world(tmp_path, **kw))
    assert e.value.code == 2 and msg in capsys.readouterr().err
