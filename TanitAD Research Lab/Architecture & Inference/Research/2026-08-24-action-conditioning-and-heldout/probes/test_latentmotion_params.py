"""Tests for the MM-E19 revision of ``latentmotion.py`` + the runner's parsers.

Covers exactly the two recorded defects (MM-C12) plus the runner's committed
spike instrument:
  * the K parameterisation REACHES THE COMPUTATION (dz = z_{t+k} - z_t), with
    K=4 the default so banked invocations reproduce;
  * the stack refusal FIRES on a bogus/tanitad-less path (exit 2, message),
    and the preflight succeeds + stamps the tree on the real run mirror;
  * the §3c spike-arrival parser reads a synthetic log to known values.

Run from a LOCAL copy (G: cannot RUN code):
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_latentmotion_params.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import latentmotion as LM  # noqa: E402  (imports torch — heavy but hermetic)
import mm_e19_read as RUN  # noqa: E402

PY = sys.executable
WT_STACK = pathlib.Path(r"C:\Users\Admin\tanitad-wt\stack")


# ---- K parameterisation ------------------------------------------------------

def test_defaults_reproduce_banked_invocation():
    cfg = LM.resolve_config([], env={})
    assert cfg.k == 4                       # the banked E-DEC-59 default
    assert (cfg.band_lo, cfg.band_hi) == (0, 8)
    assert cfg.nclips == 80
    assert cfg.device == "cuda"
    assert cfg.null is False
    assert cfg.arms == "rdw8p30k"
    assert str(cfg.corpus).replace("\\", "/").endswith(
        "sp2/cache/physicalai-val130-heldout")


def test_k_flag_env_and_precedence():
    assert LM.resolve_config(["--k", "60"], env={}).k == 60
    assert LM.resolve_config([], env={"SPD_K": "60"}).k == 60
    assert LM.resolve_config(["--k", "60"], env={"SPD_K": "4"}).k == 60  # CLI wins


def test_band_parse():
    cfg = LM.resolve_config(["--band", "8:16"], env={})
    assert (cfg.band_lo, cfg.band_hi) == (8, 16)
    cfg = LM.resolve_config([], env={"SPD_BAND": "8:16"})
    assert (cfg.band_lo, cfg.band_hi) == (8, 16)


def test_k_reaches_the_computation():
    """dz really is z_{t+K} - z_t for the K that was passed."""
    T, D = 200, 5
    zt = np.arange(T * D, dtype=np.float64).reshape(T, D)
    a = np.zeros((T, 2))
    v = np.zeros(T)
    yaw = np.linspace(0, 1, T + 1)
    for K in (4, 60):
        rows = LM.clip_rows(zt, a, v, yaw, K)
        assert rows is not None
        ztr, ego, dz = rows
        m = min(T - K, (T + 1) - K - 1)
        i = np.arange(m)
        assert len(dz) == m and len(ztr) == m and ego.shape == (m, 3)
        assert np.array_equal(dz, zt[i + K] - zt[i])
        assert np.array_equal(ztr, zt[i])
    dz4 = LM.clip_rows(zt, a, v, yaw, 4)[2]
    dz60 = LM.clip_rows(zt, a, v, yaw, 60)[2]
    assert not np.array_equal(dz4[: len(dz60)], dz60)


def test_short_clip_skipped_exactly_as_banked():
    """< 30 usable rows -> None (the banked skip), and K=60 shrinks m."""
    T, D = 33, 4
    zt = np.zeros((T, D))
    a = np.zeros((T, 2))
    v = np.zeros(T)
    yaw = np.zeros(T + 1)
    assert LM.clip_rows(zt, a, v, yaw, 4) is None       # m = 29 < 30
    T = 89                                              # m = 29 at K=60
    assert LM.clip_rows(np.zeros((T, D)), np.zeros((T, 2)), np.zeros(T),
                        np.zeros(T + 1), 60) is None
    T = 90                                              # m = 30 at K=60
    assert LM.clip_rows(np.zeros((T, D)), np.zeros((T, 2)), np.zeros(T),
                        np.zeros(T + 1), 60) is not None


# ---- stack refusal (MM-C12) --------------------------------------------------

def _preflight(stack: str):
    return subprocess.run(
        [PY, str(HERE / "latentmotion.py"), "--stack", stack,
         "--preflight-only"],
        capture_output=True, text=True, timeout=300)


def test_refusal_fires_on_nonexistent_stack(tmp_path):
    r = _preflight(str(tmp_path / "no-such-mirror" / "stack"))
    assert r.returncode == 2
    blob = r.stderr + r.stdout
    assert "REFUSED" in blob
    assert "no-such-mirror" in blob


def test_refusal_fires_on_dir_without_tanitad(tmp_path):
    (tmp_path / "stack").mkdir()
    r = _preflight(str(tmp_path / "stack"))
    assert r.returncode == 2
    assert "tanitad/__init__.py" in (r.stderr + r.stdout).replace("\\", "/")


@pytest.mark.skipif(not (WT_STACK / "tanitad" / "__init__.py").is_file(),
                    reason="run mirror not present on this box")
def test_preflight_ok_and_stamps_the_tree():
    r = _preflight(str(WT_STACK))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tanitad imported from" in r.stdout
    assert "tanitad-wt" in r.stdout


# ---- runner parsers ----------------------------------------------------------

def test_spike_arrival_on_synthetic_log(tmp_path):
    p = tmp_path / "train_log.jsonl"
    lines = ['{"run_start": {"run": "x"}}']            # header row -> ignored
    for s in range(200, 8001, 200):
        g = 100000.0 if s in (5800, 6400) else 2.0
        lines.append(json.dumps({"step": s, "gnorm": g, "o5_loss": 0.2}))
    lines.append(json.dumps({"step": 8000, "spectrum": [1]}))  # no gnorm
    p.write_text("\n".join(lines), encoding="utf-8")
    out = RUN.parse_spike_arrival(p)
    assert out["max_step"] == 8000
    assert [d["step"] for d in out["spike_steps"]] == [5800, 6400]
    w = {tuple(x["steps"]): x for x in out["windows"]}
    assert w[(0, 2000)]["spikes_gt50"] == 0
    assert w[(4000, 6000)]["spikes_gt50"] == 1
    assert w[(4000, 6000)]["rate_per_1000"] == 0.5
    assert w[(6000, 8000)]["spikes_gt50"] == 1
    assert out["spiking_regime_bounds_steps"] == [5800, 6400]
    assert out["clean_steps_since_last_spike"] == 1600
    assert out["first_half_rate_per_1000"] == 0.0
    assert out["second_half_rate_per_1000"] > 0


def test_args_diff_is_the_one_variable_check():
    d = RUN.args_diff({"o5_k": 60, "clip": 0.5, "lr": 1e-4},
                      {"o5_k": 8, "clip": 1.0, "lr": 1e-4})
    assert set(d) == {"o5_k", "clip"}
    assert d["o5_k"] == {"arm": 60, "incumbent": 8}
