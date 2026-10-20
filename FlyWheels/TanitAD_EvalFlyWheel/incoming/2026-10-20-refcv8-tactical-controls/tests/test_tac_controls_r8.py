"""Literal tests for tac_controls_r8 (synthetic decision dumps with known answers) + a deliberate-regression arm.

    PYTHONPATH=<taniteval> python -m pytest -q tests/test_tac_controls_r8.py
    TAC_CONTROLS_MUTATION=constant_from_pred ... -> test_constant_control_reads_the_label_majority must go RED
"""
import importlib, json, os, subprocess, sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")
sys.path.insert(0, CODE)


def _write_tree(tmp, seed=0, n_ep=10, run_zero=True):
    d = os.path.join(tmp, f"dump_s{seed}", "decisions")
    os.makedirs(d, exist_ok=True)
    for e in range(n_ep):
        n = 10
        lat_label = np.array([0] * 9 + [6], np.int64)            # 90 % class 0 in every episode
        lat_true = np.array([6] * 10, np.int64)                  # the head always says 6 -> acc 0.10
        lon_label = np.array([1] * 5 + [3] * 5, np.int64)
        lon_true = lon_label.copy()                               # perfect -> acc 1.0, constant 0.5
        z = dict(ws=np.arange(n), nav_cmd=np.array([2] * n, np.int64), lat_label=lat_label, lon_label=lon_label,
                 lat_pred_nav_true=lat_true, lon_pred_nav_true=lon_true,
                 lat_pred_nav_zero=(lat_label.copy() if run_zero else np.full(n, -1)),   # nav-zero is perfect on lat
                 lon_pred_nav_zero=(np.full(n, 1) if run_zero else np.full(n, -1)),      # nav-zero lon = constant 1 -> 0.5
                 lat_pred_nav_shuffled=np.full(n, -1), lon_pred_nav_shuffled=np.full(n, -1))
        np.savez(os.path.join(d, f"ep{e:03d}.npz"), **z)
    return tmp


def _run(tree, out):
    import tac_controls_r8 as T
    importlib.reload(T)
    T.main([tree, "--out", out, "--seeds", "0", "1", "--n-boot", "200"])
    return json.load(open(out, encoding="utf-8"))


def test_constant_control_reads_the_label_majority(tmp_path):
    r = _run(_write_tree(str(tmp_path)), str(tmp_path / "o.json"))["seeds"]["0"]
    lat = r["lat"]
    assert (lat["acc_nav_true"], lat["constant_class"], lat["acc_constant"]) == (0.1, 0, 0.9)
    assert lat["head_minus_constant"]["delta"] == -0.8 and lat["head_minus_constant"]["separated"] is True
    assert lat["reading"] == "BELOW the constant control, separated"


def test_intervention_and_not_run_literals(tmp_path):
    r = _run(_write_tree(str(tmp_path)), str(tmp_path / "o.json"))["seeds"]["0"]
    assert r["lat"]["acc_nav_zero"] == 1.0 and r["lat"]["head_minus_nav_zero"]["delta"] == -0.9
    assert r["lon"]["acc_nav_true"] == 1.0 and r["lon"]["acc_constant"] == 0.5 and r["lon"]["head_minus_nav_zero"]["delta"] == 0.5
    assert r["lat"]["head_minus_nav_shuffled"]["status"] == "NOT_RUN" and r["lat"]["acc_nav_shuffled"] is None
    assert r["lat"]["echo_index_pred_eq_token_mode"] == 1.0
    assert r["lon"]["macro_f1_nav_true"] == 1.0 and r["lat"]["macro_f1_nav_true"] == round((0.0 + 2 * 10 / (2 * 10 + 90)) / 2, 4)


def test_absent_seed_reads_absent(tmp_path):
    d = _run(_write_tree(str(tmp_path)), str(tmp_path / "o.json"))
    assert d["seeds"]["1"]["status"] == "ABSENT"


@pytest.mark.skipif(os.environ.get("TAC_CONTROLS_MUTATION"), reason="the mutation runner itself")
def test_mutation_constant_from_pred_goes_red():
    env = dict(os.environ, TAC_CONTROLS_MUTATION="constant_from_pred", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__ + "::test_constant_control_reads_the_label_majority"],
                       capture_output=True, text=True, env=env, cwd=os.path.dirname(HERE), timeout=600)
    assert r.returncode != 0, "the constant control taken from the PREDICTIONS stayed green"
