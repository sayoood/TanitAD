"""RA-1 evaluator: PAIRED arm - P10 deltas for the LONGITUDINAL / LATERAL families (CLAUDE.md "every eval reports four metric families";
"for two arms on the same windows use the PAIRED version, never a combination in quadrature").  CPU only.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_ra1_family_paired.py

What is pinned, and the control that must go red for each:
  * the per-window components reproduce taniteval's OWN single-arm interval points (`all_families` -> ci.components[*].mean), masked
    components included -- so the paired layer cannot drift from the blocks it pairs;
  * the paired delta's point equals the difference of the two single-arm means to 1e-9, computed by hand from the saved arrays; for the
    masked components that is the MICRO mean (sum(err)/sum(valid steps) per arm, the two arms masking differently) -- a MUTATION to a
    mean of window sums turns it RED;
  * an arm equal to P10 reads exactly +0 with the interval [0, 0] and is not "separated"; an arm that is really worse reads positive and separated;
  * `eval` writes the per-window components into `perwindow.npz` and the paired block into `eval.json`, the block is re-derivable from the
    file alone (the gap this closes: pairing used to be impossible afterwards), and the new keys are ADDITIVE -- every pre-existing key of
    eval.json and every pre-existing array of the npz is unchanged;
  * MUTATION: deleting the `arrays.update(fam_arrays)` save line leaves the paired block in eval.json but strips the arrays -- the
    contract check turns RED.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1",
    reason="RA-1 tests are CPU only: run with CUDA_VISIBLE_DEVICES=-1")

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from taniteval import ci  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
import ra1_hjepa_eval as EV  # noqa: E402
import ra1_hjepa_train as TR  # noqa: E402

torch.set_num_threads(2)

H = 30
TINY_FLAGS = ["--d-tok", "4", "--adapter-hidden", "16", "--d1", "16", "--d-up", "16", "--macro-dim", "8", "--enc-hidden", "16",
              "--pred-width", "16", "--pred-depth", "1", "--pred-heads", "2", "--readout-hidden", "16", "--sigreg-proj", "8",
              "--batch", "4", "--warmup", "1", "--log-every", "1", "--ckpt-every", "3"]
HOLD_FLAGS = ["--holdout-n", "2", "--holdout-pool", "6"]
MASKED = [(f, n) for f, n, k, *_ in EV.FAMILY_COMPONENTS if k == "ratio"]
ALL = [(f, n) for f, n, *_ in EV.FAMILY_COMPONENTS]


# ------------------------------------------------------------------------------------------------ synthetic trajectories (no model)
def _paths(v0, acc, yawr, seed=None, dt=None):
    """[N,H,2] float32 ego-frame paths at the eval's own dt: speed clipped at 0 (a stop -> zero-displacement steps -> the `min_ds` mask),
    heading integrated from a constant yaw rate."""
    dt = EV.R.DT if dt is None else dt
    t = (np.arange(H) + 1) * dt
    v = np.clip(v0[:, None] + acc[:, None] * t[None], 0.0, None)
    th = np.cumsum(np.broadcast_to(yawr[:, None] * dt, v.shape), axis=1)
    step = v * dt
    return np.stack([np.cumsum(step * np.cos(th), axis=1), np.cumsum(step * np.sin(th), axis=1)], axis=-1).astype(np.float32)


def _fixture(seed=0, n_ep=12, per=10):
    """gt + an 'arm' + a 'P10' with DIFFERENT error structure, so the two arms mask different steps (an arm that brakes to a stop where
    the human does not, etc.).  Some windows start near-stationary: their masked counts are 0."""
    rng = np.random.default_rng(seed)
    n = n_ep * per
    v0 = rng.uniform(0.0, 14.0, n)
    v0[:12] = rng.uniform(0.0, 0.4, 12)                                    # near-stationary: zero valid steps for the human
    acc = rng.uniform(-3.0, 1.5, n)
    yw = rng.normal(0.0, 0.12, n)
    gt = _paths(v0, acc, yw)
    arm = _paths(np.clip(v0 * (1 + rng.normal(0, 0.15, n)), 0, None), acc + rng.normal(0, 0.4, n), yw + rng.normal(0, 0.05, n))
    p10 = _paths(np.clip(v0 * (1 + rng.normal(0, 0.30, n)), 0, None), acc + rng.normal(0, 0.9, n), yw + rng.normal(0, 0.12, n))
    return gt, arm, p10, np.repeat(np.arange(n_ep), per)


def _pw(path, gt):
    return EV.family_per_window(torch.from_numpy(path), torch.from_numpy(gt), EV.R.DT)


def _hand_mean(c):
    """The component's full-set value from its per-window arrays, with no taniteval and no bootstrap in the way."""
    return float(c["v"].mean()) if c["kind"] == "mean" else float(c["s"].sum() / c["c"].sum())


def test_fixture_can_tell_a_micro_mean_from_a_mean_of_window_means():
    gt, arm, p10, eid = _fixture()
    pa, pp = _pw(arm, gt), _pw(p10, gt)
    for _f, n in MASKED:
        c = pa[n]
        assert c["c"].min() == 0 and c["c"].max() > 0, f"{n}: the fixture must contain windows with no valid step"
        assert (pa[n]["c"] != pp[n]["c"]).any(), f"{n}: the two arms must mask different steps"
        macro = float(np.mean(c["s"][c["c"] > 0] / c["c"][c["c"] > 0]))
        assert abs(macro - _hand_mean(c)) > 1e-3 * max(abs(macro), 1e-9), f"{n}: micro and macro coincide -- the fixture cannot bite"


@pytest.mark.parametrize("which", ["arm", "p10"])
def test_per_window_components_reproduce_taniteval_single_arm_points(which):
    gt, arm, p10, eid = _fixture()
    path = arm if which == "arm" else p10
    win = dict(pred=torch.from_numpy(path), gt=torch.from_numpy(gt), wp_steps=list(range(1, H + 1)), dt_s=EV.R.DT, eid=eid)
    fam = FF.all_families(win, tactical_from_traj=True, tier="T1", n_boot=20, seed=0)
    pw = _pw(path, gt)
    for family, name, kind, _s, _u, _ics in EV.FAMILY_COMPONENTS:
        theirs = fam[family]["ci"]["components"][name]["mean"]          # taniteval's own interval point (full-set value, 4 dp / 6 dp)
        tol = 0.6e-6 if name == "curvature_mae_1pm" else 0.6e-4
        assert abs(_hand_mean(pw[name]) - theirs) <= tol, (name, _hand_mean(pw[name]), theirs)
    # and the headline numbers the blocks print are the same micro means (the block's own rounding is 4 dp / 6 dp)
    assert abs(_hand_mean(pw["heading_mae_deg"]) - fam["lateral"]["heading_mae_deg"]) <= 0.6e-4
    assert abs(_hand_mean(pw["curvature_mae_1pm"]) - fam["lateral"]["curvature_mae_1pm"]) <= 0.6e-6
    assert abs(_hand_mean(pw["cross_mae_m"]) - fam["lateral"]["cross_mae_m"]) <= 0.6e-4
    assert abs(_hand_mean(pw["along_mae_m"]) - fam["longitudinal"]["along_mae_m"]) <= 0.6e-4


def _point_check(module, pa, pp, eid):
    """True iff every component's published arm / P10 / delta_exact equals the by-hand micro / flat means to 1e-9."""
    blk = module.paired_family_block(pa, pp, eid, 40, 0)
    for family, name in ALL:
        c = blk[family][name]
        a, b = _hand_mean(pa[name]), _hand_mean(pp[name])
        if not (abs(c["arm"] - a) < 1e-9 and abs(c["p10"] - b) < 1e-9 and abs(c["delta_exact"] - (a - b)) < 1e-9):
            return False
    return True


def test_paired_delta_point_equals_the_difference_of_the_single_arm_means_to_1e_9():
    gt, arm, p10, eid = _fixture()
    pa, pp = _pw(arm, gt), _pw(p10, gt)
    blk = EV.paired_family_block(pa, pp, eid, 60, 0)
    assert blk["estimator"] == "paired_episode_cluster_bootstrap" and blk["n_windows"] == len(eid) and blk["n_episodes"] == 12
    for family, name, kind, _s, unit, ics in EV.FAMILY_COMPONENTS:
        c = blk[family][name]
        a, b = _hand_mean(pa[name]), _hand_mean(pp[name])
        assert abs(c["delta_exact"] - (a - b)) < 1e-9, name
        # ... and against the single-arm ESTIMATOR's own point at 12 dp (ci.episode_cluster_bootstrap), not only against my arithmetic
        if kind == "mean":
            sa = ci.episode_cluster_bootstrap(pa[name]["v"], eid, n_boot=5, dp=12)["mean"]
            sb = ci.episode_cluster_bootstrap(pp[name]["v"], eid, n_boot=5, dp=12)["mean"]
        else:
            ra = FF._ratio_reducer(pa[name]["s"], pa[name]["c"], 1.0)
            rb = FF._ratio_reducer(pp[name]["s"], pp[name]["c"], 1.0)
            idx = np.arange(len(eid), dtype=np.float64)
            sa = ci.episode_cluster_bootstrap(idx, eid, reduce=ra, n_boot=5, dp=12)["mean"]
            sb = ci.episode_cluster_bootstrap(idx, eid, reduce=rb, n_boot=5, dp=12)["mean"]
        assert abs(c["delta_exact"] - (sa - sb)) < 1e-9, (name, c["delta_exact"], sa - sb)
        assert abs(c["delta"] - c["delta_exact"]) <= 0.51 * 1e-4 / ics + 1e-12, name      # `delta` is ci's rounded rendering of it
        assert c["unit"] == unit and c["estimator"] == "paired_episode_cluster_bootstrap" and c["n_episodes"] == 12
        assert c["lo"] <= c["hi"] and isinstance(c["separated"], bool)
        if kind == "ratio":
            assert c["n_steps_arm"] == int(pa[name]["c"].sum()) and c["n_steps_p10"] == int(pp[name]["c"].sum())
            assert c["n_steps_arm"] != c["n_steps_p10"], "the two arms' masks differ -- that is the case this must handle"
    assert _point_check(EV, pa, pp, eid)


def test_mutation_a_ratio_that_ignores_the_masks_turns_the_point_check_red():
    gt, arm, p10, eid = _fixture()
    pa, pp = _pw(arm, gt), _pw(p10, gt)
    line = '        red = FF._ratio_reducer(np.concatenate([a["s"], b["s"]]), np.concatenate([a["c"], b["c"]]), 1.0, "masked_mean")'
    mutant = _mutant_eval(line, '        red = (lambda S: (lambda idx: float(S[np.asarray(idx, dtype=np.int64)].mean())))(np.concatenate([a["s"], b["s"]]))')
    assert _point_check(EV, pa, pp, eid), "control: the real evaluator passes the check"
    assert not _point_check(mutant, pa, pp, eid), "a mean of window SUMS (no valid-step denominator) must make the check RED"


def test_an_arm_equal_to_p10_reads_exactly_zero_with_a_zero_width_interval():
    gt, arm, p10, eid = _fixture()
    pp = _pw(p10, gt)
    blk = EV.paired_family_block(pp, pp, eid, 80, 0)
    for family, name in ALL:
        c = blk[family][name]
        assert c["delta_exact"] == 0.0 and c["delta"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0, (name, c)
        assert c["separated"] is False and "degenerate" not in c and c["p_delta_gt0"] == 0.0
    # through the production entry point (float64 numpy in, as run_eval holds the paths) the control reads OK ...
    full, arrays = EV.paired_families(p10.astype(np.float64), p10.astype(np.float64), gt.astype(np.float64), eid, EV.R.DT, "g", 40, 0)
    assert full["controls"]["p10_minus_p10_ok"] is True and full["controls"]["n_components_checked"] == len(ALL)
    assert all(full[f][n]["delta_exact"] == 0.0 and full[f][n]["lo"] == 0.0 == full[f][n]["hi"] for f, n in ALL)
    # ... and the instrument can see a real difference: an arm that is really worse than P10 reads positive AND separated
    bad = (p10 + np.random.default_rng(1).normal(0, 1.5, p10.shape)).astype(np.float64)
    worse, _ = EV.paired_families(bad, p10.astype(np.float64), gt.astype(np.float64), eid, EV.R.DT, "g", 80, 0)
    assert worse["controls"]["p10_minus_p10_ok"] is True
    assert worse["lateral"]["cross_mae_m"]["delta_exact"] > 0 and worse["lateral"]["cross_mae_m"]["separated"] is True
    assert worse["lateral"]["cross_mae_m"]["lo"] > 0


def test_the_saved_arrays_round_trip_and_rebuild_the_same_block(tmp_path):
    gt, arm, p10, eid = _fixture()
    blk, arrays = EV.paired_families(arm.astype(np.float64), p10.astype(np.float64), gt.astype(np.float64), eid, EV.R.DT, "g", 50, 3)
    assert set(arrays) == set(EV.family_array_keys("arm")) | set(EV.family_array_keys("p10"))
    assert all(np.asarray(v).shape == (len(eid),) for v in arrays.values())
    np.savez_compressed(tmp_path / "pw.npz", **arrays)
    z = np.load(tmp_path / "pw.npz", allow_pickle=True)
    again = EV.paired_family_block(EV.family_pw_from_npz(z, "arm"), EV.family_pw_from_npz(z, "p10"), eid, 50, 3)
    for family, name in ALL:
        assert again[family][name] == blk[family][name], name


# ------------------------------------------------------------------------------------------------ end to end through `eval`
def _mutant_eval(line, replacement):
    """The evaluator's REAL source with ONE line changed."""
    src = inspect.getsource(EV)
    assert src.count(line) == 1, "the mutated line moved: update this mutation"
    m = types.ModuleType("ra1_hjepa_eval_mutant")
    m.__file__ = EV.__file__
    exec(compile(src.replace(line, replacement), EV.__file__, "exec"), m.__dict__)
    return m


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("ra1fam")
    w = D.make_synthetic_world(root, n_train=6, n_eval=3, T=70, seed=0, d_feat=32)
    prior = M.P10Prior.synthetic(0)
    M.save_p10(prior, root / "p10_prior.pt", {"synthetic": True})
    w["prior_path"] = str(root / "p10_prior.pt")
    w["grid_json"] = str(root / "eval_grid_windows.json")
    return w


@pytest.fixture(scope="module")
def ckpt(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("ra1fam_train")
    TR.main(["--arm", "R0", "--seed", "0", "--cache", world["root"], "--p10-prior", world["prior_path"], "--out", str(out), "--steps", "3",
             "--kin-mode", "steer"] + TINY_FLAGS + HOLD_FLAGS)
    assert (out / "final.pt").exists()
    return str(out / "final.pt")


def _eval(module, world, ckpt, out):
    rc = module.main(["eval", "--ckpt", ckpt, "--eval-cache", world["root"], "--p10-prior", world["prior_path"], "--grid-json", world["grid_json"],
                      "--train-cache", world["root"], "--out", str(out), "--n-boot", "20", "--fam-boot", "40"])
    return rc, json.load(open(out / "eval.json", encoding="utf-8")), np.load(out / "perwindow.npz", allow_pickle=True)


def _missing_arrays(z):
    return [k for k in EV.family_array_keys("arm") + EV.family_array_keys("p10") if k not in z.files]


def test_eval_publishes_the_paired_block_and_saves_the_per_window_components(world, ckpt, tmp_path):
    rc, r, z = _eval(EV, world, ckpt, tmp_path / "ev")
    assert rc == 0 and r["instrument_ok"]
    pb = r["families"]["paired_arm_minus_p10"]
    assert pb["status"] == "OK" and pb["estimator"] == "paired_episode_cluster_bootstrap" and pb["controls"]["p10_minus_p10_ok"] is True
    assert pb["n_windows"] == r["n_windows"] and pb["n_episodes"] == r["n_episodes"] and pb["tier"].startswith("T1")
    assert set(pb["longitudinal"]) == {"speed_mae_mps", "along_mae_m", "accel_mae_mps2"}
    assert set(pb["lateral"]) == {"heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m"}
    assert pb["tactical"]["status"] == "NOT PAIRED" and pb["tactical"]["reason"] and pb["strategic"]["status"] == "PAUSED"
    # the CONTRACT: every per-window array is in the file, one value per eval window (this is what the save line provides)
    assert _missing_arrays(z) == [], "perwindow.npz is missing the family components -- no later pairing is possible"
    assert all(z[k].shape == (r["n_windows"],) for k in EV.family_array_keys("arm") + EV.family_array_keys("p10"))
    # the published block is exactly what the saved arrays give, by hand, to 1e-9
    pa, pp = EV.family_pw_from_npz(z, "arm"), EV.family_pw_from_npz(z, "p10")
    for family, name in ALL:
        c = pb[family][name]
        assert abs(c["delta_exact"] - (_hand_mean(pa[name]) - _hand_mean(pp[name]))) < 1e-9, name
    again = EV.paired_family_block(pa, pp, z["eid"], pb["n_boot"], pb["seed"])
    assert all(again[f][n][k] == pb[f][n][k] for f, n in ALL for k in ("delta", "lo", "hi", "delta_exact", "separated")), \
        "eval.json's paired block must be re-derivable from perwindow.npz alone"
    # and it agrees with the single-arm blocks already in eval.json (their points are 4 dp / 6 dp renderings)
    for family, name, *_ in EV.FAMILY_COMPONENTS:
        for tag, key in (("arm", "arm"), ("p10", "p10")):
            single = r["families"][tag][family]["ci"]["components"][name]["mean"]
            assert abs(pb[family][name][key] - single) <= (0.6e-6 if name == "curvature_mae_1pm" else 0.6e-4), (name, tag)


def test_the_new_keys_are_additive_every_existing_key_and_array_is_unchanged(world, ckpt, tmp_path, monkeypatch):
    rc, on, zon = _eval(EV, world, ckpt, tmp_path / "on")
    assert rc == 0

    def boom(*a, **k):
        raise RuntimeError("paired layer switched off")
    monkeypatch.setattr(EV, "paired_families", boom)
    rc2, off, zoff = _eval(EV, world, ckpt, tmp_path / "off")
    assert rc2 == 0
    assert off["families"]["paired_arm_minus_p10"]["status"] == "ERROR", "a failing paired layer is recorded, never silent"
    drop = ("seconds", "families")
    assert {k: v for k, v in on.items() if k not in drop} == {k: v for k, v in off.items() if k not in drop}
    assert on["families"]["arm"] == off["families"]["arm"] and on["families"]["p10"] == off["families"]["p10"]
    assert on["families"]["lead_braking_stratum_and_distance_keeping"] == off["families"]["lead_braking_stratum_and_distance_keeping"]
    assert set(on["families"]) - {"paired_arm_minus_p10"} == set(off["families"]) - {"paired_arm_minus_p10"}
    new = set(zon.files) - set(zoff.files)
    assert new == set(EV.family_array_keys("arm")) | set(EV.family_array_keys("p10"))
    for k in zoff.files:
        assert np.array_equal(zon[k], zoff[k]), f"existing array {k} changed"


def test_mutation_deleting_the_save_line_turns_the_arrays_contract_red(world, ckpt, tmp_path):
    mutant = _mutant_eval("    arrays.update(fam_arrays)", "    pass")
    assert "arrays.update(fam_arrays)" in inspect.getsource(EV)
    rc, r, z = _eval(EV, world, ckpt, tmp_path / "real")
    assert _missing_arrays(z) == [], "control: the real evaluator saves every family array"
    rc_m, r_m, z_m = _eval(mutant, world, ckpt, tmp_path / "mut")
    assert rc_m == 0 and r_m["families"]["paired_arm_minus_p10"]["status"] == "OK", "the block is still computed in memory ..."
    assert set(_missing_arrays(z_m)) == set(EV.family_array_keys("arm")) | set(EV.family_array_keys("p10")), \
        "... but without the save line NOTHING lands in perwindow.npz: the contract must go RED"
