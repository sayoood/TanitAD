"""HiCAP dataset + training loop + evaluation smoke tests on synthetic cached clips (CPU)."""
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from tanitad.hicap import vocab as V, losses as L
from tanitad.hicap.data import (HiCAPWindows, collate, make_synth_clip, make_synth_v9, v9rows_from_dict, targets_from_distances,
                                compute_priors, ego_state, to_ego, factor_distances)
from tanitad.hicap.model import HiCAP, HiCAPConfig
from tanitad.hicap.train import train, TrainCfg
from tanitad.hicap.evaluate import evaluate

SMALL = dict(d=64, layers=2, heads=4, d_v=48, n_vlm_tokens=8, n_motion_tokens=16, d_text=32, d_audio=24, rerank_layers=1)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("cache")); rng = np.random.default_rng(0)
    shas = [f"{i:012x}" for i in range(6)]
    meta = [make_synth_clip(d, s, 1000 + i, rng) for i, s in enumerate(shas)]
    rows = v9rows_from_dict(make_synth_v9(meta, shas))
    # vocabulary from the clips' own windows (TRAIN-only rule: here all synthetic)
    paths, lat, dvs, lon, v0s = [], [], [], [], []
    for s in shas:
        c = np.load(f"{d}/{s}.npz"); P = c["poses"]
        for k in range(5, 90, 6):
            p, dv = V.extract_path_profile(to_ego(P, k), float(P[k, 3])); paths.append(p); dvs.append(dv); v0s.append(float(P[k, 3]))
            lat.append(int(np.sign(0) if abs(P[k + 30, 2] - P[k, 2]) < 0.05 else (1 if P[k + 30, 2] > P[k, 2] else 2))); lon.append(6)
    vocab = V.build_factored_vocab(np.array(paths), lat, np.array(dvs), lon, v0s, K_P=24, K_V=12, floor_P=2, floor_V=2)
    c = HiCAPConfig(**SMALL); rng = np.random.default_rng(1)
    pf = np.stack([V.path_features(p) for p in vocab.paths]); qf = np.stack([V.profile_features(x) for x in vocab.profiles])
    Vt = L.VocabTensors(rng.normal(size=(7, 32)), rng.normal(size=(7, 4)), rng.normal(size=(3, 32)), rng.normal(size=(3, 4)),
                        rng.normal(size=(7, 32)), rng.normal(size=(7, 4)), pf, vocab.path_cls, qf, vocab.prof_cls,
                        strat_mask=np.array([1, 1, 1, 1, 1, 0, 0], bool), cell_bias=V.pair_admissibility(np.ones((3, 7))))
    ds = HiCAPWindows(d, rows, vocab, train=True, stride=3)
    return dict(d=d, rows=rows, vocab=vocab, Vt=Vt, ds=ds, c=c)


def test_ego_state_has_three_channels_and_no_curvature(setup):
    it = setup["ds"][0]
    assert it["ego"].shape == (3,)


def test_item_fields_and_shapes(setup):
    it = setup["ds"][5]; vc = setup["vocab"]
    assert it["frames"].shape == (3, 3, 64, 160) and it["fut"].shape == (60, 2)
    assert it["dP"].shape == (vc.K_P,) and it["dQ"].shape == (vc.K_V,) and it["goal_y"].shape == (22,)
    assert 0.0 <= float(it["age_s"]) < 0.5 + 1e-6                      # 2 Hz refresh => age < 0.5 s
    assert float(it["goal_w"][7]) == 0.0                                # SPEED_BAND never supervised


def test_window_validity_no_future_leak(setup):
    ds = setup["ds"]
    for s, j in ds.index:
        k = j
        assert k + 60 < ds.clips[s]["T"] and k >= 3


def test_factor_distance_zero_for_own_trajectory(setup):
    vc = setup["vocab"]; it = setup["ds"][3]
    d_P, d_Q = factor_distances(it["path_log"].numpy(), it["dv_log"].numpy(), float(it["v0"]), vc.paths, vc.profiles)
    assert d_P.min() >= 0 and d_Q.min() >= 0
    d_P2, d_Q2 = factor_distances(vc.paths[2], vc.profiles[1], 5.0, vc.paths, vc.profiles)
    assert d_P2[2] < 1e-5 and d_Q2[1] < 1e-5


def test_targets_normalised_and_candidates(setup):
    b = collate([setup["ds"][i] for i in range(4)]); b = targets_from_distances(b, vocab=setup["vocab"])
    assert torch.allclose(b["yP"].sum(-1), torch.ones(4), atol=1e-5) and b["refine_t"].shape == (4, 5, 16)


def test_priors_are_log_probabilities(setup):
    p = compute_priors(setup["ds"], max_n=40)
    for k in ("s2lat", "s2lon", "s3P", "s3V"): assert abs(np.exp(p[k]).sum() - 1) < 1e-6


def test_three_stage_training_runs_and_loss_falls(setup, tmp_path):
    torch.manual_seed(0)
    m = HiCAP(setup["c"])
    cfg = TrainCfg(steps=(25, 10, 10), batch=16, workers=0, log_every=5, ckpt_dir=str(tmp_path), device="cpu")
    hist = train(m, setup["Vt"], setup["vocab"], setup["ds"], cfg, log=lambda *_: None)
    s1 = [h["loss"] for h in hist if h["stage"] == 1]
    assert s1[-1] < s1[0] and all(np.isfinite(h["loss"]) for h in hist)
    assert {h["stage"] for h in hist} == {1, 2, 3} and any("cost" in h or "ref" in h for h in hist if h["stage"] == 3)
    assert (tmp_path / "hicap_stage3_seed0.pt").exists()
    assert all(not p.requires_grad for p in m.rerank.parameters()) is False        # re-ranker unfrozen in stage 3


def test_rerank_frozen_in_stage_1_2(setup, tmp_path):
    m = HiCAP(setup["c"]); before = [p.clone() for p in m.rerank.parameters()]
    train(m, setup["Vt"], setup["vocab"], setup["ds"], TrainCfg(steps=(3, 3), batch=8, workers=0, log_every=3, ckpt_dir=str(tmp_path), device="cpu", stages=(1, 2)), log=lambda *_: None)
    assert all(torch.equal(a, b) for a, b in zip(before, m.rerank.parameters()))


def test_evaluate_four_families(setup):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "taniteval"))
    m = HiCAP(setup["c"])
    r = evaluate(m, setup["Vt"], setup["vocab"], setup["ds"], n_max=24, decider="argmax")
    assert r["n_windows"] == 24 and r["n_episodes"] >= 2
    f = r["families"]
    assert {"longitudinal", "lateral", "tactical", "strategic", "trajectory"} <= set(f)
    assert f["strategic"]["status"] == "UNAVAILABLE" and f["longitudinal"]["headway_ttc"]["status"] == "UNAVAILABLE"
    assert "paired_model_minus_hold" in f["longitudinal"]["speed_mae"]


def test_evaluate_cascade_decider(setup):
    m = HiCAP(setup["c"])
    r = evaluate(m, setup["Vt"], setup["vocab"], setup["ds"], n_max=4, decider="cascade")
    assert r["decider"] == "cascade" and r["families"]["trajectory"]["ade_2s"]["model"]["mean"] >= 0
