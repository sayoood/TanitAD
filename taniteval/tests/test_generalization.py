"""Analytic tests for the TanitEval generalization panel (generalization.py).

Synthetic inputs with hand-known answers, CPU-only, no checkpoint. Covers the
divergence-stratification, skill_vs_CTRV, quantile binning, cluster/paired
bootstrap CIs, path-curvature, maneuver-onset / commit lead-time, physical
feasibility, and the ridge probe math — plus two model-free integration tests of
the A (divergence) and B (vision-ablation) composers.

pytest is NOT installed on the eval pod, so these run standalone; they are also
plain ``test_*`` functions collectable by pytest if present.

Run:  PYTHONPATH=/root/taniteval:/root/TanitAD/stack python tests/test_generalization.py
"""
import sys

import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import generalization as G  # noqa: E402


def _approx(a, b, tol=1e-5):
    assert abs(float(a) - float(b)) <= tol, f"{a} != {b} (tol {tol})"


# --------------------------------------------------------------------------- #
# per_window_l2 / ctrv_divergence                                              #
# --------------------------------------------------------------------------- #
def test_per_window_l2_known():
    pred = torch.tensor([[[0., 0.], [3., 0.]]])          # [1,2,2]
    gt = torch.tensor([[[0., 0.], [0., 4.]]])            # step1 err = |[3,-4]|=5
    _approx(G.per_window_l2(pred, gt), 2.5)              # mean(0, 5)
    _approx(G.per_window_l2(pred, gt, idx=[1]), 5.0)     # endpoint only


def test_ctrv_divergence_is_gt_vs_ctrv():
    gt = torch.zeros(1, 3, 2)
    ctrv = torch.tensor([[[0., 0.], [1., 0.], [2., 0.]]])
    _approx(G.ctrv_divergence(gt, ctrv), (0 + 1 + 2) / 3)


# --------------------------------------------------------------------------- #
# quantile_bins                                                               #
# --------------------------------------------------------------------------- #
def test_quantile_bins_equal_count():
    vals = torch.arange(8, dtype=torch.float32)          # 0..7
    lab, edges = G.quantile_bins(vals, n_bins=4)
    counts = torch.bincount(lab, minlength=4)
    assert counts.tolist() == [2, 2, 2, 2], counts.tolist()
    assert lab.tolist() == [0, 0, 1, 1, 2, 2, 3, 3], lab.tolist()
    _approx(edges[0], 0.0); _approx(edges[-1], 7.0)


def test_quantile_bins_monotone_labels():
    vals = torch.tensor([5., 1., 9., 3., 7., 2., 8., 4.])
    lab, _ = G.quantile_bins(vals, n_bins=4)
    # the smallest value must land in bin 0, the largest in bin 3
    assert lab[vals.argmin()].item() == 0
    assert lab[vals.argmax()].item() == 3


# --------------------------------------------------------------------------- #
# skill_vs_ctrv                                                               #
# --------------------------------------------------------------------------- #
def test_skill_vs_ctrv_perfect_and_bad():
    gt = torch.zeros(2, 2, 2)
    ctrv = torch.ones(2, 2, 2) * torch.tensor([1., 0.])  # ctrv_l2 = 1 each
    pred = torch.stack([gt[0], (torch.ones(2, 2) * torch.tensor([2., 0.]))])
    r = G.skill_vs_ctrv(pred, gt, ctrv, idx=[0, 1])
    _approx(r["model_l2"][0], 0.0)                        # perfect window
    _approx(r["ctrv_l2"][0], 1.0)
    _approx(r["skill"][0], 0.0)
    _approx(r["advantage"][0], 1.0)                       # beats CTRV by 1 m
    _approx(r["model_l2"][1], 2.0)                        # worse-than-CTRV window
    _approx(r["skill"][1], 2.0)
    _approx(r["advantage"][1], -1.0)


# --------------------------------------------------------------------------- #
# bootstrap CIs                                                               #
# --------------------------------------------------------------------------- #
def test_cluster_bootstrap_constant_zero_width():
    vals = torch.full((40,), 3.0)
    eid = [i % 8 for i in range(40)]
    m, lo, hi, n = G.cluster_bootstrap_ci(vals.numpy(), eid, n_boot=200)
    _approx(m, 3.0); _approx(lo, 3.0); _approx(hi, 3.0)
    assert n == 40


def test_paired_bootstrap_delta_known():
    a = torch.full((30,), 5.0).numpy()
    b = torch.full((30,), 2.0).numpy()
    eid = [i % 6 for i in range(30)]
    d, lo, hi, p = G.paired_bootstrap_delta_ci(a, b, eid, n_boot=200)
    _approx(d, 3.0); _approx(lo, 3.0); _approx(hi, 3.0)
    _approx(p, 1.0)                                        # delta>0 in every draw


def test_paired_bootstrap_zero_delta():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, generator=g).numpy()
    eid = [i % 8 for i in range(40)]
    d, lo, hi, p = G.paired_bootstrap_delta_ci(x, x, eid, n_boot=200)
    _approx(d, 0.0)                                        # identical -> 0 delta
    _approx(lo, 0.0); _approx(hi, 0.0)


# --------------------------------------------------------------------------- #
# path_curvature_deg                                                          #
# --------------------------------------------------------------------------- #
def test_path_curvature_straight_is_zero():
    gt = torch.tensor([[[1., 0.], [2., 0.], [3., 0.], [4., 0.]]])
    _approx(G.path_curvature_deg(gt)[0], 0.0)


def test_path_curvature_right_angle_is_90():
    gt = torch.tensor([[[1., 0.], [2., 0.], [2., 1.], [2., 2.]]])
    _approx(G.path_curvature_deg(gt)[0], 90.0, tol=1e-3)


# --------------------------------------------------------------------------- #
# maneuver onset + commit lead-time                                           #
# --------------------------------------------------------------------------- #
def test_onset_step_detects_divergence():
    gt = torch.zeros(1, 4, 2)
    ctrv = torch.tensor([[[0., 0.], [0., 0.], [1., 0.], [2., 0.]]])
    assert G.maneuver_onset_step(gt, ctrv, tol_m=0.5)[0].item() == 2


def test_onset_step_no_onset_returns_H():
    gt = torch.zeros(1, 4, 2)
    ctrv = torch.zeros(1, 4, 2)
    assert G.maneuver_onset_step(gt, ctrv, tol_m=0.5)[0].item() == 4   # H, no onset


def test_commit_step_tracks_gt_direction():
    ctrv = torch.zeros(1, 4, 2)
    gt = torch.tensor([[[0., 0.1], [0., 0.4], [0., 0.8], [0., 1.2]]])   # turns +y
    pred = torch.tensor([[[0., 0.05], [0., 0.35], [0., 0.9], [0., 1.3]]])
    assert G.commit_step(pred, gt, ctrv, margin_m=0.3)[0].item() == 1   # |dev|>0.3 @1


def test_commit_step_wrong_direction_never_commits():
    ctrv = torch.zeros(1, 4, 2)
    gt = torch.tensor([[[0., 0.5], [0., 1.0], [0., 1.5], [0., 2.0]]])   # +y
    pred = torch.tensor([[[0., -0.5], [0., -1.], [0., -1.5], [0., -2.]]])  # -y
    assert G.commit_step(pred, gt, ctrv, margin_m=0.3)[0].item() == 4   # never


def test_lead_time_is_onset_minus_commit():
    # onset at step 3, commit at step 1 -> lead 2 frames (anticipates)
    ctrv = torch.zeros(1, 6, 2)
    gt = torch.tensor([[[0, 0.], [0, 0.], [0, 0.], [0, .8], [0, 1.6], [0, 2.4]]])
    pred = torch.tensor([[[0, .1], [0, .5], [0, .9], [0, 1.3], [0, 1.9], [0, 2.5]]])
    onset = G.maneuver_onset_step(gt, ctrv, tol_m=0.5)[0].item()
    commit = G.commit_step(pred, gt, ctrv, margin_m=0.3)[0].item()
    assert onset == 3, onset
    assert commit == 1, commit
    assert onset - commit == 2


# --------------------------------------------------------------------------- #
# physical feasibility                                                        #
# --------------------------------------------------------------------------- #
def test_feasibility_straight_is_feasible():
    pred = torch.tensor([[[1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.]]])
    f = G.path_feasibility(pred, torch.tensor([10.]))
    assert bool(f["feasible"][0]) is True
    _approx(f["kappa_max"][0], 0.0, tol=1e-4)


def test_feasibility_tight_turn_is_infeasible():
    # 90 deg per 0.1 s step at ~10 m/s -> kappa ~1.57 (>0.5) and a_lat huge
    pred = torch.tensor([[[1., 0.], [1., 1.], [0., 1.], [0., 0.], [1., 0.]]])
    f = G.path_feasibility(pred, torch.tensor([10.]))
    assert bool(f["feasible"][0]) is False
    assert float(f["kappa_max"][0]) > G.KAPPA_MAX


# --------------------------------------------------------------------------- #
# ridge probe (stack RidgeProbe) — decodable vs noise                          #
# --------------------------------------------------------------------------- #
def _probe_data(decodable=True, n_ep=10, per=30, f_dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats, target, eid = [], [], []
    w = torch.randn(f_dim, generator=g)
    for e in range(n_ep):
        for _ in range(per):
            x = torch.randn(f_dim, generator=g)
            y = (x @ w) if decodable else torch.randn(1, generator=g).item()
            feats.append(x)
            target.append(float(y) + 0.01 * torch.randn(1, generator=g).item())
            eid.append(e)
    return torch.stack(feats), torch.tensor(target).float(), eid


def test_probe_recovers_linear_signal():
    feats, target, eid = _probe_data(decodable=True, seed=1)
    r2, alpha = G._kfold_probe_r2(feats, target[:, None], eid)
    assert r2 > 0.8, r2                                    # linear -> high held-out R^2


def test_probe_noise_is_near_zero_r2():
    feats, target, eid = _probe_data(decodable=False, seed=2)
    r2, alpha = G._kfold_probe_r2(feats, target[:, None], eid)
    assert r2 < 0.3, r2                                    # no signal -> low R^2


# --------------------------------------------------------------------------- #
# INTEGRATION (model-free): A divergence composer + B vision-ablation composer #
# --------------------------------------------------------------------------- #
def _synthetic_col(N=200, H=20, seed=0, model="perfect"):
    """Windows where divergence (|GT-CTRV|) grows across the set and the model
    tracks GT (so advantage over CTRV == divergence). CTRV is straight (y=0);
    GT ramps in +y with a per-window slope -> divergence increases with slope."""
    g = torch.Generator().manual_seed(seed)
    steps = torch.arange(1, H + 1).float()
    ctrv = torch.zeros(N, H, 2)
    gt = torch.zeros(N, H, 2)
    pred_real = torch.zeros(N, H, 2)
    pred_mean = torch.zeros(N, H, 2)
    slopes = torch.linspace(0.0, 0.3, N)                  # low->high divergence
    for i in range(N):
        gt[i, :, 1] = slopes[i] * steps
        if model == "perfect":
            pred_real[i] = gt[i] + 0.01 * torch.randn(H, 2, generator=g)
        pred_mean[i] = ctrv[i] + 0.01 * torch.randn(H, 2, generator=g)  # ablated=CTRV
    eid = [i % 10 for i in range(N)]
    return {"gt": gt, "ctrv": ctrv, "eid": eid,
            "_preds": {"real": pred_real, "mean": pred_mean,
                       "shuffle": pred_mean.clone()}}


def test_A_composer_advantage_grows_with_divergence():
    col = _synthetic_col(seed=3)
    preds = col.pop("_preds")
    A = G.test_A_divergence(col, preds, n_bins=4)
    advs = [r["advantage_over_ctrv_m"] for r in A["divergence_bins"]]
    assert advs == sorted(advs), advs                     # monotone up
    assert A["advantage_monotone_in_divergence"] is True
    assert A["high_div_advantage_m"] > A["low_div_advantage_m"]
    assert "GENUINE" in A["anticipation_verdict"]


def test_B_composer_headline_collapse_under_ablation():
    col = _synthetic_col(seed=4)
    preds = col.pop("_preds")
    div = G.ctrv_divergence(col["gt"], col["ctrv"], G.WP_IDX)
    lab, _ = G.quantile_bins(div, 4)
    B = G.test_B_vision_ablation(col, preds, div, lab, n_bins=4)
    hl = B["headline"]
    # real vision beats CTRV on the top bin; ablated (=CTRV) does not
    assert hl["high_div_advantage_with_vision_m"] > 0
    _approx(hl["high_div_advantage_without_vision_m"], 0.0, tol=0.05)
    assert hl["high_div_vision_effect_m"] > 0
    assert hl["anticipation_is_vision_based"] is True
    assert "HEADLINE POSITIVE" in hl["verdict"]


def test_B_composer_no_collapse_when_ablation_also_good():
    """Control: if the ablated model ALSO matches GT, vision effect ~ 0 and the
    headline is NOT declared vision-based."""
    col = _synthetic_col(seed=5)
    preds = col.pop("_preds")
    preds["mean"] = preds["real"].clone()                 # ablation loses nothing
    div = G.ctrv_divergence(col["gt"], col["ctrv"], G.WP_IDX)
    lab, _ = G.quantile_bins(div, 4)
    B = G.test_B_vision_ablation(col, preds, div, lab, n_bins=4)
    assert B["headline"]["anticipation_is_vision_based"] is False


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
