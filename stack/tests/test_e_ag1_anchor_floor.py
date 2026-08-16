"""Tests for E-AG1/2/3 — the ANCHOR_GOAL quantisation floor instrument.

The properties pinned here are the ones whose failure would produce a NUMBER
rather than an error, which is the only failure mode that costs a decision:

  * the multi-target ridge must equal the programme's incumbent single-target
    solver (a fast path that changed an answer would be invisible);
  * a 2 s anchor vocabulary must REFUSE a 6 s ground truth rather than score it;
  * the 6 s branch must emit NO verdict, under every input — §5.3 fired REDERIVE;
  * ``required_top1`` must return ``None`` when the floor already exceeds the
    target, never 1.0 (a refusal rendered as "needs a perfect classifier" reads
    as difficulty when it is impossibility);
  * the LOEO vocabulary must be episode-disjoint;
  * no path to ``tanitad.data.situations`` may exist — the binding
    goal/situation disjointness rule, enforced rather than documented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e_ag1_anchor_floor as AG                                    # noqa: E402
import e_wc2_sigma_star as E                                       # noqa: E402
from probe_latent_state import RIDGE_LAMBDAS, RidgeSVD, _standardize  # noqa


# --------------------------------------------------------------------------- #
# the solver                                                                   #
# --------------------------------------------------------------------------- #
def test_fold_solver_matches_ridgesvd():
    """The multi-target fast path must be the incumbent solver, exactly."""
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(120, 17)), rng.normal(size=(30, 17))
    y = rng.normal(size=120)
    solver = AG.FoldSolver(Xtr, Xte)
    got, lam = solver.predict(y)
    Xtr_s, Xte_s = _standardize(Xtr, Xte)
    ref = RidgeSVD(Xtr_s)
    w, b = ref.fit(y, ref.best_lambda(y, RIDGE_LAMBDAS))
    assert lam == pytest.approx(ref.best_lambda(y, RIDGE_LAMBDAS))
    assert np.allclose(got[:, 0], Xte_s @ w + b, atol=1e-9)


def test_fold_solver_shares_one_factorisation_across_targets():
    """Two targets fitted through one solver must equal two independent fits —
    the property that makes the cache a speed-up and not a different model."""
    rng = np.random.default_rng(1)
    Xtr, Xte = rng.normal(size=(90, 11)), rng.normal(size=(20, 11))
    Y = rng.normal(size=(90, 2))
    both, _ = AG.FoldSolver(Xtr, Xte).predict(Y)
    a, _ = AG.ridge_multi(Xtr, Y[:, :1], Xte)
    # a shared GCV lambda is chosen over the SUM, so a single-target refit is
    # only equal when the same lambda wins; assert the columns are consistent
    # with ONE design matrix by checking the joint fit against a manual solve.
    Xtr_s, Xte_s = _standardize(Xtr, Xte)
    mx = Xtr_s.mean(0)
    U, s, Vt = np.linalg.svd(Xtr_s - mx, full_matrices=False)
    lam = AG.FoldSolver(Xtr, Xte).predict(Y)[1]
    my = Y.mean(0)
    W = Vt.T @ ((s / (s ** 2 + lam))[:, None] * (U.T @ (Y - my)))
    assert np.allclose(both, Xte_s @ W + (my - mx @ W), atol=1e-9)
    assert a.shape == (20, 1)


# --------------------------------------------------------------------------- #
# the vocabularies                                                             #
# --------------------------------------------------------------------------- #
def test_fps_prefix_property_makes_the_k_sweep_valid():
    """FPS is greedy, so the first k of a K-anchor set IS the k-anchor set.

    The K sweep compares vocabularies that are NESTED; if this failed, two rows
    of the sweep would differ by construction as well as by size and the
    monotonicity of the floor in K would be uninterpretable.
    """
    rng = np.random.default_rng(3)
    pool = rng.normal(size=(400, 2)) * np.array([12.0, 2.0])
    big = AG.fps_anchors(pool, 16, seed=0)
    small = AG.fps_anchors(pool, 4, seed=0)
    assert np.allclose(big[:4], small)


def test_kmeans_quantisation_is_not_worse_than_fps_on_a_skewed_pool():
    """Lloyd descends mean ‖x − anchor‖², which IS the σ statistic; FPS descends
    coverage of the support. On a density-skewed pool (74 % straight, the corpus'
    own shape) k-means must therefore win on σ — the trade-off the report states."""
    rng = np.random.default_rng(4)
    dense = rng.normal(size=(800, 2)) * 0.4
    tail = rng.normal(size=(40, 2)) * 0.4 + np.array([25.0, 12.0])
    pool = np.concatenate([dense, tail])
    km = AG.kmeans_anchors(pool, 8, seed=0)
    fp = AG.fps_anchors(pool, 8, seed=0)
    msq_km = float(((pool - km[AG._assign_ids(pool, km)]) ** 2).sum(1).mean())
    msq_fp = float(((pool - fp[AG._assign_ids(pool, fp)]) ** 2).sum(1).mean())
    assert msq_km <= msq_fp


def test_kmeans_is_deterministic():
    rng = np.random.default_rng(5)
    pool = rng.normal(size=(300, 2))
    assert np.allclose(AG.kmeans_anchors(pool, 6, seed=2),
                       AG.kmeans_anchors(pool, 6, seed=2))


def test_assign_runner_up_is_never_closer_than_the_oracle():
    rng = np.random.default_rng(6)
    pool, anchors = rng.normal(size=(50, 2)), rng.normal(size=(9, 2))
    a = AG.assign(pool, anchors)
    d0 = (a["resid"] ** 2).sum(1)
    d1 = (a["resid_second"] ** 2).sum(1)
    assert np.all(d1 >= d0 - 1e-12)


# --------------------------------------------------------------------------- #
# the accuracy algebra — where a refusal could be rendered as difficulty        #
# --------------------------------------------------------------------------- #
def test_required_top1_is_none_when_the_floor_already_exceeds_the_target():
    """⛔ THE REFUSAL. If a PERFECT classifier still misses the bar, the answer
    is impossibility, not "needs accuracy 1.0"."""
    assert AG.required_top1(0.80, msq_hit=8.0, msq_miss=40.0) is None


def test_required_top1_inverts_the_sigma_algebra():
    p = AG.required_top1(np.sqrt((0.6 * 2.0 + 0.4 * 30.0) / 2.0),
                         msq_hit=2.0, msq_miss=30.0)
    assert p == pytest.approx(0.6, abs=1e-9)


def test_required_top1_is_clamped_at_zero_not_negative():
    assert AG.required_top1(50.0, msq_hit=1.0, msq_miss=2.0) == 0.0


# --------------------------------------------------------------------------- #
# the verdict                                                                  #
# --------------------------------------------------------------------------- #
def test_prereg_thresholds_are_the_published_ratios_times_the_published_ade():
    assert AG.PREREG["sigma_funded_m"] == pytest.approx(1.7 * 0.4714)
    assert AG.PREREG["sigma_refused_m"] == pytest.approx(3.0 * 0.4714)
    assert AG.PREREG["sigma_funded_m"] == pytest.approx(0.80138, abs=1e-5)
    assert AG.PREREG["sigma_refused_m"] == pytest.approx(1.41420, abs=1e-5)


@pytest.mark.parametrize("best,want", [
    (0.5, "FLOOR_CLEARS"), (1.0, "INCONCLUSIVE"), (2.0, "REFUSED")])
def test_decide_ag1_is_three_sided(best, want):
    assert AG.decide_ag1({8: 3.0, 32: best})["verdict"] == want


def test_decide_ag1_refuses_a_weak_verdict_on_an_empty_sweep():
    assert AG.decide_ag1({})["verdict"] == "NO_VERDICT"


# --------------------------------------------------------------------------- #
# the 6 s rule — §5.3 fired REDERIVE, so no 6 s threshold may exist             #
# --------------------------------------------------------------------------- #
def _synthetic_dump(n_ep=6, per_ep=14, dim=9, seed=0):
    rng = np.random.default_rng(seed)
    eid, gt2, gt6 = [], [], []
    for e in range(n_ep):
        v = 4.0 + 2.0 * e
        for _ in range(per_ep):
            eid.append(e)
            gt2.append([v * 2.0 + rng.normal(0, 0.3), rng.normal(0, 0.3)])
            gt6.append([v * 6.0 + rng.normal(0, 1.0), rng.normal(0, 1.0)])
    n = len(eid)
    feats = np.asarray(gt2) @ rng.normal(size=(2, dim)) + rng.normal(size=(n, dim))
    valid = np.ones((n, 2), dtype=bool)
    valid[-3:, 1] = False                       # the 6 s tail runs past the end
    return {
        "eid": eid,
        "pooled": torch.tensor(feats, dtype=torch.float32),
        "gt_endpoint": torch.tensor(np.stack([gt2, gt6], 1), dtype=torch.float32),
        "endpoint_valid": torch.tensor(valid),
        "endpoint_steps": [20, 60],
    }


def test_no_branch_ever_emits_a_6s_verdict():
    rep = AG.run(_synthetic_dump(), features=["pooled"],
                 declared={"pooled": "VISION_ONLY"}, ks=(4,), methods=("fps",),
                 steps=(20, 60), n_boot=40)
    for meth in rep["horizons"]["6s"]["vocabularies"].values():
        assert meth["verdict"]["verdict"] is None
        assert "REDERIVE" in meth["verdict"]["meaning"]
    for meth in rep["horizons"]["2s"]["vocabularies"].values():
        assert meth["verdict"]["verdict"] in AG.VERDICTS


def test_invalid_endpoint_rows_are_excluded_with_n_reported_never_imputed():
    rep = AG.run(_synthetic_dump(), features=["pooled"],
                 declared={"pooled": "VISION_ONLY"}, ks=(4,), methods=("fps",),
                 steps=(20, 60), n_boot=40)
    assert rep["horizons"]["6s"]["endpoint_excluded"] == 3
    assert rep["horizons"]["6s"]["n"] == rep["horizons"]["2s"]["n"] - 3
    assert "NEVER imputed" in rep["horizons"]["6s"]["endpoint_exclusion_reason"]


def test_shipped_vocab_refuses_a_horizon_mismatch(tmp_path):
    """⛔ A 2 s anchor scored against a 6 s ground truth would LOOK like a
    result. It must be refused by name."""
    p = tmp_path / "v.pt"
    torch.save({"anchors": torch.zeros(4, 4, 2), "horizons": [5, 10, 15, 20]}, p)
    with pytest.raises(ValueError, match="refusing"):
        AG.shipped_vocab_arm(str(p), np.zeros((5, 2)), [0] * 5, step=60)
    out = AG.shipped_vocab_arm(str(p), np.zeros((5, 2)), [0, 0, 1, 1, 1], step=20)
    assert out["k"] == 4 and out["oracle"]["sigma_perax_m"] == 0.0


# --------------------------------------------------------------------------- #
# admissibility, enforced                                                      #
# --------------------------------------------------------------------------- #

def _imported_module_names(path) -> set[str]:
    """Every module name this file IMPORTS, from its AST — including
    function-local and lazily-deferred imports (``ast.walk`` sees the whole tree).

    ⛔ WHY AN AST AND NOT A SUBSTRING SCAN. Both modules under test DOCUMENT the
    rule in prose — their docstrings contain the literal text
    ``tanitad.data.situations`` in the sentence saying they never read it, and
    ``e_ag1_anchor_floor.run()`` emits it in a provenance string. A grep-style
    guard would match its OWN DOCUMENTATION and fire on a clean module. An AST
    walk reads only real `import` statements, so it cannot.
    """
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{a.name}" if base else a.name
                         for a in node.names)
    return names


def _reaches_situations(names) -> list[str]:
    return sorted(n for n in names
                  if n == "situations" or n.endswith(".situations")
                  or ".situations." in n)


def test_no_situation_classifier_path():
    """⛔ BINDING (PI 2026-08-03): the goal path and the situation path stay
    information-disjoint. The label deriver must not read the classifier, in any
    form — so the module must not reference it at all."""
    src = Path(AG.__file__).read_text(encoding="utf-8")
    for token in ("detect_lane_change", "detect_intersection",
                  "detect_roundabout", "situations_from_poses",
                  "anticipation_target"):
        assert token not in src, f"{token} reached the ANCHOR_GOAL instrument"
    # every real import, from the AST — immune to the module's own prose
    assert _reaches_situations(_imported_module_names(AG.__file__)) == []
    # ⚠️ and TRANSITIVELY — checked in a SUBPROCESS, because `sys.modules` in a
    # full-suite run is polluted by every other test that legitimately imports
    # the situation detectors. An in-process check would pass alone and fail in
    # the suite, i.e. it would report the SESSION's imports, not this module's.
    import subprocess
    scripts = str(Path(AG.__file__).resolve().parent)
    code = ("import sys; sys.path.insert(0, r'%s'); import e_ag1_anchor_floor; "
            "print('LEAK' if 'tanitad.data.situations' in sys.modules else 'CLEAN')"
            % scripts)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(Path(AG.__file__).resolve().parents[1]))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "CLEAN" in r.stdout, r.stdout + r.stderr[-2000:]


def test_v0_is_ADMITTED_without_the_echo_flag_after_the_PI_ruling():
    """⭐ SUPERSEDED 2026-08-16, and the supersession is the test.

    This used to assert ``pytest.raises(PermissionError, match="ECHO")`` — v0 was
    refused as an inference-time echo. Sayed ruled otherwise: *"We can use v0 as
    input since it is measured and is not the future…"* v0 is measured PRESENT
    state, so the refusal is gone and no flag is needed."""
    d = _synthetic_dump()
    d["v0"] = torch.tensor(np.linspace(4.0, 16.0, len(d["eid"])),
                           dtype=torch.float32)
    rep = AG.run(d, features=["pooled", "v0"],
                 declared={"pooled": "VISION_ONLY"},
                 ks=(4,), methods=("fps",), steps=(20,), n_boot=20)
    assert rep["_admissibility"]["features"]["any_echo"] is False
    assert rep["_admissibility"]["features"]["any_measured_present"] is True


def test_admitting_v0_stamps_the_ANTI_ECHO_OBLIGATION_not_an_adjudication():
    """⛔ …but the refusal is replaced by an OBLIGATION, not by nothing. "…we
    should assure that the model/planner later is not cheating by just outputting
    v0 as longitudinal plan." """
    d = _synthetic_dump()
    d["v0"] = torch.tensor(np.linspace(4.0, 16.0, len(d["eid"])),
                           dtype=torch.float32)
    rep = AG.run(d, features=["pooled", "v0"],
                 declared={"pooled": "VISION_ONLY"},
                 ks=(4,), methods=("fps",), steps=(20,), n_boot=20)
    st = rep["_admissibility"]["v0_status"]
    assert "PENDING_PI_ADJUDICATION" not in st          # the contradiction is gone
    assert "ADJUDICATED" in st and "2026-08-16" in st
    assert "v0_antiecho" in st and "--speed-echo-control" in st
    assert "3.527" in st and "1.1888" in st             # the deficit that remains
    # and nothing is stamped when v0 is not in the design matrix
    plain = AG.run(_synthetic_dump(), features=["pooled"],
                   declared={"pooled": "VISION_ONLY"},
                   ks=(4,), methods=("fps",), steps=(20,), n_boot=20)
    assert plain["_admissibility"]["v0_status"] is None


# --------------------------------------------------------------------------- #
# LOEO disjointness of the VOCABULARY, not only of the fit                      #
# --------------------------------------------------------------------------- #
def test_loeo_vocabulary_never_contains_a_held_out_endpoint():
    """An episode whose endpoints are far from every other episode's must be
    quantised BADLY — proof its own points were not in the pool that built the
    anchors it is scored against. (A vocabulary built on all 40 would place an
    anchor right on it and the floor would be a leak.)"""
    rng = np.random.default_rng(7)
    gt, eid = [], []
    for e in range(5):
        for _ in range(12):
            gt.append([10.0 * e, rng.normal(0, 0.05)])
            eid.append(e)
    gt.extend([[900.0, 900.0]] * 12)            # a lone, far-away episode
    eid.extend([99] * 12)
    gt = np.asarray(gt, dtype=np.float64)
    arms = AG.loeo_anchor_arms(gt, eid, None, 4, "kmeans", seed=0)
    out = np.asarray(eid) == 99
    far = np.linalg.norm(arms["resid"]["oracle"][out], axis=1)
    near = np.linalg.norm(arms["resid"]["oracle"][~out], axis=1)
    assert far.min() > 100.0 > near.max()


def test_marginal_arm_is_the_goal_echo_null():
    """`marginal` must be the vocabulary's own centroid — a per-window-constant
    goal, i.e. exactly the zero-information control §1.4 requires beside a goal."""
    gt = np.stack([np.linspace(0, 30, 40), np.zeros(40)], 1)
    eid = [i // 10 for i in range(40)]
    arms = AG.loeo_anchor_arms(gt, eid, None, 4, "kmeans", seed=0)
    marg = gt - arms["resid"]["marginal"]
    for e in set(eid):                       # constant within a fold
        rows = marg[np.asarray(eid) == e]
        assert np.allclose(rows, rows[0])
