"""Tests for E0j / C11 (gram_core, e0j_read, pgrad_gram.stage_argv). CPU only; runs the SAME vector procedure the GPU job runs on a tiny
shared-trunk model, with analytic answers and DELIBERATE-REGRESSION arms (each must make its control FAIL).
    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_e0j.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0j_read as rd  # noqa: E402
import gram_core as gc  # noqa: E402
import pgrad_gram as pg  # noqa: E402

TERMS = gc.TERMS


# ---------------------------------------------------------------------------------------------- the analytic arms
def test_share_orthogonal_closed_form():
    n = np.array([2.0, 3.0, 5.0])
    gm = np.diag(n ** 2)
    for w in (0.5, 1.0, 7.0):
        s = gc.share_from_gram(gm, [w, 1.0, 1.0])
        den = w ** 2 * 4 + 9 + 25
        assert s == pytest.approx([w ** 2 * 4 / den, 9 / den, 25 / den], abs=1e-14)
        assert s.sum() == pytest.approx(1.0, abs=1e-14)


def test_share_matches_vector_definition_with_opposing_terms():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(3, 50))
    g[1] = -0.6 * g[0] + 0.2 * g[1]                                    # box3d partly OPPOSES map -> a negative share must appear somewhere
    gm = g @ g.T
    for w in ([1, 1, 1], [3.0, 0.5, 2.0], [0.1, 1, 1]):
        G = (np.array(w)[:, None] * g).sum(0)
        want = np.array(w) * (g @ G) / (G @ G)
        assert gc.share_from_gram(gm, w) == pytest.approx(want, abs=1e-12)
    assert min(gc.share_from_gram(gm, [1, 1, 1])) < 0.0


def test_gram_checks_pass_and_catch_a_broken_matrix():
    g = np.random.default_rng(1).normal(size=(3, 20))
    assert gc.check_gram(g @ g.T)["psd"]
    bad = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])           # |cos| = 2: not a Gram matrix
    c = gc.check_gram(bad)
    assert (not c["psd"]) and (not c["cs_ok"])


def test_selector_matches_closed_form_and_flags_infeasible():
    nm, nb, na = 2.0, 3.0, 4.0
    gm = np.diag([nm ** 2, nb ** 2, na ** 2])
    gms = {"layer2": gm, "trunk": gm, "lift": gm}
    for tgt in (0.05, 0.15):
        want = np.sqrt(tgt / (1 - tgt) * (nb ** 2 + na ** 2)) / nm
        r = gc.select_w_map(gms, tgt)
        assert r["found"] and r["w_star"] == pytest.approx(want, rel=1e-6)
        assert r["shares_at_w_star"]["layer2"]["map"] == pytest.approx(tgt, abs=1e-8)
        assert r["map_share_monotone_on_grid"]
        box = nb ** 2 / (want ** 2 * nm ** 2 + nb ** 2 + na ** 2)
        assert r["floors_all_ok"] == bool(box >= 0.25)                             # lift floor 0.25 is the binding one here
    # box floor 0.25 broken at the 0.15-target weight on a gram where box3d is small: infeasible, and SAID so
    small_box = np.diag([1.0, 0.2, 5.0])
    r = gc.select_w_map({"layer2": small_box, "trunk": small_box, "lift": small_box}, 0.15)
    assert r["found"] and not r["floors_all_ok"]
    # a target that can never be reached (map gradient identically zero)
    z = np.diag([0.0, 1.0, 1.0])
    assert not gc.select_w_map({"layer2": z, "trunk": z, "lift": z}, 0.05)["found"]


# ------------------------------------------------------------------------------ the tiny shared-trunk model, the job's procedure
def tiny(seed=0, n_in=6, n_h=8):
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    m = torch.nn.ModuleDict({"l2": torch.nn.Linear(n_in, n_h), "l3": torch.nn.Linear(n_h, n_h), "lift": torch.nn.Linear(n_h, n_h),
                             "hm": torch.nn.Linear(n_h, 3), "hb": torch.nn.Linear(n_h, 3), "ha": torch.nn.Linear(n_h, 3)})
    groups = {"stage_layer2": list(m["l2"].parameters()), "stage_layer3": list(m["l3"].parameters()),
              "lift": list(m["lift"].parameters()), "map_encoder": []}
    return m, groups


def fwd_terms(m, x, y):
    h = torch.tanh(m["l2"](x))
    h = torch.tanh(m["l3"](h))
    z = torch.tanh(m["lift"](h))
    return {"map": ((m["hm"](z) - y[0]) ** 2).mean(), "box3d": ((m["hb"](z) - y[1]) ** 2).mean() + 0.3 * (h ** 2).mean(),
            "agent": ((m["ha"](h) - y[2]) ** 2).mean()}


def run_job_procedure(m, groups, n_batches=3, w2=(4.0, 0.5, 2.0), tamper=None, extra_term=False):
    """The same per-batch procedure as pgrad_gram.main: unweighted per-term grads + ONE backward of the w2-weighted total."""
    params = [p for g in groups.values() for p in g]
    idx, o = {}, 0
    for g, ps in groups.items():
        idx[g] = list(range(o, o + len(ps)))
        o += len(ps)
    V = {g: [] for g in groups if groups[g]}
    for b in range(n_batches):
        x = torch.randn(5, 6)
        y = torch.randn(3, 5, 3)
        T = fwd_terms(m, x, y)

        def gv(t):
            gr = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
            return {g: torch.cat([(gr[i] if gr[i] is not None else torch.zeros_like(params[i])).reshape(-1) for i in idx[g]]) for g in V}
        per = {t: gv(T[t]) for t in TERMS}
        direct = gv(sum(w * T[t] for w, t in zip(w2, TERMS)))
        if tamper == "scale":
            direct = {g: v * (1.0 + 2e-3) for g, v in direct.items()}
        if tamper == "other_forward":                                  # the direct backward uses a DIFFERENT forward (e.g. other RNG)
            T2 = fwd_terms(m, x + 1e-2 * torch.randn_like(x), y)
            gr = torch.autograd.grad(sum(w * T2[t] for w, t in zip(w2, TERMS)), params, allow_unused=True)
            direct = {g: torch.cat([(gr[i] if gr[i] is not None else torch.zeros_like(params[i])).reshape(-1) for i in idx[g]]) for g in V}
        for g in V:
            V[g].append(torch.stack([per[t][g] for t in TERMS] + [direct[g]]))
    return V, per


def make_rec(V, n_batches, w2=(4.0, 0.5, 2.0), z=0.0, t=0.0, val=0.0):
    K = {g: gc.block_matrix(V[g]).tolist() for g in V}
    K["map_encoder"] = [[0.0] * (4 * n_batches) for _ in range(4 * n_batches)]
    return {"K": K, "n_batches": n_batches, "batch": 5, "w2": dict(zip(TERMS, w2)), "kinds": ["map", "box3d", "agent", "direct"],
            "trunk_groups": ["stage_layer2", "stage_layer3"], "ckpt_md5": "x", "window_draw_sha256": "y",
            "w_configured": {"map": 1.0, "box3d": 1.0, "agent": 1.0},
            "groups": {g: {"n_params": 1} for g in K}, "tier": "test", "measurement_choices": {},
            "controls_raw": {"Z_zero_weight_max_abs": z, "Z_n_batches": 1, "T_hidden_term_max_rel_err": t, "T_n_batches": 1,
                             "value_check_max_rel": val},
            "per_batch": [{"n_map_labelled": 5.0, "n_map_windows": 5.0, "n_agent_labelled": 5.0, "term_present": {}}] * n_batches}


def test_end_to_end_linearity_exact_and_pooled_gram_equals_summed_vector_gram(tmp_path):
    m, groups = tiny()
    V, _ = run_job_procedure(m, groups)
    nb = 3
    K = gc.block_matrix(V["stage_layer2"])
    gm, dots, g2g2 = gc.pooled(K, nb)
    M = torch.stack(V["stage_layer2"])                                  # [nb, 4, n]
    S = M.sum(0).numpy()                                                # the SUMMED gradient per kind
    assert gm == pytest.approx(S[:3] @ S[:3].T, rel=1e-10)
    assert dots == pytest.approx(S[:3] @ S[3], rel=1e-10)
    lr = gc.linearity_report(K, nb, (4.0, 0.5, 2.0))
    # float64 autograd is linear to round-off; the rel-grad-err is read through a difference of squares, so its floor is sqrt(eps) ~ 1e-8
    assert lr["max_rel_grad_err"] < 1e-6 and lr["max_share_diff"] < 1e-12


def _write(tmp_path, rec):
    p = tmp_path / "pgrad_gram.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return str(p)


def test_reader_passes_and_selects_the_brute_force_weight(tmp_path):
    m, groups = tiny(seed=3)
    V, _ = run_job_procedure(m, groups)
    nb = 3
    res = rd.build(_write(tmp_path, make_rec(V, nb)))
    assert res["VOID"] is False and res["controls"]["all_pass"], res["controls"]
    # brute force from the raw vectors: the pooled gradient of layer2 (summed over batches)
    S = torch.stack(V["stage_layer2"]).sum(0).numpy()
    g = S[:3]

    def smap(w):
        G = w * g[0] + g[1] + g[2]
        return w * (g[0] @ G) / (G @ G)
    for arm, tgt in rd.TARGETS.items():
        s = res["selection"][arm]
        if s["found"]:
            w = s["w_star"]
            assert smap(w) == pytest.approx(tgt, abs=1e-7)
            lo = [x for x in np.exp(np.linspace(np.log(1e-3), np.log(w * 0.999), 400)) if smap(x) >= tgt - 1e-9]
            assert not lo, "a smaller weight already met the target: not the SMALLEST"
        assert 0.0 <= s["spread"]["frac_found"] <= 1.0


@pytest.mark.parametrize("tamper", ["scale", "other_forward"])
def test_deliberate_regression_linearity_failure_voids_the_reading(tmp_path, tamper):
    m, groups = tiny(seed=5)
    V, _ = run_job_procedure(m, groups, tamper=tamper)
    res = rd.build(_write(tmp_path, make_rec(V, 3)))
    assert res["controls"]["L_pass"] is False
    assert res["VOID"] is True and res["selection"] is None
    assert "falls back to the E1 dose arms" in res["verdict"]


def test_deliberate_regression_other_controls_void(tmp_path):
    m, groups = tiny(seed=6)
    V, _ = run_job_procedure(m, groups)
    for kw in ({"z": 1e-9}, {"t": 5e-4}, {"val": 1e-3}):
        res = rd.build(_write(tmp_path, make_rec(V, 3, **kw)))
        assert res["VOID"] is True and res["selection"] is None, kw


def test_zero_weight_term_reads_exactly_zero_and_a_dead_term_voids():
    m, groups = tiny(seed=7)
    params = [p for g in groups.values() for p in g]
    T = fwd_terms(m, torch.randn(4, 6), torch.randn(3, 4, 3))
    gr = torch.autograd.grad(0.0 * T["box3d"], params, allow_unused=True)
    assert max(float(g.abs().max()) for g in gr if g is not None) == 0.0
    # a term with no gradient anywhere (a mechanism that never reached the loss) -> terms_live False -> VOID
    V, _ = run_job_procedure(m, groups)
    for g in V:
        for b in V[g]:
            b[0] = 0.0                                                    # the map row (and only it) is dead
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        res = rd.build(_write(Path(d), make_rec(V, 3)))
    assert res["controls"]["terms_live"] is False and res["VOID"] is True


# -------------------------------------------------------------------------------------------------- the argv builder
BASE = ["--arm", "flat", "--map-hires", "on", "--map-hires-near-lift-m", "20", "--r9-perc-rig", "--init-from", "/x.pt", "--seed", "0"]


def test_stage_argv_banded_and_baseline():
    b = pg.stage_argv(BASE, "banded")
    assert b[b.index("--map-hires-near-lift-m") + 1] == "10"
    assert b.count("--r9-map-banded") == 1 and b.index("--r9-map-banded") == b.index("--r9-perc-rig") + 1
    assert BASE[BASE.index("--map-hires-near-lift-m") + 1] == "20"                    # the input is not mutated
    assert pg.stage_argv(BASE, "baseline8") == BASE
    with pytest.raises(SystemExit):
        pg.stage_argv(b, "banded")                                                     # already banded: refuse, never double-apply
    with pytest.raises(SystemExit):
        pg.stage_argv(b, "baseline8")
