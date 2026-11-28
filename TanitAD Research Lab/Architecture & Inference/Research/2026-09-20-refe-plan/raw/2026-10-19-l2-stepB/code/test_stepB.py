#!/usr/bin/env python3
"""Tests of the L2 Step B build.  CPU only, NO real label read, no checkpoint needed (a tiny set-conditional fake scorer stands in; the REAL scorer is exercised by g_reach_stepb.py's own run).

    CUDA_VISIBLE_DEVICES=-1 pytest test_stepB.py -q

Every control here can fail: refusals are asserted to RAISE, mutations are asserted to turn a check RED, and the decision rule has a planted world per branch plus one regression arm per term.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse_stepB as AN  # noqa: E402
import fake_scorer  # noqa: E402
import g_reach_stepb as G  # noqa: E402
import l2_scorer as L2S  # noqa: E402
import label_stepB as LB  # noqa: E402
import stepB  # noqa: E402

A, T = stepB.A, stepB.T

try:
    import torch
    torch.set_num_threads(2)
except Exception:  # pragma: no cover
    pass

FAKE = fake_scorer.FakeScorerModule()


# ------------------------------------------------------------------------------------------------------------------------------ fixtures
def _plans(rng, n_tok, speed_lo=6.0, speed_hi=14.0):
    """[n_tok,64,20,3] forward-moving plans with a little curvature; the last heading is the A7-corrupted one (differs from the previous)"""
    out = np.zeros((n_tok, 64, 20, 3), np.float32)
    for t in range(n_tok):
        for k in range(64):
            v = rng.uniform(speed_lo, speed_hi)
            kappa = rng.normal(0, 0.01)
            s = v * 0.2 * np.arange(1, 21)
            yaw = kappa * s
            out[t, k, :, 0] = s * np.cos(yaw / 2)
            out[t, k, :, 1] = s * np.sin(yaw / 2)
            out[t, k, :, 2] = yaw
            out[t, k, -1, 2] += 0.3                                                     # corrupted last heading
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    rng = np.random.default_rng(7)
    n_tok = 12
    tr = _plans(rng, n_tok)
    pick = rng.integers(0, 64, n_tok).astype(np.int64)
    logs = np.array([f"log{(i % 4):02d}" for i in range(n_tok)])
    sub = np.clip(rng.uniform(0.3, 1.0, (n_tok, 64, 6)), 0, 1)
    b = {"token": np.array([f"tok{i:03d}" for i in range(n_tok)]), "log": logs, "traj_raw": tr, "logits": np.zeros((n_tok, 64, 6), np.float32), "pick": pick, "sub": sub,
         "pdms": np.clip(rng.uniform(0.2, 1.0, (n_tok, 64)), 0, 1), "valid": np.ones((n_tok, 64), bool)}
    bp = str(d / "bundle.npz")
    np.savez(bp, **b)
    ck = str(d / "model_final.pt")
    open(ck, "wb").write(b"not a real checkpoint, the fake scorer ignores it")
    cfg = stepB.c8_from_grid()
    cp = str(d / "stepB_config.json")
    with open(cp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg.as_json(), f, indent=1)
        f.write("\n")
    ns, _ = FAKE.load_scorer(ck)
    W, bb = ns.score_head.weight.detach().numpy().copy(), ns.score_head.bias.detach().numpy().copy()
    return {"carve_sha": stepB.tokens_sha256(b["token"]), "dir": d, "bundle": bp, "bundle_d": b, "ckpt": ck, "ckpt_sha": L2S.sha256_file(ck), "cfg": cfg, "cfg_path": cp, "cfg_sha": stepB.sha256_lf(cp), "W": W, "b": bb}


def _head(world, name, ep_w=None, ep_b=None, other_noise=False):
    W, b = world["W"].copy(), world["b"].copy()
    if ep_w is not None:
        W[L2S.EP] = ep_w
    if ep_b is not None:
        b[L2S.EP] = ep_b
    if other_noise:
        W[0, 0] += 1.0
    p = str(world["dir"] / f"{name}.npz")
    np.savez(p, weight=W.astype(np.float32), bias=b.astype(np.float32), kind=np.array("test"))
    return p, L2S.sha256_file(p)


def _base(world, head, head_sha):
    return argparse.Namespace(arm="L2", head=head, head_sha256=head_sha, bundle=world["bundle"], ctx="unused", ckpt=world["ckpt"], ckpt_sha256=world["ckpt_sha"], refe_dir="unused", config=world["cfg_path"],
                              config_sha256=world["cfg_sha"], threads=2)


# ------------------------------------------------------------------------------------------------------------------------------ the head loader: refusals
def test_head_loader_accepts_a_good_file_and_installs_only_the_ep_row(world):
    p, sha = _head(world, "good", ep_w=world["W"][L2S.EP] + 0.5, ep_b=world["b"][L2S.EP] - 0.25)
    head = L2S.read_head_file(p, sha)
    ns, _ = FAKE.load_scorer("x")
    rec = L2S.install_ep_head(ns, head)
    assert rec["other_rows_equal_p2"] and not rec["ep_row_equals_p2"] and rec["ep_row_l2_delta_vs_p2"] > 0
    assert np.array_equal(ns.score_head.weight.detach().numpy()[L2S.EP], head.W[L2S.EP])
    assert np.array_equal(ns.score_head.weight.detach().numpy()[[0, 1, 3, 4, 5]], world["W"][[0, 1, 3, 4, 5]])


def test_head_loader_identity_file_is_flagged_identical(world):
    p, sha = _head(world, "ident")
    ns, _ = FAKE.load_scorer("x")
    assert L2S.install_ep_head(ns, L2S.read_head_file(p, sha))["ep_row_equals_p2"] is True


@pytest.mark.parametrize("bad", ["wrong_sha", "no_sha", "short_sha", "missing_file"])
def test_head_loader_refuses_on_sha_problems(world, bad):
    p, sha = _head(world, "sha_" + bad, ep_w=world["W"][L2S.EP] + 1)
    with pytest.raises(L2S.HeadRefused):
        if bad == "wrong_sha":
            L2S.read_head_file(p, "0" * 64)
        elif bad == "no_sha":
            L2S.read_head_file(p, None)
        elif bad == "short_sha":
            L2S.read_head_file(p, sha[:20])
        else:
            L2S.read_head_file(str(world["dir"] / "nope.npz"), sha)


def test_head_loader_refuses_bad_shapes_dtypes_nonfinite_and_changed_other_rows(world):
    d = world["dir"]
    p = str(d / "shape.npz")
    np.savez(p, weight=np.zeros((6, 128), np.float32), bias=np.zeros(6, np.float32))
    with pytest.raises(L2S.HeadRefused):
        L2S.read_head_file(p, L2S.sha256_file(p))
    p = str(d / "f64.npz")
    np.savez(p, weight=world["W"].astype(np.float64), bias=world["b"].astype(np.float64))
    with pytest.raises(L2S.HeadRefused):
        L2S.read_head_file(p, L2S.sha256_file(p))
    W = world["W"].copy()
    W[2, 0] = np.nan
    p = str(d / "nan.npz")
    np.savez(p, weight=W, bias=world["b"])
    with pytest.raises(L2S.HeadRefused):
        L2S.read_head_file(p, L2S.sha256_file(p))
    p = str(d / "nokeys.npz")
    np.savez(p, w=world["W"])
    with pytest.raises(L2S.HeadRefused):
        L2S.read_head_file(p, L2S.sha256_file(p))
    p, sha = _head(world, "other_rows", ep_w=world["W"][L2S.EP] + 1, other_noise=True)          # a head file that changed ANOTHER head
    ns, _ = FAKE.load_scorer("x")
    with pytest.raises(L2S.HeadRefused):
        L2S.install_ep_head(ns, L2S.read_head_file(p, sha))


def test_build_score_fn_refuses_if_the_installer_never_ran(world):
    class Plain:                                                                                    # a scorer module whose make_score_fn never calls load_scorer
        def load_scorer(self, *a, **k):
            return FAKE.load_scorer(*a, **k)

        def make_score_fn(self, *a, **k):
            fn = FAKE.make_score_fn(*a, **k)
            return fn
    p, sha = _head(world, "never", ep_w=world["W"][L2S.EP] + 1)
    plain = Plain()
    plain.make_score_fn = lambda *a, **k: FAKE.make_score_fn(*a, **k)                              # bypasses plain.load_scorer: the wrapper is never reached
    with pytest.raises(L2S.HeadRefused):
        L2S.build_score_fn(plain, world["ckpt"], "x", np.array(["a"]), L2S.read_head_file(p, sha))


# ------------------------------------------------------------------------------------------------------------------------------ C8: the rule, the file, the strict loader
def test_c8_rule_is_the_hull_of_the_measured_grid_and_nothing_else():
    c = stepB.c8_from_grid()
    assert c.u_lo == pytest.approx(np.log(0.85)) and c.u_hi == pytest.approx(np.log(1.15))
    assert (c.d_lo, c.d_hi) == (-0.3, 0.3)
    assert c.sigma0_u == pytest.approx(max(abs(np.log(0.85)), abs(np.log(1.15))) / 2) and c.sigma0_d == pytest.approx(0.15)
    assert (c.k_iters, c.m_pop, c.n_elite, c.widen, c.profile) == (5, 64, 8, 1.0, "ramp") and c.seeds == tuple(T.SEARCH_SEEDS)
    # the grid's extremes sit at exactly 2 sigma0, and the clips are INSIDE the placeholders (the search no longer leaves the admitted region)
    assert abs(c.u_lo) / c.sigma0_u == pytest.approx(2.0) and c.d_hi / c.sigma0_d == pytest.approx(2.0)
    assert c.u_lo > T.U_CLIP[0] and c.u_hi < T.U_CLIP[1] and c.d_lo > T.D_CLIP[0] and c.d_hi < T.D_CLIP[1]
    # regression arm: it is a function of the GRID (another grid moves it), so it cannot be a constant that outcomes could have tuned
    c2 = stepB.c8_from_grid(speed_scales=(0.8, 1.2), lat_offsets=(-0.5, 0.5))
    assert c2.d_hi == 0.5 and c2.sigma0_d == pytest.approx(0.25) and c2.u_hi == pytest.approx(np.log(1.2))


def test_config_loader_is_strict(world, tmp_path):
    good = json.load(open(world["cfg_path"]))
    for mutate in (lambda d: d.pop("sigma0_u"), lambda d: d.update(extra_lever=1)):
        d = dict(good)
        mutate(d)
        p = str(tmp_path / "c.json")
        json.dump(d, open(p, "w"))
        with pytest.raises(ValueError):
            stepB.StepBConfig.from_json(p)
    assert stepB.StepBConfig.from_json(world["cfg_path"]) == world["cfg"]


def test_validate_config_admits_the_o4_arm_and_refuses_hand_edits():
    stepB.validate_config(stepB.c8_from_grid())
    stepB.validate_config(stepB.c8_from_grid(m_pop=256, n_elite=32, widen=2.0))                          # O4: "raise M to 256 and widen the covariance once"
    for bad in (stepB.StepBConfig(**{**stepB.c8_from_grid().as_json(), "seeds": stepB.c8_from_grid().seeds, "n_elite": 7}),          # elites != M/8
                stepB.StepBConfig(**{**stepB.c8_from_grid().as_json(), "seeds": stepB.c8_from_grid().seeds, "k_iters": 6}),
                stepB.StepBConfig(**{**stepB.c8_from_grid().as_json(), "seeds": stepB.c8_from_grid().seeds, "d_hi": 0.6}),             # the placeholder clip
                stepB.StepBConfig(**{**stepB.c8_from_grid().as_json(), "seeds": stepB.c8_from_grid().seeds, "sigma0_u": 0.10})):       # the placeholder width
        with pytest.raises(SystemExit):
            stepB.validate_config(bad)


def test_the_committed_config_file_equals_the_rule_output(world):
    p = os.path.join(HERE, "..", "raw", "stepB_config.json")
    if not os.path.exists(p):
        p = os.environ.get("STEPB_CONFIG", p)
    assert os.path.exists(p), "the committed stepB_config.json is missing (raw/stepB_config.json or $STEPB_CONFIG): no skip"
    assert stepB.StepBConfig.from_json(p) == stepB.c8_from_grid()


def test_l1_code_pins_hold_and_a_wrong_pin_raises():
    assert set(stepB.verify_l1_code()) == set(stepB.L1_PINS)
    bad = dict(stepB.L1_PINS)
    bad["tts_cem.py"] = "0" * 64
    with pytest.raises(RuntimeError):
        stepB.verify_l1_code(pins=bad)


# ------------------------------------------------------------------------------------------------------------------------------ the search path on the fake scorer (what the gate also proves on the real one)
def test_search_token_identity_at_zero_iterations_and_moves_with_five(world):
    cfg0 = stepB.StepBConfig(**{**world["cfg"].as_json(), "k_iters": 0, "seeds": world["cfg"].seeds})
    fn = FAKE.make_score_fn(world["ckpt"], "x", np.array(["a"]))
    fn.bind(0)
    b = world["bundle_d"]
    r0 = stepB.search_token(fn, b["traj_raw"][0], int(b["pick"][0]), 1, cfg0)
    assert not r0["moved"] and np.array_equal(r0["exec"], A.repair_last_heading_np(b["traj_raw"][0][int(b["pick"][0])]))


def test_clips_are_restored_after_a_search(world):
    old = (T.U_CLIP, T.D_CLIP)
    fn = FAKE.make_score_fn(world["ckpt"], "x", np.array(["a"]))
    fn.bind(0)
    b = world["bundle_d"]
    stepB.search_token(fn, b["traj_raw"][0], int(b["pick"][0]), 1, world["cfg"])
    assert (T.U_CLIP, T.D_CLIP) == old


# ------------------------------------------------------------------------------------------------------------------------------ G-REACH-STEPB on the fake scorer: PASS, every mutation RED, and it can FAIL
def test_gate_passes_on_the_fake_scorer_with_every_mutation_red(world, tmp_path):
    p, sha = _head(world, "gate_head", ep_w=world["W"][L2S.EP] * 0.5 + 0.3, ep_b=world["b"][L2S.EP] + 0.2)
    st = G.run_gate(_base(world, p, sha), str(tmp_path / "gate"), scorer_mod=FAKE, verbose=False, carve_sha256=world["carve_sha"])
    assert st["verdict"] == "PASS", json.dumps({"checks": [c for c in st["checks"] if not c["ok"]], "mut": [m for m in st["mutations"] if not m["red"]]}, indent=1)[:2000]
    names = {c["name"] for c in st["checks"]}
    assert {"a_cid_identity_head_equals_p2", "b_inverted_head_negates_ep_logits_the_search_sees", "d_logit_diff_differs", "d_logit_diff_identical", "e_c8_sigma0_and_clips_reach_the_search",
            "f_end_to_end_launch_shard_shows_the_head"} <= names
    assert len(st["mutations"]) >= 8 and all(m["red"] for m in st["mutations"])
    assert [m for m in st["mutations"] if m["mutation"].startswith("M2 ")][0]["check"] == "b_inverted_head_negates_ep_logits_the_search_sees"        # install+readback gone: ONLY (b) sees it
    cid = [c for c in st["checks"] if c["name"] == "a_cid_identity_head_equals_p2"][0]
    assert cid["search_moved_the_pick_on_this_token (non-vacuous)"] is True


def test_gate_fails_for_a_head_identical_to_p2(world, tmp_path):
    """(d)'s 'differs' leg is the in-launch assertion that the head changed the EP logits: an identity head must make the gate FAIL, not pass"""
    p, sha = _head(world, "gate_ident")
    st = G.run_gate(_base(world, p, sha), str(tmp_path / "gate2"), scorer_mod=FAKE, verbose=False, carve_sha256=world["carve_sha"])
    assert st["verdict"] == "FAIL"
    assert not [c for c in st["checks"] if c["name"] == "d_logit_diff_differs"][0]["ok"]


def test_mutation_harness_refuses_a_vacuous_or_miscounted_mutation():
    with pytest.raises(RuntimeError):
        G.mutate_module(os.path.join(HERE, "l2_scorer.py"), {"NO_SUCH_TAG": ("pass", 1)}, "m_vac")
    with pytest.raises(RuntimeError):
        G.mutate_module(os.path.join(HERE, "l2_scorer.py"), {"INSTALL": ("pass", 3)}, "m_cnt")


def test_preflight_reach_reads_both_directions(world):
    b = world["bundle_d"]
    p, sha = _head(world, "pre", ep_w=world["W"][L2S.EP] + 1)
    a = G.search_args(_base(world, p, sha), "L2", p, sha)
    a_p2 = G.search_args(_base(world, p, sha), "P2HEAD", None, None)
    fa, fp = stepB.build_scorer(a, scorer_mod=FAKE), stepB.build_scorer(a_p2, scorer_mod=FAKE)
    assert stepB.preflight_reach(fa, fp, b, 0, "differs")["ok"] and not stepB.preflight_reach(fa, fp, b, 0, "identical")["ok"]
    assert stepB.preflight_reach(fp, fp, b, 0, "identical")["ok"] and not stepB.preflight_reach(fp, fp, b, 0, "differs")["ok"]


# ------------------------------------------------------------------------------------------------------------------------------ the stamp and the launch refusals
def _stamp(world, head_sha, **over):
    st = {"schema": "tanitad.g_reach_stepb/1", "verdict": "PASS", "head_sha256": head_sha, "ckpt_sha256": world["ckpt_sha"], "config_sha256": world["cfg_sha"], "bundle_sha256": L2S.sha256_file(world["bundle"]),
          "code_sha256": stepB.code_shas(), "l1_pins": dict(stepB.L1_PINS)}
    st.update(over)
    p = str(world["dir"] / "stamp.json")
    json.dump(st, open(p, "w"))
    return p


def _check(world, p, head_sha, **kw):
    return stepB.check_stamp(p, arm="L2", head_sha256=head_sha, ckpt_sha256=world["ckpt_sha"], config_sha256=world["cfg_sha"], bundle_sha256=L2S.sha256_file(world["bundle"]), **kw)


def test_stamp_accepts_its_own_binding_and_refuses_every_other(world):
    hs = "a" * 64
    assert _check(world, _stamp(world, hs), hs)["verdict"] == "PASS"
    with pytest.raises(RuntimeError):
        _check(world, str(world["dir"] / "no_such_stamp.json"), hs)
    for over in ({"verdict": "FAIL"}, {"head_sha256": "b" * 64}, {"ckpt_sha256": "c" * 64}, {"config_sha256": "d" * 64}, {"bundle_sha256": "e" * 64}, {"schema": "other"},
                 {"code_sha256": {**stepB.code_shas(), "stepB.py": "f" * 64}}, {"l1_pins": {**stepB.L1_PINS, "tts_cem.py": "1" * 64}}):
        with pytest.raises(RuntimeError):
            _check(world, _stamp(world, hs, **over), hs)


def test_launch_cli_argument_rules():
    p = stepB.build_search_parser()
    base = ["--bundle", "b", "--ctx", "c", "--ckpt", "k", "--ckpt-sha256", "x", "--refe-dir", "r", "--config", "c", "--config-sha256", "y", "--gate", "g", "--out-dir", "o"]
    with pytest.raises(SystemExit):
        stepB.validate_search_args(p.parse_args(["--arm", "L2"] + base))                                      # an L2 arm with no head
    with pytest.raises(SystemExit):
        stepB.validate_search_args(p.parse_args(["--arm", "L2", "--head", "h"] + base))                       # a head without its sha256
    with pytest.raises(SystemExit):
        stepB.validate_search_args(p.parse_args(["--arm", "P2HEAD", "--head", "h", "--head-sha256", "s"] + base))   # a flag that would be silently ignored
    with pytest.raises(SystemExit):
        p.parse_args(base)                                                                                    # no --arm
    stepB.validate_search_args(p.parse_args(["--arm", "P2HEAD"] + base))


def test_launch_refuses_without_a_stamp_before_building_any_scorer(world, tmp_path):
    p, sha = _head(world, "cli_head", ep_w=world["W"][L2S.EP] + 1)
    a = G.search_args(_base(world, p, sha), "L2", p, sha)
    a.gate, a.out_dir = str(tmp_path / "missing_stamp.json"), str(tmp_path / "o")
    with pytest.raises(RuntimeError, match="stamp"):
        stepB.run_search_cli(a, enforce_stamp=True, carve_sha256=world["carve_sha"])


def test_launch_refuses_a_hand_edited_config(world, tmp_path):
    d = world["cfg"].as_json()
    d["u_hi"] = 0.5                                                                                         # widen the clip by hand
    cp = str(tmp_path / "edited.json")
    with open(cp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f)
    p, sha = _head(world, "cli_head2", ep_w=world["W"][L2S.EP] + 1)
    a = G.search_args(_base(world, p, sha), "L2", p, sha)
    a.config, a.config_sha256, a.out_dir, a.gate = cp, stepB.sha256_lf(cp), str(tmp_path / "o"), "unused"
    with pytest.raises(SystemExit, match="C8"):
        stepB.run_search_cli(a, enforce_stamp=False, scorer_mod=FAKE)


# ------------------------------------------------------------------------------------------------------------------------------ shards
def _shard(world, tmp, arm, lo, hi, **meta):
    out = stepB.run_search(FAKE.make_score_fn(world["ckpt"], "x", np.array(["a"])), world["bundle_d"], range(lo, hi), world["cfg"])
    m = {"arm": arm, "head_sha256": "h", "ckpt_sha256": "k", "config_sha256": "c", "bundle_sha256": "b", **meta}
    p = os.path.join(tmp, f"shard_{arm}_{lo:04d}.npz")
    np.savez(p, meta=np.array(json.dumps(m)), **out)
    return p


def test_merge_joins_shards_and_refuses_disagreeing_or_duplicated_ones(world, tmp_path):
    d = str(tmp_path)
    _shard(world, d, "L2", 0, 2)
    _shard(world, d, "L2", 2, 4)
    outp = str(tmp_path / "m.npz")
    assert stepB.cmd_merge(["--out-dir", d, "--arm", "L2", "--out", outp, "--n", "4"]) == 0
    z = np.load(outp)
    assert len(z["token"]) == 4 and z["exec"].shape == (4, 3, 20, 3)
    with pytest.raises(SystemExit):
        stepB.cmd_merge(["--out-dir", d, "--arm", "L2", "--out", outp, "--n", "5"])
    d2 = str(tmp_path / "bad")
    os.makedirs(d2)
    _shard(world, d2, "L2", 0, 2)
    _shard(world, d2, "L2", 2, 4, head_sha256="OTHER")
    with pytest.raises(SystemExit):
        stepB.cmd_merge(["--out-dir", d2, "--arm", "L2", "--out", outp])
    d3 = str(tmp_path / "dup")
    os.makedirs(d3)
    _shard(world, d3, "L2", 0, 2)
    p2 = _shard(world, d3, "L2", 1, 3)
    os.replace(p2, os.path.join(d3, "shard_L2_0002.npz"))
    with pytest.raises(SystemExit):
        stepB.cmd_merge(["--out-dir", d3, "--arm", "L2", "--out", outp])


# ------------------------------------------------------------------------------------------------------------------------------ labelling: the planted identity
def test_planted_identity_plan_is_the_repaired_pick_and_the_control_is_shifted(world):
    b = world["bundle_d"]
    raw = b["traj_raw"][0][int(b["pick"][0])]
    ex, sh = LB.planted_identity_plans(raw)
    assert np.array_equal(ex, A.repair_last_heading_np(raw).astype(np.float32))                 # composite(0,0) IS the pick's executed plan, bit for bit
    assert np.allclose(sh[:, 0] - ex[:, 0], 0.5) and np.array_equal(sh[:, 1:], ex[:, 1:])
    # regression arm: a composite that is NOT the identity (a 0.001 speed scale) is not the pick's plan
    assert not np.array_equal(T.composite(raw, 0.001, 0.0)[0], ex)


def test_identity_record_reads_a_difference_and_an_exact_match():
    s = np.array([1, 1, .5, .6, 1, 1.0])
    assert LB.identity_record(s, 0.5, s, 0.5)["passed"] and LB.identity_record(s, 0.5, s, 0.5)["max_abs_diff"] == 0.0
    assert not LB.identity_record(s + 1e-3, 0.5, s, 0.5)["passed"] and not LB.identity_record(s, 0.5 + 1e-3, s, 0.5)["passed"]
    assert not LB.identity_record(np.full(6, np.nan), 0.5, s, 0.5)["passed"]


# ------------------------------------------------------------------------------------------------------------------------------ the decision
def _dp(point, lo):
    return {"point": point, "lo": lo, "hi": point + 1}


GOOD_SAFE = {"nc": 0.0, "dac": 0.0, "ttc": 0.0, "c": 0.0}
NOT_INERT = {"inert": False}


def test_decision_branches():
    assert AN.classify_stepB(_dp(0.50, 0.1), 0.2, GOOD_SAFE, NOT_INERT)["outcome"] == "O1_PASS"
    assert AN.classify_stepB(_dp(0.50, 0.1), 0.2, {**GOOD_SAFE, "dac": -0.30}, NOT_INERT)["outcome"] == "TRADE"
    assert AN.classify_stepB(_dp(0.50, 0.1), 0.2, {**GOOD_SAFE, "c": -0.25}, NOT_INERT)["outcome"] == "O1_PASS"       # the bound is inclusive
    assert AN.classify_stepB(_dp(0.29, 0.1), 0.0, GOOD_SAFE, NOT_INERT)["outcome"] == "NOT_CLEARED"                    # point under the bar
    assert AN.classify_stepB(_dp(0.50, -0.01), 0.2, GOOD_SAFE, NOT_INERT)["outcome"] == "NOT_CLEARED"                  # interval includes 0
    assert AN.classify_stepB(_dp(0.50, 0.1), 0.6, GOOD_SAFE, NOT_INERT)["outcome"] == "NOT_CLEARED"                    # an effect below the seed spread is not an effect
    assert AN.classify_stepB(_dp(0.50, 0.1), 0.2, GOOD_SAFE, {"inert": True})["outcome"] == "O4_INERT"
    assert AN.classify_stepB(_dp(-2.0, -3.0), 0.2, {**GOOD_SAFE, "nc": -5}, NOT_INERT)["outcome"] == "NOT_CLEARED"      # a failing dPDMS is never a TRADE


def _mut(drop):
    """the classifier with ONE term removed: each must disagree with the real one on its own planted world"""
    def f(dp, rng, safe, inert):
        if inert["inert"]:
            return {"outcome": "O4_INERT"}
        ok = [dp["point"] >= AN.BAR_DPDMS, dp["lo"] > 0.0 if drop != "ci" else True, dp["point"] > rng if drop != "spread" else True]
        s = all(safe[k] >= AN.BAR_SAFETY for k in AN.SAFETY_TERMS) if drop != "safety" else True
        return {"outcome": "O1_PASS" if all(ok) and s else ("TRADE" if all(ok) else "NOT_CLEARED")}
    return f


@pytest.mark.parametrize("drop,world_args", [("ci", (_dp(0.5, -0.1), 0.2, GOOD_SAFE, NOT_INERT)), ("spread", (_dp(0.5, 0.1), 0.6, GOOD_SAFE, NOT_INERT)),
                                             ("safety", (_dp(0.5, 0.1), 0.2, {**GOOD_SAFE, "ttc": -1.0}, NOT_INERT))])
def test_each_decision_term_has_a_world_where_dropping_it_changes_the_outcome(drop, world_args):
    assert _mut(drop)(*world_args)["outcome"] != AN.classify_stepB(*world_args)["outcome"]


def _planted_analysis(delta_points, safety_delta=None, moved=True, seed_jitter=0.0, n=40, S=3):
    """a synthetic Step B world with a KNOWN true effect: every search output's PDMS is the pick's + delta_points/100 (+ per-seed jitter)"""
    rng = np.random.default_rng(3)
    logs = np.array([f"log{(i % 8):02d}" for i in range(n)])
    pick = rng.integers(0, 64, n)
    sub = np.clip(rng.uniform(0.3, 0.9, (n, 64, 6)), 0, 0.9)
    pdms = np.clip(rng.uniform(0.2, 0.8, (n, 64)), 0.1, 0.8)
    b = {"token": np.array([f"t{i}" for i in range(n)]), "log": logs, "pick": pick.astype(np.int64), "sub": sub, "pdms": pdms}
    pe = np.zeros((n, 20, 3), np.float32)
    pe[:, :, 0] = np.linspace(1, 20, 20)
    ex = np.repeat(pe[:, None], S, 1).copy()
    if moved:
        ex[:, :, :, 0] += 1.0                                                     # 1.0 m ADE > 0.5 m
    th = np.zeros((n, S, 2))
    th[..., 0] = 0.05 if moved else 0.0
    so = {"token": b["token"], "bundle_index": np.arange(n), "exec": ex, "pick_exec": pe, "theta": th, "moved": np.full((n, S), moved), "arm_argmax": pick.copy(),
          "meta": np.array(json.dumps({"arm": "L2"}))}
    sd = np.zeros(S)
    sd[:] = np.linspace(-seed_jitter, seed_jitter, S)
    lab_p = pdms[np.arange(n), pick][:, None] + (delta_points + sd)[None, :] / 100.0
    lab_s = sub[np.arange(n), pick][:, None, :] + np.zeros((n, S, 6))
    for k, v in (safety_delta or {}).items():
        lab_s[:, :, A.SUBS.index(k)] += v / 100.0
    lab = {"token": b["token"], "pdms": lab_p, "sub": lab_s}
    return b, so, lab


def test_analysis_units_are_points_and_a_clear_world_reads_O1():
    b, so, lab = _planted_analysis(0.50)
    r = AN.analyse(b, so, lab, B=400)
    assert r["endpoint_dPDMS_points"]["point"] == pytest.approx(0.50, abs=1e-9)                # +0.005 on [0,1] is +0.50 points, not 0.005
    assert r["decision"]["outcome"] == "O1_PASS" and not r["inert"]["inert"]
    assert r["families"]["STRATEGIC"].startswith("PAUSED")


def test_analysis_trade_inert_and_not_cleared_worlds():
    assert AN.analyse(*_planted_analysis(0.50, {"dac": -0.40}), B=400)["decision"]["outcome"] == "TRADE"
    assert AN.analyse(*_planted_analysis(0.50, moved=False), B=400)["decision"]["outcome"] == "O4_INERT"
    assert AN.analyse(*_planted_analysis(0.10), B=400)["decision"]["outcome"] == "NOT_CLEARED"
    assert AN.analyse(*_planted_analysis(0.50, seed_jitter=0.6), B=400)["decision"]["outcome"] == "NOT_CLEARED"          # seed range 1.2 > the 0.50 effect
    r = AN.analyse(*_planted_analysis(0.50, seed_jitter=0.6), B=400)
    assert r["per_seed_points"]["range"] == pytest.approx(1.2, abs=1e-9)


def test_analysis_refuses_partial_or_misaligned_labels():
    b, so, lab = _planted_analysis(0.5)
    lab2 = {**lab, "pdms": lab["pdms"].copy()}
    lab2["pdms"][0, 0] = np.nan
    with pytest.raises(ValueError):
        AN.analyse(b, so, lab2, B=100)
    with pytest.raises(ValueError):
        AN.analyse(b, so, {**lab, "token": lab["token"][::-1]}, B=100)


def test_analysis_uses_the_banked_pick_label_not_the_arm_argmax():
    """if the arm's in-fan argmax differs from P2's pick, the endpoint is STILL against P2's pick (the descriptive decomposition separates the two)"""
    b, so, lab = _planted_analysis(0.5)
    so2 = {**so, "arm_argmax": (so["arm_argmax"] + 1) % 64}
    r1, r2 = AN.analyse(b, so, lab, B=200), AN.analyse(b, so2, lab, B=200)
    assert r1["endpoint_dPDMS_points"]["point"] == r2["endpoint_dPDMS_points"]["point"]
    assert r2["in_fan_decomposition_descriptive"]["arm_argmax_differs_from_p2_pick_share"] == 1.0


def test_combine_head_seeds_is_the_worst_outcome():
    mk = lambda o: {"decision": {"outcome": o}}  # noqa: E731
    assert AN.combine_head_seeds({"s0": mk("O1_PASS"), "s1": mk("O1_PASS")})["combined"] == "O1_PASS"
    assert AN.combine_head_seeds({"s0": mk("O1_PASS"), "s1": mk("TRADE")})["combined"] == "TRADE"
    assert AN.combine_head_seeds({"s0": mk("O1_PASS"), "s1": mk("NOT_CLEARED")})["combined"] == "NOT_CLEARED"
    assert AN.combine_head_seeds({"s0": mk("TRADE"), "s1": mk("O4_INERT")})["combined"] == "O4_INERT"


def test_analyse_cli_end_to_end_on_planted_files(tmp_path):
    """the CLI wiring: two head seeds + the P2HEAD control arm -> one RESULT json whose combined verdict ignores the control"""
    b, so, lab = _planted_analysis(0.50)
    bp = str(tmp_path / "b.npz")
    np.savez(bp, **b)
    args = ["--bundle", bp, "--out", str(tmp_path / "r.json")]
    for name, delta in (("L2s0", 0.50), ("L2s1", 0.50), ("P2HEAD", -3.0)):
        _, so_i, lab_i = _planted_analysis(delta)
        sp, lp = str(tmp_path / f"s_{name}.npz"), str(tmp_path / f"l_{name}.npz")
        np.savez(sp, **so_i)
        np.savez(lp, **lab_i)
        args += ["--search", f"{name}={sp}", "--labels", f"{name}={lp}"]
    assert AN.main(args) == 0
    r = json.load(open(tmp_path / "r.json"))
    assert r["combined"]["combined"] == "O1_PASS" and set(r["combined"]["per_head_seed"]) == {"L2s0", "L2s1"}
    assert r["arms"]["P2HEAD"]["decision"]["outcome"] == "NOT_CLEARED"                    # the descriptive control is read, never gating
    att = r["attribution_vs_P2HEAD_points (descriptive, not gating)"]
    assert att["L2s0"]["point"] == pytest.approx(0.50 - (-3.0), abs=1e-9)                  # paired difference of the two arms, in points
    with pytest.raises(SystemExit):
        AN.main(["--bundle", bp, "--out", str(tmp_path / "x.json"), "--search", f"a={sp}", "--labels", f"b={lp}"])


# ------------------------------------------------------------------------------------------------------------------------------ Revision 1: the DATA CLAUSE (the bundle's token list must equal the sealed carve)
CARVE_FILE = "C:/Users/Admin/par18/l2fresh/raw/fresh_sample_token_disjoint.json"
needs_carve = pytest.mark.skipif(not os.path.exists(CARVE_FILE), reason="the sealed carve's token list (l2fresh/raw) is not on this box")


@needs_carve
def test_data_clause_green_control_the_real_carve_token_list_hashes_to_the_sealed_pin():
    """the DEFINITION is right: sha256 of the real carve's tokens (a token list, no label) is the sealed 0629811f..., sorted or in file order; a bundle holding exactly those tokens passes the default clause"""
    toks = json.load(open(CARVE_FILE, encoding="utf-8"))["tokens"]
    assert len(toks) == 400 and toks == sorted(toks)
    assert stepB.tokens_sha256(toks) == stepB.SEALED_CARVE_SHA256 and stepB.SEALED_CARVE_SHA256.startswith("0629811f")
    rec = stepB.check_data_clause({"token": np.array(toks)})
    assert rec["sealed_default"] is True and rec["n_tokens"] == 400


@needs_carve
def test_data_clause_red_arms_one_swapped_one_dropped_one_reordered_and_a_planted_bundle_are_refused():
    toks = json.load(open(CARVE_FILE, encoding="utf-8"))["tokens"]
    swapped = list(toks)
    swapped[17] = "0" * 16                                                                                         # ONE token replaced
    reordered = list(toks)
    reordered[3], reordered[4] = reordered[4], reordered[3]                                                        # the same SET, a different order (row alignment with ctx.npy would break)
    for bad in (swapped, toks[:-1], reordered, toks + [toks[0]], []):
        with pytest.raises(SystemExit, match="DATA CLAUSE REFUSED"):
            stepB.check_data_clause({"token": np.array(bad)})


def test_data_clause_default_refuses_a_planted_bundle_and_an_explicit_planted_sha_admits_it(world):
    with pytest.raises(SystemExit, match="DATA CLAUSE REFUSED"):
        stepB.check_data_clause(world["bundle_d"])
    assert stepB.check_data_clause(world["bundle_d"], world["carve_sha"])["sealed_default"] is False
    with pytest.raises(SystemExit, match="DATA CLAUSE REFUSED"):
        stepB.check_data_clause(world["bundle_d"], "0" * 64)


def test_launch_and_gate_refuse_a_non_carve_bundle_before_any_scorer_is_built(world, tmp_path):
    p, sha = _head(world, "dc_head", ep_w=world["W"][L2S.EP] + 1)
    a = G.search_args(_base(world, p, sha), "L2", p, sha)
    a.gate, a.out_dir = "unused", str(tmp_path / "o")
    with pytest.raises(SystemExit, match="DATA CLAUSE REFUSED"):
        stepB.run_search_cli(a, enforce_stamp=False, scorer_mod=FAKE)                                              # no carve_sha256 argument = the sealed carve: the planted world is refused
    assert not os.path.exists(a.out_dir)
    with pytest.raises(SystemExit, match="DATA CLAUSE REFUSED"):
        G.run_gate(_base(world, p, sha), str(tmp_path / "g_dc"), scorer_mod=FAKE, verbose=False)                   # and no gate stamp is minted for it
    assert not list((tmp_path / "g_dc").glob("G-REACH-STEPB_*.json")) if (tmp_path / "g_dc").exists() else True


def test_data_clause_mutation_arm_deleting_the_clause_line_lets_the_planted_bundle_through(world, tmp_path):
    """MUTATE, do not inspect: with the @DATACLAUSE line deleted the SAME launch call is no longer refused for its data (it reaches the stamp check instead), so the refusal above IS that line"""
    p, sha = _head(world, "dc_head2", ep_w=world["W"][L2S.EP] + 1)
    a = G.search_args(_base(world, p, sha), "L2", p, sha)
    a.gate, a.out_dir = str(tmp_path / "no_stamp.json"), str(tmp_path / "o2")
    with pytest.raises(SystemExit, match="DATA CLAUSE REFUSED"):
        stepB.run_search_cli(a, enforce_stamp=True)                                                                # the real module refuses on the data
    mut = G.mutate_module(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stepB.py"), {"DATACLAUSE": ("clause = {}", 1)}, "stepB_mut_dataclause")
    with pytest.raises(RuntimeError, match="stamp"):
        mut.run_search_cli(a, enforce_stamp=True)                                                                  # the mutant gets past the data and fails LATER, on the missing stamp


def test_merge_refuses_shards_that_disagree_on_the_data_clause(world, tmp_path):
    d = tmp_path / "dcm"
    d.mkdir()
    for lo, clause in ((0, {"carve_sha256": "a" * 64}), (2, {"carve_sha256": "b" * 64})):
        _shard(world, d, "L2", lo, lo + 2, head_sha256="h", ckpt_sha256="c", config_sha256="f", bundle_sha256="b", data_clause=clause)
    with pytest.raises(SystemExit, match="disagree"):
        stepB.cmd_merge(["--out-dir", str(d), "--arm", "L2", "--out", str(tmp_path / "m.npz")])


# ------------------------------------------------------------------------------------------------------------------------------ Revision 1: the verdict names a next arm for every outcome
def test_every_outcome_has_a_named_next_arm_and_o4_ends_in_a_named_lever():
    assert set(AN.NEXT_ARM) == set(AN.OUTCOMES)
    o4 = AN.NEXT_ARM["O4_INERT"]
    assert "M = 256" in o4 and "widen 2.0" in o4 and "CLOSED" in o4 and "L2-BRANCH" in o4 and "P6 trunk" in o4 and "UNBLOCKED BY" in o4         # the route ends in a NAMED lever + what unblocks it
    assert "UNBLOCKED BY" in AN.NEXT_ARM["NOT_CLEARED"] and "L2-BRANCH" in AN.NEXT_ARM["NOT_CLEARED"]
    assert "BOTH banks" in AN.NEXT_ARM["O1_PASS"] and "navtest look" in AN.NEXT_ARM["O1_PASS"] and "TRADE-restricted" in AN.NEXT_ARM["TRADE"]


def test_final_verdict_is_the_worst_head_seed_carries_stamps_and_pauses_strategic():
    res = {k: AN.analyse(*_planted_analysis(d), B=200) for k, d in (("L2s0", 0.50), ("L2s1", 0.10))}
    comb = AN.combine_head_seeds(res)
    v = AN.final_verdict(res, comb, None)
    assert v["combined"] == "NOT_CLEARED" and set(v["per_head"]) == {"L2s0", "L2s1"} and v["per_head"]["L2s0"]["outcome"] == "O1_PASS"
    assert "LOG-SHARED" in v["stamps"] and "SEEN-SCENE" in v["stamps"] and "SELECTION" in v["stamps"] and v["STRATEGIC"].startswith("PAUSED")
    assert "L2-BRANCH" in v["next_arm"] and "licenses nothing alone" in v["licensing"]
    with pytest.raises(ValueError):
        AN.final_verdict(res, None, None)


def test_analyse_cli_writes_the_verdict_block(tmp_path):
    b, _, _ = _planted_analysis(0.50)
    bp = str(tmp_path / "b2.npz")
    np.savez(bp, **b)
    args = ["--bundle", bp, "--out", str(tmp_path / "r2.json")]
    for name, delta in (("L2s0", 0.50), ("L2s1", 0.50), ("P2HEAD", 0.0)):
        _, so_i, lab_i = _planted_analysis(delta)
        sp, lp = str(tmp_path / f"s2_{name}.npz"), str(tmp_path / f"l2_{name}.npz")
        np.savez(sp, **so_i)
        np.savez(lp, **lab_i)
        args += ["--search", f"{name}={sp}", "--labels", f"{name}={lp}"]
    assert AN.main(args) == 0
    v = json.load(open(tmp_path / "r2.json"))["verdict"]
    assert v["combined"] == "O1_PASS" and "BOTH banks" in v["next_arm"] and v["attribution_vs_P2HEAD"].startswith("descriptive")
