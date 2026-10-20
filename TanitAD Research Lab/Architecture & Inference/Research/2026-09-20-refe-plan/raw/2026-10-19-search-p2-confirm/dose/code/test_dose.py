#!/usr/bin/env python3
"""Tests of the SEARCH-DOSE hops (dose_chain.py).  Tiny synthetic input only: a 12-token planted world and the fake scorer; NO real bundle, NO label, NO GPU.  The real FILES read are the sealed C1 prereg, the
two config files, and the pinned code: all small and read-only.

    cd doseamend/code && CUDA_VISIBLE_DEVICES=-1 python -m pytest test_dose.py -q -p no:cacheprovider --basetemp=<scratch>
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dose_chain as DC  # noqa: E402

CH, AC, AB, A = DC.CH, DC.AC, DC.AB, DC.A
import fake_scorer  # noqa: E402
import l2_scorer as L2S  # noqa: E402
import spc_launch as SL  # noqa: E402
import stepB  # noqa: E402
import test_spc as TS  # noqa: E402  (the sealed suite's own planting helpers, imported not copied)

FAKE = fake_scorer.FakeScorerModule()
SEALED = os.path.join(DC.SPC_PKG, CH.SEALED_NAME)
real = pytest.mark.skipif(not (os.path.isfile(SEALED) and os.path.isfile(DC.CONFIG_O4) and os.path.isfile(DC.CONFIG_C8)), reason="sealed C1 prereg / config files absent")
T_TOK, S_SEEDS = 3, 3


class NoChild:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, env, log_path, cwd=None):
        self.calls.append(list(argv))
        raise AssertionError(f"a child process was started: {argv[:4]}")


class SpyScorer:
    """wraps the fake scorer module: records the leading dimension of EVERY set the search hands to score_fn"""
    def __init__(self, inner):
        self.inner, self.sizes = inner, []

    def load_scorer(self, *a, **k):
        return self.inner.load_scorer(*a, **k)

    def make_score_fn(self, *a, **k):
        fn = self.inner.make_score_fn(*a, **k)

        def spy(S):
            self.sizes.append(int(np.asarray(S).shape[0]))
            return fn(S)
        spy.bind, spy.info = fn.bind, fn.info
        return spy


def _plans(rng, n_tok):
    out = np.zeros((n_tok, 64, 20, 3), np.float32)
    for t in range(n_tok):
        for k in range(64):
            v, kappa = rng.uniform(6.0, 14.0), rng.normal(0, 0.01)
            s = v * 0.2 * np.arange(1, 21)
            yaw = kappa * s
            out[t, k, :, 0], out[t, k, :, 1], out[t, k, :, 2] = s * np.cos(yaw / 2), s * np.sin(yaw / 2), yaw
            out[t, k, -1, 2] += 0.3
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    rng = np.random.default_rng(7)
    n = 12
    b = {"token": np.array([f"tok{i:03d}" for i in range(n)]), "log": np.array([f"log{(i % 4):02d}" for i in range(n)]), "traj_raw": _plans(rng, n), "logits": np.zeros((n, 64, 6), np.float32),
         "pick": rng.integers(0, 64, n).astype(np.int64), "sub": np.clip(rng.uniform(0.3, 1.0, (n, 64, 6)), 0, 1), "pdms": np.clip(rng.uniform(0.2, 1.0, (n, 64)), 0, 1), "valid": np.ones((n, 64), bool)}
    bp = str(d / "bundle.npz")
    np.savez(bp, **b)
    ck = str(d / "model_final.pt")
    open(ck, "wb").write(b"not a real checkpoint, the fake scorer ignores it")
    ns, _ = FAKE.load_scorer(ck)
    return {"dir": d, "bundle": bp, "bundle_d": b, "ckpt": ck, "ckpt_sha": L2S.sha256_file(ck), "carve_sha": stepB.tokens_sha256(b["token"]), "n": n,
            "W": ns.score_head.weight.detach().numpy().copy(), "b": ns.score_head.bias.detach().numpy().copy()}


@pytest.fixture(scope="module")
def real_pins(tmp_path_factory):
    d = tmp_path_factory.mktemp("rp")
    ch = DC.DoseChain(DC.make_cfg(work=str(d / "dose"), c1_work=str(d / "c1"), hash_ckpt=False), runner=NoChild(), out=io.StringIO())
    return ch.check_seal()


def dose_chain(tmp, **kw):
    return DC.DoseChain(DC.make_cfg(work=str(tmp / "dose"), c1_work=str(tmp / "c1"), dose_pkg=str(tmp / "pkg"), hash_ckpt=False, **kw), runner=NoChild(), out=io.StringIO(), ram_fn=lambda: 64.0, free_fn=lambda p: 500.0)


def plant_stamp(work, which="C1"):
    d = os.path.join(str(work), f"gate_{which}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "G-REACH-STEPB_PASS_planted.json"), "w").write("{}")


# ------------------------------------------------------------------------------------------------------------------------------ the config file and the pins
@real
def test_o4_file_is_the_pinned_one_and_differs_from_c8_only_in_m_elites_widen():
    assert CH.sha_lf(DC.CONFIG_O4) == DC.O4_SHA256
    text = open(SEALED, encoding="utf-8").read()
    assert CH.parse_pins(text)["config_o4_sha256"] == DC.O4_SHA256                         # the C1 seal pins the same value
    o4, c8 = stepB.StepBConfig.from_json(DC.CONFIG_O4), stepB.StepBConfig.from_json(DC.CONFIG_C8)
    assert {k: getattr(o4, k) for k in DC.O4_LEVERS} == DC.O4_LEVERS
    stepB.validate_config(o4)
    assert o4 == stepB.c8_from_grid(m_pop=256, n_elite=32, widen=2.0)
    diff = {k for k in o4.as_json() if o4.as_json()[k] != c8.as_json()[k]}
    assert diff == {"m_pop", "n_elite", "widen"}                                           # everything else (sigma0, clips, K, profile, seeds) is C1's


@real
def test_check_seal_green_keeps_c1s_seal_and_swaps_only_the_config_pin(real_pins):
    assert real_pins["config_sha256"] == DC.O4_SHA256 and real_pins["config_c8_sha256"] == CH.sha_lf(DC.CONFIG_C8) != DC.O4_SHA256
    assert real_pins["b_rest_tokens_sha256"] == SL.B_REST_SHA256                           # the data clause is C1's


@real
def test_control_c1_default_chain_still_binds_the_c8_config(tmp_path):
    cfg = CH.Cfg(DC.SPC_PKG, work=str(tmp_path / "w"), hash_ckpt=False)
    assert cfg.config == CH.CONFIG == DC.CONFIG_C8
    ch = CH.Chain(cfg, runner=NoChild(), out=io.StringIO(), ram_fn=lambda: 64.0, free_fn=lambda p: 500.0)
    pins = ch.check_seal()
    assert pins["config_sha256"] == CH.sha_lf(DC.CONFIG_C8) and pins["config_sha256"] != DC.O4_SHA256


@real
def test_c1s_sealed_code_is_byte_identical_to_its_pins():
    pins = CH.parse_pins(open(SEALED, encoding="utf-8").read())
    n = 0
    for k, v in pins.items():
        if k.startswith("code:") and os.path.isfile(os.path.join(DC.SPC_CODE, k[5:])):
            assert CH.sha_lf(os.path.join(DC.SPC_CODE, k[5:])) == v, k
            n += 1
    assert n >= 10 and not os.path.exists(os.path.join(DC.SPC_CODE, "dose_chain.py"))


@real
def test_check_seal_red_when_the_chain_points_at_the_c8_file_or_a_tampered_o4(tmp_path):
    ch = dose_chain(tmp_path, config=DC.CONFIG_C8)                                         # the dose chain pointed at C1's own config: not the O4 pin
    with pytest.raises(CH.StopChain) as e:
        ch.check_seal()
    assert e.value.reason == "SEAL_PIN_MISMATCH" and "config_o4_sha256" in e.value.detail
    t = tmp_path / "O4_tampered.json"
    t.write_bytes(open(DC.CONFIG_O4, "rb").read().replace(b'"m_pop": 256', b'"m_pop": 255'))
    with pytest.raises(CH.StopChain) as e:
        dose_chain(tmp_path, config=str(t)).check_seal()
    assert e.value.reason == "SEAL_PIN_MISMATCH"


# ------------------------------------------------------------------------------------------------------------------------------ REACHABILITY: argv -> launcher -> config -> sha pin -> M in the search loop
def _launch(monkeypatch, world, argv, tmp):
    """run the SEALED launcher (`spc_launch.cmd_search`) on the argv the chain built, up to Step B's `run_search_cli`, with only the environment substituted (checkpoint, scorer, the planted world's data clause)"""
    real_cli, seen, spy = stepB.run_search_cli, {}, SpyScorer(FAKE)

    def shim(ns, enforce_stamp=True, carve_sha256=None, **k):
        seen.update(enforce_stamp=enforce_stamp, carve=carve_sha256, config=ns.config, config_sha=ns.config_sha256, arm=ns.arm, bundle=ns.bundle)
        ns.ckpt, ns.ckpt_sha256, ns.bundle, ns.out_dir, ns.end = world["ckpt"], world["ckpt_sha"], world["bundle"], str(tmp / "out"), T_TOK
        return real_cli(ns, enforce_stamp=False, scorer_mod=spy, carve_sha256=world["carve_sha"])
    monkeypatch.setattr(stepB, "run_search_cli", shim)
    assert SL.cmd_search(argv[argv.index("search") + 1:]) == 0
    z = np.load(str(tmp / "out" / "shard_P2HEAD_0000.npz"), allow_pickle=False)
    return seen, spy.sizes, json.loads(str(z["meta"])), int(np.asarray(z["moved"]).sum())


@real
def test_dose_argv_reaches_the_search_loop_with_m256_and_32_elites(monkeypatch, tmp_path, world, real_pins):
    ch = dose_chain(tmp_path)
    plant_stamp(ch.cfg.work)
    argv = ch.search_args("C1", real_pins, str(tmp_path / "out"))
    seen, sizes, meta, moved = _launch(monkeypatch, world, argv, tmp_path)
    assert seen["config"] == DC.CONFIG_O4 and seen["config_sha"] == DC.O4_SHA256 and seen["arm"] == "P2HEAD"      # hop 1-2: argv -> config path and sha pin
    assert seen["enforce_stamp"] is True and seen["carve"] == SL.B_REST_SHA256                                       # the C1 launcher's own data clause and stamp rule
    assert seen["bundle"] == os.path.join(str(tmp_path / "c1"), "B_REST", "bundle", "bundle.npz")                   # the bundle is C1's, not a copy
    assert meta["config_sha256"] == DC.O4_SHA256 and meta["arm"] == "P2HEAD" and meta["head_sha256"] is None       # hop 3: the sha the search recorded
    K, M, E, S, T = 5, 256, 32, S_SEEDS, T_TOK
    assert set(sizes) == {M, 64}                                                                                     # hop 4: the sets the loop scored
    assert sizes.count(M) == K * S * T                                                                               # K = 5 populations of 256 per (token, seed)
    assert sizes.count(64) == T * (1 + S * (1 + E)) + moved                                                          # 1 base + per seed (1 fan + 32 elite re-scores) + 1 per moved output


@real
def test_control_default_c1_argv_reaches_the_search_loop_with_m64_and_8_elites(monkeypatch, tmp_path, world):
    cfg = CH.Cfg(DC.SPC_PKG, work=str(tmp_path / "c1"), hash_ckpt=False)
    ch = CH.Chain(cfg, runner=NoChild(), out=io.StringIO(), ram_fn=lambda: 64.0, free_fn=lambda p: 500.0)
    plant_stamp(cfg.work)
    pins = {"ckpt_sha256": "c" * 64, "config_sha256": CH.sha_lf(DC.CONFIG_C8)}
    seen, sizes, meta, moved = _launch(monkeypatch, world, ch.search_args("C1", pins, str(tmp_path / "out")), tmp_path)
    assert seen["config"] == DC.CONFIG_C8 and seen["config_sha"] == CH.sha_lf(DC.CONFIG_C8) and meta["config_sha256"] == CH.sha_lf(DC.CONFIG_C8)
    K, E, S, T = 5, 8, S_SEEDS, T_TOK
    assert set(sizes) == {64}                                                                                        # C1 as sealed: every set has 64 members
    assert len(sizes) == T * (1 + S * (K + 1 + E)) + moved


@real
def test_o4_config_in_the_dose_chain_is_the_only_default_difference(tmp_path):
    cfg1, cfg2 = CH.Cfg(DC.SPC_PKG, work=str(tmp_path / "a")), DC.make_cfg(work=str(tmp_path / "b"), c1_work=str(tmp_path / "c"))
    diff = {k for k in cfg1.__dict__ if k in cfg2.__dict__ and cfg1.__dict__[k] != cfg2.__dict__[k]}
    assert diff == {"config", "work", "min_free_gib"} and cfg2.config == DC.CONFIG_O4 and cfg1.config == DC.CONFIG_C8


def test_step_b_gate_passes_on_the_o4_config_with_every_mutation_red(world, tmp_path):
    """Step B's own G-REACH on the pinned O4 file (planted world, fake scorer): first population bit-exact for M = 256 and widen 2.0, extremes hit the clips, every mutation arm RED"""
    import g_reach_stepb as G
    ns, _ = FAKE.load_scorer(world["ckpt"])
    W, b = world["W"].copy(), world["b"].copy()
    W[L2S.EP], b[L2S.EP] = world["W"][L2S.EP] * 0.5 + 0.3, world["b"][L2S.EP] + 0.2
    hp = str(world["dir"] / "gate_head.npz")
    np.savez(hp, weight=W.astype(np.float32), bias=b.astype(np.float32), kind=np.array("test"))
    import argparse
    a = argparse.Namespace(arm="L2", head=hp, head_sha256=L2S.sha256_file(hp), bundle=world["bundle"], ctx="unused", ckpt=world["ckpt"], ckpt_sha256=world["ckpt_sha"], refe_dir="unused",
                           config=DC.CONFIG_O4, config_sha256=DC.O4_SHA256, threads=2)
    st = G.run_gate(a, str(tmp_path / "gate"), scorer_mod=FAKE, verbose=False, carve_sha256=world["carve_sha"])
    assert st["verdict"] == "PASS", json.dumps({"checks": [c for c in st["checks"] if not c["ok"]], "mut": [m for m in st["mutations"] if not m["red"]]})[:1500]
    e = [c for c in st["checks"] if c["name"].startswith("e_")][0]
    assert e["ok"] and e["first_population_bit_exact_all_seeds"] and e["sigma0"] == list(stepB.StepBConfig.from_json(DC.CONFIG_O4).sigma0)
    assert st["config_sha256"] == DC.O4_SHA256 and all(m["red"] for m in st["mutations"])


# ------------------------------------------------------------------------------------------------------------------------------ refusals that keep the dose arm CPU-only and on C1's bundle
def test_dose_never_builds_a_bundle_or_runs_cid(tmp_path):
    ch = dose_chain(tmp_path)
    for fn, args, reason in ((ch.gpu_stage, (), "DOSE_NEVER_BUILDS_A_BUNDLE"), (ch.cid_gate, ({},), "DOSE_HAS_NO_CID")):
        with pytest.raises(CH.StopChain) as e:
            fn(*args)
        assert e.value.code == 3 and e.value.reason == reason
    assert ch.run_child_fn.calls == []


def test_dose_bundle_dir_is_c1s_and_only_c1s(tmp_path):
    ch = dose_chain(tmp_path)
    assert ch.bundle_dir("C1") == os.path.join(str(tmp_path / "c1"), "B_REST", "bundle")
    with pytest.raises(CH.StopChain):
        ch.bundle_dir("CID")


def test_emit_prefix_is_zzdose(tmp_path):
    ch = dose_chain(tmp_path)
    ch.emit("START")
    assert ch.out.getvalue() == "ZZDOSE_START\n"


def _c1_world(tmp, world, pins, done=None, outcome="O3_NULL", bundle_ok=True, c1_cfg_sha=None, terms=True):
    """plant C1's finished state under tmp/c1: stages, RESULT_SPC.json, B-REST bundle, merged search (C8 pin) and labels"""
    c1 = str(tmp / "c1")
    os.makedirs(c1, exist_ok=True)
    TS.plant_bundle(c1, "C1")
    bsha = CH.sha256_file(os.path.join(c1, "B_REST", "bundle", "bundle.npz"))
    toks = TS._tokens("C1")
    meta = {"arm": "P2HEAD", "head_sha256": None, "ckpt_sha256": pins["ckpt_sha256"], "config_sha256": c1_cfg_sha or pins["config_c8_sha256"], "data_clause": {"carve_sha256": SL.READS["C1"]}, "bundle_sha256": bsha if bundle_ok else "0" * 64}
    np.savez(os.path.join(c1, "search_C1_P2HEAD.npz"), meta=np.array(json.dumps(meta)), token=np.array(toks))
    TS.plant_labels(os.path.join(c1, "labels_C1_P2HEAD.npz"), toks)
    json.dump({"stage": "LABEL_DONE", "done": DC.C1_STAGES if done is None else done}, open(os.path.join(c1, "chain_state.json"), "w"))
    json.dump({"outcome": outcome, **({"safety_points": {k: 0.0 for k in AC.SAFETY_TERMS}, "moved_share": 0.2} if terms else {})}, open(os.path.join(c1, "RESULT_SPC.json"), "w"))
    return bsha


@real
def test_c1_done_green_and_red_arms(tmp_path, world, real_pins):
    ch = dose_chain(tmp_path)
    bsha = _c1_world(tmp_path, world, real_pins)
    c1 = ch.check_c1_done(real_pins)
    assert c1["outcome"] == "O3_NULL" and c1["moved_share"] == 0.2 and set(c1["safety_points"]) == set(AC.SAFETY_TERMS) and ch.c1_bundle_sha == bsha and "C1_DONE" in ch.st["done"]
    for kw, why in (({"done": list(DC.C1_STAGES[:-1])}, "stages"), ({"bundle_ok": False}, "bundle"), ({"outcome": "SOMETHING"}, "outcome"), ({"c1_cfg_sha": DC.O4_SHA256}, "c1_merged_not_under_c8"), ({"terms": False}, "c1_licence_terms_missing")):
        t = tmp_path / why
        ch2 = dose_chain(t)
        _c1_world(t, world, real_pins, **kw)
        with pytest.raises(CH.StopChain) as e:
            ch2.check_c1_done(real_pins)
        assert e.value.reason == "C1_NOT_DONE", why
    with pytest.raises(CH.StopChain) as e:                                                 # nothing planted at all
        dose_chain(tmp_path / "empty").check_c1_done(real_pins)
    assert e.value.reason == "C1_NOT_DONE"


@real
def test_dose_work_dir_must_not_be_c1s(tmp_path, world, real_pins):
    ch = DC.DoseChain(DC.make_cfg(work=str(tmp_path / "c1"), c1_work=str(tmp_path / "c1"), hash_ckpt=False), runner=NoChild(), out=io.StringIO())
    _c1_world(tmp_path, world, real_pins)
    with pytest.raises(CH.StopChain) as e:
        ch.check_c1_done(real_pins)
    assert e.value.reason == "DOSE_WORK_IS_C1_WORK"


@real
def test_dose_merged_valid_binds_the_o4_config_and_c1s_bundle(tmp_path, world, real_pins):
    ch = dose_chain(tmp_path)
    bsha = _c1_world(tmp_path, world, real_pins)
    ch.check_c1_done(real_pins)
    toks = TS._tokens("C1")

    def plant(**kw):
        meta = {"arm": "P2HEAD", "head_sha256": None, "ckpt_sha256": real_pins["ckpt_sha256"], "config_sha256": DC.O4_SHA256, "data_clause": {"carve_sha256": SL.READS["C1"]}, "bundle_sha256": bsha}
        meta.update(kw)
        p = str(tmp_path / "dose_merged.npz")
        np.savez(p, meta=np.array(json.dumps(meta)), token=np.array(toks))
        return p
    assert ch.merged_valid(plant(), "C1", 905, real_pins) == (True, "")
    assert not ch.merged_valid(plant(config_sha256=real_pins["config_c8_sha256"]), "C1", 905, real_pins)[0]      # a C8 file is not a dose file
    assert not ch.merged_valid(plant(bundle_sha256="1" * 64), "C1", 905, real_pins)[0]                          # not C1's bundle
    assert not ch.merged_valid(plant(head_sha256="h"), "C1", 905, real_pins)[0]                                 # C1's no-head rule is inherited
    assert not ch.merged_valid(plant(data_clause={"carve_sha256": "2" * 64}), "C1", 905, real_pins)[0]


# ------------------------------------------------------------------------------------------------------------------------------ G-SEAL-DOSE
def _amend(tmp, c1_sha, c8_sha, code_sha, status="Status: **SEALED**", o4=DC.O4_SHA256, sha=None, include_code=True, ruling_pin="ok", ruling_edit=False):
    pkg = tmp / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    rtext = "# MM ruling\nDQ1 (a).\n"
    (pkg / DC.RULING_NAME).write_bytes(rtext.encode())
    pins = [f"c1_prereg_sha256 {c1_sha}", f"config_o4_sha256 {o4}", f"config_c8_sha256 {c8_sha}"] + ([f"code:dose_chain.py {code_sha}"] if include_code else [])
    if ruling_pin == "ok":
        pins.append(f"ruling_sha256 {hashlib.sha256(rtext.encode()).hexdigest()}")
    elif ruling_pin == "wrong":
        pins.append("ruling_sha256 " + "6" * 64)
    text = f"# AMENDMENT\n\n{status} 2026-10-20\n\n<!-- PINS-BEGIN -->\n```\n" + "\n".join(pins) + "\n```\n<!-- PINS-END -->\n"
    (pkg / DC.AMEND_NAME).write_bytes(text.encode())
    (pkg / DC.AMEND_SHA).write_text((sha or hashlib.sha256(text.encode()).hexdigest()) + "\n")


@real
def test_amendment_seal_green_and_red_arms(tmp_path, real_pins):
    c1_sha = open(os.path.join(DC.SPC_PKG, "PREREG_SEARCH_P2_CONFIRM.sha256")).read().split()[0]
    code = CH.sha_lf(os.path.join(HERE, "dose_chain.py"))
    c8 = real_pins["config_c8_sha256"]
    ch = dose_chain(tmp_path)
    with pytest.raises(CH.StopChain) as e:
        ch.check_amendment(real_pins)
    assert e.value.reason == "AMEND_MISSING"
    _amend(tmp_path, c1_sha, c8, code)
    ch.check_amendment(real_pins)
    assert "AMEND_SEAL_OK" in ch.st["done"]
    reds = {"AMEND_SHA_MISMATCH": dict(sha="0" * 64), "AMEND_NOT_SEALED": dict(status="Status: **DRAFT**"), "AMEND_PIN_MISMATCH": dict(o4="3" * 64)}
    for reason, kw in reds.items():
        t = tmp_path / reason
        _amend(t, c1_sha, c8, code, **kw)
        with pytest.raises(CH.StopChain) as e:
            dose_chain(t).check_amendment(real_pins)
        assert e.value.reason == reason
    for tag, args, kw in (("wrong_c1", ("4" * 64, c8, code), {}), ("wrong_code", (c1_sha, c8, "5" * 64), {}), ("unpinned", (c1_sha, c8, code), {"include_code": False}),
                          ("ruling_unpinned", (c1_sha, c8, code), {"ruling_pin": "none"}), ("ruling_wrong_sha", (c1_sha, c8, code), {"ruling_pin": "wrong"})):
        t = tmp_path / tag
        _amend(t, *args, **kw)
        with pytest.raises(CH.StopChain) as e:
            dose_chain(t).check_amendment(real_pins)
        assert e.value.reason == "AMEND_PIN_MISMATCH", tag


# ------------------------------------------------------------------------------------------------------------------------------ analysis: C1's rules verbatim + the descriptive contrast
@pytest.fixture(scope="module")
def arms(world):
    b = world["bundle_d"]

    def run(path):
        cfg = stepB.StepBConfig.from_json(path)
        fn = FAKE.make_score_fn(world["ckpt"], "x", b["token"])
        return stepB.run_search(fn, b, range(world["n"]), cfg)
    return b, run(DC.CONFIG_C8), run(DC.CONFIG_O4)


C1READ = {"outcome": "O3_NULL", "safety_points": {k: 0.0 for k in AC.SAFETY_TERMS}, "moved_share": 0.2}


def _labels(so, rng, pdms=None, sub=None):
    n, S = so["exec"].shape[:2]
    return {"token": so["token"], "pdms": rng.uniform(0.3, 0.9, (n, S)) if pdms is None else pdms, "sub": rng.uniform(0.3, 0.9, (n, S, 6)) if sub is None else sub}


@real
def test_contrast_is_o4_minus_c8_with_the_right_sign_and_size(arms):
    b, c8, o4 = arms
    rng = np.random.default_rng(3)
    lab_c8 = _labels(c8, rng)
    lab_o4 = _labels(o4, rng, pdms=lab_c8["pdms"] + 0.02, sub=lab_c8["sub"] + 0.01)
    k = DC.contrast(b, o4, lab_o4, c8, lab_c8, B=200)
    assert k["pooled"]["point"] == pytest.approx(2.0, abs=1e-9) and k["pooled"]["lo"] <= 2.0 + 1e-9 and k["pooled"]["hi"] >= 2.0 - 1e-9
    assert all(p["point"] == pytest.approx(2.0, abs=1e-9) for p in k["per_seed"]) and len(k["per_seed"]) == S_SEEDS
    assert k["sub_scores_points"]["nc"]["point"] == pytest.approx(1.0, abs=1e-9)
    assert DC.contrast(b, c8, lab_c8, o4, lab_o4, B=200)["pooled"]["point"] == pytest.approx(-2.0, abs=1e-9)      # swapping the arms flips the sign
    assert "not a dose-response in M" in k["confound"]


@real
def test_contrast_refuses_misaligned_arms(arms):
    b, c8, o4 = arms
    rng = np.random.default_rng(4)
    lab_c8, lab_o4 = _labels(c8, rng), _labels(o4, rng)
    swapped = {**lab_c8, "token": lab_c8["token"][::-1]}
    with pytest.raises(ValueError):
        DC.contrast(b, o4, lab_o4, c8, swapped, B=50)
    short = {k: (v[:-1] if hasattr(v, "__len__") and k in ("token", "bundle_index", "exec") else v) for k, v in c8.items()}
    with pytest.raises(ValueError):
        DC.contrast(b, o4, lab_o4, short, lab_c8, B=50)


@real
def test_analyse_dose_is_c1s_classifier_with_the_dose_next_arm(arms):
    b, c8, o4 = arms
    rng = np.random.default_rng(5)
    lab_c8, lab_o4 = _labels(c8, rng), _labels(o4, rng)
    res = DC.analyse_dose(b, o4, lab_o4, c8, lab_c8, C1READ, B=200, with_concordance=False)
    ref = AC.analyse_c1(b, o4, lab_o4, B=200, with_concordance=False)
    assert res["outcome"] == ref["outcome"] and res["endpoint_dPDMS_points"] == ref["endpoint_dPDMS_points"] and res["safety_points"] == ref["safety_points"]    # C1's rules verbatim
    assert res["read"] == "DOSE" and res["next_arm"] == DC.next_arm(res["outcome"], "O3_NULL") and res["next_arm"] != ref["next_arm"]
    assert res["STRATEGIC"].startswith("PAUSED") and set(res["families"]) >= {"LONGITUDINAL", "LATERAL", "TACTICAL"}
    assert "SECOND READ OF THE SAME TOKENS" in res["stamps"]


def test_next_arm_table_covers_every_c1_outcome_and_the_inert_special_case():
    assert set(DC.DOSE_NEXT) == set(AC.OUTCOMES)
    for o in AC.OUTCOMES:
        assert DC.next_arm(o, "O3_NULL") == DC.DOSE_NEXT[o]
    assert DC.next_arm("O3_NULL", "O4_INERT").startswith("C1 read INERT; this was its committed raised-M arm, run ONCE:")
    assert "MOVED the pick" in DC.next_arm("O2_DIRECTIONAL", "O4_INERT") and "MOVED" not in DC.next_arm("NEGATIVE", "O4_INERT")
    assert "CLOSED" in DC.DOSE_NEXT["O4_INERT"] and "P6 trunk" in DC.DOSE_NEXT["O3_NULL"] and "licence" in DC.DOSE_NEXT["O1_CONFIRMED"]


# ------------------------------------------------------------------------------------------------------------------------------ the run order
@real
def test_run_order_is_the_documented_one_and_never_touches_the_gpu_or_cid(tmp_path, monkeypatch):
    ch = dose_chain(tmp_path)
    log = []
    pins = {"config_sha256": DC.O4_SHA256}

    def step(name, ret=None):
        return lambda *a, **k: (log.append(name), ret)[1]
    for nm, ret in (("check_seal", pins), ("check_amendment", None), ("check_data", None), ("check_c1_done", {"outcome": "O3_NULL"}), ("check_disk", None), ("check_bundle", None),
                    ("subid_gate", None), ("mint_gates", None), ("run_search", "m.npz"), ("selfrep_gate", None), ("run_labels", "l.npz"), ("run_analysis", None), ("gpu_stage", None), ("cid_gate", None)):
        monkeypatch.setattr(ch, nm, step(nm, ret))
    assert ch.run() == 0
    assert log == ["check_seal", "check_amendment", "check_data", "check_c1_done", "check_disk", "check_bundle", "subid_gate", "mint_gates", "run_search", "selfrep_gate", "run_labels", "run_analysis"]


@real
def test_run_refuses_at_the_first_gate_and_starts_no_child(tmp_path):
    ch = dose_chain(tmp_path)                                                              # no AMENDMENT_DOSE.md in tmp/pkg
    assert ch.run() == 5 and ch.run_child_fn.calls == []
    out = ch.out.getvalue()
    assert re.search(r"^ZZDOSE_SEAL_OK ", out, re.M) and re.search(r"^ZZDOSE_STOP AMEND_MISSING", out, re.M) and "ZZSPC_" not in out


@real
def test_check_bundle_green_and_red(tmp_path, world, real_pins):
    ch = dose_chain(tmp_path)
    _c1_world(tmp_path, world, real_pins)
    ch.check_bundle()
    assert "BUNDLE_OK" in ch.st["done"]
    TS.plant_bundle(str(tmp_path / "c1"), "C1", tamper_ctx=True)                          # ctx.npy bytes no longer match ctx.json
    with pytest.raises(CH.StopChain) as e:
        ch.check_bundle()
    assert e.value.reason == "G_BUNDLE_INVALID" and e.value.code == 3


@real
def test_mint_gates_builds_the_o4_gate_argv_and_refuses_without_a_stamp(tmp_path, real_pins):
    calls = []

    def runner(argv, env, log_path, cwd=None):
        calls.append(list(argv))
        return 0                                                                           # rc 0 but no stamp written: the gate must still be refused
    ch = dose_chain(tmp_path)
    ch.run_child_fn = runner
    with pytest.raises(CH.StopChain) as e:
        ch.mint_gates(real_pins)
    assert e.value.reason == "G_REACH_NOT_PASSED" and len(calls) == 1
    a = calls[0]
    assert a[a.index("gate") + 1:a.index("gate") + 3] == ["--read", "C1"] and "CID" not in a
    assert a[a.index("--config") + 1] == DC.CONFIG_O4 and a[a.index("--config-sha256") + 1] == DC.O4_SHA256
    assert a[a.index("--bundle") + 1] == os.path.join(str(tmp_path / "c1"), "B_REST", "bundle", "bundle.npz")
    assert a[a.index("--out-dir") + 1] == ch.cfg.d("gate_C1") and ch.cfg.work != ch.cfg.c1_work
    ch2 = dose_chain(tmp_path / "b")
    ch2.run_child_fn = lambda argv, env, log_path, cwd=None: 3                              # a failing gate
    with pytest.raises(CH.StopChain) as e:
        ch2.mint_gates(real_pins)
    assert e.value.reason == "G_REACH_NOT_PASSED"


@real
def test_g_order_analysis_refused_unless_every_dose_stage_is_done(tmp_path, real_pins):
    assert {"SUBID_PASS", "SELFREP_PASS", "REACH_PASS", "SEARCH_DONE", "LABEL_DONE", "C1_DONE", "BUNDLE_OK"} <= set(DC.DOSE_STAGES)
    ch = dose_chain(tmp_path)
    ch.st["done"] = list(DC.DOSE_STAGES[:-1])
    with pytest.raises(CH.StopChain) as e:
        ch.run_analysis("m", "l", real_pins, {"outcome": "O3_NULL"})
    assert e.value.reason == "G_ORDER" and e.value.code == 8


# ------------------------------------------------------------------------------------------------------------------------------ the seal tool: a sealed draft passes the chain's own amendment gate
import seal_dose as SD  # noqa: E402

MIN_DRAFT = "# AMENDMENT (draft)\n\nStatus: **DRAFT** the MM seals\n\n@@RULING@@\n\n@@PINS@@\n\nbody. Status: not a status line\n"


@real
def test_seal_tool_output_passes_check_amendment_and_never_overwrites(tmp_path, real_pins):
    (tmp_path / "pkg").mkdir()
    src = tmp_path / "draft.md"
    src.write_text(MIN_DRAFT, encoding="utf-8")
    ruling = tmp_path / "pkg" / DC.RULING_NAME
    ruling.write_text("# MM ruling\nDQ1 (a).\n", encoding="utf-8")
    sha = SD.seal(str(src), str(tmp_path / "pkg"), ruling=str(ruling))
    sealed = (tmp_path / "pkg" / DC.AMEND_NAME).read_text(encoding="utf-8")
    assert hashlib.sha256(sealed.encode()).hexdigest() == sha and re.search(r"^Status: \*\*SEALED\*\*", sealed, re.M) and "@@" not in sealed and "THE RULING PREVAILS" in sealed
    ch = dose_chain(tmp_path)
    ch.check_amendment(real_pins)                                                           # the dose chain's own gate accepts what the tool wrote
    assert "AMEND_SEAL_OK" in ch.st["done"]
    with pytest.raises(SD.SealRefused):
        SD.seal(str(src), str(tmp_path / "pkg"), ruling=str(ruling))                          # never overwritten
    pins = CH.parse_pins(sealed)
    assert pins["ruling_sha256"] == CH.sha_lf(str(ruling)) and set(k[5:] for k in pins if k.startswith("code:")) == set(SD.PINNED_CODE)
    (tmp_path / DC.AMEND_NAME).write_text("x")
    bad = tmp_path / "bad"
    for name, text in (("sealed", MIN_DRAFT.replace("DRAFT", "SEALED")), ("noruling", MIN_DRAFT.replace("@@RULING@@", "")), ("twopins", MIN_DRAFT + "@@PINS@@")):
        (tmp_path / name).mkdir()
        (tmp_path / name / DC.RULING_NAME).write_text("# r\n", encoding="utf-8")
        (tmp_path / f"{name}.md").write_text(text, encoding="utf-8")
        with pytest.raises(SD.SealRefused):
            SD.seal(str(tmp_path / f"{name}.md"), str(tmp_path / name), ruling=str(tmp_path / name / DC.RULING_NAME))
    (tmp_path / "np").mkdir()
    with pytest.raises(SD.SealRefused):                                                      # no ruling: refused
        SD.seal(str(src), str(tmp_path / "np"))
    stray = tmp_path / "stray.md"
    stray.write_text("x")
    with pytest.raises(SD.SealRefused):                                                      # a ruling that is not MM_RULING_DOSE.md inside the package: refused
        SD.seal(str(src), str(tmp_path / "np"), ruling=str(stray))
    assert bad.exists() is False


@real
def test_a_post_seal_edit_of_a_pinned_file_is_refused(tmp_path, real_pins):
    """copy the pinned code to a scratch dir, seal against it, edit one byte of dose_chain.py there: the sealed pins no longer match (the chain's gate is a sha check of the file at HERE)"""
    import shutil
    code = tmp_path / "code"
    code.mkdir()
    for f in SD.PINNED_CODE:
        shutil.copy(os.path.join(HERE, f), code / f)
    (tmp_path / "pkg").mkdir()
    src = tmp_path / "draft.md"
    src.write_text(MIN_DRAFT, encoding="utf-8")
    (tmp_path / "pkg" / DC.RULING_NAME).write_text("# r\n", encoding="utf-8")
    SD.seal(str(src), str(tmp_path / "pkg"), ruling=str(tmp_path / "pkg" / DC.RULING_NAME), code_dir=str(code))
    pins = CH.parse_pins((tmp_path / "pkg" / DC.AMEND_NAME).read_text(encoding="utf-8"))
    assert set(k[5:] for k in pins if k.startswith("code:")) == set(SD.PINNED_CODE)
    assert all(CH.sha_lf(str(code / f)) == pins["code:" + f] for f in SD.PINNED_CODE)
    (code / "dose_chain.py").write_bytes((code / "dose_chain.py").read_bytes() + b"\n# one edit\n")
    assert CH.sha_lf(str(code / "dose_chain.py")) != pins["code:dose_chain.py"]


def test_the_real_draft_is_sealable_when_present(tmp_path):
    draft = os.path.join(os.path.dirname(HERE), "AMENDMENT_DOSE_DRAFT.md")
    if not os.path.isfile(draft) or not os.path.isfile(SEALED):
        pytest.skip("draft or sealed C1 prereg absent")
    (tmp_path / "pkg").mkdir()
    import shutil
    ruling_src = os.path.join(os.path.dirname(HERE), DC.RULING_NAME)
    if not os.path.isfile(ruling_src):
        pytest.skip("the MM ruling is absent")
    shutil.copy(ruling_src, tmp_path / "pkg" / DC.RULING_NAME)
    SD.seal(draft, str(tmp_path / "pkg"), ruling=str(tmp_path / "pkg" / DC.RULING_NAME))
    sealed = (tmp_path / "pkg" / DC.AMEND_NAME).read_text(encoding="utf-8")
    assert "@@" not in sealed and sealed.count("<!-- PINS-BEGIN -->") == 1 and "Status: **SEALED**" in sealed and "THE RULING PREVAILS" in sealed
    assert CH.parse_pins(sealed)["ruling_sha256"] == CH.sha_lf(ruling_src)


def test_draft_quotes_the_code_next_arms_verbatim():
    """prose and code must not drift: every NEXT text the code can print appears word for word in the draft's outcome table (the registry-vs-prose class)"""
    draft = os.path.join(os.path.dirname(HERE), "AMENDMENT_DOSE_DRAFT.md")
    sealed = os.path.join(os.path.dirname(HERE), DC.AMEND_NAME)
    path = sealed if os.path.isfile(sealed) else draft
    if not os.path.isfile(path):
        pytest.skip("neither the draft nor the sealed amendment is present")
    norm = lambda t: re.sub(r"\s+", " ", t).strip()          # noqa: E731
    doc = norm(open(path, encoding="utf-8").read())
    for o, txt in DC.DOSE_NEXT.items():
        assert norm(txt) in doc, o
    for needle in (DC.O4_SHA256, "658ba1576af175a9a345692e284b1515604a6065279b921b2a4ca6dead6529d1", SL.B_REST_SHA256, "SECOND READ OF THE SAME TOKENS"):
        assert needle in doc, needle


# ------------------------------------------------------------------------------------------------------------------------------ MM_RULING_DOSE.md DQ4: the mechanical C2 config rule
def _rd(outcome, nc=0.0, moved=0.2, **safety):
    sp = {k: 0.0 for k in AC.SAFETY_TERMS}
    sp["nc"] = nc
    sp.update(safety)
    return {"outcome": outcome, "safety_points": sp, "moved_share": moved}


def test_licensed_terms_are_the_c2_licence_terms():
    assert DC.licensed(_rd("O1_CONFIRMED")) and DC.licensed(_rd("O2_DIRECTIONAL")) and not DC.licensed(_rd("O2_DIRECTIONAL"), strict=True) and DC.licensed(_rd("O1_CONFIRMED"), strict=True)
    for o in ("O3_NULL", "NEGATIVE", "TRADE", "O4_INERT"):
        assert not DC.licensed(_rd(o))
    assert DC.licensed(_rd("O2_DIRECTIONAL", nc=-0.25)) and not DC.licensed(_rd("O2_DIRECTIONAL", nc=-0.2501))        # safety bar -0.25 inclusive
    assert DC.licensed(_rd("O1_CONFIRMED", moved=0.05)) and not DC.licensed(_rd("O1_CONFIRMED", moved=0.0499))        # moved share >= 5 % inclusive
    for bad in ({"outcome": "O1_CONFIRMED"}, {**_rd("O1_CONFIRMED"), "moved_share": float("nan")}, {**_rd("O1_CONFIRMED"), "safety_points": {"nc": 0.0}}, {**_rd("O1_CONFIRMED"), "moved_share": None}):
        assert not DC.licensed(bad)                                                                                    # a missing or NaN term is never a licence


@pytest.mark.parametrize("c1,dose,lo,want,row", [
    # row 1: C1 licensed, dose not -> C8 (whatever the contrast says)
    (_rd("O1_CONFIRMED"), _rd("O3_NULL"), 9.0, "C8", 1), (_rd("O2_DIRECTIONAL"), _rd("NEGATIVE"), 9.0, "C8", 1), (_rd("O1_CONFIRMED"), _rd("TRADE"), 9.0, "C8", 1),
    (_rd("O1_CONFIRMED"), _rd("O4_INERT", moved=0.0), 9.0, "C8", 1), (_rd("O1_CONFIRMED"), _rd("O1_CONFIRMED", moved=0.03), 9.0, "C8", 1), (_rd("O2_DIRECTIONAL"), _rd("O2_DIRECTIONAL", nc=-0.3), 9.0, "C8", 1),
    # row 2: C1 not licensed, dose O1 + safety + moved -> O4 (whatever the contrast says)
    (_rd("O3_NULL"), _rd("O1_CONFIRMED"), -9.0, "O4", 2), (_rd("O4_INERT", moved=0.0), _rd("O1_CONFIRMED"), float("nan"), "O4", 2), (_rd("NEGATIVE"), _rd("O1_CONFIRMED"), 0.0, "O4", 2),
    (_rd("TRADE"), _rd("O1_CONFIRMED"), 0.0, "O4", 2), (_rd("O1_CONFIRMED", moved=0.03), _rd("O1_CONFIRMED"), 0.0, "O4", 2), (_rd("O2_DIRECTIONAL", ttc=-0.4), _rd("O1_CONFIRMED"), 0.0, "O4", 2),
    # row 3: both licensed -> C8 unless the paired O4-minus-C8 contrast has lower CI > 0
    (_rd("O1_CONFIRMED"), _rd("O1_CONFIRMED"), 0.01, "O4", 3), (_rd("O1_CONFIRMED"), _rd("O1_CONFIRMED"), 0.0, "C8", 3), (_rd("O1_CONFIRMED"), _rd("O1_CONFIRMED"), -1.0, "C8", 3),
    (_rd("O1_CONFIRMED"), _rd("O2_DIRECTIONAL"), 0.5, "O4", 3), (_rd("O2_DIRECTIONAL"), _rd("O2_DIRECTIONAL"), 0.5, "O4", 3), (_rd("O2_DIRECTIONAL"), _rd("O1_CONFIRMED"), -0.5, "C8", 3),
    (_rd("O1_CONFIRMED"), _rd("O1_CONFIRMED"), float("nan"), "C8", 3), (_rd("O2_DIRECTIONAL"), _rd("O2_DIRECTIONAL"), None, "C8", 3),
    # row 4: neither -> no C2, and NO contrast can rescue it; the O2-ONLY dose path (C1 not licensed) yields no C2
    (_rd("O3_NULL"), _rd("O3_NULL"), 9.0, None, 4), (_rd("NEGATIVE"), _rd("TRADE"), 9.0, None, 4), (_rd("O4_INERT", moved=0.0), _rd("O4_INERT", moved=0.0), 9.0, None, 4),
    (_rd("O3_NULL"), _rd("O2_DIRECTIONAL"), 9.0, None, 4), (_rd("O4_INERT", moved=0.0), _rd("O2_DIRECTIONAL"), 9.0, None, 4), (_rd("NEGATIVE"), _rd("O2_DIRECTIONAL", moved=0.9), 9.0, None, 4),
    (_rd("O1_CONFIRMED", moved=0.03), _rd("O2_DIRECTIONAL"), 9.0, None, 4), (_rd("O3_NULL"), _rd("O1_CONFIRMED", nc=-0.3), 9.0, None, 4), (_rd("O3_NULL"), _rd("O1_CONFIRMED", moved=0.03), 9.0, None, 4),
])
def test_c2_config_every_row_of_the_committed_table(c1, dose, lo, want, row):
    r = DC.c2_config(c1, dose, lo)
    assert (r["config"], r["row"]) == (want, row)


def test_c2_config_cross_product_matches_the_ruling_table():
    """all 6 x 6 outcomes (the safety and moved terms satisfied wherever the outcome allows a licence) x both signs of the contrast, against a literal transcription of the ruling's four rows"""
    for o1 in AC.OUTCOMES:
        for o2 in AC.OUTCOMES:
            for lo in (0.3, -0.3):
                c1l, dl, ds = o1 in ("O1_CONFIRMED", "O2_DIRECTIONAL"), o2 in ("O1_CONFIRMED", "O2_DIRECTIONAL"), o2 == "O1_CONFIRMED"
                want = ("C8" if (c1l and not dl) else "O4" if (not c1l and ds) else ("O4" if lo > 0 else "C8") if (c1l and dl) else None)
                assert DC.c2_config(_rd(o1), _rd(o2), lo)["config"] == want, (o1, o2, lo)
    assert DC.c2_config(_rd("O3_NULL"), _rd("O2_DIRECTIONAL"), 5.0)["config"] is None                                   # the O2-only dose path: no C2


def test_the_draft_carries_the_rulings_dq4_table_verbatim():
    ruling = os.path.join(os.path.dirname(HERE), DC.RULING_NAME)
    sealed = os.path.join(os.path.dirname(HERE), DC.AMEND_NAME)
    draft = sealed if os.path.isfile(sealed) else os.path.join(os.path.dirname(HERE), "AMENDMENT_DOSE_DRAFT.md")
    if not (os.path.isfile(ruling) and os.path.isfile(draft)):
        pytest.skip("ruling or draft absent")
    norm = lambda t: re.sub(r"[ \t]+", " ", t).strip()          # noqa: E731
    rows = [ln for ln in open(ruling, encoding="utf-8").read().splitlines() if ln.startswith("|")]
    assert len(rows) == 6                                         # header, separator, four rows
    doc_lines = {norm(ln) for ln in open(draft, encoding="utf-8").read().splitlines()}
    for ln in rows:
        assert norm(ln) in doc_lines, ln


# ------------------------------------------------------------------------------------------------------------------------------ G-SUBID and G-SELFREP: bit-for-bit replicates with RED arms
def plant_rows(n, S=3, seed=0, vacuous=False):
    rng = np.random.default_rng(seed)
    return {"token": np.array([f"t{i:04d}" for i in range(n)]), "bundle_index": np.arange(n, dtype=np.int64), "pick": rng.integers(0, 64, n).astype(np.int64),
            "seeds": np.array([20260927, 20260928, 20260929], np.int64), "exec": rng.normal(size=(n, S, 20, 3)).astype(np.float32), "scorer_in": rng.normal(size=(n, S, 20, 3)).astype(np.float32),
            "theta": rng.normal(size=(n, S, 2)), "moved": rng.random((n, S)) < 0.3, "hist_best_agg": rng.normal(size=(n, S, 5)),
            "hist_mu": np.zeros((n, S, 5, 2)) if vacuous else rng.normal(size=(n, S, 5, 2)), "pred_logit_out": rng.normal(size=(n, S, 6)), "pred_logit_pick": rng.normal(size=(n, 6)),
            "arm_argmax": rng.integers(0, 64, n).astype(np.int64), "pick_exec": rng.normal(size=(n, 20, 3)).astype(np.float32), "t_search_s": rng.random((n, S)), "t_token_s": rng.random(n)}


def flip(a):
    """one changed bit (or one changed token string) in a copy of the array"""
    a = np.array(a, copy=True)
    if a.dtype.kind == "U":
        a = a.astype("U32")
        a[0] = a[0] + "x"
        return a
    a.reshape(-1).view(np.uint8)[0] ^= 1
    return a


PIN_KEYS = ("exec", "scorer_in", "theta", "moved", "hist_best_agg", "hist_mu", "pred_logit_out", "pred_logit_pick", "arm_argmax", "pick_exec", "pick", "token", "bundle_index")


def gate_env(tmp, world, pins, ref_rows):
    """C1's finished state + the reference file + the stamps the gates need; -> (chain, c1 dict, bundle sha, reference path)"""
    ch = dose_chain(tmp)
    bsha = _c1_world(tmp, world, pins)
    c1 = ch.check_c1_done(pins)
    ref = str(tmp / "ref.npz")
    np.savez(ref, meta=np.array("{}"), **ref_rows)
    c1["merged"] = ref
    for work, cfgsha in ((str(tmp / "c1"), pins["config_c8_sha256"]), (str(tmp / "dose"), pins["config_sha256"])):
        d = os.path.join(work, "gate_C1")
        os.makedirs(d, exist_ok=True)
        json.dump({"schema": "tanitad.g_reach_stepb/1", "verdict": "PASS", "bundle_sha256": bsha, "ckpt_sha256": pins["ckpt_sha256"], "config_sha256": cfgsha}, open(os.path.join(d, "G-REACH-STEPB_PASS_planted.json"), "w"))
    return ch, c1, bsha, ref


def make_runner(bsha, pins, rows, kind, defect=None, calls=None):
    cfg = pins["config_c8_sha256"] if kind == "SUBID" else pins["config_sha256"]

    def runner(argv, env, log_path, cwd=None):
        if calls is not None:
            calls.append(list(argv))
        out = argv[argv.index("--out-dir") + 1]
        n = int(argv[argv.index("--end") + 1])
        os.makedirs(out, exist_ok=True)
        if defect == "noshard":
            return 0
        sh = {k: (v if k == "seeds" else v[:n]) for k, v in rows.items()}
        meta = {"arm": "P2HEAD", "head_sha256": None, "ckpt_sha256": pins["ckpt_sha256"], "config_sha256": cfg, "bundle_sha256": bsha, "data_clause": {"carve_sha256": SL.READS["C1"]}, "lo": 0, "hi": n}
        if defect and defect.startswith("flip:"):
            sh[defect[5:]] = flip(sh[defect[5:]])
        elif defect == "seeds":
            sh["seeds"] = flip(sh["seeds"])
        elif defect == "dropkey":
            del sh["theta"]
        elif defect == "extrakey":
            sh["extra"] = np.zeros(n)
        elif defect == "meta_cfg":
            meta["config_sha256"] = pins["config_sha256"] if kind == "SUBID" else pins["config_c8_sha256"]
        elif defect == "meta_bundle":
            meta["bundle_sha256"] = "9" * 64
        elif defect == "meta_range":
            meta["hi"] = n + 1
        elif defect == "meta_head":
            meta["head_sha256"] = "h"
        np.savez(os.path.join(out, "shard_P2HEAD_0000.npz"), meta=np.array(json.dumps(meta)), **sh)
        return 2 if defect == "rc" else 0                                                    # "rc": a PERFECT shard but a failing exit code must still be refused
    return runner


def run_gate(ch, kind, pins, c1, ref):
    return ch.subid_gate(pins, c1) if kind == "SUBID" else ch.selfrep_gate(pins, ref)


@real
@pytest.mark.parametrize("kind,n,cfgfile,cfgkey", [("SUBID", 16, DC.CONFIG_C8, "config_c8_sha256"), ("SELFREP", 8, DC.CONFIG_O4, "config_sha256")])
def test_replicate_gates_green_run_the_right_argv_and_mark_pass(tmp_path, world, real_pins, kind, n, cfgfile, cfgkey):
    rows = plant_rows(20)
    ch, c1, bsha, ref = gate_env(tmp_path, world, real_pins, rows)
    calls = []
    ch.run_child_fn = make_runner(bsha, real_pins, rows, kind, calls=calls)
    run_gate(ch, kind, real_pins, c1, ref)
    assert f"{kind}_PASS" in ch.st["done"] and re.search(rf"^ZZDOSE_{kind}_PASS {n} tokens", ch.out.getvalue(), re.M)
    (a,) = calls
    assert a[a.index("search") + 1:a.index("search") + 3] == ["--read", "C1"] and a[a.index("--arm") + 1] == "P2HEAD" and "--head" not in a
    assert a[a.index("--config") + 1] == cfgfile and a[a.index("--config-sha256") + 1] == real_pins[cfgkey]
    assert a[a.index("--start") + 1] == "0" and a[a.index("--end") + 1] == str(n) and a[a.index("--shard") + 1] == str(n) and a[a.index("--threads") + 1] == "4"
    assert a[a.index("--bundle") + 1] == os.path.join(str(tmp_path / "c1"), "B_REST", "bundle", "bundle.npz")
    gate_dir = os.path.join(str(tmp_path / ("c1" if kind == "SUBID" else "dose")), "gate_C1")
    assert os.path.dirname(a[a.index("--gate") + 1]) == gate_dir                           # SUBID uses C1's own C8 stamp; SELFREP the dose's O4 stamp
    assert a[a.index("--out-dir") + 1] == ch.cfg.d("subid_C8" if kind == "SUBID" else "selfrep_O4")
    assert ch.cfg.work == str(tmp_path / "dose") and ch.cfg.config == DC.CONFIG_O4        # the argv-building context is restored


@real
@pytest.mark.parametrize("kind", ["SUBID", "SELFREP"])
@pytest.mark.parametrize("defect", ["flip:" + k for k in PIN_KEYS] + ["seeds", "dropkey", "extrakey", "rc", "noshard", "meta_cfg", "meta_bundle", "meta_range", "meta_head"])
def test_replicate_gates_red_arms_a_planted_difference_makes_the_chain_refuse(tmp_path, world, real_pins, kind, defect):
    rows = plant_rows(20)
    ch, c1, bsha, ref = gate_env(tmp_path, world, real_pins, rows)
    ch.run_child_fn = make_runner(bsha, real_pins, rows, kind, defect=defect)
    with pytest.raises(CH.StopChain) as e:
        run_gate(ch, kind, real_pins, c1, ref)
    assert e.value.reason == f"G_{kind}_NOT_PASSED" and e.value.code == 3 and f"{kind}_PASS" not in ch.st["done"]


@real
@pytest.mark.parametrize("kind", ["SUBID", "SELFREP"])
def test_replicate_gates_refuse_a_vacuous_reference_a_stale_shard_and_compare_only_the_first_rows(tmp_path, world, real_pins, kind):
    n = 16 if kind == "SUBID" else 8                                                       # the ruling's numbers, written out: not read back from the constants under test
    out_name = "subid_C8" if kind == "SUBID" else "selfrep_O4"
    vac = plant_rows(20, vacuous=True)                                                     # the reference search never moved its mean: an identity of nothing
    ch, c1, bsha, ref = gate_env(tmp_path / "a", world, real_pins, vac)
    ch.run_child_fn = make_runner(bsha, real_pins, vac, kind)
    with pytest.raises(CH.StopChain) as e:
        run_gate(ch, kind, real_pins, c1, ref)
    assert e.value.reason == f"G_{kind}_NOT_PASSED" and "vacuous" in e.value.detail
    rows = plant_rows(20)                                                                  # a stale, perfectly matching shard on disk is NOT accepted when the replicate writes nothing
    ch2, c2, b2, ref2 = gate_env(tmp_path / "b", world, real_pins, rows)
    out = ch2.cfg.d(out_name)
    make_runner(b2, real_pins, rows, kind)(["x", "--out-dir", out, "--end", str(n)], {}, "log")
    assert os.path.isfile(os.path.join(out, "shard_P2HEAD_0000.npz"))
    ch2.run_child_fn = make_runner(b2, real_pins, rows, kind, defect="noshard")
    with pytest.raises(CH.StopChain) as e:
        run_gate(ch2, kind, real_pins, c2, ref2)
    assert "no shard" in e.value.detail
    far = {**rows, "exec": np.concatenate([rows["exec"][:n], rows["exec"][n:] + 1.0])}      # rows beyond the first n are not compared
    ch3, c3, b3, ref3 = gate_env(tmp_path / "c", world, real_pins, far)
    ch3.run_child_fn = make_runner(b3, real_pins, rows, kind)
    run_gate(ch3, kind, real_pins, c3, ref3)
    assert f"{kind}_PASS" in ch3.st["done"]


@real
def test_subid_refuses_without_c1s_c8_stamp(tmp_path, world, real_pins):
    ch, c1, bsha, ref = gate_env(tmp_path, world, real_pins, plant_rows(20))
    for f in os.listdir(os.path.join(str(tmp_path / "c1"), "gate_C1")):
        os.remove(os.path.join(str(tmp_path / "c1"), "gate_C1", f))
    ch.run_child_fn = NoChild()
    with pytest.raises(CH.StopChain) as e:
        ch.subid_gate(real_pins, c1)
    assert e.value.reason == "G_SUBID_NOT_PASSED" and "stamp" in e.value.detail and ch.run_child_fn.calls == []


def test_rows_identical_is_bitwise_including_negative_zero_and_key_sets():
    r = plant_rows(10)
    cut = lambda d, drop=(): {k: (v if k == "seeds" else v[:4]) for k, v in d.items() if k not in drop}   # noqa: E731
    assert DC.rows_identical(r, cut(r), 4)["pass"]
    a, b = cut(r), cut(r)
    a["theta"] = np.array(a["theta"], copy=True)
    b["theta"] = np.array(b["theta"], copy=True)
    a["theta"][0, 0, 0], b["theta"][0, 0, 0] = 0.0, -0.0
    assert not DC.rows_identical(a, b, 4)["pass"]                                          # -0.0 differs from 0.0
    assert not DC.rows_identical(r, cut(r, drop=("theta",)), 4)["pass"]                    # a missing array is a key-set difference
    t = cut(r)
    t["t_search_s"], t["t_token_s"] = t["t_search_s"] + 1.0, t["t_token_s"] + 1.0
    assert DC.rows_identical(r, t, 4)["pass"]                                              # wall-clock timing VALUES are never part of the identity
    assert not DC.rows_identical(r, cut(r, drop=("t_search_s",)), 4)["pass"]               # but a shard without its timing arrays is not this code's output


# ------------------------------------------------------------------------------------------------------------------------------ the C2 config reaches the chain's own output
@real
def test_run_analysis_writes_and_prints_the_c2_config(monkeypatch, tmp_path, world, real_pins, arms):
    import functools
    b, c8, o4 = arms
    rng = np.random.default_rng(11)
    lab_c8 = _labels(c8, rng)
    sign = np.where(np.arange(world["n"]) % 4 % 2 == 0, 1.0, -1.0)[:, None]                  # +0.1 on logs 0 and 2, -0.1 on logs 1 and 3: the pooled difference is exactly 0 and its CI straddles 0
    lab_o4 = _labels(o4, rng, pdms=lab_c8["pdms"] + 0.1 * sign, sub=lab_c8["sub"])
    ch = dose_chain(tmp_path)
    bd = ch.bundle_dir("C1")
    os.makedirs(bd, exist_ok=True)
    os.makedirs(ch.cfg.work, exist_ok=True)
    np.savez(os.path.join(bd, "bundle.npz"), **world["bundle_d"])
    mp, lp, c1m, c1l = (str(tmp_path / n) for n in ("m.npz", "l.npz", "c1m.npz", "c1l.npz"))
    np.savez(mp, meta=np.array("{}"), **o4)
    np.savez(lp, **lab_o4)
    np.savez(c1m, meta=np.array("{}"), **c8)
    np.savez(c1l, **lab_c8)
    ch.st["done"] = list(DC.DOSE_STAGES)
    ch.c1_bundle_sha = "b" * 64
    monkeypatch.setattr(DC, "analyse_dose", functools.partial(DC.analyse_dose, B=200, with_concordance=False))
    c1 = {**C1READ, "result_sha256": "r" * 64, "merged": c1m, "labels": c1l}
    ch.run_analysis(mp, lp, real_pins, c1)
    res = json.load(open(ch.cfg.d("RESULT_DOSE.json")))
    out = ch.out.getvalue()
    k = res["c2_config"]
    want = DC.c2_config(C1READ, {"outcome": res["outcome"], "safety_points": res["safety_points"], "moved_share": res["moved_share"]}, res["contrast_O4_minus_C8"]["pooled"]["lo"])
    assert k == json.loads(json.dumps(want))                                               # wired from the chain's own numbers
    ci = res["contrast_O4_minus_C8"]["pooled"]
    assert ci["lo"] < 0.0 < ci["hi"] and k["contrast_lower_ci_gt_0"] is False             # the LOWER bound is what enters row 3, not the upper one
    assert re.search(rf"^ZZDOSE_C2CONFIG {k['config'] or 'NONE'} row {k['row']}: ", out, re.M) and re.search(r"^ZZDOSE_NEXT ", out, re.M) and re.search(r"^ZZDOSE_DONE ", out, re.M)
    assert res["c1_outcome"] == "O3_NULL"
