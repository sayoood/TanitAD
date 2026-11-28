"""Tests of read_dscale_pres.py and podread_dscale_pres.sh (card PREREG_V8F_PROOF_ADD126). Synthetic data only: tiny worlds, no real state or result file is opened.
Evidence class: MEASURED (synthetic).

What is synthetic and what is injected: the files the controls read (done files, state records, batch logs, train logs, precheck records, the pytest output file, the previous card's result file with its
input_hashes) are written to disk in the formats the imported readers load. The per-probe read of the banked chain (tokprobe_read.read_arm) is replaced by `read_probe`, a function that returns the same
(R, out, ctrl) triple from synthetic outputs; R["M4"] is computed by the CHAIN'S OWN function (leadsup_read_n1.auroc_row) on the synthetic presence logits, so the identity control compares against
the chain, not against a copy. The pack and context build is replaced the same way through `build_ctx`. The imported readers are byte-unchanged (asserted after the tests).

Run: PYCAP_MB=3000 bash F:/TanitAD-scratch/v8proof/ops/pycap.sh <python> -m pytest test_read_dscale_pres.py -q -p no:cacheprovider --basetemp <folder>
Child processes run with cwd = PRES_TEST_CWD (the folder of dscale_lead_read.py on the pod) or the temp folder, and carry the environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
EQP_DIR = Path(os.environ.get("EQP_READ_DIR") or (HERE.parent.parent / "add125" / "read"))
OLD_DIR = Path(os.environ.get("D2X_READ_DIR") or (HERE.parent.parent / "add123" / "read"))
for _p in (str(HERE), str(EQP_DIR), str(OLD_DIR)):
    sys.path.insert(0, _p)
import numpy as np
import pytest
import read_dscale_pres as PR
import test_read_dscale_eqp as TE

T2 = TE.T2
D, N, RP, S0F = PR.D, PR.N, PR.RP, T2.S0F
SEEDS = PR.SEEDS
SCRIPT = HERE / "podread_dscale_pres.sh"
EQP_FILE = Path(PR.EQP_PATH)
READER_FILE = Path(PR.X.__file__) if getattr(PR.X, "__file__", None) else Path(TE.READER_FILE)
SHAS_AT_IMPORT = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (EQP_FILE, Path(TE.READER_FILE))}
B = 400
must_fail, _assert, wjson = T2.must_fail, T2._assert, T2.wjson


def child_cwd(tmp):
    return os.environ.get("PRES_TEST_CWD") or str(tmp)


# ------------------------------------------------------------------------------------------------ the synthetic world
def make_ctx(n_ep=48, per=12, p_lead=0.55, seed=3):
    ctx = T2.make_ctx(n_ep=n_ep, per=per)
    n = len(ctx["G15"])
    r = np.random.default_rng(seed)
    ctx["pres0"] = (r.random(n) < p_lead).astype(float)
    ctx["ep_all"], ctx["n_all"], ctx["C_all"] = ctx["ep10"].copy(), ctx["n10"], ctx["C10"].copy()
    return ctx


def logit(ctx, label, mu, salt=0):
    r = np.random.default_rng(zlib.crc32((label + str(salt)).encode()))
    sign = 2.0 * ctx["pres0"] - 1.0
    return mu * sign + r.normal(size=len(sign))


def make_scores(ctx, variant):
    S = {}
    sign = 2.0 * ctx["pres0"] - 1.0
    for label, sp in PR.EQ.probe_specs().items():
        S[label] = logit(ctx, label, 0.0 if sp["sh"] else 0.5)
    for k in SEEDS:
        S[PR.lab("U", k)] = logit(ctx, PR.lab("U", k), 0.55)
        S[PR.lab("Q", k)] = logit(ctx, PR.lab("Q", k), 0.9)
        c = S[PR.lab("C", k)]
        if variant == "scales":
            S[PR.lab("E", k)] = logit(ctx, PR.lab("E", k), 0.95)
        elif variant == "small":
            S[PR.lab("E", k)] = c + 0.03 * sign
        elif variant == "worse":
            S[PR.lab("E", k)] = logit(ctx, PR.lab("E", k), 0.15)
        else:
            S[PR.lab("E", k)] = c.copy()
    return S


def fake_reader(w):
    ctx = w["ctx"]
    y = ctx["G15"][:, S0F]

    def read_probe(label, spec, files, c, a, cache):
        p, s = w["preds"][label], w["scores"][label]
        head = N.fid_one(y, p, ctx["valid"], ctx["ep10"], ctx["n10"], ctx["C10"])
        m4 = N.auroc_row(ctx["pres0"], s, ctx["ep_all"], ctx["n_all"], ctx["C_all"])
        m4 = {**m4, "auroc": m4["auroc"] + w["m4_shift"].get(label, 0.0)}
        return {"read": True, "M2": {"head": head}, "M4": m4}, {"gap_m": p, "presence_logit": s}, {}
    return read_probe


def build_world(tmp, variant="scales", ctx=None):
    ctx = ctx or make_ctx()
    w = TE.build_world(tmp, "gap", ctx=ctx)
    w["scores"], w["m4_shift"] = make_scores(ctx, variant), {}
    w["eqp_result"] = Path(tmp) / "DSCALE_EQP_RESULT.json"
    write_eqp_result(w)
    return w


def argv(w, mode, out=None, before=None):
    return TE.argv(w, mode, out=out, before=before) + ["--eqp-result", str(w["eqp_result"])]


def write_eqp_result(w, mutate=None):
    a = PR.parse(argv(w, "controls"))
    h = PR.EQ.input_hashes(argparse_ns(a), PR.EQ.probe_specs())
    if mutate:
        mutate(h)
    wjson(w["eqp_result"], {"input_hashes": h, "name": "stand-in", "other": "never read"})


def argparse_ns(a):
    import argparse
    return argparse.Namespace(**{**vars(a), "test_out": None})


@pytest.fixture
def run(monkeypatch, capsys):
    def go(w, mode, **kw):
        monkeypatch.setattr(PR.EQ, "build_ctx", lambda a: w["ctx"])
        monkeypatch.setattr(PR.EQ, "read_probe", fake_reader(w))
        ep = Path(w["eqp_result"])
        monkeypatch.setattr(PR, "EQP_RESULT_SHA_PREFIX", hashlib.sha256(ep.read_bytes()).hexdigest()[:16] if ep.exists() else "0" * 16)
        capsys.readouterr()
        rc = PR.main(argv(w, mode, **kw))
        return rc, capsys.readouterr().out
    return go


def controls_json(w):
    return json.loads(Path(PR.X.latest_path(str(w["out"]), PR.CONTROLS_FILE)).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ the estimator
def brute_auc(y, s, w):
    num = den = 0.0
    for i in range(len(y)):
        for j in range(len(y)):
            if y[i] == 1 and y[j] == 0:
                ww = w[i] * w[j]
                den += ww
                num += ww * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
    return num / den


def test_weighted_auroc_matches_a_brute_force_count_with_ties_counted_half():
    r = np.random.default_rng(1)
    y = (r.random(60) < 0.5).astype(float)
    s = np.round(r.normal(size=60) + 0.7 * y, 1)                    # many ties
    ep = np.repeat(np.arange(6), 10)
    C = r.multinomial(6, np.full(6, 1 / 6), size=5).astype(float)
    got = PR.auc_weighted(ep, y, s, C)
    for b in range(5):
        assert abs(got[b] - brute_auc(y, s, C[b][ep])) < 1e-12
    assert PR.n_tied(ep, y, s) > 0
    must_fail(lambda: _assert(abs(got[0] - brute_auc(y, s + 0.0, C[1][ep])) < 1e-12))


def test_point_equals_the_chains_rank_form_when_there_are_no_ties_and_differs_when_there_are():
    ctx = make_ctx()
    s = logit(ctx, "x", 0.7)
    chain = RP.auc_pooled(ctx["pres0"], s)
    assert abs(PR.auc_point(ctx["ep_all"], ctx["pres0"], s) - chain) < 1e-12
    st = np.where(ctx["pres0"] == 1, 1.0, 0.0)
    st[np.flatnonzero(ctx["pres0"] == 1)[:3]] = 0.5            # three positives and three negatives share one score
    st[np.flatnonzero(ctx["pres0"] == 0)[:3]] = 0.5
    assert PR.n_tied(ctx["ep_all"], ctx["pres0"], st) > 0
    assert abs(PR.auc_point(ctx["ep_all"], ctx["pres0"], st) - RP.auc_pooled(ctx["pres0"], st)) > 1e-6      # the chain breaks ties by order, this tool counts them half


def test_integer_episode_weights_equal_the_expanded_data():
    r = np.random.default_rng(4)
    y = (r.random(40) < 0.5).astype(float)
    s = r.normal(size=40) + y
    ep = np.repeat(np.arange(4), 10)
    c = np.array([[2.0, 0.0, 1.0, 1.0]])
    got = PR.auc_weighted(ep, y, s, c)[0]
    idx = np.concatenate([np.flatnonzero(ep == e).repeat(int(c[0][e])) for e in range(4)])
    assert abs(got - PR.auc_point(np.arange(len(idx)), y[idx], s[idx])) < 1e-12


def test_estimator_forms_on_the_context_p2():
    ctx = make_ctx()
    r = PR.p2_estimator(ctx)
    assert r["pass"] is True and all(v is True for k, v in r.items() if k != "pass"), r
    y, ep, n, C = PR.pres_arrays(ctx)
    s = logit(ctx, "s", 0.8)
    assert PR.auc_point(ep, y, y) == 1.0
    p, v = PR.boot_of(ctx, {"presence_logit": s})
    d = D.two_level([p - p] * 4, np.array([v - v] * 4), 0, "shared")
    assert d["mean"] == 0.0 and d["ci95"] == [0.0, 0.0]
    perm = np.random.default_rng(1).permutation(len(s))
    q = RP.ci(PR.auc_weighted(ep, y, s[perm], C))
    assert q[0] <= 0.5 <= q[1]
    q2 = RP.ci(PR.auc_weighted(ep, y, s, C))
    assert not (q2[0] <= 0.5 <= q2[1])                            # a probe with signal does not include 0.5: the permutation form is not vacuous


def test_p2_fails_when_the_estimator_has_a_planted_defect(monkeypatch):
    ctx = make_ctx()
    real = PR.auc_weighted
    monkeypatch.setattr(PR, "auc_weighted", lambda ep, y, s, C, chunk=100: real(ep, y, s, C, chunk) * 0.99)
    r = PR.p2_estimator(ctx)
    assert r["pass"] is False and r["labels_as_scores_read_1"] is False
    monkeypatch.setattr(PR, "auc_weighted", lambda ep, y, s, C, chunk=100: np.repeat(real(ep, y, s, np.ones((1, C.shape[1])))[:1], len(C)))   # ignores the draw weights: a zero-width interval
    r = PR.p2_estimator(ctx)
    assert r["pass"] is False and r["permuted_scores_interval_includes_0.5"] is False


# ------------------------------------------------------------------------------------------------ controls
def test_clean_world_passes_every_control_and_prints_one_line_each(tmp_path, run):
    w = build_world(tmp_path)
    rc, out = run(w, "controls")
    assert rc == 0, out
    lines = out.strip().splitlines()
    assert lines[:6] == ["ctl p%d PASS" % i for i in range(6)] and lines[-1] == "ZZ DSCALE-PRES CONTROLS ALL-PASS"
    cj = controls_json(w)
    assert cj["all_pass"] is True and set(cj["controls"]) == set(PR.CONTROL_NAMES) and all(isinstance(v["pass"], bool) for v in cj["controls"].values())
    assert cj["controls"]["p1"]["max_abs_difference"] <= 1e-9 and cj["controls"]["p3"]["counts"]["episodes"] == 48


def _mut_p0_hash(w):
    write_eqp_result(w, lambda h: h.__setitem__(next(k for k in h if k.endswith("TK16_s2_eval.npz.json")), "0" * 64))


def _mut_p0_extra_inputs(w):
    write_eqp_result(w, lambda h: [h.__setitem__("extra_%d" % i, "1" * 64) for i in range(3)])


def _mut_p0_no_key(w):
    wjson(w["eqp_result"], {"name": "no input hashes"})


def _mut_p0_missing_file(w):
    w["eqp_result"] = Path(w["eqp_result"]).with_name("not_there.json")


def _mut_p0_input_changed_after(w):
    p = w["new_run"] / "DSCALE_2X_DONE.txt"
    p.write_text(p.read_text(encoding="utf-8") + "ckpt extra 00\n", encoding="utf-8")


def _mut_p1_shift(w):
    w["m4_shift"]["P16E_s1"] = 1e-6


def _mut_p3_few_leads(w):
    ctx = w["ctx"]
    ctx["pres0"] = np.where(np.arange(len(ctx["pres0"])) < 30, 1.0, 0.0)


def _mut_p3_few_episodes(w):
    ctx = w["ctx"]
    ctx["pres0"] = np.where(ctx["ep_all"] < 15, ctx["pres0"], np.nan)          # labels exist in 15 episodes only


def _mut_p4_failed_test(w):
    w["test_out"].write_text("...F\n1 failed, 3 passed in 1s\n", encoding="utf-8")


def _mut_p4_newer(w):
    os.utime(str(w["test_out"]), (time.time() + 3600, time.time() + 3600))


def _mut_p5_other_ids(w):
    T2.write_state(w["new_in"], "runQ", "TK16_s1", 61568, TE.sha_for("e", "TK16_s1"), ids=["other%03d" % i for i in range(30)])


def _mut_p5_forty_episodes(w):
    w["ctx"] = make_ctx(n_ep=40, per=14)


CASES = [("p0", _mut_p0_hash), ("p0", _mut_p0_extra_inputs), ("p0", _mut_p0_no_key), ("p0", _mut_p0_missing_file), ("p0", _mut_p0_input_changed_after), ("p1", _mut_p1_shift),
         ("p3", _mut_p3_few_leads), ("p3", _mut_p3_few_episodes), ("p4", _mut_p4_failed_test), ("p4", _mut_p4_newer), ("p5", _mut_p5_other_ids), ("p5", _mut_p5_forty_episodes)]


@pytest.mark.parametrize("name,mut", CASES, ids=[m.__name__ for _, m in CASES])
def test_each_control_fails_on_one_planted_fault_exit_2_and_names_it(tmp_path, run, monkeypatch, name, mut):
    w = build_world(tmp_path)
    if mut is _mut_p0_input_changed_after:
        mut(w)
    elif mut in (_mut_p5_forty_episodes,):
        mut(w)
        w["preds"] = TE.make_preds(w["ctx"], "gap")
        w["scores"] = make_scores(w["ctx"], "scales")
        write_eqp_result(w)
    else:
        mut(w)
        if mut is not _mut_p0_missing_file and name != "p0":
            pass
    rc, out = run(w, "controls")
    assert rc == 2, out
    assert re.search(r"^ctl %s FAIL " % name, out, re.M) and "ZZ DSCALE-PRES CONTROLS FAILED" in out
    cj = controls_json(w)
    assert cj["controls"][name]["pass"] is False and cj["all_pass"] is False and name in cj["failed"]


def test_p2_fault_in_the_estimator_fails_the_controls(tmp_path, run, monkeypatch):
    w = build_world(tmp_path)
    real = PR.auc_weighted
    calls = {"n": 0}

    def biased(ep, y, s, C, chunk=100):
        calls["n"] += 1
        return real(ep, y, s, C, chunk) * 0.99
    monkeypatch.setattr(PR, "auc_weighted", biased)
    rc, out = run(w, "controls")
    assert rc == 2 and re.search(r"^ctl p2 FAIL ", out, re.M)


def test_a_wrong_result_file_hash_prefix_fails_p0(tmp_path, run, monkeypatch):
    w = build_world(tmp_path)
    rc, out = run(w, "controls")
    assert rc == 0
    monkeypatch.setattr(PR.EQ, "build_ctx", lambda a: w["ctx"])
    monkeypatch.setattr(PR.EQ, "read_probe", fake_reader(w))
    monkeypatch.setattr(PR, "EQP_RESULT_SHA_PREFIX", "927b06ff7e4a69e9")
    rc = PR.main(argv(w, "controls"))
    assert rc == 2


def test_p1_reads_the_chains_own_m4_of_all_18_probes_and_a_probe_not_read_fails_it(tmp_path, run):
    w = build_world(tmp_path)
    ctx = w["ctx"]
    specs = PR.EQ.probe_specs()
    reader = fake_reader(w)
    P = {}
    for label, sp in specs.items():
        R, out, c = reader(label, sp, {}, ctx, None, None)
        P[label] = {"R": R, "out": out, "ctrl": c, "files": {}, "read": True}
    r = PR.p1_identity(ctx, P)
    assert r["pass"] is True and r["probes_read"] == 18 and r["max_abs_difference"] <= 1e-9
    P["P16E_s0"]["read"] = False
    assert PR.p1_identity(ctx, P)["pass"] is False
    P["P16E_s0"]["read"] = True
    P["P16E_s0"]["R"] = {**P["P16E_s0"]["R"], "M4": {"auroc": P["P16E_s0"]["R"]["M4"]["auroc"] + 1e-6}}
    assert PR.p1_identity(ctx, P)["pass"] is False


def test_controls_mode_writes_and_prints_no_measure_and_no_name(tmp_path, run):
    w = build_world(tmp_path)
    rc, out = run(w, "controls")
    assert rc == 0
    text = Path(PR.X.latest_path(str(w["out"]), PR.CONTROLS_FILE)).read_text(encoding="utf-8")
    assert all(re.match(r"^(ctl p\d (PASS|FAIL .*)|beside p[13] .*|ZZ DSCALE-PRES .*)$", ln) for ln in out.strip().splitlines())
    for bad in ("LEAD-PRES", "SCALES", '"A"', "per_seed", "ci95", "two_level", "share", "below_half", "descriptive"):
        assert bad not in text and bad not in out, bad
    ctx = w["ctx"]
    outs = {l: {"presence_logit": w["scores"][l]} for l in ("P16E_s0", "P16C_s0")}
    for l in outs:
        pt = repr(PR.auc_point(ctx["ep_all"], ctx["pres0"], outs[l]["presence_logit"]))[:8]
        assert pt not in text and pt not in out
    assert not any(f.name.startswith(("DSCALE_PRES_RESULT", "DSCALE_PRES_VERDICT")) for f in w["out"].iterdir())
    must_fail(lambda: _assert("ci95" in text))


def test_a_second_controls_call_writes_a_new_file_and_overwrites_nothing(tmp_path, run):
    w = build_world(tmp_path)
    run(w, "controls")
    first = (w["out"] / PR.CONTROLS_FILE).read_bytes()
    run(w, "controls")
    assert (w["out"] / PR.CONTROLS_FILE).read_bytes() == first and (w["out"] / "DSCALE_PRES_CONTROLS_2.json").exists()


# ------------------------------------------------------------------------------------------------ names
def test_names_refuses_without_a_passing_controls_file_after_an_input_change_and_on_a_failed_controls_file(tmp_path, run):
    w = build_world(tmp_path)
    rc, out = run(w, "names")
    assert rc == 3 and "REFUSED" in out and not (w["out"] / PR.RESULT_FILE).exists()
    assert run(w, "controls")[0] == 0
    p = w["new_run"] / "runQ" / "r1_batches.json"
    p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")
    rc, out = run(w, "names")
    assert rc == 3 and "REFUSED" in out
    w2 = build_world(tmp_path / "second")
    _mut_p4_failed_test(w2)
    assert run(w2, "controls")[0] == 2
    rc, out = run(w2, "names")
    assert rc == 3 and "ZZ DSCALE-PRES NAME" not in out and not any(f.name.startswith("DSCALE_PRES_RESULT") for f in w2["out"].iterdir())


@pytest.mark.parametrize("variant,expected", [("scales", "LEAD-PRES-EQP-SCALES"), ("small", "LEAD-PRES-EQP-SMALL"), ("worse", "LEAD-PRES-EQP-WORSE"), ("flat", "LEAD-PRES-EQP-FLAT")])
def test_each_of_the_four_names_is_reached_by_planted_data(tmp_path, run, variant, expected):
    w = build_world(tmp_path, variant)
    assert run(w, "controls")[0] == 0
    rc, out = run(w, "names")
    assert rc == 0, out
    assert out.strip().splitlines()[-1] == "ZZ DSCALE-PRES NAME " + expected
    res = json.loads((w["out"] / PR.RESULT_FILE).read_text(encoding="utf-8"))
    assert res["names"]["name"] == expected and {"A", "beside", "probes", "counts"} <= set(res["measures"]) and "input_hashes" in res
    A = res["measures"]["A"]
    assert len(A["per_seed"]) == 4 and set(res["measures"]["beside"]) >= {"Q_minus_C", "U_minus_C", "per_probe_descriptive"} and len(res["probes"]) == 18
    v = (w["out"] / PR.VERDICT_FILE).read_text(encoding="utf-8")
    assert PR.TIER_LINE in v and all(x in v for x in PR.FAMILY_LINES) and expected in v and "## Limits" in v and "P16E_s0" in v
    if variant == "scales":
        assert A["ci95"][0] > 0 and A["mean"] >= 0.02 and A["n_pos"] >= 3
    if variant == "small":
        assert A["ci95"][0] > 0 and 0 < A["mean"] < 0.02 and A["n_pos"] >= 3
    if variant == "worse":
        assert A["ci95"][1] < 0 and A["n_neg"] >= 3
    if variant == "flat":
        assert A["mean"] == 0.0 and A["ci95"] == [0.0, 0.0]


def test_decide_clauses_individually():
    mk = lambda mean, lo, hi, npos, nneg: {"mean": mean, "ci95": [lo, hi], "n_pos": npos, "n_neg": nneg}
    assert PR.decide(mk(0.03, 0.01, 0.05, 4, 0))["name"] == PR.NAMES[0]
    assert PR.decide(mk(0.03, 0.01, 0.05, 2, 2))["name"] == PR.NAMES[3]                  # interval above 0 but only two seeds above 0
    assert PR.decide(mk(0.019, 0.001, 0.05, 3, 1))["name"] == PR.NAMES[1]
    assert PR.decide(mk(0.020, 0.001, 0.05, 3, 1))["name"] == PR.NAMES[0]
    assert PR.decide(mk(0.03, -0.001, 0.05, 4, 0))["name"] == PR.NAMES[3]                # interval includes 0
    assert PR.decide(mk(-0.03, -0.05, -0.01, 0, 4))["name"] == PR.NAMES[2]
    assert PR.decide(mk(-0.03, -0.05, -0.01, 1, 3))["name"] == PR.NAMES[2]
    assert PR.decide(mk(-0.03, -0.05, -0.01, 2, 2))["name"] == PR.NAMES[3]
    must_fail(lambda: _assert(PR.decide(mk(0.019, 0.001, 0.05, 3, 1))["name"] == PR.NAMES[0]))


def test_descriptive_shares_on_a_planted_probe():
    ctx = make_ctx()
    n = len(ctx["pres0"])
    g = ctx["G15"][:, S0F]
    s = np.where(ctx["pres0"] == 1, 5.0, -5.0)
    s = np.where((ctx["pres0"] == 1) & (g < 20), -1.0, s)                    # every lead window with a gap below 20 m is scored below 0.5
    s = np.where((ctx["pres0"] == 0) & (np.arange(n) % 2 == 0), 2.0, s)      # half of the no-lead windows are scored at or above 0.5
    d = PR.descriptive(ctx, {"presence_logit": s})
    assert d["by_gap"]["gap_0_20_m"]["share_below_half"] in (1.0, None) and d["by_gap"]["gap_40_plus_m"]["share_below_half"] == 0.0 and d["by_gap"]["gap_20_40_m"]["share_below_half"] == 0.0
    nolead = np.flatnonzero(ctx["pres0"] == 0)
    assert abs(d["share_no_lead_at_or_above_half"] - float(np.mean(nolead % 2 == 0))) < 1e-12
    lead = ctx["pres0"] == 1
    assert d["n_with_lead"] == int(lead.sum()) and abs(d["share_lead_below_half_all"] - float(((g < 20) & lead).sum() / lead.sum())) < 1e-12
    assert sum(b["n"] for b in d["by_gap"].values()) + d["lead_windows_without_a_finite_gap_label"] == d["n_with_lead"]
    must_fail(lambda: _assert(d["by_gap"]["gap_40_plus_m"]["share_below_half"] == 1.0))


# ------------------------------------------------------------------------------------------------ the imported readers, the driver, the words
def test_a_changed_copy_of_the_imported_reader_is_refused_at_import(tmp_path):
    bad = tmp_path / "read_dscale_eqp.py"
    bad.write_bytes(EQP_FILE.read_bytes() + b"\n# one more line\n")
    code = "import sys; sys.path.insert(0, %r); import read_dscale_pres" % str(HERE)
    cwd = child_cwd(tmp_path)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=cwd, env={**os.environ, "EQP_READER": str(bad)})
    assert r.returncode != 0 and "REFUSED" in r.stderr
    ok = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=cwd, env={**os.environ, "EQP_READER": str(EQP_FILE)})
    assert ok.returncode == 0, ok.stderr[-400:]


def test_driver_text_never_removes_moves_or_kills_and_refuses_bad_calls(tmp_path):
    txt = SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
    for word in ("rm ", "rm\t", "mv ", "kill", "rmdir", "unlink", "ssh", "scp"):
        assert word not in code, word
    must_fail(lambda: _assert("rm " in code))
    bash = shutil.which("bash")
    assert bash
    d = tmp_path / "drv"
    d.mkdir()
    shutil.copy(str(SCRIPT), str(d / "podread_dscale_pres.sh"))
    shutil.copy(str(HERE / "read_dscale_pres.py"), str(d / "read_dscale_pres.py"))
    sp = str(d / "podread_dscale_pres.sh")
    env = {**os.environ, "PY": sys.executable, "DSCALE_LEAD_KIT_DIR": str(tmp_path / "nokit")}
    cwd = child_cwd(tmp_path)

    def call(*args):
        return subprocess.run([bash, sp, *args], capture_output=True, text=True, encoding="utf-8", env=env, cwd=cwd)
    r = call("bogus")
    assert r.returncode == 3 and "STOP" in r.stdout
    r = call("controls")
    assert r.returncode == 3 and "NO-READ-ARGS-FILE" in r.stdout
    none = tmp_path / "none"
    out = tmp_path / "outdir"
    (d / "read_args.txt").write_text("--new-run-dir\n%s\n--new-done\n%s\n" % (none.as_posix(), (none / "DONE.txt").as_posix()), encoding="utf-8")
    r = call("controls")
    assert r.returncode == 3 and "NO --out-dir" in r.stdout
    (d / "read_args.txt").write_text("--new-done\n%s\n--out-dir\n%s\n" % ((none / "DONE.txt").as_posix(), out.as_posix()), encoding="utf-8")
    r = call("controls")
    assert r.returncode == 3 and "NO-DONE-MARKER" in r.stdout and not out.exists()                  # nothing was created before the done file was found
    none.mkdir()
    (none / "DONE.txt").write_text("x\n", encoding="utf-8")
    r = call("names")
    assert r.returncode == 3 and "NO-BANKED-READER" in r.stdout and out.is_dir()
    assert "_${n}.out" in txt and "ZZ POD-PRES $MODE DONE rc $rc" in txt and "--test-before" in txt and "read_args.txt" in txt


def test_none_of_the_banned_words_occur_in_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("corre", "ction"), ("with", "drawn"), ("no lo", "nger"), ("closed", " loop")]
    for f in ("read_dscale_pres.py", "podread_dscale_pres.sh", "test_read_dscale_pres.py", "DSCALE_PRES_READ_DEFS.md"):
        p = HERE / f
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8").lower()
        for a, b in parts:
            assert (a + b) not in t, (f, a + b)


def test_the_imported_readers_are_byte_unchanged_after_the_tests():
    for f in (EQP_FILE, Path(TE.READER_FILE)):
        assert hashlib.sha256(f.read_bytes()).hexdigest() == SHAS_AT_IMPORT[str(f)]
    assert SHAS_AT_IMPORT[str(EQP_FILE)].startswith(PR.EQP_SHA_PREFIX)
