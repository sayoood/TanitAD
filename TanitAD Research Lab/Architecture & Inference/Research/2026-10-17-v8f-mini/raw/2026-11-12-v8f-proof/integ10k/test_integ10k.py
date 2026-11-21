"""test_integ10k.py -- tests of the tools of card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE). numpy only (dev box: pytest; pod: minipytest, CPU).
Files under test are taken from the folder of this file (a mutation run copies the folder and edits one file)."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.environ.get("DISTRULE_DIR") or next((c for c in (os.path.join(HERE, "..", "distrule"), os.path.join(HERE, "distrule"), HERE) if os.path.isfile(os.path.join(c, "dist_rule.py"))), "")
os.environ["DISTRULE_DIR"] = DIST
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import make_tok_probe_lr as GEN  # noqa: E402
import qmap_fit as QF  # noqa: E402
import read_i10k as RI  # noqa: E402
import reader_controls as RC  # noqa: E402
import tok_reader as TR  # noqa: E402
import v8f_t1_eval_reader_rule as WR  # noqa: E402

DR = WR.R          # dist_rule


# ------------------------------------------------------------------ wrapper: the feed, identity, one hand-computed rule case
def test_feed_forms_hand_values():
    p = np.array([0.9, 0.4, 0.7, 0.7])
    g = np.array([30.0, 12.0, 10.0, 50.0])
    ok = np.array([True, True, False, True])
    xs, ys = np.array([10.0, 50.0]), np.array([5.0, 35.0])           # Q(10)=5, Q(30)=20, Q(50)=35
    pres, used = WR.feed("RR", p, g, ok, xs, ys)
    assert pres.tolist() == [True, False, False, True]               # presence: p >= 0.5 and a full history
    assert np.allclose(used, [20.0, 6.5, 5.0, 35.0])
    pres_id, used_id = WR.feed("RRid", p, g, ok)
    assert (used_id == 1e9).all() and pres_id.tolist() == pres.tolist()
    _, used_0 = WR.feed("RR0", p, g, ok)
    assert (used_0 == 0.0).all()


def test_feed_min_is_g_when_the_map_would_lie_above():
    # a planted map above the identity (not a valid card map): g_used = min(g, Q(g)) must stay at g
    xs, ys = np.array([10.0, 20.0]), np.array([15.0, 30.0])
    _, used = WR.feed("RR", np.array([0.9, 0.9]), np.array([12.0, 18.0]), np.array([True, True]), xs, ys)
    assert np.allclose(used, [12.0, 18.0])


def _plan(v=10.0, n=30):
    return np.stack([np.arange(1, n + 1) * v * 0.2, np.zeros(n)], 1)[None].astype(np.float32)


def test_RRid_equals_R0_bit_for_bit_on_a_tiny_input():
    plan = _plan()
    pres, used = WR.feed("RRid", np.array([0.9]), np.array([20.0]), np.array([True]))
    out = DR.distance_rule(plan, np.array([10.0]), pres, used, used + 0.0, np.array([0.0]), m=10, floor_lifts=False)
    assert out.plan is plan and not out.active.any() and (out.plan == plan).all()       # the plan object itself: nothing was applied


def test_hand_computed_rule_case_stopped_lead_20_m():
    # ego 10 m/s straight (2 m per 0.2 s step), lead stopped 20 m ahead, no closing speed, D = 3, tau = 1.2, b_max = 6, floor form cap-only.
    # cap a_c(i) = 20 - max(3, 1.2 * v'(i-1)) = 8 while v' = 10; a_w(i) = 2 i, so steps 1-4 are unchanged (a' = 2 i, a'(4) = 8);
    # step 5: a_f = 8 + max(0, 10 - 6 * 0.2) * 0.2 = 9.76, a' = min(a_w = 10, max(a_f, a_c)) = 9.76.
    plan = _plan()
    out = DR.distance_rule(plan, np.array([10.0]), np.array([True]), np.array([20.0]), np.array([20.0]), np.array([0.0]), m=10, floor_lifts=False)
    assert out.active[0] and abs(out.a_new[0, 3] - 8.0) < 1e-9 and abs(out.a_new[0, 4] - 9.76) < 1e-9
    assert np.allclose(out.plan[0, :4], plan[0, :4]) and abs(float(out.plan[0, 4, 0]) - 9.76) < 1e-5 and float(out.plan[0, 4, 1]) == 0.0


# ------------------------------------------------------------------ the reader
def planted_weights(n_tok=3, W=2, hidden=8, fdim=4, hist=5, seed=0, const=None):
    r = np.random.default_rng(seed)
    z = lambda *s: np.zeros(s, np.float32)      # noqa: E731
    n = lambda *s: r.normal(0, 0.3, s).astype(np.float32)   # noqa: E731
    w = {"tok.0.weight": n(hidden, 1024) * 0.05, "tok.0.bias": n(hidden), "tok.2.weight": n(W, hidden), "tok.2.bias": n(W), "head.frame.weight": n(fdim, n_tok * W), "head.frame.bias": n(fdim),
         "head.mlp.0.weight": n(8, hist * fdim), "head.mlp.0.bias": n(8), "head.mlp.2.weight": n(3, 8), "head.mlp.2.bias": n(3)}
    if const is not None:
        w = {k: np.zeros_like(v) for k, v in w.items()}
        w["head.mlp.2.bias"] = np.array(const, np.float32)
    return w


def test_planted_constant_reader_reads_presence_half_and_gap_25():
    rd = TR.TokReader(planted_weights(const=[0.0, 0.5, 0.0]), n_tok=3)
    p, g = rd.read(np.random.default_rng(1).normal(size=(4, 5, 3, 1024)).astype(np.float32))
    assert np.allclose(p, 0.5) and np.allclose(g, 25.0)


def test_batch_path_equals_the_per_clip_path_and_the_reader_is_causal():
    rd = TR.TokReader(planted_weights(seed=3), n_tok=3)
    x = np.random.default_rng(2).normal(size=(14, 3, 1024)).astype(np.float32)
    p, g, ok = TR.read_clip(rd, x)
    assert (~ok[:4]).all() and ok[4:].all() and np.isnan(p[:4]).all()
    pb, gb = rd.read(x[None, 6:11])                   # frames 6..10 -> the reading at t = 10
    assert abs(pb[0] - p[10]) < 1e-5 and abs(gb[0] - g[10]) < 1e-3
    assert RC.causal_unchanged(rd, x, 9)              # k7: noise after the tick changes nothing
    assert not np.array_equal(p[8:], TR.read_clip(rd, x[::-1].copy())[0][8:])      # the reading does depend on the frames


def test_k7_detects_a_reader_that_looks_at_a_future_frame():
    rd = TR.TokReader(planted_weights(seed=3), n_tok=3)
    x = np.random.default_rng(2).normal(size=(14, 3, 1024)).astype(np.float32)

    def leaky(reader, tokens):
        p, g, ok = TR.read_clip(reader, tokens)
        p2, g2, _ = TR.read_clip(reader, np.concatenate([tokens[1:], tokens[-1:]]))      # the reading of t + 1 shifted back to t
        return 0.5 * (p + p2), 0.5 * (g + g2), ok
    assert not RC.causal_unchanged(rd, x, 9, read=leaky)


def test_reader_refuses_a_wrong_token_shape():
    rd = TR.TokReader(planted_weights(const=[0.0, 0.0, 0.0]), n_tok=3)
    try:
        rd.read(np.zeros((1, 4, 3, 1024), np.float32))
    except SystemExit as e:
        assert "REFUSED" in str(e)
    else:
        raise AssertionError("no refusal")


# ------------------------------------------------------------------ the held-out rule
def test_heldout_rule_is_the_card_rule_and_the_parts_do_not_overlap():
    ids = ["clip-%05d-%s" % (i, hashlib.md5(str(i).encode()).hexdigest()[:12]) for i in range(3000)]
    for c in ids:
        v = int(hashlib.sha256(c.encode("utf-8")).hexdigest()[:8], 16)
        assert TR.is_heldout(c) == (v % 10 == 0) and TR.is_fit(c) == (v % 20 == 0)
        assert not (TR.is_fit(c) and TR.is_check(c)) and (not TR.is_fit(c) or TR.is_heldout(c))
    s = TR.split_clips(ids)
    assert len(s["train"]) + len(set(s["fit"]) | set(s["check"])) == len(ids)
    assert set(s["train"]).isdisjoint(s["fit"]) and set(s["train"]).isdisjoint(s["check"]) and set(s["fit"]).isdisjoint(s["check"])
    assert 0.07 < (len(ids) - len(s["train"])) / len(ids) < 0.13 and 0.35 < len(s["fit"]) / (len(ids) - len(s["train"])) < 0.65


def test_derived_probe_tool_carries_the_split_and_its_heldout_function_equals_the_reader_rule():
    text = GEN.build_text()
    assert "clips = [c for c in clips if not is_heldout(c)]" in text and "D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)" in text
    m = re.search(r"def heldout_value\(clip_id\):.*?def is_heldout\(clip_id\):\n    return heldout_value\(clip_id\) % 10 == 0\n", text, re.S)
    assert m
    ns = {"hashlib": hashlib}
    exec(m.group(0), ns)
    for c in ["a", "b", "0123", "clip-77", "x" * 40]:
        assert ns["is_heldout"](c) == TR.is_heldout(c)
    # only the four named edits differ from the original
    src = open(GEN.SRC, encoding="utf-8", newline="").read()
    assert len(GEN.EDITS) == 4 and all(src.count(o) == 1 for o, _ in GEN.EDITS)


# ------------------------------------------------------------------ the map
def _rows(n_clips, per, rng, shift=0.0, fit_every=2):
    cols = {k: [] for k in ("ci", "g", "lab_pres", "lab_gap", "fit")}
    for c in range(n_clips):
        g = rng.uniform(3, 55, per)
        lab = np.maximum(0.3, g * rng.lognormal(0.0, 0.25, per) + shift)       # label gap scatters around the read
        cols["ci"].append(np.full(per, c))
        cols["g"].append(g)
        cols["lab_pres"].append(np.ones(per, bool))
        cols["lab_gap"].append(lab)
        cols["fit"].append(np.full(per, c % fit_every == 0))
    return {k: np.concatenate(v) for k, v in cols.items()}


def test_qmap_fit_is_monotone_covers_in_band_and_a_shifted_check_set_fails_the_band():
    rng = np.random.default_rng(5)
    R = _rows(60, 120, rng)
    rec = QF.fit_from_rows(R, "T")
    xs, ys = np.array(rec["xs"]), np.array(rec["ys"])
    assert (np.diff(ys) >= 0).all() and (ys <= xs + 1e-9).all() and rec["unit_check"]["pass"] and rec["fit_check_clips_disjoint"]
    assert rec["coverage"]["pass"] and 0.15 <= rec["coverage"]["share_below_Q"] <= 0.25
    # the same fit rows, check rows whose labels sit 8 m lower: coverage leaves the band
    R2 = {k: np.array(v, copy=True) for k, v in R.items()}
    chk = ~R2["fit"]
    R2["lab_gap"][chk] = R2["lab_gap"][chk] - 8.0
    assert not QF.fit_from_rows(R2, "T")["coverage"]["pass"]


# ------------------------------------------------------------------ the controls of the reader (k1, k6)
def test_auroc_gap_error_and_leak_count_hand_cases():
    assert RC.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0 and RC.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert RC.auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5
    assert abs(RC.auroc([0.9, 0.4, 0.6, 0.2], [1, 1, 0, 0]) - 0.75) < 1e-12
    assert abs(RC.gap_error([10.0, 100.0], [12.0, 38.0]) - 2.0) < 1e-12              # the read is clipped at 40 m
    assert RC.leak_count(["a", "b", "c"], ["x", "y"]) == 0 and RC.leak_count(["a", "b", "x"], ["x", "y"]) == 1      # a planted shared id fails the check


# ------------------------------------------------------------------ the read: controls, names in both directions
def cell(d_s, D=None, lo=None, hi=None, ex=True):
    D = float(np.mean(d_s)) if D is None else D
    return {"d_s": list(d_s), "D": D, "A_lo": D - 0.02 if lo is None else lo, "A_hi": D + 0.02 if hi is None else hi, "A_excludes_0": ex}


def good_inp():
    pairs = {RI.pk("I10RR", "I10R0"): {"cells": {"U": cell([-0.05, -0.04], ex=True)}, "Q3": {"name": "NO-COST", "cost_families": []}, "beside": {"ade6_m|all": cell([0.01, 0.02])}},
             RI.pk("I10RLb", "I10R0"): {"cells": {"U": cell([-0.08, -0.08])}},
             RI.pk("I10RR", "I10RLb"): {"cells": {"U": cell([0.03, 0.04]), "C1": cell([0.0, 0.0])}},
             RI.pk("I10RRx", "I10R0"): {"cells": {"U": cell([-0.03, -0.03])}},
             RI.pk("I10RR0", "I10R0"): {"cells": {"U": cell([0.1, 0.1])}, "beside": {"ade6_m|all": cell([1.5, 1.4])}}}
    q3pairs = {k: {"Q3": v.get("Q3", {}), "beside": v.get("beside", {})} for k, v in pairs.items()}
    rc = {k: {"k1": {"gap_mae_m_clipped_40": 4.0, "auroc": 0.9, "label_match_share": 0.998}, "k6": {"leak_count": 0, "planted_shared_id_detected": True}, "k7": {"all_unchanged": True}} for k in (0, 1)}
    return {"hw": {"pairs": pairs, "controls": {"ALL_PASS": True}, "counts": {"n_windows_total": 2399, "n_lead_windows": 642}}, "q3": {"pairs": q3pairs, "controls": {"ALL_PASS": True}},
            "rc": rc, "dump_eq": {0: (141, 0), 1: (141, 0)}, "banked": {"tolerance_m": 0.02, "UE_s0": {"fde_last_m": 9.0714, "ade_dense_m": 2.7348}, "UE_s1": {"fde_last_m": 9.2, "ade_dense_m": 2.8}},
            "r0json": {0: {"intervals": {"metrics": {"fde_last_m": 9.0801, "ade_dense_m": 2.7301}}}, 1: {"intervals": {"metrics": {"fde_last_m": 9.2, "ade_dense_m": 2.8}}}},
            "ckpt_sha": {0: "2fd4b747532f55b2" + "0" * 48, 1: "f13df8ce026eed31" + "0" * 48},
            "qmap": {k: {"coverage": {"share_below_Q": 0.2}, "fit_check_clips_disjoint": True} for k in (0, 1)}, "train": {k: {"steps": 19164, "p5_all_pass": True} for k in (0, 1)}}


def test_all_controls_pass_and_names_read_cut_reach_and_price():
    res = RI.evaluate(good_inp())
    assert res["controls_all_pass"] and res["names"]["cut"] == "INTEG10K-CUTS-UNSAFE"
    assert res["names"]["share_reached_per_seed"] == [0.625, 0.5] and res["names"]["reaches_label_rule"] == "INTEG10K-REACHES-LABEL-RULE" and res["names"]["price"] == "INTEG10K-NO-COST"
    assert "T1-kin (kinematic rollout on the model's own actions" in res["tier"] and res["strategic"][1].startswith("STRATEGIC along the rule-driven rollout (T1): NOT COMPUTED")


def test_names_flip_the_other_way_when_the_data_say_so():
    i = good_inp()
    i["hw"]["pairs"][RI.pk("I10RR", "I10R0")]["cells"]["U"] = cell([-0.05, 0.01], D=-0.02, ex=True)       # one seed above 0
    assert RI.evaluate(i)["names"]["cut"] == "INTEG10K-NO-CUT"
    i = good_inp()
    i["hw"]["pairs"][RI.pk("I10RR", "I10R0")]["cells"]["U"] = cell([-0.05, -0.04], ex=False)                # interval touches 0
    assert RI.evaluate(i)["names"]["cut"] == "INTEG10K-NO-CUT"
    i = good_inp()
    i["hw"]["pairs"][RI.pk("I10RRx", "I10R0")]["cells"]["U"] = cell([-0.03, 0.02])                           # one crossed run above R0
    assert RI.evaluate(i)["names"]["cut"] == "INTEG10K-NO-CUT"
    i = good_inp()
    i["hw"]["pairs"][RI.pk("I10RR", "I10R0")]["cells"]["U"] = cell([-0.05, -0.03])                           # share 0.625 and 0.375
    n = RI.evaluate(i)["names"]
    assert n["cut"] == "INTEG10K-CUTS-UNSAFE" and n["reaches_label_rule"] == "not named"
    i = good_inp()
    i["q3"]["pairs"][RI.pk("I10RR", "I10R0")]["Q3"] = {"name": "COSTS-LATERAL", "cost_families": ["LATERAL"]}
    assert RI.evaluate(i)["names"]["price"] == "INTEG10K-COSTS-LATERAL"


def test_a_failed_or_not_evaluable_control_withholds_every_name():
    for mut in ("k4", "k2", "k3", "k9", "k7", "k6", "k1", "k8", "k10", "k5null", "k5far"):
        i = good_inp()
        if mut == "k4":
            i["hw"]["pairs"][RI.pk("I10RLb", "I10R0")]["cells"]["U"] = cell([-0.08, 0.01], D=-0.03)
        if mut == "k2":
            i["dump_eq"][1] = (141, 3)
        if mut == "k3":
            i["q3"]["pairs"][RI.pk("I10RR0", "I10R0")]["beside"]["ade6_m|all"] = cell([-0.1, 0.1], D=0.0, ex=False)
        if mut == "k9":
            i["qmap"][0]["coverage"]["share_below_Q"] = 0.31
        if mut == "k7":
            i["rc"][0]["k7"]["all_unchanged"] = False
        if mut == "k6":
            i["rc"][1]["k6"]["leak_count"] = 1
        if mut == "k1":
            i["rc"][0]["k1"]["auroc"] = 0.80
        if mut == "k8":
            i["hw"]["counts"]["n_windows_total"] = 2398
        if mut == "k10":
            i["q3"]["controls"]["ALL_PASS"] = False
        if mut == "k5null":
            i["banked"]["UE_s1"] = None
        if mut == "k5far":
            i["r0json"][0]["intervals"]["metrics"]["fde_last_m"] = 9.2
        res = RI.evaluate(i)
        assert not res["controls_all_pass"] and res["names"] is None and "beside" not in res, mut
    # the not evaluable k5 (UE_s1 not banked yet) is a None, not a pass
    i = good_inp()
    i["banked"]["UE_s1"] = None
    assert RI.evaluate(i)["controls"]["k5_model_is_the_banked_one"]["PASS"] is None


def test_dumps_equal_counts_a_differing_file(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for d in (a, b):
        np.savez(str(d / "ep000.npz"), x=np.arange(5.0))
    np.savez(str(a / "ep001.npz"), x=np.arange(3.0))
    np.savez(str(b / "ep001.npz"), x=np.arange(3.0) + 1e-12)
    assert RI.dumps_equal(str(a), str(b)) == (2, 1)


# ------------------------------------------------------------------ the pod queue (dry run)
def _dry():
    bash = shutil.which("bash")
    if not bash:
        return None
    env = dict(os.environ, DRY="1")
    p = subprocess.run([bash, os.path.join(HERE, "pod_integ10k_queue.sh")], capture_output=True, env=env, timeout=120)
    return p.stdout.decode("utf-8", "replace").splitlines()


def test_dry_run_lists_every_command_in_order_with_a_slot_wait_and_the_right_reader_per_run():
    lines = _dry()
    if lines is None:
        return
    cmds = [l for l in lines if l.startswith("CMD:")]
    assert len(cmds) == 20 and sum(l.startswith("SLOTWAIT") for l in lines) == 13
    ro = {}
    for l in cmds:
        m = re.search(r"--form (\S+).*?--out-dir \S+/t1/(\S+) ", l)
        if m:
            ro[m.group(2)] = (m.group(1), re.findall(r"TK16_s(\d)/ckpt_last", l), re.findall(r"QMAP_LR16_s(\d)", l), re.findall(r"UE_s(\d)/A", l))
    assert len(ro) == 12
    for k in "01":
        assert ro["I10R0_s" + k][0] == "R0" and ro["I10R0_s" + k][1] == []
        for nm in ("I10RR", "I10RRid", "I10RR0"):
            assert ro["%s_s%s" % (nm, k)][1] == [k] and ro["%s_s%s" % (nm, k)][3] == [k]
        assert ro["I10RR_s" + k][2] == [k] and ro["I10RLb_s" + k][1] == []
    assert ro["I10RRx_s0"][1] == ["1"] and ro["I10RRx_s0"][3] == ["0"] and ro["I10RRx_s1"][1] == ["0"] and ro["I10RRx_s1"][3] == ["1"]
    assert [i for i, l in enumerate(lines) if "tok_probe_lr.py train" in l][0] < [i for i, l in enumerate(lines) if "tok_reader.py extract" in l][0] < [i for i, l in enumerate(lines) if "I10RR_s0" in l and l.startswith("CMD")][0]


def test_queue_has_no_pgrep_pkill_or_rm_and_uses_noclobber_start_marker():
    txt = open(os.path.join(HERE, "pod_integ10k_queue.sh"), encoding="utf-8").read()
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"\b(pgrep|pkill)\b|(^|[;&|\s])rm\s", code) and "set -C" in code and "INTEG10K_DONE.txt" in code and "INTEG10K_STOP.txt" in code
    assert "-lt \"$MAXGPU\"" in code and "sleep 45" in code and "-gt 21600" in code


def test_the_derived_probe_file_is_the_generator_output():
    assert open(os.path.join(HERE, "tok_probe_lr.py"), encoding="utf-8", newline="").read() == GEN.build_text()


def test_reader_planner_builds_after_the_wrapper_swaps_the_class_in_without_recursion():
    saved = (WR.W.RulePlanner, dict(WR.W.MODE_CFG))
    try:
        WR.W.MODE_CFG["mix"] = ("mix", {})
        WR.X.STATE["label"] = object()
        WR.W.RulePlanner = WR.ReaderPlanner          # as main() does
        rp = WR.ReaderPlanner(object(), None, "mix", 10, None, label=None, log=None, floor="cap-only")
        assert rp.source == "mix" and rp.label is WR.X.STATE["label"] and rp.m == 10
    finally:
        WR.W.RulePlanner = saved[0]
        WR.W.MODE_CFG.clear()
        WR.W.MODE_CFG.update(saved[1])
