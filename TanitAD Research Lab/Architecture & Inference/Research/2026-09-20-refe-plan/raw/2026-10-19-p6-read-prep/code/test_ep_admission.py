"""Synthetic CPU-only tests for ep_admission.py (no real bank, no GPU, no real label).  Run: python -m unittest test_ep_admission  (or pytest).

Every decision function has a RED mutation that must break the branch table; the planted cases (an over-predicting candidate must FAIL A1, a calibrated one must ADMIT, an over-confident safety row must
FAIL A2, P2 as the candidate must read exactly 0) are also run through a MUTATED copy of the module that must disagree.  Source: DRAFT_PREREG_EP_ADMISSION.md (Master Mind, 2026-10-19).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ep_admission as A                                                                  # noqa: E402

L, P, F = A.libs()
NC, DAC, EP, TTC, CMF, DDC = 0, 1, 2, 3, 4, 5
N_LOGS_SYN = 30
N_SYN = 3000


def sha_file(p):
    return A.sha256_file(p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def top8_of(z6):
    p = sigmoid(z6)
    agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0
    return np.argsort(-agg, axis=-1, kind="stable")[..., :8].astype(np.int16)


def make_case(over=0.0, frac=0.6, seed=7, noise=0.02, nc_unsafe=None, dac_unsafe=None, ttc_unsafe=None, n=N_SYN, n_logs=N_LOGS_SYN):
    """(z6_p2, z6_c, lab, log_idx): P2 picks slot 0 everywhere; the candidate picks slot 1 on `frac` of the sets (its EP row at slot 1 is 1.5 on those, -2.0 elsewhere; slot 0 is 0.0).
    Labels at the two picks: EP = candidate sigma (minus `over` at the candidate's changed pick) + noise; NC / DAC / TTC / C = Bernoulli(candidate sigma) unless a *_unsafe probability is given."""
    arr, ep, li, ch = L.synthetic_bank(N=n, n_logs=n_logs, seed=seed, frac_changed=frac)
    rng = np.random.default_rng(seed + 100)
    z6_p2 = arr["z6"].copy()                                    # slot 0 EP 2.0, slot 1 EP -2.0  -> P2 picks slot 0
    z6_p2[:, 2:, EP] = np.random.default_rng(seed + 200).normal(0.0, 1.0, z6_p2[:, 2:, EP].shape)         # the other 62 slots: an EP row on the label's scale (so a shuffled pick is well scaled)
    z6_c = z6_p2.copy()
    z6_c[:, 0, EP] = 0.0
    z6_c[:, 1, EP] = np.where(ch, 1.5, -2.0)
    lab = arr["lab"].astype(np.float32).copy()
    lab[:, 2:, EP] = np.clip(sigmoid(z6_c[:, 2:, EP]) + np.random.default_rng(seed + 300).normal(0.0, noise, lab[:, 2:, EP].shape), 0.0, 1.0)
    sc = sigmoid(z6_c)
    idx = np.arange(n)
    for slot in (0, 1):
        for col, unsafe in ((NC, nc_unsafe), (DAC, dac_unsafe), (TTC, ttc_unsafe), (CMF, None)):
            p_safe = sc[:, slot, col] if unsafe is None else np.full(n, 1.0 - unsafe)
            lab[:, slot, col] = (rng.random(n) < p_safe).astype(np.float32)
    lab[:, 0, EP] = np.clip(sc[:, 0, EP] + rng.normal(0, noise, n), 0, 1)
    lab[:, 1, EP] = np.clip(sc[:, 1, EP] - np.where(ch, over, 0.0) + rng.normal(0, noise, n), 0, 1)
    lab[~ch, 1, EP] = lab[~ch, 0, EP]                           # not used (nobody picks slot 1 there); keep deterministic
    return z6_p2, z6_c, lab, li, ch


def write_ranked(path, n_logs=N_LOGS_SYN, pct=0.1):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("rank\tlog\tkey_sha256\tfold\tfirst_fraction\tn_fit_tokens\tn_fit_sets\n")
        for r in range(n_logs):
            f.write(f"{r}\tlog{r:02d}\t{hashlib.sha256(str(r).encode()).hexdigest()}\t{r % 5}\t{pct}\t1\t1\n")
    return path


def write_bank(d, z6, lab, li, *, ckpt="synthetic_ckpt", n_units=3, shuffle_seed=None, label_order=None, drop_manifest=False, tamper=False, dup_key=False, skip_key=False):
    """S0-format unit files + MANIFEST.sha256.json.  Sets are split over n_units; `shuffle_seed` permutes the row order (a candidate bank in a different order)."""
    os.makedirs(d, exist_ok=True)
    n = len(z6)
    order = np.arange(n) if shuffle_seed is None else np.random.default_rng(shuffle_seed).permutation(n)
    toks = np.array([f"t{i:05d}" for i in range(n)])
    logs = np.array([f"log{j:02d}" for j in li])
    steps = np.arange(n, dtype=np.int32)
    ranks = np.zeros(n, np.int32)
    ck = np.full(n, 10075, np.int64)
    sel = order if not skip_key else order[:-1]
    chunks = np.array_split(sel, n_units)
    files = {}
    for u, ix in enumerate(chunks):
        if dup_key and u == 0:
            ix = np.concatenate([ix, ix[:1]])
        name = f"U{u:02d}.sets.npz"
        p = os.path.join(d, name)
        np.savez(p, log=logs[ix], token=toks[ix], step=steps[ix], rank=ranks[ix], ckpt_step=ck[ix], feat=np.zeros((len(ix), 64, 4), np.float16), logits=z6[ix].astype(np.float32), top8=top8_of(z6[ix]),
                 labels6=lab[ix].astype(np.float32), label_order=np.array(label_order or A.LABEL_ORDER), rule=np.array("navsim_v1"), ckpt=np.array(ckpt))
        files[f"work/bank/{name}"] = {"path": p.replace("\\", "/"), "sha256": sha_file(p), "bytes": os.path.getsize(p)}
    if not drop_manifest:
        with open(os.path.join(d, "MANIFEST.sha256.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"what": "synthetic", "files": files}, f)
    if tamper:
        with open(os.path.join(d, "U00.sets.npz"), "ab") as f:
            f.write(b"x")
    return d


def stats_for(li, n_logs=N_LOGS_SYN, B=500):
    return P.Stats(np.asarray(li, np.int64), F.boot_draws(n_logs, B, 11), F, n_logs=n_logs)


def analyze(case, S=None):
    z6_p2, z6_c, lab, li, ch = case
    S = S or stats_for(li)
    return A.analyze_candidate(z6_p2, z6_c, lab, S), S


class TestDecisionFunctions(unittest.TestCase):
    def test_branch_table_passes_unmutated(self):
        self.assertEqual(A.branch_table_failures(A), [])

    def test_every_decision_function_has_a_red_mutation(self):
        for old, new in A.DECISION_MUTATIONS:
            m = A.mutated_copy(old, new)
            self.assertTrue(len(m.branch_table_failures(m)) > 0, f"mutation not caught: {old!r} -> {new!r}")

    def test_mutation_report_consistent(self):
        r = A.mutation_report()
        self.assertEqual(r["unmutated_failures"], [])
        self.assertEqual(r["n_caught"], r["n_mutations"])
        self.assertEqual(r["uncaught"], [])
        self.assertGreaterEqual(r["n_mutations"], 30)

    def test_constants_are_the_drafts(self):
        self.assertEqual((A.A1_BAR, A.A2_LO, A.A2_HI, A.B_UNSAFE_UB), (0.02, 0.8, 1.25, 0.001))
        self.assertEqual((A.BOOT_B, A.BOOT_SEED, A.N_LOGS), (10000, 20260927, 117))
        self.assertEqual((L.BOOT_B, L.BOOT_SEED, L.N_LOGS), (A.BOOT_B, A.BOOT_SEED, A.N_LOGS))
        self.assertEqual(A.A2_HEADS, ("NC", "DAC", "TTC"))
        self.assertEqual((A.NULL_C_REPS, A.NULL_E_REPS, A.NULL_VOID_FIRES), (L.NULL_C_REPS, L.NULL_E_REPS, L.NULL_VOID_FIRES))
        self.assertEqual(A.PLANT, L.PLANT)


class TestPins(unittest.TestCase):
    def test_tampered_epreliab_lib_refused(self):
        with tempfile.TemporaryDirectory() as d:
            dst = os.path.join(d, "epreliab_lib.py")
            shutil.copyfile(L.__file__, dst)
            with open(dst, "ab") as f:
                f.write(b"\n# tampered\n")
            saved = sys.modules.pop("epreliab_lib", None)
            try:
                with self.assertRaises(A.AdmissionError):
                    A.load_epreliab(dst)
            finally:
                if saved is not None:
                    sys.modules["epreliab_lib"] = saved

    def test_pinned_lib_is_the_one_in_use(self):
        self.assertEqual(sha_file(L.__file__), A.EPRELIAB_LIB_SHA256)


class TestLoaderAndAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.rk = write_ranked(os.path.join(cls.tmp, "ranked.tsv"))
        cls.case = make_case()
        z6_p2, z6_c, lab, li, ch = cls.case
        cls.p2d = write_bank(os.path.join(cls.tmp, "p2"), z6_p2, lab, li, ckpt="p2")
        cls.cd = write_bank(os.path.join(cls.tmp, "cand"), z6_c, lab, li, ckpt="cand", shuffle_seed=5, n_units=4)
        cls.rows, _ = A.read_ranked(cls.rk)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_p2_bank_loads_back_exactly(self):
        z6_p2, z6_c, lab, li, ch = self.case
        b = A.load_bank_units(self.p2d, self.rows)
        self.assertTrue(np.array_equal(b["z6"], z6_p2.astype(np.float32)))
        self.assertTrue(np.array_equal(b["lab"], lab))
        self.assertTrue(np.array_equal(b["log_idx"], li))
        self.assertTrue(np.array_equal(b["fold"], li % 5))
        self.assertEqual(b["n_sets"], len(z6_p2))

    def test_candidate_in_another_order_is_aligned_by_key(self):
        z6_p2, z6_c, lab, li, ch = self.case
        p2 = A.load_bank_units(self.p2d, self.rows)
        c = A.load_bank_units(self.cd, self.rows)
        self.assertFalse(np.array_equal(c["z6"], z6_c.astype(np.float32)))                       # the file order really differs
        a = A.align_candidate(p2, c)
        self.assertTrue(np.array_equal(a["z6"], z6_c.astype(np.float32)))
        self.assertTrue(np.array_equal(a["lab"], p2["lab"]))
        self.assertTrue(a["checks"]["labels_bit_equal"])

    def test_labels_that_differ_are_refused(self):
        z6_p2, z6_c, lab, li, ch = self.case
        lab2 = lab.copy()
        lab2[3, 5, EP] = 0.123
        d = write_bank(os.path.join(self.tmp, "cand_badlab"), z6_c, lab2, li)
        with self.assertRaises(A.AdmissionError):
            A.align_candidate(A.load_bank_units(self.p2d, self.rows), A.load_bank_units(d, self.rows))

    def test_a_missing_set_is_refused(self):
        z6_p2, z6_c, lab, li, ch = self.case
        d = write_bank(os.path.join(self.tmp, "cand_skip"), z6_c, lab, li, skip_key=True)
        with self.assertRaises(A.AdmissionError):
            A.align_candidate(A.load_bank_units(self.p2d, self.rows), A.load_bank_units(d, self.rows))

    def test_a_duplicate_set_key_is_refused(self):
        z6_p2, z6_c, lab, li, ch = self.case
        d = write_bank(os.path.join(self.tmp, "cand_dup"), z6_c, lab, li, dup_key=True)
        with self.assertRaises(A.AdmissionError):
            A.load_bank_units(d, self.rows)

    def test_a_wrong_label_order_is_refused(self):
        z6_p2, z6_c, lab, li, ch = self.case
        d = write_bank(os.path.join(self.tmp, "cand_lo"), z6_c, lab, li, label_order=["dac", "nc", "ep", "ttc", "c", "ddc"])
        with self.assertRaises(A.AdmissionError):
            A.load_bank_units(d, self.rows)

    def test_a_tampered_unit_and_a_missing_manifest_are_refused(self):
        z6_p2, z6_c, lab, li, ch = self.case
        d = write_bank(os.path.join(self.tmp, "cand_tamper"), z6_c, lab, li, tamper=True)
        with self.assertRaises(A.AdmissionError):
            A.load_bank_units(d, self.rows)
        d2 = write_bank(os.path.join(self.tmp, "cand_nomanifest"), z6_c, lab, li, drop_manifest=True)
        with self.assertRaises(A.AdmissionError):
            A.load_bank_units(d2, self.rows)
        b = A.load_bank_units(d2, self.rows, require_manifest=False)                                # allowed only when asked for
        self.assertIsNone(b["manifest_sha256"])

    def test_a_stored_top8_that_is_not_the_frozen_rule_is_refused(self):
        z6_p2, z6_c, lab, li, ch = self.case
        d = write_bank(os.path.join(self.tmp, "cand_top8"), z6_c, lab, li, n_units=1)
        p = os.path.join(d, "U00.sets.npz")
        z = dict(np.load(p))
        z["top8"] = z["top8"][:, ::-1].copy()
        np.savez(p, **z)
        doc = json.load(open(os.path.join(d, "MANIFEST.sha256.json")))
        doc["files"]["work/bank/U00.sets.npz"]["sha256"] = sha_file(p)
        json.dump(doc, open(os.path.join(d, "MANIFEST.sha256.json"), "w"))
        with self.assertRaises(A.AdmissionError):
            A.load_bank_units(d, self.rows)

    def test_red_mutants_of_the_loader_checks_let_the_bad_bank_through(self):
        z6_p2, z6_c, lab, li, ch = self.case
        lab2 = lab.copy()
        lab2[3, 5, EP] = 0.123
        d = write_bank(os.path.join(self.tmp, "cand_badlab2"), z6_c, lab2, li)
        m = A.mutated_copy('if not np.array_equal(lab, p2["lab"]):', 'if False:')
        a = m.align_candidate(m.load_bank_units(self.p2d, self.rows), m.load_bank_units(d, self.rows))      # the mutant does not notice a different label source
        self.assertFalse(np.array_equal(a["lab"], lab))
        dt = write_bank(os.path.join(self.tmp, "cand_tamper2"), z6_c, lab, li, n_units=1)
        up = os.path.join(dt, "U00.sets.npz")
        z = dict(np.load(up))
        z["logits"] = (z["logits"] + 1.0).astype(np.float32)
        z["top8"] = top8_of(z["logits"])                                                                  # keep the stored top8 consistent so ONLY the sha256 check can notice
        np.savez(up, **z)                                                                                # a unit rewritten after the manifest was written
        with self.assertRaises(A.AdmissionError):
            A.load_bank_units(dt, self.rows)                                                             # the real loader refuses it on its sha256
        mt = A.mutated_copy('if sha != man["table"][n]:', 'if False:')
        b = mt.load_bank_units(dt, self.rows)                                                            # the mutant reads the rewritten logits without noticing
        self.assertFalse(np.array_equal(b["z6"], z6_c.astype(np.float32)))

    def test_the_ranked_table_fold_rule_and_pin(self):
        bad = os.path.join(self.tmp, "ranked_bad.tsv")
        txt = open(self.rk, encoding="utf-8").read().replace("\t1\t0.1\t", "\t2\t0.1\t", 1)
        open(bad, "w", encoding="utf-8", newline="\n").write(txt)
        with self.assertRaises(A.AdmissionError):
            A.read_ranked(bad)
        with self.assertRaises(A.AdmissionError):
            A.read_ranked(self.rk, expected_sha256="0" * 64)


class TestIdentity(unittest.TestCase):
    def test_p2_as_the_candidate_reads_exactly_zero(self):
        z6_p2, z6_c, lab, li, ch = make_case()
        S = stats_for(li)
        c = A.control_id(z6_p2, lab, S)
        self.assertTrue(c["pass"], c)
        self.assertEqual(c["n_changed"], 0)
        an = A.analyze_candidate(z6_p2, z6_p2, lab, S)
        for ci in (an["G"]["D_M"], an["G"]["delta"], an["A3"]["d_pdms_x100"], an["A3"]["d_nc_unsafe_rate"], an["A3"]["d_dac_unsafe_rate"]):
            self.assertEqual((ci["point"], ci["lo"], ci["hi"]), (0.0, 0.0, 0.0))
        self.assertEqual(A.a1_verdict(an["G"]["D_M"]), "ADMIT-A1")

    def test_red_mutant_of_c_id_fails(self):
        z6_p2, z6_c, lab, li, ch = make_case()
        S = stats_for(li)
        m = A.mutated_copy("an = analyze_candidate(z6_p2, z6_p2, lab, S)", "an = analyze_candidate(z6_p2, z6_p2[:, ::-1, :], lab, S)")
        self.assertFalse(m.control_id(z6_p2, lab, S)["pass"])

    def test_p2_through_the_full_run_admits_with_a_warning(self):
        z6_p2, z6_c, lab, li, ch = make_case()
        tmp = tempfile.mkdtemp()
        try:
            rows, _ = A.read_ranked(write_ranked(os.path.join(tmp, "r.tsv")))
            p2 = A.load_bank_units(write_bank(os.path.join(tmp, "p2"), z6_p2, lab, li), rows)
            res = A.run_admission(p2, {"P2": A.align_candidate(p2, p2)}, B=500, null_reps_e=0, log=lambda *_: None)
            self.assertTrue(res["controls_ok"], {k: v for k, v in res["controls"].items() if not v.get("pass")})
            self.assertTrue(res["verdict"]["label"].startswith("ADMIT"))
            self.assertTrue(any("IDENTICAL" in w for w in res["warnings"]))
            self.assertEqual(res["per_candidate"]["P2"]["A3"]["d_pdms_x100"]["point"], 0.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestPlantedA1(unittest.TestCase):
    def test_a_calibrated_candidate_is_admitted(self):
        an, S = analyze(make_case(over=0.0))
        self.assertEqual(A.a1_verdict(an["G"]["D_M"]), "ADMIT-A1", an["G"]["D_M"])

    def test_a_planted_overprediction_fails_a1(self):
        an, S = analyze(make_case(over=0.30))
        self.assertGreater(an["G"]["D_M"]["lo"], A.A1_BAR)
        self.assertEqual(A.a1_verdict(an["G"]["D_M"]), "FAIL-A1")

    def test_a_small_planted_overprediction_that_crosses_the_bar_fails(self):
        an, S = analyze(make_case(over=0.08))
        self.assertEqual(A.a1_verdict(an["G"]["D_M"]), "FAIL-A1", an["G"]["D_M"])

    def test_an_underpredicting_candidate_is_admitted(self):
        an, S = analyze(make_case(over=-0.20))
        self.assertLess(an["G"]["D_M"]["hi"], 0.0)
        self.assertEqual(A.a1_verdict(an["G"]["D_M"]), "ADMIT-A1")

    def test_the_planted_overprediction_flips_under_a_mutated_gate(self):
        an, S = analyze(make_case(over=0.30))
        m = A.mutated_copy('if ci["hi"] <= bar:\n        return "ADMIT-A1"', 'if ci["hi"] >= bar:\n        return "ADMIT-A1"')
        self.assertEqual(m.a1_verdict(an["G"]["D_M"]), "ADMIT-A1")                                    # the mutant gate admits the over-predictor: the test would catch it
        self.assertEqual(A.a1_verdict(an["G"]["D_M"]), "FAIL-A1")

    def test_d_m_equals_a_hand_computation(self):
        z6_p2, z6_c, lab, li, ch = make_case(over=0.2)
        S = stats_for(li)
        pk_p, pk_c = P.rule_pick(z6_p2), P.rule_pick(z6_c)
        sig = P.sigmoid64(z6_c[..., EP])
        y = lab[..., EP].astype(np.float64)
        idx = np.arange(len(pk_p))
        Mp = y[idx, pk_p].sum() / sig[idx, pk_p].sum()
        Me = y[idx, pk_c].sum() / sig[idx, pk_c].sum()
        an = A.analyze_candidate(z6_p2, z6_c, lab, S)
        self.assertAlmostEqual(an["G"]["D_M"]["point"], Mp - Me, places=12)
        self.assertAlmostEqual(an["G"]["delta"]["point"], float(((sig[idx, pk_c] - sig[idx, pk_p]) - (y[idx, pk_c] - y[idx, pk_p])).mean()), places=12)

    def test_the_a1_form_auto_uses_delta_only_on_scale(self):
        an, S = analyze(make_case(over=0.30))
        self.assertEqual(A.a1_statistic(an, A.a1_form(False, "auto")), an["G"]["D_M"])
        self.assertEqual(A.a1_statistic(an, A.a1_form(True, "auto")), an["G"]["delta"])
        self.assertEqual(A.a1_statistic(an, A.a1_form(True, "mass")), an["G"]["D_M"])


class TestPlantedA2(unittest.TestCase):
    def a2(self, **kw):
        an, S = analyze(make_case(**kw))
        return A.a2_verdict(an["A2_reliability"]), an

    def test_calibrated_safety_rows_are_admitted(self):
        v, an = self.a2()
        self.assertEqual(v, "ADMIT-A2", an["A2_reliability"])

    def test_an_overconfident_nc_row_fails_a2(self):
        v, an = self.a2(nc_unsafe=0.12)
        self.assertEqual(v, "FAIL-A2")
        self.assertGreater(an["A2_reliability"]["NC"]["lo"], A.A2_HI)
        self.assertTrue(A.a2_head_ok(an["A2_reliability"]["DAC"]) and A.a2_head_ok(an["A2_reliability"]["TTC"]))

    def test_an_overconfident_dac_row_fails_a2(self):
        v, an = self.a2(dac_unsafe=0.12)
        self.assertEqual(v, "FAIL-A2")
        self.assertGreater(an["A2_reliability"]["DAC"]["lo"], A.A2_HI)

    def test_an_overconfident_ttc_row_fails_a2(self):
        v, an = self.a2(ttc_unsafe=0.12)
        self.assertEqual(v, "FAIL-A2")
        self.assertGreater(an["A2_reliability"]["TTC"]["lo"], A.A2_HI)

    def test_an_overconservative_row_fails_a2_as_drafted(self):
        v, an = self.a2(ttc_unsafe=0.0)
        self.assertEqual(v, "FAIL-A2")
        self.assertLess(an["A2_reliability"]["TTC"]["hi"], A.A2_LO)

    def test_a_mutated_gate_misses_the_overconfident_row(self):
        v, an = self.a2(nc_unsafe=0.12)
        m = A.mutated_copy("if len(heads) > 0 and all((h in cis) and a2_head_ok(cis[h]) for h in heads):", "if len(heads) > 0 and any((h in cis) and a2_head_ok(cis[h]) for h in heads):")
        self.assertEqual(m.a2_verdict(an["A2_reliability"]), "ADMIT-A2")
        self.assertEqual(v, "FAIL-A2")

    def test_reliability_uses_the_deficit_form_at_the_candidates_own_picks(self):
        z6_p2, z6_c, lab, li, ch = make_case(nc_unsafe=0.12)
        S = stats_for(li)
        pk_c = P.rule_pick(z6_c)
        idx = np.arange(len(pk_c))
        obs = (1.0 - lab[idx, pk_c, NC].astype(np.float64)).sum()
        pred = (1.0 - sigmoid(z6_c[idx, pk_c, NC])).sum()
        an = A.analyze_candidate(z6_p2, z6_c, lab, S)
        self.assertAlmostEqual(an["A2_reliability"]["NC"]["point"], obs / pred, places=12)


class TestControls(unittest.TestCase):
    def test_reach_passes_and_is_not_run_without_changed_picks(self):
        z6_p2, z6_c, lab, li, ch = make_case()
        S = stats_for(li)
        r = A.control_reach(z6_p2, z6_c, lab, S)
        self.assertTrue(r["pass"] and r["status"] == "RUN", r)
        self.assertTrue(A.control_reach(z6_p2, z6_p2, lab, S)["status"].startswith("NOT RUN"))

    def test_power_sees_a_planted_overprediction_on_a_changing_candidate(self):
        z6_p2, z6_c, lab, li, ch = make_case(frac=0.6)
        S = stats_for(li)
        r = A.control_power(z6_p2, z6_c, lab, S)
        self.assertTrue(r["pass"], r)
        self.assertEqual((r["plus"]["a1"], r["minus"]["a1"], r["zero"]["a1"]), ("FAIL-A1", "ADMIT-A1", "ADMIT-A1"))
        self.assertTrue(r["zero_is_exactly_0_0"])

    def test_power_is_void_when_the_candidate_barely_changes_the_picks(self):
        z6_p2, z6_c, lab, li, ch = make_case(frac=0.004)
        S = stats_for(li)
        r = A.control_power(z6_p2, z6_c, lab, S)
        self.assertFalse(r["pass"], r)                                                               # +0.10 on ~12 picks cannot cross the bar: ADMIT would be blindness, not reliability
        self.assertIn("VOID by resolution", r["status"])

    def test_a_mutated_power_control_fails_on_a_good_candidate(self):
        z6_p2, z6_c, lab, li, ch = make_case(frac=0.6)
        S = stats_for(li)
        m = A.mutated_copy('row["plus"]["a1"] == "FAIL-A1"', 'row["plus"]["a1"] == "ADMIT-A1"')
        self.assertFalse(m.control_power(z6_p2, z6_c, lab, S)["pass"])

    def test_null_is_silent_on_a_structured_candidate(self):
        z6_p2, z6_c, lab, li, ch = make_case()
        S = stats_for(li)
        n = A.run_null(z6_p2, z6_c, lab, S, reps_e=20)
        self.assertEqual(n["status"], "RUN")
        self.assertFalse(n["void"], n["c_null"]["fires"])
        self.assertIn("e_null", n)

    def test_null_fires_when_the_bar_is_impossible(self):
        z6_p2, z6_c, lab, li, ch = make_case()
        S = stats_for(li)
        m = A.mutated_copy("A1_BAR = 0.02 ", "A1_BAR = -5.0 ")
        n = m.run_null(z6_p2, z6_c, lab, S, reps_e=0)
        self.assertTrue(n["void"] and n["c_null"]["fires"] == 5)

    def test_shuffle_keeps_the_coincidence_structure(self):
        z6_p2, z6_c, lab, li, ch = make_case()
        pk_p, pk_c = P.rule_pick(z6_p2), P.rule_pick(z6_c)
        a, b = L.shuffled_pair(pk_p, pk_c, 0)
        self.assertTrue(np.array_equal(a == b, pk_p == pk_c))


class TestPessimistic(unittest.TestCase):
    def test_lam_one_reproduces_the_candidates_own_picks(self):
        z6_p2, z6_c, lab, li, ch = make_case()
        S = stats_for(li)
        fold = li % 5
        b = A.pessimistic_picks(z6_p2, z6_c, lab, fold, S, lam_override=1.0)
        self.assertEqual(b["n_changed_vs_candidate"], 0)

    def test_lam_zero_ignores_the_ep_row(self):
        z6_p2, z6_c, lab, li, ch = make_case()
        S = stats_for(li)
        fold = li % 5
        b0 = A.pessimistic_picks(z6_p2, z6_c, lab, fold, S, lam_override=0.0)
        z6_x = z6_c.copy()
        z6_x[..., EP] = np.random.default_rng(1).normal(0, 3, z6_x[..., EP].shape)
        b1 = A.pessimistic_picks(z6_p2, z6_x, lab, fold, S, lam_override=0.0)
        self.assertTrue(np.array_equal(b0["picks"], b1["picks"]))

    def test_lam_is_the_out_of_fold_slope_of_the_training_folds_only(self):
        n, nl = 4000, 20
        rng = np.random.default_rng(3)
        li = rng.integers(0, nl, n)
        fold = li % 5
        z6 = rng.normal(0, 1, (n, 64, 6)).astype(np.float32)
        sig = sigmoid(z6[..., EP])
        slot = np.full(64, 0.5)
        lab = np.ones((n, 64, 6), np.float32)
        slope = np.where(fold == 0, 1.0, 0.3)[:, None]                                              # fold 0 is perfectly calibrated, the others have slope 0.3
        lab[..., EP] = np.clip(slot[None, :] + slope * (sig - slot[None, :]) + rng.normal(0, 1e-3, sig.shape), 0, 1)
        S = stats_for(li, nl)
        b = A.pessimistic_picks(z6, z6, lab, fold, S)
        self.assertAlmostEqual(b["lam_by_fold"][0], 0.3, delta=0.03)                                  # held-out fold 0 is scored with the slope of folds 1-4
        self.assertGreater(b["lam_by_fold"][1], 0.3)                                                  # fold 1 sees fold 0's slope-1 sets among its training folds
        m = A.mutated_copy("safe & tr[:, None]", "safe")
        bm = m.pessimistic_picks(z6, z6, lab, fold, S)
        self.assertGreater(bm["lam_by_fold"][0], b["lam_by_fold"][0] + 0.05)                          # using ALL sets leaks the held-out fold into lam: the mutant must differ

    def test_the_slot_prior_comes_from_the_training_folds_only(self):
        n, nl = 2000, 20
        rng = np.random.default_rng(4)
        li = rng.integers(0, nl, n)
        fold = li % 5
        z6 = np.zeros((n, 64, 6), np.float32)
        lab = np.ones((n, 64, 6), np.float32)
        lab[..., EP] = np.where(fold[:, None] == 0, 0.9, 0.2)                                         # fold 0 labels differ from folds 1-4
        S = stats_for(li, nl)
        b = A.pessimistic_picks(z6, z6, lab, fold, S, lam_override=0.0)
        # lam 0 -> EP_used = slot prior of the TRAINING folds: for held-out fold 0 that is 0.2, for held-out fold 1 it is a mix that includes fold 0
        zB = b["z6_B"]
        self.assertAlmostEqual(float(sigmoid(zB[fold == 0][0, 0, EP])), 0.2, places=6)
        self.assertGreater(float(sigmoid(zB[fold == 1][0, 0, EP])), 0.2 + 1e-3)

    def test_the_row_slot_source_is_selectable(self):
        z6_p2, z6_c, lab, li, ch = make_case()
        S = stats_for(li)
        fold = li % 5
        a = A.pessimistic_picks(z6_p2, z6_c, lab, fold, S, slot_source="label")
        b = A.pessimistic_picks(z6_p2, z6_c, lab, fold, S, slot_source="row")
        self.assertEqual(a["slot_source"], "label")
        self.assertEqual(b["slot_source"], "row")


class TestAnalyticOC(unittest.TestCase):
    def test_break_even_is_one_half(self):
        h = 0.0263
        oc = A.a1_analytic_oc(h, 0.02, truths=(0.02 - h,))
        self.assertAlmostEqual(oc["by_truth"][f"{0.02 - h:+g}"]["p_fail_a1"], 0.5, places=6)

    def test_monotone_in_the_truth_and_zero_halfwidth_degenerate(self):
        oc = A.a1_analytic_oc(0.0263)
        ps = [oc["by_truth"][k]["p_fail_a1"] for k in sorted(oc["by_truth"], key=float)]
        self.assertTrue(all(b >= a for a, b in zip(ps, ps[1:])))
        z = A.a1_analytic_oc(0.0, truths=(0.0, 0.05))
        self.assertEqual((z["by_truth"]["+0"]["p_fail_a1"], z["by_truth"]["+0.05"]["p_fail_a1"]), (0.0, 1.0))

    def test_the_ep_r1_anchor_says_a_calibrated_candidate_fails_about_two_in_three(self):
        oc = A.a1_analytic_oc(0.0263)
        self.assertAlmostEqual(oc["by_truth"]["+0"]["p_fail_a1"], 0.68, delta=0.02)


class TestSealing(unittest.TestCase):
    def test_prereg_must_be_hashed_and_name_the_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "PREREG_EP_ADMISSION_P6.md")
            sha_ck, sha_man = "a" * 64, "b" * 64
            open(p, "w", encoding="utf-8").write(f"EP-ADMISSION against candidate {sha_ck} bank {sha_man}\n")
            with self.assertRaises(A.AdmissionError):                                                # no sidecar
                A.check_prereg(p, sha_ck, [sha_man])
            open(os.path.join(d, "PREREG_EP_ADMISSION_P6.sha256"), "w").write(sha_file(p) + "  PREREG_EP_ADMISSION_P6.md\n")
            ok = A.check_prereg(p, sha_ck, [sha_man])
            self.assertEqual(ok["prereg_sha256"], sha_file(p))
            with self.assertRaises(A.AdmissionError):                                                # does not name a different candidate
                A.check_prereg(p, "c" * 64, [sha_man])
            with self.assertRaises(A.AdmissionError):                                                # does not name a different bank
                A.check_prereg(p, sha_ck, ["d" * 64])
            with open(p, "a", encoding="utf-8") as f:                                                # edited after sealing
                f.write("amended\n")
            with self.assertRaises(A.AdmissionError):
                A.check_prereg(p, sha_ck, [sha_man])


class TestEndToEndCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.rk = write_ranked(os.path.join(cls.tmp, "ranked.tsv"))
        good = make_case(over=0.0)
        bad = make_case(over=0.30)
        cls.p2_good = write_bank(os.path.join(cls.tmp, "p2_good"), good[0], good[2], good[3], ckpt="p2")
        cls.c_good = write_bank(os.path.join(cls.tmp, "c_good"), good[1], good[2], good[3], ckpt="cand", shuffle_seed=2)
        cls.p2_bad = write_bank(os.path.join(cls.tmp, "p2_bad"), bad[0], bad[2], bad[3], ckpt="p2")
        cls.c_bad = write_bank(os.path.join(cls.tmp, "c_bad"), bad[1], bad[2], bad[3], ckpt="cand", shuffle_seed=2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_cli(self, p2, c, out, *extra):
        return A.main(["run", "--p2-bank", p2, "--cand-bank", c, "--ranked-logs", self.rk, "--no-ranked-pin", "--enull-reps", "20", "--out", out, *extra])

    def test_a_calibrated_candidate_is_admitted_and_an_overpredictor_refused(self):
        out_g, out_b = os.path.join(self.tmp, "g.json"), os.path.join(self.tmp, "b.json")
        self.assertEqual(self.run_cli(self.p2_good, self.c_good, out_g, "--allow-unsealed"), 0)
        self.assertEqual(self.run_cli(self.p2_bad, self.c_bad, out_b, "--allow-unsealed"), 0)
        g, b = json.load(open(out_g)), json.load(open(out_b))
        self.assertTrue(g["controls_ok"] and b["controls_ok"], ({k: v for k, v in g["controls"].items() if not v.get("pass")}, {k: v for k, v in b["controls"].items() if not v.get("pass")}))
        self.assertTrue(g["verdict"]["label"].startswith("ADMIT"), g["verdict"])
        self.assertEqual(b["verdict"]["label"], "REFUSE-A1", b["verdict"])
        self.assertEqual(b["verdict"]["A2"], {"c_bad": "ADMIT-A2"})
        self.assertIn("NONE", g["authority"])
        self.assertIn("NO navtest look", b["verdict"]["next_arm"])
        self.assertIn("A1_resolution", g["per_candidate"]["c_good"])
        self.assertIn("B", g["per_candidate"]["c_good"])

    def test_a_run_without_a_registration_is_refused(self):
        with self.assertRaises(A.AdmissionError):
            self.run_cli(self.p2_good, self.c_good, os.path.join(self.tmp, "x.json"))

    def test_a_sealed_run_carries_authority(self):
        pre = os.path.join(self.tmp, "PREREG_SEALED.md")
        man = A.read_manifest(self.c_good)["sha256"]
        ck = "e" * 64
        open(pre, "w", encoding="utf-8").write(f"EP-ADMISSION sealed against {ck}; bank manifest {man}\n")
        open(os.path.join(self.tmp, "PREREG_SEALED.sha256"), "w").write(sha_file(pre) + "\n")
        out = os.path.join(self.tmp, "sealed.json")
        self.assertEqual(self.run_cli(self.p2_good, self.c_good, out, "--prereg", pre, "--candidate-ckpt-sha256", ck), 0)
        r = json.load(open(out))
        self.assertEqual(r["authority"], "SEALED REGISTRATION")
        self.assertEqual(r["seal"]["prereg_sha256"], sha_file(pre))

    def test_two_seeds_must_both_pass(self):
        out = os.path.join(self.tmp, "two.json")
        rc = A.main(["run", "--p2-bank", self.p2_good, "--cand-bank", self.c_good, "--cand-bank", self.c_good, "--cand-id", "s0", "--cand-id", "s1", "--ranked-logs", self.rk, "--no-ranked-pin",
                     "--allow-unsealed", "--enull-reps", "0", "--out", out])
        self.assertEqual(rc, 0)
        r = json.load(open(out))
        self.assertEqual(set(r["verdict"]["A1"]), {"s0", "s1"})



# =====================================================================================================================================================================================
# REVISION 1 (DRAFT_PREREG_EP_ADMISSION.md "REVISION 1", Master Mind 2026-10-19).  The 53 tests above are UNCHANGED and cover the DRAFT functions (which are untouched) and, through the new run-level
# default, the R1 rules end to end.  Everything below is new.
# =====================================================================================================================================================================================
def named_mutant(old: str, new: str):
    return A.mutated_copy(old, new)


A1R1_SRC = 'if ci["lo"] <= lo_bar and ci["point"] <= point_bar:\n        return "ADMIT-A1"'
A2R1_SRC = 'return bool(ci["lo"] <= bar)'
A2R1_HEADS_SRC = 'if len(heads) > 0 and all((h in cis) and a2_head_ok_r1(cis[h]) for h in heads):'
ANYSEED_SRC = 'return bool(len(verdicts) > 0 and all(v == admit for v in verdicts))'


class TestRevision1Constants(unittest.TestCase):
    def test_constants_defaults_and_the_draft_is_untouched(self):
        self.assertEqual((A.A1_R1_LO_BAR, A.A1_R1_POINT_BAR, A.A2_R1_RATIO_BAR), (0.0, 0.05, 1.0))
        self.assertEqual((A.DEFAULT_RULES, tuple(A.RULES)), ("r1", ("draft", "r1")))
        self.assertEqual((A.B_SLOT_SOURCE_R1, A.B_SLOT_SOURCE_DRAFT), ("row", "label"))
        self.assertEqual((A.A1_BAR, A.A2_LO, A.A2_HI, A.B_UNSAFE_UB), (0.02, 0.8, 1.25, 0.001))                  # the draft's constants did not move
        import inspect
        self.assertEqual(inspect.signature(A.run_admission).parameters["rules"].default, "r1")
        self.assertIsNone(inspect.signature(A.run_admission).parameters["slot_source"].default)                 # None = resolved by the rule set
        self.assertEqual(inspect.signature(A.pessimistic_picks).parameters["slot_source"].default, "label")     # the library-level function keeps the draft's default (reproducibility)

    def test_the_cli_defaults(self):
        pa = A.build_parser().parse_args(["run", "--p2-bank", "x", "--cand-bank", "y", "--ranked-logs", "z", "--out", "o"])
        self.assertEqual((pa.rules, pa.slot_source), ("r1", None))
        pb = A.build_parser().parse_args(["run", "--rules", "draft", "--p2-bank", "x", "--cand-bank", "y", "--ranked-logs", "z", "--out", "o"])
        self.assertEqual(pb.rules, "draft")
        pt = A.build_parser().parse_args(["selftest-real", "--p2-bank", "x", "--ranked-logs", "z", "--arrays-npz", "a", "--landed-result", "l", "--out", "o"])
        self.assertEqual(pt.rules, "r1")
        with self.assertRaises(SystemExit):
            A.build_parser().parse_args(["run", "--rules", "bogus", "--p2-bank", "x", "--cand-bank", "y", "--ranked-logs", "z", "--out", "o"])

    def test_the_r1_mutations_are_in_the_registry_and_all_caught(self):
        r1 = A.DECISION_MUTATIONS[35:]                                                  # the 35 draft mutations come first and are untouched; everything after is REVISION 1
        self.assertGreaterEqual(len(r1), 38)
        for old, new in r1:
            m = A.mutated_copy(old, new)
            self.assertTrue(len(m.branch_table_failures(m)) > 0, f"R1 mutation not caught: {old!r} -> {new!r}")

    def test_one_mutation_per_r1_conjunct_is_named_and_red(self):
        """The brief's list, each by name: every conjunct of A1-r1 and A2-r1, the any-seed quantifier, the connective, the NaN handling, the dispatch and both defaults."""
        named = {
            "A1 lower-CI conjunct dropped": (A1R1_SRC, 'if ci["point"] <= point_bar:\n        return "ADMIT-A1"'),
            "A1 point conjunct dropped": (A1R1_SRC, 'if ci["lo"] <= lo_bar:\n        return "ADMIT-A1"'),
            "A1 and -> or": (A1R1_SRC, 'if ci["lo"] <= lo_bar or ci["point"] <= point_bar:\n        return "ADMIT-A1"'),
            "A1 NaN fails open": (A1R1_SRC, 'if not (ci["lo"] > lo_bar or ci["point"] > point_bar):\n        return "ADMIT-A1"'),
            "A1 lower-CI edge opened": (A1R1_SRC, 'if ci["lo"] < lo_bar and ci["point"] <= point_bar:\n        return "ADMIT-A1"'),
            "A1 point edge opened": (A1R1_SRC, 'if ci["lo"] <= lo_bar and ci["point"] < point_bar:\n        return "ADMIT-A1"'),
            "A1 upper CI used for the lower": (A1R1_SRC, 'if ci["hi"] <= lo_bar and ci["point"] <= point_bar:\n        return "ADMIT-A1"'),
            "A2 lower CI replaced by the upper": (A2R1_SRC, 'return bool(ci["hi"] <= bar)'),
            "A2 lower CI replaced by the point": (A2R1_SRC, 'return bool(ci["point"] <= bar)'),
            "A2 made two-sided like the draft": (A2R1_SRC, 'return bool(ci["lo"] <= bar and ci["hi"] >= A2_LO)'),
            "A2 edge opened": (A2R1_SRC, 'return bool(ci["lo"] < bar)'),
            "A2 any head suffices": (A2R1_HEADS_SRC, 'if len(heads) > 0 and any((h in cis) and a2_head_ok_r1(cis[h]) for h in heads):'),
            "A2 missing head tolerated": (A2R1_HEADS_SRC, 'if len(heads) > 0 and all(a2_head_ok_r1(cis[h]) for h in heads if h in cis):'),
            "any-seed -> some seed": (ANYSEED_SRC, 'return bool(len(verdicts) > 0 and any(v == admit for v in verdicts))'),
            "any-seed empty list admits": (ANYSEED_SRC, 'return bool(all(v == admit for v in verdicts))'),
            "any-seed looks at the first seed only (A1)": ('any_seed_admits([a1_verdict_r1(c) for c in cis], "ADMIT-A1")', 'any_seed_admits([a1_verdict_r1(c) for c in cis[:1]], "ADMIT-A1")'),
            "any-seed looks at the last seed only (A2)": ('any_seed_admits([a2_verdict_r1(c) for c in cis_by_seed], "ADMIT-A2")', 'any_seed_admits([a2_verdict_r1(c) for c in cis_by_seed[-1:]], "ADMIT-A2")'),
            "dispatch: r1 returns the draft functions": ('    if rules == "r1":\n        return a1_verdict_r1, a2_verdict_r1', '    if rules == "r1":\n        return a1_verdict, a2_verdict'),
            "default rules is the draft": ('DEFAULT_RULES = "r1" ', 'DEFAULT_RULES = "draft" '),
            "B default slot source under r1 is label": ('B_SLOT_SOURCE_R1 = "row" ', 'B_SLOT_SOURCE_R1 = "label" '),
        }
        for name, (old, new) in named.items():
            m = A.mutated_copy(old, new)
            self.assertTrue(len(m.branch_table_failures(m)) > 0, f"mutation not caught: {name}")


class TestRevision1A1(unittest.TestCase):
    def d_m(self, **kw):
        an, S = analyze(make_case(**kw))
        return an["G"]["D_M"]

    def wide_case_ci(self):
        case = make_case(over=0.10, n=200, n_logs=4, noise=0.25, seed=12)
        return A.analyze_candidate(case[0], case[1], case[2], stats_for(case[3], 4, B=500))["G"]["D_M"]

    def test_a_calibrated_candidate_is_admitted_and_an_underpredictor_too(self):
        self.assertEqual(A.a1_verdict_r1(self.d_m(over=0.0)), "ADMIT-A1")
        ci = self.d_m(over=-0.20)
        self.assertEqual(A.a1_verdict_r1(ci), "ADMIT-A1")
        self.assertLess(ci["point"], 0.0)

    def test_a_significant_overprediction_below_the_point_bar_fails_through_the_lower_ci_alone(self):
        ci = self.d_m(over=0.04)
        self.assertGreater(ci["lo"], 0.0)
        self.assertLess(ci["point"], 0.05)
        self.assertEqual(A.a1_verdict_r1(ci), "FAIL-A1")
        self.assertEqual(A.a1_fails_by_r1(ci), ["lower_ci_gt_0"])
        self.assertEqual(A.a1_verdict(ci), "FAIL-A1")                                                          # (the draft also refuses this one: hi 0.038 > 0.02)

    def test_a_tiny_but_significant_overprediction_is_where_the_two_rule_sets_differ(self):
        ci = self.d_m(over=0.005)
        self.assertLessEqual(ci["hi"], 0.02)
        self.assertEqual(A.a1_verdict(ci), "ADMIT-A1")                                                         # the draft admits it (upper CI under +0.02)
        self.assertEqual(A.a1_verdict_r1(ci), "FAIL-A1")                                                       # R1 refuses it (lower CI above 0): the price of a zero-tolerance significance rule

    def test_a_material_overprediction_with_a_wide_interval_fails_through_the_point_alone(self):
        ci = self.wide_case_ci()
        self.assertLessEqual(ci["lo"], 0.0)
        self.assertGreater(ci["point"], 0.05)
        self.assertEqual(A.a1_verdict_r1(ci), "FAIL-A1")
        self.assertEqual(A.a1_fails_by_r1(ci), ["point_gt_0.05"])

    def test_a_planted_overprediction_the_size_of_ep_r1_fails_with_both_conjuncts(self):
        ci = self.d_m(over=0.135)
        self.assertEqual(A.a1_fails_by_r1(ci), ["lower_ci_gt_0", "point_gt_0.05"])

    def test_each_conjunct_mutant_disagrees_on_the_planted_case_that_needs_it(self):
        ci_lo_only, ci_pt_only = self.d_m(over=0.04), self.wide_case_ci()
        m_no_lo = named_mutant(A1R1_SRC, 'if ci["point"] <= point_bar:\n        return "ADMIT-A1"')
        self.assertEqual(m_no_lo.a1_verdict_r1(ci_lo_only), "ADMIT-A1")                                         # without the lower-CI conjunct the significant over-predictor passes
        self.assertEqual(A.a1_verdict_r1(ci_lo_only), "FAIL-A1")
        m_no_pt = named_mutant(A1R1_SRC, 'if ci["lo"] <= lo_bar:\n        return "ADMIT-A1"')
        self.assertEqual(m_no_pt.a1_verdict_r1(ci_pt_only), "ADMIT-A1")                                         # without the point conjunct the material, noisy over-predictor passes
        self.assertEqual(A.a1_verdict_r1(ci_pt_only), "FAIL-A1")
        m_or = named_mutant(A1R1_SRC, 'if ci["lo"] <= lo_bar or ci["point"] <= point_bar:\n        return "ADMIT-A1"')
        self.assertEqual((m_or.a1_verdict_r1(ci_lo_only), m_or.a1_verdict_r1(ci_pt_only)), ("ADMIT-A1", "ADMIT-A1"))


class TestRevision1A2(unittest.TestCase):
    def a2(self, **kw):
        an, S = analyze(make_case(**kw))
        return an["A2_reliability"]

    def test_calibrated_safety_rows_are_admitted(self):
        self.assertEqual(A.a2_verdict_r1(self.a2()), "ADMIT-A2")

    def test_rows_that_under_predict_risk_at_the_picks_fail_on_each_head(self):
        for kw, head in (({"nc_unsafe": 0.12}, "NC"), ({"dac_unsafe": 0.12}, "DAC"), ({"ttc_unsafe": 0.12}, "TTC")):
            r = self.a2(**kw)
            self.assertGreater(r[head]["lo"], 1.0, head)
            self.assertEqual(A.a2_verdict_r1(r), "FAIL-A2", head)

    def test_ratios_below_one_are_never_refused_where_the_draft_refused_them(self):
        for kw, head in (({"ttc_unsafe": 0.0}, "TTC"), ({"nc_unsafe": 0.0}, "NC")):
            r = self.a2(**kw)
            self.assertLess(r[head]["hi"], 1.0, head)
            self.assertEqual(A.a2_verdict(r), "FAIL-A2")                                                       # the draft's two-sided band refused it
            self.assertEqual(A.a2_verdict_r1(r), "ADMIT-A2")                                                   # R1 does not

    def test_mutants_of_a2_r1_are_wrong_on_the_planted_rows(self):
        r_bad = self.a2(nc_unsafe=0.12)
        m_any = named_mutant(A2R1_HEADS_SRC, 'if len(heads) > 0 and any((h in cis) and a2_head_ok_r1(cis[h]) for h in heads):')
        self.assertEqual(m_any.a2_verdict_r1(r_bad), "ADMIT-A2")
        m_hi = named_mutant(A2R1_SRC, 'return bool(ci["hi"] <= bar)')
        self.assertEqual(m_hi.a2_verdict_r1(self.a2()), "FAIL-A2")                                              # the upper CI of a calibrated head is above 1: the mutant refuses a calibrated candidate
        m_two = named_mutant(A2R1_SRC, 'return bool(ci["lo"] <= bar and ci["hi"] >= A2_LO)')
        self.assertEqual(m_two.a2_verdict_r1(self.a2(ttc_unsafe=0.0)), "FAIL-A2")                              # a two-sided mutant refuses the conservative row again


class TestRevision1AnySeed(unittest.TestCase):
    def test_one_failing_seed_fails_the_rule_in_either_order(self):
        good, _ = analyze(make_case(over=0.0))
        bad, _ = analyze(make_case(over=0.30))
        for order in ((good, bad), (bad, good)):
            self.assertEqual(A.a1_verdict_r1_seeds([a["G"]["D_M"] for a in order]), "FAIL-A1")
        self.assertEqual(A.a1_verdict_r1_seeds([good["G"]["D_M"], good["G"]["D_M"]]), "ADMIT-A1")
        bad2, _ = analyze(make_case(nc_unsafe=0.12))
        for order in ((good, bad2), (bad2, good)):
            self.assertEqual(A.a2_verdict_r1_seeds([a["A2_reliability"] for a in order]), "FAIL-A2")
        self.assertEqual(A.a2_verdict_r1_seeds([good["A2_reliability"], good["A2_reliability"]]), "ADMIT-A2")

    def test_mutated_quantifiers_miss_the_failing_seed(self):
        good, _ = analyze(make_case(over=0.0))
        bad, _ = analyze(make_case(over=0.30))
        cis = [good["G"]["D_M"], bad["G"]["D_M"]]
        m_any = named_mutant(ANYSEED_SRC, 'return bool(len(verdicts) > 0 and any(v == admit for v in verdicts))')
        self.assertEqual(m_any.a1_verdict_r1_seeds(cis), "ADMIT-A1")
        m_first = named_mutant('any_seed_admits([a1_verdict_r1(c) for c in cis], "ADMIT-A1")', 'any_seed_admits([a1_verdict_r1(c) for c in cis[:1]], "ADMIT-A1")')
        self.assertEqual(m_first.a1_verdict_r1_seeds(cis), "ADMIT-A1")                                          # the failing seed is the second one
        self.assertEqual(A.a1_verdict_r1_seeds(cis), "FAIL-A1")

    def test_the_seed_functions_agree_with_admission_label(self):
        """The run aggregates per-bank verdicts with `admission_label`; the simulation with `*_seeds`.  They must be the same quantifier on every combination."""
        import itertools
        for a1s in itertools.product(("ADMIT-A1", "FAIL-A1"), repeat=2):
            for a2s in itertools.product(("ADMIT-A2", "FAIL-A2"), repeat=2):
                lab = A.admission_label(list(a1s), list(a2s), True, False)
                self.assertEqual(lab.startswith("ADMIT"), A.any_seed_admits(list(a1s), "ADMIT-A1") and A.any_seed_admits(list(a2s), "ADMIT-A2"))


class TestRevision1Run(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.rk = write_ranked(os.path.join(cls.tmp, "ranked.tsv"))
        cls.rows, _ = A.read_ranked(cls.rk)
        good, bad = make_case(over=0.0), make_case(over=0.30)
        cls.g = A.load_bank_units(write_bank(os.path.join(cls.tmp, "p2g"), good[0], good[2], good[3], ckpt="p2"), cls.rows)
        cls.b = A.load_bank_units(write_bank(os.path.join(cls.tmp, "p2b"), bad[0], bad[2], bad[3], ckpt="p2"), cls.rows)
        cls.cg = A.align_candidate(cls.g, A.load_bank_units(write_bank(os.path.join(cls.tmp, "cg"), good[1], good[2], good[3], ckpt="c", shuffle_seed=2), cls.rows))
        cls.cb = A.align_candidate(cls.b, A.load_bank_units(write_bank(os.path.join(cls.tmp, "cb"), bad[1], bad[2], bad[3], ckpt="c", shuffle_seed=2), cls.rows))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_rules(self, p2, cands, **kw):
        return A.run_admission(p2, cands, B=500, null_reps_e=0, log=lambda *_: None, **kw)

    def test_the_default_is_r1_with_slot_source_row_and_the_draft_is_label(self):
        r = self.run_rules(self.g, {"c": self.cg})
        self.assertEqual((r["rules"], r["B"]["slot_source"]), ("r1", "row"))
        self.assertIn("lo_bar", r["per_candidate"]["c"]["A1"])
        self.assertNotIn("bar", r["per_candidate"]["c"]["A1"])
        d = self.run_rules(self.g, {"c": self.cg}, rules="draft")
        self.assertEqual((d["rules"], d["B"]["slot_source"]), ("draft", "label"))
        self.assertEqual(d["per_candidate"]["c"]["A1"]["bar"], 0.02)
        self.assertIn("band", d["per_candidate"]["c"]["A2"])
        e = self.run_rules(self.g, {"c": self.cg}, slot_source="label")
        self.assertEqual((e["rules"], e["B"]["slot_source"]), ("r1", "label"))                                  # an explicit choice wins over the rule set's default
        f = self.run_rules(self.g, {"c": self.cg}, rules="draft", slot_source="row")
        self.assertEqual(f["B"]["slot_source"], "row")

    def test_a_calibrated_candidate_is_admitted_and_an_overpredictor_refused_under_both_rule_sets(self):
        for rules in ("r1", "draft"):
            g = self.run_rules(self.g, {"c": self.cg}, rules=rules)
            b = self.run_rules(self.b, {"c": self.cb}, rules=rules)
            self.assertTrue(g["controls_ok"] and b["controls_ok"], rules)
            self.assertTrue(g["verdict"]["label"].startswith("ADMIT"), (rules, g["verdict"]))
            self.assertEqual(b["verdict"]["label"], "REFUSE-A1", rules)
        b = self.run_rules(self.b, {"c": self.cb})
        self.assertEqual(b["per_candidate"]["c"]["A1"]["fails_by"], ["lower_ci_gt_0", "point_gt_0.05"])

    def test_one_failing_seed_refuses_the_candidate_in_either_order(self):
        ident = A.align_candidate(self.b, self.b)                                                                # P2 as the 'good' seed on the over-prediction bank's labels: admits
        for seeds in ({"s0": ident, "s1": self.cb}, {"s0": self.cb, "s1": ident}):
            r = self.run_rules(self.b, seeds)
            self.assertEqual(r["verdict"]["label"], "REFUSE-A1", list(seeds))
            self.assertEqual(sorted(r["verdict"]["A1"].values()), ["ADMIT-A1", "FAIL-A1"])

    def test_the_draft_run_has_the_drafts_json_keys(self):
        d = self.run_rules(self.g, {"c": self.cg}, rules="draft")
        self.assertEqual(set(d), {"schema", "rules", "warnings", "controls", "controls_ok", "per_candidate", "a1_form_mode", "a1_form_used", "scale_on_all_banks", "verdict", "B", "null", "n_logs", "n_sets", "bootstrap"})
        self.assertEqual(set(d["per_candidate"]["c"]["A1"]), {"form", "statistic", "bar", "verdict", "other_form_statistic"})
        self.assertEqual(set(d["per_candidate"]["c"]["A2"]), {"verdict", "band", "heads"})

    def test_an_unknown_rule_set_is_refused(self):
        with self.assertRaises(A.AdmissionError):
            self.run_rules(self.g, {"c": self.cg}, rules="bogus")

    def test_the_power_control_and_the_null_use_the_rule_sets_a1_function(self):
        z6_p2, z6_c, lab, li, ch = make_case(frac=0.6)
        S = stats_for(li)
        r1 = A.control_power(z6_p2, z6_c, lab, S, a1_fn=A.a1_verdict_r1)
        self.assertTrue(r1["pass"], r1)
        self.assertEqual((r1["plus"]["a1"], r1["minus"]["a1"], r1["zero"]["a1"]), ("FAIL-A1", "ADMIT-A1", "ADMIT-A1"))
        always_fail = lambda ci: "FAIL-A1"                                                                       # noqa: E731
        self.assertFalse(A.control_power(z6_p2, z6_c, lab, S, a1_fn=always_fail)["pass"])                         # the control really calls the function it is given
        n = A.run_null(z6_p2, z6_c, lab, S, reps_e=0, a1_fn=always_fail)
        self.assertTrue(n["void"] and n["c_null"]["fires"] == 5)                                                 # and so does the null


if __name__ == "__main__":
    unittest.main()
