"""Tests of the search-bank adapter and the C1-geometry simulation, on PLANTED synthetic data only (no carve-B / B-REST statistic is computed).  CPU only.
    CUDA_VISIBLE_DEVICES=-1 python -I test_searchbank.py -v          (or pytest)
Every guard has (i) a RED input the real module refuses and (ii) a MUTATION arm: the guard's line is rewritten in a copy of the module, and the SAME RED input must then be accepted."""
from __future__ import annotations

import copy
import importlib.util
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
import epadm_searchbank as SB                                                                # noqa: E402
import fixture_synth as FX                                                                   # noqa: E402

SB_SRC = os.path.join(HERE, "epadm_searchbank.py")
TMP = tempfile.mkdtemp(prefix="searchbank_test_")
import atexit                                                                                # noqa: E402
atexit.register(shutil.rmtree, TMP, ignore_errors=True)                                      # C: is nearly full: the fixtures and banks are removed when the run ends
_FX = {}
_CTR = [0]


def fixture(name: str = "base", **kw) -> dict:
    """A fresh COPY of the synthetic fixture (tests plant defects into copies)."""
    if name not in _FX:
        _FX[name] = FX.make_fixture(os.path.join(TMP, f"fx_{name}"), **kw)
    _CTR[0] += 1
    d = os.path.join(TMP, f"copy_{_CTR[0]}")
    shutil.copytree(_FX[name]["dir"], d)
    return {k: (os.path.join(d, os.path.basename(v)) if k in ("bundle", "search", "labels") else (d if k == "dir" else v)) for k, v in _FX[name].items()}


def out_dir() -> str:
    _CTR[0] += 1
    return os.path.join(TMP, f"out_{_CTR[0]}")


def mutated(old: str, new: str):
    """The adapter module with ONE source text replaced (the old text must occur exactly once); loaded as a separate module sharing the sha-checked gate."""
    src = open(SB_SRC, encoding="utf-8").read()
    assert src.count(old) == 1, f"mutation anchor occurs {src.count(old)} times: {old!r}"
    mod = type(sys)("epadm_searchbank_mut")
    mod.__file__ = SB_SRC
    mod.__dict__["__name__"] = "epadm_searchbank_mut"
    exec(compile(src.replace(old, new), "epadm_searchbank_mut", "exec"), mod.__dict__)
    return mod


def build(fx, mod=SB, expect=None):
    return mod.build(fx["bundle"], fx["search"], fx["labels"], out_dir(), {"token_sha256": fx["token_sha256"], **(expect or {})})


class TestHappyPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = fixture()
        cls.out = out_dir()
        cls.rep = SB.build(cls.fx["bundle"], cls.fx["search"], cls.fx["labels"], cls.out, {"token_sha256": cls.fx["token_sha256"]})
        cls.src = SB.read_sources(cls.fx["bundle"], cls.fx["search"], cls.fx["labels"])

    def test_identity_every_unit_value_equals_the_source(self):
        res = SB.identity_check(self.out, self.src)
        self.assertTrue(res["pass"], res["mismatches"])
        self.assertTrue(all(v == 0 for v in res["mismatches"].values()))
        self.assertEqual(res["n_changed_picks_by_seed"], res["moved_by_seed"])
        self.assertTrue(res["gate_C-ID_P2_as_candidate"]["pass"])
        self.assertGreater(sum(res["moved_by_seed"]), 0)

    def test_gate_loaders_accept_the_four_banks_and_align(self):
        p2, cands, rows = SB.load_with_gate(self.out)
        self.assertEqual(p2["n_sets"], 60)
        self.assertEqual(len(cands), 3)
        for c in cands:
            self.assertEqual(c["n_sets"], 60)
            self.assertTrue(c["checks"]["keys_bijection"] and c["checks"]["labels_bit_equal"])

    def test_the_whole_frozen_gate_runs_on_the_adapted_bank(self):
        """End-to-end wiring on SYNTHETIC data: the gate's own run_admission reads the adapted bank (3 candidates) and returns a labelled verdict; this is a wiring proof, the label means nothing."""
        A = SB.gate()
        p2, cands, rows = SB.load_with_gate(self.out)
        res = A.run_admission(p2, {f"s{k}": cands[k] for k in range(3)}, null_reps_e=4, B=300, skip_controls_mutation=True, log=lambda *a: None)
        self.assertEqual(set(res["per_candidate"]), {"s0", "s1", "s2"})
        self.assertIn("label", res["verdict"])
        self.assertEqual(res["n_logs"], 6)
        for k in range(3):
            self.assertEqual(res["per_candidate"][f"s{k}"]["n_changed"], int(self.src["search"]["moved"][:, k].sum()))

    def test_deterministic_and_manifest_sha_stable(self):
        o2 = out_dir()
        rep2 = SB.build(self.fx["bundle"], self.fx["search"], self.fx["labels"], o2, {"token_sha256": self.fx["token_sha256"]})
        for b in ("p2", "s0", "s1", "s2"):
            self.assertEqual(self.rep["banks"][b]["manifest_sha256"], rep2["banks"][b]["manifest_sha256"])

    def test_host_slots_are_lowest_aggregate_non_pick_and_distinct(self):
        P = SB.pickrule()
        Z = self.src["bundle"]["logits"].astype(np.float32)
        pick = self.src["bundle"]["pick"]
        host = SB.host_slots(Z, pick)
        agg = P.rule_aggregate(Z.astype(np.float64))
        for i in range(len(pick)):
            self.assertNotIn(pick[i], host[i])
            self.assertEqual(len(set(host[i].tolist())), 3)
            rest = np.delete(agg[i], list(host[i]) + [pick[i]])
            self.assertLessEqual(agg[i][host[i]].max(), rest.min() + 1e-15)

    def _tamper(self, bank, field, fn):
        o = out_dir()
        SB.build(self.fx["bundle"], self.fx["search"], self.fx["labels"], o, {"token_sha256": self.fx["token_sha256"]})
        unit = sorted(f for f in os.listdir(os.path.join(o, bank)) if f.endswith(".sets.npz"))[0]
        p = os.path.join(o, bank, unit)
        z = np.load(p, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        z.close()
        d[field] = fn(d[field].copy())
        np.savez(p, **d)
        mp = os.path.join(o, bank, "MANIFEST.sha256.json")
        man = json.load(open(mp))
        man["files"][unit]["sha256"] = SB.sha256_file(p)                                      # regenerated: only the identity / alignment checks stand between the edit and acceptance
        json.dump(man, open(mp, "w"))
        return o

    def test_red_unedited_manifest_refuses_an_edited_unit(self):
        o = out_dir()
        SB.build(self.fx["bundle"], self.fx["search"], self.fx["labels"], o, {"token_sha256": self.fx["token_sha256"]})
        unit = sorted(f for f in os.listdir(os.path.join(o, "s1")) if f.endswith(".sets.npz"))[0]
        p = os.path.join(o, "s1", unit)
        z = np.load(p, allow_pickle=False); d = {k: z[k] for k in z.files}; z.close()
        d["logits"][0, 0, 0] += 1.0
        np.savez(p, **d)
        with self.assertRaises(Exception) as cm:
            SB.load_with_gate(o)
        self.assertIn("manifest", str(cm.exception))

    def test_red_identity_check_catches_an_edited_logit_the_gate_does_not_compare(self):
        def f(a):
            a[0, 0, 2] += np.float32(1e-3)
            return a
        o = self._tamper("s1", "logits", f)
        res = SB.identity_check(o, self.src)
        self.assertFalse(res["pass"])
        self.assertGreater(res["mismatches"]["s1_logits"], 0)

    def test_red_a_flipped_label_bit_is_caught_by_the_gates_c_labels(self):
        def f(a):
            a[0, 0, 2] = np.nextafter(a[0, 0, 2], np.float32(2.0))
            return a
        o = self._tamper("s1", "labels6", f)
        with self.assertRaises(Exception) as cm:
            SB.identity_check(o, self.src)
        self.assertIn("C-LABELS", str(cm.exception))


class TestGuardsRedAndMutation(unittest.TestCase):
    """Each case: (a) the real adapter refuses the planted defect with a BankError naming the guard; (b) the MUTATED adapter (guard removed) does not."""

    def case(self, plant, tag, old, new, expect_extra=None):
        fx = fixture()
        plant(fx)
        with self.assertRaises(SB.BankError) as cm:
            build(fx, SB, expect_extra)
        self.assertIn(tag, str(cm.exception))
        fx2 = fixture()
        plant(fx2)
        mod = mutated(old, new)
        try:
            build(fx2, mod, expect_extra)
        except Exception as e:
            if type(e).__name__ == "BankError":                                                 # the mutated module has its own BankError class
                self.fail(f"the mutated module still refused with a guard error: {e}")
            # a downstream (non-guard) failure is fine: the guard itself no longer fired

    # ---- G1: keys bijective / aligned files
    def test_g1_duplicate_token_refused(self):
        def plant(fx):                                                                          # the duplicate is planted CONSISTENTLY in all three files and pinned, so only the uniqueness guard sees it
            z = np.load(fx["bundle"], allow_pickle=False)
            t = z["token"].copy(); t[1] = t[0]
            FX.rewrite(fx["bundle"], token=t)
            FX.rewrite(fx["search"], token=t)
            FX.rewrite(fx["labels"], token=t)
            new = SB.token_list_sha256(t)
            fx["token_sha256"] = new
            FX.refresh_meta(fx, bundle_sha256=SB.sha256_file(fx["bundle"]), data_clause={"carve_sha256": new, "bundle_tokens_sha256": new})
        self.case(plant, "G1", "if len(set(tok)) != n:", "if False:")

    def test_g1_search_tokens_in_another_order_refused(self):
        def plant(fx):
            z = np.load(fx["search"], allow_pickle=False)
            t = z["token"].copy(); t[[0, 1]] = t[[1, 0]]
            FX.rewrite(fx["search"], token=t)
            FX.refresh_labels_search_sha(fx)
        self.case(plant, "G1", 'if [str(t) for t in S["token"]] != tok or [str(t) for t in L["token"]] != tok:', "if False:")

    def test_g1_search_pick_differs_from_bundle_pick_refused(self):
        def plant(fx):
            z = np.load(fx["search"], allow_pickle=False)
            p = z["pick"].copy(); p[3] = (p[3] + 1) % 64
            FX.rewrite(fx["search"], pick=p)
            FX.refresh_labels_search_sha(fx)
        self.case(plant, "G1", 'if not np.array_equal(S["pick"], B["pick"]):', "if False:")

    def test_g1_candidate_bank_with_a_missing_key_is_refused_by_the_gate_alignment(self):
        fx = fixture()
        o = out_dir()
        SB.build(fx["bundle"], fx["search"], fx["labels"], o, {"token_sha256": fx["token_sha256"]})
        unit = sorted(f for f in os.listdir(os.path.join(o, "s0")) if f.endswith(".sets.npz"))[0]
        p = os.path.join(o, "s0", unit)
        z = np.load(p, allow_pickle=False)
        d = {k: (z[k][1:] if (z[k].ndim >= 1 and len(z[k]) == len(z["token"]) and k not in ("label_order",)) else z[k]) for k in z.files}
        z.close()
        np.savez(p, **d)
        man = json.load(open(os.path.join(o, "s0", "MANIFEST.sha256.json")))
        man["files"][unit]["sha256"] = SB.sha256_file(p)
        json.dump(man, open(os.path.join(o, "s0", "MANIFEST.sha256.json"), "w"))
        with self.assertRaises(Exception) as cm:
            SB.load_with_gate(o)
        self.assertIn("C-KEYS", str(cm.exception))

    def test_g1_candidate_bank_with_swapped_keys_is_refused_by_the_gate_alignment(self):
        fx = fixture()
        o = out_dir()
        SB.build(fx["bundle"], fx["search"], fx["labels"], o, {"token_sha256": fx["token_sha256"]})
        unit = sorted(f for f in os.listdir(os.path.join(o, "s2")) if f.endswith(".sets.npz"))[0]
        p = os.path.join(o, "s2", unit)
        z = np.load(p, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        z.close()
        d["token"] = d["token"].copy(); d["token"][0] = "ffffffffffffffff"
        np.savez(p, **d)
        man = json.load(open(os.path.join(o, "s2", "MANIFEST.sha256.json")))
        man["files"][unit]["sha256"] = SB.sha256_file(p)
        json.dump(man, open(os.path.join(o, "s2", "MANIFEST.sha256.json"), "w"))
        with self.assertRaises(Exception) as cm:
            SB.load_with_gate(o)
        self.assertIn("C-KEYS", str(cm.exception))

    # ---- G2: every search output carries its own label
    def test_g2_missing_label_refused(self):
        def plant(fx):
            z = np.load(fx["labels"], allow_pickle=False)
            s = z["sub"].copy(); s[5, 1, 2] = np.nan
            FX.rewrite(fx["labels"], sub=s)
        self.case(plant, "G2", "if not np.isfinite(sub).all():", "if False:")

    def test_g2_label_array_with_a_missing_unit_refused(self):
        def plant(fx):
            z = np.load(fx["labels"], allow_pickle=False)
            FX.rewrite(fx["labels"], sub=z["sub"][:, :2, :])
        self.case(plant, "G2", "if sub.shape != (n, N_SEEDS, N_HEADS):", "if False:")

    def test_g2_label_out_of_range_refused(self):
        def plant(fx):
            z = np.load(fx["labels"], allow_pickle=False)
            s = z["sub"].copy(); s[2, 0, 1] = 1.5
            FX.rewrite(fx["labels"], sub=s)
        self.case(plant, "G2", "if sub.min() < 0.0 or sub.max() > 1.0:", "if False:")

    def test_g2_labeller_identity_failure_refused(self):
        def plant(fx):
            z = np.load(fx["labels"], allow_pickle=False)
            s = z["id_sub"].copy(); s[4, 2] = 1.0 - s[4, 2]
            FX.rewrite(fx["labels"], id_sub=s)
        self.case(plant, "G2", 'if not np.array_equal(L["id_sub"], pick_lab):', "if False:")

    def test_g2_unmoved_output_with_another_label_refused(self):
        def plant(fx):
            s0 = np.load(fx["search"], allow_pickle=False)
            mv = s0["moved"]
            i, k = np.argwhere(~mv)[0]
            z = np.load(fx["labels"], allow_pickle=False)
            s = z["sub"].copy(); s[i, k, 2] = 0.123
            FX.rewrite(fx["labels"], sub=s)
        self.case(plant, "G2", "if not np.array_equal(sub[um], np.broadcast_to(pick_lab[:, None, :], sub.shape)[um]):", "if False:")

    # ---- G3: pins (search config / checkpoint / head / arm / seeds / tokens / bundle / labels link)
    def test_g3_different_search_config_sha_refused(self):
        def plant(fx):
            FX.refresh_meta(fx, config_sha256="0" * 64)
        self.case(plant, "G3: search config_sha256", "if got != want:", "if False:")

    def test_g3_different_checkpoint_sha_refused(self):
        def plant(fx):
            FX.refresh_meta(fx, ckpt_sha256="1" * 64)
        self.case(plant, "G3: search ckpt_sha256", "if got != want:", "if False:")

    def test_g3_head_substitution_refused(self):
        def plant(fx):
            FX.refresh_meta(fx, head_sha256="a" * 64)
        self.case(plant, "G3", 'if m.get("head_sha256") is not None:', "if False:")

    def test_g3_wrong_arm_refused(self):
        def plant(fx):
            FX.refresh_meta(fx, arm="L2")
        self.case(plant, "G3: search arm", "if got != want:", "if False:")

    def test_g3_wrong_seeds_refused(self):
        def plant(fx):
            FX.rewrite(fx["search"], seeds=np.array([1, 2, 3], np.int64))
            FX.refresh_labels_search_sha(fx)
        self.case(plant, "G3: search seeds", "if got != want:", "if False:")

    def test_g3_unnamed_expected_pin_refused(self):
        fx = fixture()
        with self.assertRaises(SB.BankError):
            SB.build(fx["bundle"], fx["search"], fx["labels"], out_dir(), {"token_sha256": None})

    def test_g3_wrong_token_list_refused(self):
        def plant(fx):
            fx["token_sha256"] = "2" * 64
        self.case(plant, "token_list_sha256", "if got != want:", "if False:")

    def test_g1_bundle_index_not_arange_refused(self):
        def plant(fx):
            z = np.load(fx["search"], allow_pickle=False)
            bi = z["bundle_index"].copy(); bi[2] = 59
            FX.rewrite(fx["search"], bundle_index=bi)
            FX.refresh_labels_search_sha(fx)
        self.case(plant, "G1", 'if not (np.array_equal(S["bundle_index"], np.arange(n)) and np.array_equal(L["bundle_index"], np.arange(n))):', "if False:")

    def test_g3_labels_from_another_search_file_refused(self):
        def plant(fx):
            FX.rewrite(fx["labels"], search_sha256=np.array("3" * 64))
        self.case(plant, "G3: the labels were made from a different search file", "if lsha != src[\"search_file_sha256\"]:", "if False:")

    def test_g3_search_made_on_another_bundle_refused(self):
        def plant(fx):
            FX.refresh_meta(fx, bundle_sha256="4" * 64)
        self.case(plant, "G3: the search file was made on bundle", 'if m.get("bundle_sha256") != src["bundle_file_sha256"]:', "if False:")

    # ---- G4 / G5: the frozen pick rule reads the intended pick
    def test_g4_moved_output_that_loses_the_rule_refused(self):
        def plant(fx):
            z = np.load(fx["search"], allow_pickle=False)
            i, k = np.argwhere(z["moved"])[0]
            po = z["pred_logit_out"].copy(); po[i, k] = po[i, k] - 5.0
            FX.rewrite(fx["search"], pred_logit_out=po)
            FX.refresh_labels_search_sha(fx)
        self.case(plant, "G4", "if not np.array_equal(got, want):", "if False:")

    def test_g4_unmoved_output_with_other_logits_refused(self):
        def plant(fx):
            z = np.load(fx["search"], allow_pickle=False)
            i, k = np.argwhere(~z["moved"])[0]
            po = z["pred_logit_out"].copy(); po[i, k] = po[i, k] + 0.5
            FX.rewrite(fx["search"], pred_logit_out=po)
            FX.refresh_labels_search_sha(fx)
        self.case(plant, "G4", 'if not np.array_equal(po[~moved], np.broadcast_to(pl[:, None, :], po.shape)[~moved]):', "if False:")

    def test_g5_bundle_pick_that_is_not_the_frozen_rule_refused(self):
        """G5 is shadowed in a full build by G4 (a wrong pick also breaks the candidate pick rule), so its own line is tested on `check_p2_bundle` directly."""
        fx = fixture()
        z = np.load(fx["bundle"], allow_pickle=False)
        p = z["pick"].copy(); p[7] = (p[7] + 3) % 64
        FX.rewrite(fx["bundle"], pick=p)
        src = SB.read_sources(fx["bundle"], fx["search"], fx["labels"])
        with self.assertRaises(SB.BankError) as cm:
            SB.check_p2_bundle(src)
        self.assertIn("G5", str(cm.exception))
        mod = mutated("if not np.array_equal(P.rule_pick(Z), pick):", "if False:")
        mod.check_p2_bundle(src)                                                                # the mutated guard accepts the wrong pick
        with self.assertRaises(Exception) as cm2:                                               # with G5 removed the full build is still refused downstream by G4 (defence in depth)
            build(fx, mod)
        self.assertEqual(type(cm2.exception).__name__, "BankError")
        self.assertRegex(str(cm2.exception), r"^G[1-4]")                                       # refused by another guard (here G1: the search file's pick no longer equals the bundle's)

    def test_gate_pin_mutation_arm(self):
        src = open(os.path.join(os.path.dirname(SB.gate().__file__), "ep_admission.py"), "rb").read()
        d = tempfile.mkdtemp(dir=TMP)
        open(os.path.join(d, "ep_admission.py"), "wb").write(src + b"\n# edited\n")
        mod = mutated("if sha != GATE_SHA256:", "if False:")
        mod._GATE, mod.GATE_DIRS = None, (d,)
        self.assertIsNotNone(mod.gate())                                                        # the pin guard removed: the edited gate is imported
        sys.modules["ep_admission"] = SB.gate()                                                 # restore the sha-checked module for the other tests

    def test_gate_pin_a_changed_gate_is_refused(self):
        src = open(os.path.join(os.path.dirname(SB.gate().__file__), "ep_admission.py"), "rb").read()
        d = tempfile.mkdtemp(dir=TMP)
        open(os.path.join(d, "ep_admission.py"), "wb").write(src + b"\n# edited\n")
        saved = (SB._GATE, SB.GATE_DIRS)
        SB._GATE, SB.GATE_DIRS = None, (d,)
        try:
            with self.assertRaises(SB.BankError):
                SB.gate()
        finally:
            SB._GATE, SB.GATE_DIRS = saved

    def test_harness_a_non_guard_mutation_does_not_accept_the_defect(self):
        """The mutation harness must not accept everything: rewriting a comment-only token leaves the RED input refused."""
        fx = fixture()
        FX.refresh_meta(fx, config_sha256="0" * 64)
        mod = mutated('"""The three source files as plain arrays', '"""The three source files, as plain arrays')
        with self.assertRaises(Exception) as cm:
            build(fx, mod)
        self.assertEqual(type(cm.exception).__name__, "BankError")


class TestSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import epadm_sim_c1 as C
        cls.C = C
        cls.fx = fixture("sim", n_tokens=120, n_logs=24, seed=3)
        cls.out = out_dir()
        SB.build(cls.fx["bundle"], cls.fx["search"], cls.fx["labels"], cls.out, {"token_sha256": cls.fx["token_sha256"]})
        cls.pool = C.load_pool(cls.out)
        cls.counts = np.full(24, 10, np.int64)
        cls.R1 = C.R1

    def test_token_vectors_sum_to_the_r1_components_bit_for_bit(self):
        p2, cands, rows = SB.load_with_gate(self.out)
        for s in range(3):
            comp = self.R1.components(p2["z6"], cands[s]["z6"], p2["lab"], p2["log_idx"], 24)["vec"]
            for k, v in self.pool["tok"][s].items():
                self.assertTrue(np.array_equal(np.bincount(self.pool["log_idx"], weights=v, minlength=24), comp[k]), k)

    def test_base_is_calibrated_by_construction(self):
        C = self.C
        base = C.base_vectors(self.pool, self.counts, 0)
        for s in range(3):
            d = self.R1.build_condition(base[s], "cal")
            self.assertAlmostEqual(d["Yp"].sum() / d["Sp"].sum(), 1.0, places=12)
            self.assertAlmostEqual(d["Ye"].sum() / d["Se"].sum(), 1.0, places=12)
            for h in C.HEADS:
                self.assertAlmostEqual(d[f"OBS_{h}"].sum() / d[f"PRED_{h}"].sum(), 1.0, places=12)
            pl = self.R1.build_condition(base[s], "plant_ep", delta=0.1)
            self.assertAlmostEqual(pl["Yp"].sum() / pl["Sp"].sum() - pl["Ye"].sum() / pl["Se"].sum(), 0.1, places=10)

    def test_base_draw_has_c1_counts_per_log_and_is_seeded(self):
        C = self.C
        b0, b0b, b1 = C.base_vectors(self.pool, self.counts, 0), C.base_vectors(self.pool, self.counts, 0), C.base_vectors(self.pool, self.counts, 1)
        self.assertTrue(all(np.array_equal(b0[s][k], b0b[s][k]) for s in range(3) for k in b0[s]))
        self.assertFalse(np.array_equal(b0[0]["y_p"], b1[0]["y_p"]))
        offs = C.base_vectors(self.pool, self.counts, 0, offset=1)
        self.assertTrue(np.array_equal(offs[0]["y_p"], b1[0]["y_p"]))                         # rep + offset is the draw index

    def test_replicate_seeds_are_new(self):
        C = self.C
        self.assertNotIn(C.OUTER_SALT, C.R1_SALTS)
        self.assertNotIn(C.BASE_SALT, C.R1_SALTS)
        self.assertNotEqual(C.OUTER_SALT, C.BASE_SALT)
        self.assertEqual((C.N_LOGS, C.BOOT_B, C.BOOT_SEED, C.SEEDS), (24, 10000, 20260927, (0, 1, 2)))

    def test_pc3_shortcut_equals_the_gate_estimator_on_the_adapted_bank(self):
        """The per-log shortcut (F.diff_ci / F.ratio_ci on bincount vectors) equals the gate's own analyze_candidate on the SAME adapted bank, to 0.0."""
        C = self.C
        res = C.pc3_shortcut(self.out)
        self.assertTrue(res["pass"], res)

    def test_replicate_deterministic_and_identity_exactly_zero(self):
        C = self.C
        _, _, F = self.R1.libs()
        draws = F.boot_draws(24, 10000, 20260927)
        conds = C.conditions()
        a = C.replicate(conds["CAL"], self.pool, self.counts, 3, draws)
        b = C.replicate(conds["CAL"], self.pool, self.counts, 3, draws)
        for k in a:
            self.assertTrue(np.array_equal(a[k], b[k], equal_nan=True))
        idn = C.replicate(conds["IDENTITY"], self.pool, self.counts, 0, draws)
        self.assertTrue((idn["dm"] == 0.0).all())

    def test_decisions_module_equals_inline_including_nan_fail_closed(self):
        C = self.C
        rng = np.random.default_rng(0)
        for _ in range(300):
            dm = np.stack([np.sort(rng.normal(0.0, 0.05, 3))[[1, 0, 2]] for _ in range(3)])
            if rng.random() < 0.1:
                dm[rng.integers(0, 3), rng.integers(0, 3)] = np.nan
            self.assertEqual(C.a1_refuse_module(dm), C.a1_refuse_inline(dm))
            a2 = rng.uniform(0.8, 1.3, (3, 3, 3))
            if rng.random() < 0.1:
                a2[rng.integers(0, 3), rng.integers(0, 3), 1] = np.nan
            for m in (1.0, 1.05, 1.10):
                for heads in (["NC", "DAC", "TTC"], ["DAC"]):
                    self.assertEqual(C.a2_refuse_module(a2, heads, m), C.a2_refuse_inline(a2, heads, m))

    def test_decide_boundaries_and_each_bar_alone(self):
        C = self.C
        self.assertTrue(C.decide(0.05, 0.05, 0.70)["all_met"])
        self.assertEqual(C.decide(0.0501, 0.01, 0.9)["missed"], ["A1_r1_false_refusal_wilson_upper_le_5pct"])
        self.assertEqual(C.decide(0.01, 0.0501, 0.9)["missed"], ["A2_1.05_false_refusal_wilson_upper_le_5pct"])
        self.assertEqual(C.decide(0.01, 0.01, 0.6999)["missed"], ["DAC_power_x1.25_margin1.05_point_ge_70pct"])
        self.assertFalse(C.decide(0.2, 0.2, 0.1)["all_met"])

    def test_decide_mutation_arms_each_bar_constant_is_load_bearing(self):
        C = self.C
        src = open(os.path.join(HERE, "epadm_sim_c1.py"), encoding="utf-8").read()
        anchor = "BAR_A1_FR_UPPER, BAR_A2_FR_UPPER, BAR_DAC_POWER_125 = 0.05, 0.05, 0.70"
        self.assertEqual(src.count(anchor), 1)
        for new, args, want_met in (("0.06, 0.05, 0.70", (0.055, 0.01, 0.9), True), ("0.05, 0.06, 0.70", (0.01, 0.055, 0.9), True), ("0.05, 0.05, 0.60", (0.01, 0.01, 0.65), True)):
            ns = {}
            body = src[src.index("def decide"):src.index("def controls")]
            exec("BAR_A1_FR_UPPER, BAR_A2_FR_UPPER, BAR_DAC_POWER_125 = " + new + "\n" + body, ns)
            self.assertEqual(ns["decide"](*args)["all_met"], want_met)
            self.assertFalse(C.decide(*args)["all_met"])                                      # the real constants refuse the same inputs

    def test_geometry_loader_refuses_a_wrong_list(self):
        C = self.C
        g = json.load(open(os.path.join(HERE, "..", "raw", "geometry_c1.json")))
        names = sorted(g["b_rest_tokens_per_log"])
        self.assertEqual(int(C.load_geometry(os.path.join(HERE, "..", "raw", "geometry_c1.json"), names).sum()), 905)
        bad = copy.deepcopy(g); bad["b_rest_tokens_sha256"] = "0" * 64
        p = os.path.join(TMP, "bad_geo.json"); json.dump(bad, open(p, "w"))
        with self.assertRaises(SystemExit):
            C.load_geometry(p, names)
        with self.assertRaises(SystemExit):
            C.load_geometry(os.path.join(HERE, "..", "raw", "geometry_c1.json"), names[:-1] + ["other"])

    def test_design_guard_refuses_missing_or_unhashed_design(self):
        C = self.C
        saved = C.DESIGN
        d = tempfile.mkdtemp(dir=TMP)
        try:
            C.DESIGN = os.path.join(d, "SIM_DESIGN.md")
            with self.assertRaises(SystemExit):
                C.design_guard()
            open(C.DESIGN, "w").write("x")
            with self.assertRaises(SystemExit):
                C.design_guard()
            open(os.path.join(d, "SIM_DESIGN.sha256"), "w").write("0" * 64 + "\n")
            with self.assertRaises(SystemExit):
                C.design_guard()
            open(os.path.join(d, "SIM_DESIGN.sha256"), "w").write(C.sha256_file(C.DESIGN) + "\n")
            self.assertEqual(C.design_guard(), C.sha256_file(C.DESIGN))
        finally:
            C.DESIGN = saved


class TestA2Margin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import apply_a2_margin as M
        cls.M = M
        fx = fixture()
        o = out_dir()
        SB.build(fx["bundle"], fx["search"], fx["labels"], o, {"token_sha256": fx["token_sha256"]})
        A = SB.gate()
        p2, cands, rows = SB.load_with_gate(o)
        cls.adm = json.loads(json.dumps(A.jsonable(A.run_admission(p2, {f"s{k}": cands[k] for k in range(3)}, null_reps_e=4, B=300, skip_controls_mutation=True, log=lambda *a: None)), default=str))

    def test_margin_1_00_reproduces_the_modules_a2_verdicts(self):
        res = self.M.apply(self.adm, 1.00)
        self.assertEqual(res["A2_at_margin"], self.adm["verdict"]["A2"])
        self.assertEqual(res["label"], self.adm["verdict"]["label"])

    def test_margin_is_applied_with_the_modules_closed_edge(self):
        import copy
        for lo, m100, m105 in ((1.03, "FAIL-A2", "ADMIT-A2"), (1.05, "FAIL-A2", "ADMIT-A2"), (1.0501, "FAIL-A2", "FAIL-A2"), (0.9, "ADMIT-A2", "ADMIT-A2")):
            adm = copy.deepcopy(self.adm)
            for cid in adm["per_candidate"]:
                for h in ("NC", "DAC", "TTC"):
                    adm["per_candidate"][cid]["A2_reliability"][h] = {"point": lo, "lo": 0.5, "hi": 2.0}
                adm["per_candidate"][cid]["A2_reliability"]["DAC"]["lo"] = lo
            self.assertEqual(set(self.M.apply(adm, 1.00)["A2_at_margin"].values()), {m100})
            self.assertEqual(set(self.M.apply(adm, 1.05)["A2_at_margin"].values()), {m105})

    def test_refuses_a_draft_rules_output(self):
        import copy
        adm = copy.deepcopy(self.adm); adm["rules"] = "draft"
        with self.assertRaises(SB.BankError):
            self.M.apply(adm, 1.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
