"""CPU-only synthetic tests for FASTNEG-CENSUS, part 2: (2) the decision rule with a RED mutation arm per branch, (3) the refit REACHES its parameters, (4) the strata script cannot read labels.
`python -I test_fastneg_more.py`.  The helpers (synthetic bank and stores) live in test_fastneg.py."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastneg_lib as L                                                              # noqa: E402
from test_fastneg import make_synth, run_strata                                     # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# (2) the decision rule: the branch table, and one RED mutation arm per branch
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MUTATIONS = [
    ("deficit: hi -> lo", 'if ci["hi"] < -DEFICIT_MARGIN:', 'if ci["lo"] < -DEFICIT_MARGIN:'),
    ("deficit: sign of the margin", 'if ci["hi"] < -DEFICIT_MARGIN:', 'if ci["hi"] < DEFICIT_MARGIN:'),
    ("no-deficit: lo -> hi", 'if ci["lo"] > -DEFICIT_MARGIN:', 'if ci["hi"] > -DEFICIT_MARGIN:'),
    ("no-deficit: sign of the margin", 'if ci["lo"] > -DEFICIT_MARGIN:', 'if ci["lo"] > DEFICIT_MARGIN:'),
    ("coverage: lo -> hi", 'if ci["lo"] > SLOPE_MARGIN:\n        return "COVERAGE"', 'if ci["hi"] > SLOPE_MARGIN:\n        return "COVERAGE"'),
    ("coverage: > -> >=", 'if ci["lo"] > SLOPE_MARGIN:\n        return "COVERAGE"', 'if ci["lo"] >= SLOPE_MARGIN:\n        return "COVERAGE"'),
    ("feature: hi -> lo", 'if ci["hi"] < SLOPE_MARGIN:\n        return "FEATURE"', 'if ci["lo"] < SLOPE_MARGIN:\n        return "FEATURE"'),
    ("feature: < -> <=", 'if ci["hi"] < SLOPE_MARGIN:\n        return "FEATURE"', 'if ci["hi"] <= SLOPE_MARGIN:\n        return "FEATURE"'),
    ("row: seeds all -> any (deficit)", 'if all(d == "DEFICIT" for d in deficits):', 'if any(d == "DEFICIT" for d in deficits):'),
    ("row: seeds all -> any (coverage)", 'if all(s == "COVERAGE" for s in slopes):', 'if any(s == "COVERAGE" for s in slopes):'),
    ("row: seeds all -> any (feature)", 'if all(s == "FEATURE" for s in slopes):', 'if any(s == "FEATURE" for s in slopes):'),
    ("row: seeds all -> any (no deficit)", 'if all(d == "NO_DEFICIT" for d in deficits):', 'if any(d == "NO_DEFICIT" for d in deficits):'),
    ("outcome: COVERAGE on NC or DAC -> and", 'elif nc == "COVERAGE" or dac == "COVERAGE":', 'elif nc == "COVERAGE" and dac == "COVERAGE":'),
    ("outcome: FEATURE on both -> or", 'elif nc == "FEATURE" and dac == "FEATURE":', 'elif nc == "FEATURE" or dac == "FEATURE":'),
    ("outcome: NO_DEFICIT on both -> or", 'elif nc == "NO_DEFICIT" and dac == "NO_DEFICIT":', 'elif nc == "NO_DEFICIT" or dac == "NO_DEFICIT":'),
    ("outcome: VOID never overrides", 'if void:\n        out = "VOID"', 'if False:\n        out = "VOID"'),
    ("null fires: all seeds -> any", 'return all(ci["lo"] > SLOPE_MARGIN for ci in slope_cis_by_seed)', 'return any(ci["lo"] > SLOPE_MARGIN for ci in slope_cis_by_seed)'),
    ("null void: >= 2 -> > 2", "sum(1 for f in fires if f) >= NULL_VOID_FIRES", "sum(1 for f in fires if f) > NULL_VOID_FIRES"),
    ("null void: any row -> all rows", "return any(sum(1 for f in fires if f) >= NULL_VOID_FIRES for fires in fires_by_row.values())",
     "return all(sum(1 for f in fires if f) >= NULL_VOID_FIRES for fires in fires_by_row.values())"),
    ("constant: DEFICIT_MARGIN 0.02 -> 0.2", "DEFICIT_MARGIN = 0.02 ", "DEFICIT_MARGIN = 0.2 "),
    ("constant: SLOPE_MARGIN 0.01 -> 0.1", "SLOPE_MARGIN = 0.01 ", "SLOPE_MARGIN = 0.1 "),
]


def mutant(old: str, new: str):
    import types
    with open(os.path.join(HERE, "fastneg_lib.py"), encoding="utf-8") as f:
        src = f.read()
    assert src.count(old) == 1, f"mutation anchor not unique / absent: {old!r}"
    m = types.ModuleType("fastneg_lib_mutant")
    m.__file__ = "<mutant>"
    exec(compile(src.replace(old, new), "<mutant>", "exec"), m.__dict__)
    return m


class TestDecisionRule(unittest.TestCase):
    def test_branch_table_is_green_on_the_real_module(self):
        self.assertEqual(L.branch_table_failures(L), [])

    def test_every_mutation_goes_red(self):
        silent = []
        for name, old, new in MUTATIONS:
            if not L.branch_table_failures(mutant(old, new)):
                silent.append(name)
        self.assertEqual(silent, [], f"mutations the branch table did NOT catch: {silent}")

    def test_the_mutation_harness_can_see_a_no_op(self):
        anchor = 'if ci["hi"] < -DEFICIT_MARGIN:'
        self.assertEqual(L.branch_table_failures(mutant(anchor, anchor)), [])         # a replacement that changes nothing stays GREEN, so RED means something

    def test_next_arm_text_for_every_outcome(self):
        for k in ("COVERAGE", "FEATURE", "NO_DEFICIT", "INCONCLUSIVE", "VOID"):
            self.assertIn(k, L.NEXT_ARM)
        self.assertTrue(L.NEXT_ARM["COVERAGE"].startswith("FASTNEG-BANK"))
        self.assertTrue(L.NEXT_ARM["FEATURE"].startswith("PRIVGEOM-PROBE"))
        self.assertTrue(L.NEXT_ARM["NO_DEFICIT"].startswith("P-B is not"))
        self.assertIn("f = 25 %", L.NEXT_ARM["INCONCLUSIVE"])


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# (3) the refit reaches its parameters
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def synth_fit_data(N=48, seed=0):
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=(N, 64, 256)).astype(np.float16)
    p2_w = rng.normal(scale=0.05, size=(6, 256)).astype(np.float32)
    p2_b = rng.normal(scale=0.1, size=6).astype(np.float32)
    z6 = (feat.astype(np.float32) @ p2_w.T + p2_b).astype(np.float32)
    zst = np.ascontiguousarray(z6[..., :2])
    y = rng.choice([0.0, 0.5, 1.0], size=(N, 64, 2), p=[0.15, 0.05, 0.8]).astype(np.float32)
    code = rng.integers(0, 5, (N, 64)).astype(np.uint8)
    return feat, p2_w, p2_b, zst, y, code


class TestRefitReach(unittest.TestCase):
    def setUp(self):
        import torch
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(2)
        self.t = torch
        self.feat, self.p2_w, self.p2_b, self.zst, self.y, self.code = synth_fit_data()

    def _head(self, cold=False):
        return L.LRows(self.p2_w, self.p2_b, cols=(0, 1), cold=cold)

    def test_parameters_receive_a_nonzero_gradient_and_the_zero_control_reads_exactly_zero(self):
        t = self.t
        h = self._head()
        f, zs, yy = t.from_numpy(self.feat[:16]), t.from_numpy(self.zst[:16]), t.from_numpy(self.y[:16])
        g = h.grads(f, zs, yy, None)
        for r in (0, 1):
            self.assertGreater(float(g[r][0].abs().sum()), 1e-6, f"row {r}: w received no gradient")
            self.assertGreater(float(g[r][1].abs().sum()), 1e-6, f"row {r}: b received no gradient")
        g0 = h.grads(f, zs, yy, t.zeros(16, 64))                         # ZERO CONTROL: nothing included must read EXACTLY 0.0 (else a zero could be a broken probe)
        for r in (0, 1):
            self.assertEqual(float(g0[r][0].abs().sum()), 0.0)
            self.assertEqual(float(g0[r][1].abs().sum()), 0.0)

    def test_features_get_no_gradient_and_the_rows_are_independent(self):
        t = self.t
        h = self._head()
        f = t.from_numpy(self.feat[:8]).clone().requires_grad_(True)
        h.grads(f, t.from_numpy(self.zst[:8]), t.from_numpy(self.y[:8]), None)
        self.assertIsNone(f.grad, "the features must be detached")
        y2 = self.y[:8].copy()
        y2[..., 1] = 1.0 - y2[..., 1]
        ga = h.grads(t.from_numpy(self.feat[:8]), t.from_numpy(self.zst[:8]), t.from_numpy(self.y[:8]), None)
        gb = h.grads(t.from_numpy(self.feat[:8]), t.from_numpy(self.zst[:8]), t.from_numpy(y2), None)
        self.assertTrue(t.equal(ga[0][0], gb[0][0]) and t.equal(ga[0][1], gb[0][1]))       # the NC row does not see the DAC label
        self.assertFalse(t.equal(ga[1][0], gb[1][0]))

    def test_mask_identities_all_none_and_partial(self):
        t = self.t
        h = self._head()
        n = 24
        f, zs, yy = t.from_numpy(self.feat[:n]), t.from_numpy(self.zst[:n]), t.from_numpy(self.y[:n])
        code = self.code[:n]
        full = h.grads(f, zs, yy, None)
        ones = h.grads(f, zs, yy, t.ones(n, 64))
        for r in (0, 1):
            self.assertTrue(t.allclose(full[r][0], ones[r][0], rtol=1e-5, atol=1e-7))
        incl_np = L.make_incl(code, np.zeros(code.shape, np.float32) + 0.99, 0.0, stratum=4)       # frac 0: every Q4 entry out
        self.assertEqual(int(((code == 4) & (incl_np == 1.0)).sum()), 0)
        self.assertTrue(((code != 4) == (incl_np == 1.0)).all())
        part = h.grads(f, zs, yy, t.from_numpy(incl_np))
        keep = t.from_numpy(incl_np).double()
        for r in (0, 1):                                                  # the analytic identity: F entries out == the loss over the other entries, computed by hand in float64
            w = t.nn.Parameter(h.p2_w[r].double().clone())
            b = t.nn.Parameter(h.p2_b[r].double().clone())
            z = t.from_numpy(self.zst[:n, :, r]).double() + (f.double() @ (w - h.p2_w[r].double()) + (b - h.p2_b[r].double()))
            l = t.nn.functional.binary_cross_entropy_with_logits(z, yy[..., r].double(), reduction="none")
            ((l * keep).sum() / keep.sum()).backward()
            self.assertTrue(t.allclose(part[r][0].double(), w.grad, rtol=1e-4, atol=1e-6))
        u = L.subsample_priority(code.shape, 0)
        q = L.make_incl(code, u, 0.25)                                    # a PARTIAL truncation is neither the all-in nor the all-out gradient (a switch would be one of the two)
        nq4, nin = int((code == 4).sum()), int(((code == 4) & (q == 1.0)).sum())
        self.assertTrue(0 < nin < nq4)
        mid = h.grads(f, zs, yy, t.from_numpy(q))
        self.assertGreater(float((mid[0][0] - full[0][0]).abs().sum()), 1e-7)
        self.assertGreater(float((mid[0][0] - part[0][0]).abs().sum()), 1e-7)

    def test_subsample_is_nested_and_touches_only_the_named_stratum(self):
        code = self.code
        u = L.subsample_priority(code.shape, 1)
        i25, i50, i100 = (L.make_incl(code, u, f) for f in (0.25, 0.5, 1.0))
        self.assertTrue((i25 <= i50).all() and (i50 <= i100).all())
        self.assertTrue((i100 == 1.0).all())
        self.assertTrue((i25[code != 4] == 1.0).all())
        self.assertTrue(0.15 < float(i25[code == 4].mean()) < 0.35)
        self.assertFalse(np.array_equal(L.subsample_priority(code.shape, 0), L.subsample_priority(code.shape, 1)))
        q2 = L.make_incl(code, u, 0.25, stratum=2)                        # the placebo stratum
        self.assertTrue((q2[code != 2] == 1.0).all() and (q2[code == 2] < 1.0).any())

    def test_fit_is_deterministic_moves_parameters_and_is_the_identity_at_zero_epochs(self):
        idx = np.arange(32)
        kw = dict(seed=0, batch_sets=8)
        w0, b0 = L.fit_rows(self.feat, self.zst, self.y, idx, self.p2_w, self.p2_b, epochs=0, **kw)
        self.assertTrue(np.array_equal(w0, self.p2_w[:2]) and np.array_equal(b0, self.p2_b[:2]))        # C-ID: no step = P2's rows bit for bit
        a = L.fit_rows(self.feat, self.zst, self.y, idx, self.p2_w, self.p2_b, epochs=2, **kw)
        b = L.fit_rows(self.feat, self.zst, self.y, idx, self.p2_w, self.p2_b, epochs=2, **kw)
        c = L.fit_rows(self.feat, self.zst, self.y, idx, self.p2_w, self.p2_b, epochs=2, gen_salt=7919, **kw)
        d = L.fit_rows(self.feat, self.zst, self.y, idx, self.p2_w, self.p2_b, epochs=2, seed=1, batch_sets=8)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))                     # deterministic
        self.assertFalse(np.array_equal(a[0], c[0]))                                                     # the null's two refits differ by the batch order only
        self.assertFalse(np.array_equal(a[0], d[0]))
        self.assertGreater(float(np.abs(a[0] - self.p2_w[:2]).sum()), 1e-4)                              # the parameters moved
        u = L.subsample_priority(self.code.shape, 0)
        e = L.fit_rows(self.feat, self.zst, self.y, idx, self.p2_w, self.p2_b, epochs=2, incl=L.make_incl(self.code, u, 0.25), **kw)
        self.assertFalse(np.array_equal(a[0], e[0]))                                                     # the F-subsample reaches the fit
        wc, bc = L.fit_rows(self.feat, self.zst, self.y, idx, self.p2_w, self.p2_b, epochs=0, cold=True, **kw)
        self.assertEqual(float(np.abs(wc).sum() + np.abs(bc).sum()), 0.0)                                # cold start begins at zero

    def test_score_identity_and_optimizer_levers(self):
        h = self._head()
        for o in h.opts:                                                  # G-DVB analog: the BUILT optimiser reports the declared levers
            g = o.param_groups[0]
            self.assertEqual((g["lr"], g["weight_decay"], tuple(g["betas"]), g["eps"]), (L.LR, L.WD, L.BETAS, L.EPS))
        self.assertEqual((L.EPOCHS, L.BATCH_SETS), (20, 256))
        out = L.score_rows(self.feat, self.zst, np.arange(len(self.feat)), self.p2_w[:2], self.p2_b[:2], self.p2_w, self.p2_b)
        self.assertTrue(np.array_equal(out, self.zst))                    # scoring at P2's rows reproduces the stored logits bit for bit


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# (4) the strata script cannot read labels
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def strip_labels(src: str, dst: str):
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zo:
        for n in zi.namelist():
            if "label" not in n.lower():
                zo.writestr(n, zi.read(n))


class TestStrataIsLabelFree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fastneg_t_")
        self.sy = make_synth(self.tmp, n_logs=6, sets_per_log=6)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_speeds_and_strata_equal_the_planted_truth(self):
        out = os.path.join(self.tmp, "out")
        self.assertEqual(run_strata(self.sy, out), 0)
        z = np.load(os.path.join(out, "speeds.npz"))
        with open(os.path.join(out, "strata.json"), "rb") as f:
            raw = f.read()
        doc = json.loads(raw)
        with open(os.path.join(out, "strata.json.sha256")) as f:
            self.assertEqual(f.read().split()[0], hashlib.sha256(raw).hexdigest())
        tr, S = self.sy["truth"], 6
        for n in range(len(z["pick"])):
            li = self.sy["logs"].index(str(z["key_log"][n]))
            s = 2 * int(str(z["key_token"][n])[5:7]) + int(z["key_step"][n])
            g = li * S + s
            self.assertTrue(np.allclose(z["v"][n], tr["v"][g], atol=1e-6))
            self.assertEqual(int(z["pick"][n]), int(tr["pick"][g]))
            self.assertAlmostEqual(float(z["v_b"][n]), float(tr["v_b"][g]), places=6)
        self.assertEqual(doc["n_sets"], 36)
        self.assertEqual(doc["n_S_sets"], 2)                                                              # the two planted standstill sets
        self.assertEqual(sum(doc["rows_per_stratum"].values()), 36 * 64)
        self.assertEqual(doc["rows_per_stratum"]["S"], 2 * 64)
        self.assertAlmostEqual(doc["control_first_segment_over_later"]["median_ratio"], 1.0, places=5)
        with self.assertRaises(L.CensusError):                                                            # strata are computed ONCE
            run_strata(self.sy, out)

    def test_no_label_member_is_ever_opened_and_a_missing_label_path_is_fine(self):
        L.OPENED.clear()
        out = os.path.join(self.tmp, "out2")
        self.assertEqual(run_strata(self.sy, out, labels_path=os.path.join(self.tmp, "no", "such", "labels6.npz")), 0)
        names = {m for _, m in L.OPENED}
        self.assertTrue(names <= set(L.BANK_MEMBERS) | set(L.STORE_MEMBERS), names)
        self.assertFalse([m for m in names if "label" in m.lower()])
        with open(os.path.join(out, "strata.json")) as f:
            doc = json.load(f)
        self.assertFalse(doc["labels_path_exists_at_run"])
        self.assertEqual(doc["labels_opened"], [])
        import zipfile as zf
        seen, orig_open, orig_load = [], zf.ZipFile.open, np.load                                          # spy on EVERY zip member open and every np.load of a fresh run

        def spy_open(self_, name, *a, **k):
            seen.append(str(getattr(name, "filename", name)))
            return orig_open(self_, name, *a, **k)

        def spy_load(*a, **k):
            seen.append("np.load:" + str(a[0] if a else ""))
            return orig_load(*a, **k)
        zf.ZipFile.open, np.load = spy_open, spy_load
        try:
            run_strata(self.sy, None)
        finally:
            zf.ZipFile.open, np.load = orig_open, orig_load
        self.assertTrue(seen)
        self.assertFalse([s for s in seen if "label" in s.lower()], seen[:5])
        self.assertFalse([s for s in seen if s.startswith("np.load:")], "the strata step must not use np.load at all")

    def test_stripped_copies_give_byte_identical_strata(self):
        full = run_strata(self.sy, None)
        stripped = os.path.join(self.tmp, "stripped")
        os.makedirs(os.path.join(stripped, "bank"))
        os.makedirs(os.path.join(stripped, "sets"))
        for n in os.listdir(self.sy["bank"]):
            if n.endswith(".npz"):
                strip_labels(os.path.join(self.sy["bank"], n), os.path.join(stripped, "bank", n))
        shutil.copy(self.sy["manifest"], os.path.join(stripped, "bank", "MANIFEST.sha256.json"))
        for n in os.listdir(self.sy["stores"]):
            strip_labels(os.path.join(self.sy["stores"], n), os.path.join(stripped, "sets", n))
        for d in ("bank", "sets"):
            for n in os.listdir(os.path.join(stripped, d)):
                if n.endswith(".npz"):
                    with zipfile.ZipFile(os.path.join(stripped, d, n)) as z:
                        self.assertFalse([m for m in z.namelist() if "label" in m.lower()])
        sy2 = dict(self.sy, bank=os.path.join(stripped, "bank"), stores=os.path.join(stripped, "sets"), manifest=os.path.join(stripped, "bank", "MANIFEST.sha256.json"))
        again = run_strata(sy2, None)
        self.assertEqual(full["content_digest"], again["content_digest"])
        self.assertEqual(full["payload"], again["payload"])

    def test_the_reader_refuses_label_members_and_a_mutated_script_goes_red(self):
        p = os.path.join(self.sy["stores"], self.sy["logs"][0] + ".npz")
        for bad in (["labels6"], ["traj3", "Labels6"], ["x_label"]):
            with self.assertRaises(L.LabelAccessError):
                L.read_members(p, bad)
        old = L.STORE_MEMBERS
        try:
            L.STORE_MEMBERS = old + ("labels6",)                           # a MUTANT strata script that asks for the labels
            with self.assertRaises(L.LabelAccessError):
                run_strata(self.sy, None)
        finally:
            L.STORE_MEMBERS = old
        old_b = L.BANK_MEMBERS
        try:
            L.BANK_MEMBERS = old_b + ("labels6",)
            with self.assertRaises(L.LabelAccessError):
                run_strata(self.sy, None)
        finally:
            L.BANK_MEMBERS = old_b

    def test_static_the_strata_script_has_no_np_load_and_opens_files_only_to_write(self):
        import ast
        with open(os.path.join(HERE, "fastneg_strata.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr in ("load", "loadtxt", "genfromtxt", "fromfile") and isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy"):
                    self.fail(f"np.{f.attr} in fastneg_strata.py")
                if isinstance(f, ast.Name) and f.id == "open":
                    mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else None
                    self.assertIn(mode, ("xb", "w"), "the strata script may only OPEN files to WRITE")
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.strip() == "labels6":
                self.fail("the strata script names the labels6 member as a literal")


if __name__ == "__main__":
    unittest.main(verbosity=1)
