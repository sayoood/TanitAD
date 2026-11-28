"""Tests for epr1ub.py on SYNTHETIC banks only (never the real bank).  stdlib unittest (also collected by pytest):

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_epr1ub.py -v

(a) u = 0 is the landed loss bit for bit (loss value, EP-row gradient, the fitted row, the whole OOF read); u > 0 changes them.
(b) the u anchor: reach proof (non-zero where it acts, EXACTLY 0.0 where it must not), the analytic gradient, and MUTATION arms (delete the term, invert / drop the mask, wrong target, wrong
    denominator, ignored weight, a fit path that drops u, a fit path that ignores lr) each of which must make `reachability_check_ub` go RED.
(c) the lr lever: one Adam step moves the row by lr (1e-3, 3e-3, 1e-2), exactly 0.0 at lr 0.
(d) the selection function on canned tables with known answers, plus a mutation battery (<= -> <, all -> any, constraint dropped, lo for hi, argmin, last-max, max for mean) that must each FAIL the battery.
(e) end to end on a synthetic bank: identity arm, grid, selection routed through `select_cstar` (a sentinel proves the caller), both outcome branches, the landed-schema `epr1_result.json` accepted by
    the FROZEN `epr1_dev.check_r1_result`, the C2 stage, the identity precondition (a perturbed reference stops the run before the grid; removing the precondition goes RED).
(f) the path guard: no dev-gate / navtest path, no file access outside the landed `_io_*` functions (AST), the parent pin refuses a tampered epr1.py.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epr1ub as U   # noqa: E402  (loads the LANDED epr1.py from verified bytes as sys.modules['epr1'])
import test_epr1 as T   # noqa: E402  (the landed synthetic-world builder; same epr1 module)

E = U.E
assert T.R is E, "the landed test helpers and epr1ub must share ONE epr1 module"
SRC = os.path.join(HERE, "epr1ub.py")
PY = sys.executable


def opts_ub(w, out, mod=None, **kw):
    mod = mod or U
    base = dict(fraction=10, bank_dir=w["bank"], manifest=w["manifest"], ranked_logs=w["tsv"], p2_ckpt=w["ckpt"], out_dir=out, s0_marker=w["marker"], epochs=2, batch_sets=32,
                min_logs_per_fold=2, ranked_sha256=w["tsv_sha"], expected_n_logs=None, threads=2, reach_n_sets=64, f10_dir=os.path.join(w["root"], "f10_absent"))
    base.update(kw)
    return mod.OptionsUB(**base)


def mutate(src: str, old: str, new: str) -> str:
    assert src.count(old) == 1, f"mutation anchor must occur exactly once: {old!r} x{src.count(old)}"
    return src.replace(old, new)


def load_mut(tmp: str, name: str, edits) -> object:
    """epr1ub.py with textual edits, loaded as a separate module (its HERE points at the real code dir so the verified parent is found)."""
    src = mutate(T._rt(SRC), "HERE = os.path.dirname(os.path.abspath(__file__))", f"HERE = {HERE!r}")
    for old, new in edits:
        src = mutate(src, old, new)
    return T.load_module_from_source(src, tmp, name)


class UBase(T.Base):
    def setup_data(self, **kw):
        o = opts_ub(self.w, self.out(), **kw)
        return o, E._setup(o)

    def parts(self, **kw):
        o, p = self.setup_data(**kw)
        return o, p[8], p[9], p[10]          # opts, W6, b6, data

    def batch(self, data, n=64):
        return E._probe_batch(data, n)

    def parent_ref(self, opts, W6, b6, data):
        """The identity reference computed with the LANDED functions (the synthetic stand-in for the f10 fit)."""
        cfg = E.Cfg(lr=1e-3, anchor=0.1, epochs=opts.epochs, batch_sets=opts.batch_sets)
        draws = E.boot_draws(data.n_logs)
        per_seed, boot, rows = {}, {}, {}
        for sd in opts.seeds:
            r = E.oof_run(data, cfg, sd, W6, b6)
            per_seed[str(sd)] = {k: r[k] for k in U.OOF_KEYS}
            boot[str(sd)] = E.boot_pick_level(r, draws)
            w, b = E.fit_row(data, np.arange(data.n_sets), cfg, sd, W6, b6)
            rows[str(sd)] = E.sha256_bytes(E.row_bytes(w, b))
        return {"source": "parent-synthetic", "result_sha256": None, "per_seed": per_seed, "boot": boot, "row_sha256": rows}


def cfg_ub(**kw):
    base = dict(lr=1e-3, anchor=0.1, epochs=2, batch_sets=32, u=0.1)
    base.update(kw)
    return U.CfgUB(**base)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (a) u = 0 is the landed loss, bit for bit
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestIdentity(UBase):
    def test_a_u0_loss_value_and_gradient_equal_the_landed_loss(self):
        o, W6, b6, data = self.parts()
        batch = self.batch(data)
        for (pw, aw) in ((1.0, 0.1), (1.0, 0.3), (0.5, 0.03)):
            ref = E._probe_grads(W6, b6, batch, E.total_loss, pw, aw)
            got = U._probe_ub(W6, b6, batch, pw, aw, 0.0)
            self.assertTrue(torch.equal(ref["_gw"], got["_gw"]) and torch.equal(ref["_gb"], got["_gb"]), (pw, aw))
        z = E.forward_z6(E.RowHead(W6, b6, probe=False, perturb=True), batch["feat"].float(), batch["z6"])
        l_ref = E.total_loss(z, batch["y"], batch["safe"], batch["top8"], 0.1, 1.0)[0]
        l_ub = U.total_loss_ub(z, batch["y"], batch["safe"], batch["top8"], 0.1, 1.0, 0.0)[0]
        self.assertEqual(float(l_ref), float(l_ub))

    def test_a_u0_fitted_row_is_byte_identical_to_the_landed_fit(self):
        o, W6, b6, data = self.parts()
        c = cfg_ub(u=0.0)
        for sd in (0, 1):
            w0, b0 = E.fit_row(data, np.arange(data.n_sets), c.parent_cfg(), sd, W6, b6)
            w1, b1 = U.fit_ub(data, np.arange(data.n_sets), c, sd, W6, b6)
            self.assertEqual(E.row_bytes(w0, b0), E.row_bytes(w1, b1), sd)

    def test_a_u0_oof_read_equals_the_landed_oof_run_exactly(self):
        o, W6, b6, data = self.parts()
        c = cfg_ub(u=0.0)
        for sd in (0, 1):
            r0, r1 = E.oof_run(data, c.parent_cfg(), sd, W6, b6), U.oof_run_ub(data, c, sd, W6, b6)
            diffs: list = []
            U._deep_diff({k: r1[k] for k in U.OOF_KEYS}, {k: r0[k] for k in U.OOF_KEYS}, "oof", diffs)
            self.assertEqual(diffs, [])
            for k in ("num_log", "den_log", "n_log", "pdn_log", "pdp_log", "un_log", "up_log"):
                self.assertTrue(np.array_equal(r0[k], r1[k]), k)

    def test_a_positive_u_changes_the_row_and_the_read(self):
        o, W6, b6, data = self.parts()
        w0, _ = U.fit_ub(data, np.arange(data.n_sets), cfg_ub(u=0.0), 0, W6, b6)
        w1, _ = U.fit_ub(data, np.arange(data.n_sets), cfg_ub(u=0.3), 0, W6, b6)
        self.assertFalse(np.array_equal(w0, w1))
        r0, r1 = U.oof_run_ub(data, cfg_ub(u=0.0), 0, W6, b6), U.oof_run_ub(data, cfg_ub(u=0.3), 0, W6, b6)
        self.assertNotEqual(r0["pick_level"]["unsafe_rate_new"], r1["pick_level"]["unsafe_rate_new"])

    def test_a_larger_u_pushes_unsafe_logits_down(self):
        """The anchor does what it is for: at the same lr a dominant u pulls the mean sigmoid(EP) on the UNSAFE proposals of the TRAINING sets down (in-sample: the synthetic violations carry no
        feature signal, so only the direction and size of the push are tested, not generalisation)."""
        o, W6, b6, data = self.parts()
        p2w, p2b = np.asarray(W6, np.float32)[E.EP_COL], np.asarray(b6, np.float32)[E.EP_COL:E.EP_COL + 1]
        m = (~data.safe).numpy()
        means, losses = [], []
        for u in (0.0, 10.0):
            w, b = U.fit_ub(data, np.arange(data.n_sets), cfg_ub(lr=3e-3, epochs=4, u=u), 0, W6, b6)
            z6 = E.apply_row(data.z6, data.feat, w, b, p2w, p2b)
            means.append(float((torch.sigmoid(z6[..., E.EP_COL]).numpy() * m).sum() / m.sum()))
            losses.append(float(U.unsafe_anchor_term(z6, data.safe)))
        self.assertGreater(means[0], means[1] + 0.005)
        self.assertGreater(losses[0], losses[1] + 0.05)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (b) + (c) the u anchor and the lr lever: reach proof, analytic gradient, mutation arms
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestReachAndMutation(UBase):
    def test_b_green_and_the_report_says_why(self):
        o, W6, b6, data = self.parts()
        rep = U.reachability_check_ub(W6, b6, self.batch(data), cfg_ub())
        self.assertGreater(rep["u_only"]["ep_row_w"], 0.0)
        self.assertGreater(rep["u_only"]["ep_row_b"], 0.0)
        for k in ("full", "u_only", "zero_control"):
            self.assertEqual(rep[k]["frozen_rows_w"], 0.0, k)
            self.assertEqual(rep[k]["features"], 0.0, k)
        self.assertEqual(rep["zero_control"]["ep_row_w"], 0.0)
        self.assertEqual(rep["u_term_no_unsafe_entry_grad"], {"w": 0.0, "b": 0.0})
        for k in ("u0_gradient_bit_identical_to_landed_loss", "u_changes_full_gradient", "u_term_label_invariant_bit_identical", "u_term_safe_logit_invariant_bit_identical",
                  "u_term_unsafe_logit_changes_gradient", "all_unsafe_full_equals_u_only_bit_identical", "u_reaches_the_optimiser_rows_differ"):
            self.assertTrue(rep[k], k)
        self.assertLessEqual(rep["u_term_value_abs_err_vs_mean_softplus_on_unsafe"], 1e-6)
        steps = rep["one_step_max_abs_move_by_lr"]
        for lr in U.UB_GRID_LR:
            self.assertAlmostEqual(steps[f"{lr:g}"] / lr, 1.0, delta=0.01, msg=str(lr))
        self.assertEqual(steps["0"], 0.0)

    def test_b_analytic_gradient_of_the_u_term(self):
        """d/d(row_w) of mean softplus(z) over unsafe = sum_unsafe sigmoid(z_i) feat_i / n_unsafe  (z_i = stored + feat . (row - p2) + (b - p2b))."""
        o, W6, b6, data = self.parts()
        batch = self.batch(data)
        head = E.RowHead(W6, b6, probe=False, perturb=True)
        feat = batch["feat"].float()
        z = E.forward_z6(head, feat, batch["z6"])
        loss = U.unsafe_anchor_term(z, batch["safe"])
        loss.backward()
        zi = z[..., E.EP_COL].detach().double()
        m = (~batch["safe"]).double()
        sig = torch.sigmoid(zi) * m
        want_w = (sig.unsqueeze(-1) * feat.double()).sum((0, 1)) / m.sum()
        want_b = sig.sum() / m.sum()
        self.assertTrue(torch.allclose(head.row_w.grad.double(), want_w, rtol=1e-4, atol=1e-6))
        self.assertAlmostEqual(float(head.row_b.grad), float(want_b), places=5)

    def test_b_a_loss_that_reads_the_labels_is_caught_by_the_invariance_probe(self):
        """The probe can SEE label dependence: a u term whose target is the banked label changes its gradient when the labels change."""
        o, W6, b6, data = self.parts()
        batch = self.batch(data)

        def grads(y):
            head = E.RowHead(W6, b6, probe=True, perturb=True)
            z = E.forward_z6(head, batch["feat"].clone().float().requires_grad_(True), batch["z6"])
            m = (~batch["safe"]).float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(z[..., E.EP_COL], y, reduction="none")
            ((bce * m).sum() / m.sum()).backward()
            return head.row_w.grad.clone()
        self.assertFalse(torch.equal(grads(batch["y"]), grads(torch.rand(batch["y"].shape, generator=torch.Generator().manual_seed(5)))))

    def _red(self, name, edits, cfg_kw=None):
        o, W6, b6, data = self.parts()
        mut = load_mut(self.tmp, name, edits)
        c = mut.CfgUB(lr=1e-3, anchor=0.1, epochs=2, batch_sets=32, u=0.1)
        with self.assertRaises(E.ReachabilityError, msg=name):
            mut.reachability_check_ub(W6, b6, self.batch(data), c)

    def test_b_mutation_delete_the_u_term_goes_red(self):
        self._red("mut_u_delete", [("loss = loss + u * un", "loss = loss")])

    def test_b_mutation_invert_the_mask_goes_red(self):
        self._red("mut_u_inv_mask", [("m = (~safe).to(bce.dtype)", "m = safe.to(bce.dtype)")])

    def test_b_mutation_drop_the_mask_goes_red(self):
        self._red("mut_u_drop_mask", [("m = (~safe).to(bce.dtype)", "m = torch.ones_like(bce)")])

    def test_b_mutation_wrong_target_goes_red(self):
        self._red("mut_u_target", [("torch.full_like(z, UNSAFE_TARGET)", "torch.full_like(z, 1.0)")])

    def test_b_mutation_wrong_denominator_goes_red(self):
        self._red("mut_u_denom", [("return (bce * m).sum() / m.sum().clamp(min=1.0)", "return (bce * m).sum() / m.numel()")])

    def test_b_mutation_ignored_weight_goes_red(self):
        self._red("mut_u_weight", [("loss = loss + u * un", "loss = loss + 1.0 * un")])

    def test_b_mutation_fit_path_drops_u_goes_red(self):
        self._red("mut_u_fit", [("loss_fn=make_loss_fn(cfg.u)", "loss_fn=None")])

    def test_c_mutation_fit_path_ignores_lr_goes_red(self):
        self._red("mut_lr", [("E.fit_row(data, train_idx, cfg, seed, W6, b6, loss_fn", "E.fit_row(data, train_idx, dataclasses.replace(cfg, lr=1e-3), seed, W6, b6, loss_fn")])

    def test_c_lr_zero_leaves_the_row_exactly_unchanged(self):
        o, W6, b6, data = self.parts()
        w, b = U.fit_ub(data, np.arange(data.n_sets), cfg_ub(lr=0.0, u=0.3), 0, W6, b6)
        self.assertTrue(np.array_equal(w, np.asarray(W6)[E.EP_COL]))
        self.assertTrue(np.array_equal(b, np.asarray(b6)[E.EP_COL:E.EP_COL + 1]))

    def test_b_reach_refuses_a_batch_without_unsafe_entries(self):
        o, W6, b6, data = self.parts()
        batch = dict(self.batch(data))
        batch["safe"] = torch.ones_like(batch["safe"])
        with self.assertRaises(E.ReachabilityError):
            U.reachability_check_ub(W6, b6, batch, cfg_ub())

    def test_b_run_aborts_if_the_ub_reach_is_red(self):
        o, W6, b6, data = self.parts()
        mut = load_mut(self.tmp, "mut_run_abort", [("loss = loss + u * un", "loss = loss")])
        oo = opts_ub(self.w, self.out(), mod=mut)
        oo.ident_ref = self.parent_ref(oo, W6, b6, data)
        with self.assertRaises(E.ReachabilityError):
            mut.run_ub(oo)
        self.assertFalse(os.path.exists(os.path.join(oo.out_dir, "epr1ub_result.json")))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (d) selection
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def srow(i, hi, d, pooled=0.9, lo=None):
    """hi, d: per-seed lists (hi = upper bound of the unsafe-rate delta, d = dPDMS point)."""
    ps = {}
    for s in (0, 1):
        ps[str(s)] = {"pooled": pooled, "boot": {"delta_unsafe_rate": {"point": hi[s] - 0.001, "lo": (hi[s] - 0.002) if lo is None else lo, "hi": hi[s]}, "delta_pdms_x100": {"point": d[s]}}}
    return {"id": f"c{i}", "per_seed": ps}


def battery(mod) -> list:
    """Canned tables with known answers.  -> the names of the cases the module gets WRONG (empty = correct)."""
    bad = []

    def chk(name, got, want):
        if got != want:
            bad.append(f"{name}: got {got!r} want {want!r}")
    seeds = (0, 1)
    nan = float("nan")
    # the highest-dPDMS INFEASIBLE config is not chosen
    chk("infeasible_not_chosen", mod.select_cstar([srow(0, [0.01, 0.01], [5, 5]), srow(1, [-0.001, -0.002], [0.1, 0.2])], seeds)["id"], "c1")
    # one infeasible seed makes the config infeasible
    chk("one_seed_infeasible", mod.select_cstar([srow(0, [-0.01, 0.002], [9, 9]), srow(1, [-0.001, -0.001], [0.5, 0.5])], seeds)["id"], "c1")
    # hi exactly 0.0 is feasible; +epsilon is not
    chk("hi_zero_feasible", mod.select_cstar([srow(0, [0.0, 0.0], [1, 1])], seeds)["id"], "c0")
    chk("hi_eps_infeasible", mod.select_cstar([srow(0, [1e-12, 0.0], [1, 1])], seeds)["status"], "NONE-FEASIBLE")
    chk("hi_nan_infeasible", mod.select_cstar([srow(0, [nan, 0.0], [1, 1])], seeds)["status"], "NONE-FEASIBLE")
    # ties go to the first in grid order
    chk("tie_first", mod.select_cstar([srow(0, [-1, -1], [0.3, 0.3]), srow(1, [-1, -1], [0.3, 0.3]), srow(2, [-1, -1], [0.3, 0.3])], seeds)["id"], "c0")
    chk("tie_list", mod.select_cstar([srow(0, [-1, -1], [0.3, 0.3]), srow(1, [-1, -1], [0.3, 0.3])], seeds)["tied_at_max"], ["c0", "c1"])
    # none feasible: no c*
    r = mod.select_cstar([srow(0, [0.01, 0.01], [1, 1]), srow(1, [0.5, 0.5], [2, 2])], seeds)
    chk("none_feasible", (r["status"], r["id"], r["index"]), ("NONE-FEASIBLE", None, None))
    # the sign of dPDMS does not gate (R7): a feasible config with a negative dPDMS is still chosen
    chk("negative_dpdms_still_chosen", mod.select_cstar([srow(0, [-1, -1], [-0.5, -0.5])], seeds)["id"], "c0")
    # the objective is the MEAN over seeds
    chk("mean_not_max", mod.select_cstar([srow(0, [-1, -1], [1.0, -1.0]), srow(1, [-1, -1], [0.1, 0.1])], seeds)["id"], "c1")
    chk("max_is_largest_mean", mod.select_cstar([srow(0, [-1, -1], [0.1, 0.1]), srow(1, [-1, -1], [0.4, 0.2]), srow(2, [-1, -1], [0.2, 0.2])], seeds)["id"], "c1")
    # the constraint reads the UPPER bound, not the lower
    chk("hi_not_lo", mod.select_cstar([srow(0, [0.01, 0.01], [9, 9], lo=-0.5), srow(1, [-1, -1], [0.1, 0.1])], seeds)["id"], "c1")
    # the parent's criterion is concordance, ignoring feasibility (kept only so the gate can refuse it)
    chk("pooled_criterion", mod.select_cstar([srow(0, [0.5, 0.5], [0, 0], pooled=0.99), srow(1, [-1, -1], [9, 9], pooled=0.5)], seeds, "pooled_oof")["id"], "c0")
    return bad


class TestSelection(unittest.TestCase):
    def test_d_battery_is_green_on_the_real_function(self):
        self.assertEqual(battery(U), [])

    def test_d_feasibility_flags(self):
        self.assertTrue(U.is_feasible(srow(0, [-0.1, 0.0], [0, 0]), (0, 1)))
        self.assertFalse(U.is_feasible(srow(0, [-0.1, 1e-9], [0, 0]), (0, 1)))

    def test_d_mutation_battery_each_mutation_fails_it(self):
        tmp = tempfile.mkdtemp(prefix="epr1ub_selmut_")
        try:
            muts = {
                "le_to_lt": [("return all(h <= SEL_BOUND for h in his)", "return all(h < SEL_BOUND for h in his)")],
                "all_to_any": [("return all(h <= SEL_BOUND for h in his)", "return any(h <= SEL_BOUND for h in his)")],
                "constraint_dropped": [("feas = [i for i, r in enumerate(rows) if is_feasible(r, seeds)]", "feas = list(range(len(rows)))")],
                "lo_for_hi": [('["delta_unsafe_rate"]["hi"] for sd in seeds]', '["delta_unsafe_rate"]["lo"] for sd in seeds]')],
                "argmin": [("j = int(np.argmax(sc))", "j = int(np.argmin(sc))")],
                "last_max": [("j = int(np.argmax(sc))", "j = len(sc) - 1 - int(np.argmax(sc[::-1]))")],
                "max_for_mean": [("return float(np.mean([row[", "return float(np.max([row[")],
                "bound_flipped": [("SEL_BOUND = 0.0 ", "SEL_BOUND = 1.0 ")],
            }
            for i, (nm, edits) in enumerate(muts.items()):
                mod = load_mut(tmp, f"mut_sel_{i}_{nm}", edits)
                self.assertNotEqual(battery(mod), [], f"mutation {nm} was NOT caught by the battery")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (e) end to end on a synthetic bank
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def indep_select(grid: list, seeds=(0, 1)) -> str | None:
    """An independent re-statement of the registered rule, over a result's grid rows."""
    best, best_i = None, None
    for i, r in enumerate(grid):
        ok = all(r["per_seed"][str(s)]["boot"]["delta_unsafe_rate"]["hi"] <= 0.0 for s in seeds)
        if not ok:
            continue
        sc = sum(r["per_seed"][str(s)]["boot"]["delta_pdms_x100"]["point"] for s in seeds) / len(seeds)
        if best is None or sc > best:
            best, best_i = sc, i
    return None if best_i is None else grid[best_i]["id"]


class TestEndToEnd(UBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cache = {}

    def run_ub(self, tag, bound=None, sentinel=None, ident_tamper=False):
        """Run the full stage once per tag (cached).  `bound` temporarily replaces SEL_BOUND, `sentinel` replaces select_cstar."""
        if tag in self.cache:
            return self.cache[tag]
        o, W6, b6, data = self.parts()
        oo = opts_ub(self.w, self.out(f"e2e_{tag}"))
        ref = self.parent_ref(oo, W6, b6, data)
        if ident_tamper:
            ref["row_sha256"]["0"] = "0" * 64
        oo.ident_ref = ref
        old_b, old_s = U.SEL_BOUND, U.select_cstar
        try:
            if bound is not None:
                U.SEL_BOUND = bound
            if sentinel is not None:
                U.select_cstar = sentinel
            try:
                res = U.run_ub(oo)
                err = None
            except Exception as ex:           # noqa: BLE001
                res, err = None, ex
        finally:
            U.SEL_BOUND, U.select_cstar = old_b, old_s
        self.cache[tag] = (oo, res, err)
        return self.cache[tag]

    def test_e_identity_arm_and_grid_shape(self):
        oo, res, err = self.run_ub("all_feasible", bound=float("inf"))
        self.assertIsNone(err)
        self.assertTrue(res["identity_arm"]["bit_identical"])
        self.assertEqual(res["identity_arm"]["n_differences"], 0)
        self.assertEqual([g["id"] for g in res["grid"]], [c.id() for c in U.grid_ub()])
        self.assertEqual(len(res["grid"]), 9)
        for g in res["grid"]:
            for sd in ("0", "1"):
                self.assertEqual(set(g["per_seed"][sd]["boot"]), {"delta_pdms_x100", "delta_unsafe_rate"})
                self.assertEqual(g["per_seed"][sd]["boot"]["delta_unsafe_rate"]["B"], 10000)

    def test_e_selection_matches_the_independent_rule_and_the_caller_routes_through_select_cstar(self):
        oo, res, err = self.run_ub("all_feasible", bound=float("inf"))
        self.assertEqual(res["selection"]["status"], "FEASIBLE")
        self.assertEqual(res["selection"]["c_star"]["id"], indep_select([{**g, "per_seed": {s: {**v, "boot": {**v["boot"], "delta_unsafe_rate": {**v["boot"]["delta_unsafe_rate"], "hi": -1.0}}}
                                                                                        for s, v in g["per_seed"].items()}} for g in res["grid"]]))
        sent = lambda rows, seeds, criterion=U.CRITERION_REG: {"status": "FEASIBLE", "index": 4, "id": rows[4]["id"], "score": 0.123, "feasible_ids": [rows[4]["id"]],     # noqa: E731
                                                              "tied_at_max": [rows[4]["id"]], "criterion": criterion}
        oo2, res2, err2 = self.run_ub("sentinel", sentinel=sent)
        self.assertIsNone(err2)
        self.assertEqual(res2["selection"]["c_star"]["id"], "lr0.003_u0.1")                  # the run reports exactly what select_cstar said
        self.assertTrue(os.path.exists(os.path.join(oo2.out_dir, "ep_row_seed0.npy")))

    def test_e_mutation_bypassing_select_cstar_is_caught_by_the_sentinel(self):
        o, W6, b6, data = self.parts()
        mut = load_mut(self.tmp, "mut_bypass", [("sel = select_cstar(rows, opts.seeds, opts.criterion)", "sel = _bypass(rows, opts.seeds)")])
        mut._bypass = lambda rows, seeds: {"status": "FEASIBLE", "index": 0, "id": rows[0]["id"], "score": 0.0, "feasible_ids": [rows[0]["id"]], "tied_at_max": [rows[0]["id"]], "criterion": "x"}
        sent = lambda rows, seeds, criterion=None: {"status": "FEASIBLE", "index": 4, "id": rows[4]["id"], "score": 0.123, "feasible_ids": [rows[4]["id"]], "tied_at_max": [rows[4]["id"]], "criterion": "s"}   # noqa: E731
        mut.select_cstar = sent
        oo = opts_ub(self.w, self.out(), mod=mut)
        oo.ident_ref = self.parent_ref(oo, W6, b6, data)
        res = mut.run_ub(oo)
        self.assertNotEqual(res["selection"]["c_star"]["id"], "lr0.003_u0.1", "the sentinel WAS honoured under the bypass mutation: the sentinel test cannot detect a bypass")

    def test_e_none_feasible_writes_no_c_star_no_rows_no_compat_doc(self):
        oo, res, err = self.run_ub("none", bound=float("-inf"))
        self.assertIsNone(err)
        self.assertEqual(res["selection"]["status"], "NONE-FEASIBLE")
        self.assertIsNone(res["selection"]["c_star"])
        self.assertIsNone(res["final_rows"])
        self.assertTrue(os.path.exists(os.path.join(oo.out_dir, "epr1ub_result.json")))
        self.assertTrue(os.path.exists(os.path.join(oo.out_dir, "epr1ub_result.sha256")))
        for n in ("epr1_result.json", "ep_row_seed0.npy", "ep_row_seed1.npy"):
            self.assertFalse(os.path.exists(os.path.join(oo.out_dir, n)), n)

    def test_e_feasible_writes_rows_and_a_compat_doc_with_registered_sha(self):
        oo, res, err = self.run_ub("all_feasible", bound=float("inf"))
        out = oo.out_dir
        side = T._rt(os.path.join(out, "epr1_result.sha256")).split()[0]
        self.assertEqual(side, E.sha256_bytes(T._rb(os.path.join(out, "epr1_result.json"))))
        side_ub = T._rt(os.path.join(out, "epr1ub_result.sha256")).split()[0]
        self.assertEqual(side_ub, E.sha256_bytes(T._rb(os.path.join(out, "epr1ub_result.json"))))
        doc = T._jl(os.path.join(out, "epr1_result.json"))
        self.assertEqual((doc["schema"], doc["stage"]), (E.SCHEMA, "R1"))
        self.assertEqual(doc["epr1ub"]["derived_from_sha256"], side_ub)
        for sd in ("0", "1"):
            self.assertEqual(doc["final_rows"][sd]["sha256"], E.sha256_bytes(T._rb(os.path.join(out, doc["final_rows"][sd]["file"]))))
        cs = doc["c_star"]
        star = U.CfgUB(lr=cs["lr"], anchor=cs["anchor"], u=cs["u"], epochs=doc["params"]["epochs"], batch_sets=doc["params"]["batch_sets"])
        self.assertEqual(star.sha256(), cs["ub_config_sha256"])
        self.assertEqual(E.Cfg(lr=cs["lr"], anchor=cs["anchor"], epochs=doc["params"]["epochs"], batch_sets=doc["params"]["batch_sets"]).sha256(), cs["config_sha256"])
        self.assertEqual(doc["code_sha256"], U.PARENT_EPR1_SHA256)

    def test_e_a_perturbed_identity_reference_stops_the_run_before_the_grid(self):
        oo, res, err = self.run_ub("tamper", ident_tamper=True)
        self.assertIsInstance(err, U.IdentityError)
        self.assertTrue(os.path.exists(os.path.join(oo.out_dir, "epr1ub_ident_FAIL.json")))
        for n in ("epr1ub_result.json", "epr1_result.json", "ep_row_seed0.npy"):
            self.assertFalse(os.path.exists(os.path.join(oo.out_dir, n)), n)

    def test_e_mutation_removing_the_identity_precondition_is_caught(self):
        o, W6, b6, data = self.parts()
        mut = load_mut(self.tmp, "mut_noident", [("ident = identity_arm(data, opts, W6, b6, ref, draws)", 'ident = {"bit_identical": True}')])
        oo = opts_ub(self.w, self.out(), mod=mut)
        ref = self.parent_ref(oo, W6, b6, data)
        ref["row_sha256"]["0"] = "0" * 64
        oo.ident_ref = ref
        res = mut.run_ub(oo)                         # the run COMPLETES: the property the real test asserts (it must stop) is lost
        self.assertIn("selection", res)

    def test_e_ident_check_sees_each_kind_of_difference(self):
        o, W6, b6, data = self.parts()
        draws = E.boot_draws(data.n_logs)
        base = self.parent_ref(o, W6, b6, data)
        U.identity_arm(data, o, W6, b6, base, draws)                                           # green
        import copy
        for what in ("per_seed", "boot", "row"):
            ref = copy.deepcopy(base)
            if what == "per_seed":
                ref["per_seed"]["1"]["pick_level"]["unsafe_rate_new"] += 1e-12
            elif what == "boot":
                ref["boot"]["0"]["delta_unsafe_rate"]["hi"] += 1e-12
            else:
                ref["row_sha256"]["1"] = ref["row_sha256"]["1"][:-1] + ("0" if ref["row_sha256"]["1"][-1] != "0" else "1")
            with self.assertRaises(U.IdentityError, msg=what):
                U.identity_arm(data, o, W6, b6, ref, draws)

    def test_e_seam_the_frozen_dev_applier_accepts_the_derived_result_and_the_c2_rows(self):
        import epr1_dev as D
        sent = lambda rows, seeds, criterion=U.CRITERION_REG: {"status": "FEASIBLE", "index": 4, "id": rows[4]["id"], "score": 0.1, "feasible_ids": [rows[4]["id"]],     # noqa: E731
                                                              "tied_at_max": [rows[4]["id"]], "criterion": criterion}
        oo, res, err = self.run_ub("sentinel", sentinel=sent)
        out = oo.out_dir
        doc = T._jl(os.path.join(out, "epr1_result.json"))
        doc["inputs"]["pins_overridden"] = []                                                  # the synthetic world overrides pins by construction; the shape is what is tested
        seam = os.path.join(self.tmp, "seam")
        os.makedirs(seam, exist_ok=True)
        for sd in ("0", "1"):
            shutil.copy(os.path.join(out, doc["final_rows"][sd]["file"]), seam)
        T._jd(doc, os.path.join(seam, "epr1_result.json"))
        sha = E.sha256_bytes(T._rb(os.path.join(seam, "epr1_result.json")))
        dopts = D.DevOptions(run_dir=seam, registered_result_sha256=sha, p2_ckpt=self.w["ckpt"], out_dir=seam)
        guard = dopts.guard()
        W6, b6 = D.load_p2_head(guard, self.w["ckpt"])
        got = D.check_r1_result(guard, dopts, W6, b6)
        self.assertEqual(got["star"].lr, 3e-3)
        self.assertEqual(got["doc"]["c_star"]["u"], 0.1)
        self.assertEqual(set(got["rows"]), {"0", "1"})
        # the same doc with c* at lr = 1e-3 is refused WITHOUT a waiver and accepted WITH one naming that c* (the frozen E1 text)
        sent0 = lambda rows, seeds, criterion=U.CRITERION_REG: {"status": "FEASIBLE", "index": 1, "id": rows[1]["id"], "score": 0.1, "feasible_ids": [rows[1]["id"]],     # noqa: E731
                                                               "tied_at_max": [rows[1]["id"]], "criterion": criterion}
        oo0, res0, err0 = self.run_ub("sentinel_lo", sentinel=sent0)
        doc0 = T._jl(os.path.join(oo0.out_dir, "epr1_result.json"))
        doc0["inputs"]["pins_overridden"] = []
        seam0 = os.path.join(self.tmp, "seam0")
        os.makedirs(seam0, exist_ok=True)
        for sd in ("0", "1"):
            shutil.copy(os.path.join(oo0.out_dir, doc0["final_rows"][sd]["file"]), seam0)
        T._jd(doc0, os.path.join(seam0, "epr1_result.json"))
        sha0 = E.sha256_bytes(T._rb(os.path.join(seam0, "epr1_result.json")))
        d0 = D.DevOptions(run_dir=seam0, registered_result_sha256=sha0, p2_ckpt=self.w["ckpt"], out_dir=seam0)
        with self.assertRaises(D.DevRefusal):
            D.check_r1_result(d0.guard(), d0, W6, b6)
        wv = os.path.join(seam0, "waiver.json")
        T._jd({"waiver_of": "E1", "reason": "R7", "decided_by": "MM_RULINGS_HEADROWS R7", "c_star_config_sha256": doc0["c_star"]["config_sha256"]}, wv)
        d0w = D.DevOptions(run_dir=seam0, registered_result_sha256=sha0, p2_ckpt=self.w["ckpt"], out_dir=seam0, mm_edge_waiver=wv)
        self.assertEqual(D.check_r1_result(d0w.guard(), d0w, W6, b6)["star"].lr, 1e-3)

    def test_e_c2_stage_end_to_end_and_the_dev_applier_accepts_it(self):
        import epr1_dev as D
        oo, res, err = self.run_ub("all_feasible", bound=float("inf"))
        co = opts_ub(self.w, oo.out_dir)
        c2 = U.run_c2_ub(co)
        self.assertEqual(c2["stage"], "C2")
        for sd in ("0", "1"):
            real = T._rb(os.path.join(oo.out_dir, "ep_row_seed%s.npy" % sd))
            shuf = T._rb(os.path.join(oo.out_dir, c2["final_rows"][sd]["file"]))
            self.assertNotEqual(real, shuf)
        doc = T._jl(os.path.join(oo.out_dir, "epr1_result.json"))
        dopts = D.DevOptions(run_dir=oo.out_dir, registered_result_sha256="0" * 64, p2_ckpt=self.w["ckpt"], out_dir=oo.out_dir)
        got = D.check_c2_files(dopts.guard(), dopts, {"doc": doc})
        self.assertEqual(set(got["rows"]), {"0", "1"})
        with self.assertRaises(E.BankError):                              # never overwritten
            U.run_c2_ub(co)

    def test_e_second_run_refuses_to_overwrite(self):
        oo, res, err = self.run_ub("all_feasible", bound=float("inf"))
        o, W6, b6, data = self.parts()
        again = opts_ub(self.w, oo.out_dir)
        again.ident_ref = self.parent_ref(again, W6, b6, data)
        with self.assertRaises(E.BankError):
            U.run_ub(again)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (f) guard, CLI, parent pin
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestGuardAndPins(UBase):
    def test_f_the_parent_pin_is_the_landed_blob(self):
        self.assertEqual(U.PARENT_EPR1_SHA256, "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8")
        self.assertEqual(E.sha256_bytes(T._rb(os.path.join(HERE, "epr1.py"))), U.PARENT_EPR1_SHA256)

    def test_f_a_tampered_parent_is_refused_before_it_is_executed(self):
        d = tempfile.mkdtemp(prefix="epr1ub_tamper_")
        try:
            shutil.copy(SRC, d)
            T._wb(os.path.join(d, "epr1.py"), T._rb(os.path.join(HERE, "epr1.py")) + b"\n# tampered\n")
            cp = subprocess.run([PY, "-I", os.path.join(d, "epr1ub.py"), "run", "--fraction", "10"], capture_output=True, timeout=300,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})
            self.assertNotEqual(cp.returncode, 0)
            self.assertIn(b"refusing to import unverified code", cp.stderr)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_f_forbidden_tables_are_refused_and_legit_paths_allowed(self):
        o = opts_ub(self.w, self.out())
        g = o.guard()
        for tok in ("heldout", "navtest", "dev_gate", "table_p2"):
            with self.assertRaises(E.PathGuardError):
                g.check_read(os.path.join(o.bank_dir, f"x_{tok}_y.sets.npz"))
            with self.assertRaises(E.PathGuardError):
                g.check_write(os.path.join(o.resolved_out_dir(), f"x_{tok}_y.json"))
        g.check_read(o.manifest)
        g.check_read(os.path.join(o.f10_dir, "epr1_result.json"))
        g.check_write(os.path.join(o.resolved_out_dir(), "epr1ub_result.json"))
        with self.assertRaises(E.PathGuardError):
            g.check_read(os.path.join(tempfile.gettempdir(), "elsewhere.bin"))

    def test_f_ast_no_file_system_access_outside_the_landed_io_functions(self):
        tree = ast.parse(T._rt(SRC))
        banned_attr = {"listdir", "makedirs", "stat", "exists", "isfile", "isdir", "load", "save", "savez", "write_text", "read_text", "read_bytes", "write_bytes", "remove", "rename",
                       "replace_file", "walk", "copyfile", "rmtree", "mkdir", "fdopen"}
        offenders = []
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if fn.name == "_load_parent":
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.Call):
                    f = n.func
                    if isinstance(f, ast.Name) and f.id == "open":
                        offenders.append((fn.name, "open"))
                    if isinstance(f, ast.Attribute) and f.attr in banned_attr and not (isinstance(f.value, ast.Name) and f.value.id == "E"):
                        offenders.append((fn.name, f.attr))
        top = [n for n in tree.body if isinstance(n, ast.Expr) or isinstance(n, ast.Assign)]
        self.assertEqual(offenders, [])
        self.assertTrue(top)

    def test_f_cli_registered_argv_builds_the_registered_options(self):
        a = U.build_parser().parse_args(["run", "--fraction", "10"])
        o = U.options_from_args(a)
        self.assertEqual((tuple(o.lr_grid), tuple(o.u_grid), o.anchor, o.criterion, tuple(o.seeds), o.epochs, o.batch_sets), (U.UB_GRID_LR, U.UB_GRID_U, 0.1, "feasible_oof_dpdms", (0, 1), 10, 256))
        self.assertEqual(o.pins_overridden(), [])
        self.assertEqual([c.id() for c in o.grid()], ["lr0.001_u0.03", "lr0.001_u0.1", "lr0.001_u0.3", "lr0.003_u0.03", "lr0.003_u0.1", "lr0.003_u0.3", "lr0.01_u0.03", "lr0.01_u0.1", "lr0.01_u0.3"])
        a2 = U.build_parser().parse_args(["run", "--fraction", "10", "--u-grid", "0.05", "--criterion", "pooled_oof"])
        self.assertEqual(sorted(U.options_from_args(a2).pins_overridden()), ["criterion", "u_grid"])

    def test_f_config_sha_changes_with_u_and_the_parent_sha_does_not(self):
        a, b = cfg_ub(u=0.0), cfg_ub(u=0.1)
        self.assertNotEqual(a.sha256(), b.sha256())
        self.assertEqual(a.parent_cfg().sha256(), b.parent_cfg().sha256())
        self.assertEqual(U.grid_ub()[1].parent_cfg().sha256(), E.Cfg(lr=1e-3, anchor=0.1).sha256())


if __name__ == "__main__":
    unittest.main(verbosity=2)
