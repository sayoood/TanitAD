"""Tests for srttc_r1.py on SYNTHETIC banks only (never the real bank).  stdlib unittest (also collected by pytest):

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_srttc_r1.py -v

(a) C1 reachability: non-zero gradient on the TTC row; EXACTLY 0.0 on every other row and on the features; the zero-weight control reads 0.0; a leaky loss is caught.
(b) MUTATION: the line that adds the pairwise term is deleted from the source and the check must go RED (also: the detach line, the optimiser step).
(c) analytic identity: lr = 0 reproduces P2's picks bit-for-bit.
(d) fold rule (rank mod 5), log grouping (no log in two folds, no log in train and out-of-fold), nested LC prefixes.
(e) path guard: dev-gate / navtest tables are refused, at runtime and by an AST scan of every file-system call.
(f) determinism: same seed -> same c* and same weights sha256 (and a different seed differs: the probe can read a difference).
(g) analytic gradients of the two loss terms; the A2.4 call logic; bootstrap draws equal zero_tail.Boot's.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import srttc_r1 as R   # noqa: E402

SRC_PATH = os.path.join(HERE, "srttc_r1.py")


def _rb(p):
    with open(p, "rb") as f:
        return f.read()


def _rt(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def _wb(p, data):
    with open(p, "wb") as f:
        f.write(data)


def _wt(p, text):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _jd(obj, p):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# a synthetic world: bank units + manifest + ranked TSV + fake P2 checkpoint
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def sha256_file(p):
    return hashlib.sha256(_rb(p)).hexdigest()


def make_world(root, n_logs=40, seed=0, ckpt_W6=None, label_order=None, shuffle_ttc=False, unit_logs=3):
    rng = np.random.default_rng(seed)
    names = [f"2021.01.{i % 28 + 1:02d}.00.00.{i:02d}_veh-{i % 7}_{i:05d}_{i + 9:05d}" for i in range(n_logs)]
    keys = {n: R.log_key(n) for n in names}
    order = sorted(names, key=lambda n: keys[n])
    rank = {n: i for i, n in enumerate(order)}
    u = rng.normal(size=R.FEAT_DIM).astype(np.float32)
    u /= np.linalg.norm(u)
    W6 = (rng.normal(size=(R.N_COMP, R.FEAT_DIM)) * 0.05).astype(np.float32)
    W6[R.TTC_COL] = 0.02 * u + rng.normal(size=R.FEAT_DIM).astype(np.float32) * 0.05         # a mediocre P2 row
    b6 = (rng.normal(size=R.N_COMP) * 0.1 + 0.5).astype(np.float32)
    os.makedirs(os.path.join(root, "bank"), exist_ok=True)
    os.makedirs(os.path.join(root, "plan"), exist_ok=True)
    ck = os.path.join(root, "p2", "model_final.pt")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    cW = W6 if ckpt_W6 is None else ckpt_W6
    torch.save({"format": "refe-full-v1", "model": {"score_head.weight": torch.from_numpy(cW), "score_head.bias": torch.from_numpy(b6), "other.w": torch.zeros(3)}, "meta": {}}, ck)
    # TSV: first half of the logs -> 0.10, next quarter 0.25, ...
    tsv = ["\t".join(["rank", "log", "key_sha256", "fold", "first_fraction", "n_fit_tokens", "n_fit_sets"])]
    for n in order:
        r = rank[n]
        ff = "0.10" if r < n_logs * 0.5 else "0.25" if r < n_logs * 0.75 else "0.50" if r < n_logs * 0.9 else "1.00"
        tsv.append("\t".join([str(r), n, keys[n], str(r % 5), ff, "0", "0"]))
    tsv_path = os.path.join(root, "plan", "ranked_fit_logs.tsv")
    _wt(tsv_path, "\n".join(tsv) + "\n")
    # per-log sets
    units, files = [], {}
    chunk = []
    for li, n in enumerate(order):
        chunk.append(n)
        if len(chunk) == unit_logs or li == len(order) - 1:
            logs, toks, feats, logits, top8s, labs = [], [], [], [], [], []
            for lg in chunk:
                for si in range(int(rng.integers(8, 16))):
                    f = rng.normal(size=(R.N_PROPOSALS, R.FEAT_DIM)).astype(np.float16)
                    lo = (f.astype(np.float32) @ W6.T + b6).astype(np.float32)
                    lb = np.zeros((R.N_PROPOSALS, R.N_COMP), np.float32)
                    lb[:, 0] = rng.random(R.N_PROPOSALS) < 0.9
                    lb[:, 1] = rng.random(R.N_PROPOSALS) < 0.9
                    lb[:, 2] = rng.random(R.N_PROPOSALS)
                    lb[:, 3] = (f.astype(np.float32) @ u * 1.5 + rng.normal(size=R.N_PROPOSALS) * 0.7) > 0
                    lb[:, 4] = rng.random(R.N_PROPOSALS) < 0.8
                    lb[:, 5] = 1.0
                    if shuffle_ttc:
                        lb[:, 3] = rng.permutation(lb[:, 3])
                    logs.append(lg)
                    toks.append(f"tok{len(toks):06d}")
                    feats.append(f)
                    logits.append(lo)
                    top8s.append(R.top8_np(lo))
                    labs.append(lb)
            S = len(logs)
            nm = f"L{hashlib.sha256(''.join(chunk).encode()).hexdigest()[:10]}.sets.npz"
            p = os.path.join(root, "bank", nm)
            np.savez(p, log=np.array(logs), token=np.array(toks), step=np.zeros(S, np.int32), rank=np.zeros(S, np.int32), ckpt_step=np.zeros(S, np.int64), feat=np.stack(feats),
                     logits=np.stack(logits), top8=np.stack(top8s), labels6=np.stack(labs), label_order=np.array(label_order or list(R.LABEL_ORDER)), rule=np.array("navsim_v1"),
                     ckpt=np.array(ck.replace("\\", "/")))
            files[f"work/bank/{nm}"] = {"path": p.replace("\\", "/"), "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
            units.append(p)
            chunk = []
    man = os.path.join(root, "bank", "MANIFEST.sha256.json")
    _jd({"written": "synthetic", "n_files": len(files), "files": files}, man)
    marker = os.path.join(root, "S0_DONE")
    _wt(marker, "done\n")
    return {"root": root, "bank": os.path.join(root, "bank"), "manifest": man, "tsv": tsv_path, "ckpt": ck, "marker": marker, "W6": W6, "b6": b6, "u": u, "units": units, "order": order,
            "tsv_sha": sha256_file(tsv_path)}


def opts_for(w, out, **kw):
    base = dict(fraction=10, bank_dir=w["bank"], manifest=w["manifest"], ranked_logs=w["tsv"], p2_ckpt=w["ckpt"], out_dir=out, s0_marker=w["marker"], epochs=2, batch_sets=32,
                min_logs_per_fold=2, ranked_sha256=w["tsv_sha"], expected_n_logs=None, threads=2, reach_n_sets=64)
    base.update(kw)
    return R.Options(**base)


def load_module_from_source(src: str, tmpdir: str, name: str):
    p = os.path.join(tmpdir, name + ".py")
    _wt(p, src)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m                # dataclasses (with `from __future__ import annotations`) looks the module up here
    spec.loader.exec_module(m)
    return m


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="srttc_r1_test_")
        cls.w = make_world(os.path.join(cls.tmp, "w"))
        cls.counter = 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def out(self, name=None):
        type(self).counter += 1
        return os.path.join(self.tmp, "out", name or f"o{self.counter}")

    def setup_data(self, **kw):
        o = opts_for(self.w, self.out(), **kw)
        parts = R._setup(o)
        return o, parts

    def batch(self, data, n=64):
        return R._probe_batch(data, n)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (a) C1 reachability  +  (b) mutation
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestReachabilityAndMutation(Base):
    def test_a_green_and_controls(self):
        o, parts = self.setup_data()
        W6, b6, data = parts[8], parts[9], parts[10]
        rep = R.reachability_check(W6, b6, self.batch(data), R.Cfg(lr=1e-3, anchor=0.1))
        for name in ("full", "pair_only", "anchor_only"):
            self.assertGreater(rep[name]["ttc_row_w"], 0.0, name)                     # non-zero on the TTC row
            self.assertGreater(rep[name]["ttc_row_b"], 0.0, name)
        for name in ("full", "pair_only", "anchor_only", "zero_control"):
            self.assertEqual(rep[name]["frozen_rows_w"], 0.0, name)                   # EXACTLY 0.0 on all six frozen rows
            self.assertEqual(rep[name]["frozen_rows_b"], 0.0, name)
            self.assertEqual(rep[name]["features"], 0.0, name)                        # EXACTLY 0.0 on the features
            self.assertEqual(rep[name]["frozen_rows_w_per_row"], [0.0] * 6, name)
        self.assertEqual(rep["zero_control"]["ttc_row_w"], 0.0)                        # the control that must read 0.0
        self.assertEqual(rep["zero_control"]["ttc_row_b"], 0.0)
        self.assertGreater(rep["update_moved_ttc_row_max_abs"], 0.0)

    def test_a_probe_can_read_nonzero_on_a_frozen_row(self):
        """Positive control of the probe: a loss that leaks into another column MUST show a non-zero frozen-row gradient (else a zero reading would mean nothing)."""
        o, parts = self.setup_data()
        W6, b6, data = parts[8], parts[9], parts[10]

        def leaky(z6, y3, idx8, anchor_w, pair_w=1.0):
            loss, p, a = R.total_loss(z6, y3, idx8, anchor_w, pair_w)
            return loss + 0.1 * z6[..., 2].pow(2).mean(), p, a                       # reads column 2 = the EP row

        with self.assertRaises(R.ReachabilityError) as cm:
            R.reachability_check(W6, b6, self.batch(data), R.Cfg(lr=1e-3, anchor=0.1), loss_fn=leaky)
        self.assertIn("FROZEN row", str(cm.exception))

    def _mutated(self, old: str, new: str, marker: str, name: str):
        src = _rt(SRC_PATH)
        self.assertEqual(src.count(old), 1, f"mutation anchor {old!r} must occur exactly once (else the mutation test is vacuous)")
        self.assertIn(marker, src)
        d = os.path.join(self.tmp, "mut")
        os.makedirs(d, exist_ok=True)
        return load_module_from_source(src.replace(old, new), d, name)

    def test_b_mutation_delete_pairwise_term_goes_red(self):
        o, parts = self.setup_data()
        W6, b6, data = parts[8], parts[9], parts[10]
        M = self._mutated("    loss = loss + pair_w * pair             # MUTATION-TARGET-PAIR: the pairwise term", "    pass", "MUTATION-TARGET-PAIR", "srttc_r1_mut_pair")
        batch = self.batch(data)
        with self.assertRaises(M.ReachabilityError) as cm:
            M.reachability_check(W6, b6, batch, M.Cfg(lr=1e-3, anchor=0.1))
        msg = str(cm.exception)
        print("\n  [mutation: pairwise line deleted] ->", msg[:300])
        self.assertIn("pair_only", msg)
        R.reachability_check(W6, b6, batch, R.Cfg(lr=1e-3, anchor=0.1))                 # the unmutated module on the same batch is GREEN

    def test_b_mutation_remove_detach_goes_red(self):
        o, parts = self.setup_data()
        W6, b6, data = parts[8], parts[9], parts[10]
        M = self._mutated("    feat = feat.detach().float()                                                  # MUTATION-TARGET-DETACH: the trunk and the features are frozen",
                          "    feat = feat.float()", "MUTATION-TARGET-DETACH", "srttc_r1_mut_detach")
        with self.assertRaises(M.ReachabilityError) as cm:
            M.reachability_check(W6, b6, self.batch(data), M.Cfg(lr=1e-3, anchor=0.1))
        print("\n  [mutation: feature detach removed] ->", str(cm.exception)[:200])
        self.assertIn("features received a gradient", str(cm.exception))

    def test_b_mutation_cut_optimiser_goes_red(self):
        o, parts = self.setup_data()
        W6, b6, data = parts[8], parts[9], parts[10]
        M = self._mutated("            opt.step()", "            pass", "opt.step()", "srttc_r1_mut_step")
        with self.assertRaises(M.ReachabilityError) as cm:
            M.reachability_check(W6, b6, self.batch(data), M.Cfg(lr=1e-3, anchor=0.1))
        self.assertIn("update:", str(cm.exception))

    def test_a_run_aborts_if_reachability_is_red(self):
        """The check runs INSIDE run_r1 (before any fit): a mutated module's run_r1 must abort, not fit."""
        M = self._mutated("    loss = loss + pair_w * pair             # MUTATION-TARGET-PAIR: the pairwise term", "    pass", "MUTATION-TARGET-PAIR", "srttc_r1_mut_run")
        out = self.out()
        with self.assertRaises(M.ReachabilityError):
            M.run_r1(opts_for(self.w, out))
        self.assertFalse(os.path.exists(os.path.join(out, "r1_result.json")))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (c) analytic identity
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestIdentity(Base):
    def test_c_zero_lr_reproduces_p2_picks_bit_for_bit(self):
        o, parts = self.setup_data()
        W6, b6, data = parts[8], parts[9], parts[10]
        cfg0 = R.Cfg(lr=0.0, anchor=0.1, epochs=3, batch_sets=16)
        w, b = R.fit_row(data, np.arange(data.n_sets), cfg0, 0, W6, b6)
        self.assertTrue(np.array_equal(w, W6[R.TTC_COL]) and np.array_equal(b, b6[R.TTC_COL:R.TTC_COL + 1]), "lr = 0 must leave the row bit-identical")
        st = R.score_sets(data, np.arange(data.n_sets), w, b, W6[R.TTC_COL], b6[R.TTC_COL:R.TTC_COL + 1])
        self.assertTrue(np.array_equal(st["picks"], data.top8[:, 0].numpy()), "picks must equal the bank's stored pick (top8[:,0]) on every set")
        self.assertTrue(np.array_equal(st["picks"], st["pick_p2"]))
        r = R.oof_run(data, cfg0, 0, W6, b6)
        ref = R.p2_reference(data, W6, b6)
        self.assertEqual(r["oof_picks_changed"], 0)
        self.assertEqual(r["pooled"], ref["pooled"])                                  # exactly equal: same pairs, same logits

    def test_c_control_nonzero_lr_changes_the_row_and_picks(self):
        o, parts = self.setup_data()
        W6, b6, data = parts[8], parts[9], parts[10]
        cfg = R.Cfg(lr=1e-2, anchor=0.1, epochs=20, batch_sets=16)
        w, b = R.fit_row(data, np.arange(data.n_sets), cfg, 0, W6, b6)
        self.assertGreater(float(np.abs(w - W6[R.TTC_COL]).max()), 1e-3)
        st = R.score_sets(data, np.arange(data.n_sets), w, b, W6[R.TTC_COL], b6[R.TTC_COL:R.TTC_COL + 1])
        self.assertGreater(int((st["picks"] != st["pick_p2"]).sum()), 0, "the identity test would be vacuous if a real fit changed no pick")

    def test_c_learning_improves_and_shuffled_labels_do_not(self):
        o, parts = self.setup_data()
        W6, b6, data = parts[8], parts[9], parts[10]
        cfg = R.Cfg(lr=1e-2, anchor=0.1, epochs=30, batch_sets=16)
        ref = R.p2_reference(data, W6, b6)["pooled"]
        got = R.oof_run(data, cfg, 0, W6, b6)["pooled"]
        # calibrated on seeds 0-2 of this generator: P2 row 0.52-0.53 -> fitted 0.90-0.91 (gain 0.37-0.39)
        self.assertGreater(got - ref, 0.2, f"learnable synthetic problem: OOF {got:.4f} vs P2 {ref:.4f}")
        # C2-style negative control: TTC labels shuffled across the proposals of each set -> nothing to learn
        w2 = make_world(os.path.join(self.tmp, "w_shuf"), shuffle_ttc=True)
        o2 = opts_for(w2, self.out())
        p2 = R._setup(o2)
        ref2 = R.p2_reference(p2[10], p2[8], p2[9])["pooled"]
        got2 = R.oof_run(p2[10], cfg, 0, p2[8], p2[9])["pooled"]
        # calibrated: shuffled-label OOF concordance of the fit reads 0.48-0.52 (chance) over seeds 0-2, while the real problem reads 0.90; P2's own row reads 0.47-0.50 (noise +-0.03)
        self.assertLess(abs(got2 - 0.5), 0.06, f"shuffled labels: OOF {got2:.4f} vs P2 {ref2:.4f}: nothing to learn, so the fit must sit at chance")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (d) folds and grouping
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestFolds(Base):
    def test_d_fold_rule(self):
        for r in range(0, 3000):
            self.assertEqual(R.fold_of_rank(r), r % 5)

    def test_d_tsv_checks_refuse_tampering(self):
        g = R.PathGuard(read_dirs=[self.tmp])
        rows, sha = R.load_ranked_logs(g, self.w["tsv"], self.w["tsv_sha"], None)
        self.assertEqual([r["fold"] for r in rows], [i % 5 for i in range(len(rows))])
        lines = _rt(self.w["tsv"]).splitlines()
        for tamper, label in ((lambda c: c.__setitem__(3, str((int(c[3]) + 1) % 5)), "fold"), (lambda c: c.__setitem__(2, "0" * 64), "key")):
            ln = [x.split("\t") for x in lines]
            tamper(ln[3])
            p = os.path.join(self.tmp, f"tampered_{label}.tsv")
            _wt(p, "\n".join("\t".join(c) for c in ln) + "\n")
            with self.assertRaises(R.BankError, msg=label):
                R.load_ranked_logs(g, p, None, None)
        with self.assertRaises(R.BankError):                                           # the sha256 pin
            R.load_ranked_logs(g, self.w["tsv"], "0" * 64, None)

    def test_d_every_log_in_exactly_one_fold_and_cv_never_mixes_a_log(self):
        o, parts = self.setup_data()
        W6, b6, data = parts[8], parts[9], parts[10]
        # every log -> exactly one fold; every set carries its log's fold
        folds_of_log: dict = {}
        for li, f in zip(data.log_idx, data.fold):
            folds_of_log.setdefault(int(li), set()).add(int(f))
        self.assertTrue(all(len(s) == 1 for s in folds_of_log.values()), "a log appears in two folds")
        self.assertEqual(set(folds_of_log), set(range(data.n_logs)))
        for li, s in folds_of_log.items():
            self.assertEqual(next(iter(s)), R.fold_of_rank(int(data.log_rank[li])))
        # the out-of-fold partition covers every set exactly once
        cover = np.zeros(data.n_sets, int)
        for k in range(R.N_FOLDS):
            cover[data.fold == k] += 1
        self.assertTrue((cover == 1).all())
        # the real CV path: capture the train indices handed to fit_row
        seen = []
        orig = R.fit_row

        def spy(d, tr, cfg, seed, W6_, b6_, loss_fn=None):
            seen.append(np.array(tr))
            return orig(d, tr, cfg, seed, W6_, b6_, loss_fn)
        R.fit_row = spy
        try:
            R.oof_run(data, R.Cfg(lr=1e-3, anchor=0.1, epochs=1, batch_sets=32), 0, W6, b6)
        finally:
            R.fit_row = orig
        self.assertEqual(len(seen), R.N_FOLDS)
        for k, tr in enumerate(seen):
            tr_logs, oof_logs = set(data.log_idx[tr].tolist()), set(data.log_idx[data.fold == k].tolist())
            self.assertFalse(tr_logs & oof_logs, f"fold {k}: a log is in both the training part and the out-of-fold part")
            self.assertTrue(all(R.fold_of_rank(int(data.log_rank[i])) != k for i in tr_logs))
            self.assertEqual(len(tr), int(np.isin(data.log_idx, list(tr_logs)).sum()), "all sets of a training log are in, none of a held-out log")

    def test_d_lc_prefixes_are_nested_in_key_order(self):
        o, parts = self.setup_data()
        W6, b6, data = parts[8], parts[9], parts[10]
        seen = {}
        orig = R.fit_row

        def spy(d, tr, cfg, seed, W6_, b6_, loss_fn=None):
            seen.setdefault(cur[0], []).append(sorted(set(d.log_idx[tr].tolist())))
            return orig(d, tr, cfg, seed, W6_, b6_, loss_fn)
        R.fit_row = spy
        try:
            for p in R.LC_PREFIXES:
                cur = [p]
                R.oof_run(data, R.Cfg(lr=1e-3, anchor=0.1, epochs=1, batch_sets=64), 0, W6, b6, train_pct=p)
        finally:
            R.fit_row = orig
        for k in range(R.N_FOLDS):
            tr_all = [i for i in range(data.n_logs) if data.log_fold[i] != k]            # selected logs are in key order already
            for p in R.LC_PREFIXES:
                want = tr_all[:max(1, int(np.ceil(p / 100.0 * len(tr_all))))]
                self.assertEqual(seen[p][k], sorted(want), f"fold {k} p {p}")
            self.assertTrue(set(seen[25][k]) <= set(seen[50][k]) <= set(seen[100][k]))

    def test_d_default_min_logs_per_fold_refuses_a_small_fit_set(self):
        with self.assertRaises(R.BankError) as cm:
            R._setup(opts_for(self.w, self.out(), min_logs_per_fold=R.MIN_LOGS_PER_FOLD))
        self.assertIn("A2.2", str(cm.exception))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (e) path guard
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestPathGuard(Base):
    BAD = ("p2-heldout-table/gate_table.json", "table_p2_full.npz", "navtest/x.npz", "dev_gate/y.npz", "heldout_pdm/z.jsonl", "dgdump/q.npz", "fidelity.json", "featcheck.json",
           "s1_read_cache/labels/x/heldout_pdm/a.jsonl", "NAVTEST_tokens.npz", "p2ho_dump.npz")

    def _guard(self):
        return R.PathGuard(read_dirs=[self.tmp], read_files=[], write_dirs=[os.path.join(self.tmp, "o")])

    def test_e_forbidden_tables_are_refused_even_inside_an_allowed_root(self):
        g = self._guard()
        for rel in self.BAD:
            p = os.path.join(self.tmp, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            _wb(p, b"x")
            for fn in (R._io_read_bytes, R._io_sha256, R._io_npz, R._io_exists, R._io_stat, R._io_torch_load):
                with self.assertRaises(R.PathGuardError, msg=f"{fn.__name__} {rel}"):
                    fn(g, p)
        with self.assertRaises(R.PathGuardError):
            R._io_listdir(g, os.path.join(self.tmp, "navtest"))
        with self.assertRaises(R.PathGuardError):
            R._io_write_bytes(g, os.path.join(self.tmp, "o", "navtest_out.bin"), b"x")

    def test_e_traversal_and_outside_are_refused_and_a_legit_path_is_allowed(self):
        g = R.PathGuard(read_dirs=[self.w["bank"]], read_files=[self.w["tsv"]])
        os.makedirs(os.path.join(self.tmp, "navtest"), exist_ok=True)
        _wb(os.path.join(self.tmp, "navtest", "t.bin"), b"x")
        with self.assertRaises(R.PathGuardError):
            R._io_read_bytes(g, os.path.join(self.w["bank"], "..", "..", "navtest", "t.bin"))      # ../ traversal into a forbidden dir
        _wb(os.path.join(self.tmp, "plain.bin"), b"x")
        with self.assertRaises(R.PathGuardError):
            R._io_read_bytes(g, os.path.join(self.tmp, "plain.bin"))                                  # not forbidden, but outside the allow-list
        with self.assertRaises(R.PathGuardError):
            R._io_write_bytes(g, os.path.join(self.w["bank"], "w.bin"), b"x")                         # no write dir at all
        # CONTROL: the guard is not refusing everything
        self.assertTrue(R._io_exists(g, self.w["tsv"]))
        self.assertGreater(len(R._io_listdir(g, self.w["bank"])), 0)
        self.assertEqual(len(R._io_sha256(g, self.w["manifest"])), 64)

    def test_e_run_refuses_a_bank_or_manifest_named_like_a_forbidden_table(self):
        bad_bank = os.path.join(self.tmp, "p2-heldout-table", "bank")
        shutil.copytree(self.w["bank"], bad_bank)
        with self.assertRaises(R.PathGuardError):
            R.run_r1(opts_for(self.w, self.out(), bank_dir=bad_bank, manifest=os.path.join(bad_bank, "MANIFEST.sha256.json")))
        with self.assertRaises(R.PathGuardError):
            R.run_r1(opts_for(self.w, self.out(), p2_ckpt=os.path.join(self.tmp, "table_p2_full.npz")))
        with self.assertRaises(R.PathGuardError):
            R.run_r1(opts_for(self.w, os.path.join(self.tmp, "out", "navtest_out")))

    def test_e_ast_every_filesystem_call_lives_in_an_io_function(self):
        tree = ast.parse(_rt(SRC_PATH))
        FS_NAMES = {"open", "np.load", "numpy.load", "torch.load", "torch.save", "np.savez", "np.savez_compressed", "os.listdir", "os.scandir", "os.stat", "os.makedirs", "os.walk",
                    "os.path.exists", "os.path.isfile", "os.path.isdir", "os.path.getsize", "os.replace", "os.rename", "os.remove", "os.unlink", "glob.glob", "shutil.copy", "shutil.copyfile",
                    "shutil.move", "shutil.rmtree"}
        FS_ATTRS = {"read_bytes", "read_text", "write_bytes", "write_text", "tofile", "fromfile"}

        def dotted(n):
            if isinstance(n, ast.Name):
                return n.id
            if isinstance(n, ast.Attribute):
                b = dotted(n.value)
                return f"{b}.{n.attr}" if b else n.attr
            return ""
        bad, n_io = [], 0

        def visit(node, fn):
            nonlocal n_io
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = node.name
            if isinstance(node, ast.Call):
                name = dotted(node.func)
                hit = name in FS_NAMES or (isinstance(node.func, ast.Attribute) and node.func.attr in FS_ATTRS)
                if hit:
                    if fn and fn.startswith("_io_"):
                        n_io += 1
                    else:
                        bad.append((name, node.lineno, fn))
            for c in ast.iter_child_nodes(node):
                visit(c, fn)
        visit(tree, None)
        self.assertEqual(bad, [], f"file-system access outside an _io_* function: {bad}")
        self.assertGreater(n_io, 5, "the AST scan found almost no _io_ calls: the scanner is vacuous")
        # CONTROL: the scanner does catch a violation
        probe = ast.parse("def f(p):\n    return open(p).read()\n")
        found = []
        for nd in ast.walk(probe):
            if isinstance(nd, ast.Call) and dotted(nd.func) in FS_NAMES:
                found.append(nd)
        self.assertEqual(len(found), 1)

    def test_e_forbidden_token_list_covers_the_registered_tables(self):
        for t in ("table_p2_full", "navtest", "heldout_pdm", "p2-heldout-table", "gate_table", "dgdump", "fidelity", "featcheck", "s1_read_cache"):
            self.assertTrue(R.PathGuard.forbidden(f"C:/x/{t}/y"), t)
        self.assertFalse(R.PathGuard.forbidden("C:/Users/Admin/srttc_s0/work/bank/L1b95fda46f.sets.npz"))
        self.assertFalse(R.PathGuard.forbidden(R.DEFAULT_RANKED))
        self.assertFalse(R.PathGuard.forbidden(R.DEFAULT_P2_CKPT))
        self.assertFalse(R.PathGuard.forbidden(R.DEFAULT_MANIFEST))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# bank integrity
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestBankIntegrity(Base):
    def _copy_world(self, name):
        dst = os.path.join(self.tmp, name)
        shutil.copytree(os.path.join(self.tmp, "w"), dst)
        w = dict(self.w)
        w.update({"root": dst, "bank": os.path.join(dst, "bank"), "manifest": os.path.join(dst, "bank", "MANIFEST.sha256.json"), "tsv": os.path.join(dst, "plan", "ranked_fit_logs.tsv"),
                  "ckpt": os.path.join(dst, "p2", "model_final.pt"), "marker": os.path.join(dst, "S0_DONE")})
        return w

    def test_manifest_sha256_is_verified_before_use(self):
        w = self._copy_world("w_tamper")
        unit = sorted(n for n in os.listdir(w["bank"]) if n.endswith(".sets.npz"))[0]
        p = os.path.join(w["bank"], unit)
        b = bytearray(_rb(p))
        b[-5] ^= 0x01
        _wb(p, bytes(b))
        with self.assertRaises(R.BankError) as cm:
            R.run_r1(opts_for(w, self.out()))
        self.assertIn("sha256", str(cm.exception))

    def test_unit_missing_from_manifest_is_refused(self):
        w = self._copy_world("w_nomanifest")
        doc = _jl(w["manifest"])
        doc["files"].pop(next(iter(doc["files"])))
        _jd(doc, w["manifest"])
        with self.assertRaises(R.BankError) as cm:
            R._setup(opts_for(w, self.out()))
        self.assertIn("not in the manifest", str(cm.exception))

    def test_wrong_label_order_is_refused(self):
        w = make_world(os.path.join(self.tmp, "w_lo"), label_order=["nc", "dac", "ep", "c", "ttc", "ddc"])
        with self.assertRaises(R.BankError) as cm:
            R._setup(opts_for(w, self.out()))
        self.assertIn("label_order", str(cm.exception))

    def test_wrong_p2_row_is_refused(self):
        rng = np.random.default_rng(5)
        W = self.w["W6"].copy()
        W[R.TTC_COL] = rng.normal(size=R.FEAT_DIM).astype(np.float32) * 0.05
        w = make_world(os.path.join(self.tmp, "w_wrongrow"), ckpt_W6=W)            # same bank, a checkpoint whose TTC row is not the one that produced the logits
        with self.assertRaises(R.BankError) as cm:
            R._setup(opts_for(w, self.out()))
        self.assertIn("does not reproduce", str(cm.exception))

    def test_s0_marker_is_required_and_result_is_never_overwritten(self):
        w = self._copy_world("w_marker")
        os.remove(w["marker"])
        with self.assertRaises(R.BankError) as cm:
            R.run_r1(opts_for(w, self.out()))
        self.assertIn("S0 is not done", str(cm.exception))
        _wt(w["marker"], "done")
        out = self.out()
        R.run_r1(opts_for(w, out))
        with self.assertRaises(R.BankError) as cm:
            R.run_r1(opts_for(w, out))
        self.assertIn("ONCE", str(cm.exception))

    def _drop_unit_of(self, w, log):
        unit = f"L{hashlib.sha256(log.encode()).hexdigest()[:10]}.sets.npz"           # make_world(unit_logs=1) names a unit after its single log
        os.remove(os.path.join(w["bank"], unit))
        doc = _jl(w["manifest"])
        for k in [k for k in doc["files"] if k.endswith(unit)]:
            doc["files"].pop(k)
        _jd(doc, w["manifest"])

    def test_missing_selected_log_needs_an_explicit_replacement_decision(self):
        w = make_world(os.path.join(self.tmp, "w_missing"), unit_logs=1)
        gone, repl = w["order"][0], w["order"][25]                                   # rank 0 is in S_10 (ranks 0-19); rank 25 is a bank log outside S_10
        self._drop_unit_of(w, gone)
        with self.assertRaises(R.BankError) as cm:                                    # (1) never auto-replaced: S0 left a hole, the run stops
            R._setup(opts_for(w, self.out()))
        self.assertIn("no set in the bank", str(cm.exception))
        rf = os.path.join(self.tmp, "repl_ok.json")                                    # (2) one replacement = 5 % of 20 selected logs: admitted, and listed
        _wt(rf, json.dumps({"dropped": [gone], "added": [repl]}))
        parts = R._setup(opts_for(w, self.out(), replacements_file=rf))
        sel, rep = parts[5], parts[6]
        self.assertNotIn(gone, [r["log"] for r in sel])
        self.assertIn(repl, [r["log"] for r in sel])
        self.assertEqual(rep["dropped"], [gone])
        self.assertEqual(rep["added"], [repl])
        self.assertEqual([r["rank"] for r in sel], sorted(r["rank"] for r in sel))
        self.assertEqual(len(sel), 20)
        data = parts[10]                                                               # the added log keeps ITS OWN fold (rank mod 5), not the dropped log's
        li = [r["log"] for r in sel].index(repl)
        self.assertEqual(int(data.log_fold[li]), R.fold_of_rank(25))
        w2 = make_world(os.path.join(self.tmp, "w_missing3"), unit_logs=1)             # (3) more than 5 % replaced: STOP (A2.1 item 7)
        for lg in w2["order"][:3]:
            self._drop_unit_of(w2, lg)
        rf2 = os.path.join(self.tmp, "repl_many.json")
        _wt(rf2, json.dumps({"dropped": w2["order"][:3], "added": w2["order"][25:28]}))
        with self.assertRaises(R.BankError) as cm:
            R._setup(opts_for(w2, self.out(), replacements_file=rf2))
        self.assertIn("5 %", str(cm.exception))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (f) determinism, end to end;  (g) analytic gradients, call logic, bootstrap draws, CLI defaults
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def strip_meta(res):
    r = json.loads(json.dumps(res))
    r.pop("run_meta", None)
    return r


class TestEndToEnd(Base):
    def test_f_determinism_same_seed_same_cstar_and_weights_sha(self):
        a = R.run_r1(opts_for(self.w, self.out()))
        b = R.run_r1(opts_for(self.w, self.out()))
        self.assertEqual(strip_meta(a), strip_meta(b))
        self.assertEqual(a["c_star"]["id"], b["c_star"]["id"])
        for sd in ("0", "1"):
            self.assertEqual(a["final_rows"][sd]["sha256"], b["final_rows"][sd]["sha256"])
        self.assertEqual(len(a["grid"]), 9)
        self.assertEqual(sorted((g["lr"], g["anchor"]) for g in a["grid"]), sorted((lr, an) for lr in R.GRID_LR for an in R.GRID_ANCHOR))
        # CONTROL: a different seed pair changes the weights (the probe can read a difference)
        c = R.run_r1(opts_for(self.w, self.out(), seeds=(2, 3)))
        self.assertNotEqual(a["final_rows"]["0"]["sha256"], c["final_rows"]["2"]["sha256"])
        self.assertIn("seeds", c["inputs"]["pins_overridden"])
        # the written weights file hashes to what the result says
        o = opts_for(self.w, self.out())
        r = R.run_r1(o)
        for sd, ent in r["final_rows"].items():
            self.assertEqual(sha256_file(os.path.join(o.resolved_out_dir(), ent["file"])), ent["sha256"])
            arr = np.load(os.path.join(o.resolved_out_dir(), ent["file"]))
            self.assertEqual(arr.shape, (257,))
            self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(r["inputs"]["bank_manifest_sha256"], sha256_file(self.w["manifest"]))
        self.assertEqual(r["code_sha256"], sha256_file(SRC_PATH))
        self.assertEqual(r["inputs"]["p2_row_sha256"], R.sha256_bytes(R.row_bytes(self.w["W6"][R.TTC_COL], self.w["b6"][R.TTC_COL:R.TTC_COL + 1])))

    def test_lc_end_to_end_and_p100_reproduces_r1(self):
        o = opts_for(self.w, self.out())
        r1 = R.run_r1(o)
        lc = R.run_lc(o)
        self.assertEqual(lc["prefixes_pct"], [25, 50, 100])
        self.assertIn(lc["call"]["call"], ("RISING", "FLAT", "INCONCLUSIVE"))
        self.assertTrue(all(lc["p100_reproduces_r1"].values()), "LC p = 100 % must reproduce R1's score at c* (same fit, same seed)")
        for sd in ("0", "1"):
            d = lc["delta_100_minus_50"][sd]
            self.assertLessEqual(d["lo"], d["point"] + 1e-12)
            self.assertGreaterEqual(d["hi"], d["point"] - 1e-12)
            self.assertEqual(d["B"], R.BOOT_B)
        self.assertEqual(lc["c_star"]["id"], r1["c_star"]["id"])
        with self.assertRaises(FileExistsError):                                       # lc_result.json is written once
            R.run_lc(o)


class TestLossAndCalls(Base):
    def test_g_pairwise_gradient_is_analytic(self):
        torch.manual_seed(0)
        head = R.RowHead(np.zeros((6, R.FEAT_DIM), np.float32), np.zeros(6, np.float32))
        feat = torch.randn(1, 64, R.FEAT_DIM)
        z6s = torch.zeros(1, 64, 6)
        y3 = torch.zeros(1, 64)
        y3[0, 5] = 1.0                                                                 # contender 5 is safe, the rest are violating
        idx8 = torch.arange(8).reshape(1, 8)
        idx8[0, 0] = 5
        idx8[0, 5] = 0
        # head.row_w = 0 -> z = 0 for all; pairs: (5 safe) vs 7 violating contenders
        z6 = R.forward_z6(head, feat, z6s)
        loss, pair, anchor = R.total_loss(z6, y3, idx8, anchor_w=0.0)
        loss.backward()
        d = np.log(2.0)                                                                # softplus(0)
        self.assertAlmostEqual(float(pair), d, places=6)
        # dL/dw = mean over the 7 pairs of  -sigma(-(zi-zj)) (f_i - f_j) = -0.5 * mean_j (f_5 - f_j)
        f5 = feat[0, 5]
        others = [int(j) for j in idx8[0] if int(j) != 5]
        want = -0.5 * (f5 - feat[0, others].mean(0))
        self.assertTrue(torch.allclose(head.row_w.grad, want, atol=1e-6), float((head.row_w.grad - want).abs().max()))
        self.assertAlmostEqual(float(head.row_b.grad), 0.0, places=6)                  # the pair term is shift invariant

    def test_g_anchor_gradient_is_analytic(self):
        torch.manual_seed(1)
        head = R.RowHead(np.zeros((6, R.FEAT_DIM), np.float32), np.zeros(6, np.float32))
        feat = torch.randn(2, 64, R.FEAT_DIM)
        y3 = (torch.rand(2, 64) > 0.5).float()
        z6 = R.forward_z6(head, feat, torch.zeros(2, 64, 6))
        idx8 = torch.arange(8).repeat(2, 1)
        loss, _p, anc = R.total_loss(z6, y3, idx8, anchor_w=1.0, pair_w=0.0)
        loss.backward()
        want_w = ((torch.sigmoid(torch.zeros(2, 64)) - y3).unsqueeze(-1) * feat).mean((0, 1))
        want_b = (0.5 - y3).mean()
        self.assertTrue(torch.allclose(head.row_w.grad, want_w, atol=1e-6))
        self.assertAlmostEqual(float(head.row_b.grad), float(want_b), places=6)
        self.assertAlmostEqual(float(anc), np.log(2.0), places=6)

    def test_g_lc_call_logic(self):
        mk = lambda lo, hi: {"lo": lo, "hi": hi, "point": (lo + hi) / 2}
        self.assertEqual(R.lc_call({"0": mk(0.002, 0.03), "1": mk(0.001, 0.04)}, "flat")["call"], "RISING")
        self.assertEqual(R.lc_call({"0": mk(-0.004, 0.006), "1": mk(-0.002, 0.008)}, "flat")["call"], "FLAT")
        self.assertEqual(R.lc_call({"0": mk(-0.01, 0.03), "1": mk(-0.02, 0.02)}, "flat")["call"], "INCONCLUSIVE")
        self.assertEqual(R.lc_call({"0": mk(0.002, 0.03), "1": mk(-0.004, 0.006)}, "flat")["call"], "INCONCLUSIVE")          # the two seeds disagree
        both = {"0": mk(0.001, 0.008), "1": mk(0.002, 0.009)}                                                                  # inside (0, 0.010): RISING and FLAT
        self.assertTrue(R.lc_call(both, "flat")["both_true"])
        self.assertEqual(R.lc_call(both, "flat")["call"], "FLAT")
        self.assertEqual(R.lc_call(both, "rising")["call"], "RISING")
        self.assertEqual(R.lc_call({"0": mk(-0.01, 0.0099), "1": mk(-0.01, 0.01)}, "flat")["call"], "INCONCLUSIVE")            # the upper bound must be strictly below 0.010

    def test_g_bootstrap_draws_equal_zero_tail_boot(self):
        d = R.boot_draws(37, B=500)
        self.assertTrue(np.array_equal(d, np.random.default_rng(20260927).integers(0, 37, size=(500, 37))))
        zt = os.path.join(HERE, "zero_tail.py")
        if os.path.exists(zt):
            import zero_tail as ZT
            b = ZT.Boot(np.arange(37), 37, B=500, seed=R.BOOT_SEED)
            self.assertTrue(np.array_equal(b.draws, d), "draws must equal the registered estimator's (zero_tail.Boot)")
            num, den = np.random.default_rng(0).random(37), np.random.default_rng(1).random(37) + 1
            ref = ZT.Boot(np.arange(37), 37, B=500, seed=R.BOOT_SEED).ratio(num, den)
            mine = R.paired_delta(num, np.zeros(37), den, d)
            self.assertAlmostEqual(ref["point"], mine["point"], places=12)
            self.assertAlmostEqual(ref["lo"], mine["lo"], places=12)
            self.assertAlmostEqual(ref["hi"], mine["hi"], places=12)

    def test_g_cli_defaults_enable_every_mechanism(self):
        """The 'default=0.0 disables a wired mechanism' class: the defaults must be live."""
        a = R.build_parser().parse_args(["run", "--fraction", "10"])
        o = R.options_from_args(a)
        self.assertGreater(o.epochs, 0)
        self.assertGreater(o.batch_sets, 0)
        self.assertEqual(o.seeds, (0, 1))
        self.assertEqual(o.criterion, "pooled_oof")
        self.assertEqual(o.pins_overridden(), [])
        cfgs = R.grid(o.epochs, o.batch_sets)
        self.assertEqual([(c.lr, c.anchor) for c in cfgs], [(lr, an) for lr in (1e-4, 3e-4, 1e-3) for an in (0.03, 0.1, 0.3)])
        self.assertTrue(all(c.anchor > 0 and c.lr > 0 and c.pair_w == 1.0 and c.k == 8 and c.epochs > 0 for c in cfgs))
        with self.assertRaises(SystemExit):
            R.build_parser().parse_args(["run", "--fraction", "7"])

    def test_g_cli_pins_refuse_a_foreign_ranked_file(self):
        """The CLI cannot override the pins: a synthetic ranked file is refused (sha256 pin), proving the pin is live."""
        with self.assertRaises(R.BankError) as cm:
            R.main(["run", "--fraction", "10", "--bank-dir", self.w["bank"], "--ranked-logs", self.w["tsv"], "--p2-ckpt", self.w["ckpt"], "--s0-marker", self.w["marker"],
                    "--out-dir", self.out()])
        self.assertIn("registered", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
