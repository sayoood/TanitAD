"""Tests for epr1_dev.py on SYNTHETIC worlds only (never the real S0 bank, never the real dev table).  stdlib unittest (also collected by pytest):

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_epr1_dev.py -v

(a) KNOWN VALUES: a hand-built world whose gain is computable by hand; the identity (EP row unchanged => the dev pick EQUALS P2's pick on every token, gain exactly 0.0); the planner rule against an
    independent pure-python oracle; the bootstrap equals zero_tail.Boot; ONLY the named columns change.
(b) EVERY CHECK HAS A DELIBERATE-REGRESSION ARM: the source of epr1_dev.py is mutated at ONE marked line (the replacement must occur exactly once) and the check must go RED while the
    same check on the unmutated module is GREEN.
(c) REFUSALS fire BEFORE any dev OUTCOME is opened: the synthetic table of every refusal test has NO pdms / sub / valid / sub_names member at all, so an early read would be a KeyError, not a refusal.
(d) the read is ONCE; the path guard refuses navtest / the fit bank; an AST scan proves no file-system call lives outside an `_io_*` function.
(e) end to end: PASS / FAIL / VOID verdicts, both-seeds rule, determinism, the JOINT secondary (never moves the verdict), the PREREG s8 outcome rows.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epr1 as R          # noqa: E402
import epr1_dev as D      # noqa: E402
import zero_tail as ZT    # noqa: E402

SRC = os.path.join(HERE, "epr1_dev.py")
NP_ = R.N_PROPOSALS


def _rt(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def _rb(p):
    with open(p, "rb") as f:
        return f.read()


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def sha_f(p):
    with open(p, "rb") as f:
        return sha_b(f.read())


def wj(p, obj):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)


def load_mutant(*edits):
    """epr1_dev.py with ONE marked line edited per (marker, old, new): the marker `MUTATION-TARGET-<marker>` must sit on exactly one line and `old` must occur exactly once ON THAT LINE,
    so a refactor cannot silently neuter a mutation."""
    with open(SRC, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for marker, old, new in edits:
        idx = [i for i, ln in enumerate(lines) if f"MUTATION-TARGET-{marker}" in ln]
        assert len(idx) == 1, f"marker {marker} sits on {len(idx)} lines"
        assert lines[idx[0]].count(old) == 1, f"{old!r} occurs {lines[idx[0]].count(old)} times on the {marker} line"
        lines[idx[0]] = lines[idx[0]].replace(old, new)
    return _exec_module("\n".join(lines))


def _exec_module(src):
    name = "epr1_dev_mutant"
    mod = types.ModuleType(name)
    mod.__file__ = SRC
    sys.modules[name] = mod                     # dataclasses look the defining module up in sys.modules
    try:
        exec(compile(src, SRC, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


APPLY_OLD = "R.apply_row(z, f, w, b, W6[col], b6[col:col + 1], col=col)"


def ref_pick(z):
    """INDEPENDENT oracle of the planner rule: python floats, first strict maximum (lowest index on ties)."""
    best, bi = -1.0, 0
    for j in range(z.shape[0]):
        p = [1.0 / (1.0 + math.exp(-float(z[j, c]))) for c in range(6)]
        a = p[0] * p[1] * (5 * p[2] + 5 * p[3] + 2 * p[4]) / 12.0
        if a > best:
            best, bi = a, j
    return bi


FS_CALLS = {"open", "np.load", "np.save", "np.savez", "np.savez_compressed", "torch.load", "torch.save", "os.listdir", "os.makedirs", "os.remove", "os.rename", "os.stat", "os.unlink",
            "os.mkdir", "os.rmdir", "os.walk", "os.scandir", "os.path.exists", "os.path.isfile", "os.path.isdir", "shutil.copy", "shutil.copyfile", "shutil.rmtree", "shutil.move"}


def _dotted(f):
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def fs_violations(src):
    """[(function, call)] for every file-system call that lives outside an `_io_*` function.  `np.load(io.BytesIO(..))` in `load_row` parses bytes already read through `_io_read_bytes`."""
    bad = []
    for fn in [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)]:
        if fn.name.startswith("_io_"):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                name = _dotted(n.func)
                if name in FS_CALLS and not (fn.name == "load_row" and name == "np.load"):
                    bad.append((fn.name, name))
    return bad


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# synthetic world
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class World:
    """A dev table + P2 checkpoint + an EP-R1 run dir (+ optional SR-TTC run dir), all synthetic.  P2's NC / DAC logits are informative, its EP row is NOT; the EP label depends on feat . u,
    so a row along +u gains, a row along -u loses.  `outcomes=False` omits pdms / sub / valid / sub_names entirely."""

    def __init__(self, root, n_logs=24, per_log=6, seed=0, outcomes=True, lr=3e-4, scales=(3.0, 3.0), c2_scale=-3.0, ttc=False, ttc_scales=(2.0, 2.0)):
        self.root = root
        rng = np.random.default_rng(seed)
        self.n_logs, self.per_log = n_logs, per_log
        N = n_logs * per_log
        self.N = N
        names = [f"2021.05.{i % 28 + 1:02d}.10.00.{i:02d}_veh-{i % 7}_{i:05d}_{i + 9:05d}" for i in range(n_logs)]
        self.token = np.array([f"{i:016x}" for i in range(N)])
        self.log = np.array([names[i // per_log] for i in range(N)])
        u = rng.normal(size=R.FEAT_DIM)
        u[:3] = 0.0
        u /= np.linalg.norm(u)
        self.u = u.astype(np.float32)
        nc = np.where(rng.random((N, NP_)) < 0.9, 1.0, np.where(rng.random((N, NP_)) < 0.5, 0.5, 0.0))
        dac = (rng.random((N, NP_)) < 0.92).astype(np.float64)
        feat = rng.normal(size=(N, NP_, R.FEAT_DIM)).astype(np.float32)
        feat[..., 0] = np.where(nc >= 1.0, 3.0, -3.0) + 0.3 * rng.normal(size=(N, NP_))
        feat[..., 1] = np.where(dac >= 1.0, 3.0, -3.0) + 0.3 * rng.normal(size=(N, NP_))
        feat[..., 2] = 0.0
        self.feat = feat.astype(np.float16)
        W6 = (rng.normal(size=(R.N_COMP, R.FEAT_DIM)) * 0.02).astype(np.float32)
        W6[0], W6[1] = 0.0, 0.0
        W6[0, 0], W6[1, 1] = 1.0, 1.0
        b6 = np.array([0.0, 0.0, 0.2, 0.5, 0.5, 0.0], np.float32)
        self.W6, self.b6 = W6, b6
        f32 = self.feat.astype(np.float32)
        self.logits = (f32 @ W6.T + b6).astype(np.float32)
        safe = (nc >= 1.0) & (dac >= 1.0)
        ep = np.where(safe, 1.0 / (1.0 + np.exp(-(2.0 * (f32 @ self.u) + 0.5 * rng.normal(size=(N, NP_))))), 0.0)
        sub = np.stack([nc, dac, ep, (rng.random((N, NP_)) < 0.85).astype(np.float64), (rng.random((N, NP_)) < 0.9).astype(np.float64), (rng.random((N, NP_)) < 0.97).astype(np.float64)], -1)
        self.sub = sub
        self.pdms = sub[..., 0] * sub[..., 1] * (5.0 * sub[..., 2] + 5.0 * sub[..., 3] + 2.0 * sub[..., 4]) / 12.0
        self.props = rng.normal(size=(N, NP_, 8, 3)).astype(np.float32)
        self.pick = R.picks_np(self.logits)
        # --- files
        self.dev_dir = os.path.join(root, "dev")
        self.run_dir = os.path.join(root, "run")
        self.ttc_dir = os.path.join(root, "ttc_run")
        for d in (self.dev_dir, self.run_dir):
            os.makedirs(d, exist_ok=True)
        T = {"token": self.token, "log": self.log, "chunk": np.zeros(N, np.int32), "fingerprint": self.token, "logits": self.logits, "proposals": self.props, "pick": self.pick}
        if outcomes:
            T.update(pdms=self.pdms, sub=self.sub, valid=np.ones((N, NP_), bool), sub_names=np.array(D.SUB_NAMES), pdms_at_pick=self.pdms[np.arange(N), self.pick])
        np.savez(os.path.join(self.dev_dir, "table_p2_heldout.npz"), **T)
        np.savez(os.path.join(self.dev_dir, "prehead_features_p2_heldout.npz"), token=self.token, log=self.log, prehead_features=self.feat, what=np.array("synthetic"))
        np.savez(os.path.join(self.dev_dir, "proposals.npz"), token=self.token, log=self.log, proposals=self.props, pick=self.pick, rule=np.array("navsim_v1"), ckpt=np.array("synthetic"))
        self.pins = {n: sha_f(os.path.join(self.dev_dir, n)) for n in D.DEV_FILES}
        self.ckpt = os.path.join(root, "p2", "model_final.pt")
        os.makedirs(os.path.dirname(self.ckpt), exist_ok=True)
        torch.save({"format": "synthetic", "model": {"score_head.weight": torch.from_numpy(W6), "score_head.bias": torch.from_numpy(b6)}}, self.ckpt)
        # --- EP-R1 run dir
        self.cfg = R.Cfg(lr=lr, anchor=0.1)
        self.rows = {}
        for sd, sc in zip(D.SEED_KEYS, scales):
            self.rows[sd] = (W6[2] + np.float32(sc) * self.u + (np.float32(1e-3 * sc * int(sd)) * np.cos(np.arange(R.FEAT_DIM))).astype(np.float32), b6[2:3].copy())
        self.c2rows = {sd: (W6[2] + np.float32(c2_scale) * self.u, b6[2:3].copy()) for sd in D.SEED_KEYS}
        final, c2final = {}, {}
        for sd in D.SEED_KEYS:
            final[sd] = self._write_row(self.run_dir, f"ep_row_seed{sd}.npy", self.rows[sd], W6[2], b6[2:3])
            c2final[sd] = self._write_row(self.run_dir, f"ep_row_c2shuffle_seed{sd}.npy", self.c2rows[sd], W6[2], b6[2:3])
        self.bank_man = "b" * 64
        self.r1 = {"schema": R.SCHEMA, "stage": "R1", "fraction_pct": 10,
                   "inputs": {"bank_manifest_sha256": self.bank_man, "ranked_fit_logs_sha256": R.RANKED_LOGS_SHA256, "p2_row_sha256": sha_b(R.row_bytes(W6[2], b6[2:3])),
                              "replacements": {"n_selected": 117, "dropped": [], "added": []}, "pins_overridden": []},
                   "params": {"seeds": [0, 1], "epochs": R.DEFAULT_EPOCHS, "batch_sets": R.DEFAULT_BATCH_SETS},
                   "c_star": {"id": self.cfg.id(), "lr": self.cfg.lr, "anchor": self.cfg.anchor, "config_sha256": self.cfg.sha256(), "c_star_at_grid_edge": bool(lr == R.GRID_LR[-1])},
                   "pick_level_at_c_star": {sd: {"delta_pdms_x100": {"point": 0.1, "lo": -0.1, "hi": 0.3}, "delta_unsafe_rate": {"point": 0.0, "lo": -0.01, "hi": 0.01}} for sd in D.SEED_KEYS},
                   "final_rows": final, "code_sha256": D.EPR1_SHA256}
        self.c2 = {"schema": R.SCHEMA, "stage": "C2", "c_star": self.r1["c_star"], "bank_manifest_sha256": self.bank_man, "final_rows": c2final}
        self.write_r1()
        self.write_c2()
        if ttc:
            self.make_ttc(ttc_scales)

    @staticmethod
    def _write_row(d, name, row, p2w, p2b):
        w, b = row
        blob = R.row_bytes(w, b)
        with open(os.path.join(d, name), "wb") as f:
            f.write(blob)
        return {"file": name, "sha256": sha_b(blob), "bytes": len(blob), "max_abs_change_vs_p2": float(max(np.abs(w - p2w).max(), np.abs(b - p2b).max()))}

    def write_r1(self):
        wj(os.path.join(self.run_dir, "epr1_result.json"), self.r1)

    def write_c2(self):
        wj(os.path.join(self.run_dir, "epr1_c2_result.json"), self.c2)

    def r1_sha(self):
        return sha_f(os.path.join(self.run_dir, "epr1_result.json"))

    def make_ttc(self, ttc_scales, fraction=10, bank_man=None):
        os.makedirs(self.ttc_dir, exist_ok=True)
        # the TTC label is random in this world, so a TTC row cannot gain; the JOINT is exercised for plumbing and for the secondary's labelling
        self.ttc_rows = {sd: (self.W6[3] + np.float32(sc) * self.u, self.b6[3:4].copy()) for sd, sc in zip(D.SEED_KEYS, ttc_scales)}
        final = {sd: self._write_row(self.ttc_dir, f"ttc_row_seed{sd}.npy", self.ttc_rows[sd], self.W6[3], self.b6[3:4]) for sd in D.SEED_KEYS}
        self.ttc_r1 = {"schema": "srttc-r1-v1", "stage": "R1", "fraction_pct": fraction,
                       "inputs": {"bank_manifest_sha256": bank_man or self.bank_man, "ranked_fit_logs_sha256": R.RANKED_LOGS_SHA256, "p2_row_sha256": sha_b(R.row_bytes(self.W6[3], self.b6[3:4])),
                                  "replacements": {"n_selected": 117, "dropped": [], "added": []}},
                       "c_star": {"id": "lr0.0003_anchor0.1", "lr": 3e-4, "anchor": 0.1, "config_sha256": "c" * 64}, "final_rows": final}
        wj(os.path.join(self.ttc_dir, "r1_result.json"), self.ttc_r1)
        return sha_f(os.path.join(self.ttc_dir, "r1_result.json"))

    def opts(self, **kw):
        base = dict(run_dir=self.run_dir, registered_result_sha256=self.r1_sha(), dev_dir=self.dev_dir, p2_ckpt=self.ckpt, out_dir=os.path.join(self.root, "out"), dev_pins=dict(self.pins),
                    expect_tokens=self.N, expect_logs=self.n_logs)
        base.update(kw)
        return D.DevOptions(**base)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="epr1dev_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def world(self, name="w", **kw):
        return World(os.path.join(self.tmp, name), **kw)

    def inputs(self, w):
        return D.load_dev_inputs(w.opts().guard(), w.opts())

    def outcomes(self, w):
        return D.load_dev_outcomes(w.opts().guard(), w.opts())


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (a) known values
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class HandBuilt(Base):
    """A world computable by hand.  P2's head is the identity on the first six features (logit_k = feat_k), so the stored EP logit is feat[...,2] and a new EP row e_6 gives new EP logit feat[...,6]."""

    def build(self):
        n_tok, per = 6, 2
        rng = np.random.default_rng(5)
        feat = np.zeros((n_tok, NP_, R.FEAT_DIM), np.float32)
        feat[..., 0] = feat[..., 1] = feat[..., 3] = feat[..., 4] = 20.0                 # NC, DAC, TTC, C ~ certain: the rule reduces to argmax of pEP
        feat[..., 5] = 0.0
        a = np.array([3, 7, 11, 40, 0, 63])
        b = np.array([5, 7 + 2, 12, 41, 1, 62])
        for i in range(n_tok):
            ep_old = rng.normal(size=NP_) * 0.5
            ep_new = rng.normal(size=NP_) * 0.5
            ep_old[a[i]] = 5.0                                                          # P2 picks a_i
            ep_new[b[i]] = 5.0                                                          # the new row picks b_i
            feat[i, :, 2], feat[i, :, 6] = ep_old, ep_new
            feat[i, :, 7] = np.where(np.arange(NP_) == b[i], -20.0, 20.0)               # a TTC alternative: kills proposal b_i
        W6 = np.zeros((R.N_COMP, R.FEAT_DIM), np.float32)
        for k in range(6):
            W6[k, k] = 1.0
        b6 = np.zeros(6, np.float32)
        feat16 = feat.astype(np.float16)
        z = (feat16.astype(np.float32) @ W6.T + b6).astype(np.float32)
        x = np.array([0.50, 0.70, 0.20, 0.00, 0.90, 0.40])
        y = np.array([0.80, 0.70, 0.95, 0.60, 0.10, 0.45])
        pdms = np.full((n_tok, NP_), 0.5)
        pdms[np.arange(n_tok), a] = x
        pdms[np.arange(n_tok), b] = y
        sub = np.ones((n_tok, NP_, 6))
        sub[..., 2] = pdms                                                                # EP component := pdms (NC=DAC=TTC=C=1 -> pdms = (5 EP + 5 + 2)/12 is NOT used; the table here is hand-written)
        log = np.array(["L0"] * 2 + ["L1"] * 2 + ["L2"] * 2)
        names, idx = np.unique(log, return_inverse=True)
        inp = D.DevInputs(token=np.array([f"t{i}" for i in range(n_tok)]), log=log, log_names=names.tolist(), log_idx=idx, feat=torch.from_numpy(feat16), z6=torch.from_numpy(z),
                          pick_banked=R.picks_np(z), props_table=np.zeros((n_tok, NP_, 8, 3), np.float32), props_dump=np.zeros((n_tok, NP_, 8, 3), np.float32), pick_dump=R.picks_np(z),
                          token_feat=np.array([f"t{i}" for i in range(n_tok)]), log_feat=log, token_dump=np.array([f"t{i}" for i in range(n_tok)]), log_dump=log, provenance={})
        out = D.DevOutcomes(pdms=pdms, sub=sub, safe=np.ones((n_tok, NP_), bool), formula={})
        return inp, out, W6, b6, a, b, x, y

    def test_new_pick_and_p2_pick_are_the_hand_picks(self):
        inp, out, W6, b6, a, b, x, y = self.build()
        np.testing.assert_array_equal(R.picks_np(inp.z6.numpy()), a)
        row = np.zeros(R.FEAT_DIM, np.float32)
        row[6] = 1.0
        z = D.apply_rows(inp.z6, inp.feat, {D.EP_COL: (row, np.zeros(1, np.float32))}, W6, b6)
        np.testing.assert_array_equal(D.pick_logits(z), b)
        np.testing.assert_allclose(z[..., 2], inp.feat.float().numpy()[..., 6], atol=1e-5)        # new EP logit == feat6 (stored + (feat6 - feat2), float32)

    def test_gain_is_the_hand_value(self):
        inp, out, W6, b6, a, b, x, y = self.build()
        row = np.zeros(R.FEAT_DIM, np.float32)
        row[6] = 1.0
        z = D.apply_rows(inp.z6, inp.feat, {D.EP_COL: (row, np.zeros(1, np.float32))}, W6, b6)
        draws = R.boot_draws(inp.n_logs)
        r = D.score_arm(inp, out, D.pick_logits(z), D.pick_logits(inp.z6.numpy()), draws)
        self.assertAlmostEqual(r["gain_x100"]["point"], 100.0 * float((y - x).mean()), places=9)
        self.assertEqual(r["changed_picks"], int((a != b).sum()))
        self.assertEqual(r["changed_picks"], 6)                                           # a and b differ on all six tokens
        self.assertAlmostEqual(r["new_pdms_x100"], 100.0 * float(y.mean()), places=9)
        self.assertAlmostEqual(r["base_pdms_x100"], 100.0 * float(x.mean()), places=9)

    def test_joint_two_columns_and_independent_oracle(self):
        inp, out, W6, b6, a, b, x, y = self.build()
        r_ep = np.zeros(R.FEAT_DIM, np.float32)
        r_ep[6] = 1.0
        r_tt = np.zeros(R.FEAT_DIM, np.float32)
        r_tt[7] = 1.0
        rows = {D.EP_COL: (r_ep, np.zeros(1, np.float32)), D.TTC_COL: (r_tt, np.zeros(1, np.float32))}
        z = D.apply_rows(inp.z6, inp.feat, rows, W6, b6)
        stored = inp.z6.numpy()
        np.testing.assert_array_equal(z[..., [0, 1, 4, 5]], stored[..., [0, 1, 4, 5]])
        self.assertTrue((z[..., 2] != stored[..., 2]).any() and (z[..., 3] != stored[..., 3]).any())
        got = D.pick_logits(z)
        want = np.array([ref_pick(z[i]) for i in range(z.shape[0])])
        np.testing.assert_array_equal(got, want)                                          # the planner rule against the independent oracle
        self.assertFalse((got == b).all(), "the TTC row kills proposal b_i: the joint pick must differ from the EP-only pick")

    def test_only_named_columns_change(self):
        w = self.world()
        inp = self.inputs(w)
        z = D.apply_rows(inp.z6, inp.feat, {D.EP_COL: w.rows["0"]}, w.W6, w.b6)
        stored = inp.z6.numpy()
        for c in (0, 1, 3, 4, 5):
            np.testing.assert_array_equal(z[..., c], stored[..., c])
        self.assertTrue((z[..., 2] != stored[..., 2]).any())

    def test_apply_row_matches_float_reference(self):
        w = self.world()
        inp = self.inputs(w)
        z = D.apply_rows(inp.z6, inp.feat, {D.EP_COL: w.rows["0"]}, w.W6, w.b6)
        ref = inp.z6.numpy()[..., 2].astype(np.float64) + inp.feat.float().numpy().astype(np.float64) @ (w.rows["0"][0].astype(np.float64) - w.W6[2]) + float(w.rows["0"][1][0] - w.b6[2])
        self.assertLess(float(np.abs(z[..., 2] - ref).max()), 1e-4)

    def test_chunking_is_exact_for_the_identity_and_close_otherwise(self):
        w = self.world()
        inp = self.inputs(w)
        a = D.apply_rows(inp.z6, inp.feat, {D.EP_COL: w.rows["1"]}, w.W6, w.b6, chunk=7)
        b = D.apply_rows(inp.z6, inp.feat, {D.EP_COL: w.rows["1"]}, w.W6, w.b6, chunk=10_000)
        self.assertLess(float(np.abs(a - b).max()), 1e-5)
        ident = {D.EP_COL: (w.W6[2].copy(), w.b6[2:3].copy())}
        self.assertTrue(np.array_equal(D.apply_rows(inp.z6, inp.feat, ident, w.W6, w.b6, chunk=7), inp.z6.numpy()))


class Identity(Base):
    """The known-value identity the brief asks for: with the EP row unchanged, the dev pick equals P2's pick exactly."""

    def test_unchanged_ep_row_reproduces_p2_pick_exactly(self):
        w = self.world()
        inp = self.inputs(w)
        ident = {D.EP_COL: (w.W6[2].copy(), w.b6[2:3].copy())}
        z = D.apply_rows(inp.z6, inp.feat, ident, w.W6, w.b6)
        self.assertTrue(np.array_equal(z, inp.z6.numpy()), "logits must be BIT-identical")
        pk = D.pick_logits(z)
        self.assertTrue(np.array_equal(pk, inp.pick_banked), "dev pick must equal P2's pick on every token")
        self.assertTrue(np.array_equal(pk, np.array([ref_pick(inp.z6.numpy()[i]) for i in range(inp.n)])))
        out = self.outcomes(w)
        r = D.score_arm(inp, out, pk, inp.pick_banked, R.boot_draws(inp.n_logs))
        self.assertEqual(r["gain_x100"]["point"], 0.0)
        self.assertEqual((r["gain_x100"]["lo"], r["gain_x100"]["hi"]), (0.0, 0.0))
        self.assertEqual(r["changed_picks"], 0)
        for nm in D.COMPS:
            self.assertEqual(r["components"][nm]["delta_x100"]["point"], 0.0)

    def test_c0_green_and_gain_zero_green(self):
        w = self.world()
        inp = self.inputs(w)
        c0 = D.c0_identity(inp, w.W6, w.b6)
        self.assertTrue(c0["pass"], c0)
        self.assertEqual(c0["picks_identity_ep_row"], inp.n)
        self.assertEqual(c0["picks_identity_joint"], inp.n)
        g = D.c0_gain_zero(inp, self.outcomes(w), D.pick_logits(inp.z6.numpy()), R.boot_draws(inp.n_logs))
        self.assertTrue(g["pass"], g)

    def test_c0_red_when_the_banked_pick_is_not_p2s_rule_pick(self):
        w = self.world()
        inp = self.inputs(w)
        bad = inp.pick_banked.copy()
        bad[3] = (bad[3] + 1) % NP_
        inp2 = D.DevInputs(**{**inp.__dict__, "pick_banked": bad})
        self.assertFalse(D.c0_identity(inp2, w.W6, w.b6)["pass"])

    def test_c0_red_when_the_applier_shifts_logits_even_for_an_unchanged_row(self):
        w = self.world()
        mut = load_mutant(("APPLY", APPLY_OLD, APPLY_OLD + " + 3.0"))
        inp = self.inputs(w)
        self.assertTrue(D.c0_identity(inp, w.W6, w.b6)["pass"])
        self.assertFalse(mut.c0_identity(inp, w.W6, w.b6)["pass"])

    def test_c0_gain_zero_red_when_gain_sign_or_base_is_wrong(self):
        w = self.world()
        inp, out = self.inputs(w), self.outcomes(w)
        draws = R.boot_draws(inp.n_logs)
        base = D.pick_logits(inp.z6.numpy())
        shifted = (base + 1) % NP_
        mut = load_mutant(("GAIN-SIGN", "per_log(inp, pb)", "per_log(inp, out.pdms[ar, (base_pick + 1) % 64])"))     # the BASE leg reads another proposal
        self.assertTrue(D.c0_gain_zero(inp, out, base, draws)["pass"])
        self.assertFalse(mut.c0_gain_zero(inp, out, base, draws)["pass"])
        self.assertTrue((shifted != base).all())


class Estimator(Base):
    def test_bootstrap_equals_zero_tail_boot(self):
        w = self.world()
        inp, out = self.inputs(w), self.outcomes(w)
        draws = R.boot_draws(inp.n_logs)
        new = (D.pick_logits(inp.z6.numpy()) + 3) % NP_
        base = D.pick_logits(inp.z6.numpy())
        r = D.score_arm(inp, out, new, base, draws)["gain_x100"]
        ar = np.arange(inp.n)
        boot = ZT.Boot(inp.log_idx, inp.n_logs)
        ref = boot.ratio(out.pdms[ar, new] - out.pdms[ar, base], np.ones(inp.n), scale=100.0)
        for k in ("point", "lo", "hi"):
            self.assertAlmostEqual(r[k], ref[k], places=9, msg=k)
        self.assertEqual(r["B"], 10000)
        self.assertEqual(R.BOOT_SEED, 20260927)

    def test_gain_sign_mutant_is_red_on_the_known_value(self):
        inp, out, W6, b6, a, b, x, y = HandBuilt().build()
        row = np.zeros(R.FEAT_DIM, np.float32)
        row[6] = 1.0
        z = D.apply_rows(inp.z6, inp.feat, {D.EP_COL: (row, np.zeros(1, np.float32))}, W6, b6)
        draws = R.boot_draws(inp.n_logs)
        mut = load_mutant(("GAIN-SIGN", "R.paired_delta(per_log(inp, pn), per_log(inp, pb), den, draws)", "R.paired_delta(per_log(inp, pb), per_log(inp, pn), den, draws)"))
        want = 100.0 * float((y - x).mean())
        self.assertAlmostEqual(D.score_arm(inp, out, D.pick_logits(z), D.pick_logits(inp.z6.numpy()), draws)["gain_x100"]["point"], want, places=9)
        self.assertNotAlmostEqual(mut.score_arm(inp, out, D.pick_logits(z), D.pick_logits(inp.z6.numpy()), draws)["gain_x100"]["point"], want, places=3)

    def test_token_level_bootstrap_is_not_the_log_cluster_one(self):
        """The mutation arm for 'cluster on logs': a token-level resample gives a visibly different interval from the log-cluster one."""
        w = self.world()
        inp, out = self.inputs(w), self.outcomes(w)
        new = (D.pick_logits(inp.z6.numpy()) + 3) % NP_
        base = D.pick_logits(inp.z6.numpy())
        draws = R.boot_draws(inp.n_logs)
        r = D.score_arm(inp, out, new, base, draws)["gain_x100"]
        tok_draws = R.boot_draws(inp.n)                                                 # one cluster per token
        ar = np.arange(inp.n)
        num_a, num_b = out.pdms[ar, new], out.pdms[ar, base]
        rt = R.paired_delta(num_a, num_b, np.ones(inp.n), tok_draws)
        self.assertNotAlmostEqual(r["hi"] - r["lo"], 100.0 * (rt["hi"] - rt["lo"]), places=3)

    def test_components_report_all_six_with_intervals(self):
        w = self.world()
        inp, out = self.inputs(w), self.outcomes(w)
        r = D.score_arm(inp, out, (D.pick_logits(inp.z6.numpy()) + 5) % NP_, D.pick_logits(inp.z6.numpy()), R.boot_draws(inp.n_logs))
        self.assertEqual(tuple(r["components"]), D.COMPS)
        for nm, c in r["components"].items():
            self.assertTrue(all(k in c["delta_x100"] for k in ("point", "lo", "hi")), nm)
        self.assertIn("unsafe_pick_rate", r)
        self.assertIn("zero_pdms_pick_rate", r)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (b) controls: GREEN on the real module, RED on the mutant
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Controls(Base):
    def test_c1_green(self):
        w = self.world()
        inp = self.inputs(w)
        r = D.c1_reach(inp, {sd: w.rows[sd] for sd in D.SEED_KEYS}, w.W6, w.b6)
        self.assertTrue(r["pass"], r)
        for sd in D.SEED_KEYS:
            self.assertGreater(r["per_seed"][sd]["ep_column_entries_changed"], 0)
            self.assertTrue(r["per_seed"][sd]["other_five_columns_bit_identical"])
        self.assertGreater(r["positive_control_negated_p2_row_changed_picks"], 0)

    def test_c1_red_when_the_row_never_reaches_the_logits(self):
        w = self.world()
        inp = self.inputs(w)
        mut = load_mutant(("APPLY", "z = " + APPLY_OLD, "z = z"))
        self.assertTrue(D.c1_reach(inp, {sd: w.rows[sd] for sd in D.SEED_KEYS}, w.W6, w.b6)["pass"])
        self.assertFalse(mut.c1_reach(inp, {sd: w.rows[sd] for sd in D.SEED_KEYS}, w.W6, w.b6)["pass"])

    def test_c1_red_when_the_row_lands_in_the_wrong_column(self):
        w = self.world()
        inp = self.inputs(w)
        mut = load_mutant(("APPLY", "col=col)", "col=3)"))
        r = mut.c1_reach(inp, {sd: w.rows[sd] for sd in D.SEED_KEYS}, w.W6, w.b6)
        self.assertFalse(r["pass"])
        self.assertFalse(all(r["per_seed"][sd]["other_five_columns_bit_identical"] for sd in D.SEED_KEYS))

    def test_c1_red_when_picks_ignore_the_new_logits(self):
        w = self.world()
        inp = self.inputs(w)
        mut = load_mutant(("C1-POS", "pick_logits(z_neg)", "pick_logits(stored)"))
        self.assertFalse(mut.c1_reach(inp, {sd: w.rows[sd] for sd in D.SEED_KEYS}, w.W6, w.b6)["pass"])

    def test_c1_red_for_a_row_equal_to_p2s(self):
        """A 'fitted' row that is P2's own row reaches nothing: the reach check must say so (the fit moved nothing)."""
        w = self.world()
        inp = self.inputs(w)
        same = {sd: (w.W6[2].copy(), w.b6[2:3].copy()) for sd in D.SEED_KEYS}
        self.assertFalse(D.c1_reach(inp, same, w.W6, w.b6)["pass"])

    def test_c4_green_and_red_on_a_one_millimetre_perturbation(self):
        w = self.world()
        inp = self.inputs(w)
        self.assertTrue(D.c4_fan_frozen(inp)["pass"])
        pr = inp.props_dump.copy()
        pr[17, 5, 3, 0] += 1e-3
        bad = D.DevInputs(**{**inp.__dict__, "props_dump": pr})
        r = D.c4_fan_frozen(bad)
        self.assertFalse(r["pass"])
        self.assertGreater(r["proposals_max_abs_m"], 0.0)

    def test_c4_red_arms(self):
        w = self.world()
        inp = self.inputs(w)
        tok = inp.token_dump.copy()
        tok[[0, 1]] = tok[[1, 0]]
        self.assertFalse(D.c4_fan_frozen(D.DevInputs(**{**inp.__dict__, "token_dump": tok}))["pass"])
        pk = inp.pick_dump.copy()
        pk[0] = (pk[0] + 1) % NP_
        self.assertFalse(D.c4_fan_frozen(D.DevInputs(**{**inp.__dict__, "pick_dump": pk}))["pass"])
        lg = inp.log_feat.copy()
        lg[0] = "other"
        self.assertFalse(D.c4_fan_frozen(D.DevInputs(**{**inp.__dict__, "log_feat": lg}))["pass"])

    def test_c4_mutant_with_a_tolerance_misses_the_perturbation(self):
        w = self.world()
        inp = self.inputs(w)
        pr = inp.props_dump.copy()
        pr[17, 5, 3, 0] += 1e-3
        bad = D.DevInputs(**{**inp.__dict__, "props_dump": pr})
        mut = load_mutant(("C4-EXACT", "mx == 0.0", "mx <= 1e-2"))
        self.assertFalse(D.c4_fan_frozen(bad)["pass"])
        self.assertTrue(mut.c4_fan_frozen(bad)["pass"], "the mutant must be the one that misses it")

    def test_featcheck_green_and_red(self):
        w = self.world()
        inp = self.inputs(w)
        r = D.integrity_featcheck(inp, w.W6, w.b6, D.FEATCHECK_TOL)
        self.assertTrue(r["pass"], r)
        self.assertLess(r["max_abs_featcheck"], 1e-5)
        self.assertGreater(r["negative_control_next_token_max_abs"], 10 * D.FEATCHECK_TOL)
        bad = w.W6.copy()
        bad[2] = np.roll(bad[2], 5)
        self.assertFalse(D.integrity_featcheck(inp, bad, w.b6, D.FEATCHECK_TOL)["pass"], "a different head must not reproduce the stored logits")

    def test_formula_check_refuses_a_wrong_scale(self):
        w = self.world()
        with np.load(os.path.join(w.dev_dir, "table_p2_heldout.npz")) as z:
            t = {k: z[k] for k in z.files}
        t["pdms"] = t["pdms"] * t["sub"][..., 5]                                         # the DDC-in-product variant
        np.savez(os.path.join(w.dev_dir, "table_p2_heldout.npz"), **t)
        pins = dict(w.pins)
        pins["table_p2_heldout.npz"] = sha_f(os.path.join(w.dev_dir, "table_p2_heldout.npz"))
        o = w.opts(dev_pins=pins)
        with self.assertRaises(D.DevRefusal):
            D.load_dev_outcomes(o.guard(), o)

    def test_outcome_free_table_loads_inputs(self):
        """The inputs phase touches no outcome: it runs on a table that HAS no pdms / sub / valid / sub_names."""
        w = self.world(outcomes=False)
        inp = self.inputs(w)
        self.assertEqual(inp.n, w.N)
        with self.assertRaises(KeyError):
            self.outcomes(w)


class Bar(Base):
    GOOD = {"0": 0.5, "1": 0.31}

    def test_bar_requires_both_seeds(self):
        self.assertTrue(D.bar_check(self.GOOD)["both_seeds"])
        self.assertFalse(D.bar_check({"0": 0.5, "1": 0.29})["both_seeds"])
        self.assertFalse(D.bar_check({"0": 0.29, "1": 0.5})["both_seeds"])
        self.assertTrue(D.bar_check({"0": 0.30, "1": 0.30})["both_seeds"], "the bar is >= +0.30")
        self.assertFalse(D.bar_check({"0": 0.2999999, "1": 0.9})["both_seeds"])

    def test_bar_mutant_any_instead_of_all_is_red(self):
        mut = load_mutant(("BOTH-SEEDS", "bool(all(per[sd] for sd in SEED_KEYS))", "bool(any(per[sd] for sd in SEED_KEYS))"))
        split = {"0": 0.5, "1": 0.1}
        self.assertFalse(D.bar_check(split)["both_seeds"])
        self.assertTrue(mut.bar_check(split)["both_seeds"], "the mutant must be the one that lets one seed through")

    def test_bar_threshold_mutant_is_red(self):
        src = _rt(SRC).replace("DEV_BAR_X100 = 0.30 ", "DEV_BAR_X100 = 0.10 ")
        self.assertIn("DEV_BAR_X100 = 0.10 ", src)
        mut = _exec_module(src)
        self.assertFalse(D.bar_check({"0": 0.2, "1": 0.2})["both_seeds"])
        self.assertTrue(mut.bar_check({"0": 0.2, "1": 0.2})["both_seeds"])

    def test_verdict_logic(self):
        ok = D.bar_check(self.GOOD)
        no = D.bar_check({"0": 0.5, "1": 0.0})
        self.assertEqual(D.verdict_of(True, ok), "PASS")
        self.assertEqual(D.verdict_of(True, no), "FAIL")
        self.assertEqual(D.verdict_of(False, ok), "VOID", "a RED control voids the arm even when the bar is met")
        self.assertEqual(D.verdict_of(False, no), "VOID")


class OutcomeRows(Base):
    @staticmethod
    def seed(gain, ep, ttc):
        return {"gain_x100": {"point": gain}, "components": {"ep": {"delta_x100": {"point": ep}}, "ttc": {"delta_x100": {"point": ttc}}}}

    def r1(self, lo0=-0.01, lo1=-0.01, f=10):
        return {"fraction_pct": f, "pick_level_at_c_star": {"0": {"delta_unsafe_rate": {"lo": lo0}}, "1": {"delta_unsafe_rate": {"lo": lo1}}}}

    def test_pass(self):
        r = D.outcome_rows("PASS", {"0": self.seed(0.5, 1, -1), "1": self.seed(0.4, 1, -1)}, self.r1(), None)
        self.assertEqual(r["fired"], ["D-PASS"])

    def test_void_is_a(self):
        self.assertIn("A", D.outcome_rows("VOID", {"0": self.seed(0.5, 1, 0), "1": self.seed(0.5, 1, 0)}, self.r1(), None)["fired"])

    def test_trade_is_per_seed_and_needs_ep_up_ttc_down_and_gain_below_bar(self):
        r = D.outcome_rows("FAIL", {"0": self.seed(0.1, 2.0, -0.5), "1": self.seed(0.5, 2.0, -0.5)}, self.r1(), None)
        self.assertIn("D-TRADE", r["fired"])
        self.assertEqual(r["notes"]["d_trade_per_seed"], {"0": True, "1": False})
        r2 = D.outcome_rows("FAIL", {"0": self.seed(0.1, 2.0, 0.5), "1": self.seed(0.1, -1.0, -0.5)}, self.r1(), None)
        self.assertNotIn("D-TRADE", r2["fired"])

    def test_leak_needs_both_seeds_lo_above_zero_whatever_the_dev_gain(self):
        for verdict in ("PASS", "FAIL"):
            r = D.outcome_rows(verdict, {"0": self.seed(0.5, 1, 0), "1": self.seed(0.5, 1, 0)}, self.r1(0.001, 0.002), None)
            self.assertIn("D-LEAK", r["fired"], verdict)
        one = D.outcome_rows("FAIL", {"0": self.seed(0.1, 0, 0), "1": self.seed(0.1, 0, 0)}, self.r1(0.001, -0.002), None)
        self.assertNotIn("D-LEAK", one["fired"])

    def test_trade_mutant_is_red(self):
        mut = load_mutant(("TRADE", "tt < 0.0", "tt > 0.0"))
        args = ("FAIL", {"0": self.seed(0.1, 2.0, -0.5), "1": self.seed(0.1, 2.0, -0.5)}, self.r1(), None)
        self.assertIn("D-TRADE", D.outcome_rows(*args)["fired"])
        self.assertNotIn("D-TRADE", mut.outcome_rows(*args)["fired"])

    def test_leak_mutant_any_is_red(self):
        mut = load_mutant(("LEAK", "all(leak.values())", "any(leak.values())"))
        args = ("FAIL", {"0": self.seed(0.1, 0, 0), "1": self.seed(0.1, 0, 0)}, self.r1(0.001, -0.002), None)
        self.assertNotIn("D-LEAK", D.outcome_rows(*args)["fired"])
        self.assertIn("D-LEAK", mut.outcome_rows(*args)["fired"])

    def test_flat_and_data_follow_the_lc_call(self):
        s = {"0": self.seed(0.1, 0, 0), "1": self.seed(0.1, 0, 0)}
        self.assertIn("D-FLAT", D.outcome_rows("FAIL", s, self.r1(), {"call": {"call": "FLAT"}})["fired"])
        self.assertIn("D-DATA", D.outcome_rows("FAIL", s, self.r1(f=10), {"call": {"call": "RISING"}})["fired"])
        self.assertIn("D-DATA", D.outcome_rows("FAIL", s, self.r1(f=10), {"call": {"call": "INCONCLUSIVE"}})["fired"])
        self.assertEqual(D.outcome_rows("FAIL", s, self.r1(f=100), {"call": {"call": "RISING"}})["fired"], [])
        und = D.outcome_rows("FAIL", s, self.r1(), None)
        self.assertEqual(und["fired"], [])
        self.assertIn("UNDETERMINED", und["notes"]["d_flat_or_data"])

    def test_no_row_names_a_next_arm_that_is_not_registered(self):
        r = D.outcome_rows("FAIL", {"0": self.seed(0.1, 1, -1), "1": self.seed(0.1, 1, -1)}, self.r1(0.1, 0.1), {"call": {"call": "FLAT"}})
        for k in r["fired"]:
            self.assertTrue(r["next_arm"][k])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (c) refusals: nothing is read, and the world has no outcome arrays at all
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Refusals(Base):
    def assertRefused(self, w, opts, contains):
        with self.assertRaises(D.DevRefusal) as cm:
            D.run_dev(opts)
        self.assertIn(contains, str(cm.exception))
        out = opts.resolved_out_dir()
        self.assertFalse(os.path.exists(os.path.join(out, D.READ_STARTED)), "a refusal must not spend the read")
        self.assertFalse(os.path.exists(os.path.join(out, D.RESULT_NAME)))

    def test_unregistered_result_sha_is_refused(self):
        w = self.world(outcomes=False)
        self.assertRefused(w, w.opts(registered_result_sha256="0" * 64), "REGISTERED")

    def test_result_edited_after_registration_is_refused(self):
        w = self.world(outcomes=False)
        sha = w.r1_sha()
        w.r1["c_star"]["id"] = "edited"
        w.write_r1()
        self.assertRefused(w, w.opts(registered_result_sha256=sha), "REGISTERED")

    def test_registered_sha_mutant_is_red(self):
        w = self.world(outcomes=False)
        o = w.opts(registered_result_sha256="0" * 64)
        with self.assertRaises(D.DevRefusal):
            D.check_r1_result(o.guard(), o, w.W6, w.b6)
        mut = load_mutant(("REGSHA", "sha != opts.registered_result_sha256", "False"))
        r = mut.check_r1_result(o.guard(), mut.DevOptions(**{**o.__dict__}), w.W6, w.b6)
        self.assertEqual(set(r["rows"]), set(D.SEED_KEYS), "the mutant must be the one that reads an unregistered c*")

    def test_dev_input_sha_mutant_is_red(self):
        w = self.world(outcomes=False)
        pr = w.props.copy()
        pr[1, 1, 1, 1] += 0.5
        np.savez(os.path.join(w.dev_dir, "proposals.npz"), token=w.token, log=w.log, proposals=pr, pick=w.pick, rule=np.array("navsim_v1"), ckpt=np.array("synthetic"))   # a VALID archive, other bytes
        o = w.opts()
        with self.assertRaises(D.DevRefusal):
            D.load_dev_inputs(o.guard(), o)
        mut = load_mutant(("DEVSHA", "pin is not None and sha[n] != pin", "False"))
        inp = mut.load_dev_inputs(o.guard(), mut.DevOptions(**{**o.__dict__}))
        self.assertEqual(inp.n, w.N, "the mutant must be the one that loads an unpinned dev input")

    def test_red_control_gate_mutant_is_red(self):
        w = self.world(outcomes=False)
        pr = w.props.copy()
        pr[4, 4, 4, 1] += 1e-3
        np.savez(os.path.join(w.dev_dir, "proposals.npz"), token=w.token, log=w.log, proposals=pr, pick=w.pick, rule=np.array("navsim_v1"), ckpt=np.array("synthetic"))
        pins = dict(w.pins)
        pins["proposals.npz"] = sha_f(os.path.join(w.dev_dir, "proposals.npz"))
        o = w.opts(dev_pins=pins)
        with self.assertRaises(D.DevRefusal):
            D.run_dev(o)
        mut = load_mutant(("REDGATE", "if red:", "if False:"))
        with self.assertRaises(KeyError):                                              # it sailed past the RED control and reached the (absent) outcomes
            mut.run_dev(mut.DevOptions(**{**w.opts(out_dir=os.path.join(w.root, "o_mut"), dev_pins=pins).__dict__}))
        self.assertTrue(os.path.exists(os.path.join(w.root, "o_mut", D.READ_STARTED)), "...and spent the read doing so")

    def test_tampered_row_file_is_refused(self):
        w = self.world(outcomes=False)
        p = os.path.join(w.run_dir, "ep_row_seed0.npy")
        w.rows["0"][0][5] += np.float32(0.5)
        w._write_row(w.run_dir, "ep_row_seed0.npy", w.rows["0"], w.W6[2], w.b6[2:3])
        self.assertTrue(os.path.exists(p))
        self.assertRefused(w, w.opts(), "the row is not the one the fit wrote")

    def test_wrong_row_length_or_dtype_is_refused(self):
        w = self.world(outcomes=False)
        blob = io.BytesIO()
        np.save(blob, np.zeros(10, np.float32))
        with open(os.path.join(w.run_dir, "ep_row_seed1.npy"), "wb") as f:
            f.write(blob.getvalue())
        w.r1["final_rows"]["1"]["sha256"] = sha_b(blob.getvalue())
        w.write_r1()
        self.assertRefused(w, w.opts(), "expected finite float32")

    def test_wrong_schema_stage_seeds_pins_and_code_are_refused(self):
        for key, mutate, text in (("schema", lambda r: r.update(schema="epr1-v0"), "schema/stage"), ("stage", lambda r: r.update(stage="LC"), "schema/stage"),
                                  ("seeds", lambda r: r["params"].update(seeds=[0]), "fit seeds"), ("pins", lambda r: r["inputs"].update(pins_overridden=["seeds"]), "overrode"),
                                  ("code", lambda r: r.update(code_sha256="0" * 64), "code sha256"), ("fraction", lambda r: r.update(fraction_pct=7), "fraction"),
                                  ("cfg", lambda r: r["c_star"].update(config_sha256="0" * 64), "config sha256"),
                                  ("p2row", lambda r: r["inputs"].update(p2_row_sha256="0" * 64), "different P2")):
            w = self.world(name=f"w_{key}", outcomes=False)
            mutate(w.r1)
            w.write_r1()
            self.assertRefused(w, w.opts(), text)

    def test_different_epr1_py_is_refused(self):
        w = self.world(outcomes=False)
        self.assertRefused(w, w.opts(epr1_sha256="0" * 64), "landed")

    def test_grid_edge_is_refused_without_a_waiver(self):
        w = self.world(outcomes=False, lr=1e-3)
        self.assertRefused(w, w.opts(), "E1")

    def test_grid_edge_waiver_must_name_this_cstar(self):
        w = self.world(outcomes=False, lr=1e-3)
        wp = os.path.join(w.root, "waiver.json")
        wj(wp, {"waiver_of": "E1", "reason": "r", "decided_by": "MM", "c_star_config_sha256": "0" * 64})
        self.assertRefused(w, w.opts(mm_edge_waiver=wp), "different c*")
        wj(wp, {"waiver_of": "E2", "reason": "r", "decided_by": "MM", "c_star_config_sha256": w.cfg.sha256()})
        self.assertRefused(w, w.opts(mm_edge_waiver=wp), "not E1")
        wj(wp, {"waiver_of": "E1", "reason": "", "decided_by": "MM", "c_star_config_sha256": w.cfg.sha256()})
        self.assertRefused(w, w.opts(mm_edge_waiver=wp), "lacks")

    def test_edge_mutant_that_drops_the_refusal_is_red(self):
        w = self.world(outcomes=False, lr=1e-3)
        mut = load_mutant(("E1", "star.lr == R.GRID_LR[-1]", "False"))
        with self.assertRaises(D.DevRefusal):
            D.check_r1_result(w.opts().guard(), w.opts(), w.W6, w.b6)
        o = w.opts()
        try:
            mut.check_r1_result(o.guard(), mut.DevOptions(**{**o.__dict__}), w.W6, w.b6)
        except mut.DevRefusal as ex:
            self.fail(f"the mutant must NOT refuse the edge c*, but did: {ex}")

    def test_grid_edge_with_a_valid_waiver_passes_the_refusals_and_is_recorded(self):
        w = self.world(lr=1e-3)
        wp = os.path.join(w.root, "waiver.json")
        wj(wp, {"waiver_of": "E1", "reason": "MM: EP-R1b registered and shown not to be needed", "decided_by": "Master Mind", "c_star_config_sha256": w.cfg.sha256()})
        r = D.run_dev(w.opts(mm_edge_waiver=wp))
        self.assertEqual(r["inputs"]["mm_edge_waiver"]["decided_by"], "Master Mind")
        self.assertEqual(r["inputs"]["mm_edge_waiver"]["sha256"], sha_f(wp))

    def test_missing_c2_is_refused(self):
        w = self.world(outcomes=False)
        os.remove(os.path.join(w.run_dir, "epr1_c2_result.json"))
        self.assertRefused(w, w.opts(), "C2")

    def test_c2_of_another_cstar_or_bank_is_refused(self):
        w = self.world(outcomes=False)
        w.c2["c_star"] = dict(w.c2["c_star"], config_sha256="0" * 64)
        w.write_c2()
        self.assertRefused(w, w.opts(), "not the C2 stage of this c*")
        w2 = self.world(name="w2", outcomes=False)
        w2.c2["bank_manifest_sha256"] = "0" * 64
        w2.write_c2()
        self.assertRefused(w2, w2.opts(), "different bank")

    def test_tampered_dev_input_is_refused_before_np_load(self):
        w = self.world(outcomes=False)
        with open(os.path.join(w.dev_dir, "proposals.npz"), "ab") as f:
            f.write(b"x")
        self.assertRefused(w, w.opts(), "!= the landed")

    def test_wrong_token_or_log_count_is_refused(self):
        w = self.world(outcomes=False)
        self.assertRefused(w, w.opts(expect_tokens=w.N + 1), "tokens, expected")
        self.assertRefused(w, w.opts(expect_logs=w.n_logs + 1), "logs, expected")

    def test_a_red_control_stops_before_the_outcomes_are_opened(self):
        w = self.world(outcomes=False)
        pr = w.props.copy()
        pr[4, 4, 4, 1] += 1e-3
        np.savez(os.path.join(w.dev_dir, "proposals.npz"), token=w.token, log=w.log, proposals=pr, pick=w.pick, rule=np.array("navsim_v1"), ckpt=np.array("synthetic"))
        pins = dict(w.pins)
        pins["proposals.npz"] = sha_f(os.path.join(w.dev_dir, "proposals.npz"))
        self.assertRefused(w, w.opts(dev_pins=pins), "C4_fan_frozen")

    def test_a_wrong_head_checkpoint_is_refused(self):
        w = self.world(outcomes=False)
        sd = torch.load(w.ckpt, weights_only=True)
        sd["model"]["score_head.weight"] = sd["model"]["score_head.weight"] + 0.1
        torch.save(sd, w.ckpt)
        self.assertRefused(w, w.opts(), "different P2")

    def test_a_seven_output_head_is_refused(self):
        w = self.world(outcomes=False)
        torch.save({"model": {"score_head.weight": torch.zeros(7, R.FEAT_DIM), "score_head.bias": torch.zeros(7)}}, w.ckpt)
        self.assertRefused(w, w.opts(), "7-output")


class ReadOnce(Base):
    def test_second_read_is_refused(self):
        w = self.world()
        D.run_dev(w.opts())
        with self.assertRaises(D.DevRefusal) as cm:
            D.run_dev(w.opts())
        self.assertIn("ONCE", str(cm.exception))

    def test_read_once_mutant_is_red(self):
        w = self.world()
        D.run_dev(w.opts())
        mut = load_mutant(("ONCE", "_io_exists(guard, p)", "False"))
        with self.assertRaises(Exception) as cm:
            mut.run_dev(mut.DevOptions(**{**w.opts().__dict__}))
        self.assertNotIsInstance(cm.exception, (mut.DevRefusal, D.DevRefusal))
        self.assertNotIn("ONCE", str(cm.exception))

    def test_a_crashed_first_attempt_still_consumed_the_read(self):
        w = self.world()
        o = w.opts()
        os.makedirs(o.out_dir, exist_ok=True)
        wj(os.path.join(o.out_dir, D.READ_STARTED), {"started": "x"})
        with self.assertRaises(D.DevRefusal):
            D.run_dev(o)

    def test_marker_is_written_before_the_outcomes_and_names_the_result_hash(self):
        w = self.world()
        D.run_dev(w.opts())
        m = json.loads(_rt(os.path.join(w.opts().out_dir, D.READ_STARTED)))
        self.assertEqual(m["c_star_result_sha256"], w.r1_sha())

    def test_outcome_free_world_that_passes_every_refusal_dies_with_keyerror_after_the_marker(self):
        """The proof of ORDER: with no outcome arrays, a run that survives all refusals writes READ_STARTED and only THEN fails on the outcomes."""
        w = self.world(outcomes=False)
        with self.assertRaises(KeyError):
            D.run_dev(w.opts())
        self.assertTrue(os.path.exists(os.path.join(w.opts().out_dir, D.READ_STARTED)))

    def test_the_input_phase_never_names_an_outcome_array(self):
        src = _rt(SRC)
        tree = ast.parse(src)
        fn = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for name in ("load_dev_inputs", "integrity_featcheck", "c0_identity", "c1_reach", "c4_fan_frozen", "apply_rows"):
            txt = ast.get_source_segment(src, fn[name])
            for key in ('"pdms"', '"sub"', '"valid"', '"sub_names"'):
                self.assertNotIn(key, txt, f"{name} must not open {key}")


class Guard(Base):
    def test_guard_refuses_navtest_and_the_fit_bank(self):
        g = D.DevGuard(read_dirs=["C:/x"], write_dirs=["C:/x"])
        for p in ("C:/x/navtest/table.npz", "C:/x/table_p2_full.npz", "C:/Users/Admin/srttc_s0/work/bank/L1.sets.npz", "C:/x/a.sets.npz", "C:/x/s1_read_cache/a.npz", "C:/x/../x/ranked_fit_logs.tsv"):
            with self.assertRaises(D.DevGuardError, msg=p):
                g.check_read(p)
        with self.assertRaises(D.DevGuardError):
            g.check_read("C:/elsewhere/file.npy")
        with self.assertRaises(D.DevGuardError):
            g.check_write("C:/elsewhere/out.json")
        g.check_read("C:/x/ok.npy")
        g.check_write("C:/x/ok.json")

    def test_guard_does_not_refuse_the_heldout_dev_table(self):
        D.DevGuard(read_dirs=[D.DEV_DIR]).check_read(D.DEV_DIR + "/table_p2_heldout.npz")

    def test_symlink_free_dotdot_cannot_smuggle_navtest(self):
        g = D.DevGuard(read_dirs=["C:/x"])
        with self.assertRaises(D.DevGuardError):
            g.check_read("C:/x/sub/../navtest_stuff.npz")

    def test_no_filesystem_call_outside_an_io_function(self):
        bad = fs_violations(_rt(SRC))
        self.assertEqual(bad, [], f"file-system access outside _io_*: {bad}")

    def test_ast_scan_has_teeth(self):
        """The scan must SEE a planted violation (a probe that cannot read a defect is not a probe)."""
        base = _rt(SRC)
        self.assertEqual(fs_violations(base), [])
        for plant in ("np.load(p)", "open(p, 'rb')", "os.listdir(p)", "torch.load(p)", "os.path.exists(p)", "np.savez(p)"):
            planted = base + f"\n\ndef sneaky(p):\n    return {plant}\n"
            self.assertEqual(fs_violations(planted), [("sneaky", plant.split("(")[0])], plant)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (e) end to end
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class EndToEnd(Base):
    def test_pass(self):
        w = self.world(scales=(3.0, 3.5))
        r = D.run_dev(w.opts())
        self.assertEqual(r["dev_gate"]["verdict"], "PASS", json.dumps(r["primary"]["bar"]))
        for sd in D.SEED_KEYS:
            g = r["primary"]["per_seed"][sd]["gain_x100"]["point"]
            self.assertGreaterEqual(g, D.DEV_BAR_X100)
        self.assertTrue(r["dev_gate"]["controls_all_pass"])
        for k in ("integrity_featcheck", "C0_pick_identity", "C0_gain_zero", "C1_reach", "C2_negative", "C4_fan_frozen"):
            self.assertTrue(r["controls"][k]["pass"], k)
        self.assertEqual(r["outcome_rows"]["fired"], ["D-PASS"])
        self.assertEqual(r["estimator"]["B"], 10000)
        self.assertEqual(r["estimator"]["n_logs"], 24)
        self.assertIn("PAUSED", r["dev_gate"]["strategic_family"])
        self.assertFalse(r["joint_secondary"]["computed"])
        self.assertIn("reason", r["joint_secondary"])

    def test_result_file_and_sidecar_agree(self):
        w = self.world()
        D.run_dev(w.opts())
        out = w.opts().out_dir
        side = _rt(os.path.join(out, D.SIDECAR_NAME)).split()
        self.assertEqual(side[0], sha_f(os.path.join(out, D.RESULT_NAME)))
        doc = json.loads(_rt(os.path.join(out, D.RESULT_NAME)))
        self.assertEqual(doc["inputs"]["epr1_result_sha256"], w.r1_sha())
        self.assertEqual(doc["schema"], D.SCHEMA)

    def test_determinism_byte_identical_across_runs(self):
        w = self.world()
        D.run_dev(w.opts(out_dir=os.path.join(w.root, "out_a")))
        D.run_dev(w.opts(out_dir=os.path.join(w.root, "out_b")))
        a = _rb(os.path.join(w.root, "out_a", D.RESULT_NAME))
        b = _rb(os.path.join(w.root, "out_b", D.RESULT_NAME))
        da, db = json.loads(a), json.loads(b)
        for d in (da, db):
            d["run_meta"].pop("elapsed_s")
            d["run_meta"].pop("finished")
        self.assertEqual(da, db)

    def test_weak_row_fails_the_bar_with_controls_green(self):
        w = self.world(scales=(1e-6, 1e-6))
        r = D.run_dev(w.opts())
        self.assertEqual(r["dev_gate"]["verdict"], "FAIL")
        self.assertTrue(r["dev_gate"]["controls_all_pass"])
        self.assertLess(r["primary"]["per_seed"]["0"]["gain_x100"]["point"], D.DEV_BAR_X100)

    def test_one_seed_below_the_bar_fails(self):
        w = self.world(scales=(3.0, 1e-6))
        r = D.run_dev(w.opts())
        self.assertGreaterEqual(r["primary"]["per_seed"]["0"]["gain_x100"]["point"], D.DEV_BAR_X100)
        self.assertLess(r["primary"]["per_seed"]["1"]["gain_x100"]["point"], D.DEV_BAR_X100)
        self.assertEqual(r["dev_gate"]["verdict"], "FAIL")
        self.assertGreater(r["primary"]["bar"]["seed_gap_abs"], 0.0)

    def test_c2_with_a_positive_gain_voids_the_arm(self):
        w = self.world(c2_scale=3.0)
        r = D.run_dev(w.opts())
        self.assertFalse(r["controls"]["C2_negative"]["pass"])
        self.assertEqual(r["dev_gate"]["verdict"], "VOID")
        self.assertIn("A", r["outcome_rows"]["fired"])
        self.assertNotIn("D-PASS", r["outcome_rows"]["fired"])

    def test_c2_mutant_is_red(self):
        mut = load_mutant(("C2", "[\"point\"] <= 0.0", "[\"point\"] >= 0.0"))
        w = self.world(c2_scale=-3.0)
        self.assertEqual(D.run_dev(w.opts(out_dir=os.path.join(w.root, "o1")))["dev_gate"]["verdict"], "PASS")
        w2 = self.world(name="w2", c2_scale=-3.0)
        o2 = w2.opts(out_dir=os.path.join(w2.root, "o2"))
        r = mut.run_dev(mut.DevOptions(**{**o2.__dict__}))
        self.assertNotEqual(r["dev_gate"]["verdict"], "PASS", "the mutant must flip the C2 reading")

    def test_pick_from_stored_logits_mutant_is_red(self):
        mut = load_mutant(("PICK-NEW", "pick_logits(z_new[sd])", "pick_logits(stored)"))
        w = self.world(scales=(3.0, 3.5))
        self.assertEqual(D.run_dev(w.opts(out_dir=os.path.join(w.root, "o1")))["dev_gate"]["verdict"], "PASS")
        w2 = self.world(name="w2", scales=(3.0, 3.5))
        o2 = w2.opts(out_dir=os.path.join(w2.root, "o2"))
        r = mut.run_dev(mut.DevOptions(**{**o2.__dict__}))
        self.assertEqual(r["primary"]["per_seed"]["0"]["gain_x100"]["point"], 0.0)
        self.assertEqual(r["dev_gate"]["verdict"], "FAIL")

    def test_leak_read_is_reported_and_labelled(self):
        w = self.world()
        r = D.run_dev(w.opts())
        leak = r["primary"]["per_seed"]["0"]["leak_c7_dev"]
        self.assertIn("on_safe", leak["new"])
        self.assertIn("on_unsafe", leak["p2"])

    # ---- the JOINT secondary
    def test_joint_computed_labelled_and_never_moves_the_verdict(self):
        w = self.world(ttc=True)
        tsha = sha_f(os.path.join(w.ttc_dir, "r1_result.json"))
        r = D.run_dev(w.opts(out_dir=os.path.join(w.root, "oj"), ttc_run_dir=w.ttc_dir, ttc_registered_result_sha256=tsha))
        j = r["joint_secondary"]
        self.assertTrue(j["computed"])
        self.assertIn("SECONDARY", j["label"])
        for sd in D.SEED_KEYS:
            self.assertIn("vs_ep_r1_alone", j["per_seed"][sd])
            self.assertIn("gain_x100", j["per_seed"][sd])
            self.assertIn("meets_dev_bar_informational", j["per_seed"][sd])
        w2 = self.world(name="w2", ttc=False)
        r2 = D.run_dev(w2.opts())
        self.assertEqual(r["dev_gate"]["verdict"], r2["dev_gate"]["verdict"])
        self.assertEqual(r["primary"]["per_seed"]["0"]["gain_x100"], r2["primary"]["per_seed"]["0"]["gain_x100"])
        self.assertEqual(r["outcome_rows"]["fired"], r2["outcome_rows"]["fired"])

    def test_joint_with_a_zero_delta_ttc_row_equals_ep_alone(self):
        """Known-value identity for the joint: SR-TTC's TTC row replaced by P2's own TTC row => the joint picks == EP-R1 alone."""
        w = self.world(ttc=True, ttc_scales=(0.0, 0.0))
        tsha = sha_f(os.path.join(w.ttc_dir, "r1_result.json"))
        r = D.run_dev(w.opts(ttc_run_dir=w.ttc_dir, ttc_registered_result_sha256=tsha))
        for sd in D.SEED_KEYS:
            self.assertEqual(r["joint_secondary"]["per_seed"][sd]["vs_ep_r1_alone"]["point"], 0.0)
            self.assertEqual(r["joint_secondary"]["per_seed"][sd]["gain_x100"]["point"], r["primary"]["per_seed"][sd]["gain_x100"]["point"])

    def test_joint_not_computed_with_the_reason_named(self):
        cases = []
        w = self.world(name="w1", ttc=True)
        cases.append((w, w.opts(ttc_run_dir=w.ttc_dir), "registered beforehand"))
        w = self.world(name="w2", ttc=True)
        cases.append((w, w.opts(ttc_run_dir=w.ttc_dir, ttc_registered_result_sha256="0" * 64), "!= the registered"))
        w = self.world(name="w3", ttc=True)
        w.make_ttc((2.0, 2.0), fraction=25)
        cases.append((w, w.opts(ttc_run_dir=w.ttc_dir, ttc_registered_result_sha256=sha_f(os.path.join(w.ttc_dir, "r1_result.json"))), "fraction_pct differs"))
        w = self.world(name="w4", ttc=True)
        w.make_ttc((2.0, 2.0), bank_man="e" * 64)
        cases.append((w, w.opts(ttc_run_dir=w.ttc_dir, ttc_registered_result_sha256=sha_f(os.path.join(w.ttc_dir, "r1_result.json"))), "bank_manifest_sha256 differs"))
        w = self.world(name="w5", ttc=False)
        cases.append((w, w.opts(ttc_run_dir=w.ttc_dir, ttc_registered_result_sha256="0" * 64), "does not exist"))
        for i, (w, o, text) in enumerate(cases):
            o.out_dir = os.path.join(w.root, f"o{i}")
            r = D.run_dev(o)
            self.assertFalse(r["joint_secondary"]["computed"])
            self.assertIn(text, r["joint_secondary"]["reason"])
            self.assertEqual(r["dev_gate"]["verdict"], "PASS", "a missing joint never changes the primary")

    def test_joint_with_tampered_ttc_row_is_not_computed(self):
        w = self.world(ttc=True)
        tsha = sha_f(os.path.join(w.ttc_dir, "r1_result.json"))
        with open(os.path.join(w.ttc_dir, "ttc_row_seed0.npy"), "ab") as f:
            f.write(b"zz")
        r = D.run_dev(w.opts(ttc_run_dir=w.ttc_dir, ttc_registered_result_sha256=tsha))
        self.assertFalse(r["joint_secondary"]["computed"])
        self.assertIn("unusable", r["joint_secondary"]["reason"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# pins
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Pins(Base):
    def test_epr1_py_is_the_landed_file(self):
        self.assertEqual(sha_f(os.path.join(HERE, "epr1.py")), D.EPR1_SHA256, "epr1_dev.py is pinned to the landed epr1.py")

    def test_zero_tail_is_the_vendored_file(self):
        self.assertEqual(sha_f(os.path.join(HERE, "zero_tail.py")), "149c3e97927ed219323f8f070b67a36d9172a575e4ec23162241c417f8501a3d")

    def test_dev_pins_are_well_formed(self):
        self.assertEqual(set(D.DEV_PINS), set(D.DEV_FILES))
        for v in D.DEV_PINS.values():
            self.assertRegex(v, r"^[0-9a-f]{64}$")
        self.assertEqual(D.DEV_N_TOKENS, 1705)
        self.assertEqual(D.DEV_N_LOGS, 24)

    def test_registered_constants_come_from_epr1(self):
        self.assertEqual((D.EP_COL, D.TTC_COL, R.BOOT_B, R.BOOT_SEED), (2, 3, 10000, 20260927))
        self.assertEqual(D.COMPS, R.LABEL_ORDER)
        self.assertEqual(D.DEV_BAR_X100, 0.30)

    def test_sub_names_order_equals_label_order(self):
        self.assertEqual(tuple(D.SUB_NAMES[i] for i in range(6)), ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
                                                                  "driving_direction_compliance"))

    def test_every_mutation_marker_in_the_source_is_used_by_a_test(self):
        src = _rt(SRC)
        tests = _rt(os.path.abspath(__file__))
        markers = sorted({m[len("MUTATION-TARGET-"):] for m in re.findall(r"MUTATION-TARGET-[A-Z0-9-]+", src)})
        self.assertGreaterEqual(len(markers), 14)
        for m in markers:
            self.assertGreaterEqual(tests.count(f'"{m}"'), 1, f"marker {m} is not exercised by any load_mutant call")


if __name__ == "__main__":
    unittest.main(verbosity=2)
