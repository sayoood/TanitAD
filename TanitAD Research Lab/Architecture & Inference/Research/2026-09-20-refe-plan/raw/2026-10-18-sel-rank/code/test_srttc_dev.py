"""Tests for srttc_dev.py (the SR-TTC dev gate) and srttc_c2.py (its C2 fit) on SYNTHETIC worlds only (never the real S0 bank, never the real dev outcomes).  stdlib unittest (also collected by pytest):

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_srttc_dev.py -v

The synthetic world is built by the LANDED code, not by a mimic: `test_srttc_r1.make_world` (a bank), then the real `srttc_r1.run_r1`, `run_lc` and `srttc_c2.run_c2`, so r1_result.json / lc_result.json /
r1_c2_result.json have exactly the schema the dev gate will meet.  Dev tables are generated here from the same P2 head and the same TTC direction.

(a) IDENTITY ARM: the unchanged TTC row gives P2's pick on EVERY token (bit-identical logits), gain exactly 0.0, and C1's applier-side reach goes RED for such a no-op row.
(b) KNOWN VALUES: a hand-built world whose gain is computable by hand; the planner rule against an independent pure-python oracle; the bootstrap equals zero_tail.Boot; ONLY column 3 changes;
    the registered LC call equals srttc_r1.lc_call; c* recomputed from the grid; the HMAC equals srttc_r1_gate's.
(c) REFUSALS fire BEFORE any dev OUTCOME is opened: the synthetic table of every refusal test has NO pdms / sub / valid / sub_names member at all, so an early read would be a KeyError, not a refusal.
(d) EVERY CHECK HAS A DELIBERATE-REGRESSION ARM: the source of srttc_dev.py (and srttc_c2.py) is mutated at ONE marked line and the check must go RED while the same scenario on the unmutated module is
    GREEN; a coverage test requires every marker in the source to be mutated at least once here.
(e) CONTRACT ROUND-TRIP: `refe_head_confirm.read_dev_result` (the landed confirm applier) accepts this module's output and refuses corrupted copies; `load_arm_rows` binds the rows the dev gate applied.
(f) REACHABILITY: every CLI flag is traced into DevOptions; an AST scan proves no file-system call lives outside an `_io_*` function; the dev guard refuses navtest / the fit bank.
"""
from __future__ import annotations

import ast
import atexit
import copy
import dataclasses
import hashlib
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
EPDIR = os.path.normpath(os.path.join(HERE, "..", "..", "2026-10-18-ep-r1", "code"))
import srttc_r1 as R          # noqa: E402
import srttc_dev as D         # noqa: E402
import srttc_c2 as C2         # noqa: E402
import srttc_r1_gate as G     # noqa: E402
import zero_tail as ZT        # noqa: E402
import test_srttc_r1 as TR    # noqa: E402

SRC = os.path.join(HERE, "srttc_dev.py")
SRC_C2 = os.path.join(HERE, "srttc_c2.py")
NP_ = D.N_PROPOSALS
SK = D.SEED_KEYS


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def sha_f(p):
    with open(p, "rb") as f:
        return sha_b(f.read())


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def wj(p, obj):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# mutation harness
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _exec_module(src, path, name):
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod                     # dataclasses look the defining module up in sys.modules
    try:
        exec(compile(src, path, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


def load_mutant(*edits, path=SRC, prefix="SRDEV", name="srttc_dev_mutant"):
    """The module with ONE marked line edited per (marker, old, new): `MUTATION-TARGET-<prefix>-<marker>` must sit on exactly one line and `old` must occur exactly once ON THAT LINE."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for marker, old, new in edits:
        pat = re.compile(rf"MUTATION-TARGET-{prefix}-{re.escape(marker)}(?![A-Z0-9-])")
        idx = [i for i, ln in enumerate(lines) if pat.search(ln)]
        assert len(idx) == 1, f"marker {marker} sits on {len(idx)} lines"
        assert lines[idx[0]].count(old) == 1, f"{old!r} occurs {lines[idx[0]].count(old)} times on the {marker} line"
        lines[idx[0]] = lines[idx[0]].replace(old, new)
    return _exec_module("\n".join(lines), path, name)


def list_markers(path=SRC, prefix="SRDEV"):
    with open(path, encoding="utf-8") as f:
        return sorted(set(re.findall(rf"MUTATION-TARGET-{prefix}-([A-Z0-9-]+?)(?![A-Z0-9-])", f.read())))


MUTATED = set()          # markers exercised by a mutation test (checked by the coverage test; filled while the tests run, so the coverage test is named to sort last)


def mopts(mod, opts):
    return mod.DevOptions(**{**opts.__dict__})


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the base world: the real R1 -> LC -> C2 chain on a synthetic bank (built once)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
_BASE = None


def base():
    global _BASE
    if _BASE is None:
        root = tempfile.mkdtemp(prefix="srttc_devtest_base_")
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        w = TR.make_world(os.path.join(root, "w"))
        out = os.path.join(root, "r1run")
        o = TR.opts_for(w, out)
        r1 = R.run_r1(o)
        lc = R.run_lc(o)
        c2 = C2.run_c2(o)
        _BASE = {"root": root, "w": w, "run": out, "r1": r1, "lc": lc, "c2": c2, "opts": o}
    return _BASE


def ref_pick(z):
    """INDEPENDENT oracle of the planner rule: python floats, first strict maximum (lowest index on ties)."""
    best, bi = -1.0, 0
    for j in range(z.shape[0]):
        p = [1.0 / (1.0 + math.exp(-float(z[j, c]))) for c in range(6)]
        a = p[0] * p[1] * (5 * p[2] + 5 * p[3] + 2 * p[4]) / 12.0
        if a > best:
            best, bi = a, j
    return bi


class DW:
    """One dev world: a run dir (copied from the base chain, rows optionally replaced), dev tables for P2's head, a gate token + key, a held-out-package dir with its attestations."""

    def __init__(self, root, *, outcomes=True, scales=(3.0, 3.0), c2_scales=(-3.0, -3.0), lc="FLAT", fraction=10, n_logs=24, per_log=6, seed=11, token=True, p2ho=True):
        b = base()
        self.root, self.w, self.outcomes = root, b["w"], outcomes
        os.makedirs(root, exist_ok=True)
        self.run = os.path.join(root, "run")
        shutil.copytree(b["run"], self.run)
        self.dev = os.path.join(root, "dev")
        self.p2ho = os.path.join(root, "p2ho_pkg")
        self.gate = os.path.join(root, "gate")
        for d in (self.dev, self.p2ho, self.gate):
            os.makedirs(d, exist_ok=True)
        self.n_logs, self.per_log = n_logs, per_log
        self.N = n_logs * per_log
        self.make_tables(seed)
        if scales is not None:
            self.set_rows("real", scales)
        if c2_scales is not None:
            self.set_rows("c2", c2_scales)
        self.set_lc(lc)
        if fraction != 10:
            self.set_fraction(fraction)
        self.key = os.path.join(self.gate, "launch_gate.key")
        with open(self.key, "w", encoding="ascii") as f:
            f.write(hashlib.sha256(b"unit-test-key").hexdigest() + "\n")
        self.token_path = os.path.join(self.gate, "r1_gate_token.json")
        if token:
            self.write_token()
        if p2ho:
            self.write_p2ho()
        self._n = 0

    # ---- dev tables (a P2-head-consistent synthetic held-out table)
    def make_tables(self, seed):
        w, rng = self.w, np.random.default_rng(seed)
        N = self.N
        names = [f"2021.05.{i % 28 + 1:02d}.10.00.{i:02d}_veh-{i % 7}_{i:05d}_{i + 9:05d}" for i in range(self.n_logs)]
        self.token = np.array([f"{i:016x}" for i in range(N)])
        self.log = np.array([names[i // self.per_log] for i in range(N)])
        feat = rng.normal(size=(N, NP_, D.FEAT_DIM)).astype(np.float16)
        f32 = feat.astype(np.float32)
        self.feat = feat
        self.logits = (f32 @ w["W6"].T + w["b6"]).astype(np.float32)
        nc = (rng.random((N, NP_)) < 0.93).astype(np.float64)
        dac = (rng.random((N, NP_)) < 0.93).astype(np.float64)
        ttc = ((f32 @ w["u"]) * 1.5 + rng.normal(size=(N, NP_)) * 0.7 > 0).astype(np.float64)
        sub = np.stack([nc, dac, rng.random((N, NP_)), ttc, (rng.random((N, NP_)) < 0.85).astype(np.float64), (rng.random((N, NP_)) < 0.97).astype(np.float64)], -1)
        self.sub = sub
        self.pdms = sub[..., 0] * sub[..., 1] * (5.0 * sub[..., 2] + 5.0 * sub[..., 3] + 2.0 * sub[..., 4]) / 12.0
        self.props = rng.normal(size=(N, NP_, 8, 3)).astype(np.float32)
        self.pick = R.picks_np(self.logits)
        self.write_tables()

    def write_tables(self, pick=None, props_dump=None, feat=None, logits=None, tokens_dump=None):
        T = {"token": self.token, "log": self.log, "chunk": np.zeros(self.N, np.int32), "fingerprint": self.token, "logits": self.logits if logits is None else logits,
             "proposals": self.props, "pick": self.pick if pick is None else pick}
        if self.outcomes:
            T.update(pdms=self.pdms, sub=self.sub, valid=np.ones((self.N, NP_), bool), sub_names=np.array(D.SUB_NAMES))
        np.savez(os.path.join(self.dev, "table_p2_heldout.npz"), **T)
        np.savez(os.path.join(self.dev, "prehead_features_p2_heldout.npz"), token=self.token, log=self.log, prehead_features=self.feat if feat is None else feat, what=np.array("synthetic"))
        np.savez(os.path.join(self.dev, "proposals.npz"), token=self.token if tokens_dump is None else tokens_dump, log=self.log, proposals=self.props if props_dump is None else props_dump,
                 pick=self.pick, rule=np.array("navsim_v1"), ckpt=np.array("synthetic"))
        self.pins = {n: sha_f(os.path.join(self.dev, n)) for n in D.DEV_FILES}

    # ---- run-dir edits
    def r1doc(self):
        return rj(os.path.join(self.run, "r1_result.json"))

    def save_r1(self, doc):
        wj(os.path.join(self.run, "r1_result.json"), doc)

    def reg_sha(self):
        return sha_f(os.path.join(self.run, "r1_result.json"))

    def row_for(self, scale, sd):
        W6, b6, u = self.w["W6"], self.w["b6"], self.w["u"]
        return (W6[3] + np.float32(scale) * u + (np.float32(1e-3 * scale * int(sd)) * np.cos(np.arange(D.FEAT_DIM))).astype(np.float32), b6[3:4].copy())

    def set_rows(self, kind, scales):
        """kind 'real' | 'c2'.  scales: a pair of floats (rows P2-row + scale*u), or {sd: (w, b)}."""
        doc = self.r1doc() if kind == "real" else rj(os.path.join(self.run, "r1_c2_result.json"))
        for sd, sc in zip(SK, scales):
            w, b = self.row_for(sc, sd) if not isinstance(scales, dict) else scales[sd]
            self._put_row(doc, kind, sd, w, b)
        if kind == "real":
            self.save_r1(doc)
        else:
            wj(os.path.join(self.run, "r1_c2_result.json"), doc)

    def set_row_arrays(self, kind, rows):
        doc = self.r1doc() if kind == "real" else rj(os.path.join(self.run, "r1_c2_result.json"))
        for sd in SK:
            self._put_row(doc, kind, sd, *rows[sd])
        if kind == "real":
            self.save_r1(doc)
        else:
            wj(os.path.join(self.run, "r1_c2_result.json"), doc)

    def _put_row(self, doc, kind, sd, w, b):
        name = f"ttc_row_seed{sd}.npy" if kind == "real" else f"ttc_row_c2shuffle_seed{sd}.npy"
        blob = R.row_bytes(w, b)
        with open(os.path.join(self.run, name), "wb") as f:
            f.write(blob)
        doc["final_rows"][sd] = {"file": name, "sha256": sha_b(blob), "bytes": len(blob)}

    def set_lc(self, kind, per_seed=None):
        """Rewrite lc_result.json with intervals that give the wanted call (recorded call == recomputed call)."""
        doc = rj(os.path.join(self.run, "lc_result.json"))
        iv = {"FLAT": (-0.004, 0.004), "RISING": (0.02, 0.05), "INCONCLUSIVE": (-0.01, 0.03), "BOTH": (0.002, 0.008)}
        if per_seed is None:
            lo, hi = iv[kind]
            per_seed = {sd: {"point": (lo + hi) / 2, "lo": lo, "hi": hi, "B": 10000, "n_nan": 0} for sd in SK}
        doc["delta_100_minus_50"] = per_seed
        call = D.lc_call_registered(per_seed)
        doc["call"] = {"call": call["call"], "rising": call["rising"], "flat": call["flat"], "both_true": call["rising"] and call["flat"], "both_rule": "flat", "delta_flat_threshold": 0.01}
        wj(os.path.join(self.run, "lc_result.json"), doc)
        return call

    def set_fraction(self, f):
        d = self.r1doc()
        d["fraction_pct"] = f
        self.save_r1(d)
        for nm in ("lc_result.json", "r1_c2_result.json"):
            x = rj(os.path.join(self.run, nm))
            x["fraction_pct"] = f
            wj(os.path.join(self.run, nm), x)
        if os.path.exists(self.token_path if hasattr(self, "token_path") else ""):
            self.write_token()
        self.fraction = f

    # ---- gate token + key + held-out package
    def write_token(self, **over):
        d = self.r1doc()
        with open(self.key, encoding="ascii") as kf:
            key = bytes.fromhex(kf.read().strip())
        b = {"profile": "srttc-r1", "commit": "0" * 40, "stage": "run", "fraction": d["fraction_pct"], "argv_sha256": "a" * 64, "srttc_r1_sha256": d["code_sha256"],
             "bank_manifest_sha256": d["inputs"]["bank_manifest_sha256"]}
        b.update(over.pop("binding", {}))
        body = {"schema": G.GATE_SCHEMA, "kind": G.TOKEN_KIND, "verdict": "PASS", "binding": b, "checks": {}, "reasons": [], "gate_sha256": "b" * 64, "created_utc": "2026-10-19T00:00:00Z",
                "key_id": G.key_id(key)}
        body.update(over)
        body["hmac_sha256"] = G.sign({k: v for k, v in body.items() if k != "hmac_sha256"}, key)
        wj(self.token_path, body)

    def write_p2ho(self, done=None, rc=None, man=None):
        N = self.N
        done = done or {"when": "2026-10-18T10:25:37+0200", "n_tokens": N, "PARTIAL": False,
                        "coverage": {"n_tokens_in_plan": N, "n_logs_in_plan": self.n_logs, "n_tokens_banked": N, "n_logs_banked": self.n_logs, "n_tokens_blocked": 0, "blocked_reasons": {}},
                        "proofs": {"disjointness": "disjointness_proof.json PASS", "fidelity_A": True, "dgdump": True, "G3": True, "G4a_pick_from_logits": f"{N}/{N}", "G4b_pdms_formula_max_abs_diff": 0.0},
                        "sets_index": {"n_sets": 2 * N, "n_sets_with_row": 2 * N}, "manifest": "MANIFEST.sha256.json", "n_files": 3}
        rc = rc or {"n_old": 333, "n_new": N, "n_old_tokens_missing_in_new": 0, "n_compared": 333, "logits_max_abs": 0.0, "proposals_max_abs": 0.0, "pdms_max_abs": 0.0, "sub_max_abs": 0.0,
                    "pick_equal": 333, "pdms_at_pick_max_abs": 0.0, "log_equal": True, "PASS": True}
        man = man or {"written": "x", "what": "synthetic", "files": {n: {"path": n, "sha256": self.pins[n], "bytes": 1} for n in D.DEV_FILES}}
        wj(os.path.join(self.p2ho, "P2HO_DONE"), done)
        wj(os.path.join(self.p2ho, "recompute_check_333.json"), rc)
        wj(os.path.join(self.p2ho, "MANIFEST.sha256.json"), man)

    # ---- options
    def opts(self, **kw):
        self._n += 1
        base_ = dict(run_dir=self.run, registered_result_sha256=self.reg_sha(), dev_dir=self.dev, p2_ckpt=self.w["ckpt"], out_dir=os.path.join(self.root, f"out{self._n}"), p2ho_dir=self.p2ho,
                     r1_gate_token=self.token_path, gate_key_file=self.key, dev_pins=dict(self.pins), expect_tokens=self.N, expect_logs=self.n_logs, expect_sets=2 * self.N,
                     expect_fit_logs=self.r1doc()["inputs"]["n_logs"], reg_epochs=2, reg_batch_sets=32, allow_r1_pin_overrides=True)
        base_.update(kw)
        return D.DevOptions(**base_)

    def run_dev(self, **kw):
        return D.run_dev(self.opts(**kw))


def refused(mod, world, fragment=None, **kw) -> bool:
    """True iff `mod.run_dev` raises DevRefusal (message containing `fragment`).  Any other outcome (a result, another exception) is 'the check did not fire'."""
    try:
        mod.run_dev(mopts(mod, world.opts(**kw)))
    except mod.DevRefusal as ex:
        return fragment is None or fragment in str(ex)
    except Exception:                                                     # noqa: BLE001
        return False
    return False


def completes(mod, world, **kw):
    """-> the result dict, or None if the run did not complete."""
    try:
        return mod.run_dev(mopts(mod, world.opts(**kw)))
    except Exception:                                                     # noqa: BLE001
        return None


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="srttc_devtest_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._k = 0

    def world(self, **kw):
        self._k += 1
        return DW(os.path.join(self.tmp, f"w{self._k}"), **kw)

    def mut(self, marker, old, new, scenario, *, path=SRC, prefix="SRDEV", name="srttc_dev_mutant"):
        """scenario(mod) -> True iff the check FIRES (refuses / goes RED).  GREEN on the unmutated module, and the one-line mutant must make it stop firing."""
        MUTATED.add((prefix, marker))
        self.assertTrue(scenario(D if path == SRC else C2), f"{marker}: the scenario must fire on the UNMUTATED module")
        mod = load_mutant((marker, old, new), path=path, prefix=prefix, name=name)
        self.assertFalse(scenario(mod), f"{marker}: the one-line mutant must make the check stop firing (the check was not what caught it)")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (a) identity
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Identity(Base):
    def test_unchanged_ttc_row_gives_p2s_pick_on_every_token_and_gain_exactly_zero(self):
        w = self.world(scales=None)
        P = w.w
        row = (P["W6"][3].copy(), P["b6"][3:4].copy())
        inp = D.load_dev_inputs(w.opts().guard(), w.opts())
        z = D.apply_rows(inp.z6, inp.feat, {3: row}, P["W6"], P["b6"])
        self.assertTrue(np.array_equal(z, inp.z6.numpy()), "bit-identical logits when the row IS P2's")
        np.testing.assert_array_equal(D.pick_logits(z), inp.pick_banked)
        c0 = D.c0_identity(inp, P["W6"], P["b6"])
        self.assertTrue(c0["pass"])
        self.assertEqual(c0["picks_identity_ttc_row"], inp.n)
        out = D.read_outcomes(w.opts().guard(), os.path.join(w.dev, "table_p2_heldout.npz"))
        z0 = D.c0_gain_zero(inp, out, D.pick_logits(inp.z6.numpy()), R.boot_draws(inp.n_logs))
        self.assertTrue(z0["pass"])
        self.assertEqual((z0["gain_point"], z0["gain_lo"], z0["gain_hi"], z0["changed_picks"]), (0.0, 0.0, 0.0, 0))

    def test_rows_equal_to_p2s_make_c1_reach_red_and_the_gate_refuses_before_any_outcome(self):
        w = self.world(scales=(0.0, 0.0), outcomes=False)
        P = w.w
        for sd in SK:                                                                 # the rows ARE P2's (scale 0 plus a seed-dependent 1e-3*0 term = exactly 0)
            self.assertTrue(np.array_equal(R.row_bytes(*w.row_for(0.0, sd)), R.row_bytes(P["W6"][3], P["b6"][3:4])))
        inp = D.load_dev_inputs(w.opts().guard(), w.opts())
        c1 = D.c1_applier_reach(inp, {sd: w.row_for(0.0, sd) for sd in SK}, P["W6"], P["b6"])
        self.assertFalse(c1["pass"], "a row that equals P2's changes nothing: reach must read RED, not green")
        self.assertTrue(all(v["ttc_column_entries_changed"] == 0 for v in c1["per_seed"].values()))
        self.assertGreater(c1["positive_control_negated_p2_row_changed_picks"], 0, "the positive control proves the probe CAN read a change")
        with self.assertRaises(D.DevRefusal) as cm:
            w.run_dev()
        self.assertIn("C1_reach", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(w.opts().resolved_out_dir(), D.READ_STARTED)), "a RED control leaves the read unspent")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (b) known values
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class HandBuilt(Base):
    """P2's head is the identity on the first six features (logit_k = feat_k): the stored TTC logit is feat[...,3]; a new TTC row e_6 gives new TTC logit feat[...,6]."""

    def build(self):
        n_tok = 6
        rng = np.random.default_rng(5)
        feat = np.zeros((n_tok, NP_, D.FEAT_DIM), np.float32)
        feat[..., 0] = feat[..., 1] = feat[..., 2] = feat[..., 4] = 20.0                 # NC, DAC, EP, C ~ certain: the rule reduces to argmax of pTTC
        a = np.array([3, 7, 11, 40, 0, 63])
        b = np.array([5, 9, 12, 41, 1, 62])
        for i in range(n_tok):
            old = rng.normal(size=NP_) * 0.5
            new = rng.normal(size=NP_) * 0.5
            old[a[i]] = 5.0                                                              # P2 picks a_i
            new[b[i]] = 5.0                                                              # the new row picks b_i
            feat[i, :, 3], feat[i, :, 6] = old, new
        W6 = np.zeros((D.N_COMP, D.FEAT_DIM), np.float32)
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
        sub[..., 3] = pdms
        log = np.array(["L0"] * 2 + ["L1"] * 2 + ["L2"] * 2)
        names, idx = np.unique(log, return_inverse=True)
        tok = np.array([f"t{i}" for i in range(n_tok)])
        inp = D.DevInputs(token=tok, log=log, log_names=names.tolist(), log_idx=idx, feat=torch.from_numpy(feat16), z6=torch.from_numpy(z), pick_banked=R.picks_np(z),
                          props_table=np.zeros((n_tok, NP_, 8, 3), np.float32), props_dump=np.zeros((n_tok, NP_, 8, 3), np.float32), pick_dump=R.picks_np(z), token_feat=tok, log_feat=log,
                          token_dump=tok, log_dump=log, provenance={})
        out = D.DevOutcomes(pdms=pdms, sub=sub, safe=np.ones((n_tok, NP_), bool), formula={})
        return inp, out, W6, b6, a, b, x, y

    def test_new_pick_and_p2_pick_are_the_hand_picks(self):
        inp, out, W6, b6, a, b, x, y = self.build()
        np.testing.assert_array_equal(R.picks_np(inp.z6.numpy()), a)
        row = np.zeros(D.FEAT_DIM, np.float32)
        row[6] = 1.0
        z = D.apply_rows(inp.z6, inp.feat, {3: (row, np.zeros(1, np.float32))}, W6, b6)
        np.testing.assert_array_equal(D.pick_logits(z), b)
        np.testing.assert_allclose(z[..., 3], inp.feat.float().numpy()[..., 6], atol=1e-5)

    def test_gain_is_the_hand_value_and_the_bar_reads_it(self):
        inp, out, W6, b6, a, b, x, y = self.build()
        row = np.zeros(D.FEAT_DIM, np.float32)
        row[6] = 1.0
        z = D.apply_rows(inp.z6, inp.feat, {3: (row, np.zeros(1, np.float32))}, W6, b6)
        r = D.score_arm(inp, out, D.pick_logits(z), D.pick_logits(inp.z6.numpy()), R.boot_draws(inp.n_logs))
        self.assertAlmostEqual(r["gain_x100"]["point"], 100.0 * float((y - x).mean()), places=9)
        self.assertEqual(r["changed_picks"], 6)
        self.assertAlmostEqual(r["new_pdms_x100"], 100.0 * float(y.mean()), places=9)
        self.assertAlmostEqual(r["base_pdms_x100"], 100.0 * float(x.mean()), places=9)
        nz = x > 0
        self.assertAlmostEqual(r["pool_gain_x100"]["point"], 100.0 * float(((y - x) * nz).sum() / len(x)), places=9)
        self.assertAlmostEqual(r["components"]["ttc"]["delta_x100"]["point"], 100.0 * float((y - x).mean()), places=9)
        self.assertEqual(r["components"]["nc"]["delta_x100"]["point"], 0.0)
        self.assertTrue(D.bar_check({"0": r["gain_x100"]["point"], "1": r["gain_x100"]["point"]})["both_seeds"])
        # swapping new and base flips the sign (paired difference, new minus base)
        r2 = D.score_arm(inp, out, D.pick_logits(inp.z6.numpy()), D.pick_logits(z), R.boot_draws(inp.n_logs))
        self.assertAlmostEqual(r2["gain_x100"]["point"], -r["gain_x100"]["point"], places=9)

    def test_gain_sign_mutation(self):
        inp, out, W6, b6, a, b, x, y = self.build()
        row = np.zeros(D.FEAT_DIM, np.float32)
        row[6] = 1.0
        z = D.apply_rows(inp.z6, inp.feat, {3: (row, np.zeros(1, np.float32))}, W6, b6)

        def scen(mod):
            r = mod.score_arm(inp, out, mod.pick_logits(z), mod.pick_logits(inp.z6.numpy()), R.boot_draws(inp.n_logs))
            return r["gain_x100"]["point"] > 0                  # a flipped sign reads a negative gain on a world that gains: the check 'fires' while the sign is right
        self.mut("GAIN-SIGN", "per_log(inp, pn), per_log(inp, pb)", "per_log(inp, pb), per_log(inp, pn)", scen)

    def test_planner_rule_matches_an_independent_oracle_on_random_logits(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(40, NP_, 6)).astype(np.float32) * 2.0
        z[7] = z[7, 5]                                         # a token whose 64 proposals are exactly tied: lowest index must win
        got = D.pick_logits(z)
        self.assertEqual(int(got[7]), 0)
        for i in range(40):
            self.assertEqual(int(got[i]), ref_pick(z[i]), i)

    def test_only_column_3_changes(self):
        w = self.world()
        inp = D.load_dev_inputs(w.opts().guard(), w.opts())
        P = w.w
        z = D.apply_rows(inp.z6, inp.feat, {3: w.row_for(3.0, "0")}, P["W6"], P["b6"])
        st = inp.z6.numpy()
        self.assertTrue(np.array_equal(z[..., [0, 1, 2, 4, 5]], st[..., [0, 1, 2, 4, 5]]))
        self.assertFalse(np.array_equal(z[..., 3], st[..., 3]))

    def test_bootstrap_equals_zero_tail_boot(self):
        w = self.world()
        inp = D.load_dev_inputs(w.opts().guard(), w.opts())
        out = D.read_outcomes(w.opts().guard(), os.path.join(w.dev, "table_p2_heldout.npz"))
        P = w.w
        z = D.apply_rows(inp.z6, inp.feat, {3: w.row_for(3.0, "0")}, P["W6"], P["b6"])
        newp, basep = D.pick_logits(z), D.pick_logits(inp.z6.numpy())
        r = D.score_arm(inp, out, newp, basep, R.boot_draws(inp.n_logs))
        ar = np.arange(inp.n)
        boot = ZT.Boot(inp.log_idx, inp.n_logs)
        ref = boot.ratio(out.pdms[ar, newp] - out.pdms[ar, basep], np.ones(inp.n), scale=100.0)
        for k in ("point", "lo", "hi"):
            self.assertAlmostEqual(r["gain_x100"][k], ref[k], places=9, msg=k)
        self.assertEqual(r["gain_x100"]["B"], D.BOOT_B)
        self.assertEqual((D.BOOT_B, D.BOOT_SEED), (ZT.B_BOOT, ZT.SEED))

    def test_registered_lc_call_equals_srttc_r1s_and_the_both_rule_is_flat(self):
        grid = [(-0.004, 0.004), (0.02, 0.05), (-0.01, 0.03), (0.002, 0.008), (-0.02, -0.001), (0.0, 0.0099), (0.0, 0.01), (0.0001, 0.0101)]
        for l0, h0 in grid:
            for l1, h1 in grid:
                d = {"0": {"lo": l0, "hi": h0}, "1": {"lo": l1, "hi": h1}}
                mine, theirs = D.lc_call_registered(d), R.lc_call(d, "flat")
                self.assertEqual(mine["call"], theirs["call"], (l0, h0, l1, h1))
                self.assertEqual((mine["rising"], mine["flat"]), (theirs["rising"], theirs["flat"]))
        self.assertEqual(D.lc_call_registered({"0": {"lo": 0.002, "hi": 0.008}, "1": {"lo": 0.002, "hi": 0.008}})["call"], "FLAT", "wholly inside (0, 0.010) satisfies both: FLAT")
        self.assertEqual(D.lc_call_registered({"0": {"lo": 0.02, "hi": 0.05}, "1": {"lo": -0.01, "hi": 0.05}})["call"], "INCONCLUSIVE", "seeds disagreeing is INCONCLUSIVE")

    def test_cstar_recomputation_matches_r1_and_breaks_ties_in_grid_order(self):
        r1 = base()["r1"]
        self.assertEqual(D.recompute_cstar(r1["grid"]), r1["c_star"]["id"])
        tied = [{"lr": lr, "anchor": a, "per_seed": {"0": {"pooled": 0.7}, "1": {"pooled": 0.7}}} for lr in D.REG_GRID_LR for a in D.REG_GRID_ANCHOR]
        self.assertEqual(D.recompute_cstar(tied), "lr0.0001_anchor0.03")
        tied[4]["per_seed"] = {"0": {"pooled": 0.9}, "1": {"pooled": 0.5}}                 # mean 0.7 == the others: still the first
        self.assertEqual(D.recompute_cstar(tied), "lr0.0001_anchor0.03")
        tied[4]["per_seed"] = {"0": {"pooled": 0.9}, "1": {"pooled": 0.6}}                 # mean 0.75: wins
        self.assertEqual(D.recompute_cstar(tied), "lr0.0003_anchor0.1")

    def test_hmac_equals_the_launch_gates_and_a_tampered_token_fails(self):
        key = hashlib.sha256(b"k").digest()
        body = {"schema": G.GATE_SCHEMA, "kind": G.TOKEN_KIND, "verdict": "PASS", "binding": {"a": 1, "b": [1, 2]}, "key_id": "x"}
        tok = dict(body)
        tok["hmac_sha256"] = G.sign(body, key)
        self.assertTrue(G.signature_ok(tok, key))
        self.assertTrue(D._hmac_ok(tok, key))
        bad = copy.deepcopy(tok)
        bad["verdict"] = "FAIL"
        self.assertFalse(G.signature_ok(bad, key))
        self.assertFalse(D._hmac_ok(bad, key))
        self.assertFalse(D._hmac_ok(tok, hashlib.sha256(b"other").digest()))
        self.assertEqual(D._canonical_json(body), G.canonical_json(body))

    def test_bar_boundary_and_both_seeds(self):
        self.assertTrue(D.bar_check({"0": 0.30, "1": 0.30})["both_seeds"], "the bar is >= +0.30: exactly 0.30 clears")
        self.assertFalse(D.bar_check({"0": 0.2999999999, "1": 5.0})["both_seeds"])
        self.assertFalse(D.bar_check({"0": 5.0, "1": 0.2999999999})["both_seeds"])
        self.assertEqual(D.verdict_of(True, {"both_seeds": True}), "PASS")
        self.assertEqual(D.verdict_of(True, {"both_seeds": False}), "FAIL")
        self.assertEqual(D.verdict_of(False, {"both_seeds": True}), "VOID")
        self.assertEqual(D.DEV_BAR_X100, 0.30)

    def test_bar_mutations(self):
        MUTATED.add(("SRDEV", "BAR"))
        MUTATED.add(("SRDEV", "BOTH-SEEDS"))
        MUTATED.add(("SRDEV", "VERDICT"))
        m = load_mutant(("BAR", ">= DEV_BAR_X100", "> DEV_BAR_X100"))
        self.assertTrue(D.bar_check({"0": 0.30, "1": 0.30})["both_seeds"])
        self.assertFalse(m.bar_check({"0": 0.30, "1": 0.30})["both_seeds"], "the boundary test must catch >= -> >")
        m = load_mutant(("BOTH-SEEDS", "all(per[sd] for sd in SEED_KEYS)", "any(per[sd] for sd in SEED_KEYS)"))
        self.assertFalse(D.bar_check({"0": 1.0, "1": 0.0})["both_seeds"])
        self.assertTrue(m.bar_check({"0": 1.0, "1": 0.0})["both_seeds"], "any-seed would pass a one-seed result")
        m = load_mutant(("VERDICT", "if not controls_pass:", "if False:"))
        self.assertEqual(D.verdict_of(False, {"both_seeds": True}), "VOID")
        self.assertEqual(m.verdict_of(False, {"both_seeds": True}), "PASS", "a RED control must void the verdict")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (c)+(d) refusals fire BEFORE any outcome is opened; each has a one-line regression arm
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Refusals(Base):
    def w0(self, **kw):
        return self.world(outcomes=False, **kw)

    def test_00_the_unmutated_no_outcome_world_gets_as_far_as_opening_the_read(self):
        """Sanity: with a table that has NO outcome members the run passes EVERY refusal and every no-outcome control, writes the marker, and only then fails to find `sub_names`."""
        w = self.w0()
        o = w.opts()
        with self.assertRaises(KeyError):
            D.run_dev(o)
        self.assertTrue(os.path.exists(os.path.join(o.resolved_out_dir(), D.READ_STARTED)), "every refusal and no-outcome control passed, the marker was written, and only then the outcomes were needed")

    def test_registered_sha(self):
        w = self.w0()
        sc = lambda m: refused(m, w, "REGISTERED", registered_result_sha256="0" * 64)
        self.mut("REGSHA", "sha != opts.registered_result_sha256", "False", sc)

    def test_seeds_must_be_0_1(self):
        w = self.w0()
        d = w.r1doc()
        d["params"]["seeds"] = [0, 2]
        w.save_r1(d)
        self.mut("SEEDS", "!= REG_SEEDS", "== REG_SEEDS", lambda m: refused(m, w, "fit seeds"))

    def test_pins_overridden_refused_unless_testing(self):
        w = self.w0()
        self.assertTrue(w.r1doc()["inputs"]["pins_overridden"], "the synthetic R1 overrode pins (test rig): the registered object has none")
        self.mut("PINS", "if not opts.allow_r1_pin_overrides and", "if False and", lambda m: refused(m, w, "overrode registered pins", allow_r1_pin_overrides=False))

    def test_code_sha(self):
        w = self.w0()
        d = w.r1doc()
        d["code_sha256"] = "0" * 64
        w.save_r1(d)
        self.mut("CODESHA", "!= opts.srttc_r1_sha256", "== opts.srttc_r1_sha256 and False", lambda m: refused(m, w, "landed srttc_r1.py"))

    def test_params_must_be_the_registered_ones(self):
        w = self.w0()
        sc = lambda m: refused(m, w, "params differ", reg_epochs=10)          # the rig used 2 epochs; the registered value is 10
        self.mut("PARAMS", "if bad:", "if False:", sc)
        d = w.r1doc()
        d["params"]["grid_lr"] = [1e-4, 3e-4, 2e-3]
        w.save_r1(d)
        self.assertTrue(refused(D, w, "params differ"))

    def test_cstar_config_sha(self):
        w = self.w0()
        d = w.r1doc()
        d["c_star"]["config_sha256"] = "0" * 64
        w.save_r1(d)
        self.mut("CFGSHA", "if star.sha256() != cs[\"config_sha256\"]:", "if False:", lambda m: refused(m, w, "config sha256"))

    def test_cstar_must_be_on_the_registered_grid(self):
        w = self.w0()
        d = w.r1doc()
        cs = d["c_star"]
        star = R.Cfg(lr=2e-3, anchor=cs["anchor"], epochs=d["params"]["epochs"], batch_sets=d["params"]["batch_sets"])
        cs.update({"lr": 2e-3, "id": star.id(), "config_sha256": star.sha256()})
        w.save_r1(d)
        self.mut("GRID", "not in {(lr, a) for lr in REG_GRID_LR for a in REG_GRID_ANCHOR}", "in {(lr, a) for lr in REG_GRID_LR for a in REG_GRID_ANCHOR} and False", lambda m: refused(m, w, "registered 9"))

    def test_cstar_must_be_the_argmax_of_its_own_grid(self):
        w = self.w0()
        d = w.r1doc()
        other = [g for g in d["grid"] if g["id"] != d["c_star"]["id"]][0]
        star = R.Cfg(lr=other["lr"], anchor=other["anchor"], epochs=d["params"]["epochs"], batch_sets=d["params"]["batch_sets"])
        d["c_star"].update({"lr": star.lr, "anchor": star.anchor, "id": star.id(), "config_sha256": star.sha256()})
        w.save_r1(d)
        self.mut("CSTAR", "if recompute_cstar(grid) != cs[\"id\"]:", "if False:", lambda m: refused(m, w, "argmax of the recorded grid"))

    def test_fit_set_size_is_the_registered_one(self):
        w = self.w0()
        sc = lambda m: refused(m, w, "registered S_10", expect_fit_logs=117 if True else None)
        self.mut("NLOGS", "!= want_logs:", "== want_logs and False:", sc)
        self.assertEqual(D.LOGS_BY_FRACTION, {10: 117, 25: 292, 50: 584, 100: 1167})

    def test_rows_must_be_fitted_from_this_p2(self):
        w = self.w0()
        d = w.r1doc()
        d["inputs"]["p2_row_sha256"] = "0" * 64
        w.save_r1(d)
        self.mut("P2ROW", "!= p2_row_sha:", "== p2_row_sha and False:", lambda m: refused(m, w, "different P2"))

    def test_lc_result_is_required(self):
        w = self.w0()
        os.remove(os.path.join(w.run, "lc_result.json"))
        self.mut("LC-REQUIRED", "if not _io_exists(guard, path):", "if False:", lambda m: refused(m, w, "LC call at this size is banked"))

    def test_lc_recorded_call_must_equal_its_intervals(self):
        w = self.w0()
        doc = rj(os.path.join(w.run, "lc_result.json"))
        doc["call"]["call"] = "RISING"                       # intervals say FLAT
        wj(os.path.join(w.run, "lc_result.json"), doc)
        self.mut("LC-CALL", "if rec != mine[\"call\"]:", "if False:", lambda m: refused(m, w, "inconsistent"))

    def test_non_flat_call_below_f_max_is_not_read_and_flat_or_f_max_is(self):
        for kind in ("RISING", "INCONCLUSIVE"):
            w = self.w0(lc=kind)
            self.assertTrue(refused(D, w, "scale to the next nested size"), kind)
        w = self.w0(lc="RISING")
        self.mut("LC-SCALEUP", "or frac >= F_MAX)", "or frac >= 0)", lambda m: refused(m, w, "scale to the next nested size"))
        # the same files are READ when the call is FLAT, and when f = f_max whatever the call (the call is recorded)
        w = self.w0(lc="FLAT")
        self.assertIsNone(self._until_outcomes(w))
        w = self.w0(lc="RISING", fraction=100)
        self.assertIsNone(self._until_outcomes(w, expect_fit_logs=w.r1doc()["inputs"]["n_logs"]))

    def _until_outcomes(self, w, **kw):
        """Run; the table has no outcome members, so a run that passed every refusal ends in the KeyError of the read.  -> None iff that is what happened."""
        try:
            w.run_dev(**kw)
        except KeyError:
            return None
        except D.DevRefusal as ex:
            return str(ex)
        return "completed?"

    def test_c2_files_must_exist_and_belong_to_this_c_star(self):
        w = self.w0()
        os.remove(os.path.join(w.run, "r1_c2_result.json"))
        self.assertTrue(refused(D, w, "C2 (the shuffled-label negative control) is a dev-gate condition"))
        w = self.w0()
        c2 = rj(os.path.join(w.run, "r1_c2_result.json"))
        c2["c_star"]["config_sha256"] = "0" * 64
        wj(os.path.join(w.run, "r1_c2_result.json"), c2)
        self.mut("C2BIND", "or _need(doc, \"c_star\", \"config_sha256\") != r1[\"doc\"][\"c_star\"][\"config_sha256\"]:", ":", lambda m: refused(m, w, "not the C2 stage"))

    def test_c2_from_another_bank_or_fraction_is_refused(self):
        w = self.w0()
        c2 = rj(os.path.join(w.run, "r1_c2_result.json"))
        c2["bank_manifest_sha256"] = "0" * 64
        wj(os.path.join(w.run, "r1_c2_result.json"), c2)
        self.assertTrue(refused(D, w, "different bank"))
        w = self.w0()
        c2 = rj(os.path.join(w.run, "r1_c2_result.json"))
        c2["fraction_pct"] = 25
        wj(os.path.join(w.run, "r1_c2_result.json"), c2)
        self.assertTrue(refused(D, w, "different fraction"))

    # ---- the launch-gate token (C1)
    def test_token_missing_hmac_verdict_profile_code_and_manifest(self):
        w = self.w0()
        os.remove(w.token_path)
        self.assertTrue(refused(D, w, "C1_reach"))
        w = self.w0()
        tok = rj(w.token_path)
        tok["created_utc"] = "2026-10-20T00:00:00Z"                                            # a field no binding check reads, edited after signing: ONLY the HMAC can catch it
        wj(w.token_path, tok)
        self.mut("TOKEN-HMAC", "elif not _hmac_ok(tok, key):", "elif False:", lambda m: refused(m, w, "C1_reach"))
        w = self.w0()
        w.write_token(verdict="FAIL")                                                          # correctly SIGNED, but not a PASS
        self.mut("TOKEN-VERDICT", "if tok.get(\"verdict\") != \"PASS\":", "if False:", lambda m: refused(m, w, "C1_reach"))
        w = self.w0()
        w.write_token(binding={"profile": "srttc-r1-synthetic"})
        self.mut("TOKEN-PROFILE", "if b.get(\"profile\") != opts.gate_profile:", "if False:", lambda m: refused(m, w, "C1_reach"))
        w = self.w0()
        w.write_token(binding={"srttc_r1_sha256": "1" * 64})
        self.mut("TOKEN-CODE", "if b.get(\"srttc_r1_sha256\") != doc.get(\"code_sha256\"):", "if False:", lambda m: refused(m, w, "C1_reach"))
        w = self.w0()
        w.write_token(binding={"bank_manifest_sha256": "2" * 64})
        self.mut("TOKEN-MANIFEST", "if b.get(\"bank_manifest_sha256\") != doc[\"inputs\"].get(\"bank_manifest_sha256\"):", "if False:", lambda m: refused(m, w, "C1_reach"))
        w = self.w0()
        w.write_token(binding={"stage": "lc"})
        self.assertTrue(refused(D, w, "C1_reach"))
        w = self.w0()
        w.write_token(binding={"fraction": 25})
        self.assertTrue(refused(D, w, "C1_reach"))

    def test_token_signed_with_another_key_is_refused_and_a_missing_key_too(self):
        w = self.w0()
        with open(w.key, "w", encoding="ascii") as f:
            f.write(hashlib.sha256(b"another").hexdigest() + "\n")
        self.assertTrue(refused(D, w, "C1_reach"))
        w = self.w0()
        os.remove(w.key)
        self.assertTrue(refused(D, w, "C1_reach"))

    def test_recorded_c1_gradients_are_reevaluated_not_trusted(self):
        r1 = copy.deepcopy(base()["r1"])
        self.assertTrue(D.c1_recorded(r1)["pass"], "the real R1 record is GREEN")
        w = self.w0()
        d = w.r1doc()
        d["c1_reachability"]["full"]["ttc_row_w"] = 0.0                                       # the pairwise/anchor never reached the row
        w.save_r1(d)
        self.mut("C1-GRAD", "v > 0.0):", "v >= 0.0):", lambda m: refused(m, w, "C1_reach"))
        w = self.w0()
        d = w.r1doc()
        d["c1_reachability"]["pair_only"]["frozen_rows_w"] = 1e-9                              # a frozen row leaked
        w.save_r1(d)
        self.mut("C1-FROZEN", "if r.get(k) != 0.0:", "if False:", lambda m: refused(m, w, "C1_reach"))
        w = self.w0()
        d = w.r1doc()
        d["c1_reachability"]["zero_control"]["ttc_row_w"] = 1e-3                                # the zero-weight control read non-zero: a broken probe
        w.save_r1(d)
        self.mut("C1-ZERO", "if z.get(\"ttc_row_w\") != 0.0 or z.get(\"ttc_row_b\") != 0.0:", "if False:", lambda m: refused(m, w, "C1_reach"))
        d = w.r1doc()
        del d["c1_reachability"]
        self.assertFalse(D.c1_recorded(d)["pass"])

    # ---- A2.10
    def test_a210_four_conditions(self):
        w = self.w0()
        done = rj(os.path.join(w.p2ho, "P2HO_DONE"))
        done["PARTIAL"] = True
        wj(os.path.join(w.p2ho, "P2HO_DONE"), done)
        self.mut("A210-PARTIAL", "if done.get(\"PARTIAL\") is not False:", "if False:", lambda m: refused(m, w, "A2.10"))
        w = self.w0()
        done = rj(os.path.join(w.p2ho, "P2HO_DONE"))
        done["proofs"]["G3"] = False
        wj(os.path.join(w.p2ho, "P2HO_DONE"), done)
        self.mut("A210-PROOFS", "if not proofs_ok:", "if False:", lambda m: refused(m, w, "(2)"))
        w = self.w0()
        rc = rj(os.path.join(w.p2ho, "recompute_check_333.json"))
        rc["logits_max_abs"] = 1e-7
        wj(os.path.join(w.p2ho, "recompute_check_333.json"), rc)
        self.mut("A210-333", "and zeros):", "and True):", lambda m: refused(m, w, "(3)"))
        w = self.w0()
        man = rj(os.path.join(w.p2ho, "MANIFEST.sha256.json"))
        man["files"]["proposals.npz"]["sha256"] = "0" * 64
        wj(os.path.join(w.p2ho, "MANIFEST.sha256.json"), man)
        self.mut("A210-MANIFEST", "if landed != opts.dev_pins.get(n) or landed != dev_sha.get(n):", "if False:", lambda m: refused(m, w, "(4)"))
        w = self.w0()
        done = rj(os.path.join(w.p2ho, "P2HO_DONE"))
        done["n_tokens"] = w.N - 1
        wj(os.path.join(w.p2ho, "P2HO_DONE"), done)
        self.assertTrue(refused(D, w, "(1) tokens"))
        w = self.w0()
        done = rj(os.path.join(w.p2ho, "P2HO_DONE"))
        done["sets_index"]["n_sets"] = 603
        wj(os.path.join(w.p2ho, "P2HO_DONE"), done)
        self.assertTrue(refused(D, w, "(1) sets"))

    def test_a210_missing_package_is_a_refusal_not_a_skip(self):
        w = self.w0(p2ho=False)
        with self.assertRaises((D.DevRefusal, D.DevGuardError, FileNotFoundError)):
            w.run_dev()

    # ---- dev inputs and the no-outcome controls
    def test_dev_input_sha_pins(self):
        w = self.w0()
        pins = dict(w.pins)
        pins["proposals.npz"] = "0" * 64
        self.mut("DEVSHA", "if pin is not None and sha[n] != pin:", "if False:", lambda m: refused(m, w, "dev input", dev_pins=pins))

    def test_dev_table_shape_and_count_pins(self):
        w = self.w0()
        self.assertTrue(refused(D, w, "tokens, expected", expect_tokens=w.N + 1))
        self.assertTrue(refused(D, w, "logs, expected", expect_logs=w.n_logs + 1))

    def test_featcheck_red_when_the_features_are_not_the_input_of_this_head(self):
        w = self.w0()
        rng = np.random.default_rng(1)
        w.write_tables(feat=rng.normal(size=w.feat.shape).astype(np.float16))
        w.write_p2ho()
        self.mut("FEATCHECK", "\"pass\": bool(dev <= tol and neg > 10 * tol)}", "\"pass\": True}", lambda m: refused(m, w, "integrity_featcheck"))

    def test_c0_red_when_the_banked_pick_is_not_the_rules_pick(self):
        w = self.w0()
        bad = w.pick.copy()
        bad[3] = (bad[3] + 1) % NP_
        w.write_tables(pick=bad)
        w.write_p2ho()
        self.mut("C0", "bool(rec[\"logits_bit_identical_ttc_row_is_p2\"]", "bool(True or rec[\"logits_bit_identical_ttc_row_is_p2\"]", lambda m: refused(m, w, "C0_pick_identity"))

    def test_c4_red_when_the_dump_proposals_differ_by_one_ulp(self):
        w = self.w0()
        props = w.props.copy()
        props[5, 7, 0, 0] = np.nextafter(props[5, 7, 0, 0], np.float32(np.inf))
        w.write_tables(props_dump=props)
        w.write_p2ho()
        self.mut("C4-EXACT", "mx == 0.0 and", "mx >= 0.0 and", lambda m: refused(m, w, "C4_fan_frozen"))

    def test_c1_applier_reach_and_leak_arms(self):
        w = self.w0(scales=(0.0, 0.0))
        self.mut("C1-REACH", "ok = ok and changed > 0 and others_equal", "ok = ok", lambda m: refused(m, w, "C1_reach"))
        w = self.w0()
        sc = lambda m: refused(m, w, "C1_reach")
        MUTATED.add(("SRDEV", "APPLYCOL"))
        leak = load_mutant(("APPLYCOL", "out[..., col] = out[..., col] + (feat.float() @ dw + db)", "out[..., col] = out[..., col] + (feat.float() @ dw + db); out[..., 2] = out[..., 2] + (feat.float() @ dw + db)"))
        self.assertFalse(sc(D))
        self.assertTrue(sc(leak), "a delta leaking into the EP column must read RED (the other five columns must be bit-identical)")

    def test_c1_positive_control_is_what_proves_the_probe_can_read_a_change(self):
        """If logits -> pick is not wired (a pick function blind to the TTC column), the NEGATED P2 row changes no pick and C1's applier-side reach must read RED."""
        w = self.w0()
        inp = D.load_dev_inputs(w.opts().guard(), w.opts())
        P = w.w
        rows = {sd: w.row_for(3.0, sd) for sd in SK}
        self.assertTrue(D.c1_applier_reach(inp, rows, P["W6"], P["b6"])["pass"], "wired: GREEN, and the positive control read a change")
        self.assertGreater(D.c1_applier_reach(inp, rows, P["W6"], P["b6"])["positive_control_negated_p2_row_changed_picks"], 0)

        def blind(m):
            old = m.pick_logits
            m.pick_logits = lambda z: np.zeros(np.asarray(z).shape[0], np.int64)
            try:
                return not m.c1_applier_reach(inp, rows, P["W6"], P["b6"])["pass"]
            finally:
                m.pick_logits = old
        self.mut("C1-POS", "pos = int((pick_logits(z_neg) != pick_logits(stored)).sum())", "pos = 1", blind)

    def test_red_control_stops_before_the_marker_is_written(self):
        w = self.w0()
        props = w.props.copy()
        props[0, 0, 0, 0] += 1.0
        w.write_tables(props_dump=props)
        w.write_p2ho()
        o = w.opts()
        with self.assertRaises(D.DevRefusal):
            D.run_dev(o)
        self.assertFalse(os.path.exists(os.path.join(o.resolved_out_dir(), D.READ_STARTED)), "the read was NOT spent")
        MUTATED.add(("SRDEV", "REDGATE"))
        m = load_mutant(("REDGATE", "if red:", "if False:"))
        o2 = w.opts()
        try:
            m.run_dev(mopts(m, o2))
        except KeyError:
            pass                                                                                 # the mutant sailed through the RED control and reached the (absent) outcomes
        self.assertTrue(os.path.exists(os.path.join(o2.resolved_out_dir(), D.READ_STARTED)), "the mutant spends the read on a broken instrument: the gate is what stops it")

    # ---- once
    def test_the_read_is_once_and_a_crashed_first_attempt_still_consumed_it(self):
        w = self.w0()
        o = w.opts()
        with self.assertRaises(KeyError):                                                       # passes every refusal, writes the marker, then the outcomes are absent
            D.run_dev(o)
        self.assertTrue(os.path.exists(os.path.join(o.resolved_out_dir(), D.READ_STARTED)))
        self.assertFalse(os.path.exists(os.path.join(o.resolved_out_dir(), D.RESULT_NAME)))
        with self.assertRaises(D.DevRefusal) as cm:
            D.run_dev(o)
        self.assertIn("ONCE", str(cm.exception))
        MUTATED.add(("SRDEV", "ONCE"))
        m = load_mutant(("ONCE", "if _io_exists(guard, p):", "if False:"))
        with self.assertRaises(Exception) as cm2:
            m.run_dev(mopts(m, o))
        self.assertNotIsInstance(cm2.exception, m.DevRefusal, "the once-guard mutant does not refuse the second read")

    def test_dev_guard_refuses_navtest_and_the_fit_bank(self):
        g = D.DevGuard(read_dirs=[self.tmp], write_dirs=[self.tmp])
        for bad in ("table_p2_full.npz", "navtest_x.npz", "srttc_s0/x.npz", "bank/U.sets.npz", "s1_read_cache/x.json", "ranked_fit_logs.tsv"):
            with self.assertRaises(D.DevGuardError, msg=bad):
                g.check_read(os.path.join(self.tmp, bad))
            with self.assertRaises(D.DevGuardError, msg=bad):
                g.check_write(os.path.join(self.tmp, bad))
        g.check_read(os.path.join(self.tmp, "p2_heldout", "table_p2_heldout.npz"))
        with self.assertRaises(D.DevGuardError):
            g.check_read(os.path.join(os.path.dirname(self.tmp), "elsewhere.json"))
        # the REAL names the gate reads are allowed
        g2 = D.DevGuard(read_dirs=[D.DEV_DIR, D.P2HO_DIR], read_files=[D.DEFAULT_P2_CKPT, D.DEFAULT_GATE_ROOT + "/f10_run/r1_gate_token.json"], write_dirs=["C:/Users/Admin/srttc_r1/run/f10"])
        for p in (D.DEV_DIR + "/table_p2_heldout.npz", D.P2HO_DIR + "/P2HO_DONE", D.DEFAULT_P2_CKPT, D.DEFAULT_GATE_ROOT + "/f10_run/r1_gate_token.json"):
            g2.check_read(p)
        g2.check_write("C:/Users/Admin/srttc_r1/run/f10/srttc_dev_result.json")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class EndToEnd(Base):
    def test_pass_world(self):
        w = self.world(scales=(3.0, 3.0), c2_scales=(-3.0, -3.0))
        res = w.run_dev()
        self.assertEqual(res["dev_gate"]["verdict"], "PASS")
        g = {sd: res["primary"]["per_seed"][sd]["gain_x100"]["point"] for sd in SK}
        self.assertTrue(all(v >= 0.30 for v in g.values()), g)
        self.assertTrue(res["dev_gate"]["controls_all_pass"])
        # independent numpy recomputation of the gain from the raw arrays and the raw row files
        P = w.w
        for sd in SK:
            row = np.load(os.path.join(w.run, f"ttc_row_seed{sd}.npy"))
            f32 = w.feat.astype(np.float32)
            z = w.logits.astype(np.float64).copy()
            z[..., 3] = z[..., 3] + f32.astype(np.float64) @ (row[:256].astype(np.float64) - P["W6"][3].astype(np.float64)) + (float(row[256]) - float(P["b6"][3]))
            newp = np.array([ref_pick(z[i]) for i in range(w.N)])
            basep = np.array([ref_pick(w.logits[i].astype(np.float64)) for i in range(w.N)])
            ar = np.arange(w.N)
            self.assertAlmostEqual(g[sd], 100.0 * float((w.pdms[ar, newp] - w.pdms[ar, basep]).mean()), places=6, msg=sd)
        # the shape of the record
        self.assertEqual(res["schema"], D.SCHEMA)
        self.assertEqual(res["estimator"]["B"], 10000)
        self.assertEqual(res["estimator"]["seed"], 20260927)
        self.assertIn("DEV-PASS", res["outcome_rows"]["fired"])
        self.assertEqual(res["dev_gate"]["strategic_family"], "PAUSED (PI 2026-09-19, item 25): never a pass")
        for k in ("C0", "C1", "C2", "C3", "C4", "C5", "bar_gain_both_seeds", "gate_set", "lc_before_gate"):
            self.assertIn(k, res["registration_trace"])
        self.assertTrue(all(":" in v for v in res["registration_trace"].values()), "every bar term carries a file:line")
        self.assertEqual(set(res["registration_gaps"]), {"G1", "G2", "G3", "G4", "G5", "G6"})
        self.assertEqual(res["inputs"]["rows_sha256"], {sd: sha_f(os.path.join(w.run, f"ttc_row_seed{sd}.npy")) for sd in SK})
        self.assertEqual(res["inputs"]["r1_result_sha256"], w.reg_sha())
        self.assertIn("seed_gap_abs", res["primary"]["bar"])
        self.assertEqual(res["replicate_c3"]["rows_distinct"], True)
        # the result file and its sidecar
        # the same out dir cannot be read again (the one dev read is spent)
        with self.assertRaises(D.DevRefusal):
            w.run_dev(out_dir=os.path.join(w.root, "out1"))

    def test_fail_world_and_next_arm(self):
        w = self.world(scales=(-3.0, -3.0), c2_scales=(-3.0, -3.0), lc="FLAT")
        res = w.run_dev()
        self.assertEqual(res["dev_gate"]["verdict"], "FAIL")
        self.assertEqual(res["outcome_rows"]["fired"], ["DEV-FAIL", "HEAD/FEATURE-LIMITED"])
        self.assertIn("feature-vs-capacity probe", res["outcome_rows"]["next_arm"]["DEV-FAIL"])
        self.assertIn("0.72", res["outcome_rows"]["next_arm"]["DEV-FAIL"])

    def test_one_seed_clearing_is_a_fail_not_a_pass(self):
        w = self.world(scales=(3.0, -3.0))
        res = w.run_dev()
        self.assertEqual(res["dev_gate"]["verdict"], "FAIL")
        self.assertEqual(res["primary"]["bar"]["per_seed_meets"], {"0": True, "1": False})
        MUTATED.add(("SRDEV", "BOTH-SEEDS-E2E"))

    def test_c2_red_voids_and_the_read_is_consumed(self):
        w = self.world(scales=(3.0, 3.0), c2_scales=(3.0, 3.0))                          # the "shuffled" rows gain: the instrument reads a positive on a null
        o = w.opts()
        res = D.run_dev(o)
        self.assertEqual(res["dev_gate"]["verdict"], "VOID")
        self.assertFalse(res["controls"]["C2_negative"]["pass"])
        self.assertEqual(res["outcome_rows"]["fired"], ["VOID"])
        self.assertTrue(os.path.exists(os.path.join(o.resolved_out_dir(), D.READ_STARTED)))
        self.assertTrue(res["dev_gate"]["both_seeds_meet_bar"], "the gains cleared; the control voided them")
        self.assertTrue(res["registration_gaps"]["G3"])
        sc = lambda m: (completes(m, w) or {"dev_gate": {"verdict": "?"}})["dev_gate"]["verdict"] == "VOID"
        self.mut("C2", "<= 0.0 for sd in SEED_KEYS", "<= 1e9 for sd in SEED_KEYS", sc)

    def test_one_seed_c2_red_is_enough_to_void(self):
        w = self.world(scales=(3.0, 3.0), c2_scales=(-3.0, 3.0))
        self.assertEqual(w.run_dev()["dev_gate"]["verdict"], "VOID")

    def test_pick_new_mutation_the_pick_is_what_the_new_logits_say(self):
        w = self.world(scales=(3.0, 3.0))
        sc = lambda m: ((completes(m, w) or {"dev_gate": {"verdict": "?"}})["dev_gate"]["verdict"] == "PASS")      # 'fires' = the gate reads the new picks and reaches the PASS this world earns
        self.mut("PICK-NEW", "new_pick = pick_logits(z_new)", "new_pick = base_pick", sc)

    def test_apply_mutation_the_row_must_reach_the_logits(self):
        w = self.world(scales=(3.0, 3.0))
        sc = lambda m: ((completes(m, w) or {"dev_gate": {"verdict": "?"}})["dev_gate"]["verdict"] == "PASS")
        self.mut("APPLY", "z = apply_row(z, f, w, b, W6[col], b6[col:col + 1], col=col)", "z = z", sc)

    def test_c0_gain_zero_mutation(self):
        """C0's outcome half: identical picks score a gain of EXACTLY 0.0.  The instrument defect it exists to catch is a scoring path that reads the NEW pick from the wrong place (SCORE-LOOKUP);
        the C0-GAIN condition is then what turns that into a RED."""
        MUTATED.add(("SRDEV", "C0-GAIN"))
        MUTATED.add(("SRDEV", "SCORE-LOOKUP"))
        w = self.world()
        inp = D.load_dev_inputs(w.opts().guard(), w.opts())
        out = D.read_outcomes(w.opts().guard(), os.path.join(w.dev, "table_p2_heldout.npz"))
        base_pick = D.pick_logits(inp.z6.numpy())
        draws = R.boot_draws(inp.n_logs)
        self.assertTrue(D.c0_gain_zero(inp, out, base_pick, draws)["pass"])
        bad_lookup = ("SCORE-LOOKUP", "out.pdms[ar, new_pick], out.pdms[ar, base_pick]", "out.pdms[ar, (new_pick + 1) % N_PROPOSALS], out.pdms[ar, base_pick]")
        m1 = load_mutant(bad_lookup)
        r1 = m1.c0_gain_zero(inp, out, base_pick, draws)
        self.assertFalse(r1["pass"], "a scoring path that reads the wrong pick must make the identity arm RED")
        self.assertNotEqual(r1["gain_point"], 0.0)
        m2 = load_mutant(bad_lookup, ("C0-GAIN", "\"pass\": bool(g[\"point\"] == 0.0 and", "\"pass\": bool(True or g[\"point\"] == 0.0 and"))
        self.assertTrue(m2.c0_gain_zero(inp, out, base_pick, draws)["pass"], "with the C0-GAIN condition removed the defective scorer reads as the identity: the condition is what catches it")

    def test_determinism_two_runs_same_record(self):
        w = self.world()
        a, b = w.run_dev(), w.run_dev()
        for r in (a, b):
            r.pop("run_meta")
            r["inputs"].pop("run_dir", None)
        self.assertEqual(json.dumps(D._jsonable(a), sort_keys=True), json.dumps(D._jsonable(b), sort_keys=True))

    def test_the_real_r1_lc_c2_chain_runs_through_the_gate(self):
        """The rows and records of the LANDED code, unedited (real R1 rows, real LC, real C2): the dev gate accepts the schema and reports a verdict consistent with its own bar."""
        w = self.world(scales=None, c2_scales=None, lc="FLAT")
        # the real chain's own LC call may not be FLAT; the gate's only dependence on it is the f_max rule, so make the recorded call FLAT and keep everything else as the landed code wrote it
        res = w.run_dev()
        gains = {sd: res["primary"]["per_seed"][sd]["gain_x100"]["point"] for sd in SK}
        self.assertEqual(res["dev_gate"]["verdict"], D.verdict_of(res["dev_gate"]["controls_all_pass"], D.bar_check(gains)))
        self.assertEqual(res["inputs"]["c2_rows_sha256"], {sd: base()["c2"]["final_rows"][sd]["sha256"] for sd in SK})
        self.assertEqual(res["inputs"]["rows_sha256"], {sd: base()["r1"]["final_rows"][sd]["sha256"] for sd in SK})
        self.assertEqual(res["controls"]["C2_negative"]["shuffle_seed"], C2.C2_SHUFFLE_SEED)
        self.assertEqual(res["inputs"]["pins_overridden"], w.opts().pins_overridden())
        self.assertIn("dev_pins", res["inputs"]["pins_overridden"])
        self.assertIn("allow_r1_pin_overrides", res["inputs"]["pins_overridden"])

    def test_lc_scaleup_rule_end_to_end_for_each_size(self):
        for f in (10, 25, 50):
            w = self.world(lc="RISING", fraction=f, outcomes=False)
            with self.assertRaises(D.DevRefusal) as cm:
                w.run_dev(expect_fit_logs=w.r1doc()["inputs"]["n_logs"])
            self.assertIn("scale to the next nested size", str(cm.exception), f)
        w = self.world(lc="INCONCLUSIVE", fraction=100, scales=(-3.0, -3.0))
        res = w.run_dev(expect_fit_logs=w.r1doc()["inputs"]["n_logs"])
        self.assertEqual(res["dev_gate"]["verdict"], "FAIL")
        self.assertIn("a2_6_row3", res["outcome_rows"]["notes"], "GAP G4 is surfaced, not silently resolved")
        self.assertEqual(res["outcome_rows"]["fired"], ["DEV-FAIL"])
        w = self.world(lc="RISING", fraction=100, scales=(3.0, 3.0))
        res = w.run_dev(expect_fit_logs=w.r1doc()["inputs"]["n_logs"])
        self.assertEqual(res["dev_gate"]["verdict"], "PASS")
        self.assertIn("a2_6_row1_iv", res["outcome_rows"]["notes"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (e) the contract with the landed confirm applier
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class Contract(Base):
    def confirm(self):
        if EPDIR not in sys.path:
            sys.path.append(EPDIR)
        import refe_head_confirm as C            # noqa: E402
        return C

    def result_path(self, w, o):
        return os.path.join(o.resolved_out_dir(), D.RESULT_NAME)

    def test_read_dev_result_accepts_this_output_and_refuses_corrupted_copies(self):
        C = self.confirm()
        w = self.world(scales=(3.0, 3.0))
        o = w.opts()
        res = D.run_dev(o)
        p = self.result_path(w, o)
        g = C.ConfirmGuard(read_dirs=[o.resolved_out_dir(), self.tmp], read_files=[], write_dirs=[])
        st = C.read_dev_result(g, p, "SR-TTC")
        self.assertEqual(st["verdict"], res["dev_gate"]["verdict"])
        self.assertEqual(st["rows_sha256"], res["inputs"]["rows_sha256"])
        self.assertEqual(st["result_sha256"], res["inputs"]["r1_result_sha256"])
        self.assertEqual(st["status"], "READ")
        doc = rj(p)

        def corrupt(fn, name):
            d = copy.deepcopy(doc)
            fn(d)
            q = os.path.join(self.tmp, name + ".json")
            wj(q, d)
            g2 = C.ConfirmGuard(read_dirs=[self.tmp], read_files=[], write_dirs=[])
            with self.assertRaises(C.ConfirmRefusal, msg=name):
                C.read_dev_result(g2, q, "SR-TTC")
        corrupt(lambda d: d["dev_gate"].__setitem__("verdict", "MAYBE"), "verdict")
        corrupt(lambda d: d.pop("dev_gate"), "nodevgate")
        corrupt(lambda d: d["inputs"]["rows_sha256"].pop("1"), "rowsmissing")
        corrupt(lambda d: d["inputs"].pop("r1_result_sha256"), "noresultsha")
        corrupt(lambda d: d["inputs"].__setitem__("rows_sha256", {}), "rowsempty")

    def test_load_arm_rows_binds_the_rows_the_gate_applied_and_refuses_a_swap_after_the_gate(self):
        C = self.confirm()
        import epr1_dev as ED     # noqa: E402
        w = self.world(scales=(3.0, 3.0))
        o = w.opts()
        res = D.run_dev(o)
        g = C.ConfirmGuard(read_dirs=[w.run, o.resolved_out_dir()], read_files=[w.w["ckpt"]], write_dirs=[])
        st = C.read_dev_result(g, self.result_path(w, o), "SR-TTC")
        W6, b6 = ED.load_p2_head(g, w.w["ckpt"])
        got = C.load_arm_rows(g, "SR-TTC", w.run, w.reg_sha(), st, W6, b6, C.TTC_COL, "r1_result.json", "srttc-r1-v1")
        self.assertEqual({sd: v[2] for sd, v in got["rows"].items()}, res["inputs"]["rows_sha256"])
        # a row swapped after the gate (r1_result.json re-pointed at a different row file with a matching hash) is refused: it is not the row that passed
        w2 = self.world(scales=(3.0, 3.0))
        o2 = w2.opts()
        D.run_dev(o2)
        g2 = C.ConfirmGuard(read_dirs=[w2.run, o2.resolved_out_dir()], read_files=[w2.w["ckpt"]], write_dirs=[])
        st2 = C.read_dev_result(g2, self.result_path(w2, o2), "SR-TTC")
        w2.set_rows("real", (2.0, 2.0))
        with self.assertRaises(C.ConfirmRefusal):
            C.load_arm_rows(g2, "SR-TTC", w2.run, w2.reg_sha(), st2, W6, b6, C.TTC_COL, "r1_result.json", "srttc-r1-v1")

    def test_verdicts_drive_the_confirms_eligibility(self):
        C = self.confirm()
        for scales, c2, want_open in (((3.0, 3.0), (-3.0, -3.0), True), ((-3.0, -3.0), (-3.0, -3.0), False), ((3.0, 3.0), (3.0, 3.0), False)):
            w = self.world(scales=scales, c2_scales=c2)
            o = w.opts()
            res = D.run_dev(o)
            g = C.ConfirmGuard(read_dirs=[o.resolved_out_dir()], read_files=[], write_dirs=[])
            st = C.read_dev_result(g, self.result_path(w, o), "SR-TTC")
            ep = {"status": "READ", "verdict": "PASS"}
            if want_open:
                fams, _ = C.eligible_readings(ep, st)
                self.assertIn("SRTTC", fams)
            else:
                fams, why = C.eligible_readings(ep, st)
                self.assertNotIn("SRTTC", fams, res["dev_gate"]["verdict"])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (f) reachability and the file-system discipline
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
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


def fs_violations(src, allow=()):
    bad = []
    for fn in [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)]:
        if fn.name.startswith("_io_"):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                name = _dotted(n.func)
                if name in FS_CALLS and (fn.name, name) not in allow:
                    bad.append((fn.name, name))
    return bad


class Reachability(Base):
    def test_no_file_system_call_outside_the_io_functions(self):
        with open(SRC, encoding="utf-8") as f:
            self.assertEqual(fs_violations(f.read(), allow={("load_row", "np.load")}), [])           # load_row parses bytes already read through _io_read_bytes
        MUTATED.add(("SRDEV", "AST"))
        self.assertTrue(fs_violations("def f(p):\n    return open(p).read()\n"), "the scan can read a violation")

    def test_every_cli_flag_reaches_dev_options(self):
        ap = D.build_parser()
        a = ap.parse_args(["run", "--run-dir", "R", "--registered-result-sha256", "ab", "--r1-gate-token", "T", "--gate-key-file", "K", "--p2ho-dir", "P", "--out-dir", "O", "--dev-dir", "V",
                           "--p2-ckpt", "C"])
        dests = {act.dest for act in ap._actions if act.dest not in ("help", "cmd")}
        self.assertEqual(dests, {"run_dir", "registered_result_sha256", "r1_gate_token", "gate_key_file", "p2ho_dir", "out_dir", "dev_dir", "p2_ckpt"})
        captured = {}
        orig = D.run_dev
        D.run_dev = lambda o: captured.setdefault("o", o)
        try:
            D.main(["run", "--run-dir", "R", "--registered-result-sha256", "ab", "--r1-gate-token", "T", "--gate-key-file", "K", "--p2ho-dir", "P", "--out-dir", "O", "--dev-dir", "V", "--p2-ckpt", "C"])
        finally:
            D.run_dev = orig
        o = captured["o"]
        self.assertEqual((o.run_dir, o.registered_result_sha256, o.r1_gate_token, o.gate_key_file, o.p2ho_dir, o.out_dir, o.dev_dir, o.p2_ckpt), ("R", "ab", "T", "K", "P", "O", "V", "C"))
        self.assertEqual(o.pins_overridden(), [], "the CLI never changes a pin")

    def test_main_returns_2_on_a_refusal_and_prints_it(self):
        w = self.world(outcomes=False)
        rc = D.main(["run", "--run-dir", w.run, "--registered-result-sha256", "0" * 64, "--dev-dir", w.dev, "--p2-ckpt", w.w["ckpt"], "--p2ho-dir", w.p2ho, "--r1-gate-token", w.token_path,
                     "--gate-key-file", w.key, "--out-dir", os.path.join(w.root, "cli_out")])
        self.assertEqual(rc, 2)

    def test_the_default_path_names_are_the_real_ones_and_the_registered_constants(self):
        self.assertEqual((D.DEV_N_TOKENS, D.DEV_N_LOGS, D.DEV_N_SETS), (1705, 24, 3137))
        self.assertEqual(D.REG_GRID_LR, R.GRID_LR)
        self.assertEqual(D.REG_GRID_ANCHOR, R.GRID_ANCHOR)
        self.assertEqual((D.REG_K, D.REG_EPOCHS, D.REG_BATCH_SETS), (R.K, R.DEFAULT_EPOCHS, R.DEFAULT_BATCH_SETS))
        self.assertEqual(D.REG_FRACTIONS, R.FRACTIONS)
        self.assertEqual((D.LC_DELTA_FLAT, D.LC_PREFIXES), (R.LC_DELTA_FLAT, list(R.LC_PREFIXES)))
        self.assertEqual((D.BOOT_B, D.BOOT_SEED), (R.BOOT_B, R.BOOT_SEED))
        self.assertEqual(D.TTC_COL, R.TTC_COL)
        self.assertEqual(D.SRTTC_R1_SHA256, sha_f(os.path.join(HERE, "srttc_r1.py")), "the pin is the srttc_r1.py beside this file")
        self.assertEqual((D.GATE_PROFILE, D.GATE_KIND, D.GATE_SCHEMA), (G.PROFILE, G.TOKEN_KIND, G.GATE_SCHEMA))
        self.assertEqual(D.SUB_NAMES, ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance"))

    CITES = [("PREREG_SELRANK_ARM.md", 6, "The grid stays"), ("PREREG_SELRANK_ARM.md", 8, "written down"), ("PREREG_SELRANK_ARM.md", 9, "ONCE"), ("PREREG_SELRANK_ARM.md", 39, "Loss."),
             ("PREREG_SELRANK_ARM.md", 67, "a miss voids the arm"), ("PREREG_SELRANK_ARM.md", 69, "C0 identity"), ("PREREG_SELRANK_ARM.md", 70, "C1 reachability"),
             ("PREREG_SELRANK_ARM.md", 71, "C2 negative control"), ("PREREG_SELRANK_ARM.md", 72, "C3 replicate"), ("PREREG_SELRANK_ARM.md", 73, "C4 fan-frozen"),
             ("PREREG_SELRANK_ARM.md", 74, "C5 estimator"), ("PREREG_SELRANK_ARM.md", 93, "+0.30"), ("PREREG_SELRANK_ARM.md", 102, "DEV-FAIL"), ("PREREG_SELRANK_ARM.md", 98, "outcome"),
             ("PREREG_SELRANK_ARM.amendment2.md", 98, "10 %"), ("PREREG_SELRANK_ARM.amendment2.md", 101, "1,167"), ("PREREG_SELRANK_ARM.amendment2.md", 169, "both seeds"),
             ("PREREG_SELRANK_ARM.amendment2.md", 171, "0.010"), ("PREREG_SELRANK_ARM.amendment2.md", 177, "f_max"), ("PREREG_SELRANK_ARM.amendment2.md", 182, "banked"),
             ("PREREG_SELRANK_ARM.amendment2.md", 197, "disclosure"), ("PREREG_SELRANK_ARM.amendment2.md", 206, "FLAT"), ("PREREG_SELRANK_ARM.amendment2.md", 207, "f_max < 100 %"),
             ("PREREG_SELRANK_ARM.amendment2.md", 245, "3,137"), ("PREREG_SELRANK_ARM.amendment2.md", 247, "admissible"), ("MM_DECISIONS_R1.md", 14, "10 epochs"),
             ("MM_DECISIONS_R1.md", 15, "pooled"), ("MM_DECISIONS_R1.md", 20, "Ties"), ("MM_DECISIONS_R1.md", 24, "launch-gate")]

    def test_every_cited_registration_line_says_what_the_code_says_it_does(self):
        """Each `file:line` the trace relies on is opened in the registered document and must contain the phrase the code attributes to it.  The documents sit beside the code in the repo
        (`REFE/raw/2026-10-18-sel-rank/`); outside the repo set SRDEV_DOCS to that directory, otherwise this test reports SKIPPED (never a silent pass)."""
        cands = [os.environ.get("SRDEV_DOCS", ""), os.path.normpath(os.path.join(HERE, ".."))]
        docs = next((d for d in cands if d and os.path.exists(os.path.join(d, "PREREG_SELRANK_ARM.md"))), None)
        if docs is None:
            self.skipTest("registered documents not found beside the code (set SRDEV_DOCS)")
        cache = {}
        for nm, line, phrase in self.CITES:
            if nm not in cache:
                with open(os.path.join(docs, nm), encoding="utf-8") as f:
                    cache[nm] = f.read().split(chr(10))
            self.assertIn(phrase, cache[nm][line - 1], f"{nm}:{line} does not contain {phrase!r}")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# srttc_c2.py
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class C2Fit(Base):
    def test_shuffle_preserves_each_sets_multiset_is_independent_per_set_and_deterministic(self):
        rng = np.random.default_rng(0)
        y = torch.from_numpy((rng.random((50, NP_)) < 0.5).astype(np.float32))
        a, b = C2.shuffle_labels_within_set(y, 7), C2.shuffle_labels_within_set(y, 7)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, C2.shuffle_labels_within_set(y, 8)))
        self.assertFalse(torch.equal(a, y))
        np.testing.assert_array_equal(np.sort(a.numpy(), 1), np.sort(y.numpy(), 1))
        perms = [tuple(np.nonzero(a.numpy()[i] != y.numpy()[i])[0]) for i in range(50)]
        self.assertGreater(len(set(perms)), 25, "every set draws its own permutation")
        np.testing.assert_array_equal(y.numpy(), (rng.random((50, NP_)) * 0 + y.numpy()))      # the input was not modified

    def test_the_fit_trains_on_the_shuffled_labels_and_a_real_label_fit_differs(self):
        b = base()
        o = b["opts"]
        parts = R._setup(o)
        data, W6, b6 = parts[-1], parts[-3], parts[-2]
        cs = b["r1"]["c_star"]
        star = R.Cfg(lr=cs["lr"], anchor=cs["anchor"], epochs=2, batch_sets=32)
        seen = {}

        def spy(tr, idx, cfg, sd, W6_, b6_):
            seen[sd] = tr.y3.clone()
            return R.fit_row(tr, idx, cfg, sd, W6_, b6_)
        rows, y_shuf = C2.fit_c2_rows(data, star, (0, 1), W6, b6, fit=spy)
        self.assertTrue(torch.equal(seen[0], y_shuf))
        self.assertFalse(torch.equal(seen[0], data.y3))
        real = R.fit_row(data, np.arange(data.n_sets), star, 0, W6, b6)
        self.assertEqual(sha_b(R.row_bytes(*real)), b["r1"]["final_rows"]["0"]["sha256"], "the real-label refit reproduces R1's row (same fit path)")
        self.assertNotEqual(sha_b(R.row_bytes(*rows["0"])), b["r1"]["final_rows"]["0"]["sha256"])
        self.assertEqual(set(rows), {"0", "1"})
        # mutation: fit on the real labels -> the C2 row equals the real row (the control would test nothing)
        MUTATED.add(("C2FIT", "LABELS"))
        m = load_mutant(("LABELS", "tr = dataclasses.replace(data, y3=y_shuf)", "tr = data"), path=SRC_C2, prefix="C2FIT", name="srttc_c2_mutant")
        rows_m, _ = m.fit_c2_rows(data, star, (0,), W6, b6)
        self.assertEqual(sha_b(R.row_bytes(*rows_m["0"])), b["r1"]["final_rows"]["0"]["sha256"], "the mutant trains on the real labels")

    def test_run_c2_end_to_end_written_once_and_bound_to_the_r1_run(self):
        b = base()
        c2 = b["c2"]
        self.assertEqual((c2["schema"], c2["stage"], c2["shuffle_seed"], c2["shuffle_seed_registered"]), (R.SCHEMA, "C2", 20261018, False))
        self.assertEqual(c2["c_star"], b["r1"]["c_star"])
        self.assertEqual(c2["bank_manifest_sha256"], b["r1"]["inputs"]["bank_manifest_sha256"])
        self.assertGreater(c2["fraction_of_labels_moved"], 0.2)
        for sd in ("0", "1"):
            ent = c2["final_rows"][sd]
            self.assertEqual(sha_f(os.path.join(b["run"], ent["file"])), ent["sha256"])
            self.assertTrue(ent["differs_from_real_label_row"])
            arr = np.load(os.path.join(b["run"], ent["file"]))
            self.assertEqual((arr.dtype, arr.shape), (np.float32, (257,)))
        with self.assertRaises(C2.C2Error):                                          # written once
            C2.run_c2(b["opts"])

    def test_run_c2_refuses_without_r1_for_a_changed_bank_and_for_another_fraction(self):
        b = base()
        o = TR.opts_for(b["w"], os.path.join(self.tmp, "no_r1"))
        with self.assertRaises(C2.C2Error):
            C2.run_c2(o)
        out = os.path.join(self.tmp, "r1b")
        shutil.copytree(b["run"], out)
        for nm in ("r1_c2_result.json", "ttc_row_c2shuffle_seed0.npy", "ttc_row_c2shuffle_seed1.npy"):
            os.remove(os.path.join(out, nm))
        d = rj(os.path.join(out, "r1_result.json"))
        d["inputs"]["bank_manifest_sha256"] = "0" * 64
        wj(os.path.join(out, "r1_result.json"), d)
        with self.assertRaises(C2.C2Error):
            C2.run_c2(TR.opts_for(b["w"], out))
        d["inputs"]["bank_manifest_sha256"] = b["r1"]["inputs"]["bank_manifest_sha256"]
        d["fraction_pct"] = 25
        wj(os.path.join(out, "r1_result.json"), d)
        with self.assertRaises(C2.C2Error):
            C2.run_c2(TR.opts_for(b["w"], out))

    def test_run_c2_refuses_a_row_equal_to_the_real_row_before_writing_anything(self):
        b = base()
        out = os.path.join(self.tmp, "r1c")
        shutil.copytree(b["run"], out)
        for nm in ("r1_c2_result.json", "ttc_row_c2shuffle_seed0.npy", "ttc_row_c2shuffle_seed1.npy"):
            os.remove(os.path.join(out, nm))
        d = rj(os.path.join(out, "r1_result.json"))
        o = TR.opts_for(b["w"], out)
        # run once to learn the C2 row, then claim it is the REAL row of seed 0
        res = C2.run_c2(o)
        sha = res["final_rows"]["0"]["sha256"]
        for nm in ("r1_c2_result.json", "ttc_row_c2shuffle_seed0.npy", "ttc_row_c2shuffle_seed1.npy"):
            os.remove(os.path.join(out, nm))
        d["final_rows"]["0"]["sha256"] = sha
        wj(os.path.join(out, "r1_result.json"), d)
        MUTATED.add(("C2FIT", "DIFFERS"))
        self.assertTrue(self._c2_refuses(C2, o))
        m = load_mutant(("DIFFERS", "if real is not None and real[\"sha256\"] == R.sha256_bytes(blobs[str(sd)]):", "if False:"), path=SRC_C2, prefix="C2FIT", name="srttc_c2_mutant2")
        self.assertFalse(self._c2_refuses(m, o))
        for nm in ("r1_c2_result.json", "ttc_row_c2shuffle_seed0.npy", "ttc_row_c2shuffle_seed1.npy"):
            if os.path.exists(os.path.join(out, nm)):
                os.remove(os.path.join(out, nm))
        self.assertTrue(self._c2_refuses(C2, o))
        self.assertFalse(os.path.exists(os.path.join(out, "ttc_row_c2shuffle_seed0.npy")), "a refused C2 leaves no row behind")

    def test_a_shuffle_that_moves_no_label_is_refused(self):
        """If the shuffle is the identity the C2 fit would be a real-label fit: C2 must refuse 'moved no label' (and not merely trip the later row-equality check)."""
        b = base()
        out = os.path.join(self.tmp, "r1d")
        shutil.copytree(b["run"], out)
        for nm in ("r1_c2_result.json", "ttc_row_c2shuffle_seed0.npy", "ttc_row_c2shuffle_seed1.npy"):
            os.remove(os.path.join(out, nm))
        o = TR.opts_for(b["w"], out)

        def scen(mod):
            old = mod.shuffle_labels_within_set
            mod.shuffle_labels_within_set = lambda y, seed: y.clone()
            try:
                mod.run_c2(o)
            except mod.C2Error as ex:
                return "moved no label" in str(ex)
            finally:
                mod.shuffle_labels_within_set = old
                for nm in ("r1_c2_result.json", "ttc_row_c2shuffle_seed0.npy", "ttc_row_c2shuffle_seed1.npy"):
                    if os.path.exists(os.path.join(out, nm)):
                        os.remove(os.path.join(out, nm))
            return False
        self.mut("MOVED", "if not moved > 0.0:", "if False:", scen, path=SRC_C2, prefix="C2FIT", name="srttc_c2_mutant3")

    def _c2_refuses(self, mod, o):
        try:
            mod.run_c2(o)
        except mod.C2Error as ex:
            return "equals the real-label row" in str(ex)
        except Exception:                                                              # noqa: BLE001
            return False
        return False

    def test_cli_requires_the_gate_stage_acknowledgement(self):
        self.assertEqual(C2.main(["--fraction", "10"]), 2)

    def test_no_file_system_call_outside_the_guarded_functions_in_c2(self):
        with open(SRC_C2, encoding="utf-8") as f:
            self.assertEqual(fs_violations(f.read()), [])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# coverage: every marker is mutated (named zz so it runs last within this module)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class ZZCoverage(unittest.TestCase):
    def test_every_marker_in_the_sources_was_mutated_by_a_test_in_this_run(self):
        want = {("SRDEV", m) for m in list_markers(SRC, "SRDEV")} | {("C2FIT", m) for m in list_markers(SRC_C2, "C2FIT")}
        got = {x for x in MUTATED}
        missing = sorted(want - got)
        # markers used only inside multi-marker tests are registered by those tests; E2E-only labels are extras and harmless
        self.assertEqual(missing, [], f"markers with no mutation test: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
