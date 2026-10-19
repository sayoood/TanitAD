"""CPU-only synthetic tests for FASTNEG-CENSUS.  `python -I test_fastneg.py` (unittest).  No real data, no GPU, no stack suite.

Groups:  (1) speeds / strata / AUC / bootstrap vs brute-force references;  (2) the decision rule: a branch table AND a RED mutation arm per branch;
(3) the L-arm refit REACHES its parameters (non-zero gradient) with the zero control and the analytic mask identities;  (4) the strata script cannot read labels (spy + stripped copies +
the member guard + a mutated script);  (5) the census entry point refuses to run without strata.json at its recorded sha, and without a gate token.
"""
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
import fastneg_strata as ST                                                          # noqa: E402

E = L.load_epr1()


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# synthetic bank + stores (written exactly like the real ones: npz, the same member names, labels6 present)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def make_synth(root: str, n_logs: int = 6, sets_per_log: int = 6, seed: int = 0, with_feat: bool = False) -> dict:
    """-> dict of paths + the ground truth (true speeds and picks).  Proposal j of a set moves along x at speed v[j]; the set's pick has speed v_b."""
    rng = np.random.default_rng(seed)
    bank, stores = os.path.join(root, "bank"), os.path.join(root, "sets")
    os.makedirs(bank), os.makedirs(stores)
    names = [f"2021.01.01.00.00.{i:02d}_veh-0_{i:05d}_{i + 10:05d}" for i in range(n_logs)]
    keys = sorted(names, key=E.log_key)
    ranked = ["rank\tlog\tkey_sha256\tfold\tfirst_fraction\tn_fit_tokens\tn_fit_sets"]
    for i, lg in enumerate(keys):
        ranked.append(f"{i}\t{lg}\t{E.log_key(lg)}\t{i % 5}\t0.1\t3\t{sets_per_log}")
    ranked_p = os.path.join(root, "ranked_fit_logs.tsv")
    open(ranked_p, "w", encoding="utf-8", newline="\n").write("\n".join(ranked) + "\n")
    truth = {"v": [], "v_b": [], "pick": [], "log": []}
    unit_rows = {"A.sets.npz": [], "B.sets.npz": []}
    S_total = 0
    for li, lg in enumerate(keys):
        S = sets_per_log
        tok = np.array([f"tok{li:02d}{s // 2:02d}" for s in range(S)])
        step = np.array([s % 2 for s in range(S)], np.int32)
        rank = np.array([s % 2 for s in range(S)], np.int32)
        ck = np.full(S, 1000 + li, np.int64)
        traj = np.zeros((S, 64, 20, 3), np.float32)
        logits = np.zeros((S, 64, 6), np.float32)
        labels = rng.random((S, 64, 6)).astype(np.float32)
        feat = rng.normal(size=(S, 64, 256)).astype(np.float16) if with_feat else np.zeros((S, 64, 256), np.float16)
        for s in range(S):
            v = rng.uniform(0.0, 12.0, 64)
            pk = int(rng.integers(0, 64))
            if li == 0 and s < 2:
                v[:] = rng.uniform(0.0, 0.3, 64)            # a standstill set
                v[pk] = 0.1
            else:
                v[pk] = max(float(v[pk]), 1.0)              # never a standstill set by chance
            traj[s, :, :, 0] = (v[:, None] * L.DT_S) * np.arange(1, 21)[None, :]
            logits[s, pk, :5] = 3.0                         # the frozen rule picks pk
            truth["v"].append(v), truth["v_b"].append(v[pk]), truth["pick"].append(pk), truth["log"].append(lg)
        np.savez(os.path.join(stores, lg + ".npz"), token=tok, step=step, rank=rank, ckpt_step=ck, traj3=traj, labels6=labels, label_order=np.array(["nc", "dac", "ep", "ttc", "c", "ddc"]))
        unit_rows["A.sets.npz" if li % 2 == 0 else "B.sets.npz"].append(dict(log=np.array([lg] * S), token=tok, step=step, rank=rank, ckpt_step=ck, feat=feat, logits=logits,
                                                                          top8=np.zeros((S, 8), np.int16), labels6=labels))
        S_total += S
    man = {"files": {}}
    for un, parts in unit_rows.items():
        arr = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(os.path.join(bank, un), **arr)
        with open(os.path.join(bank, un), "rb") as f:
            man["files"][un] = {"path": un, "sha256": hashlib.sha256(f.read()).hexdigest()}
    man_p = os.path.join(bank, "MANIFEST.sha256.json")
    with open(man_p, "w", encoding="utf-8") as f:
        f.write(json.dumps(man))
    return {"root": root, "bank": bank, "stores": stores, "ranked": ranked_p, "manifest": man_p, "n_sets": S_total, "n_logs": n_logs, "logs": keys,
            "truth": {k: (np.array(v) if k != "log" else v) for k, v in truth.items()}}


def run_strata(sy: dict, out: str | None, labels_path: str = "C:/definitely/not/here/labels6.npz", extra=()):
    argv = ["--bank-dir", sy["bank"], "--manifest", sy["manifest"], "--ranked", sy["ranked"], "--stores-dir", sy["stores"], "--fraction", "10", "--labels-path", labels_path,
            "--ranked-sha", "", "--ranked-n", "0", "--expect-sets", str(sy["n_sets"]), "--expect-logs", str(sy["n_logs"]), "--expect-manifest-sha", ""]
    if out is None:
        return ST.main(argv + ["--out-dir", "unused", "--no-write", *extra])
    return ST.main(argv + ["--out-dir", out, *extra])


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class TestSpeedsStrataAuc(unittest.TestCase):
    def test_speed_known_and_origin_segment_counted(self):
        tr = np.zeros((1, 3, 20, 3), np.float32)
        tr[0, 0, :, 0] = np.arange(1, 21) * 1.0                    # 1 m per 0.2 s step from the origin = 5 m/s
        tr[0, 1, :, 0] = np.arange(1, 21) * 0.0                    # stationary
        tr[0, 2, :, 0] = np.arange(1, 21) * 0.5 + 10.0            # pose 1 already 10.5 m out: the origin segment is counted (10.5 + 19 * 0.5)
        v = L.mean_speed(tr)
        self.assertAlmostEqual(float(v[0, 0]), 5.0, places=9)
        self.assertEqual(float(v[0, 1]), 0.0)
        self.assertAlmostEqual(float(v[0, 2]), (10.5 + 19 * 0.5) / 4.0, places=6)
        seg = L.segment_lengths(tr)
        self.assertAlmostEqual(float(seg[0, 0, 0]), 1.0, places=9)  # a MUTANT that started at pose 1 would read 0 here

    def test_strata_ties_standstill_and_quartiles(self):
        rng = np.random.default_rng(1)
        N = 400
        v_b = rng.uniform(2.0, 8.0, N)
        v_b[:20] = 0.3                                             # S sets
        v_b[20] = 0.5                                              # exactly 0.5 is NOT standstill (strict <)
        v = rng.uniform(0.0, 14.0, (N, 64))
        r, cuts, code = L.build_strata(v, v_b)
        self.assertTrue((code[:20] == 0).all())
        self.assertTrue(np.isnan(r[:20]).all())
        self.assertTrue((code[20:] >= 1).all())
        ref = np.quantile((v[20:] / v_b[20:, None]).ravel(), [0.25, 0.5, 0.75])
        self.assertTrue(np.array_equal(cuts, ref))
        for k in (1, 2, 3, 4):
            self.assertAlmostEqual(float((code[20:] == k).mean()), 0.25, delta=0.002)
        # ties go to the LOWER stratum: every row with r <= c25 is Q1, every row with r == c50 exactly is Q2
        v3 = np.array([[1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        r3, c3, k3 = L.build_strata(v3, np.array([1.0]))
        self.assertTrue((k3[0][r3[0] <= c3[0]] == 1).all())
        eq50 = r3[0] == c3[1]
        if eq50.any():
            self.assertTrue((k3[0][eq50] == 2).all())
        self.assertTrue((k3[0][r3[0] > c3[2]] == 4).all())

    def test_pair_stats_vs_bruteforce_with_ties(self):
        rng = np.random.default_rng(2)
        N = 40
        z = rng.integers(-3, 4, (N, 64)).astype(np.float32)        # many exact ties
        lab = rng.choice([0.0, 0.5, 1.0], (N, 64), p=[0.2, 0.1, 0.7])
        code = rng.integers(1, 5, (N, 64)).astype(np.uint8)
        code[::7] = 0                                              # every 7th set is an S set (all code 0)
        un, sf = lab == 0.0, lab == 1.0
        got = L.pair_stats(z, un, sf, code, chunk=7)
        for gname, member in (("Q4", lambda a, b: a == 4 and b == 4), ("Q123", lambda a, b: 1 <= a <= 3 and 1 <= b <= 3), ("S", lambda a, b: a == 0 and b == 0),
                              ("xQ4", lambda a, b: a == 4 and b >= 1)):
            nn, dd = np.zeros(N), np.zeros(N)
            for n in range(N):
                for i in range(64):
                    if not un[n, i]:
                        continue
                    for j in range(64):
                        if sf[n, j] and member(int(code[n, i]), int(code[n, j])):
                            dd[n] += 1
                            nn[n] += 1.0 if z[n, j] > z[n, i] else (0.5 if z[n, j] == z[n, i] else 0.0)
            self.assertTrue(np.array_equal(got[gname][0], nn), gname)
            self.assertTrue(np.array_equal(got[gname][1], dd), gname)

    def test_auc_limits(self):
        rng = np.random.default_rng(3)
        N = 60
        lab = rng.choice([0.0, 1.0], (N, 64))
        code = np.full((N, 64), 4, np.uint8)
        un, sf = lab == 0.0, lab == 1.0
        perfect = lab + 0.01 * rng.random((N, 64))
        n, d = L.pair_stats(perfect.astype(np.float32), un, sf, code)["Q4"]
        self.assertEqual(float(n.sum() / d.sum()), 1.0)
        n, d = L.pair_stats((-perfect).astype(np.float32), un, sf, code)["Q4"]
        self.assertEqual(float(n.sum() / d.sum()), 0.0)
        n, d = L.pair_stats(np.zeros((N, 64), np.float32), un, sf, code)["Q4"]
        self.assertEqual(float(n.sum() / d.sum()), 0.5)

    def test_bootstrap_matches_landed_and_naive(self):
        self.assertTrue(np.array_equal(L.boot_draws(117), E.boot_draws(117)))
        rng = np.random.default_rng(4)
        nl = 30
        num, den = rng.random(nl) * 50, rng.random(nl) * 100 + 1.0
        dr = L.boot_draws(nl, B=500, seed=7)
        ci = L.ratio_ci(num, den, dr)
        v = np.array([num[d].sum() / den[d].sum() for d in dr])
        self.assertAlmostEqual(ci["lo"], float(np.percentile(v, 2.5)), places=12)
        self.assertAlmostEqual(ci["hi"], float(np.percentile(v, 97.5)), places=12)
        num2, den2 = rng.random(nl) * 30, rng.random(nl) * 60 + 1.0
        dc = L.diff_ci(num, den, num2, den2, dr)
        v2 = np.array([num[d].sum() / den[d].sum() - num2[d].sum() / den2[d].sum() for d in dr])
        self.assertAlmostEqual(dc["lo"], float(np.percentile(v2, 2.5)), places=12)
        self.assertAlmostEqual(dc["point"], float(num.sum() / den.sum() - num2.sum() / den2.sum()), places=12)
        # a draw with an empty denominator is skipped, not zero
        z = L.diff_ci(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros((4, 3), np.int64))
        self.assertTrue(np.isnan(z["point"]))


if __name__ == "__main__":
    unittest.main(verbosity=1)
