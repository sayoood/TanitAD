"""CPU-only synthetic END-TO-END tests of the FASTNEG-CENSUS label pass (part 3): the refusals (no strata, wrong sha, no token, stale token), the stage chain inputs -> gate -> fit -> analyze on a small
synthetic bank, the PRESENT path of HEADROW-PROBE's OOF file, and the reach of the strata precondition.  `python -I test_fastneg_census.py` (about a minute; the gate runs the two other test files)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastneg_lib as L                                                              # noqa: E402
import fastneg_strata as ST                                                          # noqa: E402
import fastneg_census as C                                                           # noqa: E402

E = L.load_epr1()


def make_full_synth(root: str, n_logs: int = 6, sets_per_log: int = 20, seed: int = 0) -> dict:
    """A tiny bank + stores that satisfy every integrity check of the real loaders (logits = feat . W + b, top8 = the frozen rule, labels in {0, .5, 1}, stores' labels equal the bank's)."""
    import torch
    rng = np.random.default_rng(seed)
    bank, stores = os.path.join(root, "bank"), os.path.join(root, "sets")
    os.makedirs(bank)
    os.makedirs(stores)
    names = [f"2021.01.01.00.00.{i:02d}_veh-0_{i:05d}_{i + 10:05d}" for i in range(n_logs)]
    keys = sorted(names, key=E.log_key)
    ranked = ["rank\tlog\tkey_sha256\tfold\tfirst_fraction\tn_fit_tokens\tn_fit_sets"] + [f"{i}\t{lg}\t{E.log_key(lg)}\t{i % 5}\t0.1\t3\t{sets_per_log}" for i, lg in enumerate(keys)]
    ranked_p = os.path.join(root, "ranked_fit_logs.tsv")
    with open(ranked_p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(ranked) + "\n")
    W6 = rng.normal(scale=0.08, size=(6, 256)).astype(np.float32)
    b6 = rng.normal(scale=0.2, size=6).astype(np.float32)
    ck = os.path.join(root, "model_final.pt")
    torch.save({"model": {"score_head.weight": torch.from_numpy(W6), "score_head.bias": torch.from_numpy(b6)}}, ck)
    parts = {"A.sets.npz": [], "B.sets.npz": []}
    for li, lg in enumerate(keys):
        S = sets_per_log
        tok = np.array([f"tok{li:02d}{s // 2:02d}" for s in range(S)])
        step = np.array([s % 2 for s in range(S)], np.int32)
        rank = np.array([s % 2 for s in range(S)], np.int32)
        cks = np.full(S, 1000 + li, np.int64)
        feat = rng.normal(size=(S, 64, 256)).astype(np.float16)
        logits = (feat.astype(np.float32) @ W6.T + b6).astype(np.float32)
        top8 = np.stack([E.top8_np(logits[i]) for i in range(S)])
        labels = np.zeros((S, 64, 6), np.float32)
        labels[..., 0] = rng.choice([0.0, 0.5, 1.0], (S, 64), p=[0.2, 0.1, 0.7])
        labels[..., 1] = rng.choice([0.0, 1.0], (S, 64), p=[0.25, 0.75])
        labels[..., 2:] = rng.random((S, 64, 4))
        # make the row informative so the AUCs are not all 0.5: shift the stored NC / DAC logit with the label (the stored logits are then NOT feat . W + b, so adjust the features instead)
        traj = np.zeros((S, 64, 20, 3), np.float32)
        v = rng.uniform(1.0, 12.0, (S, 64))
        if li == 0:
            v[:2] = rng.uniform(0.0, 0.3, (2, 64))                                  # two standstill sets
        traj[..., 0] = (v[:, :, None] * L.DT_S) * np.arange(1, 21)[None, None, :]
        np.savez(os.path.join(stores, lg + ".npz"), token=tok, step=step, rank=rank, ckpt_step=cks, traj3=traj, labels6=labels, label_order=np.array(E.LABEL_ORDER))
        parts["A.sets.npz" if li % 2 == 0 else "B.sets.npz"].append(dict(log=np.array([lg] * S), token=tok, step=step, rank=rank, ckpt_step=cks, feat=feat, logits=logits, top8=top8.astype(np.int16),
                                                                         labels6=labels))
    man = {"files": {}}
    for un, ps in parts.items():
        arr = {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}
        np.savez(os.path.join(bank, un), **arr, label_order=np.array(E.LABEL_ORDER), rule=np.array("navsim_v1"), ckpt=np.array(ck))
        man["files"][un] = {"path": un, "sha256": L.sha256_file(os.path.join(bank, un))}
    man_p = os.path.join(bank, "MANIFEST.sha256.json")
    with open(man_p, "w", encoding="utf-8") as f:
        f.write(json.dumps(man))
    return {"root": root, "bank": bank, "stores": stores, "ranked": ranked_p, "manifest": man_p, "ckpt": ck, "n_sets": n_logs * sets_per_log, "n_logs": n_logs}


class TestCensusPlumbing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="fastneg_c_")
        cls.sy = make_full_synth(cls.tmp)
        cls.rundir = os.path.join(cls.tmp, "run")
        out = os.path.join(cls.rundir, "strata")
        argv = ["--bank-dir", cls.sy["bank"], "--manifest", cls.sy["manifest"], "--ranked", cls.sy["ranked"], "--stores-dir", cls.sy["stores"], "--out-dir", out, "--fraction", "10",
                "--ranked-sha", "", "--ranked-n", "0", "--expect-sets", str(cls.sy["n_sets"]), "--expect-logs", str(cls.sy["n_logs"]), "--expect-manifest-sha", "",
                "--labels-path", "C:/no/such/labels6.npz"]
        assert ST.main(argv) == 0
        with open(os.path.join(out, "strata.json.sha256")) as f:
            cls.sha = f.read().split()[0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def argv(self, stage, sha=None, extra=()):
        return [stage, "--strata-sha", self.sha if sha is None else sha, "--run-dir", self.rundir, "--bank-dir", self.sy["bank"], "--manifest", self.sy["manifest"], "--ranked", self.sy["ranked"],
                "--stores-dir", self.sy["stores"], "--p2-ckpt", self.sy["ckpt"], "--probe-oof", os.path.join(self.tmp, "no_probe.npz"), "--ranked-sha", "", "--ranked-n", "0", "--threads", "2", *extra]

    def test_00_refusals_before_strata_and_with_a_wrong_sha_open_no_label(self):
        import zipfile as zf
        seen, orig_open, orig_load = [], zf.ZipFile.open, np.load

        def spy_open(self_, name, *a, **k):
            seen.append(str(getattr(name, "filename", name)))
            return orig_open(self_, name, *a, **k)

        def spy_load(*a, **k):
            seen.append("np.load:" + str(a[0] if a else ""))
            return orig_load(*a, **k)
        zf.ZipFile.open, np.load = spy_open, spy_load
        try:
            with self.assertRaises(L.CensusError):
                C.main(self.argv("inputs", sha=""))                               # no sha argument
            with self.assertRaises(L.CensusError):
                C.main(self.argv("inputs", sha="0" * 64))                         # a wrong sha
            empty = os.path.join(self.tmp, "run_empty")
            with self.assertRaises(L.CensusError):
                C.main(self.argv("inputs", extra=["--run-dir", empty]))           # no strata.json at all (the later --run-dir wins)
        finally:
            zf.ZipFile.open, np.load = orig_open, orig_load
        self.assertFalse([s for s in seen if "label" in s.lower()], seen[:3])
        self.assertFalse(os.path.exists(os.path.join(self.rundir, "inputs", "meta.npz")))

    def test_10_chain_inputs_gate_fit_analyze(self):
        self.assertEqual(C.main(self.argv("inputs")), 0)
        with open(os.path.join(self.rundir, "inputs", "inputs_report.json")) as f:
            rep = json.load(f)
        self.assertTrue(rep["ok"] and rep["join_proof"]["labels6_mismatches"] == 0 and rep["alignment_with_strata"]["keys_equal"])
        # a fit before the gate is refused
        with self.assertRaises(L.CensusError):
            C.main(self.argv("fit", extra=["--groups", "primary"]))
        self.assertEqual(C.main(self.argv("gate")), 0)
        with open(os.path.join(self.rundir, "gate_token.json")) as f:
            tok = json.load(f)
        self.assertTrue(tok["pass"] and set(tok["checks"]) == {"G-BANK", "G-DVB", "G-LIVE", "G-SUITE"} and all(v["pass"] for v in tok["checks"].values()))
        # a token whose binding no longer matches is refused (stale code / config)
        tpath = os.path.join(self.rundir, "gate_token.json")
        bad = json.loads(json.dumps(tok))
        bad["bindings"]["config"]["epochs"] = 19
        with open(tpath, "w") as f:
            json.dump(bad, f)
        with self.assertRaises(L.CensusError):
            C.main(self.argv("fit", extra=["--groups", "primary"]))
        with open(tpath, "w") as f:
            json.dump(tok, f)
        # the fits
        self.assertEqual(C.main(self.argv("fit", extra=["--groups", "primary,placebo,cold,null"])), 0)
        fits = sorted(n for n in os.listdir(os.path.join(self.rundir, "fits")) if n.endswith(".npy"))
        self.assertEqual(len(fits), 6 + 2 + 6 + 20)
        # a second call skips everything (resume)
        self.assertEqual(C.main(self.argv("fit", extra=["--groups", "primary"])), 0)
        # analysis without the probe file: (b)(ii) MISSING, the rule still runs on our refits
        self.assertEqual(C.main(self.argv("analyze")), 0)
        with open(os.path.join(self.rundir, "results", "census.json")) as f:
            res = json.load(f)
        self.assertEqual(res["b_ii_probe"]["status"], "MISSING")
        self.assertEqual(res["c_refit"]["status"], "PRESENT")
        self.assertEqual(res["rule"]["status"], "RAN")
        self.assertEqual(res["c0_null"]["status"], "PRESENT")
        self.assertIn(res["outcome"]["outcome"], ("COVERAGE", "FEATURE", "NO_DEFICIT", "INCONCLUSIVE", "VOID"))
        for row in L.ROWS:
            self.assertEqual(len(res["c0_null"]["rows"][row]["replicates"]), 5)
            for s in (0, 1):
                d = res["c_refit"]["rows"][row][f"s{s}"]
                self.assertEqual(d["deficit_call"], L.deficit_call(d["F_minus_Q123_at_100"]))
                self.assertEqual(d["slope_call"], L.slope_call(d["slope_F_100_minus_25"]))
        # C-AUC instrument controls: random scores ~ 0.5 (the interval brackets it), oracle scores = 1
        for row in L.ROWS:
            c = res["controls"]["C_AUC_instrument"][row]
            self.assertEqual(c["oracle_scores_F"], 1.0)
            self.assertLessEqual(c["random_scores_F"]["lo"], 0.5)
            self.assertGreaterEqual(c["random_scores_F"]["hi"], 0.5)
        # (a): the strata rows sum to every row, the per-stratum shares of unsafe rows sum to 1
        a = res["a_support"]
        self.assertEqual(sum(a[k]["rows"] for k in a), self.sy["n_sets"] * 64)
        self.assertAlmostEqual(sum(a[k]["share_of_all_unsafe_rows"]["point"] for k in a), 1.0, places=9)
        # now the PRESENT path of HEADROW-PROBE's OOF file: L arms = our own tables (C-REPRO must read EXACTLY 0), M arms random
        ctx = C.load_ctx(self_args(self), need_feat=False)
        rng = np.random.default_rng(5)
        pz = {"token": ctx["key_token"], "log": ctx["key_log"], "step": ctx["key_step"].astype(np.int32), "rank": ctx["key_rank"].astype(np.int32), "ckpt_step": ctx["key_ckpt_step"],
              "mean_speed_mps": ctx["v"].astype(np.float32), "p2_pick_mean_speed_mps": ctx["v_b"].astype(np.float32), "p2_pick_index": ctx["pick"], "labels6": ctx["labels6"],
              "p2_logits6": ctx["z6"], "fold": ctx["fold"]}
        for ri, row in enumerate(L.ROWS):
            for s in (0, 1):
                pz[f"oof_{row}_L_s{s}"] = np.load(os.path.join(self.rundir, "fits", f"L_f100_s{s}.npy"))[ri]
                pz[f"oof_{row}_M_s{s}"] = rng.normal(size=ctx["code"].shape).astype(np.float32)
        pp = os.path.join(self.tmp, "fake_probe.npz")
        np.savez(pp, **pz)
        self.assertEqual(C.main(self.argv("analyze", extra=["--probe-oof", pp])), 0)
        with open(os.path.join(self.rundir, "results", "census.json")) as f:
            res2 = json.load(f)
        self.assertEqual(res2["b_ii_probe"]["status"], "PRESENT")
        for k, v in res2["controls"]["C_REPRO_our_L100_vs_probe_L"].items():
            self.assertEqual(v["max_abs_logit_diff"], 0.0, k)
        self.assertTrue(res2["outcome_agrees_with_probe_L"])
        self.assertTrue(res2["b_ii_probe"]["cross_checks"]["pick_index_equal"] and res2["b_ii_probe"]["cross_checks"]["labels6_equal_bank"])
        self.assertLess(res2["b_ii_probe"]["cross_checks"]["mean_speed_max_abs_diff_vs_ours"], 1e-4)
        # a probe file listing the sets in a different order STOPS (b)(ii)
        pz2 = dict(pz)
        pz2["token"] = pz["token"][::-1].copy()
        pq = os.path.join(self.tmp, "fake_probe_bad.npz")
        np.savez(pq, **pz2)
        with self.assertRaises(L.CensusError):
            C.main(self.argv("analyze", extra=["--probe-oof", pq]))


def self_args(t):
    return C.build_parser().parse_args(t.argv("analyze"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
