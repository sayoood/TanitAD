#!/usr/bin/env python3
"""Tests of l2next_read.py (the generalised read; AMENDMENT_READ.md).  SYNTHETIC ONLY: every cache, head, table and run directory is fabricated in a temp dir; nothing real is
opened, nothing is trained.  Every number printed is synthetic.

    python test_l2next_read.py [--tmp <dir>]      -> prints ZZL2READ_PASS <n> (or ZZL2READ_FAIL)

Covers: (1) a schema /1 head at step 0 under --lever shmlp reads C1's picks bit for bit; (2) a schema /2 MLP head with a non-zero last layer changes logits only on its meta rows
and the read's picks equal an independent numpy MLP; (3) EP-HEAD reads chosen_step from cv_select.json (a decoy heldout_select.json is ignored) and chosen_step 0 is NULL-CV;
(4) two mutations (the meta rows ignored; heldout_select read for ephead) turn the checks RED; plus CTRL-CV, CTRL-REACH, CTRL-RED and CTRL-T2 are exercised, and the bar /
estimator / threshold lines are pinned.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import l2next_read as R  # noqa: E402

REFE = HERE.parent.parent / "refe"
sys.path.insert(0, str(REFE))
import safehead_data as D  # noqa: E402

N = 0
H = 4                                              # synthetic MLP hidden width (the card's 64 is the trainer's; the read takes it from the arrays)
NT, NLOGS_T = 24, 6                                # navtest tokens / logs
TRAIN_LOGS = [f"T{i:02d}" for i in range(10)]
HELDOUT = TRAIN_LOGS[7:]
STEPS = (0, 250)
D0 = {"oof_pdms": 90.0, "per_fold_oof_pdms": [90.0] * 5, "oof_safety_miss_rate": 0.1, "oof_ep": 80.0, "switch_rate_vs_c1": 0.0, "oof_delta_vs_step0": 0.0,
      "descriptive_logs_improved_worsened_of": [0, 0, 10]}


def ok(name):
    global N
    N += 1
    print(f"ok {N:02d} {name}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ============================================================================================ the synthetic world
def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


class World:
    def __init__(self, root: Path, seed=0):
        self.root = root
        rng = np.random.default_rng(seed)
        self.W0 = (rng.standard_normal((6, 256)) * 0.3).astype(np.float32)
        self.b0 = (rng.standard_normal(6) * 0.3).astype(np.float32)
        self.S = (rng.standard_normal((NT, 64, 256)) * 0.2).astype(np.float32)
        self.BL = D.direct_logits(self.S, self.W0, self.b0)                          # C1's banked logits: exactly the head applied to s (CTRL-FEAT-NT = 0)
        self.BP = D.select(self.BL)
        self.tok = np.array([f"tok{i:03d}" for i in range(NT)])
        self.logs = np.array([f"N{i // 4}" for i in range(NT)])
        ar = np.arange(NT)
        self.props = rng.standard_normal((NT, 64, 4, 2)).astype(np.float32)
        cache = root / "navtest_cache"
        cache.mkdir(parents=True)
        rows = []
        for lg in sorted(set(self.logs)):
            m = self.logs == lg
            np.savez(cache / f"{lg}.npz", token=self.tok[m], s=self.S[m], logits=self.BL[m], pick=self.BP[m], props=self.props[m])
            rows += [(t, lg, "navtest", f"{lg}.npz") for t in self.tok[m]]
        (cache / "index.csv").write_text("token,log,split,file\n" + "".join(",".join(r) + "\n" for r in rows), encoding="utf-8")
        D.seal_index(cache / "index.csv", root / "sealed" / "navtest_sealed.csv")
        self.nav_index = root / "sealed" / "navtest_sealed.csv"
        np.savez(root / "proposals.npz", token=self.tok, logits=self.BL, pick=self.BP, proposals=self.props)
        np.savez(root / "seam.npz", token=self.tok, poses=self.props[ar, self.BP], fingerprint=np.array("fp"), sampling=np.array("sm"))
        self.tr = {k: rng.uniform(0, 1, (NT, 64)) for k in R.SUBS + ("pdms",)}
        (root / "table").mkdir()
        with open(root / "table" / "full_C1_s0of2.jsonl", "w", encoding="utf-8") as fh:
            for i in range(NT):
                fh.write(json.dumps({"token": str(self.tok[i]), "log": str(self.logs[i]), "pick": int(self.BP[i]),
                                     "tr": {k: [float(x) for x in self.tr[k][i]] for k in self.tr}}) + "\n")
        self.c1_pdms = float((self.tr["pdms"][ar, self.BP] * 100).mean())
        tr_rows = [(f"{lg}_{j}", lg, "heldout" if lg in HELDOUT else "train", "x.npz") for lg in TRAIN_LOGS for j in range(2)]
        (root / "train_index.csv").write_text("token,log,split,file\n" + "".join(",".join(r) + "\n" for r in tr_rows), encoding="utf-8")
        (root / "sft_heldout_logs.txt").write_text("\n".join(HELDOUT) + "\n", encoding="utf-8")
        self.train_sha = sha(root / "train_index.csv")
        ev = root / "eval" / "p5c"
        ev.mkdir(parents=True)
        (ev / "p5c_read.py").write_text(
            "import numpy as np\nSEED, NBOOT = 20260927, 10000\nclass Boot:\n    def __init__(self, logs):\n        self.logs = np.asarray(logs)\n"
            "        self.u = sorted(set(self.logs.tolist()))\n        self.ix = [np.where(self.logs == l)[0] for l in self.u]\n"
            "    def draws(self, d):\n        r = np.random.default_rng(20260927)\n        d = np.asarray(d, float)\n"
            "        return np.array([d[np.concatenate([self.ix[j] for j in r.integers(0, len(self.u), len(self.u))])].mean() for _ in range(200)])\n", encoding="utf-8")
        self.eval_dir = root / "eval"
        self.nonzero = {}

    # ---- a run directory -------------------------------------------------------------------------------------------------------
    def head_arrays(self, lever, step, rng_seed, delta=True):
        rows = list(R.LEVERS[lever]["rows"])
        rng = np.random.default_rng(rng_seed)
        W, b, mlp = self.W0.copy(), self.b0.copy(), None
        if lever == "shmlp":
            nr = len(rows)
            mlp = {"mlp_w1": (rng.standard_normal((nr, H, 256)) * 0.1).astype(np.float32), "mlp_b1": (rng.standard_normal((nr, H)) * 0.1).astype(np.float32),
                   "mlp_w2": np.zeros((nr, H), np.float32), "mlp_b2": np.zeros(nr, np.float32)}
            if step and delta:
                mlp["mlp_w2"] = (rng.standard_normal((nr, H)) * 4.0).astype(np.float32)
                mlp["mlp_b2"] = (rng.standard_normal(nr)).astype(np.float32)
        elif step and delta:
            W[2] = W[2] + (rng.standard_normal(256) * 0.5).astype(np.float32)
            b[2] = b[2] + np.float32(0.7)
        return rows, W, b, mlp

    def run(self, lever, arm, chosen, *, schema1_for_shmlp=False, decoy_heldout=False, cv_red_final_up=False, name=None, delta=True):
        lv = R.LEVERS[lever]
        d = self.root / (name or f"run_{lever}_{arm}")
        d.mkdir(parents=True)
        seed, perm = {"s0": (0, -1), "s1": (1, -1), "red": (0, 20261127)}[arm]
        rows = list(lv["rows"])
        shas = {}
        for step in STEPS:
            rows, W, b, mlp = self.head_arrays(lever, step, 7 + step + seed, delta=delta and arm != "red")
            meta = {"rows": rows, "seed": seed, "init_sha256": R.C1_SHA256, "index_sha256": self.train_sha, "argv": [], "smoke": False, "permute_labels_seed": perm,
                    "loss": "t", "token": {}, "step": step}
            if lever == "shmlp" and not schema1_for_shmlp:
                meta["mlp_hidden"] = H
            if lever == "ephead":
                meta["select"] = "cv5"
            mm = mlp if not (lever == "shmlp" and schema1_for_shmlp) else None
            shas[step] = D.save_head(d / f"head_step{step:05d}.npz", W=W, b=b, W0=self.W0, b0=self.b0, meta=meta, mlp=mm)
        (d / "heads.sha256").write_text("".join(f"{shas[s]}  head_step{s:05d}.npz\n" for s in sorted(shas)), encoding="utf-8")
        (d / "run.json").write_text(json.dumps({"chosen_step": chosen, "token": {"profile": f"{lv['prefix']}_{arm}", "token_sha256": "t"}}), encoding="utf-8")
        frozen = {f"row{r}": 0.0 for r in range(6) if r not in rows}
        (d / "metrics.jsonl").write_text("".join(json.dumps({"grad_abs": {f"row{r}": 0.5 for r in rows}, "frozen_grad_abs": frozen}) + "\n" for _ in range(6)), encoding="utf-8")
        fold = D.cv_folds(TRAIN_LOGS, 5)
        (d / "built.json").write_text(json.dumps({"rows": rows, "n_train": 20, "mlp_last_layer_abs_sum_init": 0.0 if (lever == "shmlp") else None,
                                                  "cv_fold_of_log": {lg: [fold[lg]] for lg in TRAIN_LOGS} if lv["cv"] else None}), encoding="utf-8")
        if lever == "ephead":
            p250 = 89.0 if (arm == "red" or chosen == 0 and not cv_red_final_up) else 90.5
            cv = {"n_tokens": 20, "n_logs": 10, "folds": {str(f): sorted(lg for lg in TRAIN_LOGS if fold[lg] == f) for f in range(5)}, "chosen_step": chosen,
                  "per_step": {"0": copy.deepcopy(D0), "250": {**copy.deepcopy(D0), "oof_pdms": p250}}}
            (d / "cv_select.json").write_text(json.dumps(cv), encoding="utf-8")
            if decoy_heldout:                                                   # a held-out file that says "step 0": a read that opens it picks the wrong head
                (d / "heldout_select.json").write_text(json.dumps({"chosen_step": 0, "per_step": {}}), encoding="utf-8")
        else:
            ps = {"0": {"safety_miss_rate": 0.1, "auc_nc": 0.8, "auc_dac": 0.8, "auc_ttc": 0.8},
                  "250": {"safety_miss_rate": 0.1, "auc_nc": 0.8, "auc_dac": 0.8, "auc_ttc": 0.8}}
            if arm == "red":
                ps["250"] = {"safety_miss_rate": 0.2, "auc_nc": 0.5, "auc_dac": 0.5, "auc_ttc": 0.5}
            (d / "heldout_select.json").write_text(json.dumps({"chosen_step": chosen, "per_step": ps}), encoding="utf-8")
        return d

    def n(self):
        self.ctr = getattr(self, 'ctr', 0) + 1
        return self.ctr

    def read(self, lever, runs, out):
        out = self.root / out
        argv = ["read", "--lever", lever, "--table", str(self.root / "table" / "*of2.jsonl"), "--no-sha", "--proposals", str(self.root / "proposals.npz"), "--seam", str(self.root / "seam.npz"),
                "--navtest-index", str(self.nav_index), "--train-index", str(self.root / "train_index.csv"), "--heldout-logs", str(self.root / "sft_heldout_logs.txt"),
                "--run-s0", str(runs["s0"]), "--run-s1", str(runs["s1"]), "--run-red", str(runs["red"]), "--eval-dir", str(self.eval_dir), "--refe", str(REFE), "--out", str(out),
                "--c1-pdms", repr(self.c1_pdms), "--n-tokens", str(NT), "--n-logs", str(NLOGS_T), "--n-train-tokens", "20", "--n-train-logs", "10", "--reach-steps", "6"]
        rc = R.main(argv)
        rep = json.loads((out / f"READ_{R.LEVERS[lever]['tag']}.json").read_text(encoding="utf-8"))
        return rc, rep, out

    def runs(self, lever, c0, c1, tag, **kw):
        return {"s0": self.run(lever, "s0", c0, name=f"{tag}_s0", **kw), "s1": self.run(lever, "s1", c1, name=f"{tag}_s1"), "red": self.run(lever, "red", 0, name=f"{tag}_red")}

    # ---- independent expected picks (numpy, not the read's code path) --------------------------------------------------------------
    def expected_picks(self, lever, run_dir, step):
        h = D.load_head(run_dir / f"head_step{step:05d}.npz")
        L = self.BL.copy()
        if lever == "ephead":
            L[..., 2] = self.BL[..., 2] + (self.S @ (h["W"] - h["W0"])[2] + (h["b"] - h["b0"])[2])
        else:
            for i, r in enumerate(h["meta"]["rows"]):
                hid = gelu(self.S @ h["mlp"]["mlp_w1"][i].T + h["mlp"]["mlp_b1"][i]).astype(np.float32)
                L[..., r] = self.BL[..., r] + (hid @ h["mlp"]["mlp_w2"][i] + h["mlp"]["mlp_b2"][i])
        return D.select(L), L


def picks_of(out):
    return np.load(out / "seams" / "picks.npz", allow_pickle=False)


# ============================================================================================ checks (each is also run under a mutation)
def check_ephead_picks_and_step(w):
    """EP-HEAD, s0 and s1 at step 250 chosen by cv_select.json (a decoy heldout_select.json says step 0): the read must use the CV file and apply the head on its META row (2)."""
    runs = w.runs("ephead", 250, 250, f"eA{w.n()}", decoy_heldout=True)
    rc, rep, out = w.read("ephead", runs, f"outA{w.n()}")
    assert rep["M1_M3"]["s0"]["chosen_step"] == 250, rep["M1_M3"]["s0"]["chosen_step"]
    want, _ = w.expected_picks("ephead", runs["s0"], 250)
    pk = picks_of(out)
    assert np.array_equal(pk["EPH0"], want), "the EPH0 picks are not the independent head-on-meta-row picks"
    assert not np.array_equal(want, w.BP), "synthetic head too weak: the pick never moves"
    assert np.array_equal(pk["C1"], w.BP)
    return rep, out, runs


def main() -> int:
    tmp = Path(sys.argv[sys.argv.index("--tmp") + 1]) if "--tmp" in sys.argv else Path(tempfile.mkdtemp(prefix="l2read_"))
    shutil.rmtree(tmp, ignore_errors=True) if "--tmp" in sys.argv else None
    tmp.mkdir(parents=True, exist_ok=True)
    w = World(tmp / "w")
    # ---- 0. the frozen constants ------------------------------------------------------------------------------------------------
    src = (HERE / "l2next_read.py").read_text(encoding="utf-8")
    for line in ('F_REFE = 0.1158', 'DET_TOL, FEAT_MAX, FEAT_P99, T2_TOL = 1e-4, 1e-4, 1e-4, 1e-6', 'bar2 = 2 * max(F, F_REFE)', 'P1 = d0["lo"] > 0', 'P2 = d0["point"] >= bar2',
                 'R = d1["point"] > 0 and not adverse1', 'if d0["hi"] < 0 or not G:', 'elif P1 and P2 and G and R:', 'elif P1 and G:', 'if (p5c_read.SEED, p5c_read.NBOOT) != (20260927, 10000):',
                 'm = bt.draws(d)', 'np.percentile(m, 2.5)', 'np.percentile(m, 97.5)', 'abs(float(pa.mean()) - a.c1_pdms) <= 5e-5',
                 'JUDGED = ("nc", "dac", "ttc", "ep", "c")'):
        assert line in src, line
    ok("the bar, estimator and threshold lines are present verbatim")

    # ---- 1. schema /1 head at step 0 under --lever shmlp: C1's picks bit for bit -------------------------------------------------------
    runs = w.runs("shmlp", 0, 0, "s1v", schema1_for_shmlp=True)
    rc, rep, out = w.read("shmlp", runs, "out1")
    pk = picks_of(out)
    c = rep["controls"]
    assert c["CTRL-ID"]["pass"] and c["CTRL-ID"]["logits_bitexact"] and c["CTRL-ID"]["pick_agree"] == NT
    assert np.array_equal(pk["SHM0"], w.BP) and np.array_equal(pk["SHM1"], w.BP) and np.array_equal(pk["C1"], w.BP)
    assert not c["CTRL-SHA"]["pass"] and any("schema" in p for p in c["CTRL-SHA"]["problems"]), "a /1 head under shmlp (the flag ignored) must FAIL CTRL-SHA"
    assert rep["outcome_provisional"]["outcome"] == "INVALID"
    ok("shmlp, schema /1 head at step 0: picks == C1's bit for bit, CTRL-ID passes; the /1 form under shmlp is refused by CTRL-SHA")

    # ---- the clean shmlp world at step 0: every control passes, NULL-HELDOUT ----------------------------------------------------------
    runs = w.runs("shmlp", 0, 0, "s1c")
    rc, rep, out = w.read("shmlp", runs, "out1c")
    bad = [k for k, v in rep["controls"].items() if not v["pass"]]
    assert not bad, (bad, {k: rep["controls"][k] for k in bad})
    assert rep["outcome_provisional"]["outcome"] == "NULL-HELDOUT" and "DATA-SCALE" in rep["outcome_provisional"]["next_arm"]
    assert np.array_equal(picks_of(out)["SHM0"], w.BP)
    ok("shmlp, schema /2 head at step 0: all controls PASS, picks == C1's, outcome NULL-HELDOUT -> DATA-SCALE")

    # ---- 2. schema /2 head with a non-zero last layer: only the meta rows move -----------------------------------------------------------
    r2 = w.run("shmlp", "s0", 250, name="solo_s0")
    h = D.load_head(r2 / "head_step00250.npz")
    assert h["meta"]["schema"] == "safehead.head/2" and np.abs(h["mlp"]["mlp_w2"]).sum() > 0
    L = D.head_logits(w.BL, w.S, h)
    rows = list(h["meta"]["rows"])
    other = [r for r in range(6) if r not in rows]
    assert np.array_equal(L[..., other], w.BL[..., other]), "a row outside the meta rows moved"
    assert all(np.abs(L[..., r] - w.BL[..., r]).max() > 1e-3 for r in rows), "a meta row did not move"
    want, Lw = w.expected_picks("shmlp", r2, 250)
    assert np.abs(L - Lw).max() < 1e-4, np.abs(L - Lw).max()
    h0 = D.load_head(r2 / "head_step00000.npz")
    assert np.array_equal(D.head_logits(w.BL, w.S, h0), w.BL), "the step-0 MLP head is not C1 bit for bit"
    runs = w.runs("shmlp", 250, 250, "s2")
    rc, rep, out = w.read("shmlp", runs, "out2")
    assert np.array_equal(picks_of(out)["SHM0"], want) and not np.array_equal(want, w.BP)
    bad = [k for k, v in rep["controls"].items() if not v["pass"]]
    assert not bad, bad
    assert rep["M1_M3"]["s0"]["chosen_step"] == 250 and rep["outcome_provisional"]["outcome"] not in ("INVALID", "NULL-HELDOUT")
    ok("shmlp, schema /2 head: logits change only on the meta rows, step 0 bit-exact, the read's picks equal an independent numpy MLP, all controls PASS")

    # ---- 3. ephead: chosen_step from cv_select.json; chosen_step 0 is NULL-CV --------------------------------------------------------------
    rep, out, runs = check_ephead_picks_and_step(w)
    bad = [k for k, v in rep["controls"].items() if not v["pass"]]
    assert not bad, (bad, {k: rep["controls"][k] for k in bad})
    assert "CTRL-CV" in rep["controls"] and rep["outcome_provisional"]["outcome"] not in ("INVALID", "NULL-CV", "NULL-HELDOUT")
    assert "pooled_rank_corr_ep_vs_harness_tr_ep" in rep["M5"]["EPH0"] and "oof_per_step_table" in rep["M5"]["EPH0"]
    ok("ephead: the decision step is cv_select.json's (a decoy heldout_select.json ignored), the head is applied on its meta row, all controls incl. CTRL-CV PASS")
    runs0 = w.runs("ephead", 0, 250, "eB", decoy_heldout=True)
    rc, rep0, out0 = w.read("ephead", runs0, "outB")
    bad = [k for k, v in rep0["controls"].items() if not v["pass"]]
    assert not bad, (bad, {k: rep0["controls"][k] for k in bad})
    assert rep0["outcome_provisional"]["outcome"] == "NULL-CV" and "DATA-SCALE" in rep0["outcome_provisional"]["next_arm"], rep0["outcome_provisional"]
    assert np.array_equal(picks_of(out0)["EPH0"], w.BP)
    ok("ephead: chosen_step 0 -> NULL-CV -> DATA-SCALE, EPH0's picks == C1's")

    # ---- controls that must FAIL --------------------------------------------------------------------------------------------------------
    runs_f = w.runs("ephead", 250, 250, "eF")
    cvp = runs_f["s0"] / "cv_select.json"
    j = json.loads(cvp.read_text())
    j["per_step"]["0"]["oof_pdms"] = 91.0                                          # the argmax is now step 0, but chosen_step says 250
    cvp.write_text(json.dumps(j))
    rc, rf, _ = w.read("ephead", runs_f, "outF")
    assert not rf["controls"]["CTRL-CV"]["pass"] and not rf["controls"]["CTRL-CV"]["s0"]["d_chosen_is_argmax"] and rf["outcome_provisional"]["outcome"] == "INVALID"
    runs_g = w.runs("ephead", 250, 250, "eG")
    bp = runs_g["s1"] / "built.json"
    j = json.loads(bp.read_text())
    j["cv_fold_of_log"]["T00"] = [3]
    bp.write_text(json.dumps(j))
    rc, rg, _ = w.read("ephead", runs_g, "outG")
    assert not rg["controls"]["CTRL-CV"]["pass"] and not rg["controls"]["CTRL-CV"]["s1"]["a_folds_ok"]
    runs_h = w.runs("ephead", 250, 250, "eH")
    mp = runs_h["s0"] / "metrics.jsonl"
    lines = mp.read_text().splitlines()
    rec = json.loads(lines[3])
    rec["frozen_grad_abs"]["row0"] = 1e-9
    lines[3] = json.dumps(rec)
    mp.write_text("\n".join(lines) + "\n")
    rc, rh, _ = w.read("ephead", runs_h, "outH")
    assert not rh["controls"]["CTRL-REACH"]["pass"] and rh["outcome_provisional"]["outcome"] == "INVALID"
    runs_i = w.runs("ephead", 250, 250, "eI")
    ip = runs_i["red"] / "cv_select.json"
    j = json.loads(ip.read_text())
    j["per_step"]["250"]["oof_pdms"] = 91.0                                         # RED improves out of fold: a path that ignores the labels
    ip.write_text(json.dumps(j))
    rc, ri, _ = w.read("ephead", runs_i, "outI")
    assert not ri["controls"]["CTRL-RED"]["pass"]
    ok("CTRL-CV (argmax, folds), CTRL-REACH (a frozen-row gradient) and CTRL-RED (RED improving) each turn INVALID when broken")

    # ---- confirm: CTRL-T2 reads the lever's arm labels --------------------------------------------------------------------------------------
    pr = w.root / "pass_rows.jsonl"
    pk = picks_of(out)
    with open(pr, "w", encoding="utf-8") as fh:
        for i, t in enumerate(w.tok):
            arms = {nm: {"tracked": {s: float(w.tr[s][i, int(k[i])]) for s in R.SUBS + ("pdms",)}} for nm, k in (("EPH0", pk["EPH0"]), ("EPH1", pk["EPH1"]), ("C1pick", pk["C1"]))}
            fh.write(json.dumps({"token": str(t), "arms": arms}) + "\n")
    fams = w.root / "fam.json"
    fams.write_text(json.dumps({"x": 1}))
    cfmt = ["confirm", "--table", str(w.root / "table" / "*of2.jsonl"), "--read", str(out / "READ_EPHEAD.json"), "--pass-rows", str(pr), "--families", str(fams), "--tactical", str(fams),
            "--strategic", str(fams), "--out", str(out)]
    assert R.main(cfmt) == 0
    v = json.loads((out / "VERDICT_EPHEAD.json").read_text())
    assert v["controls"]["CTRL-T2"]["pass"] and set(v["controls"]["CTRL-T2"]) >= {"EPH0", "EPH1", "C1pick"} and v["outcome"]["outcome"] != "INCOMPLETE"
    pr2 = w.root / "pass_rows_wrong.jsonl"
    pr2.write_text(pr.read_text().replace("EPH0", "SH0"))
    cfmt[cfmt.index(str(pr))] = str(pr2)
    R.main(cfmt)
    v = json.loads((out / "VERDICT_EPHEAD.json").read_text())
    assert not v["controls"]["CTRL-T2"]["pass"] and v["outcome"]["outcome"] == "INVALID"
    ok("confirm: CTRL-T2 reads the arm labels EPH0 / EPH1 / C1pick (the SAFE-HEAD label SH0 is not accepted); VERDICT_EPHEAD.json")

    # ---- 4. mutations: each is applied IN PLACE to the module the read runs from, and must turn a check RED ------------------------------------------
    import contextlib

    @contextlib.contextmanager
    def patched(name, value):
        old = getattr(R, name)
        setattr(R, name, value)
        try:
            yield
        finally:
            setattr(R, name, old)

    with patched("_logits", lambda D_, base, s, h: D_.head_logits(base, s, h, rows=R.ROWS)):           # MUTATION 1: the head's meta rows ignored (SAFE-HEAD's rows used)
        try:
            check_ephead_picks_and_step(w)
        except AssertionError:
            ok("MUTATION 1 (the head's meta rows ignored) turns the ephead check RED")
        else:
            raise SystemExit("MUTATION 1 DID NOT TURN THE CHECK RED")
    check_ephead_picks_and_step(w)                                                                       # the unmutated module is GREEN again
    lv2 = copy.deepcopy(R.LEVERS)
    lv2["ephead"]["select_file"] = "heldout_select.json"                                                  # MUTATION 2: EP-HEAD reads the held-out file for its decision step
    with patched("LEVERS", lv2):
        try:
            check_ephead_picks_and_step(w)
        except (AssertionError, KeyError, FileNotFoundError):
            ok("MUTATION 2 (heldout_select.json read for ephead) turns the ephead check RED")
        else:
            raise SystemExit("MUTATION 2 DID NOT TURN THE CHECK RED")
    lv3 = copy.deepcopy(R.LEVERS)
    lv3["shmlp"]["select_file"] = "cv_select.json"                                                        # MUTATION 3: shmlp reads the CV file (the shmlp world has none)
    with patched("LEVERS", lv3):
        try:
            w.read("shmlp", w.runs("shmlp", 0, 0, "m3"), "outm3")
        except (FileNotFoundError, SystemExit):
            ok("MUTATION 3 (cv_select.json read for shmlp) turns the shmlp read RED")
        else:
            raise SystemExit("MUTATION 3 DID NOT TURN THE READ RED")
    print(f"ZZL2READ_PASS {N}")
    if "--tmp" not in sys.argv:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as e:                                                       # noqa: BLE001
        if not isinstance(e, SystemExit) or e.code not in (0, None):
            print("ZZL2READ_FAIL", type(e).__name__, str(e)[:500])
            raise
