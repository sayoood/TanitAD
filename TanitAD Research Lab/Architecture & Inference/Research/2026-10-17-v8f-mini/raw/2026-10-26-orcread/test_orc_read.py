"""Tests of orc_read.py on SYNTHETIC inputs only (no O-RECOVER run exists). CPU, small.

    ORC_READ_PATH   the reader under test (default: ./orc_read.py; mutate_orc.py points it at a mutant)
    ORC_TREE        the export of 28e28c88 holding stack/, taniteval/ and the v8f-mini raw/ readers

Synthetic worlds: the T1 tick-file analysis (matching, cells, K1/K2/K3/K5/K6/K7) needs the real eval cache, so that one step is replaced by an in-memory backend
(`SynBackend`); EVERYTHING else is real files in the programme's formats: the trainer's argv.json / summary.json / selfcheck.json, a launch-gate-token shaped JSON,
the T1 tool's ep*.npz dumps and t1v8f json, the T0 eval's json + npz. The reader's own pairing, controls, rows and outcome run unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
ORC_PATH = os.environ.get("ORC_READ_PATH", str(HERE / "orc_read.py"))
TREE = pathlib.Path(os.environ.get("ORC_TREE", "E:/Projects/TanitAD-artifacts/orcread_build/tree"))
RAW = TREE / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "raw"
T1HD, T1K = RAW / "2026-10-26-t1hdread" / "t1hd_read.py", RAW / "2026-10-26-t1kin-hacc" / "t1k_read.py"
PREREG = HERE / "PREREG_V8F_ORECOVER.28e28c88.md"
PREREG_A2 = HERE / "PREREG_V8F_ORECOVER.A2.md"           # the landed text after Amendment 2 (pinned by sha256 at landing)
PREREG_A3 = HERE / "PREREG_V8F_ORECOVER.A3.md"           # the landed text after Amendment 3: the A2 text plus the amendment, appended
PREREG_A4 = HERE / "PREREG_V8F_ORECOVER.A4.md"           # the landed text after Amendment 4: the A3 text plus the amendment, appended
NB = 100                                                                    # TEST bootstrap size (the reader labels any B != 2000 'NOT A READ')

spec_ = importlib.util.spec_from_file_location("orc_read_under_test", ORC_PATH)
orc = importlib.util.module_from_spec(spec_)
sys.modules["orc_read_under_test"] = orc
spec_.loader.exec_module(orc)

NCLS = {"lat": 5, "lon": 7}


# ======================================================================================================================== the synthetic world
class World:
    """Per-seed arrays of ORC-0 (the base) and ORC (the base plus a controlled effect)."""

    def __init__(self, lat_gain=(0.0, 0.0), lon_gain=(0.0, 0.0), ade_shift=(0.0, 0.0), t0_shift=(0.0, 0.0), hole_orc=0.0, n_ep=18, per=20,
                 permute_orc=False, ctl_fail=(), identity=False):
        self.n_ep, self.per, self.N = n_ep, per, n_ep * per
        self.ctl_fail = set(ctl_fail)
        self.s = {}
        for s in (0, 1):
            r = np.random.default_rng(100 + s)
            N = self.N
            ci = np.repeat(np.arange(n_ep), per)
            t0 = np.tile(np.arange(per) * 5 + 10, n_ep)
            w = {"ci": ci, "t0": t0, "true": {}, "pred0": {}, "pred1": {}, "true1": {}}
            for hd, n in NCLS.items():
                true = r.integers(0, n, size=(N, 3))
                true[r.random((N, 3)) < 0.08] = -1
                pred0 = np.where(r.random((N, 3)) < 0.6, np.where(true >= 0, true, 0), r.integers(0, n, size=(N, 3)))
                gain = (lat_gain if hd == "lat" else lon_gain)[s]
                pred1 = pred0.copy()
                u = r.random((N, 3))
                if gain > 0:
                    flip = (u < gain) & (pred0 != true) & (true >= 0)
                    pred1[flip] = true[flip]
                elif gain < 0:
                    flip = (u < -gain) & (pred0 == true) & (true >= 0)
                    pred1[flip] = (true[flip] + 1) % n
                pred1[:, 0] = pred0[:, 0]
                true1 = true.copy()
                if hole_orc > 0:
                    hm = r.random((N, 3)) < hole_orc
                    hm[:, 0] = False
                    true1[hm] = -1
                w["true"][hd], w["pred0"][hd], w["pred1"][hd], w["true1"][hd] = true, pred0, pred1, true1
            g = np.cumsum(r.normal(0.5, 0.1, size=(N, 30, 2)), axis=1)
            jit = r.uniform(0.8, 2.5, size=N)
            off0 = np.stack([jit, np.zeros(N)], -1)[:, None, :]
            w["g"] = g
            w["cl0"] = g + off0
            w["cl1"] = g + off0 + np.array([ade_shift[s], 0.0])
            w["fl"] = g + np.array([3.0, 0.0])
            w["t0_0"] = r.uniform(1.0, 3.0, size=N)
            w["t0_1"] = w["t0_0"] + t0_shift[s]
            if identity:
                for hd in NCLS:
                    w["pred1"][hd], w["true1"][hd] = w["pred0"][hd], w["true"][hd]
                w["cl1"], w["t0_1"] = w["cl0"], w["t0_0"]
            w["order"] = np.arange(N)
            if permute_orc:
                w["order"] = np.concatenate([c * per + np.random.default_rng(5 + c).permutation(per) for c in range(n_ep)])
            self.s[s] = w
        self.permute_orc = permute_orc


def jdump(p, o):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1), encoding="utf-8")


def jedit(p, fn):
    o = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    fn(o)
    pathlib.Path(p).write_text(json.dumps(o, indent=1), encoding="utf-8")


def orc_argv(arm, s, root, p=None):
    p = (0.3 if arm == "ORC" else 0) if p is None else p
    law = (["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-vmax", "8.0", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", str(s)]
           if arm == "ORC" else [])                                 # A3: ORC-0 names only --ego-perturb-p 0
    return ["--arm", "M2b", "--seed", str(s), "--out", f"{root}/runs/{arm}_s{s}", "--init-ckpt", f"{root}/m2b/M2b_s{s}/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4",
            "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", str(p)] + law


class Fx:
    """A built fixture on disk: paths + the manifest dict (editable before it is saved)."""

    def __init__(self, root, world, m2b=False, logged=False, nb=None):
        self.nb = NB if nb is None else nb
        self.root, self.world = pathlib.Path(root), world
        self.m2b_sha, self.manifest = {}, {"seeds": [0, 1], "arms": {"ORC": {}, "ORC0": {}}}
        self.arms = ["ORC", "ORC0"] + (["M2b"] if m2b else [])
        if m2b:
            self.manifest["arms"]["M2b"] = {}
        for s in (0, 1):
            p = self.root / "m2b" / f"M2b_s{s}" / "ckpt_last.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"synthetic M2b checkpoint seed {s}".encode())
            self.m2b_sha[s] = hashlib.sha256(p.read_bytes()).hexdigest()
            for arm in self.arms:
                self.manifest["arms"][arm][str(s)] = self.build_run(arm, s, logged)

    def build_run(self, arm, s, logged):
        root, W = self.root, self.world.s[s]
        tag = f"{arm}_s{s}"
        e = {"ckpt_tail": f"{tag}/ckpt_last.pt"}
        if arm in ("ORC", "ORC0"):
            td = root / "runs" / tag
            argv = orc_argv(arm, s, str(root))
            sha = orc.argv_sha16(argv)
            p = 0.3 if arm == "ORC" else 0.0
            lev = {"ego_perturb_p": p, "init_ckpt": {"path": argv[argv.index("--init-ckpt") + 1], "sha256": self.m2b_sha[s]}}
            jdump(td / "argv.json", {"argv": argv, "sha": sha})
            jdump(td / "summary.json", {"done": True, "steps": 3000, "arm": "M2b", "seed": s, "n_clips": 4512, "n_windows": 99999, "train_sha256": "aa" * 32,
                                        "excluded_sha256": "bb" * 32, "argv_sha": sha, "levers": lev,
                                        "ego_perturb_n": 3010 if arm == "ORC" else 0, "ego_perturb_eligible": 10000})
            jdump(td / "selfcheck.json", {"declared_vs_built": "PASS", "argv_sha": sha, "levers": lev})
            jdump(root / "tokens" / f"PASS_{tag}.json", {"kind": "launch-gate-token", "verdict": "PASS", "hmac_sha256": "ef" * 32,
                                                          "binding": {"profile": "v8f-mini", "commit": "ab" * 20, "argv_sha256": orc.gate_argv_sha256(argv), "argv": argv,
                                                                      "data_sha256": "cd" * 32}})
            e["train_dir"], e["token"] = str(td), str(root / "tokens" / f"PASS_{tag}.json")
        e["t1"] = self.write_t1(arm, s, W, f"{tag}_causal", "causal", tag)
        if logged:
            e["t1_logged"] = self.write_t1(arm, s, W, f"{tag}_logged", "logged", tag, dumps=False)
        e["t0"] = self.write_t0(arm, s, W, tag)
        e["t0_fresh"] = self.write_t0(arm, s, W, tag, ages="fresh")
        return e

    def write_t1(self, arm, s, W, t1tag, limit, tag, dumps=True):
        root = self.root
        b = "1" if arm in ("ORC", "M2b") else "0"
        cl = W["cl1"] if arm == "ORC" else W["cl0"]
        if arm == "M2b":
            cl = W["cl0"] + 0.01
        o = W["order"] if arm == "ORC" else np.arange(len(W["ci"]))
        cl, gg, flx = cl[o], W["g"][o], W["fl"][o]
        ws, ci = W["t0"][o], W["ci"][o]
        d = root / "t1" / f"dump_{t1tag}"
        if dumps:
            d.mkdir(parents=True, exist_ok=True)
            for c in range(2):
                m = (ci * 2 // (int(ci.max()) + 1)) == c
                np.savez(d / f"ep{c:04d}.npz", g=gg[m], ws=ws[m], eid=ci[m], clip_index=ci[m], v0=np.zeros(int(m.sum())), cl=cl[m], ol=cl[m], fl=flx[m])
        K = TREE_K()
        A, B = K.comps(cl, gg)["ade6_m"], K.comps(flx, gg)["ade6_m"]
        import taniteval.ci as ci_mod
        delta = ci_mod.paired_episode_cluster_bootstrap(A, B, ci, n_boot=self.nb, seed=0)["delta"]
        hdr = {"tag": t1tag, "ckpt": f"{root}/runs/{tag}/ckpt_last.pt", "limit": limit, "replan_steps": 10, "schedule": "periodic", "scene": "logged", "nav": "deadreckon",
               "kin_mode": "yaw", "ticks_heads": True, "n_windows": int(len(ci))}
        jdump(root / "t1" / f"t1v8f_{t1tag}.json", {"t1v8f": hdr, "paired_decision_grade": {"paired_model_minus_floor_t1": {"ade_m": {"delta": delta}}}})
        return {"tag": t1tag, "ticks": str(root / "t1" / f"ticks_{t1tag}.npz"), "dump_dir": str(d), "json": str(root / "t1" / f"t1v8f_{t1tag}.json")}

    def write_t0(self, arm, s, W, tag, ages="cycle"):
        root = self.root
        ade = (W["t0_1"][W["order"]] if arm == "ORC" else W["t0_0"])
        nm = f"eval_{tag}" + ("-fresh" if ages == "fresh" else "")
        if ages == "fresh":                     # A4: a different eval. ORC's is 7 m worse, so a reader that took R-T0 from it would read a +7 m point
            ade = ade + (7.0 if arm == "ORC" else 0.0)
        jdump(root / "t0" / f"{nm}.json", {"ckpt": f"{root}/runs/{tag}/ckpt_last.pt", "ages": ages, "raw": {}})
        np.savez(root / "t0" / f"{nm}.npz", ade6=ade, eid=W["ci"], ckpt=np.zeros((len(ade), 3, 3)), lat_logits=np.zeros((len(ade), 5)), lon_logits=np.zeros((len(ade), 7)))
        return {"json": str(root / "t0" / f"{nm}.json"), "npz": str(root / "t0" / f"{nm}.npz")}

    def save(self):
        jdump(self.root / "runs.json", self.manifest)
        return str(self.root / "runs.json")


_K = {}


def TREE_K():
    if "k" not in _K:
        _K["k"] = orc.load_t1k(str(T1K), str(TREE))
    return _K["k"]


class SynBackend:
    """Stands in for RealBackend's tick-file step; the packs come from the in-memory world, the K controls PASS unless the world says otherwise."""

    def __init__(self, T, world):
        self.T, self.world = T, world

    def _pack(self, entry, label, seed, light=False):
        W = self.world.s[seed]
        tag = entry["tag"].replace("_causal", "").replace("_logged", "")
        arm = tag.rsplit("_s", 1)[0]
        orc_like = arm == "ORC"
        cells, order = {}, (W["order"] if orc_like else np.arange(self.world.N))
        for hd in NCLS:
            tr, pr = (W["true1"][hd], W["pred1"][hd]) if orc_like else (W["true"][hd], W["pred0"][hd])
            if arm == "M2b":
                pr = W["pred0"][hd]
            cells[hd] = {"true": tr[order], "pred": pr[order]}
        tick0 = {hd: float(self.T.f1_red(NCLS[hd])(self.T.E0._f1_encoded(cells[hd]["true"][:, 0], cells[hd]["pred"][:, 0], NCLS[hd]))) for hd in NCLS}
        ctl = {k: (not ({(k + ("L" if light else ""), tag)} & self.world.ctl_fail), None) for k in (("K2", "K3", "GRID") if light else ("K1", "K2", "K3", "K5", "K6", "K7", "GRID"))}
        meta = {"tag": entry["tag"], "ckpt": f"/x/runs/{tag}/ckpt_last.pt", "replan_steps": 10}
        if ("TICKMETA", tag) in self.world.ctl_fail:
            meta["ckpt"] = "/x/runs/WRONG_s0/ckpt_last.pt"
        return orc.Pack(label, W["ci"][order], W["t0"][order], cells, tick0, meta, ctl, goal={"f1_ge1": 0.5, "f1_tick0": 0.5})

    def analyze_t1(self, entry, label, seed):
        return self._pack(entry, label, seed)

    def analyze_t1_light(self, entry, label, seed):
        return self._pack(entry, label, seed, light=True)

    def k4(self, entry_logged, pack, t0_entry, seed):
        tag = pack.meta["tag"].replace("_causal", "")
        fresh = str(t0_entry["json"]).endswith("-fresh.json")          # A4: the synthetic K4 holds only against the arm's fresh-age eval
        return (fresh and not ({("K4", tag)} & self.world.ctl_fail)), {"source": "synthetic", "t0_json": str(t0_entry["json"])}


def read_args(fx, out, **kw):
    a = ["--manifest", fx.save(), "--out", str(out), "--tool-root", str(TREE), "--t1hd-read", str(T1HD), "--t1k-read", str(T1K), "--eval-cache", "x", "--tracks", "x",
         "--eval-grid", "x", "--eval-join", "x", "--n-boot", str(kw.pop("n_boot", NB)), "--n-windows", str(fx.world.N), "--n-episodes", str(fx.world.n_ep),
         "--m2b-ckpt", f"0={fx.root}/m2b/M2b_s0/ckpt_last.pt", "--m2b-ckpt", f"1={fx.root}/m2b/M2b_s1/ckpt_last.pt",
         "--banked-sha", f"0={fx.m2b_sha[0]}", "--banked-sha", f"1={fx.m2b_sha[1]}", "--prereg", str(PREREG), "--work-dir", str(fx.root / "work")]
    for k, v in kw.items():
        a += [k, str(v)]
    return a


def run(tmp_path, world=None, m2b=False, logged=False, edit=None, extra=None, name="ORC_READ.json", **wk):
    world = world or World(**wk)
    fx = Fx(tmp_path, world, m2b=m2b, logged=logged)
    if edit:
        edit(fx)
    out = tmp_path / name
    ns = {}

    def factory(T, a):
        ns["T"] = T
        return SynBackend(T, world)
    rc = orc.main(read_args(fx, out, **(extra or {})), backend_factory=factory)
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return rc, res, out, fx


# strong effects (gain 0.5: ORC gets half of ORC-0's wrong cells right) -- on both seeds, both heads
STRONG = dict(lat_gain=(0.5, 0.5), lon_gain=(0.5, 0.5))


def refused(capsys, rc, out, *needles):
    err = capsys.readouterr().err
    assert rc == 2, err
    assert not out.exists(), "a REFUSED read wrote a file"
    assert "REFUSED" in err
    for n in needles:
        assert n in err, (n, err)
    return err


# ======================================================================================================================== 1. the outcome table is the PREREG's
def prereg_table():
    t = PREREG.read_text(encoding="utf-8").split("## 4. Outcomes")[1].split("## 5. Queue")[0]
    rows = [[c.strip().replace("**", "") for c in ln.strip().strip("|").split(" | ")] for ln in t.splitlines() if ln.startswith("| **")]
    return [tuple(r) for r in rows]


def test_table_verbatim_and_in_order():
    assert tuple(orc.OUTCOME_TABLE) == tuple(prereg_table())              # names, conditions and committed-next texts, in the PREREG's row order
    assert orc.OUTCOME_NAMES == ("ROBUST", "PARTIAL", "NO EFFECT", "HARMS", "REFUSED")


def test_prereg_file_is_the_28e28c88_blob():
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == "c1925795eda3247690cf44a01ac0b0d18680e1ca967db6d693a6a459a8f1e6ca"


def test_committed_numbers_are_the_preregs():
    txt = PREREG.read_text(encoding="utf-8")
    assert (orc.TWO_F["lat"], orc.TWO_F["lon"]) == (0.0248, 0.0146) and "lat 0.0124, lon 0.0073, so 2F = 0.0248 / 0.0146" in txt
    assert orc.HALF_F == 0.0604 and "ORC − ORC-0 ≤ +F/2 = 0.0604 m" in txt and orc.F_ADE == 0.1207
    assert (orc.STEPS, orc.LR, orc.WARMUP, orc.PGT_START, orc.PGT_END) == (3000, 1e-4, 100, 0.2, 0.2)
    assert orc.PERT_P == {"ORC": 0.3, "ORC0": 0.0} and (orc.RATIO_CENTRE, orc.RATIO_TOL) == (0.3, 0.02)
    assert orc.BANKED_M2B_SHA[0].startswith("601cb435") and orc.BANKED_M2B_SHA[1].startswith("6104fb15")


# ======================================================================================================================== 2. the outcome function (decide), row by row
def row(delta, sep=None):
    sep = (delta != 0) if sep is None else sep
    return {"delta": delta, "separated": bool(sep), "lo": 0.0, "hi": 0.0}


def rows_of(lat=(0.1, 0.1), lon=(0.1, 0.1), ade=(0.0, 0.0), t0=(0.0, 0.0), lat_sep=(True, True), lon_sep=(True, True), ade_sep=(False, False)):
    return {s: {"R-TLAT": row(lat[s], lat_sep[s]), "R-TLON": row(lon[s], lon_sep[s]), "R-ADE": row(ade[s], ade_sep[s]), "R-T0": {"point": t0[s]}} for s in (0, 1)}


def test_decide_robust_partial_noeffect_harms():
    assert orc.decide(rows_of())["OUTCOME"] == "ROBUST"
    assert orc.decide(rows_of(lon=(0.0, 0.0), lon_sep=(False, False)))["OUTCOME"] == "PARTIAL"
    assert orc.decide(rows_of(lat=(0.0, 0.0), lat_sep=(False, False)))["OUTCOME"] == "PARTIAL"
    assert orc.decide(rows_of(lat=(0.0, 0.0), lon=(0.0, 0.0), lat_sep=(False, False), lon_sep=(False, False)))["OUTCOME"] == "NO EFFECT"
    assert orc.decide(rows_of(ade=(0.2, 0.0), ade_sep=(True, False)))["OUTCOME"] == "HARMS"
    assert orc.decide(rows_of(t0=(0.0, 0.0605)))["OUTCOME"] == "HARMS"


def test_decide_bars_are_per_head_and_inclusive():
    f = orc.decide
    # delta exactly at 2F passes; one ulp below does not; lat and lon bars are NOT interchangeable (0.02 clears lon's 0.0146, not lat's 0.0248)
    d = f(rows_of(lat=(0.0248, 0.0248), lon=(0.0146, 0.0146)))
    assert d["OUTCOME"] == "ROBUST" and d["flags"]["head_moved_at_both_seeds"] == {"lat": True, "lon": True}
    d = f(rows_of(lat=(0.0247, 0.0247), lon=(0.0146, 0.0146)))
    assert d["flags"]["head_moved_at_both_seeds"] == {"lat": False, "lon": True} and d["OUTCOME"] == "PARTIAL"
    d = f(rows_of(lat=(0.02, 0.02), lon=(0.02, 0.02)))
    assert d["flags"]["head_moved_at_both_seeds"] == {"lat": False, "lon": True} and d["OUTCOME"] == "PARTIAL"
    d = f(rows_of(lat=(0.0248, 0.0248), lon=(0.0145, 0.0145)))
    assert d["flags"]["head_moved_at_both_seeds"] == {"lat": True, "lon": False} and d["OUTCOME"] == "PARTIAL"


def test_decide_both_seeds_and_separation_are_needed():
    f = orc.decide
    assert f(rows_of(lat=(0.1, 0.1), lat_sep=(True, False), lon=(0.0, 0.0), lon_sep=(False, False)))["OUTCOME"] == "NO EFFECT"      # s1 not separated: lat has not moved
    assert f(rows_of(lat=(0.1, -0.1), lat_sep=(True, True), lon=(0.0, 0.0), lon_sep=(False, False)))["OUTCOME"] == "NO EFFECT"     # s1 separated WORSE is not better
    assert f(rows_of(lat=(0.1, 0.1), lon=(0.1, 0.0), lon_sep=(True, False)))["OUTCOME"] == "PARTIAL"


def test_decide_harms_conditions_and_guard_boundaries():
    f = orc.decide
    assert f(rows_of(t0=(0.0604, 0.0604)))["OUTCOME"] == "ROBUST"                                  # +F/2 exactly is 'within'
    assert f(rows_of(t0=(0.0, 0.06041)))["OUTCOME"] == "HARMS"
    assert f(rows_of(ade=(0.5, 0.5), ade_sep=(False, False)))["OUTCOME"] == "ROBUST"               # worse but NOT separated: the guard holds
    assert f(rows_of(ade=(-0.5, -0.5), ade_sep=(True, True)))["OUTCOME"] == "ROBUST"               # separated BETTER is not worse
    assert f(rows_of(ade=(0.0, 0.3), ade_sep=(False, True)))["OUTCOME"] == "HARMS"                 # a single seed is enough
    assert f(rows_of(lat=(0.0, 0.0), lon=(0.0, 0.0), lat_sep=(False, False), lon_sep=(False, False), t0=(0.07, 0.0)))["OUTCOME"] == "HARMS"
    assert f(rows_of(ade=(0.3, 0.3), ade_sep=(True, True), t0=(0.07, 0.07)))["flags"]["guards_held"] is False


def test_decide_former_table_hole_is_NO_EFFECT_under_A2_and_flagged_sub_bar():
    # lat separated-better at both seeds, but below 2F at s1: the as-written table matched no row; A2-2026-10-26 makes it NO EFFECT, flagged sub-bar
    d = orc.decide(rows_of(lat=(0.1, 0.01), lon=(0.0, 0.0), lon_sep=(False, False)))
    assert d["OUTCOME"] == "NO EFFECT" and d["matching_rows_in_table_order"] == ["NO EFFECT"]
    assert d["flags"]["sub_bar_separated_better_at_both_seeds"] == {"lat": True, "lon": False}
    b = orc.outcome_block(d)
    assert b["condition"] == orc.A2_NO_EFFECT and b["condition_amended_by"] == "A2-2026-10-26" and b["committed_next"] == prereg_table()[2][2]


def test_a2_no_effect_text_is_the_amendments_verbatim():
    txt = PREREG_A2.read_text(encoding="utf-8")
    assert txt.startswith(PREREG.read_text(encoding="utf-8")[:2000])                 # the amended file extends the registered one
    assert orc.A2_NO_EFFECT in txt.split("## Amendment 2")[1] and orc.A2_NO_EFFECT not in txt.split("## Amendment 2")[0]


def test_a3_text_extends_a2_and_carries_the_pair_rule():
    a2, a3 = PREREG_A2.read_bytes(), PREREG_A3.read_bytes()
    assert a3.startswith(a2) and b"## Amendment 3" in a3[len(a2):] and b"## Amendment 3" not in a2
    t = a3[len(a2):].decode("utf-8")
    assert "ORC-0 names only `--ego-perturb-p 0`" in t and "identical except `--out`, `--ego-perturb-p`, and the ORC-only law flags" in t.replace("\n", " ")
    for name, w in orc.LAW.items():                                  # the reader's law is the amendment's, value for value
        assert f"`{name} {' '.join(str(x) for x in w)}`" in t, name
    assert "Amendment 3 (A3-2026-10-26)" in orc.PREREG_ID


def test_a4_text_extends_a3_and_carries_the_k4_rule():
    a3, a4 = PREREG_A3.read_bytes(), PREREG_A4.read_bytes()
    assert a4.startswith(a3) and b"## Amendment 4" in a4[len(a3):] and b"## Amendment 4" not in a3
    t = a4[len(a3):].decode("utf-8").replace("\n", " ")
    assert "`v8f_mini_eval.py run --ages fresh`" in t and "R-T0, T0-CKPT and the four families stay on box 2's standard eval" in t
    assert "separate manifest keys" in t and "Amendment 4 (A4-2026-10-26)" in orc.PREREG_ID


def test_a4_k4_reads_t0_fresh_and_r_t0_reads_the_standard_eval(tmp_path, capsys):
    """A4: K4's comparator is the fresh-age eval, from its own key; R-T0 stays on box 2's standard eval. The fixture's fresh eval is 7 m worse for ORC, so a
    reader that took R-T0 from it would read a +7 m point; the synthetic K4 fails unless it is handed the -fresh file."""
    rc, res, *_ = run(tmp_path, **STRONG)
    assert rc == 0, capsys.readouterr().err
    for tag in ("ORC_s0", "ORC0_s0", "ORC_s1", "ORC0_s1"):
        assert all(res["controls_PASS"][f"{k}:{tag}"] for k in ("K4", "T0-AGES", "T0F-CKPT")), tag
        assert res["controls"][f"K4:{tag}"]["detail"]["t0_json"].endswith(f"eval_{tag}-fresh.json")
    for s in ("s0", "s1"):
        assert abs(res["rows"][s]["R-T0"]["point"]) < 1.0, res["rows"][s]["R-T0"]
    assert res["OUTCOME"]["OUTCOME"] == "ROBUST"


def test_a4_refuses_a_manifest_without_t0_fresh(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, edit=lambda fx: fx.manifest["arms"]["ORC0"]["1"].pop("t0_fresh"), **STRONG)
    refused(capsys, rc, out, "ORC0_s1: manifest has no 't0_fresh'")


def test_a4_t0_ages_and_the_fresh_ckpt_are_controls(tmp_path, capsys):
    edits = chain(
        lambda fx: jedit(fx.root / "t0" / "eval_ORC_s0.json", lambda o: o.update(ages="fresh")),          # R-T0 would read a fresh-age eval
        lambda fx: jedit(fx.root / "t0" / "eval_ORC0_s1-fresh.json", lambda o: o.update(ages="cycle")),   # K4 would read a cycle-age eval
        lambda fx: jedit(fx.root / "t0" / "eval_ORC_s1-fresh.json", lambda o: o.update(ckpt="/x/runs/ORC0_s1/ckpt_last.pt")),
    )
    rc, _, out, _ = run(tmp_path, edit=edits, **STRONG)
    refused(capsys, rc, out, "T0-AGES:ORC_s0", "T0-AGES:ORC0_s1", "T0F-CKPT:ORC_s1",
            absent=("T0-AGES:ORC0_s0", "T0-AGES:ORC_s1", "T0F-CKPT:ORC_s0", "T0F-CKPT:ORC0_s1", "T0-CKPT:ORC_s1"))


def test_a2_table_is_complete_no_reachable_UNPLACED():
    # guards held: 0 / 1 / 2 heads moved -> NO EFFECT / PARTIAL / ROBUST; a guard failed -> HARMS. Every sign / separation / bar pattern places.
    import itertools
    for lat, lon, ls, ns, ade, t0 in itertools.product((0.0, 0.01, 0.1), (0.0, 0.01, 0.1), (False, True), (False, True), (0.0, 0.3), (0.0, 0.07)):
        d = orc.decide(rows_of(lat=(lat, 0.1), lon=(lon, 0.1), lat_sep=(ls, True), lon_sep=(ns, True), ade=(ade, 0.0), ade_sep=(ade > 0, False), t0=(t0, 0.0)))
        assert d["OUTCOME"] != "UNPLACED", (lat, lon, ls, ns, ade, t0)


def test_decide_lists_every_matching_row_in_table_order():
    # the four predicates are literally mutually exclusive (the guards sit inside ROBUST / PARTIAL / NO EFFECT), so a matching list has exactly one entry;
    # the reader still lists them all, and takes the FIRST in table order
    for r in (rows_of(), rows_of(ade=(0.5, 0.5), ade_sep=(True, True)), rows_of(lat=(0.0, 0.0), lon=(0.0, 0.0), lat_sep=(False, False), lon_sep=(False, False))):
        d = orc.decide(r)
        assert d["matching_rows_in_table_order"] == [d["OUTCOME"]]
        assert [n for n in orc.OUTCOME_NAMES if n in d["matching_rows_in_table_order"]] == d["matching_rows_in_table_order"]


def test_decide_needs_both_seeds():
    r = rows_of()
    del r[1]
    with pytest.raises(orc.Refusal):
        orc.decide(r)


def test_outcome_block_carries_the_committed_text_verbatim():
    for nm, cond, nxt in prereg_table()[:4]:
        d = {"OUTCOME": nm, "matching_rows_in_table_order": [nm], "flags": {}}
        b = orc.outcome_block(d)
        if nm == "NO EFFECT":                                       # A2-2026-10-26: the amended condition is in force, the as-written one rides beside it
            assert b["committed_next"] == nxt and b["condition"] == orc.A2_NO_EFFECT and b["condition_as_written"] == cond
        else:
            assert b["committed_next"] == nxt and b["condition"] == cond


# ======================================================================================================================== 3. helpers
def test_parse_flags_keeps_every_value():
    f = orc.parse_flags(["--a", "1", "--ego-perturb-vscale", "0.5", "1.0", "--flag", "--b", "x", "--a", "2"])
    assert f["--ego-perturb-vscale"] == [["0.5", "1.0"]] and f["--flag"] == [[]] and f["--a"] == [["1"], ["2"]]
    assert orc.parse_flags(["--v", "0.5", "1.0"]) != orc.parse_flags(["--v", "0.5", "9.9"])


def test_find_key_looks_in_levers_and_totals():
    src = [("summary", {"levers": {"a": 1}, "totals": {"b": 2}, "c": {"d": 3}}), ("x", None)]
    assert orc.find_key(src, "a") == [("summary:levers.a", 1)] and orc.find_key(src, "b") == [("summary:totals.b", 2)] and orc.find_key(src, "c.d") == [("summary:c.d", 3)]
    assert orc.find_key(src, "zz") == []


def test_gate_argv_hash_is_the_launch_gates():
    assert orc.gate_argv_sha256(["--a", "1"]) == hashlib.sha256(b'["--a","1"]').hexdigest()


# ======================================================================================================================== 4. synthetic worlds that must read each outcome
def test_world_ROBUST(tmp_path, capsys):
    rc, res, out, fx = run(tmp_path, **STRONG)
    assert rc == 0, capsys.readouterr().err
    assert res["OUTCOME"]["OUTCOME"] == "ROBUST" and res["OUTCOME"]["committed_next"] == prereg_table()[0][2]
    for s in ("s0", "s1"):
        for nm in ("R-TLAT", "R-TLON"):
            r = res["rows"][s][nm]
            assert r["separated"] and r["delta"] >= r["two_F"] and r["better"] and r["matched"] and r["lower_is_better"] is False
        assert res["rows"][s]["R-ADE"]["separated"] is False and res["rows"][s]["R-T0"]["point"] == pytest.approx(0.0, abs=1e-12)
    assert res["LABEL"].startswith("TEST") and all(res["controls_PASS"].values()) and res["strategic_family"].startswith("NOT PAUSED") and "2026-10-17" in res["strategic_family"]
    assert res["OUTCOME"]["matching_rows_in_table_order"] == ["ROBUST"]


def test_world_PARTIAL_lat_only_and_lon_only(tmp_path, capsys):
    rc, res, *_ = run(tmp_path / "a", lat_gain=(0.5, 0.5))
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "PARTIAL"
    assert res["OUTCOME"]["flags"]["head_moved_at_both_seeds"] == {"lat": True, "lon": False} and res["OUTCOME"]["committed_next"] == prereg_table()[1][2]
    rc, res, *_ = run(tmp_path / "b", lon_gain=(0.5, 0.5))
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "PARTIAL" and res["OUTCOME"]["flags"]["head_moved_at_both_seeds"] == {"lat": False, "lon": True}


def test_world_PARTIAL_needs_both_seeds(tmp_path):
    rc, res, *_ = run(tmp_path, lat_gain=(0.5, 0.0))                    # lat moves at s0 only
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "NO EFFECT" and res["rows"]["s1"]["R-TLAT"]["delta"] == 0.0


def test_world_NO_EFFECT(tmp_path):
    rc, res, *_ = run(tmp_path)
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "NO EFFECT" and res["OUTCOME"]["committed_next"] == prereg_table()[2][2]


def test_world_HARMS_via_ADE(tmp_path):
    rc, res, *_ = run(tmp_path, ade_shift=(0.0, 0.3), **STRONG)
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "HARMS" and res["OUTCOME"]["flags"]["R-ADE_separated_worse_per_seed"] == {"s0": False, "s1": True}
    assert res["rows"]["s1"]["R-ADE"]["separated"] and res["rows"]["s1"]["R-ADE"]["delta"] == pytest.approx(0.3, abs=1e-3)
    assert res["OUTCOME"]["committed_next"] == prereg_table()[3][2]


def test_world_HARMS_via_T0(tmp_path):
    rc, res, *_ = run(tmp_path, t0_shift=(0.1, 0.0), **STRONG)
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "HARMS" and res["OUTCOME"]["flags"]["R-T0_above_F_over_2_per_seed"] == {"s0": True, "s1": False}
    assert res["rows"]["s0"]["R-T0"]["point"] == pytest.approx(0.1, abs=1e-9)


def test_world_T0_inside_the_guard_and_ADE_better_stay_ROBUST(tmp_path):
    rc, res, *_ = run(tmp_path, t0_shift=(0.05, 0.05), ade_shift=(-0.3, -0.3), **STRONG)
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "ROBUST"
    assert res["rows"]["s0"]["R-ADE"]["delta"] < 0 and res["rows"]["s0"]["R-ADE"]["separated"]


def test_world_former_hole_reads_NO_EFFECT_under_A2(tmp_path):
    rc, res, out, _ = run(tmp_path, lat_gain=(0.5, 0.03))
    assert 0 < res["rows"]["s1"]["R-TLAT"]["delta"] < orc.TWO_F["lat"] and res["rows"]["s1"]["R-TLAT"]["separated"]
    assert rc == 0 and out.exists() and res["OUTCOME"]["OUTCOME"] == "NO EFFECT" and res["OUTCOME"]["condition"] == orc.A2_NO_EFFECT


def test_world_identity_ORC_equals_ORC0(tmp_path):
    rc, res, *_ = run(tmp_path, identity=True, lat_gain=(0.5, 0.5))
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "NO EFFECT"
    for s in ("s0", "s1"):
        for nm, r in res["rows"][s].items():
            assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0 and not r["separated"], (s, nm, r)
        assert res["rows"][s]["R-T0"]["point"] == 0.0 and res["rows"][s]["R-ADE"]["point"] == 0.0


def test_matched_cells_only(tmp_path):
    rc, res, *_ = run(tmp_path, hole_orc=0.25, **STRONG)
    assert rc == 0
    r, ro = res["rows"]["s0"]["R-TLAT"], res["rows"]["s0"]["R-TLAT_own_cells_beside"]
    assert r["matched"] and not ro["matched"]
    assert r["n_cells_a"] == r["n_cells_b"] == r["n_cells_matched_pairs"] and ro["n_cells_a"] < ro["n_cells_b"]       # ORC lost cells the matched read drops from BOTH sides


def test_ticks_ge1_only_tick0_never_enters(tmp_path):
    w = World(**STRONG)
    for s in (0, 1):                                                 # break ORC at tick 0 only: the ticks>=1 rows must not see it
        for hd in NCLS:
            w.s[s]["pred1"][hd][:, 0] = (w.s[s]["pred0"][hd][:, 0] + 1) % NCLS[hd]
    rc, res, *_ = run(tmp_path, world=w)
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "ROBUST"
    t0 = res["beside"]["per_seed"]["s0"]["tick0_f1"]
    assert t0["ORC"]["lat"] < t0["ORC0"]["lat"] - 0.3                # ... and the beside tick-0 row is where it shows
    n_ge1 = int(((w.s[0]["true"]["lat"][:, 1:] >= 0)).sum())
    assert res["rows"]["s0"]["R-TLAT_own_cells_beside"]["n_cells_b"] == n_ge1


def test_F1_sign_ORC_minus_ORC0(tmp_path):
    rc, res, *_ = run(tmp_path, lat_gain=(-0.4, -0.4), lon_gain=(0.0, 0.0))        # ORC is WORSE on lat
    assert rc == 0
    r = res["rows"]["s0"]["R-TLAT"]
    assert r["delta"] < 0 and r["separated"] and r["worse"] and not r["better"] and r["delta"] == pytest.approx(r["f1_a"] - r["f1_b"], abs=1e-4)
    assert res["OUTCOME"]["OUTCOME"] == "NO EFFECT"


def test_beside_M2b_rows_and_K4_logged_source(tmp_path):
    rc, res, *_ = run(tmp_path, m2b=True, logged=True, **STRONG)
    assert rc == 0
    b = res["beside"]["per_seed"]["s0"]
    assert set(b["ORC_minus_M2b"]) >= {"T-LAT-F1", "T-LON-F1", "T1-kin ADE6", "T0 ADE6", "tick0_f1"} and b["ORC_minus_M2b"]["T-LAT-F1"]["delta"] > 0
    assert "K4:ORC_s0" in res["controls_PASS"] and "K2-logged:ORC_s0" in res["controls_PASS"] and "K4:M2b_s0" in res["controls_PASS"]


def test_end_to_end_default_B_2000_and_sha_printed(tmp_path, capsys):
    w = World(**STRONG)
    fx = Fx(tmp_path, w, nb=2000)                                    # the t1 jsons' ADE delta at B = 2000, so R2 compares like with like
    out = tmp_path / "ORC_READ.json"

    def factory(T, a):
        return SynBackend(T, w)
    rc = orc.main(read_args(fx, out, n_boot=2000), backend_factory=factory)
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    res = json.loads(out.read_text(encoding="utf-8"))
    assert "LABEL" not in res and res["n_boot"] == 2000 and res["rows"]["s0"]["R-TLAT"]["n_boot"] == 2000
    assert f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}" in cap.out and res["OUTCOME"]["OUTCOME"] == "ROBUST"


# ======================================================================================================================== 5. unit tests of each control (function level: boundaries)
def fake_rec(argv, summary=None, selfcheck=None):
    return {"argv": argv, "flags": orc.parse_flags(argv), "summary.json": summary or {}, "selfcheck.json": selfcheck or {}, "argv.json": {"sha": orc.argv_sha16(argv)},
            "sources": [("summary", summary or {}), ("selfcheck", selfcheck or {}), ("metrics", None)]}


def test_g_pert_ratio_band_is_inclusive_and_exact():
    for n, ok in ((2800, True), (3000, True), (3200, True), (2799, False), (3201, False), (0, False)):
        r = fake_rec([], {"ego_perturb_n": n, "ego_perturb_eligible": 10000, "levers": {"ego_perturb_p": 0.3}})
        assert orc.g_pert("t", "ORC", r, "ego_perturb_n", "ego_perturb_eligible", "ego_perturb_p")[0] is ok, n


def test_g_pert_orc0_exactly_zero_and_keys_present():
    f = lambda n, extra=None: orc.g_pert("t", "ORC0", fake_rec([], {"ego_perturb_n": n, "levers": {"ego_perturb_p": 0.0}, **(extra or {})}), "ego_perturb_n", "ego_perturb_eligible", "ego_perturb_p")[0]
    assert f(0) is True and f(1) is False
    assert orc.g_pert("t", "ORC0", fake_rec([], {"levers": {"ego_perturb_p": 0.0}}), "ego_perturb_n", "x", "ego_perturb_p")[0] is False       # n not recorded
    assert orc.g_pert("t", "ORC0", fake_rec([], {"ego_perturb_n": 0}), "ego_perturb_n", "x", "ego_perturb_p")[0] is False                       # built p not recorded
    assert orc.g_pert("t", "ORC", fake_rec([], {"ego_perturb_n": 3000, "levers": {"ego_perturb_p": 0.3}}), "ego_perturb_n", "ego_perturb_eligible", "ego_perturb_p")[0] is False


def test_g_pert_conflicting_records_fail():
    r = fake_rec([], {"ego_perturb_n": 3000, "ego_perturb_eligible": 10000, "levers": {"ego_perturb_p": 0.3}}, {"totals": {"ego_perturb_n": 3001}})
    ok, det = orc.g_pert("t", "ORC", r, "ego_perturb_n", "ego_perturb_eligible", "ego_perturb_p")
    assert not ok and "conflicting" in str(det)


GOOD0 = ["--arm", "M2b", "--seed", "0", "--out", "/o", "--init-ckpt", "/x/M2b_s0/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2",
         "--p-gt-end", "0.2", "--ego-perturb-p", "0"]
LAW0 = ["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", "0"]
GOOD = [x if x != "0" or i != GOOD0.index("--ego-perturb-p") + 1 else "0.3" for i, x in enumerate(GOOD0)] + LAW0


def test_g_argv_each_registered_flag():
    assert orc.g_argv("t", "ORC", 0, fake_rec(GOOD))[0] is True
    bad = {"--steps": "2000", "--lr": "3e-4", "--warmup": "200", "--p-gt-start": "1.0", "--p-gt-end": "0.1", "--ego-perturb-p": "0.1", "--seed": "1", "--arm": "M1"}
    for flag, v in bad.items():
        a = list(GOOD)
        a[a.index(flag) + 1] = v
        ok, det = orc.g_argv("t", "ORC", 0, fake_rec(a))
        assert not ok and any(flag in d for d in det), (flag, det)
    for flag in ("--steps", "--lr", "--warmup", "--p-gt-start", "--p-gt-end", "--ego-perturb-p", "--init-ckpt"):    # not NAMED: a default is not a test of the lever
        a = list(GOOD)
        i = a.index(flag)
        del a[i:i + 2]
        assert not orc.g_argv("t", "ORC", 0, fake_rec(a))[0], flag
    assert not orc.g_argv("t", "ORC", 0, fake_rec(GOOD + ["--steps", "3000"]))[0]                                      # named twice
    assert not orc.g_argv("t", "ORC", 0, fake_rec([x.replace("M2b_s0", "M2b_s1") for x in GOOD]))[0]                  # the other seed's checkpoint
    assert orc.g_argv("t", "ORC0", 0, fake_rec(GOOD0))[0] is True and not orc.g_argv("t", "ORC", 0, fake_rec(GOOD0))[0]


def test_g_argv_a3_law_on_orc_only():
    for i in range(0, len(LAW0), 1):                                # ORC-0 naming ANY law flag (or the law seed) is refused
        if LAW0[i].startswith("--"):
            j = i + 1
            while j < len(LAW0) and not LAW0[j].startswith("--"):
                j += 1
            ok, det = orc.g_argv("t", "ORC0", 0, fake_rec(GOOD0 + LAW0[i:j]))
            assert not ok and any(LAW0[i] in d for d in det), (LAW0[i], det)
            a = GOOD[:len(GOOD) - len(LAW0)] + LAW0[:i] + LAW0[j:]  # ORC missing that law flag is refused
            ok, det = orc.g_argv("t", "ORC", 0, fake_rec(a))
            assert not ok and any(LAW0[i] in d for d in det), (LAW0[i], det)
    for flag, vals in (("--ego-perturb-vscale", ["0.5", "0.9"]), ("--ego-perturb-vmax", ["9.0"]), ("--ego-perturb-adrop", ["1.0"]), ("--ego-perturb-yawrate-std", ["0"]),
                       ("--ego-perturb-seed", ["1"])):
        a = list(GOOD)
        k = a.index(flag)
        a[k + 1:k + 1 + len(vals)] = vals
        ok, det = orc.g_argv("t", "ORC", 0, fake_rec(a))
        assert not ok and any(flag in d for d in det), (flag, det)
    r_orc, r_0 = fake_rec(GOOD), fake_rec(GOOD0)                    # the pair: identical once --out, p and the ORC-only law are set aside
    assert {k: v for k, v in r_orc["flags"].items() if k not in {"--out", "--ego-perturb-p", orc.LAW_SEED} | set(orc.LAW)} == \
           {k: v for k, v in r_0["flags"].items() if k not in {"--out", "--ego-perturb-p"}}


def test_g_token_binding():
    argv = list(GOOD)
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "hmac_sha256": "ab", "binding": {"commit": "a" * 40, "argv_sha256": orc.gate_argv_sha256(argv), "data_sha256": "c" * 64}}
    r = fake_rec(argv)
    assert orc.g_token("t", r, tok)[0] is True
    for edit in (lambda b: b.update(argv_sha256="0" * 64), lambda b: b.pop("commit"), lambda b: b.update(commit="xyz"), lambda b: b.pop("data_sha256")):
        t2 = json.loads(json.dumps(tok))
        edit(t2["binding"])
        assert not orc.g_token("t", r, t2)[0]
    for k, v in (("verdict", "FAIL"), ("verdict", "INCOMPLETE"), ("kind", "evidence")):
        assert not orc.g_token("t", r, {**tok, k: v})[0]
    assert orc.g_token("t", fake_rec(["python", "train.py"] + argv), {**tok, "binding": {**tok["binding"], "argv_sha256": orc.gate_argv_sha256(argv)}})[0]     # a leading script path is stripped


def test_g_summary():
    sm = {"done": True, "steps": 3000, "arm": "M2b", "seed": 0}
    sha = orc.argv_sha16(GOOD)
    r = fake_rec(GOOD, {**sm, "argv_sha": sha}, {"declared_vs_built": "PASS", "argv_sha": sha})
    assert orc.g_summary("t", 0, r)[0] is True
    for k, v in (("done", False), ("steps", 2999), ("arm", "M1"), ("seed", 1), ("argv_sha", "0" * 16)):
        r2 = fake_rec(GOOD, {**sm, "argv_sha": sha, k: v}, {"declared_vs_built": "PASS", "argv_sha": sha})
        assert not orc.g_summary("t", 0, r2)[0], k
    assert not orc.g_summary("t", 0, fake_rec(GOOD, {**sm, "argv_sha": sha}, {"declared_vs_built": "FAIL", "argv_sha": sha}))[0]


def test_g_init_sha_sources():
    sha = "ab" * 32
    r = fake_rec(GOOD, {"levers": {"init_ckpt": {"sha256": sha}}})
    assert orc.g_init("t", 0, r, sha, "init_ckpt.sha256")[0] is True
    assert not orc.g_init("t", 0, r, "cd" * 32, "init_ckpt.sha256")[0]
    assert not orc.g_init("t", 0, fake_rec(GOOD), sha, "init_ckpt.sha256")[0]                                          # nothing recorded, no file: unproven
    assert orc.g_init("t", 0, fake_rec(GOOD, {"init_sha": sha}), sha, "init_sha")[0] is True                            # the key name is overridable


HDR = {"t1v8f": {"tag": "T", "ckpt": "/a/ORC_s0/ckpt_last.pt", "limit": "causal", "replan_steps": 10, "schedule": "periodic", "scene": "logged", "nav": "deadreckon", "ticks_heads": True}}


def test_g_t1cfg_each_field():
    meta = {"tag": "T", "ckpt": "/b/ORC_s0/ckpt_last.pt"}
    assert orc.g_t1cfg("t", HDR, "ORC_s0/ckpt_last.pt", "causal", meta)[0] is True
    for k, v in (("limit", "logged"), ("replan_steps", 5), ("schedule", "x"), ("scene", "frozen"), ("nav", "logged"), ("ticks_heads", False), ("ckpt", "/a/ORC0_s0/ckpt_last.pt"), ("tag", "U")):
        h = json.loads(json.dumps(HDR))
        h["t1v8f"][k] = v
        assert not orc.g_t1cfg("t", h, "ORC_s0/ckpt_last.pt", "causal", meta)[0], k
    assert not orc.g_t1cfg("t", HDR, "ORC_s0/ckpt_last.pt", "causal", {**meta, "ckpt": "/b/ORC0_s0/ckpt_last.pt"})[0]
    assert orc.g_t1cfg("t", {"t1v8f": {**HDR["t1v8f"], "limit": "logged"}}, "ORC_s0/ckpt_last.pt", "logged", meta)[0] is True


def test_k1_own_rows_identity_control_fires():
    class FakeT:
        @staticmethod
        def PB(a, b, eid, n_boot, seed, reduce):
            return {"delta": 1e-4, "lo": 0.0, "hi": 0.0}
    ok, det = orc.k1_own(FakeT, [("x", np.arange(4.0), np.arange(4), "mean")], 10)
    assert not ok and det[0]["delta"] == 1e-4


# ======================================================================================================================== 6. a refusal per control, THROUGH the pipeline (rc 2, NOTHING written)
def edit_argv(arm, s, fn):
    def e(fx):
        td = fx.root / "runs" / f"{arm}_s{s}"

        def f(o):
            fn(o["argv"])
            o["sha"] = orc.argv_sha16(o["argv"])
        jedit(td / "argv.json", f)
        a = json.loads((td / "argv.json").read_text())
        for nm in ("summary.json", "selfcheck.json"):
            jedit(td / nm, lambda o: o.update(argv_sha=a["sha"]))
        jdump(fx.root / "tokens" / f"PASS_{arm}_s{s}.json", {"kind": "launch-gate-token", "verdict": "PASS", "hmac_sha256": "ef" * 32,
                                                              "binding": {"commit": "ab" * 20, "argv_sha256": orc.gate_argv_sha256(a["argv"]), "argv": a["argv"], "data_sha256": "cd" * 32}})
    return e


def chain(*edits):
    def e(fx):
        for f in edits:
            f(fx)
    return e


def set_flag(flag, v):
    return lambda a: a.__setitem__(a.index(flag) + 1, v)


def drop_flag(flag, n=2):
    def f(a):
        i = a.index(flag)
        del a[i:i + n]
    return f


def refused(capsys, rc, out, *needles, absent=()):
    err = capsys.readouterr().err
    assert rc == 2, err
    assert not out.exists(), "a REFUSED read wrote a file"
    assert "REFUSED" in err
    for n in needles:
        assert n in err, (n, err)
    for n in absent:
        assert n not in err, (n, err)
    return err


def test_refuse_every_control_is_wired_into_the_read(tmp_path, capsys):
    """One world with a defect in EVERY control family, spread over the four runs: the reader must name each one (and none of the healthy ones)."""
    fails = {("K1", "ORC_s0"), ("K2", "ORC0_s0"), ("K3", "ORC_s1"), ("K5", "ORC0_s1"), ("K6", "ORC_s0"), ("K7", "ORC0_s0"), ("GRID", "ORC_s1"), ("K4", "ORC0_s1"),
             ("TICKMETA", "ORC0_s0")}
    w = World(ctl_fail=fails, **STRONG)
    edits = chain(
        edit_argv("ORC", 0, chain_f(drop_flag("--ego-perturb-p"), set_flag("--steps", "2000"))),
        edit_argv("ORC", 1, chain_f(set_flag("--lr", "3e-4"), set_flag("--warmup", "200"), set_flag("--p-gt-start", "1.0"), set_flag("--p-gt-end", "0.1"))),
        edit_argv("ORC0", 0, chain_f(set_flag("--init-ckpt", "/x/M2b_s1/ckpt_last.pt"))),
        edit_argv("ORC0", 1, lambda a: a.extend(["--threads", "4"])),
        lambda fx: jedit(fx.root / "runs" / "ORC_s1" / "summary.json", lambda o: o.update(done=False, ego_perturb_n=5000)),
        lambda fx: jedit(fx.root / "runs" / "ORC0_s0" / "selfcheck.json", lambda o: o.update(declared_vs_built="FAIL")),
        lambda fx: jedit(fx.root / "runs" / "ORC_s0" / "summary.json", lambda o: o["levers"]["init_ckpt"].update(sha256="00" * 32)),
        lambda fx: jedit(fx.root / "runs" / "ORC0_s1" / "summary.json", lambda o: o.update(ego_perturb_n=1)),
        lambda fx: jedit(fx.root / "tokens" / "PASS_ORC_s1.json", lambda o: o.update(verdict="FAIL")),
        lambda fx: jedit(fx.root / "tokens" / "PASS_ORC0_s0.json", lambda o: o["binding"].update(data_sha256="ee" * 32)),
        lambda fx: jedit(fx.root / "t1" / "t1v8f_ORC_s0_causal.json", lambda o: o["t1v8f"].update(limit="logged")),
        lambda fx: jedit(fx.root / "t1" / "t1v8f_ORC0_s1_causal.json", lambda o: o["t1v8f"].update(ckpt="/x/runs/ORC_s1/ckpt_last.pt")),
        lambda fx: jedit(fx.root / "t0" / "eval_ORC_s1.json", lambda o: o.update(ckpt="/x/runs/ORC0_s1/ckpt_last.pt")),
        lambda fx: jedit(fx.root / "t1" / "t1v8f_ORC0_s0_causal.json",
                         lambda o: o["paired_decision_grade"]["paired_model_minus_floor_t1"]["ade_m"].update(delta=9.9)),
    )
    rc, _, out, _ = run(tmp_path, world=w, edit=edits, extra={"--banked-sha": "1=" + "11" * 32})
    err = refused(capsys, rc, out,
                  "K1:ORC_s0", "K2:ORC0_s0", "K3:ORC_s1", "K5:ORC0_s1", "K6:ORC_s0", "K7:ORC0_s0", "GRID:ORC_s1", "K4:ORC0_s1",
                  "ARGV:ORC_s0", "--ego-perturb-p is not named", "--steps 2000", "ARGV:ORC_s1", "--lr 3e-4", "--warmup 200", "--p-gt-start 1.0", "--p-gt-end 0.1",
                  "ARGV:ORC0_s0", "M2b_s0/ckpt_last.pt", "PAIR:s1", "not one lever", "PAIR:s0", "data manifests differ",
                  "SUMMARY:ORC_s1", "done is not true", "SUMMARY:ORC0_s0", "declared_vs_built",
                  "INIT-CKPT:ORC_s0", "differs from the M2b checkpoint", "PERTURB-REACHED:ORC_s1", "outside 0.3 +- 0.02", "PERTURB-REACHED:ORC0_s1", "must be exactly 0",
                  "GATE-TOKEN:ORC_s1", "not PASS", "T1-CONFIG:ORC_s0", "T1-CONFIG:ORC0_s1", "T1-CONFIG:ORC0_s0", "the ticks file says",
                  "T0-CKPT:ORC_s1", "R2-ADE6:ORC0_s0", "M2B-CKPT:s1",
                  absent=("ARGV:ORC0_s1", "SUMMARY:ORC_s0", "SUMMARY:ORC0_s1", "GATE-TOKEN:ORC_s0", "GATE-TOKEN:ORC0_s1", "T1-CONFIG:ORC_s1", "T0-CKPT:ORC_s0", "M2B-CKPT:s0",
                          "R2-ADE6:ORC_s0", "K4:ORC_s0", "SAME-WINDOWS"))
    assert "PAIR:s0" in err


def chain_f(*fs):
    def f(a):
        for g in fs:
            g(a)
    return f


def test_refuse_the_other_arms_checkpoint_and_swapped_runs(tmp_path, capsys):      # ORC and ORC-0 swapped
    rc, _, out, _ = run(tmp_path, edit=lambda fx: jedit(fx.root / "t1" / "t1v8f_ORC_s0_causal.json", lambda o: o["t1v8f"].update(ckpt="/x/runs/ORC0_s0/ckpt_last.pt")), **STRONG)
    refused(capsys, rc, out, "T1-CONFIG:ORC_s0", "ORC0_s0/ckpt_last.pt")


def test_refuse_pairing_by_different_windows(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, world=World(permute_orc=True, **STRONG))
    refused(capsys, rc, out, "SAME-WINDOWS:ORC-vs-ORC0:s0", "SAME-WINDOWS:ORC-vs-ORC0:s1", absent=("DUMP-WINDOWS",))     # ticks, dumps and T0 agree with each other, not with ORC-0's


def test_refuse_dump_windows_not_the_ticks_windows(tmp_path, capsys):
    def e(fx):
        d = fx.root / "t1" / "dump_ORC_s0_causal"
        f = sorted(d.glob("ep*.npz"))[0]
        z = dict(np.load(f))
        z["ws"] = z["ws"][::-1].copy()
        np.savez(f, **z)
    rc, _, out, _ = run(tmp_path, edit=e, **STRONG)
    refused(capsys, rc, out, "DUMP-WINDOWS:ORC_s0", "SAME-WINDOWS")


def test_refuse_t0_episodes_differ_and_grid_counts(tmp_path, capsys):
    def e(fx):
        p = fx.root / "t0" / "eval_ORC_s0.npz"
        z = dict(np.load(p))
        z["eid"] = np.roll(z["eid"], 7)
        np.savez(p, **z)
    rc, _, out, _ = run(tmp_path, edit=e, extra={"--n-windows": 2399}, **STRONG)
    refused(capsys, rc, out, "SAME-WINDOWS:ORC-vs-ORC0:s0", "GRID-COUNTS")


def test_refuse_one_seed_only(tmp_path, capsys):
    def e(fx):
        del fx.manifest["arms"]["ORC"]["1"]
    rc, _, out, _ = run(tmp_path, edit=e, **STRONG)
    refused(capsys, rc, out, "BOTH seeds")


def test_refuse_missing_file_is_a_refusal_not_a_traceback(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, edit=lambda fx: (fx.root / "t0" / "eval_ORC0_s1.npz").unlink(), **STRONG)
    refused(capsys, rc, out, "is missing")


def test_refuse_gate_token_missing_file(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, edit=lambda fx: (fx.root / "tokens" / "PASS_ORC_s1.json").unlink(), **STRONG)
    refused(capsys, rc, out, "launch-gate token", "is missing")


def test_refuse_prereg_text_mismatch(tmp_path, capsys):
    p = tmp_path / "PREREG_changed.md"
    p.write_text(PREREG.read_text(encoding="utf-8").replace("adopt for the head that moved", "adopt for any head"), encoding="utf-8")
    rc, _, out, _ = run(tmp_path, extra={"--prereg": str(p)}, **STRONG)
    refused(capsys, rc, out, "PREREG-TEXT", "PARTIAL")


def test_K_control_on_M2b_beside_run_and_logged_source_run(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, world=World(ctl_fail={("K6", "M2b_s1"), ("K3L", "ORC_s0")}, **STRONG), m2b=True, logged=True)
    err = refused(capsys, rc, out, "K6:M2b_s1", "K3-logged:ORC_s0")
    assert "K3:ORC_s0" not in err


def test_perturbation_key_names_are_cli_overridable(tmp_path, capsys):
    def e(fx):
        for arm in ("ORC", "ORC0"):
            for s in (0, 1):
                jedit(fx.root / "runs" / f"{arm}_s{s}" / "summary.json", lambda o: o.__setitem__("totals", {"pert_n": o.pop("ego_perturb_n"), "pert_elig": o.pop("ego_perturb_eligible")}))
    rc, res, *_ = run(tmp_path, edit=e, extra={"--key-perturb-n": "pert_n", "--key-perturb-eligible": "pert_elig"}, **STRONG)
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "ROBUST"
    rc, _, out, _ = run(tmp_path / "b", edit=e, name="o2.json", **STRONG)                    # without the override the default keys are not found
    refused(capsys, rc, out, "PERTURB-REACHED", "ego_perturb_n")


def test_failed_control_computes_no_rows(tmp_path, capsys):
    called = {"n": 0}
    real = orc.f1_pair_row

    def spy(*a, **k):
        called["n"] += 1
        return real(*a, **k)
    orc.f1_pair_row = spy
    try:
        rc, _, out, _ = run(tmp_path, world=World(ctl_fail={("K3", "ORC_s0")}, **STRONG))
    finally:
        orc.f1_pair_row = real
    assert rc == 2 and called["n"] == 0 and not out.exists()
