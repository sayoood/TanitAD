"""Tests of dag_read.py on SYNTHETIC inputs only (no DAgger bank, no DAG run exists). CPU, small.

    DAG_READ_PATH   the reader under test (default: ./dag_read.py; mutate_dag.py points it at a mutant)
    DAG_TREE        the read-only export of the tip holding stack/, taniteval/ and the v8f-mini raw/ readers (default E:/Projects/TanitAD-artifacts/rt4/tree)

Synthetic worlds: the T1 tick-file analysis (matching, cells, K1/K2/K3/K5/K6/K7) needs the real eval cache, so that one step is replaced by an in-memory backend
(`SynBackend`); EVERYTHING else is real files in the programme's formats: the trainer's argv.json / summary.json / selfcheck.json, a launch-gate-token shaped JSON,
the T1 tool's ep*.npz dumps and t1v8f json, the T0 eval's json + npz, and a bank report + eval-141 clip list in the (provisional) schema documented in dag_read.py.
The reader's own pairing, controls, rows and outcome run unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
DAG_PATH = os.environ.get("DAG_READ_PATH", str(HERE / "dag_read.py"))
TREE = pathlib.Path(os.environ.get("DAG_TREE", "E:/Projects/TanitAD-artifacts/rt4/tree"))
RAW = TREE / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "raw"
T1HD, T1K = RAW / "2026-10-26-t1hdread" / "t1hd_read.py", RAW / "2026-10-26-t1kin-hacc" / "t1k_read.py"
PREREG = HERE / "PREREG_V8F_DAGGER.md"
NB = 100                                                                    # TEST bootstrap size (the reader labels any B != 2000 'NOT A READ')

spec_ = importlib.util.spec_from_file_location("dag_read_under_test", DAG_PATH)
dag = importlib.util.module_from_spec(spec_)
sys.modules["dag_read_under_test"] = dag
spec_.loader.exec_module(dag)

NCLS = {"lat": 5, "lon": 7}
N_WIN = 306566                                                              # PREREG section 1 (literal here: the tests do not borrow the reader's constants)
CLIPS = [f"clip-{i:05d}" for i in range(4512)]
EVAL_IDS = [f"eval-{i:03d}" for i in range(141)]
ROLLOUT = {"replan_steps": 10, "limit": "causal", "nav": "deadreckon", "scene": "logged", "schedule": "periodic", "executor": "plan", "ticks": [1, 2]}


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def train_sha(ids):
    return hashlib.sha256("\n".join(sorted(ids, key=sha12)).encode("utf-8")).hexdigest()


TRAIN_SHA = train_sha(CLIPS)


# ======================================================================================================================== the synthetic world
def _modify(pred0, true, gain, r, n):
    """pred0 with a controlled effect: gain > 0 fixes that fraction of the wrong cells, gain < 0 breaks that fraction of the right ones (never tick 0)."""
    pred = pred0.copy()
    u = r.random(pred0.shape)
    if gain > 0:
        flip = (u < gain) & (pred0 != true) & (true >= 0)
        pred[flip] = true[flip]
    elif gain < 0:
        flip = (u < -gain) & (pred0 == true) & (true >= 0)
        pred[flip] = (true[flip] + 1) % n
    pred[:, 0] = pred0[:, 0]
    return pred


class World:
    """Per-seed arrays of DAG-0 (the base), DAG (the base plus a controlled effect) and ORC-0 (the base plus a controlled run-to-run difference: the replicate floor)."""

    def __init__(self, lat_gain=(0.0, 0.0), lon_gain=(0.0, 0.0), ade_shift=(0.0, 0.0), t0_shift=(0.0, 0.0), hole_dag=0.0, n_ep=18, per=20,
                 permute_dag=False, ctl_fail=(), identity=False, rf_lat=(0.0, 0.0), rf_lon=(0.0, 0.0), hole_orc0=0.0):
        self.n_ep, self.per, self.N = n_ep, per, n_ep * per
        self.ctl_fail = set(ctl_fail)
        self.s = {}
        for s in (0, 1):
            r = np.random.default_rng(100 + s)
            N = self.N
            ci = np.repeat(np.arange(n_ep), per)
            t0 = np.tile(np.arange(per) * 5 + 10, n_ep)
            w = {"ci": ci, "t0": t0, "true": {}, "pred0": {}, "pred1": {}, "true1": {}, "pred_o0": {}, "true_o0": {}}
            for hd, n in NCLS.items():
                true = r.integers(0, n, size=(N, 3))
                true[r.random((N, 3)) < 0.08] = -1
                pred0 = np.where(r.random((N, 3)) < 0.6, np.where(true >= 0, true, 0), r.integers(0, n, size=(N, 3)))
                gain = (lat_gain if hd == "lat" else lon_gain)[s]
                rf = (rf_lat if hd == "lat" else rf_lon)[s]
                pred1 = _modify(pred0, true, gain, r, n)
                pred_o0 = _modify(pred0, true, rf, np.random.default_rng(900 + s), n)
                true1 = true.copy()
                if hole_dag > 0:
                    hm = r.random((N, 3)) < hole_dag
                    hm[:, 0] = False
                    true1[hm] = -1
                true_o0 = true.copy()
                if hole_orc0 > 0:
                    hm0 = np.random.default_rng(700 + s).random((N, 3)) < hole_orc0
                    hm0[:, 0] = False
                    true_o0[hm0] = -1
                w["true"][hd], w["pred0"][hd], w["pred1"][hd], w["true1"][hd], w["pred_o0"][hd], w["true_o0"][hd] = true, pred0, pred1, true1, pred_o0, true_o0
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
                    w["pred1"][hd], w["true1"][hd], w["pred_o0"][hd], w["true_o0"][hd] = w["pred0"][hd], w["true"][hd], w["pred0"][hd], w["true"][hd]
                w["cl1"], w["t0_1"] = w["cl0"], w["t0_0"]
            w["order"] = np.arange(N)
            if permute_dag:
                w["order"] = np.concatenate([c * per + np.random.default_rng(5 + c).permutation(per) for c in range(n_ep)])
            self.s[s] = w


def jdump(p, o):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1), encoding="utf-8")


def jedit(p, fn):
    o = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    fn(o)
    pathlib.Path(p).write_text(json.dumps(o, indent=1), encoding="utf-8")


def dag_argv(arm, s, root, bank_sha=None):
    a = ["--arm", "M2b", "--seed", str(s), "--out", f"{root}/runs/{arm}_s{s}", "--init-ckpt", f"{root}/m2b/M2b_s{s}/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4",
         "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0"]
    if arm == "DAG":
        a += ["--dagger-p", "0.1", "--dagger-bank", f"{root}/bank/bank_s{s}.bin", "--dagger-bank-sha256", bank_sha, "--dagger-seed", str(s)]
    elif arm == "DAG0":
        a += ["--dagger-p", "0"]
    return a                                                         # ORC0: the older build -- the same argv minus --dagger-p


def bank_report(s, m2b_sha, bank_sha, n_elig=250000, clips=None):
    """A good bank report in the (provisional) schema documented in dag_read.py."""
    ids = list(CLIPS if clips is None else clips)
    before = 2 * N_WIN
    kept = max(400000, n_elig)
    off, head = 150000, 40000
    return {"kind": "dagger-bank-report",
            "header": {"seed": s, "generator_ckpt_sha256": m2b_sha, "bank_sha256": bank_sha, "window_list_sha256": train_sha(ids),
                       "matching_constants_sha256": "438366d5" + "ab" * 28, "rollout": json.loads(json.dumps(ROLLOUT)), "hmax": 16, "n_clips": len(ids), "n_windows": N_WIN},
            "clip_ids": ids,
            "coverage": {"n_windows": N_WIN, "n_eligible_windows": n_elig, "n_states_before_cuts": before, "n_states_kept": kept,
                         "cuts": {"off_route": off, "heading": head, "end_edge": before - kept - off - head}, "eligible_fraction": n_elig / N_WIN},
            "controls": {"K-B0": {"PASS": True, "n_checked": 4096, "n_mismatch": 0},
                         "K-B1": {"PASS": True, "n_replayed": 1533, "n_mismatch": 0},
                         "K-B2": {"PASS": True},
                         "K-B3": {"PASS": True, "deliberate_regression_refused": True},
                         "K-B4": {"PASS": True, "cases": {k: True for k in ("straight", "stationary_tie", "lateral_cut", "heading_cut", "end_edge", "loop")}},
                         "K-B5": {"PASS": True, "max_pose_now_err_m": 0.0, "max_rel_transform_err_m": 2e-7}}}


class Fx:
    """A built fixture on disk: paths + the manifest dict (editable before it is saved)."""

    def __init__(self, root, world, m2b=False, orc=False, logged=False, nb=None, elig=250000):
        self.nb = NB if nb is None else nb
        self.root, self.world = pathlib.Path(root), world
        self.m2b_sha, self.bank_sha = {}, {}
        self.manifest = {"seeds": [0, 1], "arms": {"DAG": {}, "DAG0": {}, "ORC0": {}}, "bank": {}, "eval141_clip_ids": str(self.root / "eval141_clip_ids.json")}
        self.arms = ["DAG", "DAG0", "ORC0"] + (["ORC"] if orc else []) + (["M2b"] if m2b else [])
        for a in ("ORC", "M2b"):
            if a in self.arms:
                self.manifest["arms"][a] = {}
        jdump(self.root / "eval141_clip_ids.json", EVAL_IDS)
        for s in (0, 1):
            p = self.root / "m2b" / f"M2b_s{s}" / "ckpt_last.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"synthetic M2b checkpoint seed {s}".encode())
            self.m2b_sha[s] = hashlib.sha256(p.read_bytes()).hexdigest()
            bf = self.root / "bank" / f"bank_s{s}.bin"
            bf.parent.mkdir(parents=True, exist_ok=True)
            bf.write_bytes(f"synthetic DAgger bank seed {s}".encode())
            self.bank_sha[s] = hashlib.sha256(bf.read_bytes()).hexdigest()
            n_e = elig[s] if isinstance(elig, dict) else elig
            jdump(self.root / "bank" / f"report_s{s}.json", bank_report(s, self.m2b_sha[s], self.bank_sha[s], n_e))
            self.manifest["bank"][str(s)] = {"report": str(self.root / "bank" / f"report_s{s}.json"), "bank_file": str(bf)}
            for arm in self.arms:
                self.manifest["arms"][arm][str(s)] = self.build_run(arm, s, logged)

    def build_run(self, arm, s, logged):
        root, W = self.root, self.world.s[s]
        tag = f"{arm}_s{s}"
        e = {"ckpt_tail": f"{tag}/ckpt_last.pt"}
        if arm in ("DAG", "DAG0", "ORC0"):
            td = root / "runs" / tag
            argv = dag_argv(arm, s, str(root), self.bank_sha[s])
            sha = dag.argv_sha16(argv)
            lev = {"init_ckpt": {"path": argv[argv.index("--init-ckpt") + 1], "sha256": self.m2b_sha[s]}}
            tot = {}
            if arm == "DAG":
                lev.update(dagger_p=0.1, dagger_bank_sha256=self.bank_sha[s])
                tot = {"dagger_n": 1003, "dagger_eligible": 10000}
            elif arm == "DAG0":
                lev.update(dagger_p=0.0)
                tot = {"dagger_n": 0, "dagger_eligible": 10000}
            jdump(td / "argv.json", {"argv": argv, "sha": sha})
            jdump(td / "summary.json", {"done": True, "steps": 3000, "arm": "M2b", "seed": s, "n_clips": 4512, "n_windows": N_WIN, "train_sha256": TRAIN_SHA,
                                        "excluded_sha256": "bb" * 32, "argv_sha": sha, "levers": lev, **tot})
            jdump(td / "selfcheck.json", {"declared_vs_built": "PASS", "argv_sha": sha, "levers": lev})
            e["train_dir"] = str(td)
            if arm != "ORC0":
                dsha = "cd" * 32 if arm == "DAG0" else "ce" * 32          # DAG's data manifest includes the bank's sha: the two tokens' manifests legitimately differ
                jdump(root / "tokens" / f"PASS_{tag}.json", {"kind": "launch-gate-token", "verdict": "PASS", "hmac_sha256": "ef" * 32,
                                                              "binding": {"profile": "v8f-mini", "commit": "ab" * 20, "argv_sha256": dag.gate_argv_sha256(argv), "argv": argv,
                                                                          "data_sha256": dsha}})
                e["token"] = str(root / "tokens" / f"PASS_{tag}.json")
        e["t1"] = self.write_t1(arm, s, W, f"{tag}_causal", "causal", tag)
        if logged:
            e["t1_logged"] = self.write_t1(arm, s, W, f"{tag}_logged", "logged", tag, dumps=False)
        e["t0"] = self.write_t0(arm, s, W, tag)
        e["t0_fresh"] = self.write_t0(arm, s, W, tag, ages="fresh")
        return e

    def write_t1(self, arm, s, W, t1tag, limit, tag, dumps=True):
        root = self.root
        cl = {"DAG": W["cl1"], "M2b": W["cl0"] + 0.01, "ORC0": W["cl0"] + 0.02, "ORC": W["cl0"] + 0.015}.get(arm, W["cl0"])
        o = W["order"] if arm == "DAG" else np.arange(len(W["ci"]))
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
        ade = (W["t0_1"][W["order"]] if arm == "DAG" else W["t0_0"])
        nm = f"eval_{tag}" + ("-fresh" if ages == "fresh" else "")
        if ages == "fresh":                     # a different eval: DAG's is 7 m worse, so a reader that took R-T0 from it would read a +7 m point
            ade = ade + (7.0 if arm == "DAG" else 0.0)
        jdump(root / "t0" / f"{nm}.json", {"ckpt": f"{root}/runs/{tag}/ckpt_last.pt", "ages": ages, "raw": {}})
        np.savez(root / "t0" / f"{nm}.npz", ade6=ade, eid=W["ci"], ckpt=np.zeros((len(ade), 3, 3)), lat_logits=np.zeros((len(ade), 5)), lon_logits=np.zeros((len(ade), 7)))
        return {"json": str(root / "t0" / f"{nm}.json"), "npz": str(root / "t0" / f"{nm}.npz")}

    def save(self):
        jdump(self.root / "runs.json", self.manifest)
        return str(self.root / "runs.json")


_K = {}


def TREE_K():
    if "k" not in _K:
        _K["k"] = dag.load_t1k(str(T1K), str(TREE))
    return _K["k"]


class SynBackend:
    """Stands in for RealBackend's tick-file step; the packs come from the in-memory world, the K controls PASS unless the world says otherwise."""

    def __init__(self, T, world):
        self.T, self.world = T, world

    def _pack(self, entry, label, seed, light=False):
        W = self.world.s[seed]
        tag = entry["tag"].replace("_causal", "").replace("_logged", "")
        arm = tag.rsplit("_s", 1)[0]
        order = W["order"] if arm == "DAG" else np.arange(self.world.N)
        cells = {}
        for hd in NCLS:
            tr, pr = {"DAG": (W["true1"][hd], W["pred1"][hd]), "ORC0": (W["true_o0"][hd], W["pred_o0"][hd])}.get(arm, (W["true"][hd], W["pred0"][hd]))
            cells[hd] = {"true": tr[order], "pred": pr[order]}
        tick0 = {hd: float(self.T.f1_red(NCLS[hd])(self.T.E0._f1_encoded(cells[hd]["true"][:, 0], cells[hd]["pred"][:, 0], NCLS[hd]))) for hd in NCLS}
        ctl = {k: (not ({(k + ("L" if light else ""), tag)} & self.world.ctl_fail), None) for k in (("K2", "K3", "GRID") if light else ("K1", "K2", "K3", "K5", "K6", "K7", "GRID"))}
        meta = {"tag": entry["tag"], "ckpt": f"/x/runs/{tag}/ckpt_last.pt", "replan_steps": 10}
        if ("TICKMETA", tag) in self.world.ctl_fail:
            meta["ckpt"] = "/x/runs/WRONG_s0/ckpt_last.pt"
        return dag.Pack(label, W["ci"][order], W["t0"][order], cells, tick0, meta, ctl, goal={"f1_ge1": 0.5, "f1_tick0": 0.5})

    def analyze_t1(self, entry, label, seed):
        return self._pack(entry, label, seed)

    def analyze_t1_light(self, entry, label, seed):
        return self._pack(entry, label, seed, light=True)

    def k4(self, entry_logged, pack, t0_entry, seed):
        tag = pack.meta["tag"].replace("_causal", "")
        fresh = str(t0_entry["json"]).endswith("-fresh.json")          # the synthetic K4 holds only against the arm's fresh-age eval
        return (fresh and not ({("K4", tag)} & self.world.ctl_fail)), {"source": "synthetic", "t0_json": str(t0_entry["json"])}


def read_args(fx, out, **kw):
    a = ["--manifest", fx.save(), "--out", str(out), "--tool-root", str(TREE), "--t1hd-read", str(T1HD), "--t1k-read", str(T1K), "--eval-cache", "x", "--tracks", "x",
         "--eval-grid", "x", "--eval-join", "x", "--n-boot", str(kw.pop("n_boot", NB)), "--n-windows", str(fx.world.N), "--n-episodes", str(fx.world.n_ep),
         "--m2b-ckpt", f"0={fx.root}/m2b/M2b_s0/ckpt_last.pt", "--m2b-ckpt", f"1={fx.root}/m2b/M2b_s1/ckpt_last.pt",
         "--banked-sha", f"0={fx.m2b_sha[0]}", "--banked-sha", f"1={fx.m2b_sha[1]}", "--prereg", str(PREREG), "--work-dir", str(fx.root / "work")]
    for k, v in kw.items():
        a += [k, str(v)]
    return a


def run(tmp_path, world=None, m2b=False, orc=False, logged=False, edit=None, extra=None, name="DAG_READ.json", fxkw=None, **wk):
    world = world or World(**wk)
    fx = Fx(tmp_path, world, m2b=m2b, orc=orc, logged=logged, **(fxkw or {}))
    if edit:
        edit(fx)
    out = tmp_path / name

    def factory(T, a):
        return SynBackend(T, world)
    rc = dag.main(read_args(fx, out, **(extra or {})), backend_factory=factory)
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return rc, res, out, fx


# strong effects (gain 0.5: DAG gets half of DAG-0's wrong cells right) -- on both seeds, both heads
STRONG = dict(lat_gain=(0.5, 0.5), lon_gain=(0.5, 0.5))


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


def chain(*edits):
    def e(fx):
        for f in edits:
            f(fx)
    return e


def chain_f(*fs):
    def f(a):
        for g in fs:
            g(a)
    return f


def set_flag(flag, v):
    return lambda a: a.__setitem__(a.index(flag) + 1, v)


def drop_flag(flag, n=2):
    def f(a):
        i = a.index(flag)
        del a[i:i + n]
    return f


def edit_argv(arm, s, fn):
    def e(fx):
        td = fx.root / "runs" / f"{arm}_s{s}"

        def f(o):
            fn(o["argv"])
            o["sha"] = dag.argv_sha16(o["argv"])
        jedit(td / "argv.json", f)
        a = json.loads((td / "argv.json").read_text())
        for nm in ("summary.json", "selfcheck.json"):
            jedit(td / nm, lambda o: o.update(argv_sha=a["sha"]))
        if arm != "ORC0":
            jdump(fx.root / "tokens" / f"PASS_{arm}_s{s}.json", {"kind": "launch-gate-token", "verdict": "PASS", "hmac_sha256": "ef" * 32,
                                                                  "binding": {"commit": "ab" * 20, "argv_sha256": dag.gate_argv_sha256(a["argv"]), "argv": a["argv"],
                                                                              "data_sha256": "cd" * 32}})
    return e


def bank_edit(s, fn):
    return lambda fx: jedit(fx.root / "bank" / f"report_s{s}.json", fn)


# ======================================================================================================================== 1. the outcome table is the PREREG's
def prereg_table():
    t = PREREG.read_text(encoding="utf-8").split("## 7. Outcomes")[1].split("## 8. Queue")[0]
    rows = [[c.strip().replace("**", "") for c in ln.strip().strip("|").split(" | ")] for ln in t.splitlines() if ln.startswith("| **")]
    return [tuple(r) for r in rows]


def test_table_verbatim_and_in_order():
    assert tuple(dag.OUTCOME_TABLE) == tuple(prereg_table())              # names, conditions and committed-next texts, in the PREREG's row order
    assert dag.OUTCOME_NAMES == ("ROBUST", "PARTIAL", "NO EFFECT", "HARMS", "BLOCKED", "REFUSED")


def test_prereg_file_is_the_07dcb513_draft():
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == "719e5f153070e9c6986c11dac9250ea3f2a2363968c9ea65225d8b582b029374"
    assert "07dcb513" in dag.PREREG_ID and "719e5f153070e9c6986c11dac9250ea3f2a2363968c9ea65225d8b582b029374" in dag.PREREG_ID and "PREREG_V8F_DAGGER" in dag.PREREG_ID


def test_committed_numbers_are_the_preregs():
    t = PREREG.read_text(encoding="utf-8").replace("**", "")
    assert (dag.TWO_F["lat"], dag.TWO_F["lon"]) == (0.0248, 0.0146) and "2F = 0.0248 (lat) / 0.0146 (lon)" in t
    assert dag.HALF_F == 0.0604 and "point ≤ +F/2 = 0.0604 m" in t and dag.F_ADE == 0.1207 and "F_M2b 0.1207 m" in t
    assert (dag.STEPS, dag.LR, dag.WARMUP, dag.PGT_START, dag.PGT_END) == (3000, 1e-4, 100, 0.2, 0.2)
    assert "--steps 3000 --lr 1e-4\n  --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2" in t
    assert dag.DAG_P == 0.1 and dag.ARM_P == {"DAG": 0.1, "DAG0": 0.0} and dag.EGO_P == 0.0
    assert "p = 0.3 if O-RECOVER read NO EFFECT; p = 0.1 if it read HARMS" in t and "That second case is \"NO EFFECT's arm at p = 0.1\"" in t
    assert dag.RATIO_TOL == 0.02 and "lie in p ± 0.02" in t
    assert dag.COVERAGE_MIN == 0.5 and "the eligible fraction of TRAIN windows is ≥ 0.5" in t
    assert (dag.N_TRAIN_CLIPS, dag.N_TRAIN_WINDOWS) == (4512, 306566) and "4,512 clips, 306,566 windows" in t
    assert dag.HMAX == 16 and "hmax = 16" in t and dag.REPLAY_FRAC == 0.005 and "a seeded 0.5 % of bank rollouts" in t and dag.KB5_TOL_M == 1e-4 and "≤ 1e-4 m" in t
    assert dag.MATCH_CONST_SHA_PREFIX == "438366d5" and "sha256 `438366d5…`" in PREREG.read_text(encoding="utf-8").replace("**", "")
    assert dag.BANK_ROLLOUT == ROLLOUT
    for frag in ("--replan-steps 10", "--limit causal", "--nav deadreckon", "--scene logged", "--schedule periodic", "executor `plan`"):
        assert frag in t, frag
    for k in dag.KB4_CASES:
        assert k.replace("_", " ") in t, k
    assert "Off-route cut" in t and "e_lat > 3.0 m" in t and "> 30°" in t and "End edge" in t and "r′ > T − 31" in t and dag.CUT_REASONS == ("off_route", "heading", "end_edge")
    assert dag.KB_IDS == ("K-B0", "K-B1", "K-B2", "K-B3", "K-B4", "K-B5")
    assert dag.DAG_ONLY_FLAGS == ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")
    for fl in ("--dagger-p", *dag.DAG_ONLY_FLAGS, "dagger_n", "dagger_eligible"):
        assert fl in t, fl
    assert "`--ego-perturb-p 0` is named on both arms" in t and "`--ego-perturb-p > 0` together with `--dagger-p > 0` is refused" in t
    assert dag.BANKED_M2B_SHA[0].startswith("601cb435") and dag.BANKED_M2B_SHA[1].startswith("6104fb15") and "601cb435…" in t and "6104fb15…" in t


def test_harms_branches_are_substrings_of_the_cell_and_resolve_by_p():
    cell = prereg_table()[3][2]
    for p, br in dag.HARMS_BRANCH.items():
        assert br in cell, (p, br)
    assert set(dag.HARMS_BRANCH) == {0.3, 0.1}
    noe = prereg_table()[2][2]
    assert dag.harms_next(0.1) == {"ran_at_p": 0.1, "branch": "NO EFFECT's next (T-EGOCUT)", "next": noe}
    assert dag.harms_next(0.3)["next"] == "this PREREG's arm at p = 0.1" and dag.harms_next(0.3)["branch"] == dag.HARMS_BRANCH[0.3]
    assert dag.harms_next(0.5)["branch"] is None and dag.harms_next(0.5)["next"] is None


# ======================================================================================================================== 2. the outcome function (decide), row by row
def row(delta, sep=None, rf=None):
    sep = (delta != 0) if sep is None else sep
    r = {"delta": delta, "separated": bool(sep), "lo": 0.0, "hi": 0.0}
    if rf is not None:
        r["RF"] = rf
    return r


def rows_of(lat=(0.1, 0.1), lon=(0.1, 0.1), ade=(0.0, 0.0), t0=(0.0, 0.0), lat_sep=(True, True), lon_sep=(True, True), ade_sep=(False, False), rf_lat=(0.0, 0.0), rf_lon=(0.0, 0.0)):
    return {s: {"R-TLAT": row(lat[s], lat_sep[s], rf_lat[s]), "R-TLON": row(lon[s], lon_sep[s], rf_lon[s]), "R-ADE": row(ade[s], ade_sep[s]), "R-T0": {"point": t0[s]}} for s in (0, 1)}


def test_decide_robust_partial_noeffect_harms():
    assert dag.decide(rows_of())["OUTCOME"] == "ROBUST"
    assert dag.decide(rows_of(lon=(0.0, 0.0), lon_sep=(False, False)))["OUTCOME"] == "PARTIAL"
    assert dag.decide(rows_of(lat=(0.0, 0.0), lat_sep=(False, False)))["OUTCOME"] == "PARTIAL"
    assert dag.decide(rows_of(lat=(0.0, 0.0), lon=(0.0, 0.0), lat_sep=(False, False), lon_sep=(False, False)))["OUTCOME"] == "NO EFFECT"
    assert dag.decide(rows_of(ade=(0.2, 0.0), ade_sep=(True, False)))["OUTCOME"] == "HARMS"
    assert dag.decide(rows_of(t0=(0.0, 0.0605)))["OUTCOME"] == "HARMS"


def test_bar_is_max_of_2F_and_2RF():
    f = dag.bar_for
    assert f("lat", 0.0) == 0.0248 and f("lon", 0.0) == 0.0146                       # no replicate gap: the bar is 2F
    assert f("lat", 0.0124) == 0.0248 and f("lon", 0.0073) == 0.0146                 # RF at the seed floor: 2RF == 2F
    assert f("lat", 0.01) == 0.0248 and f("lon", 0.001) == 0.0146                    # RF below it never LOWERS the bar
    assert f("lat", 0.02) == 0.04 and f("lon", 0.02) == 0.04                         # RF above it raises the bar by twice the measured gap


def test_decide_bars_are_per_head_and_inclusive():
    f = dag.decide
    d = f(rows_of(lat=(0.0248, 0.0248), lon=(0.0146, 0.0146)))
    assert d["OUTCOME"] == "ROBUST" and d["flags"]["head_clears_at_both_seeds"] == {"lat": True, "lon": True}
    d = f(rows_of(lat=(0.0247, 0.0247), lon=(0.0146, 0.0146)))
    assert d["flags"]["head_clears_at_both_seeds"] == {"lat": False, "lon": True} and d["OUTCOME"] == "PARTIAL"
    d = f(rows_of(lat=(0.02, 0.02), lon=(0.02, 0.02)))                                # 0.02 clears lon's 0.0146, not lat's 0.0248
    assert d["flags"]["head_clears_at_both_seeds"] == {"lat": False, "lon": True} and d["OUTCOME"] == "PARTIAL"
    d = f(rows_of(lat=(0.0248, 0.0248), lon=(0.0145, 0.0145)))
    assert d["flags"]["head_clears_at_both_seeds"] == {"lat": True, "lon": False} and d["OUTCOME"] == "PARTIAL"


def test_decide_replicate_floor_raises_the_bar_per_seed_and_inclusively():
    f = dag.decide
    # RF 0.02 => lat bar 0.04: a delta of 0.03 clears 2F but not the bar; 0.04 clears (inclusive)
    d = f(rows_of(lat=(0.03, 0.03), rf_lat=(0.02, 0.02), lon=(0.1, 0.1)))
    assert d["flags"]["head_clears_at_both_seeds"] == {"lat": False, "lon": True} and d["OUTCOME"] == "PARTIAL"
    assert d["flags"]["bar_per_seed"]["lat"] == {"s0": 0.04, "s1": 0.04} and d["flags"]["sub_bar_separated_better_at_both_seeds"] == {"lat": True, "lon": False}
    assert f(rows_of(lat=(0.04, 0.04), rf_lat=(0.02, 0.02)))["OUTCOME"] == "ROBUST"
    assert f(rows_of(lat=(0.0399, 0.04), rf_lat=(0.02, 0.02)))["OUTCOME"] == "PARTIAL"
    # the bar is per SEED: s1's larger RF binds s1 only (the head then fails at 'both seeds')
    d = f(rows_of(lat=(0.03, 0.03), rf_lat=(0.0, 0.02), lon=(0.1, 0.1)))
    assert d["flags"]["clears_per_seed"]["lat"] == {"s0": True, "s1": False} and d["OUTCOME"] == "PARTIAL"
    # RF below the floor changes nothing; lat's RF never leaks into lon's bar
    assert f(rows_of(lat=(0.0248, 0.0248), rf_lat=(0.001, 0.001), lon=(0.0146, 0.0146), rf_lon=(0.0, 0.0)))["OUTCOME"] == "ROBUST"
    assert f(rows_of(lon=(0.03, 0.03), rf_lat=(0.05, 0.05)))["OUTCOME"] == "ROBUST"
    # a missing RF is an error, never a silent 2F
    r = rows_of()
    del r[0]["R-TLAT"]["RF"]
    with pytest.raises(KeyError):
        dag.decide(r)


def test_decide_both_seeds_and_separation_are_needed():
    f = dag.decide
    assert f(rows_of(lat=(0.1, 0.1), lat_sep=(True, False), lon=(0.0, 0.0), lon_sep=(False, False)))["OUTCOME"] == "NO EFFECT"      # s1 not separated: lat has not cleared
    assert f(rows_of(lat=(0.1, -0.1), lat_sep=(True, True), lon=(0.0, 0.0), lon_sep=(False, False)))["OUTCOME"] == "NO EFFECT"     # s1 separated WORSE is not better
    assert f(rows_of(lat=(0.1, 0.1), lon=(0.1, 0.0), lon_sep=(True, False)))["OUTCOME"] == "PARTIAL"
    assert f(rows_of(lat=(0.1, 0.0), lat_sep=(True, True), lon=(0.0, 0.0), lon_sep=(False, False)))["OUTCOME"] == "NO EFFECT"       # a clear at ONE seed is not a clear at both


def test_decide_harms_conditions_and_guard_boundaries():
    f = dag.decide
    assert f(rows_of(t0=(0.0604, 0.0604)))["OUTCOME"] == "ROBUST"                                  # +F/2 exactly is 'within'
    assert f(rows_of(t0=(0.0, 0.06041)))["OUTCOME"] == "HARMS"
    assert f(rows_of(ade=(0.5, 0.5), ade_sep=(False, False)))["OUTCOME"] == "ROBUST"               # worse but NOT separated: the guard holds
    assert f(rows_of(ade=(-0.5, -0.5), ade_sep=(True, True)))["OUTCOME"] == "ROBUST"               # separated BETTER is not worse
    assert f(rows_of(ade=(0.0, 0.3), ade_sep=(False, True)))["OUTCOME"] == "HARMS"                 # a single seed is enough
    assert f(rows_of(lat=(0.0, 0.0), lon=(0.0, 0.0), lat_sep=(False, False), lon_sep=(False, False), t0=(0.07, 0.0)))["OUTCOME"] == "HARMS"
    assert f(rows_of(ade=(0.3, 0.3), ade_sep=(True, True), t0=(0.07, 0.07)))["flags"]["guards_held"] is False
    assert f(rows_of(ade=(0.3, 0.0), ade_sep=(True, False)))["OUTCOME"] == "HARMS"                 # HARMS beats a ROBUST head pattern: a failed guard is HARMS


def test_decide_a_sub_bar_head_is_NO_EFFECT_and_flagged():
    d = dag.decide(rows_of(lat=(0.1, 0.01), lon=(0.0, 0.0), lon_sep=(False, False)))
    assert d["OUTCOME"] == "NO EFFECT" and d["matching_rows_in_table_order"] == ["NO EFFECT"]
    assert d["flags"]["sub_bar_separated_better_at_both_seeds"] == {"lat": True, "lon": False}
    b = dag.outcome_block(d)
    assert b["condition"] == prereg_table()[2][1] and b["committed_next"] == prereg_table()[2][2]


def test_table_is_complete_no_reachable_UNPLACED():
    for lat, lon, ls, ns, ade, t0, rf in itertools.product((0.0, 0.01, 0.1), (0.0, 0.01, 0.1), (False, True), (False, True), (0.0, 0.3), (0.0, 0.07), (0.0, 0.05)):
        d = dag.decide(rows_of(lat=(lat, 0.1), lon=(lon, 0.1), lat_sep=(ls, True), lon_sep=(ns, True), ade=(ade, 0.0), ade_sep=(ade > 0, False), t0=(t0, 0.0), rf_lat=(rf, 0.0)))
        assert d["OUTCOME"] != "UNPLACED", (lat, lon, ls, ns, ade, t0, rf)


def test_decide_lists_every_matching_row_in_table_order():
    for r in (rows_of(), rows_of(ade=(0.5, 0.5), ade_sep=(True, True)), rows_of(lat=(0.0, 0.0), lon=(0.0, 0.0), lat_sep=(False, False), lon_sep=(False, False))):
        d = dag.decide(r)
        assert d["matching_rows_in_table_order"] == [d["OUTCOME"]]
        assert [n for n in dag.OUTCOME_NAMES if n in d["matching_rows_in_table_order"]] == d["matching_rows_in_table_order"]


def test_decide_needs_both_seeds():
    r = rows_of()
    del r[1]
    with pytest.raises(dag.Refusal):
        dag.decide(r)


def test_outcome_block_carries_the_committed_text_verbatim():
    for nm, cond, nxt in prereg_table()[:4]:
        b = dag.outcome_block({"OUTCOME": nm, "matching_rows_in_table_order": [nm], "flags": {}})
        assert b["committed_next"] == nxt and b["condition"] == cond
        assert ("committed_next_for_this_run" in b) == (nm == "HARMS")
    assert dag.outcome_block({"OUTCOME": "HARMS", "matching_rows_in_table_order": ["HARMS"], "flags": {}})["committed_next_for_this_run"] == dag.harms_next(0.1)
    cov = {s: {"eligible_fraction": 0.4 if s == 0 else 0.9} for s in (0, 1)}
    b = dag.blocked_block(cov)
    assert b["OUTCOME"] == "BLOCKED" and b["condition"] == prereg_table()[4][1] and b["committed_next"] == prereg_table()[4][2] and b["blocked_seeds"] == ["s0"]
    assert dag.blocked_block({0: {"eligible_fraction": 0.5}, 1: {"eligible_fraction": 0.4999}})["blocked_seeds"] == ["s1"]


# ======================================================================================================================== 3. helpers
def test_parse_flags_keeps_every_value():
    f = dag.parse_flags(["--a", "1", "--ego-perturb-vscale", "0.5", "1.0", "--flag", "--b", "x", "--a", "2"])
    assert f["--ego-perturb-vscale"] == [["0.5", "1.0"]] and f["--flag"] == [[]] and f["--a"] == [["1"], ["2"]]
    assert dag.parse_flags(["--v", "0.5", "1.0"]) != dag.parse_flags(["--v", "0.5", "9.9"])


def test_find_key_looks_in_levers_and_totals():
    src = [("summary", {"levers": {"a": 1}, "totals": {"b": 2}, "c": {"d": 3}}), ("x", None)]
    assert dag.find_key(src, "a") == [("summary:levers.a", 1)] and dag.find_key(src, "b") == [("summary:totals.b", 2)] and dag.find_key(src, "c.d") == [("summary:c.d", 3)]
    assert dag.find_key(src, "zz") == []


def test_gate_argv_hash_is_the_launch_gates():
    assert dag.gate_argv_sha256(["--a", "1"]) == hashlib.sha256(b'["--a","1"]').hexdigest()


def tip_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(TREE / rel))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_clip_hashing_is_the_trainers():
    """sha12 / stable_sid are the tip's v8f_mini_data functions, and the TRAIN-list hash is the formula the tip's trainer records as train_sha256."""
    D = tip_module("v8d_for_dag_test", "stack/tanitad/data/v8f_mini_data.py")
    for c in ("abc", "clip-00001", "d9c3-ff"):
        assert dag.sha12(c) == D.sha12(c) and dag.stable_sid(c) == D.stable_episode_id(c)
    src = (TREE / "stack/scripts/v8f_mini_train.py").read_text(encoding="utf-8")
    assert 'hashlib.sha256(chr(10).join(sorted(clips, key=D.sha12)).encode("utf-8")).hexdigest()' in src
    ids = ["b", "a", "c-10", "c-9"]
    assert dag.train_list_sha256(ids) == hashlib.sha256(chr(10).join(sorted(ids, key=D.sha12)).encode("utf-8")).hexdigest()
    assert dag.train_list_sha256(ids) == dag.train_list_sha256(list(reversed(ids))) != dag.train_list_sha256(ids[:3])


# ======================================================================================================================== 4. synthetic worlds that must read each outcome
def test_world_ROBUST(tmp_path, capsys):
    rc, res, out, fx = run(tmp_path, **STRONG)
    assert rc == 0, capsys.readouterr().err
    assert res["OUTCOME"]["OUTCOME"] == "ROBUST" and res["OUTCOME"]["committed_next"] == prereg_table()[0][2]
    for s in ("s0", "s1"):
        for nm in ("R-TLAT", "R-TLON"):
            r = res["rows"][s][nm]
            assert r["separated"] and r["delta"] >= r["bar"] and r["better"] and r["matched"] and r["lower_is_better"] is False and r["clears"] is True
            assert r["RF"] == 0.0 and r["bar"] == r["two_F"] and r["bar_binding"] == "2F"           # ORC-0 is a pure copy of DAG-0 here: RF reads exactly 0
        assert res["rows"][s]["R-ADE"]["separated"] is False and res["rows"][s]["R-T0"]["point"] == pytest.approx(0.0, abs=1e-12)
    assert res["LABEL"].startswith("TEST") and all(res["controls_PASS"].values()) and res["strategic_family"].startswith("NOT PAUSED") and "2026-10-17" in res["strategic_family"]
    assert res["OUTCOME"]["matching_rows_in_table_order"] == ["ROBUST"] and res["dagger_p"] == 0.1 and res["claim_row"] == "H-V8F-DAGGER-1"
    assert res["prereg"] == dag.PREREG_ID and "R73" in res["strategic_family"]
    for k in ("K-B0", "K-B1", "K-B2", "K-B3", "K-B4", "K-B5", "BANK-CONFIG", "BANK-ACCOUNTING"):
        assert res["controls_PASS"][f"{k}:s0"] and res["controls_PASS"][f"{k}:s1"], k
    for k in ("PAIR:s0", "PAIR:s1", "ORC0-REPLICATE:s0", "BANK-WINDOWS:s1", "BANK-DECLARED:DAG_s0", "DAGGER-REACHED:DAG0_s1", "GATE-TOKEN:DAG_s1"):
        assert res["controls_PASS"][k], k
    assert res["bank"]["s0"]["coverage"]["eligible_fraction"] == pytest.approx(250000 / 306566)


def test_world_PARTIAL_lat_only_and_lon_only(tmp_path):
    rc, res, *_ = run(tmp_path / "a", lat_gain=(0.5, 0.5))
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "PARTIAL"
    assert res["OUTCOME"]["flags"]["head_clears_at_both_seeds"] == {"lat": True, "lon": False} and res["OUTCOME"]["committed_next"] == prereg_table()[1][2]
    rc, res, *_ = run(tmp_path / "b", lon_gain=(0.5, 0.5))
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "PARTIAL" and res["OUTCOME"]["flags"]["head_clears_at_both_seeds"] == {"lat": False, "lon": True}


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
    assert res["OUTCOME"]["committed_next_for_this_run"]["branch"] == "NO EFFECT's next (T-EGOCUT)" and res["OUTCOME"]["committed_next_for_this_run"]["next"] == prereg_table()[2][2]


def test_world_HARMS_via_T0(tmp_path):
    rc, res, *_ = run(tmp_path, t0_shift=(0.1, 0.0), **STRONG)
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "HARMS" and res["OUTCOME"]["flags"]["R-T0_above_F_over_2_per_seed"] == {"s0": True, "s1": False}
    assert res["rows"]["s0"]["R-T0"]["point"] == pytest.approx(0.1, abs=1e-9)


def test_world_T0_inside_the_guard_and_ADE_better_stay_ROBUST(tmp_path):
    rc, res, *_ = run(tmp_path, t0_shift=(0.05, 0.05), ade_shift=(-0.3, -0.3), **STRONG)
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "ROBUST"
    assert res["rows"]["s0"]["R-ADE"]["delta"] < 0 and res["rows"]["s0"]["R-ADE"]["separated"]


def test_world_sub_bar_head_reads_NO_EFFECT(tmp_path):
    rc, res, out, _ = run(tmp_path, lat_gain=(0.5, 0.03))
    assert 0 < res["rows"]["s1"]["R-TLAT"]["delta"] < dag.TWO_F["lat"] and res["rows"]["s1"]["R-TLAT"]["separated"] and res["rows"]["s1"]["R-TLAT"]["clears"] is False
    assert rc == 0 and out.exists() and res["OUTCOME"]["OUTCOME"] == "NO EFFECT"
    assert res["OUTCOME"]["flags"]["sub_bar_separated_better_at_both_seeds"]["lat"] is True


def test_world_identity_DAG_equals_DAG0(tmp_path):
    rc, res, *_ = run(tmp_path, identity=True, lat_gain=(0.5, 0.5))
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "NO EFFECT"
    for s in ("s0", "s1"):
        for nm, r in res["rows"][s].items():
            assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0 and not r["separated"], (s, nm, r)
        assert res["rows"][s]["R-T0"]["point"] == 0.0 and res["rows"][s]["R-ADE"]["point"] == 0.0
        assert res["rows"][s]["R-TLAT"]["RF"] == 0.0 and res["rows"][s]["R-TLON"]["RF"] == 0.0


def f1_of(T, true, pred, hd):
    return T.f1_red(NCLS[hd])(T.E0._f1_encoded(true, pred, NCLS[hd])[:, 1:].reshape(-1))


def test_replicate_floor_is_the_matched_f1_gap_of_DAG0_and_ORC0_and_raises_the_bar(tmp_path):
    """The same strong DAG effect reads ROBUST when ORC-0 replicates DAG-0 and NO EFFECT when ORC-0 differs from DAG-0 by more than half the effect (2RF > delta)."""
    rc, base, *_ = run(tmp_path / "a", **STRONG)
    assert rc == 0 and base["OUTCOME"]["OUTCOME"] == "ROBUST"
    w = World(rf_lat=(-0.6, -0.6), rf_lon=(-0.6, -0.6), **STRONG)
    rc, res, *_ = run(tmp_path / "b", world=w)
    assert rc == 0
    T = dag.load_t1hd(str(T1HD), str(TREE))
    for s in (0, 1):
        W = w.s[s]
        for hd, nm in (("lat", "R-TLAT"), ("lon", "R-TLON")):
            r = res["rows"][f"s{s}"][nm]
            ea = T.E0._f1_encoded(W["true"][hd], W["pred0"][hd], NCLS[hd])[:, 1:]
            eb = T.E0._f1_encoded(W["true"][hd], W["pred_o0"][hd], NCLS[hd])[:, 1:]
            m = (ea >= 0) & (eb >= 0)
            exp = abs(T.f1_red(NCLS[hd])(np.where(m, ea, -1).reshape(-1)) - T.f1_red(NCLS[hd])(np.where(m, eb, -1).reshape(-1)))
            assert exp > 0.1 and r["RF"] == pytest.approx(exp, abs=1.5e-4) and r["RF_unrounded"] == pytest.approx(exp, abs=1e-9)
            assert r["bar"] == pytest.approx(2 * r["RF"]) and r["bar"] > r["two_F"] and r["bar_binding"] == "2RF"
            assert r["separated"] and r["delta"] >= r["two_F"] and r["meets_2F"] is True and r["delta"] < r["bar"] and r["clears"] is False
    assert res["OUTCOME"]["OUTCOME"] == "NO EFFECT" and res["OUTCOME"]["flags"]["sub_bar_separated_better_at_both_seeds"] == {"lat": True, "lon": True}
    assert res["OUTCOME"]["flags"]["bar_per_seed"]["lat"]["s0"] == res["rows"]["s0"]["R-TLAT"]["bar"]


def test_replicate_floor_is_absolute_when_ORC0_is_the_better_replicate(tmp_path):
    rc, res, *_ = run(tmp_path, rf_lat=(0.6, 0.6), rf_lon=(0.6, 0.6), **STRONG)             # ORC-0 fixes 60 % of DAG-0's wrong cells: DAG-0 - ORC-0 is NEGATIVE
    assert rc == 0
    for s in ("s0", "s1"):
        for nm in ("R-TLAT", "R-TLON"):
            r = res["rows"][s][nm]
            assert r["RF_f1_orc0"] > r["RF_f1_dag0"] + 0.05 and r["RF"] > 0.05 and r["RF"] == pytest.approx(r["RF_f1_orc0"] - r["RF_f1_dag0"], abs=1.5e-4)
            assert r["bar"] == pytest.approx(max(r["two_F"], 2 * r["RF"])) and r["bar"] > r["two_F"] and r["bar_binding"] == "2RF"


def test_replicate_floor_reads_only_the_cells_scored_in_both(tmp_path):
    w = World(rf_lat=(-0.5, -0.5), rf_lon=(-0.5, -0.5), hole_orc0=0.3, **STRONG)
    rc, res, *_ = run(tmp_path, world=w)
    assert rc == 0
    T = dag.load_t1hd(str(T1HD), str(TREE))
    for s in (0, 1):
        W = w.s[s]
        for hd, nm in (("lat", "R-TLAT"), ("lon", "R-TLON")):
            ea = T.E0._f1_encoded(W["true"][hd], W["pred0"][hd], NCLS[hd])[:, 1:]
            eb = T.E0._f1_encoded(W["true_o0"][hd], W["pred_o0"][hd], NCLS[hd])[:, 1:]
            red = T.f1_red(NCLS[hd])
            m = (ea >= 0) & (eb >= 0)
            matched = abs(red(np.where(m, ea, -1).reshape(-1)) - red(np.where(m, eb, -1).reshape(-1)))
            own = abs(red(ea.reshape(-1)) - red(eb.reshape(-1)))
            assert abs(matched - own) > 5e-4, (hd, matched, own)                            # the fixture can tell the two readings apart
            r = res["rows"][f"s{s}"][nm]
            assert r["RF"] == pytest.approx(matched, abs=1.5e-4) and r["RF_n_cells_matched"] == int(m.sum()) < int((ea >= 0).sum())


def test_matched_cells_only(tmp_path):
    rc, res, *_ = run(tmp_path, hole_dag=0.25, **STRONG)
    assert rc == 0
    r, ro = res["rows"]["s0"]["R-TLAT"], res["rows"]["s0"]["R-TLAT_own_cells_beside"]
    assert r["matched"] and not ro["matched"]
    assert r["n_cells_a"] == r["n_cells_b"] == r["n_cells_matched_pairs"] and ro["n_cells_a"] < ro["n_cells_b"]       # DAG lost cells the matched read drops from BOTH sides


def test_ticks_ge1_only_tick0_never_enters(tmp_path):
    w = World(**STRONG)
    for s in (0, 1):                                                 # break DAG at tick 0 only: the ticks>=1 rows must not see it
        for hd in NCLS:
            w.s[s]["pred1"][hd][:, 0] = (w.s[s]["pred0"][hd][:, 0] + 1) % NCLS[hd]
    rc, res, *_ = run(tmp_path, world=w)
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "ROBUST"
    t0 = res["beside"]["per_seed"]["s0"]["tick0_f1"]
    assert t0["DAG"]["lat"] < t0["DAG0"]["lat"] - 0.3                # ... and the beside tick-0 row is where it shows
    n_ge1 = int(((w.s[0]["true"]["lat"][:, 1:] >= 0)).sum())
    assert res["rows"]["s0"]["R-TLAT_own_cells_beside"]["n_cells_b"] == n_ge1


def test_F1_sign_DAG_minus_DAG0(tmp_path):
    rc, res, *_ = run(tmp_path, lat_gain=(-0.4, -0.4), lon_gain=(0.0, 0.0))        # DAG is WORSE on lat
    assert rc == 0
    r = res["rows"]["s0"]["R-TLAT"]
    assert r["delta"] < 0 and r["separated"] and r["worse"] and not r["better"] and r["delta"] == pytest.approx(r["f1_a"] - r["f1_b"], abs=1e-4) and r["clears"] is False
    assert res["OUTCOME"]["OUTCOME"] == "NO EFFECT"


def test_beside_ORC_M2b_rows_and_K4_logged_source(tmp_path):
    rc, res, *_ = run(tmp_path, m2b=True, orc=True, logged=True, **STRONG)
    assert rc == 0
    b = res["beside"]["per_seed"]["s0"]
    for key in ("DAG_minus_M2b", "DAG_minus_ORC"):
        assert set(b[key]) >= {"T-LAT-F1", "T-LON-F1", "T1-kin ADE6", "T0 ADE6", "tick0_f1"} and b[key]["T-LAT-F1"]["delta"] > 0
    assert "T1-kin ADE6" in b["DAG0_minus_M2b"] and set(b["tick0_f1"]) == {"DAG", "DAG0", "ORC0"}
    assert "K4:DAG_s0" in res["controls_PASS"] and "K2-logged:DAG_s0" in res["controls_PASS"] and "K4:M2b_s0" in res["controls_PASS"] and "K4:ORC0_s1" in res["controls_PASS"]
    assert "per-class turn F1 (TURN_L / TURN_R)" in res["beside"]["not_computed"]


def test_end_to_end_default_B_2000_and_sha_printed(tmp_path, capsys):
    w = World(**STRONG)
    fx = Fx(tmp_path, w, nb=2000)                                    # the t1 jsons' ADE delta at B = 2000, so R2 compares like with like
    out = tmp_path / "DAG_READ.json"

    def factory(T, a):
        return SynBackend(T, w)
    rc = dag.main(read_args(fx, out, n_boot=2000), backend_factory=factory)
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    res = json.loads(out.read_text(encoding="utf-8"))
    assert "LABEL" not in res and res["n_boot"] == 2000 and res["rows"]["s0"]["R-TLAT"]["n_boot"] == 2000
    assert f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}" in cap.out and res["OUTCOME"]["OUTCOME"] == "ROBUST"


# ======================================================================================================================== 5. K4 / T0 (inherited from ORC Amendments 2-4, re-proved here)
def test_k4_reads_t0_fresh_and_r_t0_reads_the_standard_eval(tmp_path, capsys):
    rc, res, *_ = run(tmp_path, **STRONG)
    assert rc == 0, capsys.readouterr().err
    for tag in ("DAG_s0", "DAG0_s0", "DAG_s1", "DAG0_s1", "ORC0_s0", "ORC0_s1"):
        assert all(res["controls_PASS"][f"{k}:{tag}"] for k in ("K4", "T0-AGES", "T0F-CKPT")), tag
        assert res["controls"][f"K4:{tag}"]["detail"]["t0_json"].endswith(f"eval_{tag}-fresh.json")
    for s in ("s0", "s1"):
        assert abs(res["rows"][s]["R-T0"]["point"]) < 1.0, res["rows"][s]["R-T0"]
    assert res["OUTCOME"]["OUTCOME"] == "ROBUST"


def test_refuses_a_manifest_without_t0_fresh(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, edit=lambda fx: fx.manifest["arms"]["DAG0"]["1"].pop("t0_fresh"), **STRONG)
    refused(capsys, rc, out, "DAG0_s1: manifest has no 't0_fresh'")


def test_t0_ages_and_the_fresh_ckpt_are_controls(tmp_path, capsys):
    edits = chain(
        lambda fx: jedit(fx.root / "t0" / "eval_DAG_s0.json", lambda o: o.update(ages="fresh")),          # R-T0 would read a fresh-age eval
        lambda fx: jedit(fx.root / "t0" / "eval_DAG0_s1-fresh.json", lambda o: o.update(ages="cycle")),   # K4 would read a cycle-age eval
        lambda fx: jedit(fx.root / "t0" / "eval_DAG_s1-fresh.json", lambda o: o.update(ckpt="/x/runs/DAG0_s1/ckpt_last.pt")),
    )
    rc, _, out, _ = run(tmp_path, edit=edits, **STRONG)
    refused(capsys, rc, out, "T0-AGES:DAG_s0", "T0-AGES:DAG0_s1", "T0F-CKPT:DAG_s1",
            absent=("T0-AGES:DAG0_s0", "T0-AGES:DAG_s1", "T0F-CKPT:DAG_s0", "T0F-CKPT:DAG0_s1", "T0-CKPT:DAG_s1"))


# ======================================================================================================================== 6. unit tests of each control (function level: boundaries)
def fake_rec(argv, summary=None, selfcheck=None):
    return {"argv": argv, "flags": dag.parse_flags(argv), "summary.json": summary or {}, "selfcheck.json": selfcheck or {}, "argv.json": {"sha": dag.argv_sha16(argv)},
            "sources": [("summary", summary or {}), ("selfcheck", selfcheck or {}), ("metrics", None)]}


def test_g_reached_ratio_band_is_inclusive_and_exact():
    for n, ok in ((800, True), (1000, True), (1200, True), (799, False), (1201, False), (0, False)):
        r = fake_rec([], {"dagger_n": n, "dagger_eligible": 10000, "levers": {"dagger_p": 0.1}})
        assert dag.g_reached("t", "DAG", r, "dagger_n", "dagger_eligible", "dagger_p")[0] is ok, n
    r = fake_rec([], {"dagger_n": 3000, "dagger_eligible": 10000, "levers": {"dagger_p": 0.1}})                # 0.3: the value of the OTHER registered p
    assert dag.g_reached("t", "DAG", r, "dagger_n", "dagger_eligible", "dagger_p")[0] is False


def test_g_reached_dag0_exactly_zero_and_keys_present():
    f = lambda n, extra=None: dag.g_reached("t", "DAG0", fake_rec([], {"dagger_n": n, "levers": {"dagger_p": 0.0}, **(extra or {})}), "dagger_n", "dagger_eligible", "dagger_p")[0]
    assert f(0) is True and f(1) is False
    assert dag.g_reached("t", "DAG0", fake_rec([], {"levers": {"dagger_p": 0.0}}), "dagger_n", "x", "dagger_p")[0] is False        # n not recorded
    assert dag.g_reached("t", "DAG0", fake_rec([], {"dagger_n": 0}), "dagger_n", "x", "dagger_p")[0] is False                       # built p not recorded
    assert dag.g_reached("t", "DAG", fake_rec([], {"dagger_n": 1000, "levers": {"dagger_p": 0.1}}), "dagger_n", "dagger_eligible", "dagger_p")[0] is False
    assert dag.g_reached("t", "DAG", fake_rec([], {"dagger_n": 1000, "dagger_eligible": 10000, "levers": {"dagger_p": 0.3}}), "dagger_n", "dagger_eligible", "dagger_p")[0] is False
    assert dag.g_reached("t", "DAG0", fake_rec([], {"dagger_n": 0, "levers": {"dagger_p": 0.1}}), "dagger_n", "dagger_eligible", "dagger_p")[0] is False


def test_g_reached_conflicting_records_fail():
    r = fake_rec([], {"dagger_n": 1000, "dagger_eligible": 10000, "levers": {"dagger_p": 0.1}}, {"totals": {"dagger_n": 1001}})
    ok, det = dag.g_reached("t", "DAG", r, "dagger_n", "dagger_eligible", "dagger_p")
    assert not ok and "conflicting" in str(det)


BSHA = "ab" * 32
GOOD0 = ["--arm", "M2b", "--seed", "0", "--out", "/o", "--init-ckpt", "/x/M2b_s0/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2",
         "--p-gt-end", "0.2", "--ego-perturb-p", "0", "--dagger-p", "0"]
GOOD = GOOD0[:-1] + ["0.1", "--dagger-bank", "/b/bank_s0.bin", "--dagger-bank-sha256", BSHA, "--dagger-seed", "0"]


def test_g_argv_each_registered_flag():
    assert dag.g_argv("t", "DAG", 0, fake_rec(GOOD))[0] is True
    bad = {"--steps": "2000", "--lr": "3e-4", "--warmup": "200", "--p-gt-start": "1.0", "--p-gt-end": "0.1", "--ego-perturb-p": "0.1", "--dagger-p": "0.3", "--seed": "1", "--arm": "M1"}
    for flag, v in bad.items():
        a = list(GOOD)
        a[a.index(flag) + 1] = v
        ok, det = dag.g_argv("t", "DAG", 0, fake_rec(a))
        assert not ok and any(flag in d for d in det), (flag, det)
    for flag in ("--steps", "--lr", "--warmup", "--p-gt-start", "--p-gt-end", "--ego-perturb-p", "--dagger-p", "--init-ckpt"):     # not NAMED: a default is not a test of the lever
        a = list(GOOD)
        i = a.index(flag)
        del a[i:i + 2]
        assert not dag.g_argv("t", "DAG", 0, fake_rec(a))[0], flag
    assert not dag.g_argv("t", "DAG", 0, fake_rec(GOOD + ["--steps", "3000"]))[0]                                      # named twice
    assert not dag.g_argv("t", "DAG", 0, fake_rec([x.replace("M2b_s0", "M2b_s1") for x in GOOD]))[0]                  # the other seed's checkpoint
    assert dag.g_argv("t", "DAG0", 0, fake_rec(GOOD0))[0] is True and not dag.g_argv("t", "DAG", 0, fake_rec(GOOD0))[0]    # DAG-0's argv is not DAG's, and vice versa
    assert not dag.g_argv("t", "DAG0", 0, fake_rec(GOOD))[0]


def test_g_argv_dagger_flags_belong_to_dag_only():
    for i, fl in enumerate(dag.DAG_ONLY_FLAGS):                      # DAG names all four: dropping any DAG-only flag is refused; DAG-0 naming any of them is refused (inert at p = 0)
        a = list(GOOD)
        j = a.index(fl)
        del a[j:j + 2]
        ok, det = dag.g_argv("t", "DAG", 0, fake_rec(a))
        assert not ok and any(fl in d for d in det), (fl, det)
        ok, det = dag.g_argv("t", "DAG0", 0, fake_rec(GOOD0 + GOOD[GOOD.index(fl):GOOD.index(fl) + 2]))
        assert not ok and any(fl in d for d in det), (fl, det)
    for badv in ("xyz", "ab" * 31, ("AB" * 32)):                     # the declared bank sha must be a 64-hex sha256
        a = list(GOOD)
        a[a.index("--dagger-bank-sha256") + 1] = badv
        assert not dag.g_argv("t", "DAG", 0, fake_rec(a))[0], badv
    a = list(GOOD)
    a[a.index("--dagger-seed") + 1] = "x"
    assert not dag.g_argv("t", "DAG", 0, fake_rec(a))[0]
    assert not dag.g_argv("t", "DAG", 0, fake_rec(GOOD + ["--dagger-bank", "/other.bin"]))[0]                         # named twice


def test_g_argv_the_ego_perturbation_levers_are_never_combined():
    for law in ("--ego-perturb-vscale", "--ego-perturb-vmax", "--ego-perturb-adrop", "--ego-perturb-yawrate-std", "--ego-perturb-seed"):
        for arm, base in (("DAG", GOOD), ("DAG0", GOOD0)):
            ok, det = dag.g_argv("t", arm, 0, fake_rec(base + [law, "1"]))
            assert not ok and any(law in d for d in det), (arm, law, det)


def test_g_token_binding():
    argv = list(GOOD)
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "hmac_sha256": "ab", "binding": {"commit": "a" * 40, "argv_sha256": dag.gate_argv_sha256(argv), "data_sha256": "c" * 64}}
    r = fake_rec(argv)
    assert dag.g_token("t", r, tok)[0] is True
    for edit in (lambda b: b.update(argv_sha256="0" * 64), lambda b: b.pop("commit"), lambda b: b.update(commit="xyz"), lambda b: b.pop("data_sha256")):
        t2 = json.loads(json.dumps(tok))
        edit(t2["binding"])
        assert not dag.g_token("t", r, t2)[0]
    for k, v in (("verdict", "FAIL"), ("verdict", "INCOMPLETE"), ("kind", "evidence")):
        assert not dag.g_token("t", r, {**tok, k: v})[0]
    assert dag.g_token("t", fake_rec(["python", "train.py"] + argv), {**tok, "binding": {**tok["binding"], "argv_sha256": dag.gate_argv_sha256(argv)}})[0]     # a leading script path is stripped


def test_g_summary():
    sm = {"done": True, "steps": 3000, "arm": "M2b", "seed": 0}
    sha = dag.argv_sha16(GOOD)
    r = fake_rec(GOOD, {**sm, "argv_sha": sha}, {"declared_vs_built": "PASS", "argv_sha": sha})
    assert dag.g_summary("t", 0, r)[0] is True
    for k, v in (("done", False), ("steps", 2999), ("arm", "M1"), ("seed", 1), ("argv_sha", "0" * 16)):
        r2 = fake_rec(GOOD, {**sm, "argv_sha": sha, k: v}, {"declared_vs_built": "PASS", "argv_sha": sha})
        assert not dag.g_summary("t", 0, r2)[0], k
    assert not dag.g_summary("t", 0, fake_rec(GOOD, {**sm, "argv_sha": sha}, {"declared_vs_built": "FAIL", "argv_sha": sha}))[0]


def test_g_init_sha_sources():
    sha = "ab" * 32
    r = fake_rec(GOOD, {"levers": {"init_ckpt": {"sha256": sha}}})
    assert dag.g_init("t", 0, r, sha, "init_ckpt.sha256")[0] is True
    assert not dag.g_init("t", 0, r, "cd" * 32, "init_ckpt.sha256")[0]
    assert not dag.g_init("t", 0, fake_rec(GOOD), sha, "init_ckpt.sha256")[0]                                          # nothing recorded, no file: unproven
    assert dag.g_init("t", 0, fake_rec(GOOD, {"init_sha": sha}), sha, "init_sha")[0] is True                            # the key name is overridable


SUM = {"steps": 3000, "n_clips": 4512, "n_windows": 306566, "train_sha256": "aa" * 32, "excluded_sha256": "bb" * 32, "seed": 0, "levers": {"init_ckpt": {"sha256": "cc" * 32}}}


def test_g_pair_drops_only_the_dagger_flags():
    rd, r0 = fake_rec(GOOD, SUM), fake_rec(GOOD0, SUM)
    assert dag.g_pair(0, rd, r0, "init_ckpt.sha256")[0] is True
    for extra in (["--lr", "3e-4"], ["--threads", "4"], ["--ego-perturb-p", "0.1"]):     # any OTHER difference is a second lever
        a = list(GOOD0)
        if extra[0] in a:
            a[a.index(extra[0]) + 1] = extra[1]
        else:
            a += extra
        ok, det = dag.g_pair(0, rd, fake_rec(a, SUM), "init_ckpt.sha256")
        assert not ok and "not one lever" in str(det), extra
    for k, v in (("steps", 2999), ("n_clips", 1), ("n_windows", 1), ("train_sha256", "dd" * 32), ("excluded_sha256", "dd" * 32), ("seed", 1)):
        ok, det = dag.g_pair(0, rd, fake_rec(GOOD0, {**SUM, k: v}), "init_ckpt.sha256")
        assert not ok and k in str(det), k
    assert not dag.g_pair(0, rd, fake_rec(GOOD0, {**SUM, "levers": {"init_ckpt": {"sha256": "dd" * 32}}}), "init_ckpt.sha256")[0]
    assert not dag.g_pair(0, fake_rec(GOOD), fake_rec(GOOD0), "init_ckpt.sha256")[0]                                    # no init recorded on either side


def test_g_replicate_orc0_is_dag0_minus_dagger_p():
    ORC0 = [x for x in GOOD0[:-2]]
    r0, ro = fake_rec(GOOD0, SUM), fake_rec(ORC0, SUM)
    assert dag.g_replicate(0, r0, ro, "init_ckpt.sha256")[0] is True
    assert not dag.g_replicate(0, r0, fake_rec(ORC0 + ["--dagger-p", "0"], SUM), "init_ckpt.sha256")[0]                  # ORC-0 already names the flag
    assert not dag.g_replicate(0, r0, fake_rec(ORC0 + ["--threads", "4"], SUM), "init_ckpt.sha256")[0]
    assert not dag.g_replicate(0, fake_rec(GOOD0 + ["--threads", "4"], SUM), ro, "init_ckpt.sha256")[0]
    a = list(ORC0)
    a[a.index("--lr") + 1] = "3e-4"
    assert not dag.g_replicate(0, r0, fake_rec(a, SUM), "init_ckpt.sha256")[0]
    assert not dag.g_replicate(0, r0, fake_rec(GOOD, SUM), "init_ckpt.sha256")[0]                                        # DAG's argv is not a replicate of anything
    for k, v in (("steps", 2999), ("train_sha256", "dd" * 32), ("seed", 1)):
        assert not dag.g_replicate(0, r0, fake_rec(ORC0, {**SUM, k: v}), "init_ckpt.sha256")[0], k
    assert not dag.g_replicate(0, r0, fake_rec(ORC0, {**SUM, "levers": {"init_ckpt": {"sha256": "dd" * 32}}}), "init_ckpt.sha256")[0]


HDR = {"t1v8f": {"tag": "T", "ckpt": "/a/DAG_s0/ckpt_last.pt", "limit": "causal", "replan_steps": 10, "schedule": "periodic", "scene": "logged", "nav": "deadreckon", "ticks_heads": True}}


def test_g_t1cfg_each_field():
    meta = {"tag": "T", "ckpt": "/b/DAG_s0/ckpt_last.pt"}
    assert dag.g_t1cfg("t", HDR, "DAG_s0/ckpt_last.pt", "causal", meta)[0] is True
    for k, v in (("limit", "logged"), ("replan_steps", 5), ("schedule", "x"), ("scene", "frozen"), ("nav", "logged"), ("ticks_heads", False), ("ckpt", "/a/DAG0_s0/ckpt_last.pt"), ("tag", "U")):
        h = json.loads(json.dumps(HDR))
        h["t1v8f"][k] = v
        assert not dag.g_t1cfg("t", h, "DAG_s0/ckpt_last.pt", "causal", meta)[0], k
    assert not dag.g_t1cfg("t", HDR, "DAG_s0/ckpt_last.pt", "causal", {**meta, "ckpt": "/b/DAG0_s0/ckpt_last.pt"})[0]
    assert dag.g_t1cfg("t", {"t1v8f": {**HDR["t1v8f"], "limit": "logged"}}, "DAG_s0/ckpt_last.pt", "logged", meta)[0] is True


def test_k1_own_rows_identity_control_fires():
    class FakeT:
        @staticmethod
        def PB(a, b, eid, n_boot, seed, reduce):
            return {"delta": 1e-4, "lo": 0.0, "hi": 0.0}
    ok, det = dag.k1_own(FakeT, [("x", np.arange(4.0), np.arange(4), "mean")], 10)
    assert not ok and det[0]["delta"] == 1e-4


# ======================================================================================================================== 7. the bank (section 1, 2, 6) at function level
def good_bank(**kw):
    return bank_report(0, "ab" * 32, "cd" * 32, **kw)


def test_clip_overlap_by_id_and_by_sid_and_empty_eval():
    assert dag.clip_overlap(CLIPS, EVAL_IDS) == []
    assert dag.clip_overlap(CLIPS + [EVAL_IDS[7]], EVAL_IDS)[0].startswith("bank and eval-141 clip ids overlap")
    assert dag.clip_overlap(CLIPS, []) and "empty" in dag.clip_overlap(CLIPS, [])[0]
    assert dag.clip_overlap(CLIPS[:5], EVAL_IDS[:2] + [CLIPS[5]]) == []                  # not in this bank
    orig = dag.stable_sid
    dag.stable_sid = lambda c: 7                                                         # every sid collides: the sid half of the check must fire on its own
    try:
        r = dag.clip_overlap(["a"], ["b"])
    finally:
        dag.stable_sid = orig
    assert r == ["bank and eval-141 stable sids overlap"]


def test_kb_records_need_pass_and_their_evidence():
    rep = good_bank()
    assert dag.g_kb0(rep)[0] and dag.g_kb1(rep)[0] and dag.g_kb2(rep, "ab" * 32)[0] and dag.g_kb3(rep, EVAL_IDS)[0] and dag.g_kb4(rep)[0] and dag.g_kb5(rep)[0]
    for kid, fn in (("K-B0", dag.g_kb0), ("K-B1", dag.g_kb1), ("K-B4", dag.g_kb4), ("K-B5", dag.g_kb5)):
        for bad in (False, None, "yes"):
            r = good_bank()
            r["controls"][kid]["PASS"] = bad
            with pytest.raises(dag.Refusal):
                fn(r)
        r = good_bank()
        del r["controls"][kid]
        with pytest.raises(dag.Refusal):
            fn(r)
    for kid, fn in (("K-B2", lambda r: dag.g_kb2(r, "ab" * 32)), ("K-B3", lambda r: dag.g_kb3(r, EVAL_IDS))):
        r = good_bank()
        r["controls"][kid]["PASS"] = False
        with pytest.raises(dag.Refusal):
            fn(r)


def test_kb0_kb1_are_not_vacuous():
    for k, v in (("n_checked", 0), ("n_mismatch", 1), ("n_checked", True)):
        r = good_bank()
        r["controls"]["K-B0"][k] = v
        with pytest.raises(dag.Refusal):
            dag.g_kb0(r)
    for n, ok in ((1533, True), (1532, True), (1531, False), (0, False)):             # a seeded 0.5 % of 306,566 rollouts = 1532.83: floor is the bar
        r = good_bank()
        r["controls"]["K-B1"]["n_replayed"] = n
        if ok:
            assert dag.g_kb1(r)[0]
        else:
            with pytest.raises(dag.Refusal):
                dag.g_kb1(r)
    r = good_bank()
    r["controls"]["K-B1"]["n_mismatch"] = 2
    with pytest.raises(dag.Refusal):
        dag.g_kb1(r)


def test_kb2_generator_checkpoint_must_be_the_m2b_checkpoint():
    with pytest.raises(dag.Refusal, match="on-policy"):
        dag.g_kb2(good_bank(), "ef" * 32)


def test_kb3_overlap_regression_and_self_test():
    r = good_bank()
    r["clip_ids"] = r["clip_ids"][:-1] + [EVAL_IDS[3]]
    with pytest.raises(dag.Refusal, match="overlap"):
        dag.g_kb3(r, EVAL_IDS)
    r = good_bank()
    r["controls"]["K-B3"]["deliberate_regression_refused"] = False
    with pytest.raises(dag.Refusal, match="deliberate regression"):
        dag.g_kb3(r, EVAL_IDS)
    r = good_bank()
    r["clip_ids"] = r["clip_ids"] + [r["clip_ids"][0]]
    with pytest.raises(dag.Refusal, match="duplicates"):
        dag.g_kb3(r, EVAL_IDS)
    r = good_bank()
    del r["clip_ids"]
    with pytest.raises(dag.Refusal):
        dag.g_kb3(r, EVAL_IDS)
    orig = dag.clip_overlap                                                              # a BLIND disjointness check is itself refused by the reader's own regression
    dag.clip_overlap = lambda a, b: []
    try:
        with pytest.raises(dag.Refusal, match="blind"):
            dag.g_kb3(good_bank(), EVAL_IDS)
    finally:
        dag.clip_overlap = orig


def test_kb4_every_matcher_case_and_kb5_tolerance():
    for case in dag.KB4_CASES:
        for bad in (False, None):
            r = good_bank()
            if bad is None:
                del r["controls"]["K-B4"]["cases"][case]
            else:
                r["controls"]["K-B4"]["cases"][case] = bad
            with pytest.raises(dag.Refusal, match=case):
                dag.g_kb4(r)
    for k in ("max_pose_now_err_m", "max_rel_transform_err_m"):
        for v, ok in ((1e-4, True), (1.0001e-4, False), (float("nan"), False), (None, False), ("0", False)):
            r = good_bank()
            r["controls"]["K-B5"][k] = v
            if ok:
                assert dag.g_kb5(r)[0]
            else:
                with pytest.raises(dag.Refusal):
                    dag.g_kb5(r)


def test_bank_config_is_section_1():
    assert dag.g_bank_config(0, good_bank())[0] is True
    edits = {"seed": lambda r: r["header"].update(seed=1), "limit": lambda r: r["header"]["rollout"].update(limit="logged"), "replan": lambda r: r["header"]["rollout"].update(replan_steps=5),
             "executor": lambda r: r["header"]["rollout"].update(executor="gt"), "ticks": lambda r: r["header"]["rollout"].update(ticks=[0, 1]),
             "nav": lambda r: r["header"]["rollout"].update(nav="logged"), "consts": lambda r: r["header"].update(matching_constants_sha256="00" * 32),
             "hmax": lambda r: r["header"].update(hmax=8), "nwin": lambda r: r["header"].update(n_windows=1000), "nclips": lambda r: r["header"].update(n_clips=10),
             "ids": lambda r: r.update(clip_ids=r["clip_ids"][:-1]), "shaform": lambda r: r["header"].update(bank_sha256="xyz")}
    for nm, fn in edits.items():
        r = good_bank()
        fn(r)
        assert not dag.g_bank_config(0, r)[0], nm


def test_bank_accounting_and_coverage_arithmetic():
    ok, det = dag.g_bank_accounting(good_bank())
    assert ok and det["n_states_before_cuts"] == 2 * 306566 and det["eligible_fraction"] == pytest.approx(250000 / 306566)
    edits = {"before": lambda r: r["coverage"].update(n_states_before_cuts=2 * N_WIN - 1),
             "before_but_consistent_with_kept_and_cuts": lambda r: (r["coverage"].update(n_states_before_cuts=2 * N_WIN - 1), r["coverage"]["cuts"].update(end_edge=r["coverage"]["cuts"]["end_edge"] - 1)),
             "kept": lambda r: r["coverage"].update(n_states_kept=r["coverage"]["n_states_kept"] + 1),
             "cut": lambda r: r["coverage"]["cuts"].update(heading=r["coverage"]["cuts"]["heading"] + 1), "frac": lambda r: r["coverage"].update(eligible_fraction=0.9),
             "elig>kept": lambda r: r["coverage"].update(n_eligible_windows=r["coverage"]["n_states_kept"] + 1, eligible_fraction=(r["coverage"]["n_states_kept"] + 1) / N_WIN),
             "nwin": lambda r: r["coverage"].update(n_windows=N_WIN - 1)}
    for nm, fn in edits.items():
        r = good_bank()
        fn(r)
        assert not dag.g_bank_accounting(r)[0], nm
    for nm, fn in {"reason": lambda r: r["coverage"]["cuts"].pop("heading"), "neg": lambda r: r["coverage"]["cuts"].update(off_route=-1), "none": lambda r: r.pop("coverage"),
                   "nwin0": lambda r: r["coverage"].update(n_windows=0)}.items():
        r = good_bank()
        fn(r)
        with pytest.raises(dag.Refusal):
            dag.coverage_of(r)
    assert dag.coverage_of(good_bank(n_elig=153283))["eligible_fraction"] == 0.5 and dag.coverage_of(good_bank(n_elig=153282))["eligible_fraction"] < 0.5


def rec_with(sums):
    return fake_rec([], sums)


def test_bank_windows_trainer_sha_is_the_banks_and_hashes_the_clip_list():
    rep = good_bank()
    k = rec_with({"train_sha256": TRAIN_SHA})
    assert dag.g_bank_windows(0, rep, k, k, "train_sha256")[0] is True
    bad = rec_with({"train_sha256": "00" * 32})
    assert not dag.g_bank_windows(0, rep, bad, k, "train_sha256")[0] and not dag.g_bank_windows(0, rep, k, bad, "train_sha256")[0]
    ok, det = dag.g_bank_windows(0, rep, bad, bad, "train_sha256")                                                 # both arms agree with each other and disagree with the bank header
    assert not ok and "!= the trainer's" in str(det)
    assert not dag.g_bank_windows(0, rep, rec_with({}), k, "train_sha256")[0]                                    # not recorded
    r2 = good_bank()
    r2["clip_ids"] = r2["clip_ids"][:-1] + ["clip-extra"]                                                          # a header that no longer hashes its own clip list
    assert not dag.g_bank_windows(0, r2, k, k, "train_sha256")[0]
    r3 = good_bank()
    r3["header"]["window_list_sha256"] = "11" * 32
    assert not dag.g_bank_windows(0, r3, k, k, "train_sha256")[0]
    assert dag.g_bank_windows(0, rep, rec_with({"tr": TRAIN_SHA}), rec_with({"tr": TRAIN_SHA}), "tr")[0]          # the key is overridable


def test_bank_declared_argv_trainer_record_init_and_file(tmp_path):
    bsha, msha = "cd" * 32, "ab" * 32
    rep = good_bank()
    argv = ["--init-ckpt", "/nope/M2b_s0/ckpt_last.pt", "--dagger-bank-sha256", bsha]
    sums = {"levers": {"dagger_bank_sha256": bsha, "init_ckpt": {"sha256": msha}}}
    f = lambda r, rp=rep, bf=None: dag.g_bank_declared("t", 0, rp, r, msha, "init_ckpt.sha256", "dagger_bank_sha256", bf)
    assert f(fake_rec(argv, sums))[0] is True
    assert not f(fake_rec(["--init-ckpt", "/n", "--dagger-bank-sha256", "11" * 32], sums))[0]                         # argv declares another bank
    assert not f(fake_rec(argv, {**sums, "levers": {**sums["levers"], "dagger_bank_sha256": "11" * 32}}))[0]          # the trainer BUILT another bank
    assert not f(fake_rec(argv, {"levers": {"init_ckpt": {"sha256": msha}}}))[0]                                      # no built record
    assert not f(fake_rec(argv, {"levers": {"dagger_bank_sha256": bsha}}))[0]                                         # no init record
    assert not f(fake_rec(argv, {"levers": {"dagger_bank_sha256": bsha, "init_ckpt": {"sha256": "11" * 32}}}))[0]     # K-B2 on the arm: init is not the generator
    r2 = good_bank()
    r2["header"]["generator_ckpt_sha256"] = "11" * 32
    assert not f(fake_rec(argv, sums), rp=r2)[0]
    p = tmp_path / "bank.bin"
    p.write_bytes(b"x")
    assert not f(fake_rec(argv, sums), bf=str(p))[0]                                                                    # the file hashes to something else
    rep_f = good_bank()
    rep_f["header"]["bank_sha256"] = hashlib.sha256(b"x").hexdigest()
    argv_f = ["--init-ckpt", "/n", "--dagger-bank-sha256", hashlib.sha256(b"x").hexdigest()]
    sums_f = {"levers": {"dagger_bank_sha256": hashlib.sha256(b"x").hexdigest(), "init_ckpt": {"sha256": msha}}}
    ok, det = f(fake_rec(argv_f, sums_f), rp=rep_f, bf=str(p))
    assert ok and det["bank_file_hashed_here"] is True
    assert not f(fake_rec(argv_f, sums_f), rp=rep_f, bf=str(tmp_path / "missing.bin"))[0]
    ok, det = f(fake_rec(argv, sums))
    assert ok and "UNVERIFIED" in det["bank_file_note"]


# ======================================================================================================================== 8. COVERAGE -> BLOCKED, and the precedence REFUSED > BLOCKED > HARMS
def test_coverage_boundary_exactly_half_launches_one_window_short_is_blocked(tmp_path, capsys):
    rc, res, *_ = run(tmp_path / "ok", fxkw={"elig": 153283}, **STRONG)                           # 153283 / 306566 = 0.5 exactly: inclusive
    assert rc == 0, capsys.readouterr().err
    assert res["OUTCOME"]["OUTCOME"] == "ROBUST" and res["bank"]["s0"]["coverage"]["eligible_fraction"] == 0.5
    rc, res, out, _ = run(tmp_path / "blk", fxkw={"elig": 153282}, **STRONG)
    assert rc == 4 and res["OUTCOME"]["OUTCOME"] == "BLOCKED" and res["OUTCOME"]["blocked_seeds"] == ["s0", "s1"]


def test_world_BLOCKED_reports_the_stats_and_reads_no_arm(tmp_path, capsys):
    rc, res, out, fx = run(tmp_path, fxkw={"elig": {0: 250000, 1: 100000}}, **STRONG)
    assert rc == 4, capsys.readouterr().err                                                         # one seed below 0.5 blocks the whole arm pair
    b = res["OUTCOME"]
    assert b["OUTCOME"] == "BLOCKED" and b["blocked_seeds"] == ["s1"] and b["condition"] == prereg_table()[4][1] and b["committed_next"] == prereg_table()[4][2]
    assert b["coverage"]["s1"]["eligible_fraction"] == pytest.approx(100000 / 306566) and b["coverage"]["s0"]["n_windows"] == 306566
    assert set(b["coverage"]["s1"]["cuts"]) == {"off_route", "heading", "end_edge"} and b["coverage_min"] == 0.5
    assert res["rows"] == {} and res["runs"] == {} and b["matching_rows_in_table_order"] == ["BLOCKED"] and all(res["controls_PASS"].values())
    assert "were NOT read" in b["warning"] and b["unverified"]
    assert res["bank"]["s1"]["header"]["generator_ckpt_sha256"] == fx.m2b_sha[1]
    assert res["OUTCOME"]["OUTCOME"] != "ROBUST" and out.exists()


def test_BLOCKED_needs_no_arm_at_all(tmp_path, capsys):
    def e(fx):
        fx.manifest["arms"] = {}
    rc, res, *_ = run(tmp_path, edit=e, fxkw={"elig": 1000}, **STRONG)
    assert rc == 4 and res["OUTCOME"]["OUTCOME"] == "BLOCKED" and "warning" not in res["OUTCOME"], capsys.readouterr().err


def test_BLOCKED_never_reads_the_arm_files(tmp_path, capsys):
    rc, res, *_ = run(tmp_path, edit=lambda fx: (fx.root / "t0" / "eval_DAG0_s1.npz").unlink(), fxkw={"elig": 1000}, **STRONG)       # a missing arm file would REFUSE a read
    assert rc == 4 and res["OUTCOME"]["OUTCOME"] == "BLOCKED", capsys.readouterr().err


def test_REFUSED_beats_BLOCKED(tmp_path, capsys):
    def e(fx):
        jedit(fx.root / "bank" / "report_s0.json", lambda o: o["controls"]["K-B0"].update(PASS=False))
    rc, _, out, _ = run(tmp_path, edit=e, fxkw={"elig": 1000}, **STRONG)
    refused(capsys, rc, out, "K-B0:s0")


def test_HARMS_is_read_only_after_coverage_passes(tmp_path):
    rc, res, *_ = run(tmp_path, fxkw={"elig": 1000}, ade_shift=(0.3, 0.3), **STRONG)                 # a would-be HARMS world: coverage decides first
    assert rc == 4 and res["OUTCOME"]["OUTCOME"] == "BLOCKED"


def test_bank_stage_names_every_failed_bank_control(tmp_path, capsys):
    def keep(o):
        return o
    edits = chain(
        bank_edit(0, lambda o: o["controls"]["K-B0"].update(PASS=False)),
        bank_edit(0, lambda o: o["clip_ids"].__setitem__(5, EVAL_IDS[9])),                                  # an eval clip inside the bank
        bank_edit(0, lambda o: o["header"]["rollout"].update(limit="logged")),
        bank_edit(1, lambda o: o["controls"]["K-B1"].update(n_replayed=1)),
        bank_edit(1, lambda o: o["controls"]["K-B4"]["cases"].update(loop=False)),
        bank_edit(1, lambda o: o["controls"]["K-B5"].update(max_rel_transform_err_m=1e-3)),
        bank_edit(1, lambda o: o["coverage"].update(n_states_before_cuts=7)),
        bank_edit(1, lambda o: o["header"].update(generator_ckpt_sha256="11" * 32)),
        bank_edit(1, lambda o: o["controls"]["K-B3"].update(deliberate_regression_refused=False)),
    )
    rc, _, out, _ = run(tmp_path, edit=edits, extra={"--banked-sha": "1=" + "11" * 32}, **STRONG)
    refused(capsys, rc, out, "K-B0:s0", "K-B3:s0", "overlap", "BANK-CONFIG:s0", "header.rollout", "K-B1:s1", "K-B4:s1", "loop", "K-B5:s1", "BANK-ACCOUNTING:s1", "K-B2:s1", "K-B3:s1",
            "deliberate regression", "M2B-CKPT:s1",
            absent=("K-B1:s0", "K-B2:s0", "K-B4:s0", "K-B5:s0", "BANK-ACCOUNTING:s0", "BANK-CONFIG:s1", "K-B0:s1", "M2B-CKPT:s0", "ARGV:"))


def test_bank_stage_refuses_missing_inputs(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path / "a", edit=lambda fx: fx.manifest["bank"].pop("1"), **STRONG)
    refused(capsys, rc, out, "no bank for seed 1")
    rc, _, out, _ = run(tmp_path / "b", edit=lambda fx: (fx.root / "bank" / "report_s0.json").unlink(), **STRONG)
    refused(capsys, rc, out, "bank seed 0 report", "is missing")
    rc, _, out, _ = run(tmp_path / "c", edit=lambda fx: fx.manifest.pop("eval141_clip_ids"), **STRONG)
    refused(capsys, rc, out, "eval141_clip_ids")
    rc, _, out, _ = run(tmp_path / "d", edit=lambda fx: jedit(fx.root / "bank" / "report_s1.json", lambda o: o.update(kind="other")), **STRONG)
    refused(capsys, rc, out, "dagger-bank-report")
    rc, _, out, _ = run(tmp_path / "e", edit=lambda fx: jdump(fx.root / "eval141_clip_ids.json", []), **STRONG)
    refused(capsys, rc, out, "K-B3:s0", "empty")


# ======================================================================================================================== 9. a refusal per control, THROUGH the pipeline (rc 2, NOTHING written)
def test_refuse_every_arm_control_is_wired_into_the_read(tmp_path, capsys):
    """One world with a defect in EVERY arm-stage control family, spread over the six runs: the reader must name each one (and none of the healthy ones)."""
    fails = {("K1", "DAG_s0"), ("K2", "DAG0_s0"), ("K3", "DAG_s1"), ("K5", "DAG0_s1"), ("K6", "ORC0_s0"), ("K7", "DAG0_s0"), ("GRID", "DAG_s1"), ("K4", "DAG0_s1"),
             ("TICKMETA", "ORC0_s0")}
    w = World(ctl_fail=fails, **STRONG)
    edits = chain(
        edit_argv("DAG", 0, chain_f(drop_flag("--dagger-p"), set_flag("--steps", "2000"))),
        edit_argv("DAG", 1, chain_f(set_flag("--lr", "3e-4"), set_flag("--warmup", "200"), set_flag("--p-gt-start", "1.0"), set_flag("--p-gt-end", "0.1"), set_flag("--dagger-p", "0.3"))),
        edit_argv("DAG0", 0, chain_f(set_flag("--init-ckpt", "/x/M2b_s1/ckpt_last.pt"))),
        edit_argv("DAG0", 1, lambda a: a.extend(["--threads", "4"])),
        lambda fx: jedit(fx.root / "runs" / "DAG_s1" / "summary.json", lambda o: o.update(done=False, dagger_n=5000)),
        lambda fx: jedit(fx.root / "runs" / "DAG0_s0" / "selfcheck.json", lambda o: o.update(declared_vs_built="FAIL")),
        lambda fx: jedit(fx.root / "runs" / "DAG_s0" / "summary.json", lambda o: o["levers"]["init_ckpt"].update(sha256="00" * 32)),
        lambda fx: jedit(fx.root / "runs" / "DAG0_s1" / "summary.json", lambda o: o.update(dagger_n=1)),
        lambda fx: jedit(fx.root / "tokens" / "PASS_DAG_s1.json", lambda o: o.update(verdict="FAIL")),
        lambda fx: jedit(fx.root / "t1" / "t1v8f_DAG_s0_causal.json", lambda o: o["t1v8f"].update(limit="logged")),
        lambda fx: jedit(fx.root / "t1" / "t1v8f_ORC0_s1_causal.json", lambda o: o["t1v8f"].update(ckpt="/x/runs/DAG_s1/ckpt_last.pt")),
        lambda fx: jedit(fx.root / "t0" / "eval_DAG_s1.json", lambda o: o.update(ckpt="/x/runs/DAG0_s1/ckpt_last.pt")),
        lambda fx: jedit(fx.root / "t1" / "t1v8f_DAG0_s0_causal.json", lambda o: o["paired_decision_grade"]["paired_model_minus_floor_t1"]["ade_m"].update(delta=9.9)),
    )
    rc, _, out, _ = run(tmp_path, world=w, edit=edits)
    refused(capsys, rc, out,
            "K1:DAG_s0", "K2:DAG0_s0", "K3:DAG_s1", "K5:DAG0_s1", "K6:ORC0_s0", "K7:DAG0_s0", "GRID:DAG_s1", "K4:DAG0_s1",
            "ARGV:DAG_s0", "--dagger-p is not named", "--steps 2000", "ARGV:DAG_s1", "--lr 3e-4", "--warmup 200", "--p-gt-start 1.0", "--p-gt-end 0.1", "--dagger-p 0.3, registered 0.1",
            "ARGV:DAG0_s0", "M2b_s0/ckpt_last.pt", "PAIR:s0", "PAIR:s1", "not one lever", "ORC0-REPLICATE:s0", "ORC0-REPLICATE:s1", "not ORC-0's plus --dagger-p 0",
            "SUMMARY:DAG_s1", "done is not true", "SUMMARY:DAG0_s0", "declared_vs_built",
            "INIT-CKPT:DAG_s0", "differs from the M2b checkpoint", "BANK-DECLARED:DAG_s0", "K-B2: ", "DAGGER-REACHED:DAG_s1", "outside 0.1 +- 0.02", "DAGGER-REACHED:DAG0_s1", "must be exactly 0",
            "GATE-TOKEN:DAG_s1", "not PASS", "T1-CONFIG:DAG_s0", "T1-CONFIG:ORC0_s1", "T1-CONFIG:ORC0_s0", "the ticks file says",
            "T0-CKPT:DAG_s1", "R2-ADE6:DAG0_s0",
            absent=("ARGV:DAG0_s1", "SUMMARY:DAG_s0", "SUMMARY:DAG0_s1", "GATE-TOKEN:DAG_s0", "GATE-TOKEN:DAG0_s1", "T1-CONFIG:DAG_s1", "T0-CKPT:DAG_s0", "M2B-CKPT",
                    "R2-ADE6:DAG_s0", "K4:DAG_s0", "BANK-WINDOWS", "K-B0:", "K-B1:", "K-B3:", "K-B4:", "K-B5:", "BANK-CONFIG", "BANK-ACCOUNTING", "DAGGER-REACHED:DAG_s0", "DAGGER-REACHED:DAG0_s0", "SAME-WINDOWS"))


@pytest.mark.parametrize("name,edit,needles,absent", [
    ("bank_sha_declared", edit_argv("DAG", 0, set_flag("--dagger-bank-sha256", "11" * 32)), ("BANK-DECLARED:DAG_s0", "!= the bank report's bank_sha256"), ("BANK-DECLARED:DAG_s1",)),
    ("bank_built_sha", lambda fx: jedit(fx.root / "runs" / "DAG_s1" / "summary.json", lambda o: o["levers"].update(dagger_bank_sha256="22" * 32)),
     ("BANK-DECLARED:DAG_s1", "bank report's bank_sha256"), ("BANK-DECLARED:DAG_s0",)),
    ("bank_file_corrupt", lambda fx: (fx.root / "bank" / "bank_s0.bin").write_bytes(b"corrupted"), ("BANK-DECLARED:DAG_s0", "hashes to"), ("BANK-DECLARED:DAG_s1",)),
    ("trainer_window_sha", lambda fx: jedit(fx.root / "runs" / "DAG0_s1" / "summary.json", lambda o: o.update(train_sha256="33" * 32)),
     ("BANK-WINDOWS:s1", "PAIR:s1", "ORC0-REPLICATE:s1"), ("BANK-WINDOWS:s0",)),
    ("bank_header_window_sha", bank_edit(0, lambda o: o["header"].update(window_list_sha256="44" * 32)), ("BANK-WINDOWS:s0", "window_list_sha256"), ("BANK-WINDOWS:s1",)),
    ("orc0_names_dagger_p", edit_argv("ORC0", 1, lambda a: a.extend(["--dagger-p", "0"])), ("ORC0-REPLICATE:s1", "already names --dagger-p"), ("ORC0-REPLICATE:s0",)),
    ("orc0_other_flag", edit_argv("ORC0", 0, set_flag("--lr", "3e-4")), ("ORC0-REPLICATE:s0", "differs in"), ("ORC0-REPLICATE:s1",)),
    ("orc0_other_init", lambda fx: jedit(fx.root / "runs" / "ORC0_s0" / "summary.json", lambda o: o["levers"]["init_ckpt"].update(sha256="55" * 32)),
     ("INIT-CKPT:ORC0_s0", "ORC0-REPLICATE:s0"), ("INIT-CKPT:ORC0_s1",)),
    ("orc0_missing", lambda fx: fx.manifest["arms"]["ORC0"].pop("1"), ("manifest has no ORC0 seed 1",), ()),
    ("dag_and_ego_perturb_combined", edit_argv("DAG", 1, lambda a: a.__setitem__(a.index("--ego-perturb-p") + 1, "0.3")), ("ARGV:DAG_s1", "--ego-perturb-p 0.3, registered 0.0", "PAIR:s1"), ("ARGV:DAG_s0",)),
])
def test_refuse_arm_stage_bank_and_replicate_controls(tmp_path, capsys, name, edit, needles, absent):
    rc, _, out, _ = run(tmp_path, edit=edit, **STRONG)
    refused(capsys, rc, out, *needles, absent=absent)


def test_gate_tokens_of_the_pair_may_differ_in_data_manifest(tmp_path, capsys):
    rc, res, *_ = run(tmp_path, **STRONG)
    assert rc == 0, capsys.readouterr().err
    assert res["controls"]["GATE-TOKEN:DAG_s0"]["detail"]["data_sha256"] != res["controls"]["GATE-TOKEN:DAG0_s0"]["detail"]["data_sha256"]
    assert res["controls_PASS"]["PAIR:s0"]


def test_refuse_the_other_arms_checkpoint_and_swapped_runs(tmp_path, capsys):      # DAG and DAG-0 swapped
    rc, _, out, _ = run(tmp_path, edit=lambda fx: jedit(fx.root / "t1" / "t1v8f_DAG_s0_causal.json", lambda o: o["t1v8f"].update(ckpt="/x/runs/DAG0_s0/ckpt_last.pt")), **STRONG)
    refused(capsys, rc, out, "T1-CONFIG:DAG_s0", "DAG0_s0/ckpt_last.pt")


def test_refuse_pairing_by_different_windows(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, world=World(permute_dag=True, **STRONG))
    refused(capsys, rc, out, "SAME-WINDOWS:DAG-vs-DAG0:s0", "SAME-WINDOWS:DAG-vs-DAG0:s1", absent=("DUMP-WINDOWS",))     # ticks, dumps and T0 agree with each other, not with DAG-0's


def test_refuse_the_replicate_on_other_windows(tmp_path, capsys):
    def e(fx):
        p = fx.root / "t0" / "eval_ORC0_s1.npz"
        z = dict(np.load(p))
        z["eid"] = np.roll(z["eid"], 7)
        np.savez(p, **z)
    rc, _, out, _ = run(tmp_path, edit=e, **STRONG)
    refused(capsys, rc, out, "SAME-WINDOWS:ORC0-vs-DAG0:s1", absent=("SAME-WINDOWS:ORC0-vs-DAG0:s0", "SAME-WINDOWS:DAG-vs"))


def test_refuse_dump_windows_not_the_ticks_windows(tmp_path, capsys):
    def e(fx):
        d = fx.root / "t1" / "dump_DAG_s0_causal"
        f = sorted(d.glob("ep*.npz"))[0]
        z = dict(np.load(f))
        z["ws"] = z["ws"][::-1].copy()
        np.savez(f, **z)
    rc, _, out, _ = run(tmp_path, edit=e, **STRONG)
    refused(capsys, rc, out, "DUMP-WINDOWS:DAG_s0", "SAME-WINDOWS")


def test_refuse_t0_episodes_differ_and_grid_counts(tmp_path, capsys):
    def e(fx):
        p = fx.root / "t0" / "eval_DAG_s0.npz"
        z = dict(np.load(p))
        z["eid"] = np.roll(z["eid"], 7)
        np.savez(p, **z)
    rc, _, out, _ = run(tmp_path, edit=e, extra={"--n-windows": 2399}, **STRONG)
    refused(capsys, rc, out, "SAME-WINDOWS:DAG-vs-DAG0:s0", "GRID-COUNTS")


def test_refuse_one_seed_only(tmp_path, capsys):
    def e(fx):
        del fx.manifest["arms"]["DAG"]["1"]
    rc, _, out, _ = run(tmp_path, edit=e, **STRONG)
    refused(capsys, rc, out, "BOTH seeds")


def test_refuse_missing_file_is_a_refusal_not_a_traceback(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, edit=lambda fx: (fx.root / "t0" / "eval_DAG0_s1.npz").unlink(), **STRONG)
    refused(capsys, rc, out, "is missing")


def test_refuse_gate_token_missing_file(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, edit=lambda fx: (fx.root / "tokens" / "PASS_DAG_s1.json").unlink(), **STRONG)
    refused(capsys, rc, out, "launch-gate token", "is missing")


def test_refuse_prereg_text_mismatch(tmp_path, capsys):
    p = tmp_path / "PREREG_changed.md"
    p.write_text(PREREG.read_text(encoding="utf-8").replace("adopt for the head that moved", "adopt for any head"), encoding="utf-8")
    rc, _, out, _ = run(tmp_path, extra={"--prereg": str(p)}, **STRONG)
    refused(capsys, rc, out, "PREREG-TEXT", "PARTIAL")
    p2 = tmp_path / "PREREG_changed2.md"
    p2.write_text(PREREG.read_text(encoding="utf-8").replace("no launch. The stats are reported.", "launch anyway."), encoding="utf-8")
    rc, _, out, _ = run(tmp_path / "b", extra={"--prereg": str(p2)}, **STRONG)
    refused(capsys, rc, out, "PREREG-TEXT", "BLOCKED")


def test_K_control_on_M2b_ORC_beside_runs_and_logged_source_run(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, world=World(ctl_fail={("K6", "M2b_s1"), ("K3L", "DAG_s0"), ("K2", "ORC_s0")}, **STRONG), m2b=True, orc=True, logged=True)
    err = refused(capsys, rc, out, "K6:M2b_s1", "K3-logged:DAG_s0", "K2:ORC_s0")
    assert "K3:DAG_s0" not in err


def test_trainer_key_names_are_cli_overridable(tmp_path, capsys):
    def e(fx):
        for arm in ("DAG", "DAG0"):
            for s in (0, 1):
                jedit(fx.root / "runs" / f"{arm}_s{s}" / "summary.json", lambda o: o.__setitem__("totals", {"rep_n": o.pop("dagger_n"), "rep_elig": o.pop("dagger_eligible")}))
        for s in (0, 1):
            jedit(fx.root / "runs" / f"DAG_s{s}" / "summary.json", lambda o: o["levers"].update(bank_sha=o["levers"].pop("dagger_bank_sha256")))
            for arm in ("DAG", "DAG0", "ORC0"):
                jedit(fx.root / "runs" / f"{arm}_s{s}" / "summary.json", lambda o: o.update(clip_list_sha=o.pop("train_sha256")))
    rc, res, *_ = run(tmp_path, edit=e, extra={"--key-dagger-n": "rep_n", "--key-dagger-eligible": "rep_elig", "--key-dagger-bank-sha": "bank_sha", "--key-window-sha": "clip_list_sha"}, **STRONG)
    assert rc == 0 and res["OUTCOME"]["OUTCOME"] == "ROBUST", capsys.readouterr().err
    rc, _, out, _ = run(tmp_path / "b", edit=e, name="o2.json", **STRONG)                    # without the overrides the default keys are not found
    refused(capsys, rc, out, "DAGGER-REACHED", "dagger_n")


def test_failed_control_computes_no_rows(tmp_path, capsys):
    called = {"n": 0}
    real = dag.f1_pair_row

    def spy(*a, **k):
        called["n"] += 1
        return real(*a, **k)
    dag.f1_pair_row = spy
    try:
        rc, _, out, _ = run(tmp_path, world=World(ctl_fail={("K3", "DAG_s0")}, **STRONG))
    finally:
        dag.f1_pair_row = real
    assert rc == 2 and called["n"] == 0 and not out.exists()


def test_strategic_label_and_tier_strings_are_pinned(tmp_path):
    assert "T1-kin: self-action OPEN loop" in dag.TIER and "never 'closed loop'" in dag.TIER and "T0 (teacher-forced own-code, eval-141) for R-T0" in dag.TIER
    rc, res, *_ = run(tmp_path, **STRONG)
    assert res["strategic_family"] == "NOT PAUSED for v8F (PI ruling 2026-10-17, R73): reported BESIDE as S-POS@30/50/80 vs CTRV per run (unpaired); decides nothing"
    assert res["tier"] == dag.TIER and "PAUSED" not in res["strategic_family"].replace("NOT PAUSED", "")
