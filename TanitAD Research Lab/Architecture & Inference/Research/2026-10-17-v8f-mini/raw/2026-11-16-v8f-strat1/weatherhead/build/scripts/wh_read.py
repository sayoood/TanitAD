"""wh_read.py -- the READ of PREREG_V8F_WEATHERHEAD form (a) (card sections 6 and 7).  Tier T0 (a per-frame perception output at the window's NOW frame); evidence class MEASURED (ours; zero training of any driving parameter).

Estimator: ``taniteval.ci.paired_episode_cluster_bootstrap``, B 2,000, seed 0, over the 141 eval episodes: it answers "another draw of EPISODES" only.  "Another TRAINING RUN" is answered by the two S1 seeds
(two trainings, two adapters), "another FIT" by F_fit (one replicate pair, a weak floor, stated).  "Another INFERENCE RUN" does not arise (deterministic head and planner, one environment, one thread count).

Reuses, does not rewrite, ``strat1_ctx_read.py`` (sha256 1b9377ba...): label_rows, split_logits, read_arm (the DETECTS rule), the K2 / K3 / K5 / K6 controls and the constants.  The rows of S1's banked ``ctx_logits``
are re-derived here and compared with ``STRAT1_CTX.json`` (K0, exact).  An H dump WITHOUT ``ctx_pool_logits`` reads "n/a, no output": the reader NEVER falls back to ``ctx_logits`` (mutation m6 -> K-ID RED).

Outcome rules (card section 7) are code here; ``ctx_output_map.json`` is written by this file from them, never by hand.
Usage: python wh_read.py --t0 DIR --out OUTDIR [--kid DIR] [--instrument FILE.json] [--smoke]"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wh_lib as W  # noqa: E402

os.environ.setdefault("STRAT1_TREE", W.TREE)
sys.path.insert(0, W.READ)
import strat1_lib as L  # noqa: E402
import strat1_ctx_read as R  # noqa: E402

M = W.M
VARS = M.CTX_VARS
SEEDS = (0, 1)
PROBE_NPZ = "F:/TanitAD-scratch/strat1/weatherprobe/wprobe_out.npz"
BANKED = W.READ + "/out"
HEAD_ARMS = ("H_s0", "H_s1", "RA_s0", "RA_s1", "R1_s0", "R1_s1", "H_s0_rep", "K4p_s0", "K4p_s1", "FEAT_s0", "FEAT_s1", "C0A_s0", "C0A_s1")
INSTRUMENT_REQUIRED = ("K0", "K2", "K3", "K4p", "K5", "K6", "K-ID", "R1", "T-ISO-a", "T-ISO-b", "T-ISO-c", "T-ISO-d", "T-VIS", "REACH-live", "REACH-census", "DISJOINT", "G-EVAL")
FORBIDDEN_WORDS = ("void", "superseded", "retracted", "formerly", "correction", "withdrawn", "no longer", "closed loop")


# ------------------------------------------------------------------------------------------------------------------------------ inputs
def head_logits(path):
    """ctx_pool_logits [N, 41] of an eval npz, or None (= no output).  NEVER ``ctx_logits``."""
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    return z["ctx_pool_logits"] if "ctx_pool_logits" in z.files else None


def shipped_logits(t0, s):
    z = np.load(f"{BANKED}/t0/eval_STRAT1-S1_s{s}_real.npz", allow_pickle=True)
    return z["ctx_logits"]


class World:
    """the eval grid, labels, constants and the rig baseline (everything the reader scores against)"""

    def __init__(self, nb=L.NB, grid=None):
        self.nb, self.CI = nb, L.load_ci(W.TREE)
        self.join = dict(np.load(L.pinned("join_eval141.npz")))
        self.tab = dict(np.load(L.pinned("strat_labels_table.npz")))
        self.plat = dict(np.load(L.INP + "/strat_platform_by_clip.npz"))
        self.ci, self.t = L.grid_windows(grid or L.pinned("eval_grid_windows.json"), self.join["clip"])
        R.check_alignment(self.tab, self.plat, self.join)
        self.r = L.rows_of(self.tab["eval_off"], self.ci, self.t)
        self.y = R.label_rows(self.tab, self.r)
        self.eid = self.ci
        self.consts = R.table_baselines(self.tab)
        self.rig_pred, self.rig_map = R.rig_baseline(self.tab, self.plat, self.ci)
        self.plat_w = np.asarray(self.plat["eval_platform_id"], np.int64)[self.ci]

    def rows(self, logits):
        return R.read_arm(self.CI, logits, self.y, self.eid, self.consts, self.rig_pred, self.plat_w, self.nb)

    def correct(self, logits):
        """[N, 5] float: 1 / 0 where the window is scorable for the variable, NaN elsewhere"""
        parts = R.split_logits(logits)
        out = np.full((len(self.y), 5), np.nan)
        for k in range(5):
            m = self.y[:, k] >= 0
            out[m, k] = (parts[k].argmax(1)[m] == self.y[m, k]).astype(np.float64)
        return out

    def paired(self, ca, cb, k):
        return L.paired(self.CI, ca[:, k], cb[:, k], self.eid, self.nb)


# ------------------------------------------------------------------------------------------------------------------------------ K0 and the controls
def _floats_equal(a, b, path=""):
    """exact (==) comparison of two JSON-like trees; -> list of differing paths"""
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ"]
        for k in a:
            bad += _floats_equal(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length"]
        for i, (x, y) in enumerate(zip(a, b)):
            bad += _floats_equal(x, y, f"{path}/{i}")
    elif a != b:
        bad.append(f"{path}: {a!r} != {b!r}")
    return bad


def k0(world, ctx_json=None, shipped=None):
    """K0 anchor: this reader on S1's banked ``ctx_logits`` reproduces EVERY row of ``STRAT1_CTX.json`` (accuracy, constant, delta, CI bounds, confusion ...) equal as floats.  -> record"""
    ref = json.load(open(ctx_json or f"{BANKED}/STRAT1_CTX.json", encoding="utf-8"))
    if world.nb != L.NB:
        return {"pass": False, "n_fields_compared": 0, "differences": [f"K0 anchors at B={L.NB}; this world has B={world.nb}"], "reference": "STRAT1_CTX.json"}
    bad, n = [], 0
    for s in SEEDS:
        lg = shipped(s) if shipped else shipped_logits(None, s)
        mine = json.loads(json.dumps(world.rows(lg), default=float))
        want = {v: {k: x for k, x in ref["arms"][f"S1_s{s}"]["rows"][v].items()} for v in VARS}
        bad += [f"S1_s{s}{p}" for p in _floats_equal(mine, want)]
        n += sum(len(r) for r in mine.values())
    return {"pass": not bad, "n_fields_compared": n, "differences": bad[:10], "reference": "STRAT1_CTX.json rows of S1_s0 and S1_s1"}


def controls(world, nb=300):
    c = R.controls(world.CI, world.y, world.eid, world.consts, world.rig_pred, world.plat_w, nb)
    return {"K2": {"pass": all(d["pass"] for d in c["K2"].values()), "rows": c["K2"]}, "K3": {"pass": all(d["pass"] for d in c["K3"].values()), "rows": c["K3"]},
            "K5": {"pass": all(d["pass"] for d in c["K5"].values()), "rows": c["K5"]}, "K6": {**c["K6"], "must_read": 0.0779, "pass": bool(c["K6"]["pass"] and abs(c["K6"]["read"] - 0.0779) < 5e-4)}}


def k_id(eval_npz, kid_npz, atol=1e-5):
    """K-ID: the evaluator's ``ctx_pool_logits`` equal the fit script's on every window (atol 1e-5)"""
    a, b = head_logits(eval_npz), (np.load(kid_npz, allow_pickle=True) if os.path.exists(kid_npz) else None)
    if a is None or b is None:
        return {"pass": False, "reason": "no output" if a is None else "no fit-side logits"}
    z = np.load(eval_npz, allow_pickle=True)
    if not (np.array_equal(z["win_ci"], b["win_ci"]) and np.array_equal(z["win_t"], b["win_t"])):
        return {"pass": False, "reason": "windows differ"}
    d = float(np.abs(a - b["ctx_pool_logits"]).max())
    return {"pass": bool(d <= atol), "max_abs_diff": d, "atol": atol, "n_windows": int(len(a))}


def window_check(eval_npz, world):
    z = np.load(eval_npz, allow_pickle=True)
    if not (np.array_equal(z["win_ci"], world.ci) and np.array_equal(z["win_t"], world.t)):
        raise RuntimeError(f"{eval_npz}: the evaluator's windows differ from the grid")


# ------------------------------------------------------------------------------------------------------------------------------ the outcome rules (section 7)
def sep_pos(r):
    return bool(r and r.get("status") == "OK" and r["separated"] and r["delta"] > 0)


def sep_neg(r):
    return bool(r and r.get("status") == "OK" and r["separated"] and r["delta"] < 0)


def detects_pair(rows, v):
    """DETECTS(v) of one arm; country also BEYOND-RIG"""
    r = rows[v]
    return bool(r["detects"]) and (bool(r.get("detects_beyond_rig")) if v == "country" else True)


def decide(A, C):
    """A: {arm: {'rows': {v: row}, 'correct': [N,5]}} plus C: {'pair': {(name, seed, v): paired-dict}, 'F_fit': {v: float}, 'instrument': {name: 'pass'|'fail'|'not-run'}}.
    -> the per-variable outcome table.  Pure function of its inputs (unit-tested with planted arms)."""
    ins = C["instrument"]
    fail = [k for k in INSTRUMENT_REQUIRED if ins.get(k) == "fail"]
    missing = [k for k in INSTRUMENT_REQUIRED if ins.get(k, "not-run") == "not-run"]
    if fail:
        return {"__all__": "INSTRUMENT-FAIL", "failed": fail, "reason": "no number is quoted; the fault is found and fixed, the failing stage re-runs, no bar changes"}
    if missing:
        return {"__all__": "INSTRUMENT-INCOMPLETE", "not_run": missing, "reason": "an outcome is not quoted before the instrument stage has been read"}
    out, F = {}, C["F_fit"]
    pair = C["pair"]
    for v in VARS:
        h = [A.get(f"H_s{s}") for s in SEEDS]
        k4 = [A.get(f"K4p_s{s}") for s in SEEDS]
        if any(x is None for x in h):
            out[v] = {"outcome": "NOT-YET", "reason": "an H arm has no ctx_pool_logits (n/a, no output)"}
            continue
        det = [detects_pair(x["rows"], v) for x in h]
        k4det = [bool(x is not None and detects_pair(x["rows"], v)) for x in k4]
        detects = all(det) and not any(k4det)
        o = {"detects_by_seed": det, "k4p_detects_by_seed": k4det}
        if v == "weather":
            if detects:
                o["outcome"] = "WH-DETECTS"
                ra = [pair.get(("H-RA", s, v)) for s in SEEDS]
                ff = F.get(v)
                if ff is None or any(r is None for r in ra):
                    o["wording"] = "pending: the RA arms or the fit replicate (F_fit) have not run"
                else:
                    adds = all(sep_pos(r) for r in ra) and all(abs(r["delta"]) > ff for r in ra)
                    o["adapter_adds"] = bool(adds)
                    o["outcome"] += " + WH-ADAPTER-ADDS" if adds else " + WH-INPUT-FEATURES"
                    o["wording"] = ("S1's adapter makes weather more readable than a random projection of the same cached features" if adds else
                                    "weather is readable from the model's vision input features (cached DINOv3 4x10 tokens over 4 frames) by the shipped pooled head; S1's training is not shown to add to it")
                o["next"] = "weather is a reported model output from ctx_pool_head.weather; then the label lever (Q3), the re-fit on the adopted driving checkpoint, the derived-goals card (its own goal-input admissibility)"
            else:
                o["outcome"] = "WH-NOT-DETECTED"
                feat = [A.get(f"FEAT_s{s}") for s in SEEDS]
                fd = None if any(x is None for x in feat) else all(detects_pair(x["rows"], v) for x in feat)
                o["feat_detects_both_seeds"] = fd
                o["next"] = ("a-W: a weather-only sub-head with the probe recipe exactly (weather-valid windows only, E grid {2, 4, 8}), same seeds, controls and reads; if a-W DETECTS weather ships from a-W and the cause (window set or E grid) is named; "
                             "if a-W is NOT-DETECTED the FEAT rows decide: " + ("FEAT arms not run" if fd is None else ("FEAT detects at both fit seeds -> the frozen S1 adapter is the bottleneck -> form (b) (PI go Q1)" if fd else
                              "FEAT does not detect -> the representation lever card (frozen-feature pooled head on pool8x20, then the 416x1024 encode, PI go)")))
        else:
            sh = [pair.get(("H-shipped", s, v)) for s in SEEDS]
            if detects and all(r is not None for r in sh):
                replaces = not any(sep_neg(r) for r in sh)
                ff = F.get(v)
                gain = replaces and all(sep_pos(r) for r in sh) and ff is not None and all(abs(r["delta"]) > ff for r in sh)
                o["outcome"] = (f"POOL-GAIN({v})" if gain else (f"POOL-REPLACES({v})" if replaces else f"SHIPPED-KEPT({v})"))
                if replaces and not gain and ff is None:
                    o["note"] = "POOL-GAIN not decidable: the fit replicate (F_fit) has not run"
            else:
                o["outcome"] = f"SHIPPED-KEPT({v})"
                o["note"] = "pooled head does not detect" if not detects else "H - shipped rows missing"
        out[v] = o
    out["driving"] = C.get("driving", {"outcome": "NOT-YET", "reason": "T-ISO (d) dumps not read"})
    return out


def output_map(outcomes):
    """ctx_output_map.json: per variable the source it is reported from -- written from the section 7 rules, never by hand"""
    m = {}
    for v in VARS:
        o = outcomes.get(v, {})
        name = o.get("outcome", "NOT-YET") if isinstance(o, dict) else str(o)
        if name.startswith("WH-DETECTS"):
            src = f"ctx_pool_head.{v}"
        elif name.startswith(("POOL-REPLACES", "POOL-GAIN")):
            src = f"ctx_pool_head.{v}"
        elif name.startswith("SHIPPED-KEPT"):
            src = "ctx_head (shipped); the pooled sub-head is reported beside"
        elif name.startswith("WH-NOT-DETECTED"):
            src = "none (weather is not detected by the pooled head; the next arm is named in the read)"
        else:
            src = "undecided"
        m[v] = {"outcome": name, "source": src}
    return m


# ------------------------------------------------------------------------------------------------------------------------------ driver
def read_all(world, t0, kid_dir="", instrument=None, smoke=False):
    rep = {"tool": "wh_read.py", "SMOKE": bool(smoke), "result": False if smoke else None, "tier": "T0 (per-frame perception output at the window's NOW frame)", "evidence_class": "MEASURED (ours; zero training of any driving parameter)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={world.nb}, seed 0 ({world.CI.SOURCE}); 'another draw of EPISODES' only", "arms": {}, "n/a": {}, "problems": []}
    A = {}
    for name in HEAD_ARMS:
        p = f"{t0}/eval_WH-{name}_real.npz"
        lg = head_logits(p)
        if lg is None:
            rep["n/a"][name] = "no output" if os.path.exists(p) else "not run"
            continue
        window_check(p, world)
        rows = world.rows(lg)
        A[name] = {"rows": rows, "correct": world.correct(lg), "logits": lg}
        rep["arms"][name] = {"rows": rows}
    S = {s: {"correct": world.correct(shipped_logits(t0, s))} for s in SEEDS}
    pair, label_ceiling = {}, {"weather": "36/40 = 90.0 % [76.9, 96.0]", "surface": "32/40", "illumination": "30/39 (PI frames, INHERITED)"}
    for s in SEEDS:
        h = A.get(f"H_s{s}")
        if h is None:
            continue
        for k, v in enumerate(VARS):
            pair[("H-shipped", s, v)] = world.paired(h["correct"], S[s]["correct"], k)
            for other, nm in (("RA", "H-RA"), ("FEAT", "H-FEAT"), ("C0A", "H-C0A")):
                o = A.get(f"{other}_s{s}")
                if o is not None:
                    pair[(nm, s, v)] = world.paired(h["correct"], o["correct"], k)
    probe = np.load(PROBE_NPZ, allow_pickle=True) if os.path.exists(PROBE_NPZ) else None
    if probe is not None:
        for s in SEEDS:
            h = A.get(f"H_s{s}")
            if h is not None and np.array_equal(probe["eval_ci"], world.ci) and np.array_equal(probe["eval_t"], world.t):
                pc = np.full((len(world.y), 5), np.nan)
                m = world.y[:, 1] >= 0
                pc[m, 1] = (probe[f"S1_s{s}_real"].argmax(1)[m] == world.y[m, 1]).astype(float)
                pair[("H-probe", s, "weather")] = world.paired(h["correct"], pc, 1)
    F = {}
    if "H_s0" in A and "H_s0_rep" in A:
        for k, v in enumerate(VARS):
            F[v] = abs(A["H_s0_rep"]["rows"][v]["accuracy"] - A["H_s0"]["rows"][v]["accuracy"])
    ins = dict(instrument or {})
    rep["K0"] = k0(world)
    ins["K0"] = "pass" if rep["K0"]["pass"] else "fail"
    c = controls(world)
    rep["controls"] = c
    for k in ("K2", "K3", "K5", "K6"):
        ins[k] = "pass" if c[k]["pass"] else "fail"
    k4 = [A.get(f"K4p_s{s}") for s in SEEDS]
    ins["K4p"] = "not-run" if any(x is None for x in k4) else ("pass" if not any(detects_pair(x["rows"], v) for x in k4 for v in VARS) else "fail")
    r1 = [A.get(f"R1_s{s}") for s in SEEDS]
    ins["R1"] = "not-run" if any(x is None for x in r1) else ("pass" if not any(detects_pair(x["rows"], v) for x in r1 for v in VARS) else "fail")
    kid = {}
    for s in SEEDS:
        if kid_dir and os.path.exists(f"{t0}/eval_WH-H_s{s}_real.npz"):
            kid[f"H_s{s}"] = k_id(f"{t0}/eval_WH-H_s{s}_real.npz", f"{kid_dir}/kid_H_s{s}.npz")
    ins["K-ID"] = "not-run" if len(kid) < 2 else ("pass" if all(x["pass"] for x in kid.values()) else "fail")
    rep["K-ID"] = kid
    rep["instrument"] = ins
    rep["F_fit"] = F
    rep["paired"] = {f"{a}|s{s}|{v}": d for (a, s, v), d in pair.items()}
    rep["label_ceiling_beside"] = label_ceiling
    rep["outcomes"] = decide(A, {"pair": pair, "F_fit": F, "instrument": ins, "driving": (instrument or {}).get("_driving")})
    if rep["outcomes"].get("__all__") == "INSTRUMENT-FAIL":                      # card 7: no number is quoted
        rep["UNQUOTABLE_arms_kept_for_the_fault_search_only"] = rep.pop("arms")
        rep["arms"] = {"withheld": "INSTRUMENT-FAIL"}
        rep["paired"] = {"withheld": "INSTRUMENT-FAIL"}
    rep["ctx_output_map"] = output_map(rep["outcomes"]) if "__all__" not in rep["outcomes"] else {"__all__": rep["outcomes"]["__all__"]}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kid", default="")
    ap.add_argument("--instrument", default="", help="the JSON the checker / iso tools wrote: {check: pass|fail|not-run, '_driving': {...}}")
    ap.add_argument("--grid", default="")
    ap.add_argument("--n-boot", type=int, default=L.NB)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w = World(a.n_boot, a.grid or None)
    ins = json.load(open(a.instrument, encoding="utf-8")) if a.instrument else {}
    rep = read_all(w, a.t0, a.kid, ins, a.smoke)
    json.dump(rep, open(f"{a.out}/WH_READ.json", "w", encoding="utf-8"), indent=1, default=float)
    json.dump(rep["ctx_output_map"], open(f"{a.out}/ctx_output_map.json", "w", encoding="utf-8"), indent=1)
    print("WROTE", a.out, "|", json.dumps({k: (v.get("outcome") if isinstance(v, dict) and "outcome" in v else v) for k, v in rep["outcomes"].items()})[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
