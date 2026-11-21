r"""s0l_read.py -- STOPPERC S0-L read (NOT RUN: L_s0 is not banked).  Card: theory/stopperc/PREREG_V8F_STOPPERC.md section 4, the "S0-L" line:
  "STOPCUE on L vs B-prime s0, both dumped T1-kin through their own caches. A gain counts only if it exceeds the AUROC floor of 0.037."

WHAT IT READS.  Two STOPDEC npz files (the per-window files stopdec_arm.py writes from an arm's T1-kin dumps; the file format STOPCUE reads), one for L, one for B-prime s0.
STOPCUE's statistic, unchanged (theory/stopcue/stopcue_read.py read_arm, lines 69-97): over the windows with `has` True, the AUROC of p(STOP_POINT) = goal_prob[:, 6]
against the label `stop_hit` (Mann-Whitney, ties one half).  The paired difference dAUC = AUROC(L) - AUROC(B-prime s0) comes from taniteval.ci.paired_stat_cluster_bootstrap
(episodes resampled, the SAME draw for both arms, B 2,000, seed 0: the estimator of every STOPPERC read).  Everything numeric is IMPORTED from the frozen read code
(stopperc_s1_read.auroc_fast / _auc_stat / cue_arrays, stopbal_read constants, taniteval.ci); nothing is copy-modified.

THE RULE (AMENDMENT_5_S0L.md, frozen 2026-11-13 before any L data).  gain = paired dAUC point > 0.037 (the card's "exceeds the floor") AND paired CI lower bound > 0,
the form of the card's sibling HEAD-UNDERUSES rule as Amendment 2 made it paired.  `decide()` implements it; the point-only reading is printed beside and decides nothing.
ESTIMATOR HONESTY: the interval answers "another draw of EPISODES" only.  It is blind to "another TRAINING run" (H-ESTIM-SEED-1); the 0.037 floor is the card's own run-to-run figure.

TIER: T1-kin (first tick, eval-141) -- a decision-head diagnostic, never driving performance.  FAMILIES: TACTICAL decision quality only.  LONGITUDINAL and LATERAL are not
covered; STRATEGIC is PAUSED (PI item 25).

  python s0l_read.py read     --L stopdec_L_s0.npz --B stopdec_BPRIME_s0.npz --out RESULT_S0L.json [--n-boot 2000]
  python s0l_read.py controls --out CONTROLS_S0L.json [--n-boot 2000]       real-data controls (B-prime vs itself on the real B-prime mini-eval scores; BAL-NULL_s0 vs itself) + planted controls
CPU only, numpy only.  Refuses (rc 2) on any misalignment, count mismatch, or a taniteval.ci that is not the pinned file.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.environ.get("S0L_THEORY") or os.path.dirname(os.path.dirname(HERE))              # .../motion_c/theory
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(THEORY, "stopperc", "stage1_A", "read"))
sys.path.insert(0, os.path.join(THEORY, "stopbal", "read"))
sys.path.insert(0, os.path.join(THEORY, "stopbal", "dose2"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

CI_TREE = os.environ.get("S0L_CI_TREE", "F:/TanitAD-scratch/stopbal_read/tree/taniteval")   # the STOPBAL read tree's taniteval (code of record, has paired_stat_cluster_bootstrap)
CI_SHA256 = "528276c5a51a94dc614406a6907b7f0da51081f6d3bd8bf5d8432857393755be"               # TREE_PINS.json: taniteval/taniteval/ci.py
if CI_TREE not in sys.path:
    sys.path.insert(0, CI_TREE)

FLOOR = 0.037                       # card section 4 (S0-L): the AUROC floor
EXPECT = {"n_windows": 2365, "n_pos": 131, "n_episodes": 139}   # STOPCUE's own windows (theory/stopcue/VERDICT_STOPCUE.md; re-counted from stopdec_PARITY_DAGOACC_s0.npz in `controls`)
N_BOOT, SEED = 2000, 0
STOP_GI, LON_STOP_I = 6, 2          # GOAL22 index of STOP_POINT; LON9 index of STOP (stopcue_read.py:19-20)
MODEL_FREE = ("clip_index", "ws", "eid", "has", "stop_hit", "v_end_gt", "v2_gt")   # arrays that come from the eval windows / GT, not from the model: must be identical across arms
NEEDED = MODEL_FREE + ("goal_prob", "lon_prob")
TIER = "T1-kin (first tick, eval-141); a decision-head diagnostic, never driving performance"
FAMILIES = {"TACTICAL": "COVERED (stop / not-stop decision quality: AUROC of p(STOP_POINT) against stop_hit)",
            "LONGITUDINAL": "NOT COVERED by S0-L (no speed, acceleration or distance-keeping cell is read; those stay with the OTBAND four-family reader)",
            "LATERAL": "NOT COVERED by S0-L (no heading, curvature, yaw-rate or cross-track cell is read)",
            "STRATEGIC": "PAUSED (PI item 25, PI_DECISION_QUEUE): reported as paused, never a pass"}


class S0LRefusal(Exception):
    """A read that cannot be trusted: raised, never softened."""


def _import_frozen():
    import stopperc_s1_read as S1          # noqa: E402  (auroc_fast, _auc_stat, cue_arrays; numpy only)
    import taniteval.ci as CI              # noqa: E402
    p = os.path.abspath(CI.__file__)
    with open(p, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha != CI_SHA256:
        raise S0LRefusal(f"taniteval.ci is {p} sha256 {sha}, not the pinned {CI_SHA256}")
    return S1, CI, sha


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------ loading
def load_arm(path):
    """A STOPDEC npz -> dict of arrays (+ its sha256).  Refuses a file without the keys STOPCUE reads."""
    z = np.load(path, allow_pickle=False)
    miss = [k for k in NEEDED if k not in z.files]
    if miss:
        raise S0LRefusal(f"{path}: missing keys {miss}")
    d = {k: np.asarray(z[k]) for k in NEEDED}
    gp = d["goal_prob"]
    if gp.ndim != 2 or gp.shape[1] <= STOP_GI or not np.all((gp >= 0) & (gp <= 1)):
        raise S0LRefusal(f"{path}: goal_prob is not a [N, >=7] array of probabilities")
    d["_path"], d["_sha256"] = str(path), sha256_file(path)
    return d


def load_arm_mini_eval(mini_npz, labels_npz):
    """CONTROL ONLY.  A v8f_mini_eval npz (goal_prob, lon_logits, eid; 2,399 windows, a different tier than T1-kin) given the model-free window arrays of a STOPDEC npz.
    Refused unless the two files' `eid` arrays are identical."""
    lab, z = np.load(labels_npz, allow_pickle=False), np.load(mini_npz, allow_pickle=False)
    if not np.array_equal(lab["eid"], z["eid"]):
        raise S0LRefusal("mini-eval eid differs from the label file's eid")
    d = {k: np.asarray(lab[k]) for k in MODEL_FREE}
    d["goal_prob"] = np.asarray(z["goal_prob"], np.float64)
    ll = np.asarray(z["lon_logits"], np.float64)
    e = np.exp(ll - ll.max(1, keepdims=True))
    d["lon_prob"] = e / e.sum(1, keepdims=True)
    d["_path"], d["_sha256"] = str(mini_npz), sha256_file(mini_npz)
    return d


def check_aligned(a, b):
    """The two arms must be scored on the SAME windows: every model-free array equal.  Refuses, naming the array."""
    bad = [k for k in MODEL_FREE if a[k].shape != b[k].shape or not np.array_equal(a[k], b[k])]
    if bad:
        raise S0LRefusal(f"arms are not window-aligned: model-free arrays differ: {bad}")


def window_set(arm):
    """STOPCUE's windows (stopcue_read.read_arm:71-78): `has` windows, score p(STOP_POINT), label stop_hit."""
    idx = np.where(np.asarray(arm["has"], bool))[0]
    return {"idx": idx, "score": np.asarray(arm["goal_prob"], np.float64)[idx, STOP_GI], "y": np.asarray(arm["stop_hit"], bool)[idx],
            "lon": np.asarray(arm["lon_prob"], np.float64)[idx, LON_STOP_I], "reg_stop": np.asarray(arm["v_end_gt"], np.float64)[idx] < 0.5,
            "eid": np.asarray(arm["eid"])[idx]}


def counts(ws):
    return {"n_windows": int(len(ws["idx"])), "n_pos": int(ws["y"].sum()), "n_episodes": int(len(np.unique(ws["eid"])))}


def assert_counts(c, expect=EXPECT):
    if c != expect:
        raise S0LRefusal(f"window counts {c} differ from STOPCUE's {expect}")


# ------------------------------------------------------------------------------------------------------------------ the read
def paired_dauc(S1, CI, sa, sb, y, eid, n_boot, seed=SEED):
    """dAUC = AUROC(sa) - AUROC(sb) on labels y; point exact; interval from the paired episode-cluster bootstrap (same resample for both arms)."""
    pa, pb = S1.auroc_fast(sa, y), S1.auroc_fast(sb, y)
    yy = np.asarray(y, np.float64)
    r = CI.paired_stat_cluster_bootstrap(list(zip(sa, yy)), list(zip(sb, yy)), eid, S1._auc_stat, n_boot=n_boot, seed=seed)
    undefined = bool(r.get("undefined", False))
    return {"auroc_a": float(pa), "auroc_b": float(pb), "dauc": float(pa - pb), "lo": None if undefined else float(r["lo"]), "hi": None if undefined else float(r["hi"]),
            "separated": bool(r["separated"]) and not undefined, "p_delta_gt0": r.get("p_delta_gt0"), "undefined": undefined, "n_boot": int(n_boot)}


def single_auc_ci(CI, S1, s, y, eid, n_boot, seed=SEED):
    yy = np.asarray(y, np.float64)
    r = CI.stat_cluster_bootstrap(list(zip(np.asarray(s, np.float64), yy)), eid, S1._auc_stat, n_boot=n_boot, seed=seed)
    return {"point": float(r["point"]), "lo": r.get("lo"), "hi": r.get("hi")}


def decide(g, floor=FLOOR):
    """AMENDMENT_5_S0L.md: GAIN iff the paired dAUC point exceeds the floor AND the paired CI lower bound is above 0.  A missing bound is NO-GAIN."""
    lo = g.get("lo")
    return bool(g["dauc"] > floor and lo is not None and lo > 0)


def read_pair(L, B, floor=FLOOR, n_boot=N_BOOT, expect=EXPECT, tier=TIER):
    """The S0-L read on two loaded arms.  Raises S0LRefusal on misalignment / counts."""
    S1, CI, ci_sha = _import_frozen()
    check_aligned(L, B)
    wl, wb = window_set(L), window_set(B)
    c = counts(wl)
    if expect is not None:
        assert_counts(c, expect)
    g = paired_dauc(S1, CI, wl["score"], wb["score"], wl["y"], wl["eid"], n_boot)
    lon = paired_dauc(S1, CI, wl["lon"], wb["lon"], wl["reg_stop"], wl["eid"], n_boot)
    return {
        "tool": "s0l_read.py", "card": "theory/stopperc/PREREG_V8F_STOPPERC.md section 4 (S0-L)", "tier": tier,
        "estimator": "taniteval.ci.paired_stat_cluster_bootstrap, B=%d, seed 0, by episode, same resample for both arms; answers 'another draw of EPISODES' only, not 'another training run'" % n_boot,
        "ci_py_sha256": ci_sha, "floor": floor, "floor_source": "card section 4; STOPBAL VERDICT_STOPBAL.md:35 'about 0.037' (INHERITED, not re-measured here)",
        "counts": c, "counts_expected_STOPCUE": expect,
        "inputs": {"L": {"path": L["_path"], "sha256": L["_sha256"]}, "B_prime_s0": {"path": B["_path"], "sha256": B["_sha256"]}},
        "stopcue_statistic": "AUROC of goal_prob[:,6] (p(STOP_POINT)) vs stop_hit on the `has` windows (stopcue_read.py:75-77)",
        "goal_stop": g,
        "gain": decide(g, floor),
        "rule": "AMENDMENT_5_S0L.md: GAIN iff paired dAUC point > 0.037 AND paired CI lower bound > 0; otherwise NO-GAIN. One training seed per arm: a GAIN licenses L-DAG, not a lever claim (H-ESTIM-SEED-1).",
        "point_only_beside": bool(g["dauc"] > floor),
        "ci_lower_above_zero": bool(g["lo"] is not None and g["lo"] > 0),
        "descriptive_dauc_below_minus_floor": bool(g["dauc"] < -floor),
        "beside_lon_stop_vs_regime_stop": lon,
        "families": FAMILIES,
        "licenses_if_gain": "L-DAG 'conditional on S0-L' (card section 4 table): proceed to DAG-OACC from L on an L-merged train cache. A non-gain does not license it; it licenses 'otherwise the 416x1024 source (Q5)' per PERC-NULL.",
    }


# ------------------------------------------------------------------------------------------------------------------ controls
def self_control(arm, n_boot=N_BOOT, mutate=None):
    """B against itself must give dAUC exactly 0.0, CI [0, 0], no gain.  `mutate` = 'invert' (the second copy's score is 1 - p) must make this control RED."""
    other = dict(arm)
    if mutate == "invert":
        other["goal_prob"] = 1.0 - np.asarray(arm["goal_prob"], np.float64)
    r = read_pair(arm, other, n_boot=n_boot, expect=None)
    g = r["goal_stop"]
    ok = (g["dauc"] == 0.0 and g["lo"] == 0.0 and g["hi"] == 0.0 and not r["gain"])
    return {"pass": bool(ok), "dauc": g["dauc"], "lo": g["lo"], "hi": g["hi"], "gain": r["gain"], "auroc": g["auroc_a"], "counts": r["counts"], "mutate": mutate}


def planted(seed=7, n_ep=139, n_pos=131, n_neg_per=16, shift_base=1.0, add_gain=0.5, add_small=0.06):
    """Planted arms on STOPCUE-shaped windows (139 episodes, 131 positives).  B = noise + shift_base*y;  L_gain = B + add_gain*y;  L_small = B + add_small*y (shared noise, so dAUC is monotone in the add)."""
    rng = np.random.default_rng(seed)
    n = n_ep * (n_neg_per + 1)
    eid = np.repeat(np.arange(n_ep), n_neg_per + 1)
    y = np.zeros(n, bool)
    y[rng.choice(n, n_pos, replace=False)] = True
    noise = rng.normal(size=n)

    def arm(score):
        z = np.zeros((n, 22))
        z[:, STOP_GI] = 1.0 / (1.0 + np.exp(-score))
        zl = np.full((n, 7), 1.0 / 7)
        return {"clip_index": np.arange(n), "ws": np.zeros(n, np.int64), "eid": eid, "has": np.ones(n, bool), "stop_hit": y, "v_end_gt": np.ones(n), "v2_gt": np.ones(n),
                "goal_prob": z, "lon_prob": zl, "_path": "planted", "_sha256": "planted"}
    base = noise + shift_base * y
    return arm(base), arm(base + add_gain * y), arm(base + add_small * y)


def planted_control(n_boot=N_BOOT, floor=FLOOR):
    B, Lg, Ls = planted()
    exp = {"n_windows": len(B["eid"]), "n_pos": int(B["stop_hit"].sum()), "n_episodes": 139}
    rg = read_pair(Lg, B, floor=floor, n_boot=n_boot, expect=exp)
    rs = read_pair(Ls, B, floor=floor, n_boot=n_boot, expect=exp)
    ok = rg["gain"] and not rs["gain"] and rg["goal_stop"]["dauc"] > floor > rs["goal_stop"]["dauc"] > 0
    return {"pass": bool(ok), "gain_dauc": rg["goal_stop"]["dauc"], "gain_counts": rg["gain"], "small_dauc": rs["goal_stop"]["dauc"], "small_counts": rs["gain"], "floor": floor}


def shuffle_control(arm, n_boot=N_BOOT, shuffle=True, seed=11):
    """Label shuffle: the single-arm AUROC CI must cover 0.5.  shuffle=False (the mutant: real labels) must NOT cover it."""
    S1, CI, _ = _import_frozen()
    w = window_set(arm)
    y = np.random.default_rng(seed).permutation(w["y"]) if shuffle else w["y"]
    r = single_auc_ci(CI, S1, w["score"], y, w["eid"], n_boot)
    return {"pass": bool(r["lo"] <= 0.5 <= r["hi"]), "auroc": r["point"], "lo": r["lo"], "hi": r["hi"], "shuffled": shuffle}


def counts_control(arm, expect=EXPECT, drop_one=False):
    """The window / positive / episode counts equal STOPCUE's.  drop_one removes one positive window: must read RED."""
    a = dict(arm)
    if drop_one:
        keep = np.ones(len(arm["has"]), bool)
        keep[np.where(arm["has"] & arm["stop_hit"])[0][0]] = False
        a = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in arm.items()}
    c = counts(window_set(a))
    return {"pass": c == expect, "counts": c, "expected": expect}


def alignment_control(arm):
    """Two arms with different windows are refused (a positive label moved in one copy)."""
    b = dict(arm)
    sh = np.array(arm["stop_hit"], copy=True)
    sh[np.where(arm["has"])[0][0]] ^= True
    b["stop_hit"] = sh
    try:
        check_aligned(arm, b)
    except S0LRefusal:
        return {"pass": True}
    return {"pass": False}


def run_controls(n_boot=N_BOOT):
    art = "E:/Projects/TanitAD-artifacts"
    paths = {"BAL_NULL_s0": THEORY + "/stopbal/read/out/stopdec/stopdec_BAL-NULL_s0.npz", "PARITY_s0": THEORY + "/stopdec/out/stopdec_PARITY_DAGOACC_s0.npz",
             "M2bp_s0": art + "/lorapod_eval/out/eval_M2bp_s0.npz", "M2bp_s1": art + "/lorapod_eval/out/eval_M2bp_s1.npz"}
    out = {"tier_note": {"BAL_NULL_s0 / PARITY": "T1-kin STOPDEC npz (real dumps of DAG-OACC-family arms)",
                         "M2bp_s0 / s1": "REAL B-prime scores but from v8f_mini_eval (mini-eval, 2,399 windows), NOT T1-kin dumps: control use only; labels taken from PARITY (eid arrays equal, asserted)"},
           "paths": paths, "controls": {}, "mutants_must_be_RED": {}}
    par = load_arm(paths["PARITY_s0"])
    bal = load_arm(paths["BAL_NULL_s0"])
    c = out["controls"]
    c["counts_equal_STOPCUE_PARITY_s0"] = counts_control(par)
    c["counts_equal_STOPCUE_BAL_NULL_s0"] = counts_control(bal)
    c["alignment_refusal"] = alignment_control(par)
    c["self_BAL_NULL_s0_T1kin"] = self_control(bal, n_boot)
    bp0 = load_arm_mini_eval(paths["M2bp_s0"], paths["PARITY_s0"])
    c["self_Bprime_s0_mini_eval_scores"] = self_control(bp0, n_boot)
    c["shuffle_Bprime_s0_mini_eval_scores"] = shuffle_control(bp0, n_boot)
    c["shuffle_BAL_NULL_s0_T1kin"] = shuffle_control(bal, n_boot)
    c["planted_gain_and_below_floor"] = planted_control(n_boot)
    m = out["mutants_must_be_RED"]
    m["self_inverted_copy"] = self_control(bal, n_boot, mutate="invert")
    m["shuffle_off_real_labels"] = shuffle_control(bal, n_boot, shuffle=False)
    m["counts_drop_one_positive"] = counts_control(par, drop_one=True)
    m["planted_floor_zero"] = planted_control(n_boot, floor=0.0)
    out["controls_all_pass"] = all(v["pass"] for v in c.values())
    out["mutants_all_RED"] = all(not v["pass"] for v in m.values())
    # descriptive, beside: B-prime s0 vs s1 on mini-eval scores. NOT T1-kin, NOT a floor, decides nothing.
    S1, CI, _ = _import_frozen()
    bp1 = load_arm_mini_eval(paths["M2bp_s1"], paths["PARITY_s0"])
    w0, w1 = window_set(bp0), window_set(bp1)
    out["descriptive_beside_Bprime_s0_vs_s1_mini_eval"] = {"auroc_s0": float(S1.auroc_fast(w0["score"], w0["y"])), "auroc_s1": float(S1.auroc_fast(w1["score"], w1["y"])),
                                                           "note": "mini-eval tier, two training seeds, point values only; not T1-kin and not the card's floor"}
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--L", required=True)
    r.add_argument("--B", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n-boot", type=int, default=N_BOOT)
    c = sub.add_parser("controls")
    c.add_argument("--out", required=True)
    c.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    try:
        if a.cmd == "read":
            res = read_pair(load_arm(a.L), load_arm(a.B), n_boot=a.n_boot)
        else:
            res = run_controls(a.n_boot)
    except S0LRefusal as e:
        print("REFUSED:", e, file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    if a.cmd == "read":
        print(json.dumps({"dauc": res["goal_stop"]["dauc"], "lo": res["goal_stop"]["lo"], "hi": res["goal_stop"]["hi"], "gain": res["gain"], "counts": res["counts"]}))
        return 0
    print(json.dumps({"controls_all_pass": res["controls_all_pass"], "mutants_all_RED": res["mutants_all_RED"]}))
    return 0 if (res["controls_all_pass"] and res["mutants_all_RED"]) else 1


if __name__ == "__main__":
    sys.exit(main())
