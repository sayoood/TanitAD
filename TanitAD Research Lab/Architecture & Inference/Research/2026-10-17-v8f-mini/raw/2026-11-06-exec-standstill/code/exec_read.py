#!/usr/bin/env python3
"""exec_read.py -- the EXEC-STANDSTILL read (PREREG_EXEC_STANDSTILL.md; definitions in DEFINITIONS_H.md). Zero training, CPU only.

Modes
  extract   per-window arrays (the reader's 19 rows, a = model, b = ITS OWN P10 floor) for every production rollout -> raw/arrays/<tag>.npz     (extract_smooth.py, pinned, --floor own)
  gate-full the full-grid G-ID evidence: the base rollouts against the BANKED C0 rollouts, executed paths and ticks                              -> raw/GATE_ID_FULL.json
  read      E = M1(C0) - M1(P10) per seed and executor, dE = E_hold - E_base (paired episode-cluster bootstrap B 2000 seed 0), all 19 rows beside,
            lateral/heading rows per start class, the yaw-rate diagnostic, the outcome of card section 5                                          -> raw/EXEC_STANDSTILL_READ.json
The read REFUSES to run unless raw/GATES.json says ALL_GATES_PASS and raw/GATE_ID_FULL.json says PASS (card section 4: no contrast is read before every gate passes).
The pure layer (outcome, diag, delta_e) is unit-tested in tests/test_exec_read.py with known answers.
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
DL = f"{ART}/dagland"
RR = f"{DL}/recover_next/read/code/recover_read.py"
EXTRACT = f"{DL}/smooth_t1/code/extract_smooth.py"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SCR = "C:/Users/Admin/exec_standstill_scratch"
OUT = f"{SCR}/out"
ARR = f"{PK}/raw/arrays"
BANKED_OUT = "C:/Users/Admin/rr_scratch/out"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
M1 = "M1 crawl share"
MODELS = {"C0": "XC0", "DAG": "XDAG"}
SEEDS = (0, 1)
EXECS = ("base", "hold")
PER_CLASS_ROWS = ("heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)", M1)
YAW_THR_DEG = 60.0
READ_ROWS_T = 20          # executed rows t0+1 .. t0+20: the rows a replan tick can read (ticks at t0, t0+10, t0+20)
SHARE_BAR = 0.5


def tags():
    return [f"{p}_s{s}_{e}" for p in MODELS.values() for s in SEEDS for e in EXECS]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ pure layer
def outcome(per_seed):
    """Card section 5, applied literally. ``per_seed`` = {seed: {"E_base": float, "dE": float, "lo": float, "hi": float}}.
    -> dict(outcome, label, per_seed flags). A separated-POSITIVE dE or E_base <= 0 is outside the card's table: label UNCOMMITTED, never forced into 1/2/3."""
    flags = {}
    for s, v in per_seed.items():
        sep = bool(v["lo"] > 0 or v["hi"] < 0)
        neg = bool(v["hi"] < 0)
        big = bool(v["E_base"] > 0 and abs(v["dE"]) >= SHARE_BAR * v["E_base"])
        flags[s] = {"separated": sep, "separated_negative": neg, "separated_positive": bool(sep and not neg), "E_base_positive": bool(v["E_base"] > 0),
                    "abs_dE_ge_half_E_base": big, "share_explained": (-v["dE"] / v["E_base"]) if v["E_base"] > 0 else None}
    ss = list(flags.values())
    if len(ss) != 2:
        return {"outcome": None, "label": "NOT COMPUTED (need both seeds)", "flags": flags}
    if any(not f["E_base_positive"] for f in ss) or any(f["separated_positive"] for f in ss):
        return {"outcome": None, "label": "UNCOMMITTED (outside the card's table: a separated-positive dE or a non-positive E_base)", "flags": flags}
    n_neg = sum(f["separated_negative"] for f in ss)
    if n_neg == 2 and all(f["abs_dE_ge_half_E_base"] for f in ss):
        return {"outcome": 1, "label": "ARTIFACT-CONTRIBUTES", "flags": flags}
    if n_neg == 2 or n_neg == 1:
        return {"outcome": 2, "label": "PARTIAL", "flags": flags}
    return {"outcome": 3, "label": "NO-EFFECT", "flags": flags}


def launch_mask(v0, g):
    """The AUDIT's standstill-launch class: logged v0 < 0.1 m/s and logged 6 s extent |g[29]| > 10 m."""
    v0 = np.asarray(v0, np.float64)
    g = np.asarray(g, np.float64)
    return (v0 < 0.1) & (np.linalg.norm(g[:, 29], axis=-1) > 10.0)


def diag_share(ch1, mask, rows, thr_deg=YAW_THR_DEG):
    """Share of the masked windows whose executed-history yaw rate exceeds ``thr_deg`` deg/s: max |ch1| (rad/s) over the first ``rows`` executed rows. -> dict."""
    ch1 = np.asarray(ch1, np.float64)[:, :rows]
    mx = np.nanmax(np.abs(np.degrees(ch1)), axis=1)
    m = np.asarray(mask, bool)
    if not m.any():
        return {"n": 0, "n_flagged": 0, "share": None, "median_max_degps": None}
    fl = mx > thr_deg
    return {"n": int(m.sum()), "n_flagged": int(fl[m].sum()), "share": float(fl[m].mean()), "median_max_degps": float(np.median(mx[m])), "rows": int(rows)}


def delta_e(GD, d_hold, d_base, eid, cls, n_boot, pool="masked"):
    """dE = mean(d_hold - d_base) over the windows where both are finite, with the paired episode-cluster bootstrap (the instrument's own draw stream)."""
    return GD.decompose(d_hold, d_base, eid, cls, pool, n_boot)


# ------------------------------------------------------------------------------------------------ heavy layer
def load_rr():
    spec = importlib.util.spec_from_file_location("recover_read", RR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def extract_one(tag, seed):
    dst = f"{ARR}/{tag}.npz"
    if os.path.exists(dst):
        return tag, "exists"
    os.makedirs(ARR, exist_ok=True)
    os.makedirs(f"{PK}/logs", exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [PY, "-I", EXTRACT, "--arm", "CTRL", "--seed", str(seed), "--out", dst, "--dump-dir", f"{OUT}/{tag}/dump_{tag}", "--ticks", f"{OUT}/{tag}/ticks/ticks_{tag}.npz",
           "--work-dir", f"{SCR}/work/{tag}", "--floor", "own", "--floor-guard-m", "1e9"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    open(f"{PK}/logs/extract_{tag}.log", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"extract {tag} failed rc={r.returncode}: {r.stderr[-400:]}")
    return tag, "ok"


def cmd_extract(a):
    jobs = [(t, int(t.split("_s")[1][0])) for t in tags() if (not a.only or a.only in t)]
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for tag, st in ex.map(lambda j: extract_one(*j), jobs):
            print("extract", tag, st, flush=True)
    return 0


def load_dump(d):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("cl", "fl", "ol", "ws", "eid", "g", "v0")}


def cmd_gate_full(a):
    res = {"note": "G-ID on the FULL eval-141 grid: the patched tool at eps 1e-3 (16 threads, as the banked runs) against the BANKED C0 rollout (tool 0721fc01, 16 threads, a different session)"}
    ok = True
    seeds = tuple(int(x) for x in a.seeds.split(","))
    for s in seeds:
        mine = load_dump(f"{OUT}/XC0_s{s}_base/dump_XC0_s{s}_base")
        ban = load_dump(f"{BANKED_OUT}/C0_s{s}/dump_C0_s{s}")
        t_m = np.load(f"{OUT}/XC0_s{s}_base/ticks/ticks_XC0_s{s}_base.npz", allow_pickle=True)
        t_b = np.load(f"{BANKED_OUT}/C0_s{s}/ticks/ticks_C0_s{s}.npz", allow_pickle=True)
        same_w = bool(all(np.array_equal(mine[k], ban[k]) for k in ("ws", "eid", "g", "v0")))
        ent = {"same_windows": same_w}
        for k in ("cl", "fl", "ol"):
            d = np.abs(mine[k] - ban[k])
            pw = (d == 0).reshape(len(d), -1).all(1)
            ent[k] = {"bit_identical": bool(pw.all()), "n_windows_unequal": int((~pw).sum()), "max_abs_dev_m": float(np.nanmax(d)), "mean_abs_dev_m": float(np.nanmean(d)),
                      "p99_window_max_dev_m": float(np.percentile(d.reshape(len(d), -1).max(1), 99))}
        ent["ticks"] = {k: {"equal": bool(np.array_equal(t_m[k], t_b[k], equal_nan=True)) if t_m[k].dtype.kind in "fc" else bool(np.array_equal(t_m[k], t_b[k]))}
                        for k in t_m.files if k in t_b.files and k not in ("tag", "ckpt", "clip_names")}
        ent["ticks_max_abs_dev"] = {k: float(np.nanmax(np.abs(t_m[k].astype(np.float64) - t_b[k].astype(np.float64)))) for k in ("plan", "P", "tau", "pose", "v0") if k in t_m.files and k in t_b.files}
        ent["strict_bit_identical"] = bool(same_w and all(ent[k]["bit_identical"] for k in ("cl", "fl", "ol")))
        res[f"s{s}"] = ent
        ok = ok and ent["strict_bit_identical"]
    res["STRICT_BIT_IDENTICAL_BOTH_SEEDS"] = bool(ok)
    res["seeds_checked"] = list(seeds)
    res["PASS"] = bool(ok and set(seeds) == set(SEEDS))              # a one-seed probe can never write a PASS
    with open(a.out or f"{PK}/raw/GATE_ID_FULL.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("STRICT_BIT_IDENTICAL_BOTH_SEEDS", "PASS")}))
    return 0 if res["PASS"] else 1


def cmd_read(a):
    for p, key in ((f"{PK}/raw/GATES.json", "ALL_GATES_PASS"), (f"{PK}/raw/GATE_ID_FULL.json", "PASS")):
        if not os.path.exists(p) or not json.load(open(p, encoding="utf-8")).get(key):
            print(f"REFUSED: {p} does not say {key}: true (card section 4: no contrast is read before every gate passes)", file=sys.stderr)
            return 3
    rr = load_rr()
    rr.heavy_init()
    GD = rr.GD
    L = GD.LABELS
    res = {"card": "PREREG_EXEC_STANDSTILL.md", "card_sha256": sha256_file(f"{PK}/PREREG_EXEC_STANDSTILL.md"), "definitions": "DEFINITIONS_H.md",
           "tier": "T1-kin (own executed motion, logged cameras), eval-141, 2,399 windows, zero training, CPU",
           "estimator": f"paired episode-cluster bootstrap, B={a.n_boot}, seed 0 (gapdecomp.decompose); answers another draw of EPISODES only; the planner is deterministic; the training-seed question is read from |dE_s1 - dE_s0|",
           "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "models": {}}
    for mname, pre in MODELS.items():
        mres = {}
        for s in SEEDS:
            z = {e: rr.read_arrays(f"{ARR}/{pre}_s{s}_{e}.npz") for e in EXECS}
            eid = z["base"]["eid"]
            for e in EXECS:
                assert np.array_equal(z[e]["eid"], eid) and np.array_equal(z[e]["ws"], z["base"]["ws"]), f"{pre}_s{s}_{e}: windows differ from base"
            cls, _ = GD.classify(z["base"]["v0"], z["base"]["g"])
            A = {e: rr.Aof(z[e]) for e in EXECS}
            B = {e: rr.Bof(z[e]) for e in EXECS}
            D = {e: [A[e][i] - B[e][i] for i in range(len(L))] for e in EXECS}          # the per-window excess of the model over its own P10, per row
            blk = {}
            blk["E_base_rows"], _ = rr.rows_of(A["base"], B["base"], eid, cls, a.n_boot)
            blk["E_hold_rows"], _ = rr.rows_of(A["hold"], B["hold"], eid, cls, a.n_boot)
            blk["dE_rows"], _ = rr.rows_of(D["hold"], D["base"], eid, cls, a.n_boot)
            blk["model_hold_minus_base_rows"], _ = rr.rows_of(A["hold"], A["base"], eid, cls, a.n_boot)
            blk["p10_hold_minus_base_rows"], _ = rr.rows_of(B["hold"], B["base"], eid, cls, a.n_boot)
            i1 = L.index(M1)
            full = {"E_base": GD.decompose(A["base"][i1], B["base"][i1], eid, cls, GD.POOL[M1], a.n_boot), "E_hold": GD.decompose(A["hold"][i1], B["hold"][i1], eid, cls, GD.POOL[M1], a.n_boot),
                    "dE": GD.decompose(D["hold"][i1], D["base"][i1], eid, cls, GD.POOL[M1], a.n_boot)}
            valid = {e: np.isfinite(A[e][i1]) & np.isfinite(B[e][i1]) for e in EXECS}
            blk["M1"] = {"E_base": full["E_base"]["delta"], "E_base_lo": full["E_base"]["lo"], "E_base_hi": full["E_base"]["hi"], "E_hold": full["E_hold"]["delta"], "E_hold_lo": full["E_hold"]["lo"], "E_hold_hi": full["E_hold"]["hi"],
                         "dE": full["dE"]["delta"], "dE_lo": full["dE"]["lo"], "dE_hi": full["dE"]["hi"], "dE_separated": bool(full["dE"]["separated"]), "n_valid_base": int(valid["base"].sum()), "n_valid_hold": int(valid["hold"].sum()),
                         "valid_sets_equal": bool(np.array_equal(valid["base"], valid["hold"])), "dE_minus_(E_hold-E_base)": float(full["dE"]["delta"] - (full["E_hold"]["delta"] - full["E_base"]["delta"])),
                         "classes_dE": full["dE"]["classes"], "classes_E_base": full["E_base"]["classes"], "classes_E_hold": full["E_hold"]["classes"]}
            blk["per_class"] = {}
            for lab in PER_CLASS_ROWS:
                i = L.index(lab)
                row = {}
                for nm, (u, v) in (("E_base", (A["base"][i], B["base"][i])), ("E_hold", (A["hold"][i], B["hold"][i])), ("dE", (D["hold"][i], D["base"][i]))):
                    r = GD.decompose(u, v, eid, cls, GD.POOL[lab], a.n_boot)
                    row[nm] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "n": r["n_windows"], "classes": {c: {k: r["classes"][c][k] for k in ("n", "delta", "delta_lo", "delta_hi")} for c in GD.CLASSES}}
                blk["per_class"][lab] = row
            # the yaw-rate diagnostic
            exc = {e: np.load(f"{OUT}/{pre}_s{s}_{e}/exec_ch_{pre}_s{s}_{e}.npz", allow_pickle=True) for e in EXECS}
            lm = launch_mask(z["base"]["v0"], z["base"]["g"])
            Lc = cls == GD.CLASSES.index("L")
            blk["diagnostic"] = {"launch_standstill_n": int(lm.sum()), "launch_standstill_episodes": int(len(np.unique(eid[lm]))), "gapdecomp_L_n": int(Lc.sum()), "threshold_degps": YAW_THR_DEG}
            for e in EXECS:
                for role in ("cl", "fl"):
                    ch = exc[e][role + "_ch1"]
                    blk["diagnostic"][f"{role}_{e}"] = {"launch_standstill_rows1to20": diag_share(ch, lm, READ_ROWS_T), "launch_standstill_rows1to30": diag_share(ch, lm, 30),
                                                        "gapdecomp_L_rows1to20": diag_share(ch, Lc, READ_ROWS_T), "all_windows_rows1to20": diag_share(ch, np.ones(len(ch), bool), READ_ROWS_T)}
            mres[f"s{s}"] = blk
        po = {s: {"E_base": mres[f"s{s}"]["M1"]["E_base"], "dE": mres[f"s{s}"]["M1"]["dE"], "lo": mres[f"s{s}"]["M1"]["dE_lo"], "hi": mres[f"s{s}"]["M1"]["dE_hi"]} for s in SEEDS}
        mres["outcome_card_section5"] = outcome(po)
        mres["seed_variance_abs_dE_s1_minus_s0"] = abs(po[1]["dE"] - po[0]["dE"])
        res["models"][mname] = mres
        print(mname, "outcome", mres["outcome_card_section5"]["label"], {f"s{s}": (round(po[s]["E_base"], 4), round(po[s]["dE"], 4), round(po[s]["lo"], 4), round(po[s]["hi"], 4)) for s in SEEDS})
    res["PRIMARY_MODEL"] = "C0 (the card's arm); DAG = DAGOACC beside, reported by the same rule but never the verdict"
    with open(f"{PK}/raw/EXEC_STANDSTILL_READ.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (bool(o) if isinstance(o, np.bool_) else str(o)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--workers", type=int, default=2)
    e.add_argument("--only", default="")
    g = sub.add_parser("gate-full")
    g.add_argument("--seeds", default="0,1")
    g.add_argument("--out", default="")
    r = sub.add_parser("read")
    r.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    return {"extract": cmd_extract, "gate-full": cmd_gate_full, "read": cmd_read}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
