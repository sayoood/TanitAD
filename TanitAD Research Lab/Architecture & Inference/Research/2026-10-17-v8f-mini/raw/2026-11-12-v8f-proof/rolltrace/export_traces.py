# export_traces.py -- pod side, CPU only. Exports per-window traces of the distance to the lead for three forms of the lead rule (RR, RLc, VD5c), both model seeds, windows with a lead.
# The series are the ENGINE'S OWN: headway_read_r2.window_cells(g, cl, L)["_gm"] (gap of the rolled-out ego to the lead, m, steps 1..30 of 0.2 s) and the flags C, U, F of that same call;
# the ego speed is headway_read_r2.step_speeds(cl). Windows, dumps and restriction are read through read_longroll10k (dump_dir, load_dump_dir, restrict_idx, cut, grid_keys), as read_visrule10k.py does.
# Control t0 (fails => no ROLLTRACE.json, exit 3): pooled contact rates recomputed from the per-window flags equal the banked values within 1e-4; windows with a lead = 650 pooled.
# Selection (fixed in code): windows (seed, key) where RR has contact and VD5c has none, ordered by the sha256 hex of "<seed>|<key>", first 6. key = "<clip id>/<window start>".
# Usage: python export_traces.py --pull <folder with I10<run>_s<k>> --grid-dir <folder with grid_p30.json> --join <join npz> --code <engine code folder> --out <new folder>
#        python export_traces.py --selftest
# Last line: ZZ EXPORT-TRACES t0 PASS windows 6   or   ZZ EXPORT-TRACES t0 FAIL <clause>
import argparse
import hashlib
import json
import math
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("SR_ESTREAD_DIR"), os.path.join(_HERE, "..", "integ10k")):
    if _p and os.path.isfile(os.path.join(_p, "read_longroll10k.py")):
        sys.path.insert(0, os.path.abspath(_p))
        break
import numpy as np  # noqa: E402

FORMS = {"RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20", "VD5c": "RQ50KF5P30Hb20vVD5c"}
FORM_ORDER = ("RR", "RLc", "VD5c")
EXPECT = {"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169}
TOL_RATE = 1e-4
N_LEAD = 650
N_WIN = 1271
SEEDS = (0, 1)
N_PICK = 6
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s)"
RULE = ("windows (model seed, window key) where RR has contact and VD5c has none; ordered by the sha256 hex of the text '<seed>|<window key>', key = '<clip id>/<window start>'; the first %d" % N_PICK)


def fin(x):
    x = float(x)
    return round(x, 4) if math.isfinite(x) else None


def row(a):
    return [fin(v) for v in np.asarray(a, np.float64).reshape(-1)]


def load_form(R, H, pull, grid_dir, J, run, seed, keys):
    """one run and seed -> dict of per-window arrays in the order of `keys`; the raw window arrays g, ws, clip_index ride along for the same-windows control"""
    raw = R.load_dump_dir(R.dump_dir(pull, run, seed))
    if raw is None:
        raise SystemExit("MISSING-DUMP %s" % R.key_of(run, seed))
    idx, rep = R.restrict_idx(raw["clip_index"], raw["ws"], keys)
    if rep["missing"] != 0:
        raise SystemExit("WINDOWS-MISSING %s %d" % (R.key_of(run, seed), rep["missing"]))
    D = R.cut(raw, idx)
    L = H.label_windows(J, D["clip_index"], D["ws"])
    W = H.window_cells(D["g"], D["cl"], L)                       # the engine's own cells: C, U, F per window and the gap series of the rolled-out ego
    return {"C": W["C"], "U": W["U"], "F": W["F"], "gm": W["_gm"], "gl": W["_gl"], "vm": H.step_speeds(D["cl"]), "vl": H.step_speeds(D["g"]),
            "lead": np.asarray(L["lead"], bool), "ws": np.asarray(D["ws"], np.int64), "ci": np.asarray(D["clip_index"], np.int64), "g": np.asarray(D["g"])}


def collect(R, H, pull, grid_dir, join_path, forms=None, seeds=SEEDS):
    forms = forms or FORMS
    J = H.load_join(join_path)
    g30 = R.jl(os.path.join(grid_dir, "grid_p30.json"))
    if g30 is None:
        raise SystemExit("MISSING-GRID %s" % os.path.join(grid_dir, "grid_p30.json"))
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    data = {f: {s: load_form(R, H, pull, grid_dir, J, forms[f], s, keys) for s in seeds} for f in forms}
    return data, [str(c) for c in J["clip"]]


def key_text(clips, ci, ws):
    return "%s/%d" % (clips[int(ci)], int(ws))


def hash_of(seed, key):
    return hashlib.sha256(("%d|%s" % (seed, key)).encode("utf-8")).hexdigest()


def contact_table(data, seeds):
    out = {}
    for s in list(seeds) + ["pooled"]:
        ss = list(seeds) if s == "pooled" else [s]
        t = {"rr_yes_vd5c_yes": 0, "rr_yes_vd5c_no": 0, "rr_no_vd5c_yes": 0, "rr_no_vd5c_no": 0}
        for m in ss:
            lead = data["RR"][m]["lead"]
            a, b = data["RR"][m]["C"] == 1.0, data["VD5c"][m]["C"] == 1.0
            t["rr_yes_vd5c_yes"] += int((lead & a & b).sum())
            t["rr_yes_vd5c_no"] += int((lead & a & ~b).sum())
            t["rr_no_vd5c_yes"] += int((lead & ~a & b).sum())
            t["rr_no_vd5c_no"] += int((lead & ~a & ~b).sum())
        out[str(s)] = t
    return out


def control_t0(data, forms, seeds, expect=None, n_lead=N_LEAD, n_win=N_WIN, tol=TOL_RATE):
    """-> (ok, failing clause or '', record). Clauses: same-windows, n-windows, n-lead, rate-<form>."""
    expect = expect or EXPECT
    rec = {"rates": {}}
    ref = data[forms[0]][seeds[0]]
    for f in forms:
        for s in seeds:
            d = data[f][s]
            if not (np.array_equal(d["ws"], ref["ws"]) and np.array_equal(d["ci"], ref["ci"]) and np.array_equal(d["g"], ref["g"]) and np.array_equal(d["lead"], ref["lead"])):
                return False, "same-windows %s s%d" % (f, s), rec
    nw = int(len(ref["ws"]))
    rec["windows_per_seed"] = nw
    if nw != n_win:
        return False, "n-windows %d != %d" % (nw, n_win), rec
    nl = int(sum(int(data[forms[0]][s]["lead"].sum()) for s in seeds))
    rec["windows_with_lead_pooled"] = nl
    if nl != n_lead:
        return False, "n-lead %d != %d" % (nl, n_lead), rec
    for f in forms:
        pooled = np.concatenate([data[f][s]["C"][data[f][s]["lead"]] for s in seeds])
        per_seed = [float(np.nanmean(data[f][s]["C"])) for s in seeds]
        r_pool, r_eng = float(np.nanmean(pooled)), float(np.mean(per_seed))
        rec["rates"][f] = {"pooled": round(r_pool, 6), "mean_of_seed_levels": round(r_eng, 6), "expected": expect[f]}
        if not (abs(r_pool - expect[f]) <= tol and abs(r_eng - expect[f]) <= tol):
            return False, "rate-%s %.4f vs %.4f" % (f, r_pool, expect[f]), rec
    return True, "", rec


def select(data, clips, seeds, n_pick=N_PICK):
    cands = []
    for s in seeds:
        rr, vd, lead = data["RR"][s], data["VD5c"][s], data["RR"][s]["lead"]
        for i in np.nonzero(lead & (rr["C"] == 1.0) & (vd["C"] == 0.0))[0].tolist():
            k = key_text(clips, rr["ci"][i], rr["ws"][i])
            cands.append((hash_of(s, k), s, i, k))
    cands.sort()
    return cands[:n_pick], len(cands)


def build_record(data, clips, seeds, forms, t0rec, table, picked, n_cand):
    dt = 0.2
    wins = []
    for hx, s, i, k in picked:
        ref = data[forms[0]][s]
        w = {"seed": int(s), "clip8": clips[int(ref["ci"][i])][:8], "ws": int(ref["ws"][i]), "key_sha256_16": hx[:16], "t_s": [round(dt * (j + 1), 4) for j in range(30)],
             "logged_gap_m": row(ref["gl"][i]), "logged_speed_mps": row(ref["vl"][i]), "forms": {}}
        for f in forms:
            d = data[f][s]
            w["forms"][f] = {"gap_m": row(d["gm"][i]), "speed_mps": row(d["vm"][i]), "C": fin(d["C"][i]), "U": fin(d["U"][i]), "F": fin(d["F"][i]), "n_scored_steps": int(np.isfinite(d["gm"][i]).sum())}
        wins.append(w)
    return {"tier": TIER, "dt_s": dt, "steps": 30, "forms": list(forms), "selection_rule": RULE, "n_windows_the_rule_chose_from": int(n_cand), "table_2x2_contact": table,
            "t0": dict(t0rec, PASS=True), "series": "gap_m = engine cell _gm: logged lead gap + logged arc - rolled arc (m), steps 1..30 of 0.2 s, null after the lead track ends; speed_mps = step length / 0.2 s",
            "windows": wins}


def run(R, H, a, forms=None, expect=None, n_lead=N_LEAD, n_win=N_WIN, seeds=SEEDS):
    forms = forms or FORMS
    data, clips = collect(R, H, a.pull, a.grid_dir, a.join, forms, seeds)
    order = [f for f in FORM_ORDER if f in forms]
    ok, why, rec = control_t0(data, order, list(seeds), expect, n_lead, n_win)
    if not ok:
        return 3, "ZZ EXPORT-TRACES t0 FAIL %s" % why, None
    picked, n_cand = select(data, clips, list(seeds))
    if len(picked) < N_PICK:
        return 3, "ZZ EXPORT-TRACES t0 FAIL only %d windows to choose from" % len(picked), None
    table = contact_table(data, list(seeds))
    return 0, "ZZ EXPORT-TRACES t0 PASS windows %d" % len(picked), build_record(data, clips, list(seeds), order, rec, table, picked, n_cand)


def get_engine(code):
    import read_longroll10k as R                                           # the reader's own module (pod: SR_ESTREAD_DIR)
    if code:
        H = R.engine(types.SimpleNamespace(code=code))[0]
    else:                                                                  # selftest on the dev box: the local copy of the engine file
        sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "headway")))
        import headway_read_r2 as H
    return R, H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for n in ("pull", "grid_dir", "join", "code", "out"):
        ap.add_argument("--" + n.replace("_", "-"), default="")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.code)
    for n in ("pull", "grid_dir", "join", "code", "out"):
        if not getattr(a, n):
            print("ZZ EXPORT-TRACES STOP missing --%s" % n.replace("_", "-"))
            return 2
    if os.path.exists(a.out):
        print("ZZ EXPORT-TRACES STOP output exists")
        return 2
    R, H = get_engine(a.code)
    rc, line, rec = run(R, H, a)
    if rec is not None:
        os.makedirs(a.out)
        with open(os.path.join(a.out, "ROLLTRACE.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
    print(line)
    return rc


# ============================================================================================ selftest on tiny synthetic data
def make_synth(root, fault=False):
    """2 clips x 20 windows, 30 steps; clip 0 has a lead, clip 1 has none. The ego path is a straight line of 2 m per step (10 m/s); a rolled ego at 20 m/s closes a 15 m gap at step 8.
    RR has contact in windows with index % 3 == 0, VD5c where index % 6 == 0, RLc nowhere. Returns the expected rates of the synthetic design."""
    os.makedirs(os.path.join(root, "grid"))
    clips = ["aaaaaaaa-0000", "bbbbbbbb-1111"]
    nclip = 60
    lead_valid = np.zeros(2 * nclip, np.int64)
    lead_valid[:nclip] = 1
    gap = np.full(2 * nclip, 15.0)
    np.savez(os.path.join(root, "join.npz"), clip=np.asarray(clips), off=np.array([0, nclip]), n=np.array([nclip, nclip]), lead_valid=lead_valid, lead_gap=gap)
    json.dump({"order": clips, "windows": {c: list(range(20)) for c in clips}}, open(os.path.join(root, "grid", "grid_p30.json"), "w"))
    kk = np.arange(1, 31)[:, None]
    g1 = np.concatenate([2.0 * kk, 0.0 * kk], axis=1)
    ci = np.repeat([0, 1], 20)
    ws = np.tile(np.arange(20), 2)
    idx = np.arange(40)
    for f, run in FORMS.items():
        for s in SEEDS:
            d = os.path.join(root, "pull", "I10%s_s%d" % (run, s), "dump_I10%s_s%d_m10_causal_heads" % (run, s))
            os.makedirs(d)
            cl = np.zeros((40, 30, 2))
            for i in idx:
                w = i % 20
                con = {"RR": w % 3 == 0, "VD5c": w % 6 == 0, "RLc": False}[f]
                v = 4.0 if con else 2.0
                if fault and f == "RR" and s == 1 and i == 1:
                    v = 4.0                                    # the planted fault: one more contact in RR, seed 1
                cl[i] = np.concatenate([v * kk, 0.0 * kk], axis=1)
            np.savez(os.path.join(d, "ep0.npz"), g=np.tile(g1, (40, 1, 1)), cl=cl, ws=ws, eid=ci, clip_index=ci, v0=np.zeros(40))
    n_lead = 40
    rr = sum(1 for w in range(20) if w % 3 == 0) / 20.0
    vd = sum(1 for w in range(20) if w % 6 == 0) / 20.0
    return {"RR": rr, "RLc": 0.0, "VD5c": vd}, n_lead


def selftest(code=""):
    import tempfile
    base = os.path.join(_HERE, "tmp_pt")
    os.makedirs(base, exist_ok=True)
    R, H = get_engine(code)
    res = []
    for fault in (False, True):
        root = tempfile.mkdtemp(prefix="selftest_export_", dir=base)
        exp, nl = make_synth(root, fault)
        a = types.SimpleNamespace(pull=os.path.join(root, "pull"), grid_dir=os.path.join(root, "grid"), join=os.path.join(root, "join.npz"))
        rc, line, rec = run(R, H, a, expect=exp, n_lead=nl, n_win=40)
        res.append((fault, rc, line, rec))
    ok_c, ok_f = res[0], res[1]
    good = ok_c[1] == 0 and ok_c[3] is not None and ok_f[1] == 3 and ok_f[3] is None and "FAIL rate-RR" in ok_f[2]
    if good:
        rec = ok_c[3]
        # RR contact in windows 0,3,6,9,12,15,18 of clip 0 = 7; VD5c in 0,6,12,18; RR-only = 3,9,15 per seed -> 6 pooled; the rule picks all 6
        good = rec["n_windows_the_rule_chose_from"] == 6 and len(rec["windows"]) == 6 and rec["table_2x2_contact"]["pooled"]["rr_yes_vd5c_no"] == 6
        good = good and all(w["forms"]["RR"]["C"] == 1.0 and w["forms"]["VD5c"]["C"] == 0.0 and len(w["forms"]["RR"]["gap_m"]) == 30 for w in rec["windows"])
        g8 = rec["windows"][0]["forms"]["RR"]["gap_m"][7]
        good = good and g8 is not None and g8 <= 0.0
    print("selftest: clean run -> %s | planted fault -> %s" % (ok_c[2], ok_f[2]))
    print("ZZ EXPORT-TRACES selftest %s" % ("PASS" if good else "FAIL"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
