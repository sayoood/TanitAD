#!/usr/bin/env python3
"""TF-N5 navtest build (PREREG_TF_N5_DRAFT.md sections 3, 4, 7): the plans of the L1 arms from the banked N4-D / N2-R plans. CPU only.

  reach   python n5_nt_build.py reach [L1 flags] [--real --fixture <json> --out <json>]
          Reads margin / cap / sign / certificate / centre_m / fallback_margin / damp / damp_cum BACK from the three built repair objects (N5-A, N5-X, N5-R),
          compares them with the card, and runs a BEHAVIOURAL SIGNATURE on synthetic corridors (a perfect-tracking fake, a ping-pong tracker, a
          certificate-worse scorer) that must equal the signature of an object built from the card constants: a parameter that is read back right but never
          reaches the executed repair changes the signature (test_n5_chain.py proves every parameter does). --real adds the two real navtrain fixture
          instances (C-REACH): zero-correction identity, sign +1 certifies, sign -1 does not, determinism. Exit 1 on any difference.
  derive  python n5_nt_build.py derive --route cache|navtrain --n2r-plans <npz> --rolls "<RJ5 rows glob>" --out <shard.jsonl> --shard i/N
              (--n4d-report <banked N4-D build report> | --l1-tokens <json>) [L1 flags] [--mutate none|cert_off|l1_all|margin_neg|label_ego_cache --smoke-allow-mutation]
          Per token (sha1 % N == i, resumable): builder TRACKED score of the N2-R plan (all 12,146: C-DACMAP); where DAC = 0: N5-R (L1 on every such token, run twice =
          C-DET) and, on the L1 set (the banked N4-D `unrepaired_dac0`, 25), N5-A (sign +1) and N5-X (sign -1). Every applied plan is re-scored by a SEPARATE
          scorer call (m_recheck). C-DERIV: the 200 smallest-sha1 tokens re-derive the N2-R label from the RJ5 target (nuPlan-DB ego).
  merge   python n5_nt_build.py merge ... -> n5_plans_{N5A,N5X,N5R}.npz, seams refe_tfn5_{n5a,n5x}.npz (W3 format, from the banked N4-D seam), n5_build_report.json
          with C-COVB, C-DACMAP, C-DERIV, C-LABELEGO, C-DET, C-SET, C-CERT-B, C-SEAMFN, C-UNCH, C-REACH. Prints ZZN5_BUILD PASS|FAIL.
  unch    python n5_nt_build.py unch --out-dir <dir> ...   -> C-UNCH again on the files ON DISK (the chain does this just before the harness reads them)."""
from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["N4_RUN"] = os.environ.get("N4_RUN") or os.path.dirname(HERE)       # n5_env reads it: the run tree this file sits in
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import n5_env  # noqa: E402,F401  (BLAS threads = 1 + the pinned tree on sys.path, before numpy)
import numpy as np  # noqa: E402

import n4_repair as RP  # noqa: E402
import n4_repair_l1 as L1  # noqa: E402
import n4_nt_build as B4  # noqa: E402  (the PINNED A1 builder: load_rolls, db_ego_fn, ctx_stream, sha1_int, csv helpers; nothing copied)

# ---- the card (PREREG_TF_N5_DRAFT.md section 3: "frozen parameters"); the ONLY place the numbers live in the chain
L1_COMMON = {"margin": 0.05, "iters": 5, "cert": "full", "centre_m": 0.05, "fallback_margin": 0.005, "damp": 0.5, "damp_cum": False}
CARD = {"N5A": dict(L1_COMMON, sign=1.0), "N5X": dict(L1_COMMON, sign=-1.0), "N5R": dict(L1_COMMON, sign=1.0)}
ARMS = tuple(CARD)
SEAM_ARMS = ("N5A", "N5X")
SEAM_LABEL = {"N5A": "refe_tfn5_n5a", "N5X": "refe_tfn5_n5x"}
N_CDERIV, CDERIV_BAR_M, N_L1_EXPECT = 200, 1e-9, 25
MUTATIONS = ("none", "cert_off", "l1_all", "margin_neg", "label_ego_cache")
TARGET = "n4_repair_l1.repair_l1"
sha1_int, sha256, load_rolls, done_tokens = B4.sha1_int, B4.sha256, B4.load_rolls, B4.done_tokens


def _cert_off(c, b):
    return True


def params_of(a) -> dict:
    """the per-arm L1 parameters AS THE CLI SAYS (defaults = the card); sign of N5-X from --sign-x"""
    base = {"margin": a.margin, "iters": a.iters, "cert": a.cert, "centre_m": a.centre_m, "fallback_margin": a.fallback_margin, "damp": a.damp, "damp_cum": a.damp_cum}
    return {"N5A": dict(base, sign=a.sign), "N5X": dict(base, sign=a.sign_x), "N5R": dict(base, sign=a.sign)}


def make_repair(spec: dict, mutation: str = "none"):
    """the repair object an arm runs: n4_repair_l1.make_l1 with EXPLICIT parameters (readback() reads what the built object uses)"""
    spec = dict(spec)
    if mutation == "margin_neg":
        spec["margin"] = -0.05
    cert = spec["cert"]
    fn = L1.make_l1(margin=spec["margin"], iters=spec["iters"], sign=spec["sign"], cert="full" if mutation == "cert_off" else cert, centre_m=spec["centre_m"],
                    fallback_margin=spec["fallback_margin"], damp=spec["damp"], damp_cum=spec["damp_cum"])
    if mutation == "cert_off":
        fn = functools.partial(fn.func, **dict(fn.keywords, cert=_cert_off))
    return fn


# ------------------------------------------------------------------------------------------------------ reachability
class _Fakes:
    """synthetic scenario factories borrowed from the PINNED test file (FakeCx, corridor, teacher): one definition, pinned"""

    def __init__(self):
        import test_n4_repair_l1 as T
        self.T, self.np = T, np
        T_ = T

        class PingPong(T_.FakeCx):
            """an adversarial tracker: the executed path is the plan plus +-0.3 m alternating per scored call, so the dominant correction flips sign"""

            def __init__(self, drivable):
                super().__init__(drivable)
                self.k = 0

            def _sim(self, plan):
                S = super()._sim(plan)
                self.k += 1
                S[:, 1] += 0.3 * (-1) ** self.k
                return S

        class NcWorse(T_.FakeCx):
            def score(self, plans):
                res, s = super().score(plans)
                return [dict(r, nc=0.0) for r in res], s

        self.PingPong, self.NcWorse = PingPong, NcWorse


_F = None


def fakes() -> _Fakes:
    global _F
    if _F is None:
        _F = _Fakes()
    return _F


def _r(x):
    return None if x is None else (round(float(x), 9) if not isinstance(x, (list, tuple)) else [round(float(v), 9) for v in x])


def _run_sig(cx, t20, fn, base):
    info = {}
    res = fn(cx, t20, base, info=info)
    its = info["iters"]
    return {"status": info["status"], "n_labels": cx.n_labels, "n_flips": info["n_flips"], "applied": res is not None,
            "plan_sha1": None if res is None else hashlib.sha1(np.ascontiguousarray(res["plan"], dtype=np.float64).tobytes()).hexdigest(),
            "width_min": [_r(e.get("interval_width_min_m")) for e in its], "s_max": [_r(e.get("s_max_abs_m")) for e in its],
            "relaxed": [len(e["relaxed_steps"]) for e in its], "factor": [_r(e.get("factor")) for e in its]}


def signature(fn) -> dict:
    """behavioural fingerprint of a built L1 object on four synthetic scenarios; every parameter of the card changes at least one entry"""
    F = fakes()
    T = F.T
    base_ok = dict(T.BASE_OK)
    t_s1 = T.teacher(lambda t: np.where(t < 2.0, 0.30, -0.16))
    sig = {}
    sig["S1_two_sided"] = _run_sig(T.FakeCx(T.corridor(0.50, 0.34)), t_s1, fn, base_ok)
    sig["S2_narrow"] = _run_sig(T.FakeCx(T.corridor(0.06, 0.06)), T.teacher(lambda t: 0.2 + 0 * t), fn, base_ok)
    sig["S3_pingpong"] = _run_sig(F.PingPong(T.corridor(0.30, 0.30)), T.teacher(lambda t: 0.0 * t), fn, base_ok)
    sig["S4_nc_worse"] = _run_sig(F.NcWorse(T.corridor(1.0, 1.0)), T.teacher(lambda t: 0.0 * t), fn, base_ok)
    return sig


def check_arm(arm: str, fn, spec: dict | None = None) -> dict:
    """read-back + behavioural signature compared with the card; ok False (with the reasons) on any difference"""
    spec = dict(CARD[arm] if spec is None else spec)
    rb, bad = L1.readback(fn), []
    for k in ("margin", "iters", "sign", "cert", "centre_m", "fallback_margin", "damp", "damp_cum"):
        if rb[k] != CARD[arm][k]:
            bad.append(f"{k}: built {rb[k]!r} != card {CARD[arm][k]!r}")
    if rb["target"] != TARGET:
        bad.append(f"repair function {rb['target']!r} is not {TARGET}")
    sg, ref = signature(fn), signature(make_repair(CARD[arm]))
    diff = [k for k in ref if sg.get(k) != ref[k]]
    if diff:
        bad.append("behavioural signature differs from the card object's on " + ",".join(diff))
    return {"arm": arm, "ok": not bad, "bad": bad, "readback": rb, "signature_sha1": hashlib.sha1(json.dumps(sg, sort_keys=True).encode()).hexdigest()}


def reach_real(fixture: str) -> dict:
    """the two real navtrain failing instances (nuPlan DB rebuild, builder scorer): zero-correction identity, + certifies, - does not, determinism"""
    import n4_geom as G
    fx = json.load(open(fixture, encoding="utf-8"))
    out = {"fixture_sha256": sha256(fixture), "instances": {}, "ok": True}
    for key in sorted(fx):
        inst = fx[key]
        cx = None
        for _lg, _tk, c in G.contexts({inst["log"]: [inst["token"]]}):
            cx = c
        t20 = np.asarray(inst["t20"], float)
        lab = cx.labels(t20)["R"]
        base = cx.score([lab])[0][0]
        rec = {"base_dac": float(base["dac"])}
        info = {}
        r = L1.make_l1()(cx, t20, base, info=info)
        r2 = L1.make_l1()(cx, t20, base)
        rx = L1.make_l1(sign=-1.0)(cx, t20, base)
        zi = {}
        z = L1.repair_l1(cx, t20, base, cert=lambda c, b: True, info=zi)
        rec.update({"l1_certified": r is not None and RP.certified(cx.score([r["plan"]])[0][0], base), "l1_status": info["status"],
                    "sign_flipped_certified": rx is not None, "deterministic": bool(r is not None and r2 is not None and np.array_equal(r["plan"], r2["plan"])),
                    "zero_identity": bool(zi["status"] == "PASSED_AT_0" and np.array_equal(z["plan"], lab))})
        rec["ok"] = bool(rec["base_dac"] == 0.0 and rec["l1_certified"] and not rec["sign_flipped_certified"] and rec["deterministic"] and rec["zero_identity"])
        out["instances"][key] = rec
        out["ok"] = out["ok"] and rec["ok"]
    return out


def cmd_reach(a) -> int:
    P = params_of(a)
    reps = [check_arm(arm, make_repair(P[arm]), P[arm]) for arm in ARMS]
    # check_arm compares the built object with the CARD (not with itself): the card is the second argument of the comparison
    ok = all(r["ok"] for r in reps)
    real = reach_real(a.fixture) if a.real else None
    ok = ok and (real is None or real["ok"])
    print("ZZN5_REACH", "PASS" if ok else "FAIL", json.dumps({r["arm"]: r["readback"] for r in reps}, default=str), flush=True)
    for r in reps:
        for b in r["bad"]:
            print("ZZN5_REACH BAD", r["arm"], b, flush=True)
    if real is not None:
        print("ZZN5_REACH REAL", json.dumps(real), flush=True)
    if a.out:
        json.dump({"ok": bool(ok), "arms": reps, "real": real}, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------ derive
def info_summary(info: dict) -> dict:
    its = info.get("iters", [])
    return {"status": info.get("status"), "n_flips": info.get("n_flips"), "n_iters": len(its), "infeasible": any(e.get("infeasible") for e in its),
            "relaxed_steps_total": int(sum(len(e.get("relaxed_steps", [])) for e in its)), "window_conflicts_total": int(sum(e.get("window_conflicts", 0) for e in its)),
            "ext_max_m_seq": [round(float(e["ext_max_m"]), 4) for e in its]}


def run_l1(fn, cx, t20, base) -> tuple:
    info = {}
    r = fn(cx, t20, base, info=info)
    return r, info_summary(info)


def l1_set_of(a, toks) -> set:
    """the tokens N5-A / N5-X run on: the banked N4-D `unrepaired_dac0` (card section 3) or an explicit smoke list"""
    if a.l1_tokens:
        s = set(json.load(open(a.l1_tokens, encoding="utf-8"))["tokens"])
    elif a.n4d_report:
        s = set(json.load(open(a.n4d_report, encoding="utf-8"))["unrepaired_dac0"]["N4D"])
    else:
        raise SystemExit("REFUSING: derive needs --n4d-report (the card's L1 set) or --l1-tokens (smoke)")
    unknown = s - set(toks)
    if unknown:
        raise SystemExit(f"REFUSING: {len(unknown)} L1-set tokens are not in the N2-R plan file, e.g. {sorted(unknown)[:3]}")
    return s


def cmd_derive(a) -> int:
    if a.mutate != "none" and not a.smoke_allow_mutation:
        raise SystemExit("REFUSING: --mutate is a navtrain-smoke instrument and needs --smoke-allow-mutation (the chain never passes either)")
    P = params_of(a)
    fns = {arm: make_repair(P[arm], a.mutate) for arm in ARMS}
    chk = {arm: check_arm(arm, fns[arm], P[arm]) for arm in ARMS}
    if not all(c["ok"] for c in chk.values()) and not a.smoke_allow_mutation:
        raise SystemExit("REFUSING: the built repair differs from the card: " + json.dumps({k: v["bad"] for k, v in chk.items() if not v["ok"]}))
    z = np.load(a.n2r_plans, allow_pickle=False)
    toks, logs, plans = [str(t) for t in z["tokens"]], [str(x) for x in z["logs"]], np.asarray(z["plans20"], float)
    ix = {t: i for i, t in enumerate(toks)}
    lset = l1_set_of(a, toks)
    rolls = load_rolls(a.rolls)
    if a.navtrain_tokens:
        keep = set(json.load(open(a.navtrain_tokens, encoding="utf-8"))["tokens"])
        toks = [t for t in toks if t in keep]
    si, sn = (int(x) for x in a.shard.split("/"))
    cder = set(sorted(toks, key=sha1_int)[:N_CDERIV])
    done = done_tokens(a.out)
    todo = {t: logs[ix[t]] for t in toks if sha1_int(t) % sn == si and t not in done and (not a.cderiv_only or t in cder)}
    missing = [t for t in todo if t not in rolls]
    if missing:
        raise SystemExit(f"REFUSING: {len(missing)} tokens have no RJ5 target row, e.g. {missing[:3]}")
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, "rb") as fb:
            fb.seek(-1, 2)
            torn = fb.read(1) != b"\n"
        if torn:
            open(a.out, "a", encoding="utf-8").write("\n")
    n, t0 = 0, time.time()
    import n4_geom as G
    with open(a.out, "a", encoding="utf-8") as fo:
        for tok, lg, cx in B4.ctx_stream(a, todo):
            ts, rec = time.time(), {"log": lg, "token": tok}
            try:
                base_plan, t20 = plans[ix[tok]], rolls[tok]
                base = cx.score([base_plan])[0][0]
                rec["base"] = base
                if tok in cder:
                    rec["cderiv_maxdev_m"] = float(np.abs(cx.labels(t20)["R"] - base_plan).max())
                dac0 = base["dac"] == 0.0
                rec["dac0"] = bool(dac0)
                rec["in_l1_set"] = tok in lset
                rec["arms"] = {}
                if (dac0 or a.mutate == "l1_all") and not a.cderiv_only:
                    run_arms = ["N5R"] + (["N5A", "N5X"] if (tok in lset or a.mutate == "l1_all") else [])
                    for arm in run_arms:
                        r, summ = run_l1(fns[arm], cx, t20, base)
                        if r is None:
                            rec["arms"][arm] = {"applied": False, "info": summ}
                        else:
                            rec["arms"][arm] = {"applied": True, "plan": np.asarray(r["plan"]).tolist(), "m": r["m"], "param": r["param"], "info": summ,
                                                "m_recheck": cx.score([r["plan"]])[0][0]}
                    # C-DET: the repair run a SECOND time (N5-R object) on every L1-run token; the N5-A / N5-R pair on the L1 set is a third bit-for-bit comparison
                    r2, _ = run_l1(fns["N5R"], cx, t20, base)
                    f = rec["arms"]["N5R"]
                    rec["det_equal"] = bool((r2 is None and not f["applied"]) or (r2 is not None and f["applied"] and np.array_equal(np.asarray(r2["plan"]), np.asarray(f["plan"]))))
                    if "N5A" in rec["arms"]:
                        a_r = rec["arms"]["N5A"]
                        rec["a_equals_r"] = bool(a_r["applied"] == f["applied"] and (not f["applied"] or a_r["plan"] == f["plan"]))
                rec["label_ego"] = cx.label_ego_src if cx.label_ego_src else "none"
                rec["t_s"] = round(time.time() - ts, 3)
            except G.LabelEgoMissing:
                raise                                                      # REFUSE: never an error row, never a cache-ego fallback
            except Exception as e:                                    # noqa: BLE001
                rec = {"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            n += 1
    print("ZZN5_DERIVE_SHARD_DONE", a.shard, n, round(time.time() - t0, 1), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------ merge
def check_unch(out_dir: str, n4d_plans: str, n4d_seam: str, n2r_plans: str, applied: dict, n_l1_run: int, n_l1_expect: int) -> dict:
    """C-UNCH on FILES: for N5-A and N5-X the plan20 AND the seam pose array of every token outside the L1-applied set is bit-identical to the banked N4-D's, the
    changed set equals the applied set exactly, and the count of L1-run tokens equals the card's 25; N5-R (builder-only, plans only) vs the banked N2-R plans."""
    zb, sb, zr = np.load(n4d_plans, allow_pickle=False), np.load(n4d_seam, allow_pickle=False), np.load(n2r_plans, allow_pickle=False)
    bt = [str(t) for t in zb["tokens"]]
    bi = {t: i for i, t in enumerate(bt)}
    sbi = {str(t): i for i, t in enumerate(sb["token"])}
    out = {"pass": True, "arms": {}, "n_l1_run": n_l1_run, "n_l1_expected": n_l1_expect}
    for arm in ARMS:
        fp = os.path.join(out_dir, f"n5_plans_{arm}.npz")
        sp = os.path.join(out_dir, SEAM_LABEL.get(arm, "none") + ".npz")
        need = (fp, sp) if arm in SEAM_ARMS else (fp,)
        if not all(os.path.exists(p) for p in need):
            out["arms"][arm] = {"pass": False, "why": "missing file"}
            out["pass"] = False
            continue
        z = np.load(fp, allow_pickle=False)
        ref, refp = (zb, bi) if arm in SEAM_ARMS else (zr, {str(t): i for i, t in enumerate(zr["tokens"])})
        zt = [str(t) for t in z["tokens"]]
        changed_p = {t for i, t in enumerate(zt) if t in refp and not np.array_equal(z["plans20"][i], ref["plans20"][refp[t]])}
        want = set(applied.get(arm, []))
        ok = changed_p == want and set(zt) == set(refp)
        rec = {"changed_plan20": len(changed_p), "applied": len(want), "tokens": len(zt), "unflagged_changed": sorted(changed_p - want)[:5], "flagged_unchanged": sorted(want - changed_p)[:5]}
        if arm in SEAM_ARMS:
            s = np.load(sp, allow_pickle=False)
            stt = [str(t) for t in s["token"]]
            changed_s = {t for i, t in enumerate(stt) if t in sbi and not np.array_equal(s["poses"][i], sb["poses"][sbi[t]])}
            ok = ok and changed_s == want and set(stt) == set(sbi)
            rec.update({"changed_seam": len(changed_s), "seam_tokens": len(stt)})
        rec["pass"] = bool(ok)
        out["arms"][arm] = rec
        out["pass"] = out["pass"] and ok
    out["l1_run_count_ok"] = bool(n_l1_run == n_l1_expect)
    out["pass"] = bool(out["pass"] and out["l1_run_count_ok"])
    return out


def read_rows(patterns) -> tuple:
    rows, n_err, n_dup = {}, 0, 0
    for pat in patterns:
        for fp in sorted(glob.glob(pat)):
            for ln in open(fp, encoding="utf-8"):
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    n_err += 1
                elif r["token"] in rows:
                    n_dup += 1
                else:
                    rows[r["token"]] = r
    return rows, n_err, n_dup


def cmd_merge(a) -> int:
    import n4_geom as G
    ND = G.ND
    P = {arm: CARD[arm] for arm in ARMS}
    rows, n_err, n_dup = read_rows(a.rows)
    zr = np.load(a.n2r_plans, allow_pickle=False)
    bt, bl, bp = [str(t) for t in zr["tokens"]], [str(x) for x in zr["logs"]], np.asarray(zr["plans20"], float)
    zd = np.load(a.n4d_plans, allow_pickle=False)
    dt, dpl = [str(t) for t in zd["tokens"]], np.asarray(zd["plans20"], float)
    assert dt == bt, "N4-D plan file token order != N2-R's"
    sb = np.load(a.n4d_seam, allow_pickle=False)
    order = [str(t) for t in sb["token"]]
    ix = {t: i for i, t in enumerate(bt)}
    rep: dict = {"n_tokens": len(bt), "n_rows": len(rows), "n_error_lines": n_err, "n_duplicate_lines": n_dup, "controls": {}, "diagnostics": {}, "files": {}}
    C = rep["controls"]
    C["C-COVB"] = {"pass": set(rows) == set(bt) and n_err == 0 and n_dup == 0, "missing": len(set(bt) - set(rows)), "extra": len(set(rows) - set(bt)),
                   "error_lines": n_err, "duplicate_lines": n_dup}
    ok_t = [t for t in bt if t in rows]
    ref = B4.load_csv_dac(a.harness_csv_n2r)
    mism = [t for t in bt if t not in ref or (t in rows and rows[t]["base"]["dac"] != ref[t])]
    C["C-DACMAP"] = {"pass": not mism and len(ref) >= len(bt), "tokens": len(bt), "mismatches": len(mism), "examples": mism[:5],
                     "what": "builder tracked DAC of the unmodified N2-R plan == harness DAC, all tokens", "reference": os.path.basename(a.harness_csv_n2r)}
    cd = sorted(bt, key=sha1_int)[:N_CDERIV]
    dev = [rows[t].get("cderiv_maxdev_m", float("nan")) for t in cd if t in rows]
    C["C-DERIV"] = {"pass": len(dev) == len(cd) and bool(np.nanmax(dev) <= CDERIV_BAR_M) and not any(np.isnan(dev)), "n": len(dev), "expected_n": len(cd),
                    "max_dev_m": float(np.nanmax(dev)) if dev else None, "bar_m": CDERIV_BAR_M}
    lab_rows = [t for t in ok_t if "cderiv_maxdev_m" in rows[t] or rows[t]["base"]["dac"] == 0.0]
    not_db = sorted(t for t in lab_rows if rows[t].get("label_ego") != "db")
    C["C-LABELEGO"] = {"pass": not not_db, "label_rows": len(lab_rows), "db": len(lab_rows) - len(not_db), "not_db": len(not_db), "examples": not_db[:5],
                       "what": "every row that built labels (C-DERIV tokens + builder DAC-0 tokens x up to 5 iterations) used the nuPlan-DB ego"}
    dac0 = sorted(t for t in ok_t if rows[t]["base"]["dac"] == 0.0)
    l1_run = sorted(t for t in ok_t if "N5A" in rows[t]["arms"])
    applied = {arm: sorted(t for t in ok_t if rows[t]["arms"].get(arm, {}).get("applied")) for arm in ARMS}
    spurious = sorted(t for arm in ARMS for t in applied[arm] if rows[t]["base"]["dac"] != 0.0)
    det = {t: rows[t]["det_equal"] for t in ok_t if "det_equal" in rows[t]}
    a_eq_r = [rows[t]["a_equals_r"] for t in ok_t if "a_equals_r" in rows[t]]
    C["C-DET"] = {"pass": bool(det) and all(det.values()) and set(det) == set(dac0) and all(a_eq_r), "checked": len(det), "equal": int(sum(det.values())), "dac0": len(dac0),
                  "n5a_equals_n5r_on_l1_set": f"{int(sum(a_eq_r))}/{len(a_eq_r)}",
                  "what": "the repair run twice on EVERY builder DAC-0 token (card: every L1-repaired + 20 of the remaining) gives bit-identical plans"}
    # C-SET: the builder DAC-0 set and the L1 set are exactly the banked N4-D build's (116 = 91 repaired + 25 unrepaired); the 25 are the card's residual set
    n4d = json.load(open(a.n4d_report, encoding="utf-8"))
    banked_dac0 = sorted(set(n4d["repaired"]["N4D"]) | set(n4d["unrepaired_dac0"]["N4D"]))
    banked_l1 = sorted(n4d["unrepaired_dac0"]["N4D"])
    res_ok, res_note = True, None
    if a.residual:
        res = json.load(open(a.residual, encoding="utf-8"))
        res_ok = sorted(res["all_25"]) == banked_l1 and sorted(res["guard_blocked"] + res["non_guard"]) == banked_l1
        res_note = {"guard_blocked": len(res["guard_blocked"]), "non_guard": len(res["non_guard"])}
    C["C-SET"] = {"pass": bool(dac0 == banked_dac0 and l1_run == banked_l1 and res_ok), "builder_dac0": len(dac0), "banked_dac0": len(banked_dac0), "l1_run": len(l1_run),
                  "banked_l1_set": len(banked_l1), "residual_split_matches": res_ok, "residual_split": res_note,
                  "what": "builder DAC-0 set == banked N4-D build's (repaired + unrepaired); L1-run set == its unrepaired_dac0 == the card's residual split"}
    bad_cert = {}
    for arm in ARMS:
        cf = RP.CERTS[P[arm]["cert"]]
        bad_cert[arm] = [t for t in applied[arm] if not cf(rows[t]["arms"][arm]["m_recheck"], rows[t]["base"]) or rows[t]["arms"][arm]["m_recheck"]["dac"] != 1.0]
    C["C-CERT-B"] = {"pass": not any(bad_cert.values()), "violations": {k: len(v) for k, v in bad_cert.items()}, "examples": {k: v[:3] for k, v in bad_cert.items()},
                     "what": "every applied plan, re-scored by a separate scorer call, has DAC 1 and passes the certificate against the N2-R base"}
    # plans / seams
    Pl = {"N5A": dpl.copy(), "N5X": dpl.copy(), "N5R": bp.copy()}
    for arm in ARMS:
        for t in applied[arm]:
            Pl[arm][ix[t]] = np.asarray(rows[t]["arms"][arm]["plan"], float)
    os.makedirs(a.out_dir, exist_ok=True)
    for arm in ARMS:
        fp = os.path.join(a.out_dir, f"n5_plans_{arm}.npz")
        np.savez(fp, tokens=np.array(bt), logs=np.array(bl), plans20=Pl[arm], arm=np.array(arm))
        rep["files"][os.path.basename(fp)] = sha256(fp)
    # C-SEAMFN: ND.to_navsim reproduces the banked N4-D seam poses bit-for-bit on every token, so the changed rows are made by the function that made the banked ones
    p8 = np.stack([ND.to_navsim(dpl[ix[t]]).astype(np.float32) for t in order])
    bad_fn = [order[i] for i in range(len(order)) if not np.array_equal(p8[i], sb["poses"][i])]
    C["C-SEAMFN"] = {"pass": not bad_fn, "tokens": len(order), "mismatches": len(bad_fn), "examples": bad_fn[:3]}
    for arm in SEAM_ARMS:
        p8a = np.stack([ND.to_navsim(Pl[arm][ix[t]]).astype(np.float32) if t in set(applied[arm]) else sb["poses"][i] for i, t in enumerate(order)])
        old = str(sb["arm"])
        new_arm = (old[: -len("refe_tfn4_n4d")] if old.endswith("refe_tfn4_n4d") else old + "_") + SEAM_LABEL[arm]
        sp = os.path.join(a.out_dir, SEAM_LABEL[arm] + ".npz")
        np.savez(sp, token=sb["token"], fingerprint=sb["fingerprint"], poses=p8a, sampling=sb["sampling"], arm=np.array(new_arm))
        rep["files"][os.path.basename(sp)] = sha256(sp)
    u = check_unch(a.out_dir, a.n4d_plans, a.n4d_seam, a.n2r_plans, {arm: applied[arm] for arm in ARMS}, len(l1_run), a.expect_l1_run)
    u["spurious_applied_on_dac1_tokens"] = len(spurious)
    u["pass"] = bool(u["pass"]) and not spurious
    C["C-UNCH"] = u
    if a.reach_json:
        rj = json.load(open(a.reach_json, encoding="utf-8"))
        C["C-REACH"] = {"pass": bool(rj.get("ok") and rj.get("real") and rj["real"].get("ok")), "arms_ok": {x["arm"]: x["ok"] for x in rj.get("arms", [])},
                        "real": {k: v.get("ok") for k, v in (rj.get("real") or {}).get("instances", {}).items()}}
    else:
        C["C-REACH"] = {"pass": False, "why": "no --reach-json (the chain's reach_real stage)"}
    rep["dac0_tokens_builder"] = len(dac0)
    rep["l1_run"] = l1_run
    rep["repaired"] = applied
    rep["unrepaired_l1"] = {arm: sorted(set(l1_run) - set(applied[arm])) for arm in ("N5A", "N5X")}
    rep["unrepaired_dac0"] = {arm: sorted(set(dac0) - set(applied[arm])) for arm in ARMS}
    rep["diagnostics"]["status_counts"] = {arm: {s: sum(1 for t in ok_t if rows[t]["arms"].get(arm, {}).get("info", {}).get("status") == s)
                                                for s in ("PASSED_AT_0", "CERTIFIED", "CERT_BLOCKED", "INFEASIBLE", "CAP_EXHAUSTED")} for arm in ARMS}
    rep["diagnostics"]["l1_set_status"] = {t: rows[t]["arms"]["N5A"]["info"] for t in l1_run}
    ts = [r["t_s"] for r in rows.values() if "t_s" in r]
    rep["timing"] = {"n": len(ts), "t_s_mean": float(np.mean(ts)) if ts else None, "cpu_h_sum": float(np.sum(ts) / 3600.0) if ts else None}
    if a.a1_base_rows:                                           # diagnostic only: builder base scores vs TF-N4 A1's (same code, same inputs)
        a1, _, _ = read_rows([a.a1_base_rows])
        worst = max((abs(rows[t]["base"][k] - a1[t]["base"][k]) for t in ok_t if t in a1 for k in ("dac", "nc", "ddc", "ttc", "c", "ep", "pdms")), default=None)
        rep["diagnostics"]["base_vs_a1_builder_max_abs"] = worst
    allpass = all(c["pass"] for c in C.values())
    rep["verdict"] = "PASS" if allpass else "FAIL"
    json.dump(rep, open(os.path.join(a.out_dir, "n5_build_report.json"), "w", encoding="utf-8"), indent=1)
    print("ZZN5_BUILD", rep["verdict"], json.dumps({k: v["pass"] for k, v in C.items()}), "repaired", {k: len(v) for k, v in applied.items()}, "dac0", len(dac0), "l1_run", len(l1_run), flush=True)
    return 0 if allpass else 1


def cmd_unch(a) -> int:
    rep = json.load(open(os.path.join(a.out_dir, "n5_build_report.json"), encoding="utf-8"))
    u = check_unch(a.out_dir, a.n4d_plans, a.n4d_seam, a.n2r_plans, rep["repaired"], len(rep["l1_run"]), a.expect_l1_run)
    print("ZZN5_UNCH", "PASS" if u["pass"] else "FAIL", json.dumps(u["arms"]), flush=True)
    if a.out:
        json.dump(u, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if u["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("reach", "derive", "merge", "unch"))
    c = CARD["N5A"]
    ap.add_argument("--margin", type=float, default=c["margin"])
    ap.add_argument("--iters", type=int, default=c["iters"])
    ap.add_argument("--cert", default=c["cert"])
    ap.add_argument("--centre-m", type=float, default=c["centre_m"])
    ap.add_argument("--fallback-margin", type=float, default=c["fallback_margin"])
    ap.add_argument("--damp", type=float, default=c["damp"])
    ap.add_argument("--damp-cum", action="store_true", default=c["damp_cum"])
    ap.add_argument("--sign", type=float, default=c["sign"])
    ap.add_argument("--sign-x", type=float, default=CARD["N5X"]["sign"])
    ap.add_argument("--route", choices=("cache", "navtrain"), default="cache")
    ap.add_argument("--n2r-plans")
    ap.add_argument("--n2r-seam", default="")
    ap.add_argument("--n4d-plans")
    ap.add_argument("--n4d-seam")
    ap.add_argument("--n4d-report", default="")
    ap.add_argument("--l1-tokens", default="")
    ap.add_argument("--residual", default="")
    ap.add_argument("--rolls", default="")
    ap.add_argument("--metric-cache-root", default="")
    ap.add_argument("--navtrain-tokens", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--mutate", choices=MUTATIONS, default="none")
    ap.add_argument("--cderiv-only", action="store_true")
    ap.add_argument("--smoke-allow-mutation", action="store_true")
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--harness-csv-n2r", default="")
    ap.add_argument("--a1-base-rows", default="")
    ap.add_argument("--reach-json", default="")
    ap.add_argument("--expect-l1-run", type=int, default=N_L1_EXPECT)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--fixture", default=os.path.join(HERE, "fixture_real_instances.json"))
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    return {"reach": cmd_reach, "derive": cmd_derive, "merge": cmd_merge, "unch": cmd_unch}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
