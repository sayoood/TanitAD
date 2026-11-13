#!/usr/bin/env python3
"""TF-N4 navtest build (PREREG_TF_N4.md sections 3, 4, 7): the plans of the three REPAIR-D arms from the banked N2-R plans. CPU only.

  reach   python n4_nt_build.py reach            -> reads margin / cap / sign / certificate BACK from the built repair objects of all three arms, compares
                                                    them with the card, and runs a stub-scorer probe that shows the margin, the cap and the sign REACH the
                                                    shift and the loop (not just the namespace). Exit 1 on any difference. Needs no data.
  derive  python n4_nt_build.py derive --route cache|navtrain --n2r-plans <n2_plans_N2R.npz> --rolls "<RJ5 rows glob>" --out <shard.jsonl> --shard i/N
              [--metric-cache-root <root>] [--navtrain-tokens <json>] [--mutate none|cert_off|repair_all|margin_neg --smoke-allow-mutation]
          Per token (sha1 % N == i, resumable): builder TRACKED score of the N2-R plan (n4_geom.Ctx.score = transp_score.score_arms on the harness's own
          metric cache); if its DAC is 0 -> REPAIR-D / N4-X / N4-DM (frozen n4_repair.repair_target, parameters read back, see reach). Each applied
          plan is re-scored by a SEPARATE scorer call (m_recheck). C-DERIV: the 200 smallest-sha1 tokens re-derive the N2-R label from the RJ5 target.
          C-DET: the first DET_PER_SHARD repaired tokens of the shard are repaired twice.
  merge   python n4_nt_build.py merge ... --rows "<shards>" --seam <T seam npz> --n2r-seam <banked N2R seam npz> --harness-csv-n2r <csv> --out-dir <dir>
          -> n4_plans_{N4D,N4X,N4DM}.npz (20-pose plans, TF-N2 A1 format), refe_tfn4_{n4d,n4x,n4dm}.npz (W3 seams), n4_build_report.json with
             C-COVB, C-DACMAP, C-DERIV, C-UNCH, C-DET, C-CERT-B. Prints ZZN4_BUILD PASS|FAIL.
  unch    python n4_nt_build.py unch --out-dir <dir> --n2r-plans <npz> --n2r-seam <npz>      -> re-runs C-UNCH on the files ON DISK (the chain does this
          just before the harness scores them, so an edit between build and score is caught)."""
from __future__ import annotations

import argparse
import csv
import functools
import glob
import hashlib
import inspect
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):          # HERE ends up first: the chain's own n4_geom / n4_repair / n4_common win
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import numpy as np

# the card (PREREG_TF_N4.md section 4), the ONLY place the numbers live in the chain
CARD = {"N4D": {"margin": 0.05, "iters": 5, "sign": 1.0, "cert": "full"},
        "N4X": {"margin": 0.05, "iters": 5, "sign": -1.0, "cert": "full"},
        "N4DM": {"margin": 0.05, "iters": 5, "sign": 1.0, "cert": "dac"}}
ARMS = tuple(CARD)
SEAM_LABEL = {"N4D": "refe_tfn4_n4d", "N4X": "refe_tfn4_n4x", "N4DM": "refe_tfn4_n4dm"}
N_CDERIV, CDERIV_BAR_M = 200, 1e-9
DET_N, DET_PER_SHARD = 50, 25
MUTATIONS = ("none", "cert_off", "repair_all", "margin_neg", "label_ego_cache")
MET8 = ("nc", "dac", "ddc", "ep", "ttc", "c", "pdms")
CSV_COL = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ep": "ego_progress", "ttc": "time_to_collision_within_bound",
           "c": "comfort", "ddc": "driving_direction_compliance", "pdms": "score"}


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha1_int(t: str) -> int:
    return int(hashlib.sha1(t.encode()).hexdigest(), 16)


def _rp():
    import n4_repair as RP
    return RP


def _cert_off(c, b):
    return True


def certs_all():
    RP = _rp()
    return dict(RP.CERTS, off=_cert_off)          # "off" exists only to be reachable by the cert_off mutation arm


def make_repair(arm: str, mutation: str = "none"):
    """the repair object an arm runs: a functools.partial of the frozen n4_repair.repair_target with EXPLICIT margin / iters / sign / cert"""
    RP = _rp()
    spec = dict(CARD[arm])
    if mutation == "margin_neg":
        spec["margin"] = -0.05
    if mutation == "cert_off":
        spec["cert"] = "off"
    return functools.partial(RP.repair_target, margin=spec["margin"], iters=spec["iters"], sign=spec["sign"], cert=certs_all()[spec["cert"]])


def readback(fn) -> dict:
    """margin / cap / sign / certificate as the BUILT object will use them (partial keywords, else the function's own defaults)"""
    RP = _rp()
    f = fn.func if isinstance(fn, functools.partial) else fn
    kw = dict(fn.keywords) if isinstance(fn, functools.partial) else {}
    sig = inspect.signature(f).parameters
    get = lambda k: kw[k] if k in kw else sig[k].default                      # noqa: E731
    cert = get("cert")
    return {"target": f"{f.__module__}.{f.__name__}", "is_frozen_repair_target": f is RP.repair_target, "margin": get("margin"), "iters": get("iters"),
            "sign": get("sign"), "cert": next((n for n, c in certs_all().items() if c is cert), "UNKNOWN")}


class _Stub:
    def __init__(self, res):
        self.res, self.targets = res, []

    def labels(self, t):
        self.targets.append(np.array(t, float))
        return {"R": np.array(t, float)}

    def score(self, plans):
        return [dict(self.res) for _ in plans], [None for _ in plans]


def reach_probe(fn) -> dict:
    """what the built repair DOES with a stub scorer: shift of the last pose after the cap, number of label builds, certificate behaviour"""
    RP = _rp()
    keep = RP.return_vector
    RP.return_vector = lambda cx, s: (0.10, 2.0, np.array([0.0, 1.0]))
    try:
        base = {"dac": 0.0, "nc": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0}
        st = _Stub(dict(base))
        out1 = fn(st, np.zeros((20, 3)), base)                                  # never certifies: runs the whole cap
        y_last = float(st.targets[-1][-1, 1])
        st2 = _Stub({"dac": 1.0, "nc": 0.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0})    # DAC repaired, NC worse
        out2 = fn(st2, np.zeros((20, 3)), base)
    finally:
        RP.return_vector = keep
    return {"never_certifies_returns_none": out1 is None, "label_builds_at_cap": len(st.targets), "last_pose_y_after_cap_m": y_last,
            "dac1_nc_worse_accepted": out2 is not None}


def check_arm(arm: str, fn) -> dict:
    """readback + probe compared with the card; ok False (with the reasons) on any difference"""
    rb, pr, spec = readback(fn), reach_probe(fn), CARD[arm]
    bad = []
    if not rb["is_frozen_repair_target"]:
        bad.append("repair function is not n4_repair.repair_target")
    for k in ("margin", "iters", "sign", "cert"):
        if rb[k] != spec[k]:
            bad.append(f"{k}: built {rb[k]!r} != card {spec[k]!r}")
    if pr["label_builds_at_cap"] != spec["iters"] + 1:
        bad.append(f"label builds at cap {pr['label_builds_at_cap']} != {spec['iters'] + 1}")
    want_y = spec["sign"] * spec["iters"] * (0.10 + spec["margin"])
    if abs(pr["last_pose_y_after_cap_m"] - want_y) > 1e-12:
        bad.append(f"shift reaching the target {pr['last_pose_y_after_cap_m']!r} != card {want_y!r}")
    if not pr["never_certifies_returns_none"]:
        bad.append("probe 1 certified")
    if pr["dac1_nc_worse_accepted"] != (spec["cert"] == "dac"):
        bad.append(f"certificate behaviour: NC-worse candidate accepted = {pr['dac1_nc_worse_accepted']}, card cert {spec['cert']!r}")
    return {"arm": arm, "ok": not bad, "bad": bad, "readback": rb, "probe": pr}


def cmd_reach(a) -> int:
    reps = [check_arm(arm, make_repair(arm)) for arm in ARMS]
    ok = all(r["ok"] for r in reps)
    print("ZZN4_REACH", "PASS" if ok else "FAIL", json.dumps({r["arm"]: r["readback"] for r in reps}), flush=True)
    for r in reps:
        for b in r["bad"]:
            print("ZZN4_REACH BAD", r["arm"], b, flush=True)
    if a.out:
        json.dump(reps, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------ rows
def load_rolls(pattern: str) -> dict:
    out = {}
    for fp in sorted(glob.glob(pattern)):
        for ln in open(fp, encoding="utf-8"):
            if ln.strip():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "traj20" in r and "error" not in r:
                    out[r["token"]] = np.asarray(r["traj20"], float)
    return out


def done_tokens(path: str) -> set:
    d = set()
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" not in r and "token" in r:
                d.add(r["token"])
    return d


def db_ego_fn(dbs: dict, lg: str, tok: str):
    """A1: lazy fetch of the token's nuPlan-DB ego (scenario.initial_ego_state, as TF-N2's tfn2_nt_build.py:153-157); only called when labels() runs.
    Raises G.LabelEgoMissing when the log has no DB or the scenario builder does not return exactly this token."""
    import n4_geom as G

    def fn():
        if lg not in dbs:
            raise G.LabelEgoMissing(f"no nuPlan DB for log {lg} (token {tok})")
        scs = list(G.NS.build_scenarios_for_log(dbs[lg], [tok]))
        if len(scs) != 1 or scs[0]._initial_lidar_token != tok:
            raise G.LabelEgoMissing(f"DB scenario for token {tok} not built ({len(scs)} returned)")
        return scs[0].initial_ego_state
    return fn


def ctx_stream(a, todo: dict):
    """yield (token, log, Ctx); cache route = the harness's own metric cache (card section 3), navtrain route = nuPlan-DB rebuild (smoke only)"""
    import n4_geom as G
    if a.route == "cache":
        import transp_score as TS
        dbs = G.NS.index_dbs(db_root=G.DB_ROOT)
        for tok, lg in sorted(todo.items()):
            cx = G.Ctx.from_parts(TS.load_cache_parts(a.metric_cache_root, lg, tok), label_ego_fn=db_ego_fn(dbs, lg, tok))
            cx.force_cache_label_ego = getattr(a, "mutate", "none") == "label_ego_cache"      # smoke-only mutation (needs --smoke-allow-mutation)
            yield tok, lg, cx
    else:
        by: dict = {}
        for t, lg in todo.items():
            by.setdefault(lg, []).append(t)
        for lg, tok, cx in G.contexts(by):
            yield tok, lg, cx


def cmd_derive(a) -> int:
    if a.mutate != "none" and not a.smoke_allow_mutation:
        raise SystemExit("REFUSING: --mutate is a navtrain-smoke instrument and needs --smoke-allow-mutation (the chain never passes either)")
    fns = {arm: make_repair(arm, a.mutate) for arm in ARMS}
    chk = {arm: check_arm(arm, fns[arm]) for arm in ARMS}
    if not all(c["ok"] for c in chk.values()) and not a.smoke_allow_mutation:
        raise SystemExit("REFUSING: the built repair differs from the card: " + json.dumps({k: v["bad"] for k, v in chk.items() if not v["ok"]}))
    z = np.load(a.n2r_plans, allow_pickle=False)
    toks, logs, plans = [str(t) for t in z["tokens"]], [str(x) for x in z["logs"]], np.asarray(z["plans20"], float)
    ix = {t: i for i, t in enumerate(toks)}
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
    n, t0, ndet = 0, time.time(), sum(1 for ln in open(a.out, encoding="utf-8") if '"det_equal"' in ln) if os.path.exists(a.out) else 0
    import n4_geom as G
    with open(a.out, "a", encoding="utf-8") as fo:
        for tok, lg, cx in ctx_stream(a, todo):
            ts, rec = time.time(), {"log": lg, "token": tok}
            try:
                base_plan, t20 = plans[ix[tok]], rolls[tok]
                base = cx.score([base_plan])[0][0]
                rec["base"] = base
                if tok in cder:
                    rec["cderiv_maxdev_m"] = float(np.abs(cx.labels(t20)["R"] - base_plan).max())
                dac0 = base["dac"] == 0.0
                rec["dac0"] = bool(dac0)
                rec["arms"] = {}
                if (dac0 or a.mutate == "repair_all") and not a.cderiv_only:
                    for arm in ARMS:
                        r = fns[arm](cx, t20, base)
                        if r is None:
                            rec["arms"][arm] = {"applied": False}
                        else:
                            rec["arms"][arm] = {"applied": True, "plan": np.asarray(r["plan"]).tolist(), "m": r["m"], "param": r["param"],
                                                "m_recheck": cx.score([r["plan"]])[0][0]}
                    d = rec["arms"].get("N4D", {})
                    if dac0 and d.get("applied") and ndet < DET_PER_SHARD:
                        r2 = fns["N4D"](cx, t20, base)
                        rec["det_equal"] = bool(r2 is not None and np.array_equal(np.asarray(r2["plan"]), np.asarray(d["plan"])))
                        ndet += 1
                rec["label_ego"] = cx.label_ego_src if cx.label_ego_src else "none"             # A1: "none" = no label was built for this token
                rec["t_s"] = round(time.time() - ts, 3)
            except G.LabelEgoMissing:
                raise                                                      # REFUSE: never an error row, never a cache-ego fallback
            except Exception as e:                                    # noqa: BLE001
                rec = {"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            n += 1
    print("ZZN4_DERIVE_SHARD_DONE", a.shard, n, round(time.time() - t0, 1), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------ merge
def load_csv_dac(p: str) -> dict:
    with open(p, encoding="utf-8", newline="") as fh:
        return {r["token"]: float(r[CSV_COL["dac"]]) for r in csv.DictReader(fh) if r["token"] != "average" and r.get("valid", "True") == "True"}


def check_unch(out_dir: str, n2r_plans: str, n2r_seam: str, applied: dict) -> dict:
    """C-UNCH on FILES: every token not applied has a plan20 AND a seam pose array bit-identical to the banked N2-R's; applied tokens are listed"""
    zb, sb = np.load(n2r_plans, allow_pickle=False), np.load(n2r_seam, allow_pickle=False)
    bt = [str(t) for t in zb["tokens"]]
    bi = {t: i for i, t in enumerate(bt)}
    sbi = {str(t): i for i, t in enumerate(sb["token"])}
    out = {"pass": True, "arms": {}}
    for arm in ARMS:
        fp, sp = os.path.join(out_dir, f"n4_plans_{arm}.npz"), os.path.join(out_dir, SEAM_LABEL[arm] + ".npz")
        if not (os.path.exists(fp) and os.path.exists(sp)):
            out["arms"][arm] = {"pass": False, "why": "missing file"}
            out["pass"] = False
            continue
        z, s = np.load(fp, allow_pickle=False), np.load(sp, allow_pickle=False)
        zt = [str(t) for t in z["tokens"]]
        stt = [str(t) for t in s["token"]]
        changed_p = {t for i, t in enumerate(zt) if t in bi and not np.array_equal(z["plans20"][i], zb["plans20"][bi[t]])}
        changed_s = {t for i, t in enumerate(stt) if t in sbi and not np.array_equal(s["poses"][i], sb["poses"][sbi[t]])}
        want = set(applied.get(arm, []))
        ok = (changed_p == want and changed_s == want and set(zt) == set(bt) and set(stt) == set(sbi))
        out["arms"][arm] = {"pass": bool(ok), "changed_plan20": len(changed_p), "changed_seam": len(changed_s), "repaired_flagged": len(want),
                            "unflagged_changed": sorted(changed_p - want)[:5], "flagged_unchanged": sorted(want - changed_p)[:5],
                            "tokens": len(zt), "seam_tokens": len(stt)}
        out["pass"] = out["pass"] and ok
    return out


def cmd_merge(a) -> int:
    import n4_geom as G
    ND = G.ND
    RP = _rp()
    certs = certs_all()
    rows, n_err, n_dup = {}, 0, 0
    for pat in a.rows:
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
    zb = np.load(a.n2r_plans, allow_pickle=False)
    bt, bl, bp = [str(t) for t in zb["tokens"]], [str(x) for x in zb["logs"]], np.asarray(zb["plans20"], float)
    if a.navtrain_tokens:
        keep = set(json.load(open(a.navtrain_tokens, encoding="utf-8"))["tokens"])
        sel = [i for i, t in enumerate(bt) if t in keep]
        bt, bl, bp = [bt[i] for i in sel], [bl[i] for i in sel], bp[sel]
    ix = {t: i for i, t in enumerate(bt)}
    rep: dict = {"n_tokens": len(bt), "n_rows": len(rows), "n_error_lines": n_err, "n_duplicate_lines": n_dup, "controls": {}, "diagnostics": {}, "files": {}}
    C = rep["controls"]
    C["C-COVB"] = {"pass": set(rows) == set(bt) and n_err == 0 and n_dup == 0, "missing": len(set(bt) - set(rows)), "extra": len(set(rows) - set(bt)),
                   "error_lines": n_err, "duplicate_lines": n_dup}
    ok_t = [t for t in bt if t in rows]
    ref = load_csv_dac(a.harness_csv_n2r)
    mism = [t for t in bt if t not in ref or (t in rows and rows[t]["base"]["dac"] != ref[t])]
    C["C-DACMAP"] = {"pass": not mism and len(ref) >= len(bt), "tokens": len(bt), "mismatches": len(mism), "examples": mism[:5],
                     "what": "builder tracked DAC of the unmodified N2-R plan == reference DAC, all tokens",
                     "reference": os.path.basename(a.harness_csv_n2r)}
    cd = sorted(bt, key=sha1_int)[:N_CDERIV]
    dev = [rows[t].get("cderiv_maxdev_m", float("nan")) for t in cd if t in rows]
    C["C-DERIV"] = {"pass": len(dev) == len(cd) and bool(np.nanmax(dev) <= CDERIV_BAR_M) and not any(np.isnan(dev)), "n": len(dev), "expected_n": len(cd),
                    "max_dev_m": float(np.nanmax(dev)) if dev else None, "bar_m": CDERIV_BAR_M}
    lab_rows = [t for t in ok_t if "cderiv_maxdev_m" in rows[t] or rows[t]["base"]["dac"] == 0.0]
    not_db = sorted(t for t in lab_rows if rows[t].get("label_ego") != "db")
    C["C-LABELEGO"] = {"pass": not not_db, "label_rows": len(lab_rows), "db": len(lab_rows) - len(not_db), "not_db": len(not_db), "examples": not_db[:5],
                       "what": "A1 read-back: every row that built labels (C-DERIV tokens + builder DAC-0 tokens) used the nuPlan-DB ego"}
    applied = {arm: sorted(t for t in ok_t if rows[t]["arms"].get(arm, {}).get("applied")) for arm in ARMS}
    dac0 = sorted(t for t in ok_t if rows[t]["base"]["dac"] == 0.0)
    repaired = {arm: sorted(t for t in applied[arm] if rows[t]["base"]["dac"] == 0.0) for arm in ARMS}
    spurious = {arm: sorted(t for t in applied[arm] if rows[t]["base"]["dac"] != 0.0) for arm in ARMS}
    det = [rows[t]["det_equal"] for t in ok_t if "det_equal" in rows[t]]
    C["C-DET"] = {"pass": bool(det) and all(det) and len(det) >= min(DET_N, len(repaired["N4D"])) if repaired["N4D"] else not det or all(det),
                  "checked": len(det), "equal": int(sum(det)), "want": min(DET_N, len(repaired["N4D"]))}
    bad_cert = {}
    for arm in ARMS:
        cf = certs[CARD[arm]["cert"]]
        bad_cert[arm] = [t for t in applied[arm] if not cf(rows[t]["arms"][arm]["m_recheck"], rows[t]["base"]) or rows[t]["arms"][arm]["m_recheck"]["dac"] != 1.0]
    C["C-CERT-B"] = {"pass": not any(bad_cert.values()), "violations": {k: len(v) for k, v in bad_cert.items()}, "examples": {k: v[:3] for k, v in bad_cert.items()},
                     "what": "every applied plan, re-scored by a separate scorer call, has DAC 1 and passes its arm's certificate against the N2-R base"}
    P, Pseam = {}, {}
    zt = np.load(a.seam, allow_pickle=False) if a.seam else None
    order = [str(t) for t in zt["token"]] if zt is not None else list(bt)
    for arm in ARMS:
        arr = bp.copy()
        for t in applied[arm]:
            arr[ix[t]] = np.asarray(rows[t]["arms"][arm]["plan"], float)
        P[arm] = arr
    os.makedirs(a.out_dir, exist_ok=True)
    for arm in ARMS:
        fp = os.path.join(a.out_dir, f"n4_plans_{arm}.npz")
        np.savez(fp, tokens=np.array(bt), logs=np.array(bl), plans20=P[arm], arm=np.array(arm))
        rep["files"][os.path.basename(fp)] = sha256(fp)
        if zt is not None:
            p8 = np.stack([ND.to_navsim(P[arm][ix[t]]).astype(np.float32) for t in order])
            sp = os.path.join(a.out_dir, SEAM_LABEL[arm] + ".npz")
            np.savez(sp, token=zt["token"], fingerprint=zt["fingerprint"], poses=p8, sampling=zt["sampling"], arm=np.array(str(zt["arm"]) + "_" + SEAM_LABEL[arm]))
            rep["files"][os.path.basename(sp)] = sha256(sp)
    if zt is not None:
        u = check_unch(a.out_dir, a.n2r_plans, a.n2r_seam, repaired)
    else:
        u = {"pass": all(np.array_equal(P[arm][[ix[t] for t in bt if t not in set(repaired[arm])]], bp[[ix[t] for t in bt if t not in set(repaired[arm])]]) for arm in ARMS),
             "arms": {}}
    u["spurious_applied_on_dac1_tokens"] = {k: len(v) for k, v in spurious.items()}
    u["pass"] = bool(u["pass"]) and not any(spurious.values())
    u["repaired_count_equation"] = {arm: {"dac0": len(dac0), "repaired": len(repaired[arm]), "unrepaired": len(dac0) - len(repaired[arm])} for arm in ARMS}
    C["C-UNCH"] = u
    rep["dac0_tokens_builder"] = len(dac0)
    rep["repaired"] = repaired
    rep["unrepaired_dac0"] = {arm: sorted(set(dac0) - set(repaired[arm])) for arm in ARMS}
    rep["diagnostics"]["builder_repair_rate"] = {arm: (len(repaired[arm]) / len(dac0) if dac0 else None) for arm in ARMS}
    rep["diagnostics"]["shift_param_examples"] = {t: rows[t]["arms"]["N4D"]["param"] for t in repaired["N4D"][:5]}
    ts = [r["t_s"] for r in rows.values() if "t_s" in r]
    rep["timing"] = {"n": len(ts), "t_s_mean": float(np.mean(ts)) if ts else None, "cpu_h_sum": float(np.sum(ts) / 3600.0) if ts else None}
    allpass = all(c["pass"] for c in C.values())
    rep["verdict"] = "PASS" if allpass else "FAIL"
    json.dump(rep, open(os.path.join(a.out_dir, "n4_build_report.json"), "w", encoding="utf-8"), indent=1)
    print("ZZN4_BUILD", rep["verdict"], json.dumps({k: v["pass"] for k, v in C.items()}), "repaired", {k: len(v) for k, v in repaired.items()}, "dac0", len(dac0), flush=True)
    return 0 if allpass else 1


def cmd_unch(a) -> int:
    rep = json.load(open(os.path.join(a.out_dir, "n4_build_report.json"), encoding="utf-8"))
    u = check_unch(a.out_dir, a.n2r_plans, a.n2r_seam, rep["repaired"])
    print("ZZN4_UNCH", "PASS" if u["pass"] else "FAIL", json.dumps(u["arms"]), flush=True)
    if a.out:
        json.dump(u, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if u["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("reach", "derive", "merge", "unch"))
    ap.add_argument("--route", choices=("cache", "navtrain"), default="cache")
    ap.add_argument("--n2r-plans")
    ap.add_argument("--n2r-seam")
    ap.add_argument("--rolls", default="")
    ap.add_argument("--metric-cache-root", default="")
    ap.add_argument("--navtrain-tokens", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--mutate", choices=MUTATIONS, default="none")
    ap.add_argument("--cderiv-only", action="store_true", help="A1 check: derive ONLY the 200 C-DERIV tokens, base score + label re-derivation, NO repair arm (the chain never passes this)")
    ap.add_argument("--smoke-allow-mutation", action="store_true")
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--seam", default="")
    ap.add_argument("--harness-csv-n2r", default="")
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    return {"reach": cmd_reach, "derive": cmd_derive, "merge": cmd_merge, "unch": cmd_unch}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
