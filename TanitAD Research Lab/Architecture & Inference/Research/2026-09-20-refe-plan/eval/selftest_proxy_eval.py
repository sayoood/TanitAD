#!/usr/bin/env python3
"""Validity tests for eval/proxy_eval.py's decision machinery (the M6 verdict). Expectations are LITERALS, never an
expression over the code under test, and every check has a deliberate-regression arm that must go RED.

  T1 decide(): the PREREG_M6_YAWLOSS section-6 truth table, case by case (ADOPT; each REFUTED trigger; each NOT PROVEN
     reason; a failed gate overrides everything). MUTATION: REFUTED on "either plain arm fails" -> the split-seed case
     flips from NOT PROVEN to REFUTED
  T2 heading_metrics(): an analytic dump -- a winner heading exactly 2*pi off reads error 0.0 (the wrap), the
     beyond-pi share is the constructed 25 %, the position error is the constructed 0.3 m (3-4-5 offsets).
     MUTATION: no wrap -> error 2*pi
  T3 boot(): a constant statistic has the zero-width interval [c, c]; two calls on one token set give identical
     intervals (the same log resample). MUTATION: token-level resampling of a 2-log set gives a different interval
  T4 fam_adverse(): an error metric whose P interval sits entirely above W's is named; overlap is not; an accuracy
     below is named; a bias farther from 0 is named. MUTATION: the direction flipped -> the lower-is-better case missed
    python eval/selftest_proxy_eval.py  -> ZZSELFTEST_PROXYEVAL_OK / _FAIL; raw/2026-09-27-m6-proxy/selftest_proxy_eval.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proxy_eval as PE  # noqa: E402

OUT = HERE.parent / "raw" / "2026-09-27-m6-proxy" / "selftest_proxy_eval.json"
checks: dict = {}


def check(name, ok, detail=""):
    checks[name] = {"ok": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


G_OK = {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True}
BOTH = {"P0": True, "P1": True}
SPLIT = {"P0": True, "P1": False}
NONE_ = {"P0": False, "P1": False}
M_OK = {"W0": True, "W1": True}
POS = {"0": True, "1": True}


def t1():
    cases = [  # (label, args, literal verdict)
        ("all pass -> ADOPT", (G_OK, BOTH, M_OK, True, True, False, POS), "ADOPT"),
        ("a failed gate -> NOT PROVEN even if all else passes",
         ({**G_OK, "G3": False}, BOTH, M_OK, True, True, False, POS), "NOT PROVEN"),
        ("PRIMARY fails for BOTH -> REFUTED", (G_OK, NONE_, M_OK, True, True, False, POS), "REFUTED"),
        ("PDMS CI upper < -1 -> REFUTED", (G_OK, BOTH, M_OK, True, False, True, POS), "REFUTED"),
        ("position guard fails for BOTH -> REFUTED", (G_OK, BOTH, M_OK, True, True, False, {"0": False, "1": False}),
         "REFUTED"),
        ("PRIMARY split -> NOT PROVEN", (G_OK, SPLIT, M_OK, True, True, False, POS), "NOT PROVEN"),
        ("control un-traps -> NOT PROVEN", (G_OK, BOTH, {"W0": True, "W1": False}, True, True, False, POS),
         "NOT PROVEN"),
        ("PDMS straddles -1 (not pass, not refute) -> NOT PROVEN", (G_OK, BOTH, M_OK, True, False, False, POS),
         "NOT PROVEN"),
        ("position split -> NOT PROVEN", (G_OK, BOTH, M_OK, True, True, False, {"0": True, "1": False}), "NOT PROVEN"),
        ("PDMS seams missing -> NOT PROVEN", (G_OK, BOTH, M_OK, False, False, False, POS), "NOT PROVEN"),
        ("a failed gate beats a REFUTED trigger -> NOT PROVEN", ({**G_OK, "G1": False}, NONE_, M_OK, True, False, True,
                                                               POS), "NOT PROVEN"),
    ]
    for label, args, want in cases:
        got, why = PE.decide(*args)
        check(f"T1 {label}", got == want, f"got {got} {why}")

    def mutated(gates_ok, primary_pass, manip_ok, have, pdms_pass, pdms_refute, pos_ok):
        if not all(gates_ok.values()):
            return "NOT PROVEN"
        if (not all(primary_pass.values())) or pdms_refute or (not any(pos_ok.values())):   # "either" instead of both
            return "REFUTED"
        return "ADOPT" if all(manip_ok.values()) and have and pdms_pass and all(pos_ok.values()) else "NOT PROVEN"
    check("T1 MUTATION (REFUTED on either plain arm) goes RED on the split-seed case",
          mutated(G_OK, SPLIT, M_OK, True, True, False, POS) != "NOT PROVEN")


def t2(tmp: Path):
    n, M = 4, 64
    traj = np.zeros((n, M, 20, 3), np.float32)
    human = np.zeros((n, 8, 3), np.float32)
    t = 0.2 * np.arange(1, 21)
    for i in range(n):
        traj[i, :, :, 0] = 10.0 * t                         # every proposal: straight at 10 m/s
        traj[i, :, :, 1] = 5.0                              # ... offset 5 m laterally, except the winner below
        traj[i, 7, :, 0] = 10.0 * t + 0.18                  # winner (slot 7): 0.18 m ahead, 0.24 m left -> 0.30 m
        traj[i, 7, :, 1] = 0.24
        human[i, :, 0] = 10.0 * 0.5 * np.arange(1, 9)
    traj[:, 7, 19, 2] = 2 * np.pi                           # winner heading at step 19 exactly one branch away
    traj[0, 8:24, 19, 2] = 4.0                              # 16 more slots of token 0 beyond pi (slot 7 untouched)
    traj[1, :48, 19, 2] = -4.0                              # 48 slots of token 1 beyond pi (slot 7 among them)
    traj[1, 7, 19, 2] = 2 * np.pi                           # (the winner stays 2*pi -- also beyond pi)
    p = tmp / "dump.npz"
    np.savez(p, token=np.array(["a", "b", "c", "d"]), traj=traj, human=human, winner=np.full(n, 7))
    h = PE.heading_metrics(p)
    # beyond pi: token0 16 slots + slot 7 (2pi) -> 17 (slot 7 is not among 0..15); token1 48 slots incl. slot 7;
    # tokens 2, 3: slot 7 only -> 17 + 48 + 1 + 1 = 67 of 256
    check("T2 a 2*pi-shifted winner heading reads error 0.0 (wrapped)", abs(h["median_winner_h19_err_rad"]) < 1e-6,
          f"median {h['median_winner_h19_err_rad']:.6f}")
    check("T2 beyond-pi share of ALL raw step-19 headings is the constructed 67/256",
          abs(h["pct_raw_h19_beyond_pi"] - 100.0 * 67 / 256) < 1e-9, f"{h['pct_raw_h19_beyond_pi']:.4f} %")
    check("T2 winner position error (ADE) is the constructed 0.30 m, L1 0.42 m",
          abs(h["winner_pos_err_ade_m"] - 0.30) < 1e-5 and abs(h["winner_pos_err_l1_m"] - 0.42) < 1e-5,
          f"ADE {h['winner_pos_err_ade_m']:.6f} L1 {h['winner_pos_err_l1_m']:.6f}")
    raw = float(np.median(np.abs(traj[np.arange(n), 7, 19, 2] - human[:, 7, 2])))
    check("T2 MUTATION (no wrap) goes RED: error 2*pi", abs(raw - 2 * np.pi) < 1e-5, f"{raw:.5f}")


def t3():
    tl = {f"t{i}": f"log{i % 5}" for i in range(40)}
    c = {t: 3.25 for t in tl}
    m, lo, hi = PE.boot(c, tl, n=2000)
    check("T3 a constant statistic has the zero-width interval [c, c]", m == 3.25 and lo == 3.25 and hi == 3.25,
          f"{m} [{lo}, {hi}]")
    rng = np.random.default_rng(1)
    v = {t: float(rng.normal()) for t in tl}
    r1, r2 = PE.boot(v, tl, n=2000), PE.boot(v, tl, n=2000)
    check("T3 two calls on one token set give identical intervals (the same log resample)", r1 == r2, f"{r1}")
    tl2 = {"a": "L1", "b": "L1", "c": "L2", "d": "L2"}
    v2 = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0}
    _, lo2, hi2 = PE.boot(v2, tl2, n=4000)
    # 2 logs -> resampled means are 0, 0.5 or 1 only; token-level resampling (the mutation) gives 0, .25, .5, ...
    rngm = np.random.default_rng(20260927)
    vals = np.array(list(v2.values()))
    tok_means = np.array([vals[rngm.integers(0, 4, 4)].mean() for _ in range(4000)])
    check("T3 cluster resampling of a 2-log set yields only {0, 0.5, 1} means", lo2 in (0.0, 0.5) and hi2 in (0.5, 1.0),
          f"[{lo2}, {hi2}]")
    check("T3 MUTATION (token-level resampling) goes RED: other means appear",
          bool(np.any((tok_means != 0.0) & (tok_means != 0.5) & (tok_means != 1.0))))


def t4():
    def blk(comps_long, comps_lat, acc=0.9):
        return {"families": {"refcv6": {
            "longitudinal": {"ci": {"components": comps_long}},
            "lateral": {"ci": {"components": comps_lat}},
            "tactical": {"lateral_decision": {"accuracy": acc, "kappa": 0.5}}}}}
    W = blk({"speed_mae_mps": {"lo": 1.0, "hi": 1.2}, "target_speed_acc.within_1.0_mps": {"lo": 0.5, "hi": 0.6},
             "along_bias_m": {"lo": 0.1, "hi": 0.2}},
            {"heading_mae_deg": {"lo": 3.0, "hi": 4.0}, "cross_mae_m": {"lo": 0.5, "hi": 0.7}})
    P = blk({"speed_mae_mps": {"lo": 1.3, "hi": 1.5}, "target_speed_acc.within_1.0_mps": {"lo": 0.3, "hi": 0.4},
             "along_bias_m": {"lo": 0.3, "hi": 0.4}},
            {"heading_mae_deg": {"lo": 3.5, "hi": 4.5}, "cross_mae_m": {"lo": 0.2, "hi": 0.4}})
    r = PE.fam_adverse(W, P)
    named = {f"{x['family']}.{x['metric']}" for x in r["adverse"]}
    want = {"longitudinal.speed_mae_mps", "longitudinal.target_speed_acc.within_1.0_mps", "longitudinal.along_bias_m"}
    check("T4 adverse separations named exactly (error above, accuracy below, bias farther from 0)", named == want,
          f"{sorted(named)}")
    check("T4 overlap (heading) and an IMPROVEMENT (cross) are not named",
          "lateral.heading_mae_deg" not in named and "lateral.cross_mae_m" not in named and r["components_compared"] == 5)
    flipped = [m for m in ("speed_mae_mps",) if P["families"]["refcv6"]["longitudinal"]["ci"]["components"][m]["hi"]
               < W["families"]["refcv6"]["longitudinal"]["ci"]["components"][m]["lo"]]
    check("T4 MUTATION (direction flipped: higher-is-better for an error) goes RED: the case is missed", flipped == [])


def t5(tmp: Path):
    """AMENDMENT 1, A1.4: PASS / NOT PROVEN / FAIL BUT STILL SHRINKING / PLATEAU from literal cases"""
    e1 = tmp / "e1"
    (e1 / "dumps").mkdir(parents=True)
    n, M = 8, 64
    t = 0.2 * np.arange(1, 21)
    human = np.zeros((n, 8, 3), np.float32)
    human[:, :, 0] = 10.0 * 0.5 * np.arange(1, 9)

    def dump(name, h19_err, n_beyond):
        traj = np.zeros((n, M, 20, 3), np.float32)
        traj[:, :, :, 0] = 10.0 * t
        traj[:, :, 19, 2] = h19_err                      # every slot's step-19 heading error = h19_err (winner too)
        traj[:, :n_beyond // n, 19, 2] = 4.0 if n_beyond else h19_err     # n_beyond slots beyond pi in total
        traj[:, 0, 19, 2] = h19_err if n_beyond == 0 else traj[:, 0, 19, 2]
        np.savez(e1 / "dumps" / f"{name}.npz", token=np.array([f"t{i}" for i in range(n)]), traj=traj, human=human,
                 winner=np.full(n, M - 1))
    # epoch 1: plain arms 0.20 rad, 4 of 8x64 = 512 beyond pi (0.78 %); wrapped 0.80 rad
    for nm in ("W0", "W1"):
        dump(nm, 0.80, 0)
    for nm in ("P0", "P1"):
        dump(nm, 0.20, 32)
    base_e1 = PE.heading_metrics(e1 / "dumps" / "P0.npz")

    def res_for(p_err, p_pct, verdict_gates=True, heading_pass=False):
        m = {f"{k}e2": {"median_winner_h19_err_rad": 0.8, "pct_raw_h19_beyond_pi": 60.0, "winner_pos_err_ade_m": 0.6}
             for k in ("W0", "W1")}
        for k in ("P0", "P1"):
            m[f"{k}e2"] = {"median_winner_h19_err_rad": p_err, "pct_raw_h19_beyond_pi": p_pct,
                           "winner_pos_err_ade_m": 0.6}
        return {"metrics": m, "gates": {"G1": {"ok": verdict_gates}}, "primary": {"pass": {"P0e2": heading_pass,
                                                                                             "P1e2": heading_pass}}}
    e1_err, e1_pct = base_e1["median_winner_h19_err_rad"], base_e1["pct_raw_h19_beyond_pi"]
    cases = [
        ("ADOPT -> PASS", res_for(0.05, 0.0, True, True), "ADOPT", "PASS -- passes after 2 proxy epochs"),
        ("heading passes, PDMS fails -> NOT PROVEN", res_for(0.05, 0.0, True, True), "NOT PROVEN", "NOT PROVEN"),
        ("a gate fails -> NOT PROVEN", res_for(0.12, 0.3, False, False), "NOT PROVEN", "NOT PROVEN"),
        ("both metrics -50 % on both arms -> SHRINKING", res_for(e1_err * 0.5, e1_pct * 0.5), "REFUTED",
         "FAIL BUT STILL SHRINKING"),
        ("median -50 % but beyond-pi -15 % -> PLATEAU", res_for(e1_err * 0.5, e1_pct * 0.85), "REFUTED", "PLATEAU"),
        ("exactly -25 % on both -> SHRINKING (the bar is >= 25 %)", res_for(e1_err * 0.75, e1_pct * 0.75), "REFUTED",
         "FAIL BUT STILL SHRINKING"),
    ]
    for label, r, verdict, want in cases:
        got = PE.amendment1_outcome(r, e1, "e2", verdict)["outcome"]
        check(f"T5 A1.4 {label}", got == want, f"got {got!r}")
    r = PE.amendment1_outcome(res_for(e1_err * 0.5, e1_pct * 0.5), e1, "e2", "REFUTED")
    tr = r["per_plain_arm_trend"]["P0"]["median_winner_h19_err_rad"]
    check("T5 A1.4 slope + prediction to epoch 4 only (log-linear): e3 = e2 x r, e4 = e2 x r^2",
          abs(tr["predicted_epoch3"] - e1_err * 0.25) < 1e-9 and abs(tr["predicted_epoch4"] - e1_err * 0.125) < 1e-9
          and "predicted_epoch5" not in tr, f"{tr}")
    # mutation: a 10 % threshold would call the PLATEAU case (median -50 %, beyond-pi -15 %) SHRINKING
    rel = PE.amendment1_outcome(res_for(e1_err * 0.5, e1_pct * 0.85), e1, "e2", "REFUTED")["per_plain_arm_trend"]
    mut = all(v["reduction_pct"] >= 10.0 for arm in rel.values() for v in arm.values())
    check("T5 MUTATION (a 10 % shrink bar) goes RED: the PLATEAU case would read SHRINKING", mut)


def t6():
    """M6b: decide_m6b's section-6 truth table (3 seeds; seed-mean + 2-of-3) and the all-slot metric (c) (analytic)"""
    G = {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True}
    P3 = {"T0": True, "T1": True, "T2": True}
    M3 = {"Wt0": True, "Wt1": True, "Wt2": True}
    S3 = {"0": True, "1": True, "2": True}
    cases = [
        ("all pass -> ADOPT", (G, P3, M3, True, -0.5, 1.0, {"0": -0.9, "1": -1.2, "2": -0.3}, S3), "ADOPT"),
        ("only 1 of 3 per-seed lowers >= -1 -> NOT PROVEN", (G, P3, M3, True, -0.5, 1.0,
                                                               {"0": -1.5, "1": -1.2, "2": -0.3}, S3), "NOT PROVEN"),
        ("seed-mean lower < -1 -> NOT PROVEN", (G, P3, M3, True, -1.4, 0.2, {"0": -0.9, "1": -0.8, "2": -0.3}, S3),
         "NOT PROVEN"),
        ("seed-mean upper < -1 -> REFUTED", (G, P3, M3, True, -4.0, -1.5, {"0": -4, "1": -4, "2": -4}, S3), "REFUTED"),
        ("PRIMARY fails for ALL T -> REFUTED", (G, {"T0": False, "T1": False, "T2": False}, M3, True, -0.5, 1.0,
                                                {"0": 0, "1": 0, "2": 0}, S3), "REFUTED"),
        ("PRIMARY split -> NOT PROVEN", (G, {"T0": True, "T1": False, "T2": True}, M3, True, -0.5, 1.0,
                                         {"0": 0, "1": 0, "2": 0}, S3), "NOT PROVEN"),
        ("a gate fails -> NOT PROVEN", ({**G, "G5": False}, P3, M3, True, -0.5, 1.0, {"0": 0, "1": 0, "2": 0}, S3),
         "NOT PROVEN"),
    ]
    for label, args, want in cases:
        got, why = PE.decide_m6b(*args)
        check(f"T6 M6b {label}", got == want, f"got {got} {why}")
    # MUTATION: '1 of 3' instead of '2 of 3' would ADOPT the second case
    lo = {"0": -1.5, "1": -1.2, "2": -0.3}
    check("T6 MUTATION (1-of-3 per-seed rule) goes RED: the 1-of-3 case would ADOPT",
          sum(v >= -1.0 for v in lo.values()) >= 1)
    # (c): a straight 10 m/s path with every step-19 heading 0.07 rad off its own tangent -> median exactly 0.07
    t = 0.2 * np.arange(1, 21)
    T = np.zeros((3, 64, 20, 3))
    T[..., 0] = 10 * t
    T[..., 19, 2] = 0.07
    th, m = PE.own_tangent19(T)
    check("T6 all-slot metric (c): a 0.07 rad offset on a straight path reads exactly 0.07, all masked in",
          abs(float(np.median(np.abs(PE.wrap(T[..., 19, 2] - th))[m])) - 0.07) < 1e-12 and bool(m.all()))
    T2 = T.copy()
    T2[..., 18, 0] = T2[..., 19, 0] - 0.1                                  # a 0.1 m last step: below the 0.2 m mask
    check("T6 (c) masks a step shorter than 0.2 m", not bool(PE.own_tangent19(T2)[1].any()))


def t7(tmp: Path):
    """the copy-history render: a copy that RECORDS its own authorisation is rendered from it (2026-09-28: the epoch-2
    copy, key copy_e2_224100, was rendered as 'RE-COPY (e2:_2:24, the PI's 21:18 decision ... floor 10 GiB' although
    fast_copy.json records 'coordinator 22:34, floor 30, emergency 15'); a copy started in error is rendered too"""
    d = tmp / "t7"
    d.mkdir()
    fc = {"path": "C:/x", "source": "D:/x", "c_free_gib_before": 1.0, "c_free_gib_after": 2.0,
          "copy_213151": {"bytes_copied": 1e9, "seconds": 10.0, "c_free_gib_before": 3.0, "c_free_gib_after": 4.0},
          "copy_e2_224100": {"authorised": "coordinator 2026-09-27 22:34 (floor >= 30 GiB after, emergency < 15 GiB)",
                             "verified_at": "2026-09-27T22:41:00", "for": "AMENDMENT 1", "bytes_copied": 1e9,
                             "seconds": 10.0, "c_free_gib_before": 5.0, "c_free_gib_after": 6.0},
          "third_copy_started_in_error_223127": {"what": "stopped at 22:31:14"}}
    json.dump(fc, open(d / "fast_copy.json", "w"))
    L = PE.predata_deviations(d, {"gates": {"G1": {}}})
    e2 = [x for x in L if "AMENDMENT 1" in x and "RE-COPY" in x]
    first = [x for x in L if "RE-COPY (21:31:51" in x]
    err = [x for x in L if "A COPY STARTED IN ERROR (22:31:27)" in x]
    check("T7 an authorised copy is rendered from ITS record (coordinator 22:34, floor 30 / 15), never the 21:18 text",
          len(e2) == 1 and "coordinator 2026-09-27 22:34" in e2[0] and "21:18" not in e2[0] and "e2:_2" not in e2[0],
          e2[0][:160] if e2 else "missing")
    check("T7 a copy WITHOUT its own record keeps the 21:18 text and its key time", len(first) == 1 and "21:18" in first[0])
    check("T7 the copy started in error is rendered", len(err) == 1 and "stopped at 22:31:14" in err[0])
    check("T7 MUTATION: slicing the time out of the epoch-2 key (the old renderer) reads 'e2:_2:24' -- the defect",
          f"{'copy_e2_224100'[5:7]}:{'copy_e2_224100'[7:9]}:{'copy_e2_224100'[9:11]}" == "e2:_2:24")


def t8(tmp: Path):
    """the RESULT names the code the ARMS ran beside the code on disk at analysis (2026-09-28: a re-rendered RESULT's
    bare 'Code sha256' line showed the NEXT experiment's bytes -- true, and wrong for the reader). Rendered from the
    banked epoch-1 result_m6.json into a temp dir."""
    src = HERE.parent / "raw" / "2026-09-27-m6-proxy" / "result_m6.json"
    r = json.load(open(src, encoding="utf-8"))
    d = tmp / "t8"
    d.mkdir()
    r = dict(r, notes_root=str(src.parent))
    PE.write_result_md(d, r)
    txt = (d / r.get("result_md", "RESULT_M6_YAWLOSS.md")).read_text(encoding="utf-8")
    check("T8 the code-on-disk line is labelled NOT what the arms ran",
          "Code on disk at analysis time (NOT what the arms ran)" in txt and "\nCode sha256:" not in txt)
    check("T8 each epoch-1 arm's OWN proxy_train.py is named beside it (d4ecc28c)",
          "What each arm ran" in txt and txt.count("proxy_train.py d4ecc28c") >= 4)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="selftest_proxyeval_"))
    t1()
    t2(tmp)
    t3()
    t4()
    t5(tmp)
    t6()
    t7(tmp)
    t8(tmp)
    n_ok = sum(v["ok"] for v in checks.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = {"checks": checks, "n_checks": len(checks), "failed": [k for k, v in checks.items() if not v["ok"]],
           "tested_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
                             for f in ("proxy_eval.py", "selftest_proxy_eval.py")}}
    json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"  {n_ok}/{len(checks)} checks pass -> {OUT}")
    print("ZZSELFTEST_PROXYEVAL_OK" if n_ok == len(checks) else "ZZSELFTEST_PROXYEVAL_FAIL")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
