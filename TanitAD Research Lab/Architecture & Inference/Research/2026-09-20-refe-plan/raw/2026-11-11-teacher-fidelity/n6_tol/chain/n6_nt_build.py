#!/usr/bin/env python3
"""TF-N6 navtest build (PREREG_TF_N6_DRAFT.md sections 3, 4, 7): the plans of the tolerance arms from the banked TF-N5 N5-A plan / seam. CPU only.

The residual set is the tokens where the BANKED N5-A plan is DAC = 0 under the builder tracked scorer (all 12,146 are scored: C-DACMAP6, C-SET). On each one the build
collects the TEACHER CANDIDATES (every iterate REPAIR-D at A1 settings and L1 at the TF-N5 frozen parameters score, plus the unrepaired plan; both repairs are the
banked functions, run through a RECORDING proxy of the scorer, so no loop is copied) and the HUMAN candidate (the logged ego future from the nuPlan DB at 0.2 s, same
`to_navsim` seam function). The tolerance certificate (builder tracked scorer, vs the N5-A builder score): DAC = 1, NC >= NC_N5A, DDC >= DDC_N5A, builder PDMS > PDMS_N5A;
TTC and C are free. Teacher first (max builder PDMS, ties to the earliest iterate, REPAIR-D before L1), the human candidate only where no teacher candidate passes.

  reach   python n6_nt_build.py reach [--real --fixture <json> --out <json>]
          the two repair objects (REPAIR-D via n4_nt_build.check_arm, L1 via n5_nt_build.check_arm), and the three SELECTORS (read back + behavioural signature on synthetic
          candidate sets equal to the card's: every clause of the certificate and the ordering changes the signature). --real: the two real navtrain fixtures (capture, selection,
          human path, determinism).
  derive  python n6_nt_build.py derive --route cache|navtrain --n5a-plans <npz> --n2r-plans <npz> --rolls "<RJ5 rows glob>" --out <shard.jsonl> --shard i/N
              [--mutate none|strict|nc_drop|ddc_drop|pdms_drop|human_first|dac_inverted|tie_latest|teacher_off|label_ego_cache --smoke-allow-mutation]
  merge   python n6_nt_build.py merge ... -> n6_plans_{N6H,N6T,N6X}.npz, seams refe_tfn6_{n6h,n6t,n6x}.npz, n6_build_report.json (C-COVB, C-DACMAP6, C-DERIV, C-LABELEGO, C-SET,
          C-REPLAY, C-DET, C-SELECT, C-CERT-B6, C-ORDER, C-X0, C-SEAMFN, C-UNCH, C-REACH). Prints ZZN6_BUILD PASS|FAIL.
  unch    python n6_nt_build.py unch --out-dir <dir> ...   C-UNCH again on the files ON DISK."""
from __future__ import annotations

import argparse
import csv
import functools
import glob
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
os.environ["N4_RUN"] = os.environ.get("N4_RUN") or RUN_DIR
for _p in (os.path.join(RUN_DIR, "n5"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import n5_env  # noqa: E402,F401  (BLAS threads = 1 + the pinned tree on sys.path, before numpy)
import numpy as np  # noqa: E402

import n4_repair as RP  # noqa: E402
import n4_nt_build as B4  # noqa: E402  (the PINNED A1 builder: load_rolls, db_ego_fn, ctx_stream, make_repair("N4D"), check_arm)
import n5_nt_build as B5  # noqa: E402  (the PINNED TF-N5 builder: make_repair / check_arm for L1, info_summary, _npz, read_rows)

sha1_int, sha256, load_rolls, done_tokens, _npz, read_rows = B4.sha1_int, B4.sha256, B4.load_rolls, B4.done_tokens, B5._npz, B5.read_rows

# ---------------------------------------------------------------------------------------------------- the card (PREREG_TF_N6_DRAFT.md section 3)
D_ARM, L1_ARM = "N4D", "N5A"                       # the banked repair objects: REPAIR-D at A1 settings, L1 at the TF-N5 frozen parameters
CERT_BASE = {"dac": 1.0, "nc": True, "ddc": True, "ttc": False, "c": False, "pdms_gt": True, "tie": "earliest"}
CARD6 = {"N6H": dict(CERT_BASE, order=("teacher", "human")),
         "N6T": dict(CERT_BASE, order=("teacher",)),
         "N6X": dict(CERT_BASE, dac=0.0, order=("teacher", "human"))}          # N6-X: the certificate inverted to require DAC = 0 (a deliberate regression)
ARMS = tuple(CARD6)
SEAM_ARMS = ARMS
SEAM_LABEL = {"N6H": "refe_tfn6_n6h", "N6T": "refe_tfn6_n6t", "N6X": "refe_tfn6_n6x"}
OLD_SEAM = "refe_tfn5_n5a"
N_CDERIV, CDERIV_BAR_M, N_RESID_EXPECT = 200, 1e-9, 20
MUTATIONS = ("none", "strict", "nc_drop", "ddc_drop", "pdms_drop", "human_first", "dac_inverted", "tie_latest", "teacher_off", "label_ego_cache")
SEL_MUTATIONS = ("strict", "nc_drop", "ddc_drop", "pdms_drop", "human_first", "dac_inverted", "tie_latest")
MET = ("nc", "dac", "ddc", "ep", "ttc", "c", "pdms")
CSV_COL = B4.CSV_COL
EPS = 1e-12


# -------------------------------------------------------------------------------------------------------------------------- the certificate and the selector
def cert_clauses(m: dict, ref: dict, spec: dict) -> dict:
    """clause -> bool for one candidate score m against the N5-A builder score ref (TTC and C are clauses only when the spec asks for them)"""
    out = {"dac": m["dac"] == spec["dac"]}
    if spec["nc"]:
        out["nc"] = m["nc"] >= ref["nc"]
    if spec["ddc"]:
        out["ddc"] = m["ddc"] >= ref["ddc"] - EPS
    if spec["ttc"]:
        out["ttc"] = m["ttc"] >= ref["ttc"] - EPS
    if spec["c"]:
        out["c"] = m["c"] >= ref["c"] - EPS
    if spec["pdms_gt"]:
        out["pdms"] = m["pdms"] > ref["pdms"]
    return out


def cert_ok(m: dict, ref: dict, spec: dict) -> bool:
    return all(cert_clauses(m, ref, spec).values())


def select(cands: list, ref: dict, spec: dict):
    """cands in generation order (REPAIR-D iterates, then L1 iterates, then the human candidate). For each class in spec['order']: the candidates of that class passing the
    certificate; the one with the maximum builder PDMS wins, ties to the EARLIEST (tie 'earliest'). Returns the candidate dict or None."""
    for cls in spec["order"]:
        best = None
        for c in cands:
            if c["cls"] != cls or not cert_ok(c["m"], ref, spec):
                continue
            if best is None or c["m"]["pdms"] > best["m"]["pdms"] or (spec["tie"] == "latest" and c["m"]["pdms"] == best["m"]["pdms"]):
                best = c
        if best is not None:
            return best
    return None


def make_selector(arm: str, mutation: str = "none"):
    spec = dict(CARD6[arm])
    if mutation == "strict":
        spec.update(ttc=True, c=True)
    elif mutation == "nc_drop":
        spec["nc"] = False
    elif mutation == "ddc_drop":
        spec["ddc"] = False
    elif mutation == "pdms_drop":
        spec["pdms_gt"] = False
    elif mutation == "human_first":
        spec["order"] = ("human", "teacher")
    elif mutation == "dac_inverted":
        spec["dac"] = 0.0 if CARD6[arm]["dac"] == 1.0 else 1.0
    elif mutation == "tie_latest":
        spec["tie"] = "latest"
    return functools.partial(select, spec=spec)


def sel_readback(fn) -> dict:
    spec = dict(fn.keywords["spec"])
    spec["order"] = tuple(spec["order"])
    return spec


def _cand(src, k, cls, **m):
    base = {"nc": 1.0, "dac": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 0.5, "pdms": 0.5}
    base.update(m)
    return {"src": src, "k": k, "cls": cls, "m": base, "plan": None}


REF0 = {"nc": 1.0, "dac": 0.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 0.0, "pdms": 0.0}
REF_NC = dict(REF0, nc=0.5)
SCENARIOS = {                          # name -> (ref, candidates): every clause of the certificate and the ordering moves at least one entry
    "max_pdms": (REF0, [_cand("D", 1, "teacher", pdms=0.3), _cand("L1", 2, "teacher", pdms=0.6), _cand("L1", 3, "teacher", pdms=0.4)]),
    "tie_earliest": (REF0, [_cand("D", 2, "teacher", pdms=0.5), _cand("L1", 1, "teacher", pdms=0.5)]),
    "tie_d_before_l1": (REF0, [_cand("L1", 1, "teacher", pdms=0.5), _cand("D", 4, "teacher", pdms=0.5)]),
    "ttc_c_free": (REF0, [_cand("L1", 1, "teacher", ttc=0.0, c=0.0, pdms=0.4)]),
    "nc_worse": (REF0, [_cand("L1", 1, "teacher", nc=0.0, pdms=0.4)]),
    "nc_equal_below_one": (REF_NC, [_cand("L1", 1, "teacher", nc=0.5, pdms=0.4)]),
    "ddc_worse": (REF0, [_cand("L1", 1, "teacher", ddc=0.5, pdms=0.4)]),
    "pdms_not_improved": (dict(REF0, pdms=0.4), [_cand("L1", 1, "teacher", pdms=0.4)]),
    "dac0_candidate": (REF0, [_cand("L1", 1, "teacher", dac=0.0, pdms=0.2)]),
    "human_only": (REF0, [_cand("L1", 1, "teacher", dac=0.0, pdms=0.0), _cand("H", 0, "human", pdms=0.7)]),
    "teacher_and_human": (REF0, [_cand("L1", 1, "teacher", pdms=0.3), _cand("H", 0, "human", pdms=0.9)]),
    "human_fails_nc": (REF0, [_cand("H", 0, "human", nc=0.0, pdms=0.9)]),
    "nothing": (REF0, [_cand("L1", 1, "teacher", dac=0.0, pdms=0.0)]),
}


def selector_signature(fn) -> dict:
    out = {}
    for name, (ref, cands) in SCENARIOS.items():
        r = fn(cands, ref)
        out[name] = None if r is None else [r["src"], r["k"]]
    return out


def check_selector(arm: str, fn) -> dict:
    """read-back + behavioural signature compared with the card (arm); ok False with the reasons on any difference"""
    rb, bad = sel_readback(fn), []
    for k, v in CARD6[arm].items():
        got = rb.get(k)
        if got != (tuple(v) if k == "order" else v):
            bad.append(f"{k}: built {got!r} != card {v!r}")
    if getattr(fn, "func", None) is not select:
        bad.append("selector function is not n6_nt_build.select")
    sg, ref = selector_signature(fn), selector_signature(make_selector(arm))
    diff = [k for k in ref if sg.get(k) != ref[k]]
    if diff:
        bad.append("behavioural signature differs from the card selector's on " + ",".join(diff))
    return {"arm": arm, "ok": not bad, "bad": bad, "readback": {k: (list(v) if isinstance(v, tuple) else v) for k, v in rb.items()},
            "signature_sha1": hashlib.sha1(json.dumps(sg, sort_keys=True).encode()).hexdigest()}


# ------------------------------------------------------------------------------------------------------------------------------------- candidate capture
class Rec:
    """recording proxy of the scorer context: every cx.score(plans) call of a repair loop is logged (plan, metrics) in call order; everything else passes through.
    The two banked repair functions are therefore run UNCHANGED, and every iterate they score is a candidate."""

    def __init__(self, cx):
        self._cx, self.log = cx, []

    def score(self, plans):
        res, sim = self._cx.score(plans)
        for p, r in zip(plans, res):
            self.log.append((np.array(p, float, copy=True), dict(r)))
        return res, sim

    def __getattr__(self, name):
        return getattr(self._cx, name)


def run_recorded(fn, cx, t20, base, wants_info: bool):
    rc, info = Rec(cx), {}
    r = fn(rc, t20, base, info=info) if wants_info else fn(rc, t20, base)
    return r, rc.log, (B5.info_summary(info) if wants_info else {})


def capture(cx, t20, base, fd, fl):
    """-> (teacher candidates in generation order, deduplicated by plan bytes keeping the first; per-repair record)"""
    cands, repl, seen = [], {}, set()
    for src, fn, wi in (("D", fd, False), ("L1", fl, True)):
        r, log, summ = run_recorded(fn, cx, t20, base, wi)
        repl[src] = {"returns_plan": r is not None, "n_scored": len(log), "info": summ}
        for k, (p, m) in enumerate(log):
            key = p.tobytes()
            if key in seen:
                continue
            seen.add(key)
            cands.append({"src": src, "k": k, "cls": "teacher", "plan": p, "m": m})
    return cands, repl


def same_cands(a: list, b: list) -> bool:
    return len(a) == len(b) and all(x["src"] == y["src"] and x["k"] == y["k"] and np.array_equal(x["plan"], y["plan"]) and x["m"] == y["m"] for x, y in zip(a, b))


def human_plan20(cx, dbs, lg: str, tok: str):
    """the logged ego future at 0.2 s (20 poses, rear axle, ego frame) from the nuPlan DB: transp_score.human_rel (E2's) sliced exactly as the harness human arm does
    (`hum[1::2][:20]`). Returns (plan20 | None, why)."""
    import n4_geom as G
    import transp_score as TS
    if cx.sc is not None:
        sc, ego = cx.sc, cx.ego
    else:
        if lg not in dbs:
            return None, f"no nuPlan DB for log {lg}"
        scs = list(G.NS.build_scenarios_for_log(dbs[lg], [tok]))
        if len(scs) != 1 or scs[0]._initial_lidar_token != tok:
            return None, f"DB scenario for token {tok} not built ({len(scs)} returned)"
        sc, ego = scs[0], scs[0].initial_ego_state
    try:
        hum = np.asarray(TS.human_rel(sc, ego), float)
    except Exception as e:                                           # noqa: BLE001
        return None, f"human future unavailable: {type(e).__name__}: {str(e)[:120]}"
    p = hum[1::2][:20]
    if p.shape != (20, 3) or not np.isfinite(p).all():
        return None, f"human future has shape {hum.shape} / non-finite values"
    return np.array(p), "ok"


def human_candidate(cx, dbs, lg: str, tok: str):
    p, why = human_plan20(cx, dbs, lg, tok)
    if p is None:
        return None, why
    return {"src": "H", "k": 0, "cls": "human", "plan": p, "m": cx.score([p])[0][0], "ego_src": "db"}, why


def choose_all(cands: list, ref: dict, sels: dict, cx) -> dict:
    out = {}
    for arm, fn in sels.items():
        c = fn(cands, ref)
        out[arm] = None if c is None else {"src": c["src"], "k": c["k"], "cls": c["cls"], "plan": np.asarray(c["plan"]).tolist(), "m": c["m"],
                                          "m_recheck": cx.score([c["plan"]])[0][0]}
    return out


def make_objects(mutation: str = "none"):
    fd = B4.make_repair(D_ARM)
    fl = B5.make_repair(B5.CARD[L1_ARM])
    sels = {arm: make_selector(arm, mutation if mutation in SEL_MUTATIONS else "none") for arm in ARMS}
    return fd, fl, sels


def reach_real(fixture: str) -> dict:
    """the two real navtrain failing instances (nuPlan DB rebuild, builder scorer): iterate 0 == the base label, the capture is deterministic, the card H selector picks a
    DAC-1 teacher candidate, N6-X picks none, the human candidate builds (shape, finite) and scores on the same scorer"""
    import n4_geom as G
    fx = json.load(open(fixture, encoding="utf-8"))
    fd, fl, sels = make_objects()
    out = {"fixture_sha256": sha256(fixture), "instances": {}, "ok": True}
    for key in sorted(fx):
        inst = fx[key]
        cx = None
        for _lg, _tk, c in G.contexts({inst["log"]: [inst["token"]]}):
            cx = c
        t20 = np.asarray(inst["t20"], float)
        lab = cx.labels(t20)["R"]
        base = cx.score([lab])[0][0]
        cands, repl = capture(cx, t20, base, fd, fl)
        cands2, _ = capture(cx, t20, base, fd, fl)
        hc, why = human_candidate(cx, None, inst["log"], inst["token"])
        allc = cands + ([hc] if hc else [])
        h, t, x = sels["N6H"](allc, base), sels["N6T"](allc, base), sels["N6X"](allc, base)
        rec = {"base_dac": float(base["dac"]), "n_teacher_cands": len(cands), "iterate0_is_label": bool(np.array_equal(cands[0]["plan"], lab)),
               "deterministic": same_cands(cands, cands2), "human_built": hc is not None, "human_why": why,
               "human_dac": None if hc is None else float(hc["m"]["dac"]),
               "n6h": None if h is None else [h["src"], h["k"]], "n6t": None if t is None else [t["src"], t["k"]], "n6x": None if x is None else [x["src"], x["k"]],
               "n6h_dac": None if h is None else float(h["m"]["dac"])}
        rec["ok"] = bool(rec["base_dac"] == 0.0 and rec["iterate0_is_label"] and rec["deterministic"] and rec["human_built"] and h is not None and rec["n6h_dac"] == 1.0
                         and h["cls"] == "teacher" and t is not None and x is None and len(cands) >= 2)
        out["instances"][key] = rec
        out["ok"] = out["ok"] and rec["ok"]
    return out


def cmd_reach(a) -> int:
    fd, fl, sels = make_objects()
    reps = [dict(B4.check_arm(D_ARM, fd), what="REPAIR-D at A1 settings"), dict(B5.check_arm(L1_ARM, fl), what="L1 at the TF-N5 frozen parameters")]
    reps += [dict(check_selector(arm, sels[arm]), what="selector " + arm) for arm in ARMS]
    ok = all(r["ok"] for r in reps)
    real = reach_real(a.fixture) if a.real else None
    ok = ok and (real is None or real["ok"])
    print("ZZN6_REACH", "PASS" if ok else "FAIL", json.dumps({r.get("what"): r["ok"] for r in reps}), flush=True)
    for r in reps:
        for b in r["bad"]:
            print("ZZN6_REACH BAD", r.get("what"), b, flush=True)
    if real is not None:
        print("ZZN6_REACH REAL", json.dumps(real), flush=True)
    if a.out:
        json.dump({"ok": bool(ok), "arms": reps, "real": real}, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------------------------------------ derive
def cmd_derive(a) -> int:
    if a.mutate != "none" and not a.smoke_allow_mutation:
        raise SystemExit("REFUSING: --mutate is a navtrain-smoke instrument and needs --smoke-allow-mutation (the chain never passes either)")
    fd, fl, sels = make_objects(a.mutate)
    chk = [B4.check_arm(D_ARM, fd), B5.check_arm(L1_ARM, fl)] + [check_selector(arm, sels[arm]) for arm in ARMS]
    if not all(c["ok"] for c in chk) and not a.smoke_allow_mutation:
        raise SystemExit("REFUSING: the built repair / selector differs from the card: " + json.dumps([c["bad"] for c in chk if not c["ok"]]))
    z5 = np.load(a.n5a_plans, allow_pickle=False)
    toks, logs, p5 = [str(t) for t in z5["tokens"]], [str(x) for x in z5["logs"]], np.asarray(z5["plans20"], float)
    z2 = np.load(a.n2r_plans, allow_pickle=False)
    t2 = [str(t) for t in z2["tokens"]]
    ix = {t: i for i, t in enumerate(toks)}
    ix2 = {t: i for i, t in enumerate(t2)}
    if set(toks) - set(ix2):
        raise SystemExit("REFUSING: N5-A plan tokens are not all in the N2-R plan file")
    p2 = np.asarray(z2["plans20"], float)
    rolls = load_rolls(a.rolls)
    if a.navtrain_tokens:
        keep = set(json.load(open(a.navtrain_tokens, encoding="utf-8"))["tokens"])
        toks = [t for t in toks if t in keep]
    si, sn = (int(x) for x in a.shard.split("/"))
    cder = set(sorted(toks, key=sha1_int)[:N_CDERIV])
    done = done_tokens(a.out)
    todo = {t: logs[ix[t]] for t in toks if sha1_int(t) % sn == si and t not in done}
    need_roll = {t for t in todo if t in cder}
    if a.resid_tokens:                                  # smoke: the residual tokens are known (their rolls are needed up front)
        need_roll |= set(json.load(open(a.resid_tokens, encoding="utf-8"))["tokens"]) & set(todo)
    missing = [t for t in need_roll if t not in rolls]
    if missing:
        raise SystemExit(f"REFUSING: {len(missing)} tokens have no RJ5 target row, e.g. {missing[:3]}")
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, "rb") as fb:
            fb.seek(-1, 2)
            torn = fb.read(1) != b"\n"
        if torn:
            open(a.out, "a", encoding="utf-8").write("\n")
    import n4_geom as G
    dbs = G.NS.index_dbs(db_root=G.DB_ROOT) if a.route == "cache" else None
    n, t0 = 0, time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        for tok, lg, cx in B4.ctx_stream(a, todo):
            ts, rec = time.time(), {"log": lg, "token": tok}
            try:
                plan5 = p5[ix[tok]]
                ref = cx.score([plan5])[0][0]
                rec["ref"] = ref
                if tok in cder:
                    rec["cderiv_maxdev_m"] = float(np.abs(cx.labels(rolls[tok])["R"] - p2[ix2[tok]]).max())
                resid = ref["dac"] == 0.0
                rec["residual"] = bool(resid)
                if resid:
                    if tok not in rolls:
                        raise SystemExit(f"REFUSING: residual token {tok} has no RJ5 target row")
                    t20 = rolls[tok]
                    rec["n5a_equals_n2r_plan"] = bool(np.array_equal(plan5, p2[ix2[tok]]))
                    cands, repl, cands2 = [], {}, []
                    if a.mutate != "teacher_off":
                        cands, repl = capture(cx, t20, ref, fd, fl)
                        cands2, _ = capture(cx, t20, ref, fd, fl)                # C-DET: the whole capture run a SECOND time
                    rec["repl"] = repl or None
                    rec["det_equal"] = bool(same_cands(cands, cands2))
                    hc, why = human_candidate(cx, dbs, lg, tok)
                    rec["human"] = None if hc is None else {"plan": hc["plan"].tolist(), "m": hc["m"], "ego_src": hc["ego_src"]}
                    rec["human_why"] = why
                    allc = cands + ([hc] if hc else [])
                    rec["cands"] = [{"src": c["src"], "k": c["k"], "cls": c["cls"], "plan": np.asarray(c["plan"]).tolist(), "m": c["m"]} for c in allc]
                    rec["chosen"] = choose_all(allc, ref, sels, cx)
                    rec["label_ego"] = cx.label_ego_src if cx.label_ego_src else "none"
                else:
                    rec["label_ego"] = cx.label_ego_src if cx.label_ego_src else "none"
                rec["t_s"] = round(time.time() - ts, 3)
            except G.LabelEgoMissing:
                raise                                                      # REFUSE: never an error row, never a cache-ego fallback
            except Exception as e:                                    # noqa: BLE001
                rec = {"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            n += 1
    print("ZZN6_DERIVE_SHARD_DONE", a.shard, n, round(time.time() - t0, 1), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------------------------------------- merge
def check_unch(out_dir: str, n5a_plans: str, n5a_seam: str, applied: dict, n_resid: int, n_expect: int) -> dict:
    """C-UNCH on FILES: for every arm the plan20 AND the seam pose array of every token outside the applied set is bit-identical to the banked N5-A's, the changed set equals
    the applied set exactly; the residual count equals the card's; N6-X (applied = []) has plan and seam arrays bit-identical to N5-A's."""
    zb, sb = _npz(n5a_plans), _npz(n5a_seam)
    bt = [str(t) for t in zb["tokens"]]
    bi = {t: i for i, t in enumerate(bt)}
    sbi = {str(t): i for i, t in enumerate(sb["token"])}
    out = {"pass": True, "arms": {}, "n_residual": n_resid, "n_expected": n_expect}
    for arm in ARMS:
        fp, sp = os.path.join(out_dir, f"n6_plans_{arm}.npz"), os.path.join(out_dir, SEAM_LABEL[arm] + ".npz")
        if not (os.path.exists(fp) and os.path.exists(sp)):
            out["arms"][arm] = {"pass": False, "why": "missing file"}
            out["pass"] = False
            continue
        z, s = _npz(fp), _npz(sp)
        zt, stt = [str(t) for t in z["tokens"]], [str(t) for t in s["token"]]
        changed_p = {t for i, t in enumerate(zt) if t in bi and not np.array_equal(z["plans20"][i], zb["plans20"][bi[t]])}
        changed_s = {t for i, t in enumerate(stt) if t in sbi and not np.array_equal(s["poses"][i], sb["poses"][sbi[t]])}
        want = set(applied.get(arm, []))
        ok = changed_p == want and changed_s == want and set(zt) == set(bi) and set(stt) == set(sbi)
        rec = {"changed_plan20": len(changed_p), "changed_seam": len(changed_s), "applied": len(want), "tokens": len(zt), "seam_tokens": len(stt),
               "unflagged_changed": sorted((changed_p | changed_s) - want)[:5], "flagged_unchanged": sorted(want - (changed_p & changed_s))[:5]}
        for k in ("token", "fingerprint", "sampling"):
            if k in sb and not np.array_equal(s[k], sb[k]):
                ok = False
                rec["seam_field_differs"] = k
        if not want:
            rec["bit_identical_to_n5a"] = bool(changed_p == set() and changed_s == set())
        rec["pass"] = bool(ok)
        out["arms"][arm] = rec
        out["pass"] = out["pass"] and ok
    out["residual_count_ok"] = bool(n_resid == n_expect)
    out["pass"] = bool(out["pass"] and out["residual_count_ok"])
    return out


def load_harness_csv(p: str) -> dict:
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8", newline="")):
        if r["token"] != "average":
            out[r["token"]] = {m: float(r[CSV_COL[m]]) for m in MET}
    return out


def recompute_chosen(row: dict, arm: str):
    """the card selector (NOT the row's recorded one) on the stored candidates of one row -> (src, k, cls) or None"""
    cands = [dict(c, m=c["m"]) for c in row["cands"]]
    c = make_selector(arm)(cands, row["ref"])
    return None if c is None else (c["src"], c["k"], c["cls"])


def cmd_merge(a) -> int:
    import n4_geom as G
    ND = G.ND
    rows, n_err, n_dup = read_rows(a.rows)
    z5 = _npz(a.n5a_plans)
    bt, bl, bp = [str(t) for t in z5["tokens"]], [str(x) for x in z5["logs"]], np.asarray(z5["plans20"], float)
    sb = _npz(a.n5a_seam)
    order = [str(t) for t in sb["token"]]
    ix = {t: i for i, t in enumerate(bt)}
    rep: dict = {"n_tokens": len(bt), "n_rows": len(rows), "n_error_lines": n_err, "n_duplicate_lines": n_dup, "controls": {}, "diagnostics": {}, "files": {}}
    C = rep["controls"]
    smoke_teacher_off = any(r.get("repl") is None for r in rows.values() if r.get("residual"))
    C["C-COVB"] = {"pass": set(rows) == set(bt) and n_err == 0 and n_dup == 0, "missing": len(set(bt) - set(rows)), "extra": len(set(rows) - set(bt)),
                   "error_lines": n_err, "duplicate_lines": n_dup}
    ok_t = [t for t in bt if t in rows]
    hc = load_harness_csv(a.harness_csv_n5a)
    mism = [t for t in bt if t not in hc or (t in rows and rows[t]["ref"]["dac"] != hc[t]["dac"])]
    C["C-DACMAP6"] = {"pass": not mism and len(hc) >= len(bt), "tokens": len(bt), "mismatches": len(mism), "examples": mism[:5],
                      "what": "builder tracked DAC of the banked N5-A plan == harness DAC of N5-A, all tokens", "reference": os.path.basename(a.harness_csv_n5a)}
    other = {m: max((abs(rows[t]["ref"][m] - hc[t][m]) for t in ok_t if t in hc), default=None) for m in ("nc", "ddc", "ttc", "c", "pdms")}
    rep["diagnostics"]["builder_vs_harness_max_abs_all_tokens"] = other
    cd = sorted(bt, key=sha1_int)[:N_CDERIV]
    dev = [rows[t].get("cderiv_maxdev_m", float("nan")) for t in cd if t in rows]
    C["C-DERIV"] = {"pass": len(dev) == len(cd) and bool(np.nanmax(dev) <= CDERIV_BAR_M) and not any(np.isnan(dev)), "n": len(dev), "expected_n": len(cd),
                    "max_dev_m": float(np.nanmax(dev)) if dev else None, "bar_m": CDERIV_BAR_M}
    resid = sorted(t for t in ok_t if rows[t].get("residual"))
    lab_rows = [t for t in ok_t if "cderiv_maxdev_m" in rows[t] or rows[t].get("residual")]
    not_db = sorted(t for t in lab_rows if rows[t].get("label_ego") != "db")
    hum_not_db = sorted(t for t in resid if rows[t].get("human") and rows[t]["human"].get("ego_src") != "db")
    C["C-LABELEGO"] = {"pass": not not_db and not hum_not_db, "label_rows": len(lab_rows), "db": len(lab_rows) - len(not_db), "not_db": len(not_db), "examples": not_db[:5],
                       "human_rows_not_db": len(hum_not_db), "what": "every row that built labels (C-DERIV tokens + residual tokens x iterates) and every human future used the nuPlan-DB ego"}
    if a.n5a_report:
        banked = json.load(open(a.n5a_report, encoding="utf-8"))
        exp_res, exp_src = sorted(banked["unrepaired_l1"]["N5A"]), "banked TF-N5 A1 build report unrepaired_l1.N5A"
    else:
        exp_res, exp_src = sorted(json.load(open(a.expect_residual_json, encoding="utf-8"))["tokens"]), "--expect-residual-json (smoke)"
    harness_dac0 = sorted(t for t in bt if t in hc and hc[t]["dac"] == 0.0)
    flag_bad = sorted(t for t in ok_t if bool(rows[t].get("residual")) != (rows[t]["ref"]["dac"] == 0.0))
    C["C-SET"] = {"pass": bool(resid == exp_res and (not a.n5a_report or harness_dac0 == exp_res) and not flag_bad), "builder_residual": len(resid), "expected": len(exp_res),
                  "residual_flag_disagrees_with_builder_dac": flag_bad[:5],
                  "harness_n5a_zero_dac": len(harness_dac0), "expected_source": exp_src,
                  "missing_from_builder": sorted(set(exp_res) - set(resid))[:5], "extra_in_builder": sorted(set(resid) - set(exp_res))[:5],
                  "what": "builder tracked N5-A DAC-0 set == the banked TF-N5 A1 residual (and, on navtest, == the harness N5-A zero-DAC set)"}
    expect = json.load(open(a.replay_expect_json, encoding="utf-8")) if a.replay_expect_json else {}
    bad_replay = []
    for t in resid:
        r = rows[t]
        if r.get("repl") is None:
            bad_replay.append((t, "no repair record"))
            continue
        for src, key in (("D", "d_returns_plan"), ("L1", "l1_returns_plan")):
            want = expect.get(key)
            if want is not None and r["repl"][src]["returns_plan"] != (t in set(want)):
                bad_replay.append((t, f"{src} returns_plan {r['repl'][src]['returns_plan']} != banked"))
        cands = [c for c in r["cands"] if c["cls"] == "teacher"]
        if not cands or cands[0]["src"] != "D" or cands[0]["k"] != 0 or not np.array_equal(np.asarray(cands[0]["plan"], float), bp[ix[t]]) or not r.get("n5a_equals_n2r_plan"):
            bad_replay.append((t, "iterate 0 is not the banked N5-A (= N2-R) plan"))
    C["C-REPLAY"] = {"pass": not bad_replay and bool(expect), "residual": len(resid), "violations": bad_replay[:6],
                     "what": "the re-run repairs reproduce the banked results (REPAIR-D and L1 return the same applied / not applied as TF-N4 A1 / TF-N5 A1 on every residual token; "
                             "iterate 0 is the banked plan)"}
    det = {t: rows[t]["det_equal"] for t in resid}
    C["C-DET"] = {"pass": bool(det) and all(det.values()) and not smoke_teacher_off, "checked": len(det), "equal": int(sum(det.values())),
                  "what": "the whole candidate capture (REPAIR-D + L1, every iterate) run twice on every residual token gives bit-identical plans and scores"}
    # selection: the card selector on the STORED candidates must equal what the row recorded (catches a built selector that differs from the card)
    bad_sel = []
    for t in resid:
        r = rows[t]
        for arm in ARMS:
            got = r["chosen"].get(arm)
            got_id = None if got is None else (got["src"], got["k"], got["cls"])
            if recompute_chosen(r, arm) != got_id:
                bad_sel.append((t, arm, got_id, recompute_chosen(r, arm)))
    C["C-SELECT"] = {"pass": not bad_sel, "tokens": len(resid), "violations": len(bad_sel), "examples": bad_sel[:4],
                     "what": "the card selector run in the merge on the stored candidates picks exactly what the derive recorded, per arm and token"}
    applied = {arm: sorted(t for t in resid if rows[t]["chosen"].get(arm) is not None) for arm in ARMS}
    cls_of = {arm: {t: rows[t]["chosen"][arm]["cls"] for t in applied[arm]} for arm in ARMS}
    bad_cert = {arm: [t for t in applied[arm] if not cert_ok(rows[t]["chosen"][arm]["m_recheck"], rows[t]["ref"], CARD6[arm])
                      or rows[t]["chosen"][arm]["m_recheck"] != rows[t]["chosen"][arm]["m"]] for arm in ARMS}
    C["C-CERT-B6"] = {"pass": not any(bad_cert.values()), "violations": {k: len(v) for k, v in bad_cert.items()}, "examples": {k: v[:3] for k, v in bad_cert.items()},
                      "what": "every applied plan, re-scored by a separate scorer call, reproduces its score and passes the arm's CARD certificate against the N5-A builder score "
                              "(DAC; NC and DDC not worse; builder PDMS strictly better; TTC and C free)"}
    bad_order = []
    for t in resid:
        ch = rows[t]["chosen"]
        h, tt = ch.get("N6H"), ch.get("N6T")
        if tt is not None and (h is None or h["plan"] != tt["plan"] or h["cls"] != "teacher"):
            bad_order.append((t, "teacher passes but N6-H did not take the teacher plan"))
        if tt is None and h is not None and h["cls"] != "human":
            bad_order.append((t, "no teacher passes but N6-H is not the human candidate"))
    C["C-ORDER"] = {"pass": not bad_order, "violations": bad_order[:4], "what": "N6-H == N6-T wherever N6-T applies; N6-H differs only through the human candidate, only where no teacher candidate passes"}
    C["C-X0"] = {"pass": not applied["N6X"], "n6x_applied": len(applied["N6X"]),
                 "what": "the deliberate-regression arm (certificate inverted: DAC = 0 required) changes 0 tokens, so its seam is bit-identical to N5-A (a DAC-0 plan has builder PDMS 0)"}
    os.makedirs(a.out_dir, exist_ok=True)
    Pl = {arm: bp.copy() for arm in ARMS}
    for arm in ARMS:
        for t in applied[arm]:
            Pl[arm][ix[t]] = np.asarray(rows[t]["chosen"][arm]["plan"], float)
        fp = os.path.join(a.out_dir, f"n6_plans_{arm}.npz")
        np.savez(fp, tokens=np.array(bt), logs=np.array(bl), plans20=Pl[arm], arm=np.array(arm))
        rep["files"][os.path.basename(fp)] = sha256(fp)
    p8 = np.stack([ND.to_navsim(bp[ix[t]]).astype(np.float32) for t in order])
    bad_fn = [order[i] for i in range(len(order)) if not np.array_equal(p8[i], sb["poses"][i])]
    C["C-SEAMFN"] = {"pass": not bad_fn, "tokens": len(order), "mismatches": len(bad_fn), "examples": bad_fn[:3],
                     "what": "ND.to_navsim reproduces the banked N5-A seam poses bit for bit on every token, so the changed rows are made by the function that made the banked ones"}
    for arm in SEAM_ARMS:
        ap = set(applied[arm])
        p8a = np.stack([ND.to_navsim(Pl[arm][ix[t]]).astype(np.float32) if t in ap else sb["poses"][i] for i, t in enumerate(order)])
        old = str(sb["arm"])
        new_arm = (old[: -len(OLD_SEAM)] if old.endswith(OLD_SEAM) else old + "_") + SEAM_LABEL[arm]
        sp = os.path.join(a.out_dir, SEAM_LABEL[arm] + ".npz")
        np.savez(sp, token=sb["token"], fingerprint=sb["fingerprint"], poses=p8a, sampling=sb["sampling"], arm=np.array(new_arm))
        rep["files"][os.path.basename(sp)] = sha256(sp)
    u = check_unch(a.out_dir, a.n5a_plans, a.n5a_seam, applied, len(resid), a.expect_residual)
    C["C-UNCH"] = u
    if a.reach_json:
        rj = json.load(open(a.reach_json, encoding="utf-8"))
        C["C-REACH"] = {"pass": bool(rj.get("ok") and rj.get("real") and rj["real"].get("ok")), "arms_ok": {x.get("what"): x["ok"] for x in rj.get("arms", [])},
                        "real": {k: v.get("ok") for k, v in ((rj.get("real") or {}).get("instances") or {}).items()}}
    else:
        C["C-REACH"] = {"pass": False, "why": "no --reach-json (the chain's reach_real stage)"}
    rep["residual"] = resid
    rep["repaired"] = applied
    rep["source_class"] = {arm: cls_of[arm] for arm in ARMS}
    rep["counts"] = {arm: {"teacher_tol": sum(1 for v in cls_of[arm].values() if v == "teacher"), "human": sum(1 for v in cls_of[arm].values() if v == "human")} for arm in ARMS}
    rep["n_human_unavailable"] = sorted(t for t in resid if not rows[t].get("human"))
    best = {}
    for t in resid:                                           # per residual token: what the best DAC = 1 candidate failed, or that none reaches DAC = 1 (the blocking class)
        r, tab = rows[t], []
        for c in r["cands"]:
            cl = cert_clauses(c["m"], r["ref"], CARD6["N6H"])
            tab.append({"src": c["src"], "k": c["k"], "cls": c["cls"], "dac": c["m"]["dac"], "nc": c["m"]["nc"], "ddc": c["m"]["ddc"], "ttc": c["m"]["ttc"], "c": c["m"]["c"],
                        "pdms": c["m"]["pdms"], "failed": sorted(k for k, v in cl.items() if not v)})
        d1 = [x for x in tab if x["dac"] == 1.0]
        ch = rows[t]["chosen"].get("N6H")
        best[t] = {"resolved_by": None if ch is None else [ch["src"], ch["k"], ch["cls"]], "n_candidates": len(tab), "n_dac1_candidates": len(d1),
                   "blocking_class": None if ch is not None else ("no candidate (teacher or human) reaches DAC = 1" if not d1 else
                                                                  "DAC-1 candidates exist but fail: " + "; ".join(sorted({",".join(x["failed"]) for x in d1}))),
                   "candidates": tab, "ref_pdms": r["ref"]["pdms"], "repl": {k: v["returns_plan"] for k, v in (r.get("repl") or {}).items()}}
    rep["residual_table"] = best
    ts = [r["t_s"] for r in rows.values() if "t_s" in r]
    rep["timing"] = {"n": len(ts), "t_s_mean": float(np.mean(ts)) if ts else None, "cpu_h_sum": float(np.sum(ts) / 3600.0) if ts else None}
    allpass = all(c["pass"] for c in C.values())
    rep["verdict"] = "PASS" if allpass else "FAIL"
    json.dump(rep, open(os.path.join(a.out_dir, "n6_build_report.json"), "w", encoding="utf-8"), indent=1)
    print("ZZN6_BUILD", rep["verdict"], json.dumps({k: v["pass"] for k, v in C.items()}), "applied", {k: len(v) for k, v in applied.items()}, "counts", json.dumps(rep["counts"]),
          "residual", len(resid), flush=True)
    return 0 if allpass else 1


def cmd_unch(a) -> int:
    rep = json.load(open(os.path.join(a.out_dir, "n6_build_report.json"), encoding="utf-8"))
    u = check_unch(a.out_dir, a.n5a_plans, a.n5a_seam, rep["repaired"], len(rep["residual"]), a.expect_residual)
    print("ZZN6_UNCH", "PASS" if u["pass"] else "FAIL", json.dumps(u["arms"]), flush=True)
    if a.out:
        json.dump(u, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if u["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("reach", "derive", "merge", "unch"))
    ap.add_argument("--route", choices=("cache", "navtrain"), default="cache")
    ap.add_argument("--n5a-plans")
    ap.add_argument("--n5a-seam")
    ap.add_argument("--n5a-report", default="")
    ap.add_argument("--n2r-plans")
    ap.add_argument("--expect-residual-json", default="")
    ap.add_argument("--replay-expect-json", default="")
    ap.add_argument("--resid-tokens", default="")
    ap.add_argument("--rolls", default="")
    ap.add_argument("--metric-cache-root", default="")
    ap.add_argument("--navtrain-tokens", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--mutate", choices=MUTATIONS, default="none")
    ap.add_argument("--smoke-allow-mutation", action="store_true")
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--harness-csv-n5a", default="")
    ap.add_argument("--reach-json", default="")
    ap.add_argument("--expect-residual", type=int, default=N_RESID_EXPECT)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--fixture", default=os.path.join(RUN_DIR, "n5", "fixture_real_instances.json"))
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()
    return {"reach": cmd_reach, "derive": cmd_derive, "merge": cmd_merge, "unch": cmd_unch}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
