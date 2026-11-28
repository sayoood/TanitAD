#!/usr/bin/env python3
"""read_selectors.py -- the navtest read of PREREG_P6_SELECTORS sections 4 and 5. CPU only, dev box. Statistics, bars helpers and the overspeed definition are those of
read_scorer1p6.py (imported, not rewritten).

    python read_selectors.py --stage e2 --nominees OUT/nominees.json --e2-tokens .../e2/e2_tokens.json --e2-table .../e2_table_p6final.npz --out RES
    python read_selectors.py --stage e3 --nominees OUT/nominees.json --e2-tokens ... --e2-table ... [--full-table F:/TanitAD_p6read_work/tactical/table_p6final.npz]
                              [--changed-csv NAME=csv ...] --out RES [--e2-result RES/selectors_read_e2.json]
    python read_selectors.py --stage e2|e3 --smoke-standin --nominees SMOKE_OUT/nominees.json --out SCRATCH/res     # real 1,058 tokens as a STAND-IN: not a result

Every selector read of navtest is counted in the persistent ledger (K, default selectors/ledger_K.json); the P bar uses the Bonferroni-adjusted interval at the running K.
E3 reads at most one nominee per family: the family's best by FIT-B (never chosen from a navtest read). Truth on E3 = the full 64-proposal table when --full-table exists
(gated against P6's banked csv), otherwise P6's csv (unchanged picks) + the E2 table + --changed-csv harness rows; missing scores stop the read with INCOMPLETE and a seam
for the harness (same layout as read_scorer1p6.py emit-changed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("SEL_THREADS", "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P6BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(P6BASE, "eval"))
import selcore as C  # noqa: E402
import read_scorer1p6 as R  # noqa: E402

PREREG = os.path.join(P6BASE, "PREREG_P6_SELECTORS.md")
PREREG_SHA = "d4778ca40be22cbe706efe68e6d3878a9b6f070179dd2b604373b5eef5fc690b"
DEFAULT_FULL_TABLE = "F:/TanitAD_p6read_work/tactical/table_p6final.npz"
DEFAULT_LEDGER = os.path.join(HERE, "ledger_K.json")
SMOKE_DUMP, SMOKE_TABLE = "F:/TanitAD_p6read_work/run/dump/dump_p6.npz", "F:/TanitAD_p6read_work/run/score/table_p6.npz"
G1_HEADS = ("TTC", "NC", "DAC")
TIER = ("T1-class open loop: NAVSIM v1.1 PDMS of the selector's own pick among P6's 64 proposals (self-action open loop, not closed loop). "
        "E2 = selection tier on 2,000 tokens, E3 = the full navtest.")
EVIDENCE = "MEASURED (ours; every input artifact and its sha256 is listed under 'inputs')"
ESTIMATOR = ("paired log-cluster bootstrap, 10,000 draws, seed 20260927, over the set's own logs (selc6_analyze.Boot). It answers 'another draw of logs' only; "
             "it is blind to training variance and to inference variance. The P bar uses the Bonferroni-adjusted interval at the running K.")
AMBIGUITIES = [
    "B1 Overspeed (G2): plan v4/v0 > 1.15 with v4 the chord of the last 0.5 s step and v0 the measured ego speed of the export, on the CRUISE tokens of the read set (read_scorer1p6 A1, A2). Difference = selector minus P6's rule, CI-hi < +0.10.",
    "B2 G1 is read on E3 as the point difference of the TTC, NC, DAC sub-scores x100 against P6's rule, each >= -0.5 (read_scorer1p6.g1_ok, 1e-9 slack in the arm's disfavour).",
    "B3 P bar: Bonferroni-adjusted CI-lo strictly > 0, with K = the running count of navtest reads of selectors on this card INCLUDING the reads of this invocation. The adjusted interval is the (2.5/K, 100-2.5/K) percentile pair of the shared bootstrap draws.",
    "B4 G2 and goal status use the plain 95 % interval (they are not best-of-K claims); the adjusted CI-hi is printed beside G2 for information.",
    "B5 'E3 at most one per family' is the family's best nominee by FIT-B (family_rank 1). It is never chosen from an E2 read.",
    "B6 S3 'gains on FIT-B and loses on E2 or E3': a nominee is a FIT-B gain by construction; 'loses' = its paired point difference vs P6's rule is below 0 on E2 or on E3. The stricter reading (CI-hi < 0) is also reported. Primary label order: S1, S3, S2; every match is listed.",
    "B7 A repeated read of the same (nominee, parameter sha256, set) is the same read and is not counted twice.",
    "B8 The family clause is applied to the selector with read_scorer1p6's components (A8, A9): ADOPT if no longitudinal/lateral component is adverse, otherwise PDMS GAIN WITH COST with each adverse component named.",
    "B9 Tokens whose truth is missing for a selector's pick stop the read (INCOMPLETE) and emit a seam; nothing is imputed.",
]
STRATEGIC = R.STRATEGIC


# ------------------------------------------------------------------------------------------------ pure decision functions (tested at exact boundaries)
def p_clears(adj_lo):
    """P: Bonferroni-adjusted CI-lo strictly > 0"""
    return bool(adj_lo is not None and adj_lo > 0)


def g1_all(points):
    return {h: R.g1_ok(points[h]) for h in G1_HEADS}


def g2_ok(hi):
    """overspeed-share difference CI-hi < +0.10 (read_scorer1p6.g3_ok: strict, 1e-9 slack in the selector's disfavour)"""
    return R.g3_ok(hi)


def decide_s(rows):
    """rows: one dict per read nominee: clears_P, guards_ok, lost_e2, lost_e3 (bools). -> dict(matches, primary)"""
    m = []
    if any(r["clears_P"] and r["guards_ok"] for r in rows):
        m.append("S1")
    if any(r.get("lost_e2") or r.get("lost_e3") for r in rows):
        m.append("S3")
    if "S1" not in m and "S3" not in m:
        m.append("S2")
    names = {"S1": "GAIN", "S2": "NO SELECTOR GAIN", "S3": "CONTRADICTION"}
    nxt = {"S1": "carry the best candidate forward as a selector and refit it on the logits of each trained arm as a counted layered candidate",
           "S2": "selection on P6's existing outputs is not a lever: the trained arms of Batch 1 and Batch 2 are, in the order of the Shapley split",
           "S3": "the same fit on the logits of a trained arm on FIT logs it never trained on (out of sample)"}
    prim = "S1" if "S1" in m else ("S3" if "S3" in m else "S2")
    return dict(matches=m, primary=prim, name=names[prim], next_step={k: nxt[k] for k in m})


# ------------------------------------------------------------------------------------------------ inputs
class ReadInputs:
    def __init__(self, world, p6_logits, p6_props, p6_pick_banked, p6_rows, e2_tokens, e2_table, full_table=None, changed=None, provenance=None, tactical=True, smoke=False):
        self.world, self.p6_logits, self.p6_props, self.p6_pick_banked = world, p6_logits, p6_props, p6_pick_banked
        self.p6_rows, self.e2_tokens, self.e2_table, self.full_table = p6_rows, list(e2_tokens), e2_table, full_table
        self.changed, self.provenance, self.tactical, self.smoke = changed or {}, provenance or {}, tactical, smoke


def _table_dict(T):
    return dict(token=np.array([str(t) for t in T["token"]]), pdms=np.asarray(T["pdms"], np.float64), sub=np.asarray(T["sub"], np.float64), valid=np.asarray(T["valid"], bool),
                red=(T["red_pick_pdms"] if "red_pick_pdms" in T.files else None))


def inputs_from_files(a):
    P6 = np.load(a.p6_proposals, allow_pickle=False)
    toks = [str(t) for t in P6["token"]]
    world = R.world_from_export(a.export, toks)
    key = "proposals" if "proposals" in P6.files else "props8"
    e2 = json.load(open(a.e2_tokens, encoding="utf-8"))["tokens"]
    e2t = _table_dict(np.load(a.e2_table, allow_pickle=False))
    full = None
    if a.full_table and os.path.exists(a.full_table):
        full = _table_dict(np.load(a.full_table, allow_pickle=False))
    changed = {}
    for s in a.changed_csv or []:
        n, p = s.split("=", 1)
        changed[n] = R.read_harness_csv(p)
    prov = {"p6_proposals": (a.p6_proposals, R.sha256_file(a.p6_proposals)), "p6_csv": (a.p6_csv, R.sha256_file(a.p6_csv)), "e2_tokens": (a.e2_tokens, R.sha256_file(a.e2_tokens)),
            "e2_table": (a.e2_table, R.sha256_file(a.e2_table)), "export": (a.export, R.sha256_file(a.export)),
            "full_table": (a.full_table, R.sha256_file(a.full_table)) if full is not None else "ABSENT: truth for changed picks comes from the harness csv"}
    return ReadInputs(world, P6["logits"], P6[key], P6["pick"].astype(np.int64) if "pick" in P6.files else None, R.read_harness_csv(a.p6_csv), e2, e2t, full, changed, prov,
                      not a.no_tactical)


def smoke_inputs():
    """the 1,058 real navtest tokens of dump_p6 / table_p6 as E2 AND E3; the P6 'csv' is synthesised from the table (circular): SMOKE ONLY"""
    Z, T = np.load(SMOKE_DUMP, allow_pickle=False), np.load(SMOKE_TABLE, allow_pickle=False)
    world = R.world_from_dump(SMOKE_DUMP)
    tab = _table_dict(T)
    pk = Z["pick"].astype(np.int64)
    r = np.arange(len(pk))
    rows = {str(t): (True, float(tab["pdms"][i, pk[i]]), tab["sub"][i, pk[i]]) for i, t in enumerate(tab["token"])}
    prov = {"SMOKE": "stand-in: dump_p6.npz + table_p6.npz, read-only; the P6 csv is synthesised from the table (the E2 gate is circular)",
            "dump": (SMOKE_DUMP, R.sha256_file(SMOKE_DUMP)), "table": (SMOKE_TABLE, R.sha256_file(SMOKE_TABLE))}
    return ReadInputs(world, Z["logits"], Z["props8"], pk, rows, list(world.token), tab, tab, {}, prov, True, True)


# ------------------------------------------------------------------------------------------------ selectors
def load_nominees(path, smoke):
    """verifies nominees.sha256 and every params sha256; returns (doc, {id: params})"""
    d = json.load(open(path, encoding="utf-8"))
    sc = os.path.join(os.path.dirname(path), "nominees.sha256")
    if os.path.exists(sc) and open(sc, encoding="utf-8").read().split()[0] != C.sha256_file(path):
        raise SystemExit("REFUSING: nominees.json does not match nominees.sha256")
    if bool(d.get("smoke")) != bool(smoke):
        raise SystemExit(f"REFUSING: nominees.json smoke={d.get('smoke')} but this read is smoke={smoke}: stand-in fits and real fits never mix")
    import pickle
    params = {}
    for n in d["nominees"]:
        pf = os.path.join(os.path.dirname(path), n["params_file"])
        blob = open(pf, "rb").read()
        if C.sha256_bytes(blob) != n["params_sha256"]:
            raise SystemExit(f"REFUSING: params of {n['id']} do not match their sha256")
        params[n["id"]] = pickle.loads(blob)
    return d, params


def select_e3_nominees(nom_doc, only=None):
    """at most one per family: the family's best by FIT-B (family_rank 1)"""
    best = {}
    for n in nom_doc["nominees"]:
        if n["family"] not in best or n["family_rank"] < best[n["family"]]["family_rank"]:
            best[n["family"]] = n
    out = list(best.values())
    if only:
        out = [n for n in out if n["id"] in only]
    fams = [n["family"] for n in out]
    assert len(fams) == len(set(fams))
    return out


def picks_of(params, inp, chunk=2000):
    L, X = inp.p6_logits, inp.p6_props
    out = []
    for s in range(0, len(L), chunk):
        v0 = inp.world.v0[s:s + chunk] if params.get("v0_source") == "measured" else None
        ctx = C.Ctx(L[s:s + chunk], X[s:s + chunk], v0)
        out.append(C.pick_of(C.apply_params(params, ctx)))
    return np.concatenate(out)


# ------------------------------------------------------------------------------------------------ truth pool
def build_pool(inp, p6pick, use_full):
    W, N = inp.world, len(inp.world.token)
    pool = R.Pool(N)
    pool.add_rows(inp.p6_rows, W, p6pick, "p6 csv")

    def fill(table, toks, src):
        pos = {t: i for i, t in enumerate(table["token"])}
        wi = np.array([W.index[t] for t in toks])
        ti = np.array([pos[t] for t in toks])
        new_p, new_s = table["pdms"][ti], table["sub"][ti]
        kn = pool.known[wi]
        if (kn & ((pool.pdms[wi] != new_p) | (pool.sub[wi] != new_s).any(-1))).any():
            pool.conflicts.append(("table-vs-pool", src))
        pool.pdms[wi], pool.sub[wi], pool.known[wi] = new_p, new_s, True

    fill(inp.e2_table, inp.e2_tokens, "e2 table")
    if use_full and inp.full_table is not None:
        fill(inp.full_table, list(W.token), "full table")
    return pool


def add_changed(pool, inp, name, pk):
    return pool.add_rows(inp.changed.get(name, {}), inp.world, pk, "csv " + name)


# ------------------------------------------------------------------------------------------------ endpoints
def rec_adj(r, K):
    lo, hi = C.interval(r, K)
    return lo, hi


def endpoints(inp, pool, ix, pk_sel, pk_p6, K, family=False):
    """selector vs P6's rule on the tokens ix (indices into the world). -> dict"""
    W = inp.world
    n = len(ix)
    r_ = np.arange(len(W.token))
    B = R.BootSet(W.log[ix])
    pd_s, pd_p = pool.pdms[ix, pk_sel[ix]] * 100.0, pool.pdms[ix, pk_p6[ix]] * 100.0
    sb_s, sb_p = pool.sub[ix, pk_sel[ix]] * 100.0, pool.sub[ix, pk_p6[ix]] * 100.0
    out = {"n_tokens": int(n), "n_logs": int(B.n_logs), "n_pick_changed_vs_p6": int((pk_sel[ix] != pk_p6[ix]).sum()), "K": int(K)}
    pt, rr, nn = B.diff(pd_s, pd_p)
    lo, hi = R.rec(pt, rr)["lo"], R.rec(pt, rr)["hi"]
    alo, ahi = rec_adj(rr, K)
    out["P"] = dict(point=R.fin(pt), lo95=lo, hi95=hi, lo_adj=alo, hi_adj=ahi, K=int(K), clears=p_clears(alo), lost_point=bool(pt < 0), lost_strict=bool(hi < 0))
    lp, lr, ln = B.stat(pd_s)
    out["level"] = dict(R.rec(lp, lr, ln), p6_level=R.fin(B.stat(pd_p)[0]), goal_status=R.goal_status(R.fin(lp), R.rec(lp, lr)["hi"]), goal=R.GOAL)
    out["G1"] = {}
    for h in G1_HEADS:
        j = R.SUBN.index(h)
        p_, r2, n2 = B.diff(sb_s[:, j], sb_p[:, j])
        out["G1"][h] = dict(point=R.fin(p_), lo=R.rec(p_, r2)["lo"], hi=R.rec(p_, r2)["hi"], ok=R.g1_ok(p_))
    out["G1_ok"] = all(v["ok"] for v in out["G1"].values())
    cru = W.cruise()[ix]
    ratio = lambda pk: R.plan_v4(inp.p6_props[ix, pk[ix]]) / np.maximum(W.v0[ix], 1e-6)   # noqa: E731
    over_s, over_p = (ratio(pk_sel) > R.OVERSPEED).astype(float), (ratio(pk_p6) > R.OVERSPEED).astype(float)
    p_, r2, n2 = B.diff(over_s, over_p, cru)
    g2 = R.rec(p_, r2, n2)
    g2.update(ok=g2_ok(g2["hi"]), hi_adj=rec_adj(r2, K)[1], arm_share=R.fin(B.stat(over_s, cru)[0]), p6_share=R.fin(B.stat(over_p, cru)[0]), n_cruise=int(cru.sum()))
    out["G2_overspeed"] = g2
    out["guards_ok"] = bool(out["G1_ok"] and g2["ok"])
    out["_diff_draws"] = None
    if family:
        out["family_table"] = family_table(inp, pool, ix, pk_sel, pk_p6, B)
    return out


def _comps(inp, pool, ix, pk):
    W = inp.world
    pose = inp.p6_props[ix, pk[ix]]
    c = {}
    for j, s in enumerate(R.SUBN):
        c[s] = (pool.sub[ix, pk[ix], j] * 100.0, "higher", {"EP": "longitudinal", "TTC": "longitudinal", "DAC": "lateral"}.get(s, "pdms-other"))
    c.update(R.geometry_components(pose, W.human8[ix]))
    if inp.tactical:
        c.update(R.tactical_components(pose, W.human8[ix], None if W.cmd is None else W.cmd[ix]))
    return c


def family_table(inp, pool, ix, pk_sel, pk_p6, B):
    a, b = _comps(inp, pool, ix, pk_sel), _comps(inp, pool, ix, pk_p6)
    rows, adv = {}, []
    for k, v in a.items():
        if not isinstance(v, tuple):
            rows[k] = {"status": str(v)}
            continue
        vals, dr, fm = v
        pt, r, n = B.diff(vals, b[k][0])
        rr = R.rec(pt, r, n)
        rr.update(direction=dr, family=fm, arm_mean=R.fin(np.nanmean(vals)), P6_mean=R.fin(np.nanmean(b[k][0])))
        rr["adverse"] = R.adverse(dr, rr["lo"], rr["hi"]) if fm in ("longitudinal", "lateral") else False
        rows[k] = rr
        if rr["adverse"]:
            adv.append(k)
    rows["_distance_keeping"] = "UNAVAILABLE at this tier (no lead-agent block in NAVSIM open loop)"
    return dict(components=rows, adverse_components=adv, strategic=STRATEGIC)


# ------------------------------------------------------------------------------------------------ gates
def run_gates(inp, stage):
    W = inp.world
    g = {}
    p6pick = C.p6_pick(inp.p6_logits)
    assert np.array_equal(p6pick, R.pick_logits(inp.p6_logits)), "selcore.p6_pick differs from read_scorer1p6.pick_logits"
    if inp.p6_pick_banked is not None:
        n_eq = int((p6pick == inp.p6_pick_banked).sum())
        g["G0-PICK"] = dict(name="G0-PICK", n=len(p6pick), n_equal_to_banked_pick=n_eq, PASS=bool(n_eq == len(p6pick)))
    else:
        g["G0-PICK"] = dict(name="G0-PICK", PASS=True, note="no banked pick in the P6 file; the rule is not cross-checked")
    miss = [t for t in (inp.e2_tokens if stage == "e2" else W.token) if t not in inp.p6_rows]
    bad = sum(1 for t in (inp.e2_tokens if stage == "e2" else W.token) if t in inp.p6_rows and not inp.p6_rows[t][0])
    g["G0-P6CSV"] = dict(name="G0-P6CSV", n_tokens_absent=len(miss), n_invalid_rows=bad, PASS=bool(not miss and not bad))
    g["G0-E2"] = R.gate_e2(inp.e2_table, inp.e2_tokens, W, inp.p6_rows, inp.p6_logits, p6pick) if g["G0-P6CSV"]["PASS"] else dict(name="G0-E2", PASS=False, why="P6 csv gate failed")
    if stage == "e3" and inp.full_table is not None:
        gf = R.gate_e2(inp.full_table, list(W.token), W, inp.p6_rows, inp.p6_logits, p6pick) if g["G0-P6CSV"]["PASS"] else dict(PASS=False, why="P6 csv gate failed")
        gf["name"] = "G0-FULLTABLE"
        g["G0-FULLTABLE"] = gf
    return g, p6pick


# ------------------------------------------------------------------------------------------------ the read
def run_read(inp, nom_doc, params, stage, ledger, e2_result=None, only=None, out_dir=None):
    t0 = time.time()
    res = dict(card="PREREG_P6_SELECTORS", stage=stage.upper(), tier=TIER, evidence_class=EVIDENCE, estimator=ESTIMATOR, strategic=STRATEGIC, ambiguities=AMBIGUITIES, smoke=bool(inp.smoke),
               inputs=inp.provenance, nominees_sha256=C.sha256_bytes(C.canon(nom_doc)), prereg=dict(path=PREREG, sha256=C.sha256_file(PREREG) if os.path.exists(PREREG) else None))
    if inp.smoke:
        res["STAMP"] = "SMOKE STAND-IN: 1,058 navtest tokens, circular P6 csv. NOT A RESULT. Do not quote any number."
    if res["prereg"]["sha256"] not in (None, PREREG_SHA):
        res.update(status="READ STOPPED", why="PREREG_P6_SELECTORS.md sha256 differs from the frozen card")
        return res
    res["code_sha256"] = {n: C.sha256_file(os.path.join(HERE, n)) for n in ("selcore.py", "read_selectors.py", "fit_selectors.py") if os.path.exists(os.path.join(HERE, n))}
    res["code_sha256"]["read_scorer1p6.py"] = R.sha256_file(os.path.join(P6BASE, "eval", "read_scorer1p6.py"))
    gates, p6pick = run_gates(inp, stage)
    res["gates"] = gates
    if not all(g["PASS"] for g in gates.values()):
        res.update(status="READ STOPPED", why="a gate failed: " + ", ".join(k for k, g in gates.items() if not g["PASS"]) + "; no verdict is written")
        return res
    W = inp.world
    use_full = stage == "e3"
    pool = build_pool(inp, p6pick, use_full)
    if pool.conflicts:
        res.update(status="READ STOPPED", why=f"truth sources disagree: {pool.conflicts[:3]}")
        return res
    ix = np.array([W.index[t] for t in inp.e2_tokens]) if stage == "e2" else np.arange(len(W.token))
    noms = nom_doc["nominees"] if stage == "e2" else select_e3_nominees(nom_doc, only)
    if not noms:
        res.update(status="READ", nominees=[], verdict=decide_s([]), note="no family nominated a variant: nothing is read, S2 by the card (no selector gain), no navtest read counted")
        return res
    picks = {n["id"]: picks_of(params[n["id"]], inp) for n in noms}
    # control: P6's own rule as a candidate goes through the same pipeline and must give exactly zero on every endpoint
    ctrl_pick = picks_of(dict(kind="rule", cand=list(C.P6_RULE)), inp)
    cends = endpoints(inp, pool, ix, ctrl_pick, p6pick, 1)
    zeros = [cends["P"][k] for k in ("point", "lo95", "hi95", "lo_adj", "hi_adj")] + [cends["G1"][h][k] for h in G1_HEADS for k in ("point", "lo", "hi")] + \
            [cends["G2_overspeed"][k] for k in ("point", "lo", "hi")]
    res["control_p6_rule_as_candidate"] = dict(n_pick_changed=cends["n_pick_changed_vs_p6"], all_endpoint_differences_exactly_zero=bool(all(z == 0.0 for z in zeros)), n_values_checked=len(zeros))
    if cends["n_pick_changed_vs_p6"] != 0 or not res["control_p6_rule_as_candidate"]["all_endpoint_differences_exactly_zero"]:
        res.update(status="READ STOPPED", why="the P6-rule control does not give exactly zero difference on every endpoint: the pipeline is broken")
        return res
    # score availability
    need = {}
    for n in noms:
        add_changed(pool, inp, n["id"], picks[n["id"]])
        m = pool.missing(picks[n["id"]])
        m = m[np.isin(m, ix)]
        if len(m):
            need[n["id"]] = m
    if need:
        res.update(status="INCOMPLETE", why="harness scores missing for changed picks: " + json.dumps({k: int(len(v)) for k, v in need.items()}) + "; score the seams, pass --changed-csv NAME=csv",
                   seams=emit_seams(inp, need, picks, out_dir) if out_dir else None)
        return res
    # count the reads (after the gates and the truth are in place; a repeated read is not a new read)
    smoke_tag = "SMOKE:" if inp.smoke else ""
    for n in noms:
        ledger.add(f"{smoke_tag}{n['id']}:{n['params_sha256']}:{stage.upper()}", nominee=n["id"], family=n["family"], set=stage.upper(), smoke=bool(inp.smoke))
    K = ledger.K
    res["ledger"] = dict(path=ledger.path, K=K, note="K counts every navtest read of a selector on this card; best-of-K statements use the Bonferroni-adjusted interval at this K")
    rows = []
    res["nominees"] = []
    e2_by = {r["id"]: r for r in (e2_result or {}).get("nominees", [])}
    for n in noms:
        e = endpoints(inp, pool, ix, picks[n["id"]], p6pick, K, family=(stage == "e3"))
        e.pop("_diff_draws", None)
        e.update(id=n["id"], family=n["family"], params_sha256=n["params_sha256"], fit_b=dict(mean=n["fit_b_mean_pdms_x100"], delta=n["delta_vs_p6_x100"], ci=[n["ci_lo"], n["ci_hi"]]))
        e["bars"] = dict(P=e["P"]["clears"], G1=e["G1_ok"], G2=e["G2_overspeed"]["ok"], goal_status=e["level"]["goal_status"])
        if stage == "e3":
            adv = e["family_table"]["adverse_components"]
            e["verdict"] = (("ADOPT" if not adv else "PDMS GAIN WITH COST") if (e["P"]["clears"] and e["guards_ok"]) else "NO ADOPTION")
            e["verdict_adverse_components"] = adv
        res["nominees"].append(e)
        rows.append(dict(id=n["id"], clears_P=e["P"]["clears"], guards_ok=e["guards_ok"], lost_e2=bool(e2_by.get(n["id"], {}).get("P", {}).get("lost_point", False)) or (stage == "e2" and e["P"]["lost_point"]),
                         lost_e3=bool(stage == "e3" and e["P"]["lost_point"])))
    if stage == "e3":
        res["outcome"] = decide_s(rows)
        res["outcome_inputs"] = rows
        res["e2_result_used"] = bool(e2_result)
    else:
        res["outcome"] = None
        res["e2_summary"] = [dict(id=r["id"], lost_e2=r["lost_e2"]) for r in rows]
        res["note"] = "E2 is a first read; the outcome S1 to S3 is decided on E3 (bars are defined there)."
    res["status"] = "READ"
    res["secs"] = round(time.time() - t0, 1)
    return res


def emit_seams(inp, need, picks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    W, out = inp.world, {}
    for nm, idx in need.items():
        poses = inp.p6_props[idx, picks[nm][idx]].astype(np.float32)
        fp = W.fingerprint[idx] if W.fingerprint is not None else np.array([""] * len(idx))
        toks = [str(W.token[i]) for i in idx]
        sp, tj = os.path.join(out_dir, f"seam_{nm}.npz"), os.path.join(out_dir, f"tokens_{nm}.json")
        np.savez(sp, token=np.array(toks), fingerprint=np.array([str(x) for x in fp]), poses=poses, sampling=np.array([8, 0.5]), arm=np.array(f"p6selectors_{nm}"))
        json.dump({"tokens": sorted(toks), "token_log": {t: str(W.log[W.index[t]]) for t in toks}}, open(tj, "w", encoding="utf-8"))
        out[nm] = dict(n_needed=int(len(idx)), seam=sp, tokens=tj, harness=f"cd <eval/harness>; python score_e.py --label p6sel_{nm} --seam {sp} --tokens {tj} --out <score dir>  (then pass the csv as --changed-csv {nm}=<csv>)")
    json.dump(out, open(os.path.join(out_dir, "needs_scoring.json"), "w", encoding="utf-8"), indent=1)
    return out


# ------------------------------------------------------------------------------------------------ output
def f(x, d=3):
    return "n/a" if x is None else f"{x:+.{d}f}"


def to_markdown(res):
    L = [f"# Selector read {res['stage']} -- PREREG_P6_SELECTORS" + ("  [SMOKE STAND-IN: NOT A RESULT]" if res.get("smoke") else ""), ""]
    if res.get("STAMP"):
        L += [f"**{res['STAMP']}**", ""]
    L += [f"Status: **{res.get('status')}** {res.get('why', '')}", "", f"Tier: {res['tier']}", f"Evidence class: {res['evidence_class']}", f"Estimator: {res['estimator']}", "",
          f"STRATEGIC: {res['strategic']}", ""]
    if "ledger" in res:
        L += [f"**K = {res['ledger']['K']}** navtest selector reads on this card so far (ledger {res['ledger']['path']}).", ""]
    if res.get("status") == "READ" and res.get("nominees"):
        L += ["| nominee | picks changed | PDMS diff x100 [95 %] | Bonferroni-adj [lo, hi] at K | P | G1 TTC / NC / DAC | G2 overspeed diff [95 %] | goal status | verdict |", "|---|---|---|---|---|---|---|---|---|"]
        for e in res["nominees"]:
            P, g2 = e["P"], e["G2_overspeed"]
            L.append(f"| {e['id']} | {e['n_pick_changed_vs_p6']} | {f(P['point'])} [{f(P['lo95'])}, {f(P['hi95'])}] | [{f(P['lo_adj'])}, {f(P['hi_adj'])}] K={P['K']} | {'CLEARS' if P['clears'] else 'no'} | "
                     f"{f(e['G1']['TTC']['point'], 2)} / {f(e['G1']['NC']['point'], 2)} / {f(e['G1']['DAC']['point'], 2)} ({'ok' if e['G1_ok'] else 'FAIL'}) | "
                     f"{f(g2['point'], 4)} [{f(g2['lo'], 4)}, {f(g2['hi'], 4)}] ({'ok' if g2['ok'] else 'FAIL'}) | {e['level']['goal_status']} | {e.get('verdict', 'E2 read')} |")
        for e in res["nominees"]:
            if "family_table" in e:
                adv = e["verdict_adverse_components"]
                L += ["", f"Family clause, {e['id']}: **{e['verdict']}**; adverse components (paired CI entirely on the worse side): {', '.join(adv) if adv else 'none'}. "
                      "Longitudinal, lateral and tactical components with CIs are in the JSON (family_table); STRATEGIC PAUSED (not a pass)."]
        if res.get("outcome"):
            o = res["outcome"]
            L += ["", f"## Outcome: {o['primary']} {o['name']}  (matches {o['matches']})", ""] + [f"- {k}: {v}" for k, v in o["next_step"].items()]
    if res.get("note"):
        L += ["", res["note"]]
    if res.get("control_p6_rule_as_candidate"):
        L += ["", f"Control, P6's rule as a candidate: {res['control_p6_rule_as_candidate']}"]
    L += ["", "## Ambiguities applied"] + [f"- {a}" for a in res["ambiguities"]]
    return "\n".join(L) + "\n"


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["e2", "e3"], required=True)
    ap.add_argument("--nominees", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p6-proposals", default=R.DEFAULT_P6_PROPOSALS)
    ap.add_argument("--p6-csv", default=R.DEFAULT_P6_CSV)
    ap.add_argument("--export", default=R.DEFAULT_EXPORT)
    ap.add_argument("--e2-tokens", default=os.path.join(P6BASE, "e2", "e2_tokens.json"))
    ap.add_argument("--e2-table", default=os.path.join(P6BASE, "e2", "e2_table_p6final.npz"))
    ap.add_argument("--full-table", default=DEFAULT_FULL_TABLE)
    ap.add_argument("--changed-csv", action="append", default=[], help="NAME=csv of harness scores for the changed picks of nominee NAME")
    ap.add_argument("--e2-result", default=None, help="selectors_read_e2.json of the E2 stage (folds S3's E2 loss into the E3 outcome)")
    ap.add_argument("--only", default=None, help="comma list of nominee ids to read on E3 (still at most one per family)")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--no-tactical", action="store_true")
    ap.add_argument("--smoke-standin", action="store_true")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.smoke_standin:
        if not os.path.abspath(a.out).replace("\\", "/").startswith("F:/tanitad_scratch/") and os.environ.get("SEL_SMOKE_ANYWHERE") != "1":
            raise SystemExit("REFUSING: --smoke-standin writes only under F:/tanitad_scratch/")
        a.ledger = os.path.join(a.out, "smoke_ledger_K.json")
        inp = smoke_inputs()
    else:
        for p in (a.e2_tokens, a.e2_table):
            if not os.path.exists(p):
                raise SystemExit(f"REFUSING: {p} does not exist (the real E2 table is required)")
        inp = inputs_from_files(a)
    nom_doc, params = load_nominees(a.nominees, a.smoke_standin)
    e2_res = json.load(open(a.e2_result, encoding="utf-8")) if a.e2_result and os.path.exists(a.e2_result) else None
    ledger = C.Ledger(a.ledger)
    res = run_read(inp, nom_doc, params, a.stage, ledger, e2_res, a.only.split(",") if a.only else None, a.out)
    os.makedirs(a.out, exist_ok=True)
    stem = f"selectors_read_{a.stage}"
    jp, mp = os.path.join(a.out, stem + ".json"), os.path.join(a.out, stem + ".md")
    C.atomic_write(jp, json.dumps(res, indent=1, default=R.jdefault).encode("utf-8"))
    C.atomic_write(mp, to_markdown(res).encode("utf-8"))
    print(json.dumps({"status": res.get("status"), "why": res.get("why"), "K": res.get("ledger", {}).get("K"), "outcome": (res.get("outcome") or {}).get("primary"), "json": jp, "md": mp,
                      "smoke": bool(inp.smoke)}, indent=1, default=R.jdefault))
    return 0 if res.get("status") == "READ" else 2


if __name__ == "__main__":
    sys.exit(main())
