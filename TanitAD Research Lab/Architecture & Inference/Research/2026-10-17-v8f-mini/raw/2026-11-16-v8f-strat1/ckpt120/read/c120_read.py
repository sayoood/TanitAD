"""c120_read.py -- the PREREG_V8F_CKPT120 read (frozen 2026-11-16; MM rulings 1-12 in rulings.md).  CPU, zero training.  -> out/RESULT_CKPT120.json + .md
Reads: the new arms' T0 passes (real / donor / zero, ``eval_CK120-<arm>_<v>.npz`` + ``_far.npz``), their T1-kin dumps (24 driving cells), the K9b tick dumps, the BANKED STRAT1 arms (S1_s0 / s1 / s10 T0 + dumps) as the base, the pinned label tables.
Tiers (every block carries its own): STRATEGIC route cells = T0 at the window's first tick; LONGITUDINAL / LATERAL / TACTICAL = T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav and nav_s, logged scene); detected info = T0.
Estimator: ``taniteval.ci`` paired episode-cluster bootstrap, B 2,000, seed 0 ("another draw of EPISODES"); another TRAINING RUN = the floors; another INFERENCE RUN does not apply (deterministic planner).
Every STRATEGIC number is worded "with the route supplied along the driven path" (card section 3, ruling 11); an H2 route-following number "follows the supplied route text" (ruling 10).  Numbers are PROVISIONAL until R1 and R2 have read (ruling 9).
Usage: python c120_read.py [--out-dir D] [--n-boot N] [--smoke] [--no-bank-check] [--arms A ...]
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c120_lib as C  # noqa: E402

SL = C.SL
sys.path.insert(0, SL.HERE)
sys.path.insert(0, os.environ.get("STRAT1_OBD_DIR", SL.HERE + "/theory/obd/read"))
sys.path.insert(0, SL.HERE + "/theory/otband/read")
import obd_read as OR  # noqa: E402
import strat1_ctx_read as CX  # noqa: E402
import strat1_read as SR  # noqa: E402
import strat1_strategic as SS  # noqa: E402

VARIANTS = ("real", "donor", "zero")
RUNGS = {"H1": {"arms": {0: "H1_s0", 1: "H1_s1"}, "base": {0: "S1_s0", 1: "S1_s1"}}, "H2": {"arms": {0: "H2_s0", 1: "H2_s1"}, "base": {0: "H1_s0", 1: "H1_s1"}}}
BANNED = ("void", "superseded", "retracted", "formerly", "correction", "withdrawn", "no longer", "closed loop")
WORDING = "with the route supplied along the driven path"
CARD_COUNTS = {"windows": 341, "episodes": 27, "left": 79, "right": 262}
K_PERFECT_CARD = {"delta": 0.232, "lo": 0.090, "hi": 0.406}
K_NAVINF_CARD = {"raw": -0.126, "balanced": 0.148}
NEXT = {
    "S120-FOLLOWS": "the T-reads-far card (S's 120 m point as a fourth checkpoint token of T; PI go) and the derived-goals card",
    "S120-FOLLOWS-STRONG": "as S120-FOLLOWS (raw separated at both seeds too)",
    "S120-PARTIAL": "routed by stratum: shortfall where the side does not reach S -> H1: H2 (in the chain); H2: the far stratum's read decides the next lever (strategic token distance resolution, a card amendment). Shortfall where the side reaches S -> the zero-training probe on the arm's frozen S hidden (fit on train once, scored on eval once; a negative probe is not a negative about learnability); positive -> a 6,000-step arm; negative -> S reader depth 2",
    "S120-IGNORES": "zero-training check: far point under zero vs real nav on RC-A120 error (does the head read nav?); then the probe above",
    "HEAD-DEAD": "build defect: fix and rerun the arm; no read",
    "FAR-SETS": "the head adds route information beyond S's 80 m point (reported with the route-following outcome)",
    "FAR-NO-GAIN": "reported as written, read together with the route-following outcome",
    "ROUTE-COST": "H1s: the far point from its own trunk (Linear(1024, 512) GELU Linear(512, 3), 526,339 params), so 30/50/80 keep their head layer",
    "KEEPS": "the arm is the strategic-recipe candidate; adoption also needs STRAT1's SEEDS verdict on S1 - C0",
    "ADDS": "as KEEPS; the gain is named",
    "COSTS-FAR": "the same arm with --s-adapter-detach on (built, default off), same seeds and reads; no adoption",
    "SEEDS-DISAGREE": "the arm at seeds 10 (init 0) and 11 (init 1), rep's form; a cell is a cost at >= 3 of 4 seeds with one per init",
}


# ------------------------------------------------------------------------------------------------------------------------------ imports from the tree of record
def load_tree_modules():
    """taniteval + ckpt120_cells + tanitad from the CKPT120 tree (asserted); a stale import path would score with other code."""
    import importlib.util
    for p in (C.TREE + "/taniteval", C.TREE + "/stack"):
        p = os.path.normpath(p)
        if p not in sys.path:
            sys.path.insert(0, p)
    import taniteval
    root = os.path.normcase(os.path.abspath(C.TREE))
    if not os.path.normcase(os.path.abspath(taniteval.__file__)).startswith(root):
        raise RuntimeError(f"taniteval imported from {taniteval.__file__}, not the CKPT120 tree {C.TREE}")
    spec = importlib.util.spec_from_file_location("ckpt120_cells", C.TREE + "/stack/scripts/ckpt120_cells.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------------------------------------ PURE labels (the unit tests plant every row)
def _ok(row, floor, need_pos=True):
    """a contrast row clears its bar: status OK, separated, point > 0, and (floor given) |point| > floor"""
    if not row or row.get("status") != "OK" or not row.get("separated"):
        return False
    if need_pos and not row["delta"] > 0:
        return False
    return floor is None or abs(row["delta"]) > floor


def route_label(per_seed, floors):
    """per_seed {0: blk, 1: blk} of ``route_following``; floors {"rd": F, "bal": F} or None (FLOOR-PENDING).  Card 8:
    IGNORES = real - donor not separated above 0 (or not above F120) at >= 1 seed;  FOLLOWS = real - donor, balanced - 0.5 separated (each above F120) and the raw point > 0 and > F120 at both seeds (STRONG when raw is separated at both too);  else PARTIAL.
    MM ruling 2026-11-17 (before any CKPT120 number): the card's 'every effect also > F120' binds the raw point too; floors['raw'] = |H1_rep - H1_s0| on the raw statistic."""
    if set(per_seed) != {0, 1}:
        return "NOT-YET", "the card needs both seeds"
    pend = " (FLOOR-PENDING)" if floors is None or any(floors.get(k) is None for k in ("rd", "bal", "raw")) else ""
    f = floors or {}
    for s, b in per_seed.items():
        if b["real_minus_donor"].get("status") != "OK" or b["raw_real_minus_best_constant"].get("status") != "OK":
            return "NOT-READ", f"seed {s}: a 120 m row is n/a (n windows {b['real_minus_donor'].get('n_windows')}, episodes {b['real_minus_donor'].get('n_episodes')})"
    if not all(_ok(b["real_minus_donor"], f.get("rd")) for b in per_seed.values()):
        return "S120-IGNORES" + pend, "real - donor is not separated above 0 (or not above F120) at >= 1 seed"
    bal_ok = all(_ok(b["balanced_minus_half"], f.get("bal")) for b in per_seed.values())
    raw_pos = all(b["raw_real_minus_best_constant"]["delta"] > 0 and (f.get("raw") is None or b["raw_real_minus_best_constant"]["delta"] > f["raw"]) for b in per_seed.values())
    if bal_ok and raw_pos:
        strong = all(_ok(b["raw_real_minus_best_constant"], None) for b in per_seed.values())
        return ("S120-FOLLOWS-STRONG" if strong else "S120-FOLLOWS") + pend, "real - donor and balanced - 0.5 separated above 0 and F120, raw point > 0 and > F120, at both seeds" + (" (raw separated too)" if strong else "")
    why = [n for n, ok in (("balanced - 0.5", bal_ok), ("raw point > 0 and > F120", raw_pos)) if not ok]
    return "S120-PARTIAL" + pend, "real - donor separated at both seeds; fails: " + ", ".join(why)


def far_label(rows, floor):
    """rows {seed: the arm - EXTRAP position-error row}; FAR-SETS = separated NEGATIVE at both seeds with |delta| > floor; else FAR-NO-GAIN"""
    if set(rows) != {0, 1}:
        return "NOT-YET"
    pend = " (FLOOR-PENDING)" if floor is None else ""
    ok = all(r.get("status") == "OK" and r["separated"] and r["delta"] < 0 and (floor is None or abs(r["delta"]) > floor) for r in rows.values())
    return ("FAR-SETS" if ok else "FAR-NO-GAIN") + pend


def cost_set(cells):
    return {c for c, v in cells.items() if v.get("cost")}


def gain_set(cells):
    return {c for c, v in cells.items() if v.get("gain")}


def driving_label(s0, s1):
    """s0 / s1 {cell: cell_delta} for seeds 0 / 1 -> (labels, detail).  KEEPS none; ADDS = KEEPS and the same cell a gain at both seeds; COSTS-FAR = a cell costs at both seeds; SEEDS-DISAGREE = a cost at exactly one seed (both labels print when both occur)."""
    if s0 is None or s1 is None:
        return ["NOT-YET"], {"reason": "the card needs both seeds"}
    c0, c1 = cost_set(s0), cost_set(s1)
    both, one = sorted(c0 & c1), sorted(c0 ^ c1)
    det = {"costs_both_seeds": both, "costs_one_seed": one}
    labels = []
    if not c0 and not c1:
        g = sorted(gain_set(s0) & gain_set(s1))
        det["gains_both_seeds"] = g
        return (["ADDS"] if g else ["KEEPS"]), det
    if both:
        labels.append("COSTS-FAR")
    if one:
        labels.append("SEEDS-DISAGREE")
    return labels, det


def route_cost_cells(rows_by_seed):
    """the RC-A cells (30 / 50 / 80 m) that cost at both seeds"""
    if set(rows_by_seed) != {0, 1}:
        return []
    return sorted(set.intersection(*[{c for c, v in r.items() if v.get("cost")} for r in rows_by_seed.values()]))


def detect_cost_labels(rows0, rows1, s1_detects):
    """detected info: a variable that DETECTS in S1 and costs at both seeds in H - S1 -> COSTS-FAR (beside)"""
    return sorted(v for v in rows0 if s1_detects.get(v) and rows0[v].get("cost") and rows1.get(v, {}).get("cost"))


def echo_share(real, token_on):
    """the share of the arm's hits (S_CKPT120 = 1) that sit on windows where the tactical token carries the side: the part that could be an echo of the tactical input rather than the route text (UNVERIFIED operationalisation)"""
    hit = np.asarray(real) == 1
    return {"hits": int(hit.sum()), "hits_on_token_windows": int((hit & np.asarray(token_on, bool)).sum()), "share": float((hit & np.asarray(token_on, bool)).sum() / hit.sum()) if hit.any() else None,
            "status": "UNVERIFIED (reader's operationalisation of the echo split; the card names the share, not its definition)"}


def floors_over(pairs, keys):
    """pairs {name: (a_cells, b_cells)} -> ({key: max |mean a - mean b| over the available pairs or None}, [pair names used])"""
    out, used = {k: None for k in keys}, []
    for name, (a, b) in pairs.items():
        if a is None or b is None:
            continue
        used.append(name)
        for k in keys:
            f = OR.floor_of(a[k], b[k])
            if f is not None and (out[k] is None or f > out[k]):
                out[k] = f
    return out, used


# ------------------------------------------------------------------------------------------------------------------------------ inputs
def ground120(CK):
    join, navtok, navtext, ci, t, eid = SS.load_inputs()
    G = SS.ground(navtok, navtext, join, ci, t)
    tr = dict(np.load(C.pinned("eval141_grid_truth.npz")))
    rc = dict(np.load(C.pinned("rc120_table.npz")))
    if not (np.array_equal(rc["eval_clip"], navtext["eval_clip"]) and np.array_equal(rc["eval_off"], navtext["eval_off"])):
        raise RuntimeError("rc120 table clip order / offsets differ from the navtext table")
    G["rc120"] = rc["eval_rc120"][G["row"]].astype(np.float64)
    G["rc120_ok"] = np.asarray(rc["eval_rc120_ok"], bool)[G["row"]]
    chk = {"rows": bool(np.array_equal(tr["rows"], G["row"])), "side": bool(np.array_equal(tr["side"], G["side"])), "d_text": bool(np.allclose(tr["d_text"], G["d_text"], equal_nan=True)),
           "eid": bool(np.array_equal(tr["eid"], eid))}
    arcs = [float(x) for x in tr["arcs"]]
    i120 = arcs.index(120.0) if 120.0 in arcs else None
    chk["rc_ok_120"] = bool(i120 is not None and np.array_equal(np.asarray(tr["rc_ok"])[:, i120], G["rc120_ok"]))
    G["token_on"], G["v_now"], G["has"] = np.asarray(tr["token_on"], bool), np.asarray(tr["v_now"], np.float64), np.asarray(tr["has"], bool)
    perfect = CK.s_ckpt120(CK.planted(G["side"], "announced"), G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"])
    ok = np.isfinite(perfect)
    kl = {"identity_with_truth_npz": chk, "n_windows": int(ok.sum()), "n_episodes": int(len(np.unique(eid[ok]))), "left": int((G["side"][ok] == 1).sum()), "right": int((G["side"][ok] == -1).sum()),
          "card": CARD_COUNTS, "matches_card": bool(ok.sum() == CARD_COUNTS["windows"] and len(np.unique(eid[ok])) == CARD_COUNTS["episodes"]),
          "rc120_valid_windows": int(G["rc120_ok"].sum()), "rc120_valid_episodes": int(len(np.unique(eid[G["rc120_ok"]]))), "windows_without_v9_row": int((~G["has"]).sum()),
          "basis": "the rc120 table of the release of record (md5 83e30bd3); the card's 341 / 27 were measured on the 2026-10-04 release (f63ece41)"}
    return G, eid, ci, t, kl


def load_t0(path_main, path_far, ci, t, need_far):
    z = dict(np.load(path_main, allow_pickle=True))
    if not (np.array_equal(z["win_ci"], ci) and np.array_equal(z["win_t"], t)):
        raise RuntimeError(f"{path_main}: the evaluator's windows differ from the grid")
    if need_far:
        zf = dict(np.load(path_far, allow_pickle=True))
        if not (np.array_equal(zf["win_ci"], ci) and np.array_equal(zf["win_t"], t)):
            raise RuntimeError(f"{path_far}: the far windows differ from the grid")
        z["ckpt_far"], z["ckpt_far_now"] = zf["ckpt_far"], zf["ckpt_far_now"]
    return z


# ------------------------------------------------------------------------------------------------------------------------------ blocks
def strata_rows(CK, cells, G, eid, nb):
    """the fixed strata of card 6.1: levels (real / donor / zero), per-side real level, best constant, real - donor (paired, n printed)"""
    out = {}
    for name, m in CK.strata(G["token_on"], G["d_text"], G["v_now"]).items():
        sub = {v: np.where(m, cells[v], np.nan) for v in VARIANTS}
        ok = np.isfinite(sub["real"])
        bc, maj = CK.best_constant(sub["real"], G["side"])
        out[name] = {"n_windows": int(ok.sum()), "n_episodes": int(len(np.unique(eid[ok]))) if ok.any() else 0,
                     "level": {v: (float(np.nanmean(sub[v])) if np.isfinite(sub[v]).any() else None) for v in VARIANTS},
                     "per_side_real": {str(s): (float(np.nanmean(sub["real"][(G["side"] == s) & ok])) if ((G["side"] == s) & ok).any() else None) for s in (-1, 1)},
                     "best_constant_level": float(np.nanmean(bc)) if np.isfinite(bc).any() else None, "real_minus_donor": CK.paired_mean(sub["real"], sub["donor"], eid, nb)}
    return out


def route_block(CK, Z, G, eid, nb, CI):
    """one arm at one seed: the 120 m cells and contrasts, strata, 30/50/80 continuity, sensitivity, route setting, echo share (card 6.1, 6.2)."""
    far = {v: Z[v]["ckpt_far_now"] for v in VARIANTS}
    cells = {v: CK.s_ckpt120(far[v], G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"]) for v in VARIANTS}
    blk = {"tier": "T0 at the window's first tick", "wording": WORDING, "route_following_120": CK.route_following(cells["real"], cells["donor"], cells["zero"], G["side"], eid, nb)}
    blk["strata"] = strata_rows(CK, cells, G, eid, nb)
    blk["echo_share"] = echo_share(cells["real"], G["token_on"])
    blk["sensitivity"] = {f"complete_m_{cm}": (lambda c: {"level": float(np.nanmean(c)) if np.isfinite(c).any() else None, "n_windows": int(np.isfinite(c).sum())})(
        CK.s_ckpt120(far["real"], G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"], complete_m=float(cm))) for cm in (10, 30)}
    c3 = {v: SS.route_cells(Z[v]["ckpt"], G) for v in VARIANTS}
    blk["continuity_30_50_80"] = SS.arm_block(CI, c3, G, eid)
    blk["route_setting_120"] = CK.route_setting_120(far["real"], Z["real"]["ckpt"], G["rc120"], G["rc120_ok"], eid, nb)
    return blk, cells


def rung_floor(name_rep, name_s0, store, key_of):
    """|rep - s0| on one statistic of the route blocks, or None"""
    if name_rep not in store or name_s0 not in store:
        return None
    a, b = key_of(store[name_rep]), key_of(store[name_s0])
    return None if a is None or b is None else abs(a - b)


def _delta(row):
    return row["delta"] if row and row.get("status") == "OK" else None


# ------------------------------------------------------------------------------------------------------------------------------ the read
def read(a):
    t_start = time.time()
    CK = load_tree_modules()
    nb = a.n_boot
    CI = SL.load_ci()
    problems, notes = [], []
    tp = C.tree_problem()
    if tp:
        problems.append("tree: " + tp)
    G, eid, ci, t, klabel = ground120(CK)
    if not all(klabel["identity_with_truth_npz"].values()):
        problems.append(f"K-LABEL: the ground truth recomputed here differs from eval141_grid_truth.npz: {klabel['identity_with_truth_npz']}")
    ci_t = {(int(c), int(tt)): i for i, (c, tt) in enumerate(zip(ci, t))}
    same_est = SL.sha256_file(SL.TREE + "/taniteval/taniteval/ci.py") == SL.sha256_file(C.TREE + "/taniteval/taniteval/ci.py")
    if not same_est:
        problems.append("same-estimator check: the STRAT1 tree's ci.py differs from the CKPT120 tree's")
    # ---- arm status: bank / reach / files
    status, reach = {}, {}
    for arm in (a.arms or C.NEW_ARMS):
        v, p = C.reach_verdict(arm, a.reach_dir)
        reach[arm] = v
        why = None if a.no_bank_check else C.bank_refusal(arm, a.ckpts)
        t0_ok = all(os.path.exists(C.new_t0(arm, x)) and os.path.exists(C.new_t0_far(arm, x)) for x in VARIANTS)
        d = C.new_t1_dump(arm)
        dump_ok = os.path.isdir(d) and len([f for f in os.listdir(d) if f.startswith("ep") and f.endswith(".npz")]) == (a.smoke_episodes if a.smoke else 141)
        if why:
            status[arm] = f"REFUSED: {why}"
        elif v == "HEAD-DEAD":
            status[arm] = "HEAD-DEAD (arm INVALID: nothing is read from it)"
        elif v != "PASS":                                  # a missing or unreadable verdict is never a PASS (--no-bank-check skips only the bank files)
            status[arm] = f"NOT-READ (K-REACH-e verdict {v})"
        elif not t0_ok:
            status[arm] = "NOT-YET (a T0 pass is missing)"
        else:
            status[arm] = "READ" + ("" if dump_ok else " (route only: the T1 dump is missing)")
    readable = {arm for arm, s in status.items() if s.startswith("READ")}
    Z, blocks, cells, arrs = {}, {}, {}, {}
    for arm in sorted(readable):
        Z[arm] = {v: load_t0(C.new_t0(arm, v), C.new_t0_far(arm, v), ci, t, True) for v in VARIANTS}
        blocks[arm], cells[arm] = route_block(CK, Z[arm], G, eid, nb, CI)
    base_real = {b: load_t0(C.banked_t0(b, "real"), None, ci, t, False) for b in ("S1_s0", "S1_s1", "S1_s10") if os.path.exists(C.banked_t0(b, "real"))}
    rep = {"tool": "c120_read.py", "card": "PREREG_V8F_CKPT120 (frozen 2026-11-16) + MM rulings 1-12", "smoke": bool(a.smoke), "code_of_record": C.COMMIT, "wording": WORDING,
           "tier": {"STRATEGIC route": "T0 at the window's first tick", "LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav and nav_s, logged scene)", "detected info": "T0"},
           "evidence_class": "MEASURED (ours; T0 evals and dumps, zero training)", "estimator": f"taniteval.ci paired episode-cluster bootstrap, B={nb}, seed 0 ({C.TREE}/taniteval/taniteval/ci.py); 'another draw of EPISODES'; another TRAINING RUN = the floors; another INFERENCE RUN does not apply",
           "arm_status": status, "k_reach_e": reach, "K_LABEL": klabel, "controls": {"same_estimator_ci_sha256": same_est}, "problems": problems, "notes": notes}
    # ---- floors (120 m route cells)
    def sel(blk):
        return blk["route_following_120"]
    fl_rd = rung_floor("H1_rep", "H1_s0", blocks, lambda b: _delta(sel(b)["real_minus_donor"]))
    fl_bal = rung_floor("H1_rep", "H1_s0", blocks, lambda b: _delta(sel(b)["balanced_minus_half"]))
    fl_pos = rung_floor("H1_rep", "H1_s0", blocks, lambda b: _delta(b["route_setting_120"]["extrap"]["arm_minus_baseline"]["pos_err_m"]) if b["route_setting_120"]["extrap"].get("arm_minus_baseline") else None)
    fl_raw = rung_floor("H1_rep", "H1_s0", blocks, lambda b: _delta(sel(b)["raw_real_minus_best_constant"]))
    F120 = {"rd": fl_rd, "bal": fl_bal, "raw": fl_raw, "pos_err_vs_extrap": fl_pos, "pair": "|H1_rep - H1_s0| on the same 120 m cell", "available": fl_rd is not None}
    rep["floors_120"] = F120
    floors120 = {"rd": fl_rd, "bal": fl_bal, "raw": fl_raw} if fl_rd is not None else None
    # ---- route following + route setting labels per rung
    rep["route_120"] = {arm: {"status": status[arm], **blocks[arm]} for arm in blocks}
    rep["route_labels"], rep["far_labels"], rep["route_cost"] = {}, {}, {}
    pairs_rc = {}
    rcs = {}
    for arm in blocks:
        rcs[arm] = SS.rc_cells(SS.rcA_errors(Z[arm]["real"]["ckpt"], G))
    for b, z in base_real.items():
        rcs[b] = SS.rc_cells(SS.rcA_errors(z["ckpt"], G))
    f_rc, f_src = floors_over({"H1_rep-H1_s0": (rcs.get("H1_rep"), rcs.get("H1_s0")), "S1_s10-S1_s0": (rcs.get("S1_s10"), rcs.get("S1_s0"))}, SS.RC_CELLS)
    rep["floors_rc_30_50_80"] = {"F": f_rc, "pairs_used": f_src, "definition": "the larger of |H1_rep - H1_s0| and |S1_s10 - S1_s0| per cell (ruling 9: after four arms the second pair alone)"}
    for rung, spec in RUNGS.items():
        per = {s: sel(blocks[arm]) for s, arm in spec["arms"].items() if arm in blocks}
        lab, why = route_label(per, floors120) if len(per) == 2 else ("NOT-YET", "an arm of this rung is not read")
        for arm in spec["arms"].values():
            if arm in status and status[arm].startswith("HEAD-DEAD"):
                lab, why = "HEAD-DEAD", f"{arm}: K-REACH-e reads HEAD-DEAD; nothing is read"
        rep["route_labels"][rung] = {"label": lab.split(" ")[0] if lab else lab, "label_full": lab, "reason": why, "next_step": NEXT.get(lab.split(" ")[0]), "wording": "H2 follows the supplied route text; it is not a perception or decision skill" if rung == "H2" else WORDING}
        ext = {s: blocks[arm]["route_setting_120"]["extrap"]["arm_minus_baseline"]["pos_err_m"] for s, arm in spec["arms"].items() if arm in blocks and "arm_minus_baseline" in blocks[arm]["route_setting_120"]["extrap"]}
        fl = far_label(ext, fl_pos) if len(ext) == 2 else "NOT-YET"
        rep["far_labels"][rung] = {"label": fl, "floor_F120_pos": fl_pos, "next_step": NEXT.get(fl.split(" ")[0])}
        rows = {}
        for s, arm in spec["arms"].items():
            ref = spec["base"][s]
            if arm in rcs and ref in rcs:
                rows[s] = SS.route_setting(CI, rcs[arm], rcs[ref], f_rc if f_src else None, eid, nb)
                pairs_rc[f"{arm}-{ref}"] = rows[s]
        rc_cost = route_cost_cells(rows)
        rep["route_cost"][rung] = {"cells_cost_at_both_seeds": rc_cost, "label": "ROUTE-COST" if rc_cost else ("none" if len(rows) == 2 else "NOT-YET"), "next_step": NEXT["ROUTE-COST"] if rc_cost else None, "floor_pending": not f_src}
    rep["route_setting_30_50_80"] = pairs_rc
    base_extrap = {}
    for b in ("S1_s0", "S1_s1"):
        if b in base_real:
            e = CK.far_errors(CK.extrap_from_80(base_real[b]["ckpt"]), G["rc120"], G["rc120_ok"])
            base_extrap[b] = {k: float(np.nanmean(v)) for k, v in e.items()} | {"n_windows": int(np.isfinite(e["pos_err_m"]).sum()), "note": "S1 has no far output: its EXTRAP is the base row"}
    rep["route_setting_120_base_rows"] = base_extrap
    # ---- driving (T1-kin dumps)
    drv = {}
    geom = None
    dump_arms = [arm for arm in readable if os.path.isdir(C.new_t1_dump(arm)) and "(route only" not in status[arm]]
    drv_cells = {}
    if dump_arms:
        import obd_geom as OG
        geom = OG.build()
        ne = a.smoke_episodes if a.smoke else 141
        ref_arm = "S1_s0"
        dg = SR.load_dump_n(C.banked_t1_dump(ref_arm), ne)
        J = geom.join(dg)
        glat, glon = geom.labels(dg["g"])
        need = list(dump_arms) + [b for b in ("S1_s0", "S1_s1", "S1_s10") if os.path.isdir(C.banked_t1_dump(b))]
        for arm in need:
            d = C.new_t1_dump(arm) if arm in C.NEW_ARMS else C.banked_t1_dump(arm)
            if arm in C.NEW_ARMS and not a.no_bank_check:
                want = OR.done_sha(d)
                if want is None or want != C.sha256_file(f"{a.ckpts or C.CKD}/{arm}/ckpt_last.pt"):
                    problems.append(f"dump {arm}: the done file's checkpoint sha256 is missing or differs from the banked checkpoint")
                    continue
            drv_cells[arm] = SR.load_arm_cells(geom, dg, J, glat, glon, d, ne)
        f_drv, f_drv_src = floors_over({"H1_rep-H1_s0": (drv_cells.get("H1_rep"), drv_cells.get("H1_s0")), "S1_s10-S1_s0": (drv_cells.get("S1_s10"), drv_cells.get("S1_s0"))}, OR.CELLS)
        rep["floors_driving"] = {"F": f_drv, "pairs_used": f_drv_src, "definition": "the larger of |H1_rep - H1_s0| and |S1_s10 - S1_s0| per cell at seed 0, applied to both seeds; a missing pair is named here"}
        slim_pairs = {}
        for rung, spec in RUNGS.items():
            per = {}
            for s, arm in spec["arms"].items():
                ref = spec["base"][s]
                if arm in drv_cells and ref in drv_cells:
                    per[s] = OR.compare_cells(CI, drv_cells[arm], drv_cells[ref], dg["eid"], f_drv if f_drv_src else None, nb)
                    slim_pairs[f"{arm}-{ref}"] = SR.slim(per[s])
            labs, det = driving_label(per.get(0), per.get(1)) if len(per) == 2 else (["NOT-YET"], {"reason": "an arm or a comparator dump of this rung is missing"})
            rep.setdefault("driving_labels", {})[rung] = {"labels": labs, "detail": det, "floor_pending": not f_drv_src, "next_step": [NEXT.get(x) for x in labs if x in NEXT]}
        rep["driving_pairs"] = slim_pairs
        fam = SR.family_blocks(slim_pairs) if slim_pairs else {}
        rep["families"] = {f: {"tier": rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "cells": c, "n_windows": int(len(dg["eid"])), "n_episodes": int(len(np.unique(dg["eid"]))), "estimator": "paired episode-cluster bootstrap"}
                           for f, c in fam.items() if f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "PRIMARY")}
        rep["controls"]["K1_driving_self_pair"] = OR.identity_control(CI, dg["eid"])
        if not rep["controls"]["K1_driving_self_pair"]["pass"]:
            problems.append("K1: a driving-cell self-pair did not read exactly 0.0")
    else:
        rep["families"] = {f: {"status": "NOT-YET", "reason": "no new T1 dump is readable"} for f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "PRIMARY")}
    rep["families"]["driving_absent_items"] = {"DK-VIOL / TTC / tactical confusion / goal selection": {"status": "not computable here", "reason": "STRAT1 work items (STRAT1 section 4), no instrument in the frozen STRAT1 reader; headway and time-gap MAE are in LONGITUDINAL", "n": 0}}
    # ---- detected info (T0), H - S1 with the cost rule
    di = {"tier": "T0", "rows": {}}
    try:
        tab = dict(np.load(SL.pinned("strat_labels_table.npz"), allow_pickle=False))
        y = CX.label_rows(tab, G["row"])
        s1_detects = {v: str(o).startswith("DETECTS") for v, o in json.load(open(SL.HERE + "/out/STRAT1_CTX.json", encoding="utf-8"))["outcome"].items()}

        def corr(z):
            parts = CX.split_logits(z["ctx_logits"])
            return {v: np.where(y[:, k] >= 0, (parts[k].argmax(1) == y[:, k]).astype(float), np.nan) for k, v in enumerate(CX.CTX_VARS)}
        cc = {arm: corr(Z[arm]["real"]) for arm in Z}
        for b, z in base_real.items():
            cc[b] = corr(z)
        f_di, f_di_src = floors_over({"H1_rep-H1_s0": (cc.get("H1_rep"), cc.get("H1_s0")), "S1_s10-S1_s0": (cc.get("S1_s10"), cc.get("S1_s0"))}, CX.CTX_VARS)
        di["floors"], di["pairs_used"] = f_di, f_di_src
        for rung in RUNGS:
            rows = {}
            for s, arm in RUNGS[rung]["arms"].items():
                if arm in cc and "S1_s%d" % s in cc:
                    rows[s] = {v: OR.cell_delta(CI, cc[arm][v], cc["S1_s%d" % s][v], eid, True, f_di[v] if f_di_src else None, nb) for v in CX.CTX_VARS}
                    di["rows"][f"{arm}-S1_s{s}"] = SR.slim(rows[s])
            di.setdefault("labels", {})[rung] = {"COSTS-FAR_trigger_variables": detect_cost_labels(rows.get(0, {}), rows.get(1, {}), s1_detects)} if len(rows) == 2 else "NOT-YET"
        di["s1_detects_banked"] = s1_detects
    except Exception as e:                                                                  # noqa: BLE001
        di["status"] = f"NOT-READ ({type(e).__name__}: {str(e)[:160]})"
    rep["detected_info"] = di
    # ---- controls
    ctl = rep["controls"]
    ctl["K7"] = CK.k7(G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"])
    ctl["K7_pass"] = bool(ctl["K7"] == {"hold": 0.0, "announced": 1.0, "mirrored": 0.0})
    from tanitad.models.hjepa import donor_indices
    import torch
    didx = donor_indices(len(eid), episode_ids=torch.as_tensor(eid), seed=0).numpy()
    r8 = CK.k8(G["side"], didx, G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"])
    ctl["K8"] = {"real": r8[0], "donor": r8[1], "expected_donor": r8[2], "pass": bool(r8[0] == 1.0 and abs(r8[1] - r8[2]) < 1e-12)}
    perfect = CK.planted(G["side"], "announced")
    navinf_far = CK.planted(G["side"] * G["token_on"], "announced")
    cp = CK.s_ckpt120(perfect, G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"])
    cn = CK.s_ckpt120(navinf_far, G["side"], G["theta"], G["d_text"], G["text_ok"], G["rc120_ok"])
    bcp, _ = CK.best_constant(cp, G["side"])
    kp = CK.paired_mean(cp, bcp, eid, nb)
    bcn, _ = CK.best_constant(cn, G["side"])
    kn = CK.paired_mean(cn, bcn, eid, nb)
    ctl["K_PERFECT"] = {"raw": kp, "card": K_PERFECT_CARD, "matches_card_3dp": bool(kp.get("status") == "OK" and all(abs(kp[k] - v) < 0.0015 for k, v in (("delta", 0.232), ("lo", 0.090), ("hi", 0.406)))) if nb == 2000 else "not compared (B != 2000)"}
    ctl["K_NAVINF"] = {"raw": kn, "balanced_minus_half": CK.balanced_minus_half(cn[np.isfinite(cn)], G["side"][np.isfinite(cn)], eid[np.isfinite(cn)], nb), "card": K_NAVINF_CARD}
    ctl["K1_route_self_pair"] = CK.k1_self_pair({"real120": next(iter(cells.values()))["real"]} if cells else {}, eid)
    if ctl["K1_route_self_pair"]:
        problems.append(f"K1: route self-pair not 0.0 on {ctl['K1_route_self_pair']}")
    for k, ok in (("K7", ctl["K7_pass"]), ("K8", ctl["K8"]["pass"])):
        if not ok:
            problems.append(f"{k} did not read its required value")
    # K9b
    k9 = {}
    for arm in sorted(readable):
        p = f"{C.T1}/k9b/ticks/ticks_CK120K9B-{arm}.npz"
        pc = f"{C.T1}/ticks/ticks_CK120-{arm}_m10_causal_heads.npz"
        row = {}
        for nm, path in (("logged_batch64", p), ("causal_beside", pc)):
            if not os.path.exists(path):
                row[nm] = {"status": "NOT-YET", "reason": f"missing {path}"}
                continue
            tk = dict(np.load(path, allow_pickle=True))
            m = np.asarray(tk["tick"]) == 0
            if "s_far" not in tk:
                row[nm] = {"status": "NOT-READ", "reason": "the tick dump carries no s_far"}
                continue
            idx = [ci_t[(int(c), int(tt))] for c, tt in zip(np.asarray(tk["ci"])[m], np.asarray(tk["t0"])[m])]
            r = CK.k9b({"ckpt": tk["s_ckpt"][m], "ckpt_far": tk["s_far"][m]}, {"ckpt": Z[arm]["real"]["ckpt"][idx], "ckpt_far": Z[arm]["real"]["ckpt_far"][idx]})
            r["n_windows"] = int(m.sum())
            r["covers_full_grid"] = bool(len(set(idx)) == len(eid))
            row[nm] = r
        k9[arm] = row
        if "pass" in row["logged_batch64"] and not (row["logged_batch64"]["pass"] and row["logged_batch64"]["covers_full_grid"]):
            problems.append(f"K9b: {arm} failed (max |delta| > 1e-6 m or not the full grid)")
    ctl["K9b"] = k9
    # R1 / R2 (must fail) and the PROVISIONAL state
    r1r2 = {}
    if "R1_s0" in blocks:
        lab, why = route_label({0: sel(blocks["R1_s0"]), 1: sel(blocks["R1_s0"])}, floors120)
        r1r2["R1_route_label_s0"] = lab
        r1r2["R1_not_FOLLOWS"] = not lab.startswith("S120-FOLLOWS")
        if "H1_s0" in cells and "R1_s0" in cells:
            pe = {arm: CK.far_errors(Z[arm]["real"]["ckpt_far_now"], G["rc120"], G["rc120_ok"])["pos_err_m"] for arm in ("H1_s0", "R1_s0")}
            row = CK.paired_mean(pe["H1_s0"], pe["R1_s0"], eid, nb)
            r1r2["R1_far_pos_err_H1_s0_minus_R1_s0"] = row
            r1r2["R1_pass"] = bool(r1r2["R1_not_FOLLOWS"] and row.get("status") == "OK" and row["separated"] and row["delta"] < 0)
    if "R2_s0" in blocks:
        st = CK.strata(G["token_on"], G["d_text"], G["v_now"])["token_does_not"]
        rd = blocks["R2_s0"]["strata"]["token_does_not"]["real_minus_donor"]
        r1r2["R2_real_minus_donor_token_off"] = rd
        if "H2_s0" in cells:
            h, r = np.where(st, cells["H2_s0"]["real"], np.nan), np.where(st, cells["R2_s0"]["real"], np.nan)
            row = CK.paired_mean(h, r, eid, nb)
            r1r2["R2_H2_s0_minus_R2_s0_real_token_off"] = row
            r1r2["R2_pass"] = bool(rd.get("status") != "OK" or not rd["separated"]) and bool(row.get("status") == "OK" and row["separated"] and row["delta"] > 0)
    ctl["R1_R2"] = r1r2
    prov = []
    if not r1r2.get("R1_pass"):
        prov.append("R1 has not read as a must-fail pass" + (" (arm not read)" if "R1_s0" not in blocks else " (it did NOT pass: the read is broken)"))
    if not r1r2.get("R2_pass"):
        prov.append("R2 has not read as a must-fail pass" + (" (arm not read)" if "R2_s0" not in blocks else " (it did NOT pass: the read is broken)"))
    if F120["rd"] is None:
        prov.append("H1_rep is not read: F120 is pending, labels carry (FLOOR-PENDING)")
    rep["PROVISIONAL"] = prov
    if "R1_pass" in r1r2 and not r1r2["R1_pass"] or "R2_pass" in r1r2 and not r1r2["R2_pass"]:
        problems.append("R1 / R2 did not read as must-fail arms: the read is broken and nothing from this card is quoted until it is fixed")
    rep["echo_share_note"] = "route-following rows are printed with the echo share beside them (ruling 10); the operationalisation is UNVERIFIED"
    rep["strategic_family"] = {"route_following_120": "see route_120 / route_labels", "route_setting": "route_setting_120 (per arm) + route_setting_30_50_80", "derived_goals": {"status": "not in this arm", "reason": "later card; goal-input rule"},
                               "intent_6_30s": {"status": "the 120 m checkpoint is the 6-30 s output", "n_windows_120_over_v_in_6_30s": int(CK.strata(G["token_on"], G["d_text"], G["v_now"])["t_6_30_s"][G["rc120_ok"]].sum())},
                               "visibility": {"status": "not built", "reason": "STRAT1 section 4.5: label below the constant"}, "detected_info": "detected_info"}
    rep["verdict_state"] = "BROKEN READ" if problems else ("PROVISIONAL" if prov else "FINAL")
    rep["wall_s"] = round(time.time() - t_start, 1)
    return rep


# ------------------------------------------------------------------------------------------------------------------------------ rendering
def _f(x, nd=3):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def row_txt(r):
    if not r or r.get("status") not in ("OK", None):
        return f"n/a (n {r.get('n_windows') if r else '-'})"
    return f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {'SEP' if r['separated'] else 'not sep'} (n {r['n_windows']}/{r['n_episodes']} ep)"


def render_md(rep):
    L = ["# RESULT_CKPT120 (PREREG_V8F_CKPT120 read)", "",
         f"State: **{rep['verdict_state']}**" + ("; " + "; ".join(rep["PROVISIONAL"]) if rep["PROVISIONAL"] else ""), "",
         f"Evidence class: {rep['evidence_class']}. Estimator: {rep['estimator']}. Every STRATEGIC number is worded \"{WORDING}\"; H2 route following is worded \"follows the supplied route text\".", ""]
    if rep["problems"]:
        L += ["## Problems (the read is not quotable until each is fixed)"] + [f"* {p}" for p in rep["problems"]] + [""]
    L += ["Side sign: +1 = left (79 scorable windows at 120 m), -1 = right (262); per-side keys read \"1\" and \"-1\".", ""]
    L += ["## Arms", "", "| arm | status | K-REACH-e |", "|---|---|---|"] + [f"| {a} | {s} | {rep['k_reach_e'].get(a)} |" for a, s in rep["arm_status"].items()] + [""]
    L += ["## Outcomes (T0 first tick for STRATEGIC; T1-kin for driving)", "", "| rung | route following 120 m | route setting 120 m | route cost 30/50/80 | driving |", "|---|---|---|---|---|"]
    for r in RUNGS:
        L.append(f"| {r} | {rep['route_labels'][r]['label_full']} | {rep['far_labels'][r]['label']} | {rep['route_cost'][r]['label']} | {', '.join(rep.get('driving_labels', {}).get(r, {}).get('labels', ['NOT-YET']))} |")
    L += [""] + [f"* {r}: {rep['route_labels'][r]['reason']}. Next: {rep['route_labels'][r]['next_step']}" for r in RUNGS] + [""]
    L += ["## STRATEGIC route following at 120 m (tier T0, first tick)", "", "| arm | n | real | donor | zero | real - donor | balanced - 0.5 | raw (real - best constant) | echo share (UNVERIFIED) |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, b in rep["route_120"].items():
        r = b["route_following_120"]
        L.append(f"| {arm} | {r['n_windows']}/{r['n_episodes']} | {_f(r['level']['real'])} | {_f(r['level']['donor'])} | {_f(r['level']['zero'])} | {row_txt(r['real_minus_donor'])} | {row_txt(r['balanced_minus_half'])} | {row_txt(r['raw_real_minus_best_constant'])} | {_f(b['echo_share']['share'])} |")
    L += ["", "Strata (n windows, real level, real - donor):", ""]
    for arm, b in rep["route_120"].items():
        L.append(f"* {arm}: " + "; ".join(f"{k}: n {v['n_windows']}, {_f(v['level']['real'])}, {row_txt(v['real_minus_donor'])}" for k, v in b["strata"].items()))
    L += ["", "## Route setting at 120 m (position error vs RC-A120, arm - EXTRAP; negative = closer to the driven route)", ""]
    for arm, b in rep["route_120"].items():
        e = b["route_setting_120"]
        L.append(f"* {arm}: arm {_f(e['arm']['pos_err_m'])} m / {_f(e['arm']['head_err_deg'])} deg; EXTRAP {_f((e['extrap'].get('level') or {}).get('pos_err_m'))} m; arm - EXTRAP pos {row_txt((e['extrap'].get('arm_minus_baseline') or {}).get('pos_err_m'))}")
    L += ["", f"Floors at 120 m (F120, |H1_rep - H1_s0|): {rep['floors_120']}", ""]
    L += ["## Driving (24 cells; cost / gain with the floor)", ""]
    for rung, d in rep.get("driving_labels", {}).items():
        L.append(f"* {rung}: {d['labels']} {d['detail']} floor pending: {d['floor_pending']}")
    L += ["", "## Controls", "", "```", json.dumps({k: v for k, v in rep["controls"].items() if k not in ("K9b",)}, indent=1, default=float)[:6000], "```", "",
          f"K-LABEL: {rep['K_LABEL']['n_windows']} windows / {rep['K_LABEL']['n_episodes']} episodes scorable (left {rep['K_LABEL']['left']} / right {rep['K_LABEL']['right']}); the card says 341 / 27 (79 / 262): matches = {rep['K_LABEL']['matches_card']}.", ""]
    txt = "\n".join(L)
    low = txt.lower()
    for w in BANNED:
        if w in low:
            txt = re.sub(re.escape(w), "[word]", txt, flags=re.I)
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=C.OUT)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true", help="a code check: fewer episodes in the dump, never a result")
    ap.add_argument("--smoke-episodes", type=int, default=141)
    ap.add_argument("--no-bank-check", action="store_true", help="smoke only: skip the bank manifest / reach PASS requirement")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--ckpts", default=None)
    ap.add_argument("--reach-dir", default=None)
    a = ap.parse_args(argv)
    rep = read(a)
    os.makedirs(a.out_dir, exist_ok=True)
    jp, mp = f"{a.out_dir}/RESULT_CKPT120.json", f"{a.out_dir}/RESULT_CKPT120.md"
    json.dump(rep, open(jp, "w", encoding="utf-8"), indent=1, default=float)
    open(mp, "w", encoding="utf-8").write(render_md(rep))
    print("WROTE", jp, "|", rep["verdict_state"], "|", {r: rep["route_labels"][r]["label_full"] for r in RUNGS}, "| problems:", len(rep["problems"]), "|", rep["wall_s"], "s")
    return 1 if rep["problems"] and not a.smoke else 0


if __name__ == "__main__":
    sys.exit(main())
