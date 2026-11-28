"""SPEC_ADDENDUM_A5: a TIME-LOCALISED nav input (nav_tl) for the pick -- arms T2 / T3 / T4 / T2c / T2h8.

Pre-registration: SPEC_ADDENDUM_A5.md (sha256 5f102584..., registered 2026-10-04T10:01:55Z). NOTHING in the
definition, horizon, arm list, reported arm or bar is chosen here from a number. Everything is fitted on TRAIN
(T4 only) or fixed by the SPEC; nothing is fitted on EVAL.

Reuses (byte-identical in definition to SPEC sec. 5):
  analyze_route.derive / pick("V2") / pick("V3") rules / masked_argmax / e9_blend / bar_check / tactical_ff,
  rescorer_a2.features / fit_arm / table_for (-> analyze_route.plan_metrics),
  the SAME bootstrap draws (route_metrics.make_draws(ep, B=2000, seed=0)).

Order of work (so the controls are reported FIRST and no arm is scored on a broken instrument):
  1. K0 join control, K1 (clock vs the replay bank's t_label_s), K2 (nav_tl == clip token at t_rel ~ 0),
     the token tally and the window counts;
  2. only if K1 and K2 pass: the arms, the reproduction controls (V2 / V3 / X1 with the CLIP token must
     reproduce the banked SPEC sec. 5 / A2 numbers), the bar and the reading rule.

Run (CPU only):
  PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" OMP_NUM_THREADS=4 \
  python nav_tl_a5.py --stage controls            # step 1 only
  python nav_tl_a5.py --stage all                 # step 1 + 2, writes raw/a5_nav_tl.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_route as A  # noqa: E402
import rescorer_a2 as R2  # noqa: E402
import route_metrics as rm  # noqa: E402

# ------------------------------------------------------------------------------------------------- #
# constants -- each traceable to the run record or to SPEC_ADDENDUM_A5                               #
# ------------------------------------------------------------------------------------------------- #
ANCHOR_S = 8.0           # RAW recording seconds of the v8 record anchor (t0_s, every record) -- A5 "t_rel = t_now_raw - 8.0"
H_DEFAULT = 6.0          # A5 horizon
H_SENS = 8.0             # A5 sensitivity (T2h8) -- reported, NEVER chosen from
W_WINDOW = 8             # cfg.core.window = refc.RefCConfig.window; K1 re-derives it from the bank (now_row - t_start_row = 7)
RAW_OFFSET = 2           # config.json label_clock.{train,eval}.raw_offsets == [2]  (= n_stack - 1, n_stack = 3)
NOMINAL_DT_S = 0.1       # clip_clock.NOMINAL_DT_S -- the trainer's fallback for a clip with no sidecar row and no pose-dt
T2_TAU = 0.05            # A5: "V2's rule (nav-side hard filter, tau 0.05, empty -> V0)"
T3_K = 10                # A5: "nav-compliance term x 10"
K1_TOL_S = 1e-3          # A5 known-value control K1
K2_BAND_S = 0.05         # A5 known-value control K2: t_rel in [-0.05, +0.05]
K2_TIME_S = 6.0          # A5 K2: nav_command.args.time_s <= 6.0
B1T_TURN_DADE = -0.566   # A4 / RESULT.md sec 2.2: B1t GT-turn dADE (the scale for "fraction of the bound captured")
TOKEN_SIDE = {"NAV_TURN_L": 1, "NAV_TURN_R": -1}        # every other token (incl. NAV_FOLLOW_ROAD) -> follow (0)
SIDE_NAME = {1: "left", 0: "follow", -1: "right"}
ARMS_ORDER = ("T2", "T3", "T4", "T2c", "T2h8")


# ------------------------------------------------------------------------------------------------- #
# identity and clock                                                                                 #
# ------------------------------------------------------------------------------------------------- #
def load_pack(D, RAW, tag):
    """analyze_route.load, except the per-pass record is raw/run_<tag>.json (the package keeps it there, not beside the npz)."""
    z = np.load(Path(D) / f"{tag}.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["_json"] = json.loads((Path(RAW) / f"run_{tag}.json").read_text(encoding="utf-8"))
    return d


def sha12(clip_id: str) -> str:
    """The bank's win_sha12: refc_v3_train.py:3580 `hashlib.sha256(str(clip).encode()).hexdigest()[:12]`."""
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]


def stable_sid(clip_id: str) -> int:
    """tanitad.data.v2_dataset.stable_episode_id (blake2b-8 >> 1). Re-implemented, not imported, so the join is
    independent of the code under test; its correctness is checked by the sidecar coverage (136/139) and by the 3
    eval sids the run's config.json names as `tactical_excluded_sids`."""
    return int.from_bytes(hashlib.blake2b(str(clip_id).encode("utf-8"), digest_size=8).digest(), "big") >> 1


def read_sidecar(path):
    table = {}
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln)
                table[int(r["sid"])] = (float(r["grid_start_s"]), float(r["dt_s"]))
    return table


def clock_for(sid, table):
    """(grid_start_s, dt_s, source). A clip with no sidecar row is clocked as the trainer's `_clock_for` does:
    grid_start_s = 0.0 and dt from the clip's own poses, or NOMINAL_DT_S where the poses are too slow for the
    identity. The run's config.json (eval label_clock) records n_pose_dt = 0, n_nominal_dt = 3 for the eval split, so
    the fallback here is (0.0, 0.1); train_diag clips are all covered by the sidecar (checked, reported)."""
    if int(sid) in table:
        g0, dt = table[int(sid)]
        return g0, dt, "sidecar"
    return 0.0, NOMINAL_DT_S, "nominal_dt"


def t_now_raw(t, g0, dt, window=W_WINDOW, raw_offset=RAW_OFFSET):
    """refc_v3_train.py::_now_s, same float expression: float(g0) + (r + raw_offset) * float(dt), r = t + w - 1."""
    r = int(t) + int(window) - 1
    return float(g0) + (r + int(raw_offset)) * float(dt)


# ------------------------------------------------------------------------------------------------- #
# nav_tl                                                                                             #
# ------------------------------------------------------------------------------------------------- #
def nav_tl_state(entries, t_rel, H=H_DEFAULT):
    """-> (side, reason). The FIRST entry with t_end_s > t_rel (not yet finished), in file order (the builder writes
    them sorted by t_start_s; asserted at load). If its t_start_s - t_rel <= H (a turn already under way has a
    negative lead and is included) the side is that of its token (NAV_TURN_L -> +1, NAV_TURN_R -> -1); every other
    token, no such entry, or a start more than H ahead -> follow (0)."""
    for e in entries:
        if e["t_end_s"] > t_rel:
            side = TOKEN_SIDE.get(e["token"], 0)
            if side == 0:
                return 0, "follow_token"
            lead = e["t_start_s"] - t_rel
            if lead <= H:
                return side, ("under_way" if lead <= 0 else "ahead_within_H")
            return 0, "beyond_H"
    return 0, "no_entry_left"


def nav_tl_side(entries, t_rel, H=H_DEFAULT):
    return nav_tl_state(entries, t_rel, H)[0]


def load_records(path):
    """sha12 -> {sid, token_side, time_s, entries, entry_tokens}. clip ids are hashed on load and NEVER stored."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            cid = r["clip_id"]
            ent = [{"token": e["token"], "t_start_s": float(e["t_start_s"]), "t_end_s": float(e["t_end_s"])}
                   for e in r["nav_30s"]["entries"]]
            st = [e["t_start_s"] for e in ent]
            if st != sorted(st):
                raise SystemExit("[A5] nav_30s entries are not sorted by t_start_s -- the 'first entry' rule is ambiguous")
            if float(r["t0_s"]) != ANCHOR_S:
                raise SystemExit(f"[A5] a record anchor is {r['t0_s']}, not {ANCHOR_S}")
            nc = r["nav_command"]
            out[sha12(cid)] = {"sid": stable_sid(cid), "token": nc["token"],
                               "token_side": TOKEN_SIDE.get(nc["token"], 0),
                               "time_s": (None if "time_s" not in nc.get("args", {}) else float(nc["args"]["time_s"])),
                               "entries": ent}
    return out


def window_clock(z, recs, sidecar):
    """per-window (t_rel, g0, dt, source) for a capture pack z (win_sha12, win_t)."""
    ws = np.asarray(z["win_sha12"]).astype(str)
    wt = np.asarray(z["win_t"]).astype(int)
    t_now = np.zeros(len(ws))
    src = []
    for i, (s, t) in enumerate(zip(ws, wt)):
        rec = recs[s]
        g0, dt, so = clock_for(rec["sid"], sidecar)
        t_now[i] = t_now_raw(t, g0, dt)
        src.append(so)
    return t_now - ANCHOR_S, np.array(src)


def nav_tl_windows(z, recs, t_rel, H=H_DEFAULT, donor=None):
    """nav_tl side and reason per window. `donor` (clip sha12 -> donor clip sha12) implements T2c: the donor
    clip's entries are evaluated at the RECIPIENT window's own t_rel."""
    ws = np.asarray(z["win_sha12"]).astype(str)
    side = np.zeros(len(ws), np.int64)
    reason = np.empty(len(ws), object)
    for i, s in enumerate(ws):
        src = recs[donor[s]] if donor is not None else recs[s]
        side[i], reason[i] = nav_tl_state(src["entries"], float(t_rel[i]), H)
    return side, reason


def derangement(keys, seed=0):
    """A fixed seeded derangement of `keys` (sorted): rng.permutation, redrawn until no fixed point."""
    keys = sorted(keys)
    n = len(keys)
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return {keys[i]: keys[int(p[i])] for i in range(n)}


# ------------------------------------------------------------------------------------------------- #
# arm rules (A5)                                                                                     #
# ------------------------------------------------------------------------------------------------- #
def navc_term_from(d, z, side, gate):
    """the nav-compliance TERM recomputed against `side`: gate * 1[dir(c; tau_c) == side] on informative windows
    (control C2 of the route package: the model's own predicate == this, 0 mismatches)."""
    return gate * ((d["dir_c"] == side[:, None]) & (side[:, None] != 0)).astype(np.float64)


def d_with_nav(d, side, z, gate):
    """A copy of derive()'s dict whose nav_side / navc_term are the recomputed ones. Every A.pick / R2.features rule
    that reads the clip token reads nav_side and navc_term and nothing else of the nav."""
    dd = dict(d)
    dd["nav_side"] = side
    dd["navc_term"] = navc_term_from(d, z, side, gate)
    return dd


def pick_T2(z, d, side, tau=T2_TAU):
    """V2's rule: nav-side hard filter within reach_keep, empty survivor set -> the unrestricted E9 argmax (= V0)."""
    return A.pick("V2", tau, z, dict(d, nav_side=side))


def pick_T3(z, d, side, gate, k=T3_K):
    """V3's rule with the predicate recomputed against nav_tl, READING (a): the shipped term (built from the CLIP
    token) is removed and replaced by k x gate x 1[dir == nav_tl side]; i.e. the term the planner sees is built from
    the time-localised nav and scaled x k. (Reading (b), `T3b`, leaves the shipped 1x clip-token term in the score and
    adds (k-1) x the new term -- the literal V3 code with the term swapped. Both are reported; (a) is T3.)"""
    s = A.e9_blend(z["s_core"] - d["navc_term"] + k * navc_term_from(d, z, side, gate), z["e9_graft"])
    return A.masked_argmax(s, d["reach"], d["reach"])[0]


def pick_T3b(z, d, side, gate, k=T3_K):
    return A.pick("V3", k, z, d_with_nav(d, side, z, gate))


# ------------------------------------------------------------------------------------------------- #
# reading rule (fixed in the code BEFORE any arm is scored; A5 "Reading rule")                        #
# ------------------------------------------------------------------------------------------------- #
def reading_rule(bar_t2, bar_t2c):
    c1, c2 = bar_t2["1_turn_ade_and_dircorrect"], bar_t2["2_straight_no_regression"]
    c3, c4 = bar_t2["3_all_ade_not_worse"], bar_t2["4_replicates_seed1"]
    if bar_t2c["CLEARS"]:
        return {"branch": "T2c PASSES", "text": "the instrument is broken; nothing from A5 is quotable"}
    if bar_t2["CLEARS"]:
        return {"branch": "T2 PASSES, T2c FAILS",
                "text": "the route-following defect is largely nav TIMING: L2 (time-localised nav) enters refcv8 as a "
                        "training input AND ships as an inference rule meanwhile"}
    out = []
    if c1 is False:
        out.append("T2 FAILS ON TURNS (no turn gain): nav timing is not the bottleneck; selector training ranks first")
    if c1 and c2 is False:
        out.append("T2 FAILS ON STRAIGHT DAMAGE despite time-localisation: the fan's right-direction candidates are "
                   "poor on straight-adjacent windows; the lever is selector training (R1), not nav")
    if c1 is False and c2 is False:
        out.append("(both named failure branches apply: no turn gain AND straight damage)")
    if c1 and c2 and (c3 is False or c4 is False):
        failed = [n for n, v in (("3 (all-window dADE <= 0)", c3), ("4 (replicates on seed 1)", c4)) if v is False]
        out.append("NO NAMED BRANCH: criteria 1 and 2 pass on seed 0 but criterion " + " and ".join(failed) +
                   " FAILS; A5's reading rule does not name this case (the pre-registered text covers a missing turn gain, "
                   "straight damage, a passing T2 and a passing T2c only)")
    return {"branch": "T2 FAILS", "text": " | ".join(out)}


# ------------------------------------------------------------------------------------------------- #
# controls and tallies (no arm is scored here)                                                         #
# ------------------------------------------------------------------------------------------------- #
def k1_control(tsv_path, recs_eval, sidecar):
    n, mx, n_fb = 0, 0.0, 0
    dnow = set()
    miss = set()
    worst = None
    per_src = {"sidecar": [0, 0.0], "nominal_dt": [0, 0.0]}
    clips = set()
    for ln in open(tsv_path, encoding="utf-8"):
        s, t, now_row, lab = ln.rstrip("\n").split("\t")
        if s not in recs_eval:
            miss.add(s)
            continue
        g0, dt, so = clock_for(recs_eval[s]["sid"], sidecar)
        mine = t_now_raw(int(t), g0, dt)
        err = abs(mine - float(lab))
        dnow.add(int(now_row) - int(t))
        n += 1
        clips.add(s)
        per_src[so][0] += 1
        per_src[so][1] = max(per_src[so][1], err)
        if err > mx:
            mx, worst = err, (s, int(t), round(mine, 5), float(lab))
    return {"n_windows": n, "n_clips": len(clips), "max_abs_diff_s": mx, "tol_s": K1_TOL_S,
            "PASS": bool(n > 0 and mx <= K1_TOL_S and not miss),
            "clips_not_in_eval_labels": sorted(miss), "worst_(sha12,t,mine,bank)": worst,
            "now_row_minus_t_start_row_values": sorted(dnow), "W_derived_from_bank": (sorted(dnow)[0] + 1 if len(dnow) == 1 else None),
            "W_used": W_WINDOW, "by_clock_source": {k: {"n": v[0], "max_abs_diff_s": v[1]} for k, v in per_src.items()},
            "note": "the bank rounds t_label_s to 4 dp, so a correct clock reads <= 5e-5 s"}


def k2_control(z, recs, t_rel, H=H_DEFAULT):
    """windows (captured) with |t_rel| <= 0.05 and nav_command.args.time_s <= 6.0: nav_tl must equal the clip token's side."""
    ws = np.asarray(z["win_sha12"]).astype(str)
    rows = []
    for i, s in enumerate(ws):
        r = recs[s]
        if abs(t_rel[i]) <= K2_BAND_S and r["time_s"] is not None and r["time_s"] <= K2_TIME_S:
            rows.append((i, r["token_side"], nav_tl_side(r["entries"], float(t_rel[i]), H), float(t_rel[i]), r["time_s"]))
    bad = [(i, cs, ts, round(tr, 4), tm) for i, cs, ts, tr, tm in rows if cs != ts]
    # only a start within (H - |t_rel|, H] can flip with the band: report those as the identified edge case
    edge = [b for b in bad if b[4] > H - K2_BAND_S]
    return {"n_windows": len(rows), "n_equal": len(rows) - len(bad), "n_unequal": len(bad),
            "VACUOUS_no_window_in_band": bool(len(rows) == 0),
            "PASS": bool(len(bad) == 0), "unequal_windows_(idx,clip_side,nav_tl_side,t_rel,time_s)": bad,
            "n_unequal_explained_by_band_edge_(time_s>H-0.05)": len(edge)}


def k2_dense(recs, sidecar, H=H_DEFAULT):
    """SUPPLEMENTARY (not a captured window): for every eval record, the integer t whose t_rel is nearest 0 on the
    clip's own clock; K2's rule on that t. Shows K2 on all clips rather than on the few the 8-per-episode grid hits."""
    rows = []
    for s, r in recs.items():
        g0, dt, so = clock_for(r["sid"], sidecar)
        t = int(round((ANCHOR_S - g0) / dt - (W_WINDOW - 1) - RAW_OFFSET))
        tr = t_now_raw(t, g0, dt) - ANCHOR_S
        if abs(tr) <= K2_BAND_S and r["time_s"] is not None and r["time_s"] <= K2_TIME_S:
            rows.append((r["token_side"], nav_tl_side(r["entries"], tr, H), tr, r["time_s"]))
    bad = [x for x in rows if x[0] != x[1]]
    return {"n_clips_with_t_in_band_and_time_s_le_6": len(rows), "n_unequal": len(bad),
            "PASS": bool(len(bad) == 0), "n_unequal_explained_by_band_edge_(time_s>H-0.05)":
                int(sum(1 for x in bad if x[3] > H - K2_BAND_S)),
            "unequal_(clip_side,nav_tl,t_rel,time_s)": [(a, b, round(c, 4), e) for a, b, c, e in bad]}


def class_masks(d):
    return {"turn": A.cls_mask(d, "turn"), "turnL": d["cls"] == "turnL", "turnR": d["cls"] == "turnR",
            "straight": A.cls_mask(d, "straight"), "gentle": A.cls_mask(d, "gentle"),
            "unclassified": d["cls"] == "unclassified", "classified_all": A.cls_mask(d, "all"),
            "every_window": np.ones(d["W"], bool)}


def tally(z, d, recs, t_rel, side, reason, side_h8=None):
    ws = np.asarray(z["win_sha12"]).astype(str)
    clip_side = np.array([recs[s]["token_side"] for s in ws])
    out = {}
    # clip level (139 eval clips)
    clips = sorted(set(ws))
    out["clips"] = {"n": len(clips),
                    "clip_token": {SIDE_NAME[k]: int(sum(1 for s in clips if recs[s]["token_side"] == k)) for k in (1, 0, -1)},
                    "entry_tokens": {}, "n_entries_per_clip": {}}
    for s in clips:
        for e in recs[s]["entries"]:
            out["clips"]["entry_tokens"][e["token"]] = out["clips"]["entry_tokens"].get(e["token"], 0) + 1
        k = str(len(recs[s]["entries"]))
        out["clips"]["n_entries_per_clip"][k] = out["clips"]["n_entries_per_clip"].get(k, 0) + 1
    dis = [s for s in clips if (recs[s]["entries"][0]["token"] != recs[s]["token"])]
    out["clips"]["nav_command_vs_nav_30s_first_entry_disagree"] = {
        "n_clips": len(dis),
        "cases": {f"{recs[s]['token']} vs entries[0]={recs[s]['entries'][0]['token']}": 0 for s in dis}}
    for s in dis:
        out["clips"]["nav_command_vs_nav_30s_first_entry_disagree"]["cases"][
            f"{recs[s]['token']} vs entries[0]={recs[s]['entries'][0]['token']}"] += 1
    out["t_rel_s"] = {"min": round(float(t_rel.min()), 3), "max": round(float(t_rel.max()), 3),
                      "median": round(float(np.median(t_rel)), 3)}
    out["window_reason"] = {r: int((reason == r).sum()) for r in sorted(set(reason))}
    M = class_masks(d)
    wc = {}
    for cn, m in M.items():
        blk = {"n": int(m.sum()),
               "nav_tl_side": {SIDE_NAME[k]: int((m & (side == k)).sum()) for k in (1, 0, -1)},
               "clip_token_side": {SIDE_NAME[k]: int((m & (clip_side == k)).sum()) for k in (1, 0, -1)},
               "reason": {r: int((m & (reason == r)).sum()) for r in sorted(set(reason)) if (m & (reason == r)).any()},
               "nav_tl_non_follow": int((m & (side != 0)).sum()),
               "clip_token_non_follow": int((m & (clip_side != 0)).sum()),
               "crosstab_clip_token_x_nav_tl": {f"{SIDE_NAME[a]}->{SIDE_NAME[b]}": int((m & (clip_side == a) & (side == b)).sum())
                                                for a in (1, 0, -1) for b in (1, 0, -1)}}
        if side_h8 is not None:
            blk["nav_tl_H8_non_follow"] = int((m & (side_h8 != 0)).sum())
        # descriptive nav_tl vs GT direction (NOT part of the bar)
        t = d["target"]
        ok = m & (t != 9)
        if cn in ("turn", "turnL", "turnR", "straight", "gentle", "classified_all") and ok.any():
            blk["descriptive_vs_GT_direction"] = {
                "nav_tl_eq_GT_dir": int((ok & (side == t)).sum()),
                "nav_tl_follow": int((ok & (side == 0)).sum()),
                "nav_tl_opposite_of_GT": int((ok & (side != 0) & (side == -t)).sum()),
                "nav_tl_non_follow_but_GT_straight": int((ok & (side != 0) & (t == 0)).sum()),
                "clip_token_eq_GT_dir": int((ok & (clip_side == t)).sum()),
                "clip_token_follow": int((ok & (clip_side == 0)).sum()),
                "clip_token_opposite_of_GT": int((ok & (clip_side != 0) & (clip_side == -t)).sum())}
        wc[cn] = blk
    out["window_counts"] = wc
    return out


# ------------------------------------------------------------------------------------------------- #
# scoring helpers                                                                                      #
# ------------------------------------------------------------------------------------------------- #
def ff_row(lt, arm, cls):
    r = lt[arm][cls]
    keys = ("ade", "fde", "along_abs_6s", "along_signed_6s", "speed_mae_0_2s", "cross_abs_6s", "heading_mae_0_2s_deg",
            "curv_mae_0_2s", "term_heading_err_deg", "dir_correct", "nav_complies")
    return {k: r[k] for k in keys}


def compare_table(t_mine, t_bank, tol=2e-4):
    """my recomputation (clip token) vs a banked lever table: max |diff| over mean / delta / CI of (ade, dir_correct)
    for turn / straight / all."""
    worst = 0.0
    for cn in ("turn", "straight", "all"):
        for m in ("ade", "dir_correct"):
            a, b = t_mine[cn][m], t_bank[cn][m]
            for k in ("mean", "delta_vs_V0"):
                if a.get(k) is not None and b.get(k) is not None:
                    worst = max(worst, abs(a[k] - b[k]))
            for k in ("ci95", "delta_ci95"):
                if a.get(k) and b.get(k) and a[k][0] is not None and b[k][0] is not None:
                    worst = max(worst, abs(a[k][0] - b[k][0]), abs(a[k][1] - b[k][1]))
    return {"max_abs_diff": worst, "tol": tol, "PASS": bool(worst <= tol)}


def fit_T4(zt, d_train_nav):
    """A2's X1 recipe on train_s0 with `nav_side` / `navc_term` swapped: same features, same standardisation rule, same
    5-fold episode-grouped CV over the same alpha grid (R2.fit_arm)."""
    dt_x = dict(d_train_nav)
    dt_x["_z"] = zt
    Xt = R2.features(zt, dt_x)
    y, ok = R2.oracle_reach(dt_x)
    mu, sd = R2.standardise(Xt, dt_x["reach"])
    cols = [R2.FEATS.index(n) for n in R2.ARMS["X1"]]
    model, info = R2.fit_arm(cols, Xt, dt_x, y, ok, mu, sd)
    return model, info


def posthoc_block(per_seed, ze, de, draws, side_e, side_c, clip_e):
    """POST-HOC, NOT PRE-REGISTERED, never a bar result: descriptive attribution that explains WHY the reported arm reads
    as it does (inference-replicate pooling, seed consistency, which turn windows carry the gain, the share of GT-turn
    windows a nav signal can touch at all)."""
    ar = np.arange(de["W"])
    ep = de["ep"]
    out = {"label": "POST-HOC descriptive; not part of A5's bar or reading rule"}
    ade = {tag: {arm: de["ade_c"][ar, per_seed[tag]["picks"][arm]] for arm in per_seed[tag]["picks"]} for tag in per_seed}
    dl = {tag: {arm: ade[tag][arm] - ade[tag]["V0"] for arm in ade[tag]} for tag in ade}
    M = {"turn": A.cls_mask(de, "turn"), "straight": A.cls_mask(de, "straight"), "all": A.cls_mask(de, "all")}

    def bm(x, m):
        pt, bs, _ = rm.boot_mean(x, ep, m, draws=draws)
        return {"mean": None if not np.isfinite(pt) else round(float(pt), 4), "ci95": rm.ci95(bs)}
    out["pooled_over_the_two_sampler_draws_dADE"] = {
        arm: {cn: bm(0.5 * (dl["eval_s0g"][arm] + dl["eval_s1"][arm]), M[cn]) for cn in M} for arm in ("T2", "T3", "T2c", "T2h8")}
    out["seed0_minus_seed1_dADE_(consistency_of_the_two_draws)"] = {
        arm: {cn: bm(dl["eval_s0g"][arm] - dl["eval_s1"][arm], M[cn]) for cn in ("turn", "all")} for arm in ("T2", "T3")}
    # is the TRUE series better than the donor series? (paired, same windows, same draws)
    out["paired_T2_minus_T2c_ADE_(negative = the true series is better)"] = {
        tag: {cn: bm(ade[tag]["T2"] - ade[tag]["T2c"], M[cn]) for cn in ("turn", "straight", "all")} for tag in ade}
    # which GT-turn windows carry T2's gain (seed 0)
    t = M["turn"]
    for arm in ("T2", "T3"):
        ch = t & (per_seed["eval_s0g"]["picks"][arm] != per_seed["eval_s0g"]["picks"]["V0"])
        g = dl["eval_s0g"][arm]
        gg = np.sort(g[ch])
        out[f"{arm}_turn_windows_seed0"] = {
            "n_turn_windows": int(t.sum()), "n_pick_changed": int(ch.sum()),
            "n_changed_improved": int((g[ch] < 0).sum()), "n_changed_worse": int((g[ch] > 0).sum()),
            "n_episodes_with_a_changed_turn_window": int(len(set(ep[ch]))),
            "sum_dADE_over_changed": round(float(g[ch].sum()), 3),
            "sum_of_5_largest_gains": round(float(gg[:5].sum()), 3) if gg.size else None,
            "median_dADE_over_changed": round(float(np.median(g[ch])), 3) if ch.any() else None}
    # how much of the GT-turn set can a nav signal touch at all?
    act = side_e != 0
    tc = {"GT_turn_windows": int(t.sum()), "nav_tl_active_on_GT_turn": int((t & act).sum()),
          "clip_token_active_on_GT_turn": int((t & (clip_e != 0)).sum()),
          "GT_turn_windows_where_nav_tl_is_follow": int((t & ~act).sum())}
    for arm in ("T2", "B1t", "T3"):
        out_ = {}
        for nm, mk in (("nav_tl_active_GT_turn", t & act), ("nav_tl_silent_GT_turn", t & ~act)):
            out_[nm] = {"n": int(mk.sum()), "seed0": bm(dl["eval_s0g"][arm], mk), "seed1": bm(dl["eval_s1"][arm], mk)}
        tc[f"{arm}_dADE_by_nav_tl_coverage"] = out_
    out["nav_coverage_of_GT_turn_windows"] = tc
    # pure timing vs extra information: partition the GT-turn windows by (clip token active?, nav_tl active?)
    part = {}
    for nm, mk in (("clip_active_and_nav_tl_active", t & (clip_e != 0) & act),
                   ("clip_FOLLOW_but_nav_tl_active_(extra_information_from_nav_30s)", t & (clip_e == 0) & act),
                   ("clip_active_but_nav_tl_follow_(time-localisation_switches_it_off)", t & (clip_e != 0) & ~act),
                   ("both_follow", t & (clip_e == 0) & ~act)):
        part[nm] = {"n": int(mk.sum())}
        for arm in ("T2", "T3"):
            for tag in ("eval_s0g", "eval_s1"):
                g = dl[tag][arm]
                part[nm][f"{arm}_{tag}_sum_dADE"] = round(float(g[mk].sum()), 3)
                part[nm][f"{arm}_{tag}_contribution_to_107_window_mean"] = round(float(g[mk].sum() / t.sum()), 4)
    out["GT_turn_partition_clip_token_vs_nav_tl"] = part
    # T2c sanity: how different is the donor series from the true one
    out["T2c_vs_T2_series"] = {"n_windows_side_differs": int((side_c != side_e).sum()),
                               "n_windows_both_active": int(((side_c != 0) & (side_e != 0)).sum()),
                               "n_windows_both_active_same_side": int(((side_c == side_e) & (side_e != 0)).sum())}
    return out


def posthoc_train_diag(zt, dtr, recs_t, t_rel_t, side_t, gate):
    """POST-HOC, NOT PRE-REGISTERED, NOT HELD-OUT: T2 / T3 have no fitted parameter (tau 0.05, k 10, H 6 are fixed by the
    SPEC), so the already-captured TRAIN-DIAG fans (139 OTHER episodes, train_s0) are a second sample for the RULE. The
    checkpoint was trained on those clips, so this is NOT a generalisation test and carries no bar verdict; it only asks
    whether the sign and size of the rule's effect repeat on different episodes (same B = 2000 estimator, own draws)."""
    g_t = zt["_json"]["gates"]["navc_gate"]
    draws = rm.make_draws(dtr["ep"], B=A.B, seed=A.SEED)
    keys = sorted(set(np.asarray(zt["win_sha12"]).astype(str)))
    der = derangement(keys, seed=0)
    side_c, _ = nav_tl_windows(zt, recs_t, t_rel_t, H_DEFAULT, donor=der)
    side_8, _ = nav_tl_windows(zt, recs_t, t_rel_t, H_SENS)
    picks = {"V0": zt["sel_idx"].copy(), "T2": pick_T2(zt, dtr, side_t), "T3": pick_T3(zt, dtr, side_t, g_t),
             "T2c": pick_T2(zt, dtr, side_c), "T2h8": pick_T2(zt, dtr, side_8), "B1t": A.pick("B1t", None, zt, dtr)}
    lt = R2.table_for(zt, dtr, picks, draws)
    cls = {c: int(A.cls_mask(dtr, c).sum()) for c in ("turn", "straight", "gentle", "all")}
    return {"label": "POST-HOC; TRAIN-DIAG split (seen by the checkpoint); not held-out; no bar verdict",
            "navc_gate_train_pass_equals_eval_pass": bool(abs(g_t - gate) < 1e-12), "class_counts": cls,
            "n_episodes": int(len(np.unique(dtr["ep"]))),
            "nav_tl_non_follow_windows": int((side_t != 0).sum()),
            "arms": {k: {"pick_changed_frac": lt[k]["pick_changed_frac"],
                         **{c: {m: lt[k][c][m] for m in ("ade", "dir_correct")} for c in ("turn", "straight", "all")}}
                     for k in lt}}


# ------------------------------------------------------------------------------------------------- #
# main                                                                                                 #
# ------------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/refcv7_route_bin/2026-10-04")
    ap.add_argument("--labels-train", default="D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz")
    ap.add_argument("--labels-eval", default="D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")
    ap.add_argument("--sidecar", default="D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl")
    ap.add_argument("--k1-cols", default=str(HERE.parent / "raw" / "a5_k1_bank_cols.tsv"),
                    help="TSV: clip_sha12, t_start_row, now_row, t_label_s -- 4 columns of the replay bank rows.jsonl "
                         "(Thor /home/nvidia/refcv7_post/replay/bank/final/rows.jsonl, 2,059 rows), extracted on Thor")
    ap.add_argument("--config", default="D:/refcv7_eval_kit/ckpt/config.json")
    ap.add_argument("--raw", default=str(HERE.parent / "raw"))
    ap.add_argument("--stage", choices=("controls", "all"), default="all")
    a = ap.parse_args()
    D, RAW = Path(a.data), Path(a.raw)
    res = {"spec": {"file": "SPEC_ADDENDUM_A5.md",
                    "sha256_expected": "5f10258477709cc4c4e7161211b2b5e8bdc514fc0ba88c37b6a7beb61a9da49b"}}
    spec_p = HERE.parent / "SPEC_ADDENDUM_A5.md"
    res["spec"]["sha256_on_disk"] = hashlib.sha256(spec_p.read_bytes()).hexdigest()
    if res["spec"]["sha256_on_disk"] != res["spec"]["sha256_expected"]:
        raise SystemExit("[A5] SPEC_ADDENDUM_A5.md does not match its registered sha256 -- refusing to run")
    # ---- inputs ----------------------------------------------------------------------------------------- #
    md5 = {}
    for p in (a.labels_train, a.labels_eval):
        md5[Path(p).name] = hashlib.md5(Path(p).read_bytes()).hexdigest()
    res["inputs_md5"] = md5
    if not md5["s2_labels_v8_train.jsonl.gz"].startswith("b45377a1") or not md5["s2_labels_v8_eval.jsonl.gz"].startswith("eefc38d1"):
        raise SystemExit("[A5] label md5 differs from the run's")
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    lc = cfg["label_clock"]
    res["run_config_label_clock"] = {sp: {k: lc[sp][k] for k in ("raw_offsets", "n_from_sidecar", "n_pose_dt_grid_start_0",
                                                                  "n_nominal_dt_grid_start_0", "n_clips")}
                                     for sp in ("train", "eval")}
    if lc["eval"]["raw_offsets"] != [RAW_OFFSET] or lc["train"]["raw_offsets"] != [RAW_OFFSET]:
        raise SystemExit("[A5] raw_offsets differs from 2")
    sidecar = read_sidecar(a.sidecar)
    recs_e = load_records(a.labels_eval)
    recs_t = load_records(a.labels_train)
    zs = {t: load_pack(D, RAW, t) for t in ("eval_s0g", "eval_s1", "train_s0")}
    for t in ("eval_s0g", "eval_s1", "train_s0"):
        ws = np.asarray(zs[t]["win_sha12"]).astype(str)
        if any(s not in (recs_t if t.startswith("train") else recs_e) for s in set(ws)):
            raise SystemExit(f"[A5] {t}: a window clip has no label record -- join broken")
    if not (np.array_equal(zs["eval_s0g"]["win_t"], zs["eval_s1"]["win_t"])
            and np.array_equal(np.asarray(zs["eval_s0g"]["win_sha12"]).astype(str), np.asarray(zs["eval_s1"]["win_sha12"]).astype(str))):
        raise SystemExit("[A5] eval_s0g and eval_s1 windows differ")
    ze, ze1, zt = zs["eval_s0g"], zs["eval_s1"], zs["train_s0"]
    de, de1, dtr = A.derive(ze), A.derive(ze1), A.derive(zt)
    # ---- K0: the join -------------------------------------------------------------------------------------- #
    ctl = {}
    for tag, z, recs in (("eval_s0g", ze, recs_e), ("eval_s1", ze1, recs_e), ("train_s0", zt, recs_t)):
        ws = np.asarray(z["win_sha12"]).astype(str)
        bank_side = np.array([rm.NAV_SIDE[int(n)] for n in z["nav"]])
        rec_side = np.array([recs[s]["token_side"] for s in ws])
        ctl[f"K0_join_bank_nav_eq_record_clip_token_{tag}"] = {"n": int(len(ws)), "n_equal": int((bank_side == rec_side).sum()),
                                                              "PASS": bool((bank_side == rec_side).all())}
    ex_sids = set(lc["eval"]["g3"]["tactical_excluded_sids"])
    nosid = sorted(r["sid"] for s, r in recs_e.items() if r["sid"] not in sidecar and s in set(np.asarray(ze["win_sha12"]).astype(str)))
    ctl["K0_sidecar_coverage"] = {"eval_clips": 139, "eval_clips_without_sidecar_row": len(nosid),
                                  "sids_equal_the_3_config_tactical_excluded_sids": bool(set(nosid) == ex_sids),
                                  "train_diag_clips_without_sidecar_row": int(len({s for s in np.asarray(zt['win_sha12']).astype(str)
                                                                                    if recs_t[s]['sid'] not in sidecar})),
                                  "fallback_used_for_eval": "(grid_start_s 0.0, dt 0.1) = the trainer's nominal_dt branch; run config: "
                                                            "n_pose_dt 0, n_nominal_dt 3"}
    ctl["K0_sidecar_coverage"]["PASS"] = bool(ctl["K0_sidecar_coverage"]["sids_equal_the_3_config_tactical_excluded_sids"]
                                              and ctl["K0_sidecar_coverage"]["train_diag_clips_without_sidecar_row"] == 0)
    # ---- K1 -------------------------------------------------------------------------------------------------- #
    ctl["K1_clock_vs_replay_bank_t_label_s"] = k1_control(a.k1_cols, recs_e, sidecar)
    t_rel_e, src_e = window_clock(ze, recs_e, sidecar)
    t_rel_t, src_t = window_clock(zt, recs_t, sidecar)
    side_e, reason_e = nav_tl_windows(ze, recs_e, t_rel_e, H_DEFAULT)
    side_e8, reason_e8 = nav_tl_windows(ze, recs_e, t_rel_e, H_SENS)
    side_t, reason_t = nav_tl_windows(zt, recs_t, t_rel_t, H_DEFAULT)
    # ---- K2 -------------------------------------------------------------------------------------------------- #
    # K2 on EVERY captured window set: the EVAL grid (8 per episode) hits t_rel in [-0.05, 0.05] on 0 windows, so the reel
    # windows (every window of 12 eval clips) and the TRAIN-DIAG grid are added; the dense one-t-per-clip check is SUPPLEMENTARY.
    zr = load_pack(D, RAW, "reel_s0")
    t_rel_r, _ = window_clock(zr, recs_e, sidecar)
    k2 = {"eval_s0g_grid": k2_control(ze, recs_e, t_rel_e), "train_s0_grid": k2_control(zt, recs_t, t_rel_t),
          "reel_s0_every_window": k2_control(zr, recs_e, t_rel_r)}
    k2["n_windows_total_captured"] = sum(v["n_windows"] for v in k2.values())
    k2["n_unequal_total_captured"] = sum(v["n_unequal"] for v in k2.values() if isinstance(v, dict))
    k2["PASS"] = bool(k2["n_windows_total_captured"] > 0 and k2["n_unequal_total_captured"] == 0)
    ctl["K2_nav_tl_eq_clip_token_at_t_rel_0_captured_windows"] = k2
    ctl["K2_supplementary_dense_one_t_per_eval_clip"] = k2_dense(recs_e, sidecar)
    res["controls"] = ctl
    res["tally_eval_s0g"] = tally(ze, de, recs_e, t_rel_e, side_e, reason_e, side_e8)
    res["tally_eval_s0g"]["clock_source_windows"] = {k: int((src_e == k).sum()) for k in sorted(set(src_e))}
    res["tally_train_s0"] = {"t_rel_s": {"min": round(float(t_rel_t.min()), 3), "max": round(float(t_rel_t.max()), 3)},
                            "nav_tl_side": {SIDE_NAME[k]: int((side_t == k).sum()) for k in (1, 0, -1)},
                            "clip_token_side": {SIDE_NAME[k]: int(sum(1 for s in np.asarray(zt['win_sha12']).astype(str)
                                                                      if recs_t[s]['token_side'] == k)) for k in (1, 0, -1)},
                            "window_reason": {r: int((reason_t == r).sum()) for r in sorted(set(reason_t))}}
    print("[A5] controls:", json.dumps({k: (v.get("PASS"), {kk: v[kk] for kk in v if kk in ("max_abs_diff_s", "n_windows", "n_unequal", "n_equal")})
                                        for k, v in ctl.items()}), flush=True)
    controls_ok = all(v.get("PASS") for v in ctl.values())
    res["controls_all_pass"] = bool(controls_ok)
    if a.stage == "controls" or not controls_ok:
        RAW.mkdir(parents=True, exist_ok=True)
        out = RAW / ("a5_controls_only.json" if controls_ok else "a5_controls_FAILED.json")
        A.jdump(out, res)
        print("[A5] wrote", out, flush=True)
        if not controls_ok:
            print("[A5] A CONTROL FAILED -- the arms are NOT scored", flush=True)
        return 0 if controls_ok else 2

    # ================================================================================================ #
    # step 2: arms                                                                                       #
    # ================================================================================================ #
    gate = ze["_json"]["gates"]["navc_gate"]
    draws = rm.make_draws(de["ep"], B=A.B, seed=A.SEED)
    der = derangement(sorted(set(np.asarray(ze["win_sha12"]).astype(str))), seed=0)
    res["derangement"] = {"seed": 0, "n_clips": len(der), "fixed_points": int(sum(1 for k, v in der.items() if k == v)),
                          "rule": "rng=default_rng(0); permutation of the sorted eval sha12 list redrawn until no fixed point; "
                                  "donor clip's nav_30s entries evaluated at the RECIPIENT window's own t_rel",
                          "digest_sha256": hashlib.sha256(json.dumps(sorted(der.items())).encode()).hexdigest()}
    side_c, reason_c = nav_tl_windows(ze, recs_e, t_rel_e, H_DEFAULT, donor=der)
    res["tally_T2c_donor_series"] = {"nav_tl_side": {SIDE_NAME[k]: int((side_c == k).sum()) for k in (1, 0, -1)},
                                     "on_GT_turn": {SIDE_NAME[k]: int((A.cls_mask(de, 'turn') & (side_c == k)).sum()) for k in (1, 0, -1)},
                                     "on_GT_straight": {SIDE_NAME[k]: int((A.cls_mask(de, 'straight') & (side_c == k)).sum()) for k in (1, 0, -1)}}
    clip_e = np.array([recs_e[s]["token_side"] for s in np.asarray(ze["win_sha12"]).astype(str)])
    clip_t = np.array([recs_t[s]["token_side"] for s in np.asarray(zt["win_sha12"]).astype(str)])

    # ---- REPRODUCTION CONTROLS FIRST (the CLIP token through THIS code path must give the banked numbers) ---------- #
    # (no A5 arm is scored until these pass: V2 / V3 / A2-X1 / B1t with the clip token == route_analysis.json / a2_rescorer.json)
    d_clip_train = d_with_nav(dtr, clip_t, zt, gate)
    mX1, infoX1 = fit_T4(zt, d_clip_train)
    a2 = json.loads((RAW / "a2_rescorer.json").read_text(encoding="utf-8"))
    bank_ra = json.loads((RAW / "route_analysis.json").read_text(encoding="utf-8"))
    res["fit_X1_reproduction"] = {"mine": infoX1, "banked": a2["fit_X1"],
                                 "alpha_equal": infoX1["alpha"] == a2["fit_X1"]["alpha"],
                                 "cv_ade_max_abs_diff": max(abs(infoX1["cv_ade_by_alpha"][k] - a2["fit_X1"]["cv_ade_by_alpha"][k])
                                                            for k in a2["fit_X1"]["cv_ade_by_alpha"]),
                                 "theta_max_abs_diff": max(abs(infoX1["theta"][k] - a2["fit_X1"]["theta"][k]) for k in a2["fit_X1"]["theta"])}
    navc_id = {tag: float(np.abs(navc_term_from(d, z, cs, gate) - d["navc_term"]).max())
               for tag, z, d, cs in (("eval_s0g", ze, de, clip_e), ("eval_s1", ze1, de1, clip_e), ("train_s0", zt, dtr, clip_t))}
    pr = {"V0": ze["sel_idx"].copy(), "V2_clip_repro": A.pick("V2", T2_TAU, ze, de), "V3_clip_repro": A.pick("V3", T3_K, ze, de),
          "X1_clip_repro": mX1.pick(R2.features(ze, d_with_nav(de, clip_e, ze, gate)), de["reach"]),
          "B1t": A.pick("B1t", None, ze, de)}
    Lr = R2.table_for(ze, de, pr, draws)
    repro = {"V2_clip_token_vs_banked_V2|0.05": compare_table(Lr["V2_clip_repro"], bank_ra["levers_eval_s0"]["V2|0.05"]),
             "V3_clip_token_vs_banked_V3|10": compare_table(Lr["V3_clip_repro"], bank_ra["levers_eval_s0"]["V3|10"]),
             "X1_clip_token_refit_vs_banked_A2_X1": compare_table(Lr["X1_clip_repro"], a2["levers_eval_s0g"]["X1"]),
             "B1t_vs_banked_A4_B1t": compare_table(Lr["B1t"], bank_ra["A4_bounds"]["eval_s0"]["B1t"]),
             "X1_fit_alpha_equal_and_cv_diff": {"alpha_equal": res["fit_X1_reproduction"]["alpha_equal"],
                                                "cv_ade_max_abs_diff": res["fit_X1_reproduction"]["cv_ade_max_abs_diff"],
                                                "PASS": bool(res["fit_X1_reproduction"]["alpha_equal"]
                                                             and res["fit_X1_reproduction"]["cv_ade_max_abs_diff"] <= 5e-4)},
             "navc_term_recomputed_from_record_clip_token_eq_shipped_term_max_abs": {**navc_id, "PASS": bool(max(navc_id.values()) <= 1e-9)}}
    res["controls"]["reproduction_with_clip_token"] = repro
    print("[A5] reproduction controls:", json.dumps({k: v["PASS"] for k, v in repro.items()}), flush=True)
    if not all(v["PASS"] for v in repro.values()):
        res["controls_all_pass"] = False
        A.jdump(RAW / "a5_controls_FAILED.json", res)
        print("[A5] A REPRODUCTION CONTROL FAILED -- the arms are NOT scored", flush=True)
        return 2

    # ---- T4: refit on train_s0 with the recomputed navc / agree_nav_side ------------------------------------- #
    d_tl_train = d_with_nav(dtr, side_t, zt, gate)
    mT4, infoT4 = fit_T4(zt, d_tl_train)
    res["fit_T4"] = infoT4

    # ---- score on eval_s0g and eval_s1 ------------------------------------------------------------------------ #
    per_seed = {}
    for tag, z, d in (("eval_s0g", ze, de), ("eval_s1", ze1, de1)):
        picks = {"V0": z["sel_idx"].copy(),
                 "T2": pick_T2(z, d, side_e),
                 "T3": pick_T3(z, d, side_e, gate),
                 "T3b": pick_T3b(z, d, side_e, gate),
                 "T4": mT4.pick(R2.features(z, d_with_nav(d, side_e, z, gate)), d["reach"]),
                 "T2c": pick_T2(z, d, side_c),
                 "T2h8": pick_T2(z, d, side_e8),
                 "B1t": A.pick("B1t", None, z, d),                  # label-side context (NOT a lever)
                 "ORACLE": d["oracle"].copy()}
        lt = R2.table_for(z, d, picks, draws)
        per_seed[tag] = {"picks": picks, "lt": lt}
        print(f"[A5] {tag}: " + " | ".join(f"{k} all {lt[k]['all']['ade']['mean']} turn {lt[k]['turn']['ade']['mean']} "
                                           f"str {lt[k]['straight']['ade']['mean']}" for k in lt), flush=True)
    L0, L1 = per_seed["eval_s0g"]["lt"], per_seed["eval_s1"]["lt"]
    arms = ["T2", "T3", "T3b", "T4", "T2c", "T2h8"]
    bars = {k: A.bar_check(L0, k, L1) for k in arms}
    res["controls_all_pass"] = bool(controls_ok)
    # ---- four families (tactical_from_trajectory) ------------------------------------------------------------------- #
    ar = np.arange(de["W"])
    tff = {k: A.tactical_ff(ze["fan"][ar, per_seed["eval_s0g"]["picks"][k]].astype(np.float64), ze)
           for k in ("V0", "T2", "T3", "T4", "T2c", "T2h8", "ORACLE")}
    # ---- B1t scale ----------------------------------------------------------------------------------------------------
    t2_turn = L0["T2"]["turn"]["ade"]["delta_vs_V0"]
    t2_turn_ci = L0["T2"]["turn"]["ade"]["delta_ci95"]
    b1t_turn = L0["B1t"]["turn"]["ade"]["delta_vs_V0"]
    frac = {"T2_dADE_turn": t2_turn, "B1t_bound_registered": B1T_TURN_DADE, "B1t_bound_recomputed_here": b1t_turn,
            "fraction_of_registered_bound": (t2_turn / B1T_TURN_DADE),
            "fraction_of_recomputed_bound": (t2_turn / b1t_turn if b1t_turn else None)}
    # fraction CI: paired bootstrap of (dADE_T2 / dADE_B1t) on the same draws
    ep = de["ep"]
    mt = A.cls_mask(de, "turn")
    P = {k: z_ for k, z_ in per_seed["eval_s0g"]["picks"].items() if k in ("V0", "T2", "B1t")}
    pm = {k: A.plan_metrics(ze["fan"][ar, P[k]].astype(np.float64), ze, de)["ade"] for k in P}
    _, bs_t2, _ = rm.boot_mean(pm["T2"] - pm["V0"], ep, mt, draws=draws)
    _, bs_b1, _ = rm.boot_mean(pm["B1t"] - pm["V0"], ep, mt, draws=draws)
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = bs_t2 / bs_b1
    frac["fraction_of_B1t_bootstrap_ci95_(recomputed_B1t_ratio_per_draw)"] = rm.ci95(fr)
    # ---- reading rule ---------------------------------------------------------------------------------------------------
    res["bar"] = {k: bars[k] for k in bars}
    reading = reading_rule(bars["T2"], bars["T2c"])
    res["reading_rule"] = reading
    res["T2_fraction_of_B1t"] = frac
    res["POSTHOC"] = posthoc_block(per_seed, ze, de, draws, side_e, side_c, clip_e)
    res["POSTHOC_train_diag"] = posthoc_train_diag(zt, dtr, recs_t, t_rel_t, side_t, gate)
    res["eval_s0g"] = {k: L0[k] for k in L0}
    res["eval_s1"] = {k: L1[k] for k in L1}
    res["tactical_from_trajectory_eval_s0g"] = tff
    res["family_tables_eval_s0g"] = {cls: {k: ff_row(L0, k, cls) for k in ("V0", "T2", "T3", "T4", "T2c", "T2h8", "B1t", "ORACLE")}
                                     for cls in ("turn", "straight", "gentle", "all")}
    res["family_notes"] = {"LONGITUDINAL distance keeping": "UNAVAILABLE in this harness (no lead tracks)",
                           "STRATEGIC decision": "UNAVAILABLE (--no-strategic arm); nav compliance reported vs the CLIP token as in RESULT sec 2.1",
                           "scope": "OPEN-LOOP, held-out eval139, one training seed, sampler-seed replicate (eval_s1); the interval answers "
                                    "'another draw of EPISODES' only"}
    res["picks_changed_vs_V0"] = {k: float(L0[k]["pick_changed_frac"]) for k in L0}
    res["picks_changed_vs_V0_seed1"] = {k: float(L1[k]["pick_changed_frac"]) for k in L1}
    res["nav_oracle_caveat"] = ("nav_tl is built from the v8 nav_30s ORACLE (provenance ego-future); an inference-time route-following "
                                "gain under that caveat, not a training lever")
    A.jdump(RAW / "a5_nav_tl.json", res)
    print("[A5] wrote", RAW / "a5_nav_tl.json", flush=True)
    print("[A5] bar:", json.dumps(bars), flush=True)
    print("[A5] reading:", json.dumps(reading), flush=True)
    return 0 if res["controls_all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
