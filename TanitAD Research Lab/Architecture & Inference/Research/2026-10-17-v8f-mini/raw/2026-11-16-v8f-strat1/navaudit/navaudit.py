"""navaudit.py -- audit of the turn windows where only v9 names the turn, a wide-announcement text nav (new file), and its scoring.
CPU only, zero training.  Imports (never edits) the landed builder copy, navecho / navecho2 / navecho3.
Classification rules (DEFAULT_RULES below), per window with the text feed not CORRECT at NOW (row r = off[ci] + ws):
  MATCH-FAILED       : the table flag at r is not 0 (no usable text row, build_navtext_nav.py:80-81), or the row is sided with a distance but the
                       countdown could not be computed (all-zero nav row, :93-94).
  ANNOUNCED-LATE-CUT : flag 0, the table side at r is the labelled side, the token read FOLLOW because countdown d > d_ann (:100-101).
  ANNOUNCED-LATE-ROW : flag 0, the table side at r is none, and a row with the labelled side and flag 0 appears at a later step within 8 s (40 steps).
  ROUTE-LACKS-TURN   : flag 0, the table side at r is none, and no later step within 8 s holds the labelled side.
  OTHER              : the table side at r is the OTHER side.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = "F:/TanitAD-scratch/strat1"
TREE = S1 + "/build/tree"
INP = S1 + "/read/inputs"
for p in (HERE, S1 + "/navecho3", S1 + "/navecho2", TREE + "/stack", TREE + "/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
import navecho3 as N3  # noqa: E402
import build_navtext_nav as B  # noqa: E402  (the landed builder, imported, not edited)

N2, N = N3.N2, N3.N
F, L = N.F, N.L
TURN_L, TURN_R = 1, 2
STEP_S = 0.2                       # one cache step = 0.2 s (join now_s diff 0.2013)
HORIZON_STEPS = 40                 # 8 s
CLASSES = ("MATCH-FAILED", "ANNOUNCED-LATE-CUT", "ANNOUNCED-LATE-ROW", "ROUTE-LACKS-TURN", "OTHER")
WIDE = (60.0, 8.0)
NARROW = (30.0, 6.0)
INPUTS = {"navtok_table.npz": "2836369e3e14a8c3c49a6b1a42116c36cda522ca42f98a6bcbc8c4e4d45c8992",
          "native_tracks.pt": "1db431f112b1bd7c426073c43c7df5fc75b2a032736c1e2020f02ffb118669e7",
          "join_train.npz": "978f41bf4f1e4c8b75bdfad31d39ee184b59634feb5da902157d306595ac3124"}


def sha(p):
    return B.sha_file(p)


# ------------------------------------------------------------------------------------------------------ classification (pure)
def _failed(c):
    return c["flag"] != 0 or c["unusable"]


def _cut(c):
    return c["side"] == c["lab"] and c["d"] > c["dann"]


def _row_later(c):
    return c["side"] == 0 and c["later_k"] is not None


def _lacks(c):
    return c["side"] == 0 and c["later_k"] is None


DEFAULT_RULES = {"failed": _failed, "cut": _cut, "row_later": _row_later, "lacks": _lacks}


def classify(c, rules=DEFAULT_RULES):
    """c: dict flag, unusable, side (0 none / 1 L / 2 R at r), lab (1 / 2), d (countdown at r), dann (cut at r), later_k (first step 1..40 with the labelled side, flag 0, else None)"""
    if rules["failed"](c):
        return "MATCH-FAILED"
    if rules["cut"](c):
        return "ANNOUNCED-LATE-CUT"
    if rules["row_later"](c):
        return "ANNOUNCED-LATE-ROW"
    if rules["lacks"](c):
        return "ROUTE-LACKS-TURN"
    return "OTHER"


# ------------------------------------------------------------------------------------------------------ builder wrapper
def build_variant(J, tab, split, tracks, dmin, dper):
    """the landed builder's build_split with ONLY the two announcement constants changed (module attributes, restored after)"""
    o = (B.D_ANN_MIN, B.D_ANN_PER_MS)
    B.D_ANN_MIN, B.D_ANN_PER_MS = dmin, dper
    try:
        return B.build_split(J, tab, split, tracks, -1)
    finally:
        B.D_ANN_MIN, B.D_ANN_PER_MS = o


def eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind == "f" else bool(np.array_equal(a, b))


def diff_proof(nav_n, nav_w, nav_d_n, nav_d_w, flag, side_step, dann_n, dann_w, d):
    """what differs between the pinned-cut build and the wide build: only token one-hots, only FOLLOW -> side, only flag-0 sided steps with dann_n < d <= dann_w"""
    changed = (nav_n != nav_w).any(1)
    dd = np.where(np.isfinite(d), d, np.inf)
    exp = (flag == 0) & (side_step > 0) & np.isfinite(d) & (dd > dann_n) & (dd <= dann_w)
    only_tokens = bool((nav_n[:, 3:] == nav_w[:, 3:]).all() and eq(nav_d_n, nav_d_w))
    ok_dir = bool((nav_n[changed, 0] == 1).all() and (nav_w[changed][np.arange(int(changed.sum())), side_step[changed]] == 1).all()) if changed.any() else True
    return {"n_steps_changed": int(changed.sum()), "changed_equals_expected_set": bool(np.array_equal(changed, exp)), "ch3_ch4_nav_d_identical": only_tokens,
            "every_change_is_FOLLOW_to_side": ok_dir}


def future_independence(J, tab, split, tracks, n_samples=200, seed=0, clip_nav=None):
    """no ego-future read: for sampled (clip, step), the step's nav from the full clip equals the nav with every track row after 2j replaced by garbage"""
    clip_nav = clip_nav or B.clip_nav
    rng = np.random.default_rng(seed)
    side = B.side_of(tab["vocab"])
    cls, dist, flag, age = (np.asarray(tab[f"{split}_{k}"]) for k in ("cls_id", "dist_m", "flag", "age_s"))
    bad, n_ok, n_tok = 0, 0, 0
    for _ in range(n_samples):
        c = int(rng.integers(len(J["n"])))
        o, n, name = int(J["off"][c]), int(J["n"][c]), str(np.asarray(J["clip"]).astype(str)[c])
        j = int(rng.integers(n))
        tr = np.asarray(tracks[name], np.float64)
        sl = slice(o, o + n)
        full = clip_nav(cls[sl], dist[sl], flag[sl], age[sl], np.asarray(J["now_s"][sl]), tr, side)[0][j]
        g = tr.copy()
        r = min(2 * j, len(tr) - 1)
        g[r + 1:] = rng.normal(0, 1e6, g[r + 1:].shape)
        garb = clip_nav(cls[sl], dist[sl], flag[sl], age[sl], np.asarray(J["now_s"][sl]), g, side)[0][j]
        n_ok += 1
        n_tok += int(full[1] + full[2] > 0)
        bad += int(not np.array_equal(full, garb))
    return {"n_sampled_steps": n_ok, "n_with_side_token": n_tok, "n_differ_when_future_track_rows_are_garbage": bad}


# ------------------------------------------------------------------------------------------------------ main
def run(dumps, out_dir, nb=N3.NB, log=print):
    from taniteval import ci as CI
    from tanitad.data import v8f_mini_data as D
    t0 = time.time()
    for k, v in INPUTS.items():
        pth = (S1 + "/ckpt120build/in/" + k) if k != "navtok_table.npz" else INP + "/" + k
        if sha(pth) != v:
            raise SystemExit(f"input {k} sha mismatch")
    tab = np.load(INP + "/navtok_table.npz", allow_pickle=False)
    tracks = D.load_native_tracks(S1 + "/ckpt120build/in/native_tracks.pt")
    Je = D.load_join(INP + "/join_eval141.npz")
    Jt = D.load_join(S1 + "/ckpt120build/in/join_train.npz")
    pin = np.load(L.pinned("navtext_nav.npz"), allow_pickle=False)
    res = {"tool": "navaudit.py", "evidence_class": "MEASURED (ours; banked T1-kin dumps + the pinned text table, zero training, CPU)", "tier": N3.TIER, "estimator": N3.EST,
           "builder_sha256": sha(B.__file__), "inputs_sha256": dict(INPUTS, **{"navtext_nav.npz": sha(L.pinned("navtext_nav.npz")), "join_eval141.npz": sha(INP + "/join_eval141.npz")})}
    # ---- repro and diff proof, both splits
    out, proof, ev = {}, {}, None
    for split, J in (("train", Jt), ("eval", Je)):
        n30 = build_variant(J, tab, split, tracks, *NARROW)
        w60 = build_variant(J, tab, split, tracks, *WIDE)
        rep = {f"{split}_nav": n30[0], f"{split}_nav_d": n30[1], f"{split}_rowclamp": n30[2]}
        proof[split] = {"narrow_reproduces_pinned": {k: eq(rep[k], pin[k]) for k in rep}}
        side = B.side_of(tab["vocab"])[np.asarray(tab[f"{split}_cls_id"])].astype(np.int64)
        flag = np.asarray(tab[f"{split}_flag"])
        v = np.zeros(len(flag))
        for c, (name, o, n) in enumerate(zip(np.asarray(J["clip"]).astype(str), J["off"], J["n"])):
            o, n = int(o), int(n)
            tr = np.asarray(tracks[name], np.float64)
            rr = np.minimum(2 * np.arange(n), len(tr) - 1)
            v[o:o + n] = np.where(np.isfinite(tr[rr, 3]), tr[rr, 3], 0.0) if tr.shape[1] > 3 else 0.0
        dann_n, dann_w = np.maximum(NARROW[0], NARROW[1] * v), np.maximum(WIDE[0], WIDE[1] * v)
        proof[split]["diff_wide_vs_pinned"] = diff_proof(n30[0], w60[0], n30[1], w60[1], flag, side, dann_n, dann_w, n30[1].astype(np.float64))
        sel = (flag == 0) & (side > 0) & np.isfinite(n30[1])
        proof[split]["K-RULE_token_is_side_iff_d_le_dann"] = bool(np.array_equal((n30[0][:, 1] + n30[0][:, 2] > 0)[sel], (n30[1].astype(np.float64) <= dann_n)[sel]))
        out.update({f"{split}_clip": np.asarray(J["clip"]).astype(str), f"{split}_off": np.asarray(J["off"]), f"{split}_n": np.asarray(J["n"]),
                    f"{split}_nav": w60[0], f"{split}_nav_d": w60[1], f"{split}_rowclamp": w60[2]})
        if split == "eval":
            ev = dict(side=side, flag=flag, v=v, dann_n=dann_n, dann_w=dann_w, nav_n=n30[0], nav_w=w60[0], d=n30[1].astype(np.float64))
        log(f"build {split} {time.time() - t0:.0f}s")
    out["meta_json"] = np.array(json.dumps({"rule": "navtext_nav.npz rule with d_ann = max(60, 8 v) (only the two constants changed)", "d_ann": list(WIDE), "left": B.LEFT, "right": B.RIGHT}))
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "navtext_nav_wide.npz")
    np.savez_compressed(dst, **out)
    res["wide_npz"] = {"path": "F:/TanitAD-scratch/strat1/navaudit/navtext_nav_wide.npz", "sha256": sha(dst), "bytes": os.path.getsize(dst)}
    res["build_proof"] = proof
    res["future_independence_eval"] = future_independence(Je, tab, "eval", tracks)
    res["future_independence_train"] = future_independence(Jt, tab, "train", tracks, 200, 1)
    log(f"proof {time.time() - t0:.0f}s")
    # ---- windows
    ref = F.load_dump(F.dump_dir(dumps, "C0_s0"))
    eid, ci_, ws, v0 = ref["eid"], ref["clip_index"], ref["ws"], ref["v0"]
    y = F.v9_rows(dict(np.load(L.pinned("join_eval141.npz"), allow_pickle=False)), ci_, ws)["lat"]
    rows = L.rows_of(Je["off"], ci_, ws)
    _, nav_v9 = N.tick0(F.ticks_path(dumps, "C0_s0"), ci_, ws)
    _, nav_dump_n1 = N.tick0(F.ticks_path(dumps, "N1_s0"), ci_, ws)
    nav_pin, nav_wide = ev["nav_n"][rows].astype(np.float64), ev["nav_w"][rows].astype(np.float64)
    res["K-FEED_pinned_table_equals_N1_dump_nav_in"] = bool(np.allclose(nav_pin, nav_dump_n1, atol=1e-6))
    turn = np.isin(y, (TURN_L, TURN_R))
    key_v9, key_p, key_w = N.nav_key(nav_v9), N.nav_key(nav_pin), N.nav_key(nav_wide)
    sp, sv = N3.cue_state(key_p[turn], y[turn]), N3.cue_state(key_v9[turn], y[turn])
    res["n_turn_windows"] = int(turn.sum())
    res["n_text_correct_pinned"], res["n_v9_correct"] = int((sp == 0).sum()), int((sv == 0).sum())
    # ---- audit of the windows where only v9 names the turn
    side_all, flag_all = ev["side"], ev["flag"]
    onlyv9 = np.flatnonzero(turn)[(sp != N3.CORRECT) & (sv == N3.CORRECT)]
    res["n_only_v9_windows"] = int(len(onlyv9))
    cases = []
    vnd = np.asarray(Je["nav_d"], np.float64)
    for w in onlyv9:
        r, c, lab = int(rows[w]), int(ci_[w]), int(y[w])
        o, n = int(Je["off"][c]), int(Je["n"][c])
        j = r - o
        unusable = bool(flag_all[r] == 0 and side_all[r] > 0 and np.isfinite(tab["eval_dist_m"][r]) and not nav_pin[w].any())
        later = None
        for k in range(1, HORIZON_STEPS + 1):
            if j + k >= n:
                break
            if flag_all[r + k] == 0 and side_all[r + k] == lab:
                later = k
                break
        case = dict(flag=int(flag_all[r]), unusable=unusable, side=int(side_all[r]), lab=lab, d=float(ev["d"][r]) if np.isfinite(ev["d"][r]) else float("inf"),
                    dann=float(ev["dann_n"][r]), later_k=later)
        cl = classify(case)
        cases.append({"window": int(w), "episode": int(eid[w]), "clip_index": c, "step": j, "label": "TURN_L" if lab == 1 else "TURN_R", "class": cl, "flag": case["flag"],
                      "table_side_at_now": int(side_all[r]), "countdown_d_m": None if not np.isfinite(ev["d"][r]) else round(float(ev["d"][r]), 2),
                      "d_ann_30_6_m": round(case["dann"], 2), "d_ann_60_8_m": round(float(ev["dann_w"][r]), 2), "later_row_step": later,
                      "later_row_s": None if later is None else round(later * STEP_S, 1),
                      "v9_nav_d_at_now_m": None if not np.isfinite(vnd[r]) else round(float(vnd[r]), 1), "ego_speed_mps": round(float(v0[w]), 2),
                      "wide_names_correct": bool(key_w[w] == lab)})
    # beside: what the table holds in the ROUTE-LACKS episodes (class groups over flag-0 steps), and whether a class the builder maps to none carries the labelled side
    vocab = [str(s) for s in tab["vocab"]]
    clsid = np.asarray(tab["eval_cls_id"])

    def grp(nm):
        return "TURN" if nm.startswith("TURN") else nm.split("_")[0] if nm.split("_")[0] in ("CONTINUE", "KEEP", "MERGE", "RAMP", "UTURN", "RB", "ROTARY", "UNPARSED") else "OTHER"
    lack_eps = sorted({x["episode"] for x in cases if x["class"] == "ROUTE-LACKS-TURN"})
    gtab, mapped_none_match = {}, 0
    for e in lack_eps:
        cc = int(ci_[np.flatnonzero(eid == e)[0]])
        o, n = int(Je["off"][cc]), int(Je["n"][cc])
        g = {}
        for j in range(n):
            if flag_all[o + j] == 0:
                k = grp(vocab[clsid[o + j]])
                g[k] = g.get(k, 0) + 1
        gtab[str(e)] = g
    for x in cases:
        if x["class"] != "ROUTE-LACKS-TURN":
            continue
        o = int(Je["off"][x["clip_index"]])
        n = int(Je["n"][x["clip_index"]])
        suf = "_L" if x["label"] == "TURN_L" else "_R"
        hit = False
        for k in range(0, HORIZON_STEPS + 1):
            if x["step"] + k >= n:
                break
            rr = o + x["step"] + k
            nm = vocab[clsid[rr]]
            if flag_all[rr] == 0 and nm.endswith(suf):
                hit = True
                break
        mapped_none_match += int(hit)
    res["audit_lacks_episode_table_class_groups_flag0_steps"] = gtab
    res["audit_lacks_windows_with_a_none_mapped_class_ending_with_labelled_side_within_8s"] = mapped_none_match
    res["audit_windows"] = cases
    res["audit_counts"] = {k: sum(1 for x in cases if x["class"] == k) for k in CLASSES}
    eps = sorted({x["episode"] for x in cases})
    byep = {}
    for e in eps:
        xs = [x for x in cases if x["episode"] == e]
        m = np.flatnonzero(eid == e)
        cc = int(ci_[m[0]])
        o, n = int(Je["off"][cc]), int(Je["n"][cc])
        byep[str(e)] = dict({"n": len(xs)}, **{k: sum(1 for x in xs if x["class"] == k) for k in CLASSES},
                            episode_steps_flag_nonzero_share=round(float((flag_all[o:o + n] != 0).mean()), 3),
                            episode_steps_with_turn_side_row_share=round(float(((flag_all[o:o + n] == 0) & (side_all[o:o + n] > 0)).mean()), 3),
                            turn_windows_in_episode=int(turn[m].sum()))
    res["audit_by_episode"] = byep
    res["audit_n_episodes"] = len(eps)
    top = sorted(byep.items(), key=lambda kv: -kv[1]["n"])
    res["audit_top5_episode_share"] = round(sum(v["n"] for _, v in top[:5]) / max(len(cases), 1), 3)
    res["audit_flag_values_in_eval_table"] = {int(k): int(v) for k, v in zip(*np.unique(flag_all, return_counts=True))}
    d_at = [x["countdown_d_m"] for x in cases if x["class"] == "ANNOUNCED-LATE-CUT"]
    res["audit_cut_windows_countdown_m"] = {"n": len(d_at), "median": None if not d_at else float(np.median(d_at)), "max": None if not d_at else float(max(d_at))}
    cutc = [x for x in cases if x["class"] == "ANNOUNCED-LATE-CUT"]
    res["audit_within_distance"] = {"text_table_correct_side_countdown_le_100m": sum(1 for x in cutc if x["countdown_d_m"] <= 100),
                                    "text_table_correct_side_countdown_le_200m": sum(1 for x in cutc if x["countdown_d_m"] <= 200),
                                    "v9_ego_turn_distance_le_100m_all_75": sum(1 for x in cases if x["v9_nav_d_at_now_m"] is not None and x["v9_nav_d_at_now_m"] <= 100),
                                    "v9_ego_turn_distance_le_200m_all_75": sum(1 for x in cases if x["v9_nav_d_at_now_m"] is not None and x["v9_nav_d_at_now_m"] <= 200)}
    res["audit_later_row_within_2_8s"] = sum(1 for x in cases if x["class"] == "ANNOUNCED-LATE-ROW" and 2.0 <= x["later_row_s"] <= 8.0)
    # upper bound on the 197: the table holds the labelled side with flag 0 at NOW (any cut)
    res["upper_bound_text_names_correct_if_cut_removed"] = int(sum(1 for w in np.flatnonzero(turn) if flag_all[int(rows[w])] == 0 and side_all[int(rows[w])] == y[w]))
    # ---- scoring of the wide feed
    yt, ets, v0t = y[turn], eid[turn], v0[turn]
    sw = N3.cue_state(key_w[turn], yt)
    d_w, d_v = N3.distance_m(nav_wide)[turn], N3.distance_m(nav_v9)[turn]
    sc = {"text_wide_names_correct": int((sw == 0).sum()), "text_wide_names_other_side": int((sw == 1).sum()), "text_pinned_names_correct": int((sp == 0).sum()),
          "text_pinned_names_other_side": int((sp == 1).sum()), "v9_names_correct": int((sv == 0).sum()), "v9_names_other_side": int((sv == 1).sum())}
    sc["wide_correct_share"] = N3.share_ci(CI, sw == 0, np.ones(len(sw)), ets, nb)
    sc["pinned_correct_share"] = N3.share_ci(CI, sp == 0, np.ones(len(sp)), ets, nb)
    sc["wide_minus_pinned_correct_paired"] = N3.mean_pair(CI, (sw == 0).astype(float), (sp == 0).astype(float), ets, nb)
    sc["v9_minus_wide_correct_paired"] = N3.mean_pair(CI, (sv == 0).astype(float), (sw == 0).astype(float), ets, nb)
    sc["by_side"] = {nm: N3.turn_tables(CI, yt[m], sw[m], sv[m], d_w[m], d_v[m], v0t[m], ets[m], nb)
                     for nm, m in (("TURN_L", yt == 1), ("TURN_R", yt == 2), ("BOTH_SIDES", np.ones(len(yt), bool)))}
    sc["wide_windows_gained_vs_pinned"] = int(((sw == 0) & (sp != 0)).sum())
    sc["wide_windows_lost_vs_pinned"] = int(((sw != 0) & (sp == 0)).sum())
    sc["wrong_side_tokens_on_labelled_turn_windows_wide"] = int((np.isin(key_w, (1, 2)) & np.isin(y, (1, 2)) & (key_w != y)).sum())
    sc["token_windows_beside"] = {"wide_token_windows_all_2399": int(np.isin(key_w, (1, 2)).sum()), "pinned_token_windows_all_2399": int(np.isin(key_p, (1, 2)).sum()),
                                  "v9_token_windows_all_2399": int(np.isin(key_v9, (1, 2)).sum()),
                                  "wide_token_where_label_is_not_a_turn_labelled": int((np.isin(key_w, (1, 2)) & (y >= 0) & ~np.isin(y, (1, 2))).sum()),
                                  "pinned_token_where_label_is_not_a_turn_labelled": int((np.isin(key_p, (1, 2)) & (y >= 0) & ~np.isin(y, (1, 2))).sum())}
    tt = N3.time_to_turn_s(d_w, v0t)
    ok = (sw == 0) & np.isfinite(tt)
    sc["wide_time_to_turn_s_over_correct_windows"] = {"n": int(ok.sum()), "median": N3.q(tt[ok], 50),
                                                      "share_in_2_to_8s": float(((tt[ok] >= 2) & (tt[ok] <= 8)).mean()) if ok.any() else None}
    res["Q_wide_vs_pinned_vs_v9"] = sc
    # ---- nav-only classifier, 5 channels
    cfg = {}
    for nm, m in (("ALL", np.ones(len(y), bool)), ("INF_ANY_pinned_mask", N.split_masks(nav_v9, nav_pin)["INF_ANY"]), ("INF_ANY_wide_mask", N.split_masks(nav_v9, nav_wide)["INF_ANY"])):
        ym = np.where(m, y, N.IGNORE)
        pp, _ = N2.crossfit_hgb(nav_pin, ym, eid)
        pw, _ = N2.crossfit_hgb(nav_wide, ym, eid)
        cfg[nm] = {"n_labelled": int((ym != N.IGNORE).sum()), "macro_f1_pinned": float(N.macro_f1(ym, pp)), "macro_f1_wide": float(N.macro_f1(ym, pw)),
                   "wide_minus_pinned_paired_macro_f1": F.conf_pair(CI, ym, pw, pp, 5, list(N.LAT9), eid, None, nb)["macro_f1"]}
        for lab, nmx in ((1, "TURN_L"), (2, "TURN_R")):
            k = ym == lab
            cfg[nm][f"recall_{nmx}_pinned"] = float((pp[k] == lab).mean()) if k.any() else None
            cfg[nm][f"recall_{nmx}_wide"] = float((pw[k] == lab).mean()) if k.any() else None
        log(f"clf {nm} {time.time() - t0:.0f}s")
    res["nav_only_classifier"] = cfg
    # ---- bar
    fi = res["future_independence_eval"]["n_differ_when_future_track_rows_are_garbage"] + res["future_independence_train"]["n_differ_when_future_track_rows_are_garbage"]
    n_ok, n_wrong = sc["text_wide_names_correct"], sc["text_wide_names_other_side"]
    res["bar"] = {"rule": ">= 160 of 197 correct, 0 wrong-side, no ego-future read", "n_correct": n_ok, "n_wrong_side": n_wrong, "no_future_read": fi == 0,
                  "met": bool(n_ok >= 160 and n_wrong == 0 and fi == 0)}
    res["problems"] = [] if res["K-FEED_pinned_table_equals_N1_dump_nav_in"] else ["K-FEED"]
    for sp_ in ("train", "eval"):
        if not all(proof[sp_]["narrow_reproduces_pinned"].values()):
            res["problems"].append(f"K-REPRO {sp_}")
        d = proof[sp_]["diff_wide_vs_pinned"]
        if not (d["changed_equals_expected_set"] and d["ch3_ch4_nav_d_identical"] and d["every_change_is_FOLLOW_to_side"] and proof[sp_]["K-RULE_token_is_side_iff_d_le_dann"]):
            res["problems"].append(f"K-DIFF {sp_}")
    res["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "RESULT_NAVAUDIT.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=N.READ + "/t1")
    ap.add_argument("--n-boot", type=int, default=N3.NB)
    a = ap.parse_args()
    r = run(a.dumps, HERE, a.n_boot)
    print(json.dumps({"counts": r["audit_counts"], "bar": r["bar"], "problems": r["problems"], "elapsed_s": r["elapsed_s"]}))
