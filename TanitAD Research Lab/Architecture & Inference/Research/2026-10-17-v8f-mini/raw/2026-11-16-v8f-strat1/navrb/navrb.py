"""navrb.py -- roundabout rows of the router text: what they carry, whether a route-only rule gives the exit side, state-only fallback.
CPU only, zero training.  Reads (never edits) the pinned navtok_table.npz, the raw router texts (texts.json, 9,864 texts) parsed by the frozen navtok grammar
(navtok_src.py = raw/2026-11-15-v8f-datascale/navtok/navtok.py, sha256 ed7e290d...), and the window set of navaudit / navecho3.
Committed reading and rule R1: PREREG_NAVRB.md (written before any number).
Rule R1 (route rows only, causal): latest router row with a roundabout exit number at or before NOW; n=1 -> R, n=3 -> L, else no side.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = "F:/TanitAD-scratch/strat1"
for p in (HERE, S1 + "/navaudit"):
    if p not in sys.path:
        sys.path.insert(0, p)
import navtok_src as NT  # noqa: E402
import navaudit as A  # noqa: E402

N3, N, F, L, B = A.N3, A.N, A.F, A.L, A.B
TURN_L, TURN_R = 1, 2
HORIZON = 40                                    # 8 s of 0.2 s steps (sensitivity only)
BAR = 0.90


# ------------------------------------------------------------------------------------------------------ rules (pure, mutable)
def exit_side(n):
    """R1 convention, right-hand traffic: exit 1 -> right, exit 3 -> left, anything else -> no side (0 none / 1 L / 2 R)"""
    return {1: 2, 3: 1}.get(int(n), 0) if n is not None and n >= 0 else 0


def latest_exit_causal(ex, flag, r, o):
    """latest exit number at rows o..r (flag 0 only), scanning back from r; -1 if none. Reads rows <= r only."""
    for k in range(r, o - 1, -1):
        if flag[k] == 0 and ex[k] >= 0:
            return int(ex[k])
    return -1


def latest_exit_current(ex, flag, r, o):
    return int(ex[r]) if flag[r] == 0 and ex[r] >= 0 else -1


def latest_exit_ahead(ex, flag, r, o, n_rows=None):
    """SENSITIVITY ONLY (reads router rows after NOW): the first exit number at r..r+HORIZON, else the causal one"""
    end = o + n_rows if n_rows is not None else len(ex)
    for k in range(r, min(r + HORIZON, end - 1) + 1):
        if flag[k] == 0 and ex[k] >= 0:
            return int(ex[k])
    return latest_exit_causal(ex, flag, r, o)


def is_state(isrb, flag, r):
    return bool(flag[r] == 0 and isrb[r])


RULES = {"exit_side": exit_side, "latest": latest_exit_causal, "state": is_state}


def predict(ex, flag, r, o, rules=RULES):
    return rules["exit_side"](rules["latest"](ex, flag, r, o))


# ------------------------------------------------------------------------------------------------------ table decoding
def parse_texts(path):
    texts = json.load(open(path, encoding="utf-8"))
    P = [NT.parse_full(t) for t in texts]
    return texts, P


def step_arrays(tab, P):
    """per eval step: exit number (-1 none), is-roundabout-class, class name"""
    vocab = [str(s) for s in tab["vocab"]]
    ti = np.asarray(tab["eval_text_i"])
    cid = np.asarray(tab["eval_cls_id"])
    ex_t = np.array([-1 if p.exit_n is None else p.exit_n for p in P], np.int32)
    ex = np.where(ti >= 0, ex_t[np.clip(ti, 0, None)], -1).astype(np.int32)
    isrb_v = np.array([c.startswith("RB_") or c.startswith("ROTARY") for c in vocab])
    isrb = np.where(cid >= 0, isrb_v[np.clip(cid, 0, None)], False)
    return vocab, ex, isrb


def grammar_fields(texts, P):
    """what the roundabout rows can carry (every distinct text of the corpus): token, exit number, distance, side/degree fields, any left/right word"""
    out = {}
    for t, p in zip(texts, P):
        if not (p.cls.startswith("RB_") or p.cls.startswith("ROTARY")):
            continue
        o = out.setdefault(p.cls, {"n_distinct_texts": 0, "with_exit_number": 0, "with_distance": 0, "side_field_nonempty": 0, "degree_field_nonempty": 0,
                                   "text_has_left_right_word": 0, "example": t})
        o["n_distinct_texts"] += 1
        o["with_exit_number"] += int(p.exit_n is not None)
        o["with_distance"] += int(p.dist_m is not None)
        o["side_field_nonempty"] += int(p.side != "")
        o["degree_field_nonempty"] += int(p.degree != "")
        o["text_has_left_right_word"] += int(bool(re.search(r"\b(left|right)\b", t, re.I)))
    unp = [t for t, p in zip(texts, P) if p.cls == "UNPARSED"]
    return out, len(unp), [t for t in unp if re.search(r"roundabout|rotary", t, re.I)][:5]


# ------------------------------------------------------------------------------------------------------ main
def run(dumps, out_dir, nb=N3.NB, log=print):
    from taniteval import ci as CI
    from tanitad.data import v8f_mini_data as D
    t0 = time.time()
    tab = np.load(S1 + "/read/inputs/navtok_table.npz", allow_pickle=False)
    texts, P = parse_texts(HERE + "/texts.json")
    vocab, ex, isrb = step_arrays(tab, P)
    flag = np.asarray(tab["eval_flag"])
    Je = D.load_join(S1 + "/read/inputs/join_eval141.npz")
    na = json.load(open(S1 + "/navaudit/out/RESULT_NAVAUDIT.json"))
    res = {"tool": "navrb.py", "evidence_class": "MEASURED (ours; pinned navtok_table.npz + frozen navtok grammar over texts.json, zero training, CPU)", "tier": N3.TIER,
           "estimator": N3.EST + " (6 episodes in the audited set: descriptive)",
           "inputs_sha256": {"navtok_table.npz": A.sha(S1 + "/read/inputs/navtok_table.npz"), "texts.json": A.sha(HERE + "/texts.json"), "navtok_src.py": A.sha(HERE + "/navtok_src.py"),
                             "RESULT_NAVAUDIT.json": A.sha(S1 + "/navaudit/out/RESULT_NAVAUDIT.json")}}
    # windows as navaudit
    ref = F.load_dump(F.dump_dir(dumps, "C0_s0"))
    eid, ci_, ws = ref["eid"], ref["clip_index"], ref["ws"]
    y = F.v9_rows(dict(np.load(L.pinned("join_eval141.npz"), allow_pickle=False)), ci_, ws)["lat"]
    rows = L.rows_of(Je["off"], ci_, ws)
    off = np.asarray(Je["off"])
    turn = np.isin(y, (TURN_L, TURN_R))
    res["n_windows"], res["n_turn_windows"] = int(len(rows)), int(turn.sum())
    # ---- 1. what the rows carry
    g, n_unp, unp_rb = grammar_fields(texts, P)
    res["grammar_roundabout_classes"] = g
    res["unparsed_texts"] = {"n": n_unp, "mention_roundabout_or_rotary": unp_rb}
    res["table_columns"] = {"eval_cls_id": "int16 index into vocab (44 classes)", "eval_dist_m": "float32, metres from the text suffix ' in <d>m' (NaN if bare)",
                            "eval_flag": "uint8 0 ok / 1 before first row / 2 no nav rows / 3 no v9 row", "eval_text_i": "int32 index into texts.json (raw router text; carries the exit number)",
                            "eval_age_s": "float32 seconds since the row's timestamp",
                            "built_by": "navtok/build_table.py:42-71 (latest nav row with ts <= now_s*1e6 - 195,000 us); classes by navtok.py:22-34 (_RULES), vocab :43-58"}
    r_all = rows
    now_cls = np.array([vocab[c] if c >= 0 else "NONE" for c in np.asarray(tab["eval_cls_id"])[r_all]])
    rb_now = isrb[r_all] & (flag[r_all] == 0)
    tab1 = {}
    for nm in sorted({c for c in now_cls[rb_now]}):
        m = rb_now & (now_cls == nm)
        tab1[nm] = {"windows": int(m.sum()), "episodes": int(len(set(eid[m]))), "with_exit_number": int((ex[r_all][m] >= 0).sum()),
                    "with_distance": int(np.isfinite(np.asarray(tab["eval_dist_m"])[r_all][m]).sum()), "exit_numbers": {str(k): int((ex[r_all][m] == k).sum()) for k in sorted(set(ex[r_all][m])) if k >= 0}}
    res["grid_roundabout_rows_at_NOW"] = {"n_windows_with_roundabout_row": int(rb_now.sum()), "episodes": int(len(set(eid[rb_now]))), "by_class": tab1,
                                          "of_turn_labelled_windows": int((rb_now & turn).sum())}
    steps_rb = (flag == 0) & isrb
    res["eval_steps_roundabout_flag0"] = {"n_steps": int(steps_rb.sum()), "with_exit_number": int((steps_rb & (ex >= 0)).sum()),
                                          "with_distance": int((steps_rb & np.isfinite(np.asarray(tab["eval_dist_m"]))).sum()),
                                          "exit_number_counts": {str(k): int((steps_rb & (ex == k)).sum()) for k in sorted(set(ex[steps_rb])) if k >= 0}}
    # ---- 2. the 43 windows
    lacks = [x for x in na["audit_windows"] if x["class"] == "ROUTE-LACKS-TURN"]
    gt = na["audit_lacks_episode_table_class_groups_flag0_steps"]
    rb_eps = sorted(int(e) for e, gg in gt.items() if (gg.get("RB", 0) + gg.get("ROTARY", 0)) > 0.5 * sum(gg.values()))
    W = [x for x in lacks if x["episode"] in rb_eps]
    res["audited_roundabout_episodes"] = rb_eps
    res["n_audited_windows"] = len(W)
    per, flags_side, flags_cur, flags_ahead, flags_state, flags_R, flags_L = [], [], [], [], [], [], []
    for x in W:
        w = x["window"]
        assert int(eid[w]) == x["episode"] and int(ci_[w]) == x["clip_index"] and int(rows[w] - off[x["clip_index"]]) == x["step"]
        r, o, lab = int(rows[w]), int(off[x["clip_index"]]), int(y[w])
        n_rows = int(Je["n"][x["clip_index"]])
        pc = predict(ex, flag, r, o)
        pcur = exit_side(latest_exit_current(ex, flag, r, o))
        pah = exit_side(latest_exit_ahead(ex, flag, r, o, n_rows))
        st = is_state(isrb, flag, r)
        flags_side.append(int(pc == lab)); flags_cur.append(int(pcur == lab)); flags_ahead.append(int(pah == lab)); flags_state.append(int(st))
        flags_R.append(int(lab == TURN_R)); flags_L.append(int(lab == TURN_L))
        ti = int(tab["eval_text_i"][r])
        per.append({"window": w, "episode": x["episode"], "step": x["step"], "label": "TURN_L" if lab == 1 else "TURN_R", "row_class_at_NOW": vocab[int(tab["eval_cls_id"][r])],
                    "row_text_at_NOW": texts[ti] if ti >= 0 else None, "exit_number_at_NOW": int(ex[r]), "latest_exit_number_causal": latest_exit_causal(ex, flag, r, o),
                    "R1_side": {0: "none", 1: "L", 2: "R"}[pc], "R1_right": bool(pc == lab), "roundabout_state_at_NOW": st, "v9_nav_d_at_now_m": x["v9_nav_d_at_now_m"]})
    ee = np.array([x["episode"] for x in W])
    ci = lambda fl: N3.share_ci(CI, fl, np.ones(len(fl), int), ee, nb)
    res["R1_causal"] = ci(flags_side)
    res["R1_current_row_only"] = ci(flags_cur)
    res["R1_router_rows_ahead_8s_NOT_CAUSAL"] = ci(flags_ahead)
    res["roundabout_state_at_NOW"] = ci(flags_state)
    res["constant_R"], res["constant_L"] = ci(flags_R), ci(flags_L)
    lab_of = np.array([p["label"] for p in per])
    ctab = {}
    for p in per:
        k = str(p["latest_exit_number_causal"])
        ctab.setdefault(k, {"TURN_L": 0, "TURN_R": 0})[p["label"]] += 1
    res["exit_number_vs_label_43_windows"] = ctab
    res["n_with_numbered_exit_row_at_or_before_NOW"] = int(sum(1 for p in per if p["latest_exit_number_causal"] >= 0))
    res["n_windows"] = len(W)
    res["n_episodes"] = int(len(set(ee)))
    res["per_window"] = per
    # wider: every labelled turn window on the grid with a roundabout row at NOW (secondary, not the reading)
    wd = [w for w in np.flatnonzero(turn & rb_now)]
    fl2 = [int(predict(ex, flag, int(rows[w]), int(off[ci_[w]])) == int(y[w])) for w in wd]
    res["secondary_all_turn_windows_with_roundabout_row_at_NOW"] = {"n_windows": len(wd), "episodes": int(len(set(eid[wd]))), "R1_right": int(sum(fl2))}
    # ---- reading
    r1 = res["R1_causal"]["point"]
    st = res["roundabout_state_at_NOW"]["point"]
    res["reading"] = "RB-SIDE-FROM-ROUTE" if r1 >= BAR else ("RB-STATE-ONLY" if st >= BAR else "RB-ABSENT")
    res["reading_inputs"] = {"R1_share": r1, "state_share": st, "bar": BAR}
    # ---- no ego-future read: rule reads ex / flag at rows <= r only
    rng = np.random.default_rng(0)
    bad, n = 0, 0
    for x in W:
        r, o = int(rows[x["window"]]), int(off[x["clip_index"]])
        ex2, fl2_ = ex.copy(), flag.copy()
        ex2[r + 1:] = rng.integers(0, 9, len(ex2[r + 1:]))
        fl2_[r + 1:] = 0
        bad += int(predict(ex, flag, r, o) != predict(ex2, fl2_, r, o))
        n += 1
    res["future_read_probe"] = {"windows": n, "router_rows_after_NOW_replaced_by_garbage_and_prediction_differs": bad,
                                "track_or_ego_arrays_read_by_the_rule": 0}
    res["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "RESULT_NAVRB.json"), "w"), indent=1, default=str)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=N.READ + "/t1")
    ap.add_argument("--n-boot", type=int, default=N3.NB)
    a = ap.parse_args()
    r = run(a.dumps, HERE + "/out", a.n_boot)
    print(json.dumps({k: r[k] for k in ("reading", "reading_inputs", "n_windows", "n_episodes", "elapsed_s")}))
