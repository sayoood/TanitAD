"""town_result: applies the controls and the outcomes of PREREG_V8F_TOWN.md MECHANICALLY to the finished raw files and writes RESULT_TOWN.json / RESULT_TOWN.md.
Nothing is re-derived that town_read.py already computed: the READ_*.json statistics are read as they are. Tables and numbers in the .md are generated from the result dict.
A control that does not read its known value makes the outcome INVALID (the statistics are then listed, not read).
usage: python -I town_result.py [--raw DIR] [--out-dir DIR]"""
import os
import sys
import json
import copy
import hashlib
import argparse

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = f"{HERE}/raw"
CARD = f"{os.path.dirname(HERE)}/PREREG_V8F_TOWN.md"
IMPL = f"{HERE}/IMPL_CHOICES.md"

CARD_PREFIX_BYTES = 4676
CARD_PREFIX_SHA256 = "707e29f388bac0cd3d0b077b5be07a93be29d82fdd7b06859070ba169bda56b2"
IMPL_SHA256 = "2c6fc876239844c9ee9e592cc0cbfcbb7202646ffe8f37741cc10bbc45d3405c"
TOOL_SHA256 = "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"
N_WINDOWS, N_EPISODES, N_TICKS, N_BOOT = 2399, 141, 3, 2000
SEEDS = ("s0", "s1")
ARMS = ("C0", "REG")
SHIFT_KEYS = ("shift_committed", "shift_beside_follow_keep_swapped", "shift_beside_three_group")
LON_ORDER = ["HOLD", "STOP", "CREEP", "DECELERATE", "FOLLOW", "KEEP", "ACCELERATE"]
OUTCOMES = ("T-FEEDBACK", "T-STABLE", "MIXED")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ inputs
def load_inputs(raw=RAW, card=CARD, impl=IMPL):
    """Every file the result rests on, with its sha256. Optional informational files are None when absent."""
    shas = {}

    def J(name, optional=False):
        p = f"{raw}/{name}"
        if not os.path.exists(p):
            if optional:
                return None
            raise FileNotFoundError(p)
        shas[name] = sha256_file(p)
        return json.load(open(p, encoding="utf-8"))

    I = {"read": {a: {s: J(f"READ_{a}_{s}.json") for s in SEEDS} for a in ARMS},
         "gate_id": {s: J(f"GATE_ID_C0_{s}.json") for s in SEEDS},
         "gate_twin": {a: {s: J(f"GATE_TWIN_{a}_{s}.json") for s in SEEDS} for a in ARMS},
         "receipt": {a: {s: J(f"receipts/receipt_{a}_{s}.json") for s in SEEDS} for a in ARMS},
         "done": {a: {s: J(f"receipts/done_{a}_{s}.txt") for s in SEEDS} for a in ARMS},
         "id_info": {s: J(f"GATE_ID_INFO_C0_{s}_vs_banked_RIC0_{s}.json") for s in SEEDS},
         "info_drift": J("INFO_DRIFT_vs_banked.json", optional=True),
         "info_dv": J("INFO_DV_SIGNS.json", optional=True),
         "banked_done": {s: J(f"receipts/banked_done_RIC0_{s}.txt", optional=True) for s in SEEDS}}
    cb = open(card, "rb").read()
    I["card_prefix_sha256"] = sha256_bytes(cb[:CARD_PREFIX_BYTES])
    I["card_bytes"] = len(cb)
    I["impl_sha256"] = sha256_file(impl)
    I["input_sha256"] = shas
    return I


# ------------------------------------------------------------------ small helpers
def sep_sign(r):
    """'n.s.' when the interval contains 0, else 'above0' / 'below0'."""
    return "n.s." if r["lo"] <= 0 <= r["hi"] else ("above0" if r["lo"] > 0 else "below0")


def f1_status(r):
    """macro-F1 own - logged: higher is better."""
    return "worse" if (r["delta"] < 0 and r["hi"] < 0) else ("better" if (r["delta"] > 0 and r["lo"] > 0) else "n.s.")


def err_status(r):
    """error (RMSE) own - logged: lower is better."""
    return "worse" if (r["delta"] > 0 and r["lo"] > 0) else ("better" if (r["delta"] < 0 and r["hi"] < 0) else "n.s.")


def slope_block(r):
    return {"point": r["slope"], "lo": r["lo"], "hi": r["hi"], "n_rows": r["n_rows"], "n_episodes": r["n_episodes"], "n_boot": r["n_boot"],
            "separated": sep_sign(r) != "n.s.", "sign_if_separated": sep_sign(r) if sep_sign(r) != "n.s." else None,
            "below0_separated": bool(r["slope"] < 0 and r["hi"] < 0), "not_separated": bool(r["lo"] <= 0 <= r["hi"]), "half_width": (r["hi"] - r["lo"]) / 2}


def delta_block(r, kind):
    st = f1_status(r) if kind == "f1" else err_status(r)
    out = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "n": r.get("n_labelled", r.get("n")), "n_episodes": r.get("n_episodes"), "status": st, "separated": st != "n.s."}
    for k in ("own", "logged", "own_rmse", "logged_rmse"):
        if k in r:
            out[k] = r[k]
    return out


# ------------------------------------------------------------------ controls
def _checks(d, values=None):
    out = {"pass": bool(all(d.values())), "checks": d}
    if values is not None:
        out["values"] = values
    return out


def control_identity(I, s):
    """Control 1 (card): the dump run's executed tracks equal the tool of record's, max |delta| exactly 0.0. Reference = the 4-thread bare tool run of the pinned tool (card Log 09:12)."""
    g = I["gate_id"][s]
    return _checks({
        "gate_PASS": g.get("PASS") is True,
        "reference_is_the_4_thread_bare_run": g.get("against") == f"BARE_{s}",
        "max_abs_delta_tracks_exactly_0.0": g.get("max_abs_delta_tracks") == 0.0,
        "tick_file_equal": g.get("ticks_equal") is True,
        "all_141_episode_dumps_equal": g.get("episode_files") == N_EPISODES and g.get("episode_files_bank") == N_EPISODES and not g.get("episode_files_unequal"),
        "no_tick_key_unequal_or_missing": not g.get("ticks_keys_unequal") and not g.get("ticks_only_in_run") and not g.get("ticks_only_in_bank")},
        values={"max_abs_delta_tracks": g.get("max_abs_delta_tracks"), "episode_files": g.get("episode_files"), "against": g.get("against")})


def control_twin_tick0(I, a, s):
    """Control 2 (card): at the first replan own == logged, d exactly 0.0 and both softmaxes equal."""
    g, r = I["gate_twin"][a][s], I["read"][a][s]
    t0 = g.get("G-TWIN0", {})
    return _checks({
        "G-TWIN0_PASS": t0.get("PASS") is True,
        "tick0_rows_equal_n_windows": t0.get("tick0_rows") == N_WINDOWS,
        "T_outputs_bitwise_equal_at_tick0": t0.get("outputs_bitwise_equal") is True,
        "swapped_inputs_bitwise_equal_at_tick0": t0.get("swapped_inputs_bitwise_equal") is True,
        "d_at_tick0_exactly_0.0_for_all_three_rank_orders": all(r[k].get("d_mean_tick0_max_abs") == 0.0 for k in SHIFT_KEYS)},
        values={"tick0_rows": t0.get("tick0_rows"), "d_tick0_max_abs": [r[k].get("d_mean_tick0_max_abs") for k in SHIFT_KEYS]})


def control_twin_logged(I, a, s):
    """Supporting gates of the twin (not a numbered card control): the twin is really the logged state, and the own outputs equal the tick file."""
    g = I["gate_twin"][a][s]
    tl, ow = g.get("G-TWINL", {}), g.get("G-OWN", {})
    return _checks({
        "G-TWINL_PASS": tl.get("PASS") is True and g.get("PASS") is True,
        "twin_mode_real": tl.get("twin_mode") == "real",
        "every_later_row_differs_in_the_swapped_inputs": tl.get("later_rows") == (N_TICKS - 1) * N_WINDOWS and tl.get("later_swap_unequal") == tl.get("later_rows"),
        "logged_p10f_mismatch_0": tl.get("logged_p10f_mismatch") == 0,
        "kept_keys_are_the_same_tensors": tl.get("keep_key_not_same_object") == 0,
        "G-OWN_PASS_rows_aligned": ow.get("PASS") is True and ow.get("rows_aligned") is True},
        values={"later_rows": tl.get("later_rows"), "later_swap_unequal": tl.get("later_swap_unequal"), "logged_p10f_mismatch": tl.get("logged_p10f_mismatch")})


def control_regression_built(I, s):
    """Control 3, part a: the regression arm is what it claims (T's ego token zeroed on every T call, own and twin; weights restored; same checkpoint as C0)."""
    rr, rc = I["receipt"]["REG"][s], I["receipt"]["C0"][s]
    dr, dc = I["done"]["REG"][s], I["done"]["C0"][s]
    return _checks({
        "arm_flag_set_on_REG_only": rr.get("zero_t_ego") is True and rc.get("zero_t_ego") is False,
        "T_zeroed_on_every_call_own_and_twin": rr.get("twin_calls", 0) > 0 and rr.get("run_t_zeroed_calls") == 2 * rr.get("twin_calls"),
        "unzeroed_C0_run_has_0_zeroed_calls": rc.get("run_t_zeroed_calls") == 0,
        "e_ego_weights_restored_equal_C0": rr.get("e_ego_weight_sha256_after") is not None and rr.get("e_ego_weight_sha256_after") == rc.get("e_ego_weight_sha256_after"),
        "same_checkpoint_sha_as_C0": dr.get("ckpt_sha256") is not None and dr.get("ckpt_sha256") == dc.get("ckpt_sha256")},
        values={"run_t_zeroed_calls": rr.get("run_t_zeroed_calls"), "expected_calls_own_plus_twin": 2 * rr.get("twin_calls", 0)})


def control_regression_slope(I, s):
    """Control 3, part b (card): the slope of the regression arm must read not separated from 0 (the committed with-intercept slope)."""
    r = I["read"]["REG"][s]["shift_committed"]["slope"]
    return _checks({"REG_slope_interval_contains_0": bool(r["lo"] <= 0 <= r["hi"])})


def integrity(I):
    """Not a numbered card control: the frozen documents hold, and every run is the same population with the same tool."""
    out = {"card_prefix_sha_holds": I["card_prefix_sha256"] == CARD_PREFIX_SHA256, "impl_choices_sha_holds": I["impl_sha256"] == IMPL_SHA256}
    for a in ARMS:
        for s in SEEDS:
            r, rc = I["read"][a][s], I["receipt"][a][s]
            sl = r["shift_committed"]["slope"]
            out[f"{a}_{s}"] = bool(r["n_rows"] == N_TICKS * N_WINDOWS and r["n_episodes"] == N_EPISODES and r["rows_per_tick"] == [N_WINDOWS] * N_TICKS and sl["n_boot"] == N_BOOT
                                   and r["rank_order_slowest_to_fastest"] == LON_ORDER and rc.get("tool_sha256") == TOOL_SHA256 and rc.get("threads") == 4 and rc.get("tool_rc") == 0)
    return _checks(out)


def evaluate_controls(I):
    C = {}
    C["C1_identity"] = {s: control_identity(I, s) for s in SEEDS}
    C["C2_twin_tick0"] = {s: _checks({f"{a}": control_twin_tick0(I, a, s)["pass"] for a in ARMS}) for s in SEEDS}
    C["C2_twin_tick0_detail"] = {s: {a: control_twin_tick0(I, a, s) for a in ARMS} for s in SEEDS}
    C["C2b_twin_is_logged"] = {s: _checks({f"{a}": control_twin_logged(I, a, s)["pass"] for a in ARMS}) for s in SEEDS}
    C["C2b_twin_is_logged_detail"] = {s: {a: control_twin_logged(I, a, s) for a in ARMS} for s in SEEDS}
    C["C3a_regression_arm_built"] = {s: control_regression_built(I, s) for s in SEEDS}
    C["C3b_regression_slope_not_separated"] = {s: control_regression_slope(I, s) for s in SEEDS}
    C["integrity"] = integrity(I)
    gate = {k: bool(all(v[s]["pass"] for s in SEEDS)) for k, v in C.items() if k not in ("integrity", "C2_twin_tick0_detail", "C2b_twin_is_logged_detail")}
    gate["integrity"] = C["integrity"]["pass"]
    C["pass_by_control"] = gate
    C["all_pass"] = bool(all(gate.values()))
    return C


# ------------------------------------------------------------------ the outcome logic (card section 'Outcomes, committed now')
def seed_facts(slope, lon_f1):
    """slope: slope_block; lon_f1: delta_block (kind f1). The three booleans the card's outcomes are written in."""
    return {"slope_below0_separated": bool(slope["below0_separated"]), "slope_not_separated": bool(slope["not_separated"]), "lon_f1_worse_separated": lon_f1["status"] == "worse"}


def decide(facts):
    """facts: {seed: seed_facts}. Both outcomes need their conditions at EVERY seed ('at both seeds'); anything else is MIXED."""
    f = [facts[s] for s in SEEDS]
    t_feedback = all(x["slope_below0_separated"] and x["lon_f1_worse_separated"] for x in f)
    t_stable = all(x["slope_not_separated"] and not x["lon_f1_worse_separated"] for x in f)
    assert not (t_feedback and t_stable)
    return {"T-FEEDBACK": t_feedback, "T-STABLE": t_stable, "label": "T-FEEDBACK" if t_feedback else ("T-STABLE" if t_stable else "MIXED")}


def outcome(I, C):
    sb = {s: slope_block(I["read"]["C0"][s]["shift_committed"]["slope"]) for s in SEEDS}
    lf = {s: delta_block(I["read"]["C0"][s]["quality"]["all_replans"]["lon_macro_f1"], "f1") for s in SEEDS}
    facts = {s: seed_facts(sb[s], lf[s]) for s in SEEDS}
    d = decide(facts)
    if not C["all_pass"]:
        return {"label": "INVALID", "logic_applied": False, "failed_controls": [k for k, v in C["pass_by_control"].items() if not v], "facts": facts}
    return {"label": d["label"], "logic_applied": True, "conditions_met": {"T-FEEDBACK": d["T-FEEDBACK"], "T-STABLE": d["T-STABLE"]}, "facts": facts, "failed_controls": []}


NEXT = {
    "T-FEEDBACK": "Card, T-FEEDBACK: T trained on its own states (the DT2 / RECOVER recipe applied to T's labels, under its own card). Until then a zero-training read: O fed T's LOGGED-state code at T1-kin; if that removes the crawl excess at T1-kin, the share T carries is measured.",
    "T-STABLE": "Card, T-STABLE: the crawl lives in O's reading of its own ego state. Next: the same twin read on O (own plan against O fed the logged ego history).",
    "INVALID": "A committed control did not read its known value: fix the failed control, re-run the affected stage, and read nothing from this run.",
}


def next_step(label, q):
    if label in NEXT:
        return {"named_by": "card", "text": NEXT[label]}
    lon_lo, lon_hi = sorted([q["s0"]["cons"]["delta"], q["s1"]["cons"]["delta"]])
    return {"named_by": "this result (the card leaves MIXED to the result); not frozen, each step needs its own card written before data by the Master Mind",
            "text": "MIXED. The card commits a next lever for T-FEEDBACK and for T-STABLE and leaves MIXED to the result; neither clean outcome fired, so both card steps stay open and are ordered by the measured effect.",
            "steps": [
                f"1. Zero-training read at T1-kin: O fed T's LOGGED-state code (the card's interim read under T-FEEDBACK). It measures the share of the crawl that T's own-state code carries WITHOUT needing the slope; "
                f"T's own-state quality is separated-worse at both seeds (T-lon macro-F1 {q['s0']['lon_f1']['delta']:+.3f} / {q['s1']['lon_f1']['delta']:+.3f}, speed-constraint RMSE {lon_lo:+.2f} to {lon_hi:+.2f} m/s), so a T-side error exists whose crawl cost is not yet measured.",
                "2. The twin read on O (own plan against O fed the logged ego history), the card's T-STABLE step. It is independent of step 1 and can run beside it; this result does not clear T, so it does not displace it.",
                "3. Only if step 1 removes crawl excess: T trained on its own states (DT2 / RECOVER recipe on T's labels), under its own card."]}


# ------------------------------------------------------------------ the result
def per_seed_block(I, s):
    c, g = I["read"]["C0"][s], I["read"]["REG"][s]
    cq, gq = c["quality"], g["quality"]
    cell = {}
    for name, r in c["constraint_rmse"]["per_cell"].items():
        if "delta" in r:
            cell[name] = delta_block(r, "err")
    n_cells = {k: sum(1 for v in cell.values() if v["status"] == k) for k in ("worse", "better", "n.s.")}
    out = {
        "slope": slope_block(c["shift_committed"]["slope"]),
        "lon_f1": delta_block(cq["all_replans"]["lon_macro_f1"], "f1"),
        "lat_f1": delta_block(cq["all_replans"]["lat_macro_f1"], "f1"),
        "cons": delta_block(c["constraint_rmse"]["speed_kind_m_per_s_per_row_mean"], "err"),
        "deficit": c["deficit"], "d_mean_all": c["shift_committed"]["d_mean_all"],
        "beside": {
            "share_argmax_slower_than_gt": {k: c["share_argmax_slower_than_gt"][k] for k in ("own", "logged", "delta", "lo", "hi", "n")} | {"separated": sep_sign(c["share_argmax_slower_than_gt"]) != "n.s."},
            "slope_through_origin": slope_block(c["shift_committed"]["slope_through_origin"]),
            "slope_dv_at_T_frame": slope_block(c["shift_committed"]["slope_dv_at_T_frame"]),
            "slope_follow_keep_swapped": slope_block(c["shift_beside_follow_keep_swapped"]["slope"]),
            "slope_three_group": slope_block(c["shift_beside_three_group"]["slope"]),
            "ticks_1_2_lon_f1": delta_block(cq["ticks_1_2"]["lon_macro_f1"], "f1"),
            "ticks_1_2_lat_f1": delta_block(cq["ticks_1_2"]["lat_macro_f1"], "f1"),
            "constraint_rmse_26_cells": {"n_cells": len(cell), "status_counts": n_cells, "cells_worse": sorted(k for k, v in cell.items() if v["status"] == "worse"),
                                         "cells_better": sorted(k for k, v in cell.items() if v["status"] == "better"), "per_cell": cell}},
        "regression_arm": {
            "slope": slope_block(g["shift_committed"]["slope"]), "slope_through_origin": slope_block(g["shift_committed"]["slope_through_origin"]),
            "lon_f1": delta_block(gq["all_replans"]["lon_macro_f1"], "f1"), "lat_f1": delta_block(gq["all_replans"]["lat_macro_f1"], "f1"),
            "dv_mean": g["deficit"]["dv_mean"], "dv_mean_tick2": g["deficit"]["dv_mean_tick2"], "share_abs_dv_ge_0.5": g["deficit"]["share_abs_dv_ge_0.5"]}}
    if I.get("info_dv"):
        out["dv_signs_C0"] = {k: v for k, v in I["info_dv"]["tags"][f"C0_{s}"].items() if k != "twin_npz_sha256"}
        out["dv_signs_REG"] = {k: v for k, v in I["info_dv"]["tags"][f"REG_{s}"].items() if k != "twin_npz_sha256"}
    return out


def identity_info(I):
    out = {"decides": "nothing (card Log 09:12: the banked 12-thread run is informational)"}
    for s in SEEDS:
        g = I["id_info"][s]
        bd = I.get("banked_done", {}).get(s) or {}
        out[s] = {"banked_threads": bd.get("threads"), "chain_threads": I["done"]["C0"][s].get("threads"), "same_ckpt_sha256": bd.get("ckpt_sha256") == I["done"]["C0"][s].get("ckpt_sha256") if bd else None,
                  "same_tool_sha256": bd.get("tool_sha256") == I["done"]["C0"][s].get("tool_sha256") if bd else None, "gate_PASS": g.get("PASS"), "max_abs_delta_tracks_m": g.get("max_abs_delta_tracks"),
                  "episode_files_unequal": len(g.get("episode_files_unequal", [])), "episode_files": g.get("episode_files"), "tick_keys_unequal": g.get("ticks_keys_unequal")}
        if I.get("info_drift"):
            d = I["info_drift"][s]
            out[s].update({"integer_or_bool_columns_unequal": d["columns_integer_or_bool_unequal"], "t_lon_argmax_agreement": d["t_lon_argmax_agreement"], "t_lat_argmax_agreement": d["t_lat_argmax_agreement"],
                           "episode_track_max_abs_delta": d["episode_track_max_abs_delta"], "t_lon_logits_max_abs": d["columns"]["t_lon_logits"]["max_abs"], "t_lon_logits_mean_abs": d["columns"]["t_lon_logits"]["mean_abs"]})
    out["reading"] = ("Same checkpoint sha256 and same tool sha256; only the thread count differs (12 against 4). Float summation order changes the bits and the rollout drifts (per-episode max track delta in the table). "
                      "The committed statistics are all paired within one run (own against logged twin), so they are statistics of the 4-thread rollout; the identity that bears on them is control 1, which reads 0.0. "
                      "The own-state T outputs barely move between the two thread counts (argmax rows and logit differences above), which bounds how much the F1 and share rows could move; the twin side was run at 4 threads only, so "
                      "whether the conclusions would hold on the 12-thread rollout is not tested here.")
    return out


def evaluate(I):
    C = evaluate_controls(I)
    O = outcome(I, C)
    seeds = {s: per_seed_block(I, s) for s in SEEDS}
    res = {
        "card": "PREREG_V8F_TOWN.md", "card_prefix_bytes": CARD_PREFIX_BYTES, "card_prefix_sha256": I["card_prefix_sha256"], "card_prefix_sha256_expected": CARD_PREFIX_SHA256,
        "impl_choices_sha256": I["impl_sha256"], "tier": "T1-kin for the own-state rows, T0 for the twin (logged-state) rows; the paired own-minus-logged difference is the card's statistic, no tier is pooled",
        "evidence_class": "MEASURED (ours; artifacts under town/raw)", "estimator": "episode-cluster bootstrap, B 2,000, seed 0, over 141 episodes (taniteval/ci.py); slope refitted on every draw; paired for F1 and RMSE",
        "n": {"windows": N_WINDOWS, "episodes": N_EPISODES, "ticks_per_window": N_TICKS, "replans": N_TICKS * N_WINDOWS}, "seeds_note": "two training seeds (v8f-recover M2b_s{0,1}_recft_C0); the interval answers another draw of episodes only (H-ESTIM-SEED-1)",
        "outcome": O["label"], "outcome_detail": O,
        "outcome_rule_reading": "'at both seeds' applies to every condition of an outcome (each condition must hold at s0 AND s1); T-lon F1 and the slope are the with-intercept slope and the all-labelled-replans macro-F1 of IMPL_CHOICES.md section 2; the slope condition for T-FEEDBACK is point < 0 and the 95 % upper bound < 0",
        "controls": C, "seeds": seeds, "identity_vs_banked_informational": identity_info(I),
        "families": {"TACTICAL": "read (T-lat, T-lon: shift d, macro-F1 own minus logged, speed-constraint RMSE)", "LONGITUDINAL / LATERAL driving metrics": "not read by this card (mechanism read, not a driving claim)",
                     "STRATEGIC": "PAUSED (PI ruling, PI_DECISION_QUEUE item 25); not read here, decides nothing, not a pass"},
        "next": next_step(O["label"], seeds) if O["label"] != "INVALID" else {"named_by": "card/controls", "text": NEXT["INVALID"]},
        "input_sha256": I["input_sha256"]}
    return res


# ------------------------------------------------------------------ the markdown (generated)
def ci(b, k="point", nd=4):
    return f"{b[k]:+.{nd}f} [{b['lo']:+.{nd}f}, {b['hi']:+.{nd}f}]"


def yn(x):
    return "PASS" if x else "FAIL"


def render_md(res):
    S = res["seeds"]
    C = res["controls"]
    O = res["outcome"]
    lines = [f"# RESULT_TOWN: does the tactical layer's speed decision follow the car's own speed on its own states?", "",
             f"Card `PREREG_V8F_TOWN.md`, frozen prefix {res['card_prefix_bytes']:,} bytes, sha256 `{res['card_prefix_sha256'][:8]}...` ({'holds' if res['card_prefix_sha256'] == res['card_prefix_sha256_expected'] else 'DOES NOT HOLD'}); implementation choices `IMPL_CHOICES.md` sha256 `{res['impl_choices_sha256'][:8]}...`.",
             f"Tier: {res['tier']}. Evidence class: {res['evidence_class']}.",
             f"Estimator: {res['estimator']}. n = {res['n']['windows']:,} windows, {res['n']['episodes']} episodes, {res['n']['ticks_per_window']} replans per window ({res['n']['replans']:,} replans). C0 s0 / s1, 4 threads, CPU.", "",
             f"## Outcome: **{O}**", ""]
    d = res["outcome_detail"]
    if O == "INVALID":
        lines += [f"A committed control did not read its known value ({', '.join(d['failed_controls'])}). The statistics below are listed, not read.", ""]
    else:
        f0, f1 = d["facts"]["s0"], d["facts"]["s1"]
        lines += [f"| condition (card) | s0 | s1 |", "|---|---|---|",
                  f"| slope below 0 and separated (T-FEEDBACK) | {f0['slope_below0_separated']} | {f1['slope_below0_separated']} |",
                  f"| slope not separated (T-STABLE) | {f0['slope_not_separated']} | {f1['slope_not_separated']} |",
                  f"| T-lon macro-F1 own - logged separated-worse (T-FEEDBACK needs True, T-STABLE needs False) | {f0['lon_f1_worse_separated']} | {f1['lon_f1_worse_separated']} |", "",
                  f"T-FEEDBACK: {d['conditions_met']['T-FEEDBACK']}. T-STABLE: {d['conditions_met']['T-STABLE']}. {res['outcome_rule_reading'].split(';')[0]}.", ""]
        if O == "MIXED":
            lines += [f"MIXED here means: the slope is not separated from 0 at either seed, so T-FEEDBACK does not hold; T-lon is separated-worse on its own states at both seeds, so T-STABLE does not hold either. "
                      f"The card names no step for MIXED; the result names it (section Next).", ""]
    lines += ["## Controls (card section Controls, binding)", "", "| control | s0 | s1 | what was read |", "|---|---|---|---|"]
    r0, r1 = S["s0"]["regression_arm"]["slope"], S["s1"]["regression_arm"]["slope"]
    ii = res["identity_vs_banked_informational"]
    cc = lambda k, s: yn(C[k][s]["pass"])
    v_t0 = C["C2_twin_tick0_detail"]["s0"]["C0"]["values"]
    v_tl = C["C2b_twin_is_logged_detail"]["s0"]["C0"]["values"]
    v_rg = [C["C3a_regression_arm_built"][s]["values"] for s in SEEDS]
    v_id = [C["C1_identity"][s]["values"] for s in SEEDS]
    lines += [f"| 1 Identity (G-ID: dump run vs the bare pinned tool, 4 threads, full grid) | {cc('C1_identity', 's0')} | {cc('C1_identity', 's1')} | max abs delta of the executed tracks {v_id[0]['max_abs_delta_tracks']} / {v_id[1]['max_abs_delta_tracks']} (must be exactly 0.0); tick file and {v_id[0]['episode_files']} episode dumps bitwise equal |",
              f"| 2 Twin identity at tick 0 (G-TWIN0; C0 and REG arms) | {cc('C2_twin_tick0', 's0')} | {cc('C2_twin_tick0', 's1')} | {v_t0['tick0_rows']:,} tick-0 rows, swapped inputs and T outputs bitwise equal, max abs d at tick 0 = {max(v_t0['d_tick0_max_abs'])} under all three rank orders (must be exactly 0.0) |",
              f"| 2b Twin is the logged state (G-TWINL, G-OWN; supporting) | {cc('C2b_twin_is_logged', 's0')} | {cc('C2b_twin_is_logged', 's1')} | {v_tl['later_swap_unequal']:,} of {v_tl['later_rows']:,} later rows differ in the swapped inputs; logged p10f mismatch {v_tl['logged_p10f_mismatch']} |",
              f"| 3a Regression arm built as declared (G-REG) | {cc('C3a_regression_arm_built', 's0')} | {cc('C3a_regression_arm_built', 's1')} | T's ego token zeroed on {v_rg[0]['run_t_zeroed_calls']} of {v_rg[0]['expected_calls_own_plus_twin']} T calls (own + twin) at s0, {v_rg[1]['run_t_zeroed_calls']} of {v_rg[1]['expected_calls_own_plus_twin']} at s1; weights restored |",
              f"| 3b Regression arm slope not separated from 0 | {cc('C3b_regression_slope_not_separated', 's0')} | {cc('C3b_regression_slope_not_separated', 's1')} | REG slope {ci(r0)} / {ci(r1)} |",
              f"| integrity (not a numbered control) | {yn(C['integrity']['pass'])} | | frozen prefix and implementation sha hold; {res['n']['replans']:,} replans, {res['n']['episodes']} episodes, B {N_BOOT:,}, the pinned tool sha, 4 threads, in all four reads |", "",
              f"Control 1 is read against the 4-thread bare run of the pinned tool, as the card's Log (09:12, before any statistic) commits. Against the banked 12-thread runs (RIC0) the same comparison reads max abs delta "
              f"{ii['s0']['max_abs_delta_tracks_m']:.4f} m (s0) and {ii['s1']['max_abs_delta_tracks_m']:.4f} m (s1), not 0.0; the Log declares that line informational. "
              f"The card's sentence says 'the tool of record's' without saying which run; the Log fixes it, and it is applied as written.", ""]
    lines += ["### The identity line that read rc 5 at both seeds (informational, decides nothing)", "",
              "| | s0 | s1 |", "|---|---|---|"]
    gv = lambda k, fmt="{}": [fmt.format(ii[s].get(k)) if ii[s].get(k) is not None else "n/a" for s in SEEDS]
    lines += [f"| banked run threads / chain threads | {ii['s0']['banked_threads']} / {ii['s0']['chain_threads']} | {ii['s1']['banked_threads']} / {ii['s1']['chain_threads']} |",
              f"| same checkpoint sha256 / same tool sha256 | {ii['s0']['same_ckpt_sha256']} / {ii['s0']['same_tool_sha256']} | {ii['s1']['same_ckpt_sha256']} / {ii['s1']['same_tool_sha256']} |",
              f"| episode dumps unequal | {ii['s0']['episode_files_unequal']} of {ii['s0']['episode_files']} | {ii['s1']['episode_files_unequal']} of {ii['s1']['episode_files']} |",
              f"| tick-file columns unequal | {len(ii['s0']['tick_keys_unequal'])} (all float) | {len(ii['s1']['tick_keys_unequal'])} (all float) |"]
    if "episode_track_max_abs_delta" in ii["s0"]:
        e0, e1 = ii["s0"]["episode_track_max_abs_delta"], ii["s1"]["episode_track_max_abs_delta"]
        lines += [f"| integer / bool columns unequal | {len(ii['s0']['integer_or_bool_columns_unequal'])} | {len(ii['s1']['integer_or_bool_columns_unequal'])} |",
                  f"| per-episode max track delta: median / p90 / max (m) | {e0['median']:.4f} / {e0['p90']:.4f} / {e0['max']:.4f} | {e1['median']:.4f} / {e1['p90']:.4f} / {e1['max']:.4f} |",
                  f"| own-state rows whose T-lon argmax differs from the banked tick file | {round((1 - ii['s0']['t_lon_argmax_agreement']) * res['n']['replans'])} of {res['n']['replans']:,} | {round((1 - ii['s1']['t_lon_argmax_agreement']) * res['n']['replans'])} of {res['n']['replans']:,} |",
                  f"| T-lon logits max / mean abs difference | {ii['s0']['t_lon_logits_max_abs']:.4f} / {ii['s0']['t_lon_logits_mean_abs']:.6f} | {ii['s1']['t_lon_logits_max_abs']:.4f} / {ii['s1']['t_lon_logits_mean_abs']:.6f} |"]
    lines += ["", ii["reading"], "",
              "## The committed statistics, per seed (C0; own = the car on its own motion, logged = the same replan with the logged ego state)", "",
              "| statistic | s0 | s1 | status s0 / s1 |", "|---|---|---|---|"]
    a, b = S["s0"], S["s1"]
    sl = lambda x: "separated below 0" if x["below0_separated"] else ("separated above 0" if x["separated"] else "not separated")
    lines += [f"| FEEDBACK slope of d on dv (rank units per m/s; rows with abs dv >= 0.5) | {ci(a['slope'])} | {ci(b['slope'])} | {sl(a['slope'])} / {sl(b['slope'])} |",
              f"| rows / episodes in the slope | {a['slope']['n_rows']:,} / {a['slope']['n_episodes']} | {b['slope']['n_rows']:,} / {b['slope']['n_episodes']} | |",
              f"| T-lon macro-F1 own - logged (own, logged) | {ci(a['lon_f1'], 'delta')} ({a['lon_f1']['own']:.4f}, {a['lon_f1']['logged']:.4f}) | {ci(b['lon_f1'], 'delta')} ({b['lon_f1']['own']:.4f}, {b['lon_f1']['logged']:.4f}) | {a['lon_f1']['status']} / {b['lon_f1']['status']} |",
              f"| T-lat macro-F1 own - logged (own, logged) | {ci(a['lat_f1'], 'delta')} ({a['lat_f1']['own']:.4f}, {a['lat_f1']['logged']:.4f}) | {ci(b['lat_f1'], 'delta')} ({b['lat_f1']['own']:.4f}, {b['lat_f1']['logged']:.4f}) | {a['lat_f1']['status']} / {b['lat_f1']['status']} |",
              f"| speed-constraint RMSE own - logged, m/s (own, logged) | {ci(a['cons'], 'delta', 3)} ({a['cons']['own_rmse']:.3f}, {a['cons']['logged_rmse']:.3f}) | {ci(b['cons'], 'delta', 3)} ({b['cons']['own_rmse']:.3f}, {b['cons']['logged_rmse']:.3f}) | {a['cons']['status']} / {b['cons']['status']} |",
              f"| labelled replans in the quality rows (lon / lat) | {a['lon_f1']['n']:,} / {a['lat_f1']['n']:,} | {b['lon_f1']['n']:,} / {b['lat_f1']['n']:,} | |",
              "", "Outcome logic uses the first row and the T-lon row only. The T-lat and constraint rows are reported as the card asks and decide nothing. F1 is higher-is-better (status worse = interval below 0); RMSE is lower-is-better (status worse = interval above 0).", "",
              "## What the read says about T's own decisions on the model's own states", ""]
    says = []
    if a["slope"]["not_separated"] and b["slope"]["not_separated"]:
        says.append(f"T-lon's expected class index is not found to follow the car's speed deficit: the slope of the shift on the deficit is not separated from 0 at either seed ({ci(a['slope'])} and {ci(b['slope'])} rank units per m/s). "
                    f"The intervals allow slopes of about {min(a['slope']['lo'], b['slope']['lo']):+.2f} to {max(a['slope']['hi'], b['slope']['hi']):+.2f} rank units per m/s, i.e. at most about {max(abs(a['slope']['lo']), abs(a['slope']['hi']), abs(b['slope']['lo']), abs(b['slope']['hi'])):.2f} of a class position per m/s of deficit. "
                    f"The mechanism in the form the card committed (a negative slope) is not found at this precision. Both signs of deficit are present: of the slope rows, {a.get('dv_signs_C0', {}).get('n_own_slower_ge_0.5', 0):,} / {b.get('dv_signs_C0', {}).get('n_own_slower_ge_0.5', 0):,} have the car slower than logged and {a.get('dv_signs_C0', {}).get('n_own_faster_ge_0.5', 0):,} / {b.get('dv_signs_C0', {}).get('n_own_faster_ge_0.5', 0):,} faster (s0 / s1).")
    else:
        says.append(f"The slope is {sl(a['slope'])} at s0 ({ci(a['slope'])}) and {sl(b['slope'])} at s1 ({ci(b['slope'])}).")
    if a["lon_f1"]["status"] == "worse" and b["lon_f1"]["status"] == "worse":
        says.append(f"T does worse on the model's own states than on the logged states at both seeds, with the same camera frames and the same route in both calls (only the ego block is swapped): T-lon macro-F1 {a['lon_f1']['own']:.3f} against {a['lon_f1']['logged']:.3f} (s0) and {b['lon_f1']['own']:.3f} against {b['lon_f1']['logged']:.3f} (s1); "
                    f"T-lat {a['lat_f1']['own']:.3f} against {a['lat_f1']['logged']:.3f} and {b['lat_f1']['own']:.3f} against {b['lat_f1']['logged']:.3f}; speed-constraint RMSE {a['cons']['own_rmse']:.2f} against {a['cons']['logged_rmse']:.2f} and {b['cons']['own_rmse']:.2f} against {b['cons']['logged_rmse']:.2f} m/s.")
        sa, sb = a["beside"]["share_argmax_slower_than_gt"], b["beside"]["share_argmax_slower_than_gt"]
        says.append(f"Beside, deciding nothing: the share of replans whose lon argmax is a slower class than the label's is {100 * sa['own']:.1f} % on own states against {100 * sa['logged']:.1f} % on logged (s0) and {100 * sb['own']:.1f} % against {100 * sb['logged']:.1f} % (s1).")
    elif a["lon_f1"]["status"] != "worse" and b["lon_f1"]["status"] != "worse":
        says.append("T-lon macro-F1 on own states is not separated-worse than on logged states at either seed.")
    else:
        says.append(f"T-lon macro-F1 own - logged is {a['lon_f1']['status']} at s0 and {b['lon_f1']['status']} at s1.")
    if a["slope"]["not_separated"] and b["slope"]["not_separated"] and a["lon_f1"]["status"] == "worse" and b["lon_f1"]["status"] == "worse":
        says.append("The two findings sit side by side: the degradation of T on its own states is measured at both seeds, and it is not carried by the linear speed-deficit channel the card tests.")
    lines += [f"* {x}" for x in says]
    lines += ["", "## What it does not say", "",
              "* It does not say T is the crawl source (T-FEEDBACK needs the slope) and it does not say T is cleared (T-STABLE needs T-lon F1 not worse on own states). The card's two clean outcomes are both unmet.",
              "* One linear slope pooled over both signs of deficit and over all classes. A threshold, a class-specific or a nonlinear dependence on speed is not read.",
              "* F1 and RMSE are scored against the label at the logged time (ego and future may be used for labels). On own states the logged driver's label is not necessarily the right target, so own - logged mixes degraded decisions with a target that fits the own state less well. It does not say which.",
              "* The twin swaps the ego block only (`p10f, base, pose_now, pose_t, pose_s, limit, limit_ok`); frames, route countdown and hold ages are the same tensors. The measured drop is therefore a drop from the ego block, not from the images. "
              f"HYPOTHESIS, untested here: T-lat loses about as much F1 as T-lon (T-lat {a['lat_f1']['delta']:+.3f} / {b['lat_f1']['delta']:+.3f}, T-lon {a['lon_f1']['delta']:+.3f} / {b['lon_f1']['delta']:+.3f}) and lat has no speed-deficit channel, so part of the drop may be a general own-state shift rather than a speed effect.",
              "* No crawl share, along-track or any driving metric of these rollouts is read, and no T-to-O link is measured: this read stops at T's outputs.",
              "* The estimator answers another draw of EPISODES. Training-run variance is read by the two seeds side by side (the outcome pattern is the same at both); the rollout is reproducible bit for bit at a fixed thread count (control 1: two separate processes, max abs delta 0.0), so there is no inference-seed variance to read, but its bits depend on the thread count (see the identity line above).",
              "* STRATEGIC family: PAUSED by PI ruling (item 25). The S layer is not read here; this is not a pass.", ""]
    ra, rb = S["s0"]["regression_arm"], S["s1"]["regression_arm"]
    bs = lambda k: f"{ci(a['beside'][k])} ({sl(a['beside'][k])}) | {ci(b['beside'][k])} ({sl(b['beside'][k])})"
    lines += ["## Beside (not committed, decides nothing)", "",
              "| | s0 | s1 |", "|---|---|---|",
              f"| slope, through origin | {bs('slope_through_origin')} |",
              f"| slope, deficit read at T's frame (j - age) | {bs('slope_dv_at_T_frame')} |",
              f"| slope, FOLLOW / KEEP swapped | {bs('slope_follow_keep_swapped')} |",
              f"| slope, three-group order (slow / middle / fast) | {bs('slope_three_group')} |",
              f"| T-lon F1 own - logged, ticks 1-2 only | {ci(a['beside']['ticks_1_2_lon_f1'], 'delta')} | {ci(b['beside']['ticks_1_2_lon_f1'], 'delta')} |",
              f"| T-lat F1 own - logged, ticks 1-2 only | {ci(a['beside']['ticks_1_2_lat_f1'], 'delta')} | {ci(b['beside']['ticks_1_2_lat_f1'], 'delta')} |",
              f"| share lon argmax slower than label: own / logged, delta [CI] | {100 * a['beside']['share_argmax_slower_than_gt']['own']:.1f} % / {100 * a['beside']['share_argmax_slower_than_gt']['logged']:.1f} %, {ci(a['beside']['share_argmax_slower_than_gt'], 'delta')} | {100 * b['beside']['share_argmax_slower_than_gt']['own']:.1f} % / {100 * b['beside']['share_argmax_slower_than_gt']['logged']:.1f} %, {ci(b['beside']['share_argmax_slower_than_gt'], 'delta')} |",
              f"| constraint RMSE, 26 cells: worse / better / n.s. | {a['beside']['constraint_rmse_26_cells']['status_counts']['worse']} / {a['beside']['constraint_rmse_26_cells']['status_counts']['better']} / {a['beside']['constraint_rmse_26_cells']['status_counts']['n.s.']} | {b['beside']['constraint_rmse_26_cells']['status_counts']['worse']} / {b['beside']['constraint_rmse_26_cells']['status_counts']['better']} / {b['beside']['constraint_rmse_26_cells']['status_counts']['n.s.']} |",
              f"| mean deficit dv (m/s); share of replans with abs dv >= 0.5 | {a['deficit']['dv_mean']:+.3f}; {a['deficit']['share_abs_dv_ge_0.5']:.3f} | {b['deficit']['dv_mean']:+.3f}; {b['deficit']['share_abs_dv_ge_0.5']:.3f} |",
              f"| mean shift d over all replans (rank units) | {a['d_mean_all']:+.4f} | {b['d_mean_all']:+.4f} |", "",
              *([f"The three-group variant of the order reads separated ABOVE 0 at both seeds, the opposite sign to the hypothesis. It is one of three orderings the card lists as beside; it was not committed, the committed ordering reads not separated, and nothing is promoted from it."]
                if a["beside"]["slope_three_group"]["sign_if_separated"] == "above0" and b["beside"]["slope_three_group"]["sign_if_separated"] == "above0" else []), "",
              "### The regression arm (T's ego token zeroed inside T), beside the C0 rows", "",
              "| | s0 | s1 |", "|---|---|---|",
              f"| REG slope (committed form) [CI]; half-width | {ci(ra['slope'])}; {ra['slope']['half_width']:.4f} | {ci(rb['slope'])}; {rb['slope']['half_width']:.4f} |",
              f"| C0 slope half-width | {a['slope']['half_width']:.4f} | {b['slope']['half_width']:.4f} |",
              f"| REG slope through origin [CI] | {ci(ra['slope_through_origin'])} | {ci(rb['slope_through_origin'])} |",
              f"| T-lon F1 on logged states: REG against C0 | {ra['lon_f1']['logged']:.3f} against {a['lon_f1']['logged']:.3f} | {rb['lon_f1']['logged']:.3f} against {b['lon_f1']['logged']:.3f} |",
              f"| T-lon F1 own - logged, REG | {ci(ra['lon_f1'], 'delta')} ({ra['lon_f1']['status']}) | {ci(rb['lon_f1'], 'delta')} ({rb['lon_f1']['status']}) |",
              f"| mean deficit dv (m/s): REG against C0 | {ra['dv_mean']:+.3f} against {a['deficit']['dv_mean']:+.3f} | {rb['dv_mean']:+.3f} against {b['deficit']['dv_mean']:+.3f} |",
              f"| mean deficit dv at tick 2 (m/s): REG against C0 | {ra['dv_mean_tick2']:+.3f} against {a['deficit']['dv_mean_tick2']:+.3f} | {rb['dv_mean_tick2']:+.3f} against {b['deficit']['dv_mean_tick2']:+.3f} |", "",
              f"Control 3b passes with the committed with-intercept slope at both seeds; at s1 the upper bound is {rb['slope']['hi']:+.4f}, a narrow margin, and the through-origin form (beside) is "
              f"{'separated below 0' if rb['slope_through_origin']['below0_separated'] else 'not separated'} at s1 and {'separated below 0' if ra['slope_through_origin']['below0_separated'] else 'not separated'} at s0. The control as committed is the with-intercept slope. "
              f"The REG interval is about {a['slope']['half_width'] / max(ra['slope']['half_width'], 1e-12):.0f}x / {b['slope']['half_width'] / max(rb['slope']['half_width'], 1e-12):.0f}x narrower than the C0 interval, so the probe resolves slopes of about {ra['slope']['half_width']:.2f} when T has no ego token; "
              f"the C0 intervals ({a['slope']['half_width']:.2f} / {b['slope']['half_width']:.2f}) are the resolution that applies to the C0 result.",
              "",
              f"Observed beside, not pre-registered and not a result: with T's ego token zeroed the rollouts {'run slower than the logged state' if ra['dv_mean'] > 0 and rb['dv_mean'] > 0 else 'differ from the logged state'} (mean deficit {ra['dv_mean']:+.2f} / {rb['dv_mean']:+.2f} m/s against {a['deficit']['dv_mean']:+.2f} / {b['deficit']['dv_mean']:+.2f} in C0) and T-lon F1 on logged states falls to {ra['lon_f1']['logged']:.3f} / {rb['lon_f1']['logged']:.3f} from {a['lon_f1']['logged']:.3f} / {b['lon_f1']['logged']:.3f}. "
              "A path from T to the car's speed exists in this model. The arm also leaves ego routes into T open (S's checkpoints, the speed-limit bucket, the route countdown), as IMPL_CHOICES.md section 5 discloses.", ""]
    nx = res["next"]
    lines += ["## Next (Rule Zero: the read does not end at its bar)", "", f"Named by: {nx['named_by']}.", "", nx["text"], ""]
    for st in nx.get("steps", []):
        lines += [st, ""]
    lines += ["", "## Families", "", *[f"* {k}: {v}" for k, v in res["families"].items()], "",
              "## Artifacts", "",
              "* Inputs (sha256 recorded in `RESULT_TOWN.json`): `raw/READ_{C0,REG}_s{0,1}.json`, `raw/GATE_ID_C0_s{0,1}.json`, `raw/GATE_TWIN_{C0,REG}_s{0,1}.json`, `raw/GATE_ID_INFO_C0_s{0,1}_vs_banked_RIC0_s{0,1}.json`, `raw/receipts/` (copies of the run receipts and done markers), `raw/INFO_DRIFT_vs_banked.json`, `raw/INFO_DV_SIGNS.json`.",
              "* Rollouts (not copied): `F:/TanitAD-scratch/motion_c/town/out/{BARE,C0,REG}_s{0,1}`. Banked runs compared: `F:/TanitAD-scratch/motion_c/ri/out/RIC0_s{0,1}`.",
              "* Generated by `town_result.py`; the deliberate-regression tests are `tests/test_town_result.py`.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args(argv)
    I = load_inputs(a.raw)
    res = evaluate(I)
    json.dump(res, open(f"{a.out_dir}/RESULT_TOWN.json", "w", encoding="utf-8", newline="\n"), indent=1)
    open(f"{a.out_dir}/RESULT_TOWN.md", "w", encoding="utf-8", newline="\n").write(render_md(res) + "\n")
    print("wrote RESULT_TOWN.json / .md:", res["outcome"], "| controls all pass:", res["controls"]["all_pass"], "| failed:", [k for k, v in res["controls"]["pass_by_control"].items() if not v])
    return 0


if __name__ == "__main__":
    sys.exit(main())
