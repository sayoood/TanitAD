"""Official NavSim v2 two-stage CSVs -> the ``summary.json`` blocks of one run.

PROMOTED (not rewritten) from
* E2 ``code/parse_scores.py``: ``SUB`` and ``s2_group_uniform`` VERBATIM (AST-pinned), and its
  paired per-scene W/T/L logic (1e-12 tie band) and refusal of a ``pdm_score`` column;
* E1 ``code/verify_controls.py``: ``epdms_formula`` and ``c4`` VERBATIM (AST-pinned) — the
  per-token EPDMS identity from ``docs/metrics.md`` — and the C5 aggregate recompute (restructured
  into :func:`c5_aggregate`, same arithmetic).

⛔ THE HEADLINE IS READ FROM THE ``score`` COLUMN OF ROW ``extended_pdm_score_combined`` — NEVER
``pdm_score`` (``PDMScorer._aggregate_pdm_scores`` masks EC out and divides by 14; the EPDMS is
assembled downstream in ``run_pdm_score.py::compute_final_scores``). It is parsed with Python's
``float()`` from the devkit's own text cell, so the value is the file's value bit-for-bit.

⚠️ AND THAT IS NOT PEDANTRY: MEASURED 2026-09-20 on E2's banked A1 CSV — the text cell reads
``0.21845081236026753`` and E2's ``scores_summary.json`` (which read it through pandas' DEFAULT csv
float parser) records ``0.2184508123602675``, ONE ULP lower. Derived statistics here still go
through pandas, EXACTLY as E2 computed them (so their numbers reproduce); a HEADLINE never does.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

HEADLINE_COLUMN = "score"
FORBIDDEN_COLUMN = "pdm_score"
HEADLINE_ROW = "extended_pdm_score_combined"
STAGE_ROWS = {"stage_one": "extended_pdm_score_stage_one", "stage_two": "extended_pdm_score_stage_two"}
TIE_EPS = 1e-12

# ---- E1 verify_controls.py constants (verbatim) ---------------------------- #
M8 = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
      "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
      "history_comfort"]
EC = "two_frame_extended_comfort"
SHORT = dict(zip(M8 + [EC], ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]))
W = {"ego_progress": 5.0, "time_to_collision_within_bound": 5.0, "lane_keeping": 2.0, "history_comfort": 2.0, EC: 2.0}
SUMMARY = ["extended_pdm_score_stage_one", "extended_pdm_score_stage_two", "extended_pdm_score_combined"]
SIGMA2 = 0.1                                               # scene_aggregator.py:18

# ---- E2 parse_scores.py constant (verbatim) --------------------------------- #
SUB = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
       "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
       "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
       "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}


class ColumnTrapError(ValueError):
    """⛔ ``pdm_score`` was offered as, or instead of, the EPDMS."""


# --------------------------------------------------------------------------- #
# VERBATIM promotions (pinned by test_bench_suite_promotion.py)                #
# --------------------------------------------------------------------------- #
def epdms_formula(r: dict, suffix: str = "") -> float:
    """docs/metrics.md: prod(NC,DAC,DDC,TLC) * sum(w*m)/sum(w); EC dropped (/14) when NaN
    (run_pdm_score_one_stage.py:177-189 zeroes EC's weight for a row with no adjacent frame)."""
    g = lambda k: float(r[k + suffix])                                    # noqa: E731
    prod = g("no_at_fault_collisions") * g("drivable_area_compliance") * g("driving_direction_compliance") * g("traffic_light_compliance")
    num = den = 0.0
    for k, w in W.items():
        v = g(k)
        if k == EC and math.isnan(v):
            continue
        num += w * v
        den += w
    return prod * num / den


def c4(df: pd.DataFrame, two_stage: bool) -> dict:
    rows = df[~df["token"].isin(SUMMARY + ["average_all_frames"]) & (df["valid"].astype(str) == "True")]
    worst, n, n_nan_ec = 0.0, 0, 0
    for _, r in rows.iterrows():
        if two_stage:
            suffix = "_stage_one" if not pd.isna(r.get("ego_progress_stage_one")) else "_stage_two"
        else:
            suffix = ""
        n_nan_ec += int(pd.isna(r[EC + suffix]))
        d = abs(epdms_formula(r, suffix) - float(r["score"]))
        worst, n = max(worst, d), n + 1
    return {"n_rows": n, "n_rows_with_nan_EC_dropped_to_14": n_nan_ec, "max_abs_diff": worst,
            "tol": 1e-9, "pass": bool(n > 0 and worst <= 1e-9)}


def s2_group_uniform(df: pd.DataFrame, mapping: list, col: str = "score") -> dict:
    s2 = df[df.stage == 2].set_index("token")[col]
    gm, sizes = [], []
    for orig, prev, pairs in mapping:
        for grp in ([p[0] for p in pairs], [p[1] for p in pairs]):
            v = s2.reindex(grp)
            if v.isna().any():
                return {"status": "UNAVAILABLE", "reason": f"{int(v.isna().sum())} group tokens "
                        f"missing/NaN", "n": int(v.notna().sum())}
            gm.append(float(v.mean()))
            sizes.append(len(grp))
    return {"value": float(np.mean(gm)), "n_groups": len(gm), "group_sizes": sizes,
            "n_scenes": int(sum(sizes))}


# --------------------------------------------------------------------------- #
# reading                                                                      #
# --------------------------------------------------------------------------- #
def read_raw_rows(path) -> tuple:
    """``(header, {token: {col: raw text}})`` — the devkit CSV as TEXT, file order preserved."""
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        header = list(rd.fieldnames or [])
        rows = {}
        for r in rd:
            rows[r["token"]] = r
    return header, rows


def headline_value(raw_rows: dict, header: list, *, column: str = None, row: str = HEADLINE_ROW) -> float:
    """⛔ The headline, from the ``score`` column only. Refuses ``pdm_score`` by name."""
    column = HEADLINE_COLUMN if column is None else column
    if column == FORBIDDEN_COLUMN:
        raise ColumnTrapError("pdm_score is NEVER the EPDMS headline (EC masked out, /14) — read `score`")
    if column not in header:
        if FORBIDDEN_COLUMN in header:
            raise ColumnTrapError(f"the CSV offers only {FORBIDDEN_COLUMN!r} and no {column!r}; refusing to substitute")
        raise KeyError(f"column {column!r} absent from the devkit CSV (header {header[:6]}…)")
    if row not in raw_rows:
        raise KeyError(f"summary row {row!r} absent from the devkit CSV")
    return float(raw_rows[row][column])


def load_scores(path, stage_of: dict) -> pd.DataFrame:
    """E2 ``parse_scores.load_arm`` (glue restructured: path + stage map as arguments): the token rows
    with short per-stage metric columns. Refuses a CSV carrying ``pdm_score`` (E2's rule)."""
    df = pd.read_csv(path)
    if FORBIDDEN_COLUMN in df.columns:
        raise ColumnTrapError(f"{path}: carries a pdm_score column — refusing (read `score`)")
    tok = df[~df.token.str.startswith("extended_pdm_score")].copy()
    tok["stage"] = tok.token.map(stage_of)
    if tok.stage.isna().any():
        raise ValueError(f"{path}: {int(tok.stage.isna().sum())} tokens not in the split's stage sets")
    for k, c in SUB.items():
        c1, c2 = f"{c}_stage_one", f"{c}_stage_two"
        tok[k] = np.where(tok.stage == 1, tok.get(c1), tok.get(c2))
    return tok


# --------------------------------------------------------------------------- #
# controls                                                                     #
# --------------------------------------------------------------------------- #
def c5_aggregate(csv_df: pd.DataFrame, frame_dump: pd.DataFrame, mapping: list) -> dict:
    """E1 C5, restructured into a function (same arithmetic): recompute the three official summary
    rows from the per-token frame (``compute_final_scores`` dump: weight, endpoints) with Gaussian
    weights (sigma^2 = 0.1, scene_aggregator.py:18) recomputed from the endpoints."""
    F = frame_dump.set_index("token")
    A1 = csv_df
    cols = M8 + [EC, "score"]
    wmax, s1_rows, s2_groups, comb = 0.0, [], [], []
    for orig, prev, pairs in mapping:
        nows, prevs = [p[0] for p in pairs], [p[1] for p in pairs]

        def wts(first, toks):
            d2 = [(F.loc[first, "endpoint_x"] - F.loc[t, "start_point_x"]) ** 2 +
                  (F.loc[first, "endpoint_y"] - F.loc[t, "start_point_y"]) ** 2 for t in toks]
            w = np.exp(-np.asarray(d2) / (2 * SIGMA2))
            return np.full(len(toks), 1.0 / len(toks)) if (np.isclose(w.sum(), 0.0) or np.isnan(w.sum())) else w / w.sum()
        w_now, w_prev = wts(orig, nows), wts(prev, prevs)
        wmax = max(wmax, float(np.abs(w_now - F.loc[nows, "weight"].to_numpy()).max()),
                   float(np.abs(w_prev - F.loc[prevs, "weight"].to_numpy()).max()))
        g1s2 = (F.loc[nows, cols].to_numpy() * w_now[:, None]).sum(0)
        g2s2 = (F.loc[prevs, cols].to_numpy() * w_prev[:, None]).sum(0)
        g1s1, g2s1 = F.loc[orig, cols].to_numpy(dtype=float), F.loc[prev, cols].to_numpy(dtype=float)
        s1_rows += [g1s1, g2s1]
        s2_groups += [g1s2, g2s2]
        comb.append((g1s1 * g1s2 + g2s1 * g2s2) / 2)
    mine = {"stage_one": np.mean(s1_rows, 0), "stage_two": np.mean(s2_groups, 0), "combined": np.mean(comb, 0)}
    diffs = {}
    for key, tok, suf in (("stage_one", SUMMARY[0], "_stage_one"), ("stage_two", SUMMARY[1], "_stage_two")):
        row = A1[A1["token"] == tok].iloc[0]
        for i, cname in enumerate(cols):
            col = "score" if cname == "score" else cname + suf
            diffs[f"{key}.{SHORT.get(cname, cname)}"] = abs(float(row[col]) - float(mine[key][i]))
    diffs["combined.score"] = abs(float(A1[A1["token"] == SUMMARY[2]].iloc[0]["score"]) - float(mine["combined"][-1]))
    worst = max(diffs.values())
    return {"max_abs_weight_diff": wmax, "max_abs_summary_diff": worst, "tol": 1e-9,
            "pass": bool(wmax <= 1e-9 and worst <= 1e-9),
            "promoted_from": "E1 code/verify_controls.py C5 (inline) -> function, same arithmetic"}


def compare_to_reference(csv_path, ref_path) -> dict:
    """Cell-level + file-level identity of a scored CSV against a banked reference CSV (ACC-4/5)."""
    import hashlib
    out = {"reference": str(ref_path).replace("\\", "/")}
    a_bytes, b_bytes = Path(csv_path).read_bytes(), Path(ref_path).read_bytes()
    out["md5"] = hashlib.md5(a_bytes).hexdigest()
    out["reference_md5"] = hashlib.md5(b_bytes).hexdigest()
    out["file_identical"] = a_bytes == b_bytes
    ha, ra = read_raw_rows(csv_path)
    hb, rb = read_raw_rows(ref_path)
    out["same_header"] = ha == hb
    out["same_token_set"] = set(ra) == set(rb)
    out["same_row_order"] = list(ra) == list(rb)
    worst, n_cells, nan_mismatch, first_diff = 0.0, 0, 0, None
    for t in sorted(set(ra) & set(rb)):
        for c in ha:
            if c in ("", "token", "valid") or c not in rb[t]:
                continue
            x, y = ra[t][c], rb[t][c]
            fx = float(x) if x not in ("", None) else float("nan")
            fy = float(y) if y not in ("", None) else float("nan")
            n_cells += 1
            if math.isnan(fx) or math.isnan(fy):
                if math.isnan(fx) != math.isnan(fy):
                    nan_mismatch += 1
                    first_diff = first_diff or {"token": t, "column": c, "ours": x, "ref": y}
                continue
            d = abs(fx - fy)
            if d > worst:
                worst = d
                first_diff = {"token": t, "column": c, "ours": x, "ref": y, "abs_diff": d}
        if ra[t].get("valid") != rb[t].get("valid"):
            nan_mismatch += 1
    out.update({"n_numeric_cells": n_cells, "max_abs_diff": worst, "nan_pattern_mismatch": nan_mismatch,
                "first_or_worst_difference": first_diff,
                "cells_identical": bool(out["same_token_set"] and worst == 0.0 and nan_mismatch == 0)})
    return out


# --------------------------------------------------------------------------- #
# the per-arm blocks                                                           #
# --------------------------------------------------------------------------- #
def _stage_block(raw_rows: dict, row: str, suffix: str, n: int) -> dict:
    r = raw_rows.get(row)
    if r is None:
        return {"status": "UNAVAILABLE", "reason": f"summary row {row} absent", "n": int(n)}
    blk = {"score": float(r[HEADLINE_COLUMN]), "column": HEADLINE_COLUMN, "row": row, "n": int(n)}
    blk["submetrics"] = {s: (float(r[c + suffix]) if r.get(c + suffix) not in (None, "") else None)
                         for s, c in SUB.items()}
    return blk


def per_log_block(tok: pd.DataFrame, log_of: dict) -> dict:
    if not log_of:
        return {"status": "UNAVAILABLE", "reason": "no token -> log_name map (wrapper hooks absent)", "n": 0}
    t = tok.assign(log=tok.token.map(log_of))
    if t.log.isna().any():
        return {"status": "UNAVAILABLE", "reason": f"{int(t.log.isna().sum())} tokens without a log_name", "n": 0}
    out = {"_statistic": ("PLAIN mean of the official per-token `score` rows, by nuPlan log and stage — NOT the "
                          "official aggregation (which weights stage-2 rows by the stage-1 endpoint kernel)")}
    for ln, g in t.groupby("log"):
        out[str(ln)] = {st_name: {"n": int((g.stage == st).sum()),
                                  "score_mean": (float(g[g.stage == st].score.mean()) if (g.stage == st).any() else None)}
                        for st_name, st in (("stage_one", 1), ("stage_two", 2))}
    return out


def paired_block(a_tok: pd.DataFrame, b_tok: pd.DataFrame, a_head, b_head, a_raw: dict, b_raw: dict,
                 a_u, b_u) -> dict:
    """Arm vs floor, on the identical tokens: aggregate deltas + per-scene W/T/L (E2's pairs)."""
    out = {"status": "OK", "headline_delta": (None if a_head is None or b_head is None else a_head - b_head),
           "S2_EPDMS_u_delta": (None if a_u is None or b_u is None else a_u - b_u), "by_stage": {}}
    n_common = 0
    for st_name, st in (("stage_one", 1), ("stage_two", 2)):
        a = a_tok[a_tok.stage == st].set_index("token")
        b = b_tok[b_tok.stage == st].set_index("token")
        common = sorted(set(a.index) & set(b.index))
        n_common += len(common)
        if not common:
            out["by_stage"][st_name] = {"status": "UNAVAILABLE", "reason": "no common tokens", "n": 0}
            continue
        d = (a.loc[common, "score"] - b.loc[common, "score"]).to_numpy(dtype=float)
        ra, rb = a_raw.get(STAGE_ROWS[st_name]), b_raw.get(STAGE_ROWS[st_name])
        out["by_stage"][st_name] = {
            "n": len(common), "score_mean_delta": float(d.mean()),
            "official_stage_row_delta": (None if ra is None or rb is None else
                                         float(ra[HEADLINE_COLUMN]) - float(rb[HEADLINE_COLUMN])),
            "wins": int((d > TIE_EPS).sum()), "ties": int((np.abs(d) <= TIE_EPS).sum()),
            "losses": int((d < -TIE_EPS).sum()),
            "submetric_mean_deltas": {k: float(pd.to_numeric(a.loc[common, k]).mean()
                                               - pd.to_numeric(b.loc[common, k]).mean()) for k in SUB}}
    # pooled per-scene counts over BOTH stages — what a leaderboard row quotes; the per-stage split
    # (where the meaning is: the stages have different token sets and different statistics) stays in
    # by_stage. Requested by W5's report, 2026-09-20.
    wtl = [b_ for b_ in out["by_stage"].values() if "wins" in b_]
    out["wins"] = int(sum(b_["wins"] for b_ in wtl))
    out["ties"] = int(sum(b_["ties"] for b_ in wtl))
    out["losses"] = int(sum(b_["losses"] for b_ in wtl))
    out["_wtl_scope"] = ("per-scene wins/ties/losses pooled over the common tokens of BOTH stages "
                         "(tie band 1e-12); per stage in by_stage.<stage>")
    out["n_common"] = int(n_common)
    return out


#: the BINDING terms per family (Sayed 2026-08-02): a missing one makes the family PARTIAL with its
#: reason — it is a WORK ITEM, never a silent omission.
FAMILY_REQUIRED = {
    "longitudinal": ("speed_mae_mps", "along_mae_m"),
    "lateral": ("cross_mae_m", "heading_mae_deg", "curvature_mae_1pm", "yaw_rate_mae_degps"),
}
#: reported when present; their absence is not a gap in the binding rule
FAMILY_EXTRA = {
    "longitudinal": ("speed_bias_mps", "along_bias_m", "accel_mae_mps2"),
    "lateral": ("cross_bias_m",),
}
FAMILY_KEYS = {k: FAMILY_REQUIRED.get(k, ()) + FAMILY_EXTRA.get(k, ()) for k in FAMILY_REQUIRED}


def families_from_artifact(art: dict, scope: str) -> dict:
    """The artifact's ``four_families`` (OUR instruments) -> per-family status + n + key metrics, or a
    refusal with reason + n. Never silently absent."""
    ff = (art or {}).get("four_families") or {}
    out = {}
    for fam in ("longitudinal", "lateral", "tactical", "strategic"):
        b = ff.get(fam)
        if not isinstance(b, dict):
            out[fam] = {"status": "UNAVAILABLE", "reason": "family block absent from the artifact", "n": 0}
            continue
        st = str(b.get("status", "OK")).upper()
        n = int(b.get("n_windows", b.get("n", 0)) or 0)
        if st not in ("OK", "PARTIAL"):
            reason = str(b.get("reason") or "UNAVAILABLE without a reason (adapter)")
            if b.get("navsim_specific_reason"):
                reason += " | NavSim: " + str(b["navsim_specific_reason"])
            out[fam] = {"status": "UNAVAILABLE", "reason": reason, "n": int(b.get("n", 0) or 0)}
            continue
        blk = {"status": "OK", "n": n, "scope": scope, "tier": b.get("tier"), "metrics": {}}
        undefined = []
        for k in FAMILY_KEYS.get(fam, ()):
            if isinstance(b.get(k), (int, float)) and not isinstance(b.get(k), bool):
                blk["metrics"][k] = b[k]
            elif k in FAMILY_REQUIRED.get(fam, ()):        # only a BINDING term makes the family PARTIAL
                v = b.get(k)
                undefined.append(f"{k}: " + (str(v.get("reason"))[:300] if isinstance(v, dict) else "absent/None"))
        if undefined:
            blk["status"] = "PARTIAL"
            blk["reason"] = "undefined or refused terms — " + " | ".join(undefined)
        if fam == "longitudinal":
            ep = b.get("ego_progress")
            if isinstance(ep, dict) and isinstance(ep.get("progress_ratio_mean"), (int, float)):
                blk["metrics"]["progress_ratio_mean"] = ep["progress_ratio_mean"]
            dk = b.get("distance_keeping")
            if not (isinstance(dk, dict) and str(dk.get("status", "")).upper() == "OK"):
                blk["status"] = "PARTIAL"
                blk["reason"] = ((blk.get("reason", "") + " | ") if blk.get("reason") else "") + (
                    "distance-keeping UNAVAILABLE: " + str((dk or {}).get("reason", "no lead block"))[:400])
        if fam == "tactical":
            for sub in ("lateral_decision", "longitudinal_decision"):
                s = b.get(sub)
                if isinstance(s, dict):
                    blk["metrics"][f"{sub}.accuracy"] = s.get("accuracy")
                    blk["metrics"][f"{sub}.kappa"] = s.get("kappa")
            gs = b.get("goal_setting")
            if isinstance(gs, dict) and isinstance(gs.get("goal_point_error_m"), (int, float)):
                blk["metrics"]["goal_setting.goal_point_error_m"] = gs["goal_point_error_m"]
            blk["_is_not"] = "trajectory-derived EXECUTED manoeuvres on both sides — NOT selected-vs-executed"
        out[fam] = blk
    return out


def families_refused(reason: str, n: int = 0) -> dict:
    return {f: {"status": "UNAVAILABLE", "reason": reason, "n": int(n)}
            for f in ("longitudinal", "lateral", "tactical", "strategic")}


# --------------------------------------------------------------------------- #
# STAGE-1 REFERENCE CHECK (navhard) -- E1 relay 2026-09-20                     #
# --------------------------------------------------------------------------- #
#: the 8 sub-metrics the two-stage EPDMS uses. EC is EXCLUDED: it is MASKED in the two-stage
#: score, so it is not part of this comparison (the paper still prints it; we do not check it).
STAGE1_SUB = ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC")


def _trunc1(x: float) -> float:
    """Truncate to 1 dp -- NEVER round. Decimal, not ``int(x*10)/10``: binary float would turn
    78.67 into 786.6999... and truncate to 78.6 by luck rather than by rule, and a value that
    lands just BELOW its decimal would truncate one step too far."""
    from decimal import Decimal, ROUND_DOWN
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_DOWN))


def stage1_submetrics(tok) -> dict:
    """Mean of each stage-1 sub-metric over the stage-1 tokens, x100 (the scale the paper prints)."""
    s1 = tok[tok.stage == 1]
    out = {"n": int(len(s1))}
    out["values_x100"] = {k: (None if k not in s1 or s1[k].isna().all() else float(s1[k].mean()) * 100.0)
                          for k in STAGE1_SUB}
    return out


def stage1_reference_check(measured_x100: dict, ref: dict, n: int | None = None) -> dict:
    """Our stage-1 sub-metrics against (a) E1's MEASURED run -- an IDENTITY check, same split, same
    scorer, same 450 scenes -- and (b) the PUBLISHED navhard leaderboard, under TRUNCATION.

    ⚠️ The truncation rule is not a stylistic choice: MEASURED 2026-09-20 against the banked PDF
    (library key 2506.04218, p.8 Table 2, column 'CV [8]'), TRUNCATING matches 8/8 and ROUNDING only
    4/8. A reproduction rule written around rounding marks a CORRECT reproduction as a MISS on half
    the metrics -- and the 4 it fails are exactly the cells whose 2nd decimal is >= 5.
    """
    e1 = (ref or {}).get("measured_e1") or {}
    pub = (ref or {}).get("published_n2") or {}
    if not e1:
        return {"status": "UNAVAILABLE", "reason": "no banked stage-1 reference for this split/arm"}
    vs_e1, vs_pub, n_trunc, n_round = {}, {}, 0, 0
    for k in STAGE1_SUB:
        ours = measured_x100.get(k)
        if ours is None:
            vs_e1[k] = {"status": "UNAVAILABLE", "reason": "not scored"}
            continue
        # (a) identity against E1 -- to the precision E1 banked (2 dp)
        vs_e1[k] = {"ours_x100": ours, "e1_x100": e1.get(k), "equal_2dp": round(ours, 2) == e1.get(k)}
        # (b) the weaker, INDEPENDENT check against the published column
        if k in pub:
            t, r = _trunc1(ours), round(ours, 1)
            n_trunc += int(t == pub[k]); n_round += int(r == pub[k])
            vs_pub[k] = {"ours_truncated_1dp": t, "ours_rounded_1dp": r, "published": pub[k],
                         "match_truncating": t == pub[k], "match_rounding": r == pub[k]}
    all_eq = all(v.get("equal_2dp") for v in vs_e1.values() if "equal_2dp" in v)
    return {
        "status": "MATCH" if all_eq else "MISMATCH",
        "n": n if n is not None else ref.get("n"),
        "expected_n": ref.get("n"),
        "vs_e1_measured": vs_e1,
        "identical_to_e1": all_eq,
        "vs_published": vs_pub,
        "n_match_truncating": n_trunc, "n_match_rounding": n_round, "n_compared": len(vs_pub),
        "comparison_rule": "TRUNCATE to 1 dp, never round",
        "published_source": ref.get("published_source"),
        "how_to_read": ("identical_to_e1 FALSE is the first thing to chase, ahead of everything else: same "
                        "split, same scorer, same scenes must give the same number. The published comparison "
                        "is the independent, weaker check."),
    }
