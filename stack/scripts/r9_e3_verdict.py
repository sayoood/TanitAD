#!/usr/bin/env python3
"""r9_e3_verdict.py -- the refcv9 E3 FINAL-READ JUDGE (REFCV9_DESIGN section 5, sealed): refcv9 EMA-73,280 vs refcv8 on the HOLD half of eval139.
Pure JSON in, JSON + md out; no GPU, no model.

    python r9_e3_verdict.py --inputs INPUTS.json --out-dir <dir>     # -> E3_VERDICT.json + E3_VERDICT.md  (rc 0 PASS, rc 1 FAIL / INCOMPLETE / VOID, rc 2 REFUSED)

THE BARS LIVE IN ONE TABLE, :data:`BARS`, and nowhere else.  No input can set, move or override a bar: INPUTS.json keys are a closed set (``bars`` /
``thresholds`` are REFUSED), and the judge reads values only from :data:`BARS`.  ``stack/tests/test_r9_e3_verdict.py`` pins the table to the
section-5 text and to ``SEAL_DECISIONS_PRE_DATA_2026-11-08.md`` (incl. the PI amendment: MAP-2 5-10 m and 10-20 m = 0.90, not 0.99, not the
provisional 0.85).

PASS = every bold section-5 bar (MAP-1, MAP-2, MAP-3, BOX-1, BOX-2, BOX-3, FAM-1, FAM-2, W4, W5) + W1, W2, MAP-4, MAP-5.  BOX-4 is "beside", never a bar.
Verdicts: PASS | FAIL (a bar missed, reported as written; the section-5.4 lever is named) | INCOMPLETE (no bar failed but one is NOT-COMPUTABLE: never a pass) |
VOID (K-BASE false: "the bars are void").  K-BASE needs the table sha256 match AND the forward re-fit reproducing it (seal decision: the hash alone does not close it).

FOUR FAMILIES (CLAUDE.md binding): LONGITUDINAL (incl. distance keeping), LATERAL, TACTICAL read row by row; STRATEGIC is PAUSED (PI 2026-09-19, ITEM 25) --
reported PAUSED, never a pass, never omitted, never extended to the other families.  A family that cannot be computed is reported per family with the reason and n.

EVERY NUMBER carries estimator, tier, split, inference seed and the artifact path + sha256 it came from.  REFUSED (rc 2, no verdict) on: a DEV input (or any split that
is not the bar's own), an input with no split / tier stamp, a planner input whose ``infer_seed`` is not the seed slot it sits in, one file used for both seeds, a
paired record with no ``episode_cluster`` estimator or with ``overlapping_holdout`` / ``jackknife`` in it, a declared scored checkpoint that is not EMA-73,280.

SEEDS.  Perception bars (MAP-2..5, BOX-*) are head output / labels: deterministic at inference, read ONCE on HOLD.  Every PLANNER bar (MAP-1, FAM-1, FAM-2, W1, W2, W4,
W5) is read at inference seeds 0 AND 1 and must hold at BOTH (design 5.1: the planner samples; D-REFAV1-SEED-GOAL-MISMATCH); a sub-bar is PASS only if PASS at both, FAIL if
it FAILS at either, else NOT-COMPUTABLE.  The paired interval answers "another draw of EPISODES" only (H-ESTIM-SEED-1): ONE training run per arm, "training variance at
full length not estimated" (design 5.3) is stamped on the verdict.

INPUTS.json (closed key set)::
    {"refcv9_label": "refcv9", "reference_label": "refcv8", "scored_ema_step": 73280, "kbase": p,
     "perception": {"eval_r9": p, "eval_r8": p, "map_paired": p},
     "seeds": {"0": {"s6c_r9": p, "compliance": p, "families": p, "distkeep": p, "w12": p, "camblind": p, "egofloor": p}, "1": {...}}}
  kbase       {"sha256_match": bool, "forward_refit_reproduces": bool|null}                       (null = re-fit not run => K-BASE OPEN => INCOMPLETE)
  eval_r9     ``r9_e1_eval.py --split hold --packs <r9> --packs-b <r8> --rast-dir ... --lift-valid ... --map-beside-d3`` single JSON (split "hold", tier T0): ``boxes.<k>``,
              ``paired_vs_B.boxes.<k>`` (ci.paired_stat_cluster_bootstrap, delta = r9 - r8), ``nearfield.ci.<k>``; ``nearfield.unobs_flags`` {sidewalk, edge} for MAP-5
  eval_r8     same script, refcv8's packs, split hold (reference point values for the 1.5x lane bar)
  map_paired  {"split": "hold", "tier": "T0 ...", "rows": {nf_prec_20_40, lane_tiou05_0_20, drv_iou_<band>: row}}  (r9 - r8)
  s6c_r9      ``s6c_hold.py`` output for refcv9 (``HOLD.subset`` "hold", ``HOLD.rate``, ``HOLD.bootstrap``, ``infer_seed``, ``tier`` T1)
  compliance  {"split": "hold", "tier": T1, "infer_seed": s, "rows": {off_drivable, edge_contact, lane_cross: row}}  (r9 - r8 on S6c-HOLD)
  families    {"split": "all", "tier": T1, "infer_seed": s, "rows": {LONGITUDINAL|LATERAL|TACTICAL: {row_id: row}}}  incl. an ``ade_m`` row (r9 - r8, eval139)
  distkeep    ``r9_e2_distkeep.py`` shape with a HOLD/all stamp: ``rows.LONGITUDINAL.LON_dk_*``
  w12         {"split": "all", ..., "rows": {"FD": row, "MD": row}}  (false decels on GT-cruise windows; missed decels on GT-decel windows; ``lon_indicators``)
  camblind / egofloor   {"split": "all", ..., "rows": {ade_m, LONGITUDINAL, LATERAL, TACTICAL: row}}   (delta = refcv9 - the floor arm)
A ``row`` = a precomputed paired record ``{delta, lo, hi, separated, estimator, ...}`` (``taniteval.ci.paired_*`` output) or the per-window form ``{a, b, eid}`` scored HERE by
``taniteval.ci.paired_episode_cluster_bootstrap`` (B 2000, seed 0).  Directions of rows whose id this table knows come from the TABLE; family rows carry ``better``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SCHEMA = "r9e3-verdict-v1"
SCORED_EMA_STEP = 73280
SEEDS = ("0", "1")
INPUT_KEYS = {"refcv9_label", "reference_label", "scored_ema_step", "kbase", "perception", "seeds"}
PERC_KEYS = {"eval_r9", "eval_r8", "map_paired"}
SEED_KEYS = {"s6c_r9", "compliance", "families", "distkeep", "w12", "camblind", "egofloor"}
FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL")
STRATEGIC_PAUSED = ("PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive with high quality; "
                    "reported PAUSED -- never a pass, never omitted, never extended to the other families. (Design 5.2 FAM-1 also states STRATEGIC n = 0: no route label.)")
CAVEAT = ("H-ESTIM-SEED-1: the paired interval answers another draw of EPISODES, not another TRAINING run; ONE training run per arm -- training variance at full length "
          "not estimated (design 5.3). Perception = T0 label level; planner = T1 self-action open loop on logged eval139 frames (not driving performance).")
DK_ROWS = ("LON_dk_headway_min_m", "LON_dk_time_gap_min_s", "LON_dk_min_ttc_s")             # headway / time-gap / TTC (higher = more distance kept)
MAP4_ROWS = ("drv_iou_0_5", "drv_iou_5_10", "drv_iou_10_20", "drv_iou_20_40")              # UNVERIFIED names: no producer exists yet (open item)

#: kinds: ge / le (point vs value, inclusive) | ge_ref (point >= value x reference point) | sep_better (paired, separated in the good direction) |
#:        not_sep_worse (paired, not separated in the bad direction) | zero (exact identity) | beside (reported, never a bar)
#: ``key`` = the metric key the producer must emit; ``dir`` = the direction in which the paired delta (refcv9 - refcv8) is an improvement.
BARS: dict = {
    "MAP-1": {"required": True, "planner": True, "split": "hold", "tier": "T1", "rule": "planner sidewalk rate on S6c-HOLD <= 9.8 % AND separated below refcv8",
              "lever": "map good / selector ignores it: critic weight + candidate generation filtered by own drv_sdf at train time (5.4 'MAP-2 PASS, MAP-1 FAIL')",
              "subs": [{"id": "rate_le", "kind": "le", "value": 0.098}, {"id": "sep_below_ref", "kind": "sep_better", "key": "off_drivable", "dir": "lower"}]},
    "MAP-2": {"required": True, "planner": False, "split": "hold", "tier": "T0", "rule": "sidewalk-only nf_prec_*: 0-5 m >= 0.60 lift-valid; 5-10 and 10-20 m >= 0.90 (PI 2026-11-08); 20-40 m separated above refcv8",
              "lever": "0-5 m only: T1 temporal (needs D3); 5-40 m: the next Q1-branch lever not yet in the run (5.4)",
              "subs": [{"id": "p_0_5_ge", "kind": "ge", "value": 0.60, "key": "nf_prec_lv_0_5"}, {"id": "p_5_10_ge", "kind": "ge", "value": 0.90, "key": "nf_prec_5_10"},
                       {"id": "p_10_20_ge", "kind": "ge", "value": 0.90, "key": "nf_prec_10_20"},
                       {"id": "p_20_40_sep_above_ref", "kind": "sep_better", "key": "nf_prec_20_40", "dir": "higher"}]},
    "MAP-3": {"required": True, "planner": False, "split": "hold", "tier": "T0", "rule": "lane tol-IoU 0.5 m at 0-20 m >= 1.5 x refcv8 and separated; drivable-boundary recall @0.3 m at 0-20 m >= 0.50",
              "lever": "map representation lever (5.4 MAP-2 FAIL at 5-40 m family): next Q1-branch lever",
              "subs": [{"id": "lane_ge_1p5x_ref", "kind": "ge_ref", "value": 1.5, "key": "lane_tiou05_0_20"},
                       {"id": "lane_sep_above_ref", "kind": "sep_better", "key": "lane_tiou05_0_20", "dir": "higher"},
                       {"id": "bnd_recall_ge", "kind": "ge", "value": 0.50, "key": "drv_bnd_rec03_0_20"}]},
    "MAP-4": {"required": True, "planner": False, "split": "hold", "tier": "T0", "rule": "drv IoU in every band not separated worse than refcv8's drivable-union IoU",
              "lever": "name the band; map representation lever", "subs": [{"id": "drv_iou_not_sep_worse", "kind": "not_sep_worse", "key": MAP4_ROWS, "dir": "higher"}]},
    "MAP-5": {"required": True, "planner": False, "split": "hold", "tier": "T0", "rule": "own sidewalk / edge flags on unobs cells = 0 (a structural identity, no seed changes it)",
              "lever": "identity broken: the writer / mask path is wrong, not a tuning matter", "subs": [{"id": "unobs_flags_zero", "kind": "zero", "value": 0}]},
    "BOX-1": {"required": True, "planner": False, "split": "hold", "tier": "T0", "rule": "AP@2 m, head output, no post-hoc NMS >= 0.45 AND separated above refcv8",
              "lever": "BOX-1 PASS / BOX-2 FAIL at 40-60 m -> LiDAR depth (D4); otherwise the box-head lever of 4.3 (B2/B3/B4)",
              "subs": [{"id": "ap_ge", "kind": "ge", "value": 0.45, "key": "ap2m_all"}, {"id": "ap_sep_above_ref", "kind": "sep_better", "key": "ap2m_all", "dir": "higher"}]},
    "BOX-2": {"required": True, "planner": False, "split": "hold", "tier": "T0", "rule": "AP@2 m by range, each separated above refcv8: 0-20 m >= 0.62; 20-40 m >= 0.32; 40-60 m >= 0.15",
              "lever": "far range: LiDAR depth (D4); long-range adaptive queries with range-modulated DN at 80-100 m (5.4)",
              "subs": [{"id": "ap_0_20_ge", "kind": "ge", "value": 0.62, "key": "ap2m_0_20"}, {"id": "ap_0_20_sep", "kind": "sep_better", "key": "ap2m_0_20", "dir": "higher"},
                       {"id": "ap_20_40_ge", "kind": "ge", "value": 0.32, "key": "ap2m_20_40"}, {"id": "ap_20_40_sep", "kind": "sep_better", "key": "ap2m_20_40", "dir": "higher"},
                       {"id": "ap_40_60_ge", "kind": "ge", "value": 0.15, "key": "ap2m_40_60"}, {"id": "ap_40_60_sep", "kind": "sep_better", "key": "ap2m_40_60", "dir": "higher"}]},
    "BOX-3": {"required": True, "planner": False, "split": "hold", "tier": "T0", "rule": "boxes per detected object at the P=R gate, no NMS <= 1.25",
              "lever": "de-dup not trained in: in-loss BEV-NMS (B3') if not in; more DN groups (5.4)", "subs": [{"id": "bpo_le", "kind": "le", "value": 1.25, "key": "boxes_per_object_noNMS"}]},
    "BOX-4": {"required": False, "planner": False, "split": "hold", "tier": "T0", "rule": "(beside, never a bar) TP class accuracy >= 0.822 - its CI half-width; person->rider confusion reported",
              "lever": "-", "subs": [{"id": "tp_class_acc_beside", "kind": "beside", "value": 0.822, "key": "tp_class_acc"}]},
    "FAM-1": {"required": True, "planner": True, "split": "all", "tier": "T1", "rule": "no registered LONGITUDINAL / LATERAL / TACTICAL / ade_m row separated worse than refcv8 at seeds 0 and 1; STRATEGIC PAUSED",
              "lever": "name the row; attribute with the trace-fixed method (MAPFT K-PLAN-ID) and E2 K0 vs K2 (5.4)",
              "subs": [{"id": "no_row_separated_worse", "kind": "not_sep_worse"}]},
    "FAM-2": {"required": True, "planner": True, "split": "hold", "tier": "T1", "rule": "MAP-COMPLIANCE (off-drivable, edge contact, lane crossing vs GT) each separated better on S6c-HOLD; "
              "LONGITUDINAL distance keeping (headway / time-gap / TTC) not separated worse", "lever": "name the row; planner wiring (C8) / critic weight",
              "subs": [{"id": "off_drivable_sep_better", "kind": "sep_better", "key": "off_drivable", "dir": "lower"},
                       {"id": "edge_contact_sep_better", "kind": "sep_better", "key": "edge_contact", "dir": "lower"},
                       {"id": "lane_cross_sep_better", "kind": "sep_better", "key": "lane_cross", "dir": "lower"},
                       {"id": "distance_keeping_not_worse", "kind": "not_sep_worse", "key": DK_ROWS, "dir": "higher", "split": "all"}]},
    "W1": {"required": True, "planner": True, "split": "all", "tier": "T1", "rule": "false decels (FD, GT-cruise windows) separated lower than refcv8 + H2cn",
           "lever": "decel gating / H2cn stage", "subs": [{"id": "fd_sep_lower", "kind": "sep_better", "key": "FD", "dir": "lower"}]},
    "W2": {"required": True, "planner": True, "split": "all", "tier": "T1", "rule": "missed decels (MD, GT-decel windows) not separated worse than refcv8 + H2cn",
           "lever": "decel gating / H2cn stage", "subs": [{"id": "md_not_sep_worse", "kind": "not_sep_worse", "key": "MD", "dir": "lower"}]},
    "W4": {"required": True, "planner": True, "split": "all", "tier": "T1", "rule": "separated BETTER than CAMERA-BLIND on ade_m and each family's primary row",
           "lever": "reported FIRST, no ship request: the planner does not use the camera beyond the ego inputs (5.4)", "subs": [{"id": "sep_better_than_camblind", "kind": "sep_better"}]},
    "W5": {"required": True, "planner": True, "split": "all", "tier": "T1", "rule": "separated BETTER than the EGO-ONLY FLOOR on ade_m and each family's primary row",
           "lever": "reported FIRST, no ship request: the planner does not use the camera beyond the ego inputs (5.4)", "subs": [{"id": "sep_better_than_egofloor", "kind": "sep_better"}]},
}
FLOOR_ROWS = ("ade_m",) + FAMILIES


class Refused(Exception):
    """An input the judge will not read (DEV for a HOLD bar, unstamped, wrong seed, forbidden estimator ...): rc 2, no verdict."""


def _ci():
    try:
        from taniteval import ci
    except ModuleNotFoundError:
        here = Path(__file__).resolve()
        for cand in (here.parents[2] / "taniteval", here.parents[1] / "taniteval"):
            if (cand / "taniteval" / "ci.py").is_file():
                sys.path.insert(0, str(cand))
                break
        from taniteval import ci
    return ci


def _fin(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def split_of(doc):
    s = doc.get("split") if isinstance(doc, dict) else None
    if isinstance(s, str) and s.lower() in ("hold", "dev", "all"):
        return s.lower()
    h = doc.get("HOLD") if isinstance(doc, dict) else None
    if isinstance(h, dict) and isinstance(h.get("subset"), str) and h["subset"].lower() in ("hold", "dev", "all"):
        return h["subset"].lower()
    return None


def load_doc(path, name, want_split, want_tier, seed=None):
    """-> (doc, provenance).  Missing / unreadable file = (None, {missing}) (NOT-COMPUTABLE downstream).  A readable file with a wrong / absent stamp = Refused."""
    prov = {"input": name, "path": str(path) if path else None, "want_split": want_split, "want_tier": want_tier, "seed": seed}
    try:
        raw = Path(path).read_bytes()
        doc = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, TypeError):
        prov["missing"] = True
        return None, prov
    prov["sha256"] = hashlib.sha256(raw).hexdigest()
    if not isinstance(doc, dict):
        raise Refused(f"{name} ({path}): not a JSON object")
    sp = split_of(doc)
    if sp is None:
        raise Refused(f"{name} ({path}): no split stamp (`split` hold|dev|all, or `HOLD.subset`): an unstamped input is not read")
    if sp == "dev":
        raise Refused(f"{name} ({path}): DEV-split input for a {want_split.upper()} bar -- refused")
    if sp != want_split:
        raise Refused(f"{name} ({path}): split {sp!r} but this bar is read on {want_split!r}")
    tier = doc.get("tier")
    if not (isinstance(tier, str) and tier[:2] in ("T0", "T1")):
        raise Refused(f"{name} ({path}): no eval-tier stamp (`tier` must start T0 or T1)")
    if tier[:2] != want_tier:
        raise Refused(f"{name} ({path}): tier {tier[:2]} but this bar is {want_tier}")
    if seed is not None and str(doc.get("infer_seed")) != str(seed):
        raise Refused(f"{name} ({path}): infer_seed {doc.get('infer_seed')!r} sits in seed slot {seed}")
    prov.update(split=sp, tier=tier)
    return doc, prov


def _est_ok(est, where):
    if not (isinstance(est, str) and "episode_cluster" in est) or any(b in est for b in ("overlapping_holdout", "jackknife")):
        raise Refused(f"{where}: estimator {est!r} is not the (paired) episode-cluster bootstrap")


def row_read(row, direction=None, where="row", boot=2000, seed=0) -> dict:
    """Normalise to {ok, delta, lo, hi, separated, separated_worse, separated_better, estimator}.  Precomputed records must name the episode-cluster estimator."""
    better = direction or (row.get("better") if isinstance(row, dict) else None)
    if not isinstance(row, dict) or better not in ("higher", "lower"):
        return {"ok": False, "why": f"{where}: row absent or has no direction"}
    if isinstance(row.get("a"), list) and "b" in row and "eid" in row:
        try:
            r = _ci().paired_episode_cluster_bootstrap(row["a"], row["b"], row["eid"], n_boot=boot, seed=seed)
        except Exception as e:                                    # noqa: BLE001 -- unaligned / empty = NOT-COMPUTABLE with its reason
            return {"ok": False, "why": f"{where}: paired bootstrap refused: {e}"}
        d = {k: r[k] for k in ("delta", "lo", "hi", "separated")}
        d.update(n=r["n_windows"], n_episodes=r["n_episodes"], estimator=r["estimator"], src="ci.paired_episode_cluster_bootstrap B2000 seed0")
    else:
        _est_ok(row.get("estimator"), where)
        d = {k: row.get(k) for k in ("delta", "lo", "hi", "separated")}
        d.update(n=row.get("n", row.get("n_windows")), n_episodes=row.get("n_episodes"), estimator=row["estimator"], src="precomputed paired record")
        if not (_fin(d["delta"]) and isinstance(d["separated"], bool)):
            return {"ok": False, "why": f"{where}: no finite delta / boolean `separated`"}
    if row.get("undefined") or row.get("degenerate"):
        return {"ok": False, "why": f"{where}: undefined / degenerate interval is not evidence"}
    good = d["delta"] > 0 if better == "higher" else d["delta"] < 0
    d.update(ok=True, better=better, separated_worse=bool(d["separated"] and not good), separated_better=bool(d["separated"] and good))
    return d


def _res(sid, status, detail, prov=None, **kw):
    return {"id": sid, "status": status, "detail": detail, "evidence": prov, **kw}


def _nc(sid, why, prov=None):
    return _res(sid, "NOT-COMPUTABLE", why, prov)


def _sub_point(sub, point, prov, rec=None):
    if not _fin(point):
        return _nc(sub["id"], f"{sub['key'] if 'key' in sub else sub['id']}: point absent / non-finite ({point!r})", prov)
    ok = point >= sub["value"] if sub["kind"] == "ge" else point <= sub["value"]
    ci = {k: rec.get(k) for k in ("lo", "hi", "estimator")} if isinstance(rec, dict) else None
    return _res(sub["id"], "PASS" if ok else "FAIL", f"point {point:.6g} {'>=' if sub['kind'] == 'ge' else '<='} bar {sub['value']}", prov, point=point, bar=sub["value"], ci=ci)


def _sub_row(sub, rr, prov, where, direction=None):
    if not rr["ok"]:
        return _nc(sub["id"], rr["why"], prov)
    want_better = sub["kind"] == "sep_better"
    ok = rr["separated_better"] if want_better else not rr["separated_worse"]
    return _res(sub["id"], "PASS" if ok else "FAIL",
                f"{where}: delta {rr['delta']:+.6g} [{rr['lo']}, {rr['hi']}] separated={rr['separated']} ({rr['better']} is better) -> "
                f"{'separated better required' if want_better else 'must not be separated worse'}", prov, row={k: rr.get(k) for k in ("delta", "lo", "hi", "separated", "n", "n_episodes", "estimator", "src")})


def _agg(subs):
    st = [s["status"] for s in subs]
    return "FAIL" if "FAIL" in st else ("NOT-COMPUTABLE" if "NOT-COMPUTABLE" in st or not st else "PASS")


# ------------------------------------------------------------------------------------------------------------------ perception (HOLD, T0, read once)
def perception_bars(docs: dict) -> dict:
    e9, e8, mp = docs.get("eval_r9"), docs.get("eval_r8"), docs.get("map_paired")
    P = lambda n: docs["_prov"].get(n)                                                                         # noqa: E731
    out = {}

    def box_rec(k):
        return _dig(e9, "boxes", k)

    def paired_box(k):
        r = _dig(e9, "paired_vs_B", "boxes", k)
        return row_read(r, "higher", f"paired_vs_B.boxes.{k}") if r is not None else {"ok": False, "why": f"paired_vs_B.boxes.{k} absent (r9_e1_eval --packs-b)"}

    def near(doc, k):
        return _dig(doc, "nearfield", "ci", k)

    def maprow(k, direction):
        r = _dig(mp, "rows", k)
        return row_read(r, direction, f"map_paired.rows.{k}") if r is not None else {"ok": False, "why": f"map_paired.rows.{k} absent (no producer yet)"}

    for bid in ("MAP-2", "MAP-3", "MAP-4", "MAP-5", "BOX-1", "BOX-2", "BOX-3", "BOX-4"):
        subs = []
        for sub in BARS[bid]["subs"]:
            k, kind = sub.get("key"), sub["kind"]
            try:
                if bid.startswith("BOX"):
                    if kind in ("ge", "le"):
                        rec = box_rec(k)
                        subs.append(_sub_point(sub, rec.get("point") if isinstance(rec, dict) else None, P("eval_r9"), rec))
                    elif kind == "sep_better":
                        subs.append(_sub_row(sub, paired_box(k), P("eval_r9"), f"paired_vs_B.boxes.{k}"))
                    else:                                                                                      # BOX-4 beside
                        rec = box_rec(k)
                        if not (isinstance(rec, dict) and all(_fin(rec.get(x)) for x in ("point", "lo", "hi"))):
                            subs.append(_nc(sub["id"], f"boxes.{k} point/lo/hi absent", P("eval_r9")))
                        else:
                            half = (rec["hi"] - rec["lo"]) / 2.0
                            subs.append(_res(sub["id"], "BESIDE", f"point {rec['point']:.6g} vs {sub['value']} - half-width {half:.4g}: " + ("met" if rec["point"] >= sub["value"] - half else "NOT met")
                                             + " (beside, never a bar; person->rider confusion to be reported by the producer)", P("eval_r9"), point=rec["point"], half_width=half))
                elif bid == "MAP-2":
                    if kind == "ge":
                        rec = near(e9, k)
                        subs.append(_sub_point(sub, rec.get("point") if isinstance(rec, dict) else None, P("eval_r9"), rec))
                    else:
                        subs.append(_sub_row(sub, maprow(k, sub["dir"]), P("map_paired"), f"map_paired.rows.{k}"))
                elif bid == "MAP-3":
                    if kind == "ge":
                        rec = near(e9, k)
                        subs.append(_sub_point(sub, rec.get("point") if isinstance(rec, dict) else None, P("eval_r9"), rec))
                    elif kind == "ge_ref":
                        a, b = (near(e9, k) or {}).get("point"), (near(e8, k) or {}).get("point")
                        if not (_fin(a) and _fin(b) and b > 0):
                            subs.append(_nc(sub["id"], f"{k}: refcv9 {a!r} / refcv8 {b!r} absent or reference not > 0", P("eval_r8")))
                        else:
                            subs.append(_res(sub["id"], "PASS" if a >= sub["value"] * b else "FAIL", f"refcv9 {a:.6g} vs {sub['value']} x refcv8 {b:.6g} = {sub['value'] * b:.6g}",
                                             [P("eval_r9"), P("eval_r8")], point=a, reference=b, bar=sub["value"]))
                    else:
                        subs.append(_sub_row(sub, maprow(k, sub["dir"]), P("map_paired"), f"map_paired.rows.{k}"))
                elif bid == "MAP-4":
                    parts = [_sub_row(dict(sub, id=f"{sub['id']}:{rk}"), maprow(rk, sub["dir"]), P("map_paired"), f"map_paired.rows.{rk}") for rk in k]
                    subs.append(_res(sub["id"], _agg(parts), "; ".join(f"{p['id'].split(':')[1]}={p['status']}" for p in parts), P("map_paired"), parts=parts))
                else:                                                                                          # MAP-5 structural identity
                    uf = _dig(e9, "nearfield", "unobs_flags")
                    if not (isinstance(uf, dict) and _fin(uf.get("sidewalk")) and _fin(uf.get("edge"))):
                        subs.append(_nc(sub["id"], "nearfield.unobs_flags {sidewalk, edge} absent (r9_e1_eval does not emit it yet: open producer item)", P("eval_r9")))
                    else:
                        subs.append(_res(sub["id"], "PASS" if uf["sidewalk"] == 0 and uf["edge"] == 0 else "FAIL",
                                         f"structural identity: own flags on unobs cells sidewalk={uf['sidewalk']} edge={uf['edge']} (must be exactly 0; no interval, no seed)", P("eval_r9")))
            except Refused:
                raise
        out[bid] = subs
    return out


# ------------------------------------------------------------------------------------------------------------------ planner bars (one seed)
def _rows_of(doc):
    r = doc.get("rows") if isinstance(doc, dict) else None
    return r if isinstance(r, dict) else {}


def families_report(fdoc):
    rows = _rows_of(fdoc)
    rep = {}
    for fam in FAMILIES:
        fr = rows.get(fam)
        if not isinstance(fr, dict) or not fr:
            rep[fam] = {"status": "NOT-COMPUTABLE", "reason": "no rows for this family in the input", "n_rows": 0, "rows": {}}
            continue
        rr = {rid: row_read(r, None, f"families.{fam}.{rid}") for rid, r in fr.items()}
        rep[fam] = {"status": "FAIL" if any(v.get("separated_worse") for v in rr.values()) else ("NOT-COMPUTABLE" if any(not v["ok"] for v in rr.values()) else "PASS"),
                    "n_rows": len(rr), "rows": rr}
    rep["STRATEGIC"] = {"status": "PAUSED", "reason": STRATEGIC_PAUSED, "rows": {}}
    return rep


def planner_bars(seed: str, docs: dict) -> tuple[dict, dict]:
    P = lambda n: docs["_prov"].get(n)                                                                         # noqa: E731
    s9, comp, fam, dk, w12, cb, ef = (docs.get(k) for k in ("s6c_r9", "compliance", "families", "distkeep", "w12", "camblind", "egofloor"))
    out, famrep = {}, None
    # MAP-1 / FAM-2 compliance
    rate = _dig(s9, "HOLD", "rate")
    subs = []
    for sub in BARS["MAP-1"]["subs"]:
        if sub["kind"] == "le":
            subs.append(_sub_point(dict(sub, key="HOLD.rate"), rate, P("s6c_r9"), _dig(s9, "HOLD", "bootstrap")))
        else:
            r = _rows_of(comp).get(sub["key"])
            subs.append(_sub_row(sub, row_read(r, sub["dir"], f"compliance.rows.{sub['key']}") if r is not None else {"ok": False, "why": f"compliance.rows.{sub['key']} absent"},
                                 P("compliance"), f"compliance.rows.{sub['key']}"))
    out["MAP-1"] = subs
    subs = []
    for sub in BARS["FAM-2"]["subs"]:
        if sub["id"] == "distance_keeping_not_worse":
            rows = _dig(dk, "rows", "LONGITUDINAL") or {}
            parts = [_sub_row(dict(sub, id=f"{sub['id']}:{rk}"), row_read(rows[rk], sub["dir"], f"distkeep.rows.LONGITUDINAL.{rk}") if rk in rows else {"ok": False, "why": f"distkeep.rows.LONGITUDINAL.{rk} absent"},
                              P("distkeep"), rk) for rk in sub["key"]]
            subs.append(_res(sub["id"], _agg(parts), "; ".join(f"{p['id'].split(':')[1]}={p['status']}" for p in parts), P("distkeep"), parts=parts))
        else:
            r = _rows_of(comp).get(sub["key"])
            subs.append(_sub_row(sub, row_read(r, sub["dir"], f"compliance.rows.{sub['key']}") if r is not None else {"ok": False, "why": f"compliance.rows.{sub['key']} absent"},
                                 P("compliance"), f"compliance.rows.{sub['key']}"))
    out["FAM-2"] = subs
    # FAM-1 + the four-family report
    famrep = families_report(fam) if fam is not None else {f: {"status": "NOT-COMPUTABLE", "reason": "families input absent", "n_rows": 0, "rows": {}} for f in FAMILIES} | {"STRATEGIC": {"status": "PAUSED", "reason": STRATEGIC_PAUSED, "rows": {}}}
    ade = next((r for f in FAMILIES for rid, r in (_rows_of(fam).get(f) or {}).items() if rid == "ade_m"), None) if fam is not None else None
    parts = []
    for f in FAMILIES:
        st = famrep[f]["status"]
        parts.append(_res(f"FAM-1:{f}", st, famrep[f].get("reason") or f"{famrep[f]['n_rows']} row(s)", P("families")))
    parts.append(_sub_row({"id": "FAM-1:ade_m", "kind": "not_sep_worse"}, row_read(ade, None, "families ade_m") if ade is not None else {"ok": False, "why": "no `ade_m` row in the families input"}, P("families"), "ade_m"))
    out["FAM-1"] = [_res("no_row_separated_worse", _agg(parts), "; ".join(f"{p['id'].split(':')[1]}={p['status']}" for p in parts), P("families"), parts=parts)]
    # W1 / W2
    for bid in ("W1", "W2"):
        sub = BARS[bid]["subs"][0]
        r = _rows_of(w12).get(sub["key"])
        out[bid] = [_sub_row(sub, row_read(r, sub["dir"], f"w12.rows.{sub['key']}") if r is not None else {"ok": False, "why": f"w12.rows.{sub['key']} absent"}, P("w12"), f"w12.rows.{sub['key']}")]
    # W4 / W5
    for bid, doc, nm in (("W4", cb, "camblind"), ("W5", ef, "egofloor")):
        sub = BARS[bid]["subs"][0]
        parts = []
        for rk in FLOOR_ROWS:
            r = _rows_of(doc).get(rk)
            parts.append(_sub_row(dict(sub, id=f"{sub['id']}:{rk}"), row_read(r, None, f"{nm}.rows.{rk}") if r is not None else {"ok": False, "why": f"{nm}.rows.{rk} absent"}, P(nm), rk))
        out[bid] = [_res(sub["id"], _agg(parts), "; ".join(f"{p['id'].split(':')[1]}={p['status']}" for p in parts), P(nm), parts=parts)]
    return out, famrep


# ------------------------------------------------------------------------------------------------------------------ assembly
def _combine(per_seed: dict) -> list:
    """Sub results per seed -> one per sub: PASS only if PASS at EVERY seed; FAIL if it FAILS at any; else NOT-COMPUTABLE."""
    ids = [s["id"] for s in per_seed[SEEDS[0]]]
    out = []
    for i, sid in enumerate(ids):
        ps = {s: per_seed[s][i] for s in SEEDS}
        out.append({"id": sid, "status": _agg(list(ps.values())), "per_seed": ps, "seeds_not_pass": [s for s in SEEDS if ps[s]["status"] != "PASS"]})
    return out


def load_all(inputs: dict):
    extra = set(inputs) - INPUT_KEYS
    if extra:
        raise Refused(f"unknown INPUTS.json key(s) {sorted(extra)}: the bars are in the judge's table, not in an input")
    if set(inputs.get("perception") or {}) - PERC_KEYS:
        raise Refused(f"unknown perception key(s) {sorted(set(inputs['perception']) - PERC_KEYS)}")
    if inputs.get("scored_ema_step") != SCORED_EMA_STEP:
        raise Refused(f"scored_ema_step {inputs.get('scored_ema_step')!r}: only the EMA at {SCORED_EMA_STEP} is a candidate on HOLD (design 5.1)")
    seeds = inputs.get("seeds") or {}
    if set(seeds) != set(SEEDS):
        raise Refused(f"inference seeds 0 AND 1 are both required, got {sorted(seeds)}")
    perc = inputs.get("perception") or {}
    pdocs = {"_prov": {}}
    for nm, tier in (("eval_r9", "T0"), ("eval_r8", "T0"), ("map_paired", "T0")):
        d, pv = load_doc(perc.get(nm), nm, "hold", tier)
        pdocs[nm], pdocs["_prov"][nm] = d, pv
    sdocs = {}
    for s in SEEDS:
        if set(seeds[s]) - SEED_KEYS:
            raise Refused(f"unknown seed-{s} key(s) {sorted(set(seeds[s]) - SEED_KEYS)}")
        sd = {"_prov": {}}
        for nm in sorted(SEED_KEYS):
            split = {"s6c_r9": "hold", "compliance": "hold"}.get(nm, "all")
            d, pv = load_doc(seeds[s].get(nm), nm, split, "T1", seed=s)
            sd[nm], sd["_prov"][nm] = d, pv
        sdocs[s] = sd
    for nm in SEED_KEYS:                                                       # one file for both seeds = one draw presented as two
        a, b = (sdocs[s]["_prov"][nm].get("sha256") for s in SEEDS)
        if a and a == b:
            raise Refused(f"{nm}: the same bytes sit in seed slots 0 and 1 (one inference draw presented as two)")
    return pdocs, sdocs


def build(inputs: dict) -> dict:
    pdocs, sdocs = load_all(inputs)
    kb, _ = (lambda p: (json.loads(Path(p).read_text(encoding="utf-8")) if p and Path(p).is_file() else None, None))(inputs.get("kbase"))
    if isinstance(kb, dict) and kb.get("sha256_match") is False or isinstance(kb, dict) and kb.get("forward_refit_reproduces") is False:
        kstat = "FALSE"
    elif isinstance(kb, dict) and kb.get("sha256_match") is True and kb.get("forward_refit_reproduces") is True:
        kstat = "TRUE"
    else:
        kstat = "OPEN"
    perc = perception_bars(pdocs)
    per_seed, famrep = {s: {} for s in SEEDS}, {}
    for s in SEEDS:
        per_seed[s], famrep[s] = planner_bars(s, sdocs[s])
    bars = {}
    for bid, spec in BARS.items():
        subs = _combine({s: per_seed[s][bid] for s in SEEDS}) if spec["planner"] else \
            [{"id": r["id"], "status": r["status"], "per_seed": {"n/a (deterministic, read once on HOLD)": r}, "seeds_not_pass": []} for r in perc[bid]]
        st = "BESIDE" if not spec["required"] else _agg(subs)
        bars[bid] = {"status": st, "required": spec["required"], "planner": spec["planner"], "split": spec["split"], "tier": "T0 (label level)" if spec["tier"] == "T0" else
                     "T1 self-action open loop (not driving performance)", "rule": spec["rule"], "subs": subs, "next_lever": spec["lever"] if st in ("FAIL", "NOT-COMPUTABLE") else None}
    req = {b: v for b, v in bars.items() if v["required"]}
    failing = [b for b, v in req.items() if v["status"] == "FAIL"]
    nc = [b for b, v in req.items() if v["status"] == "NOT-COMPUTABLE"]
    verdict = "VOID" if kstat == "FALSE" else ("FAIL" if failing else ("INCOMPLETE" if nc or kstat == "OPEN" else "PASS"))
    return {"schema": SCHEMA, "verdict": verdict, "k_base": kstat, "failing_bars": [{"id": b, "next_lever": bars[b]["next_lever"]} for b in failing],
            "not_computable_bars": nc, "bars": bars, "four_families": famrep, "strategic": STRATEGIC_PAUSED,
            "labels": f"{inputs.get('refcv9_label', 'refcv9')} (EMA-{SCORED_EMA_STEP}) vs {inputs.get('reference_label', 'refcv8')}",
            "split_policy": "perception, map and S6c bars on HOLD (eval139 odd half); four families and W1/W2/W4/W5 on all of eval139 (design 5.1); a DEV input is refused",
            "seeds": "planner bars at inference seeds 0 AND 1 (both must hold); perception read once (deterministic)", "estimator": "paired episode-cluster bootstrap B 2000 seed 0 over EPISODES (taniteval/ci.py); never overlapping_holdout_se",
            "provenance": {"perception": pdocs["_prov"], "seeds": {s: sdocs[s]["_prov"] for s in SEEDS}}, "caveat": CAVEAT,
            "rule": "PASS = every bold bar + W1, W2, MAP-4, MAP-5 (BOX-4 beside); a missed bar is reported FAILED as written, then the section-5.4 lever runs (Rule Zero)."}


def render_md(v: dict) -> str:
    L = [f"# E3 final-read verdict: {v['verdict']}  ({v['labels']})", "", v["rule"], "", f"K-BASE: {v['k_base']}", "",
         "| bar | status | split | tier | seeds not PASS |", "|---|---|---|---|---|"]
    for b, x in v["bars"].items():
        ns = sorted({s for sb in x["subs"] for s in sb["seeds_not_pass"]})
        L.append(f"| {b}{'' if x['required'] else ' (beside)'} | **{x['status']}** | {x['split']} | {x['tier']} | {','.join(ns) or '-'} |")
    for f in v["failing_bars"]:
        L += ["", f"FAILED as written: {f['id']} -- next lever: {f['next_lever']}"]
    for b in v["not_computable_bars"]:
        L += ["", f"NOT-COMPUTABLE: {b} -- {v['bars'][b]['next_lever']}"]
    for s in SEEDS:
        L += ["", f"## four families, inference seed {s}"]
        for fam, r in v["four_families"][s].items():
            L.append(f"* {fam}: {r['status']}" + (f" -- {r['reason']}" if r.get("reason") else f" (n rows {r.get('n_rows')})"))
    L += ["", f"STRATEGIC: {v['strategic']}", f"Splits: {v['split_policy']}", f"Seeds: {v['seeds']}", f"Estimator: {v['estimator']}", f"Caveat: {v['caveat']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    try:
        inputs = json.loads(Path(a.inputs).read_text(encoding="utf-8"))
        if not isinstance(inputs, dict):
            raise ValueError("not an object")
    except (OSError, ValueError) as e:
        print(f"[r9-e3-verdict] REFUSED: cannot read {a.inputs}: {e}", file=sys.stderr)
        return 2
    try:
        v = build(inputs)
    except Refused as e:
        print(f"[r9-e3-verdict] REFUSED: {e}", file=sys.stderr)
        return 2
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "E3_VERDICT.json").write_text(json.dumps(v, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "E3_VERDICT.md").write_text(render_md(v), encoding="utf-8")
    print(f"E3 verdict: {v['verdict']}" + ("" if v["verdict"] == "PASS" else "  failing: " + ", ".join(f['id'] for f in v["failing_bars"]) + "  not-computable: " + ", ".join(v["not_computable_bars"])))
    return 0 if v["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
