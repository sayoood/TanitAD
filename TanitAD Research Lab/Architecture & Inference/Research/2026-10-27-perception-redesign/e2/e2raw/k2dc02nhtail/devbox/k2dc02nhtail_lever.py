"""k2dc02nhtail_lever.py -- PREREG_K2DC02NH_2026-11-20.md sec. 3 (the LEVER RULE) and sec. 4 (the committed next-step readings), as code. Nothing here sets a constant of any earlier PREREG.

The rule is the dose rule of PREREG_K2DC / PREREG_K2DC02 (stack/scripts/r9_e2_dose_verdict.py: BAR_M, BAND_M, `decide`) applied with K2DC02nh in the candidate slot and K2DC02 (tail) in K2's slot ("same constants as the dose rule", sec. 3).
Per inference seed s in (0, 1); the outcome must hold at BOTH seeds (sec. 3: "must hold at both seeds 0 and 1"):
  SUPPORTED     K2DC02nh_t - K0_t LAT_cross_mae_m NOT separated worse (a separated, positive delta is worse), AND K2DC02nh_t - K2DC02_t LAT_cross_mae_m <= -2.5 mm,
                AND the jitter delta K2DC02nh_t - K2DC02_t < 0 (the frame-to-frame |d e| row of the dose tool)
  REFUTED       K2DC02nh_t - K2DC02_t LAT_cross_mae_m AND LAT_cross_abs_final_m both within +-1 mm
  INCONCLUSIVE  anything else
`reading` in LEVER_READING.json takes exactly the sec. 4 headings: SUPPORTED (items 1 and 2) | REFUTED (item 3) | INCONCLUSIVE (item 4) | VOID (item 5); `sec4_item` says which item, and for SUPPORTED it
splits item 1 (E2 rule GO on either read) from item 2 (the E2 clause (a) FAILS on a read, or the 0-10 m precision `nf_prec_0_10` of K2DC02nh is separated worse than K2DC02's; UNPAIRED, intervals do not overlap).
VOID (item 5) = a tail-read validity item failed (--tail-void), the reproduction check failed (--repro-void) or the zero-gradient gate failed (--zerograd-void). A VOID read makes no lever claim.
--mode ema writes the same numbers on the EMA read with `reading` = EMA-REPORTED-BESIDE (sec. 2/3: the TAIL read decides the lever).
The map trade-off (sec. 3 last bullet) is written per seed: K2DC02nh - K2DC02 on nf_swedge_prec_lv_5_20 and nf_prec_0_10, each arm's own interval beside it. The map rows come from the stage-A single-arm EVAL JSONs, so they do not depend on the inference seed.

usage: python k2dc02nhtail_lever.py --mode tail|ema --nh-pf <dir> --k2dc02-pf <dir> --k0-pf <dir> --dev-split <json> --nh-read <dir> --k2dc02-read <dir> --out-dir <dir>
         [--other-e2-verdict <E2_VERDICT.json of the other read>] [--boot 2000] [--tail-void] [--repro-void] [--zerograd-void] [--numbers-json <precomputed per-seed numbers; tests only>]
EXIT: 0 a reading was made | 5 VOID | 4 REFUSED (unreadable or inconsistent input)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SEEDS = ("0", "1")
CROSS = "LAT_cross_mae_m"
FINAL = "LAT_cross_abs_final_m"
LAT_DIST_ROWS = (CROSS, FINAL)
BAR_M = 0.0025              # PREREG_K2DC02NH sec. 3: K2DC02nh_t - K2DC02_t LAT_cross_mae_m <= -2.5 mm (the dose rule's bar)
BAND_M = 0.0010             # REFUTED: both lateral distance rows within +-1 mm
EPS = 1e-9
ARM = "K2DC02nh"
MAP_PRIMARY = "nf_swedge_prec_lv_5_20"
MAP_PREC_0_10 = "nf_prec_0_10"
READINGS = ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "VOID")


class Refused(Exception):
    pass


def _row(d: dict, key: str, where: str) -> dict:
    try:
        r = d[key]
        float(r["delta"])
    except (KeyError, TypeError, ValueError):
        raise Refused(f"{where}: no usable row {key!r}")
    return r


def seed_clauses(ps: dict, seed: str) -> dict:
    """ps = {"dc_vs_k0": {row: {delta, lo, hi, separated}}, "dc_vs_k2": {...}, "jitter_dlag_dc_vs_k2": {delta, ...}}; dc = K2DC02nh_t, k0 = K0_t, k2 = K2DC02_t."""
    w = _row(ps["dc_vs_k0"], CROSS, f"seed {seed} K2DC02nh-K0")
    d = _row(ps["dc_vs_k2"], CROSS, f"seed {seed} K2DC02nh-K2DC02")
    try:
        jd = float(ps["jitter_dlag_dc_vs_k2"]["delta"])
    except (KeyError, TypeError, ValueError):
        raise Refused(f"seed {seed}: no usable jitter row")
    not_worse = not (bool(w["separated"]) and float(w["delta"]) > 0.0)
    bar = float(d["delta"]) <= -BAR_M + EPS
    jit = jd < 0.0
    band = {r: abs(float(_row(ps["dc_vs_k2"], r, f"seed {seed} K2DC02nh-K2DC02")["delta"])) <= BAND_M + EPS for r in LAT_DIST_ROWS}
    return {"i_nh_k0_cross_not_separated_worse": not_worse, "ii_nh_k2dc02_cross_le_minus_2p5mm": bar, "iii_jitter_delta_negative": jit,
            "supported": bool(not_worse and bar and jit), "refuted_band": band, "refuted": bool(all(band.values()))}


def decide(per_seed: dict) -> dict:
    if set(per_seed) != set(SEEDS):
        raise Refused(f"inference seeds 0 AND 1 are both required, got {sorted(per_seed)}")
    cl = {s: seed_clauses(per_seed[s], s) for s in SEEDS}
    sup = all(cl[s]["supported"] for s in SEEDS)
    ref = all(cl[s]["refuted"] for s in SEEDS)
    if sup and ref:
        raise Refused("SUPPORTED and REFUTED both hold: the bar and the band overlap")
    return {"rule_outcome": "SUPPORTED" if sup else "REFUTED" if ref else "INCONCLUSIVE", "per_seed": cl}


def map_tradeoff(nh_eval: dict, k2_eval: dict) -> dict:
    """K2DC02nh - K2DC02 on the map primary and the 0-10 m precision, from the single-arm stage-A EVAL JSONs (point + episode-cluster interval each). UNPAIRED: `separated_worse` = the
    K2DC02nh interval lies wholly below the K2DC02 interval (higher is better). It is written, never used to move the lever rule."""
    out = {}
    for key in (MAP_PRIMARY, MAP_PREC_0_10):
        try:
            a, b = nh_eval["nearfield"]["ci"][key], k2_eval["nearfield"]["ci"][key]
            pa, pb = float(a["point"]), float(b["point"])
        except (KeyError, TypeError, ValueError):
            raise Refused(f"no nearfield.ci.{key} in the K2DC02nh / K2DC02 EVAL json")
        out[key] = {"nh_point": pa, "nh_lo": float(a["lo"]), "nh_hi": float(a["hi"]), "k2dc02_point": pb, "k2dc02_lo": float(b["lo"]), "k2dc02_hi": float(b["hi"]),
                    "delta_nh_minus_k2dc02": pa - pb, "separated_worse": bool(float(a["hi"]) < float(b["lo"])), "separated_better": bool(float(a["lo"]) > float(b["hi"])),
                    "estimator": "two single-arm episode-cluster intervals compared for overlap (UNPAIRED); higher is better"}
    return out


def e2_go(v: dict | None) -> dict:
    """What sec. 4 item 1 / 2 need from an E2_VERDICT.json: GO / NO-GO and whether clause (a) failed."""
    if v is None:
        return {"read": False, "go": None, "clause_a_failed": None}
    fc = [c.get("id") for c in (v.get("failing_clauses") or []) if isinstance(c, dict)]
    return {"read": True, "go": v.get("verdict") == "GO", "verdict": v.get("verdict"), "failing_clauses": fc, "clause_a_failed": "a" in fc}


def sec4_item(rule: str, e2_this: dict, e2_other: dict, tradeoff: dict, voids: list) -> tuple:
    """-> (reading, item, why)."""
    if voids:
        return "VOID", 5, "no lever claim: " + "; ".join(voids)
    if rule == "REFUTED":
        return "REFUTED", 3, "neither the map share nor HSG carries the deficit; next (zero training): a module-reset read of _map_hires, _perception, core.agent_head, core.box_tok_r9 on the tail weights, own prereg first"
    if rule == "INCONCLUSIVE":
        return "INCONCLUSIVE", 4, "next: a replicate (seed + 1), not a reinterpretation"
    if e2_this.get("go") or e2_other.get("go"):
        return "SUPPORTED", 1, "SUPPORTED and the E2 rule is GO on a read: report it; E3 with K2DC02nh goes to the PI"
    a_failed = bool(e2_this.get("clause_a_failed") or e2_other.get("clause_a_failed"))
    prec_worse = bool(tradeoff[MAP_PREC_0_10]["separated_worse"])
    if a_failed or prec_worse:
        return "SUPPORTED", 2, "SUPPORTED but E2 clause (a) fails or the 0-10 m precision is separated worse: the lateral/map trade-off is real; report both; next: HSG restricted to camera-blind cells (new code, own prereg, to the PI)"
    return "SUPPORTED", "unspecified", ("SUPPORTED, E2 not GO on the read(s) given and neither clause (a) nor the 0-10 m precision separated worse: sec. 4 names no next step for this combination "
                            "(item 1 needs GO, item 2 needs the trade-off); report to the PI" + ("" if e2_other.get("read") else "; the other read's E2 verdict was not given"))


def combine(numbers: dict, nh_eval: dict, k2_eval: dict, e2_this: dict | None, e2_other: dict | None, mode: str = "tail",
            tail_void: bool = False, repro_void: bool = False, zerograd_void: bool = False) -> dict:
    d = decide({s: {k: numbers[s][k] for k in ("dc_vs_k0", "dc_vs_k2", "jitter_dlag_dc_vs_k2")} for s in SEEDS})
    voids = [n for n, f in (("TAIL-VOID", tail_void), ("REPRO-VOID", repro_void), ("ZEROGRAD-VOID", zerograd_void)) if f]
    to = map_tradeoff(nh_eval, k2_eval)
    a, b = e2_go(e2_this), e2_go(e2_other)
    reading, item, why = sec4_item(d["rule_outcome"], a, b, to, voids)
    if mode == "ema":
        reading_out, why = "EMA-REPORTED-BESIDE", "the EMA read is reported beside the tail read (PREREG sec. 2/3); the TAIL outcome decides. This read's sec. 4 reading would be " + reading + ": " + why
    else:
        reading_out = reading
    return {"mode": mode, "arm": ARM, "reading": reading_out, "sec4_item": item, "rule_outcome": d["rule_outcome"], "why": why, "voids": voids, "per_seed": d["per_seed"],
            "constants": {"bar_m": -BAR_M, "band_m": BAND_M, "eps": EPS, "rows": list(LAT_DIST_ROWS)},
            "e2_this_read": a, "e2_other_read": b, "map_tradeoff": to,
            "tier": "T1 (self-action open loop on logged eval139 frames; the `os` arm)", "windows": "DEV (DEV_HOLD_SPLIT.json); HOLD is never read",
            "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES (clips); delta = A - B; it answers another draw of EPISODES, not another training run (H-ESTIM-SEED-1) and not another inference draw (each seed is one call)",
            "STRATEGIC": {"status": "REPORTED IN FULL under strategic/ (PI 2026-11-16 ended the pause); report-only, it decides nothing here"},
            "rule": "PREREG_K2DC02NH_2026-11-20.md sec. 3 (lever rule, tail weights, both inference seeds, the dose rule's constants) and sec. 4 (next-step readings)",
            "share_note": "the in-training map-share band check of the dose tool is NOT part of this prereg's VOID list (sec. 4 item 5: tail-read validity item or zero-gradient gate) and is not read here"}


def render(res: dict) -> str:
    L = [f"# K2DC02nh {res['mode']} lever reading: **{res['reading']}** (sec. 4 item {res['sec4_item']}; rule outcome {res['rule_outcome']})", "", res["why"], "", f"Rule: {res['rule']}", "",
         f"Voids: {', '.join(res['voids']) or 'none'}", "", "## Clauses per inference seed", "", "| seed | nh-K0 cross not separated worse | nh-K2DC02 cross <= -2.5 mm | jitter < 0 | SUPPORTED | within +-1 mm (cross, final) | REFUTED |", "|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        c = res["per_seed"][s]
        L.append(f"| {s} | {c['i_nh_k0_cross_not_separated_worse']} | {c['ii_nh_k2dc02_cross_le_minus_2p5mm']} | {c['iii_jitter_delta_negative']} | {c['supported']} | {c['refuted_band']} | {c['refuted']} |")
    L += ["", "## Map trade-off (K2DC02nh - K2DC02; UNPAIRED single-arm intervals; the same at both inference seeds)", "", "| row | nh [lo, hi] | K2DC02 [lo, hi] | delta | separated worse |", "|---|---|---|---|---|"]
    for k, v in res["map_tradeoff"].items():
        L.append(f"| {k} | {v['nh_point']:.4f} [{v['nh_lo']:.4f}, {v['nh_hi']:.4f}] | {v['k2dc02_point']:.4f} [{v['k2dc02_lo']:.4f}, {v['k2dc02_hi']:.4f}] | {v['delta_nh_minus_k2dc02']:+.4f} | {v['separated_worse']} |")
    L += ["", "STRATEGIC: reported in full under strategic/ (report-only); four-family rows: FAM_TABLE.md (K2DC02nh_t - K0_t) and `seeds.<s>.dc_vs_k2_by_family` (K2DC02nh_t - K2DC02_t).", ""]
    return "\n".join(L)


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read {p}: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("tail", "ema"), required=True)
    ap.add_argument("--nh-pf")
    ap.add_argument("--k2dc02-pf")
    ap.add_argument("--k0-pf")
    ap.add_argument("--dev-split")
    ap.add_argument("--nh-read", required=True, help="dir holding EVAL_K2.json and E2_VERDICT.json of the K2DC02nh read")
    ap.add_argument("--k2dc02-read", required=True, help="dir holding the K2DC02 read's EVAL_K2.json")
    ap.add_argument("--other-e2-verdict", help="E2_VERDICT.json of the other read (EMA for --mode tail, tail for --mode ema)")
    ap.add_argument("--numbers-json", help="precomputed per-seed numbers (tests only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tail-void", action="store_true")
    ap.add_argument("--repro-void", action="store_true")
    ap.add_argument("--zerograd-void", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.numbers_json:
            numbers = _json(Path(a.numbers_json))
        else:
            if not (a.nh_pf and a.k2dc02_pf and a.k0_pf and a.dev_split):
                raise Refused("--nh-pf, --k2dc02-pf, --k0-pf and --dev-split are required")
            import r9_e2_dose_verdict as D            # the pinned export's stack/scripts is on PYTHONPATH
            if (D.BAR_M, D.BAND_M) != (BAR_M, BAND_M):
                raise Refused(f"the dose tool's constants ({D.BAR_M}, {D.BAND_M}) differ from this lever rule's ({BAR_M}, {BAND_M})")
            numbers = {s: D.seed_numbers(a.nh_pf, a.k2dc02_pf, a.k0_pf, a.dev_split, s, a.boot, ARM) for s in SEEDS}
            ref = D.decide({s: {k: numbers[s][k] for k in ("dc_vs_k0", "dc_vs_k2", "jitter_dlag_dc_vs_k2")} for s in SEEDS})["outcome_if_valid"]
            if ref != decide({s: {k: numbers[s][k] for k in ("dc_vs_k0", "dc_vs_k2", "jitter_dlag_dc_vs_k2")} for s in SEEDS})["rule_outcome"]:
                raise Refused("this lever rule and the dose tool's own `decide` disagree on the same numbers")
        nh_read, k2_read = Path(a.nh_read), Path(a.k2dc02_read)
        nh_eval, k2_eval = _json(nh_read / "EVAL_K2.json"), _json(k2_read / "EVAL_K2.json")
        e2_this = _json(nh_read / "E2_VERDICT.json") if (nh_read / "E2_VERDICT.json").is_file() else None
        e2_other = _json(Path(a.other_e2_verdict)) if a.other_e2_verdict else None
        res = combine(numbers, nh_eval, k2_eval, e2_this, e2_other, a.mode, a.tail_void, a.repro_void, a.zerograd_void)
    except Refused as e:
        print(f"[k2dc02nh-lever] REFUSED: {e}", file=sys.stderr)
        return 4
    res["seeds"] = numbers
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "LEVER_READING.json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "LEVER_READING.md").write_text(render(res), encoding="utf-8")
    print(f"[k2dc02nh-lever] {res['mode']} reading {res['reading']} (sec. 4 item {res['sec4_item']}, rule outcome {res['rule_outcome']}) -> {out / 'LEVER_READING.json'}")
    return 5 if res["reading"] == "VOID" else 0


if __name__ == "__main__":
    sys.exit(main())
