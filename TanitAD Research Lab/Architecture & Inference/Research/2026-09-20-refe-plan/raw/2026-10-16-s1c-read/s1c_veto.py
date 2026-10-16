#!/usr/bin/env python3
"""s1c_veto.py -- arm V-S1C of eval/PREREG_AUXVETO.md (amendment A1): the S1c head's C3 rasters through the aux-veto rule, UNCHANGED.

It is a thin caller of raw/2026-10-16-auxveto-read/auxveto_read.py (the read of 2026-10-16, imported, not copied): the world (1,058 tokens, 64 proposals, the NAVSIM v1.1 table), the label
store's own signed-distance transform, geodac's footprint sampling (margin -0.25 m), the pick rule, the stratified estimator 529/12,146 * Delta_Z + 11,617/12,146 * Delta_N with the paired log-cluster
bootstrap (123 logs, 10,000 resamples, seed 20260927), and the section 8 / 9 outputs.  What is NEW here is only the raster source (a head_pred npz of the forward tool, mask = logit > 0) and the
amendment's identity check:  the read reproduces S1c's OWN C3 P1 IoU [0, 40) from aux_s1_readout (`--expect-iou`, 1e-6).

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I s1c_veto.py --head-pred H1u10k-s0=head_pred_H1u10k-s0.npz --label V-S1C --expect-iou 0.xxxxxx --out veto.json

Validity (section 9, any failure => the V-S1C row is INVALID, recorded, never silently dropped):
  1 B reproduces `pick` 1,058/1,058 | 0 the population identity (Z all NC x DAC = 0, N all > 0) | 2 [replaced by A1] the mask's IoU equals the harness's P1 | 3 zero control (an all-drivable raster vetoes
  nothing, Delta exactly 0.0, both geometries) | 4 O-DAC's Z recovery >= every raster arm's | 5 the y-flipped V-GT falls below V-GT.
INSTRUMENT identity (mine, not the prereg's; a failure is a refusal of the TOOL, not a verdict): V-GT, O-DAC and (dry-run) V-PRED reproduce the banked AUXVETO_READ.json to 1e-9.
Evidence class: MEASURED (this script on the artifacts named in the output).  Tier: NAVSIM v1.1 non-reactive table lookup, base = u400 (SFT-4 arm B).  STRATEGIC: PAUSED (PI 2026-09-19).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent
for _p in (HERE, PK / "raw/2026-10-16-auxveto-read", PK / "eval", PK / "refe", PK / "raw/2026-10-16-auxveto-feasibility"):
    sys.path.insert(0, str(_p))
import auxveto_read as AV  # noqa: E402  -- the read of 2026-10-16 (imported: its functions are the instrument)
import s1c_rules as R  # noqa: E402

TOOL = "s1c_veto/1"
BANKED_READ = PK / "raw/2026-10-16-auxveto-read/AUXVETO_READ.json"
IOU_TOL = 1e-6                                           # amendment A1: 'reproduces S1c's own C3 P1 IoU'; the same tolerance as section 3's identity check
INSTR_TOL = 1e-9
MARGIN_CURVE = (0.0, 0.25, 0.5, 1.0, 2.0)
BLOCK_KEYS = ("point", "lo", "hi", "z", "z_lo", "z_hi", "n", "n_lo", "n_hi")


class VetoRefused(RuntimeError):
    pass


def sha256_crlf_form(path) -> str:
    """sha256 of the file with every line ending normalised to CRLF (the form the read of 2026-10-16 pinned for tokens.json)."""
    import hashlib
    b = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(b).hexdigest()


# ================================================================================================================ the raster source
def load_head_masks(path: str, tokens: list) -> tuple:
    """head_pred npz (forward tool: log, token, step, logit [N,40,32], sdf_m) -> ({token: bool mask [40,32]}, sha256).  Mask = logit > 0 (section 3); every table token exactly once."""
    z = np.load(path, allow_pickle=False)
    for k in ("token", "logit"):
        if k not in z.files:
            raise VetoRefused(f"{path}: missing array {k!r}")
    tok = [str(t) for t in z["token"]]
    lg = np.asarray(z["logit"])
    step = np.asarray(z["step"]) if "step" in z.files else np.zeros(len(tok), int)
    if lg.shape != (len(tok), 40, 32):
        raise VetoRefused(f"{path}: logit {lg.shape}, want ({len(tok)}, 40, 32)")
    if not (step == 0).all() or len(set(tok)) != len(tok):
        raise VetoRefused(f"{path}: duplicate tokens or step != 0 rows")
    if set(tok) != set(tokens):
        raise VetoRefused(f"{path}: covers {len(set(tok) & set(tokens))} of the {len(tokens)} table tokens (extra {len(set(tok) - set(tokens))})")
    idx = {t: i for i, t in enumerate(tok)}
    return {t: AV.pred_mask(lg[idx[t]], None) for t in tokens}, AV.sha256_file(path)


# ================================================================================================================ the read
def _blk(b: dict) -> dict:
    return {k: float(b["d_pdms"][k]) for k in BLOCK_KEYS}


def zero_control(w, log_ix, counts) -> dict:
    zc = {}
    for nm, shape, cell, geom in (("pred_grid_40x32", (40, 32), 2.0, AV.PRED_GEOM), ("gt_grid_160x128", (160, 128), 0.5, AV.GT_GEOM)):
        masks = {t: np.ones(shape, bool) for t in w.token}
        v = AV.vetoed(AV.margins_from_masks(w, masks, cell, geom), AV.MARGIN)
        picks = AV.all_picks(w, v)
        res = AV.strat_boot(np.stack([AV.deltas(w, picks)["pdms"]]), w.is_z, log_ix, counts)
        ok, d = AV.check_zero_control(int(v.sum()), float(res["point"][0]), bool((picks == w.b_pick).all()))
        zc[nm] = {"ok": ok, **d}
    return {"ok": all(v["ok"] for v in zc.values()), **zc}


def run_veto(arm_masks: dict, arm_label: str, expect_iou: float, *, dry_run_arms: dict | None = None, prereg_text: str | None = None, log=print) -> dict:
    """arm_masks {token: bool [40,32]} = the head under test (V-S1C in the real run).  dry_run_arms {label: masks} = additional arms read through the same code (the dry run's V-PRED identity)."""
    t0 = time.time()
    # NOTE: AV.assert_prereg() is NOT called: it is bound to the pre-amendment file sha; the amended file is guarded by s1c_rules.guard_prereg_auxveto (main) instead.
    # tokens.json is pinned on the dev box's CRLF working-tree form (`03072566...`); a git archive gives the LF form: compare the CRLF-NORMALISED bytes (the same list either way)
    if not sha256_crlf_form(AV.C3 / "tokens.json").startswith(AV.TOKENS_SHA_PREFIX):
        raise VetoRefused(f"tokens.json (CRLF form) sha256 {sha256_crlf_form(AV.C3 / 'tokens.json')[:16]} != {AV.TOKENS_SHA_PREFIX}")
    if not AV.sha256_file(AV.C3 / "table.npz").startswith(AV.TABLE_SHA_PREFIX):
        raise VetoRefused(f"table.npz sha256 {AV.sha256_file(AV.C3 / 'table.npz')[:16]} != {AV.TABLE_SHA_PREFIX}")
    store_man = json.load(open(AV.STORE / "manifest.json"))
    if not store_man["store_sha256"].startswith(AV.STORE_SHA_PREFIX):
        raise VetoRefused(f"label store sha256 {store_man['store_sha256'][:16]} != {AV.STORE_SHA_PREFIX}")
    w, b_banked = AV.load_world()
    gt_masks = AV.load_gt_masks(w)
    log_ix, counts = AV._log_counts(w)
    banked = json.load(open(BANKED_READ))
    val = {}
    # ---- section 9 gates 1, 0, 2 (replaced by A1), 3
    ok1, d1 = AV.check_b_reproduces(w.b_pick, b_banked)
    val["1_B_reproduces_pick"] = {"ok": ok1, **d1}
    ncdac = w.sub[np.arange(w.n), w.b_pick, 0] * w.sub[np.arange(w.n), w.b_pick, 1]
    val["0_population_identity"] = {"ok": bool((ncdac[w.is_z] == 0).all() and (ncdac[~w.is_z] > 0).all() and w.is_z.sum() == AV.N_Z and (~w.is_z).sum() == AV.N_Z),
                                    "Z": int(w.is_z.sum()), "N_sample": int((~w.is_z).sum())}
    gt2 = np.stack([AV.gt_mask_to_2m(gt_masks[t]) for t in w.token])
    iou = AV.iou_pooled(np.stack([arm_masks[t] for t in w.token]), gt2)
    val["2_A1_mask_iou_equals_harness_P1"] = {"ok": bool(abs(iou - expect_iou) <= IOU_TOL), "iou_veto_read": iou, "iou_harness_P1_0_40": expect_iou, "abs_diff": abs(iou - expect_iou), "tol": IOU_TOL}
    val["3_zero_control"] = zero_control(w, log_ix, counts)
    # ---- the arms
    margins = {"V-GT": AV.margins_from_masks(w, gt_masks, 0.5, AV.GT_GEOM), "V-GT-YFLIP": AV.margins_from_masks(w, gt_masks, 0.5, AV.GT_GEOM, transform=lambda m: m[:, ::-1]),
               arm_label: AV.margins_from_masks(w, arm_masks, 2.0, AV.PRED_GEOM)}
    for lab, mk in (dry_run_arms or {}).items():
        margins[lab] = AV.margins_from_masks(w, mk, 2.0, AV.PRED_GEOM)
    vetoes = {k: AV.vetoed(v, AV.MARGIN) for k, v in margins.items()}
    vetoes["O-DAC"] = AV.odac_veto(w)
    picks = {k: AV.all_picks(w, v) for k, v in vetoes.items()}
    blocks = {k: AV.arm_block(w, vetoes[k], picks[k], log_ix, counts) for k in picks}
    rec = {k: float(blocks[k]["z_recovery"]["ncdac_count"]) for k in blocks}
    raster = {k: rec[k] for k in blocks if k not in ("O-DAC", "V-GT-YFLIP")}
    ok4, d4 = AV.check_odac_bounds(rec["O-DAC"], raster)
    val["4_odac_bounds_rasters"] = {"ok": ok4, **d4}
    ok5, d5 = AV.check_yflip_below(blocks["V-GT"]["d_pdms"]["point"], blocks["V-GT-YFLIP"]["d_pdms"]["point"])
    val["5_yflip_mutation_below_vgt"] = {"ok": ok5, **d5}
    val["all_pass"] = all(v["ok"] for v in val.values() if isinstance(v, dict) and "ok" in v)
    # ---- the instrument identity against the banked read (a tool check, not a prereg gate)
    ident = {}
    for lab in ("V-GT",):
        ident[lab] = {k: abs(blocks[lab]["d_pdms"][k] - banked["arms"][lab]["d_pdms"][k]) for k in BLOCK_KEYS}
    ident["O-DAC"] = {"z_recovery_abs_diff": abs(rec["O-DAC"] - banked["odac_z_recovery"]), "point_abs_diff": abs(blocks["O-DAC"]["d_pdms"]["point"] - banked["arms"]["O-DAC"]["d_pdms"]["point"])}
    for lab in (dry_run_arms or {}):
        if lab in banked["arms"]:
            ident[lab] = {k: abs(blocks[lab]["d_pdms"][k] - banked["arms"][lab]["d_pdms"][k]) for k in BLOCK_KEYS}
    worst = max(v for d in ident.values() for v in d.values())
    instrument = {"ok": bool(worst <= INSTR_TOL), "max_abs_diff": worst, "tol": INSTR_TOL, "per_arm": ident, "banked": str(BANKED_READ.relative_to(PK)), "banked_prereg_sha256": banked["prereg"]["sha256"]}
    # ---- the decision, applied mechanically (V-S1C in V-PRED's role; V-GT by the same estimator)
    prereg_text = prereg_text if prereg_text is not None else R.read_lf(PK / "eval/PREREG_AUXVETO.md")
    dec = R.veto_row(_blk(blocks[arm_label]), _blk(blocks["V-GT"]), prereg_text)
    if not val["all_pass"]:
        dec = dict(dec, row_if_valid=dec["row"], row="INVALID", invalid_checks=[k for k, v in val.items() if isinstance(v, dict) and v.get("ok") is False])
    # ---- reported, never gating
    curve = {}
    for arm in (arm_label, "V-GT"):
        curve[arm] = {}
        for m in MARGIN_CURVE:
            v = AV.vetoed(margins[arm], m)
            pk = AV.all_picks(w, v)
            bl = AV.arm_block(w, v, pk, log_ix, counts)
            curve[arm][f"{m:g}"] = {k: bl[k] for k in ("d_pdms", "z_recovery", "false_veto_N", "all_vetoed_tokens", "changed_picks", "n_vetoed_proposals", "d_dac", "d_nc", "d_ep")}
    tact = AV.tactical_levels(w, log_ix, counts)
    cities = {k: AV.city_block(w, picks[k], w.b_pick) for k in (arm_label, "V-GT")}
    out = {"tool": TOOL, "arm": arm_label, "tier": "NAVSIM v1.1 non-reactive table lookup (the programme's NAVSIM tier); base = u400 (SFT-4 arm B)",
           "inputs": {"tokens_sha256_crlf_form": sha256_crlf_form(AV.C3 / "tokens.json"), "table_sha256": AV.sha256_file(AV.C3 / "table.npz"), "label_store_sha256": store_man["store_sha256"],
                      "geodac_sha256": AV.sha256_file(PK / "raw/2026-10-16-auxveto-feasibility/geodac.py"), "auxveto_read_sha256": AV.sha256_file(AV.__file__)},
           "estimator": {"kind": "paired log-cluster bootstrap, stratified (529/12146 Z census + 11617/12146 N sample)", "n_logs": int(counts.shape[1]), "resamples": AV.B_BOOT, "seed": AV.SEED,
                         "weights": {"Z": AV.W_Z, "N": AV.W_N}, "ci": "95 % percentile",
                         "question_answered": "would another draw of LOGS say this?  BLIND to the head's TRAINING variance (one seed, H-ESTIM-SEED-1) and to the tracker; raster inference variance is zero (deterministic)"},
           "validity": val, "instrument_identity": instrument, "decision": dec, "arms": blocks, "margin_curve": curve, "tactical_levels": tact, "odac_z_recovery": rec["O-DAC"],
           "per_city_descriptive": cities, "strategic": "PAUSED (PI 2026-09-19, PI_DECISION_QUEUE.md item 25): not a pass, not omitted", "wall_s": time.time() - t0}
    log(f"s1c_veto: {arm_label} dPDMS {blocks[arm_label]['d_pdms']['point']:+.4f} [{blocks[arm_label]['d_pdms']['lo']:+.4f}, {blocks[arm_label]['d_pdms']['hi']:+.4f}]  "
        f"V-GT {blocks['V-GT']['d_pdms']['point']:+.4f}  row {dec['row']}  valid {val['all_pass']}  instrument {instrument['ok']}  {out['wall_s']:.0f} s")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--head-pred", required=True, help="ARM=head_pred.npz")
    ap.add_argument("--label", default="V-S1C")
    ap.add_argument("--expect-iou", type=float, required=True, help="the harness's P1 [0,40) point estimate of this head on the C3 tokens (amendment A1 identity)")
    ap.add_argument("--with-vpred", action="store_true", help="DRY RUN: also read the probe's V-PRED units through this code and require the banked V-PRED block (instrument identity)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    name, _, path = a.head_pred.partition("=")
    try:
        R.guard_prereg_auxveto(PK / "eval/PREREG_AUXVETO.md")
        w, _b = AV.load_world()
        masks, sha = load_head_masks(path, w.token)
        extra = None
        if a.with_vpred:
            extra = {"V-PRED": AV.load_pred_masks(w, AV.UNIT_DIRS["V-PRED"])[0]}
        out = run_veto(masks, a.label, a.expect_iou, dry_run_arms=extra)
    except (VetoRefused, ValueError) as e:
        print(f"ZZS1CVETO_REFUSED {e}", flush=True)
        return 3
    out["inputs"]["head_pred_sha256"] = sha
    out["inputs"]["head_pred_arm"] = name
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(AV.cj(out), f, indent=1)
    if not out["instrument_identity"]["ok"]:
        print(f"ZZS1CVETO_INSTRUMENT_IDENTITY_FAILED max_abs_diff {out['instrument_identity']['max_abs_diff']}", flush=True)
        return 6
    print(f"ZZS1CVETO_OK {a.label} row {out['decision']['row']} valid {out['validity']['all_pass']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
