"""s1_eval_read.py -- note N8 section 2 (PREREG_V8F_PROOF_ADD12): the r-a read of a TRAINED level-1 state on the eval bank frames. Fills gap G1 of S1_DEFS.md.

  python s1_eval_read.py run --states <ARM>_eval.npz --tag <ARM> --expect-dim 128 [--ckpt-sha256 HEX]

Out-of-sample for the model (eval clips). Bank rows, folds (by episode, 5, seed 0), probe (`fit_probe`), ego floor (`e_floor_ra`) and bootstrap (episode clusters, B 2,000, seed 0)
are those of stage 0 (spatial/s0/s0_read.py rules R2, R7, R8), by import. The state enters as given (float32, no fitting, no whitening). No outcome name of the card is assigned.
Controls first; if one fails the numbers are written, no label is printed, exit 2:
  v1 alignment   every bank row has exactly one state row with the same clip id and frame index.
  v2 identity    the ego floor of this tool on these rows reproduces the stage 0b value (lead gap R2 0.1327) at 4 decimals: same rows, folds, labels and floor code.
  v3 permuted    lead gap labels permuted within fold: the state's gap R2 at or below 0.02.
  v4 declared    state width equals --expect-dim; the record's state mode is printed; the record's checkpoint sha256 equals --ckpt-sha256 when given.
Tier T0: a world-model diagnostic, never driving performance.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "s0"))
import s0_read as S0  # noqa: E402

RP = S0.RP
REF_EGO_GAP_R2 = 0.1327          # stage 0b (VERDICT_S0.md), quoted in notes N4 / N7
REF_UNTRAINED = {"grid_gap_r2": 0.3981, "mean_gap_r2": 0.2577, "ego_gap_r2": 0.1327, "grid_presence_auc": 0.761, "mean_presence_auc": 0.742, "ego_presence_auc": 0.596}
PERM_BAR = 0.02
FAMILIES = "LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass)"


def load_bank():
    Z = np.load(S0.BANK)
    beid = Z["beid"].astype(str); bg = Z["bg"].astype(np.int64); E = Z["E"].astype(np.float64)
    keep = np.isfinite(E).all(1)                                             # S0 R2: rows with a complete ego vector
    return {"beid": beid[keep], "bg": bg[keep], "E": E[keep], "gap": Z["lead_gap"].astype(np.float64)[keep], "pres": Z["lead_present"].astype(np.float64)[keep]}


def align(bank, st):
    clips = st["clips"].astype(str); ci = st["clip_idx"].astype(np.int64); t = st["t"].astype(np.int64)
    key = {}
    dup = 0
    for r in range(len(t)):
        k = (clips[ci[r]], int(t[r]))
        dup += k in key
        key[k] = r
    rows = np.array([key.get((c, int(g)), -1) for c, g in zip(bank["beid"], bank["bg"])], np.int64)
    return rows, {"bank_rows": int(len(rows)), "missing": int((rows < 0).sum()), "duplicate_state_keys": int(dup), "pass": bool((rows >= 0).all() and dup == 0)}


def auc_row(ep, y, sa, sb, C):
    ba, bb = RP.boot_auc(ep, y, sa, C), RP.boot_auc(ep, y, sb, C)
    return RP.diff_row(ba, bb, RP.auc_pooled(y, sa), RP.auc_pooled(y, sb))


def cmd_run(a):
    t0 = time.time()
    st = np.load(a.states)
    rec = {}
    if os.path.exists(a.states + ".json"):
        with open(a.states + ".json", encoding="utf-8") as fh:
            rec = json.load(fh)
    bank = load_bank()
    rows, v1 = align(bank, st)
    res = {"tag": a.tag, "tier": "T0 (probes on a trained state; a world-model diagnostic, never driving performance)", "where": "eval bank frames, out-of-sample for the model",
           "estimator": "episode-cluster bootstrap, B 2000, seed 0, one draw shared; folds by episode (5, seed 0)", "families": FAMILIES, "controls": {"v1": v1}}
    D = int(st["Z"].shape[1])
    v4 = {"width": D, "expected": a.expect_dim, "record_state_mode": rec.get("wm_state"), "record_ckpt_sha256": rec.get("ckpt_sha256"),
          "ckpt_checked": bool(a.ckpt_sha256), "pass": bool(D == a.expect_dim and (not a.ckpt_sha256 or rec.get("ckpt_sha256") == a.ckpt_sha256))}
    res["controls"]["v4"] = v4
    out_dir = a.out_dir or HERE
    if not (v1["pass"] and v4["pass"]):
        res["controls_pass"] = False
        return finish(res, out_dir, a.tag, 2)
    S = st["Z"][rows].astype(np.float32)
    eid = bank["beid"]
    uniq = sorted(set(eid)); idx = {c: i for i, c in enumerate(uniq)}
    epi = np.array([idx[c] for c in eid], np.int64); n_ep = len(uniq)
    folds = RP.make_folds(eid)
    C = RP.boot_counts(n_ep)
    gap, pres, E = bank["gap"], bank["pres"], bank["E"]
    vg = np.isfinite(gap); va = np.isfinite(pres)
    res["counts"] = {"episodes": n_ep, "frames": int(len(eid)), "gap_frames": int(vg.sum()), "presence_frames": int(va.sum()), "state_width": D}
    # ---- ego floor (stage 0 code) and control v2
    eg, eg_d = S0.e_floor_ra(E, gap, vg, epi, folds, binary=False)
    ego_r2 = float(RP.r2_pooled(gap[vg], eg[vg]))
    res["controls"]["v2"] = {"ego_gap_r2": ego_r2, "reference": REF_EGO_GAP_R2, "chosen": eg_d["chosen"], "bar": "equal at 4 decimals", "pass": bool(round(ego_r2, 4) == REF_EGO_GAP_R2)}
    # ---- control v3
    yp = S0.permute_within_fold(gap, folds, 1)
    pp = S0.fold_cv_probe(S, yp, vg, epi, folds)
    r2p = float(RP.r2_pooled(yp[vg], pp[vg]))
    res["controls"]["v3"] = {"state_gap_r2_permuted": r2p, "bar": PERM_BAR, "pass": bool(r2p <= PERM_BAR)}
    res["controls_pass"] = bool(all(c["pass"] for c in res["controls"].values()))
    # ---- reads
    pg = S0.fold_cv_probe(S, gap, vg, epi, folds)
    pa = S0.fold_cv_probe(S, pres, va, epi, folds)
    ea, ea_d = S0.e_floor_ra(E, pres, va, epi, folds, binary=True)
    res["ra"] = {
        "state_gap": S0.single_r2(epi[vg], gap[vg], pg[vg], C, n_ep), "ego_gap": S0.single_r2(epi[vg], gap[vg], eg[vg], C, n_ep),
        "state_minus_ego_gap": S0.paired(epi[vg], gap[vg], pg[vg], eg[vg], C, n_ep),
        "state_presence": S0.single_auc(epi[va], pres[va], pa[va], C), "ego_presence": S0.single_auc(epi[va], pres[va], ea[va], C),
        "state_minus_ego_presence": auc_row(epi[va], pres[va], pa[va], ea[va], C), "ego_floor_gap": eg_d["chosen"], "ego_floor_presence": ea_d["chosen"]}
    res["untrained_references_stage_0b"] = REF_UNTRAINED
    np.savez(os.path.join(out_dir, f"OOF_S1EVAL_{a.tag}.npz"), epi=epi, folds=folds, gap=gap, pres=pres, p_gap=pg, p_pres=pa, e_gap=eg, e_pres=ea)
    res["seconds"] = round(time.time() - t0, 1)
    return finish(res, out_dir, a.tag, 0 if res["controls_pass"] else 2)


def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def finish(res, out_dir, tag, rc):
    with open(os.path.join(out_dir, f"RESULT_S1EVAL_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=_js)
    L = [f"# VERDICT_S1EVAL_{tag}: note N8, r-a read of a trained state on the eval bank", f"Tier {res['tier']}. Where: {res['where']}. Estimator: {res['estimator']}.", res["families"], "", "## CONTROLS (first)"]
    for k, c in res["controls"].items():
        L.append(f"- {k}: {'PASS' if c['pass'] else 'FAIL'} :: {json.dumps(c, default=_js)}")
    L.append(f"- controls_pass: {res.get('controls_pass')}")
    if res.get("controls_pass") and "ra" in res:
        r = res["ra"]; n = res["counts"]
        L += ["", f"## Reads (trained state, width {n['state_width']}; {n['episodes']} episodes, {n['frames']} frames, lead gap on {n['gap_frames']}); label: trained state, tag {tag}; no outcome name of the card",
              f"- lead gap R2: state {json.dumps(r['state_gap'], default=_js)}; ego floor ({r['ego_floor_gap']}) {json.dumps(r['ego_gap'], default=_js)}",
              f"- state minus ego, gap R2: {json.dumps(r['state_minus_ego_gap'], default=_js)}",
              f"- lead presence AUROC: state {json.dumps(r['state_presence'], default=_js)}; ego floor ({r['ego_floor_presence']}) {json.dumps(r['ego_presence'], default=_js)}",
              f"- state minus ego, presence AUROC: {json.dumps(r['state_minus_ego_presence'], default=_js)}",
              f"- beside, not paired (different widths), the untrained states of stage 0b on the same frames: {json.dumps(res['untrained_references_stage_0b'])}"]
    elif "ra" in res:
        L += ["", "A control failed: numbers are in the JSON, nothing is quoted."]
    with open(os.path.join(out_dir, f"VERDICT_S1EVAL_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[4:4 + len(res["controls"]) + 2]))
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--states", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--expect-dim", type=int, required=True)
    r.add_argument("--ckpt-sha256", default="")
    r.add_argument("--out-dir", default="")
    a = ap.parse_args(argv)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
