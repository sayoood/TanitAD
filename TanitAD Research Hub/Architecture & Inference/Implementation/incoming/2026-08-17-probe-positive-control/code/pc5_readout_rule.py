"""PC5 — ⛔ THE APPARATUS DEFECT THE POSITIVE CONTROL EXPOSED.

WHAT PC1/CHAIN-A FOUND. Handing the probe a memory tensor that is a direct
encoding of the frame's OWN GT BOXES gives ``lead_gap_abs_err_m = 10.18 m``
against a constant's 5.13 m — the positive control FAILS K1 harder than the
real v6 arms do. A representation that literally contains the answer does not
let this apparatus beat a constant.

⭐ WHY, AND IT IS NOT "THE PROBE CANNOT LEARN". On that same oracle arm the
probe's own banked diagnostic reads ``_diag_oracle_slot_abs_err_m`` median
**0.713 m** — the best in-corridor slot IS on the lead — and **K2 separates in
the CORRECT direction for the first time anywhere in F-18** (arm −0.75 m better
than its own C-SHUF twin, separated). The head reads its input and decodes the
boxes. What fails is the READOUT RULE that turns 74 slots into one number.

⛔ THE RULE IS NOT THE SAME FUNCTION ON BOTH SIDES, despite ``sp2_probe.py``'s
own section header saying "applied IDENTICALLY to prediction and to GT":

    GT   (``gt_lead_gap``): cx > 0  AND  |cy| <= 1.75  AND  **cx <= 30**
                            -> **min cx**              (the NEAREST agent)
    PRED (``pred_lead``):   cx > 0  AND  |cy| <= 1.75  (**no 30 m cap**)
                            -> **argmax presence**     (the MOST CONFIDENT one)

"Nearest" and "most confident" are different selections, and the decode range
runs to 60 m while the GT stratum stops at 30 m. A head that correctly and
confidently detects a car at 45 m will therefore be scored against a GT lead at
8 m — the error is the rule's, not the representation's.

THIS FILE CHANGES NOTHING IN ``sp2_probe.py``. It re-reads the BANKED heads and
applies alternative readout rules to the SAME slot outputs, on the SAME windows,
with the SAME estimator, and reports every rule side by side — including the
incumbent, which must reproduce the banked headline or the run refuses.

THE RULES (declared before the numbers):
  R0  incumbent — argmax presence over in-corridor slots (no range cap)
  R1  nearest in-corridor slot with cx <= 30 m, NO presence gate
      ⚠️ a head that SCATTERS slots gets this cheaply; it is the geometric
      ceiling, not a fair rule. Reported with the null arm beside it.
  R2  nearest in-corridor slot with cx <= 30 m AND presence >= tau
      the symmetric rule: same predicate as the GT, same tie-break, plus the
      confidence the GT does not need. Swept over tau.

⛔ AND THE CONTROL THAT DECIDES WHETHER A RULE IS A FIX OR A CHEAT: every rule
is run on the RANDOM-LATENT NULL head as well. A rule that rescues the oracle
AND the null has rescued nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp2_probe as SP                                          # noqa: E402
from taniteval.ci import (episode_cluster_bootstrap,             # noqa: E402
                          paired_episode_cluster_bootstrap)
from tanitad.models.agent_slots import (AgentSlotDecoder,        # noqa: E402
                                        SlotDecodeRanges)

CORRIDOR_M = SP.CORRIDOR_M
LEAD_MAX_M = SP.LEAD_MAX_M


def slot_dump(head, rows, ev_idx, device, batch=64):
    """One forward over the eval windows -> (cx, cy, presence) per slot."""
    head.eval()
    cxs, cys, ps = [], [], []
    with torch.no_grad():
        for s in range(0, len(ev_idx), batch):
            js = ev_idx[s:s + batch]
            mem = torch.stack([rows[i]["cells"] for i in js]).to(device).float()
            out = head(mem)
            cxs.append(out["box"][..., 0].cpu())
            cys.append(out["box"][..., 1].cpu())
            ps.append(torch.sigmoid(out["presence_logit"]).cpu())
    return (torch.cat(cxs).numpy(), torch.cat(cys).numpy(),
            torch.cat(ps).numpy())


def apply_rule(cx, cy, pres, rule, tau=0.5):
    """-> (gap, emitted). ``rule`` in {'R0', 'R1', 'R2'}."""
    ok = (cx > 0) & (np.abs(cy) <= CORRIDOR_M)
    if rule == "R0":                       # incumbent: argmax presence
        score = np.where(ok, pres, -1.0)
        best = score.argmax(axis=1)
        return cx[np.arange(cx.shape[0]), best], ok.any(axis=1)
    ok = ok & (cx <= LEAD_MAX_M)           # R1/R2 also apply the GT's cap
    if rule == "R2":
        ok = ok & (pres >= tau)
    big = np.where(ok, cx, np.inf)
    best = big.argmin(axis=1)
    return cx[np.arange(cx.shape[0]), best], ok.any(axis=1)


def score(gap, emit, gt, eid, base_common, const_m, epmean, n_boot):
    sel = base_common & emit
    n = int(sel.sum())
    if n < 30:
        return {"n_windows": n, "note": "fewer than 30 windows — no claim"}
    e_arm = np.abs(gap[sel] - gt[sel])
    e_con = np.abs(const_m - gt[sel])
    e_ep = np.abs(epmean[sel] - gt[sel])
    ee = eid[sel]
    arm = episode_cluster_bootstrap(e_arm, ee, n_boot=n_boot)
    k1 = paired_episode_cluster_bootstrap(e_arm, e_con, ee, n_boot=n_boot)
    k5 = paired_episode_cluster_bootstrap(e_arm, e_ep, ee, n_boot=n_boot)
    return {"n_windows": n, "n_clusters": int(len(np.unique(ee))),
            "abstained_of_base": int(base_common.sum() - n),
            "err_m": round(float(e_arm.mean()), 4),
            "err_ci": [arm["lo"], arm["hi"]],
            "median_m": round(float(np.median(e_arm)), 4),
            "K1_delta": k1["delta"], "K1_lo": k1["lo"], "K1_hi": k1["hi"],
            "K1_separated": k1["separated"],
            "K1_PASSES": bool(k1["separated"] and k1["delta"] < 0),
            "K5_delta": k5["delta"], "K5_lo": k5["lo"], "K5_hi": k5["hi"],
            "K5_separated": k5["separated"],
            "K5_PASSES": bool(k5["separated"] and k5["delta"] < 0),
            "pred_mean_m": round(float(gap[sel].mean()), 3),
            "pred_sd_m": round(float(gap[sel].std()), 3),
            "gt_mean_m": round(float(gt[sel].mean()), 3),
            "gt_sd_m": round(float(gt[sel].std()), 3),
            "corr_pred_gt": round(float(np.corrcoef(gap[sel], gt[sel])[0, 1]), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--expect-json", default=None,
                    help="banked sp2 result; R0 must reproduce its headline")
    ap.add_argument("--n-queries", type=int, default=74)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    decl = json.loads(Path(a.split_json).read_text("utf-8"))
    eval_clips, train_clips = set(decl["eval_clips"]), set(decl["train_clips"])
    ev_idx = [i for i, r in enumerate(rows) if r["clip_id"] in eval_clips]
    tr_idx = [i for i, r in enumerate(rows) if r["clip_id"] in train_clips]
    eid = np.array([rows[i]["episode_uid"] for i in ev_idx])
    gt = np.array([SP.gt_lead_gap(rows[i]["agents"])
                   if SP.gt_lead_gap(rows[i]["agents"]) is not None
                   else np.nan for i in ev_idx])
    has_gt = ~np.isnan(gt)
    const_m = float(np.median(np.array(
        [g for g in (SP.gt_lead_gap(rows[i]["agents"]) for i in tr_idx)
         if g is not None])))
    ep_of = np.array([rows[i]["clip_id"] for i in ev_idx])
    epmean = np.full(len(ev_idx), const_m, dtype=np.float64)
    for _c in np.unique(ep_of):
        m = (ep_of == _c) & has_gt
        pos = np.nonzero(m)[0]
        if pos.size == 0:
            continue
        tot = float(np.sum(gt[pos]))
        for k in pos:
            epmean[k] = ((tot - gt[k]) / (pos.size - 1)) if pos.size > 1 \
                else const_m

    d_mem = int(rows[0]["cells"].shape[-1])
    n_mem = int(rows[0]["cells"].shape[0])
    head = AgentSlotDecoder(d_mem, n_mem, n_queries=int(a.n_queries),
                            d_model=256, depth=3, n_heads=8,
                            ranges=SlotDecodeRanges(),
                            enforce_band=False).to(device)
    head.load_state_dict(torch.load(a.head, map_location="cpu",
                                    weights_only=False)["head"])
    cx, cy, pres = slot_dump(head, rows, ev_idx, device)

    g0, e0 = apply_rule(cx, cy, pres, "R0")
    base_common = has_gt & e0        # the incumbent's own scored set
    out = {"_evidence_class": "MEASURED (ours; alternative READOUT RULES applied "
                              "to the BANKED head's own slot outputs — the fit, "
                              "the windows, the split and the estimator are the "
                              "parity run's)",
           "eval_tier": "T0-DIAGNOSTIC", "arm": a.label,
           "run_stamp": meta.get("run_stamp"), "seed": a.seed,
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
           "n_boot": a.n_boot, "forbidden": "overlapping_holdout_se",
           "rule_defect": {
               "gt_rule": "cx>0 & |cy|<=1.75 & cx<=30 -> MIN cx (nearest)",
               "incumbent_pred_rule":
                   "cx>0 & |cy|<=1.75 (NO 30 m cap) -> ARGMAX presence",
               "_read": "different predicate AND different selection"},
           "n_gt_lead_windows": int(has_gt.sum()),
           "n_base_common": int(base_common.sum()),
           "rules": {}}

    out["rules"]["R0_incumbent_argmax_presence"] = score(
        g0, e0, gt, eid, base_common, const_m, epmean, a.n_boot)
    g1, e1 = apply_rule(cx, cy, pres, "R1")
    out["rules"]["R1_nearest_no_gate"] = score(
        g1, e1, gt, eid, base_common, const_m, epmean, a.n_boot)
    for t in a.taus:
        g2, e2 = apply_rule(cx, cy, pres, "R2", tau=t)
        out["rules"][f"R2_nearest_presence_ge_{t:g}"] = score(
            g2, e2, gt, eid, base_common, const_m, epmean, a.n_boot)

    # ---- the reproduction gate on R0 ---------------------------------------
    if a.expect_json:
        exp = json.loads(Path(a.expect_json).read_text("utf-8"))
        want = round(float(
            exp["per_arm"]["cells"]["lead_gap_abs_err_m"]["mean"]), 3)
        got = round(float(out["rules"]["R0_incumbent_argmax_presence"]
                          ["err_m"]), 3)
        out["reproduction_gate"] = {"R0_err_m": got, "banked_err_m": want,
                                    "note": "R0 is scored on the incumbent's own "
                                            "window set, so it must match the "
                                            "banked headline"}
        if abs(got - want) > 0.02:
            raise SystemExit(f"[pc5] ⛔ R0 does not reproduce the banked "
                             f"headline: {got} vs {want}")
        print(f"[pc5] R0 reproduction gate PASSED ({got} vs {want})", flush=True)

    Path(a.out).write_text(json.dumps(out, indent=1), "utf-8")
    for k, v in out["rules"].items():
        if "err_m" in v:
            print("  %-34s n=%-5d err=%7.3f  K1=%+7.3f [%+.3f,%+.3f] %-9s "
                  "K5=%+7.3f %-9s r=%+.3f" %
                  (k, v["n_windows"], v["err_m"], v["K1_delta"], v["K1_lo"],
                   v["K1_hi"], "K1 PASS" if v["K1_PASSES"] else "K1 fail",
                   v["K5_delta"], "K5 PASS" if v["K5_PASSES"] else "K5 fail",
                   v["corr_pred_gt"]), flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
    print(f"[pc5] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
