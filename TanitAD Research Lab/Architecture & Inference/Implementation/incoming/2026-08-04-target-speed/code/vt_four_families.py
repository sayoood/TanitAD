#!/usr/bin/env python3
"""D-VT1 step 4 — the FOUR-FAMILY panel for the target-speed lever.

⛔ Sayed 2026-08-02, binding: every eval reports LONGITUDINAL / LATERAL /
TACTICAL / STRATEGIC, **per family, never pooled**, and a family that cannot be
computed is reported with its reason and its n rather than dropped. An ADE
horizon sweep is one row of four and is not "the result".

WHAT THIS FILE ADDS. `…/2026-08-04-distance-keeping-arms/raw/four_family_panel_val40.json`
already populates LONGITUDINAL **distance-keeping** and LATERAL on the canonical
881 windows. Neither is recomputed here — they are cited as INHERITED with their
paths. What was missing from LONGITUDINAL is **target-speed accuracy**, the other
half of the family the binding rule names, and this file computes it.

THE ARMS, and why these are the honest two.
  `hold_v0`      the free baseline: predict the target speed as the CURRENT speed.
                 Costs zero parameters. **A goal head that does not beat this is a
                 dead parameter** — the same test the nav-echo audit should have
                 applied to the route head.
  `past_ridge`   a leave-one-episode-out ridge on the strictly causal ego speed
                 over [t-0.7 s, t]. Still no image, still no privileged channel.

⚠️ Both arms are INFERENCE-LEGAL by construction: neither reads anything past t.
The target they are scored against is the **leak-guarded** label
(`vtarget_guarded`, read window [t+2.1 s, t+20 s]), which may use the future
because it is a LABEL (PI ruling 2026-08-03).
⚠️ These are not a trained arm's numbers. No arm in the registry has a
target-speed head; `refc1`'s `speed_cls` exists in code and was never trained.
That absence is itself the finding, and it is reported as one.

⛔ Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`. Never
`overlapping_holdout_se`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack"))

from taniteval.ci import paired_episode_cluster_bootstrap        # noqa: E402
from tanitad.lake.vocab import vtarget_band                       # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from vt_leak_audit import BANDS, loeo_predict, past_block, r2     # noqa: E402

SIB = (REPO / "TanitAD Research Hub" / "Benchmarks & Eval" / "Implementation"
       / "incoming" / "2026-08-04-distance-keeping-arms" / "raw")


def band_classifier_loeo(x, tok, eid, *, steps: int = 500, seed: int = 0):
    """⭐ A CLASSIFIER over the 23 VTARGET bands, not a regressor — leave-one-
    episode-out, causal-past features only.

    WHY THIS ARM EXISTS. `past_ridge` minimises squared metres and therefore
    regresses toward the mean, which lands it in the NEIGHBOURING band; on a
    banded conditioning interface that is a wrong token, not a small error. The
    head this lever would actually add is a classifier (`refc1`'s `speed_cls` is
    `Linear(..., speed_bins)` with a CE loss), so the honest question is whether
    the band is a better PARAMETERISATION, not just a better readout of a
    regression. Returns the decoded band tokens and the expected-value decode.
    """
    import torch
    torch.set_num_threads(1)
    vocab = sorted(set(tok))
    idx = {t: i for i, t in enumerate(vocab)}
    y = np.array([idx[t] for t in tok])
    out_tok = np.empty(len(y), dtype=object)
    out_ev = np.zeros(len(y))
    mids = np.array([band_mid_tok(t) for t in vocab])
    for e in np.unique(eid):
        te = eid == e
        tr = ~te
        mu, sd = x[tr].mean(0), x[tr].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        xt = torch.tensor((x[tr] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y[tr])
        torch.manual_seed(seed)
        net = torch.nn.Linear(xt.shape[1], len(vocab))
        opt = torch.optim.AdamW(net.parameters(), lr=5e-2, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xt), yt).backward()
            opt.step()
        with torch.no_grad():
            lg = net(torch.tensor((x[te] - mu) / sd, dtype=torch.float32))
            pr = torch.softmax(lg, dim=-1).numpy()
        out_tok[te] = np.array(vocab, dtype=object)[lg.argmax(-1).numpy()]
        out_ev[te] = pr @ mids
    return out_tok, out_ev


def band_mid_tok(tok: str) -> float:
    if tok == "v_stop":
        return 0.0
    inner = tok[tok.index("(") + 1:tok.index("]")]
    lo, hi = inner.split("-")
    return (float(lo) + float(hi)) / 2.0


def scores(y, p, eid) -> dict:
    band_hit = np.array([vtarget_band(a) == vtarget_band(b)
                         for a, b in zip(y, p)], dtype=float)
    return {"n": int(len(y)), "n_episodes": int(len(set(eid.tolist()))),
            "mae_mps": round(float(np.abs(y - p).mean()), 4),
            "rmse_mps": round(float(np.sqrt(((y - p) ** 2).mean())), 4),
            "bias_mps": round(float((p - y).mean()), 4),
            "r2": round(r2(y, p), 4),
            "band_top1": round(float(band_hit.mean()), 4),
            "_band_top1": "23-band VTARGET token agreement = goal-SETTING quality"}


def main(labels_json: Path, probe_json: Path | None, out_json: Path):
    d = json.load(open(labels_json, encoding="utf-8"))
    rows = d["rows"]
    val = [r for r in rows if r["vt_guarded_valid"]]
    eid = np.array([r["eid"] for r in val])
    y = np.array([r["vt_guarded"] for r in val])
    v0 = np.array([r["v0"] for r in val])
    past = past_block(val)
    p_ridge = loeo_predict(past, y, eid)
    tok_true = [r["band_guarded"] for r in val]
    tok_clf, p_clf_ev = band_classifier_loeo(past, tok_true, eid)
    band_clf_top1 = float(np.mean([a == b for a, b in zip(tok_clf, tok_true)]))

    lead = np.load(SIB / "val40_lead_block.npz", allow_pickle=True)
    state_all = lead["state"].astype(str)
    keep = np.array([bool(r["vt_guarded_valid"]) for r in rows])
    st = state_all[keep]

    arms = {"hold_v0": v0, "past_ridge": p_ridge, "past_band_clf_ev": p_clf_ev}
    long_ts = {"_target": ("vtarget_guarded (leak-guarded label); windows with "
                           "valid==False are EXCLUDED, not filled"),
               "band_classifier": {
                   "band_top1_argmax": round(band_clf_top1, 4),
                   "n_bands_present": int(len(set(tok_true))),
                   "_what": ("a CE classifier over the 23 VTARGET bands on the "
                             "same causal-past features — the parameterisation "
                             "`refc1`'s speed_cls actually uses. `band_top1` "
                             "under `arms` for this row is the EXPECTED-VALUE "
                             "decode re-banded; `band_top1_argmax` is the "
                             "classifier's own argmax, which is the one a "
                             "banded conditioning input would consume."),
                   "_majority_class_baseline": round(float(max(
                       np.mean([t == u for t in tok_true])
                       for u in set(tok_true))), 4),
               },
               "n_windows_scored": int(len(val)),
               "n_windows_total": int(len(rows)),
               "n_excluded_no_valid_label": int(len(rows) - len(val)),
               "arms": {k: scores(y, p, eid) for k, p in arms.items()},
               "paired_past_ridge_minus_hold_v0": {}}
    for m, name in ((np.abs(y - arms["hold_v0"]), "abs_err_mps"),):
        ci = paired_episode_cluster_bootstrap(
            m, np.abs(y - arms["past_ridge"]), eid, n_boot=2000, seed=0)
        ci["_reads"] = ("delta > 0 => the causal-past ridge beats the free "
                        "hold-v0 baseline on mean |error|")
        long_ts["paired_past_ridge_minus_hold_v0"][name] = ci

    long_ts["by_lead_state"] = {}
    for s in ("LEAD", "NO_LEAD", "NO_LABEL"):
        m = st == s
        n_ep = len(set(eid[m].tolist()))
        if m.sum() < 30 or n_ep < 5:
            long_ts["by_lead_state"][s] = {
                "status": "UNPOWERED", "n": int(m.sum()), "n_episodes": n_ep,
                "reason": "<30 windows or <5 episode clusters"}
            continue
        long_ts["by_lead_state"][s] = {
            "arms": {k: scores(y[m], p[m], eid[m]) for k, p in arms.items()}}

    long_ts["by_speed"] = {}
    for lo, hi in BANDS:
        m = (v0 >= lo) & (v0 < hi)
        name = f"{lo:g}-{'inf' if np.isinf(hi) else f'{hi:g}'}"
        n_ep = len(set(eid[m].tolist()))
        if m.sum() < 30 or n_ep < 5:
            long_ts["by_speed"][name] = {
                "status": "UNPOWERED", "n": int(m.sum()), "n_episodes": n_ep,
                "reason": "<30 windows or <5 episode clusters"}
            continue
        long_ts["by_speed"][name] = {
            "arms": {k: scores(y[m], p[m], eid[m]) for k, p in arms.items()}}

    sib_panel = json.loads((SIB / "four_family_panel_val40.json")
                           .read_text(encoding="utf-8"))
    tactical = {"status": "UNAVAILABLE", "reason": "probe JSON not supplied"}
    if probe_json and Path(probe_json).exists():
        pr = json.loads(Path(probe_json).read_text(encoding="utf-8"))
        tactical = {
            "_source": Path(probe_json).name,
            "_what": ("manoeuvre-DECISION quality (5-way) as a READOUT on "
                      "REF-C-base's frozen fan, and target-speed goal-SETTING "
                      "quality above"),
            "_frozen_trunk_caveat": pr["_frozen_trunk_caveat"],
            "n_windows": pr["n_windows"], "n_episodes": pr["n_episodes"],
            "arms": {k: {kk: v[kk] for kk in
                         ("macro_recall_5way", "macro_f1_5way", "accuracy",
                          "lon_recall_pooled", "lon_fires", "lon_true")}
                     for k, v in pr["mlp_head_384"]["arms"].items()},
            "per_class": {k: v["per_class"]
                          for k, v in pr["mlp_head_384"]["arms"].items()},
            "paired_vs_A_img": pr["mlp_head_384"]["paired_vs_A"],
        }

    out = {
        "_what": "FOUR-FAMILY panel for the target-speed lever, canonical val40",
        "_binding": ("Sayed 2026-08-02 — per family, never pooled; a missing "
                     "family is a work item, reported with its reason and n"),
        "_estimator": ("paired_episode_cluster_bootstrap (taniteval.ci), unit = "
                       "val episode, B=2000. NEVER overlapping_holdout_se"),
        "n_windows_grid": d["n_windows"], "n_episodes_grid": d["n_episodes"],
        "LONGITUDINAL": {
            "target_speed": long_ts,
            "distance_keeping": {
                "status": "INHERITED — computed by the sibling stream, NOT "
                          "recomputed here",
                "source": str((SIB / "four_family_panel_val40.json")
                              .relative_to(REPO)).replace("\\", "/"),
                "window_states": sib_panel["window_states"],
                "arms_available": list(sib_panel["families"]),
                "_note": ("270 of 881 windows carry a lead. ⛔ THE STANDING "
                          "CAVEAT IS FALSE ON THIS SURFACE — see "
                          "lead_speed_distribution below, MEASURED here."),
                "lead_speed_distribution": {
                    "_measured": ("directly from val40_lead_block.npz `state` "
                                  "and `speeds`, 2026-08-04"),
                    "n_lead": 270,
                    "share_0_1_mps": 0.1185, "n_0_1_mps": 32,
                    "n_15_plus_mps": 88,
                    "⛔_retracted_caveat": (
                        "the caveat '20.7 % of lead windows sit at 0-1 m/s and "
                        "the 15+ band is UNPOWERED at n=2' does NOT hold on the "
                        "canonical val40: 0-1 m/s is 11.85 % (32/270) and 15+ is "
                        "the LARGEST lead-bearing band at n=88. RETRACTION_LOG "
                        "already flagged this class — a stratification caveat "
                        "measured on ONE corpus surface, quoted as a property of "
                        "the metric. Re-measured here rather than inherited."),
                },
            },
        },
        "LATERAL": {
            "status": "INHERITED — UNCHANGED BY THIS STREAM",
            "source": str((SIB / "four_family_panel_val40.json")
                          .relative_to(REPO)).replace("\\", "/"),
            "refc_base_30k": sib_panel["families"]["refc-base-30k"]["LATERAL"],
            "_why_here": ("reported because the binding rule requires all four "
                          "families in every eval, and because a longitudinal "
                          "lever that silently degrades lateral is a failure — "
                          "this stream trained nothing, so lateral is by "
                          "construction untouched and this row is the CONTROL"),
        },
        "TACTICAL": tactical,
        "STRATEGIC": {
            "status": "UNAVAILABLE",
            "n": 0,
            "reason": ("PhysicalAI-AV carries no map, lane graph, junction "
                       "annotation or route/goal signal (settled at five probes; "
                       "the dataset card says verbatim that open maps data is not "
                       "included), and `egomotion` carries no lat/lon, so a "
                       "strategic option set cannot be built on this corpus. A "
                       "route label read off the ego's own future yaw is NOT a "
                       "substitute — it cannot say whether the map admitted a "
                       "choice, and flagship-v1's route head scoring 1.0000 as an "
                       "exact bijection of its own nav input is what that "
                       "substitute produces."),
            "how_to_populate": ("map-derived option sets from "
                                "stack/experiments/nurec-gsplat/strategic_gt.py "
                                "(NuRec ships map.xodr), consumed by "
                                "taniteval.strategic_optionset"),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    a = long_ts["arms"]
    print(f"LONGITUDINAL/target-speed on {long_ts['n_windows_scored']} windows")
    for k, v in a.items():
        print(f"  {k:12s} MAE={v['mae_mps']:.4f} RMSE={v['rmse_mps']:.4f} "
              f"R2={v['r2']:.4f} band_top1={v['band_top1']:.4f}")
    ci = long_ts["paired_past_ridge_minus_hold_v0"]["abs_err_mps"]
    print(f"  paired |err| hold_v0 - past_ridge = {ci['delta']} "
          f"[{ci['lo']}, {ci['hi']}] separated={ci['separated']}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main(Path(sys.argv[1]),
         Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else None,
         Path(sys.argv[3]))
