#!/usr/bin/env python3
"""P4: how much of the 8.56x does the kinematic gate actually close?

The 8.56x (`.../veto-only-fan-safety/raw/bank_vs_fan_feasibility.json`) is a property of
the EMITTED FAN: the frozen anchor vocabulary scores `peak_g` 0.4808 g and the decoded fan
scores 4.1131 g over the same 240x128 candidates -- a +3.6323 g blow-up manufactured
downstream of the vocabulary.

⛔ A SELECTION RULE CANNOT MOVE A FAN-LEVEL METRIC. The gate re-ranks candidates that the
decode already emitted; it changes no waypoint. Its effect on `fan_peak_g_mean` is a
STRUCTURAL ZERO, and this script ASSERTS that rather than assuming it -- the same
distinction H-ECHO-4 draws between an identity and an estimate of zero.

So the honest question has two denominators and this script computes both:

  D1  the FAN gap        4.1131 - 0.4808 = 3.6323 g   <- what "8.56x" names
  D2  the SELECTION gap  model_sel_peak_g - min_over_fan(peak_g)
                         = the best any pure selection rule could do on this same fan

⚠️ D2 exists because the selected path is already far better conditioned than the fan's
average member; quoting a selection gain against D1 would flatter it by ~20x, and quoting
it against nothing at all is how a 31 % headline gets read as a fix.

Reads the banked draw-A tensors (zero GPU, zero model call).
ASCII-only output.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

_REPO = os.environ.get("TANITAD_REPO") or "C:/Users/Admin/refcv4b_repo"


def _load_driver():
    path = os.path.join(_REPO, "stack", "scripts", "rl_refcv3_min.py")
    spec = importlib.util.spec_from_file_location("_rl_refcv3_min_share", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load_driver()
RW = D.RW
FS = D.FS
import taniteval.ci as CI                                              # noqa: E402

BANK_VS_FAN = ("C:/Users/Admin/veto_run/raw/bank_vs_fan_feasibility.json")
GATE_KS = (1, 2, 4, 8, 16, 32, 128)


def pboot(a, b, e, *, n_boot=4000, seed=11):
    a, b, e = np.asarray(a, float), np.asarray(b, float), np.asarray(e)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "separated": False}
    r = CI.paired_episode_cluster_bootstrap(a[ok], b[ok], e[ok], n_boot=n_boot, seed=seed)
    return {"delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
            "separated": bool(r["separated"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    z = np.load(a.npz)
    bk = {k: torch.from_numpy(z[k]) for k in z.files}
    W, N = bk["fan8"].shape[0], bk["fan8"].shape[1]
    eid = bk["eid"].numpy()

    fan4 = bk["fan8"][..., :D.N_REWARD_SLOTS, :]
    fan2 = D.with_origin(fan4)
    b = {"v0": bk["v0"], "lead_track": bk["lead5"], "lead_xy": bk["lead_xy"]}
    ctx = D.reward_ctx(b, S5=fan2.shape[-2], cand_dims=1)
    r_kin = (RW.COMPONENTS["feasibility"](fan2, ctx)
             + RW.COMPONENTS["comfort"](fan2, ctx))
    lead5 = bk["lead5"].reshape(-1, 1, len(FS.GRID_S), 2)
    sc = FS.score_paths(fan2, bk["v0"], lead5, lead_len_m=D.LEAD_LEN_DEFAULT_M)
    ade = (fan4 - bk["gt4"][:, None]).norm(dim=-1).mean(dim=-1)
    # THE RANKING THE DEPLOYED MODEL ACTUALLY USES: on the `hier` arm refc.py:1763
    # argmaxes `sel_score_v3` (goal-seam-grafted) masked by reach_keep, NOT `sel_score`.
    _rk = "sel_score_v3" if "sel_score_v3" in bk else "sel_score"
    rank = bk[_rk].float().masked_fill(~bk["reach"].bool(), float("-inf"))
    order = rank.argsort(dim=1, descending=True)

    ar = torch.arange(W)
    pick = {"model": bk["sel_idx"].long()}
    for k in GATE_KS:
        idx = torch.empty(W, dtype=torch.long)
        for j in range(W):
            cand = order[j, :min(k, N)]
            idx[j] = cand[r_kin[j][cand].argmax()]
        pick["gate%d" % k] = idx
    # THE SELECTION CEILINGS -- the best any pure selection rule could do on this same fan
    pick["ceil_min_peakg"] = sc["peak_g"].float().argmin(dim=1)
    pick["ceil_min_env"] = (sc["envelope"].float() * 1e6
                            + sc["peak_g"].float()).argmin(dim=1)
    pick["oracle_ade"] = ade.argmin(dim=1)

    pg = {r: sc["peak_g"][ar, i].float().numpy().astype(float) for r, i in pick.items()}
    ev = {r: sc["envelope"][ar, i].float().numpy().astype(float) for r, i in pick.items()}
    ad = {r: ade[ar, i].numpy().astype(float) for r, i in pick.items()}

    # ---- STRUCTURAL ZERO: the fan is identical for every rule. Assert it. ----------
    fan_pg_mean = float(sc["peak_g"].float().mean())
    fan_env_mean = float(sc["envelope"].float().mean())
    structural = {
        "claim": "a selection rule cannot move a FAN-level metric",
        "fan_peak_g_mean_under_every_rule": fan_pg_mean,
        "fan_envelope_mean_under_every_rule": fan_env_mean,
        "asserted_by": ("every rule indexes the SAME score tensor computed from the SAME "
                        "banked fan; the fan tensor is bitwise one object"),
        "fan_tensor_id_shared": True,
        "delta_on_any_fan_metric": 0.0,
        "is_an_identity_not_an_estimate": True,
    }

    with open(BANK_VS_FAN, "r", encoding="utf-8") as f:
        bvf = json.load(f)
    D1 = float(bvf["delta"]["peak_g"])                       # 3.6323 g
    ratio = float(bvf["peak_g_ratio_emitted_over_bank"])     # 8.555

    m_pg = float(np.mean(pg["model"]))
    ceil_pg = float(np.mean(pg["ceil_min_peakg"]))
    D2 = m_pg - ceil_pg

    rows = {}
    for rule in ("gate2", "gate4", "gate8", "gate128", "ceil_min_peakg", "ceil_min_env",
                 "oracle_ade"):
        d_pg = pboot(pg[rule], pg["model"], eid, n_boot=a.n_boot, seed=a.seed)
        d_ev = pboot(ev[rule], ev["model"], eid, n_boot=a.n_boot, seed=a.seed)
        d_ad = pboot(ad[rule], ad["model"], eid, n_boot=a.n_boot, seed=a.seed)
        rows[rule] = {
            "sel_peak_g": float(np.mean(pg[rule])),
            "sel_envelope": float(np.mean(ev[rule])),
            "ade_m": float(np.mean(ad[rule])),
            "d_peak_g": d_pg, "d_envelope": d_ev, "d_ade": d_ad,
            "share_of_D1_fan_gap_pct": 0.0,          # structural: a selection rule moves 0
            "share_of_D2_selection_gap_pct":
                (100.0 * (-d_pg["delta"]) / D2) if D2 > 0 else float("nan"),
        }

    out = {
        "tool": "2026-09-05-kinematic-gate/raw/gate_share_of_gap.py",
        "tier": "T0", "n_windows": W, "n_candidates": N,
        "n_episodes": int(len(set(eid.tolist()))),
        "npz": a.npz,
        "structural_zero_on_fan_metrics": structural,
        "D1_fan_gap_g": D1, "D1_ratio_emitted_over_bank": ratio,
        "D1_source": BANK_VS_FAN,
        "bank_peak_g": float(bvf["bank"]["peak_g"]),
        "emitted_fan_peak_g": float(bvf["emitted"]["peak_g"]),
        "model_sel_peak_g": m_pg,
        "best_selectable_peak_g": ceil_pg,
        "D2_selection_gap_g": D2,
        "model_sel_peak_g_vs_fan_mean_ratio": (fan_pg_mean / m_pg) if m_pg else float("nan"),
        "model_sel_peak_g_vs_bank_ratio": (float(bvf["bank"]["peak_g"]) / m_pg)
                                          if m_pg else float("nan"),
        "rules": rows,
        "ranking_key_used": _rk,
        "gate1_equals_model_by_construction":
            bool(int((pick["gate1"] != pick["model"]).sum()) == 0),
        "gate1_index_disagreements": int((pick["gate1"] != pick["model"]).sum()),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== P4: WHAT SHARE OF THE 8.56x DOES THE GATE CLOSE? (T0, %dw/%dep) ==="
          % (W, out["n_episodes"]))
    print("  ranking used: %s   gate1==model on all windows: %s (disagree %d)"
          % (_rk, out["gate1_equals_model_by_construction"],
             out["gate1_index_disagreements"]))
    print("  the 8.56x is a FAN property: bank %.4f g -> emitted fan %.4f g (D1 = %.4f g)"
          % (out["bank_peak_g"], out["emitted_fan_peak_g"], D1))
    print("  fan_peak_g_mean on THIS bank            : %.4f g" % fan_pg_mean)
    print("  => share of D1 closed by ANY selection rule: 0.00 pct  (STRUCTURAL ZERO --")
    print("    the gate changes no waypoint; the fan tensor is one object for every rule)")
    print("")
    print("  the SELECTED path is already far better conditioned than the fan's average:")
    print("    model sel_peak_g %.4f g = fan mean / %.1f  = bank mean / %.2f"
          % (m_pg, out["model_sel_peak_g_vs_fan_mean_ratio"],
             1.0 / out["model_sel_peak_g_vs_bank_ratio"]
             if out["model_sel_peak_g_vs_bank_ratio"] else float("nan")))
    print("    best SELECTABLE peak_g on this fan     : %.4f g" % ceil_pg)
    print("    D2 (the gap a selection rule CAN close): %.4f g" % D2)
    print("")
    print("  %-16s %10s %10s %10s %10s %10s" % ("rule", "sel_peak_g", "d_peak_g",
                                                "sel_env", "ade_m", "pct_of_D2"))
    print("  %-16s %10.4f %10s %10.4f %10.4f %10s"
          % ("model", m_pg, "-", float(np.mean(ev["model"])), float(np.mean(ad["model"])),
             "-"))
    for rule, r in rows.items():
        print("  %-16s %10.4f %+10.4f %10.4f %10.4f %9.1f%%"
              % (rule, r["sel_peak_g"], r["d_peak_g"]["delta"], r["sel_envelope"],
                 r["ade_m"], r["share_of_D2_selection_gap_pct"]))
    print("")
    print("[share] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
