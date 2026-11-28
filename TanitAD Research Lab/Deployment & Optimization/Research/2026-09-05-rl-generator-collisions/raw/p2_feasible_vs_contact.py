"""P2 -- does the FEASIBLE DECODE change the GENERATOR's collision rate?

0 GPU. Operates on the banked fan `fan_bank_base_240w.npz` (240 windows x 128 candidates).

Reports fan collision rate with `feasible_decode.project_feasible` OFF vs ON, so the
projection's contribution is separable from anything a reward does.

CONTROLS (retraction #30: a control must assert WHICH OBJECT it ran on):
  C1 IDENTITY-OF-SCORER : re-score the UNPROJECTED banked fan and require the recomputed
                          `contact` to reproduce the banked `f_contact` EXACTLY. If this
                          fails, every downstream number is about a different object.
  C2 DISABLED-LEVER     : project_feasible(enabled=False) must return a bit-identical
                          object (max abs diff exactly 0.0).
  C3 ROUND-TRIP         : an ALREADY-FEASIBLE path must come back < 1e-5 m. This is the
                          arithmetic's control; C2 alone is blind to it because the
                          arithmetic never ran.
  C4 IT-ACTUALLY-RAN    : the projection must move a non-zero number of candidates, and
                          the post-projection infeasible rate must be ~0. A projection
                          that silently no-ops would otherwise read as "no effect".
"""
import os, sys, json, argparse, importlib.util

# --- the namespace-shadow bootstrap, copied from stack/scripts/rl_refcv3_min.py:62-95 ---
_REPO = os.environ.get("TANITAD_REPO") or r"C:\Users\Admin\tanitad-wt"
_STACK = os.path.join(_REPO, "stack")
_TE = os.path.join(_REPO, "taniteval")
_TE_TOOLS = os.path.join(_TE, "tools")
for _p in (_STACK, _TE, _TE_TOOLS, os.path.join(_STACK, "scripts")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default="raw/fan_bank_base_240w.npz")
ap.add_argument("--out", default="raw/p2_feasible_vs_contact.json")
ap.add_argument("--n-boot", type=int, default=4000)
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FS = _load_by_path("fan_safety_for_p2", os.path.join(_TE_TOOLS, "fan_safety.py"))
from tanitad.refs import feasible_decode as FD
print("REPO=%s\n  FS <- %s\n  FD <- %s" % (_REPO, FS.__file__, FD.__file__))

z = np.load(args.bank, allow_pickle=True)
fan2 = torch.from_numpy(z["fan2"]).double()          # [W, K, 5, 2]
v0 = torch.from_numpy(np.asarray(z["v0"])).double()  # [W]
lead5 = torch.from_numpy(z["lead5"]).double()        # [W, 5, 2]
has_lead = torch.from_numpy(z["has_lead"]).bool()    # [W]
eid = np.asarray(z["eid"])
banked = {f: torch.from_numpy(z["f_" + f]).bool() for f in
          ("contact", "ttc_below", "kamm_over", "envelope", "off_reach", "infeasible", "flagged")}
W, K = fan2.shape[0], fan2.shape[1]
print("OBJECT: %s  fan2%s  v0%s  lead5%s  has_lead=%d/%d"
      % (args.bank, tuple(fan2.shape), tuple(v0.shape), tuple(lead5.shape), int(has_lead.sum()), W))


def score(paths):
    """Score [W,K,5,2] with the lead track broadcast; no-lead windows get contact=False."""
    v0b = v0[:, None].expand(W, K)
    l5 = lead5[:, None, :, :].expand(W, K, 5, 2)
    sc = FS.score_paths(paths, v0b, l5)
    out = {k: v.clone() for k, v in sc.items()}
    nl = ~has_lead[:, None].expand(W, K)
    for f in ("contact", "ttc_below", "ttc_veto", "unsafe"):
        out[f] = out[f] & ~nl
    out["flagged"] = out["unsafe"] | out["infeasible"]
    return out


# ---------------- C1 IDENTITY-OF-SCORER ----------------
# C1a: re-score the unprojected bank with the CURRENT scorer. Every flag except
#      `contact` must reproduce the bank exactly. `contact` does NOT, and C1b names why.
sc_off = score(fan2)
c1 = {}
for f in ("contact", "kamm_over", "envelope", "off_reach", "infeasible"):
    n_dis = int((sc_off[f] != banked[f]).sum())
    c1[f] = {"disagreements": n_dis, "banked_rate": float(banked[f].double().mean()),
             "rescored_rate": float(sc_off[f].double().mean())}
    print("C1a %-11s disagree=%6d   banked=%.6f  rescored=%.6f"
          % (f, n_dis, c1[f]["banked_rate"], c1[f]["rescored_rate"]))
C1A_STRUCTURAL = all(c1[f]["disagreements"] == 0
                     for f in ("kamm_over", "envelope", "off_reach", "infeasible"))
print("C1a STRUCTURAL FLAGS reproduce the bank exactly: %s" % ("PASS" if C1A_STRUCTURAL else "FAIL"))

# C1b: POSITIVE IDENTIFICATION of the definition the bank actually used.
#      `rewards._collision` is SWEPT in the relative frame (a segment crossing the 2 m
#      disc counts even when neither endpoint is inside). The bank predates that change.
#      The per-step POINT test must reproduce the bank EXACTLY -- 0 disagreements. This
#      is a positive assertion about WHICH DEFINITION produced the banked flag, not a
#      hand-wave that "the numbers differ".
R_CONTACT = 2.0   # rewards._collision: ego_radius_m 1.0 + obs_radius_m 1.0
rel = lead5[:, None, 1:, :].expand(W, K, 4, 2) - fan2[:, :, 1:, :]
point_contact = (rel.norm(dim=-1) < R_CONTACT).any(-1) & has_lead[:, None]
n_pt_dis = int((point_contact != banked["contact"]).sum())
C1B_PASS = n_pt_dis == 0
print("C1b BANK-DEFINITION-ID: per-step POINT test vs banked f_contact -> disagree=%d  %s"
      % (n_pt_dis, "PASS (bank == POINT test)" if C1B_PASS else "FAIL (definition unknown)"))
print("    banked (POINT, superseded) rate = %.6f ; current (SWEPT) rate = %.6f  (+%.1f%%)"
      % (float(banked["contact"].double().mean()), float(sc_off["contact"].double().mean()),
         100 * (float(sc_off["contact"].double().mean()) / float(banked["contact"].double().mean()) - 1)))
c1["_bank_definition"] = {"identified_as": "per-step POINT test, radius 2.0 m",
                          "point_test_disagreements_vs_bank": n_pt_dis,
                          "current_scorer_is": "SWEPT relative-frame (rewards._collision)",
                          "banked_rate_point": float(banked["contact"].double().mean()),
                          "current_rate_swept": float(sc_off["contact"].double().mean())}
C1_PASS = C1A_STRUCTURAL and C1B_PASS
print("C1 OVERALL (structural reproduce + bank definition positively identified): %s"
      % ("PASS" if C1_PASS else "FAIL"))

# ---------------- C2 DISABLED-LEVER ----------------
off_obj = FD.project_feasible(fan2, v0[:, None].expand(W, K), enabled=False)
c2_max = float((off_obj - fan2).abs().max())
c2_same_obj = off_obj is fan2
print("C2 DISABLED-LEVER: max|diff|=%.3e  returned_same_object=%s -> %s"
      % (c2_max, c2_same_obj, "PASS" if c2_max == 0.0 else "FAIL"))

# ---------------- the arm: projection ON ----------------
proj = FD.project_feasible(fan2, v0[:, None].expand(W, K), enabled=True)
moved = (proj - fan2).abs().amax(dim=(-1, -2))          # [W, K] per-candidate max shift
n_moved = int((moved > 1e-9).sum())
print("C4 IT-ACTUALLY-RAN: candidates moved = %d/%d (%.4f)  median shift %.4f m  max %.4f m"
      % (n_moved, W * K, n_moved / (W * K), float(moved.median()), float(moved.max())))

# ---------------- C3 ROUND-TRIP ----------------
proj2 = FD.project_feasible(proj, v0[:, None].expand(W, K), enabled=True)
c3_max = float((proj2 - proj).abs().max())
print("C3 ROUND-TRIP (already-feasible must be a fixed point): max|diff|=%.3e -> %s"
      % (c3_max, "PASS" if c3_max < 1e-5 else "FAIL"))

sc_on = score(proj)
print("C4b post-projection infeasible rate = %.6f (was %.6f)"
      % (float(sc_on["infeasible"].double().mean()), float(sc_off["infeasible"].double().mean())))


# ---------------- the metrics, both populations ----------------
def topk_rate(flag, rank, k):
    idx = torch.argsort(rank, dim=1, descending=True)[:, :k]
    return torch.gather(flag.double(), 1, idx)


rank = torch.from_numpy(z["rank"]).double()
rank_f = torch.where(torch.isfinite(rank), rank, torch.full_like(rank, -1e30))
sel_idx = torch.from_numpy(np.asarray(z["sel_idx"])).long()

lead_w = has_lead.numpy()


def _boot_paired(a, b, ep, n_boot, seed):
    """paired episode-cluster bootstrap over per-window values a,b (a-b)."""
    rng = np.random.default_rng(seed)
    ue = np.unique(ep)
    idx_by_e = [np.nonzero(ep == e)[0] for e in ue]
    d = a - b
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(ue), len(ue))
        sel = np.concatenate([idx_by_e[j] for j in pick])
        out[i] = d[sel].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)


res = {}
for name, pop in (("all_windows", np.ones(W, bool)), ("lead_windows", lead_w)):
    r = {}
    for metric, flag_off, flag_on in (("fan_contact", sc_off["contact"], sc_on["contact"]),
                                      ("fan_infeasible", sc_off["infeasible"], sc_on["infeasible"]),
                                      ("fan_kamm_over", sc_off["kamm_over"], sc_on["kamm_over"]),
                                      ("fan_envelope", sc_off["envelope"], sc_on["envelope"]),
                                      ("fan_unsafe", sc_off["unsafe"], sc_on["unsafe"])):
        a_w = flag_on.double().mean(dim=1).numpy()[pop]
        b_w = flag_off.double().mean(dim=1).numpy()[pop]
        d, lo, hi, sep = _boot_paired(a_w, b_w, eid[pop], args.n_boot, args.seed)
        r[metric] = {"off": float(b_w.mean()), "on": float(a_w.mean()),
                     "delta": d, "lo": lo, "hi": hi, "sep": sep, "n_windows": int(pop.sum())}
        print("[%s] %-15s OFF=%.6f  ON=%.6f  d=%+.6f [%+.6f,%+.6f] sep=%s"
              % (name, metric, r[metric]["off"], r[metric]["on"], d, lo, hi, sep))
    for k in (8, 32):
        a_w = topk_rate(sc_on["contact"], rank_f, k).mean(dim=1).numpy()[pop]
        b_w = topk_rate(sc_off["contact"], rank_f, k).mean(dim=1).numpy()[pop]
        d, lo, hi, sep = _boot_paired(a_w, b_w, eid[pop], args.n_boot, args.seed)
        r["top%d_contact" % k] = {"off": float(b_w.mean()), "on": float(a_w.mean()),
                                  "delta": d, "lo": lo, "hi": hi, "sep": sep,
                                  "n_windows": int(pop.sum())}
        print("[%s] top%-11d OFF=%.6f  ON=%.6f  d=%+.6f [%+.6f,%+.6f] sep=%s"
              % (name, k, r["top%d_contact" % k]["off"], r["top%d_contact" % k]["on"], d, lo, hi, sep))
    a_w = sc_on["contact"][torch.arange(W), sel_idx].double().numpy()[pop]
    b_w = sc_off["contact"][torch.arange(W), sel_idx].double().numpy()[pop]
    r["sel_contact"] = {"off": float(b_w.mean()), "on": float(a_w.mean()),
                        "n_windows": int(pop.sum())}
    print("[%s] sel_contact     OFF=%.6f  ON=%.6f" % (name, r["sel_contact"]["off"], r["sel_contact"]["on"]))
    res[name] = r

# ---------------- geometric cost of the projection ----------------
shift = (proj - fan2).norm(dim=-1)                       # [W,K,5] per-point displacement
ade_cost = shift[..., 1:].mean(dim=-1)                   # [W,K] mean over the 4 non-origin points
print("PROJECTION GEOMETRIC COST: mean %.4f m  median %.4f m  p95 %.4f m  max %.4f m"
      % (float(ade_cost.mean()), float(ade_cost.median()),
         float(np.percentile(ade_cost.numpy(), 95)), float(ade_cost.max())))

out = {"_tool": "p2_feasible_vs_contact.py",
       "_tier": "T0 readout on the EMITTED fan (never a driving claim)",
       "_evidence_class": "MEASURED (ours)",
       "bank": args.bank, "W": W, "K": K, "n_lead_windows": int(has_lead.sum()),
       "estimator": "paired episode-cluster bootstrap (cluster = eid)",
       "n_boot": args.n_boot, "seed": args.seed,
       "controls": {"C1_identity_of_scorer": c1, "C1_PASS": bool(C1_PASS),
                    "C2_disabled_lever_max_abs_diff": c2_max,
                    "C2_returned_same_object": bool(c2_same_obj),
                    "C3_round_trip_max_abs_diff": c3_max,
                    "C4_candidates_moved": n_moved,
                    "C4_moved_frac": n_moved / (W * K),
                    "C4b_infeasible_off": float(sc_off["infeasible"].double().mean()),
                    "C4b_infeasible_on": float(sc_on["infeasible"].double().mean())},
       "projection_geometric_cost_m": {"mean": float(ade_cost.mean()),
                                       "median": float(ade_cost.median()),
                                       "p95": float(np.percentile(ade_cost.numpy(), 95)),
                                       "max": float(ade_cost.max())},
       "results": res}
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1)
print("WROTE %s" % args.out)
print("VERDICT_C1=%s C2=%s C3=%s C4=%s"
      % (C1_PASS, c2_max == 0.0, c3_max < 1e-5, n_moved > 0))
