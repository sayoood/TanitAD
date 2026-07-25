"""E1 — THE STOP GATE. Held-out confirmation of the L1_gate decision-relevance lift at 3.0 m ONLY.

Pre-registered in PRE_REGISTRATION.md. The verdict is decided at d = 3.0 m and nowhere else.
Everything after Sec 1 of the output is DESCRIPTIVE and cannot move the verdict.
"""
import json
import sys

import numpy as np
import pandas as pd

S = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad"
OUT = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
       r"\Architecture & Inference\Implementation\incoming\2026-07-25-h2-e0-e1")
sys.path.insert(0, S)
from h2e_stats import paired_cluster_diff, paired_cluster_lift   # noqa: E402

CL = "v_camera_cross_left_120fov"
CR = "v_camera_cross_right_120fov"
D_PRE = 3.0                      # THE pre-registered threshold. Not a variable.
PASS_BAR = 1.5                   # substrate agent's own stated bar for "not materially attenuated"

A = pd.read_parquet(S + r"\h2e_heldout.parquet").dropna(subset=["dmin_cf", "ego_dv"])
sweep = set(pd.read_parquet(S + r"\crux3.parquet", columns=["clip_id"]).clip_id.unique())
overlap = len(set(A.clip_id.unique()) & sweep)
assert overlap == 0, f"HELD-OUT VIOLATION: {overlap} clips also in the sweep"
print(f"held-out: {A.clip_id.nunique():,} clips / {A.chunk.nunique()} chunks / "
      f"{len(A):,} agent-frames  | overlap with sweep: {overlap}")


def frames(a, d):
    """Frame-level L1_gate (cropped definition, the substrate audit's) at conflict radius d."""
    x = a.assign(cf=a.dmin_cf <= d, unseen=~a.in_crop)
    x["Sl"] = x[CL] & x.unseen & x.cf
    x["Sr"] = x[CR] & x.unseen & x.cf
    x["S"] = x.Sl | x.Sr
    x["F"] = x.in_crop & x.cf
    g = x.groupby(["clip_id", "gi"], observed=True).agg(
        S=("S", "sum"), Sl=("Sl", "sum"), Sr=("Sr", "sum"), F=("F", "sum"),
        dv=("ego_dv", "first"), v=("ego_v", "first"),
        junction=("junction", "first"), lane_change=("lane_change", "first")).reset_index()
    g["gate"] = (g.S > 0) & (g.F == 0)
    g["resp"] = g.dv <= -1.0
    return g


res = {"pre_registered_threshold_m": D_PRE,
       "pass_bar_lift": PASS_BAR,
       "n_clips_heldout": int(A.clip_id.nunique()),
       "n_chunks_heldout": int(A.chunk.nunique()),
       "chunks_heldout": sorted(A.chunk.unique().tolist()),
       "n_agent_frames": int(len(A)),
       "overlap_clips_with_sweep": overlap}

# ---------------------------------------------------------------- 1. THE VERDICT
g = frames(A, D_PRE)
lift = paired_cluster_lift(g.resp.to_numpy(), g.gate.to_numpy(), g.clip_id.to_numpy())
diff = paired_cluster_diff(g.resp.to_numpy(), g.gate.to_numpy(), g.clip_id.to_numpy())
n_ep_pos = int(g.groupby("clip_id").gate.max().sum())
verdict = "PASS" if (lift["excludes_1_above"] and lift["lift"] >= PASS_BAR) else "FAIL"

res["E1"] = {"verdict": verdict, "lift": lift, "difference_form": diff,
             "n_frames": int(len(g)), "gate_rate": round(float(g.gate.mean()), 5),
             "n_episodes_gate_positive": n_ep_pos,
             "label_rate": round(float((g.gate & g.resp).mean()), 5),
             "n_episodes_label_positive": int(g.assign(l=g.gate & g.resp)
                                              .groupby("clip_id").l.max().sum())}
print("\n" + "=" * 78)
print(f"E1 VERDICT (pre-registered, d = {D_PRE} m only):  {verdict}")
print("=" * 78)
print(f"  held-out lift              : {lift['lift']:.2f}x   95% CI [{lift['lo']:.2f}, {lift['hi']:.2f}]")
print(f"  P(resp | gate+)            : {lift['p_resp_given_pos']:.2%}  (n = {lift['n_pos']:,} frames)")
print(f"  P(resp | gate-)            : {lift['p_resp_given_neg']:.2%}  (n = {lift['n_neg']:,} frames)")
print(f"  risk difference            : {diff['delta']:+.4f}  CI [{diff['lo']:+.4f}, {diff['hi']:+.4f}]"
      f"  separated={diff['separated']}")
print(f"  gate rate                  : {g.gate.mean():.2%} of frames")
print(f"  episode-clusters           : {lift['n_episodes']:,}  ({n_ep_pos} gate-positive)")
print(f"  estimator                  : {lift['estimator']}")
print(f"  bootstrap draws used/skip  : {lift['n_draws_used']}/{lift['n_draws_skipped']}")

# ------------------------------------------- 2. per-chunk (is one chunk driving it?) DESCRIPTIVE
per = []
for ch, gc in g.groupby("chunk" if "chunk" in g else g.clip_id.map(
        A.drop_duplicates("clip_id").set_index("clip_id").chunk)):
    if gc.gate.sum() < 5:
        per.append(dict(chunk=str(ch), n_clips=int(gc.clip_id.nunique()),
                        n_pos=int(gc.gate.sum()), lift=None))
        continue
    L = paired_cluster_lift(gc.resp.to_numpy(), gc.gate.to_numpy(), gc.clip_id.to_numpy(),
                            n_boot=500)
    per.append(dict(chunk=str(ch), n_clips=int(gc.clip_id.nunique()),
                    n_pos=int(gc.gate.sum()), lift=L["lift"], lo=L["lo"], hi=L["hi"]))
res["E1_per_chunk"] = per
print("\n-- per-chunk (descriptive; checks no single chunk drives the result) --")
for p in per:
    if p["lift"] is None:
        print(f"  chunk {p['chunk']}: n+={p['n_pos']:4d}  (too few)")
    else:
        print(f"  chunk {p['chunk']}: clips {p['n_clips']:3d}  n+={p['n_pos']:4d}  "
              f"lift {p['lift']:5.2f}x [{p['lo']:.2f}, {p['hi']:.2f}]")
pos_lifts = [p["lift"] for p in per if p["lift"] is not None]
res["E1_per_chunk_summary"] = {"n_chunks_scored": len(pos_lifts),
                               "n_above_1": int(sum(x > 1 for x in pos_lifts)),
                               "median": round(float(np.median(pos_lifts)), 3) if pos_lifts else None}

# ------------------------------------- 3. MECHANISM PROFILE (descriptive, cannot move the verdict)
prof = []
for d in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0):
    gg = frames(A, d)
    L = paired_cluster_lift(gg.resp.to_numpy(), gg.gate.to_numpy(), gg.clip_id.to_numpy(),
                            n_boot=1000)
    prof.append(dict(d_m=d, gate_rate=round(float(gg.gate.mean()), 5),
                     n_pos=int(gg.gate.sum()), lift=L["lift"], lo=L["lo"], hi=L["hi"],
                     p_pos=L["p_resp_given_pos"], p_neg=L["p_resp_given_neg"]))
res["E1_distance_profile"] = prof
print("\n-- lift as a continuous function of conflict radius (HELD-OUT, descriptive) --")
print(f"{'d(m)':>5} {'pos%':>7} {'n+':>7} {'P(r|+)':>8} {'P(r|-)':>8} {'lift':>7}  95% CI")
for p in prof:
    print(f"{p['d_m']:5.1f} {p['gate_rate']:7.2%} {p['n_pos']:7,} {p['p_pos']:8.2%} "
          f"{p['p_neg']:8.2%} {p['lift']:7.2f}x  [{p['lo']:.2f}, {p['hi']:.2f}]")

# ---------------------------------------------------------- 4. sensitivity: phase-0 clips only
sel = pd.read_parquet(r"C:\Users\Admin\tanitad-data\physicalai\r0\phase0_selection.parquet")
p0 = set(sel.clip_id.astype(str))
Ap = A[A.clip_id.isin(p0)]
if Ap.clip_id.nunique() >= 40:
    gp = frames(Ap, D_PRE)
    Lp = paired_cluster_lift(gp.resp.to_numpy(), gp.gate.to_numpy(), gp.clip_id.to_numpy())
    res["E1_phase0_subset"] = dict(n_clips=int(Ap.clip_id.nunique()), **Lp)
    print(f"\n-- sensitivity: phase-0-selected held-out clips only (n={Ap.clip_id.nunique()}) --")
    print(f"  lift {Lp['lift']:.2f}x [{Lp['lo']:.2f}, {Lp['hi']:.2f}]  n+={Lp['n_pos']:,}")

# ------------------------------------------------------- 5. sensitivity: response threshold
sens = []
for thr in (-0.5, -1.0, -1.5, -2.0):
    gg = frames(A, D_PRE)
    r = (gg.dv <= thr).to_numpy()
    L = paired_cluster_lift(r, gg.gate.to_numpy(), gg.clip_id.to_numpy(), n_boot=1000)
    sens.append(dict(dv_threshold=thr, lift=L["lift"], lo=L["lo"], hi=L["hi"],
                     p_pos=L["p_resp_given_pos"], p_neg=L["p_resp_given_neg"]))
res["E1_response_sensitivity"] = sens
print("\n-- sensitivity: deceleration-response threshold (descriptive) --")
for s in sens:
    print(f"  dv <= {s['dv_threshold']:+.1f} m/s : lift {s['lift']:5.2f}x "
          f"[{s['lo']:.2f}, {s['hi']:.2f}]")

res["published_sweep_reference"] = {"lift_3m": [2.22, 1.30, 3.14], "lift_6m": [0.43, 0.24, 0.71],
                                    "source": "H2_SUBSTRATE_AND_LABELING.md Sec 6.4 (80 clips, "
                                              "chunks 0036+0170), threshold chosen post-hoc"}
json.dump(res, open(OUT + r"\e1_heldout.json", "w"), indent=2)
print(f"\nwrote {OUT}\\e1_heldout.json")
