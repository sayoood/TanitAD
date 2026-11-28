"""GS-9 audit item A7 — THE MISSING v CONTROL.

The banked transition probe states `v is NEVER a probe input`. That is true of the FEATURE
LIST and FALSE at the MODEL BOUNDARY: `clip_features` calls `_lift3(a2, v0, "steer_accel_v")`,
whose third channel is `pose_last[:, 3] / SPEED_SCALE` (= v_t / 10) — so every `dzhat_*` cell
is produced by a predictor that was handed v_t. `dx_fwd` over one tick is v_t*dt to first
order, so a v-echo alone can score highly on it.

This script measures the CEILING of that echo, on the IDENTICAL rows, through the
IDENTICAL estimator (transition_probe's own crossfit / skill / clip bootstrap, imported by
file, never copied). It needs NO model and NO image decode — only poses and actions.

Cells:  const (must read 0.0000 exactly) | v_t (the leak channel, 1-d)
        act2 (reproduction check: must equal the banked act2 numbers exactly)
        act2+v_t (the FULL 3-channel conditioning tuple the predictor receives)
"""
import glob, importlib.util, json, os, sys, time
import numpy as np

WT = r"C:\Users\Admin\tanitad-wt"
for p in (WT + r"\stack", WT + r"\stack\scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
TOOL = os.path.join(WT, "taniteval", "tools", "transition_probe.py")
_s = importlib.util.spec_from_file_location("tp_for_vleak", TOOL)
tp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(tp)

import torch

CORPUS = r"C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901\sp2\cache\physicalai-val130-heldout"
NCLIPS, MAXF, W, NSTACK = 129, 100, 6, 3
OUT = sys.argv[1] if len(sys.argv) > 1 else "gs9_vleak.json"

def clip_rows(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = min(len(d["jpeg_len"].numpy().tolist()), MAXF)
    k = NSTACK - 1
    nz = n - k                                    # == len(z) in the banked tool
    if nz < W + 2:
        return None
    poses = np.asarray(d["poses"], dtype=np.float64)[k:k + nz]
    act = d["actions"].float().numpy().astype(np.float64)[k:k + nz]
    rows = list(range(W - 1, nz - 1))              # window ends at t, pose t+1 exists
    tg = tp.targets_from_poses(poses, 1)[rows]
    v_t = poses[rows, 3][:, None]                  # the trainer's pose_last[:, 3]
    return {"v_t": v_t, "act2": act[rows], "act2+v_t": np.concatenate([act[rows], v_t], 1),
            "const": np.ones((len(rows), 1)), "y": tg,
            "clip_id": str(d.get("clip_id", os.path.basename(path)))}

clips = sorted(glob.glob(os.path.join(CORPUS, "*.v2ep.pt")))[:NCLIPS]
print(f"clips {len(clips)}", flush=True)
F, Y, names = {}, [], []
t0 = time.time()
for i, c in enumerate(clips, 1):
    r = clip_rows(c)
    if r is None:
        continue
    for k in ("const", "v_t", "act2", "act2+v_t"):
        F.setdefault(k, []).append(r[k])
    Y.append(r["y"]); names.append(r["clip_id"])
    if i % 20 == 0 or i == len(clips):
        print(f"  {i}/{len(clips)} clips, {sum(len(y) for y in Y)} rows, {time.time()-t0:.0f}s", flush=True)

# ⛔ run_panel() iterates a FIXED FEATURE_ORDER that does not contain v_t / act2+v_t —
# calling it would SILENTLY SKIP the very cells this control exists for. So the panel body
# is reproduced here from the tool's OWN functions (crossfit / skill_score / pearson_r /
# clip_bootstrap_skill / shuffle_targets), imported by file. Same estimator, extra cells.
TARGETS = list(tp.TARGETS)
y_within = tp.shuffle_targets(Y, "within", 1)
y_global = tp.shuffle_targets(Y, "global", 2)
variants = {"within_shuffle": y_within, "global_shuffle": y_global}
cells, fits = {}, {}
for name in ("const", "v_t", "act2", "act2+v_t"):
    t1 = time.time()
    cf = tp.crossfit(F[name], Y, k_outer=5, lambdas=tp.LAMBDA_GRID, k_inner=5, seed=0,
                     pca=0, y_variants=variants)
    fits[name] = cf
    cell = {"n_score": cf["n_score"], "n_fit_per_fold": cf["n_fit_per_fold"], "d": cf["d"],
            "lambda_rel_per_fold": cf["lambda_per_fold"],
            "lambda_at_grid_edge": [bool(l in (tp.LAMBDA_GRID[0], tp.LAMBDA_GRID[-1]))
                                    for l in cf["lambda_per_fold"]],
            "seconds": round(time.time() - t1, 1)}
    for v in ("real", "within_shuffle", "global_shuffle"):
        cell[v] = {"skill": tp.skill_score(cf["pred"][v], cf["const"][v], cf["y"][v]).tolist(),
                   "r": tp.pearson_r(cf["pred"][v], cf["y"][v]).tolist()}
    cell["real"]["skill_ci"] = tp.clip_bootstrap_skill(cf["pred"]["real"], cf["const"]["real"],
                                                       cf["y"]["real"], cf["clip"], 1000, 0)
    cell["transition_specific_skill"] = [a - b for a, b in
        zip(cell["real"]["skill"], cell["within_shuffle"]["skill"])]
    cells[name] = cell
    print(f"    {name:<10} d={cf['d']:<3} n={cf['n_score']}  skill "
          + " ".join(f"{t}:{s:+.4f}" for t, s in zip(TARGETS, cell["real"]["skill"]))
          + "  | within " + " ".join(f"{s:+.3f}" for s in cell["within_shuffle"]["skill"])
          + "  | global " + " ".join(f"{s:+.3f}" for s in cell["global_shuffle"]["skill"]), flush=True)
marg = {}
for a_, b_ in (("v_t", "const"), ("act2+v_t", "act2"), ("act2+v_t", "v_t"), ("act2", "const")):
    fa, fb = fits[a_], fits[b_]
    marg[f"{a_} - {b_}"] = tp.clip_bootstrap_skill(
        fa["pred"]["real"], fa["const"]["real"], fa["y"]["real"], fa["clip"], 1000, 0,
        pred_b=fb["pred"]["real"], const_b=fb["const"]["real"])
# the raw correlation the whole audit item turns on: v_t against each target
Yc = np.concatenate([np.asarray(y) for y in Y])
Vc = np.concatenate([np.asarray(x) for x in F["v_t"]])[:, 0]
raw_r = {}
for j, t in enumerate(TARGETS):
    a = Vc - Vc.mean(); b = Yc[:, j] - Yc[:, j].mean()
    raw_r[t] = float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))
print("  raw pearson r(v_t, target):", {k: round(v, 4) for k, v in raw_r.items()}, flush=True)
panel = {"n_rows": int(sum(len(y) for y in Y)), "n_clips": len(Y), "targets": TARGETS,
         "k_outer": 5, "k_inner": 5, "lambda_grid_rel": list(tp.LAMBDA_GRID), "pca": 0,
         "cells": cells, "paired_marginals": marg,
         "raw_pearson_r_v_t_vs_target": raw_r,
         "controls": {"constant_reads_exactly_zero":
                      bool(all(s == 0.0 for s in cells["const"]["real"]["skill"]))}}
res = {"_evidence_class": "MEASURED (ours; dev-box cpu)", "eval_tier": "T0-DIAGNOSTIC",
       "purpose": "GS-9 audit A7: the v-leak ceiling at the model boundary (_lift3 channel 3)",
       "corpus": CORPUS, "n_clips": len(names), "targets": TARGETS,
       "v_policy_note": ("v_t IS deliberately an input HERE, and only here: this panel measures "
                         "what a pure v-echo can score, so the banked dzhat_* cells can be read "
                         "against it. v never enters any cell quoted as a capability."),
       "transition_probe_from": TOOL, "panel": panel,
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("WROTE", OUT, flush=True)
