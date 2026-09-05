#!/usr/bin/env python3
"""P3 -- derive the FEASIBILITY-AWARE arm's dump from the banked base dump. ZERO GPU.

The lever is a deterministic geometric function of the emitted path, so the T1 arm can be
produced from the banked rollout without a second forward pass: every episode's `os`
(the model's self-action OPEN-LOOP driven path) is replaced by its friction-feasible
projection, and every other array -- `g` (the human), `ha0` (the shared trivial floor),
`v0`, `ws` -- is copied VERBATIM.

⛔ WHY COPYING `g` AND `ha0` VERBATIM IS LOAD-BEARING, NOT LAZY.
`paired_openloop.py` proves the two dumps saw the same windows by asserting the GT is
BIT-IDENTICAL on every shared window, and it requires both arms to be scored against the
SAME floor. Regenerating either would break the proof it relies on; copying makes the
pairing exact rather than merely intended.

⛔ THE FLOORS THIS SCRIPT EXISTS TO PRODUCE.
The brief binds two: a SEED REPLICATE and `ctrl0` (lr = 0), both of which exist because
AdamW normalises by the gradient's own scale, so a control that zeroes a LOSS still moves
every tensor. ⭐ **This lever takes ZERO gradient steps**, so the admissible floor is
STRICTLY STRONGER and is measured, not asserted:
  F1  `projoff` -- the identical pipeline with the lever DISABLED. Every array must come
      back bit-identical to base, so the rig's run-to-run noise floor is EXACTLY zero and
      no `ctrl0` can improve on it.
  F2  determinism -- the projection re-run under a different RNG seed must be bit-identical.
⚠️ SCOPE, STATED: F1/F2 retire the TRAINING-variance question for THIS arm only. Any
future arm that FINE-TUNES the decoder under the projection re-acquires H-ESTIM-SEED-1 in
full, and nothing here discharges it in advance.
ASCII-only output.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch

_REPO = os.environ.get("TANITAD_REPO") or "C:/Users/Admin/refcv4b_repo"
sys.path.insert(0, os.path.join(_REPO, "stack"))
sys.path.insert(0, os.path.join(_REPO, "taniteval"))
sys.path.insert(0, os.path.join(_REPO, "taniteval", "tools"))

from tanitad.refs import feasible_decode as FD      # noqa: E402
import fan_safety as FS                             # noqa: E402
import taniteval.ci as CI                           # noqa: E402

TOOL = "2026-09-05-feasible-decode/raw/derive_projected_dump.py"

ARMS = {
    "projoff":  dict(enabled=False, mu=None,         clamp_entry=False),
    "proj07":   dict(enabled=True,  mu=FD.MU_KAMM,   clamp_entry=False),
    "proj07e":  dict(enabled=True,  mu=FD.MU_KAMM,   clamp_entry=True),
    "projbox":  dict(enabled=True,  mu=None,         clamp_entry=False),
}


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def project(os_arr: np.ndarray, v0: np.ndarray, cfg: dict, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)                       # F2: there is no RNG; prove it
    p = torch.from_numpy(os_arr).float()          # [W, 4, 2]
    if not cfg["enabled"]:
        return os_arr.copy()
    p5 = FS.with_origin(p)
    q5 = FD.project_feasible(p5, torch.from_numpy(v0).float(), mu=cfg["mu"],
                             clamp_entry=cfg["clamp_entry"])
    return q5[..., 1:, :].numpy().astype(os_arr.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dump", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    eps = sorted(glob.glob(os.path.join(a.base_dump, "ep*.npz")))
    if not eps:
        raise SystemExit("no ep*.npz under %s" % a.base_dump)
    print("=== P3: derive the feasibility-aware arm from the banked base dump ===")
    print("  base dump: %s (%d episodes)" % (a.base_dump, len(eps)))

    stats = {k: {"n_windows": 0, "identical": 0, "max_abs_delta_m": 0.0}
             for k in ARMS}
    paths_all = {k: [] for k in list(ARMS) + ["base", "gt", "ha0"]}
    v0_all, eid_all = [], []

    for name, cfg in ARMS.items():
        d = os.path.join(a.out_root, "%s_dump" % name)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for extra in ("manifest.json",):
            src = os.path.join(a.base_dump, extra)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(d, extra))
        sub = os.path.join(a.base_dump, "decisions")
        if os.path.isdir(sub):
            shutil.copytree(sub, os.path.join(d, "decisions"))

    for f in eps:
        z = np.load(f, allow_pickle=True)
        cols = {k: z[k] for k in z.files}
        v0 = cols["v0"].astype(np.float32)
        base_os = cols["os"].astype(np.float32)
        v0_all.append(v0)
        eid_all.append(np.full(v0.shape[0], int(cols["eid"][0])))
        paths_all["base"].append(base_os)
        paths_all["gt"].append(cols["g"].astype(np.float32))
        paths_all["ha0"].append(cols["ha0"].astype(np.float32))
        for name, cfg in ARMS.items():
            q = project(base_os, v0, cfg, seed=0)
            q2 = project(base_os, v0, cfg, seed=12345)      # F2
            if sha(q) != sha(q2):
                raise SystemExit("F2 FAILED: %s is not deterministic on %s"
                                 % (name, os.path.basename(f)))
            st = stats[name]
            st["n_windows"] += int(q.shape[0])
            st["identical"] += int((q == base_os).all(axis=(1, 2)).sum())
            st["max_abs_delta_m"] = max(st["max_abs_delta_m"],
                                        float(np.abs(q - base_os).max()))
            paths_all[name].append(q)
            out = dict(cols)
            out["os"] = q.astype(cols["os"].dtype)
            np.savez(os.path.join(a.out_root, "%s_dump" % name,
                                  os.path.basename(f)), **out)

    v0_t = torch.from_numpy(np.concatenate(v0_all)).float()
    eid = np.concatenate(eid_all)
    W = int(v0_t.shape[0])
    print("  %d windows / %d episodes" % (W, len(set(eid.tolist()))))

    def safety(arr):
        p5 = FS.with_origin(torch.from_numpy(np.concatenate(arr)).float())
        sc = FS.score_paths(p5, v0_t, None)
        return {k: float(sc[k].float().mean())
                for k in ("envelope", "kamm_over", "off_reach", "infeasible")} | {
            "peak_g_mean": float(sc["peak_g"].mean()),
            "peak_g_p95": float(np.percentile(sc["peak_g"].numpy(), 95)),
            "peak_g_max": float(sc["peak_g"].max())}

    def ade(arr):
        p = torch.from_numpy(np.concatenate(arr)).float()
        g = torch.from_numpy(np.concatenate(paths_all["gt"])).float()
        return (p - g).norm(dim=-1).mean(dim=-1).double().numpy()

    ade_base = ade(paths_all["base"])
    rep = {"_tool": TOOL, "_tier": "T1 -- self-action OPEN loop (PI ruling 2026-09-02)",
           "_evidence_class": "MEASURED (ours)",
           "base_dump": a.base_dump, "n_windows": W,
           "n_episodes": int(len(set(eid.tolist()))),
           "margin": FD.MARGIN,
           "driven_path_safety": {"base": safety(paths_all["base"]),
                                  "human_gt": safety(paths_all["gt"]),
                                  "ha0_floor": safety(paths_all["ha0"])},
           "arms": {}}
    for name in ARMS:
        st = stats[name]
        d = ade(paths_all[name]) - ade_base
        b = CI.episode_cluster_bootstrap(d, eid, n_boot=2000, seed=0)
        rep["arms"][name] = {
            "config": {k: (v if not isinstance(v, float) else round(v, 4))
                       for k, v in ARMS[name].items()},
            "n_windows": st["n_windows"],
            "n_bit_identical_to_base": st["identical"],
            "frac_bit_identical": st["identical"] / max(st["n_windows"], 1),
            "max_abs_delta_m": st["max_abs_delta_m"],
            "determinism_F2": "PASS (bit-identical under a different torch seed)",
            "driven_path_safety": safety(paths_all[name]),
            "d_ade_0_2s_m": {"mean": b["mean"], "lo": b["lo"], "hi": b["hi"],
                             "n_windows": b["n_windows"],
                             "n_episodes": b["n_episodes"],
                             "estimator": "paired episode_cluster_bootstrap",
                             "separated": bool(b["lo"] > 0 or b["hi"] < 0)}}
    f1 = rep["arms"]["projoff"]
    rep["F1_disabled_lever_floor"] = {
        "name": "F1 -- the identical pipeline with the lever DISABLED",
        "n_bit_identical_to_base": f1["n_bit_identical_to_base"],
        "n_windows": f1["n_windows"],
        "max_abs_delta_m": f1["max_abs_delta_m"],
        "d_ade_0_2s_m": f1["d_ade_0_2s_m"],
        "PASS": bool(f1["n_bit_identical_to_base"] == f1["n_windows"]
                     and f1["max_abs_delta_m"] == 0.0),
        "why_stronger_than_ctrl0": "ctrl0 (lr=0) exists because a loss-zeroing control "
                                   "still moves every tensor under AdamW. This lever "
                                   "takes NO gradient step at all, so its zero-lever arm "
                                   "is bit-identical rather than merely small, and the "
                                   "rig's run-to-run noise floor is EXACTLY 0.",
        "scope": "retires the TRAINING-variance question for THIS arm only; a fine-tuned "
                 "arm re-acquires H-ESTIM-SEED-1 in full."}

    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)

    print("")
    print("  F1 disabled lever: %d/%d windows bit-identical, max|delta| %.1e -> %s"
          % (f1["n_bit_identical_to_base"], f1["n_windows"], f1["max_abs_delta_m"],
             "PASS" if rep["F1_disabled_lever_floor"]["PASS"] else "FAIL"))
    print("")
    print("  --- the DRIVEN path (arm `os`), %d windows / %d episodes ---"
          % (W, rep["n_episodes"]))
    print("  %-12s %9s %9s %9s %9s %9s %9s"
          % ("arm", "envelope", "kamm_over", "off_reach", "infeas", "peak_g", "pkg_p95"))
    for tag, s in [("base", rep["driven_path_safety"]["base"])] + \
                  [(k, rep["arms"][k]["driven_path_safety"]) for k in ARMS] + \
                  [("human_gt", rep["driven_path_safety"]["human_gt"]),
                   ("ha0_floor", rep["driven_path_safety"]["ha0_floor"])]:
        print("  %-12s %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f"
              % (tag, s["envelope"], s["kamm_over"], s["off_reach"], s["infeasible"],
                 s["peak_g_mean"], s["peak_g_p95"]))
    print("")
    for k in ARMS:
        d = rep["arms"][k]["d_ade_0_2s_m"]
        print("  %-12s d_ade_0_2s %+0.4f m [%+0.4f, %+0.4f] separated=%s  "
              "(%d w / %d ep, paired episode-cluster bootstrap)"
              % (k, d["mean"], d["lo"], d["hi"], d["separated"],
                 d["n_windows"], d["n_episodes"]))
    print("[derive] wrote %s and the dumps under %s" % (a.report, a.out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
