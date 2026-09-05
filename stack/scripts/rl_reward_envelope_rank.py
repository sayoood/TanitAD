#!/usr/bin/env python3
"""P1 - DOES THE RL REWARD RANK ENVELOPE-VIOLATING CANDIDATES HIGHER?

The cheapest discriminating experiment in the REF-C RL line, and it decides whether the
composed reward is admissible at all. A group-relative advantage is a RANKING over the
candidates of one fan: whatever the reward puts on top is where probability mass goes.
So the question is not "what does the reward score" but "how does the reward ORDER the
fan with respect to the friction/comfort envelope".

  rho > 0  the reward prefers envelope-violating candidates -> DISQUALIFIED at source;
           no amount of RL repairs it, and adding a feasibility term will not either
           (the arm that made feasibility WORSE already carried `feasibility: 0.5`).
  rho < 0  the reward prefers feasible candidates; the 2026-09-05 regression came from
           somewhere else and the reward line stays open.

WHAT IS SCORED. The EMITTED fan (`anchor_traj [B, 128, 8, 2]`), 2 s prefix on the 0.5 s
grid with the ego origin prepended, exactly the object `fan_safety.py` scores and the
object the RL stage perturbs. Envelope/Kamm/reach flags come from
`taniteval.tools.fan_safety.score_paths` -- ONE definition, the same module the primary
endpoint and the paired bootstrap use. `peak_g` there is the finite-difference friction
load; `tanitad.instruments.flyability` (the EXACT, control-rolled instrument) cannot be
applied to a free-waypoint fan and is not re-implemented beside it -- `score_paths`
imports flyability's own `G`. The finite difference UNDER-reports by 1.21-1.85x on
control-rolled paths, so a LEVEL here is a lower bound; a RANK CORRELATION is unaffected
by a monotone under-report, which is the whole reason this probe reads ranks.

THE CONTROLS ARE NOT OPTIONAL (CLAUDE.md 2026-08-22: four probe failures in one
afternoon, each caught only by a control that had to read a known value):
  * `random`   a per-window uniform score  -> rho must straddle 0.
  * `constant` an identically-zero score    -> rho must be UNDEFINED (no variance), and
               the count of undefined windows is printed rather than silently dropped.
  * `peak_g`   scored against ITSELF        -> rho must read exactly +1.
  * every panel row prints its n (windows, episodes, candidates) and the number of
    windows where the flag had no variance and the correlation does not exist.

IT ALSO BANKS THE FAN. `--out-npz` writes the per-candidate arrays (fan prefix, every
reward component, every flag, peak_g, the selector's ranking score and the conf head's
logits). Any future reward design is then scorable at ZERO GPU -- the reason this probe
needed a GPU at all is that nobody had banked the fan.

Tier: T0 instrument probe on the EVAL corpus. Evidence class: MEASURED (ours).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.environ.get("TANITAD_REPO") or os.path.dirname(os.path.dirname(_HERE))


def _load_driver():
    """Load `rl_refcv3_min.py` FIRST and take the analysis chain from it.

    The outer `taniteval/` directory has no `__init__.py`, so a bare
    `import taniteval.ci` resolves to the NAMESPACE-PACKAGE SHADOW and dies with
    `No module named 'taniteval.ci'` -- a well-worn trap here. The driver already
    owns the un-shadowing (`_bootstrap_paths`) and already pays every analysis-time
    import at start-up, so this probe borrows both rather than re-deriving a second
    path bootstrap beside it.
    """
    path = os.path.join(_REPO, "stack", "scripts", "rl_refcv3_min.py")
    spec = importlib.util.spec_from_file_location("_rl_refcv3_min_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load_driver()
RW = D.RW
DEFAULT_WEIGHTS = D.DEFAULT_WEIGHTS
RewardSpec = D.RewardSpec
FS = D.FS
import taniteval.ci as CI                                              # noqa: E402

TOOL = "stack/scripts/rl_reward_envelope_rank.py"


# --------------------------------------------------------------------------- #
# Spearman, with ties and with an EXPLICIT undefined state                      #
# --------------------------------------------------------------------------- #
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation of two 1-D vectors; NaN when either has no variance.

    NaN is returned rather than 0.0 deliberately: "no variance" and "no relationship"
    are different facts, and averaging the second into the first is how a panel reports
    a null it never measured. The caller counts the NaNs and prints them.
    """
    if a.size != b.size or a.size < 3:
        return float("nan")
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def boot(vals, eids, *, n_boot: int, seed: int) -> dict:
    v = np.asarray(vals, dtype=np.float64)
    e = np.asarray(eids)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_windows": int(ok.sum()), "n_episodes": 0,
                "n_undefined": int((~ok).sum()), "estimator": "episode_cluster_bootstrap"}
    r = CI.episode_cluster_bootstrap(v[ok], e[ok], n_boot=n_boot, seed=seed)
    out = {k: r[k] for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator")}
    out["n_undefined"] = int((~ok).sum())
    out["separated"] = bool((out["lo"] > 0.0) or (out["hi"] < 0.0))
    return out


#: The candidate reward designs scored on the SAME banked fan, at zero extra GPU.
#: This is the constructive half: if a weight vector exists whose ranking prefers
#: feasible candidates, it is a DESIGN, not another 2,000-step arm.
WEIGHT_PANEL: dict[str, dict] = {
    "DEFAULT": dict(DEFAULT_WEIGHTS),
    "no_progress": {k: v for k, v in DEFAULT_WEIGHTS.items() if k != "progress"},
    "feasibility_x4": {**DEFAULT_WEIGHTS, "feasibility": 2.0},
    "feasibility_x4_no_progress": {k: v for k, v in
                                   {**DEFAULT_WEIGHTS, "feasibility": 2.0}.items()
                                   if k != "progress"},
    "progress_only": {"progress": 1.0},
    "feasibility_only": {"feasibility": 1.0},
    "comfort_only": {"comfort": 1.0},
}

FLAGS_OF_INTEREST = ("envelope", "kamm_over", "infeasible", "off_reach",
                     "ttc_below", "contact")
LEAD_ONLY = set(FS.LEAD_ONLY)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--expect-step", type=int, default=40284)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--lead-block", required=True)
    ap.add_argument("--windows", type=int, default=240)
    ap.add_argument("--window-seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lru", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-npz", default=None)
    a = ap.parse_args(argv)

    D.LEAD_MODE = "track"
    device = a.device if torch.cuda.is_available() else "cpu"

    class _A:  # the driver's `load` wants an argparse-ish namespace
        ckpt, config, expect_step = a.ckpt, a.config, a.expect_step
    model, cfg, _targs, prov = D.load(_A, device)
    model.eval()
    corp = D.open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, _lm, _li = D.load_lead_block(a.lead_block)
    pool = D.scoreable_windows(corp, lead)
    wis = sorted(random.Random(a.window_seed).sample(pool, min(len(pool), a.windows)))
    print(f"[rank] {len(wis)} windows of {len(pool)} scoreable · device={device} · "
          f"step={prov['step']}", flush=True)

    specs = {k: RewardSpec(weights=dict(v), dt=D.DT_REWARD_S) for k, v in WEIGHT_PANEL.items()}
    comp_names = sorted({n for w in WEIGHT_PANEL.values() for n in w})

    rows: list[dict] = []
    bank = {"fan2": [], "peak_g": [], "rank": [], "conf": [], "sel_idx": [],
            "v0": [], "lead5": [], "has_lead": [], "eid": [], "wi": []}
    for n in comp_names:
        bank[f"c_{n}"] = []
    for f in FS.FLAGS:
        bank[f"f_{f}"] = []

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(wis), a.batch):
            b = D.build_batch(corp, lead, wis[i:i + a.batch], device, with_gt=False)
            out = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"],
                        steps=int(prov["decoder_steps"]))
            fan = out["anchor_traj"]                                     # [B, N, 8, 2]
            fan2 = D.with_origin(fan[..., :D.N_REWARD_SLOTS, :])         # [B, N, 5, 2]
            # cand_dims=1 -- this is a FAN, not the training tensor. With the training
            # rank every window would be scored against every window's lead.
            ctx = D.reward_ctx(b, S5=fan2.shape[-2], cand_dims=1)
            per_comp = {n: RW.COMPONENTS[n](fan2, ctx) for n in comp_names}
            for n, v in per_comp.items():
                if tuple(v.shape) != tuple(fan.shape[:2]):
                    raise RuntimeError(f"component {n} returned {tuple(v.shape)} for a "
                                       f"fan of {tuple(fan.shape[:2])} — rank mismatch")
            lead5 = b["lead_track"].reshape(-1, 1, len(FS.GRID_S), 2)
            sc = FS.score_paths(fan2, b["v0"], lead5, lead_len_m=D.LEAD_LEN_DEFAULT_M)
            rank = out["sel_score"].detach().float()
            if out.get("reach_keep") is not None:
                rank = rank.masked_fill(~out["reach_keep"].bool(), float("-inf"))
            conf = out["anchor_logits"].detach().float()
            has = list(b["has_lead"])

            for j, wi in enumerate(b["wis"]):
                e_i, _t = corp.ds.index[wi]
                comp_j = {n: per_comp[n][j].cpu().numpy().astype(np.float64)
                          for n in comp_names}
                flag_j = {f: sc[f][j].cpu().numpy() for f in FS.FLAGS}
                pg = sc["peak_g"][j].cpu().numpy().astype(np.float64)
                row = {"wi": int(wi), "eid": int(e_i), "has_lead": bool(has[j])}
                # composed rewards for the whole weight panel
                rewards = {}
                for name, w in WEIGHT_PANEL.items():
                    rewards[name] = sum(w[n] * comp_j[n] for n in w)
                # ---- the correlations -------------------------------------- #
                for name, r in rewards.items():
                    row[f"rho__{name}__peak_g"] = spearman(r, pg)
                    for f in FLAGS_OF_INTEREST:
                        if f in LEAD_ONLY and not has[j]:
                            row[f"rho__{name}__{f}"] = float("nan")
                            continue
                        row[f"rho__{name}__{f}"] = spearman(r, flag_j[f].astype(np.float64))
                # per-component attribution against the envelope
                for n in comp_names:
                    row[f"rho__comp_{n}__envelope"] = spearman(
                        comp_j[n], flag_j["envelope"].astype(np.float64))
                    row[f"rho__comp_{n}__peak_g"] = spearman(comp_j[n], pg)
                # ---- controls ----------------------------------------------- #
                rng = np.random.default_rng(1000 + int(wi))
                row["rho__CTRL_random__peak_g"] = spearman(rng.random(pg.size), pg)
                row["rho__CTRL_constant__peak_g"] = spearman(np.zeros(pg.size), pg)
                row["rho__CTRL_selfpeak__peak_g"] = spearman(pg, pg)
                # ---- practical reads ---------------------------------------- #
                env = flag_j["envelope"].astype(bool)
                rd = rewards["DEFAULT"]
                row["fan_envelope_rate"] = float(env.mean())
                row["reward_mean_violating"] = float(rd[env].mean()) if env.any() else float("nan")
                row["reward_mean_feasible"] = float(rd[~env].mean()) if (~env).any() else float("nan")
                row["reward_gap_viol_minus_feas"] = (row["reward_mean_violating"]
                                                     - row["reward_mean_feasible"])
                order = np.argsort(-rd)
                row["top1_by_reward_envelope"] = float(env[order[0]])
                row["top8_by_reward_envelope"] = float(env[order[:8]].mean())
                row["top32_by_reward_envelope"] = float(env[order[:32]].mean())
                row["top1_by_reward_peak_g"] = float(pg[order[0]])
                row["fan_peak_g_mean"] = float(pg.mean())
                # the model's OWN ranking, for contrast: the selector already avoids it
                mo = np.argsort(-rank[j].cpu().numpy())
                row["top1_by_model_envelope"] = float(env[mo[0]])
                row["top8_by_model_envelope"] = float(env[mo[:8]].mean())
                rows.append(row)

                if a.out_npz:
                    bank["fan2"].append(fan2[j].cpu().numpy().astype(np.float32))
                    bank["peak_g"].append(pg.astype(np.float32))
                    bank["rank"].append(rank[j].cpu().numpy().astype(np.float32))
                    bank["conf"].append(conf[j].cpu().numpy().astype(np.float32))
                    bank["sel_idx"].append(int(out["sel_idx"][j]))
                    bank["v0"].append(float(b["v0"][j]))
                    bank["lead5"].append(b["lead_track"][j].cpu().numpy().astype(np.float32))
                    bank["has_lead"].append(bool(has[j]))
                    bank["eid"].append(int(e_i)); bank["wi"].append(int(wi))
                    for n in comp_names:
                        bank[f"c_{n}"].append(comp_j[n].astype(np.float32))
                    for f in FS.FLAGS:
                        bank[f"f_{f}"].append(flag_j[f].astype(np.uint8))
            if (i // max(a.batch, 1)) % 10 == 0:
                print(f"  [{i + len(b['wis'])}/{len(wis)}] {time.time() - t0:.0f}s", flush=True)

    eids = np.array([r["eid"] for r in rows])
    lead_mask = np.array([r["has_lead"] for r in rows])
    keys = [k for k in rows[0] if k.startswith("rho__")]
    panel = {}
    for k in keys:
        f = k.rsplit("__", 1)[-1]
        sub = lead_mask if f in LEAD_ONLY else np.ones(len(rows), bool)
        panel[k] = boot([r[k] for r, s in zip(rows, sub) if s],
                        eids[sub], n_boot=a.n_boot, seed=a.seed)
        panel[k]["population"] = "lead windows" if f in LEAD_ONLY else "all windows"
    practical = {k: boot([r[k] for r in rows], eids, n_boot=a.n_boot, seed=a.seed)
                 for k in rows[0] if not k.startswith("rho__")
                 and k not in ("wi", "eid", "has_lead")}

    res = {
        "_tool": TOOL, "_tier": "T0 instrument probe on the EVAL corpus (no driving claim)",
        "_evidence_class": "MEASURED (ours)",
        "_question": ("does the composed RL reward RANK envelope-violating candidates "
                      "above feasible ones, within a window's own fan?"),
        "ckpt": a.ckpt, "step": int(prov["step"]),
        "n_windows": len(rows), "n_episodes": int(len(set(eids.tolist()))),
        "n_lead_windows": int(lead_mask.sum()),
        "n_candidates_per_window": int(len(bank["peak_g"][0]) if bank["peak_g"] else 128),
        "n_candidate_scores": int(len(rows) * 128),
        "weight_panel": WEIGHT_PANEL, "flags": list(FLAGS_OF_INTEREST),
        "estimator": "episode_cluster_bootstrap over per-window Spearman rho",
        "n_boot": a.n_boot, "seed": a.seed,
        "kamm_limit": ("peak_g is the FINITE-DIFFERENCE friction load on a free-waypoint "
                       "fan; flyability's exact control-rolled instrument needs `controls` "
                       "and cannot be applied. It under-reports LEVELS by 1.21-1.85x; a "
                       "RANK correlation is invariant to a monotone under-report."),
        "panel": panel, "practical": practical,
        "per_window": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"[rank] -> {a.out}", flush=True)

    if a.out_npz:
        np.savez_compressed(a.out_npz, **{k: np.asarray(v) for k, v in bank.items()})
        print(f"[rank] banked fan -> {a.out_npz} "
              f"({os.path.getsize(a.out_npz) / 1e6:.1f} MB)", flush=True)

    # --- the headline, printed so a killed run still yields the answer -------- #
    print("\n=== CONTROLS (must read their known values) ===")
    for k in ("rho__CTRL_selfpeak__peak_g", "rho__CTRL_random__peak_g",
              "rho__CTRL_constant__peak_g"):
        p = panel[k]
        print(f"  {k:38s} {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] "
              f"n={p['n_windows']}/{p['n_episodes']}ep undef={p['n_undefined']}")
    print("\n=== rho(reward, flag) — POSITIVE means the reward PREFERS the violation ===")
    for name in WEIGHT_PANEL:
        for f in ("envelope", "peak_g", "kamm_over", "infeasible", "off_reach",
                  "ttc_below", "contact"):
            k = f"rho__{name}__{f}"
            if k not in panel:
                continue
            p = panel[k]
            print(f"  {name:28s} {f:11s} {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] "
                  f"n={p['n_windows']} eps={p['n_episodes']} undef={p['n_undefined']} "
                  f"{'SEP' if p.get('separated') else 'ns'}")
    print("\n=== per-component rho vs envelope (attribution) ===")
    for n in comp_names:
        p = panel[f"rho__comp_{n}__envelope"]
        q = panel[f"rho__comp_{n}__peak_g"]
        print(f"  {n:14s} envelope {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] "
              f"· peak_g {q['mean']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}] "
              f"undef={p['n_undefined']}")
    print("\n=== practical ===")
    for k in ("fan_envelope_rate", "reward_gap_viol_minus_feas", "top1_by_reward_envelope",
              "top8_by_reward_envelope", "top32_by_reward_envelope",
              "top1_by_model_envelope", "top8_by_model_envelope",
              "top1_by_reward_peak_g", "fan_peak_g_mean"):
        p = practical[k]
        print(f"  {k:30s} {p['mean']:+.5f} [{p['lo']:+.5f}, {p['hi']:+.5f}] "
              f"n={p['n_windows']} undef={p['n_undefined']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
