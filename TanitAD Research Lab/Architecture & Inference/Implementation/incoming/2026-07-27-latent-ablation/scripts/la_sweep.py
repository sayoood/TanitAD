#!/usr/bin/env python3
"""THE LATENT ABLATION sweep — ~25 GPU-min on an idle A40 (pod2 only).

THE QUESTION. Is the blind-driving horizon the world model **predicting its own
semantics in imagination**, or **kinematic integration of the action channel**?
Four MEASURED results say integrator. This sweep ablates the LATENT and keeps the
action channel, which is the contrast that can separate them.

WHAT IS AND IS NOT REIMPLEMENTED.
* ``taniteval.blindimag.blind_rollout`` is used **verbatim**. The ablations are
  new ``state_source`` values inside it; **exactly one line** of the loop differs
  between them (the latent appended at step j+1). Nothing about the action path,
  the readout, the window or the SE(2) accumulation changes.
* ``bi_run._prepare`` is imported, not copied — it owns the encode pass and
  ``build_windows``, which is what makes the window set identical to Rung 0/1's.
* The arm-loop bookkeeping is local only because the FIXED-POINT probe needs to
  store four extra per-step tensors that ``bi_run._run_arms`` does not carry.

⭐ WHY alpha = 1.0 IS THE DECISIVE ROW. At ``blend=1.0`` the fed steer/accel are
the last OBSERVED action for every arm and cannot vary with the latent
(test-pinned: ``test_at_alpha_one_the_fed_action_is_identical_across_state_sources``),
and the ``v0`` channel is the CONSTANT true speed for every arm. So at alpha = 1
any difference between two arms is **the latent and nothing else**. At alpha < 1
the own-kinematic action is a function of the decoded Delta-pose, so a latent
ablation also perturbs the action — that confound is real
(``test_at_alpha_zero_the_fed_action_DOES_depend_on_the_latent``) and alpha = 1
is what removes it.

⛔ GATES (``la_analyze.py`` blocks unless they pass): six committed anchors must
reproduce their dense ``de`` within 1e-4 m; the three identity-permutation
self-test arms must be BIT-IDENTICAL to the arms they algebraically reduce to;
and no ablation arm may be identical to the intact one.

Host: **pod2 only** (A40, verified idle 0 MiB / 0 %). pod1 (training), pod3 and
the eval pod are never touched; the val cache is read only.

Usage (pod2):
    PYTHONPATH=/root/bi:/root/taniteval:/root/TanitAD/stack:/root/TanitAD/stack/scripts \
    OMP_NUM_THREADS=8 python3 la_sweep.py --out /workspace/latab/perwindow \
        --episodes 600 --kmax 185 --batch 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

for _p in ("/root/bi", "/root/taniteval", "/root/TanitAD/stack",
           "/root/TanitAD/stack/scripts"):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                              # noqa: E402

import bi_run as R                                        # noqa: E402
from taniteval import blindimag as bi                     # noqa: E402

STR = "str"                                               # calibrated readout
SEED = 0                                                  # the reported seed

#: The latent channel. `full_obs` is PRIVILEGED (it reads the true future) and is
#: a DIAGNOSTIC ceiling, never deployable. Fixed before any number existed.
SOURCES = ("imagination", "frozen_last", "frozen_other", "shuffled",
           "shuffled_obs", "mean_latent", "zero_latent", "full_obs")
#: The action damping. 0.25 = the genuinely model-driven point; 1.0 = the
#: UNCONFOUNDED row (fed action identical across every state source).
ALPHAS = (0.0, 0.25, 0.75, 1.0)

#: (name, state_source, action_source, update_speed_channel, readout_level)
ARMS: list[tuple] = []


def _act(alpha: float) -> str:
    """alpha = 0 uses the BARE spec so the arm is bit-identical to the committed
    anchor by construction (``parse_action_source`` gives a bare source an EMPTY
    modifier dict)."""
    return "own_kinematic" if alpha == 0.0 else f"own_kinematic|blend={alpha:g}"


# --- the 8 x 4 grid ---------------------------------------------------------- #
for _s in SOURCES:
    for _a in ALPHAS:
        ARMS.append((f"{_s}__a{_a:g}", _s, _act(_a), False, STR))

# --- ANCHORS: committed arms re-rolled to gate window-set identity ----------- #
#   imagination/frozen_last at alpha=0 ARE the `own` anchors (grid rows above).
#   These two add the `hold_last` anchors, which also prove blend=1 == hold_last
#   on the FROZEN arm (Rung 1 proved it only on the imagination arm).
ARMS += [("anchor_a_hold", "imagination", "hold_last", False, STR),
         ("anchor_b_hold", "frozen_last", "hold_last", False, STR)]

# --- PLUMBING SELF-TEST: must be BIT-IDENTICAL to what they reduce to -------- #
ARMS += [("selftest__shuffled_id", "shuffled|seed=-1", _act(0.25), False, STR),
         ("selftest__shufobs_id", "shuffled_obs|seed=-1", _act(0.25), False, STR),
         ("selftest__frozother_id", "frozen_other|seed=-1", _act(0.25),
          False, STR)]

# --- SEED ROBUSTNESS of the derangement (the reported arms use seed 0) ------- #
for _sd in (1, 2):
    for _a in (0.25, 1.0):
        ARMS.append((f"shuffled_seed{_sd}__a{_a:g}", f"shuffled|seed={_sd}",
                     _act(_a), False, STR))

#: Arms whose FIXED-POINT probe is kept. All of them — 4 extra [N,185] tensors
#: per arm is ~1.8 MB, and the probe is the brief's second question.
KEEP_LATENT = tuple(n for n, *_ in ARMS)


@torch.no_grad()
def run_arms(model, sro, readouts, recs, poses_by_ep, k, batch, arms):
    """One pass over the fixed window set per arm. ``blind_rollout`` verbatim."""
    store = {n: [] for n, *_ in arms}
    psis = {n: [] for n, *_ in arms}
    spds = {n: [] for n, *_ in arms}
    lat = {n: {q: [] for q in ("lat_dz", "lat_d0", "lat_cos0", "lat_norm")}
           for n, *_ in arms if n in KEEP_LATENT}
    shared = {"gt": [], "gt_yaw": [], "cv": [], "hold_v0": [], "eid": [],
              "speed": [], "head_deg": [], "t0": [], "ep_i": []}
    nb = (len(recs) + batch - 1) // batch
    for i_b, b in enumerate(bi.batch_windows(recs, poses_by_ep, k, batch=batch)):
        for key in ("gt", "gt_yaw", "cv", "hold_v0", "speed", "head_deg"):
            shared[key].append(b[key if key != "gt" else "gt_pos"].float())
        for key in ("eid", "t0", "ep_i"):
            shared[key] += b[key]
        gtd = b["gt_dpose"].to(b["states"].device)
        vl = b["v_last"].to(b["states"].device)
        for name, ss, asrc, upd, lvl in arms:
            r = bi.blind_rollout(
                model.predictor, b["states"], b["actions"],
                sro if lvl == "op" else readouts[lvl], k,
                state_source=ss, action_source=asrc,
                future_actions=b["future_actions"], obs_states=b["obs_states"],
                gt_step_dpose=gtd, v_last=vl, update_speed_channel=bool(upd),
                latent_perm_seed=SEED, latent_stats=name in lat)
            store[name].append(r["waypoints"].cpu().float())
            psis[name].append(r["psi"].cpu().float())
            spds[name].append(r["pred_speed"].cpu().float())
            for q in lat.get(name, ()):
                lat[name][q].append(r[q].cpu().float())
        if i_b % 2 == 0:
            print(f"  [rollout] batch {i_b + 1}/{nb}", flush=True)
    out = {"pred": {k2: torch.cat(v) for k2, v in store.items()},
           "psi": {k2: torch.cat(v) for k2, v in psis.items()},
           "pred_speed": {k2: torch.cat(v) for k2, v in spds.items()},
           "latent": {n: {q: torch.cat(v) for q, v in d.items()}
                      for n, d in lat.items()}}
    for k2 in ("gt", "gt_yaw", "cv", "hold_v0", "speed", "head_deg"):
        out[k2] = torch.cat(shared[k2])
    for k2 in ("eid", "t0", "ep_i"):
        out[k2] = shared[k2]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="flagship-30k")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--kmax", type=int, default=185)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--enc-batch", type=int, default=32)
    a = ap.parse_args()
    torch.manual_seed(0)
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    print(f"[latab] {len(ARMS)} arms at K={a.kmax}, batch={a.batch}", flush=True)

    h, eps, cache, poses_by_ep, recs, parity, t_enc = R._prepare(a, "cuda")
    t0 = time.time()
    res = run_arms(h["model"], h["step_readout"], h["grounding"].step, recs,
                   poses_by_ep, a.kmax, a.batch, ARMS)
    t_roll = time.time() - t0

    meta = {"block": bi.BLOCK, "version": bi.VERSION, "stage": "latent_ablation",
            "arm_ckpt": a.arm, "ckpt_step": h.get("step"),
            "kmax": a.kmax, "stride": a.stride, "batch": a.batch,
            "episodes_requested": a.episodes, "n_windows": len(recs),
            "n_episode_clusters": len(set(res["eid"])),
            "val_parity": parity, "readout_level": STR,
            "latent_perm_seed": SEED, "dt_s": bi.DT,
            "speed_scale": bi.SPEED_SCALE,
            "state_sources": list(SOURCES), "alphas": list(ALPHAS),
            "arms": {n: {"state_source": s, "action_source": act,
                         "update_speed_channel": bool(u), "readout_level": lv}
                     for n, s, act, u, lv in ARMS},
            "derangement": ("random roll with a nonzero offset, deterministic "
                            "in (seed, step); preserves the batch multiset "
                            "EXACTLY so the per-step marginal of an ablated arm "
                            "equals the intact arm's"),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds_encode": round(t_enc, 1),
            "seconds_rollout": round(t_roll, 1)}

    torch.save({"pred": res["pred"], "psi": res["psi"],
                "pred_speed": res["pred_speed"], "latent": res["latent"],
                "gt": res["gt"], "gt_yaw": res["gt_yaw"], "cv": res["cv"],
                "hold_v0": res["hold_v0"], "speed": res["speed"],
                "head_deg": res["head_deg"], "eid": res["eid"],
                "t0": res["t0"], "ep_i": res["ep_i"], "meta": meta},
               str(outd / f"latab_sweep_K{a.kmax}.pt"))
    (outd / f"latab_meta_K{a.kmax}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[latab] done: encode {t_enc:.0f}s, rollout {t_roll:.0f}s, "
          f"{len(recs)} windows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
