#!/usr/bin/env python3
"""RA-1 evaluator + verdict.  Spec: Project Steering/PREREG_HJ_STAGE0.md (H-HJ-SENS-1 / H-HJ-K-1 / H-HJ-COST-1).  CPU is enough.

    # one arm -> <out>/eval.json + <out>/perwindow.npz
    python stack/scripts/ra1_hjepa_eval.py eval --ckpt <run>/final.pt --eval-cache D:/tanitad/cache --p10-prior <p10_prior.pt> \
        --grid-json <eval_grid_windows.json> --train-cache D:/tanitad/cache --out <run>/eval
    # all arms -> verdict.json (the bars exactly as pre-registered + the branch they select)
    python stack/scripts/ra1_hjepa_eval.py verdict --run R0@0=<dir> --run A0:b0.08@0=<dir> --run A1:b0.08g100@0=<dir> ... --out verdict.json

TIERS (EVAL_DOCTRINE, stamped on every block).  SENS and COST are **T0** (teacher-forced: the rollout consumes RECORDED future actions;
the +6 s goal state is the TRUE future frame's latent) -- a world-model diagnostic, never "driving".  K and the four families are the
readout's own trajectory from the t0 observation + t0 kinematics, no recorded future action consumed: **T1-class open-loop trajectory
prediction** (the model supplies its own output; it does not drive).
ESTIMATOR: paired episode-cluster bootstrap (taniteval/ci.py), B = 2000, over the eval episodes.  It answers "would another draw of
EPISODES say this?" ONLY.  Training-seed variance is read from the 2-seed floor in `verdict`; inference-seed variance is 0 (the readout
and the rollout are deterministic; only the donor permutation of SENS is seeded, and it is fixed).
CONTROLS that must read their known values (recorded, and a failure sets `instrument_ok=false`): SENS self-swap == exactly 0.0;
P10 - P10 == +0.0000 [0, 0]; ADE(gt, gt) == 0; cache P10 vs the banked refit predictions (when --p10-ref-pred is given).

SELECTION (PREREG_HJ_STAGE0 Amendment 1).  The best A1 (beta, gamma) cell is chosen on the fixed 60-clip TRAIN HOLDOUT that every arm
was trained WITHOUT (`eval` scores it whenever `--train-cache` is given and the checkpoint carries its holdout record; the arrays
`hold_*` and the `holdout` block of eval.json), by mean seed-0 ADE@6s there.  `verdict` has NO fallback: an A1 run without holdout
numbers is REFUSED, never selected on the eval-141 windows that score K.  Every cell's eval-141 numbers are still reported, labelled
NON_DECIDING; eval-141 scores K once, for the chosen cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tanitad.data import ra1_pooled as RD  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import ra1_hjepa as R  # noqa: E402
from tanitad.models.hjepa import rho_within_episode  # noqa: E402
from tanitad.models.v8f_mini import load_p10  # noqa: E402
from taniteval import ci  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402

PAUSE = {"status": "PAUSED", "ruling": "PI 2026-09-19, PI_DECISION_QUEUE.md ITEM 25: the strategic layer is switched off temporarily "
         "until tactical and operative drive with high quality. Reported as PAUSED -- not a pass, not omitted.",
         "unpause": "verdict_refcv6.py --unpause-strategic is the PI's call"}
SENS_BAR, SENS_K = 0.10, 10


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else {"shape": list(o.shape), "dtype": str(o.dtype)}
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return jsonable(o.detach().cpu().numpy())
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = R.RA1Config(**ck["cfg"])
    model = R.build_model(cfg, ck["model"]["act_mean"], ck["model"]["act_std"])      # G-EVAL: the identical builder
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ck


def untrained_params(ck) -> list:
    """Parameters the optimiser never allocated state for = never received a gradient tensor (the post-training gate)."""
    have = set(ck["opt"]["state"].keys())
    return [n for i, n in enumerate(ck["param_names"]) if i not in have]


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def clip_latents(model, d, device, chunk=64) -> list:
    z1 = torch.cat([model.adapter(d["pool"][i:i + chunk].to(device).float()) for i in range(0, d["T"], chunk)])
    return [z.cpu() for z in model.levels_of(z1)]


def stat(x, eid, n_boot, seed):
    r = ci.episode_cluster_bootstrap(np.asarray(x, np.float64), eid, n_boot=n_boot, seed=seed)
    return {k: r[k] for k in ("point", "mean", "lo", "hi", "ci95", "n_windows", "n_episodes", "estimator") if k in r} or r


def pair(a, b, eid, n_boot, seed):
    r = ci.paired_episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(b, np.float64), eid, n_boot=n_boot, seed=seed)
    return {k: r[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "estimator", "degenerate") if k in r}


# ============================================================================================== holdout (the A1 selection set)
@torch.no_grad()
def holdout_eval(a, model, cfg, ck, prior, device):
    """PREREG Amendment 1: this arm's ADE@6s on the fixed TRAIN holdout -- the quantity the A1 cell is SELECTED on.  Returns
    ``(block, arrays)``; the arrays are ``hold_*`` (kept apart from the eval-141 ``ade6`` so a selection cannot read the wrong one).
    The holdout is RE-DERIVED from the train manifest and checked against the checkpoint's own record (clip list + sha256 + the train
    split's sha256), and its disjointness from the train split is asserted again here."""
    if not a.train_cache:
        return dict(status="NOT COMPUTED: no --train-cache given; `verdict` will REFUSE to select an A1 cell from this run"), {}
    rec = ck.get("holdout")
    if not rec:
        return dict(status="NOT COMPUTED: the checkpoint carries no holdout record (trained before PREREG Amendment 1); "
                           "`verdict` will REFUSE to select an A1 cell from this run"), {}
    train_all = [c for c, _ in RD.list_clips(a.train_cache, "train")]
    tr, hold = RD.split_train_holdout(train_all, rec["n_clips_flag"], rec["n"], rec["pool"])
    if hold != list(rec["clips"]) or RD.clips_sha256(hold) != rec["sha256"]:
        raise AssertionError("HOLDOUT DRIFT: the holdout the train cache gives now is not the one this checkpoint was trained without")
    if RD.clips_sha256(tr) != rec["train_sha256"]:
        raise AssertionError("TRAIN-SPLIT DRIFT: the train clips the cache gives now are not the ones this checkpoint was trained on")
    if set(tr) & set(hold):
        raise AssertionError("holdout/train overlap")
    bank = RD.ClipBank(a.train_cache, "train", prior, cfg.grid)
    bank.load(hold)
    wins = RD.eval_windows(bank, hold, None, a.hold_stride)
    lat = {c: clip_latents(model, bank.clips[c], device)[0] for c in hold}
    cidx = {c: i for i, c in enumerate(hold)}
    eid = np.array([cidx[c] for c, _ in wins])
    tt = np.array([t for _, t in wins])
    z1 = torch.stack([lat[c][t] for c, t in wins])
    f8n = torch.stack([bank.clips[c]["f8n"][t] for c, t in wins])
    p10 = torch.stack([bank.clips[c]["p10_traj"][t] for c, t in wins]).double().numpy()
    gt = torch.stack([bank.clips[c]["gt"][t] for c, t in wins]).double().numpy()
    pred = model.predict_traj(z1, f8n, torch.from_numpy(p10).float()).double().numpy()
    ade6, p6 = fk.ade(pred, gt), fk.ade(p10, gt)
    nb, sd = a.n_boot, a.seed
    blk = dict(status="OK", role="A1 SELECTION BASIS (PREREG_HJ_STAGE0 Amendment 1) -- never K", tier="T1-class open-loop trajectory",
               n_clips=len(hold), n_windows=len(wins), n_episodes=len(hold), stride=a.hold_stride, n=rec["n"], pool=rec["pool"],
               clips_sha256=rec["sha256"], train_sha256=rec["train_sha256"], disjoint_from_train=True,
               ade6_arm=stat(ade6, eid, nb, sd), ade6_p10=stat(p6, eid, nb, sd), ade6_arm_minus_p10=pair(ade6, p6, eid, nb, sd))
    return blk, dict(hold_eid=eid, hold_t=tt, hold_ade6=ade6, hold_p10_ade6=p6)


# ============================================================================================== eval
def run_eval(a) -> dict:
    t0 = time.time()
    device = torch.device("cpu")
    model, cfg, ck = load_model(a.ckpt, device)
    prior = load_p10(a.p10_prior)
    bank = RD.ClipBank(a.eval_cache, "eval", prior, cfg.grid)
    order = [c for c, _ in RD.list_clips(a.eval_cache, "eval")]
    bank.load(order)
    disj = None
    if a.train_cache:
        disj = D.assert_disjoint([c for c, _ in RD.list_clips(a.train_cache, "train")], order)
    wins = RD.eval_windows(bank, order, a.grid_json, a.eval_stride)
    N = len(wins)
    cidx = {c: i for i, c in enumerate(order)}
    eid = np.array([cidx[c] for c, _ in wins])
    tt = np.array([t for _, t in wins])
    lat = {c: clip_latents(model, bank.clips[c], device) for c in order}
    nlev = len(lat[order[0]])

    def gather(fn):
        return torch.stack([fn(bank.clips[c], lat[c], t) for c, t in wins])

    z1 = gather(lambda d, z, t: z[0][t])
    f8n = gather(lambda d, z, t: d["f8n"][t])
    p10 = gather(lambda d, z, t: d["p10_traj"][t]).double().numpy()
    gt = gather(lambda d, z, t: d["gt"][t]).double().numpy()
    with torch.no_grad():
        pred = model.predict_traj(z1, f8n, torch.from_numpy(p10).float()).double().numpy()
    res: dict = dict(schema="ra1-eval-v1", arm=cfg.arm, cfg=cfg.to_json(), cfg_sha12=cfg.sha12(), ckpt=os.path.abspath(a.ckpt),
                     ckpt_md5=file_md5(a.ckpt), ckpt_step=int(ck["step"]), n_windows=N, n_episodes=len(order),
                     grid_json=a.grid_json, disjointness=disj, n_boot=a.n_boot, estimator="paired_episode_cluster_bootstrap")
    nb, sd = a.n_boot, a.seed

    # ------------------------------------------------------------------ K  (T1-class open loop)
    ade6, ade2 = fk.ade(pred, gt), fk.ade(pred[:, :10], gt[:, :10])
    p6, p2 = fk.ade(p10, gt), fk.ade(p10[:, :10], gt[:, :10])
    res["K"] = dict(tier="T1-class open-loop trajectory (readout from the t0 observation + t0 kinematics; no recorded future action)",
                    ade6_arm=stat(ade6, eid, nb, sd), ade6_p10=stat(p6, eid, nb, sd), ade6_arm_minus_p10=pair(ade6, p6, eid, nb, sd),
                    ade2_arm=stat(ade2, eid, nb, sd), ade2_p10=stat(p2, eid, nb, sd), ade2_arm_minus_p10=pair(ade2, p2, eid, nb, sd),
                    note="negative delta = the arm beats the learned kinematic floor P10 (refit in the STEER unit)")
    ctrl = dict(p10_minus_p10=pair(p6, p6, eid, nb, sd), gt_ade_max=float(fk.ade(gt, gt).max()))
    ctrl["p10_minus_p10_ok"] = ctrl["p10_minus_p10"]["delta"] == 0.0 and ctrl["p10_minus_p10"]["lo"] == 0.0 and ctrl["p10_minus_p10"]["hi"] == 0.0
    if a.p10_ref_pred and os.path.exists(a.p10_ref_pred):
        ref = np.load(a.p10_ref_pred)
        ctrl["p10_vs_banked_refit"] = dict(shape_match=bool(ref.shape == p10.shape),
                                           max_abs_diff_m=float(np.abs(ref - p10).max()) if ref.shape == p10.shape else None)
    res["controls"] = ctrl

    # ------------------------------------------------------------------ the selection holdout (NOT K; see the module docstring)
    res["holdout"], hold_arrays = holdout_eval(a, model, cfg, ck, prior, device)
    if res["holdout"].get("status") == "OK":
        res["holdout"]["disjoint_from_eval141"] = D.assert_disjoint(ck["holdout"]["clips"], order)["overlap"] == 0

    # ------------------------------------------------------------------ four families (T1-class), arm and P10 side by side
    fam = {}
    try:
        from taniteval import four_families as FF
        for nm, path in (("arm", pred), ("p10", p10)):
            win = dict(pred=torch.from_numpy(path).float(), gt=torch.from_numpy(gt).float(), wp_steps=list(range(1, 31)), dt_s=R.DT,
                       eid=eid)
            f = FF.all_families(win, tactical_from_traj=True, tier="T1", n_boot=a.fam_boot, seed=sd)
            f["strategic"] = dict(PAUSE, n=N, underlying=f.get("strategic"))
            fam[nm] = f
    except Exception as e:                                                        # noqa: BLE001 -- reported, not hidden
        fam = dict(status="ERROR", error=repr(e)[:500], reason="four_families could not run; this is a WORK ITEM, not a pass")
    fam["lead_braking_stratum_and_distance_keeping"] = ("NOT COMPUTABLE from the pooled cache: it carries no lead-agent track "
                                                       "(needs the v9 join lead_gap); n = %d windows would have been scored" % N)
    res["families"] = fam

    # ------------------------------------------------------------------ strata (labels from GT ego motion; thresholds fixed in ra1_pooled.strata_labels)
    lab = [RD.strata_labels(bank.clips[c]["pose"], t) for c, t in wins]
    strata = {}
    for name in ("turn", "stop", "brake"):
        m = np.array([l[name] for l in lab])
        e = len(np.unique(eid[m])) if m.any() else 0
        blk = dict(n_windows=int(m.sum()), n_episodes=int(e))
        if m.sum() >= 10 and e >= 3:
            blk.update(ade6_arm=stat(ade6[m], eid[m], nb, sd), ade6_p10=stat(p6[m], eid[m], nb, sd),
                       ade6_arm_minus_p10=pair(ade6[m], p6[m], eid[m], nb, sd))
        else:
            blk["status"] = "n too small for an interval (need >= 10 windows and >= 3 episodes)"
        strata[name] = blk
    strata["lead_braking"] = {"status": "NOT COMPUTABLE: no lead track in the pooled cache"}
    res["strata"] = strata

    arrays = dict(eid=eid, t=tt, ade2=ade2, ade6=ade6, p10_ade2=p2, p10_ade6=p6, clips=np.array(order), **hold_arrays)

    # ------------------------------------------------------------------ SENS (T0) + rho + COST (T0): need the world model
    if model.stack is None:
        res["SENS"] = res["COST"] = res["rho"] = {"status": "n/a: R0 has no world model"}
    else:
        fut = torch.stack([bank.clips[c]["act"][t:t + 10] for c, t in wins])
        zs = z1
        etensor = torch.from_numpy(eid)
        selfswap = R.sensitivity_per_window(model, zs, fut, etensor, perm=torch.arange(N))
        sw = {k: v for k, v in selfswap.items() if not k.startswith("win/")}
        sens_main = R.sensitivity_per_window(model, zs, fut, etensor, seed=0)
        blk = dict(tier="T0 (recorded future actions consumed by the rollout)", k_steps_s=0.2, donor="fixed derangement over episodes, seed 0",
                   self_swap_control_max=float(max(abs(v) for v in sw.values())),
                   self_swap_control_ok=bool(all(v == 0.0 for v in sw.values())), channels={})
        for ch in ("all", "a", "steer", "v"):
            for k in (1, 5, 10):
                w = sens_main[f"win/{ch}/k{k}"]
                blk["channels"][f"{ch}/k{k}"] = dict(mean_of_ratios=float(sens_main[f"sens/{ch}/k{k}"]),
                                                      ratio_of_means=float(sens_main[f"sens_rom/{ch}/k{k}"]), ci=stat(w, eid, nb, sd))
                arrays[f"sens_{ch}_k{k}"] = w
        s10 = blk["channels"][f"steer/k{SENS_K}"]["mean_of_ratios"]
        v10 = blk["channels"][f"v/k{SENS_K}"]["mean_of_ratios"]
        blk["steer_k10"], blk["speed_k10"] = s10, v10
        blk["speed_over_steer"] = (v10 / s10) if s10 > 0 else None
        res["SENS"] = blk
        res["rho"] = dict(
            tier="T0", **{f"L{l + 1}": rho_within_episode(torch.cat([lat[c][l] for c in order]),
                                                          torch.cat([torch.full((bank.clips[c]['T'],), i) for i, c in enumerate(order)]))
                          for l in range(nlev)})
        cost = {}
        for c in order:
            idx = [i for i, (cc, _) in enumerate(wins) if cc == c]
            if not idx:
                continue
            m = R.cost_windows(lat[c], tt[idx])
            for k, v in m.items():
                cost.setdefault(k, np.zeros(N))[idx] = v
        cb = dict(tier="T0 (the +6 s goal is the TRUE future frame's latent)", definition="-Spearman(sq. distance to the t+30 latent, time); "
                  "goal frame excluded; L1 on its own 0.2 s clock, L2 on s=3 (0.6 s), L3 on s=9 (exploratory); 'matched' = L1 on L2's clock")
        for k, v in cost.items():
            cb[k] = stat(v, eid, nb, sd)
            arrays[f"cost_{k}"] = v
        if "L2_native" in cost:
            cb["L2_minus_L1_native"] = pair(cost["L2_native"], cost["L1_native"], eid, nb, sd)
            cb["L2_minus_L1_matched"] = pair(cost["L2_native"], cost["L1_matched_s3"], eid, nb, sd)
        res["COST"] = cb

    res["param_audit"] = dict(n_params=len(ck["param_names"]), untrained=untrained_params(ck),
                              rule="a parameter with no Adam state never received a gradient tensor; expected: none")
    res["instrument_ok"] = bool(ctrl["p10_minus_p10_ok"] and ctrl["gt_ade_max"] == 0.0 and (
        model.stack is None or res["SENS"]["self_swap_control_ok"]) and not res["param_audit"]["untrained"])
    res["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    json.dump(jsonable(res), open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "perwindow.npz"), **arrays)
    return res


# ============================================================================================== verdict
def _parse_key(k):
    base, seed = k.rsplit("@", 1)
    arm, _, cell = base.partition(":")
    return arm, cell, int(seed)


def selection_metric(z):
    """THE quantity an A1 cell is selected on: the per-window ADE@6s on the withheld TRAIN holdout (PREREG Amendment 1).  There is
    deliberately NO fallback to ``z['ade6']`` -- the eval-141 windows score K, and a probe that tunes on the windows it scores
    manufactures a result (TRAPS.md).  A run without ``hold_ade6`` raises here and `run_verdict` refuses before it gets this far."""
    return z["hold_ade6"]


def run_verdict(a) -> dict:
    runs = {}
    for spec in a.run:
        key, path = spec.split("=", 1)
        arm, cell, seed = _parse_key(key)
        z = np.load(os.path.join(path, "perwindow.npz"), allow_pickle=True)
        runs[(arm, cell, seed)] = dict(z=z, ev=json.load(open(os.path.join(path, "eval.json"))), path=path)
    ref = next(iter(runs.values()))["z"]
    for k, r in runs.items():
        assert np.array_equal(r["z"]["eid"], ref["eid"]) and np.array_equal(r["z"]["t"], ref["t"]), f"{k}: eval windows differ"
    eid, nb, sd = ref["eid"], a.n_boot, 0
    out: dict = dict(schema="ra1-verdict-v1", runs={f"{k[0]}:{k[1]}@{k[2]}": r["path"] for k, r in runs.items()},
                     estimator="paired episode-cluster bootstrap (episodes) + the 2-seed floor |A1_s0 - A1_s1| (training seed)",
                     n_windows=int(len(eid)), instrument_ok_all=all(r["ev"]["instrument_ok"] for r in runs.values()))
    ade = lambda k: runs[k]["z"]["ade6"]                                                  # noqa: E731
    p6 = ref["p10_ade6"]

    def cells(arm, seed):
        return sorted(c for (ar, c, s) in runs if ar == arm and s == seed)

    def finish():
        json.dump(jsonable(out), open(a.out, "w"), indent=1)
        return out

    # ---------------- the A1 cell: selected on the HOLDOUT, never on the eval-141 windows that score K (PREREG Amendment 1)
    out["selection_basis"] = "the withheld TRAIN holdout (PREREG_HJ_STAGE0 Amendment 1); eval-141 scores K once, for the chosen cell"
    a1c = cells("A1", 0)
    out["A1_best_cell"] = None
    if not a1c:
        out["status"] = "NO A1 seed-0 cell supplied"
        return finish()
    lacking = [c for c in a1c if "hold_ade6" not in runs[("A1", c, 0)]["z"].files]
    if lacking:
        out["status"] = ("A1 SELECTION REFUSED: no holdout numbers for A1 cell(s) %s -- the cell is chosen on the withheld TRAIN holdout and "
                         "NEVER on the eval-141 windows that score K. Re-run `eval` with --train-cache on a checkpoint trained with --holdout-n."
                         % lacking)
        out["A1_cells_seed0_eval141_ade6_NON_DECIDING"] = {c: float(ade(("A1", c, 0)).mean()) for c in a1c}
        return finish()
    # every run that carries a holdout must have scored the SAME windows of the SAME clips (identical across arms by construction)
    hr = {k: r for k, r in runs.items() if "hold_ade6" in r["z"].files}
    h0 = next(iter(hr.values()))
    for k, r in hr.items():
        assert np.array_equal(r["z"]["hold_eid"], h0["z"]["hold_eid"]) and np.array_equal(r["z"]["hold_t"], h0["z"]["hold_t"]), \
            f"{k}: holdout windows differ"
    shas = {r["ev"].get("holdout", {}).get("clips_sha256") for r in hr.values()}
    assert len(shas) == 1 and None not in shas, f"runs were trained without the SAME holdout (clip-list sha256s: {sorted(map(str, shas))})"
    best = min(a1c, key=lambda c: float(selection_metric(runs[("A1", c, 0)]["z"]).mean()))
    hb = runs[("A1", best, 0)]["ev"].get("holdout", {})
    out["A1_best_cell"] = best
    out["A1_cells_seed0_holdout_ade6"] = {c: float(selection_metric(runs[("A1", c, 0)]["z"]).mean()) for c in a1c}
    out["A1_cells_seed0_eval141_ade6_NON_DECIDING"] = {
        c: dict(ade6=float(ade(("A1", c, 0)).mean()), arm_minus_p10=float((ade(("A1", c, 0)) - p6).mean())) for c in a1c}
    out["holdout"] = dict(clips_sha256=next(iter(shas)), n_clips=hb.get("n_clips"), n_windows=int(len(h0["z"]["hold_t"])), pool=hb.get("pool"),
                          n_runs_scored=len(hr), ade6_mean_by_run={f"{k[0]}:{k[1]}@{k[2]}": float(selection_metric(r["z"]).mean())
                                                                   for k, r in sorted(hr.items())})
    out["A1_best_rule"] = ("lowest seed-0 mean ADE@6s on the withheld TRAIN holdout (%s clips = the last %s of the first %s in sha12 order, %d windows, "
                           "clip-list sha256 %s...), withheld from training for EVERY arm -- PREREG_HJ_STAGE0 Amendment 1. NOT selected on eval-141: "
                           "eval-141 scores K once, for this cell; every cell's eval-141 ADE@6s is listed under "
                           "A1_cells_seed0_eval141_ade6_NON_DECIDING and decides nothing."
                           % (hb.get("n_clips"), hb.get("n"), hb.get("pool"), len(h0["z"]["hold_t"]), next(iter(shas))[:12]))
    seeds = [0, 1]
    # ---------------- A1x: the instrument gate
    kx = ("A1x", best, 0) if ("A1x", best, 0) in runs else (("A1x", "", 0) if ("A1x", "", 0) in runs else None)
    if kx:
        d = pair(ade(kx), ade(("A1", best, 0)), eid, nb, sd)
        out["A1x_control"] = dict(delta_A1x_minus_A1_ade6=d, loses_point=d["delta"] > 0, loses_separated=bool(d["separated"] and d["delta"] > 0),
                                  verdict=("OK: A1x loses to A1, separated" if d["separated"] and d["delta"] > 0 else
                                           "UNRESOLVED: A1x is worse but not separated -- treat as INSTRUMENT BROKEN until it is" if d["delta"] > 0 else
                                           "INSTRUMENT BROKEN: A1x does not lose to A1 -- nothing else is read until this is fixed"))
    else:
        out["A1x_control"] = {"verdict": "NOT RUN -- the instrument gate is open; no other verdict below is admissible"}
    # ---------------- SENS
    def a0_for(seed):
        want = best.split("g")[0] if "g" in best else best               # 'b0.08g100' -> 'b0.08'
        for c in cells("A0", seed):
            if c == want:
                return ("A0", c, seed)
        c0 = cells("A0", seed)
        return ("A0", c0[0], seed) if c0 else None

    sens = {}
    for s in seeds:
        k1, k0 = ("A1", best, s), a0_for(s)
        if k1 not in runs or k0 is None or k0 not in runs:
            sens[f"seed{s}"] = "NOT EVALUABLE: missing run"
            continue
        w1, w0 = runs[k1]["z"][f"sens_steer_k{SENS_K}"], runs[k0]["z"][f"sens_steer_k{SENS_K}"]
        d = pair(w1, w0, eid, nb, sd)
        v1 = float(w1.mean())
        sens[f"seed{s}"] = dict(A1_steer_k10=v1, A0_steer_k10=float(w0.mean()), A0_cell=k0[1], A1_minus_A0=d, passes_level=v1 >= SENS_BAR,
                                separated_above_A0=bool(d["separated"] and d["delta"] > 0), speed_k10_A1=float(runs[k1]["z"][f"sens_v_k{SENS_K}"].mean()))
    out["SENS"] = dict(bar=f"A1 level-1 steer-sensitivity (k={SENS_K}) >= {SENS_BAR} AND separated above A0, at BOTH seeds", per_seed=sens,
                       PASS=bool(all(isinstance(v, dict) and v["passes_level"] and v["separated_above_A0"] for v in sens.values()) and len(sens) == 2))
    # ---------------- K
    floor = None
    if ("A1", best, 0) in runs and ("A1", best, 1) in runs:
        floor = abs(float(ade(("A1", best, 0)).mean()) - float(ade(("A1", best, 1)).mean()))
    out["training_seed_floor_A1"] = floor
    cand = {}
    for arm in ("R0", "A1", "A2"):
        row = {}
        for s in seeds:
            key = (arm, best, s) if arm == "A1" else next((k for k in runs if k[0] == arm and k[2] == s), None)
            if key is None or key not in runs:
                row[f"seed{s}"] = "missing"
                continue
            d = pair(ade(key), p6, eid, nb, sd)
            row[f"seed{s}"] = dict(ade6=float(ade(key).mean()), p10_ade6=float(p6.mean()), arm_minus_p10=d, below_p10=d["delta"] < 0,
                                   separated=bool(d["separated"]), beats_2x_floor=(floor is not None and abs(d["delta"]) > 2 * floor))
        row["mean_ade6"] = float(np.mean([v["ade6"] for v in row.values() if isinstance(v, dict)])) if any(isinstance(v, dict) for v in row.values()) else None
        row["PASS"] = bool(all(isinstance(row.get(f"seed{s}"), dict) and row[f"seed{s}"]["below_p10"] and row[f"seed{s}"]["separated"]
                               and row[f"seed{s}"]["beats_2x_floor"] for s in seeds))
        cand[arm] = row
    ranked = sorted((r["mean_ade6"], arm) for arm, r in cand.items() if r["mean_ade6"] is not None)
    out["K"] = dict(bar="best(R0, A1, A2) - P10 ADE@6s < 0, separated at both seeds, |delta| > 2x the training-seed floor", candidates=cand,
                    best_arm=ranked[0][1] if ranked else None, PASS=bool(ranked and cand[ranked[0][1]]["PASS"]))
    # ---------------- COST
    cost = {}
    for arm in ("A0", "A1", "A2"):
        for key in [k for k in runs if k[0] == arm and (arm != "A1" or k[1] == best)]:
            if "cost_L2_native" in runs[key]["z"].files:
                z = runs[key]["z"]
                cost[f"{arm}:{key[1]}@{key[2]}"] = dict(L2_minus_L1_native=pair(z["cost_L2_native"], z["cost_L1_native"], eid, nb, sd),
                                                       PASS=bool(pair(z["cost_L2_native"], z["cost_L1_native"], eid, nb, sd)["separated"]
                                                                 and z["cost_L2_native"].mean() > z["cost_L1_native"].mean()))
    out["COST"] = dict(bar="-Spearman(level-2 distance to the +6 s latent, time) > the level-1 value, paired over episodes", per_run=cost)
    # ---------------- exploratory A1h
    hk = [k for k in runs if k[0] == "A1h"]
    if hk:
        h = {}
        for k in hk:
            z = runs[k]["z"]
            h[f"{k[1]}@{k[2]}"] = dict(steer_k10=float(z[f"sens_steer_k{SENS_K}"].mean()), speed_k10=float(z[f"sens_v_k{SENS_K}"].mean()),
                                       ade6=float(z["ade6"].mean()),
                                       steer_k10_minus_A1_same_seed=(pair(z[f"sens_steer_k{SENS_K}"], runs[("A1", best, k[2])]["z"][f"sens_steer_k{SENS_K}"],
                                                                          eid, nb, sd) if ("A1", best, k[2]) in runs else "missing A1"))
        out["A1h_EXPLORATORY"] = dict(note="registered 2026-10-17 before any RA-1 data; copycat remedy (arXiv 2010.14876); does NOT decide H-HJ-SENS-1", runs=h)
    # ---------------- branch (PREREG outcomes, verbatim order)
    sp, kp = out["SENS"]["PASS"], out["K"]["PASS"]
    gate = out["A1x_control"]["verdict"].startswith("OK")
    if not gate:
        branch = "A1x GATE: " + out["A1x_control"]["verdict"]
    elif sp and kp:
        branch = "SENS PASS + K PASS => frozen + adapter suffices; build refav2 (RA-2)"
    elif sp and not kp:
        branch = "SENS PASS + K FAIL => the frozen ceiling is reached; next arm = the same with LoRA r 32 (T-B)"
    else:
        branch = "SENS FAIL => raise gamma to 200, then IDM at level 2 only, then a time-repulsion floor (|lambda_sim| <= 0.06), in that order"
    if gate and cand.get("R0", {}).get("PASS") and not cand.get("A1", {}).get("PASS"):
        branch += " | R0 alone passes K while A1 does not => the WM objective is hurting the readout: report it, move the WM to v8F's chassis question"
    out["branch"] = branch
    return finish()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--eval-cache", required=True)
    e.add_argument("--p10-prior", required=True)
    e.add_argument("--grid-json", default=None, help="eval_grid_windows.json (141 eps / 2,399 windows); else every valid frame")
    e.add_argument("--eval-stride", type=int, default=1)
    e.add_argument("--train-cache", default=None, help="assert train/eval clip-id disjointness against this cache's train manifest, AND score the "
                   "withheld TRAIN holdout from it (PREREG Amendment 1: the A1 selection set; REQUIRED for any A1 cell to be selectable)")
    e.add_argument("--hold-stride", type=int, default=1, help="holdout windows: every valid frame at this stride")
    e.add_argument("--p10-ref-pred", default=None, help="banked p10_eval_pred6.npy [2399,30,2] for a cross-check")
    e.add_argument("--out", required=True)
    e.add_argument("--n-boot", type=int, default=2000)
    e.add_argument("--fam-boot", type=int, default=2000)
    e.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("verdict")
    v.add_argument("--run", action="append", required=True, help="ARM[:cell]@seed=DIR  e.g. A1:b0.08g100@0=runs/A1_s0/eval")
    v.add_argument("--out", required=True)
    v.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "eval":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        r = run_eval(a)
        print(json.dumps(jsonable({k: r[k] for k in ("arm", "ckpt_step", "n_windows", "instrument_ok")}), default=str))
        print("K ade6 arm-P10:", json.dumps(jsonable(r["K"]["ade6_arm_minus_p10"])))
        if isinstance(r.get("SENS"), dict) and "steer_k10" in r["SENS"]:
            print("SENS steer/k10 %.4f  speed/k10 %.4f  self-swap ok=%s" % (r["SENS"]["steer_k10"], r["SENS"]["speed_k10"], r["SENS"]["self_swap_control_ok"]))
        return 0 if r["instrument_ok"] else 3
    r = run_verdict(a)
    print(json.dumps(jsonable({k: r.get(k) for k in ("A1_best_cell", "A1_best_rule", "status", "A1x_control", "branch")}), indent=1, default=str))
    return 0 if r.get("A1_best_cell") else 4                  # 4 = no A1 cell could be selected (no cell / no holdout numbers): read `status`


if __name__ == "__main__":
    sys.exit(main())
