"""E1b EVAL — paired closed-loop verdict, FT vs base (run AFTER the FT completes).

Re-runs the E1a real-footage closed loop (e1a_horizon.rollout, loop body VERBATIM
+ two ADDITIVE capture lines, see make_capture_rollout.py) at K=20 and K=185 on
the SAME held-out eval episodes E1a used (physicalai-val-heldout-79d4e3d2d4c6,
MEASURED disjoint from the parity-train the FT mined from) for BOTH the base and
the FT checkpoint, then reports the PRE-REGISTERED primary + guardrails with the
PAIRED episode-cluster bootstrap (taniteval/ci.py, B=2000) on identical windows.

  PRIMARY : junction corridor-departure-rate @ K=185, paired Delta(FT - base).
            SUCCESS = CI-separated LOWER (hi < 0).
  GUARDRAIL(a): open-loop ADE@2s, PAIRED Delta(FT - base) on identical windows.
            PASS = CI includes 0 or is separated-LOWER (not separated-worse).
  GUARDRAIL(b): open-loop ANCHOR block on the same held-out windows — anchor-cls
            accuracy, anchor-cls CE and anchor traj-recon L1 toward the GT-nearest
            anchor. This is refc_train.compute_losses' anchor block (lines 268-276)
            with the GT target, i.e. exactly the block the CL-SFT re-purposed.
            PRE_REGISTRATION §3 names it the REF-C stand-in for a WM canary.
  GUARDRAIL(c): [⛔ VOID BY CONSTRUCTION — REMOVED FROM THE CONJUNCTION 2026-07-26.
            sup(ratio_arr) = 1.298888 < 1.30, so this could never fail; it passed
            139/139 repo-wide and was the ONLY guardrail that 'held' in this run.
            Kept here as a record of what was pre-registered, NOT as a live test.
            Real instrument: taniteval.ood.verdict(). RETRACTION_LOG class C13.]
            OOD-envelope ratio must stay in band (<= ~1.30, E1a's measured
            base value) — a departure improvement bought by moving the loop out of
            the measured perturbation envelope would be confounded.
  Also reported: overall + longitudinal @ K=185, peak XTE, and K=20 (the standing
            2 s instrument) so any 2 s regression is visible.
  M1 CONTRACT: no ADE is reported without its (lateral, longitudinal) split
            (taniteval/lateral.py, both ego and frenet modes).

Base per-window arrays were not persisted by the original E1a run, so BOTH arms
are re-rolled here on identical windows — that is what makes the bootstrap PAIRED.
The base arm therefore also REPRODUCES E1a's headline as a built-in control.

ESTIMATOR: episode-cluster bootstrap only (taniteval/ci.py, B=2000), paired for
every two-arm delta. `overlapping_holdout_se` is used NOWHERE.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# ORDER MATTERS. `taniteval` must bind to /workspace/TanitAD/taniteval BEFORE
# e1a_horizon is imported, because e1a_horizon prepends /root/taniteval (a stale
# checkout whose package has NO ci module) to sys.path at import time.
for _p in ("/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
           "/workspace/e1a_e2a", "/workspace/TanitAD/taniteval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# The capture-patched e1a_horizon lives BESIDE this file and MUST win over the
# pristine /workspace/e1a_e2a original. `insert(0, ...)` is not enough: this
# directory is already sys.path[0] (script dir), so the inserts above jump AHEAD
# of it and the unpatched module wins. Re-seat it explicitly.
_CAP_DIR = str(Path(__file__).resolve().parent)
while _CAP_DIR in sys.path:
    sys.path.remove(_CAP_DIR)
sys.path.insert(0, _CAP_DIR)

import taniteval  # noqa: E402
assert taniteval.__file__.startswith("/workspace/TanitAD/taniteval/"), \
    f"taniteval bound to the wrong checkout: {taniteval.__file__}"
from taniteval import ci as _pkg_ci  # noqa: E402,F401  (bind submodule now)
from taniteval import driving as _pkg_drv  # noqa: E402,F401

import e1a_horizon as e1a  # noqa: E402   (/workspace/e1b copy = E1a + capture)
# Fail LOUD at import if the pristine E1a module won the race: without the
# additive capture the M1 lateral split would silently vanish from the report.
assert str(Path(e1a.__file__).resolve().parent) == _CAP_DIR, \
    f"e1a_horizon resolved to {e1a.__file__}, not the capture copy in {_CAP_DIR}"
assert "_cap_pred" in Path(e1a.__file__).read_text(), \
    f"{e1a.__file__} lacks the E1b additive capture (run make_capture_rollout.py)"
import taniteval_ci as _ci  # noqa: E402  (md5-identical to taniteval/ci.py)
import taniteval_lateral as LAT  # noqa: E402  (byte copy of taniteval/lateral.py)
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from driving_diagnostic import gt_ego_waypoints  # noqa: E402

B_BOOT = 2000
KNOTS_S = (0.5, 1.0, 1.5, 2.0)      # WP_STEPS 5,10,15,20 @ 10 Hz
KNOT_DT = 0.5                        # spacing BETWEEN the 4 predicted knots


def _boot(v, eid, reduce="mean"):
    return _ci.episode_cluster_bootstrap(np.asarray(v, float), eid,
                                         reduce=reduce, n_boot=B_BOOT)


def _paired(a, b, eid, reduce="mean"):
    return _ci.paired_episode_cluster_bootstrap(
        np.asarray(a, float), np.asarray(b, float), eid,
        n_boot=B_BOOT, reduce=reduce)


# --------------------------------------------------------------------------- #
# OPEN LOOP — ADE@2s (guardrail a) + the anchor block (guardrail b), per window #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def openloop_full(model, episodes, device, stride=8, batch=16):
    """Per-window open-loop arrays + keys, so BOTH guardrails can be PAIRED.

    The ADE loop is e1a.openloop_canary's loop verbatim (same starts, same
    stride, same 2 denoise steps); the anchor block mirrors
    refc_train.compute_losses lines 268-276 with the GT target."""
    W, WP = e1a.W, e1a.WP_STEPS
    P, G, ade, acc, ce, l1, eids, keys, spd = [], [], [], [], [], [], [], [], []
    anchors = model.decoder.anchors.detach().float()          # [Na,4,2]
    for ei, ep in enumerate(episodes):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames.float()
        poses = ep.poses.float()
        T = fr.shape[0]
        starts = list(range(0, T - W - max(WP), stride))
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            frames = torch.stack([fr[t0:t0 + W] for t0 in ch]).to(device)
            last = torch.tensor([t0 + W - 1 for t0 in ch])
            v0 = poses[last, 3].to(device)
            out = model(frames, nav_cmd=None, v0=v0, steps=2)
            pred = out["traj"].float()                        # [B,4,2]
            gt = gt_ego_waypoints(poses, last).to(device).float()
            b = pred.shape[0]
            ade.append(torch.linalg.norm(pred - gt, dim=-1).mean(1).cpu())
            dist = ((gt[:, None] - anchors[None]) ** 2).sum(dim=(-1, -2))
            a_star = dist.argmin(dim=1)
            logits = out["anchor_logits"].float()
            acc.append((logits.argmax(dim=1) == a_star).float().cpu())
            ce.append(F.cross_entropy(logits, a_star, reduction="none").cpu())
            recon = out["anchor_traj"].float()[torch.arange(b, device=device), a_star]
            l1.append((recon - gt).abs().mean(dim=(-1, -2)).cpu())
            P.append(pred.cpu()); G.append(gt.cpu())
            spd.append(poses[last, 3].clone())
            eids += [str(ei)] * b
            keys += [(ei, int(t)) for t in ch]
    return {"key": keys, "eid": eids,
            "ade2s": torch.cat(ade).numpy(),
            "anchor_acc": torch.cat(acc).numpy(),
            "anchor_ce": torch.cat(ce).numpy(),
            "anchor_traj_l1": torch.cat(l1).numpy(),
            "speed": torch.cat(spd).numpy(),
            "pred": torch.cat(P), "gt": torch.cat(G)}


# --------------------------------------------------------------------------- #
# CLOSED LOOP — per-window observables at one horizon                          #
# --------------------------------------------------------------------------- #
def per_window(model, episodes, device, K, primary, junction_deg, stride, batch,
               ood):
    pw = e1a.rollout(model, episodes, device, K, stride, batch)
    lat = pw["lat"].numpy()                                   # [N,K] |XTE| m
    yaw = pw["yaw"].numpy()                                   # [N,K] |dpsi| deg
    ratio = ood.ratio_arr(lat, yaw)                           # P1-mapped OOD
    hd = pw["hd2s"].numpy(); spd = pw["speed"].numpy()
    junc = hd >= junction_deg
    long_ = (~junc) & (spd >= np.median(spd))
    return {
        "key": [(int(a), int(b)) for a, b in zip(pw["epi"], pw["t0"])],
        "eid": pw["eid"],
        "dep": (lat > primary).mean(1),
        "win_dep": (lat > primary).any(1).astype(float),
        "peak_xte": lat.max(1),
        "mean_xte": lat.mean(1),
        "peak_dpsi": yaw.max(1),
        "ood_peak": ratio.max(1),
        "ood_mean": ratio.mean(1),
        "out_env": ((lat > e1a.ENV_LAT_MAX) | (yaw > e1a.ENV_YAW_MAX))
                   .any(1).astype(float),
        "ade2s": pw["ade2s"].numpy(),
        "pred2s": pw["pred2s"], "gt2s": pw["gt2s"],
        "junc": junc, "long": long_,
    }


def align(a_map, b_map, mask_name=None):
    """(idx_ft, idx_base, eid) on the common windows, optionally one stratum
    (strata are taken from BASE's labels so the two arms see the same split)."""
    bk = {k: i for i, k in enumerate(b_map["key"])}
    common = [(i, bk[k]) for i, k in enumerate(a_map["key"]) if k in bk]
    ia = np.array([i for i, _ in common], dtype=int)
    ib = np.array([j for _, j in common], dtype=int)
    if mask_name:
        m = b_map[mask_name][ib]
        ia, ib = ia[m], ib[m]
    eid = [b_map["eid"][j] for j in ib]
    return ia, ib, eid


def pair_field(a_map, b_map, field, mask_name=None, reduce="mean"):
    ia, ib, eid = align(a_map, b_map, mask_name)
    if len(ia) < 2:
        return {"n": int(len(ia)), "note": "too few common windows"}
    return _paired(a_map[field][ia], b_map[field][ib], eid, reduce=reduce)


def arm_field(m, field, mask_name=None, reduce="mean"):
    idx = np.flatnonzero(m[mask_name]) if mask_name else np.arange(len(m["eid"]))
    if len(idx) < 2:
        return {"n": int(len(idx)), "note": "too few windows"}
    return _boot(m[field][idx], [m["eid"][i] for i in idx], reduce=reduce)


# --------------------------------------------------------------------------- #
# M1 — lateral / longitudinal split of every reported ADE                       #
# --------------------------------------------------------------------------- #
def lateral_block(pred_ft, gt_ft, pred_b, gt_b, eid, tag):
    """taniteval/lateral.py decomposition of the 4 predicted knots (0.5-2.0 s),
    both frames, per arm + PAIRED delta. Cross-track is the safety axis."""
    out = {"_tag": tag, "knots_s": list(KNOTS_S), "n_windows": int(len(eid)),
           "_gt_identity_max_abs_diff": round(
               float((gt_ft - gt_b).abs().max()), 6)}
    try:
        out["_axis_convention"] = LAT.assert_axis_convention(gt_b, dt=KNOT_DT)
    except Exception as e:                                    # noqa: BLE001
        out["_axis_convention"] = {"verified": False, "error": str(e)}
    j = pred_b.shape[1] - 1                                   # the 2.0 s knot
    for mode in ("ego", "frenet"):
        al_f, cr_f = LAT.decompose(pred_ft, gt_ft, mode)
        al_b, cr_b = LAT.decompose(pred_b, gt_b, mode)
        de_f = torch.linalg.norm(pred_ft - gt_ft, dim=-1)
        de_b = torch.linalg.norm(pred_b - gt_b, dim=-1)
        blk = {}
        for nm, al, cr, de in (("base", al_b, cr_b, de_b),
                               ("ft", al_f, cr_f, de_f)):
            blk[nm] = {
                "ade_over_knots": _boot(de.mean(1).numpy(), eid),
                "cross_abs@2s": _boot(cr[:, j].abs().numpy(), eid),
                "along_abs@2s": _boot(al[:, j].abs().numpy(), eid),
                "cross_p90@2s": _boot(cr[:, j].abs().numpy(), eid, reduce="p90"),
                "cross_tail@2s": LAT.tail_stats(cr[:, j].abs().numpy()),
                "energy_share": LAT.energy_share(al.numpy(), cr.numpy()),
                "mean_abs_cross_by_knot_m": [round(float(x), 4)
                                             for x in cr.abs().mean(0)],
                "mean_abs_along_by_knot_m": [round(float(x), 4)
                                             for x in al.abs().mean(0)],
            }
        blk["paired_delta_ft_minus_base"] = {
            "ade_over_knots": _paired(de_f.mean(1).numpy(),
                                      de_b.mean(1).numpy(), eid),
            "cross_abs@2s": _paired(cr_f[:, j].abs().numpy(),
                                    cr_b[:, j].abs().numpy(), eid),
            "cross_p90@2s": _paired(cr_f[:, j].abs().numpy(),
                                    cr_b[:, j].abs().numpy(), eid,
                                    reduce="p90"),
            "along_abs@2s": _paired(al_f[:, j].abs().numpy(),
                                    al_b[:, j].abs().numpy(), eid),
            "_read": "negative delta = FT better on that axis",
        }
        out[mode] = blk
    return out


def run_arm(ckpt, preset, episodes, device, Ks, primary, junction_deg, stride,
            batch, ood, label):
    t = time.time()
    model, step, cfg = e1a.load_refc(ckpt, preset, device)
    print(f"[e1b-eval] {label}: step {step} | anchors "
          f"{tuple(model.decoder.anchors.shape)}", flush=True)
    out = {"ckpt": ckpt, "step": step, "perK": {}}
    out["openloop"] = openloop_full(model, episodes, device, stride, batch)
    print(f"[e1b-eval] {label}: open-loop n={len(out['openloop']['eid'])} "
          f"ADE@2s={out['openloop']['ade2s'].mean():.4f} "
          f"anchor_acc={out['openloop']['anchor_acc'].mean():.4f} "
          f"({time.time() - t:.0f}s)", flush=True)
    for K in Ks:
        t2 = time.time()
        out["perK"][K] = per_window(model, episodes, device, K, primary,
                                    junction_deg, stride, batch, ood)
        m = out["perK"][K]
        print(f"[e1b-eval] {label}: K={K} n={len(m['eid'])} "
              f"dep={m['dep'].mean():.4f} junc_dep={m['dep'][m['junc']].mean():.4f} "
              f"peakXTE={m['peak_xte'].mean():.3f} "
              f"OODpeak={m['ood_peak'].mean():.3f} ({time.time() - t2:.0f}s)",
              flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt",
                    default="/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt")
    ap.add_argument("--ft-ckpt",
                    default="/workspace/e1b/refc-base-e1b-clsft/ckpt.pt")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--val-dir",
                    default="/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6")
    ap.add_argument("--p1-json", default="/workspace/e1a_e2a/lowood_flagship_ci.json")
    ap.add_argument("--horizons", default="20,185")
    ap.add_argument("--corridor-halfwidth", type=float, default=1.75)
    ap.add_argument("--junction-deg", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=999,
                    help="cap the held-out episode count (SMOKE ONLY — the "
                         "verdict run must use all 44, E1a's exact eval set)")
    ap.add_argument("--out", default="/workspace/e1b/e1b_eval_result.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ks = [int(x) for x in args.horizons.split(",")]
    prim = args.corridor_halfwidth
    ood = e1a.OODMap(args.p1_json)
    ep_files = sorted(Path(args.val_dir).glob("ep_*.pt"))[:args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in ep_files]
    print(f"[e1b-eval] {len(episodes)} held-out eps | K {Ks} | dev {device}",
          flush=True)

    t0 = time.time()
    with strict_numerics():
        base = run_arm(args.base_ckpt, args.preset, episodes, device, Ks, prim,
                       args.junction_deg, args.stride, args.batch, ood, "BASE")
        ft = run_arm(args.ft_ckpt, args.preset, episodes, device, Ks, prim,
                     args.junction_deg, args.stride, args.batch, ood, "FT")
    Kmax, Kmin = max(Ks), min(Ks)

    res = {
        "_experiment": "E1b paired closed-loop verdict (FT vs base)",
        "_estimator": "paired_episode_cluster_bootstrap / episode_cluster_bootstrap "
                      "(taniteval/ci.py, md5 ef925f06febd20a99f5901491fcf75cb), "
                      "B=2000, resampling the held-out EPISODES. "
                      "overlapping_holdout_se is used NOWHERE.",
        "_prereg": "PRE_REGISTRATION.md §3 — primary = junction corridor-departure "
                   "@K=185, paired Delta(FT-base); SUCCESS iff CI-separated LOWER "
                   "AND no guardrail regression.",
        "val_dir": args.val_dir, "n_episodes": len(episodes),
        "corridor_halfwidth_m": prim, "junction_deg": args.junction_deg,
        "horizons_K": Ks, "stride": args.stride,
        "base_ckpt": args.base_ckpt, "base_step": base["step"],
        "ft_ckpt": args.ft_ckpt, "ft_step": ft["step"],
    }

    # ---------------- PRIMARY + closed-loop strata ---------------------------
    KM = str(Kmax)
    res["PRIMARY_junction_corridor_departure_K" + KM] = {
        "base": arm_field(base["perK"][Kmax], "dep", "junc"),
        "ft": arm_field(ft["perK"][Kmax], "dep", "junc"),
        "paired_delta_ft_minus_base":
            pair_field(ft["perK"][Kmax], base["perK"][Kmax], "dep", "junc"),
    }
    strata = {"overall": None, "junction": "junc", "longitudinal": "long"}
    res["closed_loop_K" + KM] = {}
    for nm, mk in strata.items():
        res["closed_loop_K" + KM][nm] = {
            f: {"base": arm_field(base["perK"][Kmax], f, mk),
                "ft": arm_field(ft["perK"][Kmax], f, mk),
                "paired_delta_ft_minus_base":
                    pair_field(ft["perK"][Kmax], base["perK"][Kmax], f, mk)}
            for f in ("dep", "win_dep", "peak_xte", "mean_xte", "peak_dpsi",
                      "ood_peak", "ood_mean", "out_env", "ade2s")}
    res["closed_loop_K" + str(Kmin)] = {
        nm: {f: {"base": arm_field(base["perK"][Kmin], f, mk),
                 "ft": arm_field(ft["perK"][Kmin], f, mk),
                 "paired_delta_ft_minus_base":
                     pair_field(ft["perK"][Kmin], base["perK"][Kmin], f, mk)}
             for f in ("dep", "win_dep", "peak_xte", "ade2s")}
        for nm, mk in strata.items()}

    # ---------------- GUARDRAIL (a) + (b): open loop, PAIRED -----------------
    ob, of_ = base["openloop"], ft["openloop"]
    bk = {k: i for i, k in enumerate(ob["key"])}
    com = [(i, bk[k]) for i, k in enumerate(of_["key"]) if k in bk]
    ia = np.array([i for i, _ in com]); ib = np.array([j for _, j in com])
    oeid = [ob["eid"][j] for j in ib]
    res["GUARDRAIL_a_openloop_ade2s"] = {
        "base": _boot(ob["ade2s"][ib], oeid), "ft": _boot(of_["ade2s"][ia], oeid),
        "paired_delta_ft_minus_base": _paired(of_["ade2s"][ia], ob["ade2s"][ib], oeid),
        "_registry_reference": "REF-C base canonical val ADE@2s 0.4728 "
                               "[0.3835, 0.5699] (MODEL_REGISTRY §4.3)",
        "_pass_rule": "PASS iff the paired CI includes 0 or is separated LOWER.",
    }
    res["GUARDRAIL_b_openloop_anchor_block"] = {
        "_what": "refc_train.compute_losses anchor block (lines 268-276) with the "
                 "GT target, on the held-out windows. PRE_REGISTRATION §3 names "
                 "this the REF-C stand-in for a world-model canary.",
        **{f: {"base": _boot(ob[f][ib], oeid), "ft": _boot(of_[f][ia], oeid),
               "paired_delta_ft_minus_base": _paired(of_[f][ia], ob[f][ib], oeid)}
           for f in ("anchor_acc", "anchor_ce", "anchor_traj_l1")},
    }

    # ---------------- M1: lateral / longitudinal split of every ADE ----------
    res["M1_lateral_split_openloop"] = lateral_block(
        of_["pred"][ia], of_["gt"][ia], ob["pred"][ib], ob["gt"][ib], oeid,
        "open-loop 4 knots (0.5-2.0 s), held-out 44")
    ia2, ib2, ceid = align(ft["perK"][Kmax], base["perK"][Kmax])
    res["M1_lateral_split_closedloop_K" + KM] = lateral_block(
        ft["perK"][Kmax]["pred2s"][ia2], ft["perK"][Kmax]["gt2s"][ia2],
        base["perK"][Kmax]["pred2s"][ib2], base["perK"][Kmax]["gt2s"][ib2],
        ceid, f"closed-loop ADE@2s knots inside the K={Kmax} rollout")

    # ---------------- verdict ------------------------------------------------
    p = res["PRIMARY_junction_corridor_departure_K" + KM]["paired_delta_ft_minus_base"]
    ga = res["GUARDRAIL_a_openloop_ade2s"]["paired_delta_ft_minus_base"]
    gb = res["GUARDRAIL_b_openloop_anchor_block"]
    ood_ft = res["closed_loop_K" + KM]["overall"]["ood_peak"]["ft"]["mean"]
    ood_b = res["closed_loop_K" + KM]["overall"]["ood_peak"]["base"]["mean"]

    guard = {
        "a_openloop_ade2s_ok": bool(not (ga.get("separated") and ga["lo"] > 0)),
        "b_anchor_acc_ok": bool(not (gb["anchor_acc"]["paired_delta_ft_minus_base"]
                                     .get("separated")
                                     and gb["anchor_acc"]["paired_delta_ft_minus_base"]["hi"] < 0)),
        "b_anchor_traj_l1_ok": bool(not (gb["anchor_traj_l1"]["paired_delta_ft_minus_base"]
                                         .get("separated")
                                         and gb["anchor_traj_l1"]["paired_delta_ft_minus_base"]["lo"] > 0)),
        # ⛔ VOID BY CONSTRUCTION — kept as a RECORD of what was adjudicated, and
        # renamed so it can never again be read as a passing guardrail.
        # MEASURED 2026-07-26: sup(OODMap.ratio_arr) = 1.298888 (np.interp CLAMPS at
        # 3.0 m / 12°), so `ood_ft <= 1.30` is a TAUTOLOGY BY 0.001112 — it cannot
        # fail. Repo-wide it passed 139/139 and failed 0. ⚠️ IN THIS VERY FILE'S RUN
        # IT WAS THE ONLY GUARDRAIL THAT "HELD" WHILE THE OTHER THREE FAILED, so the
        # E1b guardrail story rests on it and must be re-read.
        # E1a's rule was a DISJUNCTION (ratio OR out-of-envelope fraction) and only
        # the dead half was evaluated. Real instrument: taniteval.ood.verdict().
        # See RETRACTION_LOG.md, class C13.
        "c_ood_in_band_VOID_DO_NOT_USE": bool(ood_ft <= 1.30 + 1e-9),
        "c_ood_void_reason": ("tautology by 0.001112: sup(ratio_arr)=1.298888 < 1.30; "
                              "use taniteval.ood.verdict() for E1a's real disjunction"),
        "c_ood_ft": ood_ft, "c_ood_base": ood_b,
        "c_ood_ratio_ft_over_base": round(float(ood_ft / max(ood_b, 1e-9)), 4),
    }
    # ⛔ `c_ood_in_band` is NO LONGER a conjunct: a criterion that cannot fail
    # contributes nothing to a conjunction except the appearance of a fourth check.
    # The three real guardrails now stand alone, and the OOD question is answered
    # separately by taniteval.ood.verdict() (E1a's actual disjunction).
    guard["all_ok"] = bool(guard["a_openloop_ade2s_ok"] and guard["b_anchor_acc_ok"]
                           and guard["b_anchor_traj_l1_ok"])
    guard["all_ok_note"] = ("3 guardrails, NOT 4 — Gc (OOD ratio <= 1.30) was VOID BY "
                            "CONSTRUCTION and has been removed from the conjunction. "
                            "Any historical all_ok that depended on it is not comparable "
                            "to this one. See RETRACTION_LOG.md, class C13.")
    res["GUARDRAIL_SUMMARY"] = guard

    if not isinstance(p, dict) or "separated" not in p:
        verdict = "INDETERMINATE"
    elif p["separated"] and p["hi"] < 0:
        verdict = ("SUCCESS (primary): junction corridor-departure@K%d CI-separated "
                   "LOWER for FT (paired). Guardrails %s."
                   % (Kmax, "HOLD" if guard["all_ok"] else "REGRESSED -> BOUND"))
        if not guard["all_ok"]:
            verdict = ("BOUND: primary separated lower BUT a guardrail regressed "
                       "(%s) — per PRE_REGISTRATION §3 this is not a SUCCESS."
                       % json.dumps(guard))
    elif p["separated"] and p["lo"] > 0:
        verdict = ("BOUND/REGRESS: FT junction departure@K%d CI-separated HIGHER "
                   "(worse) than base." % Kmax)
    else:
        verdict = ("BOUND: FT junction departure@K%d NOT CI-separated from base "
                   "(paired delta %s [%s, %s])."
                   % (Kmax, p["delta"], p["lo"], p["hi"]))
    res["verdict"] = verdict
    res["wall_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(f"[e1b-eval] PRIMARY paired delta: {p}", flush=True)
    print(f"[e1b-eval] GUARDRAILS: {json.dumps(guard)}", flush=True)
    print(f"[e1b-eval] verdict: {verdict}  ({res['wall_s']}s) -> {args.out}",
          flush=True)
    print("E1B_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
