"""Re-score the DEPLOYED `idm_head_v1` on the COMMA HALF of its OWN 9,420-window
val split, with the comma2k19 heading repair ON — COMMA_YAW_REISSUE escalation #2.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The card's headline `val_heldout_traindomain` yaw numbers (`R² 0.010433`,
`MAE 0.11814`, n = 9,420) are **POOLED over rig-A + rig-B + comma2k19**, which is
exactly why they were stale without being labelled "comma" (RETRACTION_LOG C5 +
C29). This script re-scores **the comma component of that same split**, on the
same clips, the same window construction and the same persisted head — the only
part of the pooled number the repair can move.

⛔ It does NOT re-issue the pooled number. The rig-A/rig-B half needs
`physicalai-train-e438721ae894`, which is not on this box, and **PhysicalAI is
provably unaffected anyway** (`n_pai_changed = 0`, R² bit-identical) — so
recomputing it would buy re-pooling, not knowledge. Report per corpus.

⛔ NOTHING IS RETRAINED. The head is loaded from disk and run forward twice
against two LABEL protocols on identical predictions.

SUBSTRATE (from `idm_head_v1_train.py:56-58`, `VAL_HELDOUT_TAGS`)
    comma part = `cm_[40:70]`, and `cm_{i:05d}` ≡ `ep_{i:05d}.pt` of
    `comma2k19-val-61c46fca8f7f` (`run_branchb_transfer.py:230,251`)
    -> ep_00040 … ep_00069, 30 clips, T = 300, k = 4, stride = 2
    -> `valid_centers` = arange(4, 280, 2) = 138 windows/clip = **4,140**

LABEL PROTOCOL — stated because two numbers differing only by protocol are
otherwise indistinguishable on the page:
    OFF  `heading_repair` off. The cached poses ARE the legacy label
         (`comma2k19-val-61c46fca8f7f` predates 2026-07-27). No `v_min`.
    ON   `heading_repair` on, `v_min` = 0.5 m/s
         (`comma2k19.HEADING_OBSERVABLE_V_MPS`, the MEASURED threshold).
         `v_min` is the repair's OBSERVABILITY threshold, not a scoring gate —
         **no window is dropped**, both arms score the identical 4,140.

ESTIMATOR: `taniteval.ci` episode-cluster bootstrap, unit = the 30 comma val
episodes, B = 2000; the OFF→ON contrast uses the **paired** version on the same
windows. ⛔ `overlapping_holdout_se` is not used.

Run (dev box, project venv; ~10 min on an RTX 4060, resumable):
    cd stack && python "<this file>"
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[6]
for p in (REPO / "stack", REPO / "stack" / "scripts", REPO / "taniteval"):
    sys.path.insert(0, str(p))

import idm_head as ih                                            # noqa: E402
import run_idm_proof as R                                        # noqa: E402
from taniteval import ci as tci                                  # noqa: E402
from tanitad.data.comma2k19 import (HEADING_OBSERVABLE_V_MPS,     # noqa: E402
                                    hold_heading_through_standstill)

# --------------------------------------------------------------------------- #
# pinned artifacts — FAIL LOUD, do not silently score a different head         #
# --------------------------------------------------------------------------- #
HEAD_PT = (REPO / "TanitAD Research Hub" / "Architecture & Inference" /
           "Implementation" / "incoming" / "2026-07-25-idm-youtube-validation" /
           "idm_head_v1.pt")
HEAD_MD5 = "fa4462f0b898b036be729c790278b823"          # card `/weights_md5`
ENC_CKPT = Path(r"C:\Users\Admin\tanitad-data\eval\v1_speedjerk_ckpt.pt")
ENC_MD5 = "b5f07d9e3dd2ca643949bc86832e6585"           # card `/config/encoder/ckpt_md5`
COMMA_CACHE = Path(r"C:\Users\Admin\tanitad-data\eval\comma2k19-val-61c46fca8f7f")
LAT_DIR = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
               r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
               r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\lat_cm40_69")
OUT = HERE.parent / "raw" / "idm_head_v1_comma_rescore.json"

CM_LO, CM_HI = 40, 70                                  # VAL_HELDOUT_TAGS cm slice
K, STRIDE = 4, 2
N_BOOT = 2000
SCAL = ih.SCALAR_NAMES                                 # speed, yaw_rate, steer, long_accel


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def md5_of(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# metrics — the EXACT `idm2_lib.chan_metrics` / `spearman` definitions, copied  #
# rather than imported because `idm2_lib` inserts a POD path at import time     #
# (`idm2_lib.py:19` -> `/root/taniteval`), which silently shadows the estimator. #
# --------------------------------------------------------------------------- #
def spearman(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def chan_metrics(pred, gt) -> dict:
    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)
    err = p - g
    mad = float(np.median(np.abs(g - np.median(g))))
    return {"r2": 1.0 - float((err ** 2).sum())
            / max(float(((g - g.mean()) ** 2).sum()), 1e-12),
            "rho": spearman(p, g),
            "mae": float(np.abs(err).mean()),
            "medae": float(np.median(np.abs(err))),
            "nmedae": float(np.median(np.abs(err)) / max(mad, 1e-12)),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "gt_std": float(g.std()), "gt_mad": mad, "n": int(g.size)}


def boot_r2(pred, gt, eid) -> dict:
    """Episode-cluster bootstrap on R², via ci.py's callable-reducer path: the
    per-window value is the window INDEX and the reducer recomputes R² on the
    resampled index set, so the resampling unit is the EPISODE."""
    p = np.asarray(pred, np.float64)
    g = np.asarray(gt, np.float64)

    def _r2(idx):
        i = idx.astype(np.int64)
        gg, pp = g[i], p[i]
        return float(1.0 - ((pp - gg) ** 2).sum()
                     / max(((gg - gg.mean()) ** 2).sum(), 1e-12))
    _r2.__name__ = "r2"
    return tci.episode_cluster_bootstrap(np.arange(p.size, dtype=np.float64),
                                         eid, reduce=_r2, n_boot=N_BOOT, seed=0)


def paired_delta_r2(pred, gt_a, gt_b, eid) -> dict:
    """R²(vs gt_a) − R²(vs gt_b) on the SAME windows and the SAME predictions.

    ⚠️ This is a contrast between two LABEL protocols, not between two arms. It
    is reported because a bare point move (+0.011 → +0.331) carries no interval,
    and the marginal intervals on the two arms are NOT a test of the difference.
    """
    p = np.asarray(pred, np.float64)
    ga = np.asarray(gt_a, np.float64)
    gb = np.asarray(gt_b, np.float64)

    def _mk(g):
        def _r2(idx):
            i = idx.astype(np.int64)
            gg, pp = g[i], p[i]
            return float(1.0 - ((pp - gg) ** 2).sum()
                         / max(((gg - gg.mean()) ** 2).sum(), 1e-12))
        _r2.__name__ = "r2"
        return _r2

    idx = np.arange(p.size, dtype=np.float64)
    uniq, idx_by_ep = tci.episode_index(eid)
    ra, rb = _mk(ga), _mk(gb)
    point = ra(idx) - rb(idx)
    d = np.array([ra(idx[sel]) - rb(idx[sel])
                  for sel in tci._draws(uniq, idx_by_ep, N_BOOT, 0)])
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    return {"delta": round(float(point), 4), "lo": round(lo, 4),
            "hi": round(hi, 4), "separated": bool(lo > 0 or hi < 0),
            "p_delta_gt0": round(float((d > 0).mean()), 4),
            "n_windows": int(p.size), "n_episodes": int(len(uniq)),
            "n_boot": N_BOOT,
            "estimator": "paired_episode_cluster_bootstrap (label-protocol "
                         "contrast: identical predictions, two label sets)"}


# --------------------------------------------------------------------------- #
def encode_all(device) -> list[str]:
    LAT_DIR.mkdir(parents=True, exist_ok=True)
    tags = [f"cm_{i:05d}" for i in range(CM_LO, CM_HI)]
    todo = [t for t in tags if not (LAT_DIR / f"{t}.pt").exists()]
    if todo:
        enc, ro, meta = R.load_encoder(str(ENC_CKPT), device)
        log(f"encoder: {meta}")
        for j, tag in enumerate(tags):
            lf = LAT_DIR / f"{tag}.pt"
            if lf.exists():
                continue
            ep = COMMA_CACHE / f"ep_{int(tag.split('_')[-1]):05d}.pt"
            d = R._load_ep(str(ep))
            z = R.encode_frames(enc, ro, d["frames_u8"], device, batch=8)
            torch.save({"z": z, "poses": d["poses"].float(),
                        "actions": d["actions"].float()}, lf)
            del d
            log(f"  encoded {j + 1}/{len(tags)} {tag} z{tuple(z.shape)}")
        del enc, ro
        torch.cuda.empty_cache()
    return tags


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__}")

    log("verifying pinned artifacts (fail loud)...")
    for p, want, what in ((HEAD_PT, HEAD_MD5, "idm_head_v1.pt"),
                          (ENC_CKPT, ENC_MD5, "flagship-v1 encoder ckpt")):
        got = md5_of(p)
        assert got == want, (f"{what} md5 {got} != pinned {want} — this is NOT "
                             f"the artifact the card was measured with")
        log(f"  OK {what} md5 {got}")

    tags = encode_all(device)

    d = torch.load(HEAD_PT, map_location="cpu", weights_only=False)
    head = ih.IDMHead(**d["config"]["head_kwargs"]).to(device).eval()
    head.load_state_dict(d["state_dict"])
    log(f"head: {ih.count_params(head)} params (card says {d['params']})")
    assert ih.count_params(head) == d["params"]

    # ---- windows, predictions, and BOTH label protocols ------------------- #
    preds, off, on, eids, spd = [], [], [], [], []
    n_changed_frames = n_changed_win = 0
    max_abs_label_delta = 0.0
    for tag in tags:
        L = torch.load(LAT_DIR / f"{tag}.pt", weights_only=False)
        z, poses, actions = L["z"].float(), L["poses"].float(), L["actions"].float()
        t = ih.valid_centers(z.shape[0], K, ih.DEFAULT_HORIZONS, STRIDE)
        offs = torch.arange(-K, K + 1)
        Zwin = z[t[:, None] + offs[None, :]]
        s_off = ih.scalar_targets_at(poses, actions, t)            # repair OFF

        yaw_fixed, obs = hold_heading_through_standstill(
            poses[:, 2].numpy(), poses[:, 3].numpy(),
            v_min=HEADING_OBSERVABLE_V_MPS)
        p_on = poses.clone()
        p_on[:, 2] = torch.from_numpy(yaw_fixed).float()
        s_on = ih.scalar_targets_at(p_on, actions, t)              # repair ON
        n_changed_frames += int((~obs).sum())
        n_changed_win += int((s_off[:, 1] != s_on[:, 1]).sum())
        max_abs_label_delta = max(max_abs_label_delta,
                                  float((s_off[:, 1] - s_on[:, 1]).abs().max()))
        # the other three channels MUST be untouched — the repair is yaw-only
        assert torch.equal(s_off[:, [0, 2, 3]], s_on[:, [0, 2, 3]])

        out = []
        for i in range(0, Zwin.shape[0], 512):
            out.append(head(Zwin[i:i + 512].to(device))["scalars"].cpu())
        preds.append(torch.cat(out))
        off.append(s_off)
        on.append(s_on)
        spd.append(poses[t, 3])
        eids.extend([tag] * int(t.numel()))

    P = torch.cat(preds).double().numpy()
    GO = torch.cat(off).double().numpy()
    GN = torch.cat(on).double().numpy()
    V = torch.cat(spd).double().numpy()
    eid = np.array(eids)
    n = P.shape[0]
    log(f"windows {n} over {len(tags)} clips ({n / len(tags):.0f}/clip)")
    assert n == 4140, f"expected 4140 comma windows of the 9,420 split, got {n}"

    res = {
        "what": "idm_head_v1 (DEPLOYED, nothing retrained) re-scored on the "
                "COMMA HALF of its own val_heldout_traindomain split",
        "date": "2026-07-27",
        "agent": "heading-default",
        "escalation": "COMMA_YAW_REISSUE.md §8 #2 / IDM_V3 #2",
        "evidence_class": "MEASURED (ours; dev box RTX 4060, this script)",
        "tier": "decision-grade for the comma channel on THIS substrate",
        "host": "dev box only — pod1/pod2/pod3 not touched",
        "substrate": {
            "split": "idm_head_v1_train.py VAL_HELDOUT_TAGS, comma part",
            "tags": f"cm_[{CM_LO}:{CM_HI}]",
            "cache": str(COMMA_CACHE),
            "n_clips": len(tags), "n_windows": n,
            "k": K, "stride": STRIDE, "window_frames": 2 * K + 1,
            "pooled_split_n_windows": 9420,
            "comma_share_of_pooled": round(n / 9420, 4),
            "NOT_INCLUDED": "rig-A + rig-B (5,280 windows). "
                            "physicalai-train-e438721ae894 is not on this box, "
                            "and PhysicalAI is provably unaffected "
                            "(n_pai_changed = 0). ⛔ Do not re-issue the POOLED "
                            "number from this file.",
        },
        "label_protocol": {
            "OFF": "heading_repair off (the cached poses ARE the legacy label); "
                   "no v_min — this is the card's own protocol",
            "ON": f"heading_repair on, v_min = {HEADING_OBSERVABLE_V_MPS} m/s "
                  f"(comma2k19.HEADING_OBSERVABLE_V_MPS, MEASURED)",
            "windows_dropped": 0,
            "note": "v_min is the repair's OBSERVABILITY threshold, not a "
                    "scoring gate; both arms score the identical windows",
        },
        "artifacts": {"head_md5": HEAD_MD5, "encoder_ckpt_md5": ENC_MD5,
                      "head_params": int(d["params"])},
        "repair_extent": {
            "n_frames_below_v_min": n_changed_frames,
            "n_windows_label_changed": n_changed_win,
            "frac_windows_changed": round(n_changed_win / n, 6),
            "max_abs_yaw_rate_label_delta_rad_s": max_abs_label_delta,
            "n_impossible_legacy_gt1p5": int((np.abs(GO[:, 1]) > 1.5).sum()),
            "n_impossible_repaired_gt1p5": int((np.abs(GN[:, 1]) > 1.5).sum()),
            "gt_speed_at_changed_windows_max": float(
                V[GO[:, 1] != GN[:, 1]].max()) if n_changed_win else None,
        },
        "channels": {}, "contrast": {}, "controls": {},
    }

    for c, name in enumerate(SCAL):
        res["channels"][name] = {
            "OFF": chan_metrics(P[:, c], GO[:, c]),
            "ON": chan_metrics(P[:, c], GN[:, c]),
        }
        res["channels"][name]["OFF"]["r2_ci"] = boot_r2(P[:, c], GO[:, c], eid)
        res["channels"][name]["ON"]["r2_ci"] = boot_r2(P[:, c], GN[:, c], eid)

    # the yaw contrast, properly paired on identical predictions
    res["contrast"]["yaw_rate_r2_ON_minus_OFF"] = paired_delta_r2(
        P[:, 1], GN[:, 1], GO[:, 1], eid)
    res["contrast"]["yaw_rate_absErr_ON_minus_OFF"] = tci.paired_episode_cluster_bootstrap(
        np.abs(P[:, 1] - GN[:, 1]), np.abs(P[:, 1] - GO[:, 1]), eid,
        n_boot=N_BOOT, seed=0)
    res["contrast"]["yaw_rate_medAbsErr_ON_minus_OFF"] = tci.paired_episode_cluster_bootstrap(
        np.abs(P[:, 1] - GN[:, 1]), np.abs(P[:, 1] - GO[:, 1]), eid,
        n_boot=N_BOOT, seed=0, reduce="median")

    # ---- controls -------------------------------------------------------- #
    res["controls"]["other_channels_bit_identical"] = {
        nm: bool(np.array_equal(GO[:, c], GN[:, c]))
        for c, nm in enumerate(SCAL) if nm != "yaw_rate"}
    res["controls"]["unchanged_windows"] = {
        "n": int((GO[:, 1] == GN[:, 1]).sum()),
        "max_abs_label_delta": 0.0,
        "identical": bool(np.array_equal(GO[GO[:, 1] == GN[:, 1], 1],
                                         GN[GO[:, 1] == GN[:, 1], 1]))}
    ch = GO[:, 1] != GN[:, 1]
    if ch.any():
        res["controls"]["changed_windows"] = {
            "n": int(ch.sum()),
            "legacy_label_absmax": float(np.abs(GO[ch, 1]).max()),
            "legacy_label_absmean": float(np.abs(GO[ch, 1]).mean()),
            "repaired_label_absmax": float(np.abs(GN[ch, 1]).max()),
            "repaired_label_absmean": float(np.abs(GN[ch, 1]).mean()),
            "head_pred_absmean": float(np.abs(P[ch, 1]).mean()),
            "gt_speed_mean": float(V[ch].mean()),
            "reading": "a road vehicle at v~0 has yaw_rate ~0; a small "
                       "prediction against a huge legacy label is C29 — the "
                       "MODEL was right and the LABEL was wrong"}
    # ---- ⭐ the residual defect: what the standstill repair does NOT fix --- #
    # Reported because it is the whole reason the headline R² does not move,
    # and a re-issue that omitted it would read as contradicting C29.
    imp_on = np.abs(GN[:, 1]) > 1.5
    res["residual_defect"] = {
        "why": "the standstill repair only touches frames below v_min. Any "
               "impossible yaw label at or ABOVE v_min survives it — and R² "
               "has an unbounded left tail, so a handful still dominate.",
        "n_impossible_after_repair": int(imp_on.sum()),
        "frac_of_windows": round(float(imp_on.mean()), 6),
        "their_gt_speed_min": float(V[imp_on].min()) if imp_on.any() else None,
        "their_gt_speed_median": float(np.median(V[imp_on])) if imp_on.any() else None,
        "their_abs_label_max": float(np.abs(GN[imp_on, 1]).max()) if imp_on.any() else None,
        "gt_std_rad_s": {"OFF": float(GO[:, 1].std()), "ON": float(GN[:, 1].std())},
        "gt_std_of_a_clean_yaw_channel_for_scale": {
            "physicalai_A0_v3": 0.14241782117652835,
            "source": "compare_v3.json LABEL_FIX_deployed_head/yaw_rate/legacy/pai/gt_std"},
    }
    # ---- ⭐ the MECHANISM of the residual, tested rather than narrated ------ #
    # Hypothesis: the survivors are OBSERVABILITY-BOUNDARY windows. The repair
    # holds one direction through a standstill, so a centred difference whose
    # t-1 and t+1 are anchored to DIFFERENT observable runs sees the whole
    # heading change of the stop compressed into one 0.2 s step.
    n_bd = n_res = 0
    bd_dtheta = []
    for tag in tags:
        L3 = torch.load(LAT_DIR / f"{tag}.pt", weights_only=False)
        po, ac = L3["poses"].float(), L3["actions"].float()
        t3 = ih.valid_centers(po.shape[0], K, ih.DEFAULT_HORIZONS, STRIDE)
        yf, ob = hold_heading_through_standstill(po[:, 2].numpy(),
                                                 po[:, 3].numpy(),
                                                 v_min=HEADING_OBSERVABLE_V_MPS)
        # the anchor frame each repaired sample inherited its heading from
        anc = np.where(ob, np.arange(ob.size), -1)
        np.maximum.accumulate(anc, out=anc)
        anc[anc < 0] = int(np.argmax(ob)) if ob.any() else 0
        p3 = po.clone()
        p3[:, 2] = torch.from_numpy(yf).float()
        yr = ih.scalar_targets_at(p3, ac, t3)[:, 1].numpy()
        bad = np.abs(yr) > 1.5
        ti = t3.numpy()[bad]
        n_res += int(bad.sum())
        cross = anc[ti + 1] != anc[ti - 1]
        n_bd += int(cross.sum())
        bd_dtheta.extend(np.abs(np.arctan2(
            np.sin(yf[ti + 1] - yf[ti - 1]), np.cos(yf[ti + 1] - yf[ti - 1]))).tolist())
    res["residual_defect"]["mechanism"] = {
        "hypothesis": "the survivors are OBSERVABILITY-BOUNDARY windows: t-1 and "
                      "t+1 inherit their heading from DIFFERENT observable runs, "
                      "so the entire heading change across the stop lands in one "
                      "0.2 s centred difference",
        "n_residual_impossible": n_res,
        "n_of_those_crossing_an_anchor_boundary": n_bd,
        "frac": round(n_bd / max(n_res, 1), 4),
        "median_abs_heading_jump_rad": float(np.median(bd_dtheta)) if bd_dtheta else None,
        "verdict": ("CONFIRMED — the residual is an artifact of the repair's own "
                    "boundary, not leftover standstill noise, so raising v_min "
                    "cannot remove it (see the sweep: 84-85 survivors at every "
                    "threshold tried)") if n_bd == n_res else
                   ("PARTIAL — some survivors are not boundary windows; the "
                    "mechanism does not fully explain the residual"),
    }

    # ⚠️ DIAGNOSTIC ONLY — a different v_min is a DIFFERENT PROTOCOL and this row
    # ⛔ is NOT a re-issued number. It exists to tell the owner of escalation #2
    # whether the residual is reachable by raising the threshold. PUBLISHED
    # precedent: comma.ai's own calib_challenge discards every frame below 4 m/s.
    diag = {}
    for vm in (1.0, 2.0, 4.0):
        gs = []
        for tag in tags:
            L2 = torch.load(LAT_DIR / f"{tag}.pt", weights_only=False)
            po, ac = L2["poses"].float(), L2["actions"].float()
            t2 = ih.valid_centers(po.shape[0], K, ih.DEFAULT_HORIZONS, STRIDE)
            yf, _ = hold_heading_through_standstill(po[:, 2].numpy(),
                                                    po[:, 3].numpy(), v_min=vm)
            p2 = po.clone()
            p2[:, 2] = torch.from_numpy(yf).float()
            gs.append(ih.scalar_targets_at(p2, ac, t2)[:, 1])
        g = torch.cat(gs).double().numpy()
        m = chan_metrics(P[:, 1], g)
        diag[f"v_min_{vm}"] = {
            "r2": m["r2"], "mae": m["mae"], "medae": m["medae"],
            "nmedae": m["nmedae"], "rho": m["rho"], "gt_std": m["gt_std"],
            "n_impossible_gt1p5": int((np.abs(g) > 1.5).sum())}
    diag["WARNING"] = ("⛔ DIAGNOSTIC, NOT A RE-ISSUE. Each row is a DIFFERENT "
                       "label protocol. The shipped repair is v_min "
                       f"{HEADING_OBSERVABLE_V_MPS}, which is the MEASURED "
                       "threshold; these rows say what a different threshold "
                       "would do, not what any published number should become.")
    res["diagnostic_v_min_sweep"] = diag

    # ---- the leak probe, prepared so it is a one-minute job for whoever has
    # the eval pod. episode_id is a stable sha1 of route/segment, so it is
    # CACHE-INDEPENDENT and is the right key to compare across builds.
    train_ids, val_ids = [], []
    for i in range(0, CM_HI):
        ep = torch.load(COMMA_CACHE / f"ep_{i:05d}.pt", weights_only=False,
                        map_location="cpu")
        eidv = int(ep.get("episode_id", -1)) if isinstance(ep, dict) else -1
        (train_ids if i < CM_LO else val_ids).append(eidv)
        del ep
    res["leak_probe"] = {
        "question": "are any of the v3 anchor's 22 comma val segments also "
                    "among idm_head_v1's 40 comma TRAINING segments?",
        "why_it_matters": "the anchor (+0.0114 -> +0.3308) is the number a "
                          "re-issue would be tempted to paste in. It is "
                          "measured on comma2k19-val-76b6e94a97a1 (64 segs / 21 "
                          "routes), a DIFFERENT cache from the 61c46fca8f7f (90 "
                          "eps) that A0 trained on, so the cm_ tag indices are "
                          "NOT comparable and the overlap is UNKNOWN.",
        "status": "UNRESOLVED — needs /root/idm2/manifest.json (eval pod), "
                  "which records episode_id per cm_ tag",
        "a0_comma_TRAIN_episode_ids": train_ids,
        "a0_comma_HELDOUT_episode_ids_used_here": val_ids,
        "disjoint_here": bool(not (set(train_ids) & set(val_ids))),
        "how": "intersect a0_comma_TRAIN_episode_ids with the episode_id of the "
               "22 v3 val tags (cm_00000, cm_00003, ..., cm_00063). Non-empty "
               "=> the anchor is partly IN-TRAIN.",
    }

    res["controls"]["encode_fidelity"] = {
        "why": "the latents are re-encoded here, not the pod's lat_flagshipv1, "
               "so a faithful pipeline must reproduce the head's KNOWN comma "
               "behaviour on channels the repair cannot touch",
        "comma_speed_r2_here": res["channels"]["speed"]["OFF"]["r2"],
        "comma_speed_r2_idm_v3_A0": 0.7589944415742614,
        "idm_v3_substrate": "v3 val split, 2,992 comma windows, v_min 0.5 — a "
                            "DIFFERENT substrate, so this is a plausibility "
                            "band, not an equality check",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"WROTE {OUT}")
    y = res["channels"]["yaw_rate"]
    print(f"\ncomma yaw_rate  OFF r2 {y['OFF']['r2']:+.6f}  mae {y['OFF']['mae']:.6f}"
          f"  medae {y['OFF']['medae']:.7f}  nmedae {y['OFF']['nmedae']:.5f}"
          f"  rho {y['OFF']['rho']:+.6f}")
    print(f"comma yaw_rate  ON  r2 {y['ON']['r2']:+.6f}  mae {y['ON']['mae']:.6f}"
          f"  medae {y['ON']['medae']:.7f}  nmedae {y['ON']['nmedae']:.5f}"
          f"  rho {y['ON']['rho']:+.6f}")


if __name__ == "__main__":
    main()
