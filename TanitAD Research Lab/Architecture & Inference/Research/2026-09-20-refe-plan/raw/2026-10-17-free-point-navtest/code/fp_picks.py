#!/usr/bin/env python3
"""STAGE D1 of the free-point navtest read (PLAN.md): merge the pod's chunk outputs, build the four points, pick, and write one W3 seam per point.

    python fp_picks.py build --out-dir W --chunks W/out [--limit N]      -> W/points.npz (picks, logits digests), W/seam_<POINT>.npz, W/picks_report.json
    python fp_picks.py selftest                                           -> the synthetic bars (M25 algebra, tie rule, D1 teeth)

Points (PREREG_FREEPOINT.md; `dose_rls1b.py`): BASE = u400's own scorer; C0A, R0B = the banked states; M25 = c + 0.25 (r - c) in float64 on ALL 7 outputs.
Pick = `attrib_rls1b.picks` (the trainer's sel_score in float64; ties -> lowest index).  Executed pose = proposal k through A7's last-pose heading repair (`planner.repair_last_heading`)
and the seam's `to_navsim` -- the SAME functions `refe_navtest_seam.py` applies to the planner's pick.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import importlib.util
import json
import os
import sys

import numpy as np

PK = "C:/fp/PK"
DATA = "E:/Projects/TanitAD/data/refe_navtest"
EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
BANKED_U400_SEAM = f"{DATA}/seams/stage2_sft4_B_u400_navtest_full.npz"
BANKED_U400_SEAM_MD5 = "b009e941620b6cbbbe63e1dd8a94a506"
POINTS = ("BASE", "C0A", "R0B", "M25")
ALPHA = 0.25
IC3_POSE_TOL_M = 5e-3
IC3_MIN_FRAC = 0.97              # ADDENDUM_1 amendment note 2: the gross-error detector; cross-card near-tie flips are EXPECTED (0.2-3 % on simulated logit noise)
IC3_NEARTIE_MARGIN = 0.01        # a disagreement is explained when the pod's pick beats the banked pick's proposal by <= this in sel_score (largest simulated flip: 0.0020)
IC3_MAX_UNEXPLAINED_FRAC = 0.001


def load_attrib():
    sys.path.insert(0, f"{PK}/refe")
    sys.path.insert(0, f"{PK}/eval")
    p = f"{PK}/raw/2026-10-14-rls1b-attribution/attrib_rls1b.py"
    spec = importlib.util.spec_from_file_location("attrib_rls1b", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["attrib_rls1b"] = m
    spec.loader.exec_module(m)
    return m


def merge_chunks(chunk_dir: str, order: list | None = None, limit: int = 0, present_only: bool = False):
    fs = sorted(glob.glob(f"{chunk_dir}/chunk_*.npz"))
    if not fs:
        raise SystemExit(f"no chunk_*.npz in {chunk_dir}")
    tok, traj, lg = [], [], {}
    ic2, meta = [], []
    for f in fs:
        z = np.load(f, allow_pickle=False)
        tok += [str(t) for t in z["token"]]
        traj.append(z["traj"])
        ic2.append(z["ic2_maxabs"])
        meta.append(json.loads(str(z["meta_json"])))
        for k in z.files:
            if k.startswith("logits_"):
                lg.setdefault(k[7:], []).append(z[k])
    if len(set(tok)) != len(tok):
        raise SystemExit("duplicate tokens across chunks")
    traj = np.concatenate(traj)
    lg = {k: np.concatenate(v) for k, v in lg.items()}
    ic2 = np.concatenate(ic2)
    idx = {t: i for i, t in enumerate(tok)}
    if order is not None:
        if present_only:                                      # DRY RUNS ONLY (never for the read): the tokens a smoke produced
            order = [t for t in order if t in idx]
        if limit:
            order = [t for t in order if t in idx][:limit]
        miss = [t for t in order if t not in idx]
        if miss:
            raise SystemExit(f"{len(miss)} tokens of the navtest order are missing from the chunks (first {miss[:3]})")
        sel = [idx[t] for t in order]
        traj, ic2 = traj[sel], ic2[sel]
        lg = {k: v[sel] for k, v in lg.items()}
        tok = list(order)
    return tok, traj, lg, ic2, meta


def points_logits(lg: dict) -> dict:
    c, r, b = (np.asarray(lg[k], np.float64) for k in ("C0A", "R0B", "BASE"))
    return {"BASE": b, "C0A": c, "R0B": r, "M25": c + ALPHA * (r - c)}


def pick_all(AT, pl: dict) -> dict:
    return {p: np.asarray(AT.picks(pl[p]), np.int64) for p in POINTS}


def executed_poses(traj: np.ndarray, pick: np.ndarray):
    """[N, 8, 3] float32: proposal `pick` through the seam's own conversion (A7 repair, then to_navsim)"""
    import torch
    sys.path.insert(0, f"{PK}/refe")
    sys.path.insert(0, f"{PK}/eval")
    import planner as PLM
    import refe_navtest_seam as SEAM
    out = np.zeros((len(pick), 8, 3), np.float32)
    for i, k in enumerate(pick):
        p = PLM.repair_last_heading(torch.from_numpy(traj[i, int(k)]))
        out[i] = SEAM.to_navsim(p.float().cpu().numpy()).astype(np.float32)
    return out


def all_poses(traj_tok: np.ndarray) -> np.ndarray:
    """[M, 8, 3]: every proposal of ONE token through the seam's conversion (A7 repair, to_navsim)"""
    return executed_poses(traj_tok[None].repeat(traj_tok.shape[0], 0), np.arange(traj_tok.shape[0]))


def ic3(poses: np.ndarray, tokens: list, traj: np.ndarray = None, base_logits: np.ndarray = None, AT=None, banked_path: str = BANKED_U400_SEAM) -> dict:
    """ADDENDUM_1 IC3 (as re-stated in amendment note 2): agreement >= 97 % AND every disagreement a near-tie (<= 0.01 in sel_score on the pod's own BASE logits), unexplained <= 0.1 %"""
    zb = np.load(banked_path, allow_pickle=False)
    bi = {str(t): i for i, t in enumerate(zb["token"])}
    d = np.array([np.abs(poses[i] - zb["poses"][bi[t]]).max() for i, t in enumerate(tokens)], np.float64)
    ok = d <= IC3_POSE_TOL_M
    res = {"n": len(tokens), "n_equal": int(ok.sum()), "frac_equal": float(ok.mean()), "max_dpose_m": float(d.max()), "n_exact_zero": int((d == 0.0).sum()),
           "bar_tol_m": IC3_POSE_TOL_M, "bar_frac": IC3_MIN_FRAC, "neartie_margin_bar": IC3_NEARTIE_MARGIN,
           "differing_tokens": [tokens[i] for i in np.nonzero(~ok)[0][:400]]}
    unexplained, margins = [], []
    if traj is not None and AT is not None:
        sel = AT.sel_v1_lane(AT.RV.sigmoid(base_logits))                    # [N, M] float64, the trainer's rule on the pod's BASE logits
        for i in np.nonzero(~ok)[0]:
            ap = all_poses(traj[i])
            dd = np.abs(ap - zb["poses"][bi[tokens[i]]][None]).max(axis=(1, 2))
            j = int(dd.argmin())
            if dd[j] > IC3_POSE_TOL_M:
                unexplained.append({"token": tokens[i], "why": "no pod proposal within tolerance of the banked pose", "min_dpose_m": float(dd[j])})
                continue
            pick = int(sel[i].argmax())
            m = float(sel[i, pick] - sel[i, j])
            margins.append(m)
            if m > IC3_NEARTIE_MARGIN:
                unexplained.append({"token": tokens[i], "why": "not a near-tie", "margin": m})
    res["n_disagree"] = int((~ok).sum())
    res["neartie_margin_quantiles_50_90_99_max"] = ([float(x) for x in np.quantile(margins, [0.5, 0.9, 0.99])] + [float(max(margins))]) if margins else None
    res["n_unexplained"] = len(unexplained)
    res["unexplained"] = unexplained[:50]
    res["bar_unexplained_max"] = int(np.ceil(IC3_MAX_UNEXPLAINED_FRAC * len(tokens)))
    res["pass"] = bool(ok.mean() >= IC3_MIN_FRAC and len(unexplained) <= res["bar_unexplained_max"])
    return res


def write_seam(path: str, tokens: list, poses: np.ndarray, exp: dict, arm: str):
    np.savez(path, token=np.array(tokens), fingerprint=np.array([exp[t]["fingerprint"] for t in tokens]), poses=poses.astype(np.float32),
             sampling=np.array([8, 0.5]), arm=np.array(arm))


def cmd_build(a):
    AT = load_attrib()
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    zb = np.load(BANKED_U400_SEAM, allow_pickle=False)
    order = [str(t) for t in zb["token"]]                                   # the banked u400 seam's token order == the navtest filter
    tok, traj, lg, ic2, meta = merge_chunks(a.chunks, order, a.limit, a.present_only)
    if ic2.max() != 0.0:
        print(f"INVALID: IC2 max {ic2.max()} (the replicated scorer path differs from the model's own)")
        return 2
    pl = points_logits(lg)
    pk = pick_all(AT, pl)
    seams = {}
    for p in POINTS:
        poses = executed_poses(traj, pk[p])
        seams[p] = poses
        write_seam(f"{a.out_dir}/seam_{p}.npz", tok, poses, exp, f"FREEPOINT_{p}")
    rep = {"n_tokens": len(tok), "IC2_max_abs": float(ic2.max()), "IC3_BASE_vs_banked_u400": ic3(seams["BASE"], tok, traj, np.asarray(lg["BASE"], np.float64), AT),
           "pick_changes_vs_BASE": {p: int((pk[p] != pk["BASE"]).sum()) for p in POINTS if p != "BASE"},
           "pick_changes_C0A_to_R0B": int((pk["R0B"] != pk["C0A"]).sum()), "pick_changes_M25_vs_C0A": int((pk["M25"] != pk["C0A"]).sum()),
           "pick_changes_M25_vs_R0B": int((pk["M25"] != pk["R0B"]).sum()),
           "poses_identical_to_BASE": {p: int(sum(np.array_equal(seams[p][i], seams["BASE"][i]) for i in range(len(tok)))) for p in POINTS if p != "BASE"},
           "chunks": [m.get("chunk") for m in meta], "state_info": meta[0]["states"], "device": meta[0].get("device"), "gpu": meta[0].get("gpu"),
           "torch": meta[0].get("torch"), "ckpt_md5": meta[0].get("ckpt_md5")}
    np.savez(f"{a.out_dir}/points.npz", token=np.array(tok), **{f"pick_{p}": pk[p] for p in POINTS})
    json.dump(rep, open(f"{a.out_dir}/picks_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("state_info",)}, indent=1)[:3000])
    print("ZZPICKS_OK")
    return 0


def cmd_selftest(a):
    AT = load_attrib()
    rng = np.random.default_rng(3)
    c = rng.normal(size=(5, 64, 7))
    r = c + rng.normal(size=c.shape)
    pl = points_logits({"BASE": c * 0 + 1, "C0A": c, "R0B": r})
    ok = np.array_equal(pl["M25"], c + 0.25 * (r - c)) and np.allclose(pl["M25"], 0.75 * c + 0.25 * r)
    # alpha endpoints: the dose script's s(alpha) at 0 and 1 are C0A and R0B
    ends = np.array_equal(c + 0.0 * (r - c), c) and np.allclose(c + 1.0 * (r - c), r)
    # tie rule: argmax -> lowest index
    t = np.full((1, 64, 7), 3.0)
    ties = int(AT.picks(t)[0]) == 0
    # sel_score is the trainer's: NC*DAC*(5EP+5TTC+2C)/12*lane (components 0..4, 6)
    z = np.full((1, 64, 7), -9.0)
    z[0, 17, :] = 9.0
    z[0, 40, 5] = 50.0                                            # the DDC output (index 5) does not enter the rule
    sel_ok = int(AT.picks(z)[0]) == 17
    print({"M25_algebra": ok, "alpha_endpoints": ends, "tie_lowest_index": ties, "ddc_not_in_rule": sel_ok})
    # D1: a shuffled token->logits map must collapse the pick agreement far below the IC3 bar
    base = rng.normal(size=(400, 64, 7))
    pk = np.asarray(AT.picks(base))
    sh = np.asarray(AT.picks(base[rng.permutation(400)]))
    d1 = float((pk == sh).mean())
    print({"D1_shuffled_agreement": d1, "D1_teeth": d1 < 0.5})
    good = ok and ends and ties and sel_ok and d1 < 0.5
    print("ZZPICKS_SELFTEST_" + ("PASS" if good else "FAIL"))
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "selftest"))
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--present-only", action="store_true", help="dry run on a smoke chunk: only the tokens present")
    a = ap.parse_args()
    return {"build": cmd_build, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
