"""ds_t0.py -- the DSCALE10K MECHANISM read at T0 (card section 4, "Mechanism read (stage A, T0)"; section 6 precondition and mechanism label).
It is a thin wrapper over the programme's own DSCALE instrument (ds_code/code/v8f_dscale_read.py, exported unedited from the code directory of commit 213250ce; the helpers it imports,
v8f_hist_read / v8f_grid_read / v8f_pgt_read / v8f_wc_read, are exported beside it).  Nothing of the instrument is re-implemented: load_all, check_gt_codes, present_error and mlat_row are
CALLED, and the sign rule of DSCALE Amendment 1 is the instrument's own `decide` logic.  Tier: T0 (absmask / GT-oracle evals) -- a world-model and tactical diagnostic, never driving performance.
Evidence class: MEASURED (ours; zero training).  Estimator: paired episode-cluster bootstrap B 2,000 seed 0 ("another draw of EPISODES"); another TRAINING run = the floor (B_rep vs B_s0).
Eval files per arm: eval_<NAME>_s0{,-tacabs,-tacorc,-tacorc-all-latcons}.{npz,json}, NAME = DS<arm without '_'><stage>, e.g. DSBs0A (B_s0 stage A), DSUs0A, DSBrepA, DSRs0A.
Pure helpers (decide_t, decide_reach) are numpy-only and unit-tested; mech_pair needs torch + the read tree on PYTHONPATH."""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("DS_CODE", HERE + "/ds_code/code")
BAR = 0.05            # card section 4: r <= -5 % (T-DATA-LIMITED); section 6: R worse than U by r >= +5 % (reach)


def name_of(arm, stage="A"):
    return "DS" + arm.replace("_", "") + stage


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Mods:
    HR = GR = PGT = None


def modules():
    """-> (HR, GR, PGT, WC, pairfn); private instances, their refusals routed to one exception"""
    if _Mods.HR is None:
        sys.path.insert(0, CODE)
        hr = _load("v8f_hist_read_for_ds", CODE + "/v8f_hist_read.py")

        def refuse(msg):
            raise RuntimeError(f"T0 INSTRUMENT REFUSED: {msg}")
        hr.refuse = hr.GR.refuse = hr.PGT.refuse = refuse
        _Mods.HR, _Mods.GR, _Mods.PGT = hr, hr.GR, hr.PGT
    _, WC = _Mods.PGT._modules()
    from taniteval.ci import paired_episode_cluster_bootstrap as pairfn
    return _Mods.HR, _Mods.GR, _Mods.PGT, WC, pairfn


def decide_t(r, d, floor, bar=BAR):
    """T-DATA-LIMITED (card section 4): r <= -bar, the paired MSE delta separated with a NEGATIVE point (DSCALE Amendment 1's sign rule: a separated delta whose sign contradicts r is not separated
    in r's direction), and abs(delta) > the stage-A T floor.  Returns the label and the three reasons."""
    sep = bool(d["separated"])
    sign_ok = d["point"] < 0
    above = floor is not None and abs(d["point"]) > floor
    ok = bool(r <= -bar and sep and sign_ok and above)
    return {"label": "T-DATA-LIMITED" if ok else "NOT T-DATA-LIMITED", "r_le_minus_bar": bool(r <= -bar), "separated": sep, "sign_ok": bool(sign_ok), "above_floor": bool(above), "floor": floor}


def decide_reach(r, d, bar=BAR):
    """R_s0 against U_s0 at stage A (card section 6 precondition): REACHED when R is worse than U on T's latcons by r >= +bar, separated (sign rule as above: the MSE delta R - U point > 0)."""
    ok = bool(r >= bar and d["separated"] and d["point"] > 0)
    return {"label": "REACHED" if ok else "NO-REACH", "r_ge_bar": bool(r >= bar), "separated": bool(d["separated"]), "sign_ok": bool(d["point"] > 0)}


def window_mse(Z, ref, WC, GR, name, seed=0):
    return GR.latcons_window_mse(Z[(name, seed)], ref["g_cons"], ref["has"], WC)[0]


# ---------------------------------------------------------------------------------------------------------------- GTFIX: one SD frame for every arm
# The eval writes T's code and the GT code in SD units of the CHECKPOINT's own cons_mu / cons_sd (v8f_mini.py _code_gt: (cons - cons_mu) / cons_sd), and the trainer
# sets those buffers from its TRAIN join (v8f_mini_train.py: mu, sd = M.cons_stats(J)).  Arms trained on one corpus share them; the union arms (U, R) do not, so the same
# eval labels arrive in a different frame and WC.present_error refuses (MEASURED gtfix/probe_gt.json: labels equal to 2.5e-5 in physical units, frames differ).
# The fix re-expresses an arm whose block differs into the REFERENCE arm's frame, and only after its labels are shown equal there.  Arms with bit-equal blocks are untouched.
GT_ABS_TOL, GT_REL_TOL = 1e-5, 1e-6     # SD units; MEASURED max |U re-expressed - B| = 8.2e-7 (float32 rounding of the banked codes)


def _refuse(msg):
    raise RuntimeError(f"T0 INSTRUMENT REFUSED: {msg}")


def ckpt_of(eval_dir, name, seed):
    """the one checkpoint that wrote this arm's oracle, absmask and all-latcons evals (their json 'ckpt')"""
    paths = set()
    for suf in ("-tacorc", "-tacabs", "-tacorc-all-latcons"):
        pj = os.path.join(eval_dir, f"eval_{name}_s{seed}{suf}.json")
        if not os.path.exists(pj):
            _refuse(f"{os.path.basename(pj)} is missing: the arm's checkpoint (and its constraint frame) cannot be named")
        paths.add(os.path.normcase(os.path.abspath(json.load(open(pj, encoding="utf-8"))["ckpt"])))
    if len(paths) != 1:
        _refuse(f"{name}_s{seed}: its evals name {len(paths)} different checkpoints {sorted(paths)}")
    return paths.pop()


_STATS = {}


def ckpt_stats(eval_dir, name, seed):
    """(cons_mu, cons_sd) float64 from the checkpoint's state dict: the frame its evals wrote the constraint codes in"""
    p = ckpt_of(eval_dir, name, seed)
    if p not in _STATS:
        if not os.path.exists(p):
            _refuse(f"{p} (the checkpoint of {name}_s{seed}) does not exist: its constraint frame is unknown")
        import torch
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        _STATS[p] = (sd["cons_mu"].double().numpy().copy(), sd["cons_sd"].double().numpy().copy())
    return _STATS[p]


def to_common_frame(Z, seed, ref_name, stats_of, native_check=None):
    """Re-express, in place, every arm whose GT constraint block differs from the reference arm's into the REFERENCE arm's SD frame:
    x_ref = (x_own * sd_own + mu_own - mu_ref) / sd_ref, on T's code (`t_cons`) and on the GT block (`g_cons`).  It acts only when
      * the two blocks differ and the arms' frames (cons_mu, cons_sd) differ: same frame and different blocks = different labels, left for WC.present_error to refuse;
      * `has` is identical (else left for WC.present_error to refuse);
      * the arm's own GT presence (code != 0) equals the reference's on every labelled window, and the re-expressed GT equals the reference's GT within
        GT_ABS_TOL + GT_REL_TOL * |GT| on every present cell; otherwise REFUSED here, with the count of windows and slots that differ (the labels differ: a PI question).
    Then the arm's GT block IS the reference's (equal within the tolerance), T's code is in the reference frame, cells the GT marks absent stay 0 (absmask's rule),
    and `lat_code` is checked in the arm's OWN frame first (PGT.check_gt_codes on that arm alone, via `native_check`) and set to None.
    Returns {arm: record} with the frame used and the largest deviation."""
    ref = Z[(ref_name, seed)]
    has, G = ref["has"], ref["g_cons"]
    rec = {}
    for (arm, s), v in Z.items():
        if s != seed or arm == ref_name or v is ref:
            continue
        if not np.array_equal(v["has"], has) or np.array_equal(v["g_cons"][has], G[has]):
            rec[arm] = {"frame": "native", "converted": False}
            continue
        mr, sr = stats_of(ref_name, seed)
        m, sd = stats_of(arm, s)
        if np.array_equal(m, mr) and np.array_equal(sd, sr):
            rec[arm] = {"frame": "native (same cons_mu / cons_sd as the reference: the labels differ)", "converted": False}
            continue
        if native_check is not None:
            native_check(arm, s, v)
        g = v["g_cons"]
        pa, pr = g != 0.0, G != 0.0
        if not np.array_equal(pa[has], pr[has]):
            bad = (pa != pr) & has[:, None]
            _refuse(f"{arm}_s{s}: GT presence differs from {ref_name}_s{seed}'s on {int(bad.any(1).sum())} windows / {int(bad.sum())} slots "
                    f"(slots {sorted(set(np.where(bad)[1].tolist()))}): not the same labels")
        g2 = np.where(pa, (g * sd + m - mr) / sr, 0.0)
        dev = np.abs(g2 - G)
        bad = (dev > GT_ABS_TOL + GT_REL_TOL * np.abs(G)) & has[:, None]
        if bad.any():
            _refuse(f"{arm}_s{s}: re-expressed in {ref_name}_s{seed}'s frame its GT block still differs on {int(bad.any(1).sum())} windows / {int(bad.sum())} slots "
                    f"(slots {sorted(set(np.where(bad)[1].tolist()))}, max {float(dev[has].max()):.3e} SD): not the same labels")
        t2 = (v["t_cons"] * sd + m - mr) / sr
        v["t_native"], v["stats_native"] = v["t_cons"], (m, sd)
        v["t_cons"] = np.where(has[:, None] & ~pa, 0.0, t2)
        v["g_cons"] = G.copy()
        v["lat_code"] = None
        rec[arm] = {"frame": f"{ref_name}_s{seed}", "converted": True, "max_gt_dev_sd": float(dev[has].max()), "n_cells_compared": int(has.sum() * G.shape[1]),
                    "max_abs_dmu": float(np.abs(m - mr).max()), "max_sd_ratio_dev": float(np.abs(sd / sr - 1.0).max())}
    return rec


def latcons_rmse(t, g, has, cols):
    """slot-pooled present-slot latcons RMSE: WC.present_error's rmse_group['latcons'] formula (present = has and GT code != 0)"""
    c = list(cols)
    pl = (has[:, None] & (g != 0.0))[:, c]
    return float(np.sqrt(((t[:, c] - g[:, c]) ** 2)[pl].mean()))


def frame_beside(A, Bm, ref, ref_stats, cols):
    """BESIDE the deciding r (the reference arm's frame): the point r = RMSE(A)/RMSE(B) - 1 in physical units and in A's own frame (no bootstrap)"""
    mr, sr = ref_stats
    has, G = ref["has"], ref["g_cons"]
    pres = G != 0.0
    phys = lambda v: v.get("t_native", v["t_cons"]) * v.get("stats_native", ref_stats)[1] + v.get("stats_native", ref_stats)[0]   # noqa: E731
    tA, tB, gP = phys(A), phys(Bm), np.where(pres, G * sr + mr, 0.0)
    mA, sA = A.get("stats_native", ref_stats)
    gA = np.where(pres, (gP - mA) / sA, 0.0)
    return {"physical_units": latcons_rmse(tA, gP, has, cols) / latcons_rmse(tB, gP, has, cols) - 1.0,
            "A_frame": latcons_rmse((tA - mA) / sA, gA, has, cols) / latcons_rmse((tB - mA) / sA, gA, has, cols) - 1.0}


def load_z(eval_dir, names, seed, gt_name, stats_of=None):
    """PGT.load_all for the named arms, plus the key ('M2b', seed) = the GT reference arm's entry.  The programme's WC.present_error (unedited) takes its GT block from the arm KEYED 'M2b'; every block-1 arm is
    scored on the same eval-141 windows with the same labels, and present_error itself refuses an arm whose GT block differs from that reference, so the alias is the check, not a shortcut.
    GTFIX: before the alias, an arm whose block differs ONLY by its checkpoint's constraint frame is re-expressed in the reference arm's frame (to_common_frame)."""
    HR, GR, PGT, WC, pairfn = modules()
    Z, eid = PGT.load_all(eval_dir, (seed,), tuple(dict.fromkeys(tuple(names) + (gt_name,))), WC)

    def native_check(arm, s, v):
        PGT.check_gt_codes({("M2b", s): v, (arm, s): v}, (s,), WC)
    frames = to_common_frame(Z, seed, gt_name, stats_of or (lambda n, s: ckpt_stats(eval_dir, n, s)), native_check)
    if ("M2b", seed) not in Z:
        Z[("M2b", seed)] = Z[(gt_name, seed)]
    return Z, eid, frames


def mech_pair(eval_dir, a_name, b_name, seed=0, n_boot=2000, ref_name=None):
    """arm A against arm B on T's present-slot latcons: r = RMSE(A)/RMSE(B) - 1 (slot-pooled, the deciding definition), the window-weighted r beside, the paired per-window MSE delta,
    oracle ADE6 and the per-group present-slot RMSE beside.  The GT block comes from `ref_name` (default B)."""
    HR, GR, PGT, WC, pairfn = modules()
    Z, eid, frames = load_z(eval_dir, (a_name, b_name), seed, ref_name or b_name)
    PGT.check_gt_codes(Z, (seed,), WC)
    pe, _ = WC.present_error(Z, (seed,))
    ref = Z[(ref_name or b_name, seed)]
    A, Bm = Z[(a_name, seed)], Z[(b_name, seed)]
    ml = GR.mlat_row(pairfn, A, Bm, pe[(a_name, seed)], pe[(b_name, seed)], ref, WC, eid, n_boot)
    ra, rb = ml["rmse_slot"]["G"], ml["rmse_slot"]["B"]
    wa, wb = ml["rmse_window"]["G"], ml["rmse_window"]["B"]
    out = {"a": a_name, "b": b_name, "r": ra / rb - 1.0, "r_percent": 100.0 * (ra / rb - 1.0), "rmse_slot": {"A": ra, "B": rb}, "r_window": wa / wb - 1.0, "rmse_window": {"A": wa, "B": wb},
           "mse_delta": ml["mse_delta"], "n_windows_present": ml["n_windows_present"], "n_slots_present": ml["n_slots_present"], "n_episodes": int(len(np.unique(eid))),
           "tier": "T0 (absmask / GT-oracle evals): a world-model and tactical diagnostic, not driving performance"}
    if abs(out["r"] + ml["fall"]) > 1e-12:
        raise RuntimeError(f"T0 INSTRUMENT: r {out['r']} is not minus M-LAT's fall {ml['fall']}")
    out["beside"] = {"oracle_ade6": {"A": float(A["orc_ade6"].mean()), "B": float(Bm["orc_ade6"].mean()), "delta": PGT.paired(pairfn, A["orc_ade6"], Bm["orc_ade6"], eid, n_boot)},
                     "ade6": {"A": float(A["ade6"].mean()), "B": float(Bm["ade6"].mean())},
                     "rmse_group": {"A": pe[(a_name, seed)]["rmse_group"], "B": pe[(b_name, seed)]["rmse_group"]}}
    cols = WC.GROUPS["latcons"]                          # GTFIX control: the helper reproduces the deciding RMSEs in the deciding frame
    if abs(latcons_rmse(A["t_cons"], ref["g_cons"], ref["has"], cols) - ra) > 1e-12 * ra or abs(latcons_rmse(Bm["t_cons"], ref["g_cons"], ref["has"], cols) - rb) > 1e-12 * rb:
        raise RuntimeError("T0 INSTRUMENT: latcons_rmse does not reproduce present_error's latcons RMSE")
    out["frame"] = {"deciding": f"{ref_name or b_name}_s{seed} (the reference arm's cons_mu / cons_sd)", "arms": frames}
    if any(v.get("converted") for v in frames.values()):
        out["frame"]["r_beside"] = frame_beside(A, Bm, ref, ckpt_stats(eval_dir, ref_name or b_name, seed), cols)
    return out


def mech_floor(eval_dir, rep_name, base_name, seed=0):
    """the stage-A T floor: |mean per-window latcons MSE(rep) - mean(base)| over the windows present in both, and |r| between the pair (one replicate pair: a weak floor)."""
    HR, GR, PGT, WC, pairfn = modules()
    Z, eid, _ = load_z(eval_dir, (rep_name, base_name), seed, base_name)
    ref = Z[(base_name, seed)]
    ma, mb = window_mse(Z, ref, WC, GR, rep_name, seed), window_mse(Z, ref, WC, GR, base_name, seed)
    ok = np.isfinite(ma) & np.isfinite(mb)
    pe, _ = WC.present_error(Z, (seed,))
    ra, rb = pe[(rep_name, seed)]["rmse_group"]["latcons"], pe[(base_name, seed)]["rmse_group"]["latcons"]
    return {"floor_mse": float(abs(ma[ok].mean() - mb[ok].mean())), "abs_r": float(abs(ra / rb - 1.0)), "n_windows": int(ok.sum())}


def identity_control(eval_dir, name, seed=0, n_boot=200):
    """an arm against itself reads r exactly 0 and a paired delta exactly +0 [0, 0] (the DSCALE instrument's own control)"""
    m = mech_pair(eval_dir, name, name, seed, n_boot)
    d = m["mse_delta"]
    ok = bool(m["r"] == 0.0 and m["r_window"] == 0.0 and d["point"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0)
    return {"pass": ok, "r": m["r"], "delta": d["point"]}
