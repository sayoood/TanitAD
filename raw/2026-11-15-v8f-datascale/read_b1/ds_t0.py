"""ds_t0.py -- the DSCALE10K MECHANISM read at T0 (card section 4, "Mechanism read (stage A, T0)"; section 6 precondition and mechanism label).
It is a thin wrapper over the programme's own DSCALE instrument (ds_code/code/v8f_dscale_read.py, exported unedited from the code directory of commit 213250ce; the helpers it imports,
v8f_hist_read / v8f_grid_read / v8f_pgt_read / v8f_wc_read, are exported beside it).  Nothing of the instrument is re-implemented: load_all, check_gt_codes, present_error and mlat_row are
CALLED, and the sign rule of DSCALE Amendment 1 is the instrument's own `decide` logic.  Tier: T0 (absmask / GT-oracle evals) -- a world-model and tactical diagnostic, never driving performance.
Evidence class: MEASURED (ours; zero training).  Estimator: paired episode-cluster bootstrap B 2,000 seed 0 ("another draw of EPISODES"); another TRAINING run = the floor (B_rep vs B_s0).
Eval files per arm: eval_<NAME>_s0{,-tacabs,-tacorc,-tacorc-all-latcons}.{npz,json}, NAME = DS<arm without '_'><stage>, e.g. DSBs0A (B_s0 stage A), DSUs0A, DSBrepA, DSRs0A.
Pure helpers (decide_t, decide_reach) are numpy-only and unit-tested; mech_pair needs torch + the read tree on PYTHONPATH."""
import importlib.util
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


def load_z(eval_dir, names, seed, gt_name):
    """PGT.load_all for the named arms, plus the key ('M2b', seed) = the GT reference arm's entry.  The programme's WC.present_error (unedited) takes its GT block from the arm KEYED 'M2b'; every block-1 arm is
    scored on the same eval-141 windows with the same labels, and present_error itself refuses an arm whose GT block differs from that reference, so the alias is the check, not a shortcut."""
    HR, GR, PGT, WC, pairfn = modules()
    Z, eid = PGT.load_all(eval_dir, (seed,), tuple(dict.fromkeys(tuple(names) + (gt_name,))), WC)
    if ("M2b", seed) not in Z:
        Z[("M2b", seed)] = Z[(gt_name, seed)]
    return Z, eid


def mech_pair(eval_dir, a_name, b_name, seed=0, n_boot=2000, ref_name=None):
    """arm A against arm B on T's present-slot latcons: r = RMSE(A)/RMSE(B) - 1 (slot-pooled, the deciding definition), the window-weighted r beside, the paired per-window MSE delta,
    oracle ADE6 and the per-group present-slot RMSE beside.  The GT block comes from `ref_name` (default B)."""
    HR, GR, PGT, WC, pairfn = modules()
    Z, eid = load_z(eval_dir, (a_name, b_name), seed, ref_name or b_name)
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
    return out


def mech_floor(eval_dir, rep_name, base_name, seed=0):
    """the stage-A T floor: |mean per-window latcons MSE(rep) - mean(base)| over the windows present in both, and |r| between the pair (one replicate pair: a weak floor)."""
    HR, GR, PGT, WC, pairfn = modules()
    Z, eid = load_z(eval_dir, (rep_name, base_name), seed, base_name)
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
