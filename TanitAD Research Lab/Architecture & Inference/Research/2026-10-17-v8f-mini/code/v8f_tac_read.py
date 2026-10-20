"""PREREG_V8F_TAC's reading, committed before its data: does v8F-mini's OPERATIVE layer use its TACTICAL layer, and what would a
perfect tactical layer buy?

  python v8f_tac_read.py --eval-dir <runs/v8f_mini_full/eval> --out tac_read.json

Inputs, per checkpoint c in {M1_s0, M1_s1, M2b_s0, M2b_s1}, all from ONE session and ONE code version
(`stack/scripts/v8f_mini_eval.py run --tac ...`): eval_<c>-tacpred.npz (O reads T's own code), eval_<c>-tacshuf.npz (O reads a DONOR
window's predicted code, another episode, donor seed 1), eval_<c>-tacorc.npz (O reads the GT v9 code where the window has a row), and
the banked eval_<c>.npz (the reproducibility check only).

TAC-USED (decides): shuf - pred, eval-141 ADE@6s, paired over episodes, per checkpoint.
    USED iff > 0 and separated at ALL FOUR checkpoints; DECORATIVE iff separated at NONE (O's output does not move); otherwise
    PARTIAL (named per checkpoint; a separated NEGATIVE delta -- a donor code that helps -- is PARTIAL, never USED).
TAC-HEADROOM (T0 diagnostic: the oracle reads GT; it ranks the lever and is never a result): orc - pred per checkpoint.
    LARGE for an arm iff < 0, separated, and |delta| > 2 F_arm at BOTH seeds (F_arm = |mean pred_s0 - mean pred_s1|, the tacpred runs);
    SMALL otherwise.  The lever is chosen on M2b (the K-PASS arm); M1 is reported beside it.
Beside them, deciding nothing: the paired LONGITUDINAL / LATERAL component deltas of both interventions
    (`ra1_hjepa_eval.paired_family_block`), and the intervention's own manipulation statistics.
Refusals (rc 2): a missing file; files not on one grid (eid); a shuffle that changed the code on < 0.95 of windows (the intervention
    did not happen); an oracle run that carries no fallback record; a tacpred run that does not reproduce
    the banked eval per window within 1e-3 m (same weights, same windows: a larger gap means the code path changed what it measures).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (another draw of EPISODES, models fixed); training-seed variance
    enters only through F (two seeds per arm); the model is deterministic, so there is no inference-seed variance to name."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
REPO = HERE.parents[5]
RAW = PKG / "raw"
ARMS, SEEDS = ("M1", "M2b"), (0, 1)
MODES = {"pred": "tacpred", "shuf": "tacshuf", "orc": "tacorc"}
TOL_REPRO = 1e-3
MIN_CHANGED = 0.95
B = 2000


def refuse(msg):
    print(f"TAC READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _paths():
    for p in (REPO / "stack", REPO / "taniteval"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load(eval_dir):
    runs, eid = {}, None
    for arm in ARMS:
        for s in SEEDS:
            c = f"{arm}_s{s}"
            for k, suf in [("banked", None)] + list(MODES.items()):
                p = os.path.join(eval_dir, f"eval_{c}.npz" if suf is None else f"eval_{c}-{suf}.npz")
                if not os.path.exists(p):
                    refuse(f"{os.path.basename(p)} is missing from {eval_dir}")
                z = np.load(p)
                if eid is None:
                    eid = np.asarray(z["eid"])
                elif not np.array_equal(np.asarray(z["eid"]), eid):
                    refuse(f"{os.path.basename(p)} is not on the same eval grid (eid differs)")
                runs[(c, k)] = {f: np.asarray(z[f]) for f in z.files if f in ("pred", "ade6", "tac_code_shift", "tac_oracle_fallback")}
                runs[(c, k)]["path"] = p
    return runs, eid


def checks(runs, eid):
    out = {}
    for arm in ARMS:
        for s in SEEDS:
            c = f"{arm}_s{s}"
            dev = float(np.abs(runs[(c, "pred")]["ade6"] - runs[(c, "banked")]["ade6"]).max())
            if dev > TOL_REPRO:
                refuse(f"{c}: the tacpred run differs from the banked eval by {dev:.3g} m per window (> {TOL_REPRO})")
            sh = runs[(c, "shuf")].get("tac_code_shift")
            if sh is None or float((sh > 1e-6).mean()) < MIN_CHANGED:
                refuse(f"{c}: the shuffle changed T's code on {0.0 if sh is None else float((sh > 1e-6).mean()):.3f} of windows "
                       f"(< {MIN_CHANGED}): the intervention did not happen")
            fb = runs[(c, "orc")].get("tac_oracle_fallback")
            if fb is None:
                refuse(f"{c}: the oracle run carries no tac_oracle_fallback array")
            out[c] = dict(repro_max_dev_m=dev, shuffle_frac_code_changed=float((sh > 1e-6).mean()),
                          shuffle_code_shift_mean=float(sh.mean()), oracle_fallback_windows=int(fb.sum()),
                          oracle_code_shift_mean=float(runs[(c, "orc")]["tac_code_shift"].mean()))
    return out


def read(a, pairfn, fam):
    runs, eid = load(a.eval_dir)
    chk = checks(runs, eid)
    F = {arm: abs(runs[(f"{arm}_s0", "pred")]["ade6"].mean() - runs[(f"{arm}_s1", "pred")]["ade6"].mean()) for arm in ARMS}
    rows = {}
    for arm in ARMS:
        for s in SEEDS:
            c = f"{arm}_s{s}"
            p, sh, orc = runs[(c, "pred")], runs[(c, "shuf")], runs[(c, "orc")]
            rows[c] = dict(ade6_pred=float(p["ade6"].mean()), ade6_shuf=float(sh["ade6"].mean()), ade6_orc=float(orc["ade6"].mean()),
                           shuf_minus_pred=pairfn(sh["ade6"], p["ade6"], eid, a.n_boot, 0),
                           orc_minus_pred_T0=pairfn(orc["ade6"], p["ade6"], eid, a.n_boot, 0))
            if fam is not None:
                rows[c]["families_shuf_minus_pred"] = fam(sh["pred"], p["pred"], eid)
                rows[c]["families_orc_minus_pred_T0"] = fam(orc["pred"], p["pred"], eid)
    used = {c: bool(r["shuf_minus_pred"]["delta"] > 0 and r["shuf_minus_pred"]["separated"]) for c, r in rows.items()}
    inert = {c: not bool(r["shuf_minus_pred"]["separated"]) for c, r in rows.items()}
    tac_used = "USED" if all(used.values()) else ("DECORATIVE" if all(inert.values()) else "PARTIAL")
    head = {}
    for arm in ARMS:
        ds = [rows[f"{arm}_s{s}"]["orc_minus_pred_T0"] for s in SEEDS]
        large = all(d["delta"] < 0 and d["separated"] and abs(d["delta"]) > 2 * F[arm] for d in ds)
        head[arm] = dict(verdict="LARGE" if large else "SMALL", two_F=2 * F[arm], deltas=[d["delta"] for d in ds],
                         separated=[bool(d["separated"]) for d in ds])
    if tac_used == "USED" and head["M2b"]["verdict"] == "LARGE":
        nxt = ("USED + LARGE: O uses T, and a perfect tactical layer would lower 6 s error beyond the training-seed floor. The tactical "
               "layer's QUALITY is the lever: next arm = v8F-HJ (the v9 tactical labels supervise the H-JEPA level-2 latent, T reads it; "
               "PLAN_V8F_REFAV1_HJEPA section 4.4 R3), under its own PREREG and launch gate; the weakest head first (longitudinal).")
    elif tac_used == "USED":
        nxt = ("USED + SMALL: O uses T, but a perfect tactical layer would buy less than the training-seed floor. The lever is the "
               "OPERATIVE layer (the longitudinal speed decision), not the tactical one.")
    elif tac_used == "DECORATIVE":
        nxt = ("DECORATIVE: O ignores T's output -- the hierarchy does not function end to end. Fix the T -> O wiring first (O must be "
               "trained to depend on the code: code dropout with GT-code teacher forcing, or the code as FiLM on O's residual head), "
               "then re-read TAC-USED before any tactical-quality arm.")
    else:
        nxt = ("PARTIAL: T -> O is live at " + ", ".join(c for c, u in used.items() if u) + " only. No lever is chosen on it; report "
               "per checkpoint and read the arm that is live.")
    out = dict(schema="v8f-tac-read-v1", prereg="PREREG_V8F_TAC.md", tier="T0/T1 open loop (the oracle arm is T0: it reads GT)",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |s0 - s1| per arm (training seed)",
               n_windows=int(eid.size), n_episodes=int(np.unique(eid).size), checks=chk, F=F, rows=rows,
               TAC_USED=dict(verdict=tac_used, per_checkpoint=used),
               TAC_HEADROOM_T0=head, next=nxt)
    out["reading"] = (f"TAC-USED {tac_used} (shuf - pred " + " / ".join(f"{c} {r['shuf_minus_pred']['delta']:+.4f}"
                      f"{'' if r['shuf_minus_pred']['separated'] else ' ns'}" for c, r in rows.items()) + "); TAC-HEADROOM (T0) "
                      + "; ".join(f"{arm} {head[arm]['verdict']} ({' / '.join(f'{d:+.4f}' for d in head[arm]['deltas'])}, "
                                  f"2F {head[arm]['two_F']:.4f})" for arm in ARMS) + ".")
    return out


def family_fn(a):
    """Paired LON / LAT components between two prediction sets on the same windows (GT rebuilt from the repo's eval inputs)."""
    _paths()
    E = _mod("ra1_hjepa_eval", REPO / "stack" / "scripts" / "ra1_hjepa_eval.py")
    R = _mod("v8f_mini_full_report", HERE.parent / "v8f_mini_full_report.py")
    gt, geid, dt, _prov = R.ground_truth(a)
    import torch
    g = torch.from_numpy(gt).float()

    def fam(pa, pb, eid):
        if not np.array_equal(np.asarray(geid), np.asarray(eid)):
            refuse("the rebuilt ground truth is not on the runs' grid")
        fa = E.family_per_window(torch.from_numpy(np.asarray(pa, np.float64)).float(), g, dt)
        fb = E.family_per_window(torch.from_numpy(np.asarray(pb, np.float64)).float(), g, dt)
        return E.paired_family_block(fa, fb, eid, min(a.n_boot, 2000), 0)
    return fam


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=B)
    ap.add_argument("--no-families", action="store_true", help="skip the descriptive family blocks (tests)")
    ap.add_argument("--eval-grid", default=str(RAW / "eval_grid_windows.json"))
    ap.add_argument("--eval-kin", default=str(RAW / "eval141_kin.npz"))
    ap.add_argument("--eval-join", default=str(RAW / "join_eval141.npz"))
    ap.add_argument("--gt-npz", help="TEST INPUT: npz with gt, eid, dt in place of the rebuild")
    a = ap.parse_args(argv)
    _paths()
    from taniteval.ci import paired_episode_cluster_bootstrap
    try:
        res = read(a, paired_episode_cluster_bootstrap, None if a.no_families else family_fn(a))
    except (OSError, KeyError, ValueError) as e:
        print(f"TAC READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
