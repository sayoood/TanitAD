"""PREREG_V8F_CZ stage 1, committed before its data: is the ABSENT-slot mismatch worth a training run?

  python v8f_cz_gate_read.py --eval-dir <eval> --tac-read tac_read.json --out cz_gate.json

MEASURED 2026-10-21 (PREREG_V8F_TAC Amendment 2's saved codes): T's loss MASKS an absent constraint (v8f_mini.py, `cons_absent` mask),
while _code_gt writes 0 there and O is teacher-forced on it. At inference T writes unsupervised values in those slots, and they carry
92-95 % of T's squared constraint error in latcons / stoplead and 52-54 % in lonspeed.

Per checkpoint c in {M1_s0, M1_s1, M2b_s0, M2b_s1}: eval_<c>-tacabs (`v8f_mini_eval.py run --tac absmask`: T's own code with every slot
the GT marks absent set to 0) vs eval_<c>-tacpred, paired over episodes; share = d_abs / d_all (d_all = PREREG_V8F_TAC's full oracle).
GATE, read on M2b: PASS iff d_abs < 0, separated and |d_abs| > 2F at BOTH seeds (F = |mean pred_s0 - mean pred_s1|).
  PASS -> stage 2 trains M2b-cz (T's target for an absent slot = 0), seeds 0 and 1.
  FAIL -> no training on this lever; the next lever is T's PRESENT-slot accuracy, under its own PREREG.
T0: only the PRESENCE of each constraint comes from the GT; every present value is T's own.
Refusals (rc 2): a missing file; files not on one grid; a tacabs run whose manipulation is not absmask or whose absent-slot fraction is
not in (0, 1); a tacpred run that is not the one PREREG_V8F_TAC read (ADE differs by > 1e-3 m).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (EPISODES)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ARMS, SEEDS = ("M1", "M2b"), (0, 1)
B = 2000


def refuse(msg):
    print(f"CZ GATE READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _r1():
    spec = importlib.util.spec_from_file_location("v8f_tac_groups_read_for_cz", HERE.parent / "v8f_tac_groups_read.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load(eval_dir):
    runs, eid = {}, None
    for arm in ARMS:
        for s in SEEDS:
            c = f"{arm}_s{s}"
            for suf in ("tacpred", "tacorc", "tacabs"):
                name = f"eval_{c}-{suf}"
                pz = os.path.join(eval_dir, name + ".npz")
                if not os.path.exists(pz):
                    refuse(f"{name}.npz is missing from {eval_dir}")
                z = np.load(pz)
                if eid is None:
                    eid = np.asarray(z["eid"])
                elif not np.array_equal(np.asarray(z["eid"]), eid):
                    refuse(f"{name}.npz is not on the same eval grid (eid differs)")
                rec = None
                if suf == "tacabs":
                    pj = os.path.join(eval_dir, name + ".json")
                    if not os.path.exists(pj):
                        refuse(f"{name}.json is missing (the manipulation record)")
                    rec = json.load(open(pj, encoding="utf-8")).get("tac_manipulation") or {}
                    fr = rec.get("absent_slot_frac")
                    if rec.get("mode") != "absmask" or fr is None or not 0.0 < float(fr) < 1.0:
                        refuse(f"{name}: manipulation {rec.get('mode')} with absent-slot fraction {fr}, expected absmask in (0, 1)")
                runs[(c, suf)] = {"ade6": np.asarray(z["ade6"], np.float64), "pred": np.asarray(z["pred"], np.float64), "rec": rec}
    return runs, eid


def read(a, pairfn, fam):
    runs, eid = load(a.eval_dir)
    if a.tac_read:
        prev = json.load(open(a.tac_read, encoding="utf-8"))["rows"]
        for c, r in prev.items():
            got = runs[(c, "tacpred")]["ade6"].mean()
            if abs(got - r["ade6_pred"]) > 1e-3:
                refuse(f"{c}: the tacpred run reads {got:.4f}, PREREG_V8F_TAC read {r['ade6_pred']:.4f}")
    F = {arm: abs(runs[(f"{arm}_s0", "tacpred")]["ade6"].mean() - runs[(f"{arm}_s1", "tacpred")]["ade6"].mean()) for arm in ARMS}
    rows = {}
    for arm in ARMS:
        for s in SEEDS:
            c = f"{arm}_s{s}"
            p = runs[(c, "tacpred")]
            d_all = pairfn(runs[(c, "tacorc")]["ade6"], p["ade6"], eid, a.n_boot, 0)
            d = pairfn(runs[(c, "tacabs")]["ade6"], p["ade6"], eid, a.n_boot, 0)
            rec = runs[(c, "tacabs")]["rec"]
            rows[c] = dict(ade6_pred=float(p["ade6"].mean()), ade6_abs=float(runs[(c, "tacabs")]["ade6"].mean()), absmask=d,
                           oracle_all=d_all, share=float(d["delta"] / d_all["delta"]), absent_slot_frac=rec.get("absent_slot_frac"),
                           code_shift_mean=rec.get("code_shift_mean"))
            if fam is not None:
                rows[c]["families"] = fam(runs[(c, "tacabs")]["pred"], p["pred"], eid)

    def gate(arm):
        ok = all(rows[f"{arm}_s{s}"]["absmask"]["delta"] < 0 and rows[f"{arm}_s{s}"]["absmask"]["separated"]
                 and abs(rows[f"{arm}_s{s}"]["absmask"]["delta"]) > 2 * F[arm] for s in SEEDS)
        return dict(gate="PASS" if ok else "FAIL", two_F=2 * F[arm])

    dec = {arm: gate(arm) for arm in ARMS}
    g = dec["M2b"]["gate"]
    nxt = ("PASS: zeroing the absent slots alone buys more than 2F at both M2b seeds. Stage 2 trains M2b-cz (T's target for an absent "
           "constraint = the 0 O is teacher-forced on), seeds 0 and 1, under the launch gate." if g == "PASS" else
           "FAIL: the absent-slot mismatch does not buy 2F at both M2b seeds. No training on this lever; the next lever is T's PRESENT-slot "
           "accuracy, under its own PREREG.")
    out = dict(schema="v8f-cz-gate-read-v1", prereg="PREREG_V8F_CZ.md stage 1", tier="T0 (presence from the GT; values are T's own)",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |s0 - s1| per arm", n_windows=int(eid.size),
               n_episodes=int(np.unique(eid).size), F=F, rows=rows, DECISION=dec, GATE=g, next=nxt)
    out["reading"] = (f"PREREG_V8F_CZ stage 1 (M2b): GATE {g}; absmask - pred " +
                      " / ".join(f"{rows[f'M2b_s{s}']['absmask']['delta']:+.3f}" for s in SEEDS) + " m (share " +
                      " / ".join(f"{rows[f'M2b_s{s}']['share']:+.2f}" for s in SEEDS) + f"), 2F {dec['M2b']['two_F']:.4f}; M1 " +
                      " / ".join(f"{rows[f'M1_s{s}']['absmask']['delta']:+.3f}" for s in SEEDS) + ".")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tac-read", help="PREREG_V8F_TAC's tac_read.json: the tacpred runs must be the ones it read")
    ap.add_argument("--n-boot", type=int, default=B)
    ap.add_argument("--no-families", action="store_true", help="skip the descriptive family blocks (tests)")
    ap.add_argument("--eval-grid", default=str(HERE.parents[1] / "raw" / "eval_grid_windows.json"))
    ap.add_argument("--eval-kin", default=str(HERE.parents[1] / "raw" / "eval141_kin.npz"))
    ap.add_argument("--eval-join", default=str(HERE.parents[1] / "raw" / "join_eval141.npz"))
    ap.add_argument("--gt-npz", help="TEST INPUT: npz with gt, eid, dt in place of the rebuild")
    a = ap.parse_args(argv)
    R1 = _r1()
    R1._paths()
    from taniteval.ci import paired_episode_cluster_bootstrap
    try:
        res = read(a, paired_episode_cluster_bootstrap, None if a.no_families else R1.family_fn(a))
    except (OSError, KeyError, ValueError) as e:
        print(f"CZ GATE READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
