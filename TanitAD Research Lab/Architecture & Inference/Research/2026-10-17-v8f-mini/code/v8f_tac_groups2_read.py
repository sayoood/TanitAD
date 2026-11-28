"""PREREG_V8F_TAC Amendment 2's reading, committed before its data: what must T predict better, and can partial accuracy pay?

  python v8f_tac_groups2_read.py --eval-dir <eval> --groups-read tac_groups_read.json --out tac_groups2_read.json

Amendment 1 read MIXED: the constraints TOGETHER carry 0.88-0.90 of the oracle's headroom and the decisions 0.07-0.12, while single
constraint groups HURT alone -- O was trained on whole codes only (v8f_mini.py:601-602), so one GT group beside T's own code is a mix it
never saw. Three reads, per checkpoint c in {M1_s0, M1_s1, M2b_s0, M2b_s1}, all T0 (they read GT), all paired over episodes vs `pred`:

  A  `lonspeed+londec` (eval_<c>-tacorc-lonspeed+londec): the joint oracle Amendment 1's MIXED branch committed to.
     A CARRIES iff d_A < 0, separated, |d_A| > 2F and share_A = d_A / d_all >= 0.5 at both seeds.
  B  leave-one-out (eval_<c>-tacorc-all-<g>): GT everywhere EXCEPT group g -- the mixes closest to the whole code O was trained on.
     loss_g = all-g minus all (paired); necessity_g = loss_g / |d_all|. g is NECESSARY iff loss_g > 0, separated, loss_g > 2F and
     necessity_g >= 0.25 at both seeds.
  C  dose-response (eval_<c>-tacorc-constraints-a{25,50,75}): the constraints moved a fraction alpha of the way from T's code to the GT.
     r(alpha) = d_cons(alpha) / d_cons(1), with d_cons(1) = Amendment 1's `constraints` arm. At alpha 0.5, both seeds:
     PAYS iff r >= 0.4 and d_cons(0.5) < 0 separated; THRESHOLD iff r < 0.2; else INTERMEDIATE.

The DECISION is read on M2b (F = |mean pred_s0 - mean pred_s1|):
  STRUCTURAL          iff C is THRESHOLD: partial accuracy does not pay, so a loss-level arm on T is not expected to move ADE
  LON-TACTIC          iff not STRUCTURAL and A CARRIES: the training arm targets T's longitudinal tactic (lon decision + lonspeed)
  CONSTRAINT-ACCURACY otherwise: the training arm targets T's constraint prediction, on the NECESSARY groups (all three if none is)
Beside it, deciding nothing: every row at every checkpoint, M1's decision, the LON / LAT family deltas, and T's own code error per
group (from the saved `tac_code`: T's values in g come from all-g, the GT from another all-h arm).
Refusals (rc 2): a missing file; files not on one grid; a manipulation record that is not oracle / <spec> / <n dims> / <alpha>; a
tacpred or constraints run that is not the one Amendment 1 read (ADE differs by > 1e-3 m).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (EPISODES)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
R1 = None                       # Amendment 1's reader (shared helpers), loaded in main()
ARMS, SEEDS = ("M1", "M2b"), (0, 1)
GROUPS = ("goal", "latdec", "londec", "latcons", "lonspeed", "stoplead")
CONS_GROUPS = ("latcons", "lonspeed", "stoplead")
A_SPEC = "lonspeed+londec"
ALPHAS = (25, 50, 75)
B = 2000


def refuse(msg):
    print(f"TAC GROUPS2 READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _arms(M):
    """suffix -> (groups spec, n dims, alpha) for every run this reading needs (None = not a manipulation-checked run)."""
    arms = {"tacpred": None, "tacorc": None, "tacorc-constraints": ("constraints", 26, None)}
    arms[f"tacorc-{A_SPEC}"] = (A_SPEC, len(M.code_group_index(A_SPEC)), 1.0)
    for g in GROUPS:
        arms[f"tacorc-all-{g}"] = (f"all-{g}", len(M.code_group_index(f"all-{g}")), 1.0)
    for a in ALPHAS:
        arms[f"tacorc-constraints-a{a}"] = ("constraints", 26, a / 100)
    return arms


def load(eval_dir, arms):
    runs, eid = {}, None
    for arm in ARMS:
        for s in SEEDS:
            c = f"{arm}_s{s}"
            for suf, want in arms.items():
                name = f"eval_{c}-{suf}"
                pz, pj = os.path.join(eval_dir, name + ".npz"), os.path.join(eval_dir, name + ".json")
                if not os.path.exists(pz):
                    refuse(f"{name}.npz is missing from {eval_dir}")
                z = np.load(pz)
                if eid is None:
                    eid = np.asarray(z["eid"])
                elif not np.array_equal(np.asarray(z["eid"]), eid):
                    refuse(f"{name}.npz is not on the same eval grid (eid differs)")
                if want is not None:
                    if not os.path.exists(pj):
                        refuse(f"{name}.json is missing (the manipulation record)")
                    m = json.load(open(pj, encoding="utf-8")).get("tac_manipulation") or {}
                    alpha_ok = want[2] is None or abs(float(m.get("alpha") if m.get("alpha") is not None else -1) - want[2]) < 1e-9
                    if m.get("mode") != "oracle" or m.get("groups") != want[0] or m.get("n_code_dims_replaced") != want[1] or not alpha_ok:
                        refuse(f"{name}: manipulation {m.get('mode')} / {m.get('groups')} / {m.get('n_code_dims_replaced')} dims / alpha "
                               f"{m.get('alpha')}, expected oracle / {want[0]} / {want[1]} / {want[2]}")
                runs[(c, suf)] = {"ade6": np.asarray(z["ade6"], np.float64), "pred": np.asarray(z["pred"], np.float64),
                                  "code": np.asarray(z["tac_code"], np.float64) if "tac_code" in z.files else None,
                                  "fb": np.asarray(z["tac_oracle_fallback"], bool) if "tac_oracle_fallback" in z.files else None}
    return runs, eid


def code_error(runs, c, M):
    """T's own code error per group, descriptive: T's values in g from all-g, the GT from all-h (h the next group)."""
    out = {}
    for i, g in enumerate(GROUPS):
        h = GROUPS[(i + 1) % len(GROUPS)]
        t, gt = runs[(c, f"tacorc-all-{g}")], runs[(c, f"tacorc-all-{h}")]
        if t["code"] is None or gt["code"] is None:
            return {"status": "UNAVAILABLE", "reason": "no tac_code in the runs (an eval older than Amendment 2)"}
        idx = M.CODE_GROUPS[g]
        ok = ~t["fb"] if t["fb"] is not None else np.ones(t["code"].shape[0], bool)
        e = t["code"][ok][:, idx] - gt["code"][ok][:, idx]
        kind = "constraint RMSE in standard deviations" if g in CONS_GROUPS else "mean |prob - GT| per dim"
        out[g] = {"value": float(np.sqrt((e ** 2).mean()) if g in CONS_GROUPS else np.abs(e).mean()), "unit": kind,
                  "n_windows": int(ok.sum())}
        if g == "lonspeed":
            names = [n for n in M.CONS_NAMES if M.N_GOAL + M.N_LAT + M.N_LON + M.CI[n] in idx]
            out[g]["per_dim_rmse_sd"] = {n: float(np.sqrt((e[:, k] ** 2).mean())) for k, n in enumerate(names)}
    return out


def read(a, pairfn, fam):
    R1._paths()
    from tanitad.models import v8f_mini as M
    arms = _arms(M)
    runs, eid = load(a.eval_dir, arms)
    prev = json.load(open(a.groups_read, encoding="utf-8"))["rows"]
    for c, r in prev.items():
        for suf, ref in (("tacpred", r["ade6_pred"]), ("tacorc-constraints", r["groups"]["constraints"]["ade6"])):
            got = runs[(c, suf)]["ade6"].mean()
            if abs(got - ref) > 1e-3:
                refuse(f"{c}: eval_{c}-{suf} reads {got:.4f}, Amendment 1 read {ref:.4f}")
    F = {arm: abs(runs[(f"{arm}_s0", "tacpred")]["ade6"].mean() - runs[(f"{arm}_s1", "tacpred")]["ade6"].mean()) for arm in ARMS}
    rows = {}
    for arm in ARMS:
        for s in SEEDS:
            c = f"{arm}_s{s}"
            p, full = runs[(c, "tacpred")], runs[(c, "tacorc")]
            d_all = pairfn(full["ade6"], p["ade6"], eid, a.n_boot, 0)
            dA = pairfn(runs[(c, f"tacorc-{A_SPEC}")]["ade6"], p["ade6"], eid, a.n_boot, 0)
            rowA = dict(paired=dA, share=float(dA["delta"] / d_all["delta"]))
            loo = {}
            for g in GROUPS:
                lg = pairfn(runs[(c, f"tacorc-all-{g}")]["ade6"], full["ade6"], eid, a.n_boot, 0)
                loo[g] = dict(loss=lg, necessity=float(lg["delta"] / abs(d_all["delta"])))
            d1 = pairfn(runs[(c, "tacorc-constraints")]["ade6"], p["ade6"], eid, a.n_boot, 0)
            dose = {"1.00": dict(paired=d1, r=1.0)}
            for al in ALPHAS:
                dd = pairfn(runs[(c, f"tacorc-constraints-a{al}")]["ade6"], p["ade6"], eid, a.n_boot, 0)
                dose[f"{al / 100:.2f}"] = dict(paired=dd, r=float(dd["delta"] / d1["delta"]))
            if fam is not None:
                rowA["families"] = fam(runs[(c, f"tacorc-{A_SPEC}")]["pred"], p["pred"], eid)
                for g in GROUPS:
                    loo[g]["families_vs_all"] = fam(runs[(c, f"tacorc-all-{g}")]["pred"], full["pred"], eid)
                for al in ALPHAS:
                    dose[f"{al / 100:.2f}"]["families"] = fam(runs[(c, f"tacorc-constraints-a{al}")]["pred"], p["pred"], eid)
            rows[c] = dict(ade6_pred=float(p["ade6"].mean()), oracle_all=d_all, A=rowA, leave_one_out=loo, dose=dose,
                           t_code_error=code_error(runs, c, M))

    def both(arm, fn):
        return all(fn(rows[f"{arm}_s{s}"]) for s in SEEDS)

    def decide(arm):
        two_f = 2 * F[arm]
        a_carries = both(arm, lambda r: r["A"]["paired"]["delta"] < 0 and r["A"]["paired"]["separated"]
                         and abs(r["A"]["paired"]["delta"]) > two_f and r["A"]["share"] >= 0.5)
        necessary = {g: both(arm, lambda r, g=g: r["leave_one_out"][g]["loss"]["delta"] > 0 and r["leave_one_out"][g]["loss"]["separated"]
                             and r["leave_one_out"][g]["loss"]["delta"] > two_f and r["leave_one_out"][g]["necessity"] >= 0.25)
                     for g in GROUPS}
        h = lambda r: r["dose"]["0.50"]                                     # noqa: E731
        if both(arm, lambda r: h(r)["r"] >= 0.4 and h(r)["paired"]["delta"] < 0 and h(r)["paired"]["separated"]):
            dose_cls = "PAYS"
        elif both(arm, lambda r: h(r)["r"] < 0.2):
            dose_cls = "THRESHOLD"
        else:
            dose_cls = "INTERMEDIATE"
        if dose_cls == "THRESHOLD":
            v = "STRUCTURAL"
        elif a_carries:
            v = "LON-TACTIC"
        else:
            v = "CONSTRAINT-ACCURACY"
        return dict(verdict=v, A_carries=a_carries, necessary=necessary, dose=dose_cls, two_F=two_f,
                    r_half=[rows[f"{arm}_s{s}"]["dose"]["0.50"]["r"] for s in SEEDS])

    dec = {arm: decide(arm) for arm in ARMS}
    d = dec["M2b"]
    nec = [g for g in GROUPS if d["necessary"][g]]
    tgt = [g for g in nec if g in CONS_GROUPS] or list(CONS_GROUPS)
    if d["verdict"] == "STRUCTURAL":
        nxt = ("STRUCTURAL: moving T's constraints half-way to the GT buys under a fifth of the full move at both M2b seeds, so a loss-level "
               "arm on T is not expected to move ADE. The next arm changes what T sees or how it predicts the constraint set jointly -- an "
               "architecture change: named here, built only on the PI's go.")
    elif d["verdict"] == "LON-TACTIC":
        nxt = ("LON-TACTIC: the lon decision and the speed constraints together carry at least half of the headroom at both M2b seeds "
               f"beyond 2F, and partial accuracy pays ({d['dose']}). Next arm = a training arm on T's longitudinal tactic (the lon "
               "decision + lonspeed loss terms), under its own PREREG and launch gate; PI go.")
    else:
        nxt = (f"CONSTRAINT-ACCURACY: partial accuracy pays ({d['dose']}, r(0.5) {d['r_half'][0]:+.2f} / {d['r_half'][1]:+.2f}) and the "
               f"longitudinal pair alone does not carry. Next arm = a training arm on T's constraint prediction, targeting "
               f"{' + '.join(tgt)} ({'NECESSARY by leave-one-out' if nec else 'no group is necessary alone: all three'}), under its own "
               "PREREG and launch gate; PI go.")
    out = dict(schema="v8f-tac-groups2-read-v1", prereg="PREREG_V8F_TAC.md Amendment 2", tier="T0 (every oracle arm reads GT)",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |s0 - s1| per arm",
               n_windows=int(eid.size), n_episodes=int(np.unique(eid).size), F=F, rows=rows, DECISION=dec, next=nxt)
    out["reading"] = (f"Amendment 2 (M2b): {d['verdict']}; A {A_SPEC} share " +
                      " / ".join(f"{rows[f'M2b_s{s}']['A']['share']:+.2f}" for s in SEEDS) +
                      "; leave-one-out necessity " + ", ".join(
                          f"{g} " + "/".join(f"{rows[f'M2b_s{s}']['leave_one_out'][g]['necessity']:+.2f}" for s in SEEDS) for g in GROUPS) +
                      f"; dose {d['dose']} r(0.25/0.5/0.75) " + " | ".join(
                          "/".join(f"{rows[f'M2b_s{s}']['dose'][k]['r']:+.2f}" for k in ('0.25', '0.50', '0.75')) for s in SEEDS) +
                      f" (2F {d['two_F']:.4f}).")
    return out


def main(argv=None) -> int:
    global R1
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups-read", required=True, help="Amendment 1's tac_groups_read.json: tacpred and constraints must be its runs")
    ap.add_argument("--n-boot", type=int, default=B)
    ap.add_argument("--no-families", action="store_true", help="skip the descriptive family blocks (tests)")
    ap.add_argument("--eval-grid", default=str(HERE.parents[1] / "raw" / "eval_grid_windows.json"))
    ap.add_argument("--eval-kin", default=str(HERE.parents[1] / "raw" / "eval141_kin.npz"))
    ap.add_argument("--eval-join", default=str(HERE.parents[1] / "raw" / "join_eval141.npz"))
    ap.add_argument("--gt-npz", help="TEST INPUT: npz with gt, eid, dt in place of the rebuild")
    a = ap.parse_args(argv)
    import importlib.util
    spec = importlib.util.spec_from_file_location("v8f_tac_groups_read_for_a2", HERE.parent / "v8f_tac_groups_read.py")
    R1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R1)
    R1._paths()
    from taniteval.ci import paired_episode_cluster_bootstrap
    try:
        res = read(a, paired_episode_cluster_bootstrap, None if a.no_families else R1.family_fn(a))
    except (OSError, KeyError, ValueError) as e:
        print(f"TAC GROUPS2 READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
