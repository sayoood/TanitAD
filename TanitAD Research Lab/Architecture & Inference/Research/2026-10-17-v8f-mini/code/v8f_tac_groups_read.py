"""PREREG_V8F_TAC Amendment 1's reading, committed before its data: WHICH part of the tactical code carries the oracle's headroom?

  python v8f_tac_groups_read.py --eval-dir <runs/v8f_mini_full/eval> --out tac_groups_read.json

Inputs per checkpoint c in {M1_s0, M1_s1, M2b_s0, M2b_s1}: eval_<c>-tacpred.npz and eval_<c>-tacorc.npz (PREREG_V8F_TAC's runs) and
eval_<c>-tacorc-<g>.npz for g in GROUPS + COMPOSITES (`v8f_mini_eval.py run --tac oracle --tac-groups <g>`): O reads the GT code in
group g only and T's own code everywhere else. The six groups PARTITION the code (`v8f_mini.CODE_GROUPS`).

Per checkpoint and group: d_g = orc_g - pred (eval-141 ADE@6s, paired over episodes) and share_g = d_g / d_all (d_all = the full oracle).
A group CARRIES for an arm iff d_g < 0, separated, and |d_g| > 2 F_arm at BOTH seeds (F_arm = |mean pred_s0 - mean pred_s1|).
The DECISION is read on M2b (the K-PASS arm):
  CLASSIFIER  iff `decisions` (goal + latdec + londec) CARRIES and share >= 0.5 at both seeds
  SPEED       iff not CLASSIFIER, and `lonspeed` CARRIES and share >= 0.5 at both seeds
  MIXED       otherwise (the two single groups with the largest mean share are named; no arm is chosen on them alone)
Beside it, deciding nothing: every group's row at every checkpoint, M1's decision, and each group's LON / LAT family deltas.
Refusals (rc 2): a missing file; files not on one grid; a group run that replaced a number of code dimensions other than its group's
size, or whose mode is not the oracle; a tacpred run that is not the one PREREG_V8F_TAC read (its ADE differs by > 1e-3 m).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (EPISODES). Every oracle arm is T0: it reads GT."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
RAW = HERE.parents[1] / "raw"
ARMS, SEEDS = ("M1", "M2b"), (0, 1)
GROUPS = ("goal", "latdec", "londec", "latcons", "lonspeed", "stoplead")
COMPOSITES = ("decisions", "constraints")
B = 2000


def refuse(msg):
    print(f"TAC GROUPS READ REFUSED: {msg}", file=sys.stderr)
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


def load(eval_dir, sizes):
    runs, eid = {}, None
    for arm in ARMS:
        for s in SEEDS:
            c = f"{arm}_s{s}"
            for g in ("pred", "all") + GROUPS + COMPOSITES:
                name = {"pred": f"eval_{c}-tacpred", "all": f"eval_{c}-tacorc"}.get(g, f"eval_{c}-tacorc-{g}")
                pz, pj = os.path.join(eval_dir, name + ".npz"), os.path.join(eval_dir, name + ".json")
                if not os.path.exists(pz):
                    refuse(f"{name}.npz is missing from {eval_dir}")
                z = np.load(pz)
                if eid is None:
                    eid = np.asarray(z["eid"])
                elif not np.array_equal(np.asarray(z["eid"]), eid):
                    refuse(f"{name}.npz is not on the same eval grid (eid differs)")
                if g in GROUPS + COMPOSITES:
                    if not os.path.exists(pj):
                        refuse(f"{name}.json is missing (the manipulation record)")
                    m = json.load(open(pj, encoding="utf-8")).get("tac_manipulation") or {}
                    if m.get("mode") != "oracle" or m.get("groups") != g or m.get("n_code_dims_replaced") != sizes[g]:
                        refuse(f"{name}: manipulation {m.get('mode')} / {m.get('groups')} / {m.get('n_code_dims_replaced')} dims, "
                               f"expected oracle / {g} / {sizes[g]}")
                runs[(c, g)] = {"ade6": np.asarray(z["ade6"], np.float64), "pred": np.asarray(z["pred"], np.float64), "path": pz}
    return runs, eid


def read(a, pairfn, fam):
    _paths()
    from tanitad.models import v8f_mini as M
    sizes = {g: len(M.code_group_index(g)) for g in GROUPS + COMPOSITES}
    runs, eid = load(a.eval_dir, sizes)
    if a.tac_read:
        prev = json.load(open(a.tac_read, encoding="utf-8"))["rows"]
        for c, r in prev.items():
            dev = abs(runs[(c, "pred")]["ade6"].mean() - r["ade6_pred"])
            if dev > 1e-3:
                refuse(f"{c}: the tacpred run reads {runs[(c, 'pred')]['ade6'].mean():.4f}, PREREG_V8F_TAC read {r['ade6_pred']:.4f}")
    F = {arm: abs(runs[(f"{arm}_s0", "pred")]["ade6"].mean() - runs[(f"{arm}_s1", "pred")]["ade6"].mean()) for arm in ARMS}
    rows = {}
    for arm in ARMS:
        for s in SEEDS:
            c = f"{arm}_s{s}"
            p = runs[(c, "pred")]
            d_all = pairfn(runs[(c, "all")]["ade6"], p["ade6"], eid, a.n_boot, 0)
            gr = {}
            for g in GROUPS + COMPOSITES:
                d = pairfn(runs[(c, g)]["ade6"], p["ade6"], eid, a.n_boot, 0)
                gr[g] = dict(paired=d, ade6=float(runs[(c, g)]["ade6"].mean()),
                             share=float(d["delta"] / d_all["delta"]) if d_all["delta"] != 0 else None)
                if fam is not None:
                    gr[g]["families"] = fam(runs[(c, g)]["pred"], p["pred"], eid)
            rows[c] = dict(ade6_pred=float(p["ade6"].mean()), ade6_oracle_all=float(runs[(c, "all")]["ade6"].mean()), oracle_all=d_all,
                           groups=gr)

    def carries(arm, g):
        return all(rows[f"{arm}_s{s}"]["groups"][g]["paired"]["delta"] < 0 and rows[f"{arm}_s{s}"]["groups"][g]["paired"]["separated"]
                   and abs(rows[f"{arm}_s{s}"]["groups"][g]["paired"]["delta"]) > 2 * F[arm] for s in SEEDS)

    def share_ge(arm, g, x):
        return all((rows[f"{arm}_s{s}"]["groups"][g]["share"] or 0.0) >= x for s in SEEDS)

    def decide(arm):
        mean_share = {g: float(np.mean([rows[f"{arm}_s{s}"]["groups"][g]["share"] or 0.0 for s in SEEDS])) for g in GROUPS}
        top = sorted(GROUPS, key=lambda g: -mean_share[g])[:2]
        if carries(arm, "decisions") and share_ge(arm, "decisions", 0.5):
            v = "CLASSIFIER"
        elif carries(arm, "lonspeed") and share_ge(arm, "lonspeed", 0.5):
            v = "SPEED"
        else:
            v = "MIXED"
        return dict(verdict=v, carries={g: carries(arm, g) for g in GROUPS + COMPOSITES}, mean_share=mean_share, top_two=top,
                    two_F=2 * F[arm])

    dec = {arm: decide(arm) for arm in ARMS}
    d = dec["M2b"]
    if d["verdict"] == "CLASSIFIER":
        nxt = ("CLASSIFIER: the categorical tactical DECISIONS carry at least half of the oracle's headroom at both M2b seeds, beyond 2F. "
               "Next arm = v8F-HJ: the v9 tactical labels supervise the H-JEPA level-2 latent and T reads it (PLAN_V8F section 4.4 R3), the "
               f"decision group with the largest share first ({d['top_two'][0]}), under its own PREREG and launch gate; PI go (architecture).")
    elif d["verdict"] == "SPEED":
        nxt = ("SPEED: the longitudinal SPEED constraints (target / end speed, reach time and distance, speed band) carry at least half of "
               "the headroom at both M2b seeds, beyond 2F, and the decisions do not. Next arm = a speed-target head inside T (vision + nav + "
               "ego dynamics only), trained on the v9 lon speed constraints with its own weight, O conditioned on it; under its own PREREG "
               "and launch gate; PI go.")
    else:
        nxt = (f"MIXED: neither the decisions nor the speed constraints alone carry half of the headroom at both M2b seeds beyond 2F. The "
               f"largest single groups are {d['top_two'][0]} and {d['top_two'][1]}; no arm is chosen on them alone -- their joint oracle "
               f"is the next read.")
    out = dict(schema="v8f-tac-groups-read-v1", prereg="PREREG_V8F_TAC.md Amendment 1", tier="T0 (every oracle arm reads GT)",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); F = |s0 - s1| per arm",
               n_windows=int(eid.size), n_episodes=int(np.unique(eid).size), group_sizes=sizes, F=F, rows=rows, DECISION=dec, next=nxt)
    out["reading"] = ("Amendment 1 (M2b): " + d["verdict"] + "; mean share of the full oracle: " +
                      ", ".join(f"{g} {d['mean_share'][g]:+.2f}" for g in GROUPS) +
                      "; decisions " + " / ".join(f"{rows[f'M2b_s{s}']['groups']['decisions']['share']:+.2f}" for s in SEEDS) +
                      ", constraints " + " / ".join(f"{rows[f'M2b_s{s}']['groups']['constraints']['share']:+.2f}" for s in SEEDS) +
                      f" (2F {d['two_F']:.4f}).")
    return out


def family_fn(a):
    _paths()
    E = _mod("ra1_hjepa_eval", REPO / "stack" / "scripts" / "ra1_hjepa_eval.py")
    R = _mod("v8f_mini_full_report", HERE.parent / "v8f_mini_full_report.py")
    gt, geid, dt, _ = R.ground_truth(a)
    import torch
    g = torch.from_numpy(gt).float()

    def fam(pa, pb, eid):
        if not np.array_equal(np.asarray(geid), np.asarray(eid)):
            refuse("the rebuilt ground truth is not on the runs' grid")
        f = lambda p: E.family_per_window(torch.from_numpy(np.asarray(p, np.float64)).float(), g, dt)      # noqa: E731
        return E.paired_family_block(f(pa), f(pb), eid, min(a.n_boot, 2000), 0)
    return fam


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tac-read", help="PREREG_V8F_TAC's tac_read.json: the tacpred runs must be the ones it read")
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
        print(f"TAC GROUPS READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
