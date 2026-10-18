"""PREREG_RA1_FULL verdict (../PREREG_RA1_FULL.md).  Two modes, both pure JSON, no torch:

  select  --root <ra1_yaw_full> --budgets 500 1000 3000 6000 --out full1.json
          Phase 1. Per arm (R0; A1 b0.08g50) and budget S: the seed-mean ADE@6s on the 60-clip Amendment-1 HOLDOUT.
          S*_arm = the lowest; ties (within 0.01 m of the minimum) go to the SHORTER budget.  An S* on the grid's edge is flagged and
          the pre-registered extension budget named (below the grid: S/2; above it: 2S).  ONE extension per direction: after it, re-select
          with --extended, and an S* still on the edge is read there and disclosed, never extended again.  The K arm = the lower seed-mean holdout ADE of
          R0@S*_R0 and A1@S*_A1; a tie (within 0.01 m) goes to R0, the simpler model.  eval-141 numbers are reported and decide NOTHING.
  read    --full1 full1.json --verdict <st<S*_A1>/full2_verdict.json> --out full_final.json
          Phase 2. Reading 3 (the A1x gate at S*_A1) and, only if it passes, reading 4 (K for the holdout-chosen arm, PREREG_HJ_STAGE0's
          bar with P10 = P10-yaw-4512).

Every run must be a compliant (yaw) run that scored ONE holdout, trained on ONE train list, against ONE prior, from a cache of 4,572
train clips -- a run from the 600-clip cache is refused, not averaged in.
Exit code: 0 when the inputs parse (the verdict is in the JSON); 2 on a missing / malformed input."""
from __future__ import annotations

import argparse
import json
import os
import sys

TIE_M = 0.01
CELL = "b0.08g50"
N_TRAIN_CACHE = 4572
HOLDOUT_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"   # the ES runs' 60-clip Amendment-1 holdout
ARMS = {"R0": "R0_s{seed}", "A1": "A1_" + CELL + "_s{seed}"}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run(dirpath, arm, seed):
    p = os.path.join(dirpath, ARMS[arm].format(seed=seed), "eval", "eval.json")
    if not os.path.exists(p):
        return None, p
    ev = _load(p)
    if ev.get("arm") != arm or ev.get("kin_mode") != "yaw":
        raise ValueError(f"{p} is arm={ev.get('arm')} kin_mode={ev.get('kin_mode')}; expected a compliant (yaw) {arm} run")
    if not ev.get("instrument_ok"):
        raise ValueError(f"{p} has instrument_ok=false")
    n_tr = (ev.get("disjointness") or {}).get("n_train")
    if n_tr != N_TRAIN_CACHE:
        raise ValueError(f"{p} was trained from a cache of {n_tr} train clips, not the full {N_TRAIN_CACHE} (wrong cache root?)")
    return ev, p


def select(a) -> dict:
    budgets = sorted(set(int(s) for s in a.budgets))
    rows, missing = {arm: {} for arm in ARMS}, []
    hold_sha, train_sha, p10_hold = set(), set(), set()
    for arm in ARMS:
        for S in budgets:
            d, per = os.path.join(a.root, f"st{S}"), {}
            for seed in (0, 1):
                ev, p = _run(d, arm, seed)
                if ev is None:
                    missing.append(p)
                    continue
                if int(ev.get("ckpt_step", -1)) != S:
                    raise ValueError(f"{p} reports ckpt_step {ev.get('ckpt_step')} for budget {S}")
                h, k = ev["holdout"], ev["K"]
                hold_sha.add(h.get("clips_sha256")); train_sha.add(h.get("train_sha256")); p10_hold.add(round(h["ade6_p10"]["mean"], 4))
                per[f"seed{seed}"] = dict(path=p, holdout_ade6=h["ade6_arm"]["mean"], holdout_p10_ade6=h["ade6_p10"]["mean"],
                                          holdout_delta=h.get("ade6_arm_minus_p10"),
                                          eval141_ade6_NON_DECIDING=k["ade6_arm"]["mean"], eval141_delta_NON_DECIDING=k["ade6_arm_minus_p10"])
            if len(per) == 2:
                ha = [per[f"seed{s}"]["holdout_ade6"] for s in (0, 1)]
                per["seed_mean_holdout_ade6"] = sum(ha) / 2
                per["seed_mean_holdout_minus_p10"] = sum(ha) / 2 - per["seed0"]["holdout_p10_ade6"]
                per["training_seed_floor_holdout_ade6"] = abs(ha[0] - ha[1])
            rows[arm][str(S)] = per
    if missing:
        return dict(schema="ra1-full1-v1", status="INCOMPLETE", missing=missing, budgets=rows)
    for name, s in (("holdout clip list", hold_sha), ("train clip list", train_sha), ("holdout P10 ADE@6s (= one prior)", p10_hold)):
        if len(s) != 1 or None in s:
            raise ValueError(f"the runs do not share ONE {name}: {sorted(map(str, s))}")
    if next(iter(hold_sha)) != HOLDOUT_SHA:
        raise ValueError(f"holdout clip list {next(iter(hold_sha))} is not the ES runs' Amendment-1 holdout {HOLDOUT_SHA}")
    star, edge = {}, {}
    for arm in ARMS:
        cand = {int(S): r["seed_mean_holdout_ade6"] for S, r in rows[arm].items()}
        m = min(cand.values())
        star[arm] = min(S for S, v in cand.items() if v <= m + TIE_M)
        edge[arm] = ("BELOW" if star[arm] == budgets[0] and len(budgets) > 1 else
                     "ABOVE" if star[arm] == budgets[-1] and len(budgets) > 1 else None)
    hr0, ha1 = rows["R0"][str(star["R0"])]["seed_mean_holdout_ade6"], rows["A1"][str(star["A1"])]["seed_mean_holdout_ade6"]
    k_arm = "A1" if ha1 < hr0 - TIE_M else "R0"
    ext = {arm: (star[arm] // 2 if e == "BELOW" else 2 * star[arm]) for arm, e in edge.items() if e}
    out = dict(schema="ra1-full1-v1", prereg="PREREG_RA1_FULL.md", status="OK", tier="T1-class open-loop trajectory",
               estimator="episode-cluster bootstrap per run (EPISODES); training-seed floor = |seed0 - seed1|",
               selection_basis="seed-mean ADE@6s on the 60-clip Amendment-1 holdout; eval-141 NEVER selects",
               tie_rule=f"budgets within {TIE_M} m of the minimum -> the shorter; arms within {TIE_M} m -> R0",
               holdout_clips_sha256=HOLDOUT_SHA, train_clips_sha256=next(iter(train_sha)), holdout_p10_ade6=next(iter(p10_hold)),
               S_star=star, edge=edge, extension_budgets=ext, K_arm=k_arm,
               memorisation_R0_holdout_minus_p10_by_budget={S: r["seed_mean_holdout_minus_p10"] for S, r in rows["R0"].items()},
               budgets=rows)
    if ext and not a.extended:
        out["reading"] = (f"S* on the grid edge for {sorted(ext)}: run the extension budget(s) {ext} under the same rules, then re-select "
                          "over the extended grid (PREREG_RA1_FULL §Selection). Phase 2 waits.")
        out["next"] = "EXTEND " + " ".join(f"{arm}:{S}" for arm, S in sorted(ext.items()))
    else:
        out["reading"] = (f"S*_A1 = {star['A1']}, S*_R0 = {star['R0']}; K arm = {k_arm}. Phase 2: A1x seed 0 at S*_A1, then the gate and K."
                          + (f" DISCLOSED: S* still on the grid edge after the one extension ({edge})." if ext else ""))
        out["next"] = f"full2 {star['A1']} {star['R0']} 0.08 50"
    return out


def read(a) -> dict:
    f1, v = _load(a.full1), _load(a.verdict)
    if f1.get("status") != "OK" or not str(f1.get("next", "")).startswith("full2"):
        raise ValueError(f"phase 1 is not ready for phase 2 (status {f1.get('status')}, next {f1.get('next')})")
    if v.get("kin_mode") != "yaw" or not v.get("deciding"):
        raise ValueError("the phase-2 verdict is not a deciding yaw-rate verdict")
    if v.get("A1_best_cell") != CELL:
        raise ValueError(f"phase-2 verdict cell {v.get('A1_best_cell')} != {CELL}")
    k_arm = f1["K_arm"]
    out = dict(schema="ra1-full-final-v1", prereg="PREREG_RA1_FULL.md", S_star=f1["S_star"], K_arm=k_arm, cell=CELL,
               instrument_ok_all=v.get("instrument_ok_all"), A1x_control=v.get("A1x_control"), K=v.get("K"),
               training_seed_floor_A1=v.get("training_seed_floor_A1"),
               not_read=dict(SENS="needs A0: not in this design", COST="not a PREREG_RA1_FULL reading",
                             branch_text_of_queue_verdict="assumes A0 runs; superseded by `reading` below"))
    gate = str((v.get("A1x_control") or {}).get("verdict", "")).startswith("OK")
    if not gate:
        out["outcome"] = "INSTRUMENT_BROKEN"
        out["reading"] = ("READING 3 FAILS at S*_A1 = %d on the full corpus: %s. K is NOT read. 8.4x the data did not make the time-shuffled "
                          "control lose to A1. Next lever: the A1a action-permutation control (PREREG_RA1_ES Addendum D) at full scale."
                          % (f1["S_star"]["A1"], (v.get("A1x_control") or {}).get("verdict")))
        return out
    cand = (v.get("K") or {}).get("candidates", {})
    if k_arm not in cand or not isinstance(cand[k_arm], dict) or "PASS" not in cand[k_arm]:
        raise ValueError(f"the phase-2 verdict has no K candidate for the holdout-chosen arm {k_arm}")
    kpass = bool(cand[k_arm]["PASS"])
    other = "A1" if k_arm == "R0" else "R0"
    out["K_read"] = dict(bar="K-arm - P10-yaw-4512 ADE@6s < 0, separated at both seeds, |delta| > 2x |A1_s0 - A1_s1|",
                         K_arm=k_arm, PASS=kpass, K_arm_row=cand[k_arm], other_arm_DESCRIPTIVE=cand.get(other))
    if kpass:
        out["outcome"] = "K_PASS"
        out["reading"] = (f"READING 4 PASS: the gate holds at S*_A1 = {f1['S_star']['A1']} and {k_arm} beats P10-yaw-4512 at both seeds. Data scale "
                          "is the lever. Next: A2 and SENS (with A0) at S* on the full corpus, under PREREG_HJ_STAGE0's other bars.")
    else:
        out["outcome"] = "K_FAIL"
        out["reading"] = (f"READING 4 FAILS: the gate holds, but {k_arm} does not clear K against P10-yaw-4512. Next lever: ES-5's wd 3.0 on the "
                          "full corpus (the one regulariser that separated A1 - P10 at 540 clips), then the unfrozen encoder (LoRA r 32).")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--root", required=True)
    s.add_argument("--budgets", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--extended", action="store_true", help="the grid already holds the one extension: never extend again")
    r = sub.add_parser("read")
    r.add_argument("--full1", required=True)
    r.add_argument("--verdict", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        res = select(a) if a.cmd == "select" else read(a)
    except (OSError, KeyError, ValueError) as e:
        print(f"FULL VERDICT INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res.get(k) for k in ("status", "S_star", "edge", "K_arm", "next", "outcome", "reading")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
