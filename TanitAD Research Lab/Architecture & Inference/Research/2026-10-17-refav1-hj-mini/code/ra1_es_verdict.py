"""PREREG_RA1_ES verdict (../PREREG_RA1_ES.md).  Two modes, both pure JSON, no torch:

  select  --root <ra1_yaw_es> --banked <ra1_yaw> --budgets 250 500 1000 2000 --out es1.json
          ES-1: per budget S, R0's holdout delta (R0 - P10-yaw-540, ADE@6s, the 60-clip Amendment-1 holdout) at seeds 0 and 1.
          S* = the lowest seed-mean holdout delta; ties (within 0.01 m of the minimum) go to the SHORTER budget.  eval-141 numbers are
          reported per budget and decide NOTHING here.  Stop rule: no budget with a seed-mean holdout delta < 0 => NULL, ES-2 not run.
  read    --es1 es1.json --es2-verdict <st<S*>/es2_verdict.json> --out es_final.json
          ES-2: reading 2 (the A1x instrument gate) and, only if it passes, reading 3 (K on {R0, A1} at S*, PREREG_HJ_STAGE0's bar).

Exit code: 0 always when the inputs parse (the verdict is in the JSON); 2 on a missing / malformed input."""
from __future__ import annotations

import argparse
import json
import os
import sys

TIE_M = 0.01
CELL = "b0.08g50"


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _r0(dirpath, seed):
    p = os.path.join(dirpath, f"R0_s{seed}", "eval", "eval.json")
    if not os.path.exists(p):
        return None, p
    ev = _load(p)
    if ev.get("arm") != "R0" or ev.get("kin_mode") != "yaw":
        raise SystemExit(f"REFUSED: {p} is arm={ev.get('arm')} kin_mode={ev.get('kin_mode')}; ES reads compliant (yaw) R0 runs only")
    if not ev.get("instrument_ok"):
        raise SystemExit(f"REFUSED: {p} has instrument_ok=false")
    return ev, p


def select(a) -> dict:
    budgets = sorted(set(int(s) for s in a.budgets))
    rows, missing, hold_sha = {}, [], set()
    for S in budgets + [6000]:
        d = a.banked if S == 6000 else os.path.join(a.root, f"st{S}")
        per = {}
        for seed in (0, 1):
            ev, p = _r0(d, seed)
            if ev is None:
                missing.append(p)
                continue
            if S != 6000 and int(ev.get("ckpt_step", -1)) != S:
                raise SystemExit(f"REFUSED: {p} reports ckpt_step {ev.get('ckpt_step')} for budget {S}")
            h, k = ev["holdout"], ev["K"]
            hold_sha.add(h.get("clips_sha256"))
            per[f"seed{seed}"] = dict(path=p, ckpt_step=ev.get("ckpt_step"),
                                      holdout_ade6=h["ade6_arm"]["mean"], holdout_p10_ade6=h["ade6_p10"]["mean"],
                                      holdout_delta=h["ade6_arm_minus_p10"],
                                      eval141_ade6_NON_DECIDING=k["ade6_arm"]["mean"], eval141_delta_NON_DECIDING=k["ade6_arm_minus_p10"])
        if len(per) == 2:
            hd = [per[f"seed{s}"]["holdout_delta"]["delta"] for s in (0, 1)]
            ea = [per[f"seed{s}"]["eval141_ade6_NON_DECIDING"] for s in (0, 1)]
            per["seed_mean_holdout_delta"] = sum(hd) / 2
            per["holdout_below_p10_separated_both_seeds"] = all(
                per[f"seed{s}"]["holdout_delta"]["separated"] and per[f"seed{s}"]["holdout_delta"]["delta"] < 0 for s in (0, 1))
            per["training_seed_floor_holdout_delta"] = abs(hd[0] - hd[1])
            per["training_seed_floor_eval141_ade6_NON_DECIDING"] = abs(ea[0] - ea[1])
        rows[str(S)] = per
    if missing:
        return dict(schema="ra1-es1-v1", status="INCOMPLETE", missing=missing, budgets=rows)
    if len(hold_sha) != 1 or None in hold_sha:
        raise SystemExit(f"REFUSED: the runs were not scored on ONE holdout (clip-list sha256s {sorted(map(str, hold_sha))})")
    cand = {int(S): r["seed_mean_holdout_delta"] for S, r in rows.items()}
    m = min(cand.values())
    s_star = min(S for S, v in cand.items() if v <= m + TIE_M)
    null = cand[s_star] >= 0
    out = dict(schema="ra1-es1-v1", prereg="PREREG_RA1_ES.md", status="OK", tier="T1-class open-loop trajectory",
               estimator="paired episode-cluster bootstrap (EPISODES) per run; training-seed floor = |seed0 - seed1|",
               selection_basis="seed-mean holdout delta R0 - P10-yaw-540 (ADE@6s) on the 60-clip Amendment-1 holdout; eval-141 NEVER selects",
               tie_rule=f"budgets within {TIE_M} m of the minimum -> the shorter", holdout_clips_sha256=next(iter(hold_sha)),
               seed_mean_holdout_delta_by_budget=cand, S_star=s_star, S_star_holdout_delta=cand[s_star], budgets=rows)
    if null:
        out["reading"] = ("ES-1 NULL: no training budget gives R0 a seed-mean holdout delta below 0. Memorisation is real, but stopping early "
                          "does not turn it into skill at 540 clips. ES-2 NOT RUN (stop rule). Next lever: data scale (PREREG_HJ_STAGE0: "
                          "'600 TRAIN clips first, then all 4,572').")
        out["next"] = "NULL"
    else:
        out["reading"] = (f"ES-1: S* = {s_star} (seed-mean holdout delta {cand[s_star]:+.4f}). ES-2 runs at S*: A1 {CELL} x2 seeds + A1x seed 0.")
        out["next"] = f"es2 {s_star} 0.08 50"
    return out


def read(a) -> dict:
    e1, v = _load(a.es1), _load(a.es2_verdict)
    if e1.get("next", "NULL") == "NULL":
        raise SystemExit("REFUSED: ES-1 was NULL; ES-2 is not read (stop rule)")
    s_star = e1["S_star"]
    if v.get("kin_mode") != "yaw" or not v.get("deciding"):
        raise SystemExit("REFUSED: the ES-2 verdict is not a deciding yaw-rate verdict")
    if v.get("A1_best_cell") != CELL:
        raise SystemExit(f"REFUSED: ES-2 verdict cell {v.get('A1_best_cell')} != {CELL}")
    out = dict(schema="ra1-es-final-v1", prereg="PREREG_RA1_ES.md", S_star=s_star, cell=CELL, es1_reading=e1["reading"],
               instrument_ok_all=v.get("instrument_ok_all"), A1x_control=v.get("A1x_control"), K=v.get("K"),
               training_seed_floor_A1=v.get("training_seed_floor_A1"),
               not_read=dict(SENS="needs A0 at S*: not in this sweep", COST="not a PREREG_RA1_ES reading",
                             branch_text_of_es2_verdict="assumes A0 runs (it would print SENS FAIL); superseded by `reading` below"))
    gate = str((v.get("A1x_control") or {}).get("verdict", "")).startswith("OK")
    if not gate:
        out["reading"] = ("ES-2 INSTRUMENT STILL BROKEN at S* = %d: %s. K is NOT read. Memorisation alone does not explain A1x. Next lever: what "
                          "A1x's shuffled target leaves intact -- first, whether the world-model gradient moves the latent the readout reads."
                          % (s_star, (v.get("A1x_control") or {}).get("verdict")))
        out["outcome"] = "INSTRUMENT_BROKEN"
        return out
    K = v["K"]
    cands = {arm: K["candidates"][arm] for arm in ("R0", "A1") if arm in K["candidates"]}
    ranked = sorted((r["mean_ade6"], arm) for arm, r in cands.items() if r.get("mean_ade6") is not None)
    best = ranked[0][1] if ranked else None
    kpass = bool(best and cands[best]["PASS"])
    out["K_on_R0_A1"] = dict(bar="best(R0, A1) - P10 ADE@6s < 0, separated at both seeds, |delta| > 2x |A1_s0 - A1_s1| (A2 not in this sweep)",
                             best_arm=best, PASS=kpass, candidates=cands)
    if kpass:
        out["reading"] = (f"ES-2 INSTRUMENT RESTORED and K PASS on {{R0, A1}} at S* = {s_star} (best {best}): early stopping is the lever. Next: "
                          "A2 and SENS (with A0) at S*, under PREREG_HJ_STAGE0's other bars.")
        out["outcome"] = "K_PASS"
    else:
        out["reading"] = (f"ES-2 INSTRUMENT RESTORED, K FAIL on {{R0, A1}} at S* = {s_star}: memorisation broke the instrument, but the frozen "
                          "pooled readout still does not beat P10. SENS is unread here, so the LoRA branch is not taken. Next lever: data scale.")
        out["outcome"] = "K_FAIL"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--root", required=True)
    s.add_argument("--banked", required=True)
    s.add_argument("--budgets", nargs="+", required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("read")
    r.add_argument("--es1", required=True)
    r.add_argument("--es2-verdict", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        res = select(a) if a.cmd == "select" else read(a)
    except (OSError, KeyError, ValueError) as e:
        print(f"ES VERDICT INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res.get(k) for k in ("status", "S_star", "seed_mean_holdout_delta_by_budget", "next", "outcome", "reading")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
