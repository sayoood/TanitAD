#!/usr/bin/env python3
"""P5-C read instrument B12: the TACTICAL family rows (PREREG_P5C.md section 8.2 table), via the EXISTING tools, nothing re-implemented.

Card, TACTICAL row: command adherence (turn-active accuracy headline, kappa, 3x3 confusion, random-proposal floor; `command_adherence.py`);
selection quality, random / actual / best-of-64 (`--dump-proposals`); computable as in P6's read. Tactical goal-setting is NOT computable (REFe has no tactical
goal head; NAVSIM carries no manoeuvre label beyond the driving command; n = 12,146, reason stated). STRATEGIC is PAUSED (PI ruling 2026-09-19, item 25).

Reuse (imported, not copied):
  eval/command_adherence.py   run_read(arm, comparators, dump, exp, expect_full)  -> accuracy / kappa / confusion per arm, paired deltas, the 64-proposal floor, its own gates
  eval/selection_quality.py   read_table(table.npz)                               -> random / actual / best-of-64 with paired log-cluster CIs
Both estimators are the registered ones (log-cluster bootstrap, 10,000 draws, seed 20260927, 95 %).

  python p5c_tactical.py --seam C1=SEAM.npz C0=SEAM.npz base=SEAM.npz [--dump C1=DUMP.npz] [--selq C1=TABLE.npz ...] [--tokens-from TOKENS.txt|plans.npz|subset.json]
                         [--subset-ok] --out tactical.json
  Without --subset-ok the full-navtest class counts of the P6 ruling are ENFORCED (12,146 tokens, 136 logs): a subset run is a REHEARSAL and says so.
  --tokens-from restricts the seams to those tokens (rehearsal on <= 200 tokens); it requires --subset-ok.
Output (JSON, one block per family, NEVER pooled):  tactical.command_adherence, tactical.selection_quality, tactical.goal_setting, strategic.
A family that cannot be computed carries `status: NOT COMPUTED` with the reason and n. A command_adherence gate failure makes `status: INVALID` (no numbers promoted).
`p5c_read.py --tactical tactical.json` embeds the block in the read JSON; `p5c_render.py` prints it (no tracker is inside these numbers: they read the plan).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, EVAL)

N_NAVTEST = 12146
GOAL_REASON = ("REFe has no tactical goal head, and NAVSIM carries no manoeuvre label beyond the driving command (card 8.2); "
               "the command is an INPUT of the agent, so adherence to it is the tactical row, not goal-setting")
SELQ_REASON = ("per-proposal PDMS needs the harness: 64 single-proposal seams scored per arm (~4 to 6.4 h CPU on the full navtest, P6 read gaps G2), "
               "built from the arm's --dump-proposals; no table was supplied")
STRATEGIC = {"status": "PAUSED", "reason": "PI ruling 2026-09-19 (PI_DECISION_QUEUE item 25); not a pass, not omitted"}
FAMILY_KEYS = ("command_adherence", "selection_quality", "goal_setting")


def _ca():
    import command_adherence as CA
    return CA


def _sq():
    import selection_quality as SQ
    return SQ


def tokens_from(path: str) -> list:
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        return sorted(str(t) for t in z["tokens"])
    if path.endswith(".json"):
        j = json.load(open(path, encoding="utf-8"))
        return sorted(j["tokens"] if isinstance(j, dict) else j)
    out = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            out.append(ln.split(":")[-1])
    return sorted(out)


def subset_seam(src: str, tokens, dst: str) -> str:
    z = np.load(src, allow_pickle=False)
    tok = [str(t) for t in z["token"]]
    pos = {t: i for i, t in enumerate(tok)}
    miss = [t for t in tokens if t not in pos]
    if miss:
        raise SystemExit(f"REFUSED: {src} lacks {len(miss)} requested tokens (first {miss[0]})")
    sel = np.array([pos[t] for t in tokens])
    np.savez(dst, **{k: (np.asarray(z[k])[sel] if np.asarray(z[k]).ndim >= 1 and len(z[k]) == len(tok) else np.asarray(z[k])) for k in z.files})
    return dst


def _cell(c: dict) -> dict:
    return {"n": c["n"], "accuracy": c["accuracy"], "kappa": c["kappa"], "confusion_command_rows_plan_cols": c["confusion_command_rows_plan_cols"],
            "rows": c["rows"], "cols": c["cols"], "per_command_recall": c["per_command_recall"], "per_command_n": c["per_command_n"]}


def compact_adherence(r1: dict, r2: dict | None, names: dict, n_tokens: int, registered: bool) -> dict:
    """r1 = run_read(C1 vs C0, base) result; r2 = run_read(C0 vs base) result (the C0 - base pair). -> the tactical.command_adherence block."""
    if "arms" not in r1:
        return {"status": "INVALID", "invalid": r1.get("INVALID", ["no result"]), "n": n_tokens}
    invalid = list(r1.get("INVALID") or []) + list((r2 or {}).get("INVALID") or [])
    gates = {k: {"ok": v.get("ok")} for k, v in r1["gates"].items()}
    for k, v in ((r2 or {}).get("gates") or {}).items():
        gates[f"{k}[C0 vs base]"] = {"ok": v.get("ok")}
    arms = {}
    for role, nm in names.items():
        a = r1["arms"].get(nm) or (r2 or {}).get("arms", {}).get(nm)
        if a is None:
            continue
        cl = a["executed_plan_A7"]
        arms[role] = {"turn_active": _cell(cl["turn_active"]), "all_tokens": _cell(cl["all_tokens"])}
    paired = {}
    for key, v in r1["paired_vs_comparators"].items():
        role = key.split("_minus_")[1].split("_")[0]
        paired[f"C1-{role}"] = {"turn_active": v["turn_active"], "all_tokens": v["all_tokens"]}
    if r2:
        for key, v in r2["paired_vs_comparators"].items():
            paired["C0-base"] = {"turn_active": v["turn_active"], "all_tokens": v["all_tokens"]}
    floor = r1["random_proposal_floor"]
    if "turn_active" in floor:
        fl = {"definition": floor["definition"], "dump": floor["dump"],
              "turn_active": {"floor_accuracy": floor["turn_active"]["floor"]["accuracy"], "floor_kappa": floor["turn_active"]["floor"]["kappa"],
                              "actual_minus_floor": floor["turn_active"]["actual_minus_floor"], "floor_ge_actual_point": floor["turn_active"]["floor_ge_actual_point"]},
              "all_tokens": {"floor_accuracy": floor["all_tokens"]["floor"]["accuracy"], "actual_minus_floor": floor["all_tokens"]["actual_minus_floor"]},
              "arm": "C1"}
    else:
        fl = {"status": "NOT COMPUTED", "reason": str(floor.get("status", "no dump")).replace("NOT COMPUTED: ", "", 1), "n": n_tokens}
    return {"status": "INVALID" if invalid else "COMPUTED", "invalid": invalid, "n_tokens": r1["tokens"]["n"], "n_logs": r1["tokens"]["n_logs"],
            "command_counts": r1["tokens"]["command_counts"], "n_turn_active": r1["tokens"]["turn_active_n"],
            "registered_read": bool(registered), "headline": "turn-active accuracy (LEFT u RIGHT commands)",
            "arms": arms, "paired_C1_minus_x": {k: v for k, v in paired.items()}, "random_proposal_floor": fl, "gates": gates,
            "controls": {"swap_LR_commands_drops": r1["controls"]["swap_LR_commands"]["drops"], "constant_straight_kappa_is_zero": r1["controls"]["constant_straight_plan"]["kappa_is_zero"]},
            "incomplete": list(r1.get("INCOMPLETE") or []),
            "estimator": r1["estimator"], "tier": r1["tier"], "authority": r1["authority"],
            "human_future_reference": (r1.get("reference_human_future") or {}).get("status")}


def selection_block(tables: dict, n_tokens: int) -> dict:
    out = {}
    for arm in ("C1", "C0", "base"):
        if arm in tables:
            r = _sq().read_table(tables[arm])
            out[arm] = {"status": "INVALID" if "INVALID" in r else "COMPUTED", "levels_x100": r["levels_x100"], "paired_differences_x100": r["paired_differences_x100"],
                        "random_control": r["random_control"], "n_tokens": r["n_tokens"], "n_logs": r["n_logs"], "table": r["table"], **({"invalid": r["INVALID"]} if "INVALID" in r else {})}
        else:
            out[arm] = {"status": "NOT COMPUTED", "reason": SELQ_REASON, "n": n_tokens}
    return out


def build(seams: dict, dump: dict | None = None, tables: dict | None = None, export: str | None = None, subset_ok: bool = False,
          tokens: list | None = None, workdir: str | None = None, n_boot: int | None = None, verify: bool = True) -> dict:
    """seams {C1, C0, base} -> the tactical block. Needs the labeller (repo stack/ + taniteval/) and the W3 export, as command_adherence does."""
    CA = _ca()
    for k in ("C1", "C0", "base"):
        if k not in seams:
            raise SystemExit(f"REFUSED: seam for {k} missing")
    if tokens is not None and not subset_ok:
        raise SystemExit("REFUSED: --tokens-from restricts the token set, which is a rehearsal: add --subset-ok")
    paths = dict(seams)
    if tokens is not None:
        os.makedirs(workdir, exist_ok=True)
        paths = {k: subset_seam(v, tokens, os.path.join(workdir, f"tactical_subset_{k}.npz")) for k, v in seams.items()}
    exp = CA.load_export(export) if export else CA.load_export()
    d = CA.load_dump(dump["C1"]) if dump and dump.get("C1") else None
    nb = n_boot or CA.N_BOOT
    r1 = CA.run_read({"name": "C1", "seam": paths["C1"]}, [{"role": "C0", "name": "C0", "seam": paths["C0"]}, {"role": "base", "name": "base", "seam": paths["base"]}],
                     dump=d, exp=exp, expect_full=not subset_ok, n_boot=nb, verify=verify)
    r2 = CA.run_read({"name": "C0", "seam": paths["C0"]}, [{"role": "base", "name": "base", "seam": paths["base"]}], dump=None, exp=exp,
                     expect_full=not subset_ok, n_boot=nb, verify=verify)
    ntok = len(tokens) if tokens is not None else (r1.get("tokens") or {}).get("n", N_NAVTEST)
    ca = compact_adherence(r1, r2, {"C1": "C1", "C0": "C0", "base": "base"}, ntok, registered=not subset_ok)
    return {"kind": "P5C_TACTICAL", "card": "PREREG_P5C.md section 8.2 (TACTICAL row)", "evidence": "MEASURED on the seams supplied (T1: the plan against the command, no tracker in these numbers)",
            "pooled": False, "families_never_pooled": "each block below stands alone; there is no composite",
            "command_adherence": ca, "selection_quality": selection_block(tables or {}, ntok),
            "goal_setting": {"status": "NOT COMPUTABLE", "reason": GOAL_REASON, "n": ntok if tokens is not None else N_NAVTEST},
            "strategic": STRATEGIC,
            "rehearsal": bool(subset_ok), "n_tokens": ntok}


def validate_block(t: dict) -> list:
    """what a consumer (p5c_read / p5c_render) needs: every family present, none pooled, each NOT-COMPUTED with a reason and n, STRATEGIC paused."""
    bad = []
    if not isinstance(t, dict) or t.get("kind") != "P5C_TACTICAL":
        return ["not a P5C_TACTICAL block"]
    for k in FAMILY_KEYS:
        if k not in t:
            bad.append(f"family {k} absent (never silently dropped)")
    if t.get("pooled") is not False or "composite" in t or "pooled_score" in t:
        bad.append("a pooled / composite score is present: families are never pooled")
    if (t.get("strategic") or {}).get("status") != "PAUSED":
        bad.append("STRATEGIC is not PAUSED (PI ruling 2026-09-19): it may not be a pass or absent")
    g = t.get("goal_setting") or {}
    if g.get("status") != "NOT COMPUTABLE" or not g.get("reason") or not g.get("n"):
        bad.append("goal_setting must say NOT COMPUTABLE with its reason and n")
    for arm, s in (t.get("selection_quality") or {}).items():
        if s.get("status") == "NOT COMPUTED" and (not s.get("reason") or not s.get("n")):
            bad.append(f"selection_quality[{arm}] NOT COMPUTED without reason / n")
    ca = t.get("command_adherence") or {}
    if ca.get("status") not in ("COMPUTED", "INVALID", "NOT COMPUTED"):
        bad.append("command_adherence has no status")
    if ca.get("status") == "NOT COMPUTED" and (not ca.get("reason") or not ca.get("n")):
        bad.append("command_adherence NOT COMPUTED without reason / n")
    return bad


def not_computed(reason: str, n: int) -> dict:
    """the block a read carries when B12 could not run at all (never a silent omission)."""
    return {"kind": "P5C_TACTICAL", "pooled": False, "command_adherence": {"status": "NOT COMPUTED", "reason": reason, "n": n},
            "selection_quality": {a: {"status": "NOT COMPUTED", "reason": reason, "n": n} for a in ("C1", "C0", "base")},
            "goal_setting": {"status": "NOT COMPUTABLE", "reason": GOAL_REASON, "n": n}, "strategic": STRATEGIC, "rehearsal": False, "n_tokens": n}


def _kv(items):
    out = {}
    for s in items or []:
        k, v = s.split("=", 1)
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seam", nargs="+", required=True, help="C1=seam.npz C0=seam.npz base=seam.npz")
    ap.add_argument("--dump", nargs="*", default=[], help="C1=dump.npz (--dump-proposals) for the random-proposal floor")
    ap.add_argument("--selq", nargs="*", default=[], help="ARM=table.npz from selection_quality.py table (the 64 harness columns)")
    ap.add_argument("--tokens-from", default=None)
    ap.add_argument("--subset-ok", action="store_true")
    ap.add_argument("--export", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--no-verify-boot", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    toks = tokens_from(a.tokens_from) if a.tokens_from else None
    res = build(_kv(a.seam), _kv(a.dump), _kv(a.selq), a.export, a.subset_ok, toks, a.workdir or os.path.dirname(os.path.abspath(a.out)), a.n_boot, not a.no_verify_boot)
    bad = validate_block(res)
    if bad:
        print("REFUSED:", bad, file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    ca = res["command_adherence"]
    print("ZZP5C_TACTICAL", ca["status"], "n", res["n_tokens"], "rehearsal" if res["rehearsal"] else "registered")
    return 0 if ca["status"] == "COMPUTED" else 1


if __name__ == "__main__":
    sys.exit(main())
