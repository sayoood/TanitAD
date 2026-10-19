#!/usr/bin/env python3
"""L2-OFFPROP: the REAL run (arms P2, H x seeds {0,1}, H0 x seeds {0,1}, GLOBAL-null H-null x replicates 100-104 x seeds {0,1}) and the outcome.  SEALED-ONLY.

    python run_l2.py run  --prereg <SEALED prereg .md> --prereg-sha256 <hex> [--feats F] [--out-dir D] [--cells name,name,...] [--threads N] [--recipe R.json]
    python run_l2.py read --prereg <SEALED prereg .md> --prereg-sha256 <hex> [--out-dir D] [--recipe R.json]   (evaluate whatever cells exist, write L2_RESULT.json / L2_RESULT.md, map the outcome)

THE SEAL GUARD (tested RED in test_l2.py): the run refuses (exit 5) unless (1) the prereg file's sha256 equals --prereg-sha256, (2) the file does not carry the draft marker "Status: DRAFT",
(3) gates cfeat / cid / glive / reach / leak have all written a PASS to --gates-dir, and (4) each PASS was earned on THIS recipe file and THIS code (the stamp the gate wrote: recipe sha256 + code sha256).
Nothing here is reachable from the gate commands in gates.py, so a gate run can never train H, H0 or H-null on real labels.

Cells are resumable and independent: <out>/arms/<cell>.npz holds the arm's out-of-fold EP scores.  Cell names: P2, H_s{0,1}, H0_s{0,1}, Hnull-global_r{100..104}_s{0,1}  (15 cells).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2lib  # noqa: E402
import outcome  # noqa: E402

FEATS = "D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz"
OUT = "D:/Projects/TanitAD-artifacts/l2_offprop/run"
GATES = "C:/Users/Admin/par18/l2/out/gates"
RECIPE = "D:/Projects/TanitAD-artifacts/l2_offprop/recipe_chosen.json"
REQUIRED_GATES = ("cfeat", "cid", "glive", "reach", "leak")
RECIPE_BOUND_GATES = ("cid", "glive", "reach", "leak")        # C-FEAT depends on the exported features, not on the recipe
DRAFT_MARKER = "Status: DRAFT"
NULL_GROUPS = ("global", "token", "anchor")                   # 'global' is the sealed null (MM ruling E1); the other two are DIAGNOSTICS


def null_cell_name(r, s, group=l2lib.NULL_GROUP_DEFAULT):
    return f"Hnull-{group}_r{r}_s{s}"


def all_cells(null_groups=(l2lib.NULL_GROUP_DEFAULT,)):
    """P2, H x seeds, H0 x seeds, and H-null x replicates x seeds for each requested permutation group.  'global' is the sealed null (MM ruling E1); 'token' and 'anchor' (the withdrawn draft definition)
    exist as diagnostics and never feed the outcome unless --null-group selects them."""
    cells = {"P2": l2lib.ArmSpec("P2", "P2")}
    for s in l2lib.TRAIN_SEEDS:
        cells[f"H_s{s}"] = l2lib.ArmSpec(f"H_s{s}", "H", seed=s)
    for s in l2lib.TRAIN_SEEDS:
        cells[f"H0_s{s}"] = l2lib.ArmSpec(f"H0_s{s}", "H0", seed=s)
    for g in null_groups:
        for s in l2lib.TRAIN_SEEDS:
            for r in l2lib.NULL_REPS:
                cells[null_cell_name(r, s, g)] = l2lib.ArmSpec(null_cell_name(r, s, g), "Hnull", seed=s, null_rep=r, null_group=g)
    return cells


def check_seal(prereg: str, sha_given: str, gates_dir: str, recipe_path: str | None = None) -> dict:
    """-> info dict, or raises SystemExit(5) with the reason."""
    if not prereg or not os.path.exists(prereg):
        raise SystemExit(f"SEAL REFUSED (5): prereg file not found: {prereg!r}")
    sha = l2lib.sha256_file(prereg)
    if not sha_given or sha.lower() != str(sha_given).lower():
        raise SystemExit(f"SEAL REFUSED (5): sha256 of {prereg} is {sha}, not the {sha_given!r} you passed")
    text = open(prereg, encoding="utf-8").read()
    if DRAFT_MARKER in text:
        raise SystemExit(f"SEAL REFUSED (5): {prereg} still carries {DRAFT_MARKER!r}; the Master Mind seals it (removes that line and records the sha256) BEFORE any L2 arm is trained")
    missing, stale = [], []
    now = l2lib.gate_stamp(recipe_path)
    if not now["recipe_sha256"]:
        raise SystemExit(f"SEAL REFUSED (5): recipe file {recipe_path!r} not found: the recipe is a sealed DATA file")
    for g in REQUIRED_GATES:
        p = os.path.join(gates_dir, g + ".json")
        if not os.path.exists(p):
            missing.append(g)
            continue
        j = json.load(open(p))
        if not j.get("pass"):
            missing.append(g)
            continue
        if j.get("code_sha256") != now["code_sha256"] or (g in RECIPE_BOUND_GATES and j.get("recipe_sha256") != now["recipe_sha256"]):
            stale.append(g)
    if missing:
        raise SystemExit(f"SEAL REFUSED (5): gate(s) without a PASS in {gates_dir}: {missing}")
    if stale:
        raise SystemExit(f"SEAL REFUSED (5): gate(s) {stale} passed on a different recipe file or different code than this run's: re-run them (gates.py <gate>)")
    return {"prereg": prereg, "prereg_sha256": sha, "recipe_sha256": now["recipe_sha256"]}


def outcome_from_ev(ev: dict, cfeat: bool, null_group: str, null_pool: str, void_scope: str) -> dict:
    """The mapping's inputs from the per-cell reads, ONE place (tested): A1 = the POOLED quantity (MM ruling E2); in-fan(H) is compared with in-fan(H0) AT THE SAME SEED, P2's in-fan is reported only."""
    p2 = l2lib.summarise(ev["P2"])
    reads = {}
    for s in l2lib.TRAIN_SEEDS:
        h, h0 = l2lib.summarise(ev[f"H_s{s}"]), l2lib.summarise(ev[f"H0_s{s}"])
        reads[s] = {"a1_point": h["a1_point"], "a1_lo": h["a1_lo"], "a4_point": h["a4_point"], "a5_point": h["a5_point"], "infan": h["infan"], "infan_h0": h0["infan"], "infan_p2": p2["infan"]}
    nulls = {s: [l2lib.summarise(ev[null_cell_name(r, s, null_group)])["a1_point"] for r in l2lib.NULL_REPS] for s in l2lib.TRAIN_SEEDS}
    res = outcome.l2_outcome(cfeat, p2["a1_point"], reads, nulls, null_pool, void_scope)
    res["next_arm"] = outcome.NEXT_ARM[res["outcome"]]
    res["diagnostic_null_group"] = null_group != l2lib.NULL_GROUP_DEFAULT
    res["inputs"] = {"p2_a1": p2["a1_point"], "reads": reads, "null_points": nulls, "null_pool": null_pool, "void_scope": void_scope, "null_group": null_group, "cfeat_pass": cfeat,
                     "A1_is": "POOLED lon+lat (MM ruling E2); lon / lat reported beside it, ungated",
                     "a1_lon_lat": {f"H_s{s}": {"lon": l2lib.summarise(ev[f"H_s{s}"])["a1_lon"], "lat": l2lib.summarise(ev[f"H_s{s}"])["a1_lat"]} for s in l2lib.TRAIN_SEEDS}}
    return res


def cmd_run(a):
    seal = check_seal(a.prereg, a.prereg_sha256, a.gates_dir, a.recipe)
    os.makedirs(os.path.join(a.out_dir, "arms"), exist_ok=True)
    import torch
    torch.set_num_threads(a.threads)
    D = l2lib.load_real(a.feats)
    rec = l2lib.Recipe.from_json(a.recipe)
    code = {fn: l2lib.sha256_file(os.path.join(HERE, fn)) for fn in ("l2lib.py", "outcome.py", "run_l2.py", "synthetic.py", "gates.py")}
    code["stepA.py"] = l2lib.sha256_file(os.path.join(l2lib.L1_CODE, "stepA.py"))
    json.dump({"recipe": rec.as_dict(), "seal": seal, "code_sha256": code, "feats_sha256": l2lib.sha256_file(a.feats), "argv": sys.argv[1:], "null_group": a.null_group},
              open(os.path.join(a.out_dir, "run_config.json"), "w"), indent=1)
    cells = all_cells(NULL_GROUPS)
    want = [c for c in (a.cells.split(",") if a.cells else list(all_cells((a.null_group,))))]
    for c in want:
        if c not in cells:
            raise SystemExit(f"unknown cell {c!r}")
    for c in want:
        p = os.path.join(a.out_dir, "arms", c + ".npz")
        if os.path.exists(p):
            print("ZZSKIP", c, flush=True)
            continue
        o = l2lib.oof_scores(D, cells[c], rec)
        tmp = p + ".tmp.npz"
        np.savez(tmp, ep_in=o["ep_in"], ep_p=o["ep_p"], diag=np.array(json.dumps(l2lib.jsonable(o["diag"]))))
        os.replace(tmp, p)
        print(f"ZZCELL {c} {o['seconds']:.1f}s", flush=True)
    print("ZZRUN_DONE", flush=True)
    return 0


def cmd_read(a):
    check_seal(a.prereg, a.prereg_sha256, a.gates_dir, a.recipe)
    D = l2lib.load_real(a.feats)
    cells = all_cells(NULL_GROUPS)
    outcome_cells = all_cells((a.null_group,))
    ev, have = {}, {}
    for c in cells:
        p = os.path.join(a.out_dir, "arms", c + ".npz")
        if os.path.exists(p):
            have[c] = np.load(p, allow_pickle=False)
    if "P2" not in have:
        raise SystemExit("cell P2 missing")
    ev["P2"] = l2lib.evaluate_arm(D, have["P2"]["ep_in"], have["P2"]["ep_p"])
    ref = l2lib.ref_terms_of(ev["P2"])
    for c in have:
        if c != "P2":
            ev[c] = l2lib.evaluate_arm(D, have[c]["ep_in"], have[c]["ep_p"], ref)
    cfeat = json.load(open(os.path.join(a.gates_dir, "cfeat.json")))["pass"]
    missing = [c for c in outcome_cells if c not in ev]
    res = {"stamp": "SEEN-SCENE, SELECTION tier; ONE training seed per cell; estimator = paired log-cluster bootstrap, 24 logs, 10,000, seed 20260927 (another draw of LOGS only; H-ESTIM-SEED-1)",
           "cells_present": sorted(ev), "cells_missing": missing}
    if missing:
        res["outcome"] = {"outcome": "INCOMPLETE", "why": f"{len(missing)} cells missing: no call", "per_seed": {}}
    else:
        res["outcome"] = outcome_from_ev(ev, cfeat, a.null_group, a.null_pool, a.void_scope)
    res["arms"] = {c: {k: v for k, v in e.items() if k != "_per_log"} for c, e in ev.items()}
    json.dump(l2lib.jsonable(res), open(os.path.join(a.out_dir, "L2_RESULT.json"), "w", encoding="utf-8"), indent=1)
    oc = res["outcome"]
    diag = " (DIAGNOSTIC: null group is not the sealed 'global' null: NOT the sealed outcome)" if oc.get("diagnostic_null_group") else ""
    lines = ["# L2-OFFPROP result (SEEN-SCENE, SELECTION tier; ONE training seed per cell; paired log-cluster bootstrap over 24 logs, 10,000, seed 20260927)", "",
             f"**Outcome: {oc['outcome']}**{diag} - {oc['why']}", "",
             "A1 = the POOLED lon + lat quantity Step A's gate used (MM ruling E2); lon and lat are reported beside it and gate nothing.", "",
             "| cell | A1 all | A1 lon | A1 lat | A2 all | A3 all | A4 all | A5 all | in-fan top-8 EP conc | A1 minus P2 (paired) |", "|---|---|---|---|---|---|---|---|---|---|"]
    f3 = lambda d: f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    for c, e in sorted(ev.items()):
        d = e.get("A1_minus_P2_paired")
        lines.append(f"| {c} | {f3(e['all']['A1'])} | {f3(e['lon']['A1'])} | {f3(e['lat']['A1'])} | {f3(e['all']['A2'])} | {f3(e['all']['A3'])} | {f3(e['all']['A4'])} | {f3(e['all']['A5'])} | "
                     f"{f3(e['infan_top8_ep_concordance'])} | {f3(d) if d else '-'} |")
    if "inputs" in oc:
        inp = oc["inputs"]
        lines += ["", "In-fan non-degradation (GATED: in-fan(H) >= in-fan(H0) - 0.02 at the same seed; P2's in-fan is REPORTED, never gated):", "",
                  "| seed | in-fan(H) | in-fan(H0) | H minus H0 | gate | P2 in-fan (reported) |", "|---|---|---|---|---|---|"]
        for s, r in sorted(inp["reads"].items()):
            lines.append(f"| {s} | {r['infan']:.4f} | {r['infan_h0']:.4f} | {r['infan'] - r['infan_h0']:+.4f} | {'ok' if oc['per_seed'][s]['infan_ok'] else 'FAIL'} | {r['infan_p2']:.4f} |")
    lines += ["", "TACTICAL: command-class change share is unchanged by a head retrain (reported per arm in L2_RESULT.json). STRATEGIC: PAUSED (PI ruling 2026-09-19, ITEM 25), reported as PAUSED, not as a pass."]
    open(os.path.join(a.out_dir, "L2_RESULT.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("ZZREAD_OK", oc["outcome"], flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("read", cmd_read)):
        p = sp.add_parser(name)
        p.add_argument("--prereg", required=False)
        p.add_argument("--prereg-sha256", required=False)
        p.add_argument("--gates-dir", default=GATES)
        p.add_argument("--feats", default=FEATS)
        p.add_argument("--out-dir", default=OUT)
        p.add_argument("--recipe", default=RECIPE, help="the recipe fixed in RECIPE_SELECTION.md (a JSON data file; a real run never takes the source default)")
        p.add_argument("--null-pool", choices=("all", "matched_seed"), default="all")
        p.add_argument("--void-scope", choices=("any_seed", "pooled"), default="any_seed")
        p.add_argument("--null-group", choices=NULL_GROUPS, default=l2lib.NULL_GROUP_DEFAULT, help="permutation group of H-null: 'global' = the sealed null (MM ruling E1); 'token' / 'anchor' are diagnostics")
        if name == "run":
            p.add_argument("--cells", default="")
            p.add_argument("--threads", type=int, default=4)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
