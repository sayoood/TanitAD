"""The arm-against-arm read of PREREG_T2_ATTRIB (frozen, sha256 f7d2cfa1...), "The statistic and the estimator": written 2026-11-07 BEFORE any stage-5 cell (MC or BPC) was read.

    python -I read_t2_pair.py <the read of record's own inputs, spelled as the pinned read takes them: --root2 --root3 --root4 --sources2 --sources3 --g1 --result1 --manifest1
                               --result2 --result3 --result4 --exclusions2 --exclusions3 --exclusions4>
                              --a-arm NAME [--a-stage N --a-root R --a-result STAGEN_RESULT.json --a-exclusions EXCLUSIONSN.json --a-overlay TAR::pkg/MANIFESTN.json --a-gate-log LOG]
                              --b-arm NAME [--b-stage N --b-root R --b-result ... --b-exclusions ... --b-overlay ... --b-gate-log ...]
                              --out-dir OUT [--identity-record DOMINANCE_T2.json]      (writes OUT/PAIR_T2.json and OUT/PAIR_T2.md; with --identity-record also OUT/pair_identity.json)

The pair is A - B, per seed, per each of the pinned dominance read's 19 rows, over the SAME starts, by the dominance card's estimator (paired scene-cluster bootstrap, B 2000, seed 0, through the pinned
reader). Arm names and their stage options are the pinned read's `--arm*` options, once for A (`--a-*`) and once for B (`--b-*`):
  * `P10` is the floor: stages 2 and 3, no seed (one cell read against both seeds of the other arm); it takes NO stage option.
  * `C0C` is stage 4 (`--root4`, `--result4`, `--exclusions4`); it takes no `--x-root` and the other `--x-*` inputs (the pinned read refuses them: ARM_STAGE4_EXTRA_INPUTS).
  * any other name is a stage beyond 4 (`--x-stage 5 --x-root <rollouts dir> ...`), laid out as stage 4 is. Two arms may share a stage NUMBER with different pulls (MC and BPC are both stage 5).
Seeded arms are paired seed against seed: A_s0 - B_s0 and A_s1 - B_s1. The floor is paired with both.

REUSE, NEVER RE-IMPLEMENT. The pinned `motion_c/t2v8f/code/read_t2_dominance.py` (pin 4, sha256 7fd65ffd...) is loaded BY FILE PATH after its sha256 is checked (a mismatch is a REFUSAL); its functions are
importable (`main()` is behind `__name__ == "__main__"`, line 1014), so nothing is copied: `g4_check`, `g1_check`, `g2_full`, `rscn_check`, `rscn_labels`, `g3_check`, `scenes_from`, `read_blocks`, `family_table`,
`seed_gap_table`, `arm_spec`, `arm_absent` are called UNCHANGED. Two thin pieces are this file's own, both listed here so they can be audited:
  1. `merge_rscn`: the pinned `rscn_check` reads ONE arm stage beyond 4. A pair of two stage-5 arms needs the UNION over stages 2, 3, 4, A's stage and B's stage; with one source it returns the pinned object
     itself (the identity case), with several it unions the pinned objects' `union` entries and recomputes `starts` / `scenes` with the pinned arithmetic (lines 686-691).
  2. `pair_arrays`: `arrays_for_seed` (line 821) hard-codes the floor P10 as the B side (its label `P10|T2|objects`, line 826). Where B is the floor and A is not, the pinned `arrays_for_seed` IS called
     unchanged; for every other pair `pair_arrays` repeats its checks (missing rows, windows that differ, v0 / g not identical) with the labels as parameters. The test file proves the two equal bit for bit on C0C - P10.

Scenes: the UNION of the R-SCN exclusions over EVERY stage read (2, 3, 4 and the stage of each beyond-4 arm named) leaves BOTH arms (rule 4 of AMENDMENT_T2_SCENE_REFUSAL). A pair gate is a REFUSAL: no row is
built when one fails (outcome GATE-FAIL). The seed floor of an arm is the dominance card's seed gap (`seed_gap_table`, section 7): |delta(X_s1 - P10) - delta(X_s0 - P10)| per row; P10 has none.
Tier: T2 re-perception, agents replayed at log time. The interval answers "another draw of SCENES" only (H-ESTIM-SEED-1); the seed floor is how the training-run term is read beside it. Nothing here decides:
the Read A block each seed carries is the pinned one, written for X - P10, and is kept beside as a table of the same terms for another pair, never as an outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types

sys.dont_write_bytecode = True                       # before the pinned module loads: the pinned folder gets no .pyc from this read
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
import numpy as np  # noqa: E402

DL = "E:/Projects/TanitAD-artifacts/dagland"
PIN4 = f"{DL}/motion_c/t2v8f/code/read_t2_dominance.py"
PIN4_SHA = "7fd65ffdf62ae74f5609e3ac259588826bb84b51e3a3a84968b299332a3f35f9"
ATTRIB = f"{DL}/motion_c/t2_prereg/PREREG_T2_ATTRIB.md"
ATTRIB_SHA = "f7d2cfa1dcef07bc4b8a90433325095822901677e8b1b37d7fc20d1770ae2895"
ATTRIB_FROZEN_BYTES = 4715                           # the card as frozen (sha256 ATTRIB_SHA); its log section is append-only, so later dated entries lengthen the file and never touch these bytes
FLOOR = "P10"
SEEDS = (0, 1)
SIDES = ("a", "b")
STAGE_FLAGS = ("stage", "root", "result", "exclusions", "overlay", "gate_log")
TIER = "T2 re-perception (agents replayed at log time; not a reactive-traffic result)"
ESTIMATOR = "paired scene-cluster bootstrap, B 2000, seed 0 (answers another draw of SCENES only; the seed floor is the training-run term, read beside)"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pin(path=PIN4, sha=PIN4_SHA, name="read_t2_dominance_pin4"):
    """Load the pinned dominance read BY FILE PATH, its sha256 checked FIRST. REFUSES on a mismatch (SystemExit), as `init()` does for the reader."""
    got = sha256_file(path)
    if got != sha:
        raise SystemExit(f"REFUSED: the pinned dominance read {path} is {got}, not the pinned {sha}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = load_pin()


def attrib_frozen_ok(path=ATTRIB):
    """The frozen part of PREREG_T2_ATTRIB is byte-identical: sha256 of its first ATTRIB_FROZEN_BYTES == ATTRIB_SHA (dated log entries below it are appended and do not move the check)."""
    if not os.path.isfile(path):
        return False
    b = open(path, "rb").read()
    return len(b) >= ATTRIB_FROZEN_BYTES and hashlib.sha256(b[:ATTRIB_FROZEN_BYTES]).hexdigest() == ATTRIB_SHA


# ------------------------------------------------------------------------------------------------ the arms
def arm_spec_of(a, side):
    """The arm on `side` ('a' / 'b') from the parsed args -> {arm, stage, root, result, exclusions, overlay, gate_log, floor}. The floor P10 takes no stage input; any other name is validated by the pinned
    `arm_spec` (its named refusals: ARM_NAME_FORM, ARM_NAME_SEED_SUFFIX, ARM_NAME_RESERVED, ARM_STAGE_BELOW_4, ARM_STAGE4_IS_C0C, ARM_STAGE4_EXTRA_INPUTS, ARM_ROOT_REQUIRED)."""
    name = getattr(a, f"{side}_arm", None)
    vals = {k: getattr(a, f"{side}_{k}", None) for k in STAGE_FLAGS}
    if name == FLOOR:
        extra = sorted(f"--{side}-" + k.replace("_", "-") for k, v in vals.items() if v is not None)
        if extra:
            raise SystemExit(f"REFUSED: FLOOR_EXTRA_INPUTS {extra}: {FLOOR} is read from stages 2 and 3 (--root2, --root3, --result2, --result3, --exclusions2, --exclusions3) and takes no stage input of its own")
        return {"arm": FLOOR, "stage": None, "root": None, "result": None, "exclusions": None, "overlay": None, "gate_log": None, "floor": True}
    ns = types.SimpleNamespace(arm=name, arm_stage=vals["stage"], arm_root=vals["root"], arm_result=vals["result"], arm_exclusions=vals["exclusions"], arm_overlay=vals["overlay"], arm_gate_log=vals["gate_log"])
    sp = D.arm_spec(ns)
    return {**sp, "floor": False}


def label_of(name, s):
    """The cell label the T2 row builder is handed: `P10|T2|objects` for the floor (as the pinned `arrays_for_seed`), `<arm>_s<seed>|T2|objects` for a seeded arm."""
    return f"{FLOOR}|T2|objects" if name == FLOOR else f"{name}_s{s}|T2|objects"


def sources_of(a, specs):
    """One read source per distinct non-floor arm: the stage dicts the pinned `rscn_check` / `g2_full` / `g3_check` take (stages 2, 3, 4 and, for an arm beyond 4, its own stage; two arms may share a stage number).
    With no non-floor arm the source is the pinned default (C0C at stage 4), which is the only way to load the floor's records."""
    base_roots = {2: a.root2, 3: a.root3, 4: a.root4}
    base_results = {2: a.result2 or f"{os.path.dirname(a.root2)}/STAGE2_RESULT.json", 3: a.result3 or f"{os.path.dirname(a.root3)}/STAGE3_RESULT.json", 4: a.result4 or f"{os.path.dirname(a.root4)}/STAGE4_RESULT.json"}
    base_flags = {2: getattr(a, "exclusions2", None), 3: getattr(a, "exclusions3", None), 4: getattr(a, "exclusions4", None)}
    out, seen = [], set()
    for sp in specs:
        if sp["floor"]:
            continue
        key = (sp["arm"], sp["stage"], sp["root"])
        if key in seen:
            continue
        seen.add(key)
        roots, results, flags = dict(base_roots), dict(base_results), dict(base_flags)
        if sp["stage"] > 4:
            roots[sp["stage"]], flags[sp["stage"]] = sp["root"], sp["exclusions"]
            results[sp["stage"]] = sp["result"] or f"{os.path.dirname(sp['root'])}/STAGE{sp['stage']}_RESULT.json"
        out.append({"spec": sp, "roots": roots, "results": results, "flags": flags})
    if not out:
        sp = {**D.arm_spec(types.SimpleNamespace(arm=D.ARM_DEFAULT, arm_stage=D.ARM_STAGE_DEFAULT)), "floor": False}
        out.append({"spec": sp, "roots": base_roots, "results": base_results, "flags": base_flags})
    return out


# ------------------------------------------------------------------------------------------------ R-SCN over several sources
def merge_rscn(parts):
    """parts = [(source label, arm_stage, rscn dict of the pinned `rscn_check`)] -> ONE rscn dict for g3_check. A single part is returned AS IT IS (the identity case). Several parts: the UNION of the
    excluded scenes over every stage any part read (a scene excluded in ANY stage leaves every arm), the `starts` / `scenes` blocks recomputed with the pinned arithmetic (read_t2_dominance.py lines 686-691).
    A stage 2, 3 or 4 is the same data in every part and is kept once; a stage beyond 4 is kept under `<stage>:<arm>`."""
    if len(parts) == 1:
        return parts[0][2]
    n_l, n_g = D.EXPECT["scenes_launch"], D.EXPECT["scenes_guard"]
    out = {"applied": True, "rule": parts[0][2]["rule"], "problems": [], "per_stage": {}, "union": {}, "starts": {}, "scenes": {}}
    for label, st, rs in parts:
        for p in rs["problems"]:
            if p not in out["problems"]:
                out["problems"].append(p)
        for k, info in rs["per_stage"].items():
            key = k if k in ("2", "3", "4") else f"{k}:{label}"
            out["per_stage"].setdefault(key, info)
        for sc, u in rs["union"].items():
            m = out["union"].setdefault(sc, {"scene": u["scene"], "idx": u["idx"], "stages": {}, "starts_lost": u["starts_lost"]})
            if m["starts_lost"] != u["starts_lost"]:
                out["problems"].append(f"RSCN_STARTS_DISAGREE scene {sc}: {m['starts_lost']} vs {u['starts_lost']} starts lost between the stage reads")
            for k, reason in u["stages"].items():
                m["stages"][k if k in ("2", "3", "4") else f"{k}:{label}"] = reason
    ex = D.EXPECT
    lost_l = sum(u["starts_lost"] or 0 for u in out["union"].values() if u["idx"] is not None and u["idx"] < n_l)
    lost_g = sum(u["starts_lost"] or 0 for u in out["union"].values() if u["idx"] is not None and u["idx"] >= n_l)
    out["starts"] = {"expected_before": {"launch": ex["starts_launch"], "guard": ex["starts_guard"], "total": ex["starts_launch"] + ex["starts_guard"]},
                     "lost_launch": lost_l, "lost_guard": lost_g, "lost_total": lost_l + lost_g,
                     "read_on": {"launch": ex["starts_launch"] - lost_l, "guard": ex["starts_guard"] - lost_g, "total": ex["starts_launch"] + ex["starts_guard"] - lost_l - lost_g}}
    out["scenes"] = {"of": n_l + n_g, "excluded": len(out["union"]), "read_on": n_l + n_g - len(out["union"])}
    return out


# ------------------------------------------------------------------------------------------------ rows
def pair_arrays(recs_a, recs_b, scenes, a_label, b_label):
    """The reader's array format {labels, eid, v0, g, a_i, b_i} of A against B over the SAME starts: the pinned `arrays_for_seed` with both labels as parameters (same refusals: a row missing, windows that
    differ, v0 / g not identical on the same starts). `delta` of every row built from it is mean(A - B)."""
    L = D.STATE["L"]
    GD = D.STATE["rr"].GD
    ra, ma, ia = L.rows_for_records(recs_a, scenes, a_label)
    rb, mb, ib = L.rows_for_records(recs_b, scenes, b_label)
    if ma or mb or any(lab not in ra or lab not in rb for lab in GD.LABELS):
        raise RuntimeError(f"ROWS_MISSING: {a_label} {ma}, {b_label} {mb}; the 19 rows must all be built")
    if not (np.array_equal(ia["eid"], ib["eid"]) and ia["keys"] == ib["keys"]):
        raise RuntimeError(f"WINDOWS_DIFFER between {a_label} and {b_label}")
    v0 = np.array([float(r["v0"]) for r in recs_a])
    g = np.stack([np.asarray(r["g"], np.float64) for r in recs_a])
    if not (np.array_equal(v0, np.array([float(r["v0"]) for r in recs_b])) and np.array_equal(g, np.stack([np.asarray(r["g"], np.float64) for r in recs_b]))):
        raise RuntimeError(f"WINDOWS_DIFFER: v0 / g of {a_label} and {b_label} are not identical on the same starts")
    z = {"labels": np.array(GD.LABELS), "eid": ia["eid"], "v0": v0, "g": g}
    for i, lab in enumerate(GD.LABELS):
        z[f"a_{i}"], z[f"b_{i}"] = np.asarray(ra[lab], np.float64), np.asarray(rb[lab], np.float64)
    return z


def build_arrays(a_name, b_name, recs_a, recs_b, scenes, s):
    """Seed `s` arrays of A - B. B the floor and A not: the PINNED `arrays_for_seed`, called as the pinned read calls it. Every other pair: `pair_arrays`."""
    if b_name == FLOOR and a_name != FLOOR:
        kw = {} if a_name == D.ARM_DEFAULT else {"arm_name": a_name}
        return D.arrays_for_seed(recs_a, recs_b, scenes, s, **kw)
    return pair_arrays(recs_a, recs_b, scenes, label_of(a_name, s), label_of(b_name, s))


def window_counts(z):
    """{row: {n_windows, n_scenes}}: the windows where both arms are finite and the scenes they sit in (the pool the interval resamples: 'masked' rows resample the scenes holding a valid pair, 'all' the
    scenes of the read) -- the reader's own rule (`gapdecomp.decompose`), counted here from the same arrays."""
    GD = D.STATE["rr"].GD
    out = {}
    for i, lab in enumerate(GD.LABELS):
        a, b = np.asarray(z[f"a_{i}"], np.float64), np.asarray(z[f"b_{i}"], np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        ep = np.asarray([str(x) for x in z["eid"]])
        out[lab] = {"n_windows": int(ok.sum()), "n_scenes": int(len(np.unique(ep[ok] if GD.POOL[lab] == "masked" else ep)))}
    return out


def read_pair(ctx, a_name, b_name, hooks=None):
    """{seed: array dict} -> the pinned `read_blocks` for the pair A - B, with `n_windows` / `n_scenes` added to every row. `hooks` (tests only): {'arrays': f(seed, z)} mutates a seed's arrays before they
    are read. -> {seed: {rows, read_a, classes, counts}}."""
    recs = ctx["recs"]
    arrays = {}
    for s in SEEDS:
        ra, rb = recs_of(recs, a_name, s), recs_of(recs, b_name, s)
        D.STATE["S"].pair_check(D.STATE["L"].pair_keys(ra), D.STATE["L"].pair_keys(rb))
        z = build_arrays(a_name, b_name, ra, rb, ctx["scenes"], s)
        if hooks and hooks.get("arrays"):
            hooks["arrays"](s, z)
        arrays[s] = z
    blocks = D.read_blocks(arrays)
    for s in SEEDS:
        blocks[s]["counts"] = window_counts(arrays[s])
    return blocks


def recs_of(recs, name, s):
    """The records of arm `name` at seed `s` (the floor has one set for both seeds)."""
    return recs[FLOOR] if name == FLOOR else recs[f"{name}_s{s}"]


def seed_floor(blocks):
    """The dominance card's seed gap of an arm against P10 (`seed_gap_table`, section 7): {row: {delta_s0, delta_s1, gap}}."""
    return {lab: {"delta_s0": v["delta_s0"], "delta_s1": v["delta_s1"], "gap": v["gap"]} for lab, v in D.seed_gap_table(blocks).items()}


# ------------------------------------------------------------------------------------------------ the context: gates, scenes, records
def gate_key(base, srcs, src):
    return base if len(srcs) == 1 else f"{base}[{src['spec']['arm']}@{src['spec']['stage']}]"


def build_context(a, hooks=None):
    """Everything up to the rows: the arms, the pinned gates G4, G1, G2, G3 (+ ARM) run for each source, R-SCN over the union of every stage read, the scene logs, the records of every arm.
    `hooks` (tests only): {'rscn': f(merged rscn dict)} mutates the merged R-SCN object before G3 reads it. -> ctx dict; ctx['ok'] False means NO ROW is built (GATE-FAIL)."""
    specs = {side: arm_spec_of(a, side) for side in SIDES}
    D.init()
    srcs = sources_of(a, list(specs.values()))
    ctx = {"specs": specs, "sources": srcs, "gates": {}, "recs": {}, "scenes": None, "rscn": None, "ok": False}
    G = ctx["gates"]
    card_ok = D.card_frozen_ok()
    G["G4"] = D.g4_check()
    G["G1"] = D.g1_check(a.g1)
    sids, tracks = {}, {}
    for f in (a.sources2, a.sources3):
        for e in json.load(open(f, encoding="utf-8")) if f else []:
            if e["rel"] == "extracted/sequence_tracks.json":
                tracks[int(e["idx"])] = e["source"]
            sids[int(e["idx"])] = e["scene"]
    complete = len(sids) == sum(D.EXPECT[k] for k in ("scenes_launch", "scenes_guard"))
    parts = []
    for src in srcs:
        sp = src["spec"]
        if not all(src["flags"].values()):
            G[gate_key("RSCN_FLAGS", srcs, src)] = {"PASS": False, "why": [f"RSCN_FLAGS_PARTIAL: no exclusions file for {sorted(k for k, v in src['flags'].items() if not v)}; the pair needs --exclusions2, --exclusions3, --exclusions4 and the exclusions of each stage beyond 4"]}
            continue
        if complete:
            parts.append((sp["arm"], sp["stage"], D.rscn_check(src["roots"], src["results"], sids, src["flags"], D.EXPECT, arm_name=sp["arm"], arm_stage=sp["stage"])))
    rs = merge_rscn(parts) if parts and len(parts) == len(srcs) else None
    if rs is not None and hooks and hooks.get("rscn"):
        hooks["rscn"](rs)
    ctx["rscn"] = rs
    for src in srcs:
        sp = src["spec"]
        ns = types.SimpleNamespace(root2=a.root2, root3=a.root3, root4=a.root4, result1=a.result1, manifest1=a.manifest1, arm=sp["arm"], arm_stage=sp["stage"], arm_root=sp["root"],
                                   arm_result=sp["result"], arm_exclusions=sp["exclusions"], arm_overlay=sp["overlay"], arm_gate_log=sp["gate_log"])
        G[gate_key("G2", srcs, src)] = D.g2_full(ns, src["results"], excluded=D.rscn_labels(rs))
        if rs is not None and complete:
            gate, recs = D.g3_check(src["roots"], src["results"], sids, D.EXPECT, arm_name=sp["arm"], arm_stage=sp["stage"], exclusions=src["flags"], rscn=rs)
        else:
            gate, recs = {"PASS": False, "why": [f"G3_SCENE_SOURCES {len(sids)} scenes, need {sum(D.EXPECT[k] for k in ('scenes_launch', 'scenes_guard'))}" if not complete else "R-SCN not built"]}, None
        G[gate_key("G3", srcs, src)] = gate
        absent = D.arm_absent(src["roots"][sp["stage"]], sp["arm"])
        if absent:
            G[gate_key("ARM", srcs, src)] = {"PASS": False, "why": absent}
        if recs is not None:
            for k, v in recs.items():
                if k == FLOOR and FLOOR in ctx["recs"]:
                    if D.STATE["L"].pair_keys(ctx["recs"][FLOOR]) != D.STATE["L"].pair_keys(v):
                        G["FLOOR"] = {"PASS": False, "why": ["FLOOR_RECORDS_DIFFER: the floor P10 read through two sources holds different (scene, f0) lists"]}
                else:
                    ctx["recs"][k] = v
    a_ok = attrib_frozen_ok()
    G["ATTRIB"] = {"PASS": a_ok, "why": [] if a_ok else ["the frozen part of PREREG_T2_ATTRIB (first 4715 bytes) differs from sha256 f7d2cfa1..."], "sha256_expected": ATTRIB_SHA, "frozen_bytes": ATTRIB_FROZEN_BYTES}
    if not card_ok:
        G["CARD"] = {"PASS": False, "why": ["the frozen part of the card (first 8497 bytes) differs from FROZEN_SHA_DOMINANCE.txt"]}
    ctx["card_ok"] = card_ok
    ctx["ok"] = all(g["PASS"] for g in G.values()) and card_ok and rs is not None
    if ctx["ok"]:
        by = {2: srcs[0]["roots"][2], 3: srcs[0]["roots"][3], 4: srcs[0]["roots"][4]}      # the order the pinned read hands `scenes_from`: stages 2, 3, 4, then the arm's stage
        for src in srcs:
            if src["spec"]["stage"] > 4:
                by[f"{src['spec']['stage']}:{src['spec']['arm']}"] = src["roots"][src["spec"]["stage"]]
        ctx["scenes"] = D.scenes_from(ctx["recs"], tracks, by, skip={u["idx"] for u in rs["union"].values()})
    return ctx


# ------------------------------------------------------------------------------------------------ the whole read
def rt(o):
    """A JSON round trip of the same floats (the pinned read's `rt`): equal after it means equal bit for bit."""
    return json.loads(json.dumps(o, default=D.STATE["rr"]._jd))


def run_pair(a, ctx=None, hooks=None):
    """The whole pair read. Never raises past a gate: a failed gate gives GATE-FAIL with no row read. -> result dict."""
    ctx = ctx or build_context(a, hooks)
    specs = ctx["specs"]
    A, B = specs["a"]["arm"], specs["b"]["arm"]
    res = {"card": "PREREG_T2_ATTRIB", "card_sha256_expected": ATTRIB_SHA, "script_sha256": sha256_file(os.path.abspath(__file__)), "pin4_path": PIN4, "pin4_sha256": PIN4_SHA, "pin4_sha256_now": sha256_file(PIN4),
           "dominance_card_sha_ok": ctx["card_ok"], "reader_sha256": D.RR_SHA, "tier": TIER, "estimator": ESTIMATOR, "pair": f"{A} - {B}", "null_pair": A == B,
           "arm_a": specs["a"], "arm_b": specs["b"], "argv": getattr(a, "argv", None), "gates": ctx["gates"], "r_scn": ctx["rscn"], "outcome": "GATE-FAIL" if not ctx["ok"] else "READ"}
    if not ctx["ok"]:
        return res
    blocks = read_pair(ctx, A, B, hooks)
    res["by_seed"] = {f"s{s}": {"classes": b["classes"], "rows": {lab: {**r, **b["counts"][lab]} for lab, r in b["rows"].items()}, "family_table": D.family_table(b["rows"]),
                                "read_a_beside": b["read_a"]} for s, b in blocks.items()}
    res["read_a_note"] = "Read A's bars are written for X - P10; for another pair it is a table of the same terms and decides nothing"
    vs, floors = {}, {}
    for name in dict.fromkeys(n for n in (A, B) if n != FLOOR):
        bl = blocks if (B == FLOOR and name == A) else read_pair(ctx, name, FLOOR, hooks)
        vs[name] = {f"s{s}": b["rows"] for s, b in bl.items()}
        floors[name] = seed_floor(bl)
    res["arm_vs_P10"] = vs
    res["seed_floor"] = {"a": floors.get(A), "b": floors.get(B), "definition": "the dominance card's seed gap: |delta(X_s1 - P10) - delta(X_s0 - P10)| per row (section 7); the floor P10 has none (one cell, both seeds)"}
    beside = {}
    for lab in res["by_seed"]["s0"]["rows"]:
        beside[lab] = {}
        for side, fl in (("a", floors.get(A)), ("b", floors.get(B))):
            if fl is not None:
                beside[lab][f"abs_delta_lt_floor_{side}"] = {f"s{s}": bool(abs(res["by_seed"][f"s{s}"]["rows"][lab]["delta"]) < fl[lab]["gap"]) for s in SEEDS}
    res["abs_delta_below_seed_floor_beside"] = beside
    return res


FIELDS = ("delta", "lo", "hi", "separated")


def identity_report(res, record_path):
    """With A = C0C and B = P10 the pair rows must be the read of record's 19 C0C - P10 rows at both seeds EXACTLY (a JSON round trip of the same floats). -> the dict `logs/pair_identity.json` holds."""
    rec = json.load(open(record_path, encoding="utf-8"))
    out = {"identical": False, "record": record_path, "record_sha256": sha256_file(record_path), "pair": res.get("pair"), "outcome": res.get("outcome"), "fields_compared": list(FIELDS), "per_row_max_abs_diff": {},
           "max_abs_diff": None, "why": []}
    if res.get("pair") != "C0C - P10":
        out["why"].append(f"IDENTITY_PAIR the pair read is {res.get('pair')!r}, the identity is C0C - P10")
    if "by_seed" not in res:
        out["why"].append(f"IDENTITY_NO_ROWS the pair read produced no rows (outcome {res.get('outcome')})")
        return out
    worst = 0.0
    same = True
    for s in SEEDS:
        mine, theirs = res["by_seed"][f"s{s}"]["rows"], rec["by_seed"][f"s{s}"]["rows"]
        if list(mine) != list(theirs):
            out["why"].append(f"IDENTITY_ROW_SET seed {s}: {list(mine)} vs {list(theirs)}")
            same = False
            continue
        for lab in theirs:
            d = 0.0
            for k in FIELDS:
                x, y = rt(mine[lab][k]), theirs[lab][k]
                if x != y:
                    same = False
                    out["why"].append(f"IDENTITY_DIFFERS seed {s} {lab} {k}: {x!r} vs {y!r}")
                if k != "separated":
                    d = max(d, abs(float(x) - float(y)))
            out["per_row_max_abs_diff"].setdefault(lab, {})[f"s{s}"] = d
            worst = max(worst, d)
        if rt(res["by_seed"][f"s{s}"]["classes"]) != rec["by_seed"][f"s{s}"]["classes"]:
            same = False
            out["why"].append(f"IDENTITY_CLASSES seed {s}")
        if rt(res["by_seed"][f"s{s}"]["family_table"]) != rec["by_seed"][f"s{s}"]["family_table"]:
            same = False
            out["why"].append(f"IDENTITY_FAMILY_TABLE seed {s}")
    rs, rr_ = res.get("r_scn") or {}, rec.get("r_scn") or {}
    out["scenes_and_starts_equal"] = bool(rt(rs.get("scenes")) == rr_.get("scenes") and rt(rs.get("starts")) == rr_.get("starts") and sorted(rs.get("union", {})) == sorted(rr_.get("union", {})))
    out["rows_compared"] = 2 * len(rec["by_seed"]["s0"]["rows"])
    out["max_abs_diff"] = worst
    out["identical"] = bool(same and out["scenes_and_starts_equal"] and not out["why"])
    return out


# ------------------------------------------------------------------------------------------------ output
def write_md(res, path):
    L = [f"# PAIR_T2: {res['pair']} ({res['outcome']})", "",
         f"Card PREREG_T2_ATTRIB (sha256 {res['card_sha256_expected'][:8]}...); script sha256 {res['script_sha256']}; pinned dominance read sha256 {res['pin4_sha256'][:8]}... (now {res['pin4_sha256_now'][:8]}...).",
         f"A = {res['arm_a']['arm']} (stage {res['arm_a']['stage']}), B = {res['arm_b']['arm']} (stage {res['arm_b']['stage']}). argv: `{' '.join(res.get('argv') or [])}`",
         f"Tier: {res['tier']}. Estimator: {res['estimator']}.", "", "## Gates", ""]
    for k, g in sorted(res["gates"].items()):
        L.append(f"* {k}: {'PASS' if g['PASS'] else 'FAIL'} {'; '.join(g.get('why', []))[:600]}")
    rsd = res.get("r_scn")
    if rsd and rsd.get("scenes"):
        L += ["", "## Excluded scenes (rule R-SCN, the union over every stage read)", "",
              f"* Scenes read on: {rsd['scenes']['read_on']} of {rsd['scenes']['of']} ({rsd['scenes']['excluded']} excluded). Starts read on: {rsd['starts']['read_on']['total']} of {rsd['starts']['expected_before']['total']} (lost {rsd['starts']['lost_total']}).", "",
              "| scene | idx | excluded in (stage: reason) | starts lost |", "|---|---|---|---|"]
        for sc, u in sorted(rsd["union"].items(), key=lambda kv: (kv[1]["idx"] is None, kv[1]["idx"])):
            L.append(f"| {sc} | {u['idx']} | " + ", ".join(f"{st}: {r}" for st, r in sorted(u["stages"].items())) + f" | {u['starts_lost']} |")
    if "by_seed" not in res:
        L += ["", "GATE-FAIL: no row was read."]
    else:
        fl = res["seed_floor"]
        for s in SEEDS:
            L += ["", f"## {res['pair']}, seed {s} (paired over the same starts, per row)", "",
                  "| family | row | A - B | lo | hi | separated | n windows | n scenes | seed floor A | seed floor B |", "|---|---|---|---|---|---|---|---|---|---|"]
            fam = {r["row"]: f for f, rows in res["by_seed"][f"s{s}"]["family_table"].items() for r in rows}
            for lab, r in res["by_seed"][f"s{s}"]["rows"].items():
                fa = f"{fl['a'][lab]['gap']:.4f}" if fl["a"] else "none"
                fb = f"{fl['b'][lab]['gap']:.4f}" if fl["b"] else "none"
                L.append(f"| {fam.get(lab)} | {lab} | {r['delta']:+.4f} | {r['lo']:+.4f} | {r['hi']:+.4f} | {'separated' if r['separated'] else 'not separated'} | {r['n_windows']} | {r['n_scenes']} | {fa} | {fb} |")
        L += ["", "Seed floor = the dominance card's seed gap of that arm against P10 (|delta_s1 - delta_s0|); `none` = the floor P10 has no seeds. The interval answers another draw of SCENES only; a separated row is necessary, not sufficient, for a lever claim (H-ESTIM-SEED-1)."]
    L += ["", "Disclosure: 62 scenes selected on the human's logs only, launch scenes over-represented (42 of 62); a result here is a result on these scenes at T2. A T2 pair is not a reactive-traffic result."]
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


def make_parser():
    ap = argparse.ArgumentParser()
    for k in ("root2", "root3", "root4", "sources2", "sources3", "g1", "out_dir"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    for k in ("result1", "result2", "result3", "result4", "manifest1"):
        ap.add_argument("--" + k, default=None)
    for k in ("exclusions2", "exclusions3", "exclusions4"):
        ap.add_argument("--" + k, default=None, help="EXCLUSIONS*.json of scene_exclusions.py for that stage (rule R-SCN: all of them, and each beyond-4 arm's own)")
    for side in SIDES:
        ap.add_argument(f"--{side}-arm", required=True, help=f"arm {side.upper()} of the pair A - B: P10 (the floor), C0C (stage 4) or the name of an arm in a stage beyond 4 (without its seed)")
        ap.add_argument(f"--{side}-stage", type=int, default=None, help="the stage that holds the arm's cells (default 4; a stage beyond 4 needs --%s-root)" % side)
        for k in ("root", "result", "exclusions", "overlay", "gate_log"):
            ap.add_argument(f"--{side}-{k.replace('_', '-')}", default=None, help="for an arm in a stage beyond 4: its rollouts dir / STAGE<N>_RESULT.json / EXCLUSIONS*.json / overlay '<tar>::pkg/MANIFEST<N>.json' / green pytest log")
    ap.add_argument("--identity-record", default=None, help="a DOMINANCE_T2.json of the read of record: with A = C0C and B = P10 write OUT/pair_identity.json comparing the 19 rows at both seeds")
    return ap


def main(argv=None):
    a = make_parser().parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    res = run_pair(a)
    os.makedirs(a.out_dir, exist_ok=True)
    rr = D.STATE["rr"]
    with open(f"{a.out_dir}/PAIR_T2.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=rr._jd)
    write_md(res, f"{a.out_dir}/PAIR_T2.md")
    rc = 0
    if a.identity_record:
        ident = identity_report(res, a.identity_record)
        with open(f"{a.out_dir}/pair_identity.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(ident, f, indent=1, default=rr._jd)
        print(json.dumps({"identical": ident["identical"], "max_abs_diff": ident["max_abs_diff"]}))
        rc = 0 if ident["identical"] else 3
    print(json.dumps({"pair": res["pair"], "outcome": res["outcome"], "gates": {k: g["PASS"] for k, g in res["gates"].items()}}))
    return rc if res["outcome"] == "READ" else 2


if __name__ == "__main__":
    sys.exit(main())
