"""obd_read_seed3.py -- the third-seed BAND read of AMEND_OBD_SEED3.md (sha256 a8a5848f22a3f164e4069bd7b302e7201cb05de5995494d05cf92dfde426e314), CPU, zero training.

Reuses ../read/obd_read.py (imported, NOT edited): its 24 cells, floor_of / floors_from, compare_cells (paired episode-cluster bootstrap B 2,000 seed 0), label_band, named,
check_manifest / check_selfcheck, the dump loaders and the geometry.  Floor G is |OBD-BAND_rep - OBD-BAND_s0| per cell, computed exactly as the existing read computes it and
cross-checked cell by cell against floors.G of the existing RESULT_OBD.json.
Each new pair gets a per-seed label by the card's rules (PREREG_V8F_OBD.md lines 35-47, via obd_read.label_band):
  BAND_s10 against CTL_s10 (init M2b s0) and BAND_s11 against CTL_s11 (init M2b s1).
The s0 and s1 labels are read from ../read/RESULT_OBD.json (sha256 asserted and recorded).

COMBINATION RULE (AMEND_OBD_SEED3.md lines 21-24, verbatim): four per-seed labels (s0, s1, s10, s11).  The band outcome is the label held by at least 3 of the 4,
with at least one of them at each init (M2b s0 and M2b s1).  Otherwise BAND-UNSETTLED.  A label that could not be read (no floor, arm missing) makes the outcome NOT-YET.
TRIGGER (lines 26-31): BAND-ADDS -> OBD-BAND's argv is the O/T recipe of record; BAND-NEUTRAL / BAND-COSTS -> OBD-CTL's argv, the band loss leaves the recipe, its operative gain
recorded as not carried; BAND-UNSETTLED -> OBD-CTL's argv (conservative branch), the band's operative-only gain stays recorded beside, with the code-drift caveat (VERDICT_OBD.md section 3).
STRATEGIC: PAUSED (PI item 25, PI_DECISION_QUEUE.md, 2026-09-19): reported as paused with this ruling, never as a pass, never omitted.
Usage: python obd_read_seed3.py --out RESULT_OBD_SEED3.json --md RESULT_OBD_SEED3.md      (--smoke --map smoke_map_seed3.json for the stand-in code check)
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
READ = os.path.join(os.path.dirname(HERE), "read")
sys.path.insert(0, HERE)
sys.path.insert(0, READ)
import numpy as np  # noqa: E402
import obd_read as OR  # noqa: E402
import seed3_common as C  # noqa: E402

AMEND_SHA256 = C.AMEND_SHA256
EXISTING_RESULT = os.path.join(READ, "RESULT_OBD.json")
EXISTING_SHA256 = "e05e1f5f348f1cb5d632e414784e591cf7ea943eafa7ccb7fcfc741fcdba033b"
LABELS = ("BAND-ADDS", "BAND-NEUTRAL", "BAND-COSTS")
UNSETTLED = "BAND-UNSETTLED"
SEEDS = ("s0", "s1", "s10", "s11")
TRIGGER = {
    "BAND-ADDS": "OBD-BAND's argv becomes the O/T recipe of record.",
    "BAND-NEUTRAL": "OBD-CTL's argv becomes the recipe of record. The band loss leaves the recipe, and its operative gain is recorded as not carried.",
    "BAND-COSTS": "OBD-CTL's argv becomes the recipe of record. The band loss leaves the recipe, and its operative gain is recorded as not carried.",
    UNSETTLED: "OBD-CTL's argv becomes the recipe of record, the conservative branch: four training runs could not show the band loss consistently. The band's operative-only gain stays "
               "recorded beside, with the code-drift caveat (VERDICT_OBD.md section 3).",
}
RECIPE_OF = {"BAND-ADDS": "OBD-BAND", "BAND-NEUTRAL": "OBD-CTL", "BAND-COSTS": "OBD-CTL", UNSETTLED: "OBD-CTL"}


# ----------------------------------------------------------------------------------------------------------------------------------------- the combination rule (pure)
def combine4(items):
    """items: list of (label or None, init) -- the four per-seed labels with the init (0 or 1) each was trained from.
    -> outcome.  The label held by at least 3 of the 4 with at least one of them at each init, else BAND-UNSETTLED; an unread label gives NOT-YET."""
    if len(items) != 4:
        raise ValueError("the amendment fixes four per-seed labels")
    if any(l is None for l, _ in items):
        return "NOT-YET: " + ", ".join(f"{l}@init{i}" for l, i in items) + " (a per-seed label is unread; the rule needs all four)"
    for lab in LABELS:
        held = [i for l, i in items if l == lab]
        if len(held) < 3:
            continue
        each_init = (0 in held) and (1 in held)       # INITCLAUSE  (the mutation check flips this line)
        return lab if each_init else UNSETTLED
    return UNSETTLED


def combine_seeds(labels):
    """{'s0':..,'s1':..,'s10':..,'s11':..} -> outcome, with the inits of C.SEED_LABELS."""
    return combine4([(labels.get(s), C.SEED_LABELS[s]) for s in SEEDS])


# ----------------------------------------------------------------------------------------------------------------------------------------- the read on arrays (pure)
def read_pair(CI, ctl, band, eid, floors, nb=OR.NB):
    """BAND against CTL at one new seed -> the same record shape as obd_read.read_question's per-seed entry."""
    cells = OR.compare_cells(CI, band, ctl, eid, floors, nb)
    nofl = [c for c, v in cells.items() if v.get("status") == "OK" and v.get("floor") is None]
    ok = floors is not None and not all(v.get("status") != "OK" or v.get("floor") is None for v in cells.values())
    return {"status": "READ", "reference": "CTL of the same seed", "cells": cells, "named_costs": OR.named(cells, "cost"), "named_gains": OR.named(cells, "gain"),
            "n_cells_read": sum(1 for v in cells.values() if v.get("status") == "OK"), "cells_without_floor": nofl, "label": OR.label_band(cells) if ok else None}


def spread_rows(cells_by_seed):
    """{seed: {cell: cell record}} -> {cell: {seed: delta, 'min','max','range','n'}} (beside, no bar)."""
    out = {}
    for c in OR.CELLS:
        row = {}
        for s in SEEDS:
            v = (cells_by_seed.get(s) or {}).get(c)
            row[s] = v["delta"] if v and v.get("status") == "OK" else None
        vals = [x for x in row.values() if x is not None]
        row.update({"min": min(vals) if vals else None, "max": max(vals) if vals else None, "range": (max(vals) - min(vals)) if vals else None, "n": len(vals)})
        out[c] = row
    return out


def drift_sizes(existing):
    """the code-drift row (NULL minus DAG-OACC) from the existing result, for scale: per seed the largest |delta| per family and the cell it sits on."""
    cd = ((existing.get("beside") or {}).get("code_drift_NULL_minus_DAGOACC")) or {}
    out = {}
    for s, cells in cd.items():
        fam = {}
        for c, v in cells.items():
            if v.get("status") != "OK":
                continue
            f = OR.FAMILY[c.split(":")[-1]]
            if f not in fam or abs(v["delta"]) > abs(fam[f]["delta"]):
                fam[f] = {"cell": c, "delta": v["delta"], "lo": v["lo"], "hi": v["hi"], "separated": v["separated"]}
        out[s] = {"largest_by_family": fam, "n_cells_separated": sum(1 for v in cells.values() if v.get("separated")), "ADE_operative": cells.get(OR.OP_ADE)}
    return out


def read_seed3(CI, arms, eids, existing, nb=OR.NB):
    """arms: {tag: {cell: [N]}} with OBD-BAND_rep, OBD-BAND_s0 (floor G) and OBD-CTL/BAND_s10, _s11.  eids: {0: eid, 1: eid} by init.
    existing: {'labels': {'s0','s1'}, 'cells': {'s0': {cell: rec}, 's1': ...}, 'floor_G': {cell: float} or None, 'result': the existing result dict (optional)}."""
    G = OR.floors_from(arms["OBD-BAND_rep"], arms["OBD-BAND_s0"]) if "OBD-BAND_rep" in arms and "OBD-BAND_s0" in arms else None
    per = {}
    for tag in ("s10", "s11"):
        ctl, band = arms.get(f"OBD-CTL_{tag}"), arms.get(f"OBD-BAND_{tag}")
        if ctl is None or band is None:
            per[tag] = {"status": "NOT-YET", "label": None}
        else:
            per[tag] = read_pair(CI, ctl, band, eids[C.SEED_LABELS[tag]], G, nb)
    per["s0"] = {"status": "READ (existing RESULT_OBD.json)", "label": existing["labels"].get("s0"), "cells": existing["cells"].get("s0")}
    per["s1"] = {"status": "READ (existing RESULT_OBD.json)", "label": existing["labels"].get("s1"), "cells": existing["cells"].get("s1")}
    labels = {s: per[s].get("label") for s in SEEDS}
    outcome = combine_seeds(labels)
    base = outcome if outcome in RECIPE_OF else None
    rep = {"floor_G": G, "per_seed": per, "labels": labels, "labels_by_init": {s: C.SEED_LABELS[s] for s in SEEDS}, "outcome": outcome,
           "recipe_of_record": RECIPE_OF.get(base) if base else None, "trigger": TRIGGER.get(base) if base else
           "no trigger: the outcome is not read (NOT-YET) or the read is INVALID",
           "beside": {"four_labels": labels, "per_cell_spread_BAND_minus_CTL": spread_rows({s: per[s].get("cells") for s in SEEDS}),
                      "code_drift_sizes": drift_sizes(existing.get("result") or {})}}
    if base == UNSETTLED:
        rep["beside"]["operative_only_gain_note"] = ("the band's operative-only gain stays recorded beside, with the code-drift caveat (VERDICT_OBD.md section 3); "
                                                     "operative ADE delta per seed: " + ", ".join(
                                                         f"{s} {(((per[s].get('cells') or {}).get(OR.OP_ADE)) or {}).get('delta')}" for s in SEEDS))
    return rep


# ----------------------------------------------------------------------------------------------------------------------------------------- the existing read
def load_existing(path=EXISTING_RESULT, want_sha=EXISTING_SHA256):
    """-> (existing dict for read_seed3, sha256).  Asserts the file's sha256; refuses a smoke or INVALID result."""
    sha = OR.sha_file(path)
    assert sha == want_sha, f"{path}: sha256 {sha} is not the recorded {want_sha}"
    r = json.load(open(path, encoding="utf-8"))
    assert r.get("smoke") is False, "the existing result is a smoke run"
    assert not str(r.get("outcome", "")).startswith("INVALID"), "the existing result is INVALID"
    ps = r["band"]["per_seed"]
    get = lambda s: ps.get(str(s)) or ps.get(s)  # noqa: E731
    for s in (0, 1):
        assert get(s)["status"] == "READ" and get(s)["label"] in LABELS, f"existing band label s{s} is not a read label"
    return {"labels": {"s0": get(0)["label"], "s1": get(1)["label"]}, "cells": {"s0": get(0)["cells"], "s1": get(1)["cells"]}, "floor_G": (r.get("floors") or {}).get("G"), "result": r}, sha


def floor_crosscheck(G, existing_G, tol=1e-12):
    """problems if the floor G computed here differs from floors.G of the existing result."""
    bad = []
    if existing_G is None or G is None:
        return ["floor G is absent in the existing result or here: cannot cross-check"]
    for c in OR.CELLS:
        a, b = G.get(c), existing_G.get(c)
        if (a is None) != (b is None) or (a is not None and abs(a - b) > tol):
            bad.append(f"floor G differs from the existing read on {c}: {a} vs {b}")
    return bad


# ----------------------------------------------------------------------------------------------------------------------------------------- rendering
def _f(v, nd=4):
    return "-" if v is None else f"{v:+.{nd}f}"


def _tag(v):
    return "COST" if v.get("cost") else ("gain" if v.get("gain") else ("sep, sub-floor" if v.get("separated") else ""))


def render_md(rep):
    L = [f"# OBD seed-3 BAND read: {rep['outcome_line']}", "",
         f"Amendment `AMEND_OBD_SEED3.md` (sha256 `{AMEND_SHA256[:8]}`), card `PREREG_V8F_OBD.md` (sha256 `{rep['card_sha256'][:8]}`), code of record `{rep['code_of_record'][:8]}`. "
         f"Existing read `RESULT_OBD.json` sha256 `{rep['existing_result_sha256'][:8]}`. Tier {rep['tier']}. Evidence class {rep['evidence_class']}. Estimator: {rep['estimator']}.", ""]
    if rep["smoke"]:
        L += ["**SMOKE: stand-in dumps (the existing real OBD dumps mapped onto the new tags). A code check, never a result.**", ""]
    L += ["STRATEGIC: **PAUSED** (PI item 25, `PI_DECISION_QUEUE.md`, 2026-09-19), reported as paused, not as a pass.", ""]
    if rep["controls"]["problems"]:
        L += ["## INVALID: controls that failed", *[f"* {p}" for p in rep["controls"]["problems"]], ""]
    b = rep["band"]
    L += ["## Band outcome", f"**{b['outcome']}**", "",
          "Rule (amendment): the label held by at least 3 of the 4 per-seed labels, with at least one of them at each init (M2b s0 and M2b s1), else BAND-UNSETTLED.", "",
          "| per-seed label | init | label | source |", "|---|---|---|---|"]
    for s in SEEDS:
        L.append(f"| {s} | M2b s{C.SEED_LABELS[s]} | {b['labels'][s]} | {'existing RESULT_OBD.json' if s in ('s0', 's1') else 'this read'} |")
    L += ["", f"**Recipe of record:** {b['recipe_of_record'] or 'none yet'}", f"**Triggered:** {b['trigger']}", ""]
    for s in ("s10", "s11"):
        p = b["per_seed"][s]
        L.append(f"* {s}: {p.get('label')} ({p.get('n_cells_read', 0)} cells read)" + (f", cells without floor {p['cells_without_floor']}" if p.get("cells_without_floor") else ""))
        for c in p.get("named_costs", []):
            L.append(f"    * COST {c['cell']}: {_f(c['delta'])} [{_f(c['lo'])}, {_f(c['hi'])}], floor {_f(c['floor'])}")
        for c in p.get("named_gains", []):
            L.append(f"    * gain {c['cell']}: {_f(c['delta'])} [{_f(c['lo'])}, {_f(c['hi'])}], floor {_f(c['floor'])}")
    L.append("")
    G = b["floor_G"] or {}
    for fam in ("PRIMARY", "LONGITUDINAL", "LATERAL", "TACTICAL"):
        L += [f"## Family {fam}: BAND minus CTL, per seed (floor G)", "", "| cell | floor G | " + " | ".join(SEEDS) + " |", "|---|---|" + "---|" * len(SEEDS)]
        for c in OR.CELLS:
            if OR.FAMILY[c.split(":")[-1]] != fam:
                continue
            row = [c, _f(G.get(c))]
            for s in SEEDS:
                v = (b["per_seed"][s].get("cells") or {}).get(c)
                row.append("n/a" if not v or v.get("status") != "OK" else f"{_f(v['delta'])} [{_f(v['lo'])}, {_f(v['hi'])}] {_tag(v)}".strip())
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    L += ["## STRATEGIC", "", "PAUSED (PI item 25, 2026-09-19): reported as paused, never as a pass.", "",
          "## Beside (no bar)", "", "### The four labels", "", ", ".join(f"{s}: {b['labels'][s]}" for s in SEEDS), "",
          "### Per-cell spread of BAND minus CTL across the four seeds", "", "| cell | " + " | ".join(SEEDS) + " | range |", "|---|" + "---|" * (len(SEEDS) + 1)]
    for c, r in b["beside"]["per_cell_spread_BAND_minus_CTL"].items():
        L.append(f"| {c} | " + " | ".join(_f(r[s]) for s in SEEDS) + f" | {_f(r['range'])} |")
    L += ["", "### Code-drift row (NULL minus DAG-OACC, from the existing read), for scale", ""]
    cd = b["beside"]["code_drift_sizes"]
    if not cd:
        L.append("absent in the existing result.")
    for s, d in cd.items():
        L.append(f"* {s}: {d['n_cells_separated']} of 24 cells separated; largest |delta| per family: " + "; ".join(
            f"{f} {v['cell']} {_f(v['delta'])}" for f, v in d["largest_by_family"].items()))
    if b["beside"].get("operative_only_gain_note"):
        L += ["", "### Operative-only gain (BAND-UNSETTLED branch)", "", b["beside"]["operative_only_gain_note"]]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------------------------------------------- loading real / stand-in dumps
def default_map(dumps, ref_dumps):
    d = lambda t: f"{dumps}/dump_{t}_m10_causal_heads"  # noqa: E731
    m = {t: d(t) for t in ("OBD-BAND_rep", "OBD-BAND_s0") + C.NEW_TAGS}
    for s in (0, 1):
        m[f"DAGOACC_s{s}"] = f"{ref_dumps}/dump_PARITY_DAGOACC_s{s}_m10_causal_heads"
    return m


def load_arms(amap, need, ready):
    """-> arms {tag: cells}, eids {init: eid}.  Same windows / geometry path as obd_read.main (window arrays asserted equal to the banked DAG-OACC dump)."""
    import obd_geom as G
    geom = G.build()
    banked, cache, arms, eids = {}, {}, {}, {}
    for s in (0, 1):
        banked[s] = OR.load_dump(OR.BANKED_REF.format(s=s))
        dg = OR.load_dump(amap[f"DAGOACC_s{s}"])
        for k in OR.SAME:
            assert np.array_equal(dg[k], banked[s][k]), f"window or model-free array differs from the banked DAG-OACC dump: {k} in {amap[f'DAGOACC_s{s}']}"
        cache[s] = (dg, geom.join(dg), *OR.geom_labels(geom, dg))
        eids[s] = dg["eid"]
    for tag in need:
        if not ready.get(tag):
            continue
        s = C.init_of_tag(tag)
        dg, J, glat, glon = cache[s]
        x = OR.load_dump(amap[tag])
        for k in OR.SAME:
            assert np.array_equal(x[k], banked[s][k]), f"window or model-free array differs from the banked DAG-OACC dump: {k} in {amap[tag]}"
        arms[tag] = OR.to_cells(geom.arm_arrays(x["cl"], dg["g"], J, dg["eid"]), glat, glon)
        print(f"loaded {tag}", flush=True)
    return arms, eids, geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--map", default=None, help="JSON {tag: dump_dir}; tags OBD-BAND_rep, OBD-BAND_s0, OBD-CTL/BAND_s10/s11, DAGOACC_s0/s1")
    ap.add_argument("--dumps", default=os.environ.get("OBD_DUMPS", "F:/TanitAD-scratch/otband_read/t1"))
    ap.add_argument("--ref-dumps", default=os.environ.get("OBD_REF_DUMPS", "F:/TanitAD-scratch/otband_read/t1"))
    ap.add_argument("--pull", default=os.environ.get("OBD_PULLS", "C:/Users/Admin/t2v8f_pulls"))
    ap.add_argument("--existing", default=EXISTING_RESULT)
    ap.add_argument("--smoke", action="store_true", help="code check on stand-in dumps: skips bank controls; never a result")
    ap.add_argument("--n-boot", type=int, default=OR.NB)
    a = ap.parse_args()
    t0 = time.time()
    CI = OR.load_ci()
    existing, esha = load_existing(a.existing)
    amap = json.load(open(a.map, encoding="utf-8")) if a.map else default_map(a.dumps, a.ref_dumps)
    need = ("OBD-BAND_rep", "OBD-BAND_s0") + C.NEW_TAGS
    ready = {t: (t in amap and OR.dump_ready(amap[t])) for t in list(need) + ["DAGOACC_s0", "DAGOACC_s1"]}
    rep = {"tool": "obd_read_seed3.py", "amendment": "theory/obd/AMEND_OBD_SEED3.md", "amendment_sha256": AMEND_SHA256, "card_sha256": OR.CARD_SHA256,
           "code_of_record": OR.CODE_OF_RECORD, "existing_result": os.path.relpath(a.existing, HERE).replace("\\", "/"), "existing_result_sha256": esha,
           "tier": "T1-kin (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner)",
           "evidence_class": "MEASURED (ours; dumps, zero training)", "smoke": bool(a.smoke), "STRATEGIC": {"status": "PAUSED", "ruling": "PI item 25 (2026-09-19)"},
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed {OR.SEED}; ci.py from {CI.SOURCE}. It answers 'another draw of EPISODES'; 'another TRAINING run' "
                        "is floor G (OBD-BAND_rep vs OBD-BAND_s0) and the four-seed combination rule",
           "combination_rule": "label held by at least 3 of 4 per-seed labels (s0, s1, s10, s11) with at least one at each init, else BAND-UNSETTLED (AMEND_OBD_SEED3.md lines 21-24)",
           "controls": {"problems": [], "notes": {}}, "arms_ready": ready}
    prob, notes = rep["controls"]["problems"], rep["controls"]["notes"]
    miss = [t for t in need + ("DAGOACC_s0", "DAGOACC_s1") if not ready.get(t)]
    if miss:
        rep["outcome_line"] = rep["band"] = None
        rep["outcome"] = "NOT-YET"
        rep["reason"] = f"dumps missing: {miss}"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, "NOT-YET", rep["reason"])
        return
    arms, eids, geom = load_arms(amap, need, ready)
    notes["geometry"] = {"extracted_nodes": geom.found}
    rep["controls"]["identity_control"] = OR.identity_control(CI, eids[0])
    if not rep["controls"]["identity_control"]["pass"]:
        prob.append("identity control failed: an arm minus itself did not read 0 [0, 0]")
    if not a.smoke:
        for tag in C.NEW_TAGS:
            bd = f"{a.pull}/obd_{tag}"
            mp, sp_ = f"{bd}/BANK_MANIFEST.json", f"{bd}/selfcheck.json"
            if not os.path.exists(mp):
                prob.append(f"{tag}: BANK_MANIFEST.json missing ({mp})")
            else:
                prob += OR.check_manifest(tag, json.load(open(mp, encoding="utf-8")), OR.done_sha(amap[tag]))
            if not os.path.exists(sp_):
                prob.append(f"{tag}: selfcheck.json missing ({sp_})")
            else:
                b_, n_ = OR.check_selfcheck(tag, json.load(open(sp_, encoding="utf-8")), C.init_of_tag(tag))
                prob += b_
                notes[f"selfcheck_{tag}"] = n_
            ap_ = C.argv_problem(a.pull, tag)
            if ap_:
                prob.append(ap_)
    r = read_seed3(CI, arms, eids, existing, a.n_boot)
    prob += floor_crosscheck(r["floor_G"], existing["floor_G"])
    rep["band"] = r
    if prob:
        rep["outcome"] = "INVALID"
    elif a.smoke:
        rep["outcome"] = f"SMOKE (code check, never a result): {r['outcome']}"
    else:
        rep["outcome"] = r["outcome"]
    rep["outcome_line"] = rep["outcome"]
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(render_md(rep))
    print("WROTE", a.out, "|", rep["outcome"], "| labels:", r["labels"], "| problems:", len(prob), "|", rep["wall_s"], "s")
    for p in prob:
        print("  INVALID:", p)
    print("  RECIPE OF RECORD:", r["recipe_of_record"], "|", r["trigger"])


if __name__ == "__main__":
    main()
