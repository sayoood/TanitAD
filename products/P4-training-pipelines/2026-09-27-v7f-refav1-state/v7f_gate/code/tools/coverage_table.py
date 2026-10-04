"""The G-DVB coverage table for EVERY train_v6_staged flag (the inventory's 232 at c36b6ddd + the v7F merge's
8): its registry kind, where it runs, its static read sites, its dynamic consumers on the CPU rehearsal, and a
STATUS. Writes raw/coverage_table.{json,md}."""
import json
import sys
from collections import Counter
from pathlib import Path

W = Path(r"C:/Users/Admin/v7f_gate")
sys.path.insert(0, str(W / "tree_m" / "stack"))
from tanitad.train import declared_vs_built_v6 as D6  # noqa: E402

acts = json.load(open(W / "work" / "parser_actions_merge.json"))
static = json.load(open(W / "work" / "static_reads_merge.json"))
dyn = json.load(open(W / "raw" / "dynamic_reads_merge.json"))
inv = json.load(open(r"D:/Projects/TanitAD/products/P4-training-pipelines/2026-09-27-v7f-refav1-state/raw/"
                     r"flag_inventory_train_v6_staged.json"))
inv_by = {}
for f in inv["flags"]:
    inv_by.setdefault(f["dest"], f)
MERGE_NEW = set(D6.ALL_V6_DESTS) | {"strategic_off"}
# FINDINGS are READ from the rehearsal's G-DVB evidence (never hard-coded): lever -> dest via the parser
opt2dest = {o: a["dest"] for a in acts for o in a["options"]}
finding_dests: dict = {}
for _lab, _mm in (dyn.get("dvb_mismatches") or {}).items():
    for _m in _mm:
        _lever = _m.split(":", 1)[0].strip()
        finding_dests.setdefault(opt2dest.get(_lever, _lever), []).append(f"{_lab}: {_m}")
HISTORY = {"nav_cond": ("FINDING: DECLARED-NOT-BUILT at c36b6ddd and on the first v7F merge (never mapped "
                        "into V6Config); FIXED in the merge 2026-09-27 -- now BUILT and fed on the rehearsal. "
                        "Its OPERATIVE channel then trained nothing in S-W (F7, a G-LIVE finding); FIXED in "
                        "the merge too (_NavBoundPredictor) -- the S-W G-LIVE now reaches every nav leaf")}
rows = []
for a in acts:
    d = a["dest"]
    lv = D6.REGISTRY_V6.get(d)
    kind = lv.kind if lv else "UNCOVERED"
    where = ("gate-time (check_all)" if d in D6.GATE_TIME else
             "trainer (build: check; R3 path: check_v6) + gate (check_all)" if lv else "-")
    sfun = sorted({k.split("->")[0] for k in static.get(d, {}).get("by_function", {})})
    cons = dyn["consumers"].get(d, [])
    if d in finding_dests:
        status = "FINDING: DECLARED-NOT-BUILT"
        note = "G-DVB named a declared-vs-built mismatch on the CPU rehearsal: " + "; ".join(
            finding_dests[d])[:300]
    elif d == "v2_val_cache":
        status = "REFUSED-BY-TRAINER"
        note = "train() raises if set (P4-5): dead by design, refused rather than inert"
    elif not sfun and not cons:
        status = "DEAD"
        note = "no args read site anywhere and no consumer on the exercised paths"
    elif kind in ("built", "loss"):
        status = "COVERED-CPU"
        note = "reader on the BUILT stack ran in G-DVB (check_all) on the CPU rehearsal build"
    elif kind == "elsewhere":
        if cons:
            status = "COVERED-CPU (named guard)"
            note = "its guard ran on the CPU rehearsal (G-LIVE term/knob capture or a trainer guard)"
        else:
            status = "GUARD-NOT-EXERCISED-ON-CPU"
            note = "its branch is off in the rehearsal (the guard runs when the lever is set)"
    elif kind == "data":
        status = "NOT-COVERABLE-ON-CPU (data)"
        note = "a host path: content-bound in the token's data manifest on the launch host (Thor)"
    elif kind in ("runtime", "record"):
        status = f"NON-LEVER ({kind})"
        note = "not a model lever; the registry names its role"
    else:
        status = "UNCOVERED"
        note = "no registry entry"
    iv = inv_by.get(d, {})
    rows.append({"options": a["options"], "dest": d, "default": a["default"],
                 "inventory_class": iv.get("proposed_class"), "inventory_status": iv.get("status"),
                 "in_tip_inventory": d in inv_by, "merge_new": d in MERGE_NEW,
                 "dvb_kind": kind, "runs_in": where, "reason": (lv.reason if lv else "")[:200],
                 "static_read_functions": sfun, "dynamic_consumers": cons,
                 "status": status, "note": note + ((" | HISTORY: " + HISTORY[d]) if d in HISTORY else "")})
census = Counter(r["status"] for r in rows)
out = {"dvb_findings_from_rehearsal": finding_dests, "n_option_rows": len(rows), "n_dests": len({r["dest"] for r in rows}),
       "n_tip_inventory_rows": sum(r["in_tip_inventory"] for r in rows),
       "status_census": dict(census), "rows": rows,
       "_method": "kind = declared_vs_built_v6.REGISTRY_V6 (tree_m); static = AST args-object reads in the "
                  "trainer + the helpers it hands the namespace to; dynamic = argparse.Namespace attribute "
                  "reads on the CPU rehearsal (S-W/S-T gate jobs + both dry-runs), argparse internals, "
                  "effective_weights.explicit_dests (a whole-namespace re-parse) and the gate's own reads "
                  "excluded"}
(W / "raw" / "coverage_table.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
md = ["| flag | dest | default | inventory class | DVB kind | status | static read sites | CPU consumers |",
      "|---|---|---|---|---|---|---|---|"]
order = ["FINDING: DECLARED-NOT-BUILT", "REFUSED-BY-TRAINER", "DEAD", "UNCOVERED", "COVERED-CPU",
         "COVERED-CPU (named guard)", "GUARD-NOT-EXERCISED-ON-CPU", "NOT-COVERABLE-ON-CPU (data)",
         "NON-LEVER (runtime)", "NON-LEVER (record)"]
for r in sorted(rows, key=lambda r: (order.index(r["status"]) if r["status"] in order else 99, r["dest"])):
    md.append("| `{}` | `{}` | `{}` | {} | {} | **{}** | {} | {} |".format(
        "/".join(r["options"])[:44], r["dest"], str(r["default"])[:24],
        r["inventory_class"] or ("(merge)" if r["merge_new"] else "-"), r["dvb_kind"], r["status"],
        ", ".join(r["static_read_functions"])[:70] or "-",
        ", ".join(c.split(":")[1] for c in r["dynamic_consumers"])[:60] or "-"))
(W / "raw" / "coverage_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(out["n_option_rows"], "option rows,", out["n_dests"], "dests,", out["n_tip_inventory_rows"], "in the tip inventory")
print(dict(census))
