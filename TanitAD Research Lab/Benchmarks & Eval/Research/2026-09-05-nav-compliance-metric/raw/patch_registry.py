"""Register the nav-compliance criteria in CRITERIA_REGISTRY.json (2.5.0 -> 2.6.0)
and point the route_head_echo leak guard at them. Idempotent.

Structure harvested from the predecessor's draft: ONE criterion for the metric
and ONE PER INTERVENTION CONTROL, so a missing control is a NAMED violation.
Keys are the ones taniteval/taniteval/nav_compliance.py::compliance_arm writes.
"""
import json
import sys

path = sys.argv[1]
d = json.load(open(path, encoding="utf-8"))
if any(c.get("id") == "strat.nav_compliance"
       for c in d["families"]["STRATEGIC"]["criteria"]):
    raise SystemExit("already registered")
d["version"] = "2.6.0"
PREFIXES = ("strategic", "refcv3.strategic", "four_families.strategic")


def keys(tail):
    return [f"{p}.nav_compliance.readouts.plan.{tail}" for p in PREFIXES]


ADDED = ("2026-09-05, Benchmarks & Eval FlyWheel. PI rulings 2026-09-04: 'we are not "
         "evaluating the nav command itself, we are evaluating the fact that the model is "
         "following the nav command in consistency to the strategic goals'.")
d["families"]["STRATEGIC"]["criteria"] += [
    {
        "id": "strat.nav_compliance",
        "required": True,
        "label": "nav-COMPLIANCE — the emitted BEHAVIOUR follows the route command (imminent windows)",
        "keys": keys("conditionings.nav_true.compliance_with_TRUE_command"),
        "partial_keys": [f"{p}.nav_compliance.readouts.gstr.conditionings.nav_true.compliance_with_TRUE_command"
                         for p in PREFIXES],
        "refused_as": ["nav_compliance"],
        "emitter": "taniteval/taniteval/nav_compliance.py::compliance_arm (via the refcv3_arm.py sidecar; standalone: taniteval/tools/nav_compliance_report.py)",
        "added": ADDED,
        "note": ("Replaces the route metric that could not fail: D-REFAV1-ROUTE-LABEL-IS-THE-NAV "
                 "MEASURED the route LABEL as an exact bijection of the fed nav token (141/141), so "
                 "a head copying its input scored 1.0000 and NO behaviour could lower it. This "
                 "criterion scores the emitted 6 s path (with g_str and the SELECTED anchor as "
                 "sibling readouts) against the COMMAND on windows where the commanded turn lies "
                 "inside the plan horizon, n always reported. ⛔ The RATE is NOT the result — it is "
                 "admissible ONLY with strat.nav_compliance_ctrl_shuffle and _ctrl_zero on the SAME "
                 "windows; a rate that does not drop under them is coincidence with the scene."),
    },
    {
        "id": "strat.nav_compliance_ctrl_shuffle",
        "required": True,
        "label": "nav-compliance CONTROL: paired drop under nav-SHUFFLE (token permuted across windows)",
        "keys": keys("paired_true_minus_shuffled"),
        "refused_as": ["nav_compliance_controls", "nav_compliance"],
        "emitter": "taniteval/taniteval/nav_compliance.py::compliance_arm -> paired_true_minus_shuffled (taniteval.ci paired episode-cluster bootstrap)",
        "added": ADDED,
        "note": ("If the model follows the command, compliance with the TRUE command must DROP when the "
                 "token is permuted; the changed subset must follow the FED command. No drop = the high "
                 "rate is scene coincidence. Known values on the panel: ha0 reads 0.0000 compliance "
                 "exactly; every nav-blind control (ha, ha0_ext) reads a delta of exactly 0.0000."),
    },
    {
        "id": "strat.nav_compliance_ctrl_zero",
        "required": True,
        "label": "nav-compliance CONTROL: paired drop under nav-ZERO (nav withheld, E13 skipped)",
        "keys": keys("paired_true_minus_zero"),
        "refused_as": ["nav_compliance_controls", "nav_compliance"],
        "emitter": "taniteval/taniteval/nav_compliance.py::compliance_arm -> paired_true_minus_zero",
        "added": ADDED,
        "note": ("The robustness ablation (D-NAV-IS-GT: os is the deployment arm, os_navzero the "
                 "ablation). nav_cmd=None collapses the core onto the follow embedding, so this is a "
                 "LOWER bound on nav dependence; BACKLOG R39 binds it beside the shuffle, never instead."),
    },
]
for g in d["leak_guards"]["guards"]:
    if g.get("id") == "route_head_echo":
        g.setdefault("keys", [])
        for k in keys("paired_true_minus_shuffled")[:2]:
            if k not in g["keys"]:
                g["keys"].append(k)
        g["note"] = (g.get("note", "") + " 2026-09-05: the nav-compliance intervention pair "
                     "(taniteval.nav_compliance) IS the behavioural echo test and satisfies this guard.").strip()
json.dump(d, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(path, "a", encoding="utf-8").write("\n")
print("registered strat.nav_compliance + 2 controls; version", d["version"])
