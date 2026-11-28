"""SPEC_ADDENDUM_A6: announced(entry, record) and the gating control G1.

announced() CALLS the nav_command builder's own functions (`stack/scripts/s2_geom_emit_v7.py::nav_command` and
`is_turn`; PI 2026-08-29 suppression included) on the entry's manoeuvre handed over "as if it were the next
manoeuvre". No threshold is written here: they live in the builder (TURN_MIN_DYAW_DEG 15 / TURN_MAX_ARC_R_M 140 /
TURN_MAX_VMIN_MS 8.0) and are read from it.

⚠️ Run with PYTHONPATH="D:/Projects/TanitAD/stack" (a WINDOWS path). The venv's editable `tanitad` install maps to the
abandoned G: checkout; this module puts D:/Projects/TanitAD/stack first on sys.path BEFORE importing the builder and
refuses to continue if any imported module lives on G:. Do NOT combine this process with the launch-tree
(C:/Users/Admin/ev7) `taniteval` import: they resolve different `tanitad` trees. The scoring process therefore reads the
table this module writes (`raw/a6_announced_table.json`), never the builder.

Run:  python announced_a6.py --stage g1      -> raw/a6_g1.json (G1 + its decomposition), exit 0 iff G1 passes
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

STACK = "D:/Projects/TanitAD/stack"
BUILDER = Path(STACK) / "scripts" / "s2_geom_emit_v7.py"
HERE = Path(__file__).resolve().parent
TURN = ("NAV_TURN_L", "NAV_TURN_R")
SIDE = {"NAV_TURN_L": 1, "NAV_TURN_R": -1}
G1_ALL_MIN = 0.99          # SPEC_ADDENDUM_A6: >= 99 % of records
G1_SUPPRESSED_MIN = 1.00   # SPEC_ADDENDUM_A6: 100 % of records whose nav_command.reason names a suppression

_B = None


def builder():
    """Import the nav_command builder from D: (never from G:)."""
    global _B
    if _B is not None:
        return _B
    if STACK not in sys.path:
        sys.path.insert(0, STACK)
    if "tanitad" in sys.modules and not str(sys.modules["tanitad"].__file__).replace("\\", "/").startswith(STACK):
        raise SystemExit("[A6] `tanitad` was already imported from another tree: " + str(sys.modules["tanitad"].__file__))
    spec = importlib.util.spec_from_file_location("s2_geom_emit_v7", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s2_geom_emit_v7"] = mod
    spec.loader.exec_module(mod)
    on_g = sorted(f for f in {getattr(m, "__file__", None) for m in sys.modules.values()} if f and f.upper().startswith("G:"))
    if on_g:
        raise SystemExit("[A6] a module was imported from G: " + str(on_g[:3]))
    _B = mod
    return mod


def builder_md5():
    return hashlib.md5(BUILDER.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------- #
# announced()                                                                                        #
# ------------------------------------------------------------------------------------------------- #
_DUMMY_POSES = np.zeros((1200, 4))      # nav_command reads poses only to fill `args.distance_m` (not used here)


def seq_element(entry, record):
    """The record's `manoeuvre_sequence` element of this nav_30s entry (same t_start_s and dyaw), as the builder's
    seq tuple (t_start, t_end, dyaw, R, vmin). Raises if there is none: never guess."""
    for m in record.get("manoeuvre_sequence") or []:
        if m["t_start_s"] == entry["t_start_s"] and (entry.get("dyaw_deg") is None or m["dyaw_deg"] == entry["dyaw_deg"]):
            return (m["t_start_s"], m["t_end_s"], m["dyaw_deg"], m["radius_m"], m["v_min_ms"])
    raise LookupError("no manoeuvre_sequence element for the entry")


def suppression_applies(record):
    """The emitter writes the key `applied` (s2_geom_emit_v7.py:893); build_v8_nav30s.py:83 read `suppressed`, which does not
    exist -- that is why nav_30s still lists the contested turns."""
    return bool((record.get("turn_suppression") or {}).get("applied"))


def command_token(seg, suppress):
    """The builder's own nav_command() applied to ONE manoeuvre as the next one (seg=None: nothing ahead)."""
    return builder().nav_command(_DUMMY_POSES, 0, [] if seg is None else [seg], suppress_turn=bool(suppress))["token"]


def announced(entry, record):
    """True iff the builder would COMMAND this entry's turn if it were the next manoeuvre: it is a turn by the builder's
    is_turn (not a road curve) and the record's turn is not a contested / obstacle-pass one (PI 2026-08-29).
    A NAV_FOLLOW_ROAD entry announces nothing."""
    if entry["token"] not in TURN:
        return False
    return command_token(seq_element(entry, record), suppression_applies(record)) == entry["token"]


def shipped_side(record):
    return SIDE.get(record["nav_command"]["token"], 0)


def announced_flags(record):
    """per-entry announced flags, in the order of nav_30s.entries."""
    return [bool(announced(e, record)) for e in record["nav_30s"]["entries"]]


# ------------------------------------------------------------------------------------------------- #
# G1                                                                                                  #
# ------------------------------------------------------------------------------------------------- #
def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def sha12(clip_id):
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]


def g1_control(train_path, eval_path):
    recs = [("train", r) for r in load(train_path)] + [("eval", r) for r in load(eval_path)]
    out = {"builder": str(BUILDER), "builder_md5": builder_md5(),
           "thresholds_read_from_builder": {"TURN_MIN_DYAW_DEG": builder().TURN_MIN_DYAW_DEG,
                                            "TURN_MAX_ARC_R_M": builder().TURN_MAX_ARC_R_M,
                                            "TURN_MAX_VMIN_MS": builder().TURN_MAX_VMIN_MS},
           "criteria": {"all_records_min": G1_ALL_MIN, "suppression_named_min": G1_SUPPRESSED_MIN}}
    res = {"all": [0, 0], "train": [0, 0], "eval": [0, 0]}
    sup = [0, 0]
    seq0 = [0, 0]
    flag_mis = 0
    kinds = {}
    firstonly_mis = 0
    ex = {}
    for split, r in recs:
        ship = shipped_side(r)
        e0 = r["nav_30s"]["entries"][0]
        a = (1 if announced(e0, r) and e0["token"] == "NAV_TURN_L" else
             -1 if announced(e0, r) and e0["token"] == "NAV_TURN_R" else 0)
        ok = (a == ship)
        for k in ("all", split):
            res[k][0] += int(ok)
            res[k][1] += 1
        reason = str(r["nav_command"].get("reason") or "")
        if "contested" in reason or "suppress" in reason.lower():
            sup[1] += 1
            sup[0] += int(ok)
        # diagnostic (NOT a substitute for the registered control): the builder's own input is seq[0]
        seq = r.get("manoeuvre_sequence") or []
        s0 = command_token(None if not seq else (seq[0]["t_start_s"], seq[0]["t_end_s"], seq[0]["dyaw_deg"],
                                                 seq[0]["radius_m"], seq[0]["v_min_ms"]), suppression_applies(r))
        seq0[0] += int(SIDE.get(s0, 0) == ship)
        seq0[1] += 1
        for m in seq:
            flag_mis += int(builder().is_turn((m["t_start_s"], m["t_end_s"], m["dyaw_deg"], m["radius_m"], m["v_min_ms"]))
                            != bool(m["is_turn"]))
        if not ok:
            if ship == 0 and e0["token"] in TURN and "curve" in reason:
                kind = "curve_first: nav_command looked at seq[0] (a road curve) -> FOLLOW; entries[0] is a LATER real turn"
            elif ship != 0 and e0["token"] == "NAV_FOLLOW_ROAD":
                t = r["nav_command"].get("args", {}).get("time_s")
                kind = ("turn_beyond_nav_30s_horizon: nav_command TURN with time_s %s > 30.0 (no nav_30s entry)"
                        % ("> 30" if t is not None and t > 30.0 else "<= 30 ?!"))
            else:
                kind = "OTHER"
            kinds[kind] = kinds.get(kind, 0) + 1
            ex.setdefault(kind, []).append(sha12(r["clip_id"]))
            if kind.startswith("OTHER"):
                pass
        # variant (ii): an entry counts as announced only when it is ALSO the first sustained segment (nothing before it)
        first_ok = (seq and seq[0]["t_start_s"] == e0["t_start_s"] and e0["token"] in TURN and announced(e0, r))
        a2 = SIDE.get(e0["token"], 0) if first_ok else 0
        firstonly_mis += int(a2 != ship)
    n = res["all"][1]
    out["n_records"] = {"all": n, "train": res["train"][1], "eval": res["eval"][1]}
    out["G1_literal_entries0"] = {
        "n_match": res["all"][0], "rate": res["all"][0] / n,
        "train": {"n_match": res["train"][0], "n": res["train"][1], "rate": res["train"][0] / res["train"][1]},
        "eval": {"n_match": res["eval"][0], "n": res["eval"][1], "rate": res["eval"][0] / res["eval"][1]},
        "PASS_all_records_ge_99pct": bool(res["all"][0] / n >= G1_ALL_MIN)}
    out["G1_suppression_named_records"] = {"n": sup[1], "n_match": sup[0], "rate": (sup[0] / sup[1] if sup[1] else None),
                                           "PASS": bool(sup[1] > 0 and sup[0] == sup[1])}
    out["G1_PASS"] = bool(out["G1_literal_entries0"]["PASS_all_records_ge_99pct"] and out["G1_suppression_named_records"]["PASS"])
    out["mismatch_decomposition"] = {"n_mismatch": n - res["all"][0], "by_kind": kinds,
                                     "examples_sha12_(first_6_each)": {k: v[:6] for k, v in ex.items()}}
    out["diagnostics_NOT_the_registered_control"] = {
        "builder_rule_on_seq0_reproduces_shipped_nav_command": {"n_match": seq0[0], "n": seq0[1], "rate": seq0[0] / seq0[1]},
        "builder_is_turn_on_every_manoeuvre_sequence_element_equals_the_stored_is_turn_flag_mismatches": flag_mis,
        "variant_entry_announced_only_if_also_seq0_literal_entries0_rate": 1 - firstonly_mis / n,
    }
    return out


def g1prime_control(train_path, eval_path):
    """SPEC_ADDENDUM_A7 G1' (replaces A6's G1).
    (1) the builder's rule on manoeuvre_sequence[0] reproduces nav_command on >= 99 % of the 4,719 records;
    (2) on records where entries[0] IS seq[0] (start time within 0.05 s) and nav_command.args.time_s <= 30,
        announced(entries[0]) reproduces the nav_command side on >= 99 %, and on 100 % of suppression-named records.
    READING of 'time_s <= 30' for records WITHOUT a time_s (every FOLLOW command has args {}): the condition is
    satisfied (there is no time to exceed 30); for a TURN command it excludes beyond-cap records, which the start-time
    condition already excludes. Both sub-populations are reported separately so the verdict does not hinge on this."""
    recs = [("train", r) for r in load(train_path)] + [("eval", r) for r in load(eval_path)]
    c1 = [0, 0]
    pop = {"all": [0, 0], "turn_commands": [0, 0], "follow_commands": [0, 0]}
    sup_in, sup_all = [0, 0], [0, 0]
    excluded = {"entries0_is_not_seq0": 0, "time_s_gt_30": 0}
    mism = []
    for split, r in recs:
        ship = shipped_side(r)
        seq = r.get("manoeuvre_sequence") or []
        s0 = command_token(None if not seq else (seq[0]["t_start_s"], seq[0]["t_end_s"], seq[0]["dyaw_deg"],
                                                 seq[0]["radius_m"], seq[0]["v_min_ms"]), suppression_applies(r))
        c1[0] += int(SIDE.get(s0, 0) == ship)
        c1[1] += 1
        e0 = r["nav_30s"]["entries"][0]
        a = 1 if (announced(e0, r) and e0["token"] == "NAV_TURN_L") else -1 if (announced(e0, r) and e0["token"] == "NAV_TURN_R") else 0
        reason = str(r["nav_command"].get("reason") or "")
        named = "contested" in reason or "suppress" in reason.lower()
        if named:
            sup_all[1] += 1
            sup_all[0] += int(a == ship)
        is_seq0 = bool(seq) and abs(seq[0]["t_start_s"] - e0["t_start_s"]) <= 0.05
        t_s = r["nav_command"].get("args", {}).get("time_s")
        if not is_seq0:
            excluded["entries0_is_not_seq0"] += 1
            continue
        if t_s is not None and t_s > 30.0:
            excluded["time_s_gt_30"] += 1
            continue
        kind = "turn_commands" if t_s is not None else "follow_commands"
        for k in ("all", kind):
            pop[k][0] += int(a == ship)
            pop[k][1] += 1
        if named:
            sup_in[1] += 1
            sup_in[0] += int(a == ship)
        if a != ship:
            mism.append(sha12(r["clip_id"]))
    out = {"builder": str(BUILDER), "builder_md5": builder_md5(), "n_records": c1[1],
           "criteria": {"c1_min": 0.99, "c2_min": 0.99, "c2_suppression_named": 1.0}}
    out["G1prime_1_builder_rule_on_seq0"] = {"n_match": c1[0], "n": c1[1], "rate": c1[0] / c1[1], "PASS": bool(c1[0] / c1[1] >= 0.99)}
    out["G1prime_2_announced_entries0_on_records_where_entries0_is_seq0_and_time_s_le_30"] = {
        "n_match": pop["all"][0], "n": pop["all"][1], "rate": pop["all"][0] / pop["all"][1],
        "by_population": {k: {"n_match": v[0], "n": v[1], "rate": (v[0] / v[1] if v[1] else None)} for k, v in pop.items() if k != "all"},
        "excluded_records": excluded, "mismatch_sha12": mism[:20],
        "PASS": bool(pop["all"][0] / pop["all"][1] >= 0.99)}
    out["G1prime_2_suppression_named"] = {
        "inside_the_G1prime_population": {"n": sup_in[1], "n_match": sup_in[0], "PASS": bool(sup_in[1] > 0 and sup_in[0] == sup_in[1])},
        "all_suppression_named_records": {"n": sup_all[1], "n_match": sup_all[0], "PASS": bool(sup_all[1] > 0 and sup_all[0] == sup_all[1])}}
    out["G1prime_PASS"] = bool(out["G1prime_1_builder_rule_on_seq0"]["PASS"]
                               and out["G1prime_2_announced_entries0_on_records_where_entries0_is_seq0_and_time_s_le_30"]["PASS"]
                               and out["G1prime_2_suppression_named"]["inside_the_G1prime_population"]["PASS"])
    out["modules_imported_from_G_mount"] = sorted(f for f in {getattr(m, "__file__", None) for m in sys.modules.values()} if f and f.upper().startswith("G:"))
    return out


def table_stage(train_path, eval_path, out):
    """Per-entry announced flags for every record (sha12 -> flags) + census; read by the scoring process (which must not
    import the builder). Written even when G1 fails: it is a DESCRIPTIVE table, no arm is scored from it here."""
    tab = {"eval": {}, "census": {}}
    for split, path in (("train", train_path), ("eval", eval_path)):
        n_ent = n_ann = n_turn_ent = n_sup_rec = n_sup_entries_removed = n_multi_sup = 0
        for r in load(path):
            fl = announced_flags(r)
            if split == "eval":
                tab["eval"][sha12(r["clip_id"])] = fl
            ents = r["nav_30s"]["entries"]
            n_ent += len(ents)
            n_turn_ent += sum(e["token"] in TURN for e in ents)
            n_ann += sum(fl)
            if suppression_applies(r):
                n_sup_rec += 1
                n_sup_entries_removed += sum(1 for e, f in zip(ents, fl) if e["token"] in TURN and not f)
                n_multi_sup += int(sum(e["token"] in TURN for e in ents) > 1)
        tab["census"][split] = {"entries": n_ent, "turn_entries": n_turn_ent, "announced": n_ann,
                                "turn_entries_not_announced": n_turn_ent - n_ann,
                                "records_with_turn_suppression_applied": n_sup_rec,
                                "turn_entries_removed_on_suppressed_records": n_sup_entries_removed,
                                "suppressed_records_with_more_than_one_turn_entry": n_multi_sup}
    tab["builder_md5"] = builder_md5()
    Path(out).write_text(json.dumps(tab, indent=1), encoding="utf-8")
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("g1", "g1prime", "table"), default="g1")
    ap.add_argument("--labels-train", default="D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz")
    ap.add_argument("--labels-eval", default="D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")
    ap.add_argument("--out", default=str(HERE.parent / "raw" / "a6_g1.json"))
    a = ap.parse_args()
    md5 = {Path(p).name: hashlib.md5(Path(p).read_bytes()).hexdigest() for p in (a.labels_train, a.labels_eval)}
    if not md5["s2_labels_v8_train.jsonl.gz"].startswith("b45377a1") or not md5["s2_labels_v8_eval.jsonl.gz"].startswith("eefc38d1"):
        raise SystemExit("[A6] label md5 differs from the run's")
    if a.stage == "table":
        t = table_stage(a.labels_train, a.labels_eval, HERE.parent / "raw" / "a6_announced_table.json")
        print(json.dumps(t["census"], indent=1))
        return 0
    if a.stage == "g1prime":
        res = g1prime_control(a.labels_train, a.labels_eval)
        res["inputs_md5"] = md5
        res["spec"] = {"file": "SPEC_ADDENDUM_A7.md",
                       "sha256_expected": "8e846a842018ad8c21a953ae8db3827a6be6e6b6343bce97fb8b1ac67164e5a5",
                       "sha256_on_disk": hashlib.sha256((HERE.parent / "SPEC_ADDENDUM_A7.md").read_bytes()).hexdigest()}
        if res["spec"]["sha256_on_disk"] != res["spec"]["sha256_expected"]:
            raise SystemExit("[A7] SPEC_ADDENDUM_A7.md does not match its registered sha256")
        (HERE.parent / "raw" / "a7_g1prime.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return 0 if res["G1prime_PASS"] else 2
    res = g1_control(a.labels_train, a.labels_eval)
    res["inputs_md5"] = md5
    res["spec"] = {"file": "SPEC_ADDENDUM_A6.md",
                   "sha256_expected": "b409957816248d333886ad694c5e3061cc26c9a29d115b246de4c91b1c5fd736",
                   "sha256_on_disk": hashlib.sha256((HERE.parent / "SPEC_ADDENDUM_A6.md").read_bytes()).hexdigest()}
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_records", "G1_literal_entries0", "G1_suppression_named_records", "G1_PASS",
                                          "mismatch_decomposition", "diagnostics_NOT_the_registered_control")}, indent=1,
                     ensure_ascii=False))
    return 0 if res["G1_PASS"] else 2


if __name__ == "__main__":
    sys.exit(main())
