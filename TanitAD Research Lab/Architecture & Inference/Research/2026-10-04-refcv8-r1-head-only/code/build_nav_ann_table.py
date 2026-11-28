"""H3's nav input source: per record, the nav_30s entries + A6's announced() flags, for train AND eval. CPU, dev box.

announced() is the route package's `code/announced_a6.py` (SPEC_ADDENDUM_A6: it CALLS the nav_command builder
`stack/scripts/s2_geom_emit_v7.py` from D:, never re-derives a threshold) -- imported by path, not copied. A6's G1 was
mis-specified and REPLACED by SPEC_ADDENDUM_A7's G1' (semantics fixed there: a real turn behind a road curve IS
announced; the obstacle-pass suppression is clip-level, as the builder applies it). G1' is run here on all 4,719 records
and must PASS before the table is written:
  1. the builder's rule on manoeuvre_sequence[0] reproduces nav_command on >= 99 % of records;
  2. on records where entries[0] IS seq[0] (start within 0.05 s) and nav_command.args.time_s <= 30 (records with no
     time_s -- FOLLOW -- reported both included and excluded, both must pass), announced(entries[0]) reproduces the
     nav_command side on >= 99 %, and on 100 % of suppression-named records.
Output: raw/r1_nav_ann_table.json  {sha12: {"entries": [{token, t_start_s, t_end_s}], "announced": [bool],
"nav_command_side": int}} -- sha12 only, never a clip id.
Run with PYTHONPATH="D:/Projects/TanitAD/stack" (a WINDOWS path).
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTE = Path("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
             "2026-10-04-refcv7-route-following/code/announced_a6.py")


def main():
    spec = importlib.util.spec_from_file_location("announced_a6", ROUTE)
    A6 = importlib.util.module_from_spec(spec)
    sys.modules["announced_a6"] = A6
    spec.loader.exec_module(A6)
    tr = "D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz"
    ev = "D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"
    md5 = {Path(p).name: hashlib.md5(Path(p).read_bytes()).hexdigest() for p in (tr, ev)}
    if not md5["s2_labels_v8_train.jsonl.gz"].startswith("b45377a1") or \
            not md5["s2_labels_v8_eval.jsonl.gz"].startswith("eefc38d1"):
        raise SystemExit("[r1] label md5 differs from the run's")
    g1 = A6.g1_control(tr, ev)
    c1 = g1["diagnostics_NOT_the_registered_control"]["builder_rule_on_seq0_reproduces_shipped_nav_command"]
    c2 = {"with_no_time": [0, 0], "time_le_30_only": [0, 0]}
    sup = [0, 0]
    for path in (tr, ev):
        for r in A6.load(path):
            seq = r.get("manoeuvre_sequence") or []
            e0 = r["nav_30s"]["entries"][0]
            if not seq or abs(float(seq[0]["t_start_s"]) - float(e0["t_start_s"])) > 0.05:
                continue
            ts = (r["nav_command"].get("args") or {}).get("time_s")
            if ts is not None and float(ts) > 30.0:
                continue
            a_ = (A6.SIDE.get(e0["token"], 0) if A6.announced(e0, r) else 0)
            ok = int(a_ == A6.shipped_side(r))
            c2["with_no_time"][0] += ok
            c2["with_no_time"][1] += 1
            if ts is not None:
                c2["time_le_30_only"][0] += ok
                c2["time_le_30_only"][1] += 1
            reason = str(r["nav_command"].get("reason") or "")
            if "contested" in reason or "suppress" in reason.lower():
                sup[0] += ok
                sup[1] += 1
    g1p = {"1_seq0_rule": c1, "2_entries0_is_seq0": {k: {"n_match": v[0], "n": v[1], "rate": v[0] / max(v[1], 1)}
                                                       for k, v in c2.items()},
           "2_suppression_named": {"n_match": sup[0], "n": sup[1]}}
    g1p["PASS"] = bool(c1["rate"] >= 0.99 and all(v["rate"] >= 0.99 for v in g1p["2_entries0_is_seq0"].values())
                       and sup[1] > 0 and sup[0] == sup[1])
    print(json.dumps(g1p, indent=1))
    if not g1p["PASS"]:
        raise SystemExit("[r1] G1' (SPEC_ADDENDUM_A7) FAILS -- the announced table is not written")
    out = {"_meta": {"announced_from": str(ROUTE), "announced_md5": hashlib.md5(ROUTE.read_bytes()).hexdigest(),
                     "builder_md5": A6.builder_md5(), "labels_md5": md5,
                     "G1_prime_A7": g1p,
                     "A6_G1_as_registered_(superseded)": {k: g1[k] for k in ("G1_literal_entries0", "G1_PASS")}}}
    n = {"train": 0, "eval": 0}
    for split, path in (("train", tr), ("eval", ev)):
        for r in A6.load(path):
            s = A6.sha12(r["clip_id"])
            ents = [{"token": e["token"], "t_start_s": float(e["t_start_s"]), "t_end_s": float(e["t_end_s"])}
                    for e in r["nav_30s"]["entries"]]
            out[s] = {"split": split, "entries": ents, "announced": A6.announced_flags(r),
                      "nav_command_side": A6.shipped_side(r), "t0_s": float(r["t0_s"])}
            n[split] += 1
    out["_meta"]["n_records"] = n
    p = HERE.parent / "raw" / "r1_nav_ann_table.json"
    p.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(out["_meta"], indent=1))


if __name__ == "__main__":
    main()
