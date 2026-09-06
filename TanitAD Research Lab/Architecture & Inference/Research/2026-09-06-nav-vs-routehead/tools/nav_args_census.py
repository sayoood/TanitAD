"""Census of the v7.2 nav_command record: token cardinality and the two arg slots.

ASCII-only prints (cp1252 dev box). Pure stdlib.
Answers the PI's P0 question: does time/distance travel with the nav command,
and how far is the next nav command in time and distance?
"""
import collections
import gzip
import json
import statistics
import sys

import os
_BD = os.environ["BLOBDIR"]
BLOBS = {
    "eval": os.path.join(_BD, "s2_labels_v7.2_eval.jsonl.gz"),
    "train": os.path.join(_BD, "s2_labels_v7.2_train.jsonl.gz"),
}


def q(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def census(name, path):
    out = {"split": name, "path": path}
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out["n_records"] = len(rows)
    out["schema_version"] = collections.Counter(
        r.get("schema_version") for r in rows).most_common()
    out["vocab"] = collections.Counter(r.get("vocab") for r in rows).most_common()

    navs = [(r.get("clip_id"), r.get("nav_command") or {}) for r in rows]
    out["n_with_nav"] = sum(1 for _, n in navs if n)
    out["token"] = collections.Counter(n.get("token") for _, n in navs).most_common()
    out["provenance"] = collections.Counter(
        n.get("provenance") for _, n in navs).most_common()
    out["oracle_flag"] = collections.Counter(
        repr(n.get("oracle", "<KEY ABSENT>")) for _, n in navs).most_common()

    # ---- the arg slots: the load-bearing question -------------------------
    argkeys = collections.Counter()
    empty_args = collections.Counter()
    per_tok = collections.defaultdict(lambda: {"d": [], "t": []})
    for _, n in navs:
        tok = n.get("token")
        a = n.get("args")
        if a is None:
            argkeys["<args KEY ABSENT>"] += 1
            empty_args[tok] += 1
            continue
        if not a:
            argkeys["<args EMPTY DICT {}>"] += 1
            empty_args[tok] += 1
            continue
        argkeys["|".join(sorted(a))] += 1
        if "distance_m" in a:
            per_tok[tok]["d"].append(float(a["distance_m"]))
        if "time_s" in a:
            per_tok[tok]["t"].append(float(a["time_s"]))
    out["args_key_shapes"] = argkeys.most_common()
    out["tokens_with_no_args"] = empty_args.most_common()

    stats = {}
    for tok, dd in sorted(per_tok.items()):
        e = {}
        for lab, xs in (("distance_m", dd["d"]), ("time_s", dd["t"])):
            if xs:
                e[lab] = {
                    "n": len(xs), "min": round(min(xs), 3),
                    "p25": round(q(xs, .25), 3), "median": round(q(xs, .50), 3),
                    "p75": round(q(xs, .75), 3), "max": round(max(xs), 3),
                    "mean": round(statistics.fmean(xs), 3),
                }
        stats[tok] = e
    out["arg_stats_by_token"] = stats

    # what a t0_constant consumer would actually see, per record
    seen_d, seen_t = [], []
    for _, n in navs:
        a = n.get("args") or {}
        seen_d.append(float(a.get("distance_m", 0.0)))
        seen_t.append(float(a.get("time_s", 0.0)))
    zd = sum(1 for x in seen_d if x == 0.0)
    out["as_consumed_t0_constant"] = {
        "note": "_args_for_window defaults a missing slot to 0.0",
        "n": len(seen_d),
        "distance_m_zero": zd,
        "distance_m_zero_frac": round(zd / max(len(seen_d), 1), 4),
        "time_s_zero": sum(1 for x in seen_t if x == 0.0),
        "distance_m_nonzero_median": round(
            q([x for x in seen_d if x != 0.0], .5), 3) if zd < len(seen_d) else None,
        "time_s_nonzero_median": round(
            q([x for x in seen_t if x != 0.0], .5), 3) if any(seen_t) else None,
    }
    return out


if __name__ == "__main__":
    res = {}
    for nm, p in BLOBS.items():
        for attempt in range(6):
            try:
                res[nm] = census(nm, p)
                break
            except OSError as exc:
                res[nm] = {"split": nm, "READ_ERROR": repr(exc),
                           "attempt": attempt}
        else:
            pass
    print(json.dumps(res, indent=2))
    sys.stdout.flush()
