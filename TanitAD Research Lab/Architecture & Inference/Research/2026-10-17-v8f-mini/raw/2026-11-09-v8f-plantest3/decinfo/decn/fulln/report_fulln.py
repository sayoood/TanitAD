"""DECN full-n probe read-out (card PREREG_V8F_DECN.md, RISES trigger; amendment 1's row rule applies). Tier T0, a readout diagnostic. Pooled held-out R2 on the 17 held-out episodes, no interval; differences under 0.05 carry no weight. Writes RESULT_FULLN.md / .json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
DECN = os.path.abspath(os.path.join(HERE, ".."))
TAGS, RUNGS, TGTS = ("M2b_s0", "M2b_s1"), ("R0", "R2"), ("speed", "yaw")
BAR = {"speed": 0.70, "yaw": 0.50}
NAME = {"R0": "M2b z1", "R2": "pooled token"}
LIM = 0.05


def load(rung, tag, N):
    d = f"{HERE}/runs" if N == "N4565" else f"{DECN}/runs"
    return json.load(open(f"{d}/{rung}_{tag}_{N}.json", encoding="utf-8"))


def best(rows, tgt):
    """the maximum over reader and m, over rows whose shuffled control is at or below 0.05 (amendment 1). Returns (value, reader, m, n rows out of the maximum)."""
    r = [x for x in rows if x["target"] == tgt]
    ok = [x for x in r if x["r2_shuffled"] <= LIM]
    b = max(ok, key=lambda x: x["r2"])
    return b["r2"], b["reader"], b["m"], len(r) - len(ok)


def main():
    out = dict(card="PREREG_V8F_DECN.md cfa5bb1f, amendment 1", tier="T0", cells={}, controls={})
    fail_shuf, planted_bad, n_rows = [], [], 0
    digests = set()
    for rung in RUNGS:
        for tag in TAGS:
            f = load(rung, tag, "N4565")
            digests.add(f["meta"]["held_digest"])
            digests.add(load(rung, tag, "N1053")["meta"]["held_digest"])
            for r in f["rows"]:
                n_rows += 1
                if r["r2_shuffled"] > LIM:
                    fail_shuf.append(f"{rung} {tag} {r['reader']} m={r['m']} {r['target']}: {r['r2_shuffled']:+.3f}")
            for p in f["planted"]:
                if not p["passed"]:
                    planted_bad.append(f"{rung} {tag} m={p['m']}: {p['r2_signal']:.3f} / {p['r2_zeroed']:.3f}")
            for tgt in TGTS:
                c = {}
                for N in ("N53", "N1053", "N4565"):
                    v, rd, m, drop = best(load(rung, tag, N)["rows"], tgt)
                    c[N] = dict(r2=v, reader=rd, m=m, rows_out_of_maximum=drop)
                rr = {f"{x['reader']}_m{x['m']}": x["r2"] for x in f["rows"] if x["target"] == tgt}
                c["N4565_all_readers"] = rr
                out["cells"][f"{rung}|{tag}|{tgt}"] = c
    out["controls"] = dict(n_rows=n_rows, shuffled_above_005=fail_shuf, planted_failed=planted_bad, same_held_digest=len(digests) == 1, held_digest=sorted(digests))
    outcome = {}
    for tgt in TGTS:
        clears = {rung: all(out["cells"][f"{rung}|{t}|{tgt}"]["N4565"]["r2"] >= BAR[tgt] for t in TAGS) for rung in RUNGS}
        rise = {rung: [out["cells"][f"{rung}|{t}|{tgt}"]["N4565"]["r2"] - out["cells"][f"{rung}|{t}|{tgt}"]["N1053"]["r2"] for t in TAGS] for rung in RUNGS}
        outcome[tgt] = dict(clears_bar_both_seeds=clears, rise_N1053_to_N4565=rise, label="READ AT FULL N" if any(clears.values()) else "BELOW THE BAR AT FULL N")
    out["outcome"] = outcome
    json.dump(out, open(f"{HERE}/RESULT_FULLN.json", "w", encoding="utf-8"), indent=1)
    a = ["# RESULT_FULLN: the deciding inputs at every trained clip (N4565)", "Card `../../../PREREG_V8F_DECN.md` (`cfa5bb1f`, amendment 1). Tier **T0**, a readout diagnostic, never a driving result. MEASURED: `runs/*.json` here, N53 and N1053 from `../runs/`. Pooled held-out R2, no interval, the same 17 held-out episodes; differences under 0.05 carry no weight. Best over four readers, so optimistic. LONGITUDINAL, LATERAL, TACTICAL have no metric here; STRATEGIC is PAUSED (PI item 25).", "", "## Controls"]
    a.append(f"* Same held-out digest over every N1053 and N4565 unit: {out['controls']['same_held_digest']}.")
    a.append(f"* Shuffled targets at or below 0.05: {n_rows - len(fail_shuf)} of {n_rows} rows at N4565; rows above: {', '.join(fail_shuf) if fail_shuf else 'none'} (amendment 1: a failing row leaves the deciding maximum).")
    a.append(f"* Planted signal: {'all pass' if not planted_bad else '; '.join(planted_bad)}.")
    a += ["", "## Best held-out R2 per deciding cell", "| target (bar) | input | seed | N53 | N1053 | N4565 | reader at N4565 | rise N1053 to N4565 |", "|---|---|---|---|---|---|---|---|"]
    for tgt in TGTS:
        for rung in RUNGS:
            for t in TAGS:
                c = out["cells"][f"{rung}|{t}|{tgt}"]
                a.append(f"| {tgt} ({BAR[tgt]:.2f}) | {rung} {NAME[rung]} | {t[-2:]} | {c['N53']['r2']:.3f} | {c['N1053']['r2']:.3f} | {c['N4565']['r2']:.3f} | {c['N4565']['reader']} m={c['N4565']['m']} | {c['N4565']['r2'] - c['N1053']['r2']:+.3f} |")
    a += ["", "## Every reader at N4565", "| target | input | seed | ridge m=1 | MLP m=1 | ridge m=4 | MLP m=4 |", "|---|---|---|---|---|---|---|"]
    for tgt in TGTS:
        for rung in RUNGS:
            for t in TAGS:
                r = out["cells"][f"{rung}|{t}|{tgt}"]["N4565_all_readers"]
                a.append(f"| {tgt} | {rung} | {t[-2:]} | {r['ridge_m1']:+.3f} | {r['mlp_m1']:+.3f} | {r['ridge_m4']:+.3f} | {r['mlp_m4']:+.3f} |")
    a += ["", "## Outcome per target", ""]
    for tgt, o in outcome.items():
        a.append(f"* **{tgt}: {o['label']}.** Clears the bar at both seeds: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in o["clears_bar_both_seeds"].items()) + ".")
    open(f"{HERE}/RESULT_FULLN.md", "w", encoding="utf-8").write("\n".join(a) + "\n")
    print(json.dumps(outcome, indent=0))
    print(fail_shuf, planted_bad, out["controls"]["same_held_digest"])


if __name__ == "__main__":
    main()
