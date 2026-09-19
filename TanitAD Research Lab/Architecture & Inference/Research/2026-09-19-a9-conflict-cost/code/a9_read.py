"""A9 reader — overhead of `--conflict-detector on` and its three identity controls.

⛔ Every number is read from the two runs' OWN metrics.jsonl, never from the wrapper's exit
codes. The rate is the median of `elapsed_s` MARGINAL deltas per step, rows at step <= 30
excluded as warm-up. The controls are EXACT identities (float64 contract): a tolerated epsilon
would void every conflict number, so none is applied.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

WARM = 30


def rows(p: pathlib.Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def rate(rs: list[dict]) -> dict:
    tr = [r for r in rs if "elapsed_s" in r and "step" in r
          and not any(k.startswith("eval_") for k in r)]
    tr.sort(key=lambda r: int(r["step"]))
    d = []
    for a, b in zip(tr, tr[1:]):
        ds = int(b["step"]) - int(a["step"])
        if ds > 0 and int(a["step"]) > WARM:
            d.append((float(b["elapsed_s"]) - float(a["elapsed_s"])) / ds)
    d.sort()
    return {"n_deltas": len(d), "last_step": int(tr[-1]["step"]) if tr else None,
            "median_s_per_step": d[len(d) // 2] if d else None,
            "mean_s_per_step": sum(d) / len(d) if d else None}


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    off = rows(root / "off" / "metrics.jsonl")
    on = rows(root / "on" / "metrics.jsonl")
    if not off or not on:
        print(f"ZZA9-VERDICT NO-METRICS off={len(off)} on={len(on)}ZZ")
        return 4
    r_off, r_on = rate(off), rate(on)
    ctl_on = [r["conflict_controls"] for r in on if "conflict_controls" in r]
    ctl_off = [r for r in off if "conflict_controls" in r]
    cd_rows_on = sum(1 for r in on if any(k.startswith("conflict_cos") or k.startswith("cd_")
                                          for k in r))
    ok = True
    notes = []
    if not ctl_on:
        ok = False
        notes.append("the ON arm wrote NO conflict_controls row")
    else:
        c = ctl_on[0]
        s, n, dc = c.get("self_cos"), c.get("negated_cos"), c.get("detached_cos")
        if s != 1.0:
            ok = False
            notes.append(f"self_cos={s!r}, not EXACTLY 1.0")
        if n != -1.0:
            ok = False
            notes.append(f"negated_cos={n!r}, not EXACTLY -1.0")
        if not (isinstance(dc, float) and math.isnan(dc)):
            ok = False
            notes.append(f"detached_cos={dc!r}, not NaN")
    if ctl_off:
        ok = False
        notes.append("the OFF arm wrote a conflict_controls row -- the detector was not off")
    if not (r_off["median_s_per_step"] and r_on["median_s_per_step"]):
        ok = False
        notes.append("a rate could not be formed")
        overhead = None
    else:
        overhead = r_on["median_s_per_step"] / r_off["median_s_per_step"] - 1.0
    out = {"_what": "A9 -- conflict-detector cost at 416x1024, 200 steps on vs off",
           "_evidence_class": "MEASURED (ours)",
           "_not_a_capability_claim": "200 steps x batch 2 = 0.075 of one halfA epoch",
           "rate_off": r_off, "rate_on": r_on,
           "overhead_frac_median": overhead,
           "controls_on": ctl_on[0] if ctl_on else None,
           "off_arm_has_controls_row": bool(ctl_off),
           "on_rows_with_conflict_keys": cd_rows_on,
           "banked_small_grid_overhead": "+71.7 ... +104.2 % (NOT comparable: smaller grid)",
           "notes": notes, "verdict": "OK" if ok else "FAIL"}
    pathlib.Path("C:/Users/Admin/qland/a9_conflict_cost.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    print("ZZA9-OK" if ok else "ZZA9-FAIL")
    return 0 if ok else 6


if __name__ == "__main__":
    raise SystemExit(main())
