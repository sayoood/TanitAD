#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare two `numerics_ab.py` runs, parameter by parameter, step by step.

Verdict fields (all counts, never a summary adjective):
  * `n_steps` of each side and whether they agree;
  * for every parameter NOT under the excluded prefixes (default: the ten frozen modules): the
    number of (step, param) pairs whose PARAMETER bytes differ and whose GRADIENT bytes differ
    (grad state compared as none / zero:<sha> / nz:<sha>);
  * for every parameter UNDER an excluded prefix, per side: every step's grad state (must be
    `none` for the freeze to be bit-identical) and whether the value ever left its INIT digest;
  * AdamW state tensors compared by parameter NAME (the frozen side has none for the ten);
  * the run's own metrics.jsonl: every numeric key both sides logged at the same step, compared
    EXACTLY (`losses_differ`), with the key and step of the first difference.

Usage: compare_digests.py <A dir> <B dir> [--exclude-ten | --exclude-none] [--out verdict.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEN = ("core.strategic.gru", "core.strategic.proj", "nav_to_str", "str_goal_head", "gstr_embed",
       "gstr_film", "core.decoder.ctx_to_cond", "core.route_head", "core.decoder.lat_to_anchor",
       "core.decoder.lon_to_anchor")


def under(n: str, pre) -> bool:
    return any(n == p or n.startswith(p + ".") for p in pre)


def rows(d: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    p = d / "run" / "metrics.jsonl"
    if not p.is_file():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
        except Exception:                                  # noqa: BLE001
            continue
        if isinstance(r, dict) and isinstance(r.get("step"), int):
            out.setdefault(r["step"], {}).update(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--exclude-none", action="store_true")
    ap.add_argument("--out", default=None)
    x = ap.parse_args()
    A = json.loads((Path(x.a) / "digests.json").read_text(encoding="utf-8"))
    B = json.loads((Path(x.b) / "digests.json").read_text(encoding="utf-8"))
    excl = () if x.exclude_none else TEN
    v: dict = {"a": x.a, "b": x.b, "excluded": list(excl),
               "errors": [A.get("error"), B.get("error")],
               "n_steps": [A["n_opt_steps"], B["n_opt_steps"]]}
    n = min(A["n_opt_steps"], B["n_opt_steps"])
    names = sorted(set(A["init"]) & set(B["init"]))
    v["n_params_common"] = len(names)
    v["params_only_in"] = [sorted(set(A["init"]) - set(B["init"]))[:10],
                           sorted(set(B["init"]) - set(A["init"]))[:10]]
    live = [nm for nm in names if not under(nm, excl)]
    v["n_live_params_compared"] = len(live)
    v["init_differ"] = [nm for nm in live if A["init"][nm] != B["init"][nm]][:10]
    pdiff, gdiff, first = 0, 0, None
    for i in range(n):
        sa, sb = A["steps"][i], B["steps"][i]
        for nm in live:
            if sa["param"][nm] != sb["param"][nm]:
                pdiff += 1
                first = first or {"step": i, "param": nm}
            if sa["grad"][nm] != sb["grad"][nm]:
                gdiff += 1
    v["param_step_pairs_compared"] = n * len(live)
    v["param_bytes_differ"] = pdiff
    v["grad_bytes_differ"] = gdiff
    v["first_param_difference"] = first
    # the excluded (frozen) modules, per side
    for tag, D in (("a", A), ("b", B)):
        ex = [nm for nm in D["init"] if under(nm, TEN)]
        states = sorted({D["steps"][i]["grad"][nm].split(":")[0] for i in range(D["n_opt_steps"])
                         for nm in ex})
        moved = [nm for nm in ex if any(D["steps"][i]["param"][nm] != D["init"][nm]
                                        for i in range(D["n_opt_steps"]))]
        v[f"ten_{tag}"] = {"n_tensors": len(ex), "grad_states_seen": states,
                           "moved_from_init": moved,
                           "requires_grad": sorted({str(D["requires_grad"][nm]) for nm in ex}),
                           "in_optimizer": len([nm for nm in ex if nm in set(D["in_opt"])])}
    # optimiser state by name
    oa, ob = A.get("opt_state") or {}, B.get("opt_state") or {}
    common = sorted((set(oa) & set(ob)) - {nm for nm in set(oa) | set(ob) if under(nm, excl)})
    v["opt_state_compared"] = len(common)
    v["opt_state_differ"] = [nm for nm in common if oa[nm] != ob[nm]][:10]
    v["opt_entries"] = [len(oa), len(ob)]
    v["opt_state_nonempty_on_ten"] = [[nm for nm in oa if under(nm, TEN) and oa[nm]],
                                      [nm for nm in ob if under(nm, TEN) and ob[nm]]]
    # the run's own log rows
    ra, rb = rows(Path(x.a)), rows(Path(x.b))
    steps = sorted(set(ra) & set(rb))
    ndiff, nkeys, fd = 0, 0, None
    for s in steps:
        for k in sorted(set(ra[s]) & set(rb[s])):
            va, vb = ra[s][k], rb[s][k]
            if k in ("elapsed_s",) or k.endswith(("_s", "_ms")) or "time" in k:   # wall clocks
                continue
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                nkeys += 1
                if va != vb and not (va != va and vb != vb):
                    ndiff += 1
                    fd = fd or {"step": s, "key": k, "a": va, "b": vb}
    v["log_steps_common"] = steps
    v["log_values_compared"] = nkeys
    v["log_values_differ"] = ndiff
    v["first_log_difference"] = fd
    v["BIT_IDENTICAL_LIVE_TRAJECTORY"] = (v["n_steps"][0] == v["n_steps"][1] and n > 0
                                          and pdiff == 0 and gdiff == 0 and not v["init_differ"]
                                          and not v["opt_state_differ"])
    txt = json.dumps(v, indent=1)
    if x.out:
        Path(x.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
