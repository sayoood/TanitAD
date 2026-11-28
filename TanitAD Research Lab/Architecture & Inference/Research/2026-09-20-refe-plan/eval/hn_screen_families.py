#!/usr/bin/env python3
"""P6-HN-S: the FOUR METRIC FAMILIES on RL-val (binding rule; PREREG_P6HN_SCREEN.md section 10; SCREEN_NOTES.md checklist item 6).  The families6 pattern (the instrument that produced
`sft4_B_u400_families.json`): `taniteval/adapters/navsim.py` `scenes_to_win(...)` -> `four_families_block(win, tier="T1")`, IMPORTED, never re-implemented, on the EXECUTED 8 NAVSIM poses of each arm's pick against the
LOGGED HUMAN FUTURE (RL-val scenes are navtrain scenes, which carry one).

    python eval/hn_screen_families.py --inputs families_inputs.npz --picks picks.npz --out families.json --per-set-out families_per_set.npz     # npz formats below

  inputs  .npz  tokens, logs (str [N]), v0 float [N] (ego speed, m/s), human float [N, 8, 3] (the logged future on the NAVSIM 0.5 s grid, ego frame), plans8 float [N, M, 8, 3] (the executed fan plans)
  picks   .npz  one int array [N] per scorer, named base, A0.., B0..  (rl_val_eval.select's pick, ties to the lowest index)

WHAT IT REPORTS.  Per scorer the adapter's own block (its own single-arm episode-cluster intervals, cluster unit = log_name), as before, PLUS (amendment B4, the registered pairing) the OPT-IN PER-SET EXPORT of the
instrument: `four_families_block(..., per_window=True)` (taniteval/per_window.py) returns, for every set, each trajectory-level component's per-set value (LONGITUDINAL speed / along-track MAE and bias, target-speed bands,
accel, ego progress; LATERAL heading, curvature, yaw rate [masked at standstill, RETR-2026-09-26-YAWMASK], cross-track, last-pose heading; TACTICAL the class ids of the executed lateral / longitudinal / 5-way decisions, so the
correctness flags and kappa can be recomputed on any resample).  This script writes that export, for every arm, to `--per-set-out` (npz, exact float64, one row per set, with the set's log and token).
THE PAIRED READ LIVES IN hn_screen_read.py: it aligns the export to the PDM rows by (log, token) and takes the seed-mean paired difference B_s - A_s on the SAME joint log draws (10,000, seed 20260927) as the PDM sub-scores.
This script no longer applies the unpaired non-overlap rule (`proxy_eval.fam_adverse`) when the per-set export is on: B4 says that deviation is NOT adopted.  Called without the export (`build(per_set=False)`, kept for the
adapter checks) it still returns the legacy unpaired reading, stamped `adverse_reading.paired = False`; `hn_screen_read` will not publish families from it.
These trajectory-level components are REPORTED, NOT GATING (section 10).  Families that cannot be computed are reported with their reason and n: LEAD-AGENT distance keeping (the lead block is UNAVAILABLE in the adapter),
command adherence of the pick (RL-val inputs carry no navigation command).  STRATEGIC is PAUSED (PI 2026-09-19, PI_DECISION_QUEUE.md item 25): stamped, not a pass, not omitted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
for _p in (HERE, os.path.join(PKG, "refe")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TOOL = "hn_screen_families/1"
PAUSED = {"status": "PAUSED", "ruling": "PI 2026-09-19, PI_DECISION_QUEUE.md item 25", "note": "reported as PAUSED: not a pass, not omitted"}


def repo_root():
    """the repo root of this checkout (the file sits at <root>/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/)"""
    return os.path.abspath(os.path.join(PKG, *([".."] * 4)))


def load_adapter(taniteval_dir=None):
    """taniteval/adapters/navsim.py by file, as families6.py does; `taniteval_dir` is the directory that holds `adapters/` and the `taniteval` package"""
    d = taniteval_dir or os.environ.get("TANITEVAL_DIR") or os.path.join(repo_root(), "taniteval")
    if d not in sys.path:
        sys.path.append(d)
    spec = importlib.util.spec_from_file_location("navsim_adapter_hn", os.path.join(d, "adapters", "navsim.py"))
    ad = importlib.util.module_from_spec(spec)
    sys.modules["navsim_adapter_hn"] = ad
    spec.loader.exec_module(ad)
    return ad


def pick_plans(plans8, picks):
    p = np.asarray(plans8, np.float64)
    return p[np.arange(len(picks)), np.asarray(picks, np.int64)]


def arm_block(ad, plans_pick, human, v0, logs, tokens, n_boot=2000, per_window=False):
    w = ad.scenes_to_win(plans_pick, human, frame="ego", origin_included=False, dt_s=0.5, scene_tokens=tokens, ego_speed_mps=v0, log_names=logs)
    return ad.four_families_block(w, tier="T1", n_boot=n_boot, **({"per_window": True} if per_window else {}))        # the flag is passed ONLY when asked for: the default call is the one that always existed


def seed_mean_block(blocks):
    """average every numeric leaf that all blocks carry (means and interval ends alike); non-numeric leaves come from the first block"""
    def walk(objs):
        o0 = objs[0]
        if isinstance(o0, dict):
            return {k: walk([o[k] for o in objs]) if all(isinstance(o, dict) and k in o for o in objs) else o0[k] for k in o0}
        if all(isinstance(o, (int, float)) and not isinstance(o, bool) and np.isfinite(o) for o in objs):
            return float(np.mean(objs))
        return o0
    return walk(blocks)


def adverse(fa, fb):
    """B against A by proxy_eval.fam_adverse (the landed rule), on the envelope {'families': {'arm': block}}"""
    import proxy_eval as PE
    return PE.fam_adverse({"families": {"arm": fa}}, {"families": {"arm": fb}}, arm_key="arm")


def build(ad, inputs, picks, n_boot=2000, per_set=False):
    """-> the families json.  With `per_set=True` the per-set export of every arm is returned under `out["_per_set_export"]` ({arm: export}); the caller writes it with `write_per_set` and pops the key."""
    toks, logs, v0, human, plans8 = ([str(t) for t in inputs["tokens"]], [str(x) for x in inputs["logs"]], np.asarray(inputs["v0"], np.float64),
                                     np.asarray(inputs["human"], np.float64), np.asarray(inputs["plans8"], np.float64))
    blocks, exports = {}, {}
    for name, pk in picks.items():
        try:
            blocks[name] = arm_block(ad, pick_plans(plans8, pk), human, v0, logs, toks, n_boot, per_window=per_set)
            if per_set:
                exports[name] = blocks[name].pop("_per_window")                  # the block itself is then exactly the default one
        except Exception as e:                                              # noqa: BLE001 -- reported per scorer, never dropped
            blocks[name] = {"status": "RAISED", "error": f"{type(e).__name__}: {e}"}
    seeds = sorted({int(n[1:]) for n in blocks if n[:1] in "AB" and n[1:].isdigit() and f"A{n[1:]}" in blocks and f"B{n[1:]}" in blocks})
    ok = [s for s in seeds if "status" not in blocks[f"A{s}"] and "status" not in blocks[f"B{s}"]]
    out = {"tool": TOOL, "tier": "T1-family on a SELECTION-tier lookup (the executed plan of the pick vs the logged human future; the loop is open)", "n": len(toks), "n_logs": len(set(logs)),
           "adapter": "taniteval/adapters/navsim.py (imported)", "blocks": blocks, "seeds": ok, "strategic": PAUSED,
           "not_computable": {"lead_distance_keeping": {"status": "UNAVAILABLE", "reason": "the adapter builds no lead block (needs the annotation stream)", "n": len(toks)},
                              "command_adherence_of_pick": {"status": "UNAVAILABLE", "reason": "the RL-val lookup carries no navigation command", "n": len(toks)}}}
    if per_set:
        out["_per_set_export"] = exports
        out["per_set"] = {"requested": True, "arms": {n: {"status": e.get("status"), "n": e.get("n"), "grid": e.get("grid"), "components": {f: sorted(e.get(f, {})) for f in ("longitudinal", "lateral")},
                                                           "unavailable": sorted(e.get("unavailable", {})), "tactical": (e.get("tactical") or {}).get("status")} for n, e in sorted(exports.items())},
                          "file": None}
        out["adverse_reading"] = {"paired": True, "where": "hn_screen_read.py: seed-mean paired B_s - A_s differences on the joint log draws of the PDM sub-scores (amendment B4)",
                                  "unpaired_legacy": "NOT computed when the per-set export is on"}
    else:
        out["adverse_reading"] = {"paired": False, "status": "UNPAIRED LEGACY READING -- amendment B4 does NOT adopt it; hn_screen_read will not publish families from this file (no per-set export)"}
    if ok and not per_set:
        per_seed = {str(s): adverse(blocks[f"A{s}"], blocks[f"B{s}"]) for s in ok}
        sm_a = seed_mean_block([blocks[f"A{s}"] for s in ok])
        sm_b = seed_mean_block([blocks[f"B{s}"] for s in ok])
        sm = adverse(sm_a, sm_b)
        out.update(adverse_per_seed=per_seed, adverse_seed_mean=sm,
                   watch_items=sorted(f"{x['family']}.{x['metric']}" for x in sm.get("adverse", [])),
                   watch_items_per_seed_count={m: sum(any(f"{x['family']}.{x['metric']}" == m for x in v.get("adverse", [])) for v in per_seed.values())
                                               for m in sorted({f"{x['family']}.{x['metric']}" for v in per_seed.values() for x in v.get("adverse", [])})},
                   note="trajectory-level components: reported, NOT gating; unpaired non-overlap on the seed-mean block (see module docstring)")
    first = next((b for b in blocks.values() if "status" not in b), None)
    out["families_present"] = {k: ({"status": "OK"} if (first is not None and k != "strategic") else PAUSED if k == "strategic" else {"status": "UNAVAILABLE"}) for k in ("longitudinal", "lateral", "tactical", "strategic")}
    out["families_present"]["strategic"] = PAUSED
    return out


# ================================================================================================ the per-set file (npz, exact float64; numpy only -- hn_screen_read loads it without torch or taniteval)
PER_SET_SCHEMA = "hn_screen_per_set/1"
_SEP = "__"


def _flatten(exp, prefix):
    """one arm's export -> ({key: ndarray}, json-able spec).  Keys: `<arm>__<family>.<component>__<field>`; tactical: `<arm>__tactical.<decision>__gt|pred`."""
    arrays, spec = {}, {"schema": exp.get("schema"), "n": exp.get("n"), "status": exp.get("status"), "grid": exp.get("grid"), "dt_s": exp.get("dt_s"), "unavailable": exp.get("unavailable", {}),
                        "strategic": exp.get("strategic"), "components": {}, "tactical": None}
    if exp.get("reason"):
        spec["reason"] = exp["reason"]
    for fam in ("longitudinal", "lateral"):
        for name, c in (exp.get(fam) or {}).items():
            key = f"{fam}.{name}"
            spec["components"][key] = {k: v for k, v in c.items() if not isinstance(v, np.ndarray)}
            for k, v in c.items():
                if isinstance(v, np.ndarray):
                    arrays[f"{prefix}{key}{_SEP}{k}"] = v
    t = exp.get("tactical")
    if isinstance(t, dict) and t.get("status") == "OK":
        ts = {"status": "OK", "source": t.get("source"), "decisions": {}, "goal_setting": {}}
        for d in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
            ts["decisions"][d] = {"classes": list(t[d]["classes"])}
            arrays[f"{prefix}tactical.{d}{_SEP}gt"], arrays[f"{prefix}tactical.{d}{_SEP}pred"] = np.asarray(t[d]["gt"]), np.asarray(t[d]["pred"])
        for name, c in t["goal_setting"].items():
            ts["goal_setting"][name] = {k: v for k, v in c.items() if not isinstance(v, np.ndarray)}
            for k, v in c.items():
                if isinstance(v, np.ndarray):
                    arrays[f"{prefix}tactical.goal_setting.{name}{_SEP}{k}"] = v
        spec["tactical"] = ts
    else:
        spec["tactical"] = t
    if exp.get("tokens") is not None:
        arrays[f"{prefix}tokens"] = np.asarray(exp["tokens"], dtype=str)
    if exp.get("eid") is not None:
        arrays[f"{prefix}eid"] = np.asarray(exp["eid"], dtype=str)
    return arrays, spec


def _unflatten(z, spec, prefix):
    names = set(z.files)
    out = {k: spec.get(k) for k in ("schema", "n", "status", "grid", "dt_s", "strategic")}
    out.update(unavailable=spec.get("unavailable", {}), longitudinal={}, lateral={}, tokens=None, eid=None)
    if spec.get("reason"):
        out["reason"] = spec["reason"]
    if f"{prefix}tokens" in names:
        out["tokens"] = [str(x) for x in z[f"{prefix}tokens"]]
    if f"{prefix}eid" in names:
        out["eid"] = [str(x) for x in z[f"{prefix}eid"]]

    def comp(base, rec):
        c = dict(rec)
        for k in ("value", "num", "den", "rows"):
            if f"{base}{_SEP}{k}" in names:
                c[k] = np.asarray(z[f"{base}{_SEP}{k}"])
        return c

    for key, rec in spec.get("components", {}).items():
        fam, _, name = key.partition(".")
        out[fam][name] = comp(f"{prefix}{key}", rec)
    ts = spec.get("tactical")
    if isinstance(ts, dict) and ts.get("status") == "OK":
        t = {"status": "OK", "source": ts.get("source"), "goal_setting": {n: comp(f"{prefix}tactical.goal_setting.{n}", r) for n, r in ts["goal_setting"].items()}}
        for d, meta in ts["decisions"].items():
            t[d] = {"classes": meta["classes"], "gt": np.asarray(z[f"{prefix}tactical.{d}{_SEP}gt"]), "pred": np.asarray(z[f"{prefix}tactical.{d}{_SEP}pred"])}
        out["tactical"] = t
    else:
        out["tactical"] = ts
    return out


def write_per_set(path, exports):
    """{arm: export} -> one npz.  Arm names may not contain the separator."""
    arrays, specs = {}, {}
    for arm, e in sorted(exports.items()):
        assert _SEP not in arm, f"arm name {arm!r} contains {_SEP!r}"
        a, sp = _flatten(e, f"{arm}{_SEP}")
        arrays.update(a)
        specs[arm] = sp
    arrays["__spec__"] = np.asarray(json.dumps({"schema": PER_SET_SCHEMA, "arms": specs}, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    np.savez(path, **arrays)


def read_per_set(path):
    """the npz -> {arm: export}"""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__spec__"]))
    if meta.get("schema") != PER_SET_SCHEMA:
        raise ValueError(f"per-set file schema {meta.get('schema')!r} != {PER_SET_SCHEMA!r}")
    return {arm: _unflatten(z, sp, f"{arm}{_SEP}") for arm, sp in meta["arms"].items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-set-out", required=True, help="npz of the per-set component values of every arm (taniteval.per_window); hn_screen_read pairs them on the joint log draws")
    ap.add_argument("--taniteval-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    ad = load_adapter(a.taniteval_dir)
    z = np.load(a.inputs, allow_pickle=False)
    pk = np.load(a.picks, allow_pickle=False)
    out = build(ad, {k: z[k] for k in ("tokens", "logs", "v0", "human", "plans8")}, {k: pk[k] for k in pk.files}, a.n_boot, per_set=True)
    exports = out.pop("_per_set_export")
    write_per_set(a.per_set_out, exports)
    out["per_set"]["file"] = os.path.basename(a.per_set_out)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    bad = [k for k, v in out["blocks"].items() if v.get("status") == "RAISED"]
    print("ZZHNFAM " + ("DONE" if not bad else "RAISED " + ",".join(bad)), flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
