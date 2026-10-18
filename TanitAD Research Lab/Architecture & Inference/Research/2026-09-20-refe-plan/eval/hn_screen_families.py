#!/usr/bin/env python3
"""P6-HN-S: the FOUR METRIC FAMILIES on RL-val (binding rule; PREREG_P6HN_SCREEN.md section 10; SCREEN_NOTES.md checklist item 6).  The families6 pattern (the instrument that produced
`sft4_B_u400_families.json`): `taniteval/adapters/navsim.py` `scenes_to_win(...)` -> `four_families_block(win, tier="T1")`, IMPORTED, never re-implemented, on the EXECUTED 8 NAVSIM poses of each arm's pick against the
LOGGED HUMAN FUTURE (RL-val scenes are navtrain scenes, which carry one).

    python eval/hn_screen_families.py --inputs families_inputs.npz --picks picks.npz --out families.json     # npz formats below

  inputs  .npz  tokens, logs (str [N]), v0 float [N] (ego speed, m/s), human float [N, 8, 3] (the logged future on the NAVSIM 0.5 s grid, ego frame), plans8 float [N, M, 8, 3] (the executed fan plans)
  picks   .npz  one int array [N] per scorer, named base, A0.., B0..  (rl_val_eval.select's pick, ties to the lowest index)

WHAT IT REPORTS.  Per scorer the adapter's own block (its own episode-cluster intervals, cluster unit = log_name).  The binding 'seed-mean paired difference on the section-6 draws' is exact for the PDM sub-scores
(EP, TTC, DAC) and the tactical selection quality, and those live in hn_screen_read.py on the joint draws.  For the TRAJECTORY-LEVEL components the instrument exposes aggregates and its own unpaired intervals only, so
the adverse-separation watch-items here use the established rule `proxy_eval.fam_adverse` (a component separates adversely when B's interval lies entirely on the worse side of A's; unpaired non-overlap, the conservative
reading RESULT_SFT4 used), applied to (a) each seed and (b) the SEED-MEAN BLOCK (every numeric leaf averaged across the seeds).  ⚠️ This is a stated deviation from 'paired'; per-window component values are not exported by the instrument.
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


def arm_block(ad, plans_pick, human, v0, logs, tokens, n_boot=2000):
    w = ad.scenes_to_win(plans_pick, human, frame="ego", origin_included=False, dt_s=0.5, scene_tokens=tokens, ego_speed_mps=v0, log_names=logs)
    return ad.four_families_block(w, tier="T1", n_boot=n_boot)


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


def build(ad, inputs, picks, n_boot=2000):
    toks, logs, v0, human, plans8 = ([str(t) for t in inputs["tokens"]], [str(x) for x in inputs["logs"]], np.asarray(inputs["v0"], np.float64),
                                     np.asarray(inputs["human"], np.float64), np.asarray(inputs["plans8"], np.float64))
    blocks = {}
    for name, pk in picks.items():
        try:
            blocks[name] = arm_block(ad, pick_plans(plans8, pk), human, v0, logs, toks, n_boot)
        except Exception as e:                                              # noqa: BLE001 -- reported per scorer, never dropped
            blocks[name] = {"status": "RAISED", "error": f"{type(e).__name__}: {e}"}
    seeds = sorted({int(n[1:]) for n in blocks if n[:1] in "AB" and n[1:].isdigit() and f"A{n[1:]}" in blocks and f"B{n[1:]}" in blocks})
    ok = [s for s in seeds if "status" not in blocks[f"A{s}"] and "status" not in blocks[f"B{s}"]]
    out = {"tool": TOOL, "tier": "T1-family on a SELECTION-tier lookup (the executed plan of the pick vs the logged human future; the loop is open)", "n": len(toks), "n_logs": len(set(logs)),
           "adapter": "taniteval/adapters/navsim.py (imported)", "blocks": blocks, "seeds": ok, "strategic": PAUSED,
           "not_computable": {"lead_distance_keeping": {"status": "UNAVAILABLE", "reason": "the adapter builds no lead block (needs the annotation stream)", "n": len(toks)},
                              "command_adherence_of_pick": {"status": "UNAVAILABLE", "reason": "the RL-val lookup carries no navigation command", "n": len(toks)}}}
    if ok:
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--taniteval-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    ad = load_adapter(a.taniteval_dir)
    z = np.load(a.inputs, allow_pickle=False)
    pk = np.load(a.picks, allow_pickle=False)
    out = build(ad, {k: z[k] for k in ("tokens", "logs", "v0", "human", "plans8")}, {k: pk[k] for k in pk.files}, a.n_boot)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    bad = [k for k, v in out["blocks"].items() if v.get("status") == "RAISED"]
    print("ZZHNFAM " + ("DONE" if not bad else "RAISED " + ",".join(bad)), flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
