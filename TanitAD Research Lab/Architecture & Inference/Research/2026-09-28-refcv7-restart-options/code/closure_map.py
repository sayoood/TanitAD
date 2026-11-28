#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv7 restart options, item 1 -- the CLOSURE MAP.

Reads the two BINDING closure records of the live run (pulled read-only from Thor 2026-09-28,
md5-verified; banked with clip ids redacted -- the module lists are untouched) and the launch
commit, and answers, for every candidate change, which launch-gate checks a restart would have to
RE-RUN rather than merely re-judge.

The rules are the gate's own (stack/scripts/launch_gate.py at the launch commit fec3a0d):
* the PASS TOKEN binds (profile, commit, tree_sha256, argv_sha256) + every data input
  (`verify_token`): ANY change to stack/ taniteval/ tools/ or to the argv needs a NEW gate run;
* the overfit PASS records bind (a) the LAUNCH argv sha256 (`_overfit_binding_reasons`,
  `judge_box_overfit`: "the record is bound to argv X, not the launch argv Y") and (b) the CODE
  CLOSURE by blob (`judge_closure`: "one changed blob refuses"), plus data sha256 and the venv;
* ⇒ a change touching a closure module OR the argv needs the binding RUNS re-done (exclusive Thor
  GPU), not just the record re-judged.

Usage: closure_map.py --gmo <gmo_closure.json> --gbo <gbo_closure.json> --git-dir <dir>
                      --commit <launch commit> --out <closure_map.json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

#: each candidate: the repo files it changes, and the argv edits it makes
CANDIDATES = {
    "freeze": {"what": "model-side freeze + declaration of G-LIVE's 10 admitted dead groups",
               "files": ["stack/tanitad/refs/refc_v3.py", "stack/tanitad/train/declared_vs_built.py"],
               "argv": []},
    "conflict50": {"what": "--conflict-every 10 -> 50 (instrument cadence)", "files": [],
                   "argv": [("--conflict-every", "10", "50")]},
    "lever2": {"what": "per-class 10 cm signal + box census on LOGGED steps only",
               "files": ["stack/scripts/refc_v3_train.py"], "argv": []},
    "getstate": {"what": "V2CompressedCache pickles newest_frame_only",
                 "files": ["stack/tanitad/data/v2_dataset.py"], "argv": []},
    "i3_patch": {"what": "refcv6_loader rebuilds old records at the STAMPED query count",
                 "files": ["stack/tanitad/eval/refcv6_loader.py", "stack/scripts/g_box_overfit.py",
                           "stack/tests/test_g_box_overfit.py"], "argv": []},
    "ckpt_off": {"what": "--map-hires-grad-ckpt on -> off", "files": [],
                 "argv": [("--map-hires-grad-ckpt", "on", "off")]},
    "eval1000": {"what": "--eval-every 500 -> 1000 (in-training monitor cadence)", "files": [],
                 "argv": [("--eval-every", "500", "1000")]},
    "tests_or_new_tool": {"what": "a NEW test / tool file under stack/ (no closure module edited)",
                          "files": ["stack/tests/<new file>"], "argv": []},
    "research_lab_only": {"what": "package / doc files outside stack/ taniteval/ tools/",
                          "files": ["TanitAD Research Lab/<...>"], "argv": []},
}
ALL_CHECKS = ["G-HYG", "G-DVB", "G-LIVE", "G-CKPT", "G-EVAL", "G-CLOCK", "G-SUITE",
              "G-SUITE-PINNED", "G-MAP-OVERFIT", "G-BOX-OVERFIT"]
TREE_ROOTS = ("stack/", "taniteval/", "tools/")


def blob(git_dir: str, commit: str, rel: str) -> str | None:
    p = subprocess.run(["git", f"--git-dir={git_dir}", "rev-parse", f"{commit}:{rel}"],
                       capture_output=True, text=True)
    s = p.stdout.strip()
    return s if len(s) == 40 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmo", required=True)
    ap.add_argument("--gbo", required=True)
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clo = {"map": json.loads(Path(a.gmo).read_text(encoding="utf-8")),
           "box": json.loads(Path(a.gbo).read_text(encoding="utf-8"))}
    mods = {k: {r: b for r, b in c["modules"]} for k, c in clo.items()}
    # every recorded blob must equal the launch commit's (else the live token could not exist)
    at_launch = {}
    for k, m in mods.items():
        diff = [r for r, b in m.items() if blob(a.git_dir, a.commit, r) != b]
        at_launch[k] = {"n_modules": len(m), "n_equal_at_launch": len(m) - len(diff),
                        "differ": diff}
    union = sorted(set(mods["map"]) | set(mods["box"]))
    membership = [{"path": r, "map": r in mods["map"], "box": r in mods["box"],
                   "blob": mods["map"].get(r) or mods["box"].get(r)} for r in union]
    res = {}
    for name, c in CANDIDATES.items():
        files = c["files"]
        in_map = [f for f in files if f in mods["map"]]
        in_box = [f for f in files if f in mods["box"]]
        tree_change = any(f.startswith(TREE_ROOTS) for f in files)
        argv_change = bool(c["argv"])
        token = tree_change or argv_change
        rebind_map = bool(in_map) or argv_change
        rebind_box = bool(in_box) or argv_change
        rerun = []
        if token:
            rerun = [ch for ch in ALL_CHECKS if ch not in ("G-MAP-OVERFIT", "G-BOX-OVERFIT")]
            rerun += ["G-MAP-OVERFIT (re-judge only)" if not rebind_map else
                      "G-MAP-OVERFIT (binding RUN re-done)",
                      "G-BOX-OVERFIT (re-judge only)" if not rebind_box else
                      "G-BOX-OVERFIT (binding RUN re-done)"]
        why = []
        if in_map or in_box:
            why.append(f"closure module(s) changed: map {in_map}, box {in_box}")
        if argv_change:
            why.append("argv sha256 changes: both overfit records bind the LAUNCH argv sha256 "
                       f"({clo['map'].get('argv', [])[clo['map']['argv'].index('--launch-argv-sha256') + 1][:12]}"
                       f"...) -- a record bound to another argv is refused")
        res[name] = {"what": c["what"], "files": files, "argv_edits": c["argv"],
                     "closure_map": in_map, "closure_box": in_box,
                     "closure_touching": bool(in_map or in_box),
                     "new_token_needed": token, "rebind_map_run": rebind_map,
                     "rebind_box_run": rebind_box, "checks_to_run": rerun, "why": why}
    out = {"schema": "refcv7-restart-options/closure-map/1", "launch_commit": a.commit,
           "closures": {k: {"script": c["script"], "argv_sha256_of_harness_argv": c["argv_sha256"],
                            "n_modules": c["n_modules"], "n_data": c["n_data"], "env": c["env"],
                            "elapsed_s": c["elapsed_s"], "written_utc": c["written_utc"],
                            "closure_sha256": c["closure_sha256"]}
                        for k, c in clo.items()},
           "bound_launch_argv_sha256": clo["map"]["argv"][clo["map"]["argv"].index(
               "--launch-argv-sha256") + 1],
           "at_launch": at_launch, "membership": membership,
           "only_in_box": [m["path"] for m in membership if m["box"] and not m["map"]],
           "only_in_map": [m["path"] for m in membership if m["map"] and not m["box"]],
           "candidates": res}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{'candidate':<20} {'closure':<9} {'token':<6} {'map run':<8} {'box run':<8}")
    for n, r in res.items():
        print(f"{n:<20} {str(r['closure_touching']):<9} {str(r['new_token_needed']):<6} "
              f"{str(r['rebind_map_run']):<8} {str(r['rebind_box_run']):<8}")
    print("at launch:", {k: (v["n_equal_at_launch"], v["n_modules"]) for k, v in at_launch.items()})
    print("only in box:", out["only_in_box"])
    print("only in map:", out["only_in_map"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
