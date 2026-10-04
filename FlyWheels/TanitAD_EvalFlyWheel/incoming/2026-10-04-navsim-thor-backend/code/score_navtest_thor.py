#!/usr/bin/env python3
# THOR VARIANT (2026-10-04, EvalFlyWheel navsim-thor-backend) of refcv7's
# ``…/2026-09-28-refcv7-standard-tests/navsim/code/score_navtest7.py``. W3's ``run_v1.py`` is STILL
# loaded by path and its ``cmd_score`` called UNCHANGED (byte-identical copy on Thor, sha256 recorded);
# only W3's module globals that name dev-box PATHS are re-pointed (each listed in ``THOR_GLOBALS``
# and written to ``<label>.thor.json``), CUDA_VISIBLE_DEVICES is "" (POSIX keeps an empty var), and the
# rasterio import stub joins PYTHONPATH. The dev-box RAM governor is not used: run this driver under
# ``nice -n 19 ionice -c3 setsid`` (children inherit both) and let the Thor pool's watchdog kill the
# recorded PIDs. Everything that decides a score is W3's, untouched.
# PORTED from the refcv6 NavSim suite: FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/score_navtest6.py (git blob 376474105b3b6c48a76b1adef066f58662e86e78 at 0c444082a08e) by code/port_from_refcv6.py; substitutions: [["if not a.label.startswith(\"r6\"):", "if not a.label.startswith(\"r7\"):", 1], ["⛔ labels must start with 'r6'", "⛔ labels must start with 'r7'", 1]]
"""Score ONE refcv7 seam on NAVSIM v1.1 ``navtest`` with W3's harness, IMPORTED -- on Thor.

    nice -n 19 ionice -c3 setsid python score_navtest_thor.py --label r7s50400_R7_A1 \
        --seam /dev/shm/navsim/seams/step50400/seam_R7_A1__navtest.npz --out /dev/shm/navsim/out/<dir> \
        [--tokens /dev/shm/navsim/subsets/A1_sub200_tokens.json]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
W3_CODE = "/home/nvidia/navsim-thor/code/w3"                        # THOR: byte-identical copy of W3's code
STUBS = "/home/nvidia/navsim-thor/code/stubs"                       # THOR: rasterio import stub
THOR_GLOBALS = {                                                    # THOR: every re-pointed W3 global
    "NV_PY": "/home/nvidia/venvs/navsim-cpu/bin/python",
    "V11_TREE": "/home/nvidia/navsim-thor/src/navsim-v11",
    "EXP": "/dev/shm/navsim/exp/w3_navtest_v1",
    "DATA": "/dev/shm/navsim/data/openscene",
    "NAVTEST_YAML": ("/home/nvidia/navsim-thor/src/navsim-v11/navsim/planning/script/config/common/"
                     "train_test_split/scene_filter/navtest.yaml"),
    "WRAP": os.path.join(HERE, "navsim_v1_thor.py"),
    "HERE": W3_CODE,                                                # PYTHONPATH entry for w3_agents_v1
}


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seam", default="")
    ap.add_argument("--official", default="", choices=("", "STOP", "CV"))
    ap.add_argument("--tokens", default=None, help="W3-format {'tokens', 'token_log'} subset")
    ap.add_argument("--out", required=True)                          # THOR: always explicit
    ap.add_argument("--agent-target", default="w3_agents_v1.SeamAgentV1",
                    help="THOR candidate: w3_agents_v1_ulp.SeamAgentV1ULP (reference-checked fingerprint)")
    ap.add_argument("--extra-override", action="append", default=[],
                    help="additional Hydra override(s), e.g. +agent.ulp_reference=<json>")
    a = ap.parse_args(argv)
    if not a.label.startswith("r7"):
        sys.exit("⛔ labels must start with 'r7' so the devkit scratch never collides with W3's")
    spec = importlib.util.spec_from_file_location("w3_run_v1", os.path.join(W3_CODE, "run_v1.py"))
    w3 = importlib.util.module_from_spec(spec)
    sys.modules["w3_run_v1"] = w3
    spec.loader.exec_module(w3)
    for k, v in THOR_GLOBALS.items():                               # THOR
        setattr(w3, k, v)
    w3.ENV = dict(w3.ENV, NUPLAN_MAPS_ROOT="/dev/shm/navsim/data/maps", NAVSIM_EXP_ROOT=w3.EXP,
                  NAVSIM_DEVKIT_ROOT=w3.V11_TREE, OPENSCENE_DATA_ROOT=w3.DATA,
                  CUDA_VISIBLE_DEVICES="")                           # THOR
    _orig_env_for = w3.env_for

    def env_for(hashseed: str = "1") -> dict:                       # THOR: + rasterio stub
        e = _orig_env_for(hashseed)
        e["PYTHONPATH"] = e["PYTHONPATH"] + os.pathsep + STUBS
        if os.environ.get("THOR_AUDIT_SITE"):        # file-open audit (code/audit_site), never in production
            e["PYTHONPATH"] = os.environ["THOR_AUDIT_SITE"] + os.pathsep + e["PYTHONPATH"]
        return e

    w3.env_for = env_for
    if a.agent_target != "w3_agents_v1.SeamAgentV1" or a.extra_override:      # THOR: candidate agent
        _orig_ao = w3.agent_overrides

        def agent_overrides(arm, call_log):
            ov, cls = _orig_ao(arm, call_log)
            ov = [f"agent._target_={a.agent_target}" if o == "agent._target_=w3_agents_v1.SeamAgentV1" else o
                  for o in ov] + list(a.extra_override)
            return ov, a.agent_target

        w3.agent_overrides = agent_overrides
    os.makedirs(a.out, exist_ok=True)
    w3.RAW = os.path.abspath(a.out)                  # where every artifact lands
    w3.PKG = PKG                                     # only used for a relpath in the counts
    if bool(a.seam) == bool(a.official):
        sys.exit("⛔ exactly one of --seam / --official")
    arm = a.official if a.official else f"SEAM:{os.path.abspath(a.seam)}"
    odir = os.path.join(os.path.abspath(a.out), a.label)
    # THOR: same idempotence rule as the dev box (never rescore = truncate a banked PASS), keyed by seam sha
    seam_sha = _sha(a.seam) if a.seam else None
    thor_path = os.path.join(odir, f"{a.label}.thor.json")
    if a.seam and os.environ.get("R7_FORCE_RESCORE") != "1":
        try:
            c = json.load(open(os.path.join(odir, f"{a.label}.counts.json"), encoding="utf-8"))
            t = json.load(open(thor_path, encoding="utf-8"))
            if c.get("status") == "PASS" and t.get("seam_sha256") == seam_sha \
                    and os.path.isfile(os.path.join(odir, f"{a.label}.csv")):
                print(json.dumps({"label": a.label, "status": "PASS", "skipped": "banked PASS, same seam"}))
                return 0
        except (OSError, ValueError):
            pass
    ns = types.SimpleNamespace(label=a.label, arm=arm,
                               tokens=a.tokens, cache_name="navtest", worker="sequential",
                               hashseed="1", record_poses=True, patch_loader=True)
    os.makedirs(odir, exist_ok=True)
    prov = {"backend": "thor", "driver": os.path.abspath(__file__), "driver_sha256": _sha(__file__),
            "w3_run_v1_sha256": _sha(os.path.join(W3_CODE, "run_v1.py")),
            "w3_agents_v1_sha256": _sha(os.path.join(W3_CODE, "w3_agents_v1.py")),
            "wrapper_sha256": _sha(THOR_GLOBALS["WRAP"]), "seam": a.seam, "seam_sha256": seam_sha,
            "tokens": a.tokens, "thor_globals": THOR_GLOBALS, "pid": os.getpid(),
            "agent_target": a.agent_target, "extra_overrides": list(a.extra_override),
            "ulp_agent_sha256": (_sha(os.path.join(HERE, "w3_agents_v1_ulp.py")) if "ulp" in a.agent_target else None),
            "ulp_guard_sha256": (_sha(os.path.join(HERE, "ulp_guard.py")) if "ulp" in a.agent_target else None)}
    json.dump(prov, open(thor_path, "w", encoding="utf-8"), indent=1)
    rc = w3.cmd_score(ns)
    return rc


if __name__ == "__main__":
    sys.exit(main())
