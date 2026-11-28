#!/usr/bin/env python3
"""A FIXTURE test of a8_analyze.py -- synthetic seams / inputs / scores on the 422 registered token ids, written to a
SCRATCH dir (never the real RESULT). No model, no harness. Each case must produce its known decision:
  ADOPT      ON = OFF + 8 PDMS on every token (families identical -> no adverse separation)
  REFUTED    ON = OFF - 8
  NOTPROVEN  ON = OFF (+ zero-mean noise)
  GATE_C     ADOPT's data, but ONE token's OFF pose differs from its dumped pick -> gate (c) FAILS -> NOT PROVEN
  GATE_E     ADOPT's data, but ONE token's ego vector differs between the arms -> gate (e) FAILS -> NOT PROVEN
  GATE_A     ADOPT's data, but ONE token's ON diag is untriggered -> gate (a) FAILS -> NOT PROVEN
    python a8_analyze_fixture.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "eval"))
sys.path.insert(0, str(PKG / "refe"))
SIB_PLANNER = Path("C:/Users/Admin/AppData/Local/Temp/claude/D--Projects-TanitAD/91effc67-8c1e-4b66-9a63-341a3109dfc1/"
                   "scratchpad/pkg_a8/refe/planner.py")
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
        "comfort", "driving_direction_compliance"]


def build(root: Path, effect: float, mut: str | None, seed=0):
    import torch
    import planner as PL
    rng = np.random.default_rng(seed)
    fresh = json.load(open(HERE / "amendment8_fresh_set.json", encoding="utf-8"))
    census = json.load(open(HERE / "route_cover_census_navtest_full.json", encoding="utf-8"))["rows"]
    toks = fresh["fresh_tokens"]
    work, a8 = root / "work", root / "a8"
    for d in (work, a8 / "score", a8 / "families"):
        d.mkdir(parents=True, exist_ok=True)
    n = len(toks)
    props = rng.normal(size=(n, 64, 8, 3)).astype(np.float32)
    logits = rng.normal(size=(n, 64, 6)).astype(np.float32)
    stub = type("S", (), {"rule": "navsim_v1", "V1_W": PL.REFePlanner.V1_W, "PDM_W": PL.REFePlanner.PDM_W})()
    pick = np.array([int(PL.REFePlanner.aggregate(stub, torch.from_numpy(logits[i])[None])[0].argmax()) for i in range(n)])
    np.savez(work / "refe_a8_off_props.npz", token=np.array(toks), proposals=props, logits=logits, pick=pick,
             rule=np.array("navsim_v1"), repair_last_heading=np.array(True))
    off_poses = props[np.arange(n), pick].copy()
    if mut == "GATE_C":
        off_poses[0, 0, 0] += 1.0
    np.savez(work / "refe_a8_off.npz", token=np.array(toks), poses=off_poses)
    np.savez(work / "refe_a8_on.npz", token=np.array(toks), poses=off_poses + 0.5)
    rec = {"off": {}, "on": {}}
    for i, t in enumerate(toks):
        ego = [float(x) for x in rng.normal(size=7)]
        fr = [f"{t}_CAM_{c}" for c in "FLRB"]
        rec["off"][t] = {"ego": ego, "goal": [400.0, 0.0, 430.0, 0.0], "frames": fr, "goal_diag": None, "pick": int(pick[i])}
        rec["on"][t] = {"ego": list(ego), "goal": [30.0, 0.0, 60.0, 0.0], "frames": list(fr), "pick": 0,
                        "goal_diag": {"triggered": True, "ego_to_route_m": census[t]["ego_to_route_m"], "fallback": "lane"}}
    if mut == "GATE_E":
        rec["on"][toks[1]]["ego"][0] += 1e-3
    if mut == "GATE_A":
        rec["on"][toks[2]]["goal_diag"]["triggered"] = False
    for arm in ("off", "on"):
        json.dump(rec[arm], open(a8 / f"refe_a8_{arm}.inputs.json", "w"))
        if arm == "off":
            off_sc = sc = rng.uniform(0.3, 0.7, size=n)
        else:                                          # an EXACTLY zero-mean spread (no clip): the effect is `effect`
            d = rng.normal(0, 0.03, n)
            sc = off_sc + effect / 100.0 + (d - d.mean())
        lines = ["token," + "score," + ",".join(SUBS) + ",valid"]
        for i, t in enumerate(toks):
            lines.append(f"{t},{sc[i]:.6f}," + ",".join(f"{sc[i]:.6f}" for _ in SUBS) + ",True")
        (a8 / "score" / f"refe_a8_{arm}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        st = {"label": f"refe_a8_{arm}", "status": "PASS", "csv_valid_rows": n, "C1_max_abs_delta": 0.0}
        (a8 / "score" / f"refe_a8_{arm}.log").write_text(json.dumps(st) + "\n", encoding="utf-8")
        fam = PKG / "raw" / "2026-09-28-m6b-tangent" / "families" / "refe_m6_Wt0_on.json"
        shutil.copy2(fam, a8 / "families" / f"refe_a8_{arm}.json")
    return root


def main() -> int:
    import hashlib
    sib_sha = hashlib.sha256(SIB_PLANNER.read_bytes()).hexdigest()
    cases = {"ADOPT": (8.0, None, "ADOPT"), "REFUTED": (-8.0, None, "REFUTED"), "NOTPROVEN": (0.0, None, "NOT PROVEN"),
             "GATE_C": (8.0, "GATE_C", "NOT PROVEN"), "GATE_E": (8.0, "GATE_E", "NOT PROVEN"),
             "GATE_A": (8.0, "GATE_A", "NOT PROVEN")}
    tmp = Path(tempfile.mkdtemp(prefix="a8_fixture_"))
    out = {}
    for name, (eff, mut, want) in cases.items():
        root = build(tmp / name, eff, mut)
        p = subprocess.run([sys.executable, str(HERE / "a8_analyze.py"), "--work", str(root / "work"), "--a8",
                            str(root / "a8"), "--out", str(root / "out"), "--selftest",
                            str(HERE / "a8_selftest_sibling.json"), "--tested-planner", sib_sha,
                            "--landed-planner", str(SIB_PLANNER)], capture_output=True, text=True, cwd=str(PKG / "eval"))
        r = json.load(open(root / "out" / "result_a8.json")) if (root / "out" / "result_a8.json").exists() else {}
        got = r.get("verdict")
        out[name] = {"want": want, "got": got, "ok": got == want, "rc": p.returncode,
                     "gates": {g: v["ok"] for g, v in (r.get("gates") or {}).items()},
                     "why": r.get("verdict_reason"), "stderr_tail": p.stderr[-600:] if p.returncode else ""}
        print(f"  [{'PASS' if got == want else 'FAIL'}] {name}: want {want}, got {got} gates {out[name]['gates']} "
              f"{out[name]['stderr_tail']}")
    json.dump(out, open(HERE / "a8_analyze_fixture.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    ok = all(v["ok"] for v in out.values())
    print("ZZA8_ANALYZE_FIXTURE_" + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
