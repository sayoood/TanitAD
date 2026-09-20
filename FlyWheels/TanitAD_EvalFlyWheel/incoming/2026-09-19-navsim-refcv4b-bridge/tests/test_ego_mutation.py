#!/usr/bin/env python3
"""K5 / K6 — the ego-status ENFORCEMENT evidence (gate ``navsim.ego_enforcement``). TANITAD VENV, CPU.

One process, one model load; every comparison is between two forwards of the
SAME model in the SAME process, so byte-identity is a fair demand (checked
first: two identical calls must agree bit-for-bit — K0).

K5a  A1: randomise every EgoStatus field A1 does NOT declare (all ego_pose; the
     t-1..t-3 velocity / acceleration / command) -> poses BYTE-IDENTICAL.
K5b  A3: as K5a plus the t0 command (A3 does not declare it) -> BYTE-IDENTICAL.
K5c  A2: randomise EVERY ego field, t0 included -> BYTE-IDENTICAL (A2 declares none).
K6   the control that must go RED: mutate a DECLARED field (ego_velocity[t0] x 1.5)
     -> A1 poses must CHANGE on >= 90 % of scenes. Without it, a K5 "pass" could
     be a probe that cannot see anything.

Writes raw/K5_K6_ego_mutation.json.
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=D:/Projects/TanitAD/stack python tests/test_ego_mutation.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, "code"))
import tanitad_navsim_bridge as B  # noqa: E402

RAW = os.path.join(PKG, "raw")
BANK = "C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus"
CKPT = "D:/Projects/TanitAD-artifacts/refcv4b_final/ckpt_40284_FINAL.pt"


def _rand_status(rng, es: dict, keep_t0: tuple) -> dict:
    """A copy of one EgoStatus dict with every field NOT in keep_t0 randomised."""
    out = copy.deepcopy(es)
    for f in B.FIELDS:
        if f in keep_t0:
            continue
        n = len(es[f])
        if f == "driving_command":
            v = [0.0] * 4
            v[int(rng.integers(0, 4))] = 1.0
            out[f] = v
        else:
            out[f] = (rng.normal(0, 5, n)).tolist()
    return out


def mutate(statuses: list, rng, declared_t0: tuple) -> list:
    """Randomise every field of frames t-3..t-1, and at t0 every field whose name is
    not in ``declared_t0`` (field names without the [t0] suffix)."""
    out = [_rand_status(rng, es, ()) for es in statuses[:-1]]
    out.append(_rand_status(rng, statuses[-1], declared_t0))
    return out


def main(n_scenes: int = 20, seed: int = 20260919) -> dict:
    import torch
    if torch.cuda.is_available():
        sys.exit("⛔ CUDA visible — set CUDA_VISIBLE_DEVICES=-1")
    torch.set_num_threads(8)
    doc = json.load(open(os.path.join(RAW, "navsim_agent_inputs.json"), encoding="utf-8"))
    s2 = sorted(t for t, r in doc["tokens"].items() if r["stage"] == 2)
    rng = np.random.default_rng(seed)
    left = [t for t in s2 if int(np.argmax(doc["tokens"][t]["ego_statuses"][-1]["driving_command"])) != 1]
    pick = sorted(set(left[:6]) | set(rng.choice(s2, n_scenes - min(6, len(left)), replace=False)))
    bank = B.FrameBank.open(BANK)
    t0 = time.time()
    model, cfg, targs, prov, mod = B.load_refcv4b(CKPT)
    tr = mod.trainer()
    steps, W = int(prov["decoder_steps"]), int(prov["window"])

    def run(arm, statuses, frames):
        decl = B.declare(statuses, arm)
        rows = B.pack_frames(frames, B.slot_sources([-1.5, -1.0, -0.5, 0.0], B.ARMS[arm]["frames"], W))
        r = B.run_model(model, tr, rows, decl, steps)
        return B.knots_to_navsim(r["traj"]).astype(np.float32), r["traj"]

    res = {"n_scenes": len(pick), "tokens": pick, "seed": seed, "cases": {}}
    cases = {
        "K0_determinism_A1": ("A1_ego_cmd", None),
        "K5a_A1_undeclared": ("A1_ego_cmd", ("ego_velocity", "ego_acceleration", "driving_command")),
        "K5b_A3_undeclared": ("A3_ego_nocmd", ("ego_velocity", "ego_acceleration")),
        "K5c_A2_all_ego": ("A2_vision_pure", ()),
        "K6_A1_declared_v0x1p5": ("A1_ego_cmd", "SCALE_V0"),
    }
    base = {}
    for name, (arm, keep) in cases.items():
        same, n = 0, 0
        maxdiff = 0.0
        for tok in pick:
            r = doc["tokens"][tok]
            fr, _, _ = bank.load_wide(r["scene_token"])
            key = (arm, tok)
            if key not in base:
                base[key] = run(arm, r["ego_statuses"], fr)
            if keep is None:
                st = r["ego_statuses"]
            elif keep == "SCALE_V0":
                st = copy.deepcopy(r["ego_statuses"])
                st[-1]["ego_velocity"] = [1.5 * x for x in st[-1]["ego_velocity"]]
            else:
                st = mutate(r["ego_statuses"], rng, keep)
            p, _ = run(arm, st, fr)
            b = base[key][0]
            n += 1
            same += int(p.tobytes() == b.tobytes())
            maxdiff = max(maxdiff, float(np.abs(p.astype(np.float64) - b).max()))
        res["cases"][name] = {"arm": arm, "n": n, "byte_identical": same,
                              "changed": n - same, "max_abs_diff": maxdiff}
    c = res["cases"]
    res["verdict"] = {
        "K0_deterministic": c["K0_determinism_A1"]["byte_identical"] == c["K0_determinism_A1"]["n"],
        "K5a_pass": c["K5a_A1_undeclared"]["byte_identical"] == c["K5a_A1_undeclared"]["n"],
        "K5b_pass": c["K5b_A3_undeclared"]["byte_identical"] == c["K5b_A3_undeclared"]["n"],
        "K5c_pass": c["K5c_A2_all_ego"]["byte_identical"] == c["K5c_A2_all_ego"]["n"],
        "K6_can_fail": c["K6_A1_declared_v0x1p5"]["changed"] >= 0.9 * c["K6_A1_declared_v0x1p5"]["n"],
    }
    res["seconds"] = round(time.time() - t0, 1)
    return res


def test_ego_enforcement():
    v = main()["verdict"]
    assert all(v.values()), v


if __name__ == "__main__":
    out = main()
    B.json_dump(out, os.path.join(RAW, "K5_K6_ego_mutation.json"))
    print(json.dumps({"cases": out["cases"], "verdict": out["verdict"],
                      "seconds": out["seconds"]}, indent=1))
