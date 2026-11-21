"""D6 common loaders -- refcv7 NavSim failure anatomy (refcv8 data audit, 2026-10-04).

CPU only, light.  Reads ONLY banked artifacts:
  PKG   = FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim
  INPUT = the navhard agent-inputs export (per-token stage / log / driving command / ego status)
Every loader asserts the SHAPE of what it read (a count of 0 from a file that could not be read is not
absence).  No GPU, no torch.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pandas as pd

PKG = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
       "2026-09-28-refcv7-standard-tests/navsim")
INPUTS = ("C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/"
          "navsim_agent_inputs.json")
NAVSIM_CI = "D:/Projects/TanitAD/taniteval/adapters/navsim_ci.py"
SUB = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
       "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
       "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
       "LK": "lane_keeping", "HC": "history_comfort"}
MULT = ("NC", "DAC", "DDC", "TLC")
WEIGHTED = {"EP": 5.0, "TTC": 5.0, "LK": 2.0, "HC": 2.0, "EC": 2.0}
CMD_NAME = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT", 3: "UNKNOWN"}   # NavSim one-hot order; verified, see d6_controls


def milestone_dir(step: int) -> str:
    return f"{PKG}/raw/milestones/step{step}"


def floors_dir() -> str:
    return f"{PKG}/raw/floors/navhard_two_stage"


def load_ci():
    spec = importlib.util.spec_from_file_location("navsim_ci_d6", NAVSIM_CI)
    m = importlib.util.module_from_spec(spec)
    sys.modules["navsim_ci_d6"] = m
    spec.loader.exec_module(m)
    return m


def load_inputs() -> dict:
    d = json.load(open(INPUTS, encoding="utf-8"))
    toks = d["tokens"]
    assert len(toks) == 5912 and d["n_stage1"] == 450 and d["n_stage2"] == 5462, "inputs shape"
    out = {}
    for t, r in toks.items():
        es = r["ego_statuses"][-1]
        dc = np.asarray(es["driving_command"], dtype=float)
        assert dc.sum() == 1.0, f"driving_command not one-hot for a token: {dc}"
        vel = es["ego_velocity"]
        out[t] = {"stage": int(r["stage"]), "log_name": r["log_name"], "map": r["map_name"],
                  "cmd": int(np.argmax(dc)), "v0_inputs": float(math.hypot(vel[0], vel[1])),
                  "orig_init": r.get("corresponding_original_initial_token"),
                  "scene_token": r["scene_token"],
                  "cv_poses": r.get("cv_poses"),
                  "human_future_poses": r.get("human_future_poses")}
    return out, d["reactive_all_mapping"]


def _frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert len(df) == 5912 and df["token"].is_unique, f"{path}: {len(df)} rows"
    return df.set_index("token")


def load_arm(step: int, arm: str, with_hooks: bool = True) -> pd.DataFrame:
    """One model arm at one milestone: sub-scores, weight, score (the official per-token EPDMS with
    the two-frame EC injected) from the PRE-CSV frame; plan / human poses / v0 from the hooks."""
    base = f"{milestone_dir(step)}/scores_navhard/score_{arm}__navhard_two_stage_wrapper"
    tag = f"{arm}__navhard_two_stage"
    df = _frame(f"{base}/{tag}_final_scores_frame.csv")
    df = df.rename(columns={v: k for k, v in SUB.items()})
    df = df.rename(columns={"two_frame_extended_comfort": "EC"})
    if with_hooks:
        h = json.load(open(f"{base}/{tag}_hooks.json", encoding="utf-8"))["pdm_score_calls"]
        assert len(h) == 5912, "hooks shape"
        plan, hum, v0, rowd = {}, {}, {}, {}
        for c in h:
            plan[c["token"]] = np.asarray(c["agent_poses"], dtype=np.float64)
            # ORIGINAL (stage-1) scenes bank the human future; SYNTHETIC (stage-2) scenes bank NONE (null)
            hum[c["token"]] = (np.asarray(c["human_poses"], dtype=np.float64)
                               if c["human_poses"] is not None else None)
            v0[c["token"]] = float(c["v0_mps"])
            rowd[c["token"]] = c["row"]
            assert c["agent_sampling"] == [8, 0.5]
            assert (c["human_poses"] is None) == (c["scene_type"] == "SceneFrameType.SYNTHETIC")
        df["plan"] = pd.Series(plan)
        df["human"] = pd.Series(hum, dtype=object)
        df["v0"] = pd.Series(v0)
        assert df["plan"].notna().all()
    return df


def load_floor(name: str) -> pd.DataFrame:
    """STOP_zero / CV_official / ECHO_ha0_ext frames (no hooks banked for floors)."""
    p = f"{floors_dir()}/score_{name}__navhard_two_stage_wrapper/{name}__navhard_two_stage_final_scores_frame.csv"
    df = _frame(p)
    df = df.rename(columns={v: k for k, v in SUB.items()})
    df = df.rename(columns={"two_frame_extended_comfort": "EC"})
    return df


def official_epdms(df: pd.DataFrame, mapping, ci, col: str = "score") -> float:
    rows = {t: {"score": float(r[col]), "weight": float(r["weight"])} for t, r in df[[col, "weight"]].iterrows()}
    c = ci.two_stage_key_contributions(rows, mapping, "score")
    return ci.official_aggregate(c)


def heading_wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
