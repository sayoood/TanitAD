"""Build nuPlan scenarios for NAVSIM **navtrain** frames, straight from the DB token list.

⛔ WHY A SECOND SCENARIO SOURCE EXISTS. `build_teacher_rollouts.py` reads its scenarios out of
CLOSED-LOOP SIM LOGS (`--run <dir>/**/*.msgpack.xz`) -- the artifact their val14 harness leaves
behind. **navtrain has no such runs**, so the paper's 100K-frame training split is unreachable by
that path. This module supplies the same `AbstractScenario` objects from `(log_name, token)`
alone, so the rollout code above it is used UNCHANGED. That is deliberate: one rollout
implementation is what makes the equivalence control worth running.

## ⭐ The design decision that is NOT a preference

A whole-log scenario (one `NuPlanScenario` per log, navtrain tokens as iteration indices) is the
obvious cheap shape -- it amortises every per-scenario cost over ~85 frames and mirrors the
existing builder's loop exactly. **It is wrong, and it fails silently.**

    NuPlanScenario.get_route_roadblock_ids():
        roadblock_ids = get_roadblock_ids_for_lidarpc_token_from_db(self._log_file,
                                                                   self._initial_lidar_token)

The route is keyed on the scenario's INITIAL token. A whole-log scenario's initial token is the
log's FIRST frame, so every navtrain frame in that log would hand the teacher the route from the
start of the log -- MEASURED median 2,300 iterations away, i.e. ~230 s of driving. The teacher
would roll out toward a goal belonging to a different part of the drive.
⚠️ Nothing downstream would catch it: the rollouts stay self-consistent, the ego and the camera
still describe the same logged pose, and every gate we own is about consistency, not about whether
the goal is the RIGHT goal. (Class: the R6/R11 family -- a parameter that lives somewhere else in
the NEXT construction.)

⇒ **one scenario per navtrain token**, `initial_lidar_token = token`. The route is then the route
the DB stored for that frame.

## Why the 2.0 s history still arrives

A per-token scenario starts AT the token, so iteration 0 has nothing before it *inside the
scenario*. That is fine, and it is worth stating why, because the 0.355 m fidelity miss came from
exactly this area:

    NuPlanScenario.get_ego_past_trajectory(iteration, ...) ->
        get_sampled_ego_states_from_db(self._log_file, self._lidarpc_tokens[iteration], ...,
                                       future=False)

It queries the DB BACKWARD from the token, independent of the extraction window. MEASURED over all
18,179 locally-resolvable navtrain frames: the minimum iteration index within its log is **30**, so
every frame has >= 2.0 s of real history and >= 4.0 s of tail. No clamped buffers, no padding.
"""
from __future__ import annotations

import glob
import os
import sqlite3
from pathlib import Path

# The devkit default (20.0 s) is kept rather than trimmed to the 4.0 s the rollout needs: the
# teacher's route-goal extraction looks `route_goal_horizon_s = 12.0` ahead, and a scenario cut to
# the rollout span would be a silent change to THEIR planner's input.
SCENARIO_DURATION_S = 20.0
# ⭐ SIMULATION RATE OF THE PER-FRAME ROLLOUTS. nuPlan's own scenario mapping subsamples the 20 Hz DB
# by 0.5 (`nuplan_scenario_mapping.yaml: subsample_ratio_override: 0.5`), so the teacher's published
# val14 closed-loop behaviour was produced at **10 Hz** -- while this module built its scenarios at
# `subsample_ratio=1.0`, i.e. the teacher replanned at 20 Hz (80 planner calls per 4 s instead of 40).
# REFE_SIM_HZ selects the rate; everything downstream derives its rows from `database_interval`
# (build_teacher_rollouts.geometry_for), so 10 Hz gives stride 2 / span 40 with no other change.
# Default 20 = every row banked before 2026-09-24. See diag_sim_rate.py for what 10 Hz changes.
SIM_HZ = int(os.environ.get("REFE_SIM_HZ", "20"))
if SIM_HZ not in (10, 20):
    raise ValueError(f"REFE_SIM_HZ={SIM_HZ}: only 20 (the DB rate) or 10 (nuPlan's sim rate)")
# ⛔ POD PORTABILITY: every default below was a dev-box Windows path, and two of them are reached on
# the navtrain path with NO override -- `build_scenarios_for_log` is never handed a map root, and
# `index_dbs()` is called bare by three modules. On a pod they resolve to nothing. Env first; the
# dev-box path stays the fallback so the running bank is untouched.
DEFAULT_YAML = os.environ.get("REFE_NAVTRAIN_YAML",
                              "D:/Archive/devbox-C/navsim/devkit/navsim/planning/script/config/common/"
                              "train_test_split/scene_filter/navtrain.yaml")
DEFAULT_DB_ROOT = os.environ.get("REFE_NUPLAN_DB_ROOT", "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")
DEFAULT_MAP_ROOT = os.environ.get("NUPLAN_MAPS_ROOT", "D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
MAP_VERSION = "nuplan-maps-v1.0"


def load_split(yaml_path: str = DEFAULT_YAML) -> tuple[list[str], set[str]]:
    """(log_names, tokens) of a NAVSIM split file. Tokens are a FLAT list with no log label."""
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    return list(cfg["log_names"] or []), {t.lower() for t in (cfg["tokens"] or [])}


def index_dbs(db_root: str = DEFAULT_DB_ROOT, splits=("mini", "test", "trainval")) -> dict[str, str]:
    """{log_name: absolute .db path} over the local nuPlan splits."""
    out: dict[str, str] = {}
    for s in splits:
        for p in glob.glob(os.path.join(db_root, s, "*.db")):
            out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p))
    # ⛔ AND a FLAT directory. `fetch_navtrain_dbs.py` writes every DB into one dir; the split-only
    # glob would have found ZERO of them on the pod and every navtrain log would have read as
    # "not held" -- a pipeline that runs, reports 0 frames, and exits cleanly.
    for p in glob.glob(os.path.join(db_root, "*.db")):
        out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p))
    return out


def resolve(yaml_path: str = DEFAULT_YAML, db_root: str = DEFAULT_DB_ROOT
            ) -> tuple[dict[str, list[str]], dict[str, int]]:
    """{log_name: [navtrain tokens present in that DB]} plus a coverage report.

    ⚠️ The split file does NOT say which log a token belongs to, so the pairing is done by reading
    each DB's `lidar_pc` table. Do not infer it positionally -- MEASURED len(tokens)=103,288 vs
    len(log_names)=1,192, so there is no positional pairing to infer.
    """
    logs, want = load_split(yaml_path)
    local = index_dbs(db_root)
    covered = sorted(set(logs) & set(local))
    per_log: dict[str, list[str]] = {}
    hit = 0
    for lg in covered:
        con = sqlite3.connect(f"file:{local[lg]}?mode=ro", uri=True)
        try:
            toks = [r[0] for r in con.execute(
                "SELECT lower(hex(token)) FROM lidar_pc ORDER BY timestamp")]
        finally:
            con.close()
        keep = [t for t in toks if t in want]
        if keep:
            per_log[lg] = keep
            hit += len(keep)
    report = {"split_logs": len(logs), "split_tokens": len(want), "local_dbs": len(local),
              "covered_logs": len(covered), "logs_with_frames": len(per_log), "frames": hit}
    return per_log, report


# ⛔⛔ THESE WERE CONSTANTS "IN ROWS AT 10 Hz" AND BOTH WERE HALF THE TRUTH. The raw nuPlan
# `lidar_pc` stream is **20 Hz** -- MEASURED dt 0.0500 s, `scenario.database_interval == 0.05` --
# while val14's closed-loop sim logs report 0.1 because THEIR SIMULATION SUBSAMPLES the database.
# So 2.0 s of history is **40 rows, not 20**, and 4.0 s of future is **80 rows, not 40**.
# The consequence was not cosmetic. Three of four generation shards died at ~23:07 on
#     AttributeError: 'NoneType' object has no attribute 'hex'   (lidar_pc.py:37, prev_token)
# because `get_past_tracked_objects(2.0 s)` walked back 40 rows from frames admitted at index >= 20
# and fell off the front of the log, where `prev_token` is NULL. The devkit's guard tests that the
# COLUMN exists, not that the VALUE is non-null.
# ⭐ Seconds are the requirement; rows are a derived quantity. Derive them from the log's own
# measured interval rather than encoding a rate.
HISTORY_S = 2.0     # `build_teacher_rollouts.BUFFER_S`
FUTURE_S = 4.0      # `build_teacher_rollouts.HORIZON_S`
HISTORY_ROWS = 40   # 2.0 s at the MEASURED 20 Hz; overridden per log by _rows_for()
FUTURE_ROWS = 80    # 4.0 s at the MEASURED 20 Hz; overridden per log by _rows_for()


def _rows_for(rows) -> tuple[int, int]:
    """(history_rows, future_rows) from a log's OWN median lidar_pc interval."""
    if len(rows) < 3:
        return HISTORY_ROWS, FUTURE_ROWS
    ts = [t for _tok, t in rows[:200]]
    d = sorted((ts[i + 1] - ts[i]) / 1e6 for i in range(len(ts) - 1))
    dt = d[len(d) // 2]
    if not (0.01 <= dt <= 0.5):
        return HISTORY_ROWS, FUTURE_ROWS
    return int(round(HISTORY_S / dt)), int(round(FUTURE_S / dt))


def _log_rows(db_path: str) -> tuple[str, list[tuple[str, int]]]:
    """(map_name, [(token, timestamp)] in time order) -- one pass, no devkit filtering."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        map_name = con.execute("SELECT map_version FROM log").fetchone()[0]
        rows = con.execute(
            "SELECT lower(hex(token)), timestamp FROM lidar_pc ORDER BY timestamp").fetchall()
    finally:
        con.close()
    return map_name, rows


def _scenario_types(db_path: str, tokens: list[str]) -> dict[str, str]:
    """{token: scenario_type} from `scenario_tag`, mirroring the devkit's `MAX(st.type)` pick.

    ⚠️ `MAX` is not a quality choice -- the devkit's own comment says "scenarios can have multiple
    tags, pick one arbitrarily". Reproduced exactly so a scenario built here carries the same label
    it would have carried through their path.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT lower(hex(lidar_pc_token)), MAX(type) FROM scenario_tag "
            "GROUP BY lidar_pc_token").fetchall()
    finally:
        con.close()
    want = set(tokens)
    return {t: ty for t, ty in rows if t in want and ty is not None}


def build_scenarios_for_log(db_path: str, tokens: list[str], map_root: str = DEFAULT_MAP_ROOT,
                            duration_s: float = SCENARIO_DURATION_S,
                            history_rows: int = HISTORY_ROWS, future_rows: int = FUTURE_ROWS):
    """Yield one `NuPlanScenario` per token, with REFe's OWN margin guard.

    ⛔⛔ THIS DELIBERATELY DOES NOT USE `get_scenarios_from_db`, THE DEVKIT'S OWN TOKEN QUERY, AND
    THE REASON IS A MEASURED 25.4 % SILENT DATA LOSS.

    That query restricts every result to `valid_scenes`:

        WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1   -- "at least 2 scenes before and 2 after"

    I first read that as cheap -- a nuPlan scene is commonly ~20 frames, so 4 scenes is ~80 frames
    of a multi-thousand-frame log. **It is not.** MEASURED on these DBs a scene is ~330-380
    lidar_pc rows, so the rule discards ~1,400 frames per log, and our navtrain DBs are short
    slices (median ~3,580 rows ~ 10 scenes):

        scenes in log     4      8      10     16
        kept              0/4    4/8    6/10   12/16
        i.e. the loss is 4 / n_scenes -- **worst on the SHORTEST logs**

    Aggregate over the 214 local navtrain logs: **13,565 of 18,179 frames survive (74.62 %)**, and
    **19 logs yield ZERO**. ⚠️ The loss is therefore BIASED, not random: it removes short logs
    preferentially, so the surviving bank would silently over-represent long continuous drives.
    Nothing downstream could have detected it -- the bank would simply have been smaller.

    ⭐ The rule is a coarse PROXY for "this frame has room for history and future". REFe's real
    requirement is exact and small: `BUFFER_S = 2.0 s` back and `SPAN = 40` steps forward. So the
    proxy is replaced by the requirement itself, asserted per frame against the row index. MEASURED:
    all 18,179 frames have iteration index >= 30 and a full 40-step tail, so the exact guard keeps
    **18,179 / 18,179** where the proxy kept 13,565.

    Camera availability is likewise NOT gated here. The devkit's `include_cameras` filter is
        INNER JOIN image AS img ON img.ego_pose_token = lp.ego_pose_token
    -- an EXACT ego_pose_token identity -- while `build_targets.camera_index` indexes each channel
    on its own timestamps and `build_teacher_rollouts` pairs all four independently within 60 ms,
    because (its own comment) "the cameras are NOT synchronised to a common clock". MEASURED:
        REFe's 4-camera <= 60 ms pairing   18,179 / 18,179  = 100.00 %
        devkit exact-pose join             17,785 / 18,179  =  97.83 %
    ⇒ the strict join would discard a further **394 usable frames** (~2.2 %; ~2,240 on the full
    103,288-frame split). The guard REFe needs already runs per frame downstream.
    """
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
        DEFAULT_SCENARIO_NAME, ScenarioExtractionInfo)

    data_root = str(Path(db_path).parent)
    veh = get_pacifica_parameters()
    map_name, rows = _log_rows(db_path)
    types = _scenario_types(db_path, list(tokens))
    pos = {t: i for i, (t, _) in enumerate(rows)}
    n = len(rows)
    # derived from THIS log's measured interval unless the caller pinned them explicitly
    if history_rows == HISTORY_ROWS and future_rows == FUTURE_ROWS:
        history_rows, future_rows = _rows_for(rows)
    for tok in tokens:
        i = pos.get(tok)
        if i is None:
            continue
        # the exact margin REFe needs, asserted on this frame rather than proxied by scene index.
        # ⛔ STRICTLY GREATER, and the off-by-one was measured, not reasoned. A 2.0 s history reads
        # rows [i-40, i]; if that span includes row 0, the devkit reads row 0's `prev_token`, which
        # is NULL, and raises. MEASURED on one log, same query at consecutive positions:
        #     pos 38/39/40 -> AttributeError ('NoneType' has no attribute 'hex')
        #     pos 41/42/43 -> OK, 40 samples
        # The first version of this guard used `i < history_rows`, which ADMITS i == 40 -- and 13
        # navtrain frames sit at exactly position 40. Any shard reaching one died.
        if i <= history_rows or i + future_rows >= n:
            continue
        stype = types.get(tok, DEFAULT_SCENARIO_NAME)
        yield NuPlanScenario(
            data_root=data_root,
            log_file_load_path=db_path,
            initial_lidar_token=tok,
            initial_lidar_timestamp=rows[i][1],
            scenario_type=stype,
            map_root=map_root,
            map_version=MAP_VERSION,
            map_name=map_name,
            # ⛔ offset 0.0: the scenario must START at the navtrain token, because the ROUTE is
            # keyed on `initial_lidar_token`. The history comes from the DB, not from the window.
            scenario_extraction_info=ScenarioExtractionInfo(
                scenario_name=stype, scenario_duration=duration_s,
                extraction_offset=0.0, subsample_ratio=SIM_HZ / 20.0),
            ego_vehicle_parameters=veh,
            sensor_root=None,
        )
