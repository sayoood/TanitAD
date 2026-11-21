"""Add an opt-in --goal-min-speed to reroll_twins.py and long_rollout.py (addendum 5).  argv: <dir holding both>.
Unset => byte-identical behaviour (the config's own route_goal_min_speed_mps, captured once at startup, is restored for every twin).
'v0max' => floor = max(the twin's frame speed ego[6], the config default): the goal can no longer move closer as the ego slows."""
import sys
from pathlib import Path
D = Path(sys.argv[1])

def sub(name, pairs):
    p = D / name
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert t.count(old) == 1, (name, t.count(old), old[:70])
        t = t.replace(old, new)
    p.write_bytes(t.encode("utf-8"))

MS = ('def _min_speed(spec, ego, ms0):\n'
      '    """None -> the config default; a float -> that floor; \'v0max\' -> max(frame speed, default) (addendum 5)."""\n'
      '    if spec is None:\n'
      '        return ms0\n'
      '    if spec == "v0max":\n'
      '        return max(float(ego[6]), ms0)\n'
      '    return float(spec)\n\n\n')

sub("long_rollout.py", [
    ('def main() -> int:\n', MS + 'def main() -> int:\n'),
    ('    ap.add_argument("--goal-horizon", type=float, default=12.0)\n',
     '    ap.add_argument("--goal-horizon", type=float, default=12.0)\n'
     '    ap.add_argument("--goal-min-speed", default=None, help="LEVER (addendum 5): float m/s or v0max; unset = the config default")\n'),
    ('    planner = BR.build_planner("cpu")\n',
     '    planner = BR.build_planner("cpu")\n'
     '    MS0 = float(planner._feature_builder.config.route_goal_min_speed_mps)\n'),
    ('            planner._feature_builder.config.route_goal_horizon_s = a.goal_horizon\n',
     '            planner._feature_builder.config.route_goal_horizon_s = a.goal_horizon\n'
     '            rec["goal_min_speed"] = planner._feature_builder.config.route_goal_min_speed_mps = _min_speed(a.goal_min_speed, r["ego"], MS0)\n'),
])
sub("reroll_twins.py", [
    ("def cmd_run(a) -> int:\n", MS + "def cmd_run(a) -> int:\n"),
    ("    planner = BR.build_planner(a.device)\n",
     "    planner = BR.build_planner(a.device)\n"
     "    MS0 = float(planner._feature_builder.config.route_goal_min_speed_mps)\n"),
    ('            planner._feature_builder.config.route_goal_horizon_s = float(a.goal_horizon)\n',
     '            planner._feature_builder.config.route_goal_horizon_s = float(a.goal_horizon)\n'
     '            rec["goal_min_speed"] = planner._feature_builder.config.route_goal_min_speed_mps = _min_speed(a.goal_min_speed, r["ego"], MS0)\n'),
    ('            inv["goal"] = rec["goal_max_abs_diff"] <= GOAL_TOL_M if a.goal_horizon == 12.0 else True',
     '            inv["goal"] = rec["goal_max_abs_diff"] <= GOAL_TOL_M if (a.goal_horizon == 12.0 and a.goal_min_speed is None) else True'),
    ('            if a.goal_horizon != 12.0:                                   # LEVER ARM ONLY: the row carries the goal the teacher received\n',
     '            if a.goal_horizon != 12.0 or a.goal_min_speed is not None:  # LEVER ARM ONLY: the row carries the goal the teacher received\n'),
    ('            if a.goal_horizon != 12.0:\n                out["aug"]["goal_horizon_s"] = float(a.goal_horizon)\n',
     '            if a.goal_horizon != 12.0:\n                out["aug"]["goal_horizon_s"] = float(a.goal_horizon)\n'
     '            if a.goal_min_speed is not None:\n                out["aug"]["goal_min_speed"] = rec["goal_min_speed"]\n'),
    ('    r.add_argument("--goal-horizon", type=float, default=12.0, help="LEVER ARM ONLY (default 12.0 = the bank\'s base horizon)")\n',
     '    r.add_argument("--goal-horizon", type=float, default=12.0, help="LEVER ARM ONLY (default 12.0 = the bank\'s base horizon)")\n'
     '    r.add_argument("--goal-min-speed", default=None, help="LEVER (addendum 5): float m/s or v0max; unset = the config default")\n'),
])
print("patched")
