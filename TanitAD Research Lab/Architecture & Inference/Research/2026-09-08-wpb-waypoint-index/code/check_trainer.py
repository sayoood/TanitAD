"""WP-B trainer-wiring probe: the two refusals, the stamp, the seam assertion,
and the LAUNCH-COMMAND diff.

⛔ THE ONE-VARIABLE RULE IS CHECKED ON THE ACTUAL PARSED LAUNCH COMMANDS, NOT ON
THE INTENT. This prints the symmetric difference of the two arms' namespaces, so
a reader can see that exactly the wp-index knobs differ and nothing else.
"""
import json
import sys

sys.path.insert(0, r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
sys.path.insert(0, r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/scripts")

import refc_v3_train as T                                          # noqa: E402
from tanitad.refs import refc_v3 as v3                             # noqa: E402

BASE = ["--arm", "hier", "--out", "/tmp/wpb", "--steps", "40284",
        "--agents", "head", "--w-agent", "1.0",
        "--agent-join", "joins/train2400_agents.jsonl.xz"]
out = {}


def parse(extra):
    return T.build_parser().parse_args(BASE + extra)


# ---- 1. the launch-command DIFF (the one-variable proof, on the commands) -- #
a0, a1 = vars(parse([])), vars(parse(["--wp-index", "on"]))
out["launch_diff_B0_vs_B1"] = {
    k: [a0.get(k, "<UNSET>"), a1.get(k, "<UNSET>")]
    for k in set(a0) | set(a1)
    if a0.get(k, "<UNSET>") != a1.get(k, "<UNSET>")}


# ---- 2. the two refusals ------------------------------------------------- #
def refuses(extra, needle):
    args = T.build_parser().parse_args(BASE[:4] + ["--steps", "10"] + extra)
    cfg = v3.RefCV3Config()
    try:
        T._pin_refcv5_seams(cfg, args)
    except SystemExit as exc:
        return {"refused": True, "matched": needle in str(exc),
                "msg": " ".join(str(exc).split())[:260]}
    return {"refused": False, "matched": False, "msg": "DID NOT REFUSE"}


out["refusal_index_without_agents"] = refuses(
    ["--agents", "off", "--wp-index", "on"], "--agents off")
out["refusal_noop_knobs_with_index_off"] = refuses(
    ["--agents", "head", "--w-agent", "1.0", "--agent-join", "j.xz",
     "--wp-index", "off", "--wp-index-mode", "shuffle"], "--wp-index off")

# ---- 3. the stamp -------------------------------------------------------- #
cfg = v3.RefCV3Config()
args = parse(["--wp-index", "on", "--wp-index-mode", "const"])
T._pin_refcv5_seams(cfg, args)
stamp = T._seam_stamp(cfg, args)
out["stamp_wp_index"] = stamp["wp_index"]
out["stamp_cross_agent"] = stamp["cross_agent"]
out["knob_dests_wp_index"] = sorted(d for d in T.agent_knob_dests()
                                    if d.startswith("wp_index"))


# ---- 4. assert_seams_are_built, ALL FOUR directions ---------------------- #
class _L:
    def __init__(self, ca, wpi):
        self.cross_agent, self.wp_index = ca, wpi


class _D:
    def __init__(self, layers, wp_cfg):
        self.layers, self.wp_index_cfg = layers, wp_cfg
        self.control_head = None
        self.gp_point_gate = None


class _M:
    def __init__(self, dec):
        self.decoder = dec
        self.agent_head = object()
        self.agent_embed = object()
        self.gp_head = None


def seam_check(layers, wp_cfg, override):
    st = dict(stamp)
    st.update(override)
    try:
        T.assert_seams_are_built(_M(_D(layers, wp_cfg)), st)
        return "PASSED"
    except SystemExit as exc:
        return " ".join(str(exc).split())[:240]
    except Exception as exc:                                       # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"[:240]


live = [_L(object(), object()), _L(object(), object())]
dead = [_L(object(), None), _L(object(), None)]
out["seam_A_stamped_and_built"] = seam_check(
    live, type("C", (), {"mode": "const"})(), {})
out["seam_B_stamped_but_NOT_built"] = seam_check(dead, None, {})
out["seam_C_built_but_NOT_stamped"] = seam_check(
    live, type("C", (), {"mode": "const"})(), {"wp_index": None})
out["seam_D_mode_mismatch"] = seam_check(
    live, type("C", (), {"mode": "geom"})(), {})

print(json.dumps(out, indent=2, default=str))
