"""static_gate_check.py <ARM> -- CPU-ONLY static admission of an r9e2 argv under the refcv9-e2 launch-gate profile (MM 4a3d05, 2026-11-21; K2SG form, derived from the K2DC02nh kit's file). NO token, NO GPU, NO data.

Runs, on stack/ops/runs.d/r9e2_<ARM>.argv.json: launch_gate.profile_argv_rules (required / forbidden flags, forbidden levers, schedule, map-weight and stop-gradient share-alarm rules,
PI-pending reasons), the trainer's own parser (map_trunk_gs.pin_args via the trainer), and the T1v / banded pins the build would hit. It is NOT G-DVB (that needs the built model and the data on
Thor): it lists what the STATIC half refuses. The K2M1N20 argv (K2SG's base: K2SG = K2M1N20 + --map-hires-trunk-grad-scale 0.0) is checked in the same breath as the CONTROL: its reasons are the baseline.
usage: python static_gate_check.py <ARM> [<stack dir>]; exit 0 = no refusal beyond the K2M1N20 control's (on a dev box the control itself lists the HOST reasons: --nav-tau-record not given, --init-from not readable; Thor supplies both), 1 = a NEW refusal.
The body is in main() under a __main__ guard, so importing this file (the lander's import probe) runs nothing.
"""
import json
import sys
from pathlib import Path


def main(argv):
    arm = argv[1]
    stack = Path(argv[2]) if len(argv) > 2 else Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(stack), str(stack / "scripts")]
    import launch_gate as LG  # noqa: E402
    import refc_v3_train as T  # noqa: E402
    from tanitad.train import bev_warp_flag as BWF  # noqa: E402

    def one(name):
        doc = json.loads((stack / "ops" / "runs.d" / f"r9e2_{name}.argv.json").read_text(encoding="utf-8"))
        prof = LG.PROFILES[doc["profile"]]
        refusals, pending = LG.profile_argv_rules(prof, doc["argv"])
        ns = T.build_parser().parse_args(doc["argv"])
        BWF.pin_args(ns)                                           # the T1 pins (a no-op when neither T1h nor T1v is declared)
        return {"arm": name, "profile": doc["profile"], "refusals": refusals, "pi_pending": pending, "hsg_on": bool(BWF.geom_on(ns)), "seed": ns.seed,
                "trunk_grad_scale": ns.map_hires_trunk_grad_scale, "share_target": ns.map_hires_share_target,
                "w_map_hires": ns.w_map_hires, "map_banded": bool(ns.r9_map_banded), "w_r8_drivable": ns.w_r8_drivable}

    res = {n: one(n) for n in ("K2M1N20", arm)}
    new = [r for r in res[arm]["refusals"] if r not in res["K2M1N20"]["refusals"]]
    res["refusals_not_in_the_K2M1N20_control"] = new
    print(json.dumps(res, indent=1, default=str))
    return 0 if not new else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
