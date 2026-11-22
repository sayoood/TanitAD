"""static_gate_check.py <ARM> -- CPU-ONLY static admission of an r9e2 argv under the refcv9-e2 launch-gate profile (MM 4a3d05, 2026-11-20). NO token, NO GPU, NO data.

Runs, on stack/ops/runs.d/r9e2_<ARM>.argv.json: launch_gate.profile_argv_rules (required / forbidden flags, forbidden levers, schedule, map-weight and K2DC dose-controller
rules, PI-pending reasons), the trainer's own parser, and the T1v / banded / share-controller pins the build would hit. It is NOT G-DVB (that needs the built model and the data on Thor):
it lists what the STATIC half refuses. The K2DC02 argv is checked in the same breath as the CONTROL: its reasons are the baseline the arm is compared to.
usage: python static_gate_check.py <ARM> [<stack dir>]; exit 0 = no refusal beyond the K2DC02 control's (on a dev box the control itself lists the two HOST reasons: --nav-tau-record not given, --init-from not readable; Thor supplies both), 1 = a NEW refusal.
MM 11-20: the body is in main() under a __main__ guard, so importing this file (the lander's import probe) runs nothing.
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
    from tanitad.train import map_dose_ctl as MDC  # noqa: E402

    def one(name):
        doc = json.loads((stack / "ops" / "runs.d" / f"r9e2_{name}.argv.json").read_text(encoding="utf-8"))
        prof = LG.PROFILES[doc["profile"]]
        refusals, pending = LG.profile_argv_rules(prof, doc["argv"])
        ns = T.build_parser().parse_args(doc["argv"])
        BWF.pin_args(ns)                                           # the T1 pins (a no-op when neither T1h nor T1v is declared)
        out = {"arm": name, "profile": doc["profile"], "refusals": refusals, "pi_pending": pending, "hsg_on": bool(BWF.geom_on(ns)), "seed": ns.seed,
               "share_target": ns.map_hires_share_target, "dose_ctl_static": LG.dose_controller_static_reasons(doc["argv"])[0], "dc_firings": len(LG.dc_reading_rows(doc["argv"])),
               "w_map_hires": ns.w_map_hires, "map_banded": bool(ns.r9_map_banded), "w_r8_drivable": ns.w_r8_drivable}
        MDC.parse_bounds(ns.map_hires_w_bounds) if ns.map_hires_share_target is not None else None
        return out

    res = {n: one(n) for n in ("K2DC02", arm)}
    new = [r for r in res[arm]["refusals"] if r not in res["K2DC02"]["refusals"]]
    res["refusals_not_in_the_K2DC02_control"] = new
    print(json.dumps(res, indent=1, default=str))
    return 0 if not new else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
