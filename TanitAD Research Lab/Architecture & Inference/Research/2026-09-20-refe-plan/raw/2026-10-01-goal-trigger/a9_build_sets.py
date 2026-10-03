#!/usr/bin/env python3
"""Amendment 9's token sets, from CPU artifacts only (no model output): per arm, the CONFIRMATION tokens (goal changes
under the arm, census) minus the 1,123 selection tokens minus the 36 frame-control failures, plus 24 CONTROL tokens whose
goal does not change (seed 20260927). Writes a9/tokens_<arm>.json in W3's {'tokens', 'token_log'} format + the split."""
import gzip
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRV = os.environ.get("REFE_DRIVE", "D:")
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
SEL = HERE.parent / "2026-09-28-m6b-tangent" / "tokens_1123.json"
FC = Path(os.environ.get("REFE_FC_JSON", str(HERE.parent / "2026-10-04-navtest-final" / "frame_control_navtest_full.json")))
ARMS = ("pdm_route", "navgoal_straight", "navgoal_arc", "a8_lane")


def main():
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    G = json.load(open(HERE / "goal_fix_census.json", encoding="utf-8"))
    C = json.load(open(HERE / "goal_census_navtest_full.json", encoding="utf-8"))["rows"]
    sel = set(json.load(open(SEL, encoding="utf-8"))["tokens"])
    fc_bad = set(json.load(open(FC, encoding="utf-8"))["over_bar"])
    sel_logs = {E[t]["log_name"] for t in sel}
    (HERE / "a9").mkdir(exist_ok=True)
    summ = {"n_selection": len(sel), "n_frame_control_fail": len(fc_bad), "arms": {}}
    for arm in ARMS:
        if arm == "a8_lane":
            changed = {t for t, r in C.items() if r.get("ego_to_route_m", 0) > 20.0}
        else:
            changed = set(G["summary"][arm]["changed_tokens"])
        conf = sorted(changed - sel - fc_bad)
        # FRESH-LOG (as Amendment 8): the logs where this arm's defect was never SEEN during exploration, i.e. that hold no
        # selection token the arm changes. ("no selection token at all" is empty: the 1,123 touch all 136 navtest logs.)
        seen_logs = {E[t]["log_name"] for t in changed & sel}
        pool = sorted(t for t in E if t not in changed and t not in sel and t not in fc_bad)
        ctrl = sorted(random.Random(20260927).sample(pool, 24))
        toks = conf + ctrl
        json.dump({"rule": f"Amendment 9 arm {arm}: confirmation tokens (goal changes under the arm, census) minus the 1,123 "
                           f"selection tokens minus the 36 frame-control failures, + 24 unchanged controls (seed 20260927)",
                   "tokens": toks, "token_log": {t: E[t]["log_name"] for t in toks}, "confirm": conf, "controls": ctrl,
                   "seen_logs": sorted(seen_logs)},
                  open(HERE / "a9" / f"tokens_{arm}.json", "w", encoding="utf-8", newline="\n"), indent=0)
        cmd = {}
        for t in conf:
            c = ("L", "S", "R", "U")[max(range(4), key=lambda i: E[t]["ego_statuses"][-1]["driving_command"][i])]
            cmd[c] = cmd.get(c, 0) + 1
        summ["arms"][arm] = {"changed_all": len(changed), "changed_in_selection": len(changed & sel),
                             "changed_frame_fail": len(changed & fc_bad), "confirm": len(conf),
                             "confirm_logs": len({E[t]["log_name"] for t in conf}),
                             "fresh_log_tokens": sum(E[t]["log_name"] not in seen_logs for t in conf),
                             "fresh_logs": len({E[t]["log_name"] for t in conf} - seen_logs), "commands_LSRU": cmd}
    json.dump(summ, open(HERE / "a9" / "token_sets_summary.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
