"""make_windows_r8.py -- build the S2 and S6 window files for the refcv8 kit and PROVE they are the refcv7 baseline's windows.

    python make_windows_r8.py E:/refcv8_eval_kit/windows [--bank <pkg raw dir for the sha12-only report>]

S2 = `r7_roll.s2_window_list` over the banked refcv4b grid restricted to the kit's 139 clips (the SAME function
`run_battery_r7` and `make_s2_windows.py` use).  S6 = the 6-s subset (windows with a valid 60-step future); its identity is
READ FROM the banked refcv7-50,400 S6 dumps (`s6_s{0,1}`), and asserted to be a subset of S2 -- after the refcv8 roll,
`pair_read_r8.py` re-asserts that refcv8's own S6 set equals it.

ASSERTED, not assumed (each a set-equality of (clip_id, ws) pairs; counts printed):
  * refcv8 S2 == refcv7 `windows/s2_windows.json` (the file refcv7's chain rolled on)
  * refcv8 S2 == refcv7-50,400 banked dump_s0 == dump_s1 == panel_s0 == panel_s1
  * S6 file == refcv7-50,400 banked s6_s0 == s6_s1, and S6 is a subset of S2
  * counts == the SPEC quotes (S2 4,754 windows / 139 episodes; S6 3,668 / 139) -- reported, a mismatch is a loud FAIL
The report file carries sha12-keyed material only when written under the package; the raw-id files stay in the kit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_battery_patch as BP  # noqa: E402

L, P = BP.apply()
import r7_roll as RR  # noqa: E402
from tanitad.data.v2_dataset import load_or_build_manifest  # noqa: E402

SPEC_QUOTE = {"S2": {"n_windows": 4754, "n_episodes": 139}, "S6": {"n_windows": 3668, "n_episodes": 139}}
R7_WIN_FILE = os.environ.get("R8_R7_WINDOWS", "D:/refcv7_eval_kit/windows/s2_windows.json")


def _sha(d: dict) -> str:
    return hashlib.sha256(json.dumps({L.sha12(c): v for c, v in d.items()}, sort_keys=True).encode()).hexdigest()


def _eq(a: dict, b: dict) -> dict:
    sa, sb = BP.window_set(a), BP.window_set(b)
    return {"equal": sa == sb, "n_a": len(sa), "n_b": len(sb), "only_a": len(sa - sb), "only_b": len(sb - sa),
            "clips_a": len(a), "clips_b": len(b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--r7base", default=BP.R7_BASE_ROOT)
    ap.add_argument("--bank", default=None, help="write the sha12-only report here too")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kit_eval = str(L.KIT / "data/refcv6-b1-416x1024-eval139")
    ids = {str(c) for c in load_or_build_manifest(kit_eval, verbose=False)["clip_id"]}
    s2 = RR.s2_window_list(P.BASELINES["b_refcv4b"], ids)
    r7 = Path(a.r7base)
    s6_s0 = BP.windows_of(r7 / "s6_s0")
    rep = {"tool": "make_windows_r8.py", "kit_eval": kit_eval, "n_kit_clips": len(ids),
           "s2_source": P.BASELINES["b_refcv4b"], "r7base": str(r7), "checks": {}}
    rep["S2"] = {"n_windows": len(BP.window_set(s2)), "n_episodes": len(s2), "sha256_of_sha12_view": _sha(s2)}
    rep["S6"] = {"n_windows": len(BP.window_set(s6_s0)), "n_episodes": len(s6_s0), "sha256_of_sha12_view": _sha(s6_s0)}
    ck = rep["checks"]
    ck["S2_vs_refcv7_windows_file"] = _eq(s2, json.load(open(R7_WIN_FILE, encoding="utf-8"))) if os.path.exists(R7_WIN_FILE) \
        else {"equal": None, "why": f"{R7_WIN_FILE} absent"}
    for tag in ("dump_s0", "dump_s1", "panel_s0", "panel_s1"):
        ck[f"S2_vs_refcv7_50400_{tag}"] = _eq(s2, BP.windows_of(r7 / tag))
    for tag in ("s6_s0", "s6_s1"):
        ck[f"S6_vs_refcv7_50400_{tag}"] = _eq(s6_s0, BP.windows_of(r7 / tag))
    ck["S6_subset_of_S2"] = {"equal": BP.window_set(s6_s0) <= BP.window_set(s2),
                             "n_s6_not_in_s2": len(BP.window_set(s6_s0) - BP.window_set(s2))}
    for k, quote in SPEC_QUOTE.items():
        ck[f"{k}_counts_match_SPEC_quote"] = {"equal": (rep[k]["n_windows"] == quote["n_windows"]
                                                         and rep[k]["n_episodes"] == quote["n_episodes"]),
                                              "measured": {x: rep[k][x] for x in ("n_windows", "n_episodes")},
                                              "quoted": quote}
    # r7base's own analysis json (the artifact the paired read is made against) quotes the counts too
    ra = Path(os.environ.get("R8_R7BASE_PKG", "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                              "2026-10-11-r7base-fourfamily")) / "raw"
    for tag, key in (("analysis_s2_s0.json", "S2"), ("analysis_s6_s0.json", "S6")):
        if (ra / tag).exists():
            j = json.load(open(ra / tag, encoding="utf-8"))
            ck[f"{key}_counts_match_{tag}"] = {"equal": (j["n_windows"], j["n_episodes"]) ==
                                               (rep[key]["n_windows"], rep[key]["n_episodes"]),
                                               "in_analysis": [j["n_windows"], j["n_episodes"]]}
    rep["ALL_EQUAL"] = all(v.get("equal") is True for v in ck.values())
    rep["baselines_repointed"] = BP.describe()["baseline_repoint"]
    json.dump(s2, open(out / "s2_windows.json", "w", encoding="utf-8"))
    json.dump({L.sha12(c): v for c, v in s2.items()}, open(out / "s2_windows.json.sha12.json", "w", encoding="utf-8"))
    json.dump(s6_s0, open(out / "s6_windows.json", "w", encoding="utf-8"))
    json.dump({L.sha12(c): v for c, v in s6_s0.items()}, open(out / "s6_windows.json.sha12.json", "w", encoding="utf-8"))
    json.dump(rep, open(out / "windows_equality.json", "w", encoding="utf-8"), indent=1)
    if a.bank:
        Path(a.bank).mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(Path(a.bank) / "windows_equality.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({"ALL_EQUAL": rep["ALL_EQUAL"], "S2": rep["S2"], "S6": rep["S6"],
                      "checks": {k: v.get("equal") for k, v in ck.items()}}, indent=1))
    print("ZZR8WINDOWS_%s_ZZ" % ("OK" if rep["ALL_EQUAL"] else "MISMATCH"))


if __name__ == "__main__":
    main()
