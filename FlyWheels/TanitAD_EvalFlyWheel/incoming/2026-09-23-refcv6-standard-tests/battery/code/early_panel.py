"""Analyse ONE finished roll dump through the full panel, on CPU, while the next seed rolls on the GPU.
Same functions run_battery.py calls; used to surface any full-scale defect in the panel stage early.
usage: python early_panel.py <dump_dir> <out_dir> [n_boot]"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import run_battery as RB  # noqa: E402
import refcv6_panel as P  # noqa: E402


def main():
    d, out = sys.argv[1], Path(sys.argv[2])
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    out.mkdir(parents=True, exist_ok=True)
    pd = str(out / "panel")
    prec = P.build_panel_dump(d, pd, os.path.join(d, "refcv6_extras.npz"))
    json.dump({k: prec[k] for k in ("pairing", "void_gates_3_4", "arms", "n_windows", "n_episodes")},
              open(out / "panel_record.json", "w", encoding="utf-8"), indent=1)
    print("[early] panel", prec["n_windows"], prec["void_gates_3_4"], flush=True)
    an = P.analyze(pd, str(out / "analysis.json"), RB.KIT_LABELS, n_boot=nb)
    vg = P.void_gate_checks(pd, an)
    json.dump(vg, open(out / "void_gates.json", "w", encoding="utf-8"), indent=1)
    cp = P.cross_paired(pd, RB.PAIRS, n_boot=nb)
    json.dump(cp, open(out / "cross_paired.json", "w", encoding="utf-8"), indent=1)
    man = json.load(open(os.path.join(pd, "manifest.json"), encoding="utf-8"))
    clip_eid = {int(e["episode_index"]): int(e["file_index"]) for e in man["episodes"]}
    tac = P.tactical_v6(os.path.join(d, "refcv6_extras.npz"), clip_eid, n_boot=nb)
    json.dump(tac, open(out / "tactical_v6.json", "w", encoding="utf-8"), indent=1)
    acc = P.acceptance(an, os.path.join(d, "refcv6_extras.npz"), clip_eid, pd, n_boot=nb)
    acc["tzero"] = P.tzero_from_tactical(tac)
    json.dump(acc, open(out / "acceptance.json", "w", encoding="utf-8"), indent=1, default=str)
    s6 = P.build_s6_dump(d, str(out / "s6"))
    P.analyze(str(out / "s6"), str(out / "analysis_s6.json"), RB.KIT_LABELS, n_boot=nb)
    c6 = P.cross_paired(str(out / "s6"), RB.PAIRS, n_boot=nb)
    json.dump(c6, open(out / "cross_paired_s6.json", "w", encoding="utf-8"), indent=1)
    print("[early] DONE", json.dumps({"void": vg.get("pass_1_2"), "s6": s6["n_windows"],
                                      "tflip": acc["tflip"].get("verdict"),
                                      "obedience": acc["obedience"].get("verdict")}), flush=True)


if __name__ == "__main__":
    main()
