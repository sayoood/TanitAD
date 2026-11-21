"""SPEC A4 lever L3: deterministic DDIM (anchored-Gaussian draw = 0) on a battery tag's own windows.

usage: python lever_eps0.py <tag_dir> --ckpt <ckpt.pt> [--config <config.json>]

GPU gate (the dev-box rule, own pid excluded; this parent never holds CUDA) -> `roll_eps0.py` child
on the battery's `windows_s0.json` -> the battery's own `build_panel_dump` -> VOID gate: `g` and
the model-free arms bit-identical to the battery's `panel_s0` -> paired cells (ADE 0-2 s, T1):
`os_eps0 - ha0_ext`, `os_eps0 - ha`, `os_eps0 - os_s0`, `os_eps0 - os_s1`.
Writes `<tag_dir>/levers/eps0.json` + `EPS0.md`. The eps0 dump/panel stay in `<tag_dir>/levers/eps0/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_panel as P  # noqa: E402
import run_battery as RB  # noqa: E402
import lever_panel as LP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_dir")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="D:/refcv6_eval_kit/ckpt/config.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-gate", action="store_true", help="SMOKE ONLY")
    ap.add_argument("--smoke-cpu-fp32-trunk", action="store_true", help="SMOKE ONLY")
    ap.add_argument("--reuse-dump", default=None,
                    help="TEST ONLY: skip gate + roll and treat this existing dump as the L3 dump "
                         "(with the battery's own dump_s0 every os_eps0 - os_s0 cell must read 0.0)")
    a = ap.parse_args()
    root = Path(a.tag_dir)
    lv = root / "levers"
    wd = lv / "eps0"
    wd.mkdir(parents=True, exist_ok=True)
    rec = {"tool": "lever_eps0.py", "amendment": "A4", "lever": "L3 deterministic DDIM (eps = 0)",
           "tag_dir": str(root), "ckpt": a.ckpt, "ckpt_md5": P.L.md5_file(a.ckpt),
           "spec_sha256": hashlib.sha256((HERE.parent / "SPEC.md").read_bytes()).hexdigest(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    bsum = json.load(open(root / "battery_summary.json", encoding="utf-8"))
    if bsum.get("ckpt_md5") != rec["ckpt_md5"]:
        raise SystemExit(f"[eps0] ckpt md5 {rec['ckpt_md5']} != the battery's {bsum.get('ckpt_md5')}")
    dump = wd / "dump"
    rj = wd / "roll.json"
    if a.reuse_dump:
        rec["TEST_reuse_dump"] = a.reuse_dump
        dump = Path(a.reuse_dump)
        rj = wd / "roll_reuse_TEST.json"
        json.dump({"TEST": "reuse-dump", "eps0_randn_like_calls": None}, open(rj, "w", encoding="utf-8"))
    else:
        rec["gate"] = ({"skipped": "--skip-gate (smoke)"} if a.skip_gate
                       else RB.gate_wait(str(wd / "gate.json")))
    cmd = [sys.executable, str(HERE / "roll_eps0.py"), "--ckpt", a.ckpt, "--config", a.config,
           "--seed", "0", "--dump-dir", str(dump), "--windows-json", str(root / "windows_s0.json"),
           "--device", a.device, "--out-json", str(rj)]
    if a.smoke_cpu_fp32_trunk:
        cmd.append("--smoke-cpu-fp32-trunk")
    if not a.reuse_dump:
        with open(wd / "roll.log", "w", encoding="utf-8") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if not rj.exists():
        raise SystemExit(f"[eps0] the roll wrote no record -- see {wd / 'roll.log'}")
    rec["roll"] = json.load(open(rj, encoding="utf-8"))
    pd = wd / "panel"
    prec = P.build_panel_dump(str(dump), str(pd), str(dump / "refcv6_extras.npz"))
    rec["panel_void_gates_3_4"] = prec["void_gates_3_4"]
    man, E, eid = P.load_panel(str(pd))
    _, B0, eid0 = P.load_panel(str(root / "panel_s0"))
    same = (np.array_equal(eid, eid0) and np.array_equal(E["g"], B0["g"])
            and all(np.array_equal(E[k], B0[k]) for k in ("ha", "ha0", "ha0_ext")))
    rec["void_gate_same_windows_as_battery"] = bool(same)
    if not same or not prec["void_gates_3_4"]["pass"]:
        json.dump(rec, open(lv / "eps0.json", "w", encoding="utf-8"), indent=1, default=str)
        raise SystemExit("[eps0] VOID: not the battery's windows / controls")
    dt = float(man["grid"]["dt_s"])
    G = E["g"]
    a_e = LP.ade(E["os"], G, dt)
    a_echo, a_ha = LP.ade(E["ha0_ext"], G, dt), LP.ade(E["ha"], G, dt)
    cells = {"os_eps0_minus_ha0_ext": LP.paired(a_e, a_echo, eid),
             "os_eps0_minus_ha": LP.paired(a_e, a_ha, eid),
             "os_eps0_minus_os_s0": LP.paired(a_e, LP.ade(B0["os"], G, dt), eid)}
    if (root / "panel_s1").exists():
        _, B1, _ = P.load_panel(str(root / "panel_s1"))
        cells["os_eps0_minus_os_s1"] = LP.paired(a_e, LP.ade(B1["os"], G, dt), eid)
    rec["cells"] = cells
    rec["n_windows"], rec["n_episodes"] = int(len(eid)), int(len(set(eid.tolist())))
    rec["ade_mean"] = {"os_eps0": float(a_e.mean()), "ha0_ext": float(a_echo.mean()), "ha": float(a_ha.mean())}
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(rec, open(lv / "eps0.json", "w", encoding="utf-8"), indent=1, default=str)
    md = [f"### A4 lever L3 — deterministic DDIM (eps = 0), {rec['n_windows']} windows / {rec['n_episodes']} "
          f"episodes, S2 ADE 0–2 s, T1; a DIFFERENT planner from the registered (sampling) arm\n",
          f"randn_like calls zeroed: {rec['roll'].get('eps0_randn_like_calls')}; VOID gates: same windows "
          f"{rec['void_gate_same_windows_as_battery']}, pairing {prec['void_gates_3_4']['pass']}\n",
          "| cell | ADE m [CI] |", "|---|---|"]
    md += [f"| {k.replace('_minus_', ' − ')} | {LP._c(v)} |" for k, v in cells.items()]
    (lv / "EPS0.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
