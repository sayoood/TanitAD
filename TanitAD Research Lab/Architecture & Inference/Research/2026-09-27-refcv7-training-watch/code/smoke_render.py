"""Smoke renders of the refcv7 Training Watch, and the launch gate's own reading of them.

    python smoke_render.py <tree> <watch6 dir> <gate launch_gate.py> <out dir>

1. FIXTURE: the synthetic run of `test_build_watch_refcv7.py` (its own `_run_dir`), rendered twice --
   healthy, and with alarms on (a thin class under its bar, a box3d ratio of 3.4, a dead ga_mh_refine).
   Named FIXTURE_*: these are not a run.
2. BACK-COMPAT: the REAL refcv6-r101-s0 artifacts (a COPY of <watch6 dir>: metrics.jsonl, config.json,
   sup.log, remote_state.json, stderr_tail.log), rendered by the refcv7 builder with --arm refcv6-r101-s0
   and refcv6's own planned segments + diagnosed stderr line (read from the tip's refcv6 builder).
   The shared keys must render; the refcv7-only sections must say UNAVAILABLE.
3. THE GATE: `launch_gate.map_watch_reasons(PROFILES["refcv7"], page)` -- the gate's OWN code, not a replica
   -- on both, and the gate's builder invocation run exactly as `launch_gate.py` formats it, with the
   command template proposed for PROFILES["refcv7"]["map_hires"]["watch_builder"].
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEMPLATE = ["{python}", "{tree}/taniteval/tools/training_watch/build_watch_refcv7.py",
            "--metrics", "{metrics}", "--out", "{out}"]


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m               # a module's dataclasses resolve their own module while it executes
    spec.loader.exec_module(m)
    return m


def main(argv) -> int:
    tree, watch6, gate_py, out = (Path(a) for a in argv[:4])
    out.mkdir(parents=True, exist_ok=True)
    for k in ("REFCV7_EVAL_PKG", "REFCV7_EVAL_LIVE", "REFCV7_RUN", "REFCV7_ARM", "REFCV7_WATCH_DIR"):
        os.environ.pop(k, None)
    builder = tree / "taniteval" / "tools" / "training_watch" / "build_watch_refcv7.py"
    tests = _mod(tree / "taniteval" / "tests" / "test_build_watch_refcv7.py", "t7")
    gate = _mod(gate_py, "launch_gate_under_smoke")
    prof = gate.PROFILES["refcv7"]
    rec = {"tree": str(tree), "gate_py": str(gate_py), "template": TEMPLATE, "renders": {}}
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # ---- 1. fixtures -------------------------------------------------------------------------
        for name, kw in (("FIXTURE_synthetic_healthy", {}),
                         ("FIXTURE_synthetic_alarms_on", dict(map_over={("lane", "0_20"): 0.03,
                                                                        ("edge", "20_40"): 0.03,
                                                                        ("arrow", "80_100"): 0.0},
                                                              map_over_steps={5500}, ratios={"box3d": 3.4},
                                                              ga_last={"ga_mh_refine": 0.0}))):
            d = tests._run_dir(tdp, name=name, **kw)
            mod = _mod(builder, f"b7_{name}")
            page, summ = mod.build(watch_dir=str(d), arm="refcv7-r101-s0")
            p = out / f"{name}.html"
            p.write_text(page, encoding="utf-8", newline="\n")
            (out / f"{name}.summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8", newline="\n")
            reasons, det = gate.map_watch_reasons(prof, str(p))
            rec["renders"][name] = {"bytes": len(page.encode("utf-8")), "map": summ["map"]["state"],
                                    "box3d": summ["box"]["box3d"]["state"], "ga_alarm": summ["ga"]["alarm"],
                                    "gate_map_watch_reasons": reasons, "gate_alarm_tile": det.get("alarm_tile")}
        # ---- 3a. the gate's builder invocation, exactly as launch_gate.py formats it ----------------
        d = tests._run_dir(tdp, name="gate_form")
        for f in ("sup.log", "remote_state.json", "stderr_tail.log"):
            if (d / f).exists():
                (d / f).unlink()
        out_w = out / "FIXTURE_gate_form_watch_refcv7.html"
        cmd = [str(x).format(python=sys.executable, metrics=str(d / "metrics.jsonl"), out=str(out_w),
                             tree=str(tree)) for x in TEMPLATE]
        r = subprocess.run(cmd, cwd=str(tree), timeout=1800, capture_output=True)
        reasons, det = gate.map_watch_reasons(prof, str(out_w) if out_w.is_file() else None)
        rec["gate_form"] = {"rc": r.returncode, "stdout_tail": r.stdout.decode("utf-8", "replace").splitlines()[:1],
                            "stderr": r.stderr.decode("utf-8", "replace")[-400:],
                            "gate_map_watch_reasons": reasons, "gate_alarm_tile": det.get("alarm_tile"),
                            "series_missing": det.get("series_missing")}
        # ---- 2. back-compat: the real refcv6 artifacts ----------------------------------------------
        w6 = tdp / "watch6_copy"
        w6.mkdir()
        for f in ("metrics.jsonl", "config.json", "sup.log", "remote_state.json", "stderr_tail.log"):
            shutil.copyfile(watch6 / f, w6 / f)
        r6 = _mod(tree / "taniteval" / "tools" / "training_watch" / "build_watch_refcv6.py", "b6_facts")
        mod = _mod(builder, "b7_backcompat")
        mod.FACTS["segments"] = [(c, w) for c, w, _ in r6.FACTS["segments"]]
        mod.FACTS["stderr_diagnosed"] = dict(r6.FACTS["stderr_diagnosed"])
        page, summ = mod.build(watch_dir=str(w6), arm="refcv6-r101-s0")
        p = out / "BACKCOMPAT_refcv6-r101-s0_log_by_the_refcv7_builder.html"
        p.write_text(page, encoding="utf-8", newline="\n")
        (out / "BACKCOMPAT_refcv6-r101-s0_log_by_the_refcv7_builder.summary.json").write_text(
            json.dumps(summ, indent=1), encoding="utf-8", newline="\n")
        reasons, det = gate.map_watch_reasons(prof, str(p))
        drawn = [c for c in ("c1", "c2", "c3", "c4", "c5", "c6", "c8", "c9", "c10", "c11", "c12")
                 if f'data-chart="{c}"' in page]
        rec["renders"]["BACKCOMPAT_refcv6"] = {
            "bytes": len(page.encode("utf-8")), "step": summ["step"], "segments": summ["segments"],
            "unplanned": summ["unplanned"], "stopped": summ["stopped"], "map": summ["map"]["state"],
            "map_series_keys": summ["map"]["series_keys"], "box3d": summ["box"]["box3d"]["state"],
            "agent": summ["box"]["agent"]["state"], "ga_declared": summ["ga"]["declared"],
            "prior_mode": summ["prior"]["mode"], "shared_charts_drawn": drawn,
            "stderr_undiagnosed": summ["stderr_undiagnosed"], "n_err": summ["n_err"],
            "gate_map_watch_reasons_n": len(reasons), "gate_alarm_tile": det.get("alarm_tile")}
    ok = (not rec["renders"]["FIXTURE_synthetic_healthy"]["gate_map_watch_reasons"]
          and not rec["renders"]["FIXTURE_synthetic_alarms_on"]["gate_map_watch_reasons"]
          and rec["gate_form"]["rc"] == 0 and not rec["gate_form"]["gate_map_watch_reasons"]
          and rec["renders"]["BACKCOMPAT_refcv6"]["gate_map_watch_reasons_n"] >= 1
          and len(rec["renders"]["BACKCOMPAT_refcv6"]["shared_charts_drawn"]) == 11)
    rec["verdict"] = "PASS" if ok else "FAIL"
    (out / "smoke_record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(rec, indent=1)[:4000])
    print(f"ZZSMOKE-{rec['verdict']}ZZ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
