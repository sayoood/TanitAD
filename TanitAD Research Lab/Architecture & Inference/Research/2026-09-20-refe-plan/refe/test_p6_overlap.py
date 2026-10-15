#!/usr/bin/env python3
"""Tests for refe/p6_overlap.py (eval/P6_ON_11VCPU.md section 9): the helpers of P6's OVERLAP mode.  No pod, no GPU, no NAVSIM, no network.

Each function in both directions and at its boundary: the running trainer's step time from a metrics.jsonl (event rows ignored, a resume reset not averaged in, a torn last line,
a stale file = no live trainer), the 5 % guard at 1.049 / 1.05 / 1.051, the worker cap from a fake cgroup tree whose CPU counter advances (v1 and v2, clamps, fallback), the
G-LIVE-LAB bar read from a lab.json (identity AND stale refusal AND throughput), the constants pinned to the gate's own, and every CLI marker + rc.
Run: python refe/test_p6_overlap.py   -> ZZTEST_PASS <n>
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p6_overlap as O  # noqa: E402

CLI = HERE / "p6_overlap.py"
checks = []


def t(name, ok):
    checks.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


def cli(*args):
    r = subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def write_metrics(path, steps, dt=60.0, t0=1000.0, extra_events=True, torn=False):
    """steps: iterable of step numbers, each `dt` seconds after the previous (dt may be a list)."""
    lines, el = [], t0
    if extra_events:
        lines.append(json.dumps({"event": "start", "at": "2026-10-14T00:00:00Z"}))
    dts = dt if isinstance(dt, (list, tuple)) else None
    for i, s in enumerate(steps):
        el += (dts[i] if dts else dt)
        lines.append(json.dumps({"step": s, "epoch": 0, "pos": 0, "traj_L1": 1.0, "score": 0.1, "elapsed_s": el, "samples": s * 256}))
        if extra_events and s % 3 == 0:
            lines.append(json.dumps({"event": "ckpt", "step": s, "elapsed_s": el}))     # an event row: no traj_L1, must be ignored
    text = "\n".join(lines) + "\n"
    if torn:
        text += '{"step": 99, "traj_L1": 1.0, "elaps'
    Path(path).write_text(text, encoding="utf-8")


def cgtree(root, v, usage_s, nr_throttled=0):
    root = Path(root)
    if v == 1:
        (root / "cpuacct").mkdir(parents=True, exist_ok=True)
        (root / "cpu").mkdir(parents=True, exist_ok=True)
        (root / "cpuacct" / "cpuacct.usage").write_text(f"{int(usage_s * 1e9)}\n")
        (root / "cpu" / "cpu.stat").write_text(f"nr_periods 100\nnr_throttled {nr_throttled}\nthrottled_time 0\n")
    else:
        root.mkdir(parents=True, exist_ok=True)
        (root / "cpu.stat").write_text(f"usage_usec {int(usage_s * 1e6)}\nnr_periods 100\nnr_throttled {nr_throttled}\nthrottled_usec 0\n")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="p6ovl_"))
    # ---- constants == the gate's
    gc = (HERE / "gate_checks.py").read_text(encoding="utf-8")
    m = re.search(r"^LAB_NEED_SETS_S, LAB_MARGIN, LAB_STALL_MAX, MEM_MAX_GB, MEM_MIN_MICRO = ([\d.]+), ([\d.]+), ", gc, re.M)
    t("const/equal_the_gates_own", bool(m) and (float(m.group(1)), float(m.group(2))) == (O.NEED_SETS_S, O.MARGIN))
    t("const/gate_bar_is_6_735", O.BAR_GATE == 6.735)

    # ---- trainer rate
    f = tmp / "m.jsonl"
    write_metrics(f, range(1, 12), 60.0)
    cur, n, last, why = O.trainer_rate(f, 5)
    t("rate/median_of_last_5_is_60", cur == 60.0 and n == 5 and last == 11)
    write_metrics(f, range(1, 12), [60, 60, 60, 60, 60, 60, 90, 60, 61, 60, 59])
    t("rate/median_ignores_one_outlier", O.trainer_rate(f, 5)[0] == 60.0)
    write_metrics(f, range(1, 12), [60] * 6 + [80, 81, 79, 82, 80])
    t("rate/median_follows_a_real_slowdown", O.trainer_rate(f, 5)[0] == 80.0)
    write_metrics(f, [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7], 60.0)                       # a resume rewinds the step counter: that interval is skipped, not averaged in
    dts, last = O.step_times(f, 20)
    t("rate/resume_rewind_skipped", len(dts) == 9 and all(abs(x - 60.0) < 1e-9 for x in dts) and last == 7)
    write_metrics(f, [1, 3, 5, 7, 9], 120.0)                                          # log-every 2: 120 s per 2 steps = 60 s per step
    t("rate/log_every_2_divides_by_the_step_delta", O.trainer_rate(f, 5)[0] == 60.0)
    write_metrics(f, range(1, 12), 60.0, torn=True)
    t("rate/torn_last_line_tolerated", O.trainer_rate(f, 5)[0] == 60.0)
    write_metrics(f, [1, 2], 60.0)
    c2, n2, _, why2 = O.trainer_rate(f, 5)
    t("rate/fewer_than_3_intervals_is_unknown", c2 is None and "(< 3)" in why2)
    write_metrics(f, range(1, 12), 60.0)
    os.utime(f, (time.time() - 5000, time.time() - 5000))
    c3, _, _, why3 = O.trainer_rate(f, 5, max_age_s=900)
    t("rate/stale_file_is_no_live_trainer", c3 is None and why3.startswith("stale"))
    c4, _, _, why4 = O.trainer_rate(tmp / "absent.jsonl", 5)
    t("rate/missing_file_is_unreadable_not_zero", c4 is None and why4.startswith("unreadable"))
    big = tmp / "big.jsonl"                                                           # a long file: only the tail is read, and a cut first line is dropped
    write_metrics(big, range(1, 6000), 60.0, extra_events=False)
    t("rate/long_file_tail_window", abs(O.trainer_rate(big, 5)[0] - 60.0) < 1e-9 and Path(big).stat().st_size > 262144)

    # ---- the 5 % guard at its boundary
    write_metrics(f, range(1, 12), 60.0)
    t("guard/unchanged_is_ok", O.guard(f, 60.0, 0.05)[0] == "ok")
    for cur_s, want in ((62.94, "ok"), (63.0, "ok"), (63.06, "slow"), (70.0, "slow"), (55.0, "ok")):
        write_metrics(f, range(1, 12), cur_s)
        t(f"guard/{cur_s}_vs_60_is_{want}", O.guard(f, 60.0, 0.05)[0] == want)
    t("guard/no_baseline_is_unknown", O.guard(f, 0.0, 0.05)[0] == "unknown" and O.guard(f, None, 0.05)[0] == "unknown")
    t("guard/missing_file_is_unknown", O.guard(tmp / "absent.jsonl", 60.0)[0] == "unknown")

    # ---- the worker cap from a cgroup whose counter advances
    for v in (1, 2):
        root = tmp / f"cg{v}"
        cgtree(root, v, 100.0)
        state = {"n": 0}

        def advance(secs, root=root, v=v, burn=3.2):
            state["n"] += 1
            cgtree(root, v, 100.0 + burn * secs)
        r = O.overlap_cap(11, 11, secs=60, margin=1.0, root=str(root), sleep=advance)
        t(f"cap/v{v}_burn_3_2_leaves_6_of_11", r["cap"] == 6 and abs(r["burn"] - 3.2) < 1e-6 and r["src"] == "census")    # floor(11 - 3.2 - 1) = 6
        cgtree(root, v, 100.0)
        r = O.overlap_cap(11, 5, secs=60, margin=1.0, root=str(root), sleep=lambda s, root=root, v=v: cgtree(root, v, 100.0 + 3.2 * s))
        t(f"cap/v{v}_clamped_to_max", r["cap"] == 5)
        cgtree(root, v, 100.0)
        r = O.overlap_cap(11, 11, secs=60, margin=1.0, root=str(root), sleep=lambda s, root=root, v=v: cgtree(root, v, 100.0 + 10.5 * s))
        t(f"cap/v{v}_trainer_eats_everything_leaves_one", r["cap"] == 1)
        cgtree(root, v, 100.0)
        r = O.overlap_cap(11, 11, secs=60, margin=1.0, root=str(root), sleep=lambda s, root=root, v=v: cgtree(root, v, 100.0 + 0.0 * s))
        t(f"cap/v{v}_idle_pod_leaves_cores_minus_margin", r["cap"] == 10)
    r = O.overlap_cap(11, 11, secs=1, root=str(tmp / "nothing_here"), sleep=lambda s: None)
    t("cap/unreadable_counters_fall_back_to_half_and_say_UNVERIFIED", r["cap"] == 5 and r["burn"] is None and "UNVERIFIED" in r["src"] and r["src"].startswith("fallback"))
    root = tmp / "cgback"
    cgtree(root, 2, 500.0)
    r = O.overlap_cap(11, 11, secs=10, root=str(root), sleep=lambda s: cgtree(root, 2, 100.0))      # a counter that went backwards (container restart)
    t("cap/backwards_counter_falls_back", r["cap"] == 5 and r["src"].startswith("fallback"))

    # ---- the lab bar
    lab = tmp / "lab.json"
    base = {"identity_ok": True, "n_sets": 256, "identity_equal_sets": 256, "stale_refused": True, "workers": 7}
    for thr, want, rc in ((6.735, "ok", 0), (6.734, "below", 1), (11.9, "ok", 0), (4.6, "below", 1)):
        lab.write_text(json.dumps({**base, "throughput_sets_s": thr}))
        v, det = O.lab_bar(lab)
        t(f"labbar/thr_{thr}_is_{want}", v == want)
    # a compared set that differs makes the PRODUCER (diag_p6_labeller_identity.py) set identity_ok False; < 64 compared fails judge_lab's D1(a)
    for nm, patch in (("identity_failed", {"identity_ok": False}), ("only_32_sets_compared", {"n_sets": 32, "identity_equal_sets": 32}), ("stale_not_refused", {"stale_refused": False}),
                      ("some_sets_differ", {"identity_ok": False, "identity_equal_sets": 63}), ("fewer_than_64_compared", {"identity_equal_sets": 63})):
        lab.write_text(json.dumps({**base, "throughput_sets_s": 12.0, **patch}))
        t(f"labbar/{nm}_is_bad_whatever_the_throughput", O.lab_bar(lab)[0] == "bad")
    # the REAL producer's shape (Master Mind 2026-10-15): the pod's clean early lab, 256 labelled, 64 compared (min(n_sets, 64)), all equal.  The old
    # `identity_equal_sets == n_sets` rule logged it "bad identity=False" while the chain's own refusal read ZZLAB_IDENTITY_PASS and judge_lab passes it.
    real = {"n_sets": 256, "distinct_tokens": 86, "workers": 7, "labelled": 256, "identity_equal_sets": 64, "identity_ok": True,
            "throughput_sets_s": 10.887, "throughput_bar_sets_s": 6.735, "throughput_ok": True, "stale_refused": True}
    lab.write_text(json.dumps(real))
    v, det = O.lab_bar(lab)
    t("labbar/real_producer_shape_256_labelled_64_compared_is_ok", v == "ok" and det["identity"] is True)
    # lab_bar exists to PREDICT the launch gate: its identity verdict must equal judge_lab's D1(a)/D1(c) on every fixture
    import gate_checks as G  # noqa: E402
    fx = [real, {**real, "identity_ok": False}, {**real, "identity_equal_sets": 63}, {**real, "n_sets": 32, "identity_equal_sets": 32},
          {**real, "stale_refused": False}, {**base, "throughput_sets_s": 12.0}]
    agree = []
    for d in fx:
        lab.write_text(json.dumps(d))
        g_ok = not any(s.startswith(("D1(a)", "D1(c)")) for s in G.judge_lab(d, [], d.get("workers", 7)))
        agree.append(O.lab_bar(lab)[1]["identity"] is g_ok)
    t("labbar/identity_verdict_equals_judge_lab_on_every_fixture", all(agree) and len(agree) == 6)
    lab.write_text(json.dumps(base))
    t("labbar/no_throughput_is_bad", O.lab_bar(lab)[0] == "bad")
    t("labbar/missing_file_is_bad", O.lab_bar(tmp / "nope.json")[0] == "bad")

    # ---- the CLI: markers and exit codes
    write_metrics(f, range(1, 12), 60.0)
    rc, out = cli("trainer-rate", "--metrics", str(f))
    t("cli/trainer_rate_marker", rc == 0 and re.fullmatch(r"ZZTRAINER_RATE step_s=60\.000 n=5 last_step=11", out) is not None)
    rc, out = cli("trainer-rate", "--metrics", str(tmp / "absent.jsonl"))
    t("cli/trainer_rate_unknown_rc2", rc == 2 and out.startswith("ZZTRAINER_RATE unknown"))
    rc, out = cli("trainer-guard", "--metrics", str(f), "--base", "60")
    t("cli/guard_ok_rc0", rc == 0 and out.startswith("ZZTRAINER_GUARD ok ratio=1.000"))
    write_metrics(f, range(1, 12), 66.0)
    rc, out = cli("trainer-guard", "--metrics", str(f), "--base", "60")
    t("cli/guard_slow_rc1", rc == 1 and out.startswith("ZZTRAINER_GUARD slow ratio=1.100"))
    rc, out = cli("trainer-guard", "--metrics", str(tmp / "absent.jsonl"), "--base", "60")
    t("cli/guard_unknown_rc2", rc == 2 and out.startswith("ZZTRAINER_GUARD unknown"))
    cgtree(tmp / "cgcli", 1, 5.0)
    rc, out = cli("overlap-cap", "--cores", "11", "--max", "9", "--secs", "0.2", "--root", str(tmp / "cgcli"))
    t("cli/overlap_cap_marker_burn_0_gives_9", rc == 0 and re.match(r"ZZOVERLAP_CAP 9 burn=0\.0 cores=11 margin=1 max=9 src=census", out) is not None)
    rc, out = cli("overlap-cap", "--cores", "11", "--max", "9", "--secs", "0.1", "--root", str(tmp / "nowhere"))
    t("cli/overlap_cap_fallback_marker", rc == 0 and out.startswith("ZZOVERLAP_CAP 5 burn=na") and "UNVERIFIED" in out)
    lab.write_text(json.dumps({**base, "throughput_sets_s": 7.0}))
    rc, out = cli("lab-bar", "--json", str(lab))
    t("cli/lab_bar_ok_rc0", rc == 0 and out.startswith("ZZLAB_BAR ok thr=7.0 bar=6.735"))
    lab.write_text(json.dumps({**base, "throughput_sets_s": 5.0}))
    rc, out = cli("lab-bar", "--json", str(lab))
    t("cli/lab_bar_below_rc1", rc == 1 and out.startswith("ZZLAB_BAR below thr=5.0"))
    rc, out = cli("lab-bar", "--json", str(tmp / "nope.json"))
    t("cli/lab_bar_bad_rc2", rc == 2 and out.startswith("ZZLAB_BAR bad"))
    src = CLI.read_text(encoding="utf-8")
    t("source/no_pkill_no_token_no_network", "pkill" not in src and "HF_TOKEN" not in src and "urllib" not in src and "requests" not in src and "socket" not in src)

    failed = [n for n, ok in checks if not ok]
    if failed:
        print("ZZTEST_FAIL", len(failed), failed[:6])
        return 1
    print(f"ZZTEST_PASS {len(checks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
