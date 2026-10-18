#!/usr/bin/env python3
"""Tests for refe/p6_cpu_probe.sh + refe/p6_cpu_probe.py (eval/P6_ON_11VCPU.md): the CPU-only probe the Master Mind runs on the pod between chains.

No pod, no GPU, no network, no NAVSIM stack. Three layers:
  1. `check_probe(text)` -- static properties the shell script must hold (it refuses beside a running chain with the READ-WRITE lock form, never kills by pattern, writes only
     scratch, bounds its time, closes fd 200 for every child, mirrors the chain's worker formulas and environment, holds no secret); each property is BROKEN in a copy of the
     script (a mutation) and `check_probe` must say so. A check that cannot go RED is not a check.
  2. the Python helper, each function in both directions: the cgroup readers, the bank scan (exact counts on a fixture), the verdict rules at their boundaries, the pool
     harness on the SYNTHETIC backend (thru / latency / paced / chunks), the constants pinned to the gate's own.
  3. the REAL p6_cpu_probe.sh under bash in a sandbox with stubbed tools: FIT / MARGINAL / SHORT / VOID, every refusal (each of the three chain locks, the GPU, nvidia-smi
     absent, a loaded cgroup, a second probe), a spent budget, the dd test, the scratch removed, nothing written to the real metric cache, no token reaching a child.

What ONLY the pod can prove (printed as ZZINFO, never skipped): the real NAVSIM labeller's cost, the pod's cgroup counters, NFS flock semantics, the real dd.
Run: python refe/test_p6_cpu_probe.py   -> ZZTEST_PASS <n>   (needs bash on PATH; Git Bash on the dev box). Never prints ZZTEST_SKIP.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p6_cpu_probe as P  # noqa: E402
import test_p2r_chain as T2  # noqa: E402   (logical_commands)

SCRIPT = HERE / "p6_cpu_probe.sh"
HELPER = HERE / "p6_cpu_probe.py"
CHAIN = HERE / "p6_pod.sh"
checks = []


def t(name, ok):
    checks.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


def posix(p):
    return str(p).replace("\\", "/")


def code_of(text):
    return "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("#"))


# ------------------------------------------------------------------------------------------------ layer 1: static properties
CHILD_RX = re.compile(r'"\$PY"(?!\s*\])|\bdd +if=|\bsleep +["\d$]|\btimeout\b|\bdu +-sk\b|"\$nv"|"\$\{POOL_ENV\[@\]\}"')


def check_probe(text):
    v = []
    if "\r" in text:
        v.append("CRLF")
    code = code_of(text)
    if re.search(r"\bpkill\b|\bkillall\b|\bpgrep\s+-f|\bkill\s+-9|\bkill\s+-KILL", code):
        v.append("KILL_BY_PATTERN")
    # the lock probe: read-write fd, one form only; the read-only exclusive form is EBADF on NFSv4 (MEASURED 2026-10-13)
    if re.search(r"flock[^\n]*\b9<(?![>&])", code):
        v.append("READONLY_LOCK_FD")
    if not re.search(r'\[ -e "\$lk" \] && ! \( flock -n 9 \) 9<> "\$lk"', code):
        v.append("LOCK_PROBE_FORM")
    m = re.search(r'^LOCKS="\$\{P6CPU_LOCKS:-([^"]*)\}"', code, re.M)
    locks = m.group(1) if m else ""
    for need in ("/rls1/chain.lock", "/s1aux/chain.lock", "/refe_p2rep/chain.lock"):
        if need not in locks:
            v.append("LOCK_MISSING:" + need)
    if "ZZP6CPU_REFUSED" not in code or 'refuse "the chain lock' not in code:
        v.append("NO_LOCK_REFUSAL")
    if "gpu_busy && refuse" not in code or "rc 0 = a process is on the GPU, OR the GPU cannot be asked" not in text:
        v.append("NO_GPU_REFUSAL")
    if "flock -n 200 ||" not in code:
        v.append("NO_OWN_LOCK")
    if "not a quiet window" not in code:
        v.append("NO_QUIET_REFUSAL")
    # fd 200 closed for every child
    for c in T2.logical_commands(text):
        if CHILD_RX.search(c) and "200>&-" not in c and not c.startswith(("have_time()", "gpu_busy()", "cleanup()")):
            v.append("FD200:" + c[:70])
    # no secret, no network, no address
    for ln in code.split("\n"):
        if "HF_TOKEN" in ln and "env -u HF_TOKEN" not in ln and not ln.strip().startswith("unset HF_TOKEN "):
            v.append("HF_TOKEN_USE:" + ln.strip()[:50])
    if re.search(r"hf_[A-Za-z0-9]{10,}|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\bssh\b|\bscp\b|\bcurl\b|\bwget\b|https?://", code):
        v.append("NETWORK_OR_ADDRESS")
    first = [ln for ln in code.split("\n") if ln.strip()][:3]
    if not any(ln.startswith("unset HF_TOKEN ") for ln in first):
        v.append("NO_TOKEN_DROP")
    if re.search(r"^\s*set\s+-[a-wyz]*x|\bbash\s+-x\b", text, re.M):
        v.append("XTRACE")
    # scratch only: the D9 build writes under $SCR; the real metric cache is never an output
    if 'MC="$SCR/metric_cache"' not in code or '--out "$MC"' not in code:
        v.append("D9_NOT_SCRATCH")
    for c in T2.logical_commands(text):
        if re.search(r"--out\s+\S*refe_navtrain/metric_cache|rm\s+-rf?\s+\S*refe_navtrain", c) or re.search(r"\brm\b", c) and "metric_cache" in c and "$SCR" not in c and "$MC" not in c:
            v.append("TOUCHES_REAL_CACHE:" + c[:60])
    if "trap cleanup EXIT" not in code or 'rm -f "$DATA/.p6cpu_dd"' not in code or 'rm -rf "$SCR"' not in code:
        v.append("NO_CLEANUP")
    if len(re.findall(r"\bhave_time \d+ ", code)) < 7:
        v.append("BUDGET_NOT_ENFORCED_PER_STAGE")
    if not re.search(r"BUDGET_S=\$\(\( \$\{P6CPU_BUDGET_MIN:-60\} \* 60 \)\)", code):
        v.append("BUDGET_DEFAULT_NOT_60MIN")
    # the same worker formulas and environment as p6_pod.sh (the probe must measure what the chain would run)
    if "CORES - 2 < 1 ? 1 : (CORES - 2 > 16 ? 16 : CORES - 2)" not in code:
        v.append("LW_FORMULA")
    if "CORES < 1 ? 1 : (CORES > 24 ? 24 : CORES)" not in code:
        v.append("CW_FORMULA")
    if not re.search(r"OMP_NUM_THREADS=4[^\n]*diag_p6_labeller_identity\.py", code):
        v.append("DIAG_NOT_UNDER_OMP4")
    if not re.search(r"OMP_NUM_THREADS=1 [^\n]*nice -n 10[^\n]*build_metric_cache\.py build", code.replace("\\\n", " ")):
        v.append("D9_ENV_NOT_THE_CHAINS")
    if 'P6CPU_LWS:-7 8 9 10' not in code:
        v.append("LWS_DEFAULT")
    m = re.search(r'NSETS="\$\{P6CPU_SETS:-(\d+)\}"', code)
    if not m or int(m.group(1)) < 256:
        v.append("SETS_BELOW_256")
    if not re.search(r'diag_p6_labeller_identity\.py[^\n]*--workers "\$W"[^\n]*--n-sets "\$NSETS"', code.replace("\\\n", " ")):
        v.append("DIAG_ARGS")
    if 'refe/diag_p6_labeller_identity.py' not in code:
        v.append("DIAG_NOT_REQUIRED")
    # the dd test: chunked, stops at the first error, bounded, the file removed
    if not re.search(r"oflag=append conv=notrunc,fsync", code) or "write_error" not in code or "time_cap" not in code:
        v.append("DD_NOT_CHUNKED_BOUNDED")
    # the verdict is one of the named lines, from the helper
    htxt = HELPER.read_text(encoding="utf-8")
    if 'f"ZZP6CPU_{v} ' not in htxt:
        v.append("VERDICT_MISSING:line")
    for k in ('return "FIT", ', 'return "MARGINAL", ', 'return "SHORT", ', 'return "VOID", '):
        if k not in htxt:
            v.append("VERDICT_MISSING:" + k)
    # order of the stages
    order = ["flock -n 200 ||", "gpu_busy && refuse", "ZZP6CPU_ENV", "ZZP6CPU_QUIET", " sample-bank ", " build_metric_cache.py build", "diag_p6_labeller_identity.py --cache",
            "--mode \"$M\"", "ZZP6CPU_AB", "ZZP6CPU_CHUNK2", "ZZP6CPU_DISK", "ZZP6CPU_DD", " verdict "]
    flat = code.replace("\\\n", " ")
    pos = [flat.find(k) for k in order]
    for k, p in zip(order, pos):
        if p < 0:
            v.append("MISSING:" + k.strip())
    if all(p >= 0 for p in pos) and pos != sorted(pos):
        v.append("ORDER")
    return v


def mutations(text):
    d = lambda s, sub: s.replace(sub, "", 1)           # noqa: E731
    return [
        ("no lock refusal", text.replace('refuse "the chain lock $lk is held', 'log "the chain lock $lk is held'), "NO_LOCK_REFUSAL"),
        ("a read-only lock fd", text.replace('( flock -n 9 ) 9<> "$lk"', '( flock -n 9 ) 9< "$lk"'), "READONLY_LOCK_FD"),
        ("no -e guard (the rw open would create a lock file)", text.replace('[ -e "$lk" ] && ! ( flock', '! ( flock'), "LOCK_PROBE_FORM"),
        ("s1aux lock dropped", text.replace("$DATA/s1aux/chain.lock ", ""), "LOCK_MISSING:/s1aux/chain.lock"),
        ("p2 replicate lock dropped", text.replace("$DATA/refe_p2rep/chain.lock ", ""), "LOCK_MISSING:/refe_p2rep/chain.lock"),
        ("rls1 lock dropped", text.replace("$DATA/rls1/chain.lock ", ""), "LOCK_MISSING:/rls1/chain.lock"),
        ("pkill", text + "\npkill -f labeller\n", "KILL_BY_PATTERN"),
        ("pgrep -f", text + "\npgrep -f train.py\n", "KILL_BY_PATTERN"),
        ("a child without fd 200 closed", text.replace('--workers "$W" --n-sets "$NSETS" --out "$PR/lab_w$W.json" > "$PR/lab_w$W.log" 2>&1 200>&-',
                                                       '--workers "$W" --n-sets "$NSETS" --out "$PR/lab_w$W.json" > "$PR/lab_w$W.log" 2>&1'), "FD200"),
        ("the GPU is not asked", text.replace('gpu_busy && refuse', 'gpu_busy || true #'), "NO_GPU_REFUSAL"),
        ("the quiet check is gone", text.replace("not a quiet window", "ignored"), "NO_QUIET_REFUSAL"),
        ("D9 writes the real cache", text.replace('--out "$MC"', '--out "$DATA/refe_navtrain/metric_cache"'), "D9_NOT_SCRATCH"),
        ("no cleanup trap", d(text, "trap cleanup EXIT"), "NO_CLEANUP"),
        ("a token reaches a child", text.replace("env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK \"${SIMHZ_ENV[@]}\")", "env \"${SIMHZ_ENV[@]}\")\nexport HF_TOKEN_X=$HF_TOKEN"), "HF_TOKEN_USE"),
        ("the token is not dropped", text.replace("unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN\nHERE=", "HERE=", 1), "NO_TOKEN_DROP"),
        ("a network call", text + "\ncurl -s https://example.org\n", "NETWORK_OR_ADDRESS"),
        ("an address", text + "\nX=10.1.2.3\n", "NETWORK_OR_ADDRESS"),
        ("workers from nproc", re.sub(r"^LWD=.*$", 'LWD="$(nproc)"', text, flags=re.M), "LW_FORMULA"),
        ("the diag under another OMP", text.replace("OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 timeout \"${P6CPU_LAB_TMO", "OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 timeout \"${P6CPU_LAB_TMO"), "DIAG_NOT_UNDER_OMP4"),
        ("64 sets only", text.replace('P6CPU_SETS:-256', 'P6CPU_SETS:-64'), "SETS_BELOW_256"),
        ("one worker count", text.replace('P6CPU_LWS:-7 8 9 10', 'P6CPU_LWS:-7'), "LWS_DEFAULT"),
        ("no time budget", text.replace("P6CPU_BUDGET_MIN:-60", "P6CPU_BUDGET_MIN:-6000"), "BUDGET_DEFAULT_NOT_60MIN"),
        ("a stage without its budget check", text.replace("have_time 420 d9;", "true;"), "BUDGET_NOT_ENFORCED_PER_STAGE"),
        ("a dd in one piece", text.replace("oflag=append conv=notrunc,fsync", "conv=fsync"), "DD_NOT_CHUNKED_BOUNDED"),
        ("the verdict before the dd", text.replace("ZZP6CPU_DD", "ZZP6CPU_DX"), "MISSING:ZZP6CPU_DD"),
        ("xtrace", text.replace("set -u\n", "set -ux\n", 1), "XTRACE"),
    ]


def layer1():
    text = SCRIPT.read_text(encoding="utf-8")
    base = check_probe(text)
    t("probe/unmutated_script_has_no_violation", base == [])
    if base:
        print("      ", base)
    for name, mt, want in mutations(text):
        got = check_probe(mt)
        t(f"probe/mutation_caught: {name}", any(g.startswith(want) for g in got))
    # the chain's formulas, read from p6_pod.sh itself (the probe must not drift from it)
    chain = CHAIN.read_text(encoding="utf-8")
    t("probe/lw_formula_equals_p6_pod_sh", "CORES - 2 < 1 ? 1 : (CORES - 2 > 16 ? 16 : CORES - 2)" in chain)
    t("probe/cw_formula_equals_p6_pod_sh", "CORES < 1 ? 1 : (CORES > 24 ? 24 : CORES)" in chain)
    t("probe/chain_exports_omp4_before_the_lab_stage", re.search(r"export OMP_NUM_THREADS=4[^\n]*\n(?:.*\n)*?.*diag_p6_labeller_identity\.py", chain) is not None)
    t("probe/chain_sets_no_sim_hz_so_neither_does_the_probe_by_default", "REFE_SIM_HZ" not in code_of(chain) and not re.search(r"REFE_SIM_HZ=\d", code_of(text)))
    t("probe/shell_syntax", subprocess.run([shutil.which("bash"), "-n", str(SCRIPT)], capture_output=True).returncode == 0)
    r = subprocess.run([sys.executable, str(HERE / "test_flock_rw.py")], capture_output=True, text=True)
    t("probe/test_flock_rw_passes_with_the_probe_in_the_tree", r.returncode == 0 and "ZZTEST_PASS" in r.stdout)


# ------------------------------------------------------------------------------------------------ layer 2: the helper
def tree(root, files):
    for rel, body in files.items():
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")


def layer2():
    tmp = Path(tempfile.mkdtemp(prefix="p6cpu_unit_"))
    try:
        # constants == the gate's
        gc = (HERE / "gate_checks.py").read_text(encoding="utf-8")
        m = re.search(r"^LAB_NEED_SETS_S, LAB_MARGIN, LAB_STALL_MAX, MEM_MAX_GB, MEM_MIN_MICRO = ([\d.]+), ([\d.]+), ([\d.]+), ", gc, re.M)
        t("helper/constants_equal_the_gates_own", bool(m) and (float(m.group(1)), float(m.group(2)), float(m.group(3))) == (P.NEED_SETS_S, P.MARGIN, P.STALL_MAX))
        t("helper/gate_bar_is_6_735", P.BAR_GATE == 6.735)
        t("helper/micro_batch_is_57_over_64", abs(P.MICRO_S - 57.0 / 64.0) < 1e-12)
        # cgroup readers
        tree(tmp / "v2", {"cpu.max": "1100000 100000\n"})
        tree(tmp / "v2max", {"cpu.max": "max 100000\n"})
        tree(tmp / "v1", {"cpu/cpu.cfs_quota_us": "1105000\n", "cpu/cpu.cfs_period_us": "100000\n"})
        tree(tmp / "v1unl", {"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"})
        t("cgroup/v2_cpu_max", P.cgroup_cores(str(tmp / "v2")) == (11.0, "cgroup v2 cpu.max"))
        c, src = P.cgroup_cores(str(tmp / "v1"))
        t("cgroup/v1_cfs_is_11_05_not_96", abs(c - 11.05) < 1e-9 and src == "cgroup v1 cfs")
        c, src = P.cgroup_cores(str(tmp / "v2max"), fallback=96)
        t("cgroup/max_falls_back_and_says_UNVERIFIED", c == 96.0 and "UNVERIFIED" in src)
        c, src = P.cgroup_cores(str(tmp / "v1unl"), fallback=96)
        t("cgroup/unlimited_v1_falls_back_and_says_UNVERIFIED", c == 96.0 and "UNVERIFIED" in src)
        c, src = P.cgroup_cores(str(tmp / "nothing"), fallback=7)
        t("cgroup/no_files_falls_back", c == 7.0 and "UNVERIFIED" in src)
        tree(tmp / "cg1", {"cpuacct/cpuacct.usage": "5000000000\n", "cpu/cpu.stat": "nr_periods 100\nnr_throttled 7\nthrottled_time 3000000000\n",
                           "memory/memory.limit_in_bytes": "70999998464\n", "memory/memory.usage_in_bytes": "1000\n", "memory/memory.stat": "total_rss 400\ntotal_cache 600\n"})
        a = P.cg_counters(str(tmp / "cg1"))
        t("cgroup/v1_counters", a["version"] == 1 and a["usage_s"] == 5.0 and a["nr_throttled"] == 7 and a["throttled_s"] == 3.0 and a["mem_limit"] == 70999998464
          and a["mem_rss"] == 400 and a["mem_cache"] == 600)
        tree(tmp / "cg2", {"cpu.stat": "usage_usec 7000000\nnr_periods 10\nnr_throttled 1\nthrottled_usec 500000\n", "memory.max": "max\n", "memory.current": "42\n",
                           "memory.stat": "anon 10\nfile 20\n"})
        b = P.cg_counters(str(tmp / "cg2"))
        t("cgroup/v2_counters_and_unlimited_memory", b["version"] == 2 and b["usage_s"] == 7.0 and b["mem_limit"] is None and b["mem_usage"] == 42 and b["mem_rss"] == 10)
        t("cgroup/unreadable_is_empty_not_zero", P.cg_counters(str(tmp / "nothing")) == {} and P.cg_delta({}, a) == {})
        dl = P.cg_delta(a, dict(a, usage_s=9.5, nr_throttled=9, nr_periods=130, throttled_s=4.25))
        t("cgroup/delta", dl == {"cpu_s": 4.5, "nr_throttled": 2, "nr_periods": 30, "throttled_s": 1.25})
        # the bank scan: exact counts on a fixture whose key fields sit at the TAIL of long lines, as the real bank's do
        def row(lg, tk, rank=0, aug=None, pad=3300):
            d = {"image": ["x" * pad], "ego": [0.0] * 4, "rank": rank, "step": 0, "token": tk, "log_name": lg, "source": "s"}
            if aug is not None:
                d["aug"] = aug
            return json.dumps(d)
        bank = tmp / "bank"
        bank.mkdir()
        logs = [f"L{i}" for i in range(6)]
        with open(bank / "targets_rank0.jsonl", "w", encoding="utf-8") as fh:
            for lg in logs:
                for k in range(5):
                    fh.write(row(lg, f"{lg}t{k}") + "\n")
        with open(bank / "targets_rank1.jsonl", "w", encoding="utf-8") as fh:
            fh.write(row("L0", "L0t0", 1, {"kind": "goal_horizon_s", "value": 16.0}) + "\n")      # shares the logged route's parts: NOT a twin key
            fh.write(row("L0", "L0t1", 1, {"kind": "lane_rank", "value": 2, "divergence_m": 0.3, "lanes": 4}) + "\n")
            fh.write(row("L0", "L0t1", 1, {"kind": "lane_rank", "value": 3, "divergence_m": 0.3, "lanes": 4}) + "\n")
            fh.write(row("L1", "L1t0", 1, {"value": 2, "kind": "lane_rank"}) + "\n")               # key order reversed
        for lg in logs:
            (tmp / "dbs").mkdir(exist_ok=True)
            (tmp / "dbs" / f"{lg}.db").write_text("", encoding="utf-8")
        res = P.sample_bank(str(bank), str(tmp / "dbs"), 3, str(tmp / "samp"))
        t("bank/exact_workload_counts", res["plain_tokens"] == 30 and res["lane_twin_keys"] == 3 and res["lane_values"] == [2, 3] and res["bank_logs"] == 6 and res["complete"])
        t("bank/sample_is_whole_logs_evenly_spaced", res["sample_logs"] == 3 and res["sample_tokens"] == 15 and res["sample_log_names"] == ["L0", "L2", "L4"])
        import build_metric_cache as BM
        keys = BM.keys_of_bank(str(tmp / "samp"))
        t("bank/the_sample_is_a_bank_build_metric_cache_reads", sorted(keys) == ["L0", "L2", "L4"] and all(len(v) == 5 for v in keys.values()))
        r2 = P.sample_bank(str(bank), str(tmp / "dbs"), 3, str(tmp / "samp2"), max_scan_s=-1.0)
        t("bank/a_capped_scan_says_INCOMPLETE", r2["complete"] is False and r2["rows_seen"] >= 20000 or r2["complete"] is True and r2["rows_seen"] < 20000)
        t("bank/head_row_none_on_garbage", P.head_row("not a row") is None and P.head_row('{"log_name": null, "token": "t"}') is None)
        # the verdict, at its boundaries
        ok = {"identity_ok": True, "stale_refused": True}
        pl = lambda thr, st: dict(ok, gate_sets_s=thr, paced_stall_frac=st)    # noqa: E731
        v = lambda per, lwd=7: P.decide(11.05, lwd, per)[0]                     # noqa: E731
        t("verdict/FIT", v({7: pl(7.0, 0.01), 8: pl(7.5, 0.01)}) == "FIT")
        t("verdict/FIT_at_exactly_the_bar", v({7: pl(6.735, 0.05)}) == "FIT")
        t("verdict/just_below_the_bar_is_not_FIT", v({7: pl(6.734, 0.01)}) == "MARGINAL")
        t("verdict/stall_over_5pct_is_not_FIT", v({7: pl(9.0, 0.0501)}) == "MARGINAL")
        t("verdict/no_paced_data_is_not_FIT", v({7: dict(ok, gate_sets_s=9.0, paced_stall_frac=None)}) == "MARGINAL")
        t("verdict/MARGINAL_when_another_W_passes", P.decide(11.05, 7, {7: pl(5.0, 0.2), 8: pl(7.0, 0.01)}) == ("MARGINAL", P.decide(11.05, 7, {7: pl(5.0, 0.2), 8: pl(7.0, 0.01)})[1], 8))
        t("verdict/MARGINAL_at_exactly_the_need", v({7: pl(4.49, 0.5)}) == "MARGINAL")
        t("verdict/SHORT_just_below_the_need", v({7: pl(4.489, 0.0), 8: pl(4.0, 0.0)}) == "SHORT")
        t("verdict/VOID_on_identity_failure_even_with_a_great_rate", v({7: pl(20.0, 0.0), 8: dict(pl(20.0, 0.0), identity_ok=False)}) == "VOID")
        t("verdict/VOID_when_the_stale_arm_is_not_refused", v({7: dict(pl(20.0, 0.0), stale_refused=False)}) == "VOID")
        t("verdict/VOID_when_nothing_was_measured", v({}) == "VOID")
        t("verdict/default_W_missing_cannot_be_FIT", v({8: pl(9.0, 0.0)}, lwd=7) == "MARGINAL")
        # P6's wall time if the labeller lags
        t("days/at_the_need_it_is_the_accepted_6_8", abs(P.step_time_days(4.49) - 6.65 * 1.02) < 1e-9)
        t("days/half_the_rate_doubles_it", abs(P.step_time_days(4.49 / 2) - 2 * 6.65 * 1.02) < 1e-9 and P.step_time_days(100.0) == P.step_time_days(4.49))
        t("helper/percentile", P.pct([1, 2, 3, 4], 0.5) == 2.5 and P.pct([], 0.5) is None and P.pct([5], 0.95) == 5)
        # d9 arithmetic
        cache = tmp / "mc"
        for i in range(10):
            (cache / "L0").mkdir(parents=True, exist_ok=True)
            (cache / "L0" / f"t{i}.pkl.xz").write_bytes(b"x" * 1000)
        (tmp / "bs.json").write_text(json.dumps({"plain_tokens": 100000, "lane_twin_keys": 60000}), encoding="utf-8")
        d9 = P.d9_summary({"usage_s": 100.0}, {"usage_s": 120.0}, 4.0, 10, str(tmp / "bs.json"), str(cache))
        t("d9/arithmetic", d9["tokens_built"] == 10 and d9["cpu_s_per_token"] == 2.0 and d9["full_tokens"] == 160000 and d9["full_cpu_h"] == round(160000 * 2.0 / 3600, 2)
          and d9["full_wall_h"] == round(160000 * 2.0 / (10 * 0.9) / 3600, 2) and d9["bytes_per_token"] == 1000 and d9["full_cache_gb"] == 0.2)
        # the pool harness, synthetic backend: instrumentation, all three modes, chunks
        toks = ",".join(f"L:{i}" for i in range(8))
        r = P.run_pool(str(tmp), toks, 4, 24, "thru", backend="synthetic")
        t("pool/thru_labels_everything", r["labelled_jobs"] == 24 and r["expected_jobs"] == 24 and r["sets_per_s"] > 0 and r["mode"] == "thru")
        r = P.run_pool(str(tmp), toks, 4, 24, "thru", backend="synthetic", chunks=2)
        t("pool/chunks_2_makes_2_jobs_per_set_and_labels_all", r["chunks"] == 2 and r["expected_jobs"] == 48 and r["labelled_jobs"] == 48 and r["n_sets"] == 24)
        r = P.run_pool(str(tmp), toks, 4, 24, "latency", backend="synthetic")
        t("pool/latency_records_one_value_per_micro_batch", r["micro_latency_s"]["n"] == 6 and r["micro_latency_s"]["p95"] >= r["micro_latency_s"]["p50"] >= 0)
        r = P.run_pool(str(tmp), toks, 4, 24, "paced", backend="synthetic", period=0.03, n_micro=8)
        t("pool/paced_reports_a_stall_fraction_in_0_1", r["stall_frac"] is not None and 0.0 <= r["stall_frac"] <= 1.0 and r["period_s"] == 0.03 and r["n_sets"] == 32)
        try:
            P.run_pool(str(tmp), toks, 4, 24, "thru", backend="synthetic", chunks=3)
            t("pool/chunks_must_divide_64", False)
        except SystemExit:
            t("pool/chunks_must_divide_64", True)
        # the process tree reader sees a real child
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
        try:
            import time as _t
            _t.sleep(0.6)
            s = P.tree_snapshot()
            t("tree/sees_a_real_child_with_rss", s["n"] >= 1 and s["rss_max"] > 0 and s["threads_max"] >= 1)
        finally:
            child.terminate()
            child.wait()
        # the verdict CLI end to end on a directory of stage JSONs
        d = tmp / "pr"
        d.mkdir()
        for w, thr, st in ((7, 5.2, 0.12), (8, 6.9, 0.03), (9, 7.0, 0.03), (10, 7.0, 0.03)):
            (d / f"lab_w{w}.json").write_text(json.dumps({"identity_ok": True, "stale_refused": True, "throughput_sets_s": thr, "n_sets": 256, "identity_equal_sets": 64}))
            (d / f"pool_paced_w{w}.json").write_text(json.dumps({"stall_frac": st}))
            (d / f"pool_thru_w{w}.json").write_text(json.dumps({"cpu_per_set_tree_s": 0.9, "rss_worker_max_mb": 300.0, "sets_per_s": thr, "cg": {"nr_throttled": 0}, "threads_worker_max": 11}))
        (d / "dd.json").write_text(json.dumps({"gb_ok": 96, "gb_target": 96}))
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = P.main(["verdict", "--dir", str(d), "--cores", "11.05", "--lw-default", "7", "--workers", "7,8,9,10"])
        out = buf.getvalue().strip().splitlines()
        t("verdict_cli/one_line_MARGINAL_with_the_numbers", rc == 0 and len(out) == 1 and out[0].startswith("ZZP6CPU_MARGINAL ") and "bar_gate=6.735" in out[0] and "best_lw=8" in out[0]
          and "dd_gb_ok=96/96" in out[0])
        t("verdict_cli/writes_verdict_json", json.loads((d / "verdict.json").read_text())["verdict"] == "MARGINAL")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ layer 3: the real script in a sandbox
FAKEPY = r'''
import json, os, runpy, sys
from pathlib import Path
REAL = Path(os.environ["STUB_REAL_REFE"])
argv = sys.argv[1:]
script = os.path.basename(argv[0]) if argv else ""
rest = argv[1:]
with open(os.environ["STUB_CALLS"], "a", encoding="utf-8") as f:
    f.write("%s %s | OMP=%s OPENBLAS=%s HF=%s SIMHZ=%s\n" % (script, " ".join(rest), os.environ.get("OMP_NUM_THREADS", ""), os.environ.get("OPENBLAS_NUM_THREADS", ""),
                                                          os.environ.get("HF_TOKEN", ""), os.environ.get("REFE_SIM_HZ", "")))
def opt(name, default=None):
    return rest[rest.index(name) + 1] if name in rest else default
def thr_of(w):
    d = dict(x.split(":") for x in os.environ.get("STUB_THRU", "7:9.0,8:9.5,9:9.5,10:9.5").split(","))
    return float(d.get(str(w), 9.0))
def stall_of(w, chunks):
    base = float(os.environ.get("STUB_STALL", "0.01"))
    d = dict(x.split(":") for x in os.environ.get("STUB_STALL_W", "").split(",") if x)
    s = float(d.get(str(w), base))
    return s / 3 if chunks == 2 else s
def real(path, args):
    sys.path.insert(0, str(REAL))
    sys.argv = [str(path)] + args
    runpy.run_path(str(path), run_name="__main__")
if script == "p6_cpu_probe.py":
    sub = rest[0]
    if sub in ("cores", "d9-summary", "verdict", "cg-rate") or (sub == "cg" and not os.environ.get("STUB_CG_BUSY")):
        real(REAL / "p6_cpu_probe.py", rest)
    elif sub == "cg":
        c = Path(os.environ["STUB_COUNT"]); n = int(c.read_text()) if c.exists() else 0; c.write_text(str(n + 1))
        print("ZZCG " + json.dumps({"version": 1, "usage_s": 20.0 * n, "nr_periods": 0, "nr_throttled": 0, "throttled_s": 0.0}))
    elif sub == "sample-bank":
        out = Path(opt("--out")); out.mkdir(parents=True, exist_ok=True)
        with open(out / "targets_rank0.jsonl", "w", encoding="utf-8") as fh:
            for lg in ("L0", "L1"):
                for k in range(3):
                    fh.write(json.dumps({"log_name": lg, "token": "%st%d" % (lg, k), "rank": 0, "aug": None}) + "\n")
        r = {"complete": True, "plain_tokens": 103037, "lane_twin_keys": 65000, "sample_tokens": 6, "sample_logs": 2, "bank_logs": 1192}
        Path(opt("--json")).write_text(json.dumps(r)); print("ZZBANK_SAMPLE " + json.dumps(r))
    elif sub == "pool":
        w = int(opt("--workers")); mode = opt("--mode"); chunks = int(opt("--chunks", "1"))
        if os.environ.get("STUB_POOL_FAIL"):
            sys.exit(3)
        r = {"mode": mode, "workers": w, "chunks": chunks, "n_sets": int(opt("--n-sets")), "sets_per_s": thr_of(w) if mode == "thru" else None,
             "cpu_per_set_tree_s": 0.5, "rss_worker_max_mb": 281.0, "threads_worker_max": 11, "cg": {"nr_throttled": 0}}
        if mode in ("latency", "paced"):
            r["micro_latency_s"] = {"p50": 0.5, "p95": 0.6, "max": 0.7, "n": 5}
        if mode == "paced":
            r["stall_frac"] = stall_of(w, chunks)
        Path(opt("--out")).write_text(json.dumps(r)); print("ZZPOOL_OK " + json.dumps({"mode": mode}))
    else:
        sys.exit(2)
elif script == "p6_prep.py":
    real(REAL / "p6_prep.py", rest)
elif script == "build_metric_cache.py":
    out = Path(opt("--out")); n = 0; logs = set()
    for ln in open(Path(opt("--targets")) / "targets_rank0.jsonl", encoding="utf-8"):
        r = json.loads(ln); p = out / r["log_name"] / (r["token"] + ".pkl.xz"); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"p" * 2048); n += 1; logs.add(r["log_name"])
    print("ZZMETRIC_CACHE_DONE built=%d skipped_5s_guard=0 failed=%d logs=%d 1s" % (n, 1 if os.environ.get("STUB_D9_FAIL") else 0, len(logs)))
elif script == "diag_p6_labeller_identity.py":
    w = int(opt("--workers")); bad = [int(x) for x in os.environ.get("STUB_IDENT_BAD", "").split(",") if x]
    ok = w not in bad
    Path(opt("--out")).write_text(json.dumps({"identity_ok": ok, "n_sets": int(opt("--n-sets")), "identity_equal_sets": 64 if ok else 60, "stale_refused": True,
                                              "throughput_sets_s": thr_of(w), "workers": w}))
    print("ZZLAB_IDENTITY_PASS" if ok else "ZZLAB_IDENTITY_FAIL"); print("ZZLAB_THROUGHPUT %s" % thr_of(w))
else:
    print("fakepy: unknown stub " + script); sys.exit(2)
'''
BASH_ENV = r'''
# flock stub: the real one is not in Git Bash.  `flock -n N` fails iff the file behind fd N is in STUB_HELD (by the name of its parent directory, or "probe" for probe.lock).
flock() {
  local fd="${@: -1}" tgt who
  tgt="$(readlink /dev/fd/$fd 2> /dev/null)"
  case "$(basename "$tgt")" in probe.lock) who=probe ;; *) who="$(basename "$(dirname "$tgt")")" ;; esac
  case " ${STUB_HELD:-} " in *" $who "*) return 1 ;; esac
  return 0
}
'''
NVSMI = '#!/bin/bash\n[ -n "${STUB_GPU_PIDS:-}" ] && echo "$STUB_GPU_PIDS"\nexit 0\n'
REQUIRED = ["refe/p6_cpu_probe.py", "refe/diag_p6_labeller_identity.py", "refe/build_metric_cache.py", "refe/p6_prep.py", "refe/online_labeller.py", "refe/navsim_pdm_targets.py",
            "refe/navtrain_scenarios.py", "splits/navtrain.yaml"]


class Sandbox:
    def __init__(self, root):
        self.root = Path(root)
        self.ws = self.root / "ws"
        self.data = self.ws / "data"
        self.pkg = self.ws / "refe-plan-p6"
        self.stub = self.root / "stub"
        self.calls = self.root / "calls.log"
        self.cg = self.root / "cg"
        self.pr = self.data / "refe_p6cpu"
        for d in (self.pkg / "refe", self.pkg / "splits", self.stub, self.data / "navtrain_dbs", self.data / "refe_navtrain" / "train_grow", self.cg / "cpu", self.cg / "cpuacct"):
            d.mkdir(parents=True, exist_ok=True)
        self.calls.write_text("", encoding="utf-8")
        for f in REQUIRED:
            (self.pkg / f).write_text("", encoding="utf-8")
        shutil.copy(HELPER, self.pkg / "refe" / "p6_cpu_probe.py")
        shutil.copy(SCRIPT, self.pkg / "refe" / "p6_cpu_probe.sh")
        (self.pkg / "refe" / "p6_cpu_probe.sh").write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        (self.data / "refe_navtrain" / "train_grow" / "targets_rank0.jsonl").write_text("{}\n", encoding="utf-8")
        (self.cg / "cpu" / "cpu.cfs_quota_us").write_text("1105000\n")
        (self.cg / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
        (self.cg / "cpuacct" / "cpuacct.usage").write_text("0\n")
        for who in ("rls1", "s1aux", "refe_p2rep"):
            (self.data / who).mkdir(parents=True, exist_ok=True)
            (self.data / who / "chain.lock").write_text("", encoding="utf-8")
        (self.stub / "fakepy.py").write_text(FAKEPY, encoding="utf-8")
        py = posix(sys.executable)
        (self.stub / "fakepy").write_text(f'#!/bin/bash\nif [ "$1" = "-" ]; then exec "{py}" "$@"; fi\nexec "{py}" "{posix(self.stub / "fakepy.py")}" "$@"\n', encoding="utf-8", newline="\n")
        (self.stub / "nvsmi").write_text(NVSMI, encoding="utf-8", newline="\n")
        (self.stub / "bash_env.sh").write_text(BASH_ENV, encoding="utf-8", newline="\n")
        subprocess.run([shutil.which("bash"), "-c", f"chmod +x '{posix(self.stub / 'fakepy')}' '{posix(self.stub / 'nvsmi')}'"], check=True)
        (self.ws / "teacher_env.sh").write_text(f'export DRIVERL_EVAL_PYTHON="{posix(self.stub / "fakepy")}" NUPLAN_MAPS_ROOT="{posix(self.data / "maps")}"\n', encoding="utf-8", newline="\n")

    def run(self, **env):
        e = dict(os.environ)
        e.update(WORK=posix(self.ws), P6CPU_PKG=posix(self.pkg), P6CPU_CG_ROOT=posix(self.cg), P6CPU_NVSMI=posix(self.stub / "nvsmi"), STUB_CALLS=posix(self.calls),
                 STUB_REAL_REFE=posix(HERE), STUB_COUNT=posix(self.root / "count"), BASH_ENV=posix(self.stub / "bash_env.sh"), P6CPU_QUIET_S="1", P6CPU_DD_MB="24",
                 P6CPU_DD_CHUNK_MB="8", HF_TOKEN="hf_" + "Q" * 24, PYTHONIOENCODING="utf-8")
        for k in ("STUB_HELD", "STUB_GPU_PIDS", "STUB_CG_BUSY", "STUB_THRU", "STUB_STALL", "STUB_STALL_W", "STUB_IDENT_BAD", "STUB_D9_FAIL", "STUB_POOL_FAIL"):
            e.pop(k, None)
        e.update({k: str(v) for k, v in env.items()})
        (self.root / "count").unlink(missing_ok=True)
        self.calls.write_text("", encoding="utf-8")
        r = subprocess.run([shutil.which("bash"), posix(self.pkg / "refe" / "p6_cpu_probe.sh")], env=e, capture_output=True, text=True, timeout=300)
        return r.returncode, r.stdout + r.stderr

    def calls_text(self):
        return self.calls.read_text(encoding="utf-8")

    def log(self):
        p = self.pr / "probe.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""


def verdict_line(out):
    ls = [ln for ln in out.splitlines() if re.search(r"(^|\s)ZZP6CPU_(FIT|MARGINAL|SHORT|VOID|REFUSED) ", ln)]
    return re.sub(r"^\S+T\S+ ", "", ls[-1]) if ls else ""


def layer3():
    if not shutil.which("bash"):
        t("sandbox/bash_is_on_PATH (Git Bash on the dev box)", False)
        return
    root = Path(tempfile.mkdtemp(prefix="p6cpu_sbx_"))
    try:
        sb = Sandbox(root)
        # ---- FIT: every worker count passes, the default one included
        rc, out = sb.run()
        vl = verdict_line(out)
        t("sandbox/FIT_path_prints_ZZP6CPU_FIT_and_exits_0", rc == 0 and vl.startswith("ZZP6CPU_FIT "))
        t("sandbox/the_verdict_line_carries_the_numbers", all(k in vl for k in ("cores=11.05", "lw_default=9", "bar_gate=6.735", "bar_need=4.49", "gate_sets_s=", "paced_stall=", "cpu_s_per_set=",
                                                                                 "d9_full_wall_h=", "dd_gb_ok=", "p6_days_if_best_rate=")))
        calls = sb.calls_text()
        t("sandbox/diag_ran_once_per_worker_count_with_256_sets", all(re.search(rf"diag_p6_labeller_identity\.py .*--workers {w} --n-sets 256", calls) for w in (7, 8, 9, 10)))
        t("sandbox/the_diag_ran_under_the_chains_OMP_4_and_the_D9_build_under_1", all("OMP=4" in ln for ln in calls.splitlines() if ln.startswith("diag_p6_labeller_identity.py"))
          and all("OMP=1" in ln for ln in calls.splitlines() if ln.startswith("build_metric_cache.py")))
        t("sandbox/the_AB_arm_ran_with_1_BLAS_thread", any(ln.startswith("p6_cpu_probe.py pool") and "OPENBLAS=1" in ln and "_omp1.json" in ln for ln in calls.splitlines()))
        t("sandbox/the_chunk_2_arm_ran_at_the_default_W", sum(1 for ln in calls.splitlines() if "--chunks 2" in ln and "--workers 9" in ln) == 2)    # the default W = CORES - 2 (p6_pod.sh, 2026-10-18)
        t("sandbox/D9_build_used_the_cgroup_cores_as_workers_and_the_scratch_dir", any(ln.startswith("build_metric_cache.py build") and "--workers 11" in ln and "/scratch/metric_cache" in ln
                                                                                        for ln in calls.splitlines()))
        t("sandbox/no_token_reached_any_child", all(re.search(r"HF=(\S*)", ln).group(1) == "" for ln in calls.splitlines() if ln.strip()))
        t("sandbox/REFE_SIM_HZ_is_not_set_unless_asked", all(re.search(r"SIMHZ=\s*$", ln) for ln in calls.splitlines() if ln.strip()))
        t("sandbox/scratch_is_removed", not (sb.pr / "scratch").exists())
        t("sandbox/the_dd_file_is_removed", not (sb.data / ".p6cpu_dd").exists())
        t("sandbox/the_real_metric_cache_dir_was_never_created", not (sb.data / "refe_navtrain" / "metric_cache").exists())
        t("sandbox/no_lock_file_was_created_by_the_probe", not (sb.data / "refe_p2" / "chain.lock").exists() and not (sb.data / "refe_p6" / "chain.lock").exists())
        dd = json.loads((sb.pr / "dd.json").read_text())
        t("sandbox/dd_reached_its_target_in_chunks", dd["mb_ok"] >= 24 and dd["stopped_by"] == "target" and dd["gb_target"] == 0)
        lg = sb.log()
        idx = [lg.find(k) for k in ("ZZP6CPU_START", "ZZP6CPU_ENV", "ZZP6CPU_QUIET", "ZZBANK_SAMPLE", "ZZP6CPU_D9", "ZZP6CPU_LW w=7", "ZZP6CPU_AB", "ZZP6CPU_CHUNK2", "ZZP6CPU_DISK", "ZZP6CPU_DD",
                                    "ZZP6CPU_FIT", "ZZP6CPU_DONE")]
        t("sandbox/log_markers_in_stage_order", all(i >= 0 for i in idx) and idx == sorted(idx))
        t("sandbox/the_probe_log_holds_no_token", "hf_" + "Q" * 24 not in lg and "hf_QQQ" not in lg)
        # ---- MARGINAL: the default W is under the gate's bar but over the need
        rc, out = sb.run(STUB_THRU="7:5.0,8:6.0,9:6.5,10:6.5")
        t("sandbox/MARGINAL_when_the_default_W_is_under_the_bar", rc == 0 and verdict_line(out).startswith("ZZP6CPU_MARGINAL "))
        # ---- MARGINAL: the gate passes but the trainer's clock stalls over 5 %
        rc, out = sb.run(STUB_STALL_W="9:0.2")                                     # at the default W (CORES - 2 = 9 on 11.05 cores)
        t("sandbox/MARGINAL_when_throughput_passes_but_the_paced_stall_does_not", verdict_line(out).startswith("ZZP6CPU_MARGINAL "))
        # ---- SHORT
        rc, out = sb.run(STUB_THRU="7:3.0,8:3.5,9:3.6,10:3.6")
        t("sandbox/SHORT_when_no_W_reaches_the_need", rc == 0 and verdict_line(out).startswith("ZZP6CPU_SHORT "))
        # ---- VOID: an identity failure voids the throughput
        rc, out = sb.run(STUB_IDENT_BAD="8")
        t("sandbox/VOID_when_identity_fails_at_any_W", verdict_line(out).startswith("ZZP6CPU_VOID "))
        # ---- VOID: the D9 build failed -> no parts -> no pool
        rc, out = sb.run(STUB_D9_FAIL="1")
        t("sandbox/VOID_and_no_pool_when_D9_failed", verdict_line(out).startswith("ZZP6CPU_VOID ") and "diag_p6_labeller_identity.py" not in sb.calls_text())
        # ---- refusals: each lock, the GPU, nvidia-smi absent, a loaded cgroup, a second probe -> exit 1, a REFUSED line, and NO work done
        for who in ("rls1", "s1aux", "refe_p2rep"):
            rc, out = sb.run(STUB_HELD=who)
            c = sb.calls_text()
            t(f"sandbox/REFUSED_while_the_{who}_chain_holds_its_lock", rc == 1 and verdict_line(out).startswith("ZZP6CPU_REFUSED ") and who in out and "build_metric_cache" not in c
              and "diag_p6" not in c and "pool" not in c)
        rc, out = sb.run(STUB_GPU_PIDS="4242")
        t("sandbox/REFUSED_when_a_process_is_on_the_GPU", rc == 1 and "GPU" in verdict_line(out) and "build_metric_cache" not in sb.calls_text())
        rc, out = sb.run(P6CPU_NVSMI="/nonexistent/nvidia-smi")
        t("sandbox/REFUSED_when_nvidia_smi_cannot_be_asked", rc == 1 and verdict_line(out).startswith("ZZP6CPU_REFUSED "))
        rc, out = sb.run(STUB_CG_BUSY="1")
        t("sandbox/REFUSED_when_the_cgroup_is_already_burning_cores", rc == 1 and "quiet" in verdict_line(out) and "build_metric_cache" not in sb.calls_text())
        (sb.pr / "scratch").mkdir(parents=True, exist_ok=True)
        (sb.pr / "scratch" / "keep.txt").write_text("x")
        (sb.data / ".p6cpu_dd").write_text("x")
        rc, out = sb.run(STUB_HELD="probe")
        t("sandbox/REFUSED_when_another_probe_runs", rc == 1 and "another p6_cpu_probe.sh" in out)
        t("sandbox/the_refused_second_probe_deleted_nothing_of_the_running_one", (sb.pr / "scratch" / "keep.txt").exists() and (sb.data / ".p6cpu_dd").exists())
        (sb.data / ".p6cpu_dd").unlink()
        shutil.rmtree(sb.pr / "scratch")
        rc, out = sb.run(STUB_CG_BUSY="1", P6CPU_MAX_BG="50")
        t("sandbox/the_quiet_bar_is_a_knob_not_a_constant", rc == 0)
        # ---- a spent budget skips every stage, loudly, and cannot be FIT
        rc, out = sb.run(P6CPU_BUDGET_MIN="0")
        lg = sb.log()
        t("sandbox/a_spent_budget_skips_stages_and_says_so", "ZZP6CPU_BUDGET skip=bank" in lg and "ZZP6CPU_BUDGET skip=dd" in lg)
        t("sandbox/a_spent_budget_is_VOID_and_stale_files_of_the_previous_run_do_not_leak", verdict_line(out).startswith("ZZP6CPU_VOID ") and not (sb.pr / "lab_w7.json").exists())
        # ---- dd bounded by its time cap, stops cleanly
        rc, out = sb.run(P6CPU_DD_MAX_S="0")
        dd = json.loads((sb.pr / "dd.json").read_text())
        t("sandbox/dd_stops_at_its_time_cap_and_reports_the_GB_proven", dd["stopped_by"] == "time_cap" and dd["mb_ok"] == 0 and not (sb.data / ".p6cpu_dd").exists())
        # ---- P6CPU_KEEP keeps the scratch; the default removes it
        rc, out = sb.run(P6CPU_KEEP="1")
        t("sandbox/P6CPU_KEEP_keeps_the_scratch", (sb.pr / "scratch" / "metric_cache").is_dir())
        rc, out = sb.run()
        t("sandbox/the_next_default_run_clears_it", not (sb.pr / "scratch").exists())
        # ---- P6CPU_SIM_HZ is passed to the D9 build and the pool when (and only when) asked
        rc, out = sb.run(P6CPU_SIM_HZ="10")
        t("sandbox/P6CPU_SIM_HZ_reaches_the_build_and_the_pool", all("SIMHZ=10" in ln for ln in sb.calls_text().splitlines() if ln.startswith(("build_metric_cache.py", "diag_p6"))))
        # ---- stages the pod could not run stay visible
        rc, out = sb.run(P6CPU_LWS="7 8 20")
        t("sandbox/a_worker_count_above_cores_plus_1_is_skipped_and_logged", "ZZP6CPU_SKIP lw=20" in sb.log())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    layer1()
    layer2()
    layer3()
    print("ZZINFO what only the pod can prove: the real NAVSIM labeller's cost on its vCPUs, its cgroup counters (cpu.stat, memory.stat), NFSv4 flock semantics, the real dd, "
          "and the real bank's row layout (this fixture puts the key fields at the tail, as the dev-box bank does)")
    failed = [n for n, ok in checks if not ok]
    if failed:
        print("ZZTEST_FAIL", len(failed), failed[:6])
        return 1
    print(f"ZZTEST_PASS {len(checks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
