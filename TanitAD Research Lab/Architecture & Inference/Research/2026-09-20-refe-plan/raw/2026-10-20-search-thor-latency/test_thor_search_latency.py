"""CPU tests of thor_search_latency.py on tiny SYNTHETIC inputs (a deterministic stand-in scorer, a fake clock).  No checkpoint, no ctx, no GPU, no Thor, no label.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest test_thor_search_latency.py -q -p no:cacheprovider
`mutation_check.py` deletes each `@GUARD` line in turn in a scratch copy; this suite must then go RED."""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import thor_search_latency as T  # noqa: E402

SB = T.stepb()
A = SB.A
N_FAN = 8


# ------------------------------------------------------------------------------------------------------------------------------ synthetic world
class FakeClock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t

    def advance(self, ns):
        self.t += int(ns)


class TickClock(FakeClock):
    """a clock that moves 1 ms on EVERY read: a timer with a huge overhead"""

    def __call__(self):
        self.t += 1_000_000
        return self.t


class SynthScorer:
    """deterministic stand-in for the real scorer: logits = f(set, bound token).  Advances the fake clock by base_ns + row_ns * rows per call."""

    def __init__(self, clock=None, base_ns=1_000_000, row_ns=1_000, nondet=False, seed=7):
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((60, 6)) * 0.05).astype(np.float32)
        self.off = rng.standard_normal((16, 6)).astype(np.float32)
        self.clock, self.base_ns, self.row_ns, self.nondet = clock, base_ns, row_ns, nondet
        self.n, self.calls, self.info = 0, 0, {}

    def bind(self, n):
        self.n = int(n)

    def __call__(self, S):
        S = np.asarray(S, np.float32)
        K = S.shape[0]
        self.calls += 1
        if self.clock is not None:
            self.clock.advance(self.base_ns + self.row_ns * K)
        z = np.tanh(S.reshape(K, -1) @ self.W) * 1.5 + self.off[self.n % 16]
        if self.nondet:
            z = z + 1e-6 * (self.calls % 3)
        return z.astype(np.float32)


def make_fans(n_tokens: int, n_fan: int = N_FAN, seed: int = 3):
    rng = np.random.default_rng(seed)
    fans = []
    for _ in range(n_tokens):
        f = np.zeros((n_fan, 20, 3), np.float32)
        for k in range(n_fan):
            v = 6.0 + 1.5 * k + rng.uniform(-0.5, 0.5)
            x = np.cumsum(np.full(20, v * 0.2))
            y = 0.02 * x * rng.uniform(-1, 1) + 0.1 * np.sin(np.arange(20) / 5.0)
            h = np.arctan2(np.gradient(y), np.gradient(x))
            f[k] = np.stack([x, y, h], 1)
        fans.append(f)
    return fans


CFG_B = SB.c8_from_grid(m_pop=16, n_elite=2)                       # tiny stand-ins for C8 / O4: same code path, cheap
CFG_C = SB.c8_from_grid(m_pop=48, n_elite=6, widen=2.0)


def run_measure(*, n=3, warmup=2, repeats=2, clock=None, scorer=None, wrap=T.CountingScorer, cfg_b=CFG_B, cfg_c=CFG_C, SB_=None, **kw):
    clock = clock or FakeClock()
    sc = scorer or SynthScorer(clock=clock)
    fans = make_fans(n)
    picks = [i % N_FAN for i in range(n)]
    m = T.measure(sc, sc.bind, fans, picks, cfg_b, cfg_c, seed=20260927, warmup=warmup, repeats=repeats, clock_ns=clock, sync=lambda: None, wrap=wrap, SB=(SB_ or SB), **kw)
    return m, sc


@pytest.fixture(scope="module")
def honest():
    m, sc = run_measure(n=3, warmup=2, repeats=2)
    return m, sc, T.report(m, CFG_B, CFG_C)


class Perturbing(T.CountingScorer):
    """a timed path that CHANGES the answer: a row-dependent offset on the logits"""

    def __call__(self, S):
        out = super().__call__(S)
        return np.asarray(out, np.float32) + np.float32(1e-3) * np.arange(out.shape[0], dtype=np.float32)[:, None]


class Undercount(T.CountingScorer):
    def __call__(self, S):
        out = super().__call__(S)
        self.rows -= 1
        return out


# ------------------------------------------------------------------------------------------------------------------------------ percentile maths
def test_percentile_known_values():
    x = list(range(1, 101))
    assert T.percentile(x, 50) == pytest.approx(50.5)
    assert T.percentile(x, 95) == pytest.approx(95.05)
    assert T.percentile(x, 99) == pytest.approx(99.01)
    assert T.percentile(x, 0) == 1 and T.percentile(x, 100) == 100


def test_percentile_edges_and_order_and_numpy_crosscheck():
    assert T.percentile([5.0], 99) == 5.0
    assert T.percentile([2.0, 4.0], 50) == 3.0
    assert T.percentile([9, 1, 5, 3, 7], 50) == 5.0                                      # unsorted input
    rng = np.random.default_rng(0)
    v = rng.exponential(2.0, 257)
    for q in (1, 25, 50, 90, 95, 99, 99.9):
        assert T.percentile(v, q) == pytest.approx(float(np.percentile(v, q)), rel=1e-12)


def test_percentile_refuses_bad_input():
    with pytest.raises(ValueError):
        T.percentile([], 50)
    with pytest.raises(ValueError):
        T.percentile([1.0, float("nan")], 50)                                            # @GUARD:FINITE
    with pytest.raises(ValueError):
        T.percentile([1.0, float("inf")], 50)
    with pytest.raises(ValueError):
        T.percentile([1.0], 101)


def test_summarize_and_ratios():
    s = T.summarize([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["mean"] == 2.5 and s["min"] == 1.0 and s["max"] == 4.0 and s["p50"] == 2.5
    r = T.ratios({"p50": 10.0, "p95": 30.0, "p99": 90.0}, {"p50": 2.0, "p95": 3.0, "p99": 9.0})
    assert r == {"p50": 5.0, "p95": 10.0, "p99": 10.0}
    with pytest.raises(ValueError):
        T.ratios({"p50": 1.0, "p95": 1.0, "p99": 1.0}, {"p50": 0.0, "p95": 1.0, "p99": 1.0})   # a zero baseline is refused, not divided


# ------------------------------------------------------------------------------------------------------------------------------ call counts: C8 vs O4 match the config
def test_expected_counts_formula_for_the_real_configs():
    c8, o4 = SB.c8_from_grid(), SB.c8_from_grid(m_pop=256, n_elite=32, widen=2.0)
    assert T.expected_counts(c8, 64) == {"calls": 14, "rows": 896}
    assert T.expected_counts(o4, 64) == {"calls": 38, "rows": 3392}
    assert T.expected_counts(dataclasses.replace(c8, k_iters=0), 64) == {"calls": 1, "rows": 64}


def test_real_config_files_are_the_pinned_ones(tmp_path):
    """the C8 rule reproduces the pinned config bytes: the pin is the rule's output, not a hand-edited file"""
    for name, cfg, key in (("c8.json", SB.c8_from_grid(), "c8_config_sha256"), ("o4.json", SB.c8_from_grid(m_pop=256, n_elite=32, widen=2.0), "o4_config_sha256")):
        p = tmp_path / name
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg.as_json(), f, indent=1)
            f.write("\n")
        assert T.sha256_lf(str(p)) == T.PINS[key]
        SB.validate_config(SB.StepBConfig.from_json(str(p)))


def test_observed_counts_match_the_config_for_c8_and_o4_on_the_real_search():
    """one real search at the real shapes: the counter reads exactly 14 calls / 896 rows (C8) and 38 / 3392 (O4)"""
    clock = FakeClock()
    sc = SynthScorer(clock=clock)
    fan = make_fans(1, n_fan=64)[0]
    for cfg, want in ((SB.c8_from_grid(), (14, 896)), (SB.c8_from_grid(m_pop=256, n_elite=32, widen=2.0), (38, 3392))):
        cnt = T.CountingScorer(sc, clock, lambda: None)
        _, _, _, calls, rows = T.time_one(lambda sf: SB.search_token(sf, fan, 5, 20260927, cfg), cnt, clock, lambda: None)
        assert (calls, rows) == want
        assert T.expected_counts(cfg, 64) == {"calls": want[0], "rows": want[1]}


def test_call_count_control_is_green_when_honest_and_red_when_a_count_is_off(honest):
    m, sc, rep = honest
    for a in T.ARMS:
        assert rep["controls"][f"call_counts_{a}"]["ok"], rep["controls"][f"call_counts_{a}"]
    assert rep["arms"]["b"]["scorer_calls_per_token"] == [8] and rep["arms"]["b"]["scorer_rows_per_token"] == [104]
    assert rep["arms"]["c"]["scorer_calls_per_token"] == [12] and rep["arms"]["c"]["scorer_rows_per_token"] == [296]
    m2, _ = run_measure(wrap=Undercount)
    rep2 = T.report(m2, CFG_B, CFG_C)
    assert not rep2["controls"]["call_counts_b"]["ok"] and not rep2["controls"]["call_counts_c"]["ok"] and not rep2["controls_ok"]    # @GUARD:COUNTS


# ------------------------------------------------------------------------------------------------------------------------------ the identity control
def test_identity_control_green_on_an_honest_timed_path(honest):
    m, sc, rep = honest
    for a in T.ARMS:
        assert rep["controls"][f"identity_{a}"]["ok"], rep["controls"][f"identity_{a}"]
        assert rep["controls"][f"identity_{a}"]["of"] == 3 * 2
        assert rep["controls"][f"determinism_{a}"]["ok"]
    assert rep["controls_ok"], {k: v for k, v in rep["controls"].items() if not v["ok"]}


def test_identity_control_goes_red_when_the_timed_path_is_perturbed():
    m, sc = run_measure(wrap=Perturbing)
    rep = T.report(m, CFG_B, CFG_C)
    for a in T.ARMS:
        c = rep["controls"][f"identity_{a}"]
        assert not c["ok"] and c["mismatched_samples"] == 3 * 2, (a, c)                  # every sample of every arm differs
    assert rep["controls_ok"] is False


def test_determinism_floor_goes_red_on_a_nondeterministic_device():
    clock = FakeClock()
    m, sc = run_measure(clock=clock, scorer=SynthScorer(clock=clock, nondet=True))
    rep = T.report(m, CFG_B, CFG_C)
    assert not rep["controls"]["determinism_a"]["ok"]                                    # @GUARD:DETERMINISM
    assert not rep["controls"]["identity_a"]["ok"]                                       # the identity control is then unattributable, and says so by being red too
    assert rep["controls_ok"] is False


def test_outputs_identical_is_bitwise():
    x = {"raw": np.array([0.0], np.float32), "theta": (0.0, 1.0), "moved": False, "h": [{"v": 1.0}]}
    assert T.outputs_identical(x, {"raw": np.array([0.0], np.float32), "theta": (0.0, 1.0), "moved": False, "h": [{"v": 1.0}]})
    assert not T.outputs_identical(x, {**x, "raw": np.array([-0.0], np.float32)})        # -0.0 is not 0.0 bit for bit
    assert not T.outputs_identical(x, {**x, "moved": True})
    assert not T.outputs_identical(x, {**x, "h": [{"v": 1.0 + 1e-16 * 3}]})
    assert not T.outputs_identical(x, {**x, "raw": np.array([0.0], np.float64)})


def test_n0_control_returns_the_pick_with_one_call(honest):
    m, sc, rep = honest
    assert rep["controls"]["n0_pick"]["ok"] and rep["controls"]["n0_pick"]["tokens"] == 3


class Overcount(T.CountingScorer):
    def __call__(self, S):
        out = super().__call__(S)
        self.calls += 1
        return out


class _SBMovesAtN0:
    """a stand-in for Step B whose k_iters=0 search wrongly MOVES the pick"""
    A = SB.A

    @staticmethod
    def search_token(sf, fan, pick, seed, cfg):
        r = SB.search_token(sf, fan, pick, seed, cfg)
        if cfg.k_iters == 0:
            r = dict(r, moved=True)
        return r


class _SBShiftsAtN0(_SBMovesAtN0):
    @staticmethod
    def search_token(sf, fan, pick, seed, cfg):
        r = SB.search_token(sf, fan, pick, seed, cfg)
        if cfg.k_iters == 0:
            r = dict(r, raw=np.asarray(r["raw"]) + np.float32(1e-6))
        return r


def test_n0_control_goes_red_when_the_call_count_or_the_pick_is_off():
    m, _ = run_measure(wrap=Overcount)
    assert not T.report(m, CFG_B, CFG_C)["controls"]["n0_pick"]["ok"]                    # @GUARD:N0 (one extra call)
    for fake in (_SBMovesAtN0, _SBShiftsAtN0):
        m, _ = run_measure(SB_=fake)
        assert not T.report(m, CFG_B, CFG_C)["controls"]["n0_pick"]["ok"], fake           # moved flag / raw bytes differ from the pick


# ------------------------------------------------------------------------------------------------------------------------------ warm-up, the floor, the timer sees the work
def test_warmup_is_run_but_excluded_from_every_statistic():
    n, w, r = 4, 3, 2
    clock = FakeClock()
    sc = SynthScorer(clock=clock)
    m, _ = run_measure(n=n, warmup=w, repeats=r, clock=clock, scorer=sc)
    per_token = 1 + 8 + 12                                                                # calls of (a) + (b) + (c) per token
    noop_calls = 0                                                                        # the stub scorer is not the synthetic one
    n0_calls = min(5, n) * 1
    scaling_calls = 2 * 15
    assert sc.calls == 2 * n * per_token + w * per_token + r * n * per_token + n0_calls + scaling_calls + noop_calls    # @GUARD:WARMUP
    for a in T.ARMS:
        assert len(m["lat_ms"][a]) == r and all(len(rep) == n for rep in m["lat_ms"][a])      # n x repeats timed samples, no warm-up sample
        assert len(m["calls"][a]) == n * r
    assert m["warmup_tokens"] == w


def test_warmup_larger_than_the_token_list_is_capped_not_an_error():
    m, _ = run_measure(n=2, warmup=30, repeats=1)
    assert m["warmup_tokens"] == 2


def test_noop_timer_floor_reads_zero_on_a_quiet_clock_and_goes_red_on_a_heavy_one(honest):
    m, sc, rep = honest
    assert m["noop_floor"]["ns"]["p99"] == 0 and rep["controls"]["timer_floor"]["ok"]
    clock = TickClock()
    sc2 = SynthScorer(clock=clock, base_ns=0, row_ns=0)                                   # the model adds ~nothing; every clock read costs 1 ms
    m2, _ = run_measure(clock=clock, scorer=sc2)
    assert m2["noop_floor"]["ns"]["p99"] == 1_000_000
    rep2 = T.report(m2, CFG_B, CFG_C)
    assert not rep2["controls"]["timer_floor"]["ok"]                                      # @GUARD:FLOOR


def test_timer_sees_the_work_a_lt_b_lt_c_and_goes_red_if_not(honest):
    m, sc, rep = honest
    p = rep["controls"]["sees_work"]["p50_a_b_c_ms"]
    assert rep["controls"]["sees_work"]["ok"] and p[0] < p[1] < p[2]
    m2 = json.loads(json.dumps(m))
    m2["lat_ms"]["c"] = m2["lat_ms"]["a"]                                                 # a search as cheap as the pick: the timer did not see the work
    rep2 = T.report(m2, CFG_B, CFG_C)
    assert not rep2["controls"]["sees_work"]["ok"] and rep2["controls_ok"] is False       # @GUARD:SEESWORK


def test_controls_ok_is_the_conjunction(honest):
    m, sc, rep = honest
    m2 = json.loads(json.dumps(m))
    m2["mismatches"]["b"] = [{"repeat": 0, "token_index": 0}]
    rep2 = T.report(m2, CFG_B, CFG_C)
    assert rep2["controls"]["identity_b"]["ok"] is False and rep2["controls_ok"] is False   # @GUARD:CONJ


# ------------------------------------------------------------------------------------------------------------------------------ numbers: exact on the fake clock
def test_report_numbers_are_exact_on_the_fake_clock(honest):
    m, sc, rep = honest
    a_ms, b_ms, c_ms = 1.008, 8.104, 12.296                                               # base 1 ms + 1 us per row: (1 call, 8 rows), (8, 104), (12, 296)
    arms = rep["arms"]
    for arm, want in (("a", a_ms), ("b", b_ms), ("c", c_ms)):
        for q in ("p50", "p95", "p99", "mean"):
            assert arms[arm]["ms"][q] == pytest.approx(want, abs=1e-9)
        assert arms[arm]["host_rest_ms"]["p50"] == pytest.approx(0.0, abs=1e-9)            # the fake clock only moves inside the scorer
    assert arms["b"]["ratio_to_a"]["p50"] == pytest.approx(b_ms / a_ms)
    assert arms["c"]["ratio_to_a"]["p99"] == pytest.approx(c_ms / a_ms)
    assert arms["c"]["ratio_to_b"]["p95"] == pytest.approx(c_ms / b_ms)
    assert arms["b"]["added_ms_vs_a"]["p50"] == pytest.approx(b_ms - a_ms)


def test_pooled_percentiles_match_numpy_over_the_raw_samples():
    clock = FakeClock()
    rng = np.random.default_rng(5)

    class Jitter(SynthScorer):
        def __call__(self, S):
            self.base_ns = int(rng.integers(100_000, 5_000_000))
            return super().__call__(S)

    j = Jitter(clock=clock)
    m, _ = run_measure(n=4, repeats=3, clock=clock, scorer=j)
    rep = T.report(m, CFG_B, CFG_C)
    flat = np.array([v for r in m["lat_ms"]["b"] for v in r])
    for q, k in ((50, "p50"), (95, "p95"), (99, "p99")):
        assert rep["arms"]["b"]["ms"][k] == pytest.approx(float(np.percentile(flat, q)), rel=1e-12)
    assert len(rep["arms"]["b"]["per_repeat_ms"]) == 3


def test_record_is_json_serialisable_and_has_no_pass_fail(honest):
    m, sc, rep = honest
    env = {"target": "devbox-cpu", "cpu_name": "x"}
    rec = T.build_record(target="devbox-cpu", env=env, pins_seen={}, inputs={}, cfgs={}, rep=rep, args_echo={}, preflight=None, preflight_end=None, thermal_end=None)
    json.dumps(rec)
    assert rec["verdict"] is None and rec["no_pass_fail"] is True
    assert "DQ4" in rec["statements"][0] and "NO claim of deployability" in rec["statements"][0]
    assert any("Warm-up" in s for s in rec["statements"])
    assert rec["is_thor_number"] is False
    assert any("devbox-cpu" in ln for ln in T.summary_lines(rec))


def test_is_thor_number_needs_thor_green_controls_and_a_passed_preflight(honest):
    m, sc, rep = honest
    kw = dict(env={"target": "thor"}, pins_seen={}, inputs={}, cfgs={}, args_echo={}, preflight_end=None, thermal_end=None)
    assert T.build_record(target="thor", rep=rep, preflight={"ok": True}, **kw)["is_thor_number"] is True
    assert T.build_record(target="thor", rep=rep, preflight={"ok": False}, **kw)["is_thor_number"] is False
    assert T.build_record(target="thor", rep=rep, preflight=None, **kw)["is_thor_number"] is False
    red = {**rep, "controls_ok": False}
    r = T.build_record(target="thor", rep=red, preflight={"ok": True}, **kw)
    assert r["is_thor_number"] is False and "not quotable" in r["evidence_class"]          # @GUARD:THORNUMBER


def test_informative_blocks_present(honest):
    m, sc, rep = honest
    inf = rep["informative"]
    assert inf["noop_scorer_search_ms"]["b"]["calls_per_token"] == [8] and inf["noop_scorer_search_ms"]["c"]["calls_per_token"] == [12]
    assert set(inf["per_call_ms_by_population"]) == {"64", "256"}
    assert inf["row_scaling_visible_256_vs_64"] == pytest.approx((1_000_000 + 1000 * 256) / (1_000_000 + 1000 * 64), rel=1e-9)


# ------------------------------------------------------------------------------------------------------------------------------ device fields: the refusal
THOR_ENV = {"hostname": "thor6", "platform": "Linux-aarch64", "machine": "aarch64", "python": "3.12.1", "numpy": "1.26", "torch": "2.9", "torch_threads": 4, "torch_cuda": "13.0",
            "cuda_device_name": "NVIDIA Thor", "cuda_capability": [11, 0], "nvpmodel_q": "NV Power Mode: 120W\n1", "jetson_clocks_show": "GPU MinFreq=1 MaxFreq=2 CurrentFreq=2",
            "devfreq": {"gpu-gpc-0": {"cur": "1386000000"}}, "thermal": {"tj-thermal": "58500"}}
CPU_ENV = {"hostname": "box", "platform": "Windows", "machine": "AMD64", "python": "3.13", "numpy": "2.0", "torch": "2.11", "torch_threads": 4, "cpu_name": "Intel"}


def test_complete_device_fields_pass():
    T.check_device_fields(THOR_ENV, "thor")
    T.check_device_fields(CPU_ENV, "devbox-cpu")


@pytest.mark.parametrize("field", T.required_fields("thor"))
@pytest.mark.parametrize("blank", [None, "", {}, []])
def test_every_missing_thor_field_is_refused(field, blank):
    env = dict(THOR_ENV)
    env[field] = blank
    with pytest.raises(T.DeviceFieldsMissing, match=field):
        T.check_device_fields(env, "thor")


def test_absent_key_is_refused_and_cpu_target_has_its_own_list():
    env = {k: v for k, v in THOR_ENV.items() if k != "nvpmodel_q"}
    with pytest.raises(T.DeviceFieldsMissing, match="nvpmodel_q"):
        T.check_device_fields(env, "thor")
    env = {k: v for k, v in CPU_ENV.items() if k != "cpu_name"}
    with pytest.raises(T.DeviceFieldsMissing, match="cpu_name"):
        T.check_device_fields(env, "devbox-cpu")
    assert "nvpmodel_q" in T.required_fields("thor") and "jetson_clocks_show" in T.required_fields("thor") and "nvpmodel_q" not in T.required_fields("devbox-cpu")


class FakeTorch:
    __version__ = "2.9.0"

    class version:
        cuda = "13.0"

    class cuda:
        @staticmethod
        def is_available():
            return False

    @staticmethod
    def get_num_threads():
        return 4


def test_device_state_reads_live_then_falls_back_to_fresh_files_and_refuses_stale_ones(tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "nvpmodel_q.txt").write_text("NV Power Mode: 120W\n1\n")
    (env_dir / "jetson_clocks_show.txt").write_text("GPU MinFreq=306 MaxFreq=1386 CurrentFreq=1386\n")
    dev = tmp_path / "devfreq" / "gpu"
    dev.mkdir(parents=True)
    for fn, v in (("cur_freq", "1386000000"), ("min_freq", "1"), ("max_freq", "1386000000"), ("governor", "nvhost_podgov")):
        (dev / fn).write_text(v)
    th = tmp_path / "thermal" / "thermal_zone0"
    th.mkdir(parents=True)
    (th / "type").write_text("tj-thermal")
    (th / "temp").write_text("58500")
    sysfs = {"devfreq": {"root": str(tmp_path / "devfreq")}, "thermal": {"root": str(tmp_path / "thermal")}}

    def dead(argv):                                                                       # nvpmodel / jetson_clocks need root here: the live read fails
        assert tuple(argv) in T.READONLY_ALLOW
        return (1, "") if argv[0] in ("nvpmodel", "jetson_clocks") else (0, "58, 33.5 W, 0 %\n")

    import time as _t
    env = T.device_state("thor", FakeTorch, str(env_dir), _t.time(), runner=dead, sysfs=sysfs)
    assert env["nvpmodel_q_source"].startswith("file:") and env["jetson_clocks_show_source"].startswith("file:")
    assert env["jetson_clocks_parsed"]["gpu_at_max"] is True
    assert env["devfreq"]["gpu"]["cur"] == "1386000000" and env["thermal"] == {"tj-thermal": "58500"}
    old = _t.time() - 7200
    os.utime(env_dir / "nvpmodel_q.txt", (old, old))
    env2 = T.device_state("thor", FakeTorch, str(env_dir), _t.time(), runner=dead, sysfs=sysfs)
    assert env2["nvpmodel_q"] is None                                                     # a 2-hour-old capture is not a status
    with pytest.raises(T.DeviceFieldsMissing, match="nvpmodel_q"):
        T.check_device_fields({**THOR_ENV, **{k: env2[k] for k in ("nvpmodel_q",)}}, "thor")

    def live(argv):
        assert tuple(argv) in T.READONLY_ALLOW
        return 0, {"nvpmodel": "NV Power Mode: MAXN\n0", "jetson_clocks": "GPU MaxFreq=5 CurrentFreq=4"}.get(argv[0], "x")

    env3 = T.device_state("thor", FakeTorch, None, _t.time(), runner=live, sysfs=sysfs)
    assert env3["nvpmodel_q_source"] == "live" and env3["jetson_clocks_parsed"]["gpu_at_max"] is False


# ------------------------------------------------------------------------------------------------------------------------------ read-only commands only
def test_readonly_allowlist_refuses_everything_else(monkeypatch):
    seen = []
    monkeypatch.setattr(T.subprocess, "run", lambda argv, **k: seen.append(tuple(argv)) or type("P", (), {"returncode": 0, "stdout": "ok"})())
    for bad in (("jetson_clocks",), ("jetson_clocks", "--store"), ("nvpmodel", "-m", "0"), ("free", "-m"), ("tegrastats",), ("nvidia-smi", "--query-gpu=memory.used", "--format=csv"),
                ("sudo", "nvpmodel", "-q"), ("rm", "-rf", "/")):
        with pytest.raises(T.ForbiddenCommand):                                           # @GUARD:READONLY
            T.run_readonly(bad)
    assert seen == []                                                                     # nothing forbidden ever reached subprocess
    for good in T.READONLY_ALLOW:
        assert T.run_readonly(good) == (0, "ok")
    assert seen == list(T.READONLY_ALLOW)
    assert ("jetson_clocks", "--show") in T.READONLY_ALLOW and ("jetson_clocks",) not in T.READONLY_ALLOW
    assert not any("memory" in x or "tegrastats" in x or x == "free" for argv in T.READONLY_ALLOW for x in argv)    # the lying Thor memory probes are not on the list


# ------------------------------------------------------------------------------------------------------------------------------ the Thor-busy preflight
PS_IDLE = """\
    1     0 root      99000 /sbin/init
  500     1 root      98000 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers
  800     1 root      97000 /usr/sbin/sshd -D
 2000   800 nvidia      500 sshd: nvidia@pts/0
 2100  2000 nvidia      499 -bash
 2150  2100 nvidia        9 python3 launcher.py
 2200  2150 nvidia        8 python3 /home/nvidia/thorlat/code/thor_search_latency.py preflight --target thor
"""
TRAINER = " 3000  2100 nvidia      600 /home/nvidia/venvs/tanitad-train/bin/python scripts/train_v6_staged.py --stage 3 --steps 30000\n"
ARM_EVAL = " 3001  2100 nvidia      600 /home/nvidia/venvs/tanitad-train/bin/python taniteval/tools/refav1_arm.py --ckpt x\n"
SUPERVISOR = " 3002     1 nvidia    90000 /bin/bash /home/nvidia/ops/supervise_run.sh refcv8-r101-s0\n"
ROOT_TRAINER = " 3003     1 root      600 python scripts/train_x.py\n"


def runner_for(ps=PS_IDLE, utils=("0", "0", "0"), apps="", ps_rc=0, smi_rc=0, apps_rc=0):
    cyc = itertools.cycle(utils)
    log = []

    def runner(argv):
        t = tuple(argv)
        assert t in T.READONLY_ALLOW, t
        log.append(t)
        if t[0] == "ps":
            return ps_rc, ps
        if t[1].startswith("--query-gpu=utilization.gpu"):
            return smi_rc, str(next(cyc)) + "\n"
        if t[1].startswith("--query-compute-apps"):
            return apps_rc, apps
        raise AssertionError(t)

    runner.log = log
    return runner


def pf(runner=None, **kw):
    args = dict(runner=runner or runner_for(), sleep=lambda s: None, self_pid=2200, me="nvidia", gpu_lock="/nonexistent/thor_gpu.lock", expect_hostname="thor6", hostname="thor6", samples=3)
    args.update(kw)
    return T.preflight_thor(**args)


def test_preflight_passes_on_an_idle_box_and_excludes_self_and_ancestors():
    r = pf()
    assert r["ok"], r["reasons"]
    assert r["evidence"]["busy_processes"] == [] and r["evidence"]["gpu_util_pct_samples"] == [0.0, 0.0, 0.0]
    assert "NOT USED" in r["evidence"]["memory_probes"]


@pytest.mark.parametrize("extra,why", [(TRAINER, "busy process"), (ARM_EVAL, "busy process"), (SUPERVISOR, "busy process"), (ROOT_TRAINER, "busy process")])
def test_preflight_refuses_a_running_trainer_eval_or_supervisor(extra, why):
    r = pf(runner_for(ps=PS_IDLE + extra))
    assert not r["ok"] and any(why in x for x in r["reasons"])                            # @GUARD:BUSY


def test_preflight_ignores_a_root_system_python_daemon_but_not_a_trainer_of_any_user():
    assert pf()["ok"]                                                                     # pid 500 is root's networkd-dispatcher (python3): not this user's job
    assert not pf(runner_for(ps=PS_IDLE + ROOT_TRAINER))["ok"]


def test_preflight_refuses_a_busy_gpu():
    r = pf(runner_for(utils=("0", "97", "0")))
    assert not r["ok"] and any("utilisation" in x for x in r["reasons"])                  # @GUARD:UTIL
    assert pf(runner_for(utils=("4.9", "4.9", "4.9")))["ok"]                              # below 5 % is idle
    assert not pf(runner_for(utils=("5", "5", "5")))["ok"]


@pytest.mark.parametrize("bad", ["[N/A]", "", "N/A"])
def test_preflight_fails_closed_when_gpu_utilisation_is_unreadable(bad):
    r = pf(runner_for(utils=(bad,)))
    assert not r["ok"] and any("unreadable" in x for x in r["reasons"])
    r = pf(runner_for(smi_rc=127))
    assert not r["ok"]                                                                    # nvidia-smi missing is not 'idle'


def test_preflight_fails_closed_when_the_process_table_is_unreadable():
    assert not pf(runner_for(ps_rc=1, ps=""))["ok"]
    assert not pf(runner_for(ps=""))["ok"]                                                # @GUARD:PSREAD


def test_preflight_refuses_listed_compute_apps():
    r = pf(runner_for(apps="2999, python\n"))
    assert not r["ok"] and any("compute app" in x for x in r["reasons"])                  # @GUARD:APPS
    assert pf(runner_for(apps_rc=127))["ok"]                                              # unavailable is recorded, not a pass or a block (the util + ps + lock reads decide)


def test_preflight_binds_the_run_to_a_named_box():
    assert not pf(hostname="thor7")["ok"]                                                 # @GUARD:HOST
    r = pf(expect_hostname=None)
    assert not r["ok"] and any("no --expect-hostname" in x for x in r["reasons"])         # @GUARD:NOHOST
    r = pf(hostname="thor7")
    assert any("thor7" in x and "thor6" in x for x in r["reasons"])


class FakeFlock:
    LOCK_EX, LOCK_NB, LOCK_UN = 2, 4, 8

    def __init__(self, held):
        self.held, self.ops = held, []

    def flock(self, fd, op):
        self.ops.append(op)
        if self.held and op & self.LOCK_NB:
            raise BlockingIOError()


def test_preflight_refuses_a_held_gpu_lock_and_holds_a_free_one(tmp_path):
    lock = tmp_path / "thor_gpu.lock"
    lock.write_text("")
    r = pf(gpu_lock=str(lock), flock_mod=FakeFlock(held=True))
    assert not r["ok"] and any("HELD" in x for x in r["reasons"])                         # @GUARD:LOCK
    ff = FakeFlock(held=False)
    r = pf(gpu_lock=str(lock), flock_mod=ff, hold_lock=True)
    assert r["ok"] and r["_lock"]["held_by_us"] is True and ff.ops == [6]
    os.close(r["_lock"]["_fd"])
    ff2 = FakeFlock(held=False)
    r = pf(gpu_lock=str(lock), flock_mod=ff2, hold_lock=False)
    assert r["ok"] and r["_lock"]["held_by_us"] is False and ff2.ops == [6, 8]            # a preflight-only run releases it
    r = pf(gpu_lock=str(tmp_path / "absent.lock"))
    assert r["ok"] and r["evidence"]["gpu_lock"]["exists"] is False                       # absence is recorded, never read as 'idle'


def test_preflight_only_ever_issues_allowlisted_read_only_commands():
    rn = runner_for()
    pf(rn)
    assert all(t in T.READONLY_ALLOW for t in rn.log) and rn.log                          # the runner asserts membership; no memory probe is among them


# ------------------------------------------------------------------------------------------------------------------------------ main(): refusals end to end (no checkpoint, no GPU)
def _argv(cmd, target, tmp, extra=()):
    base = [cmd, "--target", target, "--expect-hostname", "thor6", "--gpu-lock", str(tmp / "nolock")]
    if cmd == "run":
        base += ["--bundle", str(tmp / "nobundle.npz"), "--ctx", str(tmp / "noctx.npy"), "--ckpt", str(tmp / "nockpt.pt"), "--refe-dir", str(tmp), "--c8", str(tmp / "c8.json"), "--o4", str(tmp / "o4.json"),
                 "--out-dir", str(tmp / "out"), "--autocast", "bf16"]
    return base + list(extra)


def test_main_preflight_returns_3_when_thor_is_busy_and_0_when_idle(tmp_path, capsys):
    kw = dict(sleep=lambda s: None, self_pid=2200, me="nvidia", hostname="thor6", samples=2)
    assert T.main(_argv("preflight", "thor", tmp_path), runner=runner_for(ps=PS_IDLE + TRAINER), **kw) == T.EXIT["thor_busy"]
    assert "ZZTHORLAT_STOP THOR_BUSY" in capsys.readouterr().out
    assert T.main(_argv("preflight", "thor", tmp_path), runner=runner_for(), **kw) == 0
    assert "ZZTHORLAT_PREFLIGHT_OK" in capsys.readouterr().out


def test_main_run_refuses_a_busy_thor_before_touching_any_input(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    kw = dict(sleep=lambda s: None, self_pid=2200, me="nvidia", hostname="thor6", samples=2)
    rc = T.main(_argv("run", "thor", tmp_path), runner=runner_for(utils=("93",)), **kw)
    assert rc == T.EXIT["thor_busy"] and not (tmp_path / "out").exists()                  # the bundle / ckpt paths do not exist: reaching them would have crashed


def test_main_run_refuses_when_device_fields_are_missing_and_writes_the_refusal(tmp_path, monkeypatch, capsys):
    import torch
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    incomplete = {k: v for k, v in THOR_ENV.items() if k not in ("nvpmodel_q", "jetson_clocks_show")}
    monkeypatch.setattr(T, "device_state", lambda *a, **k: dict(incomplete))
    kw = dict(sleep=lambda s: None, self_pid=2200, me="nvidia", hostname="thor6", samples=2)
    rc = T.main(_argv("run", "thor", tmp_path), runner=runner_for(), **kw)
    assert rc == T.EXIT["device_fields"]
    j = json.load(open(tmp_path / "out" / "REFUSED_DEVICE_FIELDS.json"))
    assert j["status"] == "REFUSED" and "nvpmodel_q" in j["reason"] and "jetson_clocks_show" in j["reason"]
    assert "ZZTHORLAT_STOP DEVICE_FIELDS_MISSING" in capsys.readouterr().out
    assert not list((tmp_path / "out").glob("THOR_SEARCH_LATENCY_*"))                     # no timing was taken


def test_main_cpu_target_requires_a_cpu_only_process(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert T.main(_argv("run", "devbox-cpu", tmp_path), runner=runner_for()) == T.EXIT["usage"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert T.main(_argv("run", "thor", tmp_path), runner=runner_for()) == T.EXIT["usage"]


# ------------------------------------------------------------------------------------------------------------------------------ inputs: pins, the carve, labels
def test_labels_are_never_indexed_and_requesting_one_raises(tmp_path, monkeypatch):
    p = tmp_path / "b.npz"
    np.savez(p, token=np.array(["a", "b", "c"]), log=np.array(["l"] * 3), traj_raw=np.zeros((3, 4, 20, 3), np.float32), logits=np.zeros((3, 4, 6), np.float32), pick=np.zeros(3, np.int64),
             sub=np.ones((3, 4, 6)), pdms=np.ones((3, 4)), valid=np.ones((3, 4), bool))
    real_load = np.load
    seen = []

    class Spy:
        def __init__(self, z):
            self.z = z

        @property
        def files(self):
            return self.z.files

        def __getitem__(self, k):
            seen.append(k)
            return self.z[k]

        def close(self):
            self.z.close()

    monkeypatch.setattr(np, "load", lambda *a, **k: Spy(real_load(*a, **k)))
    out = T.load_fan(str(p), 2)
    assert set(seen) <= set(T.BUNDLE_ALLOWED) and not (set(seen) & set(T.BUNDLE_LABEL_KEYS))
    assert out["traj_raw"].shape == (2, 4, 20, 3) and out["token"] == ["a", "b"] and len(out["token_all"]) == 3
    seen.clear()
    for k in T.BUNDLE_LABEL_KEYS:
        with pytest.raises(T.LabelAccess):                                                # @GUARD:NOLABEL
            T.load_fan(str(p), 2, keys=("token", k))
    assert seen == []                                                                     # refused before any array was indexed


def test_pin_check_refuses_a_different_file():
    T._check_pin("x", "a" * 64, "a" * 64)
    with pytest.raises(T.Refusal):
        T._check_pin("x", "a" * 64, "b" * 64)                                             # @GUARD:PIN


def test_tokens_must_be_sorted_and_the_carve_is_the_sealed_one():
    T.check_tokens_sorted(["a", "b", "c"])
    with pytest.raises(T.Refusal):
        T.check_tokens_sorted(["b", "a"])                                                 # @GUARD:SORTED
    assert T.PINS["carve_b_tokens_sha256"] == SB.SEALED_CARVE_SHA256
    with pytest.raises(SystemExit):
        SB.check_data_clause({"token": np.array(["a", "b"])})                             # Step B's own data clause refuses any other token list
    assert T.tokens_sha256(["x", "y"]) == SB.tokens_sha256(["x", "y"])


def test_stepb_code_pins_equal_the_sealed_pins_and_a_tampered_copy_is_refused(tmp_path):
    d = T.locate_code()["stepB_dir"]
    for fn in T.STEPB_CODE_PINS:
        shutil.copy(os.path.join(d, fn), tmp_path / fn)
    assert T.verify_stepb_files(str(tmp_path)) == T.STEPB_CODE_PINS
    s = (tmp_path / "stepB.py").read_bytes()
    (tmp_path / "stepB.py").write_bytes(s.replace(b"K_ITERS", b"K_ITERz", 1) if b"K_ITERS" in s else s + b"\n# edited\n")
    with pytest.raises(T.Refusal):
        T.verify_stepb_files(str(tmp_path))                                               # @GUARD:CODEPIN
    (tmp_path / "stepB.py").unlink()
    with pytest.raises(T.Refusal):
        T.verify_stepb_files(str(tmp_path))                                               # a missing file is a mismatch, not a pass


def test_l1_files_are_the_landed_ones():
    got = SB.verify_l1_code(T.locate_code()["l1_dir"])
    assert set(got) == {"stepA.py", "tts_cem.py", "real_scorer.py", "label_perturbations.py"}


def test_this_script_reads_no_label_name_outside_the_guard_lists():
    src = open(os.path.join(HERE, "thor_search_latency.py"), encoding="utf-8").read()
    for lab in ('["sub"]', '["pdms"]', '["valid"]', "z['sub']", "z['pdms']", "z['valid']"):
        assert lab not in src
    assert "load_npz" not in src                                                          # stepA.load_npz would read EVERY array, labels included


# ------------------------------------------------------------------------------------------------------------------------------ the timed bracket, the arm rotation, pick-only, the exit code
def test_every_timed_call_is_bracketed_by_a_device_sync():
    log = []

    class Clock(FakeClock):
        def __call__(self):
            log.append("clock")
            return super().__call__()

    clock = Clock()

    def scorer(S):
        log.append("score")
        clock.advance(1000)
        return np.zeros((np.shape(S)[0], 6), np.float32)

    cnt = T.CountingScorer(scorer, clock, lambda: log.append("sync"))
    cnt(np.zeros((4, 20, 3), np.float32))
    assert log == ["sync", "clock", "score", "sync", "clock"]                             # sync, t0, work, sync, t1: the work is fully inside the bracket
    log.clear()
    T.time_one(lambda sf: sf(np.zeros((4, 20, 3), np.float32)), cnt, clock, lambda: log.append("sync"))
    assert log[:2] == ["sync", "clock"] and log[-2:] == ["sync", "clock"] and log.count("sync") == 4    # the outer bracket as well as the per-call one


def test_counting_scorer_changes_nothing_it_returns_the_same_object_and_leaves_the_input_alone():
    S = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    S0 = S.copy()
    out = np.ones((2, 6), np.float32)
    cnt = T.CountingScorer(lambda x: out, lambda: 0, lambda: None)
    assert cnt(S) is out and np.array_equal(S, S0) and cnt.calls == 1 and cnt.rows == 2


def test_the_three_arms_rotate_through_every_position():
    seen = []

    class Rec(T.CountingScorer):
        def __call__(self, S):
            if self.calls == 0:
                seen.append(int(np.shape(S)[0]))                                          # the first call of an arm identifies it: 8 rows = (a), 16 = (b), 48 = (c)
            return super().__call__(S)

    run_measure(n=3, warmup=1, repeats=2, wrap=Rec)
    ident = {8: "a", 16: "b", 48: "c"}
    timed = [ident[x] for x in seen[3 * 1:3 * 1 + 3 * 3 * 2]]                              # drop the warm-up (1 token x 3 arms), then repeats x tokens x arms
    base = ["a", "b", "c"]
    want = []
    for r in range(2):
        for i in range(3):
            k = (i + r) % 3
            want += base[k:] + base[:k]
    assert timed == want                                                                  # @GUARD:ROTATE
    for pos in range(3):
        assert {a: sum(1 for j in range(pos, len(timed), 3) if timed[j] == a) for a in base} == {"a": 2, "b": 2, "c": 2}


def test_pick_only_is_the_argmax_of_the_aggregate():
    logits = np.zeros((4, 6))
    logits[:, :2] = 5.0
    logits[:, 2:5] = [[-3.0] * 3, [0.0] * 3, [4.0] * 3, [1.0] * 3]
    out = T.pick_only(lambda S: logits, np.zeros((4, 20, 3), np.float32), A)
    assert out["pick"] == 2 and out["logits"].dtype == np.float64                        # @GUARD:PICKONLY


def test_finalize_returns_zero_only_when_every_control_is_green_and_always_writes_the_record(honest, tmp_path):
    m, sc, rep = honest
    kw = dict(env={"target": "devbox-cpu", "cpu_name": "x"}, pins_seen={}, inputs={}, cfgs={}, args_echo={}, preflight=None, preflight_end=None, thermal_end=None)
    green = T.build_record(target="devbox-cpu", rep=rep, **kw)
    assert T.finalize(green, str(tmp_path / "g"), "g") == T.EXIT["ok"]
    red_rep = json.loads(json.dumps(rep))
    red_rep["controls_ok"] = False
    red_rep["controls"]["identity_b"]["ok"] = False
    red = T.build_record(target="devbox-cpu", rep=red_rep, **kw)
    assert T.finalize(red, str(tmp_path / "r"), "r") == T.EXIT["controls_red"] == 6        # @GUARD:EXITCODE
    assert json.load(open(tmp_path / "r" / "THOR_SEARCH_LATENCY_r.json"))["controls_ok"] is False     # a RED record is still written, and says so


def test_end_scan_marks_a_job_that_started_during_the_run_as_contamination(honest):
    m, sc, rep = honest
    r1 = json.loads(json.dumps(rep))
    end = T.apply_end_scan(r1, PS_IDLE, 0, 2200, "nvidia")
    assert r1["controls_ok"] is True and end["foreign_busy_processes_at_end"] == [] and end["ps_readable"]
    r2 = json.loads(json.dumps(rep))
    end = T.apply_end_scan(r2, PS_IDLE + TRAINER, 0, 2200, "nvidia")
    assert r2["controls_ok"] is False and not r2["controls"]["no_foreign_job_at_end"]["ok"] and len(end["foreign_busy_processes_at_end"]) == 1    # @GUARD:ENDSCAN


def test_n_tokens_range_is_checked():
    T.check_n_tokens(1, 400)
    T.check_n_tokens(400, 400)
    for bad in (0, -3, 401):
        with pytest.raises(T.Refusal):
            T.check_n_tokens(bad, 400)                                                    # @GUARD:NTOK


def test_the_search_seed_is_the_first_of_the_config_and_one_seed_only():
    cfg = SB.c8_from_grid()
    assert int(cfg.seeds[0]) == 20260927 and len(cfg.seeds) == 3
    src = open(os.path.join(HERE, "thor_search_latency.py"), encoding="utf-8").read()
    assert "cfg_b.seeds[int(a.seed_index)]" in src and "for s in cfg" not in src
