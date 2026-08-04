"""Tests for the kill-forensics probe.

These lock down the two readings that each cost the programme a wrong diagnosis
of the same ``rc=137``:

* ``memory.usage_in_bytes`` at 98-100 % of the cap read as pressure, when it
  counts reclaimable page cache (retracted as ``R-2026-08-03-mem``);
* ``memory.failcnt == 0`` read as proof the cap was never hit, when on this
  cgroup that counter is **structurally unable to move**.

Both are the same failure — a counter quoted without establishing what it can
say — so the verdict logic is tested directly rather than through a pod. The
fixtures carry the real numbers MEASURED on ``tanitad-new`` 2026-08-04.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pod_kill_forensics import (  # noqa: E402
    decode_exit_code, inflight_sample_bytes, live_failcnt, lru_ram_bytes,
    memory_headroom, oom_window, parse_kv, project_config, summarize_samples,
    unreclaimable_bytes,
)

# --- the real thing, MEASURED on tanitad-new 2026-08-04 -------------------
MEM_LIMIT = 49_999_998_976          # memory.limit_in_bytes
MEMSW_LIMIT = 49_999_998_976        # memory.memsw.limit_in_bytes — EQUAL
MEAN_PAYLOAD = int(33.4 * 2**20)    # v2 clip, n=40 sample of the train cache

MEMORY_STAT = """cache 43713548288
rss 5246406656
rss_huge 0
shmem 5385515008
mapped_file 5435219968
swap 0
active_anon 9950978048
inactive_anon 680673280
active_file 13523894272
inactive_file 24804139008
"""


# ---------------------------------------------------------------- parsing

def test_parse_kv_skips_non_integer_fields():
    """These files gain fields across kernels; a probe must not die on one."""
    got = parse_kv("cache 10\nhierarchical_limit max\nrss 20\nblank\n")
    assert got == {"cache": 10, "rss": 20}


# ------------------------------------------------------------ exit codes

def test_137_is_sigkill_and_never_a_cuda_oom():
    got = decode_exit_code(137)
    assert got["kind"] == "signal" and got["signal"] == 9
    assert got["name"] == "SIGKILL"
    # The confusion that sent the v5f diagnosis sideways twice.
    assert "exits 1" in got["meaning"]


def test_cuda_oom_lands_on_rc_1_with_a_traceback():
    got = decode_exit_code(1)
    assert got["kind"] == "exception"
    assert "CUDA OOM" in got["meaning"]


def test_sigterm_and_clean_exit_are_distinguished():
    assert decode_exit_code(143)["name"] == "SIGTERM"
    assert decode_exit_code(0)["kind"] == "clean"
    assert decode_exit_code(139)["name"] == "SIGSEGV"


# -------------------------------------------------- WHICH failcnt is live

def test_memory_failcnt_is_frozen_when_memsw_equals_memory_and_no_swap():
    """THE finding. memory.failcnt cannot move here, so 0 proves nothing."""
    got = live_failcnt(MEM_LIMIT, MEMSW_LIMIT, swap_total=0)
    assert got["live"] == "memory.memsw.failcnt"
    assert "memory.failcnt" in got["frozen"]
    assert "FROZEN" in got["reason"]


def test_memory_failcnt_is_live_when_swap_headroom_exists():
    got = live_failcnt(MEM_LIMIT, MEM_LIMIT * 2, swap_total=0)
    assert got["live"] == "memory.failcnt"
    assert got["frozen"] == []


def test_no_memsw_accounting_leaves_memory_failcnt_live():
    got = live_failcnt(MEM_LIMIT, None)
    assert got["live"] == "memory.failcnt"


def test_unknown_limit_does_not_pretend_to_know():
    assert live_failcnt(None, None)["live"] == "unknown"


# ------------------------------------------------------------- headroom

def test_unreclaimable_excludes_page_cache():
    """43.7 GB of `cache` is reclaimable and must not count as pressure."""
    stat = parse_kv(MEMORY_STAT)
    assert unreclaimable_bytes(stat) == 5_246_406_656 + 5_385_515_008


def test_headroom_reports_the_number_that_decides():
    hr = memory_headroom(parse_kv(MEMORY_STAT), MEM_LIMIT)
    # ~9.9 GiB unreclaimable against a ~46.6 GiB cap: ample, despite
    # usage_in_bytes simultaneously reading 98 % of that cap.
    assert 9.5 < hr["unreclaimable_gb"] < 10.5
    assert hr["headroom_gb"] > 35
    assert hr["unreclaimable_pct_of_limit"] < 25


# ------------------------------------------------------------- oom window

def test_oom_kill_without_a_container_start_is_not_quotable():
    """It reset 6 -> 0 on pod2 and was quoted as history. Never again."""
    got = oom_window({"oom_kill": 1, "under_oom": 0}, None, now=1_000_000.0)
    assert got["quotable"] is False
    assert "MUST NOT be quoted" in got["statement"]


def test_oom_kill_with_a_container_start_carries_its_window():
    got = oom_window({"oom_kill": 1, "under_oom": 0, "oom_kill_disable": 0},
                     1_000_000.0 - 3600 * 31.8, now=1_000_000.0)
    assert got["quotable"] is True
    assert got["window_hours"] == 31.8
    assert "31.8 h" in got["statement"]


def test_oom_counter_does_not_claim_which_killer_fired():
    got = oom_window({"oom_kill": 1}, 1_000_000.0 - 3600, now=1_000_000.0)
    assert "does NOT by itself say which" in got["caveat"]


# ------------------------------------------------- the DataLoader RAM model

def test_lru_ram_counts_the_main_process_too():
    """LRU is per-process: workers + 1, not workers."""
    assert lru_ram_bytes(4, 4, MEAN_PAYLOAD) == 5 * 4 * MEAN_PAYLOAD


def test_the_attempted_lru_is_a_double_digit_gb_change():
    """--v2-lru 4 -> 64 with --workers 4 -> 8 is ~+18 GiB of ANON memory."""
    shipped = lru_ram_bytes(4, 4, MEAN_PAYLOAD)
    attempted = lru_ram_bytes(8, 64, MEAN_PAYLOAD)
    delta_gb = (attempted - shipped) / 2**30
    assert 17.0 < delta_gb < 20.0
    assert shipped / 2**30 < 1.0        # the shipped config is under a GiB


def test_docstring_payload_estimate_would_have_under_budgeted_by_over_10x():
    """The class docstring says 2-4 MB/clip; this cache MEASURED 33.4 MB."""
    optimistic = lru_ram_bytes(8, 64, 3 * 2**20)
    real = lru_ram_bytes(8, 64, MEAN_PAYLOAD)
    assert real / optimistic > 10


def test_inflight_sample_cost_backsolves_from_observed_shmem():
    per = inflight_sample_bytes(5_385_515_008, workers=4, batch=4, prefetch_factor=2)
    assert 150 * 2**20 < per < 175 * 2**20      # ~160 MiB per in-flight sample
    assert inflight_sample_bytes(1234, workers=0, batch=4) == 0.0


def test_the_attempted_config_is_projected_OVER_the_cap():
    """workers 8 + batch 8 + lru 64 does not fit. That is the rc=137."""
    got = project_config(
        unreclaimable_bytes(parse_kv(MEMORY_STAT)),
        from_workers=4, from_batch=4, from_lru=4,
        to_workers=8, to_batch=8, to_lru=64,
        mean_payload_bytes=MEAN_PAYLOAD,
        observed_shmem_bytes=5_385_515_008,
        per_worker_anon_bytes=2_127_360 * 1024)
    assert got["projected_unreclaimable_bytes"] > MEM_LIMIT
    assert got["evidence_class"].startswith("ESTIMATED")


def test_batch_only_change_is_projected_WELL_UNDER_the_cap():
    """Holding workers and lru, --batch 8 --accum 8 costs only the transport."""
    got = project_config(
        unreclaimable_bytes(parse_kv(MEMORY_STAT)),
        from_workers=4, from_batch=4, from_lru=4,
        to_workers=4, to_batch=8, to_lru=4,
        mean_payload_bytes=MEAN_PAYLOAD,
        observed_shmem_bytes=5_385_515_008,
        per_worker_anon_bytes=2_127_360 * 1024)
    assert got["delta_lru_gb"] == 0.0
    assert got["delta_worker_anon_gb"] == 0.0
    assert got["projected_unreclaimable_bytes"] < MEM_LIMIT / 2


# --------------------------------------------------------------- sampling

def test_summarize_needs_and_uses_many_samples():
    """MEASURED spread on the live trainer: 21 % to 100 % within 20 s."""
    vals = [27, 34, 29, 26, 75, 100, 34, 100, 100, 29,
            39, 32, 26, 39, 31, 37, 27, 25, 21, 35]
    got = summarize_samples(vals)
    assert got["n"] == 20
    assert got["median"] == 33.0
    assert got["min"] == 21.0 and got["max"] == 100.0
    # a single sample could have been any of these — hence the >=10 rule
    assert got["max"] - got["min"] > 70


def test_summarize_handles_empty():
    assert summarize_samples([])["median"] is None
