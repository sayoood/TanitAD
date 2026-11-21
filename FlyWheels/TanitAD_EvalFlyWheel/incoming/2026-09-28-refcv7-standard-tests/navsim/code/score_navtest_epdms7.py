#!/usr/bin/env python3
"""NAVSIM v2 EPDMS on ``navtest`` (ONE stage) for the refcv7 suite -- by RE-SCORING banked plans (TANITAD VENV).

    python code/score_navtest_epdms7.py plan                      # readiness, gates, addendum pin (no compute)
    python code/score_navtest_epdms7.py prep                      # one-off: compact token meta from the v2 export
    python code/score_navtest_epdms7.py score --arm R7_A1 --step 5000          # ONE full-split arm
    python code/score_navtest_epdms7.py score --arm STOP                       # a floor (checkpoint-independent)
    python code/score_navtest_epdms7.py score --arm R7_A1 --step 5000 --tokens-file T.json --tag smoke_a   # SMOKE
    python code/score_navtest_epdms7.py summarize --step 5000     # statistics + guards -> summary_navtest_epdms.json
    python code/score_navtest_epdms7.py compare --a X.csv --b Y.csv   # cell-by-cell identity (controls)

PRE-REGISTRATION: ``SPEC_ADDENDUM_NAVTEST_EPDMS.md`` (same package). This tool REFUSES to run when that file's
sha256 is not the LAST hash line recorded in ``raw/SPEC_ADDENDUM_NAVTEST_EPDMS_SHA256.txt`` (an edit without an
amendment hash line is a silent goalpost move).

NOTHING HERE IS A RE-IMPLEMENTATION OF THE METRIC. The score is the devkit's own: ``navsim@0a380a9``
``run_pdm_score_one_stage`` driven through the suite's promoted scorer
(``taniteval.bench.navsim.scoring.score_arm``: E1's wrapper, E2's seam agent, the RAM guard, the count guards)
on the profile ``navtest_single_stage``. What lives here is: the pins, the seam handling (a banked seam is read,
never rewritten), the guards of the addendum (G1-G5), the controls (KE1-KE7), the statistics (W2's
``navsim_ci`` estimators, reused through W8's ``single_stage`` helpers) and the summary.

Resource rules (addendum section 7): one scorer at a time; the wrapper's RAM guard is 4,000 MB (the other jobs'
floor is 3,000 MB, so THIS job yields first); BELOW_NORMAL priority; no GPU; the pre-aggregation dump is deleted
after a PASS (disk is scarce).
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TANITEVAL = "D:/Projects/TanitAD/taniteval"
if TANITEVAL not in sys.path:
    sys.path.insert(0, TANITEVAL)

ADDENDUM = os.path.join(PKG, "SPEC_ADDENDUM_NAVTEST_EPDMS.md")
SHAFILE = os.path.join(PKG, "raw", "SPEC_ADDENDUM_NAVTEST_EPDMS_SHA256.txt")
OUT = os.path.join(PKG, "raw", "navtest_epdms")
MS = os.path.join(PKG, "raw", "milestones")
W3_RAW = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw")
EXPORT_JSON = "C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navtest_single_stage/navsim_agent_inputs.json"
SCHEMA = "navtest-epdms7/1"
PROTOCOL = "EPDMS_v2_navtest_single_stage"

#: Everything below is pinned by the addendum (section 0 F1/F2/F3). A cache or seam that does not match is REFUSED.
PINS = {
    "tokens_sha256": "8d42fef68542f095d0e03894fdaf38babfdc89f3fd7cdc26080c3a6009cb4108",
    "cache_manifest_sha256": "b02d1a833f205035421e5ce5b180a8d41657bd7d234dd5866e2aa3df6ca04706",
    "cache_metadata_sha256": "c6b9c4bf6807da78e1770f564fb469d0eef3dfab5b77f5553a01acc07f1bf840",
    "export_sha256": "a41795496e00cb62ec7719781b744e2338c09340b2d55834be2f12a7328ade11",
    "n_tokens": 12146,
    "n_logs": 136,
    "devkit_sha": "0a380a9063d7162ec93d0f51e9990ebac585f720",
}
FLOORS = ("STOP", "CV", "HUMAN")
MODEL_ARMS = ("R7_A1", "R7_A1_s1", "R7_CEILDECL_d", "PRIOR_ha0p")
#: scoring priority inside a milestone (a killed job still yields the most valuable rows first)
STEP_ORDER = ("R7_A1", "R7_A1_s1", "PRIOR_ha0p", "R7_CEILDECL_d")
FLOOR_ORDER = ("STOP", "CV", "HUMAN")
OFFICIAL_AGENT = {"CV": "constant_velocity_agent", "HUMAN": "human_agent"}
COMPONENTS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
              "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
              "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "LK": "lane_keeping",
              "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
MULTIPLIERS = ("NC", "DAC", "DDC", "TLC")
WEIGHTS = {"EP": 5.0, "TTC": 5.0, "LK": 2.0, "HC": 2.0, "EC": 2.0}
SUMMARY_ROW = "average_all_frames"
GATE_PHYS_GB = 9.0
GATE_VIRT_GB = 6.0
GATE_SAMPLES = 5
GATE_EVERY_S = 30
SMOKE_PHYS_GB = 4.0
SMOKE_VIRT_GB = 4.0
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fwd(p) -> str:
    return str(p).replace(os.sep, "/")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tokens_sha(tokens) -> str:
    """sha256 of the sorted, newline-joined token list (the cache's own ``tokens_sha256`` definition)."""
    return hashlib.sha256("\n".join(sorted(tokens)).encode("utf-8")).hexdigest()


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# 0. the metric, derived INDEPENDENTLY of the devkit (KE1)                     #
# --------------------------------------------------------------------------- #
def epdms_from_components(r: dict) -> float:
    """docs/metrics.md L28-33 written out: prod(NC,DAC,DDC,TLC) * sum(w*m)/sum(w), weights EP 5, TTC 5, LK 2,
    HC 2, EC 2; EC is dropped (denominator 14) when it is NaN, as ``compute_final_scores`` zeroes its weight.
    ``r`` maps SHORT names (NC..EC) to floats."""
    prod = 1.0
    for k in MULTIPLIERS:
        prod *= float(r[k])
    num = den = 0.0
    for k, w in WEIGHTS.items():
        v = float(r[k])
        if k == "EC" and math.isnan(v):
            continue
        num += w * v
        den += w
    return prod * num / den


def max_formula_gap(tok: dict) -> tuple:
    """KE1 over a ``load_token_scores`` dict -> (max |formula - score|, n rows)."""
    worst, n = 0.0, 0
    for r in tok.values():
        row = {k: r[c] for k, c in COMPONENTS.items()}
        worst = max(worst, abs(epdms_from_components(row) - r["score"]))
        n += 1
    return worst, n


def classify(d_x100: float, interval: dict, floor_x100: float) -> str:
    """Addendum section 5. ``interval`` is a paired-bootstrap block in SCORE units (fractions of 1).
    SEPARATED iff the interval excludes 0 and |d| > 2F; NOT PROVEN iff it excludes 0 but |d| <= 2F;
    otherwise NOT SEPARATED. An unavailable interval is reported as such, never guessed."""
    if not interval or interval.get("status") != "OK" or interval.get("lo") is None:
        return "INTERVAL UNAVAILABLE"
    excl = (interval["lo"] > 0.0) or (interval["hi"] < 0.0)
    if not excl:
        return "NOT SEPARATED"
    return "SEPARATED" if abs(d_x100) > 2.0 * floor_x100 else "NOT PROVEN"


# --------------------------------------------------------------------------- #
# 1. the addendum pin, memory and disk gates                                   #
# --------------------------------------------------------------------------- #
def addendum_state() -> dict:
    cur = sha256_file(ADDENDUM)
    rec = []
    try:
        for ln in open(SHAFILE, encoding="utf-8"):
            m = re.match(r"^sha256\S*\s+([0-9a-f]{64})\s", ln)
            if m:
                rec.append(m.group(1))
    except FileNotFoundError:
        pass
    last = rec[-1] if rec else None
    return {"current": cur, "recorded": rec, "last_recorded": last, "ok": bool(last and cur == last)}


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def mem_status() -> dict:
    """``FreePhysicalMemory`` (available RAM incl. standby) and ``FreeVirtualMemory`` (commit headroom =
    commit limit - committed) from ``GlobalMemoryStatusEx`` -- the same two quantities the brief names, read
    without spawning PowerShell (a PowerShell probe timed out under load in W8's waiter). GB = GiB."""
    s = _MemStatus()
    s.dwLength = ctypes.sizeof(_MemStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        raise OSError("GlobalMemoryStatusEx failed")
    g = 2.0 ** 30
    return {"free_phys_gb": s.ullAvailPhys / g, "free_virt_gb": s.ullAvailPageFile / g,
            "total_phys_gb": s.ullTotalPhys / g, "load_pct": int(s.dwMemoryLoad)}


def gate_ok(sample: dict, min_phys: float = GATE_PHYS_GB, min_virt: float = GATE_VIRT_GB) -> bool:
    return sample["free_phys_gb"] >= min_phys and sample["free_virt_gb"] >= min_virt


def gate_sustained(samples: list, min_phys: float = GATE_PHYS_GB, min_virt: float = GATE_VIRT_GB,
                   n_ok: int = GATE_SAMPLES) -> bool:
    """True iff the LAST ``n_ok`` samples all pass (consecutive, not cumulative)."""
    return len(samples) >= n_ok and all(gate_ok(s, min_phys, min_virt) for s in samples[-n_ok:])


def disk_free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 2.0 ** 30


def lower_priority() -> None:
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:                                                       # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# 2. seams, cache identity, token meta                                         #
# --------------------------------------------------------------------------- #
def seam_path(step: int, arm: str) -> str:
    return os.path.join(MS, f"step{step}", "bridge_navtest", f"seam_{arm}.npz")


def seam_manifest_path(step: int, arm: str) -> str:
    return os.path.join(MS, f"step{step}", "bridge_navtest", f"seam_{arm}.manifest.json")


def seam_info(path: str) -> dict:
    """Identity of a banked seam. Reads it (never writes it)."""
    import numpy as np
    z = np.load(path, allow_pickle=False)
    toks = [str(x) for x in z["token"]]
    poses = np.asarray(z["poses"])
    return {"path": fwd(path), "sha256": sha256_file(path), "n": len(toks), "n_unique": len(set(toks)),
            "tokens_sha256": tokens_sha(toks), "poses_shape": list(poses.shape),
            "finite": bool(np.isfinite(poses).all()), "sampling": [float(x) for x in z["sampling"]],
            "sources": sorted({str(x) for x in z["source"]}), "arm_key": str(z["arm"]) if "arm" in z.files else None}


def seam_refusals(info: dict, full: bool = True) -> list:
    bad = []
    if info["n"] != info["n_unique"]:
        bad.append("duplicate tokens")
    if not info["finite"]:
        bad.append("non-finite poses")
    if info["poses_shape"][1:] != [8, 3] or info["sampling"] != [8.0, 0.5]:
        bad.append(f"geometry {info['poses_shape']} / sampling {info['sampling']} != [N,8,3] / [8,0.5]")
    if full and info["tokens_sha256"] != PINS["tokens_sha256"]:
        bad.append(f"token-set sha256 {info['tokens_sha256'][:12]} != cache {PINS['tokens_sha256'][:12]}")
    if full and info["n"] != PINS["n_tokens"]:
        bad.append(f"{info['n']} tokens != {PINS['n_tokens']}")
    return bad


def seam_stop_fraction(path: str) -> float:
    import numpy as np
    p = np.load(path, allow_pickle=False)["poses"]
    return float((np.hypot(p[:, -1, 0], p[:, -1, 1]) < 1.0).mean())


def cache_identity(prof) -> dict:
    """G3: ``CACHE_DONE.json`` and the manifest / metadata hashes equal the addendum's pins."""
    cd = json.load(open(os.path.join(str(prof.cache), "CACHE_DONE.json"), encoding="utf-8"))
    man = sha256_file(os.path.join(str(prof.cache), "CACHE_MANIFEST.json"))
    meta = sha256_file(os.path.join(str(prof.cache), "metadata", "metric_cache_navtest_v2_metadata_node_0.csv"))
    rec = {"tokens_sha256": cd.get("tokens_sha256"), "manifest_sha256": man, "metadata_csv_sha256": meta,
           "token_set": cd.get("token_set"), "n_cached": (cd.get("n_cached") or {}).get("total")}
    bad = []
    if rec["tokens_sha256"] != PINS["tokens_sha256"]:
        bad.append("CACHE_DONE tokens_sha256")
    if man != PINS["cache_manifest_sha256"]:
        bad.append("CACHE_MANIFEST sha256")
    if meta != PINS["cache_metadata_sha256"]:
        bad.append("metadata csv sha256")
    if rec["token_set"] != "FULL_SPLIT" or rec["n_cached"] != PINS["n_tokens"]:
        bad.append("not a FULL_SPLIT cache of 12,146")
    rec["ok"] = not bad
    rec["mismatches"] = bad
    return rec


def meta_path() -> str:
    return os.path.join(OUT, "inputs", "token_meta.json")


def cmd_prep(a) -> int:
    """Compact per-token meta {log, fingerprint, command, v0} from the FULL v2 export (61 MB json, ~0.5 GB RAM
    transient). Needed for the STOP seam, the per-command / speed-band split and nothing else."""
    import numpy as np
    sha = sha256_file(EXPORT_JSON)
    if sha != PINS["export_sha256"]:
        print(f"REFUSED: export sha256 {sha} != pinned {PINS['export_sha256']}")
        return 2
    ex = json.load(open(EXPORT_JSON, encoding="utf-8"))["tokens"]
    meta = {}
    for t, r in ex.items():
        es = r["ego_statuses"][-1]
        oh = [float(x) for x in es["driving_command"]]
        vx, vy = es["ego_velocity"]
        meta[t] = {"log": r["log_name"], "fp": r["fingerprint"], "cmd": int(np.argmax(oh)),
                   "onehot": bool(sorted(oh) == [0.0] * (len(oh) - 1) + [1.0]), "v0": float(math.hypot(vx, vy))}
    if tokens_sha(meta) != PINS["tokens_sha256"] or len(meta) != PINS["n_tokens"]:
        print("REFUSED: export token set != pinned token set")
        return 2
    if len({m["log"] for m in meta.values()}) != PINS["n_logs"]:
        print("REFUSED: export does not cover 136 logs")
        return 2
    write_json(meta_path(), {"schema": "navtest-epdms7-token-meta/1", "export_sha256": sha, "written_utc": now(),
                             "command_order": ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"], "tokens": meta})
    print(f"token meta: {len(meta)} tokens, {len({m['log'] for m in meta.values()})} logs -> {meta_path()}")
    return 0


def load_meta() -> dict:
    p = meta_path()
    if not os.path.exists(p):
        raise SystemExit("REFUSED: raw/navtest_epdms/inputs/token_meta.json absent -- run `prep` first")
    d = json.load(open(p, encoding="utf-8"))
    if d.get("export_sha256") != PINS["export_sha256"]:
        raise SystemExit("REFUSED: token meta was not built from the pinned export")
    return d["tokens"]


# --------------------------------------------------------------------------- #
# 3. log hygiene                                                               #
# --------------------------------------------------------------------------- #
def sanitize_text(text: str) -> tuple:
    """The devkit logs ``thread_id=<uuid4>`` per worker (a random id, NOT a PhysicalAI clip id). The package's
    landing scan refuses any file carrying a canonical UUID, so the banked copy carries ``<thread-uuid>``."""
    new, n = UUID_RE.subn("<thread-uuid>", text)
    return new, n


def bank_text_file(src: str, dst: str) -> dict:
    raw = open(src, encoding="utf-8", errors="replace").read()
    new, n = sanitize_text(raw)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new)
    return {"src_sha256": sha256_file(src), "dst": fwd(dst), "n_uuid_replaced": n, "dst_sha256": sha256_file(dst)}


# --------------------------------------------------------------------------- #
# 4. one arm                                                                   #
# --------------------------------------------------------------------------- #
def scope_of(arm: str, step, tag) -> tuple:
    """-> (scope dir name, label). Subsets are ALWAYS smoke scope."""
    if tag:
        return os.path.join("smoke", tag), f"smoke_{tag}"
    if arm in FLOORS:
        return "floors", "floors"
    return f"step{step}", f"step{step}"


LOCK_ROOT = "C:/Users/Admin/navsim-crun/exp/tanitad_bench/runs/epdms7/locks"      # scratch, NOT in the package


def lock_path(scope: str, arm: str) -> str:
    return os.path.join(LOCK_ROOT, f"{scope.replace(os.sep, '_').replace('/', '_')}__{arm}.json")


def take_lock(scope: str, arm: str) -> bool:
    """One scorer per (scope, arm). The lock lives in scratch so a landing sweep of the package never sees it."""
    import psutil
    os.makedirs(LOCK_ROOT, exist_ok=True)
    lp = lock_path(scope, arm)
    if os.path.exists(lp):
        try:
            pid = int(json.load(open(lp, encoding="utf-8")).get("pid", -1))
        except Exception:                                                   # noqa: BLE001
            pid = -1
        if pid > 0 and psutil.pid_exists(pid):
            return False
        os.remove(lp)                                                       # stale: its owner is gone
    fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "utc": now()}, fh)
    return True


def subset_seam(parent: str, tokens: list, dst: str) -> dict:
    """A SUBSET seam for the smoke: the parent's rows, bit-identical, in the parent's order."""
    import numpy as np
    z = np.load(parent, allow_pickle=False)
    toks = [str(x) for x in z["token"]]
    want = set(tokens)
    idx = [i for i, t in enumerate(toks) if t in want]
    if len(idx) != len(want):
        raise SystemExit(f"REFUSED: {len(want) - len(idx)} subset tokens are absent from {parent}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    per_row = ("token", "fingerprint", "source", "poses", "knots")          # the E2 seam layout (seams.py)
    missing = [k for k in per_row if k not in z.files]
    if missing:
        raise SystemExit(f"REFUSED: {parent} lacks seam keys {missing}")
    np.savez(dst, **{k: np.asarray(z[k])[idx] for k in per_row},
             sampling=np.asarray(z["sampling"]), arm=np.asarray(str(z["arm"]) if "arm" in z.files else "?"))
    y = np.load(dst, allow_pickle=False)
    if not np.array_equal(y["poses"], np.asarray(z["poses"])[idx]):
        raise SystemExit("REFUSED: subset seam poses are not bit-identical to the parent's rows")
    return {"parent": fwd(parent), "parent_sha256": sha256_file(parent), "n": len(idx), "path": fwd(dst),
            "sha256": sha256_file(dst)}


def stop_seam(path: str, tokens, meta: dict) -> str:
    """The suite's OWN STOP builder (``seams.make_stop_seam``) on a doc-shaped dict derived from the pinned export."""
    from taniteval.bench.navsim import seams as SM
    doc = {"tokens": {t: {"fingerprint": meta[t]["fp"]} for t in tokens}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    SM.make_stop_seam(doc, path, arm="STOP")
    return path


def postcheck(csv_path: str, expected_tokens, full: bool) -> dict:
    """G2 + G5 on a devkit CSV, read as TEXT."""
    from taniteval.bench.navsim import single_stage as SS, summarize as S
    hdr, raw = S.read_raw_rows(csv_path)
    if "pdm_score" in hdr and "score" not in hdr:
        return {"ok": False, "failures": ["CSV offers only pdm_score (forbidden column)"]}
    head = S.headline_value(raw, hdr, row=SUMMARY_ROW)                       # refuses pdm_score by name
    tok = SS.load_token_scores(csv_path)
    fails = []
    n = len(tok)
    n_valid = sum(1 for r in tok.values() if str(r.get("valid")) in ("True", "true", "1"))
    if n != len(expected_tokens) or n_valid != n:
        fails.append(f"G1 token rows {n} / valid {n_valid} != expected {len(expected_tokens)}")
    if set(tok) != set(expected_tokens):
        fails.append(f"G2 token set differs ({len(set(tok) - set(expected_tokens))} extra, "
                     f"{len(set(expected_tokens) - set(tok))} missing)")
    tsha = tokens_sha(tok)
    if full and tsha != PINS["tokens_sha256"]:
        fails.append(f"G2 token-set sha256 {tsha[:12]} != pinned {PINS['tokens_sha256'][:12]}")
    gap, n_f = max_formula_gap(tok)
    if not (gap <= 1e-9 and n_f == n and n > 0):
        fails.append(f"KE1 formula gap {gap}")
    avg = SS.average_row_check(tok, head)
    if not avg["pass"]:
        fails.append(f"G5 average row != skipna mean (|d| {avg['abs_diff']})")
    n_ec_nan = sum(1 for r in tok.values() if math.isnan(r.get("two_frame_extended_comfort", float("nan"))))
    return {"ok": not fails, "failures": fails, "n_tokens": n, "tokens_sha256": tsha,
            "headline_score": head, "headline_x100": round(100.0 * head, 4), "KE1_max_abs_gap": gap,
            "C_AVG": avg, "n_ec_nan": n_ec_nan}


def run_arm(*, arm: str, step=None, tokens=None, tag: str = "", **kw) -> dict:
    """Score ONE arm. -> the ARM_DONE / ARM_FAILED record (also written to disk). EVERY return -- a PASS, a
    FAIL, a closed gate -- also leaves ``ARM_LAST_ATTEMPT.json`` beside it: the waiter reads its verdict from
    ARTIFACTS, and a gate that closed (no PASS, no FAIL) must not look like a crash or cost a retry."""
    rec = _run_arm(arm=arm, step=step, tokens=tokens, tag=tag, **kw)
    scope, _ = scope_of(arm, step, tag)
    write_json(os.path.join(OUT, scope, arm, "ARM_LAST_ATTEMPT.json"),
               {"status": rec.get("status"), "retryable": rec.get("retryable"), "arm": arm, "scope": scope,
                "utc": now(), "epoch": time.time(), "counts_only_if": ("RAM_GUARD_ABORT", "FAIL", "FAIL_POSTCHECK",
                                                                       "AGGREGATION_FAILED", "SEAM_ABSENT")})
    return rec


def _run_arm(*, arm: str, step=None, tokens=None, tag: str = "", ram_floor_mb: float = 4000.0,
             log=print, min_phys: float = None, min_virt: float = None) -> dict:
    from taniteval.bench.navsim import profiles as P, scoring as SC
    lower_priority()
    full = tokens is None
    if not full and not tag:
        raise SystemExit("REFUSED: a token subset is a SMOKE and needs --tag (it is never a result)")
    if arm in MODEL_ARMS and step is None:
        raise SystemExit(f"REFUSED: {arm} needs --step")
    if arm not in FLOORS and arm not in MODEL_ARMS:
        raise SystemExit(f"REFUSED: unknown arm {arm!r} (floors {FLOORS}, model arms {MODEL_ARMS})")
    scope, label = scope_of(arm, step, tag)
    out_dir = os.path.join(OUT, scope, arm)
    ads = addendum_state()
    if not ads["ok"]:
        raise SystemExit(f"REFUSED: the addendum's sha256 {ads['current'][:12]} is not the last recorded hash line "
                         f"{(ads['last_recorded'] or 'NONE')[:12]} -- an edit without an amendment hash line")
    # ---- memory / disk gate, read and LOGGED before anything starts
    ms = mem_status()
    mp = (GATE_PHYS_GB if full else SMOKE_PHYS_GB) if min_phys is None else min_phys
    mv = (GATE_VIRT_GB if full else SMOKE_VIRT_GB) if min_virt is None else min_virt
    log(f"[epdms7] {arm} {scope}: FreePhysicalMemory {ms['free_phys_gb']:.2f} GB, FreeVirtualMemory "
        f"{ms['free_virt_gb']:.2f} GB (gate {mp} / {mv})")
    if not gate_ok(ms, mp, mv):
        return {"status": "GATE_CLOSED", "retryable": True, "arm": arm, "scope": scope, "mem": ms,
                "gate": [mp, mv]}
    prof = P.SPLITS["navtest_single_stage"]
    scratch_root = P.EXP_ROOT / "runs" / "epdms7" / label.replace("/", "_") / arm
    need = 1.5 if full else 0.5
    os.makedirs(str(scratch_root), exist_ok=True)
    if disk_free_gb(str(scratch_root)) < need:
        return {"status": "DISK_LOW", "retryable": True, "arm": arm, "scope": scope,
                "free_gb": disk_free_gb(str(scratch_root)), "need_gb": need}
    if not take_lock(scope, arm):
        return {"status": "ALREADY_RUNNING", "retryable": True, "arm": arm, "scope": scope}
    try:
        cid = cache_identity(prof)
        if not cid["ok"]:
            raise SystemExit(f"REFUSED (G3): metric cache identity {cid['mismatches']}")
        meta = load_meta()
        t2l = P.token_to_log(prof)
        pre = P.preflight(prof, tokens=tokens)
        expected = set(tokens) if tokens is not None else set(t2l)
        # ---- the plan source
        seam, official, seam_rec = None, None, None
        raw_dir = scratch_root / "raw"
        exp_dir = scratch_root / "exp"
        if arm in OFFICIAL_AGENT:
            official = OFFICIAL_AGENT[arm]
        elif arm == "STOP":
            seam = stop_seam(fwd(scratch_root / "seam_STOP.npz"), sorted(expected), meta)
            seam_rec = {"built_by": "taniteval.bench.navsim.seams.make_stop_seam", "path": fwd(seam),
                        "sha256": sha256_file(seam)}
        else:
            parent = seam_path(step, arm)
            if not os.path.exists(parent):
                return {"status": "SEAM_ABSENT", "retryable": False, "arm": arm, "scope": scope, "seam": parent}
            info = seam_info(parent)
            bad = seam_refusals(info, full=True)                              # the PARENT is always a full seam
            if bad:
                raise SystemExit(f"REFUSED (G4): {parent}: {bad}")
            if full:
                seam, seam_rec = parent, info
            else:
                seam_rec = subset_seam(parent, sorted(expected), fwd(scratch_root / f"seam_{arm}_subset.npz"))
                seam = seam_rec["path"]
                seam_rec["parent_info"] = info
        t0 = time.time()
        rep = SC.score_arm(arm=f"{label.replace('/', '_')}_{arm}", prof=prof, raw_dir=raw_dir, exp_dir=exp_dir,
                           seam=seam, official_agent=official, worker="sequential", ram_floor_mb=ram_floor_mb,
                           log=log, tokens=(sorted(tokens) if tokens is not None else None),
                           token_log=(t2l if tokens is not None else None))
        rec = {"schema": SCHEMA, "arm": arm, "scope": scope, "step": step, "tag": tag or None, "full_split": full,
               "utc": now(), "scoring_status": rep["status"], "wall_s": rep.get("wall_s"),
               "counts": rep, "mem_at_launch": ms, "addendum": ads, "cache": cid, "seam": seam_rec}
        if rep["status"] != "PASS":
            rec["status"] = rep["status"]
            rec["retryable"] = bool(rep.get("retryable"))
            tail = ""
            lp = raw_dir / f"{label.replace('/', '_')}_{arm}.score.log"
            if lp.exists():
                tail = sanitize_text(lp.read_text(encoding="utf-8", errors="replace")[-3000:])[0]
            rec["log_tail"] = tail
            write_json(os.path.join(out_dir, "ARM_FAILED.json"), rec)
            return rec
        csv_src = rep["csv"]
        pc = postcheck(csv_src, expected, full)
        # G4 / G3 AGAIN, after the run: the inputs did not move while the scorer was reading them
        if seam_rec and arm in MODEL_ARMS and full and sha256_file(seam) != seam_rec["sha256"]:
            pc["ok"] = False
            pc["failures"].append("G4 seam changed while it was being scored")
        cid2 = cache_identity(prof)
        if not cid2["ok"]:
            pc["ok"] = False
            pc["failures"].append(f"G3 cache identity changed during the run {cid2['mismatches']}")
        rec["postcheck"] = pc
        if not pc["ok"]:
            rec["status"] = "FAIL_POSTCHECK"
            rec["retryable"] = False
            write_json(os.path.join(out_dir, "ARM_FAILED.json"), rec)
            return rec
        # ---- bank the deliverables (the CSV UNMODIFIED; logs sanitised of thread-id UUIDs)
        os.makedirs(out_dir, exist_ok=True)
        csv_dst = os.path.join(out_dir, f"{arm}.devkit.csv")
        shutil.copyfile(csv_src, csv_dst)
        banked = {"csv": fwd(os.path.relpath(csv_dst, PKG)), "csv_sha256": sha256_file(csv_dst)}
        lbl = f"{label.replace('/', '_')}_{arm}"
        banked["score_log"] = bank_text_file(str(raw_dir / f"{lbl}.score.log"), os.path.join(out_dir, f"{arm}.score.log"))
        for src, dst in ((raw_dir / f"{lbl}_manifest.json", f"{arm}.wrapper_manifest.json"),
                         (raw_dir / f"{lbl}.counts.json", f"{arm}.counts.json")):
            if src.exists():
                banked[dst] = bank_text_file(str(src), os.path.join(out_dir, dst))
        man = {}
        try:
            man = json.load(open(raw_dir / f"{lbl}_manifest.json", encoding="utf-8"))
        except Exception:                                                   # noqa: BLE001
            pass
        rec.update({"status": "PASS" if full else "PASS_SUBSET", "retryable": False, "banked": banked,
                    "resources": man.get("resources"), "devkit": P.devkit_sha_measured(),
                    "tokens_sha256": pc["tokens_sha256"], "n_tokens": pc["n_tokens"],
                    "headline_x100": pc["headline_x100"], "wall_total_s": round(time.time() - t0, 1)})
        write_json(os.path.join(out_dir, "ARM_DONE.json"), rec)
        # ---- a FULL-split PASS frees its scratch (every deliverable is banked above; ~0.1 GB per arm on a drive
        # with ~2.5 GB free); a FAIL and a smoke keep theirs for diagnosis
        if full:
            shutil.rmtree(str(scratch_root), ignore_errors=True)
        if os.path.exists(os.path.join(out_dir, "ARM_FAILED.json")):
            os.remove(os.path.join(out_dir, "ARM_FAILED.json"))
        return rec
    finally:
        try:
            os.remove(lock_path(scope, arm))
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# 5. controls                                                                  #
# --------------------------------------------------------------------------- #
def compare_csv(a: str, b: str, tol: float = 1e-9) -> dict:
    """Cell-by-cell identity of two one-stage devkit CSVs on ``score`` and the nine components (NaN == NaN)."""
    from taniteval.bench.navsim import single_stage as SS
    ta, tb = SS.load_token_scores(a), SS.load_token_scores(b)
    cols = ["score"] + list(COMPONENTS.values())
    out = {"n_a": len(ta), "n_b": len(tb), "same_token_set": set(ta) == set(tb), "tol": tol, "max_abs": {}}
    worst = 0.0
    for c in cols:
        m = 0.0
        for t in set(ta) & set(tb):
            x, y = ta[t].get(c, float("nan")), tb[t].get(c, float("nan"))
            if math.isnan(x) and math.isnan(y):
                continue
            d = float("inf") if (math.isnan(x) != math.isnan(y)) else abs(x - y)
            m = max(m, d)
        out["max_abs"][c] = m
        worst = max(worst, m)
    out["max_abs_all"] = worst
    out["identical"] = bool(out["same_token_set"] and worst <= tol)
    return out


def ke7_dac(v2_csv: str, v1_csv: str) -> dict:
    """KE7 (REPORTED, not a gate): v2 DAC vs v1.1 DAC for the same arm and token. The human filter is one-sided
    (v2 >= v1), so `v2 < v1` would mean the two scorers did not see the same plan."""
    import pandas as pd
    a = pd.read_csv(v2_csv, usecols=["token", "drivable_area_compliance"])
    a = a[a.token != SUMMARY_ROW].set_index("token")["drivable_area_compliance"]
    b = pd.read_csv(v1_csv, usecols=["token", "drivable_area_compliance"])
    b = b[b.token != "average"].set_index("token")["drivable_area_compliance"]
    common = a.index.intersection(b.index)
    d = a.loc[common].to_numpy(float) - b.loc[common].to_numpy(float)
    return {"n_common": int(len(common)), "equal": int((abs(d) <= 1e-12).sum()),
            "v2_gt_v1": int((d > 1e-12).sum()), "v2_lt_v1": int((d < -1e-12).sum()), "expected_v2_lt_v1": 0}


def ke5_human(tok: dict) -> dict:
    bad = 0
    for r in tok.values():
        ones = all(abs(r[COMPONENTS[k]] - 1.0) <= 1e-12 for k in ("NC", "DAC", "TTC", "TLC", "LK"))
        ddc_ok = r[COMPONENTS["DDC"]] in (0.5, 1.0)
        bad += 0 if (ones and ddc_ok) else 1
    return {"n": len(tok), "exceptions": bad, "pass": bad == 0}


# --------------------------------------------------------------------------- #
# 6. summary                                                                   #
# --------------------------------------------------------------------------- #
def arm_dir(step, arm: str) -> str:
    return os.path.join(OUT, "floors" if arm in FLOORS else f"step{step}", arm)


def read_done(step, arm: str):
    p = os.path.join(arm_dir(step, arm), "ARM_DONE.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return d if d.get("status") == "PASS" and d.get("full_split") else None


def v1_csv_for(step, arm: str):
    if arm in FLOORS:
        return f"{W3_RAW}/{arm}_navtest/{arm}_navtest.csv"
    p = os.path.join(MS, f"step{step}", "scores_navtest", f"r7s{step}_{arm}", f"r7s{step}_{arm}.csv")
    return p if os.path.exists(p) else None


def band(v: float) -> str:
    for lo, hi in ((0, 2), (2, 5), (5, 8), (8, 12), (12, 999)):
        if lo <= v < hi:
            return f"{lo}-{hi} m/s" if hi < 999 else ">=12 m/s"
    return "unknown"


#: smoke runs whose tokens are WHOLE logs (so EC adjacency equals the full split's): arm -> (tag, step or None)
SMOKE_OF = {"R7_A1": ("ke3a", 5000), "STOP": ("ke2", None), "HUMAN": ("ke5_HUMAN", None), "CV": ("ke5_CV", None)}


def ke4_subset_vs_full(arm: str, step, tok_full: dict) -> dict:
    """KE4: the smoke's whole-log rows equal the same tokens' rows in the FULL-split run (|d| <= 1e-9)."""
    from taniteval.bench.navsim import single_stage as SS
    m = SMOKE_OF.get(arm)
    if m is None or (m[1] is not None and m[1] != step):
        return {"status": "UNAVAILABLE", "reason": "no whole-log smoke for this arm/step"}
    p = os.path.join(OUT, "smoke", m[0], arm, f"{arm}.devkit.csv")
    if not os.path.exists(p):
        return {"status": "UNAVAILABLE", "reason": f"{p} absent"}
    ts = SS.load_token_scores(p)
    worst, miss = 0.0, 0
    for t, r in ts.items():
        if t not in tok_full:
            miss += 1
            continue
        for c in ["score"] + list(COMPONENTS.values()):
            x, y = r.get(c, float("nan")), tok_full[t].get(c, float("nan"))
            if math.isnan(x) and math.isnan(y):
                continue
            worst = max(worst, float("inf") if math.isnan(x) != math.isnan(y) else abs(x - y))
    try:
        shown = fwd(os.path.relpath(p, PKG))
    except ValueError:                                                      # another drive (tests only)
        shown = fwd(p)
    return {"status": "OK", "smoke": shown, "n_tokens": len(ts), "tokens_missing_in_full": miss,
            "max_abs": worst, "pass": bool(miss == 0 and worst <= 1e-9)}


def cmd_summarize(a) -> int:                                                 # noqa: C901
    import numpy as np
    from taniteval.bench.navsim import profiles as P, single_stage as SS, summarize as S
    step = a.step
    ads = addendum_state()
    prof = P.SPLITS["navtest_single_stage"]
    ci = SS._ci()
    meta = load_meta()
    t2l = {t: m["log"] for t, m in meta.items()}
    refusals, arms, tok = [], {}, {}
    cid = cache_identity(prof)
    if not cid["ok"]:
        refusals.append(f"G3 cache identity {cid['mismatches']}")
    if not ads["ok"]:
        refusals.append("the addendum's sha256 is not the last recorded hash line")
    wanted = list(FLOOR_ORDER) + list(STEP_ORDER)
    for arm in wanted:
        d = read_done(step, arm)
        if d is None:
            arms[arm] = {"status": "NOT_SCORED"}
            continue
        csv_path = os.path.join(PKG, d["banked"]["csv"])
        bad = []
        if not os.path.exists(csv_path) or sha256_file(csv_path) != d["banked"]["csv_sha256"]:
            bad.append("banked CSV missing or its sha256 changed since the run")
        if arm in MODEL_ARMS:
            sp = seam_path(step, arm)
            if not os.path.exists(sp) or sha256_file(sp) != d["seam"]["sha256"]:
                bad.append("G4 seam changed since it was scored")
        if d.get("cache", {}).get("tokens_sha256") != PINS["tokens_sha256"] or d["cache"]["manifest_sha256"] != PINS["cache_manifest_sha256"]:
            bad.append("G3 recorded cache identity != pins")
        if bad:
            arms[arm] = {"status": "REFUSED", "why": bad}
            refusals.append(f"{arm}: {bad}")
            continue
        t = SS.load_token_scores(csv_path)
        pc = postcheck(csv_path, set(t2l), True)
        if not pc["ok"]:
            arms[arm] = {"status": "REFUSED", "why": pc["failures"]}
            refusals.append(f"{arm}: {pc['failures']}")
            continue
        tok[arm] = t
        hdr, raw = S.read_raw_rows(csv_path)
        head = S.headline_value(raw, hdr, row=SUMMARY_ROW)
        iv = SS.interval_single(ci, PROTOCOL, t, t2l, head, PINS["n_logs"])
        comp = {k: (float(np.nanmean([r[c] for r in t.values()])) if t else None) for k, c in COMPONENTS.items()}
        zero = {k: float(np.mean([r[COMPONENTS[k]] <= 1e-12 for r in t.values()])) for k in MULTIPLIERS}
        any_zero = float(np.mean([any(r[COMPONENTS[k]] <= 1e-12 for k in MULTIPLIERS) for r in t.values()]))
        sm = {}
        if arm in MODEL_ARMS:
            mp = seam_manifest_path(step, arm)
            m = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
            dev = m.get("device")
            pre_ = m.get("precision")
            sm = {"device": "/".join(dev) if isinstance(dev, list) else dev,
                  "precision": "/".join(pre_) if isinstance(pre_, list) else pre_,
                  "stop_fraction_4s_endpoint_lt_1m": seam_stop_fraction(seam_path(step, arm))}
        elif arm == "STOP":
            sm = {"stop_fraction_4s_endpoint_lt_1m": 1.0}
        v1 = v1_csv_for(step, arm)
        ctl = {"KE1_max_abs_gap": pc["KE1_max_abs_gap"], "C_AVG": pc["C_AVG"],
               "KE7_dac_v2_vs_v1": (ke7_dac(csv_path, v1) if v1 and os.path.exists(v1) else
                                    {"status": "UNAVAILABLE", "reason": f"no v1.1 CSV for {arm} at this step"})}
        ctl["KE4_subset_vs_full"] = ke4_subset_vs_full(arm, step, t)
        if arm == "HUMAN":
            ctl["KE5_human_analytic"] = ke5_human(t)
        arms[arm] = {"status": "OK", "epdms": head, "epdms_x100": round(100.0 * head, 4), "n": len(t),
                     "interval": iv, "components": comp, "multiplier_zero_rate": zero,
                     "any_multiplier_zero_rate": any_zero, "n_ec_nan_dropped_to_14": pc["n_ec_nan"],
                     "csv": d["banked"]["csv"], "csv_sha256": d["banked"]["csv_sha256"],
                     "wall_s": d.get("wall_s"), "resources": d.get("resources"), "controls": ctl, **sm}
    ok = {k: v for k, v in tok.items()}
    # ---- pairs
    pair_list = []
    for x, ys in (("R7_A1", ("STOP", "CV", "HUMAN", "PRIOR_ha0p", "R7_A1_s1", "R7_CEILDECL_d")),
                  ("PRIOR_ha0p", ("STOP",)), ("R7_A1_s1", ("STOP",)), ("STOP", ("CV",))):
        for y in ys:
            if x in ok and y in ok:
                pair_list.append((x, y))
    pairs = {}
    for x, y in pair_list:
        pairs[f"{x}__minus__{y}"] = SS.paired_single(ci, ok[x], ok[y], t2l, arms[x]["epdms"], arms[y]["epdms"])
    # ---- seed floor and classification
    floor = None
    if "R7_A1" in ok and "R7_A1_s1" in ok:
        floor = round(abs(arms["R7_A1"]["epdms_x100"] - arms["R7_A1_s1"]["epdms_x100"]), 4)
    classes = {}
    for k, v in pairs.items():
        x, y = k.split("__minus__")
        if y == "HUMAN":
            classes[k] = "CONTEXT (privileged reference; no classification word)"
        elif floor is None:
            classes[k] = "SEED FLOOR UNAVAILABLE (R7_A1 or R7_A1_s1 not scored)"
        else:
            d = 100.0 * v.get("headline_delta", float("nan"))
            iv = v.get("interval") or {}
            word = classify(d, iv, floor)
            lo = iv.get("lo")
            hi = iv.get("hi")
            classes[k] = {"word": word, "delta_x100": round(d, 4),
                          "interval_x100": [None if lo is None else round(100 * lo, 4),
                                            None if hi is None else round(100 * hi, 4)],
                          "two_x_seed_floor_x100": round(2 * floor, 4),
                          "separated_log_name": iv.get("separated_log_name"),
                          "separated_log_AND_drive_conjunction": iv.get("separated"),
                          "training_variance": "UNTESTED (H-ESTIM-SEED-1)"}
    # ---- per command / speed band (R7_A1 and the others against STOP)
    CMD = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")
    decomp = {}
    if "STOP" in ok:
        for arm in (x for x in ok if x != "STOP"):
            blk = {"by_command": {}, "by_speed_band": {}}
            for name, keyf in (("by_command", lambda t: CMD[meta[t]["cmd"]]),
                               ("by_speed_band", lambda t: band(meta[t]["v0"]))):
                groups = {}
                for t in ok[arm]:
                    groups.setdefault(keyf(t), []).append(t)
                for g, ts in sorted(groups.items()):
                    xa = np.array([ok[arm][t]["score"] for t in ts])
                    xs = np.array([ok["STOP"][t]["score"] for t in ts])
                    blk[name][g] = {"n": len(ts), "epdms_x100": round(100 * float(xa.mean()), 4),
                                    "STOP_x100": round(100 * float(xs.mean()), 4),
                                    "delta_vs_STOP_x100": round(100 * float((xa - xs).mean()), 4),
                                    "wtl_vs_STOP": [int((xa - xs > 1e-12).sum()), int((abs(xa - xs) <= 1e-12).sum()),
                                                    int((xa - xs < -1e-12).sum())]}
            decomp[arm] = blk
    # ---- P1 (addendum section 10)
    p1 = {"statement": "sign(EPDMS(R7_A1)-EPDMS(STOP)) == sign(PDMS(R7_A1)-PDMS(STOP))"}
    pdms_p = os.path.join(MS, f"step{step}", "summary_navtest.json")
    pdms_delta = None
    if os.path.exists(pdms_p):
        try:
            pdms_delta = json.load(open(pdms_p, encoding="utf-8"))["pairs"]["R7_A1__minus__STOP"]["delta_x100"]
        except Exception:                                                   # noqa: BLE001
            pdms_delta = None
    ep_delta = None
    if "R7_A1__minus__STOP" in pairs:
        ep_delta = round(100.0 * pairs["R7_A1__minus__STOP"]["headline_delta"], 4)
    p1.update({"pdms_delta_x100": pdms_delta, "pdms_source": fwd(pdms_p), "epdms_delta_x100": ep_delta})
    if pdms_delta is not None and ep_delta is not None:
        p1["holds"] = bool(np.sign(pdms_delta) == np.sign(ep_delta))
        if not p1["holds"]:
            ds = {k: round(arms["R7_A1"]["components"][k] - arms["STOP"]["components"][k], 6) for k in COMPONENTS}
            p1["flip_terms_component_delta_R7_A1_minus_STOP"] = ds
            p1["next_lever"] = ("the term(s) with the most negative weighted contribution above; the multiplier "
                                "zero-rates are in arms[*].multiplier_zero_rate")
    else:
        p1["holds"] = None
    status = "REFUSED" if refusals else ("COMPLETE" if all(arms[x]["status"] == "OK" for x in wanted) else "PARTIAL")
    summary = {"schema": SCHEMA, "status": status, "step": step, "written_utc": now(),
               "label": "RESULT (secondary read; no bar)", "protocol": PROTOCOL, "n_tokens": PINS["n_tokens"],
               "n_logs": PINS["n_logs"], "refusals": refusals, "addendum": ads, "cache": cid,
               "devkit": {"sha": PINS["devkit_sha"], "note": "autonomousvision/navsim@0a380a9 = post-#151; runner "
                          "run_pdm_score_one_stage; traffic_agents=non_reactive"},
               "stamps": {"tier": "NavSim open-loop benchmark (zero-shot, never closed loop)",
                          "launch_tree": "the speed ceiling does not reach the emitted plan (SPEC_REFCV7 26.1; SPEC A1)",
                          "evidence_class": "MEASURED (ours + artifact paths in arms[*].csv)",
                          "estimator": ci.ESTIMATOR + " / " + ci.PAIRED_ESTIMATOR,
                          "question_answered": ci.QUESTION_ANSWERED},
               "headline_column": "score (average_all_frames); pdm_score is forbidden",
               "arms": arms, "pairs": pairs, "seed_floor_x100": floor, "classification": classes,
               "decomposition": decomp, "P1": p1}
    outp = os.path.join(OUT, f"step{step}", "summary_navtest_epdms.json")
    write_json(outp, summary)
    lines = [f"navtest EPDMS (NAVSIM v2, one stage) -- step {step} -- {status}  (SECONDARY READ, no bar)"]
    for arm in wanted:
        v = arms[arm]
        if v["status"] == "OK":
            iv = v["interval"]
            lines.append(f"{arm:14s} EPDMS={v['epdms_x100']:8.4f}  CI95=[{100 * iv.get('lo', float('nan')):.4f}, "
                         f"{100 * iv.get('hi', float('nan')):.4f}]  n={v['n']}")
        else:
            lines.append(f"{arm:14s} {v['status']}")
    for k, v in classes.items():
        lines.append(f"{k:34s} {v if isinstance(v, str) else v['word'] + ' d=' + str(v['delta_x100']) + ' ci=' + str(v['interval_x100'])}")
    lines.append(f"seed floor (|A1-A1_s1|) x100 = {floor}; P1 holds = {p1.get('holds')}")
    with open(os.path.join(OUT, f"step{step}", "summary_navtest_epdms.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if status in ("COMPLETE", "PARTIAL") else 3


# --------------------------------------------------------------------------- #
# 7. CLI                                                                       #
# --------------------------------------------------------------------------- #
def cmd_plan(a) -> int:
    ads = addendum_state()
    rep = {"addendum": ads, "mem": mem_status(), "disk_free_gb": {d: round(disk_free_gb(d + "/"), 2) for d in ("C:", "D:")},
           "pins": PINS, "arms": {}}
    for step in (5000, 30000, 50400):
        rep["arms"][f"step{step}"] = {arm: os.path.exists(seam_path(step, arm)) for arm in MODEL_ARMS}
    rep["done"] = {f"floors/{x}": bool(read_done(None, x)) for x in FLOORS}
    for step in (5000, 30000, 50400):
        for arm in MODEL_ARMS:
            rep["done"][f"step{step}/{arm}"] = bool(read_done(step, arm))
    print(json.dumps(rep, indent=1, default=str))
    return 0 if ads["ok"] else 2


def cmd_score(a) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    tokens = None
    if a.tokens_file:
        from taniteval.bench.navsim import profiles as P
        tokens = P.read_tokens_file(a.tokens_file)
    rec = run_arm(arm=a.arm, step=a.step, tokens=tokens, tag=a.tag, ram_floor_mb=a.ram_floor_mb)
    keep = {k: rec.get(k) for k in ("status", "arm", "scope", "retryable", "headline_x100", "wall_s", "n_tokens")}
    print(json.dumps(keep, default=str))
    if rec.get("postcheck") and not rec["postcheck"]["ok"]:
        print(json.dumps(rec["postcheck"]["failures"]))
    return 0 if rec.get("status") in ("PASS", "PASS_SUBSET") else 1


def cmd_controls(a) -> int:
    """Recompute the SMOKE / known-value control record from the banked smoke artifacts (VALIDATION ONLY)."""
    from taniteval.bench.navsim import single_stage as SS
    sm = os.path.join(OUT, "smoke")
    ctl = os.path.join(OUT, "controls")
    rec = {"schema": SCHEMA + "/controls", "stamp": "VALIDATION ONLY -- smoke subsets, never a result", "utc": now(),
           "addendum": addendum_state(), "controls": {}}

    def done(tag, arm):
        return json.load(open(os.path.join(sm, tag, arm, "ARM_DONE.json"), encoding="utf-8"))
    ke2 = compare_csv(os.path.join(ctl, "W8_smoke218_STOP.csv"), os.path.join(sm, "ke2", "STOP", "STOP.devkit.csv"))
    d2 = done("ke2", "STOP")
    rec["controls"]["KE2_known_value_W8_STOP_218"] = {
        "what": "STOP scored through THIS driver on W8's 218 whole-log tokens against the FULL v2 cache == W8's banked per-token rows",
        "banked": "raw/navtest_epdms/controls/W8_smoke218_STOP.csv", "banked_sha256": sha256_file(os.path.join(ctl, "W8_smoke218_STOP.csv")),
        "banked_headline_literal": 0.579805532200227, "this_driver_headline": d2["postcheck"]["headline_score"],
        "compare": ke2, "pass": bool(ke2["identical"] and d2["postcheck"]["headline_score"] == 0.579805532200227)}
    ka, kb = done("ke3a", "R7_A1"), done("ke3b", "R7_A1")
    ke3 = compare_csv(os.path.join(sm, "ke3a", "R7_A1", "R7_A1.devkit.csv"), os.path.join(sm, "ke3b", "R7_A1", "R7_A1.devkit.csv"))
    rec["controls"]["KE3_determinism_R7_A1_step5000_43tokens"] = {
        "csv_sha256_a": ka["banked"]["csv_sha256"], "csv_sha256_b": kb["banked"]["csv_sha256"], "compare": ke3,
        "pass": bool(ke3["identical"] and ka["banked"]["csv_sha256"] == kb["banked"]["csv_sha256"])}
    hum = SS.load_token_scores(os.path.join(sm, "ke5_HUMAN", "HUMAN", "HUMAN.devkit.csv"))
    rec["controls"]["KE5_human_analytic_43tokens"] = ke5_human(hum)
    v1 = os.path.join(MS, "step5000", "scores_navtest", "r7s5000_R7_A1", "r7s5000_R7_A1.csv")
    rec["controls"]["KE7_plan_identity_DAC_v2_vs_v1.1"] = {
        "R7_A1_step5000": ke7_dac(os.path.join(sm, "ke3a", "R7_A1", "R7_A1.devkit.csv"), v1),
        "HUMAN": ke7_dac(os.path.join(sm, "ke5_HUMAN", "HUMAN", "HUMAN.devkit.csv"), f"{W3_RAW}/HUMAN_navtest/HUMAN_navtest.csv"),
        "CV": ke7_dac(os.path.join(sm, "ke5_CV", "CV", "CV.devkit.csv"), f"{W3_RAW}/CV_navtest/CV_navtest.csv")}
    rec["measured_cost"] = {}
    for tag, arm in (("ke3a", "R7_A1"), ("ke3b", "R7_A1"), ("ke5_HUMAN", "HUMAN"), ("ke5_CV", "CV"), ("ke2", "STOP")):
        d = done(tag, arm)
        rec["measured_cost"][f"{tag}/{arm}"] = {"n_tokens": d["n_tokens"], "scorer_wall_s": d["wall_s"],
                                                "peak_rss_mb": (d.get("resources") or {}).get("peak_rss_mb"),
                                                "min_system_available_mb": (d.get("resources") or {}).get("min_system_available_mb"),
                                                "headline_x100_VALIDATION_ONLY": d["headline_x100"]}
    rec["all_pass"] = bool(rec["controls"]["KE2_known_value_W8_STOP_218"]["pass"]
                           and rec["controls"]["KE3_determinism_R7_A1_step5000_43tokens"]["pass"]
                           and rec["controls"]["KE5_human_analytic_43tokens"]["pass"]
                           and all(v["v2_lt_v1"] == 0 for v in rec["controls"]["KE7_plan_identity_DAC_v2_vs_v1.1"].values()))
    write_json(os.path.join(ctl, "CONTROLS_SMOKE.json"), rec)
    print(json.dumps({k: v for k, v in rec.items() if k in ("all_pass",)}), flush=True)
    return 0 if rec["all_pass"] else 1


def cmd_compare(a) -> int:
    r = compare_csv(a.a, a.b, a.tol)
    print(json.dumps(r, indent=1))
    return 0 if r["identical"] else 1


def cmd_gate(a) -> int:
    ms = mem_status()
    print(json.dumps({"mem": ms, "full_gate_open_now": gate_ok(ms), "smoke_gate_open_now": gate_ok(ms, SMOKE_PHYS_GB, SMOKE_VIRT_GB)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("prep")
    sub.add_parser("gate")
    sub.add_parser("controls")
    s = sub.add_parser("score")
    s.add_argument("--arm", required=True)
    s.add_argument("--step", type=int, default=None)
    s.add_argument("--tokens-file", default=None)
    s.add_argument("--tag", default="")
    s.add_argument("--ram-floor-mb", type=float, default=4000.0)
    m = sub.add_parser("summarize")
    m.add_argument("--step", type=int, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "prep": cmd_prep, "gate": cmd_gate, "score": cmd_score,
            "summarize": cmd_summarize, "compare": cmd_compare, "controls": cmd_controls}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
