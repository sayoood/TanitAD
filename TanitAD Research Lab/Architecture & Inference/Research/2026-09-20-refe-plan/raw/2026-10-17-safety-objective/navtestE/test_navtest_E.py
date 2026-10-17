#!/usr/bin/env python3
"""Self-test of navtest_E.py (IMPL_PLAN step 9 of PREREG_SAFEOBJ): the navtest T1 read -> navtest_E.json (OUTCOME_BUILD.md section 7) + navtest_families.json.
Run (Git bash):  "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe" -I -X utf8 test_navtest_E.py        -> ZZTEST_PASS <n>      (no pod, no GPU, no network; at most 2 processes)

WHICH FIXTURE IS REAL AND WHICH IS SYNTHETIC (the "built, tested, unreachable" rule: a reader tested on a schema its author invented passes its test and reads nothing)
  REAL (read-only, from disk; MEASURED by earlier programme work)
    * the full-navtest harness tables (12,146 tokens, 136 logs): u400 (md5 b17c4fba...), deployed (2101bc7f...), P2-s0, P2-s1 -- under C:/fp/PK/raw (the free-point archive checkout)
      They are NOT SAFEOBJ states.  They stand in for the four states because they are real harness output on the real token/log structure.  Two pair-sets are built from them:
        ANCHOR  R0d = u400 - deployed  (equals RESULT_SFT4 / the free-point selftest KNOWN to 1e-12);   R1d = P2-s1 - P2-s0  (RESULT_P2REP.md RR2 column, 2 dp)
        P2REP   R0d = P2-s0 - u400     (RESULT_P2REP.md "banked" column, 2 dp);                          R1d = P2-s1 - u400  (RR1 column, 2 dp)
      The RESULT_P2REP.md numbers are typed in below with their source line: INHERITED numbers, compared against a MEASURED recomputation (agreement at 2 dp is the check).
    * the families6 JSONs of u400 and deployed (real navtest families6 output), for navtest_families.json
    * the free-point dry-run chunk (work/dry/out/chunk_000.npz: a REAL CPU forward of 2 tokens, logits BASE / C0A / R0B) and the seams the live free-point picker wrote from it (work/dry/seam_*.npz)
    * the banked C0 / R0 scorer-state files (rls1_adam_dl) and the sha256 of their tensor sets from the free-point EXPECTED_STATES.json
    * the REAL outcome reader outcome/safeobj_outcome.py (bar_E, constants), run in a clean subprocess, imported read-only
  SYNTHETIC (no producer exists yet for these, or the numbers must be exactly controlled)
    * the 1,000-token / 40-log tables with a built-in log-level effect (pairs positively correlated across logs, so the joint and the independent resample DIFFER); an ADOPT-sized effect
    * irep.json; the RL-val seed floor handed to the reader's bar_E as a bare number (0.05 / 100.0: the READER derives it from bar L's inputs, the producer never writes it);
      n_tokens / n_logs 12146 / 136 stamped on the synthetic contract that is fed to the reader (a FIXTURE OVERRIDE: the synthetic tables are 1,000 tokens / 40 logs)
    * the gate files (picks_report_E.json, EXPECTED_STATES_E.json, harness status files) for the gate tests and for the CLI end to end (stamped FIXTURE)
    * tiny torch state files for the expected-states error paths
  NO REAL SAFEOBJ NUMBER EXISTS: no R0d / R1d state exists yet.  Every verdict below is a fixture verdict.

The mutations rewrite the tool's SOURCE text (old -> new, each `old` occurring exactly once) and re-run the checks of the groups the mutation exists for; each must go RED AT the named checks.
"""
from __future__ import annotations

import concurrent.futures as cf
import copy
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TOOL_PATH = HERE / "navtest_E.py"
PY = sys.executable
O = "C:/Users/Admin/sobj_pkg/raw/2026-10-17-safety-objective"
PKB = "C:/fp/PK/raw/p2_r2_instrument/baseline"
REAL_CSV = {"u400": f"{PKB}/u400_navtest_full.csv", "deployed": f"{PKB}/deployed_navtest_full.csv",
            "p2s0": "C:/fp/PK/raw/p2_readout/p2_full/arm/refe_stage2_p2_full.csv", "p2s1": "C:/fp/PK/raw/p2_readout/p2s1_full/arm/refe_stage2_p2s1_full.csv"}
REAL_CSV_MD5 = {"u400": "b17c4fbac26d064f17ce790dca4d9d08", "deployed": "2101bc7f2084195b2536e5ee09c6be32"}
FAMS = {"u400": f"{PKB}/u400_navtest_full_families.json", "deployed": f"{PKB}/deployed_navtest_full_families.json"}
DRY = "C:/Users/Admin/freept/work/dry"
C0_STATES = "C:/Users/Admin/rls1_adam_dl/rls1-c0/scorer_states/scorer_states_u1000.pt"
R0_STATES = "C:/Users/Admin/rls1_adam_dl/rls1-r0/scorer_states/scorer_states_u1000.pt"
# sha256 of the tensor sets, from the free-point package's EXPECTED_STATES.json (computed by the free-point agent BEFORE its run)
EXPECT_HASH = {"C0A": "2dd96344f9d5f6ab0864ba067da52279d119791b4753986f3e30055a0dd6d692", "C0B": "60502b910ce2b872cd3dcc1abd7b59fb28d8a5e2b5d17d44e7caf45a073e0910",
               "R0A": "276b97065e6d9c9282d5da1ff7845455486e567797c59c45fd1f1abfe95712eb", "R0B": "ea019cb25ea37a6ebbc9bdf284a87c4718567ce6df0e247a2fbff895c286bf23"}
# the REAL outcome reader this tool was tested against (read-only; OUTCOME_BUILD.md section 1 states the same digest).  A change to the reader means this suite must be re-run, so the pin is a hard check.
READER_SHA256 = "e534e691d69f202b172a6057aa30a396e6b116d102b92a4502aa00f07e333faf"
FP_COPY_MD5 = {"fp_chain.py": "11ae185a9cb0206f02ca6cd3b0d157cc", "fp_checks.py": "b5e5b25786d15d0cd7d29508075789f8", "fp_forward.py": "82bf21514f919ee4eafab72690004e34",
               "fp_picks.py": "eb6d263707fc9d0972de8cfd10b306b0", "fp_read.py": "22328746bb5c8520c412b59e2f1ef9e6", "fp_score.py": "5ef408e5d1e585b122c883f94337282b",
               "fp_ship.py": "70b7738dbb10128d49d68dc28125d803"}

# ---- literal anchors (INHERITED numbers, each with its source; the test compares a MEASURED recomputation with them)
KNOWN_SFT4 = {"mean": 3.7908187056340563, "lo": 3.0470289517787528, "hi": 4.542726430876108}           # RESULT_SFT4 / P2_READOUT_RUNBOOK s.0: u400 - deployed, PDMS x100 (free-point fp_read.KNOWN)
KEYS = ("PDMS", "NC", "DAC", "EP", "TTC", "C", "DDC")
# RESULT_P2REP.md "The numbers" table, (mean, lo, hi) x100, 2 dp
P2REP_S0_MINUS_U400 = {"PDMS": (0.57, 0.02, 1.18), "NC": (-0.13, -0.31, 0.08), "DAC": (0.30, -0.23, 0.89), "EP": (1.14, 0.61, 1.70), "TTC": (-0.31, -0.80, 0.14), "C": (0.01, 0.00, 0.03), "DDC": (-0.16, -0.43, 0.13)}
P2REP_S1_MINUS_U400 = {"PDMS": (0.90, 0.36, 1.59), "NC": (-0.17, -0.36, 0.03), "DAC": (0.65, 0.11, 1.37), "EP": (1.80, 1.33, 2.41), "TTC": (-0.49, -0.97, -0.07), "C": (0.02, 0.00, 0.05), "DDC": (-0.07, -0.36, 0.23)}
P2REP_S1_MINUS_S0 = {"PDMS": (0.33, 0.04, 0.64), "NC": (-0.04, -0.16, 0.09), "DAC": (0.35, 0.09, 0.64), "EP": (0.65, 0.33, 1.01), "TTC": (-0.18, -0.47, 0.09), "C": (0.01, 0.00, 0.03), "DDC": (0.08, -0.07, 0.24)}
N_BOOT, SEED = 10000, 20260927
CONTRACT_ESTIMATOR = "paired log-cluster bootstrap, 10000 resamples, 95 %, seed 20260927"           # OUTCOME_BUILD.md section 7, typed independently of the tool
STATES = ("R0d_A", "R0d_B", "R1d_A", "R1d_B")
S7_TOP = {"tier", "estimator", "n_tokens", "n_logs", "pairs", "pooled", "floors"}               # OUTCOME_BUILD.md section 7, typed independently of the tool


# ============================================================ independent references (NOT the tool's code)
def _groups(tl, toks):
    by = {}
    for i, t in enumerate(toks):
        by.setdefault(tl[t], []).append(i)
    return [np.asarray(v) for _k, v in sorted(by.items())]


def ref_boot(d, tl, n=N_BOOT, seed=SEED):
    """paired log-cluster bootstrap of the mean of {token: value}: the estimator written out again (default_rng(seed); per draw ONE index set over the logs)"""
    toks = sorted(d)
    gs = _groups(tl, toks)
    v = np.asarray([d[t] for t in toks], dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = np.empty(n)
    for b in range(n):
        m[b] = v[np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))])].mean()
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ref_joint(d0, d1, tl, n=N_BOOT, seed=SEED):
    """the pooled contrast by the JOINT resample of addendum 1 R4: ONE index set per draw, applied to BOTH pairs; pooled = (mean over the draw of d0 + mean of d1) / 2"""
    toks = sorted(d0)
    gs = _groups(tl, toks)
    v0, v1 = (np.asarray([d[t] for t in toks], dtype=np.float64) for d in (d0, d1))
    rng = np.random.default_rng(seed)
    m = np.empty(n)
    for b in range(n):
        idx = np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))])
        m[b] = (v0[idx].mean() + v1[idx].mean()) / 2.0
    return float((v0.mean() + v1.mean()) / 2.0), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ref_indep(d0, d1, tl, n=N_BOOT, seed=SEED):
    """the WRONG alternative the test must tell apart: two INDEPENDENT bootstraps (separate generators), averaged draw by draw"""
    toks = sorted(d0)
    gs = _groups(tl, toks)
    v0, v1 = (np.asarray([d[t] for t in toks], dtype=np.float64) for d in (d0, d1))
    r0, r1 = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    m = np.empty(n)
    for b in range(n):
        a0 = v0[np.concatenate([gs[j] for j in r0.integers(0, len(gs), len(gs))])].mean()
        a1 = v1[np.concatenate([gs[j] for j in r1.integers(0, len(gs), len(gs))])].mean()
        m[b] = (a0 + a1) / 2.0
    return float((v0.mean() + v1.mean()) / 2.0), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def read_csv_x100(path):
    """token -> {PDMS, NC, DAC, EP, TTC, C, DDC} x100, read with the csv module (the harness columns), independent of stage3_full.load_table"""
    out = {}
    cols = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "C": "comfort", "DDC": "driving_direction_compliance"}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["token"] in ("average", ""):
            continue
        row = {"PDMS": 100.0 * float(r["score"])}
        row.update({k: 100.0 * float(r[c]) for k, c in cols.items()})
        out[r["token"]] = row
    return out


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================ the synthetic world
HDR = [",token,valid,no_at_fault_collisions,drivable_area_compliance,ego_progress,time_to_collision_within_bound,comfort,driving_direction_compliance,score"]


def write_harness_csv(path, toks, cols):
    """cols: dict of arrays in the x1 NAVSIM scale; a trailing `average` row as the harness writes it"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HDR[0] + "\n")
        for i, t in enumerate(toks):
            f.write(f"{i},{t},True,{cols['NC'][i]!r},{cols['DAC'][i]!r},{cols['EP'][i]!r},{cols['TTC'][i]!r},{cols['C'][i]!r},{cols['DDC'][i]!r},{cols['PDMS'][i]!r}\n")
        f.write(f"{len(toks)},average,True,0,0,0,0,0,0,0\n")


def pdms_x1(c):
    return c["NC"] * c["DAC"] * c["DDC"] * (5 * c["EP"] + 5 * c["TTC"] + 2 * c["C"]) / 12.0


def make_syn(td, effect0=0.050, effect1=0.040, seed=20261017):
    """40 logs x 25 tokens.  A tables are independent draws; B = A + a log-level effect on EP (pairs correlated 0.8 across logs) + small flips towards 1 on NC / DAC / TTC.  Nothing gets worse."""
    rng = np.random.default_rng(seed)
    L, K = 40, 25
    logs = [f"log{i:02d}" for i in range(L)]
    toks = [f"t{i:02d}_{j:02d}" for i in range(L) for j in range(K)]
    tl = {t: logs[i // K] for i, t in enumerate(toks)}
    N = len(toks)
    lid = np.repeat(np.arange(L), K)
    g, gp = rng.normal(size=L), rng.normal(size=L)
    eff = {"R0d": effect0 + 0.06 * g, "R1d": effect1 + 0.06 * (0.8 * g + 0.6 * gp)}

    def mk_a():
        c = {"NC": np.where(rng.random(N) > 0.04, 1.0, 0.5), "DAC": (rng.random(N) > 0.05).astype(float), "EP": np.clip(rng.beta(6, 2, size=N), 0, 1),
             "TTC": (rng.random(N) > 0.08).astype(float), "C": np.ones(N), "DDC": np.where(rng.random(N) > 0.01, 1.0, 0.5)}
        c["PDMS"] = pdms_x1(c)
        return c

    def mk_b(a, e):
        c = {k: v.copy() for k, v in a.items()}
        c["EP"] = np.clip(a["EP"] + e[lid] + rng.normal(0, 0.03, size=N), 0, 1)
        for k, pr in (("NC", 0.25), ("DAC", 0.10), ("TTC", 0.15), ("DDC", 0.5)):          # rare improvements: every sub-score has its own, small, nonzero delta
            fix = (a[k] < 1.0) & (rng.random(N) < pr)
            c[k] = np.where(fix, 1.0, a[k])
        c["PDMS"] = pdms_x1(c)
        return c

    tabs = {}
    for p in ("R0d", "R1d"):
        a = mk_a()
        tabs[f"{p}_A"], tabs[f"{p}_B"] = a, mk_b(a, eff[p])
    paths = {}
    for s, c in tabs.items():
        paths[s] = os.path.join(td, f"syn_{s}.csv")
        write_harness_csv(paths[s], toks, c)
    tokf = os.path.join(td, "syn_tokens.json")
    json.dump({"rule": "synthetic", "tokens": toks, "token_log": tl}, open(tokf, "w"))
    return {"paths": paths, "toks": toks, "tl": tl, "tokens_file": tokf, "cols": tabs}


def write_irep(td, name="irep.json", v=0.01, unit="PDMS x100"):
    p = os.path.join(td, name)
    json.dump({"irep_abs_delta": v, "unit": unit, "kind": "fixture"}, open(p, "w"))
    return p


def fake_state_info(tag="a"):
    return {s: {"file": "scorer_states_last.pt", "key": s[-1], "update": 1000, "sha256_tensors": hashlib.sha256(f"{tag}{s}".encode()).hexdigest()} for s in STATES}


def write_gate_dir(root, *, ic2=0.0, n_tokens=12146, swap_hashes=False, harness_fail=None, ic3=True, ic5=True, drop_expected=False, keys_wrong=False, fixture=True):
    os.makedirs(os.path.join(root, "score_logs"), exist_ok=True)
    info = fake_state_info()
    exp = {s: {"file": "scorer_states_u1000.pt", "key": s[-1], "update": 1000, "sha256_tensors": info[s]["sha256_tensors"]} for s in STATES}
    if keys_wrong:
        exp["R0d_A"]["key"] = "B"
    got = copy.deepcopy(info)
    if keys_wrong:
        got["R0d_A"]["key"] = "B"            # consistent end to end: the A/B binding gate (expected == got, hash and key) is SATISFIED; only keys_A_B_as_named can refuse this
    if swap_hashes:
        got["R0d_A"]["sha256_tensors"], got["R0d_B"]["sha256_tensors"] = info["R0d_B"]["sha256_tensors"], info["R0d_A"]["sha256_tensors"]
    rep = {"n_tokens": n_tokens, "IC2_max_abs": ic2, "IC3_BASE_vs_banked_u400": {"pass": ic3}, "ic5_forward": {"bit_identical": ic5}, "state_info": got, "FIXTURE": fixture}
    json.dump(rep, open(os.path.join(root, "picks_report_E.json"), "w"))
    if not drop_expected:
        json.dump(exp, open(os.path.join(root, "EXPECTED_STATES_E.json"), "w"))
    for s in STATES:
        json.dump({"status": "FAIL" if s == harness_fail else "PASS"}, open(os.path.join(root, "score_logs", f"refe_sobj_{s}.status.json"), "w"))
    return root


# ============================================================ the reader (REAL outcome/safeobj_outcome.py) in a clean subprocess
PROBE = r'''
import sys, json
sys.dont_write_bytecode = True
O = sys.argv[1]
for p in (O + "/outcome", O + "/deliverables/eval", O + "/deliverables/refe"):
    sys.path.insert(0, p)
import safeobj_outcome as SO
cmd = sys.argv[2]
if cmd == "bar_E":
    rl = None if len(sys.argv) < 5 or sys.argv[4] == "none" else {"value": float(sys.argv[4]), "terms": {"fixture": float(sys.argv[4])}}
    out = SO.bar_E(sys.argv[3], SO.Ctx(), rl, None if rl is not None else "no RL-val floor was handed to the probe")
elif cmd == "constants":
    out = {"N_BOOT": SO.N_BOOT, "SEED": SO.SEED, "SUBS": list(SO.SUBS), "CONSTRAINED": list(SO.CONSTRAINED), "E_EACH_MIN": SO.E_EACH_MIN, "E_MEAN_MIN": SO.E_MEAN_MIN,
           "E_REFUTED_UPPER": SO.E_REFUTED_UPPER, "E_FLOOR_MULT": SO.E_FLOOR_MULT, "NAVTEST_TOKENS": SO.NAVTEST_TOKENS, "NAVTEST_LOGS": SO.NAVTEST_LOGS, "E_POOL_TOL": SO.E_POOL_TOL,
           "file": SO.__file__}
print("ZZPROBE " + json.dumps(out))
'''


def reader(td, cmd, *args):
    pp = os.path.join(td, "reader_probe.py")
    if not os.path.exists(pp):
        open(pp, "w", encoding="utf-8").write(PROBE)
    r = subprocess.run([PY, "-I", "-X", "utf8", pp, O, cmd] + [str(a) for a in args], capture_output=True, text=True, encoding="utf-8", timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith("ZZPROBE ")), None)
    if line is None:
        raise AssertionError(f"reader probe produced no result (rc {r.returncode}): {r.stderr[-600:]}")
    return json.loads(line[len("ZZPROBE "):])


def reader_E(td, contract, name="c.json", rlval=0.05):
    """the REAL bar_E on `contract`; rlval = the RL-val seed floor bar L would hand it (None = clause 3 cannot be read)"""
    p = os.path.join(td, name)
    json.dump(contract, open(p, "w"))
    return reader(td, "bar_E", p, "none" if rlval is None else repr(float(rlval)))


def section7_block():
    """the JSON block of OUTCOME_BUILD.md section 7, parsed (the reader's author's statement of the contract)"""
    doc = open(f"{O}/outcome/OUTCOME_BUILD.md", encoding="utf-8").read()
    i = doc.index("`navtest_E.json` (IMPL step 9 must write exactly this")
    j = doc.index("```json", i) + 7
    return json.loads(doc[j: doc.index("```", j)])


# ============================================================ tool loading (mutation = source text rewrite) and the lazy world
def load_tool(src):
    mod = types.ModuleType("navtest_E_under_test")
    mod.__file__ = str(TOOL_PATH)
    exec(compile(src, str(TOOL_PATH), "exec"), mod.__dict__)
    return mod


class Ctx:
    """lazy, cached fixtures for ONE tool instance (a build that raises re-raises for every check that needs it)"""

    def __init__(self, T, td):
        self.T, self.td = T, td
        self._c = {}

    def memo(self, key, fn):
        if key not in self._c:
            try:
                self._c[key] = ("ok", fn())
            except BaseException as e:  # noqa: BLE001  (a check must see the failure, not the harness)
                self._c[key] = ("err", e)
        st, v = self._c[key]
        if st == "err":
            raise v
        return v

    # --- inputs
    @property
    def S3(self):
        return self.memo("S3", self.T.get_s3)

    @property
    def export_logs(self):
        return self.memo("export", lambda: {t: v["log_name"] for t, v in json.load(__import__("gzip").open(self.T.EXPORT, "rt", encoding="utf-8"))["tokens"].items()})

    def real_tl(self):
        return {t: self.export_logs[t] for t in json.load(open(self.T.TOKENS_FILE, encoding="utf-8"))["tokens"]}

    @property
    def p2rep_tables(self):
        return {"R0d_A": REAL_CSV["u400"], "R0d_B": REAL_CSV["p2s0"], "R1d_A": REAL_CSV["u400"], "R1d_B": REAL_CSV["p2s1"]}

    @property
    def anchor_tables(self):
        return {"R0d_A": REAL_CSV["deployed"], "R0d_B": REAL_CSV["u400"], "R1d_A": REAL_CSV["p2s0"], "R1d_B": REAL_CSV["p2s1"]}

    @property
    def irep(self):
        return self.memo("irep", lambda: write_irep(self.td))

    @property
    def syn(self):
        return self.memo("syn", lambda: make_syn(self.td))

    # --- builds (the tool's own build_E)
    def real_build(self):
        return self.memo("real_build", lambda: self.T.build_E(self.p2rep_tables, tokens_file=self.T.TOKENS_FILE, export_logs=self.export_logs, irep_path=self.irep))

    def syn_build(self):
        s = self.syn
        return self.memo("syn_build", lambda: self.T.build_E(s["paths"], tokens_file=s["tokens_file"], export_logs=s["tl"], pins=None, irep_path=self.irep))

    def anchor_pairs(self):
        def f():
            tabs = self.T.load_tables(self.S3, self.anchor_tables)
            toks = sorted(json.load(open(self.T.TOKENS_FILE, encoding="utf-8"))["tokens"])
            tl = {t: self.export_logs[t] for t in toks}
            return self.T.contrasts(self.S3, tabs, toks, tl, N_BOOT, SEED)
        return self.memo("anchor_pairs", f)


CHECKS = {}                                     # name -> (group, fn)


def check(name, group):
    def deco(fn):
        assert name not in CHECKS, name
        CHECKS[name] = (group, fn)
        return fn
    return deco


def refused(fn, needle=None):
    """True iff fn() raises the tool's Refused (and the message contains `needle`)"""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__ == "Refused" and (needle is None or needle in str(e))
    return False


def close2(row, want):
    """row = {'mean', 'ci95'}; want = (mean, lo, hi) at 2 dp"""
    got = (round(row["mean"], 2), round(row["ci95"][0], 2), round(row["ci95"][1], 2))
    return got == tuple(want), got


def tables_match(pairobj, want):
    bad = []
    for k in KEYS:
        ok, got = close2(pairobj[k], want[k])
        if not ok:
            bad.append(f"{k}: got {got} want {tuple(want[k])}")
    assert not bad, "; ".join(bad)


# ---------------------------------------------------------------- real fixture: build, anchors, contract, pooled, floors, reader
@check("real:build_runs", "real")
def _(c):
    contract, prov, ex = c.real_build()
    assert prov["token_gate"]["ok"] and prov["token_gate"]["n"] == 12146 and prov["token_gate"]["n_logs"] == 136 and prov["pinned_tokens"]


@check("real:anchor_u400_minus_deployed_equals_RESULT_SFT4_1e-12", "anchor")
def _(c):
    r = c.anchor_pairs()["R0d"]["primary"]
    dev = max(abs(r["mean"] - KNOWN_SFT4["mean"]), abs(r["ci95"][0] - KNOWN_SFT4["lo"]), abs(r["ci95"][1] - KNOWN_SFT4["hi"]))
    assert dev <= 1e-12, f"u400 - deployed deviates from RESULT_SFT4 by {dev}: {r['mean']} {r['ci95']}"


@check("real:anchor_P2s1_minus_P2s0_equals_RESULT_P2REP_RR2_2dp", "anchor")
def _(c):
    tables_match(c.T.pair_object(c.anchor_pairs()["R1d"]), P2REP_S1_MINUS_S0)


@check("real:R0d_matches_RESULT_P2REP_P2s0_minus_u400_2dp", "real")
def _(c):
    tables_match(c.real_build()[0]["pairs"]["R0d"], P2REP_S0_MINUS_U400)


@check("real:R1d_matches_RESULT_P2REP_RR1_P2s1_minus_u400_2dp", "real")
def _(c):
    tables_match(c.real_build()[0]["pairs"]["R1d"], P2REP_S1_MINUS_U400)


@check("real:contract_exact_keys_types_order", "real")
def _(c):
    k = c.real_build()[0]
    assert set(k) == S7_TOP, sorted(k)
    assert k["tier"] == "T1" and k["estimator"] == CONTRACT_ESTIMATOR
    assert type(k["n_tokens"]) is int and k["n_tokens"] == 12146 and type(k["n_logs"]) is int and k["n_logs"] == 136, (k["n_tokens"], k["n_logs"])
    assert list(k["pairs"]) == ["R0d", "R1d"]
    for p in ("R0d", "R1d"):
        assert set(k["pairs"][p]) == set(KEYS), sorted(k["pairs"][p])
        for key in KEYS:
            o = k["pairs"][p][key]
            assert set(o) == {"mean", "ci95"} and type(o["mean"]) is float and len(o["ci95"]) == 2 and all(type(x) is float for x in o["ci95"]), (p, key)
    assert set(k["pooled"]) == {"PDMS", "resample"} and k["pooled"]["resample"] == "joint", k["pooled"]
    assert set(k["floors"]) == {"irep_abs_delta"} and type(k["floors"]["irep_abs_delta"]) is float and "seed_floor_abs" not in json.dumps(k), k["floors"]


@check("contract:ci_order_in_emitted_file", "real")
def _(c):
    k = c.real_build()[0]
    objs = [k["pairs"][p][key] for p in k["pairs"] for key in k["pairs"][p]] + [k["pooled"]["PDMS"]]
    assert all(o["ci95"][0] <= o["ci95"][1] for o in objs) and len(objs) == 15


@check("contract:subscores_all_seven_in_emitted_file", "real")
def _(c):
    k = c.real_build()[0]
    for p in ("R0d", "R1d"):
        assert sorted(k["pairs"][p]) == sorted(KEYS), f"{p}: {sorted(k['pairs'][p])} (EP is not optional: clause 5)"


@check("real:pooled_equals_joint_reference_1e-9", "real")
def _(c):
    k, _p, ex = c.real_build()
    tl, toks = ex["tl"], ex["toks"]
    tabs = {s: read_csv_x100(p) for s, p in c.p2rep_tables.items() if s in ("R0d_B", "R1d_B")}
    ua = read_csv_x100(REAL_CSV["u400"])
    d0 = {t: tabs["R0d_B"][t]["PDMS"] - ua[t]["PDMS"] for t in toks}
    d1 = {t: tabs["R1d_B"][t]["PDMS"] - ua[t]["PDMS"] for t in toks}
    ref = ref_joint(d0, d1, tl)
    got = k["pooled"]["PDMS"]
    assert abs(got["mean"] - ref[0]) <= 1e-9 and abs(got["ci95"][0] - ref[1]) <= 1e-9 and abs(got["ci95"][1] - ref[2]) <= 1e-9, f"got {got} ref {ref}"


@check("real:pooled_mean_is_mean_of_the_two_pair_means", "real")
def _(c):
    k = c.real_build()[0]
    assert abs(k["pooled"]["PDMS"]["mean"] - (k["pairs"]["R0d"]["PDMS"]["mean"] + k["pairs"]["R1d"]["PDMS"]["mean"]) / 2.0) <= 1e-12


@check("real:accepted_by_the_REAL_outcome_reader", "real")
def _(c):
    k = c.real_build()[0]
    r = reader_E(c.td, k)                                                       # RL-val seed floor handed in: 0.05 (a fixture number; the reader derives the real one from bar L)
    assert r["verdict"] in ("ADOPT", "NOT_PROVEN", "REFUTED") and r.get("problems") == [], r
    cl = r["clauses"]
    # this fixture: P2-s1's TTC upper bound is -0.07 < 0 (RESULT_P2REP: NOT REPLICATED on an adverse TTC) -> clause 4 False; everything else the reader recomputes from OUR file
    assert cl["4_ttc_nc_upper_ge_0_each_pair"] is False and cl["1_each_ge_0.25_and_mean_ge_0.5"] is True and cl["2_pooled_lower_gt_0"] is True and cl["3_mean_contrast_gt_2x_max_floor"] is True, cl
    n = r["numbers"]
    assert abs(n["mean_contrast"]["value"] - (k["pairs"]["R0d"]["PDMS"]["mean"] + k["pairs"]["R1d"]["PDMS"]["mean"]) / 2.0) <= 1e-12
    assert n["pooled_pdms_lower"]["value"] == k["pooled"]["PDMS"]["ci95"][0] and n["pooled_pdms_upper"]["value"] == k["pooled"]["PDMS"]["ci95"][1]
    assert n["irep_abs_delta"]["value"] == k["floors"]["irep_abs_delta"]


@check("floors:file_carries_only_irep_and_the_sidecar_seed_floor_equals_the_readers", "real")
def _(c):
    k, prov, _ = c.real_build()
    assert k["floors"] == {"irep_abs_delta": 0.01} and "seed_floor_abs" not in json.dumps(k), k["floors"]
    r = reader_E(c.td, k, "fl.json")
    want = r["numbers"]["navtest_seed_floor"]["value"]                          # the READER's derivation: |PDMS(R0d) - PDMS(R1d)|
    got = prov["floors_components"]["navtest_seed_floor_abs_R0d_vs_R1d"]        # the producer's sidecar record of the same quantity
    assert abs(got - want) <= 1e-12 and want > 0.3, (got, want)
    assert abs(r["floor"]["max_of_all"] - max(want, 0.01, 0.05)) <= 1e-12 and r["floor"]["rlval_seed_floor"]["value"] == 0.05


@check("floors:irep_required_refuses_missing", "light")
def _(c):
    T = c.T
    assert refused(lambda: T.read_irep(None), "NO default") and refused(lambda: T.read_irep(os.path.join(c.td, "nope.json")), "NO default")


@check("floors:irep_unit_slip_refused", "light")
def _(c):
    T = c.T
    assert refused(lambda: T.read_irep(write_irep(c.td, "irep_x1.json", 0.0001, "PDMS x1")), "unit")


@check("floors:irep_negative_or_nonfinite_refused", "light")
def _(c):
    T = c.T
    assert refused(lambda: T.read_irep(write_irep(c.td, "irep_neg.json", -0.1))) and refused(lambda: T.read_irep(write_irep(c.td, "irep_nan.json", float("nan"))))
    assert T.read_irep(write_irep(c.td, "irep_zero.json", 0.0))["value"] == 0.0


@check("floors:irep_from_tables_anchor", "light")
def _(c):
    r = c.T.irep_from_tables(REAL_CSV["u400"], REAL_CSV["p2s0"])
    assert r["unit"] == "PDMS x100" and r["n_tokens"] == 12146 and abs(r["irep_abs_delta"] - (88.8184 - 88.2469)) < 2e-4, r         # RESULT_P2REP: PDMS means u400 88.2469, P2-s0 88.8184


# ---------------------------------------------------------------- synthetic fixture: labels, sign, sub-scores, pooled, units, degenerate, pins
def syn_refs(c):
    s = c.syn
    cols = s["cols"]
    d = {}
    for p in ("R0d", "R1d"):
        d[p] = {key: {t: 100.0 * (cols[f"{p}_B"][key][i] - cols[f"{p}_A"][key][i]) for i, t in enumerate(s["toks"])} for key in KEYS}
    return d


@check("syn:build_runs", "syn")
def _(c):
    k, prov, ex = c.syn_build()
    assert prov["pinned_tokens"] is False and len(ex["toks"]) == 1000 and prov["status"].startswith("UNGATED")


@check("syn:pair_labels_and_sign_B_minus_A", "syn")
def _(c):
    k = c.syn_build()[0]
    d = syn_refs(c)
    tl = c.syn["tl"]
    r0, r1 = ref_boot(d["R0d"]["PDMS"], tl), ref_boot(d["R1d"]["PDMS"], tl)
    assert r0[0] > 0.5 and r1[0] > 0.3 and abs(r0[0] - r1[0]) > 0.15, "fixture must separate the labels and have a positive sign"
    for p, r in (("R0d", r0), ("R1d", r1)):
        o = k["pairs"][p]["PDMS"]
        assert abs(o["mean"] - r[0]) <= 1e-9 and abs(o["ci95"][0] - r[1]) <= 1e-9 and abs(o["ci95"][1] - r[2]) <= 1e-9, (p, o, r)


@check("syn:all_seven_subscores_match_reference", "syn")
def _(c):
    k = c.syn_build()[0]
    d = syn_refs(c)
    tl = c.syn["tl"]
    for p in ("R0d", "R1d"):
        for key in KEYS:
            r = ref_boot(d[p][key], tl)
            o = k["pairs"][p][key]
            assert abs(o["mean"] - r[0]) <= 1e-9 and abs(o["ci95"][0] - r[1]) <= 1e-9 and abs(o["ci95"][1] - r[2]) <= 1e-9, (p, key, o, r)


@check("syn:pooled_equals_joint_reference_1e-9", "syn")
def _(c):
    k = c.syn_build()[0]
    d = syn_refs(c)
    ref = ref_joint(d["R0d"]["PDMS"], d["R1d"]["PDMS"], c.syn["tl"])
    g = k["pooled"]["PDMS"]
    assert abs(g["mean"] - ref[0]) <= 1e-9 and abs(g["ci95"][0] - ref[1]) <= 1e-9 and abs(g["ci95"][1] - ref[2]) <= 1e-9, (g, ref)


@check("syn:fixture_discriminates_joint_from_independent", "syn")
def _(c):
    d = syn_refs(c)
    j, i = ref_joint(d["R0d"]["PDMS"], d["R1d"]["PDMS"], c.syn["tl"]), ref_indep(d["R0d"]["PDMS"], d["R1d"]["PDMS"], c.syn["tl"])
    wj, wi = j[2] - j[1], i[2] - i[1]
    assert abs(wj - wi) / wj > 0.10 and abs(j[1] - i[1]) > 0.05 and abs(j[2] - i[2]) > 0.05, f"the fixture cannot tell the joint resample from independent ones: widths joint {wj:.4f} independent {wi:.4f}"


@check("syn:pooled_is_not_two_independent_bootstraps", "syn")
def _(c):
    k = c.syn_build()[0]
    d = syn_refs(c)
    i = ref_indep(d["R0d"]["PDMS"], d["R1d"]["PDMS"], c.syn["tl"])
    g = k["pooled"]["PDMS"]["ci95"]
    assert abs(g[0] - i[1]) > 0.05 and abs(g[1] - i[2]) > 0.05, "the pooled interval equals the independent-draws interval: it was not a joint resample"


@check("syn:reader_reaches_ADOPT_on_a_strong_synthetic_effect", "syn")
def _(c):
    k = good_contract(c)
    r = reader_E(c.td, k, "syn_c.json")
    assert r["verdict"] == "ADOPT" and r["problems"] == [] and all(v for kk, v in r["clauses"].items()), r


@check("reader:clause3_is_unread_without_the_rlval_floor_and_decisive_with_it", "syn")
def _(c):
    k = good_contract(c)
    c3 = "3_mean_contrast_gt_2x_max_floor"
    r0 = reader_E(c.td, k, "c3_none.json", rlval=None)
    assert r0["verdict"] == "NOT_READ" and r0["clauses"][c3] is None and any("clause 3 is unread" in p for p in r0["problems"]), r0
    r1 = reader_E(c.td, k, "c3_small.json", rlval=0.05)
    assert r1["verdict"] == "ADOPT" and r1["clauses"][c3] is True, r1
    r2 = reader_E(c.td, k, "c3_big.json", rlval=100.0)
    assert r2["verdict"] == "NOT_PROVEN" and r2["clauses"][c3] is False and r2["clauses"]["2_pooled_lower_gt_0"] is True, r2


@check("contract:a_subset_file_carries_subset_counts_and_the_reader_refuses_it", "syn")
def _(c):
    sc = c.syn_build()[0]
    assert sc["n_tokens"] == 1000 and sc["n_logs"] == 40, (sc["n_tokens"], sc["n_logs"])
    c.T.validate_contract(sc, full=False)
    assert refused(lambda: c.T.validate_contract(sc), "n_tokens must be 12146")
    r = reader_E(c.td, sc, "subset.json")
    assert r["verdict"] == "NOT_READ" and any("n_tokens / n_logs" in p for p in r["problems"]), r


@check("contract:emitted_structure_equals_the_section7_block", "syn")
def _(c):
    k, blk = c.syn_build()[0], section7_block()
    assert set(k) == set(blk) == S7_TOP and k["tier"] == blk["tier"] and k["estimator"] == blk["estimator"]
    assert set(k["pairs"]) == set(blk["pairs"]) and set(k["pairs"]["R0d"]) == set(blk["pairs"]["R0d"]) and set(k["pairs"]["R0d"]["PDMS"]) == set(blk["pairs"]["R0d"]["PDMS"])
    assert set(k["pooled"]) == set(blk["pooled"]) and k["pooled"]["resample"] == blk["pooled"]["resample"] and set(k["pooled"]["PDMS"]) == set(blk["pooled"]["PDMS"])
    assert set(k["floors"]) == set(blk["floors"])


def _x1(tab):
    return {t: {k: v / 100.0 for k, v in row.items()} for t, row in tab.items()}


@check("unit:refuses_x1_tables", "light")
def _(c):
    T = c.T
    tab = c.memo("u400_tab", lambda: read_csv_x100(REAL_CSV["u400"]))
    assert T.unit_guard("ok", tab)["pdms_mean_x100"] > 80
    assert refused(lambda: T.unit_guard("x1", _x1(tab)), "UNIT SLIP"), "a x1-scale table must be refused"
    # end to end: a x1 table fed through build_E is refused, not read
    s = c.syn
    real = c.S3

    class Shim:
        def __getattr__(self, n):
            return getattr(real, n)

        def load_table(self, p):
            return _x1(real.load_table(p))
    old = c.T._S3
    c.T._S3 = Shim()
    try:
        assert refused(lambda: c.T.build_E(s["paths"], tokens_file=s["tokens_file"], export_logs=s["tl"], pins=None, irep_path=c.irep), "UNIT SLIP")
    finally:
        c.T._S3 = old


@check("unit:refuses_scale_above_100", "light")
def _(c):
    tab = c.memo("u400_tab", lambda: read_csv_x100(REAL_CSV["u400"]))
    big = {t: dict(row, PDMS=row["PDMS"] * 100.0) for t, row in tab.items()}          # only PDMS is on the wrong scale: the sub-score bound cannot be what refuses it
    assert refused(lambda: c.T.unit_guard("big", big), "PDMS outside [0, 100]")


@check("unit:levels_recorded_x100_in_sidecar", "syn")
def _(c):
    prov = c.syn_build()[1]
    assert all(30 < v["level_x100"]["pdms_mean_x100"] <= 100 for v in prov["tables"].values())


@check("tokens:refuses_missing_token_rows", "syn")
def _(c):
    s = c.syn
    p = dict(s["paths"])
    broken = os.path.join(c.td, "syn_broken.csv")
    lines = open(p["R1d_B"], encoding="utf-8").read().splitlines()
    open(broken, "w", encoding="utf-8").write("\n".join(lines[:5] + lines[6:]) + "\n")           # drop one token row
    p["R1d_B"] = broken
    assert refused(lambda: c.T.build_E(p, tokens_file=s["tokens_file"], export_logs=s["tl"], pins=None, irep_path=c.irep), "no row")


@check("degenerate:refuses_B_equals_A", "syn")
def _(c):
    s = c.syn
    p = dict(s["paths"])
    p["R0d_B"] = p["R0d_A"]
    assert refused(lambda: c.T.build_E(p, tokens_file=s["tokens_file"], export_logs=s["tl"], pins=None, irep_path=c.irep), "identically 0")


@check("degenerate:refuses_two_identical_pairs", "syn")
def _(c):
    s = c.syn
    p = dict(s["paths"])
    p["R1d_A"], p["R1d_B"] = p["R0d_A"], p["R0d_B"]
    assert refused(lambda: c.T.build_E(p, tokens_file=s["tokens_file"], export_logs=s["tl"], pins=None, irep_path=c.irep), "one pair counted twice")


@check("pins:refuses_an_unpinned_token_set_on_the_pinned_path", "syn")
def _(c):
    s = c.syn
    assert refused(lambda: c.T.build_E(s["paths"], tokens_file=s["tokens_file"], export_logs=s["tl"], pins=c.T.FULL_PINS, irep_path=c.irep), "pinned navtest set")


@check("pins:refuses_a_nondefault_estimator_on_the_pinned_path", "syn")
def _(c):
    assert refused(lambda: c.T.build_E(c.anchor_tables, tokens_file=c.T.TOKENS_FILE, export_logs=c.export_logs, pins=c.T.FULL_PINS, irep_path=c.irep, n_boot=500), "10,000")


@check("pins:instrument_md5_and_estimator_constants", "light")
def _(c):
    T = c.T
    assert T.INSTRUMENT_MD5 == {"stage3_full.py": "a501ef4eac66f58b9e6115f61ef48ba5", "proxy_eval.py": "d177f0c9ab493390cb2b5bb8811c7cd3"}
    assert T.instrument_pins()["ok"] and (T.N_BOOT, T.SEED) == (N_BOOT, SEED) and T.ESTIMATOR == CONTRACT_ESTIMATOR and T.FULL_PINS == {"md5": "3a8b3653fd70891e50a230a37eb76ae8", "n": 12146, "logs": 136}
    assert md5_file(T.TOKENS_FILE) == T.FULL_PINS["md5"]


# ---------------------------------------------------------------- the contract validator (hand-broken files) and the REAL reader as the oracle for what is broken
def good_contract(c):
    """the synthetic contract with the REAL navtest counts stamped on it: a FIXTURE OVERRIDE (the synthetic tables are 1,000 tokens / 40 logs; the validator and the reader both pin 12,146 / 136)"""
    def f():
        k = copy.deepcopy(c.syn_build()[0])
        k["n_tokens"], k["n_logs"] = 12146, 136
        return k
    return c.memo("good_contract", f)


def broken_variants(k):
    v = {}
    x = copy.deepcopy(k)
    x["pairs"]["R0d"]["TTC"]["ci95"] = x["pairs"]["R0d"]["TTC"]["ci95"][::-1] if x["pairs"]["R0d"]["TTC"]["ci95"][0] != x["pairs"]["R0d"]["TTC"]["ci95"][1] else [1.0, -1.0]
    v["ci_order"] = x
    x = copy.deepcopy(k); del x["pairs"]["R1d"]["EP"]; v["missing_EP"] = x
    x = copy.deepcopy(k); x["tier"] = "T0"; v["tier"] = x
    x = copy.deepcopy(k); x["estimator"] = "paired log-cluster bootstrap, 2000 resamples, 95 %, seed 20260927"; v["estimator"] = x
    x = copy.deepcopy(k); x["extra"] = 1; v["extra_key"] = x
    x = copy.deepcopy(k); del x["pooled"]["PDMS"]; v["pooled_missing"] = x
    x = copy.deepcopy(k); x["floors"]["irep_abs_delta"] = -0.1; v["floor_negative"] = x
    x = copy.deepcopy(k); x["pairs"]["R0d"]["PDMS"]["mean"] = float("nan"); v["nan"] = x
    x = copy.deepcopy(k); x["pairs"]["R2d"] = x["pairs"]["R0d"]; v["extra_pair"] = x
    x = copy.deepcopy(k); x["pooled"]["resample"] = "independent"; v["resample_not_joint"] = x
    x = copy.deepcopy(k); del x["pooled"]["resample"]; v["resample_missing"] = x
    x = copy.deepcopy(k); x["pooled"]["PDMS"]["mean"] += 0.5; x["pooled"]["PDMS"]["ci95"] = [a + 0.5 for a in x["pooled"]["PDMS"]["ci95"]]; v["pooled_mean_mismatch"] = x
    x = copy.deepcopy(k); x["n_tokens"] = 12000; v["n_tokens_wrong"] = x
    x = copy.deepcopy(k); x["n_logs"] = 135; v["n_logs_wrong"] = x
    x = copy.deepcopy(k); del x["n_tokens"]; v["n_tokens_missing"] = x
    x = copy.deepcopy(k); x["floors"]["seed_floor_abs"] = 0.4; v["seed_floor_leak"] = x
    x = copy.deepcopy(k); del x["floors"]["irep_abs_delta"]; v["irep_missing"] = x
    return v


@check("contract:validator_accepts_good", "light")
def _(c):
    c.T.validate_contract(good_contract(c))


@check("contract:validator_refuses_ci_order", "light")
def _(c):
    assert refused(lambda: c.T.validate_contract(broken_variants(good_contract(c))["ci_order"]), "ci95")


@check("contract:validator_refuses_missing_subscore", "light")
def _(c):
    assert refused(lambda: c.T.validate_contract(broken_variants(good_contract(c))["missing_EP"]), "EP")


@check("contract:validator_refuses_tier_estimator_extra_pooled_floors_nan_extra_pair", "light")
def _(c):
    v = broken_variants(good_contract(c))
    for name in ("tier", "estimator", "extra_key", "pooled_missing", "floor_negative", "nan", "extra_pair", "irep_missing"):
        assert refused(lambda n=name: c.T.validate_contract(v[n])), name


@check("contract:validator_refuses_resample_not_joint", "light")
def _(c):
    v = broken_variants(good_contract(c))
    assert refused(lambda: c.T.validate_contract(v["resample_not_joint"]), "resample") and refused(lambda: c.T.validate_contract(v["resample_missing"]), "resample")


@check("contract:validator_refuses_pooled_mean_that_is_not_the_mean_of_the_pairs", "light")
def _(c):
    assert refused(lambda: c.T.validate_contract(broken_variants(good_contract(c))["pooled_mean_mismatch"]), "mean of the two pairs")


@check("contract:validator_pins_n_tokens_and_n_logs", "light")
def _(c):
    v = broken_variants(good_contract(c))
    assert refused(lambda: c.T.validate_contract(v["n_tokens_wrong"]), "n_tokens must be 12146") and refused(lambda: c.T.validate_contract(v["n_logs_wrong"]), "n_logs must be 136")
    assert refused(lambda: c.T.validate_contract(v["n_tokens_missing"]), "exactly")


@check("contract:validator_refuses_a_seed_floor_in_the_file_and_a_missing_irep", "light")
def _(c):
    v = broken_variants(good_contract(c))
    assert refused(lambda: c.T.validate_contract(v["seed_floor_leak"]), "floors must be exactly") and refused(lambda: c.T.validate_contract(v["irep_missing"]), "floors must be exactly")


@check("reader:REAL_outcome_reader_refuses_every_broken_variant_for_its_reason", "light")
def _(c):
    v = broken_variants(good_contract(c))
    # name -> a fragment of the reader's own reason: the variant must be refused FOR THAT, not for some accident of the fixture
    why = {"ci_order": "malformed", "missing_EP": "EP", "tier": "tier T1", "estimator": "registered one", "pooled_missing": "pooled.PDMS missing", "floor_negative": "irep_abs_delta", "nan": "malformed",
           "resample_not_joint": "joint", "resample_missing": "joint", "pooled_mean_mismatch": "mean of the two pairs", "n_tokens_wrong": "n_tokens / n_logs", "n_logs_wrong": "n_tokens / n_logs",
           "n_tokens_missing": "n_tokens / n_logs", "irep_missing": "irep_abs_delta"}
    for name, frag in why.items():
        r = reader_E(c.td, v[name], f"bad_{name}.json")
        assert r["verdict"] == "NOT_READ" and any(frag in p for p in r["problems"]), (name, frag, r)
    # the reader only WARNS about a leaked seed floor ("IGNORED"): our validator is stricter, so the leak can never be emitted
    r = reader_E(c.td, v["seed_floor_leak"], "bad_leak.json")
    assert r["verdict"] in ("ADOPT", "NOT_PROVEN", "REFUTED") and any("IGNORED" in p for p in r["problems"]), r
    assert reader_E(c.td, good_contract(c), "good.json")["verdict"] in ("ADOPT", "NOT_PROVEN", "REFUTED")


@check("pin:contract_vs_OUTCOME_BUILD_md_and_reader_constants", "light")
def _(c):
    T = c.T
    blk = section7_block()
    assert set(blk) == S7_TOP, sorted(blk)
    assert blk["estimator"] == T.ESTIMATOR == CONTRACT_ESTIMATOR and blk["tier"] == T.TIER == "T1", (blk["estimator"], T.ESTIMATOR)
    assert (blk["n_tokens"], blk["n_logs"]) == (T.FULL_PINS["n"], T.FULL_PINS["logs"]) == (12146, 136)
    assert set(blk["pairs"]) == set(T.PAIRS) and set(blk["pairs"]["R0d"]) == set(T.KEYS) == set(KEYS)
    assert set(blk["pooled"]) == {"PDMS", "resample"} and blk["pooled"]["resample"] == T.POOLED_RESAMPLE == "joint"
    assert set(blk["floors"]) == {"irep_abs_delta"} and "seed_floor_abs" not in json.dumps(blk)
    k = reader(c.td, "constants")
    assert (k["N_BOOT"], k["SEED"]) == (T.N_BOOT, T.SEED) and tuple(k["SUBS"]) == T.SUBS and tuple(k["CONSTRAINED"]) == T.PAIRS and tuple(KEYS) == T.KEYS
    assert (k["NAVTEST_TOKENS"], k["NAVTEST_LOGS"]) == (T.FULL_PINS["n"], T.FULL_PINS["logs"]) and k["E_POOL_TOL"] == 1e-9
    prereg = open(f"{O}/PREREG_SAFEOBJ.md", encoding="utf-8").read()
    assert "10,000 resamples, seed 20260927" in prereg and "STRATEGIC" in prereg


@check("pin:reader_is_the_one_pinned_and_documented_in_OUTCOME_BUILD", "light")
def _(c):
    got = hashlib.sha256(open(f"{O}/outcome/safeobj_outcome.py", "rb").read()).hexdigest()
    doc = open(f"{O}/outcome/OUTCOME_BUILD.md", encoding="utf-8").read()
    row = next(l for l in doc.splitlines() if l.startswith("| `safeobj_outcome.py`"))
    assert READER_SHA256 in row, ("OUTCOME_BUILD.md section 1 does not state the pinned reader digest", row[:200])
    assert got == READER_SHA256, f"the reader on disk ({got}) is not the one this suite was written and run against ({READER_SHA256}): re-run the whole suite, then re-pin"


@check("pin:free_point_copies_byte_identical", "light")
def _(c):
    for f, h in FP_COPY_MD5.items():
        assert md5_file(str(HERE / "fp" / f)) == h, f


# ---------------------------------------------------------------- run gates
@check("gates:valid_when_everything_passes", "light")
def _(c):
    g = c.T.read_run_gates(write_gate_dir(os.path.join(c.td, "g_ok")))
    assert g["status"] == "VALID" and all(g["hard"].values()) and all(g["soft"].values()) and len(g["hard"]) == 10, g


@check("gates:ic2_nonzero_refuses", "light")
def _(c):
    T = c.T
    root = write_gate_dir(os.path.join(c.td, "g_ic2"), ic2=1e-7)
    g = T.read_run_gates(root)
    assert g["status"] == "INVALID" and g["hard"]["IC2_replicated_scorer_path_bit_equal"] is False, g
    s = c.syn
    assert refused(lambda: T.build_E(s["paths"], tokens_file=s["tokens_file"], export_logs=s["tl"], pins=None, irep_path=c.irep, run_gates=g), "hard gate")


@check("gates:state_binding_mismatch_refuses", "light")
def _(c):
    g = c.T.read_run_gates(write_gate_dir(os.path.join(c.td, "g_swap"), swap_hashes=True))
    assert g["status"] == "INVALID" and g["hard"]["states_bound_to_expected_hashes"] is False, g
    g2 = c.T.read_run_gates(write_gate_dir(os.path.join(c.td, "g_noexp"), drop_expected=True))
    assert g2["status"] == "INVALID" and g2["hard"]["states_bound_to_expected_hashes"] is False


@check("gates:state_key_must_be_the_name_suffix", "light")
def _(c):
    g = c.T.read_run_gates(write_gate_dir(os.path.join(c.td, "g_keys"), keys_wrong=True))
    assert g["status"] == "INVALID" and g["hard"]["keys_A_B_as_named"] is False, g
    assert g["hard"]["states_bound_to_expected_hashes"] is True, ("the fixture must satisfy the binding gate so that ONLY the name-suffix rule can refuse it", g["hard"])


@check("gates:harness_not_PASS_refuses", "light")
def _(c):
    g = c.T.read_run_gates(write_gate_dir(os.path.join(c.td, "g_h"), harness_fail="R1d_B"))
    assert g["status"] == "INVALID" and g["hard"]["harness_PASS_R1d_B"] is False and g["hard"]["harness_PASS_R0d_A"] is True, g


@check("gates:token_count_must_be_12146", "light")
def _(c):
    g = c.T.read_run_gates(write_gate_dir(os.path.join(c.td, "g_n"), n_tokens=12000))
    assert g["status"] == "INVALID" and g["hard"]["n_tokens_12146"] is False


@check("gates:soft_failures_stamp_IDENTITY_FAILED_and_do_not_refuse", "syn")
def _(c):
    g = c.T.read_run_gates(write_gate_dir(os.path.join(c.td, "g_soft"), ic3=False))
    assert g["status"] == "IDENTITY-FAILED" and g["soft"]["IC3_BASE_vs_banked_u400"] is False
    s = c.syn
    k, prov, _ = c.T.build_E(s["paths"], tokens_file=s["tokens_file"], export_logs=s["tl"], pins=None, irep_path=c.irep, run_gates=g)
    assert prov["status"] == "IDENTITY-FAILED"


# ---------------------------------------------------------------- families
def fam_real(c):
    return c.memo("fams", lambda: {k: json.load(open(p, encoding="utf-8")) for k, p in FAMS.items()})


def fams_for(c, a="deployed", b="u400", a2="u400", b2="deployed"):
    f = fam_real(c)
    return {"R0d_A": f[a], "R0d_B": f[b], "R1d_A": f[a2], "R1d_B": f[b2]}


@check("families:strategic_paused_never_a_pass", "light")
def _(c):
    k, prov, ex = c.syn_build()
    fam = c.T.build_families(c.S3, ex["read_pair"], fams_for(c), n_tokens=1000, n_logs=40)
    for p in ("R0d", "R1d"):
        s = fam["pairs"][p]["STRATEGIC"]
        assert s["status"] == "PAUSED" and "item 25" in s["ruling"] and "pass" not in json.dumps(s).replace("never a pass", "").lower(), s
    assert fam["strategic_pause"]["status"] == "PAUSED"


@check("families:components_and_adverse_match_PE_fam_adverse", "light")
def _(c):
    S3 = c.S3
    k, prov, ex = c.syn_build()
    f = fams_for(c)
    fam = c.T.build_families(S3, ex["read_pair"], f, n_tokens=1000, n_logs=40)
    total_adv = 0
    for p, (a, b) in (("R0d", ("R0d_A", "R0d_B")), ("R1d", ("R1d_A", "R1d_B"))):
        want = S3.PE.fam_adverse(f[a], f[b])                      # the program's own function, called by the test
        want_set = {(x["family"], x["metric"]) for x in want["adverse"]}
        got_set = {(fm.lower(), m) for fm in ("LONGITUDINAL", "LATERAL") for m, row in fam["pairs"][p][fm]["families6"]["components"].items() if row["adverse"]}
        assert got_set == want_set, (p, got_set ^ want_set)
        assert fam["pairs"][p]["LONGITUDINAL"]["families6"]["n_components_compared"] == want["components_compared"]
        total_adv += len(want_set)
        for fm in ("LONGITUDINAL", "LATERAL"):
            for m, row in fam["pairs"][p][fm]["families6"]["components"].items():
                assert abs(row["delta_B_minus_A"] - (row["B"]["mean"] - row["A"]["mean"])) <= 1e-12
    assert total_adv >= 2, "the fixture must contain adverse components in both directions, or the flag cannot be tested"


@check("families:kappa_rule_fall_gt_0.03", "light")
def _(c):
    k, prov, ex = c.syn_build()
    f = {k_: copy.deepcopy(v_) for k_, v_ in fams_for(c).items()}

    def set_kappa(fam, v):
        fam["families"]["refcv6"]["tactical"]["lateral_decision"]["kappa"] = v
    set_kappa(f["R0d_A"], 0.5)
    set_kappa(f["R0d_B"], 0.46875)                            # -0.03125: a fall by more than 0.03
    set_kappa(f["R1d_A"], 0.5)
    set_kappa(f["R1d_B"], 0.484375)                           # -0.015625: not
    fam = c.T.build_families(c.S3, ex["read_pair"], f, n_tokens=1000, n_logs=40)
    ld0, ld1 = fam["pairs"]["R0d"]["TACTICAL"]["lateral_decision"], fam["pairs"]["R1d"]["TACTICAL"]["lateral_decision"]
    assert ld0["kappa_fell_by_more_than_0.03"] is True and ld1["kappa_fell_by_more_than_0.03"] is False and abs(ld0["delta_kappa_B_minus_A"] + 0.03125) < 1e-12


@check("families:missing_is_not_computed_never_pass", "light")
def _(c):
    k, prov, ex = c.syn_build()
    fam = c.T.build_families(c.S3, ex["read_pair"], {s: None for s in STATES}, n_tokens=1000, n_logs=40)
    assert fam["status"].startswith("PDM_MEMBERS_ONLY")
    for p in ("R0d", "R1d"):
        b = fam["pairs"][p]
        assert b["TACTICAL"]["status"] == "NOT_COMPUTED" and b["LONGITUDINAL"]["families6"]["status"] == "NOT_COMPUTED" and b["LATERAL"]["families6"]["status"] == "NOT_COMPUTED", b
        assert b["STRATEGIC"]["status"] == "PAUSED"


@check("families:pdm_members_use_the_paired_estimator_and_the_free_point_membership", "light")
def _(c):
    k, prov, ex = c.syn_build()
    fam = c.T.build_families(c.S3, ex["read_pair"], fams_for(c), n_tokens=1000, n_logs=40)
    for p in ("R0d", "R1d"):
        for fm, members in (("LONGITUDINAL", ("EP", "TTC", "C")), ("LATERAL", ("DAC", "DDC"))):
            pm = fam["pairs"][p][fm]["pdm_members"]
            assert tuple(pm) == members, (fm, tuple(pm))
            for m in members:
                assert pm[m]["mean"] == k["pairs"][p][m]["mean"] and pm[m]["ci95"] == k["pairs"][p][m]["ci95"]
    assert "paired log-cluster" in fam["estimators"]["pdm_members"] and "UNPAIRED" in fam["estimators"]["families6_components"]


# ---------------------------------------------------------------- seams (the free-point picker, state-generic) on the REAL dry-run chunk
def dry_chunk_copy(c, rename=None, ic2=None):
    d = os.path.join(c.td, "chunks" + ("_r" if rename else "") + ("_i" if ic2 is not None else ""))
    os.makedirs(d, exist_ok=True)
    z = np.load(f"{DRY}/out/chunk_000.npz", allow_pickle=False)
    arrs = {k: z[k] for k in z.files}
    if rename:
        arrs = {(("logits_" + rename[k[7:]]) if k.startswith("logits_") and k[7:] in rename else k): v for k, v in arrs.items()}
        if "R0d_A" in rename.values():
            arrs["logits_R1d_A"], arrs["logits_R1d_B"] = arrs["logits_R0d_A"].copy(), arrs["logits_R0d_B"].copy()
    if ic2 is not None:
        arrs["ic2_maxabs"] = np.full_like(arrs["ic2_maxabs"], ic2)
    np.savez(os.path.join(d, "chunk_000.npz"), **arrs)
    return d


@check("seams:reproduce_the_live_picker_poses_bit_for_bit", "seams")
def _(c):
    d = dry_chunk_copy(c)
    w = os.path.join(c.td, "seams_w")
    rep = c.T.build_seams(d, w, names=("C0A", "R0B"), present_only=True)
    assert rep["n_tokens"] == 2 and rep["IC2_max_abs"] == 0.0
    for k in ("BASE", "C0A", "R0B"):
        mine, live = np.load(f"{w}/seam_{k}.npz", allow_pickle=False), np.load(f"{DRY}/seam_{k}.npz", allow_pickle=False)
        assert [str(t) for t in mine["token"]] == [str(t) for t in live["token"]] and np.array_equal(mine["poses"], live["poses"]), k
    assert rep["pairs"] == {}, "a partial name list has no complete pair: no pair counts"


@check("seams:default_names_are_the_four_states_and_the_pair_counts_exist", "seams")
def _(c):
    d = dry_chunk_copy(c, rename={"C0A": "R0d_A", "R0B": "R0d_B"})
    w = os.path.join(c.td, "seams_w2")
    rep = c.T.build_seams(d, w, present_only=True)
    for k in ("BASE",) + STATES:
        assert os.path.exists(f"{w}/seam_{k}.npz"), k
    assert set(rep["pairs"]) == {"R0d", "R1d"} and rep["pairs"]["R0d"]["pick_changes_B_vs_A"] == 2 and rep["pairs"]["R1d"]["pick_changes_B_vs_A"] == 2     # the live dry run: pick_changes_C0A_to_R0B == 2
    live = np.load(f"{DRY}/seam_C0A.npz", allow_pickle=False)
    assert np.array_equal(np.load(f"{w}/seam_R0d_A.npz", allow_pickle=False)["poses"], live["poses"])


@check("seams:refuses_missing_state_logits", "seams")
def _(c):
    d = dry_chunk_copy(c)
    assert refused(lambda: c.T.build_seams(d, os.path.join(c.td, "seams_w3"), present_only=True), "no logits")


@check("seams:refuses_ic2_nonzero", "seams")
def _(c):
    d = dry_chunk_copy(c, ic2=1e-9)
    assert refused(lambda: c.T.build_seams(d, os.path.join(c.td, "seams_w4"), names=("C0A", "R0B"), present_only=True), "IC2")


# ---------------------------------------------------------------- the pod/dev-box plan and the shipper wrapper
@check("plan:state_specs_bind_A_to_A_and_B_to_B", "light")
def _(c):
    T = c.T
    specs = T.state_specs("/workspace/data/rls1/arms")
    assert len(specs) == 4
    want = {"R0d_A": ("R0d", "A"), "R0d_B": ("R0d", "B"), "R1d_A": ("R1d", "A"), "R1d_B": ("R1d", "B")}
    got = {}
    for s in specs:
        assert s.startswith("--state ")
        name, _, rest = s[len("--state "):].partition("=")
        path, _, key = rest.rpartition(":")
        got[name] = (path.split("/")[-2], key)
        assert path.endswith("/scorer_states_last.pt")
    assert got == want, got


@check("plan:refuses_the_live_free_point_directory", "light")
def _(c):
    assert refused(lambda: c.T.plan_text("/workspace/data/rls1/arms", "W", "/workspace/ops/freept"), "free-point")
    txt = c.T.plan_text("/workspace/data/rls1/arms", "W", "/workspace/ops/sobjE")
    assert "/workspace/ops/freept" not in txt


@check("plan:script_names_every_stage_and_label_prefix", "light")
def _(c):
    txt = c.T.plan_text("/workspace/data/rls1/arms", "W", "/workspace/ops/sobjE")
    for needle in ("expected-states", "sobj_ship.py prep-pod", "fp_forward.py serve", "--repeat-first 128", "sobj_ship.py run", "navtest_E.py seams", "fp_score.py score", "--label-prefix refe_sobj_",
                   "navtest_E.py families", "navtest_E.py build", "--irep-json", "safeobj_outcome.py", "--navtest-e",
                   "--addendum raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.addendum1.md"):
        assert needle in txt, needle
    assert c.T.LABEL_PREFIX.startswith("refe") and c.T.label_of("R1d_B") == "refe_sobj_R1d_B"
    assert "--points R0d_A,R0d_B,R1d_A,R1d_B" in txt
    assert "--rlval-arm" not in txt, "the producer takes no RL-val input: the reader derives that floor from bar L"


@check("plan:ship_wrapper_overrides_remote_and_states_and_refuses_live", "light")
def _(c):
    code = (f"import sys; sys.dont_write_bytecode=True; sys.path.insert(0, {str(HERE)!r}); import sobj_ship as W, fp_ship as SH; "
            "orig=(SH.REMOTE, dict(SH.STATES)); W.configure('/workspace/ops/sobjE'); "
            "import json; print('ZZ'+json.dumps([orig[0], len(orig[1]), SH.REMOTE, SH.STATES])); "
            "bad=[]\n"
            "for r in ('/workspace/ops/freept','/workspace/ops/freept/','/tmp/x'):\n"
            "    try: W.configure(r); bad.append(r)\n"
            "    except SystemExit: pass\n"
            "print('ZZBAD'+json.dumps(bad))")
    r = subprocess.run([PY, "-I", "-X", "utf8", "-c", code], capture_output=True, text=True, encoding="utf-8")
    z = next(l for l in r.stdout.splitlines() if l.startswith("ZZ") and not l.startswith("ZZBAD"))
    orig_remote, n_states, new_remote, new_states = json.loads(z[2:])
    assert orig_remote == "/workspace/ops/freept" and n_states == 2 and new_remote == "/workspace/ops/sobjE" and new_states == {}
    assert json.loads(next(l for l in r.stdout.splitlines() if l.startswith("ZZBAD"))[5:]) == [], r.stdout


# ---------------------------------------------------------------- expected-states (REAL banked C0 / R0 state files; tiny synthetic ones for the error paths)
CLI_MUT_SRC = None          # set by mutation_job: path of a file holding the MUTATED tool text; None = run the tool file itself
CLI_RUNNER = ("import sys\nsrc = open(sys.argv[1], encoding='utf-8').read()\npath = sys.argv[2]\nsys.argv = [path] + sys.argv[3:]\n"
              "exec(compile(src, path, 'exec'), {'__name__': '__main__', '__file__': path})\n")


def run_cli(args, td, timeout=900):
    """the tool's CLI in a clean interpreter.  Under a mutation job it executes the MUTATED text as the tool's own path (so its relative imports resolve), never the unmutated file"""
    cmd = [PY, "-I", "-X", "utf8", str(TOOL_PATH)] if CLI_MUT_SRC is None else [PY, "-I", "-X", "utf8", "-c", CLI_RUNNER, CLI_MUT_SRC, str(TOOL_PATH)]
    r = subprocess.run(cmd + args, capture_output=True, text=True, encoding="utf-8", cwd=td, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


@check("states:expected_hashes_equal_the_free_point_EXPECTED_STATES_for_real_files", "states")
def _(c):
    out = os.path.join(c.td, "exp_real.json")
    rc, txt = run_cli(["expected-states", "--state", f"R0d_A={C0_STATES}:A", "--state", f"R0d_B={C0_STATES}:B", "--state", f"R1d_A={R0_STATES}:A", "--state", f"R1d_B={R0_STATES}:B", "--out", out], c.td)
    assert rc == 0 and "ZZEXPECTED_E_OK" in txt, txt[-400:]
    d = json.load(open(out))
    assert d["R0d_A"]["sha256_tensors"] == EXPECT_HASH["C0A"] and d["R0d_B"]["sha256_tensors"] == EXPECT_HASH["C0B"]
    assert d["R1d_A"]["sha256_tensors"] == EXPECT_HASH["R0A"] and d["R1d_B"]["sha256_tensors"] == EXPECT_HASH["R0B"]
    assert all(v["update"] == 1000 for v in d.values())


@check("states:expected_states_refuses_key_name_mismatch_update_and_identical_tensors", "states")
def _(c):
    import torch
    mk = lambda v: {m: {"w": torch.full((3,), float(v))} for m in ("score_q_mlp", "score_dec", "score_head")}     # noqa: E731
    f_ok = os.path.join(c.td, "ss_ok.pt")
    torch.save({"A": mk(1), "B": mk(2), "update": 1000, "EPO_CON": {"lam": {"ttc": 0.1}}}, f_ok)
    f_same = os.path.join(c.td, "ss_same.pt")
    torch.save({"A": mk(1), "B": mk(1), "update": 1000}, f_same)
    f_u = os.path.join(c.td, "ss_u.pt")
    torch.save({"A": mk(1), "B": mk(2), "update": 750}, f_u)

    # the good case needs four DISTINCT tensor sets: R1d from the other file
    g = os.path.join(c.td, "ss_ok2.pt")
    torch.save({"A": mk(3), "B": mk(4), "update": 1000}, g)
    out = os.path.join(c.td, "exp_good.json")
    args = ["expected-states"]
    for s_ in [f"R0d_A={f_ok}:A", f"R0d_B={f_ok}:B", f"R1d_A={g}:A", f"R1d_B={g}:B"]:
        args += ["--state", s_]
    rc, txt = run_cli(args + ["--out", out], c.td)
    assert rc == 0 and os.path.exists(out), txt[-300:]
    # key != the name's suffix (A state read from key B): REFUSED
    bad = ["expected-states"]
    for s_ in [f"R0d_A={f_ok}:B", f"R0d_B={f_ok}:B", f"R1d_A={g}:A", f"R1d_B={g}:B"]:
        bad += ["--state", s_]
    rc1, _t = run_cli(bad + ["--out", os.path.join(c.td, "e1.json")], c.td)
    # update != 1000
    upd = ["expected-states"]
    for s_ in [f"R0d_A={f_u}:A", f"R0d_B={f_u}:B", f"R1d_A={g}:A", f"R1d_B={g}:B"]:
        upd += ["--state", s_]
    rc2, _t = run_cli(upd + ["--out", os.path.join(c.td, "e2.json")], c.td)
    # A and B tensor-identical in one file
    same = ["expected-states"]
    for s_ in [f"R0d_A={f_same}:A", f"R0d_B={f_same}:B", f"R1d_A={g}:A", f"R1d_B={g}:B"]:
        same += ["--state", s_]
    rc3, _t = run_cli(same + ["--out", os.path.join(c.td, "e3.json")], c.td)
    # fewer than four states
    rc4, _t = run_cli(["expected-states", "--state", f"R0d_A={f_ok}:A", "--out", os.path.join(c.td, "e4.json")], c.td)
    assert (rc1, rc2, rc3, rc4) == (2, 2, 2, 2) and not any(os.path.exists(os.path.join(c.td, f"e{i}.json")) for i in (1, 2, 3, 4))


# ---------------------------------------------------------------- the CLI end to end
def subset_tokens_file(c, n_logs=30):
    """a REAL subset of the navtest: the tokens of the first `n_logs` logs, as a token file with its own token_log (not the pinned set: only a --dry-run-subset run may read it)"""
    def f():
        full = json.load(open(c.T.TOKENS_FILE, encoding="utf-8"))
        logs = sorted(set(full["token_log"].values()))[:n_logs]
        keep = [t for t in sorted(full["tokens"]) if full["token_log"][t] in logs]
        p = os.path.join(c.td, "subset_tokens.json")
        json.dump({"rule": "real subset (dry-run fixture)", "tokens": keep, "token_log": {t: full["token_log"][t] for t in keep}}, open(p, "w"))
        return p
    return c.memo("subset_tokens", f)


def cli_world(c, real: bool):
    """-> (argv for `build`, out_dir, work_dir).  real=True: the P2REP pairs on the pinned full navtest; real=False: a REAL 30-log subset of the same tables (needs --dry-run-subset)."""
    c._n = getattr(c, "_n", 0) + 1
    tag = f"{'real' if real else 'subset'}_{c._n}"
    work = os.path.join(c.td, f"cli_work_{tag}")
    write_gate_dir(work)
    out = os.path.join(c.td, f"cli_out_{tag}")
    args = ["build", "--work", work, "--out-dir", out, "--irep-json", c.irep]
    tabs = c.p2rep_tables
    for s, p in tabs.items():
        args += ["--table", f"{s}={p}"]
    if real:
        os.makedirs(os.path.join(work, "families"), exist_ok=True)
        for s, k in (("R0d_A", "u400"), ("R0d_B", "deployed"), ("R1d_A", "u400"), ("R1d_B", "deployed")):
            shutil.copy(FAMS[k], os.path.join(work, "families", f"sobj_{s}_families.json"))
    else:
        args += ["--tokens", subset_tokens_file(c), "--dry-run-subset"]
    return args, out, work


@check("cli:real_pinned_build_end_to_end_writes_the_three_files", "cli")
def _(c):
    args, out, work = cli_world(c, True)
    rc, txt = run_cli(args, c.td)
    assert rc == 0 and "ZZNAVTEST_E_OK VALID" in txt, txt[-600:]
    assert sorted(os.listdir(out)) == ["navtest_E.json", "navtest_E.provenance.json", "navtest_families.json"], os.listdir(out)
    k = json.load(open(os.path.join(out, "navtest_E.json")))
    c.T.validate_contract(k)
    tables_match(k["pairs"]["R0d"], P2REP_S0_MINUS_U400)
    prov = json.load(open(os.path.join(out, "navtest_E.provenance.json")))
    assert prov["status"] == "VALID" and prov["run_gates"]["fixture"] is True and prov["pinned_tokens"] is True and "FIXTURE" not in k
    fam = json.load(open(os.path.join(out, "navtest_families.json")))
    assert fam["status"] == "COMPUTED" and fam["pairs"]["R1d"]["STRATEGIC"]["status"] == "PAUSED"
    assert k["n_tokens"] == 12146 and k["n_logs"] == 136 and k["pooled"]["resample"] == "joint" and set(k["floors"]) == {"irep_abs_delta"}
    r = reader_E(c.td, k, "cli_real.json")
    assert r["verdict"] == "NOT_PROVEN" and r["problems"] == [] and r["clauses"]["3_mean_contrast_gt_2x_max_floor"] is True, r


@check("cli:subset_refused_without_the_dry_run_flag", "cli")
def _(c):
    args, out, work = cli_world(c, False)
    args = [a for a in args if a != "--dry-run-subset"]
    rc, txt = run_cli(args, c.td)
    assert rc == 2 and "REFUSED" in txt and not os.path.exists(out), (rc, txt[-300:])
    assert not os.path.exists(os.path.join(out, "navtest_E.json"))


@check("cli:dry_run_names_only_DRYRUN_files", "cli")
def _(c):
    args, out, work = cli_world(c, False)
    rc, txt = run_cli(args, c.td)
    assert rc == 0 and "ZZNAVTEST_E_DRYRUN" in txt, txt[-400:]
    names = sorted(os.listdir(out))
    assert names == ["navtest_E.DRYRUN.json", "navtest_E.DRYRUN.provenance.json", "navtest_families.DRYRUN.json"], names
    prov = json.load(open(os.path.join(out, "navtest_E.DRYRUN.provenance.json")))
    assert "DRY_RUN" in prov and prov["pinned_tokens"] is False and prov["token_gate"]["n"] < 12146
    dk = json.load(open(os.path.join(out, "navtest_E.DRYRUN.json")))
    assert dk["n_tokens"] < 12146 and dk["n_logs"] < 136
    r = reader_E(c.td, dk, "dry_to_reader.json")                               # "the reader must never be given this file": and if it is, it refuses it
    assert r["verdict"] == "NOT_READ" and any("n_tokens / n_logs" in p for p in r["problems"]), r


@check("cli:refuses_without_irep_and_writes_nothing", "cli")
def _(c):
    args, out, work = cli_world(c, False)
    i = args.index("--irep-json")
    rc, txt = run_cli(args[:i] + args[i + 2:], c.td)
    assert rc == 2 and "NO default" in txt and not os.path.exists(out), (rc, txt[-300:])


@check("cli:hard_gate_failure_exits_2_and_writes_no_contract", "cli")
def _(c):
    args, out, work = cli_world(c, False)
    write_gate_dir(work, ic2=3e-6)
    rc, txt = run_cli(args, c.td)
    assert rc == 2 and "hard gate" in txt and not os.path.exists(out), (rc, txt[-300:])


# ============================================================ mutations: (name, [(old, new), ...], {checks that must go RED})
BOOT_INDEP = '''    import numpy as _np

    def _draws(d, sd):
        ts = sorted(d)
        by = {}
        for i, t in enumerate(ts):
            by.setdefault(tl[t], []).append(i)
        gs = [_np.asarray(v) for _k, v in sorted(by.items())]
        vv = _np.asarray([d[t] for t in ts], dtype=_np.float64)
        r = _np.random.default_rng(sd)
        return _np.array([vv[_np.concatenate([gs[j] for j in r.integers(0, len(gs), len(gs))])].mean() for _ in range(n)])
    _m = (_draws(d0, seed) + _draws(d1, seed + 1)) / 2.0
    mu, lo, hi = float(_np.mean(list(series.values()))), float(_np.percentile(_m, 2.5)), float(_np.percentile(_m, 97.5))'''

MUTATIONS = [
    # --- pairs swapped
    ("pairs_swapped:R0d_reads_R1d_tables_and_vice_versa",
     [('PAIR_STATES = {"R0d": ("R0d_A", "R0d_B"), "R1d": ("R1d_A", "R1d_B")}', 'PAIR_STATES = {"R0d": ("R1d_A", "R1d_B"), "R1d": ("R0d_A", "R0d_B")}')],
     {"real:R0d_matches_RESULT_P2REP_P2s0_minus_u400_2dp", "real:R1d_matches_RESULT_P2REP_RR1_P2s1_minus_u400_2dp", "syn:pair_labels_and_sign_B_minus_A", "real:anchor_u400_minus_deployed_equals_RESULT_SFT4_1e-12"}),
    ("pairs_swapped:A_and_B_exchanged_in_the_contrast_(B-A becomes A-B)",
     [("out[p] = S3.read_pair(tabs[b], tabs[a], toks, tl, n, seed)", "out[p] = S3.read_pair(tabs[a], tabs[b], toks, tl, n, seed)")],
     {"real:anchor_u400_minus_deployed_equals_RESULT_SFT4_1e-12", "real:R0d_matches_RESULT_P2REP_P2s0_minus_u400_2dp", "syn:pair_labels_and_sign_B_minus_A", "real:pooled_mean_is_mean_of_the_two_pair_means"}),
    ("pairs_swapped:pooled_series_uses_A_minus_B",
     [("return {t: tabs[b][t][key] - tabs[a][t][key] for t in toks}", "return {t: tabs[a][t][key] - tabs[b][t][key] for t in toks}")],
     {"real:pooled_equals_joint_reference_1e-9", "syn:pooled_equals_joint_reference_1e-9", "real:pooled_mean_is_mean_of_the_two_pair_means"}),
    # --- pooled
    ("pooled_as_two_independent_bootstraps",
     [("    mu, lo, hi = S3.PE.boot(series, tl, n=n, seed=seed)", BOOT_INDEP)],
     {"syn:pooled_equals_joint_reference_1e-9", "syn:pooled_is_not_two_independent_bootstraps", "real:pooled_equals_joint_reference_1e-9"}),
    ("pooled_bootstrap_with_a_different_seed",
     [("    mu, lo, hi = S3.PE.boot(series, tl, n=n, seed=seed)", "    mu, lo, hi = S3.PE.boot(series, tl, n=n, seed=seed + 1)")],
     {"syn:pooled_equals_joint_reference_1e-9", "real:pooled_equals_joint_reference_1e-9"}),
    # --- unit slip x1 vs x100
    ("unit_slip:guard_disabled",
     [("    if hi <= 1.0 + 1e-9:", "    if False:")],
     {"unit:refuses_x1_tables"}),
    ("unit_slip:tables_loaded_at_x1_and_guard_disabled",
     [("        out[s] = S3.load_table(paths[s])", "        out[s] = {t: {k: v / 100.0 for k, v in r.items()} for t, r in S3.load_table(paths[s]).items()}"), ("    if hi <= 1.0 + 1e-9:", "    if False:")],
     {"real:anchor_u400_minus_deployed_equals_RESULT_SFT4_1e-12", "real:R0d_matches_RESULT_P2REP_P2s0_minus_u400_2dp", "unit:refuses_x1_tables", "syn:pair_labels_and_sign_B_minus_A"}),
    ("unit_slip:upper_bound_removed",
     [("    if hi > 100.0 + 1e-6 or lo < -1e-9:", "    if False:")],
     {"unit:refuses_scale_above_100"}),
    ("unit_slip:irep_unit_not_checked",
     [("    if unit != UNIT:", "    if False:")],
     {"floors:irep_unit_slip_refused"}),
    # --- CI order
    ("ci_order_broken:emitter_writes_[hi,lo]_validator_intact",
     [('return {"mean": float(r["mean"]), "ci95": [float(r["ci95"][0]), float(r["ci95"][1])]}', 'return {"mean": float(r["mean"]), "ci95": [float(r["ci95"][1]), float(r["ci95"][0])]}')],
     {"real:build_runs", "syn:build_runs"}),
    ("ci_order_broken:validator_no_longer_checks_lo_le_hi",
     [('and all(_num(v) for v in x["ci95"]) and x["ci95"][0] <= x["ci95"][1])', 'and all(_num(v) for v in x["ci95"]))')],
     {"contract:validator_refuses_ci_order"}),
    ("ci_order_broken:emitter_AND_validator_broken_(only_the_file_and_the_reader_can_see_it)",
     [('return {"mean": float(r["mean"]), "ci95": [float(r["ci95"][0]), float(r["ci95"][1])]}', 'return {"mean": float(r["mean"]), "ci95": [float(r["ci95"][1]), float(r["ci95"][0])]}'),
      ('and all(_num(v) for v in x["ci95"]) and x["ci95"][0] <= x["ci95"][1])', 'and all(_num(v) for v in x["ci95"]))')],
     {"contract:ci_order_in_emitted_file", "contract:validator_refuses_ci_order", "real:accepted_by_the_REAL_outcome_reader"}),
    # --- a missing sub-score accepted
    ("missing_subscore_accepted:validator_no_longer_requires_all_seven",
     [("            if not isinstance(o, dict) or set(o) != set(KEYS):", "            if not isinstance(o, dict):"), ("            for k in KEYS:\n                if not _ci_ok(o[k]):", "            for k in o:\n                if not _ci_ok(o[k]):")],
     {"contract:validator_refuses_missing_subscore"}),
    ("missing_subscore_accepted:EP_dropped_by_the_emitter_and_the_validator_weakened",
     [("            if not isinstance(o, dict) or set(o) != set(KEYS):", "            if not isinstance(o, dict):"), ("            for k in KEYS:\n                if not _ci_ok(o[k]):", "            for k in o:\n                if not _ci_ok(o[k]):"),
      ("    for k in SUBS:\n        o[k] = ci_obj(rp[\"subscores\"][k])", "    for k in [x for x in SUBS if x != \"EP\"]:\n        o[k] = ci_obj(rp[\"subscores\"][k])")],
     {"contract:subscores_all_seven_in_emitted_file", "real:accepted_by_the_REAL_outcome_reader", "contract:validator_refuses_missing_subscore"}),
    # --- floors
    ("floors:seed_floor_leaks_into_the_file_(emitter_only)",
     [('    return {"irep_abs_delta": float(irep["value"])}', '    return {"irep_abs_delta": float(irep["value"]), "seed_floor_abs": 0.0}')],
     {"real:build_runs"}),
    ("floors:seed_floor_leaks_into_the_file_(emitter_AND_validator_weakened)",
     [('    return {"irep_abs_delta": float(irep["value"])}', '    return {"irep_abs_delta": float(irep["value"]), "seed_floor_abs": 0.0}'), ('set(fl) != {"irep_abs_delta"}', '"irep_abs_delta" not in fl')],
     {"real:contract_exact_keys_types_order", "real:accepted_by_the_REAL_outcome_reader", "contract:validator_refuses_a_seed_floor_in_the_file_and_a_missing_irep"}),
    ("floors:sidecar_navtest_seed_floor_wrong",
     [('"navtest_seed_floor_abs_R0d_vs_R1d": abs(pairs["R0d"]["PDMS"]["mean"] - pairs["R1d"]["PDMS"]["mean"])', '"navtest_seed_floor_abs_R0d_vs_R1d": abs(pairs["R0d"]["PDMS"]["mean"] + pairs["R1d"]["PDMS"]["mean"])')],
     {"floors:file_carries_only_irep_and_the_sidecar_seed_floor_equals_the_readers"}),
    ("floors:irep_defaults_to_zero_when_missing",
     [('    if not path or not os.path.exists(path):\n        raise Refused("floors.irep_abs_delta has NO default: pass --irep-json (a defaulted 0 would silently weaken E clause 3)")',
       '    if not path or not os.path.exists(path):\n        return {"value": 0.0, "source": None, "sha256": None, "kind": "DEFAULTED"}')],
     {"floors:irep_required_refuses_missing"}),
    # --- the section 7 elements that arrived with the 14:15 revision
    ("pooled:resample_label_not_joint_(emitter_and_validator_agree_on_the_wrong_word)",
     [('POOLED_RESAMPLE = "joint"', 'POOLED_RESAMPLE = "independent"')],
     {"real:contract_exact_keys_types_order", "real:accepted_by_the_REAL_outcome_reader", "pin:contract_vs_OUTCOME_BUILD_md_and_reader_constants"}),
    ("pooled:validator_no_longer_checks_the_resample_label",
     [('or po.get("resample") != POOLED_RESAMPLE:', ':')],
     {"contract:validator_refuses_resample_not_joint"}),
    ("pooled:validator_no_longer_checks_pooled_mean_against_the_pairs",
     [('/ 2.0) > 1e-9:\n        pb.append("pooled.PDMS.mean is not the mean', '/ 2.0) > 1e9:\n        pb.append("pooled.PDMS.mean is not the mean')],
     {"contract:validator_refuses_pooled_mean_that_is_not_the_mean_of_the_pairs"}),
    ("counts:validator_no_longer_pins_n_tokens_n_logs",
     [("or (full and v != want):", "or False:")],
     {"contract:validator_pins_n_tokens_and_n_logs", "contract:a_subset_file_carries_subset_counts_and_the_reader_refuses_it"}),
    ("counts:n_tokens_off_by_one_in_the_emitter_(validator_unpinned)",
     [('"n_tokens": len(toks), "n_logs"', '"n_tokens": len(toks) - 1, "n_logs"'), ("or (full and v != want):", "or False:")],
     {"real:contract_exact_keys_types_order", "real:accepted_by_the_REAL_outcome_reader"}),
    # --- tokens / degenerate / pins
    ("tokens:missing_rows_tolerated",
     [("        miss = [x for x in toks if x not in t]\n        if miss:", "        miss = [x for x in toks if x not in t]\n        if False:")],
     {"tokens:refuses_missing_token_rows"}),
    ("degenerate:B_equal_A_not_refused",
     [("        if same:", "        if False:")],
     {"degenerate:refuses_B_equals_A"}),
    ("degenerate:identical_pairs_not_refused",
     [("    if d0 == d1:", "    if False:")],
     {"degenerate:refuses_two_identical_pairs"}),
    ("pins:token_pin_not_enforced",
     [("    if not t[\"gate\"][\"ok\"]:", "    if False:")],
     {"pins:refuses_an_unpinned_token_set_on_the_pinned_path"}),
    ("pins:nondefault_n_boot_allowed_on_the_pinned_path",
     [("        if pins is not None:\n            raise Refused(\"the registered estimator", "        if False:\n            raise Refused(\"the registered estimator")],
     {"pins:refuses_a_nondefault_estimator_on_the_pinned_path"}),
    ("estimator:N_BOOT_2000",
     [("N_BOOT, SEED = 10000, 20260927", "N_BOOT, SEED = 2000, 20260927")],
     {"real:contract_exact_keys_types_order", "real:accepted_by_the_REAL_outcome_reader", "pin:contract_vs_OUTCOME_BUILD_md_and_reader_constants", "pins:instrument_md5_and_estimator_constants"}),
    # --- gates
    ("gates:IC2_not_enforced",
     [('hard["IC2_replicated_scorer_path_bit_equal"] = bool(rep.get("IC2_max_abs") == 0.0)', 'hard["IC2_replicated_scorer_path_bit_equal"] = True')],
     {"gates:ic2_nonzero_refuses"}),
    ("gates:state_binding_not_enforced",
     [('hard["states_bound_to_expected_hashes"] = bool(ok)', 'hard["states_bound_to_expected_hashes"] = True')],
     {"gates:state_binding_mismatch_refuses"}),
    ("gates:harness_status_not_read",
     [('hard[f"harness_PASS_{s}"] = bool(st.get("status") == "PASS")', 'hard[f"harness_PASS_{s}"] = True')],
     {"gates:harness_not_PASS_refuses"}),
    ("gates:token_count_not_enforced",
     [('hard["n_tokens_12146"] = bool(rep.get("n_tokens") == FULL_PINS["n"])', 'hard["n_tokens_12146"] = True')],
     {"gates:token_count_must_be_12146"}),
    ("gates:key_suffix_not_enforced",
     [('hard["keys_A_B_as_named"] = bool(all(exp.get(s, {}).get("key") == s[-1] for s in STATES))', 'hard["keys_A_B_as_named"] = True')],
     {"gates:state_key_must_be_the_name_suffix"}),
    # --- families
    ("families:strategic_reported_as_PASS",
     [('STRATEGIC = {"status": "PAUSED",', 'STRATEGIC = {"status": "PASS",')],
     {"families:strategic_paused_never_a_pass", "families:missing_is_not_computed_never_pass"}),
    ("families:kappa_threshold_0.3",
     [("KAPPA_FALL_MAX = 0.03", "KAPPA_FALL_MAX = 0.3")],
     {"families:kappa_rule_fall_gt_0.03"}),
    ("families:missing_families_reported_OK",
     [('{"status": "NOT_COMPUTED", "reason": "families6 JSON of A and/or B absent: run `navtest_E.py families` (never read as a pass)"}', '{"status": "OK", "reason": "families6 JSON of A and/or B absent: run `navtest_E.py families` (never read as a pass)"}')],
     {"families:missing_is_not_computed_never_pass"}),
    ("families:adverse_flag_always_False",
     [('"adverse": (fam, m) in adverse_set}', '"adverse": False}')],
     {"families:components_and_adverse_match_PE_fam_adverse"}),
    ("families:pdm_members_wrong_membership",
     [('("lateral", "LATERAL", ("DAC", "DDC"))', '("lateral", "LATERAL", ("DAC", "NC"))')],
     {"families:pdm_members_use_the_paired_estimator_and_the_free_point_membership"}),
    # --- seams
    ("seams:missing_state_logits_not_refused",
     [("    if miss:\n        raise Refused(f\"the chunks carry no logits", "    if False:\n        raise Refused(f\"the chunks carry no logits")],
     {"seams:refuses_missing_state_logits"}),
    ("seams:IC2_not_enforced",
     [("    if ic2.max() != 0.0:\n        raise Refused", "    if False:\n        raise Refused")],
     {"seams:refuses_ic2_nonzero"}),
    ("seams:pair_counts_use_A_for_B",
     [('changes[p] = {"pick_changes_B_vs_A": int((pk[b] != pk[a]).sum())', 'changes[p] = {"pick_changes_B_vs_A": int((pk[a] != pk[a]).sum())')],
     {"seams:default_names_are_the_four_states_and_the_pair_counts_exist"}),
    # --- plan / shipper
    ("plan:live_directory_allowed",
     [('    if remote.rstrip("/") == LIVE_REMOTE:\n        raise Refused(f"{LIVE_REMOTE} is the free-point run', '    if False:\n        raise Refused(f"{LIVE_REMOTE} is the free-point run')],
     {"plan:refuses_the_live_free_point_directory"}),
    ("plan:state_key_swapped_A_for_B",
     [('out.append(f"--state {s}={arms_root}/{p}/scorer_states_last.pt:{s[-1]}")', 'out.append(f"--state {s}={arms_root}/{p}/scorer_states_last.pt:{\'A\' if s[-1] == \'B\' else \'B\'}")')],
     {"plan:state_specs_bind_A_to_A_and_B_to_B"}),
    ("plan:harness_label_prefix_not_refe",
     [('LABEL_PREFIX = "refe_sobj_"', 'LABEL_PREFIX = "sobj_"')],
     {"plan:script_names_every_stage_and_label_prefix"}),
    ("plan:reader_command_without_the_addendum",
     [('"--addendum raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.addendum1.md --out ...",', '"--out ...",')],
     {"plan:script_names_every_stage_and_label_prefix"}),
    # --- CLI
    # three independent layers refuse a non-pinned token set on the pinned path: cmd_build's own check, token_gate (pins), and validate_contract(full) on n_tokens / n_logs.  Removing one or two stays GREEN BY DESIGN
    # (MEASURED: logs/dev_atrisk4.log, the two-layer form); only the removal of ALL THREE can show that the check sees the guard.
    ("cli:subset_allowed_without_the_dry_run_flag_(all_three_guard_layers_removed)",
     [("        elif a.tokens and os.path.abspath(a.tokens) != os.path.abspath(TOKENS_FILE):", "        elif False:"), ("    if not t[\"gate\"][\"ok\"]:", "    if False:"),
      ("or (full and v != want):", ":")],
     {"cli:subset_refused_without_the_dry_run_flag"}),
    ("cli:dry_run_writes_the_real_file_name",
     [('name = "navtest_E.DRYRUN.json" if dry_run else "navtest_E.json"', 'name = "navtest_E.json"')],
     {"cli:dry_run_names_only_DRYRUN_files"}),
]


# ============================================================ runner
def run_checks(T, td, groups, only=None):
    """-> {name: None | error text} for the checks in `groups`"""
    c = Ctx(T, td)
    res = {}
    for name, (g, fn) in CHECKS.items():
        if g not in groups or (only is not None and name not in only):
            continue
        try:
            fn(c)
            res[name] = None
        except BaseException as e:  # noqa: BLE001
            res[name] = f"{type(e).__name__}: {str(e)[:240]}"
    return res


def groups_for(expect):
    return {"light"} | {CHECKS[n][0] for n in expect}


def mutation_job(args):
    i, name, edits, expect, src = args
    src = mutated_source(src, name, edits)
    unknown = [n for n in expect if n not in CHECKS]
    assert not unknown, f"mutation '{name}' expects checks that do not exist: {unknown}"
    global CLI_MUT_SRC
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as md:
        mp = os.path.join(md, "mutated_navtest_E.py")
        open(mp, "w", encoding="utf-8", newline="\n").write(src)
        CLI_MUT_SRC = mp
        try:
            T = load_tool(src)
            t0 = time.time()
            res = run_checks(T, td, groups_for(expect))
        finally:
            CLI_MUT_SRC = None
    red = sorted(k for k, v in res.items() if v is not None)
    return i, name, red, sorted(expect), sorted(groups_for(expect)), time.time() - t0, {k: res[k] for k in red[:4]}


def mutated_source(src, name, edits):
    for old, new in edits:
        assert src.count(old) == 1, f"mutation anchor x{src.count(old)}: {name}: {old[:90]!r}"
        src = src.replace(old, new)
    compile(src, f"<mutation {name}>", "exec")                       # a mutation that does not even compile would be RED for the wrong reason
    return src


def main():
    t_all = time.time()
    src = TOOL_PATH.read_text(encoding="utf-8")
    for m in MUTATIONS:
        mutated_source(src, m[0], m[1])
    if "--anchors" in sys.argv:
        print(f"ZZANCHORS_OK {len(MUTATIONS)} mutations: every anchor occurs exactly once and every mutated source compiles")
        return 0
    sha = hashlib.sha256(src.encode()).hexdigest()
    names = [n for n in CHECKS]
    # sanity of the mutation table itself, before anything runs
    for m in MUTATIONS:
        assert all(n in CHECKS for n in m[2]), f"mutation '{m[0]}' names a check that does not exist"
        assert m[2], f"mutation '{m[0]}' has no expected RED location"
    reader_sha = hashlib.sha256(open(f"{O}/outcome/safeobj_outcome.py", "rb").read()).hexdigest()
    print(f"navtest_E.py sha256 {sha}\nsafeobj_outcome.py (REAL reader, read-only) sha256 {reader_sha}\n{len(names)} checks, {len(MUTATIONS)} RED mutations", flush=True)
    n_pass = 0
    keep_going = "--keep-going" in sys.argv                         # development: collect every problem instead of stopping at the first
    problems = []
    # ---- 1. the unmutated tool: every check of every group
    if "--skip-unmutated" not in sys.argv:
        with tempfile.TemporaryDirectory() as td:
            T = load_tool(src)
            t0 = time.time()
            res = run_checks(T, td, {g for g, _ in CHECKS.values()})
            bad = {k: v for k, v in res.items() if v is not None}
            for k, v in res.items():
                print(f"  {'PASS' if v is None else 'FAIL'}  {k}" + ("" if v is None else f"   <- {v}"), flush=True)
            print(f"unmutated suite: {len(res) - len(bad)}/{len(res)} checks green in {time.time() - t0:.0f} s", flush=True)
            assert not bad, f"the UNMUTATED tool fails {len(bad)} check(s): {sorted(bad)}"
            n_pass += 1
    if "--unmutated" in sys.argv:                                  # development: the unmutated suite only
        print(f"ZZTEST_UNMUTATED_PASS {len(names)} checks (NOT the full form: the RED mutations did not run)")
        return 0
    # ---- 2. the mutations (at most 2 processes)
    jobs = [(i, m[0], m[1], m[2], src) for i, m in enumerate(MUTATIONS)]
    done = []
    with cf.ProcessPoolExecutor(max_workers=2) as ex:
        for i, name, red, expect, groups, secs, why in ex.map(mutation_job, jobs):
            missing = [e for e in expect if e not in red]
            if keep_going and (not red or missing):
                problems.append(name)
                print(f"  !!!  {name}: " + ("stayed GREEN" if not red else f"RED but NOT at {missing}; got {red}; reasons {why}"), flush=True)
                continue
            assert red, f"mutation '{name}' stayed GREEN"
            assert not missing, f"mutation '{name}' went RED but NOT where it exists to: expected {expect}, missing {missing}; got {red}; first reasons {why}"
            print(f"  RED  {name}  ->  {len(red)} check(s) [{','.join(groups)}] {secs:.0f}s: {', '.join(red[:3])}{' ...' if len(red) > 3 else ''}", flush=True)
            done.append(name)
            n_pass += 1
    if problems:
        print(f"{len(problems)} mutation(s) with a problem: {problems}")
        return 1
    assert len(done) == len(MUTATIONS)
    print(f"ZZTEST_PASS {n_pass}  (1 unmutated suite of {len(names)} checks incl. the CLI end to end, {len(MUTATIONS)} RED mutations)  total {time.time() - t_all:.0f} s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--write-example":
        # development helper: run the real pinned CLI build on the P2REP stand-in tables into DIR (the example files of the package)
        dest = Path(sys.argv[2])
        with tempfile.TemporaryDirectory() as td:
            T = load_tool(TOOL_PATH.read_text(encoding="utf-8"))
            c = Ctx(T, td)
            args, out, work = cli_world(c, True)
            rc, txt = run_cli(args, td)
            print(txt[-1500:])
            assert rc == 0
            dest.mkdir(parents=True, exist_ok=True)
            for f in os.listdir(out):
                shutil.copy(os.path.join(out, f), dest / f"FIXTURE_P2REP_{f}")
        sys.exit(0)
    sys.exit(main())
