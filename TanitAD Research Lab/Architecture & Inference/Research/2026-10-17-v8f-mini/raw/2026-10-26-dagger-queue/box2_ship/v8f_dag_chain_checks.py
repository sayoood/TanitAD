#!/usr/bin/env python3
"""PREREG_V8F_DAGGER on box 2: the two DECISIONS the chain makes on files, in one place so a test can run both sides of each (+ two argv helpers).

    argv-sha LINEFILE          launch_gate.argv_sha256 of the TRAINING argv of a printed queue line (everything after v8f_mini_train.py; double quotes group and are removed, nothing else is interpreted)
    argv-same GATE TRAIN       SAME iff gate argv minus (--out <v>, --selfcheck-only) == train argv minus (--out <v>), with --selfcheck-only exactly once in the gate line; else DIFFERENT

    orc-predicate ORC_READ.json
        PREREG_V8F_DAGGER opening section and section 7: the arm runs iff O-RECOVER read NO EFFECT (p = 0.3) or HARMS (p = 0.1). ROBUST / PARTIAL: parked, not run. REFUSED / UNPLACED / anything else: wait for ORC's re-run.
        Reads OUTCOME.OUTCOME (the reader's own table row), requires claim_row H-V8F-ORECOVER-1, every control in controls_PASS true, and exactly one matching table row. Prints `<OUTCOME> <p> <sha256 of the file>`.
        exit 0 = run (p printed) | 4 = PARKED (ROBUST / PARTIAL) | 2 = not a usable read (REFUSED, UNPLACED, a failed control, a foreign file, unreadable)

    bank-judge --banked-sha0 S --banked-sha1 S --seed0-dir D0 --seed1-dir D1 [--expect-clips 4512] [--expect-windows 306566]
        PREREG_V8F_DAGGER sections 6 and 7 on the two PULLED banks (bank.json, PROVENANCE.json, shard_KKKofNNN.provenance.json in each dir). Order of precedence is the PREREG table's: a failed CONTROL is REFUSED
        (fix and re-run), K-B4 and K-B5 included: PROVENANCE.json controls["K-B4"]["PASS"] is True with all six cases True, and controls["K-B5"]["PASS"] is True with max_pose_now_err_m and
        max_rel_transform_err_m both finite and <= 1e-4 (dag_read.py g_kb4 / g_kb5), a MISSING record being a failed control; only with every control passing does COVERAGE decide: the eligible fraction of TRAIN windows >= 0.5 (inclusive) at BOTH seeds, else BLOCKED (no gate, no training). The fraction is recomputed
        here from n_eligible / n_windows; the bank's own `blocked` flag must agree with it.
        exit 0 = PASS | 3 = BLOCKED | 2 = REFUSED. Everything it saw is printed as one JSON line `ZZDAG-JUDGE {...}`.

    bank-files DIR [--kind all|small|provenance|manifest|shard_provenance|shard_npz|stage_log] [--expect-shards 4]
        The file names the chain PUBLISHES to the dev box for the bank in DIR, one per line, derived from DIR/bank.json (n_shards, shards[].file) and DIR/PROVENANCE.json (shard_provenance): the FOUR kinds the dev-box reader
        (dag_read.py load_bank) needs -- PROVENANCE.json, bank.json, the shard provenance files, the shard .npz files (the only place the bank's clips are named: K-B3.1) -- and bank_stage.log. `small` = all but the npz (what the
        judge reads). rc 2 + a reason on stderr if the bank cannot be named (unreadable, n_shards != 4, names that are not the generator's).
    bank-verify DIR [--require-sums] [--expect-shards 4]
        The PUBLISHED directory, by content: every file present and non-empty, bank.json == PROVENANCE.bank_sha256, every shard npz == bank.json's and its shard provenance's file_sha256; with --require-sums SHA256SUMS.txt must list
        EXACTLY those files (11 for 4 shards) with the right hashes. One JSON line `ZZDAG-BANKFILES {...}`; rc 0 / 2.
    bank-plan
        The planned file names and the ESTIMATED sizes (printed by the chain's --dry-run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

SCHEMA = "v8f-dagger-bank-v1"
ROLLOUT = {"replan_steps": 10, "limit": "causal", "nav": "deadreckon", "scene": "logged", "schedule": "periodic", "executor": "plan", "post_map": "none (PLAIN)"}
COVERAGE_MIN = 0.5
P_OF = {"HARMS": "0.1", "NO EFFECT": "0.3"}
PARKED = ("ROBUST", "PARTIAL")


def sha256_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def orc_predicate(path) -> tuple:
    """(rc, message). message = `<OUTCOME> <p> <sha256>` on rc 0."""
    try:
        d = json.load(open(path, encoding="utf-8"))
        sha = sha256_file(path)
    except Exception as e:                                                  # noqa: BLE001
        return 2, f"REFUSED: {path} is not a readable ORC read ({type(e).__name__}: {e})"
    o = d.get("OUTCOME")
    name = (o.get("OUTCOME") if isinstance(o, dict) else o)
    if not isinstance(name, str):
        return 2, f"REFUSED: {path} carries no OUTCOME row ({o!r})"
    name = name.strip().upper().replace("_", " ")
    if d.get("claim_row") != "H-V8F-ORECOVER-1":
        return 2, f"REFUSED: {path} is not O-RECOVER's read (claim_row {d.get('claim_row')!r})"
    ctl = d.get("controls_PASS")
    if not isinstance(ctl, dict) or not ctl or not all(v is True for v in ctl.values()):
        failed = [k for k, v in (ctl or {}).items() if v is not True]
        return 2, f"REFUSED: O-RECOVER's read has failed or missing controls {failed[:6]}: a read with a failed control is not an outcome"
    rows = (o.get("matching_rows_in_table_order") if isinstance(o, dict) else None)
    if rows is not None and [str(x).strip().upper().replace("_", " ") for x in rows] != [name]:
        return 2, f"REFUSED: O-RECOVER's read matches the table rows {rows}, not exactly one ({name})"
    if name in PARKED:
        return 4, f"PARKED: O-RECOVER read {name}: PREREG_V8F_DAGGER is parked, not run (sha256 {sha})"
    if name not in P_OF:
        return 2, f"REFUSED: O-RECOVER read {name!r}: the arm runs only after NO EFFECT (p 0.3) or HARMS (p 0.1); wait for the re-run / the amendment"
    return 0, f"{name} {P_OF[name]} {sha}"


def argv_of(text: str) -> list:
    import re
    t = [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', text.strip())]
    return t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]


def argv_sha256(argv) -> str:
    return hashlib.sha256(json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _drop(x, bare=()):
    o, k = [], 0
    while k < len(x):
        if x[k] == "--out":
            k += 2
        elif x[k] in bare:
            k += 1
        else:
            o.append(x[k])
            k += 1
    return o


def argv_same(gate_text: str, train_text: str) -> bool:
    g, t = argv_of(gate_text), argv_of(train_text)
    return _drop(g, {"--selfcheck-only"}) == _drop(t) and g.count("--selfcheck-only") == 1 and "--selfcheck-only" not in t


def _j(p):
    return json.load(open(p, encoding="utf-8"))


KB4_CASES = ("straight", "stationary_tie", "lateral_cut", "heading_cut", "end_edge", "loop")      # dag_read.py KB4_CASES (PREREG section 6, K-B4)
KB5_TOL_M = 1e-4                                                                                        # dag_read.py KB5_TOL_M (K-B5)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def kb_record_problem(c, kid: str):
    """dag_read.py kb_record: the record exists, is a dict and carries PASS exactly True. None = fine, else why not."""
    if not isinstance(c, dict):
        return f"PROVENANCE.json carries no controls.{kid} record (the generator does not write it): the chain STOPS"
    if c.get("PASS") is not True:
        return f"{kid} is recorded PASS = {c.get('PASS')!r}"
    return None


def kb4_problem(c):
    """dag_read.py g_kb4: every one of the six synthetic cases recorded exactly True."""
    if not isinstance(c, dict):
        return None                                                          # already reported by kb_record_problem
    cases = c.get("cases") if isinstance(c.get("cases"), dict) else {}
    miss = [k for k in KB4_CASES if cases.get(k) is not True]
    return f"every case must be recorded True; not True / absent: {miss}" if miss else None


def kb5_problem(c):
    """dag_read.py g_kb5: max_pose_now_err_m and max_rel_transform_err_m are finite numbers <= 1e-4."""
    if not isinstance(c, dict):
        return None
    for k in ("max_pose_now_err_m", "max_rel_transform_err_m"):
        if not (_num(c.get(k)) and c[k] <= KB5_TOL_M):
            return f"{k} must be recorded <= {KB5_TOL_M}: {c.get(k)!r}"
    return None


def judge_seed(d: str, seed: int, banked_sha: str, n_clips: int, n_windows: int) -> dict:
    """One seed's bank: {controls: {name: bool}, coverage: {...}, bad: [why a control failed]}. Nothing here raises on a malformed file: a malformed bank is a failed control."""
    r = {"seed": seed, "controls": {}, "bad": [], "coverage": None, "extra_controls_in_provenance": None}

    def ctl(name, ok, why=""):
        r["controls"][name] = bool(ok)
        if not ok:
            r["bad"].append(f"{name}: {why}")
    try:
        man, prov = _j(os.path.join(d, "bank.json")), _j(os.path.join(d, "PROVENANCE.json"))
        common = man["common"]
        sh = [_j(os.path.join(d, f"shard_{k:03d}of{man['n_shards']:03d}.provenance.json")) for k in range(man["n_shards"])]
    except Exception as e:                                                  # noqa: BLE001
        ctl("BANK-FILES", False, f"{type(e).__name__}: {e}")
        return r
    ctl("BANK-FILES", True)
    ctl("SCHEMA", man.get("schema") == SCHEMA and common.get("schema") == SCHEMA, f"{man.get('schema')!r}")
    ctl("COMPLETE", man.get("partial") is False and common.get("partial") is False and common.get("limit_clips") in (0, None), f"partial {man.get('partial')} limit_clips {common.get('limit_clips')}")
    ctl("SHARDS", man["n_shards"] >= 1 and len(sh) == man["n_shards"] and sorted(s["k"] for s in man["shards"]) == list(range(man["n_shards"])), f"n_shards {man['n_shards']}")
    ctl("ROLLOUT=PREREG-1", common.get("rollout") == ROLLOUT, f"{common.get('rollout')!r}")
    ctl("K-B2 on-policy: the bank's checkpoint is the banked M2b", common.get("ckpt_sha256") == banked_sha, f"{str(common.get('ckpt_sha256'))[:12]} != {banked_sha[:12]}")
    tr = common.get("train") or {}
    ctl("TRAIN list is M2b's (clips, windows)", tr.get("n_clips") == n_clips and tr.get("n_windows") == n_windows and tr.get("exclude") == "540:600", f"{tr.get('n_clips')} clips / {tr.get('n_windows')} windows / exclude {tr.get('exclude')!r}")
    ev = common.get("eval") or {}
    ctl("K-B3 eval-141 disjoint", ev.get("overlap") == 0 and all((p.get("kb3_eval_disjoint") or {}).get("overlap") == 0 for p in sh) and (ev.get("n_clips") or 0) > 0, f"overlap {ev.get('overlap')} n_eval {ev.get('n_clips')}")
    ctl("K-B0 tick-0 identity on every window", all((p.get("kb0_tick0_identity") or {}).get("mismatch") == 0 for p in sh) and sum((p.get("kb0_tick0_identity") or {}).get("checked", 0) for p in sh) == n_windows,
        f"checked {sum((p.get('kb0_tick0_identity') or {}).get('checked', 0) for p in sh)} mismatch {[(p.get('kb0_tick0_identity') or {}).get('mismatch') for p in sh]}")
    ctl("K-B1 replay bit-exact", all((p.get("kb1_replay") or {}).get("mismatch_chunks") == 0 and (p.get("kb1_replay") or {}).get("chunks", 0) >= 1 for p in sh), f"{[p.get('kb1_replay') for p in sh]}")
    r["warnings"] = [w for p in sh for w in (p.get("warnings") or [])]        # REPORTED (a NONFINITE count is a cut, not a PREREG control; a K-B1 failure is already caught above)
    cov = man.get("coverage") or {}
    n_w, n_e = cov.get("n_windows"), cov.get("n_eligible")
    ok_cov_shape = isinstance(n_w, int) and isinstance(n_e, int) and n_w == n_windows and 0 <= n_e <= n_w
    ctl("COVERAGE record is well-formed (denominator = the TRAIN window count)", ok_cov_shape, f"{cov}")
    if ok_cov_shape:
        frac = n_e / n_w
        ctl("COVERAGE `blocked` flag agrees with n_eligible / n_windows < 0.5", cov.get("blocked") is (frac < COVERAGE_MIN) and cov.get("min") == COVERAGE_MIN, f"flag {cov.get('blocked')} frac {frac}")
        ctl("PROVENANCE.json agrees with bank.json on coverage", (prov.get("coverage") or {}).get("n_eligible") == n_e and (prov.get("coverage") or {}).get("n_windows") == n_w, f"{prov.get('coverage')}")
        r["coverage"] = {"n_windows": n_w, "n_eligible": n_e, "fraction": frac, "min": COVERAGE_MIN, "blocked": frac < COVERAGE_MIN}
    r["bank_sha256"] = sha256_file(os.path.join(d, "bank.json"))
    # K-B4 / K-B5: GATED (Master Mind 2026-10-26, O6), by the reader's own rules (dag_read.py g_kb4 / g_kb5 / kb_record). A missing record is a failed control (REFUSED, the chain STOPS), never a skipped one.
    kbs = prov.get("controls") if isinstance(prov.get("controls"), dict) else {}
    kb4, kb5 = kbs.get("K-B4"), kbs.get("K-B5")
    why4 = kb_record_problem(kb4, "K-B4") or kb4_problem(kb4)
    why5 = kb_record_problem(kb5, "K-B5") or kb5_problem(kb5)
    ctl("K-B4 matcher synthetic cases (PROVENANCE.json controls.K-B4: PASS True and all six cases True)", why4 is None, why4 or "")
    ctl("K-B5 re-anchoring (PROVENANCE.json controls.K-B5: PASS True, both errors <= 1e-4 m)", why5 is None, why5 or "")
    r["extra_controls_in_provenance"] = prov.get("controls")             # the whole controls block, REPORTED (any key beyond K-B4 / K-B5 is gated on nothing here)
    return r


def bank_judge(a) -> tuple:
    seeds = [judge_seed(a.seed0_dir, 0, a.banked_sha0, a.expect_clips, a.expect_windows), judge_seed(a.seed1_dir, 1, a.banked_sha1, a.expect_clips, a.expect_windows)]
    refused = [b for r in seeds for b in r["bad"]]
    blocked = [r["seed"] for r in seeds if r["coverage"] and r["coverage"]["blocked"]]
    if refused:
        outcome, rc = "REFUSED", 2
    elif blocked:
        outcome, rc = "BLOCKED", 3
    else:
        outcome, rc = "PASS", 0
    res = {"outcome": outcome, "blocked_seeds": blocked, "coverage_min": COVERAGE_MIN, "seeds": seeds, "refused_because": refused}
    print("ZZDAG-JUDGE " + json.dumps(res, sort_keys=True))
    return rc, res


# ---------------------------------------------------------------------------------------------------------------- the bank files the chain PUBLISHES to the dev box
# The dev-box reader (dag_read.py load_bank) needs, per seed, FOUR kinds of file in one directory: PROVENANCE.json (`bank.{s}.report`), bank.json (`bank_file`; its sha256 IS the bank's identity), the shard provenance
# files, and the shard .npz files (their `header_json` member is the ONLY place the bank's clips are listed by name: K-B3.1). The stage log is published beside them (audit; the reader does not read it).
# The judge (above) reads only the small files, so the chain pulls those first, judges, and pulls the shard npz ONLY after the judge passed (a REFUSED / BLOCKED bank is never moved).
BANK_SHARDS = 4                                       # `bankdag` runs `bank --n-shards 4` (mutant 14 of mutate_dag_queue.py pins it)
BANK_KINDS = ("provenance", "manifest", "shard_provenance", "shard_npz", "stage_log")
# ESTIMATE basis: the 12-clip smoke bank (smoke_kb45: 804 windows, 1,608 records = 2 ticks per window, ONE shard): shard npz 1,750,642 B, of which header_json 7,774 B; PROVENANCE.json 1,809 B; bank.json 9,079 B; shard provenance 5,722 B.
SMOKE_BANK = {"windows": 804, "records": 1608, "npz_bytes": 1750642, "npz_header_bytes": 7774, "npz_zip_overhead_bytes": 1476, "provenance_bytes": 1809, "bank_json_bytes": 9079, "shard_prov_bytes": 5722}


class BankFilesError(Exception):
    pass


def bank_names(n: int) -> dict:
    """{kind: [file names]} of a bank of n shards, in the generator's naming (v8f_dagger_bank.py: shard_KKKofNNN.{npz,provenance.json})."""
    return {"provenance": ["PROVENANCE.json"], "manifest": ["bank.json"],
            "shard_provenance": [f"shard_{k:03d}of{n:03d}.provenance.json" for k in range(n)],
            "shard_npz": [f"shard_{k:03d}of{n:03d}.npz" for k in range(n)], "stage_log": ["bank_stage.log"]}


def sha256_stream(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bank_published(d: str, expect_shards=BANK_SHARDS) -> tuple:
    """(names, man, prov) of the bank in directory ``d``: the file names come from bank.json (n_shards, shards[].file) and PROVENANCE.json (shard_provenance), and must be the generator's canonical ones for that n.
    Raises BankFilesError with the reason (a bank this function cannot name has nothing to publish)."""
    try:
        man, prov = _j(os.path.join(d, "bank.json")), _j(os.path.join(d, "PROVENANCE.json"))
    except Exception as e:                                                  # noqa: BLE001
        raise BankFilesError(f"cannot read bank.json / PROVENANCE.json in {d}: {type(e).__name__}: {e}")
    n = man.get("n_shards") if isinstance(man, dict) else None
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise BankFilesError(f"bank.json n_shards is {n!r}")
    if expect_shards is not None and n != expect_shards:
        raise BankFilesError(f"bank.json n_shards is {n}, the queue's bankdag runs {expect_shards}")
    names = bank_names(n)
    shards = man.get("shards")
    if not isinstance(shards, list) or [s.get("file") for s in sorted(shards, key=lambda s: s.get("k", -1))] != names["shard_npz"]:
        raise BankFilesError(f"bank.json shards name {[s.get('file') for s in shards] if isinstance(shards, list) else shards!r}, not {names['shard_npz']}")
    if not isinstance(prov, dict) or prov.get("shard_provenance") != names["shard_provenance"]:
        raise BankFilesError(f"PROVENANCE.json shard_provenance is {prov.get('shard_provenance') if isinstance(prov, dict) else prov!r}, not {names['shard_provenance']}")
    if prov.get("manifest", "bank.json") != "bank.json":
        raise BankFilesError(f"PROVENANCE.json names the manifest {prov.get('manifest')!r}, not bank.json")
    return names, man, prov


def bank_flat(names: dict, kinds=BANK_KINDS) -> list:
    return [n for k in kinds for n in names[k]]


def parse_sums(path: str) -> tuple:
    """(listed {name: sha256}, problem|None) of a sha256sum file: ends with a newline, every line `<64 hex>  <name>`, no name twice."""
    try:
        text = open(path, encoding="utf-8", errors="replace", newline="").read()
    except OSError as e:
        return {}, f"SHA256SUMS.txt unreadable: {e}"
    if not text.endswith("\n"):
        return {}, "SHA256SUMS.txt does not end with a newline (its last line may be partial)"
    listed = {}
    for raw in text.splitlines():
        m = re.match(r"^([0-9a-f]{64}) [ *](.+)$", raw.rstrip("\r"))
        if not m:
            return {}, f"SHA256SUMS.txt has a line that is not '<sha256>  <name>': {raw[:80]!r}"
        if m.group(2).strip() in listed:
            return {}, f"SHA256SUMS.txt lists {m.group(2).strip()} twice"
        listed[m.group(2).strip()] = m.group(1)
    return listed, None


def bank_verify(d: str, require_sums: bool = False, expect_shards=BANK_SHARDS) -> dict:
    """One seed's PUBLISHED bank directory, by content: all four kinds (+ the stage log) present and non-empty; bank.json hashes to PROVENANCE.bank_sha256; every shard npz hashes to bank.json's AND its shard
    provenance's file_sha256; with ``require_sums`` also SHA256SUMS.txt lists EXACTLY those files (no more, no fewer) with the right hashes. -> {ok, bad, n_files, bytes_by_kind, ...}. Never raises."""
    r = {"ok": False, "bad": [], "dir": d, "require_sums": bool(require_sums)}
    try:
        names, man, prov = bank_published(d, expect_shards)
    except BankFilesError as e:
        r["bad"].append(str(e))
        return r
    allnames = bank_flat(names)
    r["n_files"] = len(allnames)
    r["bytes_by_kind"] = {}
    for kind in BANK_KINDS:
        tot = 0
        for nm in names[kind]:
            p = os.path.join(d, nm)
            if not os.path.isfile(p):
                r["bad"].append(f"{kind}: {nm} is absent")
            elif os.path.getsize(p) == 0:
                r["bad"].append(f"{kind}: {nm} is empty")
            else:
                tot += os.path.getsize(p)
        r["bytes_by_kind"][kind] = tot
    if r["bad"]:
        return r
    msha = sha256_stream(os.path.join(d, "bank.json"))
    r["bank_sha256"] = msha
    if prov.get("bank_sha256") != msha:
        r["bad"].append(f"bank.json hashes to {msha[:12]}, not PROVENANCE.bank_sha256 {str(prov.get('bank_sha256'))[:12]}")
    by_k = {s.get("k"): s for s in man["shards"]}
    for k, (npz, pn) in enumerate(zip(names["shard_npz"], names["shard_provenance"])):
        got = sha256_stream(os.path.join(d, npz))
        if got != by_k[k].get("file_sha256"):
            r["bad"].append(f"shard npz {npz} hashes to {got[:12]}, not bank.json's {str(by_k[k].get('file_sha256'))[:12]}")
        try:
            sp = _j(os.path.join(d, pn))
        except Exception as e:                                              # noqa: BLE001
            r["bad"].append(f"shard provenance {pn} does not parse: {type(e).__name__}")
            continue
        if not isinstance(sp, dict) or sp.get("file_sha256") != got:
            r["bad"].append(f"shard provenance {pn} records file_sha256 {str(sp.get('file_sha256') if isinstance(sp, dict) else None)[:12]}, the npz hashes to {got[:12]}")
    if require_sums:
        sp_ = os.path.join(d, "SHA256SUMS.txt")
        if not os.path.isfile(sp_):
            r["bad"].append("SHA256SUMS.txt is absent")
        else:
            listed, why = parse_sums(sp_)
            if why:
                r["bad"].append(why)
            elif sorted(listed) != sorted(allnames):
                r["bad"].append(f"SHA256SUMS.txt lists {len(listed)} names, expected exactly {len(allnames)} (missing {sorted(set(allnames) - set(listed))}, unexpected {sorted(set(listed) - set(allnames))})")
            else:
                for nm in allnames:
                    h = msha if nm == "bank.json" else sha256_stream(os.path.join(d, nm))
                    if h != listed[nm]:
                        r["bad"].append(f"SHA256SUMS.txt lists {listed[nm][:12]} for {nm}, the file hashes to {h[:12]}")
    r["ok"] = not r["bad"]
    return r


def est_bank_bytes(n_windows: int = 306566, n_shards: int = BANK_SHARDS, clip_name_chars: int = 36, n_clips: int = 4512) -> dict:
    """ESTIMATED bytes of one seed's published bank: the shard arrays scale linearly with the number of records (2 ticks per window), the per-shard header with its clip names (real clip ids are UUIDs: 36 chars)."""
    s = SMOKE_BANK
    per_window = (s["npz_bytes"] - s["npz_header_bytes"] - s["npz_zip_overhead_bytes"]) / s["windows"]
    arrays = per_window * n_windows
    header = n_shards * s["npz_header_bytes"] + n_clips * (clip_name_chars + 8)            # constant block per shard + one `"idx": "name",` entry per clip
    npz = arrays + header + n_shards * s["npz_zip_overhead_bytes"]
    small = s["provenance_bytes"] + s["bank_json_bytes"] + n_shards * (s["shard_prov_bytes"] + 100) + 4096 + 1024      # + bank_stage.log (a few KB) + SHA256SUMS.txt
    return {"per_window_bytes": per_window, "npz_total_bytes": int(npz), "npz_each_bytes": int(npz / n_shards), "small_total_bytes": int(small), "seed_total_bytes": int(npz + small)}


def bank_plan(n_shards: int = BANK_SHARDS) -> list:
    names = bank_names(n_shards)
    est = est_bank_bytes()
    out = [f"kind {k}: {', '.join(names[k])}" for k in BANK_KINDS]
    out.append(f"SHA256SUMS.txt: exactly {len(bank_flat(names))} names (PROVENANCE.json 1 + bank.json 1 + shard provenance {n_shards} + shard npz {n_shards} + bank_stage.log 1), written LAST, after every shard npz verified")
    out.append(f"ESTIMATED size per seed (scaled from the 12-clip smoke, 804 windows -> 306,566): shard npz {est['npz_total_bytes'] / 1e9:.3f} GB ({est['npz_each_bytes'] / 1e6:.0f} MB each x {n_shards}), "
               f"small files {est['small_total_bytes'] / 1e3:.0f} KB, total {est['seed_total_bytes'] / 1e9:.3f} GB; both seeds {2 * est['seed_total_bytes'] / 1e9:.3f} GB")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    bf = sub.add_parser("bank-files")
    bf.add_argument("dir")
    bf.add_argument("--kind", default="all", help="all | small (everything but the shard npz: what the judge reads) | " + " | ".join(BANK_KINDS))
    bf.add_argument("--expect-shards", type=int, default=BANK_SHARDS)
    bv = sub.add_parser("bank-verify")
    bv.add_argument("dir")
    bv.add_argument("--require-sums", action="store_true")
    bv.add_argument("--expect-shards", type=int, default=BANK_SHARDS)
    sub.add_parser("bank-plan")
    sub.add_parser("argv-sha").add_argument("line_file")
    q = sub.add_parser("argv-same")
    q.add_argument("gate_file")
    q.add_argument("train_file")
    o = sub.add_parser("orc-predicate")
    o.add_argument("orc_read")
    b = sub.add_parser("bank-judge")
    b.add_argument("--banked-sha0", required=True)
    b.add_argument("--banked-sha1", required=True)
    b.add_argument("--seed0-dir", required=True)
    b.add_argument("--seed1-dir", required=True)
    b.add_argument("--expect-clips", type=int, default=4512)
    b.add_argument("--expect-windows", type=int, default=306566)
    a = ap.parse_args(argv)
    if a.mode == "bank-files":
        try:
            names = bank_published(a.dir, a.expect_shards)[0]
        except BankFilesError as e:
            print(f"bank-files: {e}", file=sys.stderr)
            return 2
        if a.kind == "all":
            sel = bank_flat(names)
        elif a.kind == "small":
            sel = bank_flat(names, [k for k in BANK_KINDS if k != "shard_npz"])
        elif a.kind in BANK_KINDS:
            sel = names[a.kind]
        else:
            print(f"bank-files: unknown --kind {a.kind!r}", file=sys.stderr)
            return 2
        sys.stdout.buffer.write(("\n".join(sel) + "\n").encode("utf-8"))        # LF bytes: a Windows text-mode print ends every line in CR LF, and the chain word-splits this output into file names
        sys.stdout.flush()
        return 0
    if a.mode == "bank-verify":
        res = bank_verify(a.dir, a.require_sums, a.expect_shards)
        print("ZZDAG-BANKFILES " + json.dumps(res, sort_keys=True))
        return 0 if res["ok"] else 2
    if a.mode == "bank-plan":
        print("\n".join(bank_plan()))
        return 0
    if a.mode == "argv-sha":
        print(argv_sha256(argv_of(open(a.line_file, encoding="utf-8").read())))
        return 0
    if a.mode == "argv-same":
        same = argv_same(open(a.gate_file, encoding="utf-8").read(), open(a.train_file, encoding="utf-8").read())
        print("SAME" if same else "DIFFERENT")
        return 0 if same else 1
    if a.mode == "orc-predicate":
        rc, msg = orc_predicate(a.orc_read)
        print(msg)
        return rc
    return bank_judge(a)[0]


if __name__ == "__main__":
    sys.exit(main())
