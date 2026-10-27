#!/usr/bin/env python3
"""PREREG_V8F_DAG_OACC and PREREG_V8F_FLATDAG on box 2: the DECISIONS the two chains make on files, in one place so a test can run both sides of each. DERIVED from the DAG2 build's v8f_dag2_chain_checks.py by make_helpers.py
(the bank judge and the bank-file code are the DAG2 ones; only the names that said DAG say generator now, and the DAG-read predicate is gone).

    argv-sha LINEFILE / argv-same GATE TRAIN        as DAG2 (launch_gate.argv_sha256 of the training argv; the gate line == the train line with --out and --selfcheck-only changed)
    bank-judge --banked-sha0 S --banked-sha1 S --seed0-dir D0 --seed1-dir D1      PREREG_V8F_DAGGER sections 6-7 on two PULLED banks (K-B0..K-B5, COVERAGE >= 0.5 inclusive); exit 0 PASS | 3 BLOCKED | 2 REFUSED. K-B2 reads the
                                                                                   GENERATOR's checkpoint: for the FLAT bank that is FLAT_s<seed>
    bank-files DIR [--kind ...] / bank-verify DIR [--require-sums] / bank-plan   the published bank files
    wacc-check --box w_acc.json --offline wacc_offline_DAG_s0.json [--offline-sha SHA] [--txt w_acc.txt] [--expect-ckpt-sha SHA] [--rtol 1e-4]
        C-O3: the box-2 value of the w_acc rule equals the offline recomputation on DAG_s0's banked fit plans to within 1e-4 relative (w_acc, L_traj, L_acc, L_ade, L_fde; n_windows exactly), the box record names the rule's
        constants (frac 0.5, w_traj 1.0), its model-route cross-check against the banked fit windows PASSED on the same windows, its checkpoint is DAG_s0's, and w_acc.txt is the bare number. exit 0 = equal | 2 = REFUSED.
    init-match TRAINER.py       the F1 lever-in-tip check: INIT_CKPT_MATCH (read by ast, whatever the formatting) names o_interface. exit 0 | 2
    disk-check --dir D --size N [--floor B] [--free-bytes F]      the disk floor: free(D) - N >= B (default 1 GiB). exit 0 | 2. Prints `free=... size=... after=... floor=...`
    pred-verdict --out LOG --busy FILE           the predecessor rule: prints GO | WAIT | BUSY | HALT and a reason (GO: ZZV8FMINI-DONE with BUSY released, or a ZZV8FMINI-STOP with no ZZGUARD-STOP line and BUSY released; BUSY: a stop whose
                                                 BOX2_BUSY.txt is still there (the caller retries a few times, then HALTs); HALT: a guard stop, or a DONE with BUSY still held; WAIT: no end marker yet)
    pin-sha FILE                 sha256 of a file
"""
from __future__ import annotations

import argparse
import ast
import shutil
import hashlib
import json
import math
import os
import re
import sys

SCHEMA = "v8f-dagger-bank-v1"
ROLLOUT = {"replan_steps": 10, "limit": "causal", "nav": "deadreckon", "scene": "logged", "schedule": "periodic", "executor": "plan", "post_map": "none (PLAIN)"}
COVERAGE_MIN = 0.5


def sha256_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


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
    ctl("K-B2 on-policy: the bank's checkpoint is the banked GENERATOR's (DAG_s<seed> / FLAT_s<seed>)", common.get("ckpt_sha256") == banked_sha, f"{str(common.get('ckpt_sha256'))[:12]} != {banked_sha[:12]}")
    tr = common.get("train") or {}
    ctl("TRAIN list is M2b's (clips, windows; the generator's data flags are M2b's)", tr.get("n_clips") == n_clips and tr.get("n_windows") == n_windows and tr.get("exclude") == "540:600", f"{tr.get('n_clips')} clips / {tr.get('n_windows')} windows / exclude {tr.get('exclude')!r}")
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
    print("ZZOAF-JUDGE " + json.dumps(res, sort_keys=True))
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


# ---------------------------------------------------------------------------------------------------------------- C-O3, the F1 lever, the disk floor, the predecessor rule
WACC_REL_TOL = 1e-4


def wacc_check(box_json: str, offline_json: str, offline_sha: str = "", txt: str = "", expect_ckpt_sha: str = "", rtol: float = WACC_REL_TOL) -> tuple:
    """(rc, record). C-O3 (PREREG_V8F_DAG_OACC section 3): the box-2 w_acc record against the offline recomputation on DAG_s0's banked fit plans. Anything that cannot be read or does not match is REFUSED (rc 2)."""
    bad, rec = [], {}
    try:
        if offline_sha and sha256_file(offline_json) != offline_sha.lower():
            bad.append(f"the offline record {offline_json} hashes to {sha256_file(offline_json)[:16]}, not the pinned {offline_sha[:16]}")
        b, o = _j(box_json), _j(offline_json)
    except Exception as e:                                                  # noqa: BLE001
        return 2, {"outcome": "REFUSED", "bad": [f"cannot read the records: {type(e).__name__}: {e}"]}
    for k in ("w_acc", "L_traj", "L_acc", "L_ade", "L_fde"):
        bv, ov = b.get(k), o.get(k)
        if not (_num(bv) and _num(ov) and ov != 0):
            bad.append(f"{k}: box {bv!r} / offline {ov!r} are not comparable numbers")
            continue
        rel = abs(bv - ov) / abs(ov)
        rec[k] = {"box": bv, "offline": ov, "rel": rel}
        if rel > rtol:
            bad.append(f"{k}: box {bv!r} vs offline {ov!r}: relative difference {rel:.3g} > {rtol}")
    if b.get("n_windows") != o.get("n_windows") or b.get("n_windows") != 9582:
        bad.append(f"n_windows box {b.get('n_windows')!r} / offline {o.get('n_windows')!r} (must be equal, 9,582)")
    if (b.get("frac"), b.get("w_traj")) != (0.5, 1.0) or b.get("schema") != "v8f-oacc-wacc-v1":
        bad.append(f"the box record does not carry the rule's constants: frac {b.get('frac')!r} w_traj {b.get('w_traj')!r} schema {b.get('schema')!r}")
    bc = b.get("banked_crosscheck")
    if not isinstance(bc, dict) or bc.get("pass") is not True or bc.get("same_windows") is not True:
        bad.append(f"the model route's cross-check against DAG_s0's banked fit windows did not pass on the same windows: {bc if not isinstance(bc, dict) else {k: bc.get(k) for k in ('pass', 'same_windows')}}")
    elif o.get("source", {}).get("fit_windows_sha256") and bc.get("fit_windows_sha256") != o["source"]["fit_windows_sha256"]:
        bad.append("the cross-check ran against another banked fit-windows file than the offline recomputation's")
    src = b.get("source") or {}
    if expect_ckpt_sha and src.get("ckpt_sha256") != expect_ckpt_sha.lower():
        bad.append(f"the rule was evaluated with checkpoint {str(src.get('ckpt_sha256'))[:16]}, not DAG_s0's {expect_ckpt_sha[:16]}")
    if txt:
        try:
            tv = open(txt, encoding="ascii", newline="").read()
        except OSError as e:
            tv = None
            bad.append(f"w_acc.txt unreadable: {e}")
        if tv is not None and tv != f"{b.get('w_acc')!r}\n":
            bad.append(f"w_acc.txt {tv!r} is not the bare number {b.get('w_acc')!r} + LF (what the queue reads into --w-acc)")
    rec.update({"outcome": "REFUSED" if bad else "PASS", "bad": bad, "rtol": rtol})
    return (2 if bad else 0), rec


def init_match_has(src_text: str, key: str = "o_interface") -> tuple:
    """(ok, why): the module-level assignment INIT_CKPT_MATCH is a tuple of string literals that contains ``key`` (read by ast: formatting and comments do not matter)."""
    try:
        tree = ast.parse(src_text)
    except SyntaxError as e:
        return False, f"the trainer does not parse: {e}"
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "INIT_CKPT_MATCH" for t in n.targets):
            try:
                v = ast.literal_eval(n.value)
            except Exception as e:                                          # noqa: BLE001
                return False, f"INIT_CKPT_MATCH is not a literal: {type(e).__name__}"
            return (key in v), (f"INIT_CKPT_MATCH has {len(v)} keys and {'names' if key in v else 'does NOT name'} {key}")
    return False, "no INIT_CKPT_MATCH assignment"


FLOOR_BYTES = 1 << 30                                                       # 1 GiB: a pull must leave at least this much free on the destination drive


def disk_check(d: str, size: int, floor: int = FLOOR_BYTES, free=None) -> tuple:
    f = shutil.disk_usage(d).free if free is None else int(free)
    after = f - int(size)
    return (0 if after >= floor else 2), f"free={f} size={int(size)} after={after} floor={int(floor)}"


def pred_verdict(out_text: str, busy_exists: bool) -> tuple:
    """The predecessor rule (the brief): a predecessor STOP must not idle box 2. The arms are independent, so go when the predecessor ended with DONE, or with a STOP of its own read / judge / run, once its BOX2_BUSY is
    released. Halt on a guard stop (the box may be hot) or a BOX2_BUSY that is not released. Markers are ANCHORED at the start of a line (a 'waiting for ^ZZV8FMINI-STOP' echo starts with a date)."""
    lines = out_text.replace("\r", "").split("\n")
    done = [x for x in lines if x.startswith("ZZV8FMINI-DONE")]
    stop = [x for x in lines if x.startswith("ZZV8FMINI-STOP")]
    guard = [x for x in lines if "ZZGUARD-STOP" in x]
    if done:
        return ("HALT", "the predecessor ended DONE but BOX2_BUSY.txt is still there") if busy_exists else ("GO", done[-1][:200])
    if stop:
        if guard:
            return "HALT", "the predecessor was stopped by the GUARD (" + guard[-1][:160] + "): the box may be hot"
        if busy_exists:
            return "BUSY", "the predecessor stopped but its BOX2_BUSY.txt is not released (yet)"
        return "GO", "the predecessor STOPPED on its own (" + stop[-1][:160] + ") and released the box: the arms are independent"
    return "WAIT", "no end marker yet"


def json_same(a: str, b: str) -> str:
    """BYTES (identical bytes) | JSON (different bytes, equal parsed JSON: line endings / spacing) | DIFFERENT. FLAT's argv.json / summary.json are pinned by the sha256 of the REPO's blob; box 2 wrote its own copies."""
    ba, bb = open(a, "rb").read(), open(b, "rb").read()
    if ba == bb:
        return "BYTES"
    try:
        return "JSON" if json.loads(ba.decode("utf-8")) == json.loads(bb.decode("utf-8")) else "DIFFERENT"
    except Exception:                                                       # noqa: BLE001
        return "DIFFERENT"


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
    w = sub.add_parser("wacc-check")
    w.add_argument("--box", required=True)
    w.add_argument("--offline", required=True)
    w.add_argument("--offline-sha", default="")
    w.add_argument("--txt", default="")
    w.add_argument("--expect-ckpt-sha", default="")
    w.add_argument("--rtol", type=float, default=WACC_REL_TOL)
    im = sub.add_parser("init-match")
    im.add_argument("trainer")
    dc = sub.add_parser("disk-check")
    dc.add_argument("--dir", required=True)
    dc.add_argument("--size", type=int, required=True)
    dc.add_argument("--floor", type=int, default=FLOOR_BYTES)
    dc.add_argument("--free-bytes", default="")
    pv = sub.add_parser("pred-verdict")
    pv.add_argument("--out", required=True)
    pv.add_argument("--busy", required=True)
    sub.add_parser("pin-sha").add_argument("file")
    js = sub.add_parser("json-same")
    js.add_argument("a")
    js.add_argument("b")
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
        print("ZZOAF-BANKFILES " + json.dumps(res, sort_keys=True))
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
    if a.mode == "wacc-check":
        rc, rec = wacc_check(a.box, a.offline, a.offline_sha, a.txt, a.expect_ckpt_sha, a.rtol)
        print("ZZDAGOACC-CO3 " + json.dumps(rec, sort_keys=True))
        return rc
    if a.mode == "init-match":
        ok, why = init_match_has(open(a.trainer, encoding="utf-8").read())
        print(why)
        return 0 if ok else 2
    if a.mode == "disk-check":
        rc, msg = disk_check(a.dir, a.size, a.floor, a.free_bytes or None)
        print(msg)
        return rc
    if a.mode == "pred-verdict":
        try:
            txt = open(a.out, encoding="utf-8", errors="replace").read() if os.path.exists(a.out) else ""
        except OSError:
            txt = ""
        v, why = pred_verdict(txt, os.path.exists(a.busy))
        print(v, why)
        return 0
    if a.mode == "json-same":
        v = json_same(a.a, a.b)
        print(v)
        return 0 if v != "DIFFERENT" else 1
    if a.mode == "pin-sha":
        print(sha256_file(a.file))
        return 0
    return bank_judge(a)[0]


if __name__ == "__main__":
    sys.exit(main())
