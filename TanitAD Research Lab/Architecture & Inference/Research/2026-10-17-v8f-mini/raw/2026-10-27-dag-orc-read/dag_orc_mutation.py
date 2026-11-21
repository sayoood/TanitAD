#!/usr/bin/env python3
"""dag_orc_mutation.py -- the mutation proof for dag_orc_read.py.

Each mutant is ONE hand-written textual defect in a COPY of the reader (anchor must occur exactly once, else ANCHOR ERROR).  A mutant is KILLED iff dag_orc_selftest.py (run with --reader <mutant>)
exits non-zero.  The UNMUTATED control must exit 0.  A mutant that does not parse is reported INVALID (not counted as killed).  At most 2 workers.

    python dag_orc_mutation.py [--workers 2] [--only ID,ID] [--out mutation.json]       exit 0 iff the control is green and every mutant is killed

Scope (said plainly): the mutants cover the PURE and RECORD-LEVEL code the self-test drives with synthetic inputs (the bar, clears / fires, the domain, the per-seed composition, the outcome
table and its decision, the cascade, the PREREG pin, the RF0 pin, K-PAIR / K-INIT / K-REACH / K-BUILD / K-P0 / identity / RC, slot discovery).  The heavy path (analyze_slot / read_core /
main_read / main_rc: the readers run on real dumps) is exercised on REAL banked data by `--rc-dag2` (K-RC) and `--dry-run`, not by mutation.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PY = sys.executable
SCR = "F:/TanitAD-scratch/dag_orc_read_tmp/mut"
TIMEOUT_S = 600

# (id, description, anchor, replacement)
MUTANTS = [
    # ---- the bar
    ("bar-drop-RF0", "bar ignores RF0", 'return 2.0 * max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]), abs(rf[label]))', 'return 2.0 * max(F_ADE6 if label == "ADE6" else 0.0, abs(rf[label]))'),
    ("bar-drop-RF", "bar ignores RF (the floor term deleted)", 'return 2.0 * max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]), abs(rf[label]))', 'return 2.0 * max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]))'),
    ("bar-drop-F", "bar ignores F on ADE6", 'return 2.0 * max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]), abs(rf[label]))', 'return 2.0 * max(0.0, abs(rf0[label]), abs(rf[label]))'),
    ("bar-factor-1", "bar factor 1 instead of 2", 'return 2.0 * max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]), abs(rf[label]))', 'return 1.0 * max(F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label]), abs(rf[label]))'),
    ("bar-F-everywhere", "F applied to every row", 'F_ADE6 if label == "ADE6" else 0.0, abs(rf0[label])', 'F_ADE6 if True else 0.0, abs(rf0[label])'),
    ("bar-no-abs-rf0", "RF0 enters signed", 'abs(rf0[label]), abs(rf[label]))', 'rf0[label], abs(rf[label]))'),
    ("bar-no-abs-rf", "RF enters signed", 'abs(rf0[label]), abs(rf[label]))', 'abs(rf0[label]), rf[label])'),
    ("F-value", "F_ADE6 0.1207 -> 0.12", 'F_ADE6 = 0.1207 ', 'F_ADE6 = 0.12 '),
    ("halfF-value", "HALF_F 0.0604 -> 0.0605", 'HALF_F = 0.0604 ', 'HALF_F = 0.0605 '),
    # ---- clears / fires / domain / worsening
    ("clears-strict", "clears is strict at the bar", 'return bool(sep_better(label, row) and -worsening(label, row) >= bar)', 'return bool(sep_better(label, row) and -worsening(label, row) > bar)'),
    ("clears-no-sep", "clears drops separation", 'return bool(sep_better(label, row) and -worsening(label, row) >= bar)', 'return bool(-worsening(label, row) >= bar)'),
    ("fires-strict", "fires is strict at the bar", 'and worsening(label, x_ctrl) >= bar)', 'and worsening(label, x_ctrl) > bar)'),
    ("fires-no-domain", "fires drops the domain condition", ' and not sep_worse(label, dag_p10) and sep_worse(label, x_ctrl)', ' and sep_worse(label, x_ctrl)'),
    ("fires-no-xp10", "fires drops X - P10 worse", 'return bool(sep_worse(label, x_p10) and not sep_worse', 'return bool(not sep_worse'),
    ("fires-no-ctrl-sep", "fires drops X - CTRL separated", ' and sep_worse(label, x_ctrl) and worsening(label, x_ctrl)', ' and worsening(label, x_ctrl)'),
    ("domain-no-exempt", "domain keeps the exempt rows", 'if l not in EXEMPT and not sep_worse(l, dag_p10[l])]', 'if not sep_worse(l, dag_p10[l])]'),
    ("domain-inverted", "domain keeps the DAG-worse rows", 'if l not in EXEMPT and not sep_worse(l, dag_p10[l])]', 'if l not in EXEMPT and sep_worse(l, dag_p10[l])]'),
    ("worsening-dir", "worsening ignores direction", 'return d if DIRECTION[label] == "lower" else -d', 'return d'),
    ("worsening-dir-swap", "worsening direction swapped", 'return d if DIRECTION[label] == "lower" else -d', 'return d if DIRECTION[label] == "higher" else -d'),
    ("sepworse-ge", "separated-worse includes zero", 'and worsening(label, row) > 0)', 'and worsening(label, row) >= 0)'),
    ("sepbetter-le", "separated-better includes zero", 'and worsening(label, row) < 0)', 'and worsening(label, row) <= 0)'),
    ("notcomputed-pass", "a NOT COMPUTED row is judged 0", 'if row is None or row.get("status") == "NOT COMPUTED":\n        raise RowNotComputed(label)\n    return row', 'if row is None or row.get("status") == "NOT COMPUTED":\n        return {"delta": 0.0, "separated": False}\n    return row'),
    # ---- per-seed composition
    ("crawl-uses-ctrl", "R-CRAWL reads X - CTRL", 'crawl_rows = {l: not sep_worse(l, t["X_P10"][l]) for l in c}', 'crawl_rows = {l: not sep_worse(l, t["X_CTRL"][l]) for l in c}'),
    ("crawl-inverted", "R-CRAWL inverted", 'crawl_rows = {l: not sep_worse(l, t["X_P10"][l]) for l in c}', 'crawl_rows = {l: sep_worse(l, t["X_P10"][l]) for l in c}'),
    ("move-uses-p10", "R-MOVE reads X - P10", 'move_rows = {l: clears(l, t["X_CTRL"][l], bars[l]) for l in c}', 'move_rows = {l: clears(l, t["X_P10"][l], bars[l]) for l in c}'),
    ("move-no-bar", "R-MOVE ignores the bar", 'move_rows = {l: clears(l, t["X_CTRL"][l], bars[l]) for l in c}', 'move_rows = {l: clears(l, t["X_CTRL"][l], 0.0) for l in c}'),
    ("crawl-any", "R-CRAWL any instead of all", '"R-CRAWL": all(crawl_rows.values())', '"R-CRAWL": any(crawl_rows.values())'),
    ("move-any", "R-MOVE any instead of all", '"R-MOVE": all(move_rows.values())', '"R-MOVE": any(move_rows.values())'),
    ("keep-strict", "R-KEEP strict", '"R-KEEP": bool(g["KEEP_point"] <= HALF_F)', '"R-KEEP": bool(g["KEEP_point"] < HALF_F)'),
    ("keep-reads-t0", "R-KEEP reads the T0 point", '"R-KEEP": bool(g["KEEP_point"] <= HALF_F)', '"R-KEEP": bool(g["T0_point"] <= HALF_F)'),
    ("t0-strict", "R-T0 strict", '"R-T0": bool(g["T0_point"] <= HALF_F)', '"R-T0": bool(g["T0_point"] < HALF_F)'),
    ("t0-reads-keep", "R-T0 reads the KEEP point", '"R-T0": bool(g["T0_point"] <= HALF_F)', '"R-T0": bool(g["KEEP_point"] <= HALF_F)'),
    ("tlat-sign", "R-TLAT/R-TLON sign flipped", 'f1 = lambda r: bool(r["separated"] and float(r["delta"]) < 0)', 'f1 = lambda r: bool(r["separated"] and float(r["delta"]) > 0)'),
    ("tlat-no-sep", "R-TLAT/R-TLON ignores separation", 'f1 = lambda r: bool(r["separated"] and float(r["delta"]) < 0)', 'f1 = lambda r: bool(float(r["delta"]) < 0)'),
    ("ade-reads-p10", "R-ADE reads X - P10", '"R-ADE": not sep_worse("ADE6", t["X_CTRL"]["ADE6"])', '"R-ADE": not sep_worse("ADE6", t["X_P10"]["ADE6"])'),
    ("new-not-all", "R-NEW: not all instead of not any", '"R-NEW": not any(fire_rows.values())', '"R-NEW": not all(fire_rows.values())'),
    ("nobase-not-all", "R-NOBASE: not all instead of not any", '"R-NOBASE": not any(nobase_rows.values())', '"R-NOBASE": not all(nobase_rows.values())'),
    ("nobase-no-dag", "R-NOBASE drops X - DAG separated-worse", 'and sep_worse(l, t["X_DAG"][l]) and worsening(l, t["X_DAG"][l]) >= bars[l])', 'and worsening(l, t["X_DAG"][l]) >= bars[l])'),
    ("nobase-strict", "R-NOBASE strict at the bar", 'worsening(l, t["X_DAG"][l]) >= bars[l])', 'worsening(l, t["X_DAG"][l]) > bars[l])'),
    ("nobase-no-p10", "R-NOBASE drops X - P10 separated-worse", 'nobase_rows = {l: bool(sep_worse(l, t["X_P10"][l]) and sep_worse', 'nobase_rows = {l: bool(sep_worse'),
    ("nobase-reads-ctrl", "R-NOBASE worsening from X - CTRL", 'and worsening(l, t["X_DAG"][l]) >= bars[l])', 'and worsening(l, t["X_CTRL"][l]) >= bars[l])'),
    ("fires-wrong-arg", "R-NEW passes X - DAG as X - CTRL", 'fires(l, t["X_P10"][l], t["DAG_P10"][l], t["X_CTRL"][l], bars[l])', 'fires(l, t["X_P10"][l], t["DAG_P10"][l], t["X_DAG"][l], bars[l])'),
    ("bars-rf-twice", "bars use RF for RF0", 'bars = {l: bar_for(l, rf0, rf) for l in LABELS}', 'bars = {l: bar_for(l, rf, rf) for l in LABELS}'),
    ("bars-rf0-twice", "bars use RF0 for RF", 'bars = {l: bar_for(l, rf0, rf) for l in LABELS}', 'bars = {l: bar_for(l, rf0, rf0) for l in LABELS}'),
    ("crawlset-s1-yaw", "C_1 drops yaw-rate", '"M1 crawl share", "yaw-rate MAE")}', '"M1 crawl share")}'),
    ("crawlset-s0-m1", "C_0 drops M1", 'CRAWL_SET = {0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),', 'CRAWL_SET = {0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)"),'),
    ("exempt-accel", "EXEMPT drops accel", 'EXEMPT = ("accel MAE", "DK-VIOL")', 'EXEMPT = ("DK-VIOL",)'),
    ("direction-tac", "TAC-LON lower-is-better", 'DIRECTION.update({"TAC-LAT": "higher", "TAC-LON": "higher"})', 'DIRECTION.update({"TAC-LAT": "higher"})'),
    ("labels-drop", "LABELS drops DK-TG", '"accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE"', '"accel MAE", "DK-VIOL", "DK-HW", "heading MAE"'),
    ("domain-n", "|D_1| 13 -> 14", 'DOMAIN_N = {0: 14, 1: 13}', 'DOMAIN_N = {0: 14, 1: 14}'),
    ("rep-seed", "REP seed 11 -> 10", 'REP_SEED = {0: 10, 1: 11}', 'REP_SEED = {0: 10, 1: 10}'),
    ("p-b", "P_B 0.1 -> 0.2", 'P_A, P_B, REACH_TOL = 0.3, 0.1, 0.02', 'P_A, P_B, REACH_TOL = 0.3, 0.2, 0.02'),
    ("tol", "REACH_TOL 0.02 -> 0.03", 'P_A, P_B, REACH_TOL = 0.3, 0.1, 0.02', 'P_A, P_B, REACH_TOL = 0.3, 0.1, 0.03'),
    # ---- the outcome table
    ("guards-any", "guards held: any instead of all", 'guards_held = all(held[g][s] for g in GUARDS for s in SEEDS)', 'guards_held = any(held[g][s] for g in GUARDS for s in SEEDS)'),
    ("harms-no-ctrl", "HARMS without the controls condition", '"HARMS": ctrl and not guards_held,', '"HARMS": not guards_held,'),
    ("rmboth-any", "R-MOVE both: any", 'rm_both, rm_n, rc_both = all(rm[s] for s in SEEDS),', 'rm_both, rm_n, rc_both = any(rm[s] for s in SEEDS),'),
    ("rcboth-any", "R-CRAWL both: any", 'sum(bool(rm[s]) for s in SEEDS), all(rc[s] for s in SEEDS)', 'sum(bool(rm[s]) for s in SEEDS), any(rc[s] for s in SEEDS)'),
    ("adopt-any", "adoption: any instead of all", 'adopt_ok = all(keep[s] and nobase[s] for s in SEEDS)', 'adopt_ok = any(keep[s] and nobase[s] for s in SEEDS)'),
    ("adopt-or", "adoption: keep OR nobase", 'adopt_ok = all(keep[s] and nobase[s] for s in SEEDS)', 'adopt_ok = all(keep[s] or nobase[s] for s in SEEDS)'),
    ("oneseed-ge", "ONE SEED: >= 1", '"PARTIAL / ONE SEED": ctrl and guards_held and rm_n == 1,', '"PARTIAL / ONE SEED": ctrl and guards_held and rm_n >= 1,'),
    ("noeffect-le", "NO EFFECT: <= 1", '"NO EFFECT": ctrl and guards_held and rm_n == 0,', '"NO EFFECT": ctrl and guards_held and rm_n <= 1,'),
    ("refused-never", "REFUSED never", '"REFUSED": not ctrl,', '"REFUSED": False,'),
    ("fixedadopted-no-rc", "FIXED / ADOPTED drops R-CRAWL", '"FIXED / ADOPTED": ctrl and guards_held and rm_both and rc_both and adopt_ok,', '"FIXED / ADOPTED": ctrl and guards_held and rm_both and adopt_ok,'),
    ("partialadopted-rc", "PARTIAL / ADOPTED needs R-CRAWL", '"PARTIAL / ADOPTED": ctrl and guards_held and rm_both and not rc_both and adopt_ok,', '"PARTIAL / ADOPTED": ctrl and guards_held and rm_both and rc_both and adopt_ok,'),
    ("partialnot-adopt", "PARTIAL / NOT ADOPTED ignores adoption", '"PARTIAL / NOT ADOPTED": ctrl and guards_held and rm_both and not rc_both and not adopt_ok,', '"PARTIAL / NOT ADOPTED": ctrl and guards_held and rm_both and not rc_both,'),
    ("decide-ge", "decide accepts several matches", 'out = matching[0] if len(matching) == 1 else "UNPLACED"', 'out = matching[0] if len(matching) >= 1 else "UNPLACED"'),
    ("decide-adopts", "adopts = FIXED", '"adopts": out in ADOPTING,', '"adopts": out.startswith("FIXED"),'),
    ("adopting-set", "ADOPTING drops PARTIAL", 'ADOPTING = ("FIXED / ADOPTED", "PARTIAL / ADOPTED")', 'ADOPTING = ("FIXED / ADOPTED",)'),
    ("cascade-eq", "cascade: B decides when A is not HARMS", 'if a_block["OUTCOME"] != "HARMS":', 'if a_block["OUTCOME"] == "HARMS":'),
    ("cascade-final", "cascade: final arm A after A's HARMS", 'return {"final_arm": "B", "final": b_block["OUTCOME"]', 'return {"final_arm": "A", "final": b_block["OUTCOME"]'),
    ("cascade-pending", "cascade: HARMS without B is not pending", '"b_status": "PENDING: train B', '"b_status": "decided: train B'),
    ("next-swap", "next_for_arm swaps A and B", 'return nxt[:i].strip() if arm == "A" else nxt[i:].strip()', 'return nxt[:i].strip() if arm == "B" else nxt[i:].strip()'),
    # ---- the PREREG pin, the RF0 pin
    ("prereg-no-tables", "prereg_problems skips the tables", 'return table_problems(text) + phrase_problems(text)', 'return phrase_problems(text)'),
    ("prereg-no-phrases", "prereg_problems skips the phrases", 'return table_problems(text) + phrase_problems(text)', 'return table_problems(text)'),
    ("prereg-no-sha", "prereg_problems skips the raw sha256", 'if hashlib.sha256(raw).hexdigest() != pin:', 'if False:'),
    ("prereg-pin-default", "prereg_problems default pin is empty, not the pinned sha", 'pin = PREREG_SHA256 if pin is None else pin', 'pin = "" if pin is None else pin'),
    ("table-no-cell", "table_problems skips the cell compare", '            if g != w:', '            if False:'),
    ("table-no-len", "table_problems skips the row count", 'if len(got) != len(want["rows"]):', 'if False:'),
    ("table-no-heading", "table_problems ignores a missing heading", 'if got_h is None:', 'if False:'),
    ("phrase-inverted", "phrase_problems inverted", ' not in re.sub(r"\\s+", " ", sec.replace("**", "")):', ' in re.sub(r"\\s+", " ", sec.replace("**", "")):'),
    ("phrase-no-sec", "phrase_problems ignores a missing section", 'if sec is None or re.sub', 'if re.sub'),
    ("rf0-no-pin", "rf0_problems skips the pin", 'if got != pin:', 'if False:'),
    ("rf0-no-shafile", "rf0_problems skips the .sha256 compare", 'if not rec or rec[0] != got:', 'if not rec:'),
    ("rf0-no-max", "rf0_problems skips RF0 = max over seeds", 'elif v != max(ds):', 'elif False:'),
    ("rf0-no-labels", "rf0_problems skips the label check", 'if list(rows) != list(LABELS) or list(rf0) != list(LABELS):', 'if False:'),
    ("rf0-no-schema", "rf0_problems skips the schema", 'if d.get("schema") != RF0_SCHEMA:', 'if False:'),
    ("rf0-no-controls", "rf0_problems skips P-0's controls", 'if failed or not d.get("controls"):', 'if False:'),
    ("rf0-pin-const", "RF0_SHA256 changed", 'RF0_SHA256 = "1608d70b', 'RF0_SHA256 = "1608d70c'),
    ("prereg-pin-const", "PREREG_SHA256 changed", 'PREREG_SHA256 = "38bc688b', 'PREREG_SHA256 = "38bc688c'),
    ("p0-strict", "K-P0 mtime: < instead of <=", 'os.path.getmtime(f) <= t_frozen]', 'os.path.getmtime(f) < t_frozen]'),
    ("p0-never", "K-P0 mtime never late", 'os.path.getmtime(f) <= t_frozen]', 'os.path.getmtime(f) <= 0]'),
    # ---- run-record controls
    ("pair-A-vs-ctrl", "K-PAIR(A) compares with DAG2-0's argv (no lever)", "want = a_flags(ctrl_flags, seed, P_A)\n        d = flag_diff(flags, want)", "want = a_flags(ctrl_flags, seed, P_A)\n        d = flag_diff(flags, ctrl_flags)"),
    ("pair-REP-seed", "K-PAIR(REP) leaves --seed unchecked", 'want["--seed"] = [[str(REP_SEED[seed])]]', 'want["--seed"] = flags.get("--seed", [])'),
    ("pair-B-vs-A", "K-PAIR(B) skips the A-record comparison", "if d2 != [\"--ego-perturb-p\"]:", "if False:"),
    ("pair-ignore-out", "flag_diff compares --out", 'def flag_diff(fa, fb, ignore=("--out",)):', 'def flag_diff(fa, fb, ignore=()):'),
    ("pair-norm", "flag_diff compares strings, not numbers", 'return repr(float(x))', 'return str(x)'),
    ("pair-law", "a_flags drops --ego-perturb-vmax", '    for k, v in LAW_A.items():\n        fl[k] = [list(v)]', '    for k, v in LAW_A.items():\n        if k != "--ego-perturb-vmax":\n            fl[k] = [list(v)]'),
    ("pair-law-seed", "a_flags: wrong --ego-perturb-seed", 'fl["--ego-perturb-seed"] = [[str(seed)]]', 'fl["--ego-perturb-seed"] = [["0"]]'),
    ("reg-rep-p", "registered p of REP is 0.3", 'p = {"A": P_A, "B": P_B, "REP": 0.0, "CTRL": 0.0}[arm]', 'p = {"A": P_A, "B": P_B, "REP": P_A, "CTRL": 0.0}[arm]'),
    ("reg-no-dagger", "registered argv skips --dagger-p", '("--ego-perturb-p", p), ("--dagger-p", 0.0), ("--seed", want_seed))', '("--ego-perturb-p", p), ("--seed", want_seed))'),
    ("reg-seed", "registered seed of REP is the base seed", 'want_seed = REP_SEED[seed] if arm == "REP" else seed', 'want_seed = seed'),
    ("reg-init-tail", "registered init tail unchecked", 'endswith(f"M2b_s{seed}_dag/ckpt_last.pt")', 'endswith("ckpt_last.pt")'),
    ("reg-steps", "registered steps 3000 -> 2999", 'STEPS, LR, WARMUP, PGT = 3000, 1e-4, 100, 0.2', 'STEPS, LR, WARMUP, PGT = 2999, 1e-4, 100, 0.2'),
    ("init-any", "K-INIT accepts any sha", 'if str(v) != DAG_CKPT_SHA[seed]]', 'if False]'),
    ("init-none", "K-INIT accepts no record", 'return False, "no record of the init checkpoint\'s sha256: the init is unproven"', 'return True, "no record"'),
    ("init-other-seed", "K-INIT compares with the other seed's DAG", 'if str(v) != DAG_CKPT_SHA[seed]]', 'if str(v) != DAG_CKPT_SHA[1 - seed]]'),
    ("reach-rep", "K-REACH(REP) accepts n > 0", '        if n != 0:\n            bad.append(f"REP ego_perturb_n', '        if n < 0:\n            bad.append(f"REP ego_perturb_n'),
    ("reach-band", "K-REACH band widened", 'want + REACH_TOL + 1e-12', 'want + 0.05'),
    ("reach-lower", "K-REACH lower band dropped", 'if not (want - REACH_TOL - 1e-12 <= ratio <= want + REACH_TOL + 1e-12):', 'if not (ratio <= want + REACH_TOL + 1e-12):'),
    ("reach-builtp", "K-REACH ignores the built p", 'if built_p is None or abs(float(built_p) - want) > 1e-12:', 'if False:'),
    ("reach-elig", "K-REACH accepts eligible 0", 'and isinstance(eligible, (int, float)) and eligible > 0 and 0 <= n <= eligible):', 'and isinstance(eligible, (int, float)) and 0 <= n <= eligible):'),
    ("build-identical", "K-BUILD ignores a changed file", 'if fr["sha256_launch"] != fr["sha256_base"]:', 'if False:'),
    ("build-nohunk", "K-BUILD accepts a changed file with no hunk", 'if not hunks:\n                bad.append', 'if False:\n                bad.append'),
    ("build-bit", "K-BUILD skips the bit-identity test", ' or h.get("bit_identity_test") != "PASS":', ':'),
    ("build-default", "K-BUILD skips default_off", 'if h.get("default_off") is not True or not h.get("behind_flag")', 'if not h.get("behind_flag")'),
    ("build-commit", "K-BUILD accepts any launch commit", 'r"[0-9a-f]{40}"', 'r"[0-9a-f]{3}"'),
    ("build-base", "K-BUILD accepts any base commit", 'startswith(BUILD_BASE_COMMIT)', 'startswith("")'),
    ("zero-hi", "identity ignores the CI upper end", 'r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0)', 'r["delta"] == 0.0 and r["lo"] == 0.0)'),
    ("zero-empty", "identity accepts an empty table", 'return bool(rows) and all(', 'return all('),
    ("rc-tol", "RC tolerance huge", 'if abs(float(ra[k]) - float(rb[k])) > tol:', 'if abs(float(ra[k]) - float(rb[k])) > 1e9:'),
    ("rc-sep", "RC ignores the separated flag", 'if bool(ra["separated"]) != bool(rb["separated"]):', 'if False:'),
    ("rc-missing", "RC ignores a missing row", 'bad.append(f"{lab}: present on one side only")', 'pass'),
    ("floors-min", "floors_from takes min", 'return {l: max(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS) for l in LABELS}', 'return {l: min(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS) for l in LABELS}'),
    ("floors-signed", "floors_from signed", 'max(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS)', 'max(float(_row(l, tables[s][l])["delta"]) for s in SEEDS)'),
    ("floors-one-seed", "floors_from reads seed 0 only", 'max(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS)', 'max(abs(float(_row(l, tables[s][l])["delta"])) for s in SEEDS[:1])'),
    ("disc-partial", "discover: a short dump dir is PRESENT", '"ep*.npz"))) < N_EPISODES:', '"ep*.npz"))) < 1:'),
    ("disc-dag", "discover: DAG needs the T0 files too", '    if arm != "DAG":\n        td = e.get("train_dir")', '    if True:\n        td = e.get("train_dir")'),
    ("disc-noentry", "discover: absent arm is PRESENT", 'out[arm][s] = {"state": "MISSING", "missing": ["no manifest entry"]}', 'out[arm][s] = {"state": "PRESENT", "missing": []}'),
    ("rcev-sha", "RC evidence accepted from another reader", 'if d.get("reader_sha256") != me:', 'if False:'),
    ("rcev-pass", "RC evidence PASS flag ignored", 'if d.get("differences") != 0 or d.get("PASS") is not True:', 'if d.get("differences") != 0:'),
    ("rcev-diff", "RC evidence differences ignored", 'if d.get("differences") != 0 or d.get("PASS") is not True:', 'if d.get("PASS") is not True:'),
    ("compose-rf-twice", "compose_all passes RF as RF0", 'res["floors"]["RF0"], res["floors"]["RF"])', 'res["floors"]["RF"], res["floors"]["RF"])'),
    ("compose-rf0-twice", "compose_all passes RF0 as RF", 'res["floors"]["RF0"], res["floors"]["RF"])', 'res["floors"]["RF0"], res["floors"]["RF0"])'),
    ("compose-ctrl-ok", "compose_all hard-codes the controls flag", 'out[x] = {"per_seed": per_seed, "decision": outcome_block(decide_arm(ctl_ok, per_seed), x)}', 'out[x] = {"per_seed": per_seed, "decision": outcome_block(decide_arm(True, per_seed), x)}'),
    ("parse-flags", "parse_flags drops multi-value flags", 'while j < len(argv) and not argv[j].startswith("--"):', 'while j < i + 2 and j < len(argv) and not argv[j].startswith("--"):'),
]


def run_one(args):
    mid, desc, old, new, src = args
    n = src.count(old)
    if n != 1:
        return {"id": mid, "desc": desc, "result": "ANCHOR ERROR", "anchor_count": n}
    mut = src.replace(old, new)
    try:
        ast.parse(mut)
    except SyntaxError as e:
        return {"id": mid, "desc": desc, "result": "INVALID", "error": str(e)}
    p = f"{SCR}/m_{mid}.py"
    open(p, "w", encoding="utf-8", newline="\n").write(mut)
    return {"id": mid, "desc": desc, "file": p, "result": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only")
    ap.add_argument("--out", default=f"{HERE}/mutation.json")
    ap.add_argument("--reader", default=f"{HERE}/dag_orc_read.py")
    a = ap.parse_args()
    os.makedirs(SCR, exist_ok=True)
    src = open(a.reader, encoding="utf-8").read()
    only = set(a.only.split(",")) if a.only else None
    ctrl = f"{SCR}/m_CONTROL.py"
    open(ctrl, "w", encoding="utf-8", newline="\n").write(src)
    jobs = [("CONTROL", "unmutated control", None, None, ctrl)]
    prep = []
    for mid, desc, old, new in MUTANTS:
        if only and mid not in only:
            continue
        prep.append(run_one((mid, desc, old, new, src)))
    ids = [m[0] for m in MUTANTS]
    assert len(ids) == len(set(ids)), "duplicate mutant ids"
    queue = [("CONTROL", ctrl)] + [(p["id"], p["file"]) for p in prep if p["result"] is None]
    results = {p["id"]: p for p in prep if p["result"] is not None}
    running, done, t0 = {}, {}, time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", DAGORC_TMP="F:/TanitAD-scratch/dag_orc_read_tmp")
    while queue or running:
        while queue and len(running) < a.workers:
            mid, f = queue.pop(0)
            of = open(f"{SCR}/out_{mid}.txt", "wb")      # a file, not a pipe: a Windows pipe holds 4 KB and a failing mutant prints more (a PIPE deadlocked one mutant in the first run)
            running[mid] = (subprocess.Popen([PY, "-I", f"{HERE}/dag_orc_selftest.py", "--reader", f, "--quiet"], stdout=of, stderr=subprocess.STDOUT, env=env, cwd=HERE), of, time.time())
        for mid in list(running):
            pr, of, t_start = running[mid]
            if pr.poll() is None and time.time() - t_start > TIMEOUT_S:
                pr.kill()
                pr.wait()
                of.close()
                done[mid] = ("TIMEOUT", "")
                del running[mid]
            elif pr.poll() is not None:
                of.close()
                done[mid] = (pr.returncode, open(f"{SCR}/out_{mid}.txt", "rb").read().decode("utf-8", "replace"))
                del running[mid]
        time.sleep(0.2)
    ctl_rc, ctl_out = done["CONTROL"]
    res = []
    for mid, desc, old, new in MUTANTS:
        if only and mid not in only:
            continue
        if mid in results:
            res.append(results[mid])
            continue
        rc, out = done[mid]
        last = [ln for ln in out.splitlines() if ln.startswith("FAIL") or ln.startswith("SELFTEST") or "Error" in ln][:3]
        res.append({"id": mid, "desc": desc, "result": "TIMEOUT" if rc == "TIMEOUT" else ("KILLED" if rc != 0 else "SURVIVED"), "selftest_rc": rc, "first_red": last[:2]})
    n_k = sum(r["result"] == "KILLED" for r in res)
    bad = [r for r in res if r["result"] != "KILLED"]
    summary = {"control_rc": ctl_rc, "control_last": ctl_out.strip().splitlines()[-1] if ctl_out.strip() else "", "mutants": len(res), "killed": n_k, "not_killed": [(r["id"], r["result"]) for r in bad],
               "wall_s": round(time.time() - t0, 1), "reader_sha256": __import__("hashlib").sha256(src.encode("utf-8")).hexdigest()}
    json.dump({"summary": summary, "results": res}, open(a.out, "w", encoding="utf-8"), indent=1)
    for r in res:
        print(f"{r['result']:13s} {r['id']:22s} {r['desc']}" + (f"   <- {r['first_red'][0][:90]}" if r.get("first_red") and r["result"] == "KILLED" else ""))
    print(f"CONTROL rc={ctl_rc} ({summary['control_last']})")
    print(f"MUTATION {n_k}/{len(res)} killed" + (f"; NOT KILLED: {summary['not_killed']}" if bad else "") + f"; {summary['wall_s']} s")
    return 0 if (ctl_rc == 0 and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
