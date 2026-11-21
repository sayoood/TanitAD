"""Mutation proof for dag_read.py: every mutant is a textual edit of a COPY of the reader; the test suite (pointed at the copy through DAG_READ_PATH)
must turn RED. A mutant that leaves the suite GREEN is reported SURVIVED and the script exits 1.

    python mutate_dag.py [--only M01,N05] [--full] [--out out/mutation.out]

Each mutant is first run against its TARGETED tests (`-k`, a few seconds); one that survives them is re-run against the WHOLE suite before it is called
survived. The pytest cap (never a third concurrent pytest process) is honoured by polling `pytest_runs.py` before every run.
The reader and the tests are NEVER modified: the edits go to copies under WORK.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "dag_read.py"
WORK = pathlib.Path("E:/Projects/TanitAD-artifacts/dagread_build/mut")
PT = pathlib.Path("E:/Projects/TanitAD-artifacts/dagread_build/pt")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
RUNS = "C:/Users/Admin/orcread/pytest_runs.py"


def sub(old, new, count=1):
    def f(s):
        assert s.count(old) == count, (s.count(old), old)
        return s.replace(old, new)
    return f


def swap_table_rows(s):
    a, b = '("ROBUST",\n     "R-TLAT and R-TLON both clear', '("PARTIAL",\n     "exactly one head clears'
    assert s.count(a) == 1 and s.count(b) == 1
    return s.replace(a, "@@A@@").replace(b, "@@B@@").replace("@@A@@", b).replace("@@B@@", a)


BANK_FAIL = 'if ctl.failed():\n        raise Refusal([f"{k}: {v[\'detail\']}" for k, v in ctl.failed().items()])\n'
# id, what it breaks, edit, targeted -k
MUTANTS = [
    # ------------------------------------------------------------------------------------------------ inherited from orc_read.py (re-proved on the DAG reader)
    ("M01", "flip a bar sign (>= bar becomes <= bar)", sub('return bool(sep_better(s, hd) and r["delta"] >= bar(s, hd))', 'return bool(sep_better(s, hd) and r["delta"] <= bar(s, hd))'), "decide"),
    ("M02", "drop a seed (SEEDS = (0,))", sub("SEEDS = (0, 1)", "SEEDS = (0,)"), "decide or world_ROBUST"),
    ("M03", "swap the lat / lon bars", sub('TWO_F = {"lat": 0.0248, "lon": 0.0146}', 'TWO_F = {"lat": 0.0146, "lon": 0.0248}'), "decide or committed"),
    ("M04", "skip a control (the replaced-fraction band)", sub("if not (DAG_P - RATIO_TOL - 1e-12 <= ratio <= DAG_P + RATIO_TOL + 1e-12):", "if False:"), "g_reached"),
    ("M05", "off-by-one tick >= 1 (ticks >= 2)", sub('n)[:, 1:]', 'n)[:, 2:]', count=2), "ticks_ge1 or matched_cells or world_ROBUST"),
    ("M23", "off-by-one tick, non-crashing (ticks 0-1 instead of 1-2)", sub('n)[:, 1:]', 'n)[:, :2]', count=2), "ticks_ge1 or matched_cells or world_ROBUST"),
    ("M06", "wrong reducer direction (DAG0 - DAG)", sub("a, b, eid2, n_boot, red, record=False,\n                  lower_is_better=False", "b, a, eid2, n_boot, red, record=False,\n                  lower_is_better=False"),
     "F1_sign or world_ROBUST"),
    ("M07", "outcome order swap (ROBUST and PARTIAL rows trade places)", swap_table_rows, "table_verbatim"),
    ("M08", "pair runs by different windows (the SAME-WINDOWS control is blind)", sub('ctl.add(f"SAME-WINDOWS:{arm}-vs-DAG0:s{s}", same,', 'ctl.add(f"SAME-WINDOWS:{arm}-vs-DAG0:s{s}", True,'),
     "different_windows or other_windows or t0_episodes"),
    ("M09", "R-ADE guard direction (separated-BETTER counts as worse)", sub('rows[s]["R-ADE"]["delta"] > 0) for s in SEEDS}', 'rows[s]["R-ADE"]["delta"] < 0) for s in SEEDS}'), "decide or HARMS"),
    ("M10", "R-T0 bar is exclusive (>= F/2 is above)", sub('rows[s]["R-T0"]["point"] > HALF_F', 'rows[s]["R-T0"]["point"] >= HALF_F'), "guard_boundaries"),
    ("M11", "matched cells off (each run's own cells)", sub("    if matched:\n        m = (ea >= 0)", "    if False:\n        m = (ea >= 0)"), "matched_cells"),
    ("M12", "the K1-K7 results of a T1 run are ignored", sub('ctl.add(f"{k}:{tag}", ok, det)', 'ctl.add(f"{k}:{tag}", True, det)'), "every_arm_control"),
    ("M14", "the init-checkpoint sha control always passes", sub("    bad = [f\"{w} = {str(v)[:16]}\" for w, v in vals if str(v) != m2b_sha]\n    return not bad,", "    bad = []\n    return not bad,"),
     "g_init_sha or every_arm_control"),
    ("M15", "DAG-0 may replace samples (n == 0 not required)", sub('if not (isinstance(n, (int, float)) and n == 0):', "if False:"), "g_reached_dag0 or every_arm_control"),
    ("M16", "the launch-gate verdict is not read", sub('if tok.get("verdict") != "PASS":', "if False:"), "g_token or every_arm_control"),
    ("M17", "the reader's own K1 identity control is blind", sub('if not (r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0):', "if False:"), "k1_own"),
    ("M18", "the guards ignore R-T0", sub("guards_held = not any(ade_worse.values()) and not any(t0_above.values())", "guards_held = not any(ade_worse.values())"), "decide or HARMS"),
    ("M19", "a head that clears at ONE seed counts as clearing at both", sub('clear_both = {hd: all(clears(s, hd) for s in SEEDS) for hd in ("lat", "lon")}',
                                                                          'clear_both = {hd: any(clears(s, hd) for s in SEEDS) for hd in ("lat", "lon")}'), "decide or world_PARTIAL"),
    ("M20", "separation is not required for 'better'", sub('return bool(r["separated"] and r["delta"] > 0)', 'return bool(r["delta"] > 0)'), "decide"),
    ("M21", "the registered DAG --dagger-p is not checked against the arm (0.0 for both)", sub('"--dagger-p": ARM_P[arm]}', '"--dagger-p": 0.0}'), "g_argv or world_ROBUST"),
    ("M22", "a failed control still computes rows (REFUSED does not stop the read)", sub("    if ctl.failed():\n        raise Refusal([f\"{k}: {v['detail']}\" for k, v in ctl.failed().items()])\n    # ---------------- the rows",
                                                                                       "    if False:\n        raise Refusal([f\"{k}: {v['detail']}\" for k, v in ctl.failed().items()])\n    # ---------------- the rows"),
     "failed_control or every_arm_control"),
    ("M26", "K4 reads the standard T0 eval", sub('backend.k4(lg, pk, rec["entry"]["t0_fresh"], s)', 'backend.k4(lg, pk, rec["entry"]["t0"], s)'), "k4_reads or t0_ages or world_ROBUST"),
    ("M27", "the T0-AGES control is blind", sub('ages == ("cycle", "fresh"),', 'True,'), "t0_ages"),
    ("M28", "the fresh eval's checkpoint is not checked", sub('tail2(str(rec["t0fjson"].get("ckpt", ""))) == ckpt_tail,', 'True,'), "t0_ages"),
    ("M29", "a manifest without t0_fresh falls back to t0", sub("""            if not t0f:
                raise Refusal(f"{tag}: manifest has no 't0_fresh' (K4""", """            if not t0f:
                e["t0_fresh"] = t0f = e["t0"]
            if False:
                raise Refusal(f"{tag}: manifest has no 't0_fresh' (K4"""), "without_t0_fresh"),
    ("M30", "R-T0 reads the fresh eval", sub('            t0e = e.get("t0")\n', '            t0e = e.get("t0_fresh")\n'), "k4_reads or world_ROBUST"),
    # ------------------------------------------------------------------------------------------------ new in dag_read.py: the replicate floor (section 5)
    ("N01", "the bar ignores RF (always 2F)", sub("    return max(TWO_F[hd], 2.0 * rf)", "    return TWO_F[hd]"), "bar_is or replicate_floor"),
    ("N02", "the bar uses RF, not 2 x RF", sub("    return max(TWO_F[hd], 2.0 * rf)", "    return max(TWO_F[hd], rf)"), "bar_is or replicate_floor"),
    ("N03", "RF is the signed gap (ORC-0 better than DAG-0 reads as no floor)", sub('rf = abs(float(rf_row["delta"]))', 'rf = float(rf_row["delta"])'), "replicate_floor_is_absolute"),
    ("N04", "RF is read on each run's own cells, not the cells scored in both", sub('rf_row, (ra, _rb, reid, rred) = f1_pair_row(T, z["pack"], r0["pack"], hd, nb, matched=True)',
                                                                                    'rf_row, (ra, _rb, reid, rred) = f1_pair_row(T, z["pack"], r0["pack"], hd, nb, matched=False)'), "replicate_floor_reads_only"),
    ("N05", "RF is DAG vs ORC-0 (the lever's own effect) instead of DAG-0 vs ORC-0", sub('rf_row, (ra, _rb, reid, rred) = f1_pair_row(T, z["pack"], r0["pack"]',
                                                                                          'rf_row, (ra, _rb, reid, rred) = f1_pair_row(T, o["pack"], r0["pack"]'), "world_ROBUST or replicate_floor"),
    ("N45", "bar_binding label inverted", sub('"2RF" if 2.0 * rf > TWO_F[hd] else "2F"', '"2RF" if 2.0 * rf < TWO_F[hd] else "2F"'), "world_ROBUST or replicate_floor"),
    ("N46", "'clears' is judged against 2F, not the bar", sub('r["clears"] = bool(r["better"] and r["delta"] >= r["bar"])', 'r["clears"] = bool(r["better"] and r["delta"] >= r["two_F"])'), "replicate_floor"),
    ("N47", "decide() uses 2F, not the bar", sub('return bar_for(hd, rows[s][hd_row[hd]]["RF"])', 'return TWO_F[hd]'), "decide_replicate or replicate_floor_raises"),
    ("N44", "ORC-0 is not read (no replicate arm)", sub('arms = ["DAG", "DAG0", "ORC0"] + ', 'arms = ["DAG", "DAG0"] + '), "world_ROBUST"),
    # ------------------------------------------------------------------------------------------------ new: the DAG arms' argv / pair / replicate / reached controls (sections 4, 6)
    ("N06", "p is 0.3, not the registered 0.1", sub("\nDAG_P = 0.1\n", "\nDAG_P = 0.3\n"), "committed_numbers"),
    ("N07", "DAG-0 is expected at the DAG p", sub('ARM_P = {"DAG": DAG_P, "DAG0": 0.0}', 'ARM_P = {"DAG": DAG_P, "DAG0": DAG_P}'), "world_ROBUST or g_argv"),
    ("N08", "--ego-perturb-p 0 is not required on the arms", sub('"--ego-perturb-p": EGO_P, "--dagger-p": ARM_P[arm]}', '"--dagger-p": ARM_P[arm]}'), "g_argv_each"),
    ("N09", "the ego-perturbation law may ride along on an arm", sub("    for name in LAW:\n        if name in fl:", "    for name in ():\n        if name in fl:"), "ego_perturbation_levers"),
    ("N10", "DAG-0 may name the DAG-only flags", sub("        for name in DAG_ONLY_FLAGS:\n            if name in fl:", "        for name in DAG_ONLY_FLAGS:\n            if False:"), "dagger_flags_belong"),
    ("N11", "DAG may omit --dagger-bank", sub('if not ok or not v:\n            bad.append(f"--dagger-bank is not named', 'if False:\n            bad.append(f"--dagger-bank is not named'), "dagger_flags_belong"),
    ("N12", "DAG may declare a malformed bank sha256", sub("if not ok or not is_hex(v):", "if False:"), "dagger_flags_belong"),
    ("N13", "DAG may omit --dagger-seed", sub("if not seed_ok:", "if False:"), "dagger_flags_belong"),
    ("N14", "the pair rule does not set the DAG-only flags aside", sub('drop = {"--out", "--dagger-p"} | set(DAG_ONLY_FLAGS)', 'drop = {"--out", "--dagger-p"}'), "world_ROBUST or g_pair"),
    ("N15", "the pair rule sets --lr aside too (a second lever slips through)", sub('drop = {"--out", "--dagger-p"} | set(DAG_ONLY_FLAGS)', 'drop = {"--out", "--dagger-p"} | set(DAG_ONLY_FLAGS) | {"--lr"}'), "g_pair"),
    ("N16", "SUMMARY_SAME forgets the TRAIN-list sha", sub('SUMMARY_SAME = ("steps", "n_clips", "n_windows", "train_sha256", "excluded_sha256", "seed")', 'SUMMARY_SAME = ("steps", "n_clips", "n_windows", "excluded_sha256", "seed")'),
     "g_pair or g_replicate"),
    ("N17", "the ORC0-REPLICATE control is not wired", sub('ctl.run(f"ORC0-REPLICATE:s{s}", lambda r0=r0, ro=ro, s=s: g_replicate(s, r0["train"], ro["train"], ki))', 'ctl.add(f"ORC0-REPLICATE:s{s}", True, None)'),
     "arm_stage_bank or every_arm_control"),
    ("N18", "ORC-0 may already name --dagger-p", sub('    if "--dagger-p" in fo:\n        bad.append(f"ORC-0\'s argv already names', '    if False:\n        bad.append(f"ORC-0\'s argv already names'), "g_replicate"),
    # ------------------------------------------------------------------------------------------------ new: the bank controls (section 6) and COVERAGE -> BLOCKED (section 7)
    ("N19", "COVERAGE_MIN is 0.4", sub("COVERAGE_MIN = 0.5", "COVERAGE_MIN = 0.4"), "coverage_boundary or committed_numbers"),
    ("N20", "COVERAGE is exclusive (exactly 0.5 is blocked)", sub('if any(cov[s]["eligible_fraction"] < COVERAGE_MIN for s in SEEDS):', 'if any(cov[s]["eligible_fraction"] <= COVERAGE_MIN for s in SEEDS):'), "coverage_boundary"),
    ("N21", "COVERAGE never blocks", sub('if any(cov[s]["eligible_fraction"] < COVERAGE_MIN for s in SEEDS):', 'if False:'), "BLOCKED"),
    ("N22", "COVERAGE reads seed 0 only", sub('if any(cov[s]["eligible_fraction"] < COVERAGE_MIN for s in SEEDS):', 'if any(cov[s]["eligible_fraction"] < COVERAGE_MIN for s in (0,)):'), "world_BLOCKED"),
    ("N23", "a failed bank control does not stop the read before COVERAGE (BLOCKED beats REFUSED)", sub(BANK_FAIL + "    cov = {s: coverage_of", "if False:\n        raise Refusal([f\"{k}: {v['detail']}\" for k, v in ctl.failed().items()])\n    cov = {s: coverage_of"), "REFUSED_beats"),
    ("N24", "the eligible fraction counts kept states, not eligible windows", sub('"eligible_fraction": cv["n_eligible_windows"] / cv["n_windows"],', '"eligible_fraction": cv["n_states_kept"] / cv["n_windows"],'), "world_BLOCKED or coverage_boundary"),
    ("N25", "BLOCKED returns rc 0", sub('{"UNPLACED": 3, "BLOCKED": 4}', '{"UNPLACED": 3, "BLOCKED": 0}'), "BLOCKED"),
    ("N26", "the BLOCKED row's condition text is not the PREREG's", sub('"COVERAGE < 0.5 (§6)"', '"COVERAGE < 0.4 (§6)"'), "table_verbatim"),
    ("N27", "K-B3 overlap by clip id is blind", sub('    if t & e:\n        bad.append(f"bank and eval-141 clip ids overlap', '    if False:\n        bad.append(f"bank and eval-141 clip ids overlap'), "clip_overlap"),
    ("N28", "K-B3 overlap by stable sid is blind", sub("if {stable_sid(c) for c in t} & {stable_sid(c) for c in e}:", "if False:"), "clip_overlap"),
    ("N29", "an empty eval-141 list is accepted", sub('    if not e:\n        bad.append("the eval-141 clip-id list is empty', '    if False:\n        bad.append("the eval-141 clip-id list is empty'), "clip_overlap or bank_stage_refuses"),
    ("N30", "duplicate bank clips are accepted", sub('    if len(set(ids)) != len(ids):\n        bad.append("the bank\'s clip_ids contain duplicates")', '    if False:\n        bad.append("the bank\'s clip_ids contain duplicates")'), "kb3"),
    ("N31", "the reader's own K-B3 deliberate regression is dropped", sub("if bad == [] and not clip_overlap(", "if False and not clip_overlap("), "kb3"),
    ("N32", "K-B3's recorded deliberate regression is not required", sub('if c.get("deliberate_regression_refused") is not True:', "if False:"), "kb3"),
    ("N33", "K-B0 may check zero states", sub('c["n_checked"] > 0', 'c["n_checked"] >= 0'), "kb0_kb1"),
    ("N34", "K-B1 may replay a single rollout", sub('c["n_replayed"] >= max(1, need_n)', 'c["n_replayed"] >= 1'), "kb0_kb1"),
    ("N35", "K-B2 on the report does not compare the generator with M2b", sub("    if g != m2b_sha:", "    if False:"), "kb2"),
    ("N36", "K-B4 does not require every matcher case", sub("miss = [k for k in KB4_CASES if cases.get(k) is not True]", "miss = []"), "kb4"),
    ("N37", "K-B5 tolerance is 1.0 m", sub("if not (num(c.get(k)) and c[k] <= KB5_TOL_M):", "if not (num(c.get(k)) and c[k] <= 1.0):"), "kb4"),
    ("N38", "a bank control record need not say PASS", sub('    if c.get("PASS") is not True:\n        raise Refusal(f"{kid} is recorded', '    if False:\n        raise Refusal(f"{kid} is recorded'), "kb_records"),
    ("N39", "BANK-CONFIG ignores the rollout configuration", sub('if h.get("rollout") != BANK_ROLLOUT:', "if False:"), "bank_config"),
    ("N40", "BANK-CONFIG ignores the matching-constants sha256", sub('if not str(h.get("matching_constants_sha256") or "").lower().startswith(MATCH_CONST_SHA_PREFIX):', "if False:"), "bank_config"),
    ("N41", "BANK-CONFIG ignores the window count", sub('if h.get("n_windows") != N_TRAIN_WINDOWS:', "if False:"), "bank_config"),
    ("N42", "BANK-CONFIG ignores the bank's seed", sub('if h.get("seed") != seed:', "if False:"), "bank_config"),
    ("N43", "BANK-CONFIG ignores the clip count", sub('if h.get("n_clips") != N_TRAIN_CLIPS or not isinstance(ids, list) or len(ids) != N_TRAIN_CLIPS:', "if False:"), "bank_config"),
    ("N48", "BANK-CONFIG ignores hmax", sub('if h.get("hmax") != HMAX:', "if False:"), "bank_config"),
    ("N49", "BANK-ACCOUNTING ignores 2 x windows", sub('if cv["n_states_before_cuts"] != 2 * cv["n_windows"]:', "if False:"), "bank_accounting"),
    ("N50", "BANK-ACCOUNTING ignores kept + cuts", sub('if cv["n_states_kept"] + sum(cv["cuts"].values()) != cv["n_states_before_cuts"]:', "if False:"), "bank_accounting"),
    ("N51", "BANK-ACCOUNTING ignores a conflicting stated fraction", sub('if sf is not None and not (num(sf) and abs(sf - cv["eligible_fraction"]) <= 1e-9):', "if False:"), "bank_accounting"),
    ("N52", "BANK-WINDOWS ignores the trainer's TRAIN-list sha", sub('        if v != h.get("window_list_sha256"):', "        if False:"), "bank_windows"),
    ("N53", "BANK-WINDOWS does not hash the bank's clip list", sub('    if got != h.get("window_list_sha256"):', "    if False:"), "bank_windows"),
    ("N54", "BANK-DECLARED ignores the argv's bank sha256", sub('if not ok or declared != h.get("bank_sha256"):', "if False:"), "bank_declared"),
    ("N55", "BANK-DECLARED ignores the trainer's built bank sha256", sub('        if str(v) != h.get("bank_sha256"):', "        if False:"), "bank_declared"),
    ("N56", "BANK-DECLARED (K-B2 on the arm) ignores the init checkpoint", sub('if str(v) != h.get("generator_ckpt_sha256") or str(v) != m2b_sha:', "if False:"), "bank_declared"),
    ("N57", "BANK-DECLARED ignores the bank file's hash", sub('if fsha != h.get("bank_sha256"):', "if False:"), "bank_declared"),
    ("N58", "BANK-DECLARED is not wired into the read", sub('g_bank_declared(tag, s, banks[s]["report"], tr, cfg["m2b_sha"].get(s, ""), ki, kb,', '(True, None) or g_bank_declared(tag, s, banks[s]["report"], tr, cfg["m2b_sha"].get(s, ""), ki, kb,'),
     "arm_stage_bank or every_arm_control"),
    ("N59", "BANK-WINDOWS is not wired into the read", sub('lambda rd=rd, r0=r0, s=s: g_bank_windows(', 'lambda rd=rd, r0=r0, s=s: (True, None) or g_bank_windows('), "arm_stage_bank"),
    ("N60", "K-B5 is not wired into the bank stage", sub('    ctl.run(f"K-B5:s{s}", lambda: g_kb5(rep))\n', "    pass\n"), "bank_stage_names"),
    ("N61", "BANK-ACCOUNTING is not wired into the bank stage", sub('    ctl.run(f"BANK-ACCOUNTING:s{s}", lambda: g_bank_accounting(rep))\n', "    pass\n"), "bank_stage_names"),
    ("N62", "K-B3 is not wired into the bank stage", sub('    ctl.run(f"K-B3:s{s}", lambda: g_kb3(rep, eval_ids))\n', "    pass\n"), "bank_stage_names"),
    # ------------------------------------------------------------------------------------------------ new: the outcome texts and labels
    ("N63", "HARMS at p = 0.1 resolves to the wrong branch", sub('"next": table_row("NO EFFECT")[1] if p == 0.1 else br}', '"next": table_row("NO EFFECT")[1] if p == 0.3 else br}'), "harms_branches or outcome_block or world_HARMS"),
    ("N64", "the PREREG id names another commit", sub("@ commit 07dcb513", "@ commit deadbeef"), "prereg_file"),
    ("N65", "the STRATEGIC family is labelled paused", sub('"NOT PAUSED for v8F (PI ruling 2026-10-17, R73)', '"PAUSED for v8F (PI ruling 2026-10-17, R73)'), "strategic_label or world_ROBUST"),
    ("N66", "T1-kin is called a closed loop", sub("T1 (T1-kin: self-action OPEN loop,", "T1 (T1-kin: self-action closed loop,"), "strategic_label or committed"),
]


def n_pytest():
    """Concurrent pytest RUNS, counted by the harness's own script (a venv launcher and its child count as ONE run)."""
    r = subprocess.run([PY, RUNS], capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def run_pytest(mut_path, k, tag, timeout=1500):
    while n_pytest() >= 2:
        time.sleep(30)
    env = dict(os.environ, DAG_READ_PATH=str(mut_path), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    PT.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={PT}/mut_{tag}_{int(time.time())}", str(HERE / "test_dag_read.py")]
    if k:
        cmd += ["-k", k]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-3:]
    return r.returncode, time.time() - t, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--full", action="store_true", help="run every mutant against the whole suite, not the targeted tests")
    ap.add_argument("--out", default=str(HERE / "out" / "mutation.out"))
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x}
    WORK.mkdir(parents=True, exist_ok=True)
    src = SRC.read_text(encoding="utf-8")
    ids = [m[0] for m in MUTANTS]
    assert len(ids) == len(set(ids)), "duplicate mutant ids"
    lines, survived = [], []
    for mid, what, edit, k in MUTANTS:
        if only and mid not in only:
            continue
        mut = edit(src)
        assert mut != src, mid
        p = WORK / f"dag_read_{mid}.py"
        p.write_text(mut, encoding="utf-8", newline="\n")
        rc, dt, tail = run_pytest(p, None if a.full else k, mid)
        scope = "whole suite" if a.full else f"-k '{k}'"
        if rc == 0 and not a.full:                       # survived the targeted tests: the whole suite decides
            rc, dt2, tail = run_pytest(p, None, mid + "_full")
            dt += dt2
            scope = "whole suite (after the targeted tests stayed green)"
        if rc == 5:                                      # pytest: no tests collected -- a typo in -k must not read as a kill
            verdict = "ERROR (no test selected)"
            survived.append(mid)
        else:
            verdict = "KILLED (RED)" if rc != 0 else "SURVIVED (GREEN)"
            if rc == 0:
                survived.append(mid)
        ln = f"{mid} {verdict:17s} {dt:5.1f}s  [{scope}]  {what}\n      pytest tail: {' | '.join(tail)}"
        print(ln, flush=True)
        lines.append(ln)
    outp = pathlib.Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if only else "w"
    with open(outp, mode, encoding="utf-8", newline="\n") as f:
        f.write(f"# mutation proof of dag_read.py (sha256 of the unmutated reader: {hashlib.sha256(SRC.read_bytes()).hexdigest()})\n")
        f.write("\n".join(lines) + "\n")
        f.write(f"# {len(lines) - len(survived)} of {len(lines)} mutants KILLED; survived: {survived or 'none'}\n")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
