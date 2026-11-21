#!/usr/bin/env python3
"""ROUND 2 of the RL-S1 fan-equivalence calibration: the per-set MEAN |delta| bound, tested on DISJOINT keys.

THE RULE LIVES IN `PREREG_ROUND2.md` (this directory), written and staged BEFORE any round-2 dump existed.  This file only APPLIES it.  Nothing here may be re-tuned after
the dump: the analysis prints the verdict the pre-registration commits to, and a code bug found later is fixed in a NEW commit that names it, never silently.

    python run_calibration2.py --phase prep      # CPU: assert ZERO key overlap with round 1, build the variants in <work>/atol_cal2 (never <work>/atol_cal)
    python run_calibration2.py --phase dump      # GPU: round 1's own dump(), every arm behind wait_idle.py (a non-zero exit ABORTS, exit 7); writes DUMP_DONE.json / DUMP_ABORT.json
    python run_calibration2.py --phase analyse   # CPU: the verdict (PASS / FAIL / INCOMPLETE) -> verdict_round2.json, arms_summary_round2.json, RESULTS_TABLE_round2.md
    python run_calibration2.py --phase selftest  # CPU: proves the wait_idle guard is reachable (RED when the waiter fails, GREEN control when it passes)

Python: C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe, PYTHONIOENCODING=utf-8, TMP/TEMP on D:.  Round 1's `run_calibration.py` is imported UNCHANGED;
only `pick_keys` (rows[14::28][:48]), the work dir, the roles (A1 demoted) and the analysis differ.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, R1)
import run_calibration as RC  # noqa: E402  (round 1, unchanged)

RF = RC.RF
STEP = RC.STEP
N_KEYS = RC.N_KEYS
STRIDE = RC.STRIDE
OFFSET = 14                                   # round 1 used rows[0::28]
DEFAULT_WORK = "D:/Projects/TanitAD-artifacts/_rlfan_work"
CATASTROPHE_ATOL = 0.5                        # kept in the CLI runs exactly as in round 1; its exit codes are recorded, never decisive
REF_SHA256_ROUND1 = "d2c4e47468ee3ba7b6c07c30ed68c496e26dfdd483387572323a7161106832d7"
ROUND1_CKPT_SHA = {"ckpt_head_noise10.pt": "1bc91b07e92bc9013551e202e8ef7ab4fbb45e1f0ccac9c636c20a07b6938901",
                   "ckpt_head_noise1.pt": "4860499fcdf574c9f91b1eb4f453332b7339e9d70add07d5cd19acaedcda4091"}

# ---- PREREG sections 3-5 (frozen) --------------------------------------------------------------------------------------------------------------------
NULL_R2 = ["N0_b4", "N1_b1", "N2_b2"]
NULL_ARMS = NULL_R2 + ["N_full"]
DECISION_ARMS = ["A2_calib_half_scale", "B1_img_hflip", "B2_img_bgr_rgb", "C1_head_noise10", "C2_snap_epoch021"]
INFO_ARMS = ["A1_calib_other_vehicle", "N3_jpeg", "S_focal105", "S_yaw3", "S_head_noise1", "S_snap027", "S_wrong_scene", "S_cam_reversed"]   # A1 DEMOTED (PREREG section 6)
N_FULL_SETS = 1365
M_FACTOR, M_QUANTUM = 2.0, 0.005
MIN_FRAC_ABOVE, WORST_OVER_M = 0.25, 5.0
IDENT_REPORT_ONLY = ["N_b1", "N_b2", "N_b4", "N_b4shift", "N_b4w2"]

_all = {n for n, _, _ in RC.ARMS}
assert set(DECISION_ARMS) | set(INFO_ARMS) | set(NULL_R2) == _all, sorted(_all ^ (set(DECISION_ARMS) | set(INFO_ARMS) | set(NULL_R2)))   # round 1's arm list must still be what we assume


# ------------------------------------------------------------------------------------------------------------------------------------ prep
def rows_of(W):
    return [ln.rstrip("\n").split("\t") for ln in open(f"{W}/identity/keys.tsv", encoding="utf-8")][1:]


def pick_keys2(W):
    sample = rows_of(W)[OFFSET::STRIDE][:N_KEYS]
    assert len(sample) == N_KEYS, len(sample)
    return [(t, l) for t, l in sample]


def round1_keys():
    return [tuple(ln.rstrip("\n").split("\t")) for ln in open(os.path.join(R1, "keys48.tsv"), encoding="utf-8")][1:]


def overlap_check(W):
    """Returns (overlap count over (token, log) pairs, shared log names).  Aborts when round 1's file is not what round 1's rule says (a stale or edited file)."""
    rows = rows_of(W)
    r1 = round1_keys()
    assert len(r1) == N_KEYS and r1 == [tuple(r) for r in rows[::STRIDE][:N_KEYS]], "round 1's keys48.tsv is not rows[::28][:48] of identity/keys.tsv"
    r2 = [tuple(k) for k in pick_keys2(W)]
    assert len(set(r2)) == N_KEYS
    ov = set(r1) & set(r2)
    return len(ov), sorted({l for _, l in r1} & {l for _, l in r2})


def phase_prep(W, A, out):
    assert os.path.abspath(A) != os.path.abspath(f"{W}/atol_cal"), "refusing to touch round 1's work dir"
    n_ov, shared_logs = overlap_check(W)
    assert n_ov == 0, f"ABORT: {n_ov} (token, log) pairs overlap with round 1's keys; round 2 must be disjoint"
    print(f"[prep] key overlap with round 1: {n_ov} (token, log) pairs; {len(shared_logs)} log names shared (different tokens)", flush=True)
    RC.pick_keys = pick_keys2                       # prep() looks pick_keys up in RC's globals at call time
    RC.prep(W, A)
    man = json.load(open(f"{A}/prep_manifest.json", encoding="utf-8"))
    man.update({"round": 2, "offset": OFFSET, "stride": STRIDE, "key_overlap_with_round1_token_log_pairs": n_ov, "shared_log_names_with_round1": len(shared_logs),
                "round1_keys48_sha256": RC.sha256_file(os.path.join(R1, "keys48.tsv"))})
    assert man["proxy_sets_sha256"] == REF_SHA256_ROUND1, ("reference sets differ from round 1's", man["proxy_sets_sha256"])
    man["variants_equal_round1"] = {f: (man["variants"].get(f) == h) for f, h in ROUND1_CKPT_SHA.items()}
    r1m = json.load(open(os.path.join(R1, "prep_manifest.json"), encoding="utf-8"))
    man["calib_variants_equal_round1"] = {f: (h == r1m["variants"].get(f)) for f, h in man["variants"].items() if f.startswith("calib_")}
    RC.wr_json(f"{A}/prep_manifest.json", man)
    RC.wr_json(os.path.join(out, "prep_manifest_round2.json"), man)
    shutil.copyfile(f"{A}/keys48.tsv", os.path.join(out, "keys48_round2.tsv"))
    print("[prep] manifest:", json.dumps({k: man[k] for k in ("keys48_sha256", "key_overlap_with_round1_token_log_pairs", "variants_equal_round1", "calib_variants_equal_round1")}), flush=True)


# ------------------------------------------------------------------------------------------------------------------------------------ dump
def phase_dump(W, A, only, py):
    """Round 1's dump() unchanged: every arm blocks on wait_idle.py and a non-zero exit from it aborts the whole dump with exit 7 (no arm runs on a GPU that is not provably idle)."""
    assert os.path.abspath(A) != os.path.abspath(f"{W}/atol_cal"), "refusing to touch round 1's work dir"
    assert os.path.exists(f"{A}/keys48.tsv"), "run --phase prep first"
    for m in ("DUMP_DONE.json", "DUMP_ABORT.json"):
        if os.path.exists(f"{A}/{m}"):
            os.remove(f"{A}/{m}")
    t0 = time.time()
    RC.wr_json(f"{A}/DUMP_RUNNING.json", {"pid": os.getpid(), "started_unix": t0, "started_local": time.strftime("%Y-%m-%d %H:%M:%S"), "only": sorted(only)})
    try:
        RC.dump(W, A, only, CATASTROPHE_ATOL, py)
    except SystemExit as e:
        RC.wr_json(f"{A}/DUMP_ABORT.json", {"exit": e.code, "reason": "wait_idle exited non-zero" if e.code == 7 else "SystemExit", "wall_sec": round(time.time() - t0, 1)})
        if os.path.exists(f"{A}/DUMP_RUNNING.json"):
            os.remove(f"{A}/DUMP_RUNNING.json")
        raise
    except BaseException as e:
        RC.wr_json(f"{A}/DUMP_ABORT.json", {"exit": 1, "reason": repr(e), "wall_sec": round(time.time() - t0, 1)})
        if os.path.exists(f"{A}/DUMP_RUNNING.json"):
            os.remove(f"{A}/DUMP_RUNNING.json")
        raise
    arms = {}
    for name, kind, var in RC.ARMS:
        if only and name not in only:
            continue
        cj = f"{A}/runs/{name}/cli.json"
        arms[name] = ({k: json.load(open(cj))[k] for k in ("exit", "wall_sec")} if os.path.exists(cj) else {"exit": None, "missing": True})
    missing = sorted(n for n, a in arms.items() if a.get("missing"))
    if os.path.exists(f"{A}/DUMP_RUNNING.json"):             # left in place only if the process is killed: a stale RUNNING marker then proves the death
        os.remove(f"{A}/DUMP_RUNNING.json")
    RC.wr_json(f"{A}/DUMP_DONE.json" if not missing else f"{A}/DUMP_ABORT.json",
               {"arms": arms, "missing": missing, "wall_sec": round(time.time() - t0, 1), "finished_local": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"[dump] {'DONE' if not missing else 'ABORT (missing arms: ' + ','.join(missing) + ')'} in {time.time() - t0:.0f}s", flush=True)
    if missing:
        sys.exit(8)


# ------------------------------------------------------------------------------------------------------------------------------------ analyse
def m_values(d):
    return sorted(x[3] for x in d.values())


def bound_M(F):
    return round(math.ceil(M_FACTOR * F / M_QUANTUM - 1e-9) * M_QUANTUM, 6)


def count_above(ms, M):
    return sum(1 for v in ms if v > M)


def phase_analyse(W, A, out, label):
    proxy = f"{W}/sets_10075_proxy.jsonl"
    ref_sha = RC.sha256_file(proxy)
    ref = RC.load_sets(proxy, STEP)
    incomplete, dl, arms = [], {}, {}
    if ref_sha != REF_SHA256_ROUND1:
        incomplete.append(f"reference sets sha256 {ref_sha} != round 1's {REF_SHA256_ROUND1}")
    # ---- N_full (round 1's identity-run dump, ALL 1,365 keys, CPU re-analysis) and the report-only invariance dumps
    idn = f"{W}/identity"
    nf = f"{idn}/new_full/onpolicy_rlfan_full_s0of1.jsonl"
    dl["N_full"] = RC.arm_deltas(nf, STEP, ref) if os.path.exists(nf) else {}
    for tag in IDENT_REPORT_ONLY:
        p = f"{idn}/new_{tag[2:]}/onpolicy_rlfan_{tag[2:]}_s0of1.jsonl"
        if os.path.exists(p):
            dl[tag] = RC.arm_deltas(p, STEP, ref)
    # ---- this round's arms
    clis = {}
    for name, kind, var in RC.ARMS:
        out_a = f"{A}/runs/{name}"
        f = f"{out_a}/onpolicy_rlfan_cal_s0of1.jsonl"
        cj, mj = f"{out_a}/cli.json", f"{out_a}/rl_fan_dump_cal_s0of1.manifest.json"
        if not (os.path.exists(f) and os.path.exists(cj) and os.path.exists(mj)):
            arms[name] = {"missing": True, "cli_exit": (json.load(open(cj))["exit"] if os.path.exists(cj) else None)}
            continue
        cli, man = json.load(open(cj)), json.load(open(mj))
        step = int(man["ckpt_step"])
        path = f
        if step != STEP:                                   # a refused checkpoint (exit 4) is judged offline with the step relabelled, as in round 1
            path = f"{out_a}/relabelled_step{STEP}.jsonl"
            RC.relabel_step(f, path, STEP)
        dl[name] = RC.arm_deltas(path, STEP, ref)
        cc = man.get("compare_sets") or {}
        clis[name] = {"cli_exit": cli["exit"], "ckpt_step": step, "gpu": man.get("gpu"), "torch": man.get("torch"), "ckpt_sha256": man.get("ckpt_sha256"),
                      "cli_max_abs_delta": cc.get("max_abs_delta"), "wall_sec": cli.get("wall_sec")}
        if step == STEP and cc and cc.get("max_abs_delta") is not None:
            mx = max((x[0] for x in dl[name].values()), default=0.0)
            clis[name]["cli_vs_analysis_max_delta_equal"] = bool(abs(cc["max_abs_delta"] - mx) <= 1e-12)
    # ---- completeness (PREREG section 8)
    for n in NULL_ARMS + DECISION_ARMS:
        want = N_FULL_SETS if n == "N_full" else N_KEYS
        got = len(dl.get(n, {}))
        if n != "N_full" and arms.get(n, {}).get("missing"):
            incomplete.append(f"{n}: NOT RUN (cli exit {arms[n].get('cli_exit')})")
        elif got != want:
            incomplete.append(f"{n}: {got} matched sets, need {want}")
    ms = {n: m_values(d) for n, d in dl.items()}
    verdict = {"round": 2, "rule_file": "PREREG_ROUND2.md", "a_dir": A, "label": label, "reference_sha256": ref_sha, "step": STEP}
    F = M = None
    if not incomplete:
        F = max(max(ms[n]) for n in NULL_ARMS)
        M = bound_M(F)
    need = math.ceil(MIN_FRAC_ABOVE * N_KEYS - 1e-9)
    failures, dec, nul, info = [], {}, {}, {}
    if M is not None:
        for n in NULL_ARMS:
            nul[n] = {"sets": len(ms[n]), "max_m": ms[n][-1], "above_M": count_above(ms[n], M)}
            if nul[n]["above_M"]:
                failures.append(f"null arm {n}: {nul[n]['above_M']} sets above M {M}")
        for n in DECISION_ARMS:
            v = sorted(ms[n], reverse=True)
            above, worst = count_above(v, M), v[0]
            thr = round(WORST_OVER_M * M, 9)
            ok2, ok3 = above >= need, worst >= thr
            m_prime = min(v[need - 1], worst / WORST_OVER_M) if len(v) >= need else None
            dec[n] = {"sets": len(v), "above_M": above, "need_above": need, "frac_above": above / len(v), "worst_m": worst, "worst_over_M": worst / M, "need_worst_m": thr,
                      "p50_m": RF.quantile_nearest_rank(sorted(v), 0.5), "separates": bool(ok2 and ok3), "largest_M_prime_that_would_separate": m_prime,
                      "M_prime_above_2F": (None if m_prime is None else bool(m_prime > M_FACTOR * F))}
            if not (ok2 and ok3):
                failures.append(f"decision arm {n}: {above}/{len(v)} sets above M {M} (need >= {need}), worst m {worst:.5g} = {worst / M:.2f}x M (need >= {WORST_OVER_M:g}x); "
                                f"largest M' that would separate it: {m_prime if m_prime is None else format(m_prime, '.5g')} (2F = {M_FACTOR * F:.5g})")
        for n in INFO_ARMS:
            if n in ms:
                info[n] = {"sets": len(ms[n]), "above_M": count_above(ms[n], M), "worst_m": ms[n][-1], "worst_over_M": ms[n][-1] / M, "p50_m": RF.quantile_nearest_rank(ms[n], 0.5)}
            else:
                info[n] = {"missing": True, "cli_exit": arms.get(n, {}).get("cli_exit")}
    ident = {n: {"sets": len(ms[n]), "max_m": ms[n][-1]} for n in IDENT_REPORT_ONLY if n in ms}
    verdict.update({"state": ("INCOMPLETE" if incomplete else ("PASS" if not failures else "FAIL")), "incomplete_reasons": incomplete, "F_max_null_m": F, "M": M,
                    "M_rule": f"ceil({M_FACTOR:g} x F / {M_QUANTUM} - 1e-9) x {M_QUANTUM}", "null_arms": nul, "decision_arms": dec, "info_arms": info,
                    "report_only_identity_invariance_dumps": ident, "failures": failures, "CHOSEN_MEAN_ATOL": (M if (M is not None and not failures) else None),
                    "catastrophe_atol_kept": CATASTROPHE_ATOL, "cli": clis})
    summary = {}
    for n, d in dl.items():
        v = ms[n]
        summary[n] = {"sets": len(v), "m_p50": RF.quantile_nearest_rank(v, 0.5), "m_p90": RF.quantile_nearest_rank(v, 0.9), "m_p99": RF.quantile_nearest_rank(v, 0.99), "m_max": v[-1],
                      "max_abs_delta": max((x[0] for x in d.values()), default=0.0), "above_M": (count_above(v, M) if M is not None else None), **clis.get(n, {})}
    RC.wr_json(f"{out}/arms_summary_round2.json", summary)
    RC.wr_json(f"{out}/verdict_round2.json", verdict)
    print(json.dumps({k: verdict[k] for k in ("state", "incomplete_reasons", "F_max_null_m", "M", "failures", "CHOSEN_MEAN_ATOL")}, indent=1))
    lines = ["| arm | role | CLI exit | sets | m p50 | m max (worst) | worst / M | sets above M | max-abs delta (round-1 statistic) |", "|---|---|---|---|---|---|---|---|---|"]
    for n in NULL_ARMS + DECISION_ARMS + INFO_ARMS:
        role = "NULL" if n in NULL_ARMS else ("DECISION" if n in DECISION_ARMS else "info")
        if n not in summary:
            lines.append(f"| {n} | {role} | {arms.get(n, {}).get('cli_exit', 'n/a')} | not run / no sets | | | | | |")
            continue
        s = summary[n]
        lines.append(f"| {n} | {role} | {s.get('cli_exit', 'n/a (CPU)')} | {s['sets']} | {s['m_p50']:.5g} | {s['m_max']:.5g} | {(s['m_max'] / M if M else float('nan')):.2f} | {s['above_M']} | {s['max_abs_delta']:.4g} |")
    lines.append(f"\nstate {verdict['state']}; F = {F}; M = {M}; need >= {need} sets above M and worst m >= {WORST_OVER_M:g} x M on every DECISION arm.")
    open(f"{out}/RESULTS_TABLE_round2.md", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    # ---- copy the CLI evidence (small) next to the verdict, as round 1 did
    for name, kind, var in RC.ARMS:
        for fn in ("cli.json", "cli.log", "rl_fan_dump_cal_s0of1.manifest.json"):
            s = f"{A}/runs/{name}/{fn}"
            if os.path.exists(s):
                os.makedirs(f"{out}/arms/{name}", exist_ok=True)
                shutil.copyfile(s, f"{out}/arms/{name}/{fn}")
    return verdict


# ------------------------------------------------------------------------------------------------------------------------------------ selftest
def phase_selftest(py):
    """MUTATE, do not inspect: with a waiter that exits 3 the dump must abort (SystemExit 7) and must NOT have reached the dumper (no cli.json); with a waiter that exits 0 it must
    reach the dumper (cli.json exists; the dumper itself fails on the missing inputs, which is irrelevant here)."""
    res = {}
    real_pkg = RC.PKG
    for label, code in (("waiter_fails", 3), ("waiter_passes", 0)):
        W = tempfile.mkdtemp(prefix="rlfan_selftest_", dir=os.environ.get("TMP"))
        A = f"{W}/atol_cal2"
        os.makedirs(A)
        os.makedirs(f"{W}/pkg/refe")
        open(f"{W}/wait_idle.py", "w").write(f"import sys; sys.exit({code})\n")
        open(f"{W}/pkg/refe/rl_fan_dump.py", "w").write("import sys; sys.exit(9)\n")      # a stub dumper: the control must not touch the real model or the GPU
        RC.PKG = f"{W}/pkg"
        try:
            RC.dump(W, A, {"N0_b4"}, CATASTROPHE_ATOL, py)
            exited = None
        except SystemExit as e:
            exited = e.code
        finally:
            RC.PKG = real_pkg
        cj = f"{A}/runs/N0_b4/cli.json"
        res[label] = {"waiter_exit": code, "dump_exit": exited, "reached_dumper": os.path.exists(cj), "stub_dumper_exit": (json.load(open(cj))["exit"] if os.path.exists(cj) else None)}
        shutil.rmtree(W, ignore_errors=True)
    ok = (res["waiter_fails"]["dump_exit"] == 7 and not res["waiter_fails"]["reached_dumper"] and res["waiter_passes"]["dump_exit"] is None and res["waiter_passes"]["reached_dumper"]
          and res["waiter_passes"]["stub_dumper_exit"] == 9)
    print(json.dumps({"selftest": res, "guard_reachable_and_control_green": ok}, indent=1))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["prep", "dump", "analyse", "selftest"])
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--a-dir", default="", help="default <work>/atol_cal2; pointing it elsewhere (e.g. round 1's atol_cal) is a CODE SELF-TEST and labels the output so")
    ap.add_argument("--only", default="")
    ap.add_argument("--py-dumper", default="C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    W = a.work.replace("\\", "/")
    A = (a.a_dir or f"{W}/atol_cal2").replace("\\", "/")
    os.makedirs(f"{W}/tmp", exist_ok=True)
    os.environ.update(PYTHONIOENCODING="utf-8", TMP=f"{W}/tmp", TEMP=f"{W}/tmp")
    only = {x for x in a.only.split(",") if x}
    if a.phase == "prep":
        phase_prep(W, A, a.out)
    elif a.phase == "dump":
        phase_dump(W, A, only, a.py_dumper)
    elif a.phase == "analyse":
        label = "ROUND 2" if os.path.abspath(A) == os.path.abspath(f"{W}/atol_cal2") else f"CODE SELF-TEST on {A} (NOT round 2; decides nothing)"
        phase_analyse(W, A, a.out, label)
    else:
        sys.exit(phase_selftest(a.py_dumper))


if __name__ == "__main__":
    main()
