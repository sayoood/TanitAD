"""pa_verdict_amend2.py -- PREREG_REFCV9_INMODEL_PA AMENDMENT2 change 3: the Phase A word READ from the already-computed bars
(result/R9PF_PA_BARS.json 497a551d...) with K-EQ replaced by the device-matched K-EQ (result/R9PF_KEQ_AMEND2.json), labelled
"K-EQ under AMENDMENT2 (post-failure, device-matched)" -> result/R9PF_PA_VERDICT_AMEND2.{json,md}.  No bar, arm, population, estimator or tolerance is
recomputed or changed; the word logic is a3_bars.py's, verbatim.  Refuses unless K-EQ(AMEND2) PASSED.  The run-1 VOID verdict is not touched.
CPU only, no model.  NEW file (EvalFlyWheel run 2, 2026-10-26).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

BARS = RC.RESULT / "R9PF_PA_BARS.json"
BARS_SHA = "497a551d72ab9a92f6bed7ab56cc4d1156a0fe48d6cb55514ef3df9fd51bcad6"
KEQA2 = RC.RESULT / "R9PF_KEQ_AMEND2.json"
AMEND2 = RC.WORK / "PREREG_REFCV9_INMODEL_PA_AMENDMENT2.md"
AMEND2_SHA = "865c95692383f78a0c6355003c6d1b0ec63413a5816827babacb28cc5674cc3d"
VOID = {"R9PF_PA_VERDICT.json": "8c6f3c4602aebc30c12e5a0ed70ba24f37b54a51bf804c483b937c6dcc3bdf65",
        "R9PF_PA_VERDICT.md": "0867012cb54f7bb791b18c79fb8846969c0619032c98f2c26150e0b7acdf69cd",
        "R9PF_KEQ.json": "d1de8a28b421379659b24bfa38787e30b3f8ce0452268db367d90faf7b281da7"}
LABEL = "K-EQ under AMENDMENT2 (post-failure, device-matched)"
TEST = "--render-test" in sys.argv      # RENDER TEST ONLY: reads the plumbing-test K-EQ file, ignores its PASS, writes to scratch -- never result/
if TEST:
    KEQA2 = RC.SCRATCH / "KEQ_AMEND2_PLUMBING_TEST.json"
OUTCOME = {"PASS": "PASS (A) -> R9PF is refcv9's speed decode; the Phase-B compute request goes to the PI with these numbers; a refcv8 deliverable needs its OWN SHIP-R8 amendment.",
           "PARTIAL": "PARTIAL (PA-1a xor PA-1b) -> a new operating point, not a frontier move; next: a larger clean split (V-C) and the decision's AUC vs the lon head's beside.",
           "INERT": "INERT -> the decision SOURCE adds nothing at 297 clips; the lever moves to training the lon decision in the trunk (PLAN s12, Phase B).",
           "FAIL-GUARD": "FAIL-GUARD (PA-G) -> name the rows (below); next a constrained stage-2 objective, pre-registered first.",
           "FAIL": "FAIL -> no named branch beyond the failed bars.", "VOID": "VOID -> a control failed; the bars are BESIDE only."}


def main() -> int:
    RC.check_prereg()
    amend1 = RC.check_amendment1()
    got = RC.sha256_file(AMEND2)
    seal = RC.hex64(AMEND2.with_suffix(".sha256").read_text(encoding="utf-8-sig"))
    if not (len(got) == 64 and got == seal == AMEND2_SHA):
        raise SystemExit("[pa2] AMENDMENT2 seal INCONCLUSIVE / mismatch -- refusing")
    if RC.sha256_file(BARS) != BARS_SHA:
        raise SystemExit("[pa2] R9PF_PA_BARS.json is not 497a551d... -- refusing")
    for k, v in VOID.items():
        if RC.sha256_file(RC.RESULT / k) != v:
            raise SystemExit(f"[pa2] run-1 record {k} changed -- refusing")
    b = json.loads(BARS.read_text(encoding="utf-8"))
    k2 = json.loads(KEQA2.read_text(encoding="utf-8"))
    keq_ok = bool(k2["K-EQ_AMEND2"]["PASS"])
    if not keq_ok and not TEST:
        raise SystemExit("[pa2] K-EQ under AMENDMENT2 FAILED -- the VOID verdict stands; nothing to write")
    failed = [k for k in b["failed_controls"] if k != "K-EQ"]
    p, pa_r_holds, inert_flag = b["bars_primary"], b["PA-R"]["holds"], b["inert_read"]["INERT"]
    PASS = bool(p["PA-1a"] and p["PA-1b"] and p["PA-2"] and p["PA-G"] and pa_r_holds)
    # a3_bars.py's word logic, verbatim
    if failed:
        word = "VOID"
    elif PASS:
        word = "PASS"
    elif p["PA-1a"] != p["PA-1b"]:
        word = "PARTIAL"
    elif inert_flag:
        word = "INERT"
    elif not p["PA-G"]:
        word = "FAIL-GUARD"
    else:
        word = "FAIL"
    prim = b["primary_arm"]
    ba = b["bars_all_heads"]
    pag_rows = {t: ba[t]["PA-G_worse_rows"] for t in ("R9PF_fs0", "R9PF_fs1", "R9PF_fs2")}
    ks = k2["K-EQ_AMEND2"]["per_seed"]
    bs = k2["BESIDE_cuda_vs_cpu_residual"]["per_seed"]
    out = {"tool": "pa_verdict_amend2.py", "script_sha256": RC.sha256_file(__file__), "label": LABEL, "word": word, "PASS": PASS,
           "prereg": {"sha256": RC.PREREG_SHA}, "amendment1": amend1, "amendment2": {"file": str(AMEND2), "sha256": got, "sealed": "2026-10-26 16:20:56 Berlin, post-failure, disclosed"},
           "tier": b["tier"], "estimator": b["estimator"], "primary_arm": prim,
           "bars_primary_PA-G_row_names": pag_rows[prim],
           "bars_primary": p, "PA-R": b["PA-R"], "inert_read": b["inert_read"], "PA-G_worse_rows_by_fit_seed": pag_rows,
           "controls": {"K-EQ_run1_as_written": {"PASS": False, "file": str(RC.RESULT / "R9PF_KEQ.json")},
                        "K-EQ_AMEND2": {"PASS": keq_ok, "file": str(KEQA2), "sha256": RC.sha256_file(KEQA2),
                                        "per_seed": {s: {x: v[x] for x in ("max_abs_m", "max_ratio_to_2ulp", "n_windows_over_tol", "decision_agreement",
                                                                             "n_windows_joined_to_eval_inputs")} for s, v in ks.items()}},
                        "BESIDE_cuda_vs_cpu": bs,
                        "other_controls_from_bars": {k: (v if isinstance(v, bool) else v.get("PASS")) for k, v in b["controls"].items()
                                                     if k not in ("K-EQ",)},
                        "failed_controls_under_AMENDMENT2": failed},
           "committed_outcome_s5": OUTCOME[word],
           "bars_record": {"file": str(BARS), "sha256": BARS_SHA},
           "void_record_kept": {k: {"file": str(RC.RESULT / k), "sha256": v} for k, v in VOID.items()},
           "generated": time.strftime("%FT%T%z")}
    OUTD = RC.SCRATCH / "render_test" if TEST else RC.RESULT
    OUTD.mkdir(exist_ok=True)
    RC.jdump(out, OUTD / "R9PF_PA_VERDICT_AMEND2.json")
    # ---- md
    def yn(x):
        return "HOLDS" if x else "FAILS"
    L = [f"# R9PF_PA_VERDICT_AMEND2 -- {word}", "",
         f"**label:** {LABEL}. The run-1 verdict `result/R9PF_PA_VERDICT.json` (8c6f3c46...) is **VOID (K-EQ FAIL)** and is kept unchanged; this is a "
         f"SECOND, labelled reading of the SAME bars (`R9PF_PA_BARS.json` {BARS_SHA[:8]}...), with K-EQ replaced by AMENDMENT2's device-matched form.", "",
         f"**tier:** {b['tier']}  ", f"**estimator:** {b['estimator']}  ",
         "**questions answered:** another draw of EPISODES (bootstrap); another TRAINING run = R9 fit seeds 0/1/2 (PA-R); another INFERENCE run = seeds 0 / 1 apart.", "",
         f"**AMENDMENT2** sha256 {got} (sealed 16:20:56 Berlin, AFTER K-EQ failed -- disclosed as post-failure). Bars, arms, population, estimator and the "
         "2-ulp tolerance unchanged; only the device of the in-forward roll changed (CUDA -> CPU).", "",
         "## K-EQ under AMENDMENT2 (CPU in-forward vs CPU offline; the first eval139 clips in sha12 order until >= 500 windows: 515 windows / 15 clips; both inference seeds)", "",
         "| seed | max abs d (m) | max d / 2 ulp | windows over 2 ulp | decision agreement | PASS |", "|---|---|---|---|---|---|"]
    for s, v in ks.items():
        L.append(f"| {s} | {v['max_abs_m']:.3g} | {v['max_ratio_to_2ulp']:.3g} | {v['n_windows_over_tol']} / {v['n_windows_joined_to_eval_inputs']} | "
                 f"{v['decision_agreement']:.4f} | {v['PASS']} |")
    L += ["", "**Beside -- the CUDA-vs-CPU residual (the device a CUDA deployment would run on):**", "",
          "| seed | CUDA in-forward vs CPU offline: max abs d / windows over 2 ulp / decision agreement | CUDA vs CPU in-forward directly: path bit-equal / plan max abs d / decision agreement |",
          "|---|---|---|"]
    for s, v in bs.items():
        a, d = v["a_cuda_inforward_vs_cpu_offline"], v["b_cuda_inforward_vs_cpu_inforward"]
        dd = (f"{d['selected_path_traj_preretime']['n_bit_equal']} / {d['n_windows']}; {d['in_forward_plan']['max_abs_m']:.3g} m; {d['decision_agreement']:.4f}"
              if d.get("same_window_sequence") else "window sequence differs")
        L.append(f"| {s} | {a['max_abs_m']:.3g} m / {a['n_windows_over_tol']} of {a['n_windows_joined_to_eval_inputs']} / {a['decision_agreement']:.4f} | {dd} |")
    L += ["", f"## Word: **{word}** (a3_bars.py's word logic, verbatim; failed controls under AMENDMENT2: {failed or 'none'})", "",
          f"**every bar, as written (primary {prim}):** PA-1a {yn(p['PA-1a'])}, PA-1b {yn(p['PA-1b'])}, PA-2 {yn(p['PA-2'])}, PA-G {yn(p['PA-G'])}, "
          f"PA-R {yn(pa_r_holds)} -> PASS = {PASS}; INERT = {inert_flag}.", "",
          "| fit seed | PA-1a | PA-1b | PA-2 | PA-G (beside for fs1/fs2) |", "|---|---|---|---|---|"]
    for k in ("fs0", "fs1", "fs2"):
        v = b["PA-R"]["per_fit_seed"][k]
        L.append(f"| {k} | {v['PA-1a']} | {v['PA-1b']} | {v['PA-2']} | {ba['R9PF_' + k]['PA-G']} ({len(pag_rows['R9PF_' + k])} rows) |")
    L += ["", f"**committed outcome (prereg s5):** {OUTCOME[word]}", "",
          f"## PA-G rows separated WORSE than H2CN_DEPLOY8 -- primary {prim} ({len(pag_rows[prim])} rows; 95 % paired interval)", ""]
    L += [f"* `{r}`" for r in pag_rows[prim]]
    for k in ("R9PF_fs1", "R9PF_fs2"):
        L += ["", f"Beside -- {k}: {len(pag_rows[k])} rows"] + [f"* `{r}`" for r in pag_rows[k]]
    L += ["", "## What this reading does NOT change", "",
          "* The bars, cells and four-family tables are those of `R9PF_PA_BARS.json` / the run-1 `R9PF_PA_VERDICT.md` (both unchanged); only the control "
          "row K-EQ is re-read.",
          "* Every number is T1 self-action OPEN LOOP on logged eval139 frames, paired episode-cluster bootstrap over EPISODES (B 2000 seed 0); one refcv8 "
          "trunk seed; R9 profile fit seeds 0/1/2 are the training-variance read.",
          "* PREREG_R9PF_LH's s2 precondition (R9-PF's K-EQ / K-OFF PASSED) is satisfied in the same labelled sense -> R9PF_LH runs "
          "(`result/R9PF_LH_VERDICT_RUN2.{json,md}`)."]
    (OUTD / "R9PF_PA_VERDICT_AMEND2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[pa2] word {word}  ZZPA2_{word}ZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
