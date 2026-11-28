"""run_mutations_ec.py -- the mutation runner of the generated wrapper v8f_t1_eval_reader_rule_rzgfhqe.py (card ADD79, form RRe). Each mutation changes ONE fragment of the new code in a COPY of
the wrapper inside tmp_mut/ (the real file is never touched); test_estclose_wrapper.py is run against the copy (ESTCLOSE_WRAPPER) and must go RED; the unmutated suite (control) must be GREEN
before and after. Sequential, one pytest process at a time, CPU only. Pytest base folders are new names inside tmp_pt/ (nothing is deleted or moved).
  bash ../ops/pycap.sh run_mutations_ec.py [--only m1,m2] [--out MUTATIONS_EC.json]
Writes MUTATIONS_EC.json (the list, the result of each run, the two control runs). Exit 0 = every mutation red and both control runs green."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUT = HERE / "tmp_mut"
PT = HERE / "tmp_pt"
WRAP = HERE / "v8f_t1_eval_reader_rule_rzgfhqe.py"

# name -> (what it breaks, exact text of ONE fragment of the wrapper, its replacement)
MUTATIONS = {
    "m01_history_boundary_off_by_one": ("an estimate is given one frame before the history is complete", 'if e - need + 1 >= 0 and e < len(s.d["feats"]):', 'if e - need >= 0 and e < len(s.d["feats"]):'),
    "m02_clip_end_allowed": ("a look on the frame after the last frame of the clip still gets an estimate", 'if e - need + 1 >= 0 and e < len(s.d["feats"]):', 'if e - need + 1 >= 0 and e <= len(s.d["feats"]):'),
    "m03_short_history_fed_as_nan": ("a look without a full history feeds NaN instead of 0.0", "    return np.where(est_ok, c_est, 0.0)", "    return np.where(est_ok, c_est, np.nan)"),
    "m04_short_history_fed_as_tenth": ("a look without a full history feeds 0.1 m/s instead of 0.0", "    return np.where(est_ok, c_est, 0.0)", "    return np.where(est_ok, c_est, 0.1)"),
    "m05_presence_before_gap": ("the first two input columns (gap, presence) are swapped", 'S["reads"][k] = (float(a), float(b), float(d))', 'S["reads"][k] = (float(b), float(a), float(d))'),
    "m06_frames_newest_first": ("the history is stacked newest frame first", "for f in range(e - H + 1, e + 1)], np.float64)", "for f in range(e, e - H, -1)], np.float64)"),
    "m07_closing_scale_one": ("the closing output of a frame is scaled by 1 instead of 10 m/s", "    c = (o[:, 2].astype(np.float64) * CLOS_SCALE_MS).astype(np.float32)\n", "    c = (o[:, 2].astype(np.float64) * 1.0).astype(np.float32)\n"),
    "m08_clip_not_in_the_cache_key": ("the read cache is keyed by frame only, clips share reads", 'wanted[(int(s.ci), f)] = s.d["feats"]', 'wanted[(0, f)] = s.d["feats"]'),
    "m09_mid_look_closing_zero": ("a later look feeds 0.0 instead of the estimate", '(est_fed(c_est, est_ok) if form == "RRe" else np.zeros(B, np.float64))', "np.zeros(B, np.float64)"),
    "m10_mid_look_estimate_at_the_tick_frame": ("a later look takes the estimate of frame j instead of j + q", '_est_at(st, rd, q) if form == "RRe"', '_est_at(st, rd, 0) if form == "RRe"'),
    "m11_tick_estimate_one_frame_late": ("the tick look takes the estimate of frame j + 1", '_est_at(states, STATE["reader"], 0)', '_est_at(states, STATE["reader"], 1)'),
    "m12_count_line_full_short_swapped": ("the scored full and short counts are swapped in the run line", '% (sf, ss, pf, ps, S["sha256"])', '% (ss, sf, pf, ps, S["sha256"])'),
    "m13_flag_accepted_with_other_forms": ("--closing-est is accepted with every form", 'if mine.form != "RRe" and mine.closing_est:', "if False:"),
    "m14_pre_file_keeps_main_arrays": ("the pre-roll rule log keeps the main-phase c_est / est_ok arrays", '            meta.pop("c_est", None)\n', "            pass\n"),
    "m15_mid_row_logs_a_wrong_estimate": ("the later-look log carries zeros and True instead of the estimate and its flag", 'row["c_est"], row["est_ok"] = c_est, est_ok', 'row["c_est"], row["est_ok"] = np.zeros(B), np.ones(B, bool)'),
    "m16_tick_returns_zero_closing": ("the tick look returns the form-RR closing instead of the estimate", "(c_estv if c_estv is not None else c_fed)", "c_fed"),
    "m17_head_returns_presence_first": ("the head returns (presence, gap, closing) instead of (gap, presence, closing)", "    return g, p, c\n", "    return p, g, c\n"),
    "m18_tick_estimate_not_logged": ("the tick-level estimate is not written to the log column", '    STATE["cols"]["c_est"].append(c_est)\n', "    pass\n"),
    "m19_feed_without_RRe": ("the reader feed does not know form RRe", 'elif form in ("RR", "RRc", "RRh", "RRd", "RRe"):', 'elif form in ("RR", "RRc", "RRh", "RRd"):'),
    "m20_mid_branch_without_RRe": ("the later-look reader branch does not take form RRe", 'if form in ("RR", "RRc", "RRh", "RRe"):', 'if form in ("RR", "RRc", "RRh"):'),
    "m21_embeddings_never_cached": ("the frame embeddings of earlier frames are recomputed at every look", '            if (ci, ff) not in S["emb"]:\n', "            if True:\n"),
    "m22_planner_does_not_bind_states": ("the planner does not hand its states to the estimate", '            STATE["est_states"] = self.states ', '            STATE["est_states"] = None '),
    "m23_save_mid_without_estimate_keys": ("the later-look file does not write c_est and est_ok", 'for f in MID_FIELDS + (MID_EST if STATE["form"] == "RRe" else ())}', "for f in MID_FIELDS}"),
    "m24_mix_closing_zero_for_RRe": ("the mix closing source of RRe is zero", '"RRd": "head", "RRe": "head"}', '"RRd": "head"}'),
    "m25_rule_log_without_estimate": ("the rule log file is written without the estimate arrays", 'dict(meta, **est_tick_arrays(log)) if f == "RRe" else meta', "meta"),
    "m26_no_run_line_at_the_end": ("the run text line is not written", "        finish_preroll()\n        write_est_line()\n", "        finish_preroll()\n        pass\n"),
}


def apply(name, text):
    _, old, new = MUTATIONS[name]
    n = text.count(old)
    if n != 1:
        raise SystemExit("mutation %s: the fragment occurs %d times, expected exactly 1" % (name, n))
    return text.replace(old, new)


def run_suite(tag, wrapper, timeout=1800):
    env = dict(os.environ)
    env.update({"ESTCLOSE_WRAPPER": str(wrapper), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    PT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", str(HERE / "test_estclose_wrapper.py"), "--basetemp", str(PT / tag)], env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=timeout, cwd=str(HERE))
    out = r.stdout + r.stderr
    failed = re.findall(r"^FAILED (\S+)", out, re.M) + re.findall(r"^ERROR (\S+::\S+)", out, re.M)
    summ = re.findall(r"(\d+) (passed|failed|error)", out)
    return {"rc": r.returncode, "failed_tests": failed, "summary": " ".join("%s %s" % x for x in summ), "seconds": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-3:]}


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / "MUTATIONS_EC.json"))
    a = ap.parse_args(argv)
    stamp = time.strftime("%H%M%S")
    names = [n for n in MUTATIONS if not a.only or n.split("_")[0] in a.only.split(",") or n in a.only.split(",")]
    base = WRAP.read_text(encoding="utf-8")
    MUT.mkdir(parents=True, exist_ok=True)
    res = {"wrapper": str(WRAP), "wrapper_sha256_16": hashlib.sha256(base.encode("utf-8")).hexdigest()[:16], "mutations": {}}
    res["control_before"] = run_suite("ctl_before_" + stamp, WRAP)
    print("control before:", res["control_before"]["summary"], flush=True)
    for n in names:
        d = MUT / (n + "_" + stamp)
        d.mkdir(parents=True, exist_ok=True)
        f = d / WRAP.name
        f.write_text(apply(n, base), encoding="utf-8", newline="\n")
        r = run_suite("mut_%s_%s" % (n, stamp), f)
        r["what"], r["old"], r["new"] = MUTATIONS[n][0], MUTATIONS[n][1], MUTATIONS[n][2]
        r["red"] = r["rc"] != 0 and bool(r["failed_tests"])
        res["mutations"][n] = r
        print("%-44s %s  %s" % (n, "RED" if r["red"] else "GREEN (not caught)", ",".join(x.split("::")[-1] for x in r["failed_tests"])[:100]), flush=True)
    res["control_after"] = run_suite("ctl_after_" + stamp, WRAP)
    print("control after:", res["control_after"]["summary"], flush=True)
    ok = res["control_before"]["rc"] == 0 and res["control_after"]["rc"] == 0 and all(v["red"] for v in res["mutations"].values())
    res["all_red_and_controls_green"] = ok
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print("ZZ MUTATIONS-EC", "PASS" if ok else "FAIL", "%d mutations" % len(res["mutations"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
