"""run_mutations_sc.py -- the mutation runner of the generated wrapper v8f_t1_eval_reader_rule_rzgfhqet.py (card ADD83, form RRe with --closing-table). Each mutation changes ONE fragment of the new code in a COPY of
the wrapper inside tmp_mut/ (the real file is never touched); test_shuf_wrapper.py is run against the copy (SHUFCLOSE_WRAPPER) and must go RED; the unmutated suite (control) must be GREEN
before and after. Sequential, one pytest process at a time, CPU only. Pytest base folders are new names inside tmp_pt/ (nothing is deleted or moved).
  bash ../ops/pycap.sh run_mutations_sc.py [--only m1,m2] [--out MUTATIONS_SC.json]
Writes MUTATIONS_SC.json (the list, the result of each run, the two control runs). Exit 0 = every mutation red and both control runs green."""
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
WRAP = HERE / "v8f_t1_eval_reader_rule_rzgfhqet.py"

# name -> (what it breaks, exact text of ONE fragment of the wrapper, its replacement)
MUTATIONS = {
    "s01_table_frame_one_late": ("the look takes the table value of frame j + e_off + 1", "e = np.array([int(s.j) + int(e_off) for s in states], np.int64)", "e = np.array([int(s.j) + int(e_off) + 1 for s in states], np.int64)"),
    "s02_table_frame_ignores_the_later_look_offset": ("a later look takes the table value of frame j instead of j + q", "e = np.array([int(s.j) + int(e_off) for s in states], np.int64)", "e = np.array([int(s.j) + 0 for s in states], np.int64)"),
    "s03_position_instead_of_clip_index": ("the table is read by the position of the window in the batch, not by its clip index", "ci = np.array([int(s.ci) for s in states], np.int64)", "ci = np.arange(len(states), dtype=np.int64)"),
    "s04_nan_value_counts_as_full": ("a table row that holds NaN is marked full", "est_ok = hit & np.isfinite(v)", "est_ok = hit"),
    "s05_every_look_marked_full": ("every look is marked full, rows or not", "est_ok = hit & np.isfinite(v)", "est_ok = np.ones(B, bool)"),
    "s06_small_stride_key": ("the key of the table uses a stride of 16 instead of 2**20 in the lookup", "key = np.where(inside, ci * TABLE_STRIDE + e, -1)", "key = np.where(inside, ci * 16 + e, -1)"),
    "s07_table_path_never_taken": ("the estimate does not look at the table", '    if S.get("table") is not None:', "    if False:"),
    "s08_duplicate_rows_accepted": ("two rows with the same clip index and frame are accepted", "    if len(key) > 1 and bool(np.any(key[1:] == key[:-1])):", "    if False:"),
    "s09_table_not_sorted": ("the table keys are not sorted before the search", "    key, v = key[order], v[order]\n", "    pass\n"),
    "s10_table_sha_not_its_own": ("the run line carries another file hash than the table's", '"sha256": W.sha256_file(path), "emb": {}, "reads": {}, "cache": True, "embed_calls": 0, "rule_log": rule_log}\n    for k in MID_EST:\n        STATE["cols"][k] = []\n        STATE["mid"]["rows"][k] = []\n    return STATE["est"]["table"]', '"sha256": "0" * 64, "emb": {}, "reads": {}, "cache": True, "embed_calls": 0, "rule_log": rule_log}\n    for k in MID_EST:\n        STATE["cols"][k] = []\n        STATE["mid"]["rows"][k] = []\n    return STATE["est"]["table"]'),
    "s11_table_flag_accepted_with_other_forms": ("--closing-table is accepted with every form", 'if mine.form != "RRe" and mine.closing_table:', "if False:"),
    "s12_both_sources_accepted": ("--closing-est and --closing-table together are accepted", "    if mine.closing_est and mine.closing_table:", "    if False:"),
    "s13_table_flag_not_loaded": ("form RRe with --closing-table loads the estimator path", "            if mine.closing_table:\n                load_table(", "            if False:\n                load_table("),
    "s14_no_source_accepted": ("form RRe without any source is accepted", '    if mine.form == "RRe" and not (mine.closing_est or mine.closing_table):', "    if False:"),
    "s15_infinite_value_kept_in_the_column": ("the logged estimate keeps an infinite table value", "        c_est = np.where(est_ok, v, np.nan)\n", "        c_est = v\n"),
    "s16_missing_array_not_refused": ("a table without the est array is not refused cleanly", '        for k in ("clip_index", "frame", "est"):', '        for k in ("clip_index", "frame"):'),
    "s17_negative_index_accepted": ("negative clip index, negative frame and a frame at or above the stride are accepted", "    if len(ci) and (ci.min() < 0 or fr.min() < 0 or fr.max() >= TABLE_STRIDE):", "    if False:"),
    "s18_missing_row_takes_the_neighbour": ("a look without a row takes the value of the next row", "        hit = inside & (pos < len(T[\"key\"])) & (T[\"key\"][posc] == key)", "        hit = inside & (pos < len(T[\"key\"]))"),
    "s19_nan_to_zero_in_table": ("NaN values of the table are read as 0.0 and marked full", "        v = np.where(hit, T[\"val\"][posc], np.nan)\n", "        v = np.where(hit, np.nan_to_num(T[\"val\"][posc]), np.nan)\n"),
    "s20_table_rows_not_sized_checked": ("arrays of different lengths are accepted", "    if not (ci.ndim == fr.ndim == v.ndim == 1 and len(ci) == len(fr) == len(v)):", "    if False:"),
    "s21_est_file_branch_dropped": ("form RRe with --closing-est does not load the estimator", "                load_estimator(mine.closing_est, STATE[\"reader\"], mine.rule_log)\n        if f == \"RRd\":", "                pass\n        if f == \"RRd\":"),
}


def apply(name, text):
    _, old, new = MUTATIONS[name]
    n = text.count(old)
    if n != 1:
        raise SystemExit("mutation %s: the fragment occurs %d times, expected exactly 1" % (name, n))
    return text.replace(old, new)


def run_suite(tag, wrapper, timeout=1800):
    env = dict(os.environ)
    env.update({"SHUFCLOSE_WRAPPER": str(wrapper), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    PT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", str(HERE / "test_shuf_wrapper.py"), "--basetemp", str(PT / tag)], env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=timeout, cwd=str(HERE))
    out = r.stdout + r.stderr
    failed = re.findall(r"^FAILED (\S+)", out, re.M) + re.findall(r"^ERROR (\S+::\S+)", out, re.M)
    summ = re.findall(r"(\d+) (passed|failed|error)", out)
    return {"rc": r.returncode, "failed_tests": failed, "summary": " ".join("%s %s" % x for x in summ), "seconds": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-3:]}


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / "MUTATIONS_SC.json"))
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
    print("ZZ MUTATIONS-SC", "PASS" if ok else "FAIL", "%d mutations" % len(res["mutations"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
