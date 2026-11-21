"""run_mutations_tb.py -- mutation runner of card ADD84 (TABCLOSE10K): the reader (through its --selftest), the queue, the pod read script and the chain (through test_tabclose.py).
Each mutation changes ONE fragment of one tool in a COPY inside tmp_mut/ (the real file is never touched); the check of the copy has to FAIL. The unmutated checks (control) have to pass before and after.
Sequential, one process at a time, CPU only. Writes MUTATIONS_TB.json. Exit 0 = every mutation red and the control runs green.
  bash ../ops/pycap.sh run_mutations_tb.py [--only m1,m2] [--out MUTATIONS_TB.json]"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUT = HERE / "tmp_mut"
PT = HERE / "tmp_pt"
V = HERE.parent
RD, QU, PR, CH = "read_tabclose10k.py", "pod_tabclose10k_queue.sh", "podread_tabclose10k.sh", "chain_read_tabclose10k.sh"
KW_ = ('TAB="$TABLES/tab_%s_s${k}.npz"')

# name -> (file, what it breaks, exact text of ONE fragment, its replacement)
M = {
    # ---- reader: constants and controls
    "r01_a4u3_sd_upper": (RD, "a4_u3 sd ratio may reach 2.0", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.90", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 2.0, 0.15, 0.90"),
    "r02_a4u3_sd_lower": (RD, "a4_u3 sd ratio may go down to 0.5", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.90", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.50, 1.25, 0.15, 0.90"),
    "r03_a4u3_mean": (RD, "a4_u3 mean may reach 0.5", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.90", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.50, 0.90"),
    "r04_a4u3_share": (RD, "a4_u3 share of full looks may be 0.5", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.90", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25, 0.15, 0.50"),
    "r05_a4u3_share_not_judged": (RD, "a4_u3 does not judge the share", '"PASS": bool(c1 and c2 and c3), "complete": True', '"PASS": bool(c1 and c2), "complete": True'),
    "r06_a3_tolerance": (RD, "a3 allows two windows", "A3_TOL = 1.0 / 325.0", "A3_TOL = 2.0 / 325.0"),
    "r07_half_is_strict": (RD, "a share of exactly 0.5 does not reach half", "if sh_ is not None and sh_ >= HALF:", "if sh_ is not None and sh_ > HALF:"),
    "r08_half_value": (RD, "half is 0.4", "HALF = 0.5", "HALF = 0.4"),
    "r09_u2_tolerance": (RD, "a2 tolerance 1e-5", "TOL_U2 = 1e-9", "TOL_U2 = 1e-5"),
    "r10_rec_hash_not_judged": (RD, "record control does not compare the table hash", "bad = sorted(n for n in want if shas.get(n) is None or tabs.get(n) != shas.get(n))", "bad = sorted(n for n in want if shas.get(n) is None)"),
    "r11_rec_pass_not_judged": (RD, "record control ignores the record's PASS", 'if not isinstance(d, dict) or d.get("PASS") is not True:', "if not isinstance(d, dict):"),
    "r12_rec_no_table_expected": (RD, "record control passes with no table expected", 'if not isinstance(tabs, dict) or not want:', "if not isinstance(tabs, dict):"),
    "r13_a3_cutoff": (RD, "a3 judges 2 tolerances", '"PASS": bool(mx <= tol)', '"PASS": bool(mx <= 2 * tol)'),
    "r14_r1_hash_not_judged": (RD, "r1 does not compare the est line's file hash", 'sha_ok = bool(pe is not None and pe["file_sha256"]', 'sha_ok = True or bool(pe is not None and pe["file_sha256"]'),
    "r15_x1_nonzero_not_judged": (RD, "x1 accepts a closing that is 0 everywhere", 'g1 = r["present"] > 0 and r["present_nonzero"] > 0 and', 'g1 = r["present"] > 0 and'),
    "r16_m1_rows_not_compared": (RD, "m1 does not compare the row count with RR", "good = bool(n and base and n == base)", "good = bool(n and base)"),
    # ---- reader: verdicts
    "v01_tk_agec_not_judged": (RD, "TK-SUPPORTED ignores AGE C", "sup = decide([c_lower(r_a), c_nh(r_a, \"U\")", "sup = decide([True, c_nh(r_a, \"U\")"),
    "v02_tk_ageu_not_judged": (RD, "TK-SUPPORTED ignores AGE U", 'c_lower(r_a), c_nh(r_a, "U"), no_worse(r_a)', 'c_lower(r_a), True, no_worse(r_a)'),
    "v03_tk_agefam_not_judged": (RD, "TK-SUPPORTED ignores the AGE families", 'c_nh(r_a, "U"), no_worse(r_a), c_nh(r_f, "C")', 'c_nh(r_a, "U"), True, c_nh(r_f, "C")'),
    "v04_tk_freshc_not_judged": (RD, "TK-SUPPORTED ignores FRESH C", 'no_worse(r_a), c_nh(r_f, "C"), c_nh(r_f, "U")', 'no_worse(r_a), True, c_nh(r_f, "U")'),
    "v05_tk_freshu_not_judged": (RD, "TK-SUPPORTED ignores FRESH U", 'c_nh(r_f, "C"), c_nh(r_f, "U"), no_worse(r_f)', 'c_nh(r_f, "C"), True, no_worse(r_f)'),
    "v06_tk_freshfam_not_judged": (RD, "TK-SUPPORTED ignores the FRESH families", 'c_nh(r_f, "U"), no_worse(r_f)])', 'c_nh(r_f, "U"), True])'),
    "v07_tk_freshc_needs_lower": (RD, "FRESH C has to be LOWER", 'c_nh(r_f, "C"), c_nh(r_f, "U")', 'c_lower(r_f), c_nh(r_f, "U")'),
    "v08_tk_beyond_wrong_pair": (RD, "TK-BEYOND-L25 reads TOK-VS-RR", 'decide([c_lower(wd_get(words, "AGE", "TOK-VS-EST-L25"))])', 'decide([c_lower(wd_get(words, "AGE", "TOK-VS-RR"))])'),
    "v09_gain_u_not_judged": (RD, "GAIN ignores U", 'decide([c_lower(sh), c_nh(sh, "U")])', 'decide([c_lower(sh)])'),
    "v10_gain_u_needs_lower": (RD, "GAIN needs U LOWER", 'decide([c_lower(sh), c_nh(sh, "U")])', 'decide([c_lower(sh), c_lower(sh)])'),
    "v11_gain_beside_name": (RD, "the name beside GAIN is the U name", 'cname("SHUF-VS-RR", "AGE", "C", sr["C"])', 'cname("SHUF-VS-RR", "AGE", "U", sr["U"])'),
    "v12_f_beside_name": (RD, "the F name beside TK-SUPPORTED is the C name", 'cname("TOK-VS-RR", st, "F", r["F"])', 'cname("TOK-VS-RR", st, "C", r["C"])'),
    "v13_removes_u_not_judged": (RD, "REMOVES-CONTACT ignores U", "rem[x] = decide([c_lower(rr), c_nh(rr, \"U\")])", "rem[x] = decide([c_lower(rr)])"),
    "v14_removes_c_not_lower": (RD, "REMOVES-CONTACT needs only C not HIGHER", 'rem[x] = decide([c_lower(rr), c_nh(rr, "U")])', 'rem[x] = decide([c_nh(rr, "C"), c_nh(rr, "U")])'),
    "v15_label_level_needs_lower": (RD, "AT-LABEL-LEVEL needs C LOWER", 'at[x] = decide([c_nh(lc, "C")])', 'at[x] = decide([c_lower(lc)])'),
    "v16_requirement_arm_not_skipped": (RD, "an arm that does not remove contact can set the requirement", "        if rem[x] is False:\n            continue\n", ""),
    "v17_requirement_order": (RD, "the requirement is searched from 90 down", '((50, "W50"), (70, "W70"), (90, "W90"))', '((90, "W90"), (70, "W70"), (50, "W50"))'),
    "v18_requirement_unresolved": (RD, "an unavailable lower arm does not block the requirement", "        if rem[x] is None:\n            unresolved = x\n            break\n", ""),
    "v19_no_tested_name": (RD, "the none name is another", 'PRE + "-NO-TESTED-LEVEL-REACHES-HALF"', 'PRE + "-NO-LEVEL-REACHES-HALF"'),
    "v20_spread_wrong_pair": (RD, "SPREAD-ALONE reads W50-VS-RR", 'decide([c_lower(wd_get(words, "AGE", "W00-VS-RR"))])', 'decide([c_lower(wd_get(words, "AGE", "W50-VS-RR"))])'),
    "v21_mark_all": (RD, "the mark is written only when all names differ", "if any(a_ != b_ for a_, b_ in pair_):", "if all(a_ != b_ for a_, b_ in pair_):"),
    "v22_mark_label_not_compared": (RD, "the mark ignores the label-level names", 'pair_ = [(rem["W70"], rem["S70"]), (at["W70"], at["S70"])]', 'pair_ = [(rem["W70"], rem["S70"])]'),
    "v23_put_inverted": (RD, "a name reads the opposite", 'names[key] = PRE + "-" + (yes if cond else no)', 'names[key] = PRE + "-" + (no if cond else yes)'),
    "v24_decide_none_first": (RD, "an unavailable word decides before a failing one", "    if any(c is False for c in conds):\n        return False\n    if any(c is None for c in conds):\n        return None\n",
                              "    if any(c is None for c in conds):\n        return None\n    if any(c is False for c in conds):\n        return False\n"),
    "v25_worse_word": (RD, "no_worse looks for another word", 'w["fam"][f] != "WORSE"', 'w["fam"][f] != "HIGHER"'),
    # ---- reader: withholding and pairs
    "w01_b_not_withheld": (RD, "a failed part B control withholds nothing", 'if part == "B" and failed["B"]:', 'if False:'),
    "w02_shuf_pair_not_withheld": (RD, "SHUF-VS-RR is not withheld by a failed a4", 'if pair in ("TOK-VS-SHUF", "SHUF-VS-RR") and failed["SHUF"]:', 'if pair in ("TOK-VS-SHUF",) and failed["SHUF"]:'),
    "w03_a2_not_withheld": (RD, "a failed a2 withholds nothing", 'bad = failed["a2"].get(st) or []', "bad = []"),
    "w04_a2_first_form_only": (RD, "a failed a2 looks only at the first form of a pair", 'used = [f[len("RRe_"):] for f in forms if f in NEW_FORMS]', 'used = [f[len("RRe_"):] for f in forms[:1] if f in NEW_FORMS]'),
    "w05_a3_is_inherited": (RD, "a failed a3 is treated as inherited", 'if k in ("a0_tables", "a1_tables") or k.startswith("a3_"):', 'if k in ("a0_tables", "a1_tables"):'),
    "w06_a4u4_is_inherited": (RD, "a failed a4_u4 is treated as inherited", 'if k in ("a4_u1", "a4_u4") or k.startswith("a4_u3_"):', 'if k in ("a4_u1",) or k.startswith("a4_u3_"):'),
    "w07_a2_without_runs": (RD, "a failed a2 without a run list withholds nothing", 'f["a2"][k[3:]] = bad or list(NEW_ARMS)', 'f["a2"][k[3:]] = bad'),
    "w08_inherited_keeps_payload": (RD, "an inherited failure keeps the names", 'if payload and not fs["glob"]:', "if payload:"),
    "w09_blocked_pair_words_used": (RD, "a withheld pair still feeds a verdict", "                w2[st][pair] = None\n                continue\n", "                w2[st][pair] = w\n                continue\n"),
    "p01_shuf_pair_swapped": (RD, "TOK-VS-SHUF is STK minus TK", 'add("TOK-VS-SHUF", "RRe_TK", "RRe_STK", "A")', 'add("TOK-VS-SHUF", "RRe_STK", "RRe_TK", "A")'),
    "p02_shuf_vs_rr_swapped": (RD, "SHUF-VS-RR is RR minus STK", 'add("SHUF-VS-RR", "RRe_STK", "RR", "A")', 'add("SHUF-VS-RR", "RR", "RRe_STK", "A")'),
    "p03_s70_w70_swapped": (RD, "S70-VS-W70 is W70 minus S70", 'add("S70-VS-W70", "RRe_S70", "RRe_W70", "B")', 'add("S70-VS-W70", "RRe_W70", "RRe_S70", "B")'),
    "p04_label_pair_vs_rr": (RD, "X-VS-LABEL-CLOSING compares with RR", 'add(x + "-VS-LABEL-CLOSING", "RRe_" + x, "RRc", "B")', 'add(x + "-VS-LABEL-CLOSING", "RRe_" + x, "RR", "B")'),
    "p05_fresh_has_stk": (RD, "FRESH runs STK as well", 'arms = [x for x in arms_of(part) if st == "AGE" or x == "TK"]', 'arms = [x for x in arms_of(part) if st == "AGE" or x in ("TK", "STK")]'),
    "p06_run_name": (RD, "the run names use another stem", '"RRe_" + x: "RQ50e%sKF5%s" % (x, suf)', '"RRe_" + x: "RQ50c%sKF5%s" % (x, suf)'),
    "p07_name_pattern": (RD, "the setting and the pair swap places in a name", 'return "%s-%s-%s-%s-%s" % (PRE, pair, setting, kind, word)', 'return "%s-%s-%s-%s-%s" % (PRE, setting, pair, kind, word)'),
    "p08_share_sign": (RD, "the share has the opposite sign", "else (c_rr - c_x) / gap", "else (c_x - c_rr) / gap"),
    "p09_share_gap": (RD, "the gap is RRc minus RR", "gap = c_rr - c_rc", "gap = c_rc - c_rr"),
    "p10_share_zero_gap": (RD, "a zero gap gives a share of 0", "return None if gap == 0.0 else", "return 0.0 if gap == 0.0 else"),
    # ---- queue
    "q01_part_suffix": (QU, "the part suffix of the markers is empty", 'PS=""; [ "$PART" = ALL ] || PS="_$PART"', 'PS=""'),
    "q02_want_always": (QU, "a part switch runs everything", 'want() { [ "$PART" = ALL ] || [ "$PART" = "$1" ]; }', "want() { true; }"),
    "q03_stk_every": (QU, "STK rule every 10", 'KAPPA=0.2 EVERY=5 QMAP_FILE="$Q" TAB="$TABLES/tab_STK_s${k}.npz"', 'KAPPA=0.2 EVERY=10 QMAP_FILE="$Q" TAB="$TABLES/tab_STK_s${k}.npz"'),
    "q04_w00_no_pathexec": (QU, "W00 without the path executor", 'PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$Q" TAB="$TABLES/tab_W00_s${k}.npz"', 'PATHEXEC=0 KAPPA=0.2 EVERY=5 QMAP_FILE="$Q" TAB="$TABLES/tab_W00_s${k}.npz"'),
    "q05_w90_table": (QU, "W90 reads the W50 table", 'TAB="$TABLES/tab_W90_s${k}.npz"', 'TAB="$TABLES/tab_W50_s${k}.npz"'),
    "q06_s70_fresh": (QU, "S70 runs on the fresh grid", 'RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$Q" TAB="$TABLES/tab_S70_s${k}.npz"', 'READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$Q" TAB="$TABLES/tab_S70_s${k}.npz"'),
    "q07_tk_fresh_aged": (QU, "TK FRESH runs with a pre-roll", '    READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$Q" TAB="$TABLES/tab_TK_s${k}.npz" rollout I10RQ50eTKKF5Hb20', '    RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE="$Q" TAB="$TABLES/tab_TK_s${k}.npz" rollout I10RQ50eTKKF5Hb20'),
    "q08_tables_folder": (QU, "the table folder is est_shuf", "TABLES=${I10_TABLES:-$V/closeread/est_tab}", "TABLES=${I10_TABLES:-$V/closeread/est_shuf}"),
    "q09_done_marker": (QU, "the DONE marker has no part suffix", 'echo "$(ts) all steps done (part $PART)" > "$W/TABCLOSE10K_DONE$PS.txt"', 'echo "$(ts) all steps done (part $PART)" > "$W/TABCLOSE10K_DONE.txt"'),
    "q10_admit_pass_not_judged": (QU, "the admission ignores the record's pass for ALL", 'if part == "ALL" and rec.get("pass") is not True:', "if False:"),
    "q10b_admit_pass_for_parts": (QU, "the admission needs the record's pass for a part too", 'if part == "ALL" and rec.get("pass") is not True:', 'if rec.get("pass") is not True:'),
    "q11_admit_a4_not_needed": (QU, "part A is admitted without a4", 'need += ["a4_u1", "a4_u4"]', "need += []"),
    "q12_admit_a1_not_needed": (QU, "part B is admitted without a1", 'need += ["a0", "a1"]', 'need += ["a0"]'),
    "q13_admit_hash_not_judged": (QU, "the admission ignores a table hash", "if h != tabs[fn]:", "if False:"),
    "q14_admit_not_in_record": (QU, "the admission ignores a table that is not in the record", "if fn not in tabs:", "if False:"),
    "q15_admit_missing_file": (QU, "the admission ignores a missing table", "if not os.path.isfile(p):", "if False:"),
    "q16_admit_control_value": (QU, "the admission ignores a control that says fail", 'if not isinstance(d, dict) or d.get("PASS") is not True:', "if not isinstance(d, dict):"),
    "q17_tab_check_removed": (QU, "the admission is not run", '  adm=$(admit "$TABLES/TAB_TABLES_RECORD.json" "$TABLES" "$PART" "$ARMLIST") || stop "TABLE-ADMISSION $adm"\n', '  adm=skipped\n'),
    "q18_wrapper_check_removed": (QU, "the pod check of the wrapper is not required", '|| stop "NO-PASS-OF-THE-WRAPPER-CHECK"', "|| true"),
    "q19_tag_fixed": (QU, "the tag of the pod check cannot be changed", "TAG=${SC_TAG:-sk1}", "TAG=sk1"),
    "q20_banked_rr_not_needed": (QU, "the banked RR run at age is not required", "for nm in I10RQ50KF5P30Hb20 I10RLcKF5P30Hb20", "for nm in I10RLcKF5P30Hb20"),
    # ---- pod read scripts
    "d01_podread_arms_a": (PR, "part A does not need the STK table", 'ARMS_N="TK STK"', 'ARMS_N="TK"'),
    "d02_podread_part_option": (PR, "the part is not handed to the reader", '--tab-dir "$TABDIR" --part "$PART"', '--tab-dir "$TABDIR"'),
    "d03_podread_done_marker": (PR, "the read ignores the part marker", '[ -f "$W/TABCLOSE10K_DONE$PS.txt" ] || stop "QUEUE-NOT-DONE"', '[ -f "$W/TABCLOSE10K_DONE.txt" ] || stop "QUEUE-NOT-DONE"'),
    "d04_chain_links": (CH, "the chain does not link the S70 runs", "I10RQ50e{TK,STK,L100,W00,W50,W70,W90,S70}*_s[01]", "I10RQ50e{TK,STK,L100,W00,W50,W70,W90}*_s[01]"),
    "d05_chain_done_marker": (CH, "the chain waits for the full marker with a part switch", '[ -f "$W/TABCLOSE10K_DONE$PS.txt" ] && [ -f "$W/GO_READ_TABCLOSE10K.txt" ] && break', '[ -f "$W/TABCLOSE10K_DONE.txt" ] && [ -f "$W/GO_READ_TABCLOSE10K.txt" ] && break'),
    "d06_chain_go_file": (CH, "the chain does not wait for the go file", '[ -f "$W/TABCLOSE10K_DONE$PS.txt" ] && [ -f "$W/GO_READ_TABCLOSE10K.txt" ] && break', '[ -f "$W/TABCLOSE10K_DONE$PS.txt" ] && break'),
    "d07_chain_out_folder": (CH, "the part folder is not used", "OUT=$P/out/read_tabclose10k$PS\n", "OUT=$P/out/read_tabclose10k\n"),
}


def run_reader(path, tag):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", TB_ESTREAD_DIR=str(V / "estclose_read"), CLOSEREAD_DIR=str(V / "closeread"))
    t0 = time.time()
    r = subprocess.run([sys.executable, str(path), "--selftest"], env=env, capture_output=True, text=True, encoding="utf-8", timeout=900, cwd=str(Path(path).parent))
    return {"rc": r.returncode, "seconds": round(time.time() - t0, 1), "tail": (r.stdout + r.stderr).strip().splitlines()[-2:]}


def run_pytest(env_over, tag):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", TB_ESTREAD_DIR=str(V / "estclose_read"), **env_over)
    base = PT / ("mut_%s_%s" % (tag, time.strftime("%H%M%S")))
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "test_tabclose.py"), "-q", "-x", "-p", "no:cacheprovider", "--basetemp", str(base)], env=env, capture_output=True, text=True, encoding="utf-8", timeout=1800)
    return {"rc": r.returncode, "seconds": round(time.time() - t0, 1), "tail": (r.stdout + r.stderr).strip().splitlines()[-2:]}


def check(fn, path, tag):
    if fn == RD:
        return run_reader(path, tag)
    key = {QU: "TB_QUEUE", PR: "TB_PODREAD", CH: "TB_CHAIN"}[fn]
    return run_pytest({key: str(path)}, tag)


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / "MUTATIONS_TB.json"))
    a = ap.parse_args(argv)
    stamp = time.strftime("%H%M%S")
    names = [n for n in M if not a.only or n.split("_")[0] in a.only.split(",") or n in a.only.split(",")]
    res = {"mutations": {}, "controls_before": {}, "controls_after": {}}
    files = sorted({M[n][0] for n in names})
    for fn in files:
        res["controls_before"][fn] = check(fn, HERE / fn, "ctl_before")
        print("control before", fn, res["controls_before"][fn]["rc"], flush=True)
    for n in names:
        fn, what, old, new = M[n]
        base = (HERE / fn).read_text(encoding="utf-8")
        if base.count(old) != 1:
            raise SystemExit("mutation %s: the fragment occurs %d times, expected exactly 1" % (n, base.count(old)))
        if old == new:
            raise SystemExit("mutation %s: no change" % n)
        d = MUT / (n + "_" + stamp)
        d.mkdir(parents=True, exist_ok=True)
        f = d / fn
        f.write_text(base.replace(old, new), encoding="utf-8", newline="\n")
        r = check(fn, f, n)
        r["what"], r["old"], r["new"], r["file"] = what, old, new, fn
        r["red"] = r["rc"] != 0
        res["mutations"][n] = r
        print("%-36s %s" % (n, "RED" if r["red"] else "GREEN (not caught)"), flush=True)
    for fn in files:
        res["controls_after"][fn] = check(fn, HERE / fn, "ctl_after")
        print("control after", fn, res["controls_after"][fn]["rc"], flush=True)
    ok = all(v["rc"] == 0 for v in res["controls_before"].values()) and all(v["rc"] == 0 for v in res["controls_after"].values()) and all(v["red"] for v in res["mutations"].values())
    res["red"], res["total"] = sum(1 for v in res["mutations"].values() if v["red"]), len(res["mutations"])
    res["all_red_and_controls_green"] = ok
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print("ZZ MUTATIONS-TB", "PASS" if ok else "FAIL", "%d of %d red" % (res["red"], res["total"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
