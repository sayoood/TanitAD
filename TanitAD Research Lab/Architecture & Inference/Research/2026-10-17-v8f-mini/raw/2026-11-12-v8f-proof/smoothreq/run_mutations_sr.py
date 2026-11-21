"""run_mutations_sr.py -- mutation runner of the kit of card ADD85. Every mutant (one counted text replacement in a copy of a file) has to turn its test red; a mutant that stays green is reported as SURVIVED.
Targets: the table tool and the reader (their own --selftest), the queue and the pod scripts (test_smoothreq.py, the shell tests). Writes MUTATIONS_SR.json. Run from F:/TanitAD-scratch/v8proof:
  PYCAP_MB=3000 bash ops/pycap.sh smoothreq/run_mutations_sr.py [--only tool|reader|shell]"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
CAP = os.path.join(V, "ops", "capmem.py")
TMP = os.path.join(HERE, "tmp_mut_sr")
TOOL, READER = "make_smooth_tables.py", "read_smoothreq10k.py"
QUEUE, CHAIN, PODREAD = "pod_smoothreq10k_queue.sh", "chain_read_smoothreq10k.sh", "podread_smoothreq10k.sh"
M = []   # (target file, label, old, new)


def m(f, label, old, new):
    M.append((f, label, old, new))


# ---- table tool
for lab_, old, new in (("target r of S50", '"S50": 0.50, "S90"', '"S50": 0.55, "S90"'), ("target r of S90", '"S90": 0.90, "S7B"', '"S90": 0.85, "S7B"'), ("smooth arms list loses S90 and S7B", 'SMOOTH_ARMS = ("S50", "S90", "S7B")', 'SMOOTH_ARMS = ("S50",)'),
                       ("shrink term r^2 (L - m) becomes r (L - m)", "m + r * r * (L - m) + s * np.sqrt", "m + r * (L - m) + s * np.sqrt"), ("noise scale", "s * np.sqrt(r * r * (1.0 - r * r)) * e\n    return out", "s * np.sqrt(r * r) * e\n    return out"),
                       ("seed index S50/S90 swapped", '{"S50": 0, "S90": 1,', '{"S50": 1, "S90": 0,'), ("seed base", "RNG_BASE = 85000", "RNG_BASE = 84000"),
                       ("Z00 scale z = mean instead of sd", "z = float(fin[np.isfinite(fin)].std())", "z = float(np.abs(fin[np.isfinite(fin)]).mean())"), ("Z00 unit sd lost", "e = e / e.std()", "e = e / (2.0 * e.std())"),
                       ("look mean ignores est_ok", '& np.asarray(g["est_ok"], bool)]', ']'), ("d_k without c_k", "d_k = m_stk - c_k", "d_k = m_stk"), ("STKM sign", 'arms["STKM"] = stk - d_k', 'arms["STKM"] = stk + d_k'),
                       ("TKD not centred", 'arms["TKD"] = tk - c_k', 'arms["TKD"] = tk'), ("b0 r tolerance", "B0_R_TOL, B0_SD_BAND, B0_MEAN_TOL = 0.02,", "B0_R_TOL, B0_SD_BAND, B0_MEAN_TOL = 0.5,"),
                       ("b0 second draw band", "SECOND_DRAW_BAND = (-0.10, 0.10)", "SECOND_DRAW_BAND = (-1.0, 1.0)"), ("b0 white autocorrelation band", "B0_AC_WHITE = (-0.05, 0.05)", "B0_AC_WHITE = (-1.0, 1.0)"),
                       ("b0 smooth autocorrelation band", "B0_AC_SMOOTH, B0_AC_WHITE = (0.85, 0.95),", "B0_AC_SMOOTH, B0_AC_WHITE = (-1.0, 1.0),"), ("b0 sd band", "B0_SD_BAND, B0_MEAN_TOL = 0.02, (0.95, 1.05)", "B0_SD_BAND, B0_MEAN_TOL = 0.02, (0.5, 2.0)"),
                       ("b1 mean tolerance", "B1_MEAN_TOL, B1_SD_BAND, B1_AC, B1_R = 0.01,", "B1_MEAN_TOL, B1_SD_BAND, B1_AC, B1_R = 5.0,"), ("b1 label correlation bound", "(0.85, 0.95), 0.15\n", "(0.85, 0.95), 5.0\n"),
                       ("b1 autocorrelation band", "(0.95, 1.05), (0.85, 0.95), 0.15", "(0.95, 1.05), (-1.0, 1.0), 0.15"), ("b1 sd band", "B1_SD_BAND, B1_AC, B1_R = 0.01, (0.95, 1.05)", "B1_SD_BAND, B1_AC, B1_R = 0.01, (0.1, 10.0)"),
                       ("b2 constants tolerance", "B2_CONST_TOL = 1e-12", "B2_CONST_TOL = 1e-3"), ("b2 r of STKM against the label dropped", '"PASS": bool(ok and ok4), "runs": out}\n\n\n# ====', '"PASS": bool(ok), "runs": out}\n\n\n# ===='),
                       ("refusal on a1 of the record", 'for c in ("a0", "a1"):', 'for c in ("a0",):'), ("refusal on a table hash", '!= sha256_file(p):', '== sha256_file(p):'),
                       ("b0 not part of the failed list", '(("b0", b0), ("b1", b1), ("b2", b2))', '(("b1", b1), ("b2", b2))'), ("valued frames of L100 not compared in b0", "np.array_equal(np.isfinite(v), V))\n            if not same", "True)\n            if not same")):
    m(TOOL, lab_, old, new)
# ---- reader
for lab_, old, new in (("b4 sd lower bound", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80,", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.30,"), ("b4 sd upper bound", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 1.25,", "SD_LO, SD_HI, MEAN_MAX, SHARE_MIN = 0.80, 3.0,"),
                       ("b4 mean bound", "0.80, 1.25, 0.15, 0.90", "0.80, 1.25, 0.60, 0.90"), ("b4 share bound", "0.80, 1.25, 0.15, 0.90", "0.80, 1.25, 0.15, 0.50"), ("b4 mean against zero instead of the TK mean", "abs(mean_s - mean_t) <= MEAN_MAX", "abs(mean_s) <= MEAN_MAX"),
                       ("half of the gap", "HALF = 0.5", "HALF = 0.6"), ("mean of the two shares becomes the larger", "sum(shares) / len(shares) >= HALF", "max(shares) >= HALF"),
                       ("level 70 needs only the banked draw", '(70, ("S70", "S7B"))', '(70, ("S70",))'), ("level 70 needs only the new draw", '(70, ("S70", "S7B"))', '(70, ("S7B",))'),
                       ("a share of None not any more blocks", "if any(s is None for s in shares):\n        return False", "if False:\n        return False"), ("time course word", 'return None if w is None or w["C"] is None else bool(w["C"] == "HIGHER")', 'return None if w is None or w["C"] is None else bool(w["C"] == "LOWER")'),
                       ("c_lower word", 'bool(w["C"] == "LOWER")\n\n\ndef c_higher', 'bool(w["C"] == "HIGHER")\n\n\ndef c_higher'), ("(c) reads another pair", 'words.get("Z00-VS-RR")', 'words.get("S50-VS-RR")'),
                       ("(e) reads C HIGHER", 'decide([c_lower(words.get("TKD-VS-TOK"))])', 'decide([c_higher(words.get("TKD-VS-TOK"))])'), ("(d) ignores U", "decide([c_lower(sh), c_nh(sh, \"U\")])", "decide([c_lower(sh)])"),
                       ("D pairs lose TKD-VS-SHUFM", 'D_PAIRS = ("TOK-VS-SHUFM", "SHUFM-VS-RR", "TKD-VS-SHUFM")', 'D_PAIRS = ("TOK-VS-SHUFM", "SHUFM-VS-RR")'), ("b2 not any more withholds TKD", 'if failed["b2"] and "TKD" in used:', 'if failed["b2"] and "ZZZ" in used:'),
                       ("b0 arms lose W7B", 'B0_ARMS = ("S50", "S90", "S7B", "W7B")', 'B0_ARMS = ("S50", "S90", "S7B")'), ("b4 not any more withholds (d)", '(failed["b2"] or failed["b4"]) and pair in D_PAIRS', 'failed["b2"] and pair in D_PAIRS'),
                       ("b1 not any more withholds Z00", 'failed["b1"] and "Z00" in used', 'failed["b1"] and "ZZZ" in used'), ("b3 failed runs not used", "if used & set(failed[\"b3\"].get(st) or []):", "if False:"),
                       ("k4 admits any failed control", 'others = [k for k in notpass if k != "a4_u3_AGE"]', "others = []"), ("k4 card check", 'tbj.get("card") != TB_CARD', 'tbj.get("card") == TB_CARD'),
                       ("k4 tolerance", "TOL_K4 = 1e-9", "TOL_K4 = 1e-2"), ("withheld rows need only one withheld pair", "if fp and all(n in wset for n in fp):", "if fp and any(n in wset for n in fp):"),
                       ("withheld pairs stay in pairs", 'clean[key] = {n: v for n, v in (part.get(key) or {}).items() if n not in wset}', 'clean[key] = dict(part.get(key) or {})'),
                       ("b4 not classified as a card control", 'elif k.startswith("b4_"):', 'elif k.startswith("b5_"):'), ("share formula", "(c_rr - c_x) / gap", "(c_x - c_rr) / gap"),
                       ("pair STKM named SHUFM", 'LABEL = {"STKM": "SHUFM", "TK": "TOK"}', 'LABEL = {"STKM": "STKM", "TK": "TOK"}'), ("beside name of the gain verdict", 'names["VERDICT_TK_GAIN_BESIDE_SHUFM_VS_RR"] = cname("SHUFM-VS-RR", "AGE", "C", sr["C"])', 'names["VERDICT_TK_GAIN_BESIDE_SHUFM_VS_RR"] = cname("SHUFM-VS-RR", "AGE", "U", sr["C"])')):
    m(READER, lab_, old, new)
# ---- queue and pod scripts
for lab_, f, old, new in (("seed switch ignored", QUEUE, 'ARMS=("$SEEDSW")', "ARMS=(0 1)"), ("seed marker suffix lost", QUEUE, 'PS="_s$SEEDSW"', 'PS=""'), ("seed value 2 accepted", QUEUE, "case \"$SEEDSW\" in 0|1|ALL)", "case \"$SEEDSW\" in 0|1|2|ALL)"),
                          ("pre-roll lost", QUEUE, "PREROLL=30", "PREROLL=0"), ("admission: top-level pass", QUEUE, 'if rec.get("pass") is not True:', "if False:"), ("admission: b2", QUEUE, 'for c in ("b0", "b1", "b2"):', 'for c in ("b0", "b1"):'),
                          ("admission: hash", QUEUE, "if h != tabs[fn]:", "if False:"), ("admission: wrapper hash", QUEUE, '[ "$got" = "02a58e655e5c1979" ]', '[ "$got" = "x" ]'), ("wrong table folder", QUEUE, "closeread/est_smooth}", "closeread/est_tab}"),
                          ("chain: second seed marker", CHAIN, ' && [ -f "$W/SMOOTHREQ10K_DONE_s1.txt" ]', ""), ("chain: stop glob", CHAIN, 'SMOOTHREQ10K_STOP*.txt', "SMOOTHREQ10K_STOP.txt"), ("chain: go file", CHAIN, '&& [ -f "$W/GO_READ_SMOOTHREQ10K.txt" ] && break', "&& break"),
                          ("chain: link list", CHAIN, "STKM,TKD,Z00,S90,S50,S7B,W7B,TK", "STKM,TKD,Z00,S90,S50,S7B,TK"), ("podread: second seed marker", PODREAD, '&& [ -f "$W/SMOOTHREQ10K_DONE_s1.txt" ]', ""),
                          ("podread: tabclose json arg", PODREAD, '--tabclose-json "$TBJ" --tab-dir "$TABDIR"', '--tab-dir "$TABDIR"')):
    m(f, lab_, old, new)


def run_one(i, f, label, old, new):
    src = os.path.join(HERE, f)
    txt = open(src, encoding="utf-8", newline="").read()
    if txt.count(old) != 1:
        return {"id": i, "file": f, "label": label, "result": "BAD-MUTANT", "count": txt.count(old)}
    d = os.path.join(TMP, "m%03d" % i)
    os.makedirs(d, exist_ok=True)
    mp = os.path.join(d, f)
    open(mp, "w", encoding="utf-8", newline="").write(txt.replace(old, new))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SMOOTH_TOOL_DIR=os.path.join(V, "tabclose"), SR_ESTREAD_DIR=os.path.join(V, "estclose_read"))
    if f == TOOL:
        cmd = [PY, CAP, "3000", mp, "--selftest"]
    elif f == READER:
        cmd = [PY, CAP, "3000", mp, "--selftest"]
    else:
        key = {QUEUE: "SR_QUEUE", CHAIN: "SR_CHAIN", PODREAD: "SR_PODREAD"}[f]
        env[key] = mp
        cmd = [PY, CAP, "3000", "-m", "pytest", os.path.join(HERE, "test_smoothreq.py"), "-q", "-p", "no:cacheprovider", "-x", "--basetemp", os.path.join(d, "pt"), "-k", "queue or admission or chain or podread or pod_scripts"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=1500)
    red = r.returncode != 0
    return {"id": i, "file": f, "label": label, "result": "RED" if red else "SURVIVED", "rc": r.returncode, "tail": (r.stdout.strip().splitlines() or [""])[-1][:160]}


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "all"
    sel = {"tool": TOOL, "reader": READER}
    out = []
    for i, (f, label, old, new) in enumerate(M):
        if only == "shell" and f in (TOOL, READER) or only in sel and f != sel[only]:
            continue
        r = run_one(i, f, label, old, new)
        out.append(r)
        print("%s %3d %-28s %s" % (r["result"], i, f[:28], label), flush=True)
    res = {"card": "PREREG_V8F_PROOF_ADD85", "total": len(out), "red": sum(1 for r in out if r["result"] == "RED"), "survived": [r for r in out if r["result"] == "SURVIVED"], "bad_mutants": [r for r in out if r["result"] == "BAD-MUTANT"],
           "sources": {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()[:16] for f in (TOOL, READER, QUEUE, CHAIN, PODREAD)}, "mutants": out}
    name = "MUTATIONS_SR.json" if only == "all" else "tmp_MUTATIONS_SR_%s.json" % only
    json.dump(res, open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZ MUTATIONS-SR red %d of %d survived %d bad %d" % (res["red"], res["total"], len(res["survived"]), len(res["bad_mutants"])))


if __name__ == "__main__":
    main()
