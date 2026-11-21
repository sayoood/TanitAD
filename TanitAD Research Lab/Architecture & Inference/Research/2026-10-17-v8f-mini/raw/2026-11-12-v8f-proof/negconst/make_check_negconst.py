"""make_check_negconst.py -- writes check_negconst.py (pod check of card ADD82, control k1) from constclose/check_constclose.py
(sha256 prefix 25185711c68ad55b) by counted, asserted text edits. Refuses to overwrite its output (NC_OVERWRITE_OWN=1 rewrites it).
Usage: bash ops/pycap.sh make_check_negconst.py"""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "constclose", "check_constclose.py"))
OUT = os.path.join(HERE, "check_negconst.py")
raw = io.open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest()[:16] == "25185711c68ad55b", "parent changed"
if os.path.exists(OUT) and os.environ.get("NC_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists" % OUT)
T = raw.decode("utf-8")
n_edits = 0


def once(a, b, n=1):
    global T, n_edits
    assert T.count(a) == n, (a[:70], T.count(a))
    T = T.replace(a, b)
    n_edits += 1


# --- the docstring (everything up to the closing quotes of the first line block)
end = T.index('<names>"""') + len('<names>"""')
DOC = '''"""check_negconst.py -- pod check of card ADD82 (NEGCONST10K), control k1: the constant reaches the rule WITH ITS SIGN. One small rollout of form RRe (wrapper v8f_t1_eval_reader_rule_rzgfhqe.py,
hash a747abeb51ef041f) with the constant file const_M20_H1.npz (-2.0 m/s, history one frame) on the first --n-clips clips of the pinned grid (all their windows), fresh (no pre-roll), model seed 0,
readout K (--readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2), --rule-every 5, q50 map, reader seed 0. Derived from the check of card ADD80 (sha256 prefix 25185711c68ad55b) by make_check_negconst.py.
Checks on the two logs of the run (replan log rule_log_<name>.npz and mid-look log rule_log_mid_<name>.npz; both hold ci, t0, j, k, present, c_src, c_est, est_ok, v_l, v_L):
    K1a  the closing column c_src is -2.0 on every row that is present (lead read present) and est_ok, in both logs; and on the present rows without est_ok it is 0.0
    K1b  c_est is -2.0 on every est_ok row of both logs (replan and second look) and NaN on the others
    K1c  no look at clip frame index j >= 4 is marked short (est_ok False): a full history needs the frame j-4..j; looks at j < 4 are counted
    K1d  v_L = max(0, v_l + c_src) on the present rows
    K1e  THE SIGN CHECK of the card: on the present rows with est_ok, the imagined lead speed v_L equals max(0, v_l - 2.0) and is below v_l wherever v_l is above zero
    L1   the run line "est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>": the counts equal those recomputed from the two logs (preroll 0 0) and the sha256 equals that of the file
    F0   the constant file loaded by closest.py returns -2.0 for random inputs of shape [1, 3] and [n, 1, 3] and needs 5 frames (control k0 on the file the rollout used)
    W0   the wrapper has the hash of the card
  Nothing here removes, moves or stops anything; the rollout writes only inside --workdir (refused if it exists, except with --analyze-only, which only reads the run folder already there).
Usage on the pod:
  python check_negconst.py --new WRAPPER.py --tree TREE --ckpt CKPT --tracks native_tracks.pt --eval-cache DIR --eval-join J.npz --grid PINNED_GRID.json --reader READER.pt --qmap MAP.json
         --workdir NEWDIR --distrule-dir DIR --closest-dir DIR --const-file const_M20_H1.npz [--python PY] [--device cuda] [--n-clips 3] [--dry] [--analyze-only]
  python check_negconst.py --selftest
Last line: ZZ CHECK-NEGCONST PASS <counts>  or  ZZ CHECK-NEGCONST FAIL <names>"""'''
T = DOC + T[end:]
n_edits += 1

once("VALUE = 2.0\n", "VALUE = -2.0\n")
once('RUNNAME = "k1_c20"', 'RUNNAME = "k1_m20"')
once("ZZ CHECK-CONSTCLOSE", "ZZ CHECK-NEGCONST", 6)
once("ZZ SELFTEST-CHECK-CONSTCLOSE", "ZZ SELFTEST-CHECK-NEGCONST")
once('g2["constclose_subset"]', 'g2["negconst_subset"]')
once('"K1a closing column is 2.0 at every present look with est_ok"', '"K1a closing column is -2.0 at every present look with est_ok"')
once('"K1b c_est is 2.0 on every est_ok row"', '"K1b c_est is -2.0 on every est_ok row"')
once('("K1d v_L = max(0, v_l + closing)", check_d)):', '("K1d v_L = max(0, v_l + closing)", check_d),\n                   ("K1e SIGN: v_L = max(0, v_l - 2.0) and below v_l wherever v_l > 0 (looks with a full history)", check_e)):')
once('"""-> list of (name, ok, detail) for K1a, K1b, K1c, K1d (both logs each) and L1"""', '"""-> list of (name, ok, detail) for K1a, K1b, K1c, K1d, K1e (both logs each) and L1"""')
once('"L1 run line counts equal the counts recomputed from the logs, file sha256 equals the constant file"', '"L1 run line counts equal the counts recomputed from the logs, file sha256 equals the constant file"')
once("def check_line(", '''def check_e(z, value=VALUE):
    """K1e, the sign check of the card (ADD82 section 4, k1): on the present rows with est_ok the imagined lead speed is max(0, v_l + value) with value = -2.0 (taken from the card, not from the c_src column)
    and is below v_l wherever v_l is above zero. A rule that read the constant with the wrong sign would put v_L above v_l."""
    sel = _b(z, "present") & _b(z, "est_ok")
    v_l, v_L = np.asarray(z["v_l"], np.float64)[sel], np.asarray(z["v_L"], np.float64)[sel]
    pos = v_l > 0.0
    eq = np.allclose(v_L, np.maximum(0.0, v_l + value))
    below = bool(np.all(v_L[pos] < v_l[pos]))
    return bool(sel.any() and pos.any() and eq and below), {"rows": int(sel.sum()), "rows_v_l_positive": int(pos.sum()), "v_L_not_max0_vl_minus_2": int((~np.isclose(v_L, np.maximum(0.0, v_l + value))).sum()), "v_L_not_below_v_l": int((~(v_L[pos] < v_l[pos])).sum())}


def check_line(''')
# --- selftest plants
once('''    r1["c_src"][i] = 1.9
    r1["v_L"][i] = max(0.0, r1["v_l"][i] + 1.9)''', '''    r1["c_src"][i] = -1.9
    r1["v_L"][i] = max(0.0, r1["v_l"][i] - 1.9)''')
once('chk("planted 1.9 in the closing column of the replan log: K1a fails (and only the closing check %s)" % bad1, not ok1 and bad1 == ["K1a (replan log)"])',
     'chk("planted -1.9 in the closing column of the replan log (v_L consistent with it): K1a and the sign check K1e fail, nothing else %s" % bad1, not ok1 and bad1 == ["K1a (replan log)", "K1e (replan log)"])')
once('r4["c_est"][i] = 2.0001', 'r4["c_est"][i] = -2.0001')
once('"planted c_est 2.0001 on one est_ok row: K1b fails %s"', '"planted c_est -2.0001 on one est_ok row: K1b fails %s"')
once('''    r6["c_src"][i] = 2.0
    r6["v_L"][i] = r6["v_l"][i] + 2.0''', '''    r6["c_src"][i] = -2.0
    r6["v_L"][i] = max(0.0, r6["v_l"][i] - 2.0)''')
once('"a closing of 2.0 on a present look without a full history: K1a fails"', '"a closing of -2.0 on a present look without a full history: K1a fails"')
once('chk("the constants of the card: closing 2.0 m/s, a full history from frame index 4 on", VALUE == 2.0 and FULL_FROM == 4)',
     '''chk("the constants of the card: closing -2.0 m/s (the cautious sign), a full history from frame index 4 on", VALUE == -2.0 and FULL_FROM == 4)
    # the planted failures of the card: the wrong sign, a v_L above v_l, a short look at frame 7 (above), a -1.9 (above)
    r10 = {k: v.copy() for k, v in RL.items()}
    i = int(np.nonzero(r10["present"] & r10["est_ok"])[0][4])
    r10["c_src"][i] = 2.0
    r10["c_est"][i] = 2.0
    r10["v_L"][i] = r10["v_l"][i] + 2.0
    ok10, bad10 = all_ok(r10, ML, line(r10, ML))
    chk("planted +2.0 (the wrong sign) in c_src and c_est with v_L = v_l + 2.0: K1a, K1b and the sign check K1e fail, K1d (the formula) does not %s" % bad10,
        not ok10 and bad10 == ["K1a (replan log)", "K1b (replan log)", "K1e (replan log)"])
    r11 = {k: v.copy() for k, v in ML.items()}
    i = int(np.nonzero(r11["present"] & r11["est_ok"])[0][5])
    r11["v_L"][i] = r11["v_l"][i] + 0.5
    ok11, bad11 = all_ok(RL, r11, line(RL, r11))
    chk("planted v_L above v_l on a full look of the mid-look log: K1d and the sign check K1e fail %s" % bad11, not ok11 and [x[:3] for x in bad11] == ["K1d", "K1e"])
    r12 = {k: v.copy() for k, v in RL.items()}
    i = int(np.nonzero(r12["present"] & r12["est_ok"])[0][6])
    r12["c_src"][i] = 0.0
    r12["v_L"][i] = r12["v_l"][i]
    ok12, bad12 = all_ok(r12, ML, line(r12, ML))
    chk("planted closing 0.0 (a rule that dropped the constant) on a full look: K1a and K1e fail %s" % bad12, not ok12 and bad12 == ["K1a (replan log)", "K1e (replan log)"])
    r13 = {k: v.copy() for k, v in RL.items()}
    r13["v_l"][:] = 0.0
    r13["v_L"][:] = 0.0
    chk("a log in which every lead speed is 0 has no row to test the sign on: K1e fails (no row is not a pass)", not check_e(r13)[0])''')
# --- the rest of the text must not carry a trace of the parent card
assert "CONSTCLOSE" not in T and "constclose" not in T and "C20" not in T and "const_C" not in T, "a name of the parent card is left"
io.open(OUT, "w", encoding="utf-8", newline="\n").write(T)
print("wrote %s sha256 %s edits %d" % (os.path.basename(OUT), hashlib.sha256(T.encode("utf-8")).hexdigest()[:16], n_edits))
