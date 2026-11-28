"""make_unsafe_cause_f5.py -- writes unsafe_cause_f5.py from unsafe_cause_b8.py by asserted text edits (card PREREG_V8F_PROOF_ADD62, RULEFAST10K; the split of the unsafe windows,
descriptive, names nothing). The original is not changed. Pure text generator. Usage: python make_unsafe_cause_f5.py (in this folder)"""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_unsafe_cause_f5.py"


def rep(text, old, new, n=1):
    c = text.count(old)
    assert c == n, "edit anchor %r occurs %d times, expected %d" % (old[:70], c, n)
    return text.replace(old, new)


def span(text, start, end, new):
    assert text.count(start) == 1 and text.count(end) == 1, (start[:50], text.count(start), end[:50], text.count(end))
    i, j = text.index(start), text.index(end) + len(end)
    assert i < j
    return text[:i] + new + text[j:]


raw = open(os.path.join(HERE, "unsafe_cause_b8.py"), "rb").read()
assert bytes([13]) not in raw
u = raw.decode("utf-8")
u = span(u, "# unsafe_cause_b8.py : the split of the unsafe windows repeated on the runs of card ADD57", "python unsafe_cause_b8.py\n",
         "# unsafe_cause_f5.py -- written by make_unsafe_cause_f5.py from unsafe_cause_b8.py. The split of the unsafe windows repeated on the runs of card ADD62 (RULEFAST10K: the lead rule applied again\n"
         "# inside the 2.0 s block), as that card commits it: descriptive, names nothing. Tier T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.\n"
         "# Convention of the unsafe-window split of the TIMES_NOTE.md line of 2026-11-19T16:28:23Z. Per arm and model seed (arms R0, RLc, RLcF5, RLcF2, RQ50, RQ50F5): first unsafe step k*, tick in force\n"
         "# q = (k* - 1) // 10, kind, depth, rule state at the tick in force; for the arms with later looks also the rule state at the later look in force (the last look before step k* of its block;\n"
         "# tick_rule = none yet). Per pair (banked rule at the tick only -> faster rule: RLc -> RLcF5, RLc -> RLcF2, RQ50 -> RQ50F5) and model seed, on the windows scored in both runs:\n"
         "#   both / removed / added   unsafe in both runs / only with the rule at the tick only / only with the faster rule; for the added windows the split in the faster-rule run, for the removed\n"
         "#   windows the split in the banked run, the smallest time gap of the window in both runs, the ego speed and gap at the first unsafe step (faster minus banked), and over all windows scored in\n"
         "#   both runs the mean ego speed and gap difference at the steps 5, 10, 15, 20, 25, 30.\n"
         "# Run from the headway folder of the pod read kit (numpy; roots from PS_ROOT / PS_OUT): python unsafe_cause_f5.py\n")
u = rep(u, "read_brake/UNSAFE_CAUSE_B8.json", "read_rulefast/UNSAFE_CAUSE_F5.json")
u = rep(u, '("I10R0", "I10RLc", "I10RLcB8", "I10RLcB10", "I10RQ50", "I10RQ50B8")', '("I10R0", "I10RLc", "I10RLcF5", "I10RLcF2", "I10RQ50", "I10RQ50F5")')
u = rep(u, 'PAIRS = (("I10RLc", "I10RLcB8"), ("I10RLc", "I10RLcB10"), ("I10RQ50", "I10RQ50B8"))',
        'PAIRS = (("I10RLc", "I10RLcF5"), ("I10RLc", "I10RLcF2"), ("I10RQ50", "I10RQ50F5"))\nEVERY = {"I10RLcF5": 5, "I10RLcF2": 2, "I10RQ50F5": 5}')
u = rep(u, "split of the unsafe windows on the runs of card ADD57 and the windows added or removed by the higher braking limit; descriptive, names nothing",
        "split of the unsafe windows on the runs of card ADD62 and the windows added or removed by the faster rule; descriptive, names nothing")
for old, new in (("added_windows_in_the_higher_limit_run", "added_windows_in_the_faster_rule_run"), ("removed_windows_in_the_6.0_run", "removed_windows_in_the_banked_run"),
                 ("higher_minus_6.0", "faster_minus_banked"), ("ZZ UNSAFE-CAUSE-B8 DONE", "ZZ UNSAFE-CAUSE-F5 DONE")):
    assert u.count(old) >= 1, old
    u = u.replace(old, new)
EXTRA = """def look_rows(arm, k):
    \"\"\"the later-look log of an arm: key (window, tick, q) -> row, with the rule state columns\"\"\"
    L = np.load(os.path.join(ROOT, "%s_s%d" % (arm, k), "ticks", "rule_log_mid_%s_s%d.npz" % (arm, k)))
    lkey = np.asarray(L["ci"], np.int64) * 1000000 + np.asarray(L["t0"], np.int64)
    row = {(int(x), int(t), int(q)): i for i, (x, t, q) in enumerate(zip(lkey, np.asarray(L["tick"], np.int64), np.asarray(L["q"], np.int64)))}
    return row, np.asarray(L["present"]).astype(bool), np.asarray(L["changed"]).astype(bool), np.asarray(L["sat_exec"], np.float64)


def look_state(lr, h, w, t, s):
    \"\"\"rule state at the later look in force for the step s (1 to 10) of the block t: the last look q < s (q a multiple of h); tick_rule when there is none\"\"\"
    q = ((int(s) - 1) // h) * h
    if q == 0:
        return "tick_rule"
    row, pres, chg, sx = lr
    i = row.get((int(w), int(t), int(q)))
    if i is None or not pres[i]:
        return "no_lead"
    if sx[i] > 0:
        return "sat"
    return "capped" if chg[i] else "unchanged"


def med(x):"""
u = rep(u, "def med(x):", EXTRA)
OLD = '            rec["rule_state_at_tick_in_force"] = counts([state_of(rr, w, t) for w, t in zip(wkey[idx], q)], STATES)\n'
NEW = (OLD + '        if EVERY.get(a, 10) < 10:\n            lr = look_rows(a, k)\n'
       '            rec["rule_state_at_look_in_force"] = counts([look_state(lr, EVERY[a], w, t, (ks - 1) % M_STEPS + 1) for w, t, ks in zip(wkey[idx], q, kstar)], STATES + ("tick_rule",))\n')
u = rep(u, OLD, NEW)
compile(u, "unsafe_cause_f5.py", "exec")
p = os.path.join(HERE, "unsafe_cause_f5.py")
if os.path.exists(p):
    assert ME in io.open(p, encoding="utf-8", errors="replace").read(3000), "unsafe_cause_f5.py exists and was not written by this script"
with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(u)
print("wrote unsafe_cause_f5.py (%d bytes) sha256 %s" % (len(u.encode("utf-8")), hashlib.sha256(u.encode("utf-8")).hexdigest()[:16]))
