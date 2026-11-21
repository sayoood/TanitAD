"""make_read_accwin10k.py -- writes the read tools of card PREREG_V8F_PROOF_ADD55 (ACCWIN10K) from those of card ADD54 by asserted text edits:
  read_handover10k.py -> read_accwin10k.py ; podread_handover.sh -> podread_accwin.sh ; wait_podread_handover.sh -> wait_podread_accwin.sh
The originals are not changed. Usage: python make_read_accwin10k.py (in this folder)"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ME = "make_read_accwin10k.py"


def load(name):
    raw = open(os.path.join(HERE, name), "rb").read()
    assert bytes([13]) not in raw, "%s has CR line ends" % name
    return raw.decode("utf-8")


def save(name, text):
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        assert ME in io.open(p, encoding="utf-8", errors="replace").read(3000), "%s exists and was not written by this script" % name
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s (%d bytes)" % (name, len(text.encode("utf-8"))))


def rep(text, old, new, n=1):
    c = text.count(old)
    assert c == n, "edit anchor %r occurs %d times, expected %d" % (old[:80], c, n)
    return text.replace(old, new)


def span(text, start, end, new):
    assert text.count(start) == 1 and text.count(end) == 1, (start[:50], text.count(start), end[:50], text.count(end))
    i, j = text.index(start), text.index(end) + len(end)
    assert i < j
    return text[:i] + new + text[j:]


r = load("read_handover10k.py")
r = rep(r, '"""read_handover10k.py -- card PREREG_V8F_PROOF_ADD54 (HANDOVER10K): the read (numpy only). Controls h1 to h9 first;',
        '"""read_accwin10k.py -- written by make_read_accwin10k.py from read_handover10k.py. Card PREREG_V8F_PROOF_ADD55 (ACCWIN10K): the read (numpy only). The readouts of this\n'
        'card: WINDOW (causal_w: the acceleration at every executed step is the executed speed change over the last 1.0 s) and ZERO (causal_z: no acceleration read); both keep\n'
        'the causal speed. Runs R0W5, R0Z5 (5 steps), R0W10 (10 steps), control R0ZID (causal readout, must equal R0C5 of card ADD54). Controls z1 to z9 first. Text of the parent:\n'
        'read_handover10k.py -- card PREREG_V8F_PROOF_ADD54 (HANDOVER10K): the read (numpy only). Controls h1 to h9 first;')
r = rep(r, 'WRAPPER_SHA_PREFIX = "4a96e1c941d8275c"', 'WRAPPER_SHA_PREFIX = "fa7207d72311b5cf"')
r = rep(r, 'READOUTS = (("CAUSAL", "I10R0C5"), ("BACKWARD", "I10R0B5"))', 'READOUTS = (("WINDOW", "I10R0W5"), ("ZERO", "I10R0Z5"))')
r = rep(r, 'RUNS = {"I10R0HID": ("pinned", 10), "I10R0C5": ("causal", 5), "I10R0B5": ("backward", 5), "I10R0C10": ("causal", 10)}',
        'RUNS = {"I10R0ZID": ("causal", 5), "I10R0W5": ("causal_w", 5), "I10R0Z5": ("causal_z", 5), "I10R0W10": ("causal_w", 10)}')
r = rep(r, 'NEW_RUNS = [("I10R0HID", 0)] + [(arm, k) for arm in ("I10R0C5", "I10R0B5", "I10R0C10") for k in SEEDS]',
        'NEW_RUNS = [("I10R0ZID", 0)] + [(arm, k) for arm in ("I10R0W5", "I10R0Z5", "I10R0W10") for k in SEEDS]')
r = rep(r, 'BANKED_OF = {"I10R0HID": "I10R0", "I10R0C5": "I10R0M5", "I10R0B5": "I10R0M5", "I10R0C10": "I10R0"}',
        'BANKED_OF = {"I10R0ZID": "I10R0C5", "I10R0W5": "I10R0M5", "I10R0Z5": "I10R0M5", "I10R0W10": "I10R0"}')
r = rep(r, 'os.path.join(a.pull, "small_handover", "CMP_R0HID.txt")', 'os.path.join(a.pull, "small_accwin", "CMP_R0ZID.txt")')
r = rep(r, '    for arm in ("I10R0", "I10R0M5"):\n        for k in SEEDS:\n            D[(arm, k)] = dump(a.pull, arm, k)\n',
        '    for arm in ("I10R0", "I10R0M5", "I10R0C5"):\n        for k in SEEDS:\n            D[(arm, k)] = dump(a.pull, arm, k)\n')
r = span(r, '        elif mode == "backward":\n', '"mean_acceleration_diff": L["mean_abs_acceleration_minus_executed"]}\n',
         '        elif mode == "causal_w":\n'
         '            dw = float(L["max_abs_acceleration_minus_window"])\n'
         '            h3[key] = {"max_speed_diff": dv, "max_acceleration_minus_window": dw, "mean_abs_acceleration_read": L["mean_abs_acceleration_read"], "PASS": bool(dv <= 1e-6 and dw <= 1e-6)}\n'
         '        else:\n'
         '            az = float(L["max_abs_acceleration_read"])\n'
         '            h3[key] = {"max_speed_diff": dv, "max_abs_acceleration_read": az, "PASS": bool(dv <= 1e-6 and az <= 1e-6)}\n')
for i, nm in ((1, "rerun_equals_banked"), (2, "parameters_applied"), (3, "readout_holds"), (4, "first_block_equal"), (5, "counts"), (6, "reused_and_known_values"), (7, "reader_controls"), (8, "model"), (9, "tool_check")):
    assert r.count('"h%d_%s"' % (i, nm)) >= 1, (i, nm)
    r = r.replace('"h%d_%s"' % (i, nm), '"z%d_%s"' % (i, nm))
r = rep(r, 'print("== controls (h1 to h9)")', 'print("== controls (z1 to z9)")')
r = rep(r, '"ZZ CHECK-TICK-READOUT formula holds: True" in t_txt', '"ZZ CHECK-TICK-READOUT-RZ all hold: True" in t_txt')
r = rep(r, '"card": "PREREG_V8F_PROOF_ADD54"', '"card": "PREREG_V8F_PROOF_ADD55"')
r = rep(r, '"HANDOVER10K-MODEL-', '"ACCWIN10K-MODEL-', 2)
r = rep(r, '"HANDOVER10K-PRIOR-', '"ACCWIN10K-PRIOR-', 3)
r = rep(r, '"HANDOVER10K-CAUSAL-%s-AT-2S"', '"ACCWIN10K-WINDOW-%s-AT-2S"')
r = rep(r, 'pr = {"headway": ucf(hw, "I10R0C10", "I10R0"), "families": fams(q3, "I10R0C10", "I10R0")}', 'pr = {"headway": ucf(hw, "I10R0W10", "I10R0"), "families": fams(q3, "I10R0W10", "I10R0")}')
r = rep(r, 'd10 = D[("I10R0C10", 0)]', 'd10 = D[("I10R0W10", 0)]')
r = rep(r, 'prior["CAUSAL_10_steps"]', 'prior["WINDOW_10_steps"]')
r = rep(r, '("I10R0", "I10R0M5", "I10R0C5", "I10R0B5", "I10R0C10")', '("I10R0", "I10R0M5", "I10R0C5", "I10R0W5", "I10R0Z5", "I10R0W10")', 2)
r = rep(r, '        cb = {"headway": ucf(hw, "I10R0C5", "I10R0B5"), "families": fams(q3, "I10R0C5", "I10R0B5")}\n',
        '        cb = {"headway": ucf(hw, "I10R0W5", "I10R0Z5"), "families": fams(q3, "I10R0W5", "I10R0Z5")}\n'
        '        vc = {rk: {"headway": ucf(hw, arm, "I10R0C5"), "families": fams(q3, arm, "I10R0C5")} for rk, arm in READOUTS}\n')
r = rep(r, 'model_ADE6_m=model_ade, causal_minus_backward=cb)', 'model_ADE6_m=model_ade, window_minus_zero=cb, against_causal_readout=vc)')
r = rep(r, '        print("== beside: causal minus backward at 1.0 s")\n        print("   %s" % line(cb["headway"]))\n',
        '        print("== beside: WINDOW minus ZERO at 1.0 s")\n        print("   %s" % line(cb["headway"]))\n'
        '        print("== beside: each readout against the causal readout of card ADD54 at 1.0 s")\n'
        '        for rk, v in vc.items():\n'
        '            d = v["families"][SPEED]\n'
        '            print("   %-8s speed error %+.4f [%+.4f, %+.4f] %s | %s" % (rk, d["D"], d["interval"][0], d["interval"][1], d["per_seed"], line(v["headway"])))\n')
r = rep(r, '[("PR", pr["families"]), ("causal_minus_backward", cb["families"])]', '[("PR", pr["families"]), ("window_minus_zero", cb["families"])] + [("vs_causal_" + rk, v["families"]) for rk, v in vc.items()]')
r = rep(r, 'print("== PR: the causal readout at the programme\'s cycle (R0C10 minus R0)")', 'print("== PR: the WINDOW readout at the programme\'s cycle (R0W10 minus R0)")')
assert "HANDOVER10K-" not in r and "I10R0B5" not in r and "I10R0HID" not in r
save("read_accwin10k.py", r)

p = load("podread_handover.sh")
for old, new, n in (("PODREAD-HANDOVER", "PODREAD-ACCWIN", None), ("HANDOVER10K_DONE", "ACCWIN10K_DONE", 1), ("e1h", "e1z", None), ("read_handover", "read_accwin", None), ("small_handover", "small_accwin", None),
                    ("CMP_R0HID.txt", "CMP_R0ZID.txt", 1), ("READ_HANDOVER10K", "READ_ACCWIN10K", None), ("v8f_t1_eval_reader_rule_ro.py", "v8f_t1_eval_reader_rule_rz.py", 2),
                    ("CHECK_TICK_READOUT.txt", "CHECK_TICK_READOUT_RZ.txt", 1), ("card ADD54", "card ADD55", None)):
    c = p.count(old)
    assert c >= 1 and (n is None or c == n), (old, c)
    p = p.replace(old, new)
lines = p.split("\n")
k = [i for i, ln in enumerate(lines) if ln.startswith("PAIRS=")]
assert len(k) == 1
lines[k[0]] = 'PAIRS="I10R0W5:I10R0M5,I10R0Z5:I10R0M5,I10R0W5:I10R0,I10R0Z5:I10R0,I10R0W10:I10R0,I10R0W5:I10R0Z5,I10R0W5:I10R0C5,I10R0Z5:I10R0C5"'
p = "\n".join(lines)
p = rep(p, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# podread_accwin.sh -- written by make_read_accwin10k.py from the pod read script of card ADD54: pod-side read of card PREREG_V8F_PROOF_ADD55 (ACCWIN10K).\n")
assert "HANDOVER10K_REUSED.json" in p and "handover_test" in p          # the reused-file record and the test folder are those of card ADD54
save("podread_accwin.sh", p)

w = load("wait_podread_handover.sh")
for old, new in (("wait_podread_handover", "wait_podread_accwin"), ("HANDOVER10K_DONE", "ACCWIN10K_DONE"), ("HANDOVER10K_STOP", "ACCWIN10K_STOP"), ("WAIT-PODREAD-HANDOVER", "WAIT-PODREAD-ACCWIN"),
                 ("podread_handover", "podread_accwin"), ("read_handover", "read_accwin"), ("READ_HANDOVER10K", "READ_ACCWIN10K"), ("e1h", "e1z"), ("card ADD54", "card ADD55")):
    assert w.count(old) >= 1, old
    w = w.replace(old, new)
w = rep(w, "#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# wait_podread_accwin.sh -- written by make_read_accwin10k.py from the waiter of card ADD54: dev-box waiter of card ADD55.\n")
assert "HANDOVER" not in "\n".join(ln for ln in w.split("\n") if not ln.startswith("#")) and "handover" not in "\n".join(ln for ln in w.split("\n") if not ln.startswith("#"))
save("wait_podread_accwin.sh", w)
print("ZZ MAKE-READ-ACCWIN10K DONE")
