# append_r184.py -- paper v11: append the fragments of rows 183 and 184 to frag_row127.tex, replace the fork's status sentence, add two spec bullets.
# The cost card is not changed (no decision with a cost arises). Every step is asserted. Run once.
import io
import re
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"
NEW = (183, 184)


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def has_crlf(p):
    return "\r\n" in io.open(p, encoding="utf-8", newline="").read()


def stop(msg):
    print("ZZ APPEND-R184 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex")
if t.count("H-V8F-PROOF-182") != 1 or any("H-V8F-PROOF-%d" % n in t for n in NEW) or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 182")
t2 = t
for n in NEW:
    x = rd(D + "frag_row%d.tex" % n)
    if x.count("H-V8F-PROOF-%d" % n) != 1:
        stop("fragment %d does not carry its row once" % n)
    t2 = t2 + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex")
pat = re.compile(r"Open on this fork after row 182.*?is armed behind it\.")
hits = pat.findall(s)
if len(hits) != 1:
    stop("status sentence found %d times" % len(hits))
old = hits[0]
head_old = "Open on this fork after row 182:"
tail_old = ("Card ADD117, a margin on the camera gap, runs on the fork pod; card ADD118, a presence head on the existing narrow-window tokens against the bar of row 180, "
            "is armed behind it.")
if old.count(head_old) != 1 or old.count(tail_old) != 1 or not old.endswith(tail_old):
    stop("status sentence head or tail not found once")
tail_new = (r"Row 183, card ADD117: a margin of 10 per cent on the camera gap closes the unsafe rate under readout K (0.0374 against 0.0365 of the label form) at $+0.0078$ in "
            r"falling behind; at the convention no margin closes it. Row 184, card ADD118 at \tier{T0}: a presence head on the existing narrow-window tokens, joined with the "
            r"reader by OR, misses 0.0338 of the lead frames against 0.1558 at 0.3217 extra marks against 0.2244, inside the bar of row 180. Card ADD119, the margin at the "
            r"pre-rolls of 30 and of 60 steps, and card ADD120, the candidate with the joined presence input alone and with the margin in the rollout, run on the fork pod.")
new = old.replace(head_old, "Open on this fork after row 184:").replace(tail_old, tail_new)
s2 = s.replace(old, new)
old_nums = set(re.findall(r"ADD\d+|\d+\.\d+|\d{2,}", old))
lost = sorted(n for n in old_nums if n not in new and n not in ("182",))
if lost:
    stop("status sentence lost " + ", ".join(lost))

sp = rd(SPF)
a = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out"
if sp.count(a) != 1 or any("* Row %d (" % n in sp for n in NEW) or sp.count("* Row 182 (") != 1:
    stop("spec is not in the state after row 182")
b183 = ("* Row 183 (card ADD117, tier T1-kin/hb0.2; eight of eight controls passed): `GAPMARGIN10K-K0-MARGIN-CLOSES-UNSAFE-AT-A-PRICE` (closing arm A90), "
        "`GAPMARGIN10K-CONV-MARGIN-DOES-NOT-CLOSE-UNSAFE`. Study-arm option, not adopted: the camera gap scaled by 0.90 before the rule (arm A90; A80 scales by 0.80). Price "
        "table (contact / unsafe / falling behind by more than 10 m). Readout K, no pre-roll: the candidate 0.0000 / 0.0498 / 0.1651; A90 0.0000 / 0.0374 / 0.1729; A80 0.0000 / "
        "0.0302 / 0.1916; label form 0.0000 / 0.0365 / 0.1550. A90 against the candidate: unsafe -0.0125 [-0.0203, -0.0057], falling behind +0.0078 [+0.0016, +0.0153]; A80: "
        "-0.0196 [-0.0330, -0.0080] and +0.0265 [+0.0132, +0.0432], with the longitudinal and tactical reads worse. Convention: the candidate 0.0016 / 0.0569 / 0.2329; A90 "
        "0.0008 / 0.0463 / 0.2539; A80 0.0008 / 0.0383 / 0.2788; label form 0.0016 / 0.0320 / 0.2243. A90 against the candidate: unsafe -0.0107 [-0.0284, +0.0040], falling "
        "behind +0.0210 [+0.0086, +0.0348]; A80: -0.0187 [-0.0315, -0.0078] and +0.0460 [+0.0234, +0.0690], with the longitudinal, tactical and strategic reads worse. Taking the "
        "margin of 10 per cent for readout K, or carrying the unsafe rate as a stated limit, is a design decision for the PI and the other session; the aged settings are read by "
        "card ADD119.\n")
b184 = ("* Row 184 (card ADD118, tier T0, recorded evaluation frames, not driving performance; seven of seven controls passed): `PRESHEAD-T0-BAR-MET`, form HEAD-OR-READER. "
        "Study-arm option, not adopted: the presence of the rule read as the larger of the token reader's p and the p of a presence head on the existing narrow-window tokens "
        "(dumps of card ADD118, head seed k for model seed k), tested at 0.5 as before. On the frames of row 180: misses 0.0338 [0.0145, 0.0553] against the reader's 0.1558, "
        "extra marks 0.3217 [0.2638, 0.3875] against 0.2244; the bar was 0.0779 and 0.4488. The head alone (misses 0.1007, extra marks 0.1814) is not separated from the reader. "
        "What the extra marks cost in the rollout is read by card ADD120 through a study wrapper generated from the unchanged wrapper. The form of the rule is unchanged.\n")
sp2 = sp.replace(a, b183 + b184 + a)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline=("\r\n" if has_crlf(SPF) else "\n")).write(sp2)
print("ZZ APPEND-R184 DONE frag_row127 %d -> %d, status %d -> %d, spec %d -> %d chars" % (len(t), len(t2), len(old), len(new), len(sp), len(sp2)))
