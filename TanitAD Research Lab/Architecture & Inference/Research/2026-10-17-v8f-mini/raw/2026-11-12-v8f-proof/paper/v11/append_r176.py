# append_r176.py -- paper v11: append the fragment of row 176 to frag_row127.tex, replace the fork's status sentence, add one spec bullet. The cost card is not changed (no decision arises).
# Every step is asserted. Run once.
import io
import re
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def has_crlf(p):
    return "\r\n" in io.open(p, encoding="utf-8", newline="").read()


def stop(msg):
    print("ZZ APPEND-R176 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex")
if t.count("H-V8F-PROOF-175") != 1 or "H-V8F-PROOF-176" in t or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 175")
x6 = rd(D + "frag_row176.tex")
if x6.count("H-V8F-PROOF-176") != 1:
    stop("the fragment does not carry its row once")
t2 = t + "\n" + x6.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex")
pat = re.compile(r"Open on this fork after row 175.*?runs on the fork pod\.")
hits = pat.findall(s)
if len(hits) != 1:
    stop("status sentence found %d times" % len(hits))
old = hits[0]
head_old = "Open on this fork after row 175:"
tail_old = "Card ADD110, rollouts with gap files composed from the label gap and the camera gap and no encode, runs on the fork pod."
if old.count(head_old) != 1 or old.count(tail_old) != 1 or not old.endswith(tail_old):
    stop("status sentence head or tail not found once")
tail_new = ("Row 176 reads card ADD110, rollouts with gap files composed from the label gap and the camera gap and no encode: with the label gap on every label row and the reader's own "
            "presence the price in falling behind is gone ($-0.0277$ [$-0.0539$, $-0.0060$] against the camera-fed form), so it sits in the gap value, and no single range of the gap "
            "carries it. Card ADD111, the split of the camera gap error into a slow and a fast part and two smoothing forms with no label, is next on the fork pod.")
new = old.replace(head_old, "Open on this fork after row 176:").replace(tail_old, tail_new)
s2 = s.replace(old, new)
old_nums = set(re.findall(r"ADD\d+|\d+\.\d+|\d{2,}", old))
lost = sorted(n for n in old_nums if n not in new and n != "175")
if lost:
    stop("status sentence lost " + ", ".join(lost))

sp = rd(SPF)
a = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out"
if sp.count(a) != 1 or "Row 176" in sp or sp.count("* Row 175 (") != 1:
    stop("spec is not in the state after row 175")
b176 = ("* Row 176 (card ADD110, tier T1-kin/hb0.2, pre-roll of 30 steps (6 s); six of six controls passed, control a4 with two readings): `FALLATTR10K-GAP-CARRIES-THE-PRICE`, "
        "`FALLATTR10K-NO-RANGE-SEPARATED`. An attribution with label values, no candidate: with the label gap on every label row and the reader's own presence the lead rule falls "
        "behind on 0.4877 of the windows with a lead against 0.5154 for the camera-fed form (-0.0277 [-0.0539, -0.0060]) and 0.4908 for the label form; contact 0 of 650. The price "
        "of the gap input of this spec in falling behind sits in the gap value, not in the reader's presence. The gap input of this spec is unchanged. Next: card ADD111 (slow and "
        "fast part of the camera gap error; two smoothing forms of the camera gap with no label, as study arms; a smoothed gap as an input of the rule would be a design decision "
        "for the PI and the other session).\n")
sp2 = sp.replace(a, b176 + a)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline=("\r\n" if has_crlf(SPF) else "\n")).write(sp2)
print("ZZ APPEND-R176 DONE frag_row127 %d -> %d, status %d -> %d, spec %d -> %d chars" % (len(t), len(t2), len(old), len(new), len(sp), len(sp2)))
