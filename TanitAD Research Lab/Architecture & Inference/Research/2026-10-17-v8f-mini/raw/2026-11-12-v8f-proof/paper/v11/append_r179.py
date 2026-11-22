# append_r179.py -- paper v11: append the fragments of rows 178 and 179 to frag_row127.tex, replace the fork's status sentence, add two spec bullets (the second carries the table
# of decision D16-A for the lead rule). The cost card is not changed (no decision arises). Every step is asserted. Run once.
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
    print("ZZ APPEND-R179 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex")
if t.count("H-V8F-PROOF-177") != 1 or "H-V8F-PROOF-178" in t or "H-V8F-PROOF-179" in t or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 177")
x8, x9 = rd(D + "frag_row178.tex"), rd(D + "frag_row179.tex")
if x8.count("H-V8F-PROOF-178") != 1 or x9.count("H-V8F-PROOF-179") != 1:
    stop("a fragment does not carry its row once")
t2 = t + "\n" + x8.rstrip("\n") + "\n\n" + x9.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex")
pat = re.compile(r"Open on this fork after row 177.*?is next on CPU\.")
hits = pat.findall(s)
if len(hits) != 1:
    stop("status sentence found %d times" % len(hits))
old = hits[0]
head_old = "Open on this fork after row 177:"
tail_old = r"Card ADD112, a bound at \tier{T0} on how much of the slow part a monotone map of the gap value can remove, is next on CPU."
if old.count(head_old) != 1 or old.count(tail_old) != 1 or not old.endswith(tail_old):
    stop("status sentence head or tail not found once")
tail_new = (r"Row 178 reads card ADD112, a bound at \tier{T0} on how much of the slow part a monotone map of the gap value can remove: 0.035 [0.011, 0.056] in sample and 0.009 "
            r"cross-fitted, below the bar of 0.10, so the recalibration ends with no pod spend. Row 179 reads card ADD113: the lead rule with no label at inference cuts contact to the "
            r"level of the label form at every setting (0.0016 against 0.0280 under the convention, 0.0000 against 0.0296 under readout K with no pre-roll, 0.0137 against 0.1096 at a "
            r"pre-roll of 60 steps (12 s)), with its unsafe rate still above the label form. Card ADD114, whether that unsafe rate sits in the presence input or in the gap value, "
            r"runs on the fork pod.")
new = old.replace(head_old, "Open on this fork after row 179:").replace(tail_old, tail_new)
s2 = s.replace(old, new)
old_nums = set(re.findall(r"ADD\d+|\d+\.\d+|\d{2,}", old))
lost = sorted(n for n in old_nums if n not in new and n != "177")
if lost:
    stop("status sentence lost " + ", ".join(lost))

sp = rd(SPF)
a = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out"
if sp.count(a) != 1 or "Row 178" in sp or "Row 179" in sp or sp.count("* Row 177 (") != 1:
    stop("spec is not in the state after row 177")
b178 = ("* Row 178 (card ADD112, tier T0, recorded evaluation frames, not driving performance; six of six controls passed): `GAPMAP-T0-MAP-DOES-NOT-CUT-SLOW-PART`. A monotone map of "
        "the camera gap value cuts the slow part of its error by 0.035 [0.011, 0.056] in sample and by 0.009 [-0.019, +0.030] cross-fitted over clips, below the bar of 0.10. No "
        "recalibrated gap is proposed; no fold fits and no rollouts were run. The price in falling behind of the gap input of this spec stays a stated limit of that input; what "
        "remains for it is the reader's own input (a new encode, a PI decision).\n")
b179 = ("* Row 179 (card ADD113, tier T1-kin/hb0.2; seven of seven controls passed): `CANDSET10K-CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT`. The table of decision D16-A for the lead "
        "rule of this spec (candidate / reader-fed form / label form; contact, unsafe, falling behind by more than 10 m; windows with a lead per model seed). Convention, no pre-roll, "
        "pinned readout, 642 windows: contact 0.0016 / 0.0280 / 0.0016, unsafe 0.0569 / 0.0783 / 0.0320, falling behind 0.2329 / 0.2593 / 0.2243. Readout K, no pre-roll, 642 "
        "windows: contact 0.0000 / 0.0296 / 0.0000, unsafe 0.0498 / 0.0738 / 0.0365, falling behind 0.1651 / 0.1682 / 0.1550. Readout K, pre-roll of 30 steps (6 s), 325 windows: "
        "contact 0.0062 / 0.0954 / 0.0000, unsafe 0.0839 / 0.1339 / 0.0714, falling behind 0.5154 / 0.4954 / 0.4908. Readout K, pre-roll of 60 steps (12 s), 73 windows: contact "
        "0.0137 / 0.1096 / 0.0137, unsafe 0.0738 / 0.1230 / 0.0410, falling behind 0.6438 / 0.6575 / 0.6301. The contact cut against the reader-fed form is separated at every "
        "setting (-0.0265 [-0.0517, -0.0089], -0.0296 [-0.0525, -0.0105], -0.0892 [-0.1488, -0.0353], -0.0959 [-0.1649, -0.0373]). Open in this table: the unsafe rate of the "
        "candidate above the label form (card ADD114), and two model seeds (a further model seed of the 10k line would be the replicate; a PI decision). The form of the rule is "
        "unchanged.\n")
sp2 = sp.replace(a, b178 + b179 + a)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline=("\r\n" if has_crlf(SPF) else "\n")).write(sp2)
print("ZZ APPEND-R179 DONE frag_row127 %d -> %d, status %d -> %d, spec %d -> %d chars" % (len(t), len(t2), len(old), len(new), len(sp), len(sp2)))
