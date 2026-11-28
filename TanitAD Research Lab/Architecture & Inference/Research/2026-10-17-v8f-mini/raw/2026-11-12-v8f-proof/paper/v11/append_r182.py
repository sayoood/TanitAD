# append_r182.py -- paper v11: append the fragments of rows 180, 181 and 182 to frag_row127.tex, replace the fork's status sentence, add three spec bullets.
# The cost card is not changed (no decision arises). Every step is asserted. Run once.
import io
import re
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"
NEW = (180, 181, 182)


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def has_crlf(p):
    return "\r\n" in io.open(p, encoding="utf-8", newline="").read()


def stop(msg):
    print("ZZ APPEND-R182 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex")
if t.count("H-V8F-PROOF-179") != 1 or any("H-V8F-PROOF-%d" % n in t for n in NEW) or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 179")
t2 = t
for n in NEW:
    x = rd(D + "frag_row%d.tex" % n)
    if x.count("H-V8F-PROOF-%d" % n) != 1:
        stop("fragment %d does not carry its row once" % n)
    t2 = t2 + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex")
pat = re.compile(r"Open on this fork after row 179.*?runs on the fork pod\.")
hits = pat.findall(s)
if len(hits) != 1:
    stop("status sentence found %d times" % len(hits))
old = hits[0]
head_old = "Open on this fork after row 179:"
tail_old = "Card ADD114, whether that unsafe rate sits in the presence input or in the gap value, runs on the fork pod."
if old.count(head_old) != 1 or old.count(tail_old) != 1 or not old.endswith(tail_old):
    stop("status sentence head or tail not found once")
tail_new = (r"Rows 180 to 182 read where that unsafe rate sits. Card ADD116 at \tier{T0}: the reader's presence does not mark 0.1558 of the frames with a lead of the label and marks "
            r"0.2244 of the frames with none, the misses sit at gaps from 20 m, and a threshold has no room. Card ADD114 under readout K with no pre-roll: the unsafe rate comes with "
            r"the camera gap value ($+0.0133$ at the reader's presence) and not with the presence (0.0000 and $+0.0009$). Card ADD115 at the convention: both inputs carry it, the gap "
            r"value $+0.0142$ and $+0.0169$, the presence $+0.0080$ and $+0.0107$. Card ADD117, a margin on the camera gap, runs on the fork pod; card ADD118, a presence head on the "
            r"existing narrow-window tokens against the bar of row 180, is armed behind it.")
new = old.replace(head_old, "Open on this fork after row 182:").replace(tail_old, tail_new)
s2 = s.replace(old, new)
old_nums = set(re.findall(r"ADD\d+|\d+\.\d+|\d{2,}", old))
lost = sorted(n for n in old_nums if n not in new and n not in ("179", "ADD114"))
if lost:
    stop("status sentence lost " + ", ".join(lost))
if "ADD114" not in new:
    stop("status sentence lost ADD114")

sp = rd(SPF)
a = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out"
if sp.count(a) != 1 or any("* Row %d (" % n in sp for n in NEW) or sp.count("* Row 179 (") != 1:
    stop("spec is not in the state after row 179")
b180 = ("* Row 180 (card ADD116, tier T0, recorded evaluation frames, not driving performance; six of six controls passed): `PRESERR-T0-THRESHOLD-HAS-NO-ROOM`, "
        "`PRESERR-T0-MISSES-AT-LARGE-GAPS`. The presence input of the lead rule of this spec (the token reader's p at 0.5 or more) does not mark 0.1558 [0.0959, 0.2162] of the "
        "frames with a lead of the label within 60 m and marks 0.2244 [0.1762, 0.2811] of the frames with none (10,865 frames per model seed, 141 clips). The misses sit at gaps "
        "from 20 m (0.2278 against 0.0656 below 20 m). A threshold of 0.1 leaves 0.5370 of the misses at 1.5983 times the extra marks, so the threshold of this spec is not "
        "changed. Bar committed for a presence head before the rows below: at most half of 0.1558 misses at no more than twice 0.2244 extra marks on these frames.\n")
b181 = ("* Row 181 (card ADD114, tier T1-kin/hb0.2; eight of eight controls passed): `PRESATTR10K-K0-GAP-CARRIES-UNSAFE`, `PRESATTR10K-K30-UNSAFE-NOT-ATTRIBUTED`. Presence and "
        "gap each from the label or from the camera, all four cells. Readout K, no pre-roll (contact / unsafe / falling behind by more than 10 m): label form 0.0000 / 0.0365 / "
        "0.1550; reader's presence with label gap 0.0000 / 0.0365 / 0.1550; label presence with camera gap 0.0000 / 0.0489 / 0.1667; the candidate 0.0000 / 0.0498 / 0.1651. The "
        "unsafe rate of the candidate above the label form comes with the camera gap value (+0.0125 [+0.0026, +0.0243] and +0.0133 [+0.0036, +0.0250]), not with the presence "
        "(0.0000 [0.0000, 0.0000] and +0.0009 [0.0000, +0.0037]). At a pre-roll of 30 steps (6 s) no contrast is separated on the unsafe rate. Cells with a label input are "
        "attribution arms and never candidates.\n")
b182 = ("* Row 182 (card ADD115, tier T1-kin/hb0.2; eight of eight controls passed): `PRESATTRC10K-CONV-BOTH-CARRY-UNSAFE`, `PRESATTRC10K-ATTRIBUTION-NOT-AS-AT-K0`. The same four "
        "cells at the convention: label form 0.0016 / 0.0320 / 0.2243; reader's presence with label gap 0.0016 / 0.0427 / 0.2165; label presence with camera gap 0.0008 / 0.0489 / "
        "0.2399; the candidate 0.0016 / 0.0569 / 0.2329. The camera gap value raises the unsafe rate (+0.0169 [+0.0034, +0.0333] and +0.0142 [+0.0007, +0.0292]) and so does the "
        "reader's presence (+0.0080 [+0.0009, +0.0171] at the camera gap; +0.0107 [0.0000, +0.0232] at the label gap). The candidate against the label form: unsafe +0.0249 "
        "[+0.0072, +0.0464]. Open levers, each a study arm whose adoption is a design decision for the PI and the other session: a margin on the camera gap (card ADD117) and a "
        "presence head on the existing narrow-window tokens (card ADD118). The form of the rule is unchanged.\n")
sp2 = sp.replace(a, b180 + b181 + b182 + a)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline=("\r\n" if has_crlf(SPF) else "\n")).write(sp2)
print("ZZ APPEND-R182 DONE frag_row127 %d -> %d, status %d -> %d, spec %d -> %d chars" % (len(t), len(t2), len(old), len(new), len(sp), len(sp2)))
