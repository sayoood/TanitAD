# append_r201.py -- paper v11: append the fragment of row 201 to frag_row127.tex and extend the fork's status sentence; adds the spec bullet of row 201 (asserted). Run once.
import io
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def stop(msg):
    print("ZZ APPEND-R201 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex").replace("\r\n", "\n")
if t.count("H-V8F-PROOF-200") != 1 or "H-V8F-PROOF-201" in t or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 200")
x = rd(D + "frag_row201.tex").replace("\r\n", "\n")
if x.count("H-V8F-PROOF-201") != 1:
    stop("fragment 201 does not carry its row once")
t2 = t + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex").replace("\r\n", "\n")
head_old = "The fork after row 200:"
tail_old = r"no detector is proposed for the perception path."
if s.count(head_old) != 1 or s.count(tail_old) != 1:
    stop("status sentence head or tail not found once")
tail_new = tail_old + (r" Row 201: at \tier{T0} the spread of the three gap fits marks the long reads (0.7841 [0.6837, 0.8834] of the long part of a random placement), "
                       r"and a 10 per cent margin placed by it is not separated from the flat margin on the long part (1.0472 [0.8725, 1.3655]).")
s2 = s.replace(head_old, "The fork after row 201:").replace(tail_old, tail_new)

sp = rd(SPF)
nl = "\r\n" if "\r\n" in sp else "\n"
anchor = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out:"
if sp.count(anchor) != 1 or sp.count("* Row 200 (card ADD131") != 1 or "Row 201 (cards ADD132" in sp:
    stop("spec is not in the state after row 200")
add = ("* Row 201 (cards ADD132 and ADD133, tier T0; card ADD132 failed its control s3 and gives no name, card ADD133 passed six of six controls): `GAPSPREADB-T0-SPREAD-MARKS-LONG-READS`, `GAPSPREADB-T0-PLACED-NOT-SEPARATED`. "
       "A 10 per cent margin placed by the spread of the three gap fits against the same factors placed at random: long part 0.7841 [0.6837, 0.8834], short part 0.8822 [0.8231, 0.9408]; "
       "against the flat margin: long part 1.0472 [0.8725, 1.3655], short part 0.8010 [0.7400, 0.8613]. "
       "By the card's committed reading no rollout is spent on the spread. The gap input and the margin of the candidate are unchanged." + nl)
sp2 = sp.replace(anchor, add + anchor)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline="").write(sp2)
print("ZZ APPEND-R201 DONE frag_row127 %d -> %d, status %d -> %d chars, spec %d -> %d" % (len(t), len(t2), len(s), len(s2), len(sp), len(sp2)))
