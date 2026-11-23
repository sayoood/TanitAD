# append_r202.py -- paper v11: append the fragment of row 202 to frag_row127.tex and extend the fork's status sentence; adds the spec bullet of row 202 (asserted). Run once.
import io
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def stop(msg):
    print("ZZ APPEND-R202 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex").replace("\r\n", "\n")
if t.count("H-V8F-PROOF-201") != 1 or "H-V8F-PROOF-202" in t or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 201")
x = rd(D + "frag_row202.tex").replace("\r\n", "\n")
if x.count("H-V8F-PROOF-202") != 1:
    stop("fragment 202 does not carry its row once")
t2 = t + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex").replace("\r\n", "\n")
head_old = "The fork after row 201:"
tail_old = r"is not separated from the flat margin on the long part (1.0472 [0.8725, 1.3655])."
if s.count(head_old) != 1 or s.count(tail_old) != 1:
    stop("status sentence head or tail not found once")
tail_new = tail_old + (r" Row 202: in the lead rule at \tier{T1-kin/hb0.2} on four model seeds the placed margin reads as the flat one (unsafe 0.0000 [$-0.0053$, $+0.0053$], "
                       r"falling behind 0.0000 [$-0.0066$, $+0.0064$]); the flat margin stays, and the gap line is read out as far as existing files go.")
s2 = s.replace(head_old, "The fork after row 202:").replace(tail_old, tail_new)

sp = rd(SPF)
nl = "\r\n" if "\r\n" in sp else "\n"
anchor = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out:"
if sp.count(anchor) != 1 or sp.count("* Row 201 (cards ADD132") != 1 or "Row 202 (card ADD134" in sp:
    stop("spec is not in the state after row 201")
add = ("* Row 202 (card ADD134, tier T1-kin/hb0.2, four model seeds, readout K with no pre-roll; eight of eight controls passed): `GAPSPREAD10K-K0-PLACED-NOT-SEPARATED-FROM-FLAT`. "
       "The margin placed by the spread of the three gap fits minus the flat margin A90: unsafe 0.0000 [-0.0053, +0.0053], contact +0.0004 [0.0000, +0.0017], falling behind 0.0000 [-0.0066, +0.0064]; "
       "levels (contact / unsafe / falling behind) SP90 0.0004 / 0.0365 / 0.1651, A90 0.0000 / 0.0365 / 0.1651. "
       "The flat margin stays the form under readout K; the gap input and the margin of the candidate are unchanged." + nl)
sp2 = sp.replace(anchor, add + anchor)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline="").write(sp2)
print("ZZ APPEND-R202 DONE frag_row127 %d -> %d, status %d -> %d chars, spec %d -> %d" % (len(t), len(t2), len(s), len(s2), len(sp), len(sp2)))
