# append_r200.py -- paper v11: append the fragment of row 200 to frag_row127.tex and extend the fork's status sentence; adds the spec bullet of row 200 (asserted). Run once.
import io
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def stop(msg):
    print("ZZ APPEND-R200 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex").replace("\r\n", "\n")
if t.count("H-V8F-PROOF-199") != 1 or "H-V8F-PROOF-200" in t or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 199")
x = rd(D + "frag_row200.tex").replace("\r\n", "\n")
if x.count("H-V8F-PROOF-200") != 1:
    stop("fragment 200 does not carry its row once")
t2 = t + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex").replace("\r\n", "\n")
head_old = "The fork after row 199:"
tail_old = r"so it is quoted as about ten per cent."
if s.count(head_old) != 1 or s.count(tail_old) != 1:
    stop("status sentence head or tail not found once")
tail_new = tail_old + (r" Row 200: in the candidate's lead rule at \tier{T1-kin/hb0.2} the joined gap moves no measure on two models (unsafe $-0.0018$ [$-0.0060$, $+0.0026$] at readout K with no pre-roll), "
                       r"and against the candidate's own gap file its gain at \tier{T0} is 3.3 per cent and not separated; no detector is proposed for the perception path.")
s2 = s.replace(head_old, "The fork after row 200:").replace(tail_old, tail_new)

sp = rd(SPF)
nl = "\r\n" if "\r\n" in sp else "\n"
anchor = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out:"
if sp.count(anchor) != 1 or sp.count("* Row 199 (card ADD130") != 1 or "Row 200 (card ADD131" in sp:
    stop("spec is not in the state after row 199")
add = ("* Row 200 (card ADD131, tiers T0 and T1-kin/hb0.2, models UE_s0 and UE_s1; all controls passed): `BOXFUSE10K-T0-JOINED-NOT-SEPARATED` and, for each of the three pairs, `JOINED-GAP-NOT-SEPARATED`. "
       "Joined minus candidate at readout K with no pre-roll: unsafe -0.0018 [-0.0060, +0.0026], contact +0.0008 [0.0000, +0.0030], falling behind -0.0055 [-0.0133, +0.0015]; "
       "at the convention: unsafe -0.0080 [-0.0209, +0.0041], falling behind -0.0031 [-0.0151, +0.0083]; with the margin of 0.90 against A90: unsafe 0.0000 [-0.0095, +0.0084], falling behind -0.0062 [-0.0135, +0.0008]. "
       "T0 against the candidate's own gap file: 3.028 m against 3.132 m, ratio 0.9669 [0.9177, 1.0091]. "
       "No detector is proposed for the perception path on this evidence. The gap input of the candidate is unchanged." + nl)
sp2 = sp.replace(anchor, add + anchor)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline="").write(sp2)
print("ZZ APPEND-R200 DONE frag_row127 %d -> %d, status %d -> %d chars, spec %d -> %d" % (len(t), len(t2), len(s), len(s2), len(sp), len(sp2)))
