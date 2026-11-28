# append_r203.py -- paper v11: append the fragment of row 203 to frag_row127.tex and extend the fork's status sentence; adds the spec bullet of row 203 (asserted). Run once.
import io
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def stop(msg):
    print("ZZ APPEND-R203 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex").replace("\r\n", "\n")
if t.count("H-V8F-PROOF-202") != 1 or "H-V8F-PROOF-203" in t or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 202")
x = rd(D + "frag_row203.tex").replace("\r\n", "\n")
if x.count("H-V8F-PROOF-203") != 1:
    stop("fragment 203 does not carry its row once")
t2 = t + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex").replace("\r\n", "\n")
head_old = "The fork after row 202:"
tail_old = r"the flat margin stays, and the gap line is read out as far as existing files go."
if s.count(head_old) != 1 or s.count(tail_old) != 1:
    stop("status sentence head or tail not found once")
tail_new = tail_old + (r" Row 203: the unsafe rate that remains with label inputs (0.0369 of the candidate's 0.0480) lies in 31 distinct windows of slow, close following in which the rollout ends "
                       r"more than 0.2 s closer than the logged driver (0.9518 [0.8084, 1.0000] of 83 windows over four seeds); the rule's response is the lever, and its logs are read next.")
s2 = s.replace(head_old, "The fork after row 203:").replace(tail_old, tail_new)

sp = rd(SPF)
nl = "\r\n" if "\r\n" in sp else "\n"
anchor = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out:"
if sp.count(anchor) != 1 or sp.count("* Row 202 (card ADD134") != 1 or "Row 203 (card ADD135" in sp:
    stop("spec is not in the state after row 202")
add = ("* Row 203 (card ADD135, tier T1-kin/hb0.2, banked runs of four model seeds at readout K with no pre-roll; five of five controls passed): `LABELUNSAFE10K-K0-RULE-FOLLOWS-CLOSER-THAN-LOGGED`, "
       "`LABELUNSAFE10K-K0-SAME-WINDOWS-ACROSS-SEEDS`. The label form has 83 unsafe windows over four seeds, 31 distinct; in 0.9518 [0.8084, 1.0000] the lowest time gap of the rollout lies more than 0.2 s below "
       "the logged driver's, in 0.1084 [0.0000, 0.3865] the driver's own lowest time gap is below 1.2 s; 0.6129 [0.3750, 0.7778] of the distinct windows are unsafe at three or four seeds. "
       "Stated limit of the candidate: 0.0369 of its unsafe rate of 0.0480 is the rule's response with exact inputs, in slow, close following; a change of the rule is a design decision for the PI and the other session." + nl)
sp2 = sp.replace(anchor, add + anchor)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline="").write(sp2)
print("ZZ APPEND-R203 DONE frag_row127 %d -> %d, status %d -> %d chars, spec %d -> %d" % (len(t), len(t2), len(s), len(s2), len(sp), len(sp2)))
