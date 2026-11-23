# append_r199.py -- paper v11: append the fragment of row 199 to frag_row127.tex and extend the fork's status sentence; adds the spec bullet of row 199 (asserted). Run once.
import io
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def stop(msg):
    print("ZZ APPEND-R199 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex").replace("\r\n", "\n")
if t.count("H-V8F-PROOF-198") != 1 or "H-V8F-PROOF-199" in t or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 198")
x = rd(D + "frag_row199.tex").replace("\r\n", "\n")
if x.count("H-V8F-PROOF-199") != 1:
    stop("fragment 199 does not carry its row once")
t2 = t + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex").replace("\r\n", "\n")
head_old = "The fork after row 198:"
tail_old = r"and its use in rollouts is a decision of the PI."
if s.count(head_old) != 1 or s.count(tail_old) != 1:
    stop("status sentence head or tail not found once")
tail_new = tail_old + r" Row 199: the gain holds under the locator's two further fits (ratios 0.9007 and 0.9068, each interval below 1), so it is quoted as about ten per cent."
s2 = s.replace(head_old, "The fork after row 199:").replace(tail_old, tail_new)

sp = rd(SPF)
nl = "\r\n" if "\r\n" in sp else "\n"
anchor = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out:"
if sp.count(anchor) != 1 or sp.count("* Row 198 (card ADD129") != 1 or "Row 199 (card ADD130" in sp:
    stop("spec is not in the state after row 198")
add = ("* Row 199 (card ADD130, tier T0; six of six controls passed for each of two further fits of the locator): `BOXFUSE-T0-LOCFIT-REPLICATED`, `BOXFUSE-T0-LOCFIT-BAR-NOT-3-OF-3`. "
       "R(FG) against arm D is 0.8961 [0.8449, 0.9407], 0.9007 [0.8509, 0.9456] and 0.9068 [0.8545, 0.9537] with the picks of `LOC_s0`, `LOC_s1` and `LOC_s2`: below 1 for each fit, at most 0.90 for the first alone. "
       "The gain of the joined read is quoted as about ten per cent at T0. The gap input of the candidate is unchanged." + nl)
sp2 = sp.replace(anchor, add + anchor)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline="").write(sp2)
print("ZZ APPEND-R199 DONE frag_row127 %d -> %d, status %d -> %d chars, spec %d -> %d" % (len(t), len(t2), len(s), len(s2), len(sp), len(sp2)))
