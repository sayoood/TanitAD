# append_r198.py -- paper v11: append the fragments of rows 196 to 198 to frag_row127.tex and extend the fork's status sentence. The spec bullets are in the spec already (asserted). Run once.
import io
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"
NEW = (196, 197, 198)


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def stop(msg):
    print("ZZ APPEND-R198 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex")
if t.count("H-V8F-PROOF-195") != 1 or any("H-V8F-PROOF-%d" % n in t for n in NEW) or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 195")
t2 = t
for n in NEW:
    x = rd(D + "frag_row%d.tex" % n)
    if x.count("H-V8F-PROOF-%d" % n) != 1:
        stop("fragment %d does not carry its row once" % n)
    t2 = t2 + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex")
head_old = "The fork after row 195:"
tail_old = "a read of banked states with no new run."
if s.count(head_old) != 1 or s.count(tail_old) != 1:
    stop("status sentence head or tail not found once")
tail_new = (tail_old + r" Rows 196 to 198: a pretrained detector's box read of the lead, alone, is worse than the token head (ratio 1.21) because one pick in eleven lands on another vehicle; "
            r"joined with the token head under an agreement gate it is better (ratio 0.8961 [0.8449, 0.9407], \tier{T0}), and its use in rollouts is a decision of the PI.")
s2 = s.replace(head_old, "The fork after row 198:").replace(tail_old, tail_new)

sp = rd(SPF)
for n, c in ((196, "ADD127"), (197, "ADD128"), (198, "ADD129")):
    if sp.count("* Row %d (card %s" % (n, c)) != 1:
        stop("spec does not carry the bullet of row %d once" % n)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
print("ZZ APPEND-R198 DONE frag_row127 %d -> %d, status %d -> %d chars" % (len(t), len(t2), len(s), len(s2)))
