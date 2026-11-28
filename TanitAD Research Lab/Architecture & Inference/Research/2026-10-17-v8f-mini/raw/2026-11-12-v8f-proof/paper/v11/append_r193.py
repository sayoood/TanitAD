# append_r193.py -- paper v11: append the fragment of row 193 to frag_row127.tex and extend the fork's status sentence. The spec bullet of row 193 is in the spec already (asserted). Run once.
import io
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"
NEW = (193,)


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def stop(msg):
    print("ZZ APPEND-R193 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex")
if t.count("H-V8F-PROOF-192") != 1 or any("H-V8F-PROOF-%d" % n in t for n in NEW) or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 192")
t2 = t
for n in NEW:
    x = rd(D + "frag_row%d.tex" % n)
    if x.count("H-V8F-PROOF-%d" % n) != 1:
        stop("fragment %d does not carry its row once" % n)
    t2 = t2 + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex")
head_old = "The fork after row 192:"
tail_old = "the run at equal passes (card ADD125) is frozen and is a decision of the PI."
if s.count(head_old) != 1 or s.count(tail_old) != 1:
    stop("status sentence head or tail not found once")
tail_new = (tail_old + r" The PI gave that go on 2026-11-23 and the run is on the pod of the other session. Row 193: the replicate of card ADD121 is read; the gap margin lowers the unsafe rate "
            r"under readout K on a second pair of training runs at both settings, five of seven claims are replicated on the new pair alone, and the price table of the candidate is restated on four seeds.")
s2 = s.replace(head_old, "The fork after row 193:").replace(tail_old, tail_new)

sp = rd(SPF)
if sp.count("* Row 193 (card ADD121") != 1:
    stop("spec does not carry the bullet of row 193 once")

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
print("ZZ APPEND-R193 DONE frag_row127 %d -> %d, status %d -> %d chars" % (len(t), len(t2), len(s), len(s2)))
