# append_r191.py -- paper v11: append the fragments of rows 189 and 191 to frag_row127.tex and extend the fork's status sentence.
# Rows 187, 188 and 190 are rows of cards frozen before their runs; they get one clause each in the status sentence. The spec bullets of rows 189 and 191 are in the spec already (asserted). Run once.
import io
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"
NEW = (189, 191)


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def stop(msg):
    print("ZZ APPEND-R191 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex")
if t.count("H-V8F-PROOF-186") != 1 or any("H-V8F-PROOF-%d" % n in t for n in NEW) or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 186")
t2 = t
for n in NEW:
    x = rd(D + "frag_row%d.tex" % n)
    if x.count("H-V8F-PROOF-%d" % n) != 1:
        stop("fragment %d does not carry its row once" % n)
    t2 = t2 + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex")
head_old = "The fork after row 186:"
tail_old = "the gap value (a new encode) and the number of model seeds, both decisions of the PI."
if s.count(head_old) != 1 or s.count(tail_old) != 1:
    stop("status sentence head or tail not found once")
tail_new = (tail_old + r" The PI decided both on 2026-11-22. Two more model seeds of the 10k line are in training and the candidate set is run again on them under card ADD121 (row 187, "
            r"frozen before the runs). The new encode was the front tele camera (rows 188 and 189, card ADD122): at fixed windows it makes the gap head worse (ratio 1.3579 with four "
            r"windows), and on label tracks the size of the lead's box alone fixes the gap to 1.611~m. Row 191, card ADD124: a head that reads that size from the pooled crop tokens "
            r"is worse than the metre head (ratio 1.2348), because it misses the size by 14 to 20 per cent; a box read at pixel level is the lever that follows and a decision of the "
            r"PI. Row 190, card ADD123: the union probe of the data-scale lead line with twice the steps at four seeds is frozen and runs on the pod of the other session.")
s2 = s.replace(head_old, "The fork after row 191:").replace(tail_old, tail_new)

sp = rd(SPF)
if sp.count("* Row 189 (card ADD122") != 1 or sp.count("* Row 191 (card ADD124") != 1:
    stop("spec does not carry the bullets of rows 189 and 191 once each")

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
print("ZZ APPEND-R191 DONE frag_row127 %d -> %d, status %d -> %d chars" % (len(t), len(t2), len(s), len(s2)))
