# append_r177.py -- paper v11: append the fragment of row 177 to frag_row127.tex, replace the fork's status sentence, add one spec bullet. The cost card is not changed (no decision arises).
# Every step is asserted. Run once.
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
    print("ZZ APPEND-R177 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex")
if t.count("H-V8F-PROOF-176") != 1 or "H-V8F-PROOF-177" in t or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 176")
x7 = rd(D + "frag_row177.tex")
if x7.count("H-V8F-PROOF-177") != 1:
    stop("the fragment does not carry its row once")
t2 = t + "\n" + x7.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex")
pat = re.compile(r"Open on this fork after row 176.*?is next on the fork pod\.")
hits = pat.findall(s)
if len(hits) != 1:
    stop("status sentence found %d times" % len(hits))
old = hits[0]
head_old = "Open on this fork after row 176:"
tail_old = "Card ADD111, the split of the camera gap error into a slow and a fast part and two smoothing forms with no label, is next on the fork pod."
if old.count(head_old) != 1 or old.count(tail_old) != 1 or not old.endswith(tail_old):
    stop("status sentence head or tail not found once")
tail_new = (r"Row 177 reads card ADD111, the split of the camera gap error into a slow and a fast part and two smoothing forms with no label: neither part carries the price in falling "
            r"behind alone ($-0.0092$ and $-0.0138$ against the camera-fed form, neither separated), the trailing median pays with contact ($+0.0277$ [$+0.0064$, $+0.0593$]) and the "
            r"trailing line fit gives no gain. Card ADD112, a bound at \tier{T0} on how much of the slow part a monotone map of the gap value can remove, is next on CPU.")
new = old.replace(head_old, "Open on this fork after row 177:").replace(tail_old, tail_new)
s2 = s.replace(old, new)
old_nums = set(re.findall(r"ADD\d+|\d+\.\d+|\d{2,}", old))
lost = sorted(n for n in old_nums if n not in new and n != "176")
if lost:
    stop("status sentence lost " + ", ".join(lost))

sp = rd(SPF)
a = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out"
if sp.count(a) != 1 or "Row 177" in sp or sp.count("* Row 176 (") != 1:
    stop("spec is not in the state after row 176")
b177 = ("* Row 177 (card ADD111, tier T1-kin/hb0.2, pre-roll of 30 steps (6 s); six of six controls passed): `GAPERR10K-PARTS-NOT-SEPARATED`, `GAPERR10K-SM-PAYS-SAFETY`, "
        "`GAPERR10K-SL-NO-GAIN`. Neither the slow nor the fast part of the camera gap error carries the price in falling behind alone (label gap plus the slow part -0.0092 "
        "[-0.0322, +0.0125], plus the fast part -0.0138 [-0.0441, +0.0080], against 0.5154; attribution arms with label values, no candidates). Of the two smoothing forms with no "
        "label the trailing median raises contact to 0.0338 from 0.0062 (+0.0277 [+0.0064, +0.0593]) and is not carried further; the trailing line fit gives no gain (falling behind "
        "+0.0000 [-0.0153, +0.0158]). The gap input of this spec is unchanged and no smoothed gap is proposed. Next: card ADD112 (tier T0, CPU): how much of the slow part a monotone "
        "map of the gap value can remove, read before fold fits and rollouts for a recalibrated gap are paid.\n")
sp2 = sp.replace(a, b177 + a)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline=("\r\n" if has_crlf(SPF) else "\n")).write(sp2)
print("ZZ APPEND-R177 DONE frag_row127 %d -> %d, status %d -> %d, spec %d -> %d chars" % (len(t), len(t2), len(old), len(new), len(sp), len(sp2)))
