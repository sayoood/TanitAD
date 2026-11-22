# append_r186.py -- paper v11: append the fragments of rows 185 and 186 to frag_row127.tex, replace the fork's status sentence, add the spec bullet of row 186.
# The spec bullet of row 185 and the PI decision bullet of 2026-11-22 are in the spec already (asserted here). The cost card is not changed. Every step is asserted. Run once.
import io
import re
import sys

V = "F:/TanitAD-scratch/v8proof/"
D = V + "paper/v11/"
SPF = V + "CANDIDATE_SPEC_V8F_10K_2026-11-20.md"
NEW = (185, 186)


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def has_crlf(p):
    return "\r\n" in io.open(p, encoding="utf-8", newline="").read()


def stop(msg):
    print("ZZ APPEND-R186 STOP: " + msg)
    sys.exit(2)


t = rd(D + "frag_row127.tex")
if t.count("H-V8F-PROOF-184") != 1 or any("H-V8F-PROOF-%d" % n in t for n in NEW) or not t.endswith("\n"):
    stop("frag_row127.tex is not in the state after row 184")
t2 = t
for n in NEW:
    x = rd(D + "frag_row%d.tex" % n)
    if x.count("H-V8F-PROOF-%d" % n) != 1:
        stop("fragment %d does not carry its row once" % n)
    t2 = t2 + "\n" + x.rstrip("\n") + "\n"

s = rd(D + "frag_status_run.tex")
pat = re.compile(r"Open on this fork after row 184.*?alone and with the margin in the rollout, run on the fork pod\.")
hits = pat.findall(s)
if len(hits) != 1:
    stop("status sentence found %d times" % len(hits))
old = hits[0]
head_old = "Open on this fork after row 184:"
tail_old = ("Card ADD119, the margin at the pre-rolls of 30 and of 60 steps, and card ADD120, the candidate with the joined presence input alone and with the margin in the rollout, "
            "run on the fork pod.")
if old.count(head_old) != 1 or old.count(tail_old) != 1 or not old.endswith(tail_old):
    stop("status sentence head or tail not found once")
tail_new = (r"Row 185, card ADD119: at the pre-roll of 30 steps the margin of 10 per cent lowers the unsafe rate from 0.0839 to 0.0571 with falling behind not separated; the "
            r"pre-roll of 60 steps has too few windows to decide alone. Row 186, card ADD120: in the rollout the joined presence input is not separated on the unsafe rate at the "
            r"convention ($-0.0071$) and moves nothing under readout K; with the margin the candidate stands at 0.0383 at the convention, not separated from the label form and "
            r"above the card's bound, at a price in falling behind. The PI adopted the margin of 10 per cent for readout K on 2026-11-22. Open on the unsafe rate at the convention: "
            r"the gap value (a new encode) and the number of model seeds, both decisions of the PI.")
new = old.replace(head_old, "The fork after row 186:").replace(tail_old, tail_new)
s2 = s.replace(old, new)
old_nums = set(re.findall(r"ADD\d+|\d+\.\d+|\d{2,}", old))
lost = sorted(n for n in old_nums if n not in new and n not in ("184",))
if lost:
    stop("status sentence lost " + ", ".join(lost))

sp = rd(SPF)
a = "* Open: the question of what reaches the requirement of r 0.9 with no new encode is read out"
if sp.count(a) != 1 or sp.count("* Row 185 (") != 1 or sp.count("* PI decision, 2026-11-22") != 1 or "* Row 186 (" in sp:
    stop("spec is not in the state after row 185 and the PI decision")
b186 = ("* Row 186 (card ADD120, tier T1-kin/hb0.2; nine of nine controls passed): `PRESROLL10K-CONV-BOTH-LEVERS-DO-NOT-REACH-LABEL-FORM`, "
        "`PRESROLL10K-CONV-PRESENCE-UNSAFE-NOT-SEPARATED`, `PRESROLL10K-K0-BOTH-LEVERS-REACH-LABEL-FORM-AT-A-PRICE`, `PRESROLL10K-K0-PRESENCE-UNSAFE-NOT-SEPARATED`. The joined "
        "presence input of row 184 in the rollout (contact / unsafe / falling behind by more than 10 m). Convention: the candidate 0.0016 / 0.0569 / 0.2329; with the presence "
        "input 0.0008 / 0.0498 / 0.2368; with the margin 0.0008 / 0.0463 / 0.2539; with both 0.0000 / 0.0383 / 0.2609; label form 0.0016 / 0.0320 / 0.2243. The presence input "
        "alone: unsafe -0.0071 [-0.0161, 0.0000], not separated. Both levers against the label form: unsafe +0.0062 [-0.0042, +0.0165], falling behind +0.0366 [+0.0169, "
        "+0.0602]; the level 0.0383 is above the card's bound of 0.0365. Readout K, no pre-roll: with the presence input 0.0000 / 0.0489 / 0.1659, with both 0.0000 / 0.0374 / "
        "0.1737; the presence input moves nothing there (0.0000 [0.0000, 0.0000] on top of the margin). The presence input is not taken into the candidate: the presence lever "
        "with the existing tokens ends here. Stated limit at the convention: the candidate of this spec (no margin there) has an unsafe rate of 0.0569 against 0.0320 of the "
        "label form; what is left is the gap value (a new encode) and the number of model seeds, both decisions of the PI.\n")
sp2 = sp.replace(a, b186 + a)

io.open(D + "frag_row127.tex", "w", encoding="utf-8", newline="\n").write(t2)
io.open(D + "frag_status_run.tex", "w", encoding="utf-8", newline="\n").write(s2)
io.open(SPF, "w", encoding="utf-8", newline=("\r\n" if has_crlf(SPF) else "\n")).write(sp2)
print("ZZ APPEND-R186 DONE frag_row127 %d -> %d, status %d -> %d, spec %d -> %d chars" % (len(t), len(t2), len(old), len(new), len(sp), len(sp2)))
