"""patch_tex_v4.py -- the TanitAD-4B paper, version 4 (2026-11-13): v3's tex plus where the tactical headroom sits (oracle reads),
the stop-decision chain and its perception ceiling, and the closed recipe levers.  Every edit must match exactly once.
Usage: python -I patch_tex_v4.py <v3.tex> <out.tex> <stage1_status>
  stage1_status: "training" if the STOPPERC stage-1 chain has started, else "queued" (sets one phrase in the text and one roadmap cell).
Sources of every new number (MEASURED; T1-kin first tick unless marked; folder raw/ of the v8f-mini programme):
  2026-11-10-v8f-consorc/VERDICT_TOPK_PAIR_CONSORC.md   selector share 0.239/0.228; GT-ALL - TOP1 -1.625/-1.773; cost shares CONS 0.872/0.892, DEC 0.111/0.094
  2026-11-10-v8f-speedsrc/VERDICT_SPEEDSRC.md           STOP_POINT mass 0.200/0.197 on 0.050 of windows
  2026-11-10-v8f-stopdec/VERDICT_STOPDEC.md             goal-head hit 0.103/0.128 on 117 stop-point windows
  2026-11-10-v8f-stopcue/VERDICT_STOPCUE.md             AUROC 0.836/0.857 (131 positives / 2,365 windows)
  2026-11-10-v8f-stopbal/oracle/VERDICT_STOPORC.md      forced-minus-own -1.539/-1.477 m (5.02 -> 3.48, 4.94 -> 3.47)
  2026-11-10-v8f-stopbal/oracle/VERDICT_STOPHARD.md     hardening recovers 3-6 % of the ceiling; all-window costs
  2026-11-10-v8f-stopbal/oracle/VERDICT_STOPTAU.md      TAU-TRADES / TAU-NULL
  2026-11-10-v8f-stopbal/read/VERDICT_STOPBAL.md and dose2/read/VERDICT_DOSE2.md   recall NULL 0.060-0.145 -> dose 2 0.453-0.615;
                                                        false stops 0.005-0.013 -> 0.120-0.145; AUROC inside its floor; P floor 0.703
  2026-11-11-v8f-stopperc/VERDICT_S0.md                 c 0.7 (82 of 117): -0.957/-0.855 m; yaw rate +2.166/+2.178 deg/s
  2026-11-11-v8f-stopperc/VERDICT_S0P.md                probe AUROC 0.8387-0.8653 over all input sets (pool8x20 included)
  2026-11-10-v8f-obd/VERDICT_OBD.md, read_seed3/VERDICT_OBD_SEED3.md   DETACH-KEEPS; operative gain 0.0014-0.0020 m; BAND-UNSETTLED
  2026-11-09-v8f-plantest3/decinfo/egowm/VERDICT_EGOWM.md   NO-EFFECT (T0)"""
import hashlib
import sys

SRC, OUT, ST1 = sys.argv[1], sys.argv[2], sys.argv[3]
assert ST1 in ("training", "queued"), ST1
b = open(SRC, "rb").read()
assert hashlib.sha256(b).hexdigest() == "1b783e91bc86ce772617f06f5d3ff8c4912692bce00dab759a40e3145b0ebbaa", "base is not v3's tex"
s = b.decode("utf-8")
STAT = "now in training" if ST1 == "training" else "queued to train"
CELL = "in training, two seeds" if ST1 == "training" else "queued, two seeds"
EDITS = []

# A. abstract: one sentence on where the tactical headroom sits
EDITS.append((
    "90\\,\\% of windows. All of it rests on an evaluation",
    "90\\,\\% of windows. Oracle reads place 87--89\\,\\% of the remaining tactical headroom in the tactical level's continuous "
    "constraints, chiefly target speed, and its sharpest cause in missed stop decisions: the operative level executes a correct stop "
    "(1.5\\,m lower tactical error on stop-point windows), while loss re-weighting only moves the stop decision along one ROC curve, so "
    "the next lever is stop perception. All of it rests on an evaluation"))

# B. headline table: four rows after the top-3 row
EDITS.append((
    "best of three plans $-0.35$ to $-0.39$\\,m 2--6\\,s ADE (oracle) & T1-kin \\\\\n",
    "best of three plans $-0.35$ to $-0.39$\\,m 2--6\\,s ADE (oracle) & T1-kin \\\\\n"
    "Tactical headroom: 87\\,/\\,89\\,\\% in the continuous constraints; a forced correct stop lowers stop-point tactical ADE by "
    "1.54\\,/\\,1.48\\,m; stop re-weighting moves recall only along one ROC curve (oracle) & T1-kin \\\\\n"))

# C. results: a paragraph after "Which lever carries which band"
EDITS.append((
    "trained selection over the tactical codes, and the next lever.\n",
    "trained selection over the tactical codes, and the next lever.\n\n"
    "\\paragraph{Where the tactical headroom sits.} We measured that lever before building it, with zero-training oracle reads on the "
    "adopted recipe (\\tier{T1-kin}, first tick, both seeds, every control at its known value; the strategic family paused). A perfect "
    "selector over the top three manoeuvres captures only 24/23\\,\\% of the best-of-three gain, so the choice among manoeuvres is not "
    "where the headroom lies. Writing the ground-truth tactical code in place of the model's own lowers tactical ADE by 1.63/1.77\\,m. "
    "Leaving out its continuous constraints forfeits 87/89\\,\\% of that gain and leaving out the discrete decision 11/9\\,\\%; target "
    "speed is the costliest single constraint. The speed error spreads over every stratum, but stop-point windows hold 20\\,\\% of its "
    "squared mass on 5\\,\\% of the windows. There the decision misses: the goal head says stop in 10/13\\,\\% of the 117 stop-point "
    "windows, although its stop score separates them from the rest with AUROC 0.84/0.86. Forcing the correct stop into the tactical code "
    "lowers their tactical ADE by 1.54/1.48\\,m (30--31\\,\\%), so the operative level executes a stop once it is decided.\n\n"
    "Three pre-registered levers then acted on the decision alone. Hardening the model's own decision at inference recovers 3--6\\,\\% of "
    "that gain at a cost on the other windows, and a threshold chosen on training clips does not help. Re-weighting the stop classes at "
    "two doses raises stop recall from 0.06--0.15 to 0.45--0.62 and the false-stop rate from 0.5--1.3\\,\\% to 12--14.5\\,\\%, with "
    "AUROC unchanged within its replicate floor: the weights slide the operating point along one ROC curve and add no discrimination, "
    "and stop-point ADE does not improve beyond its 0.70\\,m floor. The missing piece is discrimination, which is a perception question. "
    "Its ceiling was fixed before any training: a correct stop on the 82 of 117 stop-point windows that the model already ranks highest "
    "lowers tactical ADE there by 0.96/0.86\\,m, beyond the floor at both seeds, with a separated lateral cost (yaw-rate error "
    "$+2.2^{\\circ}$/s) that comes with the forced stop. Linear and small MLP probes on the frozen features rank stops no better than "
    "the model (AUROC 0.84--0.87 for every input set, a 4$\\times$ finer token grid included), so the first perception arm trains instead "
    "of probing: a vision-only stop-location auxiliary loss, " + STAT + " at two seeds against same-seed controls and two replicates. "
    "Two recipe levers closed on the way: on the DAgger base, detaching the operative level costs nothing, and the band loss's operative "
    "gain shrinks to 1--2\\,mm and does not settle over four seeds.\n"))

# D. roadmap rows
EDITS.append((
    "do not yet beat holding the logged action (\\tier{T0}) \\\\\n",
    "do not yet beat holding the logged action (\\tier{T0}); giving its predictor the measured speed does not raise its skill beyond "
    "the seed floor (\\tier{T0}); a slow-feature basis is being read \\\\\n"))
EDITS.append((
    "Band retraining \\textit{(read, two seeds)} & a reproduced 0--2\\,s gain; the 2--6\\,s band moves with on-policy data, so the detached recipe is next on the DAgger base \\\\\n"
    "Tactical selection \\textit{(designed)} & a trained choice among the top-K tactical codes; measured oracle headroom 0.35--0.39\\,m of 2--6\\,s ADE \\\\\n",
    "Band retraining \\textit{(read, four seeds)} & on the DAgger base the detach costs nothing; the band loss's 1--2\\,mm operative gain does not settle, so it is closed as a recipe lever \\\\\n"
    "Tactical selection \\textit{(oracle read)} & a perfect top-3 selector captures 24/23\\,\\% of the best-of-three headroom; the headroom sits in the speed constraint, so no selector is built \\\\\n"
    "Stop perception \\textit{(" + CELL + ")} & a vision-only stop-location auxiliary on the tactical level; measured ceiling 0.86--0.96\\,m tactical ADE on stop-point windows (oracle) \\\\\n"))

# E. outlook bullet
EDITS.append((
    "\\textbf{perception adaptation} for motion cues;",
    "\\textbf{perception adaptation} for motion and stop cues;"))

for old, new in EDITS:
    n = s.count(old)
    assert n == 1, (n, old[:80])
    s = s.replace(old, new)
open(OUT, "wb").write(s.encode("utf-8"))
print("ok", len(EDITS), "edits ->", OUT, hashlib.sha256(s.encode("utf-8")).hexdigest())
