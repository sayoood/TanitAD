"""patch_tex_v6b.py -- second edit of the version 6 tex (lead, 2026-11-15): the read of the six-seed replicate (card PREREG_V8F_PROOF_ADD11).
Source of every number: hiergap/read_pod/VERDICT_HIERREP.md (written by the pod chain on 2026-11-14T16:04Z, sha256 a6a7525e479dac61...; reader hierrep_read.py, rc 0).
Input: tanitad_4b_paper.tex as the agent build left it (sha256 asserted). Every replacement must match exactly once. The earlier file is kept as tanitad_4b_paper_v6a.tex."""
import hashlib
import io
import shutil

P = "tanitad_4b_paper.tex"
raw = io.open(P, "rb").read()
assert hashlib.sha256(raw).hexdigest() == "574c9574eb1f5af92e02bf4b97d7649a353eb725f1aaafb9025a62276ffac0d4", hashlib.sha256(raw).hexdigest()
shutil.copyfile(P, "tanitad_4b_paper_v6a.tex")
s = raw.decode("utf-8")
NL = "\r\n" if "\r\n" in s else "\n"
s = s.replace("\r\n", "\n")


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


# 1 abstract
rep("The hierarchy is ahead of a flat control on selection-critical windows and not established on all windows; the six-seed replicate that decides it is trained and its read is pending.",
    "In the deciding six-seed replicate the hierarchy is ahead of its flat control under the primary protocol, the own-state fine-tune "
    "($-0.09$\\,m on all windows and $-0.20$\\,m on selection-critical windows, six of six seed differences negative, both intervals excluding zero); "
    "for training from scratch the comparison is not established.")

# 2 Table 1, row of T2
rep("open: top-1 selection 57--59\\,\\%; the hierarchy is ahead of a flat control on selection-critical windows ($-0.48$/$-0.16$\\,m) and not established on all windows \\\\",
    "premise open (top-1 selection 57--59\\,\\%); measured at six seeds: hierarchy ahead under the primary protocol ($-0.09$\\,m on all windows, six of six seeds), not established from scratch \\\\")

# 3 Section 8.1, the deciding experiment
rep("with seeds as the unit; all 24 arms were trained on 14 Nov and the read is pending, so there is no verdict here. A window class\n"
    "reads for the hierarchy only when all six seed differences agree in sign and the two-level interval excludes zero.",
    "with seeds as the unit. A window class\n"
    "reads for the hierarchy only when all six seed differences agree in sign and the two-level interval (seeds, then episodes) excludes zero.\n"
    "\n"
    "\\paragraph{The six-seed replicate.} All 24 arms are trained and read (\\tier{T1-kin}, 2,399 windows of 141 episodes, 7,214,495 parameters in\n"
    "every arm, every control at its known value, the strategic family paused). Under the primary protocol, the own-state fine-tune, the\n"
    "outcome is HIER-DOMINATES: 6\\,s ADE, hierarchy minus flat, is $-0.092$\\,m $[-0.168,-0.025]$ on all windows and $-0.195$\\,m\n"
    "$[-0.293,-0.098]$ on selection-critical windows, with all six seed differences negative in both cells (exact sign-flip $p=0.031$, the\n"
    "smallest value six seeds allow); on common-case windows it is $+0.0305$\\,m $[-0.0541,+0.1202]$, open. By family on all windows:\n"
    "cross-track error $-0.165$\\,m $[-0.219,-0.097]$ and heading error $-1.76^\\circ$ $[-2.24,-1.28]$ read for the hierarchy; the\n"
    "longitudinal, lateral and joint manoeuvre error rates are lower by $0.100$ $[0.063,0.135]$, $0.045$ $[0.027,0.061]$ and $0.104$\n"
    "$[0.065,0.143]$; tactical-band ADE is $-0.144$\\,m $[-0.255,-0.043]$; speed and along-track error are open ($-0.087$\\,m/s\n"
    "$[-0.197,+0.003]$ and $+0.024$\\,m $[-0.049,+0.103]$). Under the secondary protocol, training from scratch without own-state data, the\n"
    "outcome is NOT-ESTABLISHED: $-0.158$\\,m $[-0.266,-0.034]$ on selection-critical windows with five of six seed differences negative,\n"
    "$+0.3025$\\,m $[+0.1437,+0.4550]$ on common-case windows, where the flat control is ahead, and $+0.052$\\,m $[-0.061,+0.169]$ on all\n"
    "windows; there the flat control is also ahead on speed error ($+0.298$\\,m/s) and on the longitudinal manoeuvre error rate ($+0.097$),\n"
    "and the hierarchy on cross-track error ($-0.184$\\,m). The label-swap control reads open in 20 of 20 draws. The advantage of the\n"
    "hierarchy is therefore established where the model has been trained on its own executed motion and is not established without it.\n"
    "Limits: one corpus of 4,512 training clips, one evaluation set, a kinematic rollout on logged frames; headway and the stop-goal\n"
    "decision are not in this read, and the planner is deterministic, so inference variance is not sampled.")

# 4 outlook
rep("Three further reads are committed: the deciding six-seed replicate of the hierarchy\n"
    "against its flat control (all 24 arms were trained on 14 Nov; the read is pending and this paper states no verdict), the\n"
    "dominance premise on those six seeds, and a go-or-stay cue at rest from perception.",
    "The deciding six-seed replicate of the hierarchy against its flat control is read: the hierarchy dominates under the primary\n"
    "protocol and the comparison from scratch is not established. Two further reads are committed: the\n"
    "dominance premise on those six seeds, and a go-or-stay cue at rest from perception.")

io.open(P, "w", encoding="utf-8", newline="").write(s.replace("\n", NL))
print("patched", P, hashlib.sha256(io.open(P, "rb").read()).hexdigest())
