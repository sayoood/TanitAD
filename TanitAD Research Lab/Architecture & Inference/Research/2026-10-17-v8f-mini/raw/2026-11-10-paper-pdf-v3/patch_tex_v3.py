"""patch_tex_v3.py -- the TanitAD-4B paper, version 3 (2026-11-10): v2's tex plus the band-retrain read and the per-band attribution of
the adopted base.  Every edit must match exactly once.  Usage: python -I patch_tex_v3.py <v2.tex> <out.tex>
Sources of every new number (MEASURED, T1-kin): raw/2026-11-09-v8f-otband/read/out/{READ_FULL,FLOOR_TABLE,TOPK_FULL}.json, register
H-V8F-OTBAND-1, programme paper section 7.24; the LoRA step time from the pod's train log (2026-11-09 23:18 UTC)."""
import hashlib
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
b = open(SRC, "rb").read()
assert hashlib.sha256(b).hexdigest() == "fb1d7ae82815856e988fa604bbdf98edc1717de47f8168500607e67995c46404", "base is not v2's tex"
s = b.decode("utf-8")
EDITS = []

# A. abstract: one sentence on the per-band reading
EDITS.append((
    "least one row in each of the four metric families: longitudinal, lateral, tactical and strategic (a measurement; the strategic bars are paused by decision). All of it rests on an evaluation",
    "least one row in each of the four metric families: longitudinal, lateral, tactical and strategic (a measurement; the strategic bars are paused by decision). "
    "Split by horizon band, on-policy (DAgger) data carries the 2--6\\,s band and an acceleration-matching term the 0--2\\,s band; a per-band loss with the "
    "operative level detached from the tactical one adds a reproduced 0--2\\,s gain, and the tactical level's top three manoeuvres contain the logged one in "
    "90\\,\\% of windows. All of it rests on an evaluation"))

# B. headline table: three rows after the adopted base
EDITS.append((
    "Adopted base: 6\\,s ADE $-1.26$\\,/\\,$-1.19$\\,m vs P10; every family has a separated-better row & T1-kin \\\\\n",
    "Adopted base: 6\\,s ADE $-1.26$\\,/\\,$-1.19$\\,m vs P10; every family has a separated-better row & T1-kin \\\\\n"
    "Per band: DAgger rows lower 2--6\\,s ADE by 0.82\\,/\\,0.62\\,m; the acceleration term lowers 0--2\\,s ADE by 0.062\\,/\\,0.079\\,m and 0--2\\,s acceleration error by 0.97\\,/\\,1.14\\,m/s$^2$ & T1-kin \\\\\n"
    "Band loss, operative level detached: 0--2\\,s ADE $-0.017$\\,/\\,$-0.015$\\,m beyond a same-recipe replicate; no 2--6\\,s row moves & T1-kin \\\\\n"
    "Top-3 tactical manoeuvres contain the logged one in 90\\,\\% of windows (top-1: 57--59\\,\\%); best of three plans $-0.35$ to $-0.39$\\,m 2--6\\,s ADE (oracle) & T1-kin \\\\\n"))

# C. results: a paragraph after "Every level contributes."
EDITS.append((
    "convex, that headroom concentrates in the final gains in tactical precision.\n",
    "convex, that headroom concentrates in the final gains in tactical precision.\n\n"
    "\\paragraph{Which lever carries which band.} Under the band definition of \\cref{sec:bands}, we retrained the operative and tactical levels "
    "with a per-band loss and with the operative level's loss detached from the tactical level, at two seeds and with a same-recipe replicate "
    "(\\tier{T1-kin} throughout). The 0--2\\,s plan improves at both seeds beyond the replicate: ADE by 0.017/0.015\\,m, about 6\\,\\%, and FDE at "
    "2\\,s by 0.039/0.027\\,m. No row of the 2--6\\,s band moves, and the replicate says why: two runs of the same recipe differ there by 0.29\\,m. "
    "A ladder of banked fine-tunes splits the adopted base into its three changes. The DAgger rows carry the tactical band (2--6\\,s ADE "
    "$-0.82$/$-0.62$\\,m, heading $-4.6$/$-2.6^{\\circ}$). The acceleration-matching term carries the operative band and acceleration (0--2\\,s "
    "ADE $-0.062$/$-0.079$\\,m; acceleration error $-0.97$/$-1.14$\\,m/s$^2$ at 0--2\\,s and $-2.0$/$-2.6$\\,m/s$^2$ at 2--6\\,s). Decoding the "
    "operative level once per top-three tactical manoeuvre, the logged manoeuvre is among the three in 90\\,\\% of windows against 57--59\\,\\% for "
    "the first choice, and the best of the three plans would lower 2--6\\,s ADE by 0.35--0.39\\,m. That oracle gap is the measured headroom of a "
    "trained selection over the tactical codes, and the next lever.\n"))

# D. roadmap rows
EDITS.append((
    "World-model planner \\textit{(built)} & top-down planning over three latent levels, the baseline for comparing other approaches \\\\\n",
    "World-model planner \\textit{(built)} & top-down planning over three latent levels, the baseline for comparing other approaches; its first gates on the M2b latents do not yet beat holding the logged action (\\tier{T0}) \\\\\n"))
EDITS.append((
    "Band retraining \\textit{(in training)} & world models, operative and tactical levels on the 2\\,s, 6\\,s and 30\\,s bands \\\\\n",
    "Band retraining \\textit{(read, two seeds)} & a reproduced 0--2\\,s gain; the 2--6\\,s band moves with on-policy data, so the detached recipe is next on the DAgger base \\\\\n"
    "Tactical selection \\textit{(designed)} & a trained choice among the top-K tactical codes; measured oracle headroom 0.35--0.39\\,m of 2--6\\,s ADE \\\\\n"))
EDITS.append((
    "Perception adaptation & LoRA on the visual encoder for motion cues; about 100 GPU-h per seed on an A40-class GPU (estimate) \\\\\n",
    "Perception adaptation \\textit{(in training, seed 0)} & LoRA on the visual encoder for motion cues; about 184\\,h per seed on the 20\\,GB pod GPU at the measured 34.5\\,s per step \\\\\n"))

for old, new in EDITS:
    n = s.count(old)
    assert n == 1, (n, old[:80])
    s = s.replace(old, new)
open(OUT, "w", encoding="utf-8", newline="\n").write(s)
print("edits", len(EDITS), "bytes", len(s.encode("utf-8")), "sha256", hashlib.sha256(s.encode("utf-8")).hexdigest())
