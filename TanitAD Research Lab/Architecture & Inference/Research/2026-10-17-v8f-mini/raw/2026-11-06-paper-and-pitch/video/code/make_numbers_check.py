"""Generate video/NUMBERS_CHECK.md: every on-screen number -> its FACTS_PITCH.md line (searched, not typed)."""
import re
FACTS = "E:/Projects/TanitAD-artifacts/dagland/paper_v8f/FACTS_PITCH.md"
OUT = "E:/Projects/TanitAD-artifacts/dagland/paper_v8f/video/NUMBERS_CHECK.md"
L = open(FACTS, encoding="utf-8").read().split("\n")


def find(key):
    for i, l in enumerate(L, 1):
        if key in l:
            return i
    raise SystemExit("NOT FOUND in FACTS: " + key)


# (scene, on-screen text, tier shown, FACTS search key)
ROWS = [
    (2, "under 300 M parameters", "programme claim", "sub-300 M-parameter"),
    (3, "strategic: \u2264 0.5 Hz, 6\u201330 s", "design", "**strategic:** \u2264 0.5 Hz"),
    (3, "tactical: 1\u20132 Hz, 2\u20136 s", "design", "**tactical:** 1\u20132 Hz"),
    (3, "operative: 10\u201320 Hz, 0.1\u20132 s", "design", "**operative:** 10\u201320 Hz"),
    (3, "fallback: continuous", "design", "**fallback:** continuous"),
    (3, "sub-300 M-parameter, self-supervised, hierarchical latent world model", "design", "sub-300 M-parameter"),
    (4, "Frozen DINOv3, L/16 features", "architecture", "frozen DINOv3-L/16"),
    (4, "route checkpoints 30 / 50 / 80 m, about 1 Hz", "architecture", "S: route checkpoints at 30 / 50 / 80 m, about 1 Hz"),
    (4, "typed 60-d code; 22 goals, 5 lateral, 7 longitudinal, 26 constraints; about 2 Hz", "architecture", "a typed 60-dimensional tactical code (22 goals"),
    (4, "6 s trajectory, 10 Hz", "architecture", "O: the 6 s trajectory, 10 Hz"),
    (4, "residual on kinematic prior P10; zero residual = the prior exactly, bit for bit", "architecture", "A zero-initialised head equals the prior bit for"),
    (4, "H-JEPA world-model auxiliary on the shared adapter", "architecture", "**Plus** an H-JEPA world-model auxiliary"),
    (4, "7.2 M trainable parameters", "size", "7.2 M (M2b) trainable parameters"),
    (4, "\u2248 4 h on one 8 GB consumer GPU", "cost", "about 4 h on one 8 GB consumer GPU"),
    (4, "4,512 training clips", "data", "4,512 training clips"),
    (4, "141 held-out episodes", "data", "141 held-out episodes"),
    (4, "inputs: front-wide camera, ego motion incl. yaw rate, nav, speed bucket", "architecture", "**Inputs at inference:**"),
    (5, "S about 1 Hz, routing 6\u201330 s; T about 2 Hz, manoeuvres 2\u20136 s; O 10 Hz, control 0.1\u20132 s", "T0 diagnostic, qualitative", "T: a typed 60-dimensional tactical code"),
    (6, "K-1: \u22120.31 / \u22120.43 m 6 s ADE vs P10 (0.312 / 0.432), two seeds", "open loop, model's own held outputs", "improves 6 s ADE over P10 by 0.312 / 0.432 m"),
    (6, "pre-registered bar 0.2415 m", "open loop", "2F = 0.2415 m"),
    (6, "12.5\u201317.4 % lower 6 s error", "open loop", "12.5\u201317.4 % lower 6 s error"),
    (6, "HACC rows: ADE 2 s \u22120.011 / \u22120.008 m", "open loop, zero-training post-map", "ADE2 \u22120.011 / \u22120.008 m"),
    (6, "HACC rows: ADE 6 s \u22120.50 / \u22120.47 m", "open loop, zero-training post-map", "ADE6 \u22120.50 / \u22120.47 m"),
    (6, "HACC rows: acceleration error \u22120.022 / \u22120.040 m/s\u00b2", "open loop, zero-training post-map", "accel MAE \u22120.022 / \u22120.040 m/s\u00b2"),
    (6, "HACC rows: curvature, speed, heading, yaw rate, decision agreement, consistency (check marks)", "open loop", "plus curvature, speed, heading, yaw rate,"),
    (6, "Nothing is separated-worse; distance keeping is not separated", "open loop", "Distance keeping is not separated, and nothing is separated-worse"),
    (6, "own executed motion: 6 s ADE \u22121.26 / \u22121.19 m (1.2556 / 1.1903), DAG-OACC vs P10", "T1-kin, own executed motion, logged cameras", "6 s ADE \u22121.2556 / \u22121.1903 m"),
    (6, "four of four metric families improved somewhere", "T1-kin", "Every one of the four families (longitudinal, lateral, tactical, strategic) has a separated-better row"),
    (6, "route checkpoints beat a kinematic arc by up to 8.8 m: +1.2 / +3.3 / +7.8 and +1.3 / +3.5 / +8.8 m at 30 / 50 / 80 m", "T1-kin", "1.2 / 3.3 / 7.8 m and"),
    (6, "typed interface about 1.1 m of headroom; about 2.1 m to about 1.0 m", "diagnostic T0, oracle by interface", "from\n" if False else "about 2.1 m to about 1.0 m"),
    (7, "three variance questions (episodes, training seeds, inference)", "method", "three variance questions (episodes, training seeds, inference)"),
    (7, "four metric families on every evaluation", "method", "Four metric families are reported on every eval"),
    (7, "append-only log of revised claims", "method", "append-only retraction log"),
    (8, "62 scenes (42 launch, 20 guard) selected by a rule frozen before any log was read", "T2, in preparation", "62 scenes (42 launch, 20 guard)"),
    (8, "harness reproduces the plans bit for bit", "T2, in preparation", "The harness reproduces the T1-kin plans bit for bit"),
    (8, "one open behaviour: assertive launch from standstill; every zero-training lever measured", "T1-kin", "Every zero-training CPU lever has been measured"),
    (9, "LoRA fine-tune of the visual encoder; A40-class GPU, about 100 GPU-h per seed (estimated)", "Outlook", "about 100 GPU-h per seed (ESTIMATED)"),
    (9, "from 7 M toward the 300 M 4B budget; far larger than the 4,512 clips used", "Outlook", "from 7 M toward the 300 M 4B budget"),
    (9, "Jetson Thor, at most 100 ms end to end with 4 cameras", "Outlook", "at most 100 ms end to end with 4 cameras"),
    (9, "ISMR / DSSAD logging; minimal-risk manoeuvre", "Outlook", "ISMR / DSSAD"),
    (9, "NavSim / navtest benchmarks, T2 reads", "Outlook", "NavSim / navtest"),
]

NOTE = """# NUMBERS_CHECK: every on-screen number in tanitad_pitch.mp4 and its FACTS line

Source sheet: `paper_v8f/FACTS_PITCH.md` (tip `24b97ba9`). Line numbers below are searched from the sheet by this script
(`video/code/make_numbers_check.py`), not typed. Scene kickers `01`-`10` are scene indices, not data.

Qualitative on-screen text with no number (problem scene, "Seeing in layers", rigour wording, closing sentence) follows the brief
(`BRIEF_VIDEO.md`) and the FACTS wording rules; the problem statement in scene 2 and the closing sentence in scene 10 are the brief's own
text and are not backed by a FACTS line.

Attention clip (scene 5): the programme's own `v8f_hierarchy_attention.mp4` (v8f report 2026-10-25), cropped to the three camera panels.
The clip's own white percentage labels were removed by inpainting so the film shows no number that is not in FACTS; its captions are cropped
out. Two example clips, qualitative, T0 diagnostic.

Wording rules checked: no "closed loop", no "dominant", no state-of-the-art or rank claim; every result carries a tier chip
(OPEN LOOP / T1-KIN / DIAGNOSTIC T0 / T0 DIAGNOSTIC / T2 in preparation); outlook items sit under the OUTLOOK badge, the T2 harness under
IN PREPARATION.

| scene | on-screen number or claim | tier shown | FACTS_PITCH.md line |
|---|---|---|---|
"""
rows = [f"| {s} | {txt} | {tier} | L{find(key)} |" for s, txt, tier, key in ROWS]
open(OUT, "w", encoding="utf-8", newline="\n").write(NOTE + "\n".join(rows) + "\n")
print("rows", len(rows))
