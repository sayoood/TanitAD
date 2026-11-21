# NUMBERS_CHECK: every on-screen number in tanitad_pitch.mp4 and its FACTS line

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
| 2 | under 300 M parameters | programme claim | L17 |
| 3 | strategic: ≤ 0.5 Hz, 6–30 s | design | L21 |
| 3 | tactical: 1–2 Hz, 2–6 s | design | L20 |
| 3 | operative: 10–20 Hz, 0.1–2 s | design | L19 |
| 3 | fallback: continuous | design | L22 |
| 3 | sub-300 M-parameter, self-supervised, hierarchical latent world model | design | L17 |
| 4 | Frozen DINOv3, L/16 features | architecture | L34 |
| 4 | route checkpoints 30 / 50 / 80 m, about 1 Hz | architecture | L35 |
| 4 | typed 60-d code; 22 goals, 5 lateral, 7 longitudinal, 26 constraints; about 2 Hz | architecture | L36 |
| 4 | 6 s trajectory, 10 Hz | architecture | L38 |
| 4 | residual on kinematic prior P10; zero residual = the prior exactly, bit for bit | architecture | L40 |
| 4 | H-JEPA world-model auxiliary on the shared adapter | architecture | L39 |
| 4 | 7.2 M trainable parameters | size | L43 |
| 4 | ≈ 4 h on one 8 GB consumer GPU | cost | L43 |
| 4 | 4,512 training clips | data | L46 |
| 4 | 141 held-out episodes | data | L46 |
| 4 | inputs: front-wide camera, ego motion incl. yaw rate, nav, speed bucket | architecture | L44 |
| 5 | S about 1 Hz, routing 6–30 s; T about 2 Hz, manoeuvres 2–6 s; O 10 Hz, control 0.1–2 s | T0 diagnostic, qualitative | L36 |
| 6 | K-1: −0.31 / −0.43 m 6 s ADE vs P10 (0.312 / 0.432), two seeds | open loop, model's own held outputs | L51 |
| 6 | pre-registered bar 0.2415 m | open loop | L52 |
| 6 | 12.5–17.4 % lower 6 s error | open loop | L52 |
| 6 | HACC rows: ADE 2 s −0.011 / −0.008 m | open loop, zero-training post-map | L54 |
| 6 | HACC rows: ADE 6 s −0.50 / −0.47 m | open loop, zero-training post-map | L54 |
| 6 | HACC rows: acceleration error −0.022 / −0.040 m/s² | open loop, zero-training post-map | L54 |
| 6 | HACC rows: curvature, speed, heading, yaw rate, decision agreement, consistency (check marks) | open loop | L54 |
| 6 | Nothing is separated-worse; distance keeping is not separated | open loop | L55 |
| 6 | own executed motion: 6 s ADE −1.26 / −1.19 m (1.2556 / 1.1903), DAG-OACC vs P10 | T1-kin, own executed motion, logged cameras | L70 |
| 6 | four of four metric families improved somewhere | T1-kin | L71 |
| 6 | route checkpoints beat a kinematic arc by up to 8.8 m: +1.2 / +3.3 / +7.8 and +1.3 / +3.5 / +8.8 m at 30 / 50 / 80 m | T1-kin | L63 |
| 6 | typed interface about 1.1 m of headroom; about 2.1 m to about 1.0 m | diagnostic T0, oracle by interface | L62 |
| 7 | three variance questions (episodes, training seeds, inference) | method | L100 |
| 7 | four metric families on every evaluation | method | L101 |
| 7 | append-only log of revised claims | method | L102 |
| 8 | 62 scenes (42 launch, 20 guard) selected by a rule frozen before any log was read | T2, in preparation | L82 |
| 8 | harness reproduces the plans bit for bit | T2, in preparation | L83 |
| 8 | one open behaviour: assertive launch from standstill; every zero-training lever measured | T1-kin | L75 |
| 9 | LoRA fine-tune of the visual encoder; A40-class GPU, about 100 GPU-h per seed (estimated) | Outlook | L88 |
| 9 | from 7 M toward the 300 M 4B budget; far larger than the 4,512 clips used | Outlook | L89 |
| 9 | Jetson Thor, at most 100 ms end to end with 4 cameras | Outlook | L93 |
| 9 | ISMR / DSSAD logging; minimal-risk manoeuvre | Outlook | L22 |
| 9 | NavSim / navtest benchmarks, T2 reads | Outlook | L94 |
