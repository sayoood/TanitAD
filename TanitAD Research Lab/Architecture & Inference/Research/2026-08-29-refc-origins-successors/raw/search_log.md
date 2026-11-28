# Search log — refc origins & successors (2026-08-29)

Chronological; every probe named so absence claims are auditable. Session ran through a full
G:-mount outage (content reads errno-22/"Unzulässige Funktion" while Glob/metadata worked) —
all repo reads below the outage line went through the Google Drive API or the off-Drive mirror.

## 1. Repo identity probes (STEP 1)
- `Glob stack/tanitad/refs/refc*.py` → refc.py, refc_select.py, refc_tactical.py, refc_v3.py. Direct Read → EISDIR (mount flap).
- MSYS cp retry ×10 (~6 s spacing) FAILED for all 5 targets; PowerShell Copy-Item retry ×8 FAILED; Get-Content error "Unzulässige Funktion" while Get-Item returned correct sizes ⇒ known G: hard-failure mode (memory: gdrive-mount-hard-failure). Monitor armed on CLAUDE.md readability (20 s poll, 1 h timeout) — never fired during the session.
- Off-Drive mirror `C:\Users\Admin\tanitad-wt\stack\tanitad\refs\` — refc.py 134,405 B (byte-identical size to G: metadata), mirrored 2026-08-23 → read docstrings there. refc.py:1-8 names TCP-C (2206.08129) → DiffusionDrive (2411.15139). refc_v3.py docstring read (hierarchy, GoalDistanceScorer, E-edges).
- Drive API `search_files title='REFC_V3_DESIGN.md'` → 5 copies (worktree backups), all 37,980 B; newest (2026-08-27, post-rename tree) read in full via read_file_content.
- Drive API: REFCV3_REVIEW.md (12,352 B) read in full — carries the "what we changed from DiffusionDrive" table and admits its DD comparison was abstract-level only.
- Drive API: LIBRARY.md (live copy, modified 2026-08-29 11:29 UTC, 97,213 B) downloaded, base64-decoded locally. Greps: 2411.15139 BANKED (sha12 6ad4f8a37949, note "THE paper REF-C is derived from"); Hydra-MDP/++, GoalFlow, DriveSuprim, WoTE, VADv2, TransFuser, NAVSIM, TrajHF, SimWAM, Drive-JEPA, Latent-WAM, FROST-Drive all banked. NOT found: 2512.07745, 2507.04049, 2504.19580, 2206.08129, ReCogDrive, FeaXDrive, ReflectDrive-2, TransDiffuser.
- Drive API: NAVSIM_PROTOCOL.md (57,145 B) downloaded/decoded; §6 read in full (P1–P4 protocol families, published-baseline tables with per-row arXiv+table citations), §7 ego-shortcut section read.

## 2. Primary verification (sha-match against Library hashes)
Downloads from arxiv.org/pdf/<id> (curl --ssl-no-revoke), sha256 vs LIBRARY.md sha12:

| id | expect | got | verdict |
|---|---|---|---|
| 2411.15139 DiffusionDrive | 6ad4f8a37949 | 6ad4f8a37949… | MATCH — local copy IS the banked primary |
| 2506.06659 DriveSuprim | 33cf828e4a5c | MATCH | |
| 2503.05689 GoalFlow | 4206cc1b3aa4 | MATCH | |
| 2503.12820 Hydra-MDP++ | e70fea2dddbf | MATCH | |
| 2406.06978 Hydra-MDP | 8927b0ffbfb0 | MATCH | |
| 2504.01941 WoTE | 04569f633b3c | MATCH | |
| 2503.10434 TrajHF | eb18c94e3eaf | MATCH | |
| 2601.22032 Drive-JEPA | d88c053a67ac | MATCH | |
| 2608.07468 SimWAM | 03ce230505c4 | 6f55da6df120 (latest), c4469a44a324 (v1) | MISMATCH both — arXiv re-render or version bump; banked bytes NOT reproduced; SimWAM numbers kept INHERITED |
| 2512.07745 DiffusionDriveV2 | (new) | 076ce47e0b0a | downloaded for banking |

Text extracted with PyMuPDF (system python `fitz`); DiffusionDrive read 15/15 pages (100 %);
DiffusionDriveV2 read: abstract/intro (p1-2), method §4.4-4.6, experiments §5 incl. Tab. 1/2/3/4/5/6, conclusion.
Spot-greps on banked-matched copies: DriveSuprim (93.5 headline, coarse-to-fine/rotation/self-distill), GoalFlow
(goal-point vocabulary + flow matching, 90.3), TrajHF (94.x rows are "PDMS selector" rows), Drive-JEPA (93.7 P1 / 87.8 P4 abstract; 93.3 exists as a table value — library-note reconcile flagged).

## 3. Web searches / fetches
- WebSearch "DiffusionDrive successor improved diffusion planner NAVSIM PDMS 2025 …" → found DiffusionDriveV2 (2512.07745), FeaXDrive (2604.12656), ReflectDrive-2 (2605.04647), GitHub hustvl/DiffusionDrive (CVPR 2025 Highlight).
- WebFetch arxiv.org/abs/2512.07745 → author overlap confirmed (Liao, S. Chen, Q. Zhang, X. Wang; HUST+Horizon), limitations-of-original quotes, GRPO method sketch. Full details then taken from the downloaded PDF, not the fetch.

## 4. Sources found but DROPPED (named per instruction, not chased)
- **ReCogDrive** (VLM + diffusion planner + RL, hustvl) — identified as a likely successor line; never fetched/verified this session.
- **FeaXDrive** 2604.12656 (feasibility-aware diffusion planning) — thematically adjacent to our reach-clamp; not fetched.
- **ReflectDrive-2** 2605.04647 (RL-aligned discrete diffusion) — not fetched.
- **TransDiffuser** 2505.09315 — not fetched.
- **hustvl/DiffusionDrive GitHub README** (release notes / variants) — not fetched; outlook was sourced from the papers instead.
- **Latent-WAM / FROST-Drive / World4Drive / DeepSight** — banked already; not re-read this session (out of the 3-page budget); their library notes were not load-bearing for any RESULT number except the Latent-WAM 89.3 P4 mention, which was dropped from the final tables.

## 5. Decisions of record
- The 85.5 "NAVSIM v2" number of DDv2 is **P4** (single-stage EPDMS on navtest — DDv2 Tab. 2 caption "NAVSIM v2 navtest split"); labeled accordingly, per NAVSIM_PROTOCOL.md §6.4 and the [N2] §4.2 discouragement.
- TrajHF's ~94 PDMS rows are PDMS-selector rows (metric-informed selection) — cited qualitatively only.
- GoalFlow† 92.1 (GT-endpoint goal) marked privileged, never deployable — matches the standing NAVSIM_PROTOCOL.md warning and our 2026-08-03 goal ruling.
- DD Tab. 4 step-3 row prints EP 92.2 with PDMS 88.1 — flagged as a paper typo (EP 82.2 consistent).
- DIVER/ARTEMIS numbers quoted from DDv2 Tab. 1 (which states it uses official same-R34-backbone scores) — marked INHERITED, banks queued so they can be upgraded to PRIMARY.

## 6. Environment notes for the landing agent
- G: down the whole session (metadata OK, content errno-22). Everything authored on C: scratchpad.
- Monitor task buxlrn1nj (G: recovery probe) may still be running at session end — kill or let it time out (1 h).
- PDFs + extracted txt for all verified papers: `<scratchpad>/papers/` (2411.15139.pdf/.txt, 2512.07745.pdf/.txt, 2506.06659, 2503.05689, 2503.12820, 2406.06978, 2504.01941, 2503.10434, 2601.22032, 2608.07468 both non-matching versions).
