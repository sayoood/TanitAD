# Research Lab audit — 2026-09-02 (~23:10 Berlin)

**Provenance.** A read-only audit by an analyst agent on the Master Mind's brief, against the constitution
(`Project Steering/TANITAD_PROGRAMME.md` §2/§3/§6) and the Lab charter files. Every count was MEASURED on G:
tonight (two probes for every absence); statements taken from documents are tagged INHERITED. The Master Mind
reviewed the report; nothing here is a model or capability claim. Question answered: *"are we running the daily
research lab with literature research, experiments and frontier proposals for the programme?"*

## A. Verdict

**Partially running — the literature half is real and now above the constitution's bar; the experiment half and
the proposal-to-register loop are not.** The daily trigger exists and fires (scheduled task
`tanitad-daily-research-lab`, cron `53 7 * * *` + 494 s jitter, enabled, lastRun 2026-09-02 06:01 UTC; brief at
`C:\Users\Admin\.claude\scheduled-tasks\tanitad-daily-research-lab\SKILL.md`). Since 08-20 the Lab authored
**29 of 48** dated `Research/` packages (MM 12, FlyWheels 7), ran on **6 of the 7 days 08-27 … 09-02** (08-27 has
no Lab output), banked **298 primaries in 14 days** (Library 313 entries; `kb_add.py --verify` at 22:45:
*"verified 313 entries, 0 orphan(s), 0 problem(s)"*), appended **62 KNOWLEDGE_BASE entries** (29 citing a Library
key, 0 URL-only), and produced three charter-grade frontier scans (08-31, 09-01, 09-02) with search logs, ledgers
and 7-step opponent adjudications. But: only **2 GPU experiments** in 12 days (both 4060 microbenchmarks), **0**
reverse-engineered papers (constitution §2 job 2), **H-RANK-17** — run 001's designed arm — unexecuted since
08-23; **0 Lab hypotheses entered `GOALS_AND_CLAIMS.md` after 08-23** (5 did on run 001, one with an ID collision);
**63 PROPOSED backlog rows, 0 ranked** since the section was created 08-31; and the brief prescribes
`RESULT + search_log`, not the §3 schema (SPEC appears in 0/15 Lab packages dated 08-28…08-30, 13/14 from 08-31 on).

## B. Activity since 2026-08-20

| domain | pkgs (Lab / MM / FW) | Lab pkgs with SPEC / raw / tests | KB entries 14 d (lib-key) | Library adds (cited_by, top) | proposals |
|---|---|---|---|---|---|
| Data Engineering | 6 (6/0/0) | 3 / 4 / 1 | 13 (5) | label-source 16, corridor 6 | H-DATA-1 (register, run 001); L-5…L-7, L-10…L-12, P-16, P-18 |
| Architecture & Inference | 27 (9/12/6) | 3 / 7 / 1 | 16 (11) | input-pipeline 65, data-efficiency 36, t1-floor 16 | H-RANK-17 (register); P-1/2/4/15, L-1/2, FS-2/3, GS-1…GS-4, MM-1 |
| Deployment & Optimization | 6 (6/0/0) | 3 / 4 / 1 | 11 (2) — empty by declaration until 08-30 | b1-gate 8 | H-DEPLOY-1 (register); P-6/7, L-3/4/14, FS-4 |
| Opponent Analysis | 4 (4*/0/0) | 2 / 4 / 1 | 10 (5) | waymo 10, action-provenance 9 | H-OPP-1 (register); 36 register rows W/N/M/D; P-3/12/14, L-15, GS-5 |
| Benchmarks & Evals | 5 (4/0/1) | 2 / 2 / 1 | 12 (6) | anti-echo 15, portfolio 6 | H-EVAL-1 (register, ID collision C-11); P-5/9/13, L-8/9/17/18, FS-1 |
| Frontier Scan / Daily | 3 (Lab; 08-31 executed by the MM) | — / 3 search logs | 9 ledgers, 22-track `TRACKS.md` | 08-31: 8, 09-01: 1, 09-02: 3 | GS-1…GS-7 |

\* the Waymo adjudication (LAB-RUN-005) was written by the Master Mind under the Lab charter. Library `banked` per
day: 08-28 18, 08-29 28, **08-30 128**, 08-31 34, 09-01 4, 09-02 3. Lab-authored package class: 20
literature/adjudication, 9 measurement (7 CPU, 2 RTX-4060).

**Daily trigger evidence** (LAB-RUN files two-probed; program reports two-probed, last one 2026-08-30 01:47 —
none since, until the fixed-clock cron was re-created on 2026-09-02 evening):

| day | evidence | verdict |
|---|---|---|
| 08-27 | only `2026-08-27-jepa-driving-survey` — "Author Master Mind" | no Lab run |
| 08-28 | LAB-RUN-002 (respawn; first spawn died on CLI restart) | ran |
| 08-29 | LAB-RUN-003 ("third spawn — first two died") | ran |
| 08-30 | 4 packages self-labelled "daily run 004" (commit `eaaf4f7`), **no LAB-RUN file** | ran, unsummarised |
| 08-31 | scheduled pass 08:45–09:26 + MM charter pass + PI-directed pass | ran ×3 |
| 09-01 | LAB-RUN-006 (4 pkgs, 4060) + 007 (frontier scan) | ran ×2 |
| 09-02 | 5 packages 01:53–02:32 (MM-spawned, commit `37ede7c`) + cron pass = LAB-RUN-008 frontier scan 12:47 | ran ×2 |

## C. Workflow conformance (§2/§3/§6, charter, AGENT_CHARTERS §1)

| clause | status | evidence |
|---|---|---|
| one agent, ~08:00, MM-triggered | matches | cron above; two-part dedup gate in the brief |
| sequential across FIVE fields | deviates | the brief says "FOUR work packages … Opponent Analysis / Benchmarks & Evals" merged |
| literature + banking | matches (strong) | 298 banked/14 d; `--verify` clean; KB entries cite `lib` keys; Band-C sources stamped PUBLISHED-BLOG/RELAYED |
| small dev-box GPU experiments incl. reverse-engineering | deviates | 2 GPU pkgs (rollout-depth 08-31, terminal-value 09-01); 0 reverse-engineering; AGENT_CHARTERS:52 "≥1 experiment per day with a SPEC" unmet 08-28/29/30 |
| §3 schema | partial | Lab SPEC.md 13/29 (all ≥ 08-31); raw/ 21/29; tests/ 5/29 (all 09-02); the brief prescribes only RESULT+search_log; MM packages 5/12 SPEC; FlyWheels use `PREREG_*.md` names |
| §6 spec-before-code, controls, n, tier | matches on ≥ 08-31 packages | see D |
| hypotheses registered in GOALS_AND_CLAIMS (AGENT_CHARTERS:60–62) | fails | 5 rows from run 001, 0 since; ~35 package-local `H-*` IDs; two probes for GS-/FS-/L-/P-1x/MM-1..4 → none |
| backlog contract (same-turn motions, MM ranks) | half | motions recorded in 006/007/008; ranking: 63 PROPOSED, 0 ranked; row 3 still "⛔PI" though the register (D-BENCH-PORT) records PI approval 08-29; "P-1…P-8" IDs used by two different blocks |
| charter breadth (22 tracks/day) | fails 2 days running | `TRACKS.md`: 19/22 (08-31), 10/22 (09-01), 11/22 (09-02); escalated as GS-7 |
| run summary per day, numbered | deviates | 08-30 missing; packages say "run 004" while LAB-RUN-004 = 08-31 |
| KB per field | matches since 08-30 | 0 entries 08-20…08-28 (audit S-1), 62 after; the router still says "175 entries across 7 areas" |
| brief as a versioned, rewritten handoff (§4.4) | deviates | static SKILL.md outside the repo; no copy found in `Project Steering/` or `.claude/` — ⚠️ ONE completed probe (Select-String); the repo-wide grep was killed before finishing, so this absence is single-probe until a second mechanism confirms it (correction 2026-09-02 23:55 Berlin) |
| skills (§8) | 3 of 7 exist; none for the Lab | `.claude/skills/`: BenchmarkCriteria, Review, ValidateAIDesign only |

## D. Content spot-check — the two most recent Lab packages (both 09-02, full schema + tests)

**`Architecture & Inference/Research/2026-09-02-scene-matched-action-criterion/`** — SPEC commits H-SMAS-1…3
with both outcomes (SPEC:28–34) and controls that must read known values (SPEC:41–48), n stated (SPEC:49), tier
T0-DIAGNOSTIC (SPEC:61). RESULT is MEASURED from banked JSONs (`raw/smas.json`), all four controls pass
(RESULT:116–123: no-information 0.0, synthetic factors recovered < 1e-12, h2/h4 floor, C0 identity), and it refuses
to oversell: "the fix does not rescue the k=60 arm" (RESULT:93–105); it declines to quote a CI rather than a wrong
one and proposes L-13 (RESULT:160–164); the inherited window-set confound is named (RESULT:165–168). COMMS
escalates E1–E3 with owners. Test `tests/test_smas.py` present. Gap: hypotheses are package-local, not registered.

**`Data Engineering/Research/2026-09-02-future-action-horizon-and-window-budget/`** — SPEC H-DH-1…4 both-outcome
(SPEC:26–31); success criterion 1 forbids reading T from a cache name (SPEC:35–37); scope stated up front
(SPEC:60–66). RESULT gives two independent lines for T ≈ 199 (banked `o4_n` arithmetic → 198.92 exactly; 24
episodes' bytes min 200 / max 207), refutes `train_v6_staged.py:2864-2890`'s inherited 120, quotes an
autocorrelation table with n = 4,444 pairs, recommends `--max-horizon 45` at −14.5 % windows, and marks F3 as
HYPOTHESIS. The MM registered the finding the same day (D-EPISODE-LENGTH) — the loop closed, but under the MM's ID,
not H-DH-3. Minor: no `raw/search_log.md`.

Both meet §6 (spec-driven, controls, printed n, tier, escalation with owners). The 09-02 frontier scan is of the
same grade (6 full-text reads, a counter-search flipped a verdict to CONTESTED, a fabricated "sub-50 ms" number
struck, its own false negative logged).

## E. Gaps and structural fixes, ranked

1. **Lab hypotheses never reach the register.** Extend `stack/scripts/lab_backlog_drift.py` to assert every `H-*`
   in a Lab `SPEC.md` exists in `GOALS_AND_CLAIMS.md` or is tagged `package-local`; run it in the trigger's END
   step. Fix the H-EVAL-1 collision (open since 08-29) and the duplicate "P-1…P-8" series with an ID allocator.
2. **The ranking half of the backlog contract is dead** (63 PROPOSED, 0 ranked; row 3 stale-⛔PI). A weekly MM
   skill `/TanitAD_LabRank` plus a test that fails when PROPOSED rows exceed 7 days or a ⛔PI row contradicts a
   register approval.
3. **The experiment half is under-delivered.** Put AGENT_CHARTERS:52 into the brief and into the LAB-RUN
   completeness table ("experiment with SPEC: y/n, GPU y/n"); make dedup-gate part (c) an experiment package;
   reserve a daily 4060 window. First candidates already free: GS-2 (0 GPU), then H-RANK-17.
4. **The brief lives outside the repo and is static.** Version it at `Project Steering/AgentSchedule/LAB_TRIGGER.md`,
   make the scheduled task read it, and rewrite the DEEP list per run (§4.4).
5. **Schema drift.** Brief → §3; a `lab_package_lint.py` (SPEC/RESULT/raw + evidence-class + tier line) at END.
6. **Breadth/depth tension (GS-7).** A ruling is needed; make `TRACKS.md` staleness a generated test (> 3 days
   unscanned fails).
7. **Banked-but-unread** (6/8 primaries "needed" on 09-02 already banked). Add `read_status` to `library.json`
   written by `kb_add`, and generate the FS-5 count in `TRACKS.md`.
8. **Run-summary integrity.** Generate `LAB-RUN-<NNN>.md` from the day's packages; gate part (c) checks
   yesterday's summary exists.
9. **Program reports stopped 08-30**; Lab activity was invisible except via commits. The fixed-clock cron was
   re-created on 2026-09-02 evening (07:57/12:57/17:57); pull `LAB_DIGEST.md` counts automatically.
10. Dedup gate part (a) accepts any dated dir, including MM/FlyWheel ones. Add an `owner:` line to every package
    and check it.

## F. PI decisions (defaults in brackets)

1. Four packages/day with Opp and B&E merged, or five as constituted [keep four; alternate the fifth field daily,
   stated in the brief].
2. GS-7 ruling [SCAN 8 tracks/day so all 22 land every 3 days; ≥ 3 Band-B DEEP; Band D when open; ≥ 2 full-text].
3. A protected Lab GPU window on the 4060 and a go for the first two Lab experiments (GS-2, H-RANK-17)
   [08:00–12:00 daily unless the MM books it].
4. Do Lab hypotheses enter `GOALS_AND_CLAIMS.md` [yes, `LAB-` prefix, status PROPOSED, generated-checked] or a
   Lab-local register?
5. Ranking cadence and owner [MM, weekly; overdue PROPOSED rows escalate in the program report].
6. Ratify charter v1/v2 (PI-1, PI-4 still open in the run files), confirm backlog row 3 is unparked, and supply a
   route to the UNECE GRVA primary (D-4, three HTTP 403s).
