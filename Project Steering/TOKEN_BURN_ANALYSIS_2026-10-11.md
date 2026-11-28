# Why the weekly token budget burns in 1–2 days — analysis and fixes (2026-10-11)

**Evidence class.** MEASURED, from every Claude Code transcript under `C:/Users/Admin/.claude/projects/` with a
call after 2026-09-21: 35,293 model calls, after de-duplication. The `D--` and `E--` folders hold the same
sessions twice, because D: is a `subst` of E:. Each call's `usage` record gives its fresh input, cache-write,
cache-read and output tokens.

**Weighting.** The budget cost is weighted by the API price ratios: fresh input 1, cache write 1.25, cache read
0.1, output 5, with Sonnet at 0.6× and Haiku at 0.2× of Opus. The absolute scale of the subscription meter is
not public, so only the SHARES below are quotable. The scripts are in the session scratchpad
(`usage_scan.py`, `usage_final.py`) and are proposed for `tools/token_meter/` (P8).

## 1. What happened

| window | weighted units | calls | main session share |
|---|---|---|---|
| Sep 21–27 | 1,746 M | 23,595 | 34 % |
| Sep 28–Oct 3 | 189 M | 2,782 | 25 % (mostly locked out) |
| **Oct 4–5** | **704 M** | 8,913 | 15 % (limit hit on the morning of Oct 5) |

- The last weekly budget was gone after about 1.5 days. Oct 4 alone used about 4–5 days' worth at a sustainable
  pace.
- The same thing happened the week before: Sep 26–27 used 1.68 B, then almost nothing ran until Oct 3.

## 2. Where it went (all calls, de-duplicated)

| driver | share of cost | mechanism |
|---|---|---|
| **Re-reading huge contexts** | **61 %** cache read | Every call re-reads the agent's whole context. Calls above 300k context are 92 % of the cost; calls below 100k are 0.4 %. Agents ran to 870–970k context. |
| … of which context **above 200k** | **37 %** | The part a 200k context cap would remove. |
| **Cache expiry while waiting** | **29 %** cache write after an idle gap ≥ 5 min | Subagents' prompt cache lives **5 minutes**. Every time an agent waited 5–60 min on a Thor job and woke up, it re-wrote its full 400–900k context at 1.25×. That happened 1,545 times. The R1 agent spent **89 % of its cost** on these re-writes: 126 re-writes of about 260k–650k tokens in 375 calls. |
| other cache writes | 5 % | Normal growth. |
| output (what we actually produce) | **4 %** | |
| Subagents vs main sessions | 72 % / 28 % | |
| Opus vs Sonnet/Haiku | **96 %** Opus | Even though the PI ruled on 10-04 that Opus is for complex tasks only. |
| CLAUDE.md (22k tokens, 88 KB) re-read on every call | 2.8 % | It also inflates every agent's starting context. |

**Top consumers on Oct 4–5:**

| agent | share | calls | avg context | what it was mostly doing |
|---|---|---|---|---|
| R1 agent | 19.6 % | 375 | 658k | waiting on Thor lock slots |
| WP-B | 18.2 % | 1,399 | 537k | 38 h of life |
| WP-D | 10.6 % | 536 | 511k | waiting on P-BOX jobs |
| this Master Mind session | 7.9 % | 837 | 537k | relaying agent messages |
| the parallel E: Master Mind session | 7.5 % | 826 | 493k | |

**One-line diagnosis.** We run **long-lived Opus agents that wait for GPU jobs with 500–900k contexts**. Each
wait longer than 5 minutes re-buys the whole context, and each step re-reads it. Real work (output) is 4 % of
the bill.

## 3. Fixes, ranked by measured effect

| # | fix | removes (est.) | how |
|---|---|---|---|
| **P1** | **No agent waits on a long job while holding a big context.** | most of the **29 %** | When an agent hands work to Thor that takes more than ~4 minutes, it writes a ≤ 2k-token `HANDOFF.md` (state, artifact paths, next step) and ENDS. The job's completion (background task notification or a file watcher) launches a FRESH agent that starts from the handoff, not from the old transcript. Also: never `SendMessage`-resume a big agent after a long gap. That resume is exactly the 1.25× full re-write. |
| **P2** | **Cap agent context at ~200k; one agent = one scoped step.** | most of the **37 %** | Briefs name ONE deliverable (e.g. "land L5", "run the S3 read"), not a whole work package's lifecycle. At ~200k context the agent hands off and ends. WP-B's 38-hour, 1,399-call life becomes 6–10 short agents. |
| **P3** | **Sonnet by default; Opus only for diagnosis and design.** | ~40 % of the agent-side cost that moves | Landing, staging, monitoring, Thor ops, test gates, scoring runs: Sonnet (or Haiku for pure mechanics). Opus: root-cause diagnosis, pre-registration design, architecture. Enforce it by setting `model` on every Agent call (already a PI rule; it was not applied). |
| **P4** | **One Master Mind session at a time, compacted at ~300k.** | ~15–25 % of the main share | On Sep 26–27 three main sessions ran in parallel at ~550k each. Every relayed agent message costs the main session a full re-read (this session sat at 760k). Run `/compact` at ~300k, or start a fresh Master Mind each morning from `LOOP_STATE.md`. |
| **P5** | **Agents report less, the main session reads less.** | part of P4 | Hand-backs ≤ 300 words plus a file path. The full report lives in the package and the Master Mind opens it only when deciding. |
| **P6** | **Trim CLAUDE.md. APPLIED: canonical D: file 25.4k → 13.5k tokens.** | ~1.5 % directly, plus a smaller start for every agent | Moved verbatim "Traps preflight" (10.9k tokens) and "Git hygiene" (2.5k) into `Project Steering/TRAPS.md`, plus a `tanitad-traps` skill that loads on demand. Kept the PI's binding rules, Rule Zero and the invariants, plus a 34-line trap index. Further cuts would touch PI-binding rule text, which is the PI's call. |
| **P7** | **Amend "≥ 5 parallel streams" to "≥ 5 streams of work, run by short-lived scoped agents".** | enables P1/P2 | Five live Opus agents at 500k+ each, all waiting on one Thor GPU, is the measured burn pattern. Streams can be parked as handoff files while their job runs: a parked stream costs zero tokens. This is the PI's rule, so the PI must approve the change. |
| **P8** | **Daily budget meter.** | guard | Commit `tools/token_meter/` (the two scripts above). The Master Mind reads yesterday's spend every morning against a daily budget of 1/6 of the weekly limit. If a day exceeds it, switch everything except diagnosis to Sonnet and park the waiting streams. |

**Expected combined effect (ESTIMATED).**
- P1 + P2 + P3 cut the weighted burn per unit of work by roughly **3–5×**. P1 alone removes most of the
  29 % idle-expiry share; P2 removes most of the 37 % large-context share; P3 roughly halves what is left on
  moved work.
- At the Oct 4 pace of work, the week would last **5–7 days** instead of 1.5.
- To be verified with the P8 meter after the first two days.

**Applied 2026-10-11:**
- the TOKEN BUDGET rule in CLAUDE.md (P1–P5, P7);
- preamble rule 6 in `AGENT_OPERATING_STANDARD.md`;
- `Project Steering/TRAPS.md` and the `tanitad-traps` skill;
- `tools/token_meter/token_meter.py`.

Its first run (Oct 3–11) gives these shares for the limit week alone: **idle-expiry cache writes 42.9 %**, reads above 200k context 28.4 %, output 2.8 %, subagents 83.6 %, Opus 87 %.

## 4. What this session itself shows

This Master Mind session reached 760k tokens. Every agent message it relayed re-read that context.
⇒ **This session should be compacted or replaced by a fresh one right after this analysis** (P4).
