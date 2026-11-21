# Cover note of the coordinator to the lever sheet (2026-11-19)

Sheet: `LEVER_SHEET_V8F_PROOF_2026-11-19.md` (sha256 prefix 65c0ea6ad6862caa; compiled by a build agent from register rows 52 and 56 to 91 at commit 85f85844 and from the lever folders; reviewed by the coordinator; the agent read the rows in part, so a number that matters for a decision is checked against its row before use). Decisions D3 and D4 of `dagland/COORD_V8F_2026-11-19.md`.

## 1. State after the rows (MEASURED today, T0, not yet in a row; lands with hold 65)

* Level term: both weights of note ADD37_N6 lose as written (0.001 keeps the response and misses the level at one base, ratio 1.198 against 1.15; 0.01 keeps the level and loses the response, slope -0.40). Two interior weights are training under note ADD37_N7. The level term stays NOT READY.
* Control p5b at the new bases: the low consistency weight reads 0.4649 and 0.4877 on the model's own plans (bar 0.5), so the miss of the sheet's block 2.2 holds on four bases (row 91).

## 2. What the coordinator proposes for the integrated tree (plan step 2)

| lever | in the integrated tree | default in the integrated candidate | why |
|---|---|---|---|
| supervised lead head | yes, behind its flags | on (`--lead-sup on --w-lead-sup 1.0`) | the rule's presence and gap reads come from it; `SUP-NO-COST` |
| consistency term | yes, behind its flag | on at 0.0001 | no cost in three families on four bases; it is the only confirmed lever on the imagination; no driving metric is expected to move from it alone, and the sheet says so |
| distance rule with the trained gap head | yes, as inference-time modules | read both ways (off and on) on every eval | it lowers the unsafe rate and has a price; the candidate must show both columns |
| level term, planner on imagined rollouts, finer grid | no | off | not confirmed (level term), no gain over extrapolation (planner), not separated (grid; card ADD43 running) |

Launch gate (D3 c): the four lead-head flags and the two consistency flags need declared-against-built entries (the owner's work; the lead-head patch carries its own trainer-side check). The rule and the gap head act at inference: the coordinator asks for one equality check in the eval path instead (the eval loader builds the same rule constants, the same head file by sha256 and the same threshold the argv declares), with a deliberate-regression arm, as for any gate check.

Not established by the sheet and owed by this study before the port: counted mutation lists for the lead head, the consistency term and the rule (their hand-off files name the mutations but give no list file), and pins files for the rule and the gap-head tools. The fork produces these with the hold after 65.

## 3. Proposal for one common floor (D4), for the integrated candidate

* Evaluation set: the 141 evaluation episodes of the v8F-mini study (2,399 windows; 642 lead windows of 56 episodes for the distance cells), paired episode-cluster bootstrap, B 2,000, on every arm of both sessions; four families, STRATEGIC computed where the arm has a strategic layer.
* Floor per cell: the larger of the two seed differences of the control arm at the candidate's own step count and corpus, `max(|C_s1 - C_s0|, |C_rep - C_s0|)`; the floors of the 4,572-clip studies are printed beside and decide nothing at 10k.
* A lever effect is quotable when it is separated at every seed and its mean is at or above the floor; the seed spread of the paired difference is printed beside (the R2 form of amendment UM-A3), and a result below it is called narrow.
* The Master Mind says where its strat1 and data-scale reads differ from this (cells, estimator, windows) before the candidate card is written; the card and its cost go to the PI.
