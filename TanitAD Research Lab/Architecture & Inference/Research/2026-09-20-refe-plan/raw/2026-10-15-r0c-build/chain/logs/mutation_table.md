| # | mutation (a copy of the chain / helper with ONE property broken) | static catcher (check_chain) | behaviour catcher (r0c_behaviour group: violation) |
|---|---|---|---|
| 1 | the R0c argv line is deleted | `R0C_NO_ARGV` | `argv: R0C_NO_ARGV` |
| 2 | R0c is not registered | `R0C_NOT_REGISTERED` | `arms: R0C_REFUSED_AT_PRECHECK` |
| 3 | R0c is registered under a typo (R0C) | `R0C_NOT_REGISTERED` | `arms: MISSPELLED_ARM_ACCEPTED` |
| 4 | R0c is judged under the original profile | `R0C_NO_PROFILE` | `profile: R0C_WRONG_PROFILE` |
| 5 | R0c is judged under the RL-S1b profile | `R0C_NO_PROFILE` | `profile: R0C_WRONG_PROFILE` |
| 6 | R0c has no profile line (falls through to refe_rls1) | `R0C_NO_PROFILE` | `profile: R0C_WRONG_PROFILE` |
| 7 | C1 is judged under refe_rls1b whenever R0c is listed | - | `profile: R0C_WRONG_PROFILE` |
| 8 | --epo-ref armA is dropped from R0c's argv | `R0C_NO_EPO_REF_ARMA` | `argv: R0C_ARGV_NOT_THE_JSON` |
| 9 | --epo-ref armA is dropped (the real gate says so too) | - | `argv: R0C_REFUSED_BY_THE_REAL_GATE` |
| 10 | --epo-ref base is named instead of armA | `R0C_NO_EPO_REF_ARMA` | `argv: R0C_ARGV_NOT_THE_JSON` |
| 11 | a second lever rides along (--epo-lam 0.25) | `R0C_EXTRA_LEVER:--epo-lam` | `argv: R0C_NOT_R0_PLUS_ONE_FLAG` |
| 12 | R0c runs with seed 1 | `R0C_WRONG_SEED` | `argv: R0C_ARGV_NOT_THE_JSON` |
| 13 | R0 is no longer the banked token's argv | - | `argv: R0_ARGV_NOT_THE_BANKED_TOKEN` |
| 14 | R0c is read against C0's dumps when they are there (the own-A override is deleted) | `R0C_NOT_READ_AGAINST_ITS_OWN_A` | `control: R0C_CONTROL` |
| 15 | the own-A override runs BEFORE the generic assignment and is overwritten | `R0C_NOT_READ_AGAINST_ITS_OWN_A` | `control: R0C_CONTROL` |
| 16 | C0 is not named as R0c's second control (static only: ${ARM:1:1} happens to give C0 as well) | `R0C_SECOND_CONTROL_NOT_NAMED` | - |
| 17 | R0c falls under the RL-S1b control guard (refused against its own arm A) | `RLS1B_GUARD_EDITED` | `control: R0C_CONTROL` |
| 18 | R0b / R1b lose their guard (a missing control is silently replaced by the arm's own A) | `RLS1B_GUARD_EDITED` | `control: CONTROL_DRIFT` |
| 19 | no loud marker when C0's banked dumps are missing | `R0C_SILENT_WITHOUT_SECOND_CONTROL` | - |
| 20 | the end-of-loop kill refusal skips R0c | `KILL_RULE_ARM_GATED` | `kill: R0C_KILL_DOES_NOT_REFUSE` |
| 21 | the restart-after-kill refusal skips R0c | `KILL_RULE_ARM_GATED` | `kill: R0C_KILL_DOES_NOT_REFUSE` |
| 22 | the gate's suite files are not required before the queue wait | `R0C_SUITE_FILE_NOT_REQUIRED` | - |
| 23 | arm names are matched as substrings | - | `arms: MISSPELLED_ARM_ACCEPTED` |
| 24 | the precheck no longer validates the arm names | `ARM_NAMES_NOT_REFUSED_TWICE` | `arms: R0C_NO_PRECHECK` |

QUEUE
| R0c is read against C0's dumps whenever they are on the pod | "order", "mixed" |
| R0c is judged under the original profile | "order" |
| R0c's argv loses --epo-ref armA | "order" |
| the loud marker for a missing second control is gone | "no_c0" |
| the second control (C0) is never read | "order" |
| a killed R0c does not stop the chain | "kill", "kmon" |
| a killed R0c re-trains on a re-run | "rerun_after_kill" |
| a misspelled arm reaches the preflight | "unknown_arm" |
