# ERRATUM (MM, 2026-11-16): BEHAVIOUR_SCORECARD.md section 5 proxies are T1, not T0

**Scope.** `BEHAVIOUR_SCORECARD.md` stays frozen (sha256 `fac6c7f1...`, landed d641517e) and is not edited. This note corrects one tier label. No threshold, definition, reference value or decision rule changes.

**Text affected.**
- Section 0, line 10: "Open-loop proxies (section 5) are **T0**".
- Section 5: its title "(T0)" and the table header "proxy (T0)".

**Correction.** P-1 to P-5 come from NAVSIM navtest plans:
- each plan is made from t0 perception and the command;
- no recorded future action is fed to the model.

By `Project Steering/EVAL_DOCTRINE.md`:
- T0 means "teacher-forced: predictor consumes recorded future actions";
- T1 means "self-action OPEN loop ... perception context fixed at t0".

So every section 5 proxy is **T1 (self-action open loop)**. It is still open loop, so it may never be quoted as driving performance. That restriction is unchanged.

**Consistent with the instruments.** `PK/eval/command_adherence.py:30` and `:79` already stamp the navtest plan as T1.

**Unchanged.**
- The closed-loop stamp for sections 2 to 4 ("T2-class, closed-loop sim").
- The open question in that section about naming the tier.
- H-PROXY-1, which stays a HYPOTHESIS.

**Source.** Flagged by the v8 flagship session on 2026-11-16 and verified by the MM against both documents on tip 86c5c7ec.

**Root-cause class:** a tier mislabel. "Open loop" was read as "T0", when the doctrine splits open loop into T0 (recorded actions) and T1 (own actions).
