"""RESULT.md: the PI's E1 ruling (2026-09-26) -- all three ON for refcv7, the speed ceiling
inference-only, REFCV7_REQUIRED_ON, tau banked, the Thor gate plan."""
import sys

P = sys.argv[1]
s = open(P, encoding="utf-8").read()


def edit(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch] {tag}: anchor found {n} times")
    s = s.replace(old, new)


# E1 row: replace its last cell (the decision) -- anchored on its opening words
i = s.index("| **E1** |")
j = s.index("| **E2** |")
row = s[i:j]
cells = row.rstrip("\n").split(" | ")
cells[-1] = ("**DECIDED — PI RULING 2026-09-26, verbatim:** *\"go with your recommendation for E1, assure that "
             "the three selection mechanism are on\"* (SPEC_REFCV7 §7 / A2, GOALS D-REFCV7-E1). Implemented: "
             "trainer defaults stay OFF (other arms bit-identical); the refcv7 argv carries `--graft-tac8-prior "
             "--graft-nav-compliance --nav-compliance-tau-rad <τ> --speed-ceiling-filter`; the ceiling is now "
             "INFERENCE-ONLY (`refc.py`, `not self.training` guard, class switch "
             "`AnchoredDiffusionDecoder.speed_ceiling_in_training = False`); G-DVB exposes "
             "`REFCV7_REQUIRED_ON` + `check_refcv7_required(..., tau_file=)`; τ is banked in "
             "`raw/nav_compliance_tau_train.json` (FULL train split). |")
s = s[:i] + " | ".join(cells) + "\n" + s[j:]

edit('''`selection_inputs_not_built`, and carries the facts by key (`selection_mechanisms_built`) for G-DVB. **Wiring is done;
switching ON is E1.**''', '''`selection_inputs_not_built`, and carries the facts by key (`selection_mechanisms_built`) for G-DVB. **Wiring is done;
switching ON was E1** — and the PI ruled all three ON for refcv7 (2026-09-26): the refcv7 argv passes the three flags
(defaults stay OFF), `check_refcv7_required` refuses a refcv7 build missing any of them, and the speed ceiling became
INFERENCE-ONLY (it no longer changes the training-time selection that feeds the LAW head).''', "s3")

edit('''drivort_levers_set(args) -> list[str]    # DrivoR-T flags set away from their literal defaults
Mismatch(lever, declared, built, read_from, why)
```''', '''drivort_levers_set(args) -> list[str]    # DrivoR-T flags set away from their literal defaults
REFCV7_REQUIRED_ON = ("graft_tac8_prior", "graft_nav_compliance", "speed_ceiling_filter")   # PI 2026-09-26
check_refcv7_required(model, args, *, tau_file=None) -> list[Mismatch]   # any OFF / unbuilt / tau != file
Mismatch(lever, declared, built, read_from, why)
```
A refcv7 launch calls `check(model, args, build_parser(), forbid_kinds=("drivort",))` AND
`check_refcv7_required(model, args, tau_file="…/raw/nav_compliance_tau_train.json")`; both must return `[]`. The
`--speed-ceiling-filter` entry also refuses a built decoder whose `speed_ceiling_in_training` is True.''', "s5 api")

edit('''| `taniteval/tests/test_refcv3_arm_equalize_as_trained.py` (3) |''',
     '''| `stack/tests/test_speed_ceiling_inference_only.py` (4) | on the REAL `RefCModel.forward`, a crafted per-row ceiling that excludes the unmasked winner: at EVAL the pick moves onto a compliant candidate; in TRAINING (same seed) the pick and the fan are unchanged and the LAW head's input is the UNMASKED selection | `speed_ceiling_in_training = True` (the pre-ruling behaviour) → the training assertion fails |
| `taniteval/tests/test_refcv3_arm_equalize_as_trained.py` (3) |''', "s7 new row")

edit('''| `stack/tests/test_declared_vs_built.py` (24) |''', '''| `stack/tests/test_declared_vs_built.py` (32) |''', "s7 count")
edit('''one selection term UNWIRED; the STATIC declaration restored |''',
     '''one selection term UNWIRED; the STATIC declaration restored; REFCV7: each of the three flags missing from argv (×3), refcv6's own build under a refcv7 argv (all three "not built"), a τ ≠ the banked file, and a decoder carrying the in-training ceiling switch |''', "s7 red arms")

edit('''6. The E1 decision is carried in argv (and, if nav compliance is ON, `--nav-compliance-tau-rad` equals the value in
   `raw/navc_tau_train.json`, derived on the FULL train split).''',
     '''6. PI E1 ruling: `check_refcv7_required(model, args, tau_file=raw/nav_compliance_tau_train.json) == []` — all three
   selection mechanisms ON in argv and BUILT, and `--nav-compliance-tau-rad` EQUAL to the banked τ (FULL train split).
7. The speed ceiling is inference-only: `core.decoder.speed_ceiling_in_training` is False (G-DVB refuses otherwise).''', "s8")

edit('''**MEASURED on a dev tree (tip + these files):** the five new files plus the closest neighbours''',
     '''**Second runner (Master Mind):** `raw/gate_plan.json` is the machine-readable plan for the same three stages on
Thor (base commit, per-stage ordered files, the 14-file overlay with md5s, environment-bound tests, expected
results).
**MEASURED on a dev tree (tip + these files, BEFORE the PI-ruling additions):** the five new files plus the closest neighbours''', "s7 note")

open(P, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
