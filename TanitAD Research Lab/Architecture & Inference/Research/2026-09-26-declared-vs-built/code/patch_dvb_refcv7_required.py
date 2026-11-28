"""PI RULING 2026-09-26 (SPEC_REFCV7 §7 / A2, D-REFCV7-E1): all three refcv6 selection mechanisms
are ON for refcv7. G-DVB exposes the frozen list and a helper that FAILS when any is OFF in argv or
unbuilt in the model; and it refuses a decoder carrying the test-only in-training ceiling switch."""
import sys

P = sys.argv[1]
s = open(P, encoding="utf-8").read()


def edit(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch] {tag}: anchor found {n} times")
    s = s.replace(old, new)


edit('''__all__ = ["Mismatch", "Lever", "REGISTRY", "check", "refuse_on_mismatch", "coverage",
           "register", "KINDS", "DRIVORT_DEFAULTS", "drivort_levers_set"]
''', '''__all__ = ["Mismatch", "Lever", "REGISTRY", "check", "refuse_on_mismatch", "coverage",
           "register", "KINDS", "DRIVORT_DEFAULTS", "drivort_levers_set",
           "REFCV7_REQUIRED_ON", "check_refcv7_required"]
''', "all")

edit('''def _c_selection(dest):
    key = {v: k for k, v in SELECTION_KEYS.items()}[dest]

    def chk(m, a):
        want = bool(_a(a, dest, False))
        return _eq(dest, want, _sel_built(m)[key], f"selection mechanism `{key}` (built)",
                   "a selection mechanism the ranked score does not have (D-REFCV6-CONFIG-BUILD)"
                   if want else "a live selection mechanism no flag asked for")
    return chk
''', '''def _c_selection(dest):
    key = {v: k for k, v in SELECTION_KEYS.items()}[dest]

    def chk(m, a):
        want = bool(_a(a, dest, False))
        out = _eq(dest, want, _sel_built(m)[key], f"selection mechanism `{key}` (built)",
                  "a selection mechanism the ranked score does not have (D-REFCV6-CONFIG-BUILD)"
                  if want else "a live selection mechanism no flag asked for")
        if dest == "speed_ceiling_filter":
            # PI 2026-09-26 A2: INFERENCE-ONLY. The class switch that restores the pre-ruling
            # in-training mask exists for the regression test alone; a built decoder carrying it
            # is refused.
            out += _eq(dest, False, bool(getattr(_dec(m), "speed_ceiling_in_training", False)),
                       "core.decoder.speed_ceiling_in_training",
                       "the ceiling would mask the TRAINING selection (the LAW input)")
        return out
    return chk
''', "ceiling in-training check")

edit('''def drivort_levers_set(args) -> list[str]:''', '''#: ⛔⛔ PI RULING 2026-09-26, verbatim: *"go with your recommendation for E1, assure that the three
#: selection mechanism are on"* (SPEC_REFCV7 §7 / A2, GOALS D-REFCV7-E1). The trainer's defaults
#: stay OFF (other arms bit-identical); a refcv7 launch must carry all three, built.
REFCV7_REQUIRED_ON: tuple[str, ...] = ("graft_tac8_prior", "graft_nav_compliance",
                                       "speed_ceiling_filter")


def check_refcv7_required(model, args, *, tau_file: str | None = None) -> list[Mismatch]:
    """-> a Mismatch for every REFCV7_REQUIRED_ON lever that is OFF in argv or UNBUILT in the
    model ([] == all three on and built). With ``tau_file`` (the banked
    `raw/nav_compliance_tau_train.json`), argv's `--nav-compliance-tau-rad` must equal its `tau`.
    """
    out: list[Mismatch] = []
    built = _sel_built(model)
    for dest in REFCV7_REQUIRED_ON:
        key = {v: k for k, v in SELECTION_KEYS.items()}[dest]
        if not bool(_a(args, dest, False)):
            out.append(Mismatch(_flag(dest), "ON (PI 2026-09-26, D-REFCV7-E1)", "OFF in argv",
                                "argv", "refcv7 requires all three selection mechanisms"))
        if not built[key]:
            out.append(Mismatch(_flag(dest), "BUILT", "not built",
                                f"selection mechanism `{key}` (built)",
                                "refcv7 requires all three selection mechanisms"))
    if tau_file is not None:
        import json as _json
        want = float(_json.load(open(tau_file, encoding="utf-8"))["tau"])
        got = float(_a(args, "nav_compliance_tau_rad", 0.0) or 0.0)
        if abs(got - want) > 1e-12:
            out.append(Mismatch("--nav-compliance-tau-rad", want, got, f"argv vs {tau_file}",
                                "tau must be the value derived on the TRAIN split"))
    return out


def drivort_levers_set(args) -> list[str]:''', "refcv7 required")

open(P, "w", encoding="utf-8", newline="\n").write(s)
print("declared_vs_built: REFCV7_REQUIRED_ON + check_refcv7_required + in-training ceiling check")
