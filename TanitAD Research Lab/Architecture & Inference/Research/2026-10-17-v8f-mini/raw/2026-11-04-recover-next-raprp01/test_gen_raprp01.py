"""Mutation tests of gen_raprp01_results.py: every addendum 3 row (M, I, N, J, K) is reached from a mutated copy of a real read JSON,
and an inconsistent reader JSON is refused. Run: python -I test_gen_raprp01.py (CPU, no network)."""
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gen_raprp01_results", HERE / "gen_raprp01_results.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
RAPR = json.load(open(G.RAW / "RECOVER_READ_RAPR.json", encoding="utf-8"))
FAILS = []


def guards_pass(d, seeds=("s0", "s1")):
    for s in seeds:
        g = d["guards"]["per_seed"][s]
        g["ade6_within_guard"], g["non_crawl_separated_worse"] = True, []
    d["guards"]["PASS"] = all(G.guard_ok(d, s) for s in G.SS)
    return d


def composite(d, dom, status):
    d["composite"]["DOMINANT"], d["composite"]["status"] = dom, status
    d["composite"]["counts"] = status == "DOMINANT, COUNTED"
    return d


def case(name, d, want):
    with tempfile.TemporaryDirectory() as td:
        jp, out = Path(td) / "arm.json", Path(td) / "out.md"
        json.dump(d, open(jp, "w", encoding="utf-8"))
        try:
            got = G.main(["gen", "--json", str(jp), str(out)])
            txt = out.read_text(encoding="utf-8")
            ok = got == want and G.ROWS[want][0] in txt and "DRY RUN" in txt
        except AssertionError as e:
            got, ok = f"AssertionError({e})", want == "refuse"
    print(("PASS " if ok else "FAIL ") + name, "->", got)
    if not ok:
        FAILS.append(name)


m = copy.deepcopy
case("RAPR as read (guards fail at both seeds, NOT DOMINANT) -> K", m(RAPR), "K")
case("guards pass at both, NOT DOMINANT -> J", guards_pass(m(RAPR)), "J")
case("guards pass at s0 only -> K", guards_pass(m(RAPR), ("s0",)), "K")
case("DOMINANT, COUNTED with PARTIAL -> I", composite(m(RAPR), True, "DOMINANT, COUNTED"), "I")
case("DOMINANT, COUNTED with guards failing -> I (I precedes K)", composite(m(RAPR), True, "DOMINANT, COUNTED"), "I")
case("DOMINANT, RIG ARTEFACT -> N", composite(m(RAPR), True, "DOMINANT, RIG ARTEFACT"), "N")
case("DOMINANT, UNREAD -> N", composite(m(RAPR), True, "DOMINANT, UNREAD (frame-freeze controls failed or reading not computable)"), "N")
d = m(RAPR); d["controls_failed"] = ["KA1_bit_identity_rerun_of_R_s0"]
case("a failed control -> M", d, "M")
d = m(RAPR); d["refused"] = "a pin mismatch"
case("refused read -> M", d, "M")
d = m(RAPR); d["composite"] = {"status": None, "note": "NOT COMPUTED"}
case("composite not computed -> M", d, "M")
d = m(RAPR); d["composite"]["counts"] = True
case("counts True with status NOT DOMINANT is refused", d, "refuse")
d = composite(m(RAPR), True, "DOMINANT, COUNTED"); d["frame_freeze"]["reading"] = "ARTEFACT"
case("counted with an ARTEFACT reading is refused", d, "refuse")
d = guards_pass(m(RAPR)); d["guards"]["PASS"] = False
case("per-seed guards disagreeing with guards.PASS is refused", d, "refuse")
print("RESULT:", "PASS" if not FAILS else "FAIL", f"({len(FAILS)} failed)")
sys.exit(1 if FAILS else 0)
