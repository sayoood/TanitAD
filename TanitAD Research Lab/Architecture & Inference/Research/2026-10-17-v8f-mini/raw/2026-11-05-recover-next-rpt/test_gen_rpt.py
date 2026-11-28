"""Mutation tests of gen_rpt_results.py: every addendum 4 row (M, I, N, J, K1, K2) is reached from a mutated copy of a real read JSON
(RAPR-P01's), an inconsistent reader JSON is refused, and the per-tick table record is read (and refused when absent in a production
read). Run: python -I test_gen_rpt.py (CPU, no network)."""
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gen_rpt_results", HERE / "gen_rpt_results.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
BASE = json.load(open(G.RAW / "RECOVER_READ_RAPRP01.json", encoding="utf-8"))
R_SHARE = {s: float(json.load(open(G.RREF, encoding="utf-8"))["primary"]["share_closed"][s]) for s in G.SS}
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


def share(d, s0, s1):
    d["primary"]["share_closed"] = {"s0": s0, "s1": s1}
    return d


def fake_runs(td, dropped=0, now_shift=0.0, drop_rpt_s1=False):
    """A runs dir with RPT and R gate selfchecks per seed: R's table record, RPT's = R's + per-tick record."""
    runs = Path(td) / "runs"
    for s in G.SS:
        rt = {"n": 1000, "counts": {"records_ok": 1000, "kept": 1000, "dropped_logged_row_unreal": 0},
              **{k: 0.1 * (i + 1) for i, k in enumerate(G.NOW_KEYS)}}
        pt = copy.deepcopy(rt)
        pt["n"] = pt["counts"]["kept"] = 1000 - dropped
        pt["counts"]["dropped_per_tick_row_unreal"] = dropped
        pt["per_tick"] = {"ticks": 10, "mean_abs_dv_t": [0.1] * 10, "mean_abs_da_t": [0.1] * 10, "mean_abs_dom_t": [0.1] * 10}
        pt["mean_dv"] += now_shift
        for arm, t in (("R", rt), ("RPT", pt)):
            if arm == "RPT" and s == "s1" and drop_rpt_s1:
                continue
            p = runs / f"M2b_{s}_recft_{arm}"
            p.mkdir(parents=True)
            json.dump({"levers": {"recover_table": t}}, open(p / "gate_selfcheck.json", "w", encoding="utf-8"))
    return runs


def case(name, d, want, runs=None, must=()):
    with tempfile.TemporaryDirectory() as td:
        jp, out = Path(td) / "arm.json", Path(td) / "out.md"
        json.dump(d, open(jp, "w", encoding="utf-8"))
        try:
            got = G.main(["gen", "--json", str(jp), str(out)] + (["--runs", str(runs)] if runs else []))
            txt = out.read_text(encoding="utf-8")
            ok = got == want and G.ROWS[want][0] in txt and "DRY RUN" in txt and all(m in txt for m in must)
        except AssertionError as e:
            got, ok = f"AssertionError({e})", want == "refuse"
    print(("PASS " if ok else "FAIL ") + name, "->", got)
    if not ok:
        FAILS.append(name)


m = copy.deepcopy
above = {s: R_SHARE[s] + 0.05 for s in G.SS}
case("RAPR-P01 as read (guards fail, s0 share 0.2147 < R's 0.2198) -> K2", m(BASE), "K2")
case("guards fail, share above R's at both seeds -> K1", share(m(BASE), above["s0"], above["s1"]), "K1")
case("guards fail, share above R's at s0 only -> K2", share(m(BASE), above["s0"], R_SHARE["s1"] - 0.01), "K2")
case("guards fail, share EQUAL to R's at both seeds (does not exceed) -> K2", share(m(BASE), R_SHARE["s0"], R_SHARE["s1"]), "K2")
case("guards pass at both, NOT DOMINANT -> J", guards_pass(m(BASE)), "J")
case("guards pass at both, share below R's -> J (J precedes K)", share(guards_pass(m(BASE)), 0.01, 0.01), "J")
case("guards pass at s0 only, share above R's at both -> K1", share(guards_pass(m(BASE), ("s0",)), above["s0"], above["s1"]), "K1")
case("DOMINANT, COUNTED -> I", composite(m(BASE), True, "DOMINANT, COUNTED"), "I")
case("DOMINANT, COUNTED with guards failing and share below R's -> I (I precedes K)", share(composite(m(BASE), True, "DOMINANT, COUNTED"), 0.0, 0.0), "I")
case("DOMINANT, RIG ARTEFACT -> N", composite(m(BASE), True, "DOMINANT, RIG ARTEFACT"), "N")
case("DOMINANT, UNREAD -> N", composite(m(BASE), True, "DOMINANT, UNREAD (frame-freeze controls failed or reading not computable)"), "N")
d = m(BASE); d["controls_failed"] = ["KA1_bit_identity_rerun_of_R_s0"]
case("a failed control -> M", d, "M")
d = m(BASE); d["refused"] = "a pin mismatch"
case("refused read -> M", d, "M")
d = m(BASE); d["composite"] = {"status": None, "note": "NOT COMPUTED"}
case("composite not computed -> M", d, "M")
d = m(BASE); d["composite"]["counts"] = True
case("counts True with status NOT DOMINANT is refused", d, "refuse")
d = composite(m(BASE), True, "DOMINANT, COUNTED"); d["frame_freeze"]["reading"] = "ARTEFACT"
case("counted with an ARTEFACT reading is refused", d, "refuse")
d = guards_pass(m(BASE)); d["guards"]["PASS"] = False
case("per-seed guards disagreeing with guards.PASS is refused", d, "refuse")
with tempfile.TemporaryDirectory() as td:
    case("per-tick record read: no drop, NOW columns equal R's", m(BASE), "K2", fake_runs(td), ("| s0 | 1000 | 1000 | 0 | 10 | 1000 | all five |",))
with tempfile.TemporaryDirectory() as td:
    case("per-tick record read: 7 dropped, mean_dv differs", m(BASE), "K2", fake_runs(td, dropped=7, now_shift=1e-3), ("| s1 | 993 | 993 | 7 | 10 | 1000 | mean_dv differ |",))
with tempfile.TemporaryDirectory() as td:
    runs = fake_runs(td, drop_rpt_s1=True)
    case("per-tick record absent at s1 in a non-production read is reported as absent", m(BASE), "K2", runs, ("| s1 | absent (",))
    try:
        G.pertick_counts(runs, strict=True)
        ok = False
    except AssertionError:
        ok = True
    print(("PASS " if ok else "FAIL ") + "per-tick record absent in a production read is refused")
    if not ok:
        FAILS.append("strict pertick")
print("RESULT:", "PASS" if not FAILS else "FAIL", f"({len(FAILS)} failed)")
sys.exit(1 if FAILS else 0)
