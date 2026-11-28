"""Known-answer test: P6's run/metrics.jsonl holds two NUL-padded lines (PREREG_P6 Amendment (d) item 4).
MEASURED 2026-11-06 on the pod's file (19,219 lines): file line 12,660 = 634 NULs + an intact {"event": "resume", "step": 4136, ...};
line 13,263 = 91 NULs + an intact {"event": "resume", "step": 4315, ...}.  The NULs overwrote the torn tail of the append BEFORE each resume.
What this pins (stdlib only; run:  python -I test_metrics_nul.py --mods <dir holding p6_overlap.py and restage_helpers.py>):
  1. restage_helpers.last_ckpt (the restage resume-point check) reads the right step through the NUL lines;
  2. restage_helpers' guard reads labeller events after an offset: unaffected (the NUL lines precede every offset it is ever given);
  3. p6_overlap.step_times (the overlap guard's per-step clock) skips them without error;
  4. CONTROL: the strict reader pattern of diag_onpolicy_train.py:305 (json.loads on every non-blank line) RAISES -- str.strip() does not remove NUL;
  5. a startswith("{") reader silently DROPS the two resume events; only a NUL-stripping reader counts them.  -> an interruption count must strip NULs.
Prints ZZNUL PASS n/n or ZZNUL FAIL <name>."""
import argparse, json, os, sys, tempfile, importlib.util

ap = argparse.ArgumentParser(); ap.add_argument("--mods", required=True); a = ap.parse_args()
def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(a.mods, name + ".py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
RH, OV = load("restage_helpers"), load("p6_overlap")

def row(step, t): return {"step": step, "epoch": 10, "traj_L1": 1.0, "elapsed_s": t}
recs = [b'{"event": "start"}', json.dumps(row(4134, 100.0)).encode(), json.dumps({"event": "ckpt", "step": 4135}).encode(),
        json.dumps(row(4135, 158.0)).encode()[:40],                                    # the torn append ...
        ]
lines = [recs[0], recs[1], recs[2],
         b"\x00" * 634 + json.dumps({"event": "resume", "step": 4136}).encode(),       # ... overwritten by NULs, then the resume event (file line 12,660's shape)
         json.dumps(row(4136, 10.0)).encode(), json.dumps(row(4137, 68.0)).encode(), json.dumps({"event": "ckpt", "step": 4138}).encode(),
         b"\x00" * 91 + json.dumps({"event": "resume", "step": 4315}).encode(),        # line 13,263's shape
         json.dumps(row(4315, 5.0)).encode(), json.dumps(row(4316, 63.5)).encode(), json.dumps({"event": "ckpt", "step": 6266}).encode()]
d = tempfile.mkdtemp(); p = os.path.join(d, "metrics.jsonl"); open(p, "wb").write(b"\n".join(lines) + b"\n")

res = []
def t(name, ok): res.append((name, bool(ok)))
ck, tail = RH.last_ckpt(p)
t("last_ckpt_reads_6266_through_NUL_lines", ck == 6266 and tail == ("ckpt", 6266))
dts, last = OV.step_times(p, last=5)
t("step_times_no_error_and_no_cross_resume_pair", last == 4316 and all(abs(x - 58.0) < 1e-9 or abs(x - 58.5) < 1e-9 for x in dts) and len(dts) >= 2)
try:
    [json.loads(l) for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]; strict_raised = False
except json.JSONDecodeError:
    strict_raised = True
t("CONTROL_strict_reader_raises_on_NUL_line", strict_raised)
brace = [json.loads(l) for l in open(p, encoding="utf-8").read().splitlines() if l.strip().startswith("{")]
t("brace_reader_drops_both_resume_events", sum(r.get("event") == "resume" for r in brace) == 0)
nul = [json.loads(l.replace("\x00", "")) for l in open(p, encoding="utf-8").read().splitlines() if l.replace("\x00", "").strip()]
t("NUL_stripping_reader_counts_both_resumes", [r["step"] for r in nul if r.get("event") == "resume"] == [4136, 4315])
bad = [n for n, ok in res if not ok]
print(("ZZNUL PASS %d/%d" % (len(res), len(res))) if not bad else "ZZNUL FAIL " + " ".join(bad))
for n, ok in res: print("  %-55s %s" % (n, "ok" if ok else "FAIL"))
sys.exit(1 if bad else 0)
