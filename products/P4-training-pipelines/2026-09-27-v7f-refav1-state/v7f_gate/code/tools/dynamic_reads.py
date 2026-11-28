"""DYNAMIC dest-read trace for train_v6_staged.py: which parsed flags are actually READ (attribute access
on ANY argparse.Namespace -- the parsed args, and the eval loader's Namespace(**config['args'])) on the
paths a CPU rehearsal exercises: the gate's S-W and S-T jobs (preflight, train() with the synthetic
corpus, the resume run, the uninterrupted reference, the build capture, the eval rebuild) and the
trainer's own --dry-run for both stages. `vars(a)` (the config.json stamp) is NOT a read.
Writes raw/dynamic_reads.json."""
import argparse
import json
import sys
from pathlib import Path

TREE = Path(r"C:/Users/Admin/v7f_gate/tree_m")
sys.path.insert(0, str(TREE / "stack"))
sys.path.insert(0, str(TREE / "stack" / "scripts"))
sys.path.insert(0, str(TREE / "stack" / "tests"))
import launch_gate as LG  # noqa: E402
from test_launch_gate_v7f import ST_ARGV, SW_ARGV, _ctx  # noqa: E402

sys.path.insert(0, str(TREE / "stack" / "scripts"))
import train_v6_staged as _T  # noqa: E402
_ap = _T.build_parser()
_ap.add_argument("--i-know-this-is-the-control-arm", action="store_true", dest="control_arm_ack")
DESTS = {x.dest for x in _ap._actions if x.dest != "help"}
READS: dict[str, set] = {}
CALLERS: dict[str, set] = {}
PHASE = {"now": None}
_orig = argparse.Namespace.__getattribute__


GATE_FILES = ("launch_gate.py", "declared_vs_built_v6.py", "dynamic_reads.py")


def _rec(self, name):
    if name in DESTS and PHASE["now"] is not None:
        caller = sys._getframe(1).f_code.co_filename
        # the GATE's own reads (its registry readers, its argv rules) are not the trainer's
        if not caller.endswith(GATE_FILES):
            READS.setdefault(name, set()).add(PHASE["now"])
            fr = sys._getframe(1)
            CALLERS.setdefault(name, set()).add(f"{Path(fr.f_code.co_filename).name}:{fr.f_code.co_name}")
    return _orig(self, name)


argparse.Namespace.__getattribute__ = _rec
out_root = Path(r"C:/Users/Admin/v7f_gate/gate_runs/dynamic_trace_merge")
import shutil  # noqa: E402
shutil.rmtree(out_root, ignore_errors=True)
phases = []
DVB_MISMATCHES: dict = {}
for label, argv in (("S-W", SW_ARGV), ("S-T", ST_ARGV)):
    ctx = _ctx(out_root / label, argv)
    PHASE["now"] = f"{label}:smoke"
    ev = LG.job_smoke_v6(ctx, ["G-LIVE", "G-CKPT"])
    PHASE["now"] = f"{label}:model"
    ev.update(LG.job_model_v6(ctx, ["G-HYG", "G-DVB", "G-EVAL"]))
    phases.append({label: {c: e["status"] for c, e in ev.items()}})
    DVB_MISMATCHES[label] = list((ev.get("G-DVB") or {}).get("details", {}).get("mismatches") or [])
    # the trainer's own --dry-run on the same rehearsal argv
    T = LG.load_trainer_v6(ctx)
    tiny, _ = LG.v6_rehearsal_argv(ctx.prof, argv, out_root / label)
    # nav is KEPT (the merge maps it; the rehearsal serves synthetic nav records via v6_corpus_seam)
    tiny = LG.set_flag(LG.set_flag(tiny, "--init-from", None), "--prev-gate", None)
    tiny = LG.set_flag(LG.set_flag(tiny, "--out", [str(out_root / label / "dry")]), "--dry-run", [])
    PHASE["now"] = f"{label}:dry_run"
    with LG.v6_corpus_seam(ctx, {}):
        try:
            rc = T.main(tiny)
        except SystemExit as e:
            rc = f"SystemExit: {e}"
        except Exception as e:  # noqa: BLE001 -- a crash is a RESULT here (the reads before it count)
            rc = f"RAISED {type(e).__name__}: {str(e)[:160]}"
    phases.append({f"{label}:dry_run": rc})
PHASE["now"] = None
argparse.Namespace.__getattribute__ = _orig
res = {"phases": phases, "n_dests": len(DESTS), "dvb_mismatches": DVB_MISMATCHES,
       "read": {d: sorted(READS.get(d, ())) for d in sorted(DESTS)},
       "never_read": sorted(d for d in DESTS if d not in READS),
       "callers": {d: sorted(CALLERS.get(d, ())) for d in sorted(DESTS)}}
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamic_consumers import add_consumers  # noqa: E402
res = add_consumers(res)
Path(r"C:/Users/Admin/v7f_gate/raw/dynamic_reads_merge.json").write_text(json.dumps(res, indent=1),
                                                                    encoding="utf-8")
print("phases", phases)
print("never read on the exercised paths:", len(res["never_read"]))
print(res["never_read"])
