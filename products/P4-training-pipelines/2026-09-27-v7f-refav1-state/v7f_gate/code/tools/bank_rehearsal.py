"""Bank a dev-box CLI rehearsal run into the deliverable: the evidence files, the CLI log, the launch argv, and a
SUMMARY of the token (the token itself is never banked -- it stays in the agent's scratch copy)."""
import json
import shutil
import sys
from pathlib import Path

W = Path(r"C:/Users/Admin/v7f_gate")
DEL = Path(r"D:/Projects/TanitAD/products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_gate/raw")

for s in ("st", "sw"):
    run = W / "gate_runs" / f"m_{s}_devbox_nav"
    dst = DEL / f"merge_rehearsal_{s}"
    dst.mkdir(parents=True, exist_ok=True)
    for ev in sorted((run / "evidence").glob("G-*.json")):
        shutil.copyfile(ev, dst / ev.name)
    shutil.copyfile(W / "raw" / f"cli_rehearsal_merge_{s}_nav.log", dst / "cli.log")
    shutil.copyfile(W / "argv" / f"v7f_{s}_merge_compliant.argv.json", dst / "launch_argv.json")
    toks = sorted(run.glob("*_7f7f7f7f7f7f.json"))
    if len(toks) != 1:
        sys.exit(f"{s}: expected ONE token in {run}, found {[t.name for t in toks]}")
    tok = json.loads(toks[0].read_text(encoding="utf-8"))
    summary = {"_read": "a SUMMARY of a dev-box REHEARSAL token (the token itself is NOT banked: tokens stay in "
                        "the agent's scratch copy). Signed with a throwaway test key, fake commit 7f7f..., never "
                        "verifiable. Re-run 2026-09-27 on the merge WITH the nav mapping (nav kept end to end).",
               "verdict": tok.get("verdict"), "profile": tok.get("profile"),
               "checks_in_token": {c: v.get("status") for c, v in (tok.get("checks") or {}).items()},
               "token_reasons": tok.get("reasons"), "rehearsal": tok.get("rehearsal")}
    (dst / "rehearsal_verdicts.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(s, summary["verdict"], {c: v.get("status") for c, v in
                                  ((summary["rehearsal"] or {}).get("checks") or {}).items()})
