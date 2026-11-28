"""S4c: detached post-build runner for the eval-141 matched rebuild. CPU only, polls with a script (zero model tokens).

Waits for `[build] DONE` in the build log (or a `_failures.json`), then runs, in order:
  1. check_eval141_rebuild.py  -> <stage>/check_report.json  (token ZZEVALCHECK-OKZZ / -FAILZZ)
  2. pull_encode_pooled.py --split eval --source local --src-dir <out> --src-geom 256 --dry-run  -> <stage>/dry_run_eval.out
Everything is appended to <stage>/post_build.log; the last line is ZZPOST-DONEZZ. Run:  python post_build_eval141.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = Path("C:/Users/Admin/tanitad-data/refav1-eval141-256rb_stage")
OUT = Path("C:/Users/Admin/tanitad-data/refav1-eval141-256rb")
LOG = STAGE / "post_build.log"
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH="E:/Projects/TanitAD/stack;E:/Projects/TanitAD/taniteval", PYTHONIOENCODING="utf-8",
           CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1")


def say(m):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%F %T')}] {m}\n")


def main():
    say("post-build runner started; waiting for '[build] DONE'")
    t0 = time.time()
    while time.time() - t0 < 6 * 3600:
        txt = (STAGE / "build.log").read_text(encoding="utf-8", errors="ignore") if (STAGE / "build.log").exists() else ""
        if "[build] DONE" in txt:
            break
        time.sleep(20)
    else:
        say("TIMEOUT waiting for build DONE")
        say("ZZPOST-DONEZZ")
        return 1
    say("build DONE seen; running check_eval141_rebuild.py")
    r = subprocess.run([PY, str(HERE / "check_eval141_rebuild.py"), "--workers", "4"], capture_output=True, text=True,
                       env=ENV, encoding="utf-8", errors="replace")
    (STAGE / "check_stdout.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-3000:], encoding="utf-8")
    say(f"check rc={r.returncode}; last stdout line: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else '<none>'}")
    say("running eval dry-run preflight")
    d = subprocess.run([PY, str(HERE / "pull_encode_pooled.py"), "--source", "local", "--src-dir", str(OUT).replace("\\", "/"),
                        "--src-geom", "256", "--split", "eval", "--grids", "4x10,8x20", "--dry-run",
                        "--out", str(STAGE / "_dry").replace("\\", "/")],
                       capture_output=True, text=True, env=ENV, encoding="utf-8", errors="replace")
    (STAGE / "dry_run_eval.out").write_text(d.stdout + "\n--- stderr ---\n" + d.stderr[-3000:], encoding="utf-8")
    say(f"dry-run rc={d.returncode}; lines: " + " | ".join(l for l in d.stdout.splitlines() if l.startswith(("ZZ", "[plan]", "[source]", "[parity]"))))
    # bank the evidence into the package (raw/s4c) so the next agent only has to read it
    import json, shutil
    raw = HERE.parent / "raw" / "s4c"
    raw.mkdir(parents=True, exist_ok=True)
    for src, dst in ((STAGE / "check_report.json", "check_report_eval141.json"), (STAGE / "build.log", "build_eval141.log"),
                     (STAGE / "dry_run_eval.out", "dry_run_eval_post.out"), (STAGE / "stage_report.json", "stage_report_eval141.json"),
                     (STAGE / "stage.log", "stage_eval141.log"), (OUT / "_manifest_sha256.jsonl", "manifest_sha256_eval141.jsonl")):
        if src.exists():
            shutil.copy2(src, raw / dst)
    try:
        rep = json.loads((STAGE / "check_report.json").read_text(encoding="utf-8"))
        keep = {k: v for k, v in rep.items() if k != "per_clip"}
        (raw / "post_summary.json").write_text(json.dumps({"check": keep, "check_rc": r.returncode, "dry_rc": d.returncode,
                                                           "dry_lines": [l for l in d.stdout.splitlines() if l.startswith(("ZZ", "[plan]", "[source]", "[parity]"))]},
                                                          indent=1), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        say(f"summary failed: {e!r}")
    say("evidence banked to " + str(raw))
    say("ZZPOST-DONEZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
