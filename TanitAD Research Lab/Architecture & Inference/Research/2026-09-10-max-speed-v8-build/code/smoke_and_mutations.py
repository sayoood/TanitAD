"""The 20-step smoke, asserted ON THE ARTIFACTS, plus two deliberate regressions.

⛔⛔ WHY THIS SCRIPT ASSERTS ON FILES AND NEVER ON AN EXIT CODE. MEASURED TWICE
on 2026-09-07: a wrapper read ``$?`` through a pipe and reported **exit 0** for a
tool that had exited 1, and a 25-minute ``timeout`` killed a pre-launch gate with
**zero output and no JSON written** while its wrapper printed ``GATE_EXIT=0``.
⇒ *"The admissible evidence that this gate did not run is the MISSING JSON, never
the exit code."* Every check below opens a file and reads its content.

⛔⛔ AND WHY A GREEN SMOKE IS NOT ENOUGH. The refusal this build must not weaken
was proven by mutation on 2026-09-10's census: with the guard at
``refc_v3_train.py:1596`` disabled, the trainer printed
``0/2 clips carry a ceiling, 0/2 windows fed`` **and trained anyway** -- the
`tac_goal` zero-gradient failure exactly, an arm whose ``config.json`` would read
``max_speed_input: true`` while nothing reached the model. ⇒ A smoke that only
proves "it ran" cannot tell those two runs apart. This one asserts the census
line's OWN numbers, so ``0/N windows fed`` fails here even when the run exits 0.

THE FOUR ARMS
  A  ON      -- real v8 labels; must FEED windows, loss finite and NOT constant,
                and config.json must carry the provenance stamp.
  B  OFF     -- the single-lever twin; must carry NO stamp and NO ceiling.
  M1 MUTANT  -- the v8 block stripped from the labels; the loader's refusal at
                :1596 must fire. ⛔ Proves the guard still has teeth WITHOUT
                touching the guard.
  M2 MUTANT  -- `speed_max_derivation` deleted from the config dict; the new
                `_assert_speed_max_stamp` must refuse to start. ⛔ Proves the
                stamp is a precondition and not decoration.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
TRAINER = REPO / "stack" / "scripts" / "refc_v3_train.py"
V8_DIR = (REPO / "TanitAD Research Lab" / "Data Engineering" / "Implementation"
          / "incoming" / "2026-09-10-v8-speed-max-label-release" / "raw")

#: ⛔ LITERALS, never an expression over the code under test. A check written as
#: `tok in T.SPEED_MAX_DERIVATION` passes for whatever the constant happens to
#: say, which is the "green forever" failure.
REQUIRED_STAMP_TOKENS = ("oracle", "ego-future", "SPEED_BAND.v_hi_ms",
                         "[t0+2 s, +6 s]")

CENSUS_RE = re.compile(
    r"max_speed_input \((\w+)\): (\d+)/(\d+) clips carry a ceiling, "
    r"(\d+)/(\d+) windows fed, (\d+) over")


def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = f"{REPO / 'stack'}{os.pathsep}{REPO / 'taniteval'}"
    e["PYTHONIOENCODING"] = "utf-8"
    e["OMP_NUM_THREADS"] = "6"          # ⛔ torch spawns ~113 threads/process
    return e


def run_arm(out: Path, labels: Path | None, max_speed: bool, steps: int,
            trainer: Path = TRAINER) -> dict:
    """Run one arm. Returns the raw evidence; JUDGES NOTHING here."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cmd = [sys.executable, str(trainer), "--arm", "hier", "--out", str(out),
           "--smoke", "--device", "cpu", "--synth-episodes", "4",
           "--steps", str(steps), "--log-every", "1"]
    if labels is not None:
        cmd += ["--v7-labels", str(labels)]
    if max_speed:
        cmd += ["--max-speed-input"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                       encoding="utf-8", errors="replace", timeout=1800)
    log = (p.stdout or "") + (p.stderr or "")
    (out.parent / f"{out.name}.log").write_text(log, encoding="utf-8")
    return {"argv": cmd[2:], "rc": p.returncode, "log": log, "out": out}


def read_artifacts(out: Path) -> dict:
    """⛔ EVERY judgement below reads a FILE. Absence is reported as absence."""
    cfg_p, met_p, sum_p = (out / "config.json", out / "metrics.jsonl",
                           out / "summary.json")
    cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else None
    losses = []
    if met_p.exists():
        for ln in met_p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                v = json.loads(ln).get("loss")
            except json.JSONDecodeError:
                continue
            if isinstance(v, (int, float)):
                losses.append(float(v))
    return {
        "config_json_exists": cfg_p.exists(),
        "metrics_jsonl_exists": met_p.exists(),
        "summary_json_exists": sum_p.exists(),
        "config": cfg, "losses": losses,
    }


def judge_on(ev: dict, art: dict) -> dict:
    """Arm A. ⭐ The load-bearing check is `windows_fed > 0`, not `rc == 0`."""
    m = CENSUS_RE.search(ev["log"])
    cfg = art["config"] or {}
    stamp = cfg.get("speed_max_derivation")
    ls = art["losses"]
    finite = bool(ls) and all(math.isfinite(x) for x in ls)
    checks = {
        "rc_is_0": ev["rc"] == 0,
        "config_json_written": art["config_json_exists"],
        "metrics_jsonl_written": art["metrics_jsonl_exists"],
        "summary_json_written": art["summary_json_exists"],
        "census_line_present": m is not None,
        # ⛔⛔ THE ONE THAT SEPARATES THIS RUN FROM THE `0/2 windows fed`
        # MUTANT. A run can exit 0, write every artifact, and have fed nothing.
        "windows_fed_GT_0": bool(m) and int(m.group(4)) > 0,
        "every_window_fed": bool(m) and m.group(4) == m.group(5),
        "clips_with_ceiling_GT_0": bool(m) and int(m.group(2)) > 0,
        "loss_rows_present": len(ls) > 0,
        "loss_all_finite": finite,
        # ⛔ a CONSTANT loss is the dead-channel signature (`tac_goal` took
        # grad_abs_sum 0 for 40,284 steps and its loss never moved).
        "loss_is_NOT_constant": len(set(ls)) > 1,
        "config_max_speed_input_true": cfg.get("max_speed_input") is True,
        "config_stamp_present": isinstance(stamp, str) and bool(stamp.strip()),
        "config_stamp_declares_all_tokens":
            isinstance(stamp, str)
            and all(t in stamp for t in REQUIRED_STAMP_TOKENS),
        "config_window_ceiling_frac_GT_0":
            float((((cfg.get("max_speed_stats") or {}).get("train")) or {})
                  .get("window_ceiling_frac", 0.0) or 0.0) > 0.0,
        "config_provenance_is_ego_future":
            (((cfg.get("max_speed_stats") or {}).get("train")) or {})
            .get("provenance", "").startswith("ego-future"),
    }
    return {"checks": checks, "PASS": all(checks.values()),
            "census": (m.groups() if m else None),
            "n_loss_rows": len(ls), "n_distinct_losses": len(set(ls)),
            "loss_min": min(ls) if ls else None,
            "loss_max": max(ls) if ls else None,
            "stamp": stamp}


def judge_off(ev: dict, art: dict) -> dict:
    """Arm B. The OFF twin must be clean in BOTH directions."""
    cfg = art["config"] or {}
    checks = {
        "rc_is_0": ev["rc"] == 0,
        "config_json_written": art["config_json_exists"],
        "config_max_speed_input_false": cfg.get("max_speed_input") is False,
        # ⛔ the mirror failure: a control stamped as max-speed-conditioned.
        "config_stamp_is_None": cfg.get("speed_max_derivation") is None,
        "config_max_speed_stats_is_None": cfg.get("max_speed_stats") is None,
        "no_census_line": CENSUS_RE.search(ev["log"]) is None,
        "loss_rows_present": len(art["losses"]) > 0,
    }
    return {"checks": checks, "PASS": all(checks.values())}


def judge_refusal(ev: dict, art: dict, must_contain: tuple[str, ...]) -> dict:
    """M1 / M2. A refusal is proven by a NON-ZERO rc, a NAMED message, and --
    the part that matters -- NO artifact left behind."""
    checks = {
        "rc_is_NONZERO": ev["rc"] != 0,
        "no_config_json_written": not art["config_json_exists"],
        "no_summary_json_written": not art["summary_json_exists"],
    }
    for tok in must_contain:
        checks[f"message_says_{tok.replace(' ', '_')}"] = tok in ev["log"]
    tail = [ln for ln in ev["log"].splitlines() if ln.strip()][-4:]
    return {"checks": checks, "PASS": all(checks.values()), "tail": tail}


def strip_block(src: Path, dst: Path) -> int:
    """M1's corpus: the v8 blob with `speed_max_input` REMOVED from every
    record -- i.e. a v7.2-shaped blob. ⛔ The GUARD is not touched."""
    n = 0
    with gzip.open(src, "rt", encoding="utf-8") as fh, \
            gzip.open(dst, "wt", encoding="utf-8", newline="\n") as out:
        for ln in fh:
            if not ln.strip():
                continue
            r = json.loads(ln)
            n += int(r.pop("speed_max_input", None) is not None)
            out.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return n


def mutate_trainer_drop_stamp(tmp: Path) -> Path:
    """M2's trainer: a COPY with the `speed_max_derivation` config line
    deleted, so `_assert_speed_max_stamp` meets a config that lacks it.

    ⛔ The original file is never edited -- a `sed -i` on a live source tree is
    how a running shell ends up executing garbage from mid-line.

    ⚠️⚠️ THE COPY MUST LIVE BESIDE THE REAL TRAINER, NOT IN A TEMP DIR.
    MEASURED 2026-09-10, and it is exactly the failure this whole file exists to
    catch: the first version of this mutation wrote the copy to ``%TEMP%``, and
    the mutant died at ``import refb_labels`` (``refc_v3_train.py:74`` resolves
    its siblings relative to its OWN path) -- **before reaching the guard**. It
    exited non-zero and wrote no ``config.json``, so a refusal check looking
    only at "rc != 0 and no artifact" would have scored it **PASS** and
    certified a guard that had never run. Only the message assertion separated
    "refused" from "crashed on the way to the refusal". ⇒ the caller deletes
    this file in a ``finally``; it is never staged.
    """
    text = TRAINER.read_text(encoding="utf-8")
    needle = ('        "speed_max_derivation": (SPEED_MAX_DERIVATION\n'
              '                                 if getattr(args, '
              '"max_speed_input", False)\n'
              '                                 else None),\n')
    if needle not in text:
        raise SystemExit(
            "[M2] the config stamp line is not where this mutation expects "
            "it. Refusing to run a mutation that may be mutating nothing -- a "
            "regression arm that does not regress is green forever.")
    dst = TRAINER.parent / "_MUTANT_no_stamp_refc_v3_train.py"
    dst.write_text(text.replace(needle, ""), encoding="utf-8")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--work", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    v8_train = V8_DIR / "s2_labels_v8.0_train.jsonl.gz"
    if not v8_train.exists():
        raise SystemExit(f"[smoke] the v8 blob is missing: {v8_train}")

    rep: dict = {"instrument": "smoke_and_mutations.py", "steps": a.steps,
                 "trainer": str(TRAINER), "labels": str(v8_train), "arms": {}}

    # ---- A: ON -----------------------------------------------------------
    ev = run_arm(work / "A_on", v8_train, True, a.steps)
    rep["arms"]["A_on"] = {"argv": ev["argv"],
                           **judge_on(ev, read_artifacts(work / "A_on"))}

    # ---- B: OFF (the single-lever twin) ----------------------------------
    ev = run_arm(work / "B_off", v8_train, False, a.steps)
    rep["arms"]["B_off"] = {"argv": ev["argv"],
                            **judge_off(ev, read_artifacts(work / "B_off"))}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # ---- M1: the block removed -> the LOADER's refusal must fire ------
        stripped = tmp / "s2_labels_NO_BLOCK.jsonl.gz"
        n_removed = strip_block(v8_train, stripped)
        ev = run_arm(work / "M1_no_block", stripped, True, a.steps)
        rep["arms"]["M1_no_block"] = {
            "argv": ev["argv"], "n_blocks_removed": n_removed,
            **judge_refusal(ev, read_artifacts(work / "M1_no_block"),
                            ("v8 addition", "NOT ONE"))}

        # ---- M2: the stamp deleted -> the NEW guard must refuse -----------
        mutant = mutate_trainer_drop_stamp(tmp)
        try:
            ev = run_arm(work / "M2_no_stamp", v8_train, True, a.steps,
                         trainer=mutant)
            rep["arms"]["M2_no_stamp"] = {
                "argv": ev["argv"], "mutant_trainer": str(mutant),
                **judge_refusal(ev, read_artifacts(work / "M2_no_stamp"),
                                ("speed_max_derivation", "Refusing to start"))}
        finally:
            # ⛔ the mutant lived in stack/scripts/ so its sibling imports
            # resolved; it must not survive the run and must never be staged.
            mutant.unlink(missing_ok=True)
            rep["arms"].setdefault("M2_no_stamp", {})[
                "mutant_removed"] = not mutant.exists()

    rep["OVERALL_PASS"] = all(v["PASS"] for v in rep["arms"].values())
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(txt, encoding="utf-8")
    for name, v in rep["arms"].items():
        bad = [k for k, ok in v["checks"].items() if not ok]
        print(f"{name:14s} {'PASS' if v['PASS'] else 'FAIL'}"
              + (f"  failed: {bad}" if bad else ""))
    print(f"\nOVERALL: {'PASS' if rep['OVERALL_PASS'] else 'FAIL'}")
    return 0 if rep["OVERALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
