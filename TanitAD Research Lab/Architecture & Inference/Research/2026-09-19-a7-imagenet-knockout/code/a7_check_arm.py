"""A7 per-arm VALIDITY gate -- PREREG_REFCV6_DEVBOX_PREPARATION.md A7.2 / A7.3 / A7.8.

Validity, never a verdict: an arm that fails here is VOID (A7.6 table, last row) and the
panel stops rather than spending GPU on arms that cannot be read.

Every expectation is written as a LITERAL (CLAUDE.md: "a cross-check must be derived
independently of the value it checks"), never as an expression over the run's output.

Exit 0 = VALID · 1 = INVALID (reasons listed) · 2 = INCONCLUSIVE (an artifact could not
be read -- never reported as valid). Writes ``<arm_dir>/a7_arm_check.json`` either way;
⛔ the admissible evidence that this gate ran is that JSON, not the exit code.

Usage: python a7_check_arm.py <arm_dir> --pretrained {0,1} --seed N
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

N_WINDOWS_EVAL = 1000          # --eval-batches 500 x --batch 2
N_RECAL = 256                  # A7.2 item 4
RECAL_SEED = 0
N_BN_RESNET34 = 36             # MEASURED 2026-09-19 (A7.1): stem + 16 blocks x 2 + 3 downsample
STEPS = 2000
REBUILD_REL = 1e-4             # A7.8 item 5


def _rebuild_eval_traj(rows, batch=2):
    """A7.8 item 3: the loss's own rule -- slot-weighted per consecutive pair, mean over pairs."""
    vals = []
    for i in range(0, len(rows), batch):
        grp = rows[i:i + batch]
        w = sum(r["slot_valid_frac"] for r in grp)
        vals.append(sum(r["traj"] * r["slot_valid_frac"] for r in grp) / w if w > 0 else 0.0)
    return sum(vals) / len(vals)


def check(arm_dir: Path, pretrained: int, seed: int) -> dict:
    run = arm_dir / "run"
    bad, inconclusive, facts = [], [], {}

    def need(ok, msg):
        if not ok:
            bad.append(msg)

    def load_json(p):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:                      # noqa: BLE001 (recorded)
            inconclusive.append("%s unreadable: %s: %s" % (p.name, type(exc).__name__, exc))
            return None

    def load_jsonl(p):
        try:
            return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        except Exception as exc:                      # noqa: BLE001 (recorded)
            inconclusive.append("%s unreadable: %s: %s" % (p.name, type(exc).__name__, exc))
            return None

    summ = load_json(run / "summary.json")
    if summ is not None:
        need(summ.get("done") is True, "summary.json done is not True")
        need(summ.get("step") == STEPS, "summary.json step %r != 2000" % summ.get("step"))

    cfg = load_json(run / "config.json")
    if cfg is not None:
        argv = cfg.get("argv") or []

        def val(flag):
            return argv[argv.index(flag) + 1] if flag in argv else None
        need(val("--seed") == str(seed), "--seed %r != %d" % (val("--seed"), seed))
        want, other = (("--trunk-pretrained", "--no-trunk-pretrained") if pretrained
                       else ("--no-trunk-pretrained", "--trunk-pretrained"))
        need(want in argv and other not in argv, "pretrained flag: want %s only" % want)
        need(val("--trunk-bn-recalib") == str(N_RECAL), "--trunk-bn-recalib != 256")
        need(val("--trunk-bn-recalib-seed") == str(RECAL_SEED), "--trunk-bn-recalib-seed != 0")
        need(val("--eval-every") == "2000", "--eval-every != 2000")
        need(val("--eval-batches") == "500", "--eval-batches != 500")
        need(val("--steps") == "2000" and val("--batch") == "2", "--steps/--batch not 2000/2")
        need(val("--trunk-name") == "resnet34.a1_in1k", "--trunk-name not resnet34.a1_in1k")
        need("--trunk-frozen-bn" in argv and val("--trunk-chunk-ckpt") == "1",
             "memory levers missing")
        st = cfg.get("trunk_bn_recalib")
        need(isinstance(st, dict), "config.json trunk_bn_recalib is not a stamp")
        if isinstance(st, dict):
            facts["stamp"] = {k: st.get(k) for k in (
                "n_windows", "seed", "windows_sha12", "n_batches", "n_images", "n_bn",
                "stats_sha12", "changed", "var_median_before", "var_median_after",
                "chunk_bypassed")}
            need(st.get("changed") is True, "recalibration did not change the statistics")
            need(st.get("n_windows") == N_RECAL, "n_windows %r != 256" % st.get("n_windows"))
            need(st.get("seed") == RECAL_SEED, "recal seed %r != 0" % st.get("seed"))
            need(st.get("n_bn") == N_BN_RESNET34, "n_bn %r != 36" % st.get("n_bn"))
            need(st.get("chunk_bypassed") is True, "chunking was not bypassed")
            need(isinstance(st.get("windows_sha12"), str) and len(st["windows_sha12"]) == 12,
                 "windows_sha12 missing")
            if not pretrained:
                # A7.1, the confound itself: a random-init trunk must START at the identity
                need(st.get("var_median_before") == 1.0,
                     "RND arm did not start at the identity (var_median_before %r)"
                     % st.get("var_median_before"))
            fin = st.get("final") or {}
            need(fin.get("freeze_held") is True, "config final.freeze_held is not True")

    bnr = load_json(run / "bn_recalib.json")
    if bnr is not None:
        facts["bn_recalib"] = bnr
        need(bnr.get("freeze_held") is True, "⛔ the frozen statistics MOVED -- arm VOID")
        need(bnr.get("staleness_error") is None, "staleness failed: %s" % bnr.get("staleness_error"))
        need(isinstance(bnr.get("staleness"), dict), "no staleness record")
    need((run / "bn_recalib_stats.pt").exists(), "bn_recalib_stats.pt missing")

    met = load_jsonl(run / "metrics.jsonl")
    if met is not None:
        ev = [r for r in met if "eval_traj" in r]
        need(len(ev) == 1, "expected exactly ONE eval row (--eval-every 2000), got %d" % len(ev))
        if ev:
            e = ev[-1]
            facts["eval_row"] = e
            need(e.get("step") == STEPS, "eval row step %r != 2000" % e.get("step"))
            need(e.get("eval_windows") == N_WINDOWS_EVAL, "eval_windows %r != 1000"
                 % e.get("eval_windows"))
            need(isinstance(e.get("eval_traj"), (int, float)) and math.isfinite(e["eval_traj"]),
                 "eval_traj not finite")
        errs = [r for r in met if "eval_error" in r or r.get("eval_window_dump_error")]
        need(not errs, "an eval pass recorded an error: %s" % errs[:1])

    rows = load_jsonl(arm_dir / "eval_windows.jsonl")
    if rows is not None:
        need(len(rows) == N_WINDOWS_EVAL, "dump rows %d != 1000" % len(rows))
        need(all(r.get("step") == STEPS for r in rows), "dump rows not all from step 2000")
        need(all("traj" in r and "slot_valid_frac" in r and "episode_id" in r for r in rows),
             "dump rows lack traj / slot_valid_frac / episode_id")
        if met is not None and ev and len(rows) == N_WINDOWS_EVAL:
            rb = _rebuild_eval_traj(rows)
            agg = ev[-1]["eval_traj"]
            rel = abs(rb - agg) / max(abs(agg), 1e-12)
            facts["dump_rebuild"] = {"rebuilt": rb, "aggregate": agg, "rel": rel,
                                     "plain_row_mean": sum(r["traj"] for r in rows) / len(rows),
                                     "n_partial_future": sum(r["slot_valid_frac"] < 1.0
                                                             for r in rows)}
            need(rel <= REBUILD_REL, "A7.8(5): dump rows rebuild eval_traj only to rel %.3g "
                 "(> 1e-4) -- the dump is VOID for the bootstrap" % rel)

    status = "INCONCLUSIVE" if inconclusive else ("VALID" if not bad else "INVALID")
    return {"arm": arm_dir.name, "status": status, "failures": bad,
            "inconclusive": inconclusive, "facts": facts,
            "expected": {"pretrained": pretrained, "seed": seed}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--pretrained", type=int, choices=(0, 1), required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    d = Path(a.arm_dir)
    rep = check(d, a.pretrained, a.seed)
    (d / "a7_arm_check.json").write_text(json.dumps(rep, indent=1, default=str),
                                         encoding="utf-8")
    print("ZZA7CHECK-%s-%sZZ" % (rep["arm"], rep["status"]))
    for m in rep["failures"] + rep["inconclusive"]:
        print("   -", m)
    return {"VALID": 0, "INVALID": 1}.get(rep["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
