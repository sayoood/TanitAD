"""A/B: the scorer's EGO VIEW (`score_proposals.EGO_VIEW`) must write rows BYTE-IDENTICAL to the full path.

Each arm is a FRESH `build_scorer_targets.py` process on the SAME per-frame bank:
  REF      REFE_SCORER_EGO_VIEW=0                          the full-object path (what is banked)
  EGO      REFE_SCORER_EGO_VIEW=1                          the lever
  MUT      REFE_SCORER_EGO_VIEW=1 + _MUTATE=agent1         the view on the WRONG agent: MUST differ
  AUG_REF / AUG_EGO   (with --aug-file)                    the augmented path: rank 1, per-row routes
  REF2     REFE_SCORER_EGO_VIEW=0 again                     the DETERMINISM control: REF vs REF2
PASS = REF == REF2 == EGO in CONTENT and in BYTES with rows > 0, the EGO arm's `view` count > 0 AND
`fallback` 0 (a run that fell back on every prefix IS the reference path and passes vacuously --
MEASURED on the first run: 1,980/1,980 prefixes fell back on `agent_nearest_indices`), REF's view
count 0, and MUT != REF in CONTENT. Frames come from several logs, one per shard file.
⛔ EVERY ARM RUNS WITH PYTHONHASHSEED=0. MEASURED on the first run: REF and an all-fallback EGO had
0/198 rows differing in content yet different sha256 -- `score_proposal_rollout` builds each row's
`targets` from a SET of key strings, so key ORDER follows the per-process string-hash seed. Bytes
are only comparable under a fixed seed; content is compared regardless.

  python refe/diag_scorer_ego_view.py --bank-glob "<bank>/r0_s*/targets_rank0.jsonl" \
      --logs 3 --frames-per-log 6 --work <scratch dir> [--aug-file <dir>/targets_aug.jsonl]
Prints ZZEGOVIEW_EXACT <rows> <frames> <speedup>x ZZ, or ZZEGOVIEW_FAIL <reason> ZZ; exit 0 / 1.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_rows(pattern: str, n_logs: int, per_log: int) -> list[str]:
    """The first `per_log` complete rows of ONE log per shard file, until `n_logs` logs."""
    rows, used = [], set()
    for fp in sorted(glob.glob(pattern)):
        by_log: dict = {}
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.endswith("\n") or not line.strip():
                    continue                     # a torn tail is not a row
                lg = json.loads(line)["log_name"]
                if lg in used:
                    continue
                by_log.setdefault(lg, []).append(line)
                if len(by_log[lg]) >= per_log:
                    rows.extend(by_log[lg])
                    used.add(lg)
                    break
        if len(used) >= n_logs:
            break
    return rows


def canonical(data: bytes) -> str:
    """sha256 of the rows' CONTENT: every row re-serialised with sorted keys (NaN kept as NaN), so
    key order cannot matter and every value still does."""
    h = hashlib.sha256()
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            h.update(json.dumps(json.loads(line), sort_keys=True).encode("utf-8") + b"\n")
    return h.hexdigest()


def run_arm(name: str, env_over: dict, bank: str, work: str, rank: int, pf_file: str | None):
    out = os.path.join(work, name)
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, os.path.join(HERE, "build_scorer_targets.py"), "--source", "navtrain",
           "--perframe-bank", bank, "--out", out, "--rank", str(rank), "--frame-stride", "1"]
    if pf_file:
        cmd += ["--perframe-file", pf_file]
    env = dict(os.environ)
    env.pop("REFE_SCORER_EGO_VIEW_MUTATE", None)
    env["PYTHONHASHSEED"] = "0"
    env.update(env_over)
    p = subprocess.run(cmd, cwd=HERE, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    open(os.path.join(work, name + ".log"), "w", encoding="utf-8").write(p.stdout)
    fn = "scorer_targets.jsonl" if rank == 0 else f"scorer_targets_rank{rank}.jsonl"
    path = os.path.join(out, fn)
    data = open(path, "rb").read() if os.path.exists(path) else b""
    sp = os.path.join(out, "scorer_targets_stats.json")
    st = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    r = {"rc": p.returncode, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
         "rows": data.count(b"\n"), "frames": st.get("frames"), "seconds": st.get("seconds"),
         "ego_view": st.get("ego_view"), "ego_view_stats": st.get("ego_view_stats"),
         "content": canonical(data)}
    print(f"  {name:8s} rc {r['rc']}  rows {r['rows']:5d}  frames {r['frames']}  "
          f"{(r['seconds'] or 0):7.1f} s  view {r['ego_view']} {r['ego_view_stats']}  "
          f"sha {r['sha256'][:12]}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-glob", required=True)
    ap.add_argument("--logs", type=int, default=3)
    ap.add_argument("--frames-per-log", type=int, default=6)
    ap.add_argument("--work", required=True)
    ap.add_argument("--aug-file", default=None)
    ap.add_argument("--no-replicate", action="store_true",
                    help="skip REF2, the determinism control (only when REF is already proven "
                         "deterministic on this machine)")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    bank = os.path.join(a.work, "bank")
    os.makedirs(bank, exist_ok=True)
    rows = pick_rows(a.bank_glob, a.logs, a.frames_per_log)
    if not rows:
        print("ZZEGOVIEW_FAIL no_rows ZZ"); return 1
    open(os.path.join(bank, "targets_rank0.jsonl"), "w", encoding="utf-8").write("".join(rows))
    print(f"  bank: {len(rows)} frames over {len({json.loads(r)['log_name'] for r in rows})} logs")
    arms = {"REF": run_arm("REF", {"REFE_SCORER_EGO_VIEW": "0"}, bank, a.work, 0, None),
            "EGO": run_arm("EGO", {"REFE_SCORER_EGO_VIEW": "1"}, bank, a.work, 0, None),
            "MUT": run_arm("MUT", {"REFE_SCORER_EGO_VIEW": "1",
                                   "REFE_SCORER_EGO_VIEW_MUTATE": "agent1"}, bank, a.work, 0, None)}
    if not a.no_replicate:
        arms["REF2"] = run_arm("REF2", {"REFE_SCORER_EGO_VIEW": "0"}, bank, a.work, 0, None)
    if a.aug_file:
        arms["AUG_REF"] = run_arm("AUG_REF", {"REFE_SCORER_EGO_VIEW": "0"}, bank, a.work, 1,
                                  a.aug_file)
        arms["AUG_EGO"] = run_arm("AUG_EGO", {"REFE_SCORER_EGO_VIEW": "1"}, bank, a.work, 1,
                                  a.aug_file)
    R, E, M = arms["REF"], arms["EGO"], arms["MUT"]
    fails = []
    if R["rc"] != 0 or R["rows"] == 0:
        fails.append("ref_empty_or_failed")
    if (R["ego_view_stats"] or {}).get("view", 0) != 0:
        fails.append("ref_used_the_view")
    if "REF2" in arms and (arms["REF2"]["content"] != R["content"] or arms["REF2"]["sha256"] != R["sha256"]):
        fails.append("ref_not_deterministic")          # then no byte-level claim is possible at all
    if E["content"] != R["content"]:
        fails.append("ego_content_differs")
    elif E["sha256"] != R["sha256"]:
        fails.append("ego_bytes_differ_content_equal")
    if (E["ego_view_stats"] or {}).get("view", 0) <= 0 or (E["ego_view_stats"] or {}).get("fallback", 1):
        fails.append("ego_view_not_exercised")
    if M["content"] == R["content"]:
        fails.append("mutation_not_detected")
    if a.aug_file:
        AR_, AE = arms["AUG_REF"], arms["AUG_EGO"]
        if AR_["rc"] != 0 or AR_["rows"] == 0:
            fails.append("aug_ref_empty_or_failed")
        if AE["content"] != AR_["content"] or AE["sha256"] != AR_["sha256"]:
            fails.append("aug_ego_differs")
        if (AE["ego_view_stats"] or {}).get("view", 0) <= 0 or (AE["ego_view_stats"] or {}).get("fallback", 1):
            fails.append("aug_ego_view_not_exercised")
    speed = (R["seconds"] or 0) / max(E["seconds"] or 1e-9, 1e-9)
    json.dump({"arms": arms, "fails": fails, "speedup": speed, "frames": len(rows)},
              open(os.path.join(a.work, "diag_scorer_ego_view.json"), "w"), indent=1)
    if fails:
        print(f"ZZEGOVIEW_FAIL {','.join(fails)} ZZ")
        return 1
    print(f"ZZEGOVIEW_EXACT {R['rows']} {len(rows)} {speed:.2f}x ZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
