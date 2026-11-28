"""fresh2_refusals.py -- PREREG_FRESH2 s1.3, mechanical: a FRESH2 clip that any L1C chain step refuses is logged by sha12 + step + reason and EXCLUDED.
There is NO replacement (the pool is fixed at candidate ranks 421-1020) -- this is l1c_replace_refused.py's logic with the replacement removed.

Steps read (each from its ARTIFACT, never from an exit code):
  episode build   no <clip>.v2ep.pt in F:/l1c/kit_fresh2/data/refcv6-b1-416x1024-eval139
  labels          labels_report.json 'failed' (s2_geom_emit_v7.emit_one raised)
  clock sidecar   clock_sidecar.jsonl.meta.json 'refused_by_sid'
Writes: F:/l1c/fresh2/fresh2_ids.json rewritten to the USABLE clips (the pool copy kept as fresh2_ids.pool.json; raw ids, F: only); the refused clips' episode
files moved to F:/l1c/kit_fresh2_discarded/v2; _v2manifest.pt removed (rebuilt by the next step); <pkg>/select/chain_failures_sha12.json (sha12 + step + reason).
STOPS (exit 5, file REFUSAL_STOP.json) if more than 15 % of the fetched pool is refused (the Master Mind's stop rule).  Prints ZZF2REFUSEDn/NZZ.
NEW file (FRESH2 agent, 2026-10-27).  No clip id is printed.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58/stack")
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402

KIT = Path("F:/l1c/kit_fresh2")
EPS = KIT / "data" / "refcv6-b1-416x1024-eval139"
IDS = Path("F:/l1c/fresh2/fresh2_ids.json")
POOL = Path("F:/l1c/fresh2/fresh2_ids.pool.json")
DISC = Path("F:/l1c/kit_fresh2_discarded/v2")
PKG = Path(__file__).resolve().parent.parent
MAX_FRAC = 0.15


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def check() -> int:
    """--check (after labels / speed / clock were RE-RUN on the usable set): every usable clip has an episode file, an emitted label and a clock row; nothing else is in the kit."""
    usable = json.load(open(IDS, encoding="utf-8"))
    eps = {p.name[:-8] for p in EPS.glob("*.v2ep.pt")}
    lr = json.load(open(KIT / "labels_report.json", encoding="utf-8"))
    m = json.load(open(KIT / "data" / "clock_sidecar.jsonl.meta.json", encoding="utf-8"))
    n_clock = sum(1 for _ in open(KIT / "data" / "clock_sidecar.jsonl", encoding="utf-8"))
    ok = {"eps_eq_usable": eps == {r["clip_id"] for r in usable}, "labels_failed_0": len(lr["failed"]) == 0, "labels_n_eq": lr["n_emitted"] == len(usable),
          "clock_refused_0": len(m["refused_by_sid"]) == 0, "clock_rows_eq": n_clock == len(usable)}
    print(f"ZZF2REFCHECK{'PASS' if all(ok.values()) else 'FAIL'}ZZ n_usable {len(usable)} {ok}", flush=True)
    return 0 if all(ok.values()) else 6


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        return check()
    if POOL.exists():
        raise SystemExit("[refusals] fresh2_ids.pool.json exists: the exclusion round already ran -- use --check (refusals are logged ONCE, on the full pool)")
    if not POOL.exists():
        shutil.copyfile(IDS, POOL)
    pool = json.load(open(POOL, encoding="utf-8"))
    by12 = {sha12(r["clip_id"]): r for r in pool}
    refused = {}                                     # clip_id -> (step, reason)
    built = {p.name[:-8] for p in EPS.glob("*.v2ep.pt")}
    for r in pool:
        if r["clip_id"] not in built:
            refused[r["clip_id"]] = ("episode build", "no episode file written by v2_compressed.py build")
    lr = json.load(open(KIT / "labels_report.json", encoding="utf-8"))
    for f in lr["failed"]:
        c = by12[f["sha12"]]["clip_id"]
        refused.setdefault(c, ("labels", f["why"]))
    m = json.load(open(KIT / "data" / "clock_sidecar.jsonl.meta.json", encoding="utf-8"))
    sid2 = {str(stable_episode_id(r["clip_id"])): r["clip_id"] for r in pool}
    unknown = [s for s in m["refused_by_sid"] if s not in sid2]
    if unknown:
        raise SystemExit(f"[refusals] {len(unknown)} refused sids are not FRESH2 pool clips")
    for s, why in m["refused_by_sid"].items():
        refused.setdefault(sid2[s], ("clock sidecar", why))
    n_pool = len(pool)
    frac = len(refused) / n_pool
    rows = sorted(({"sha12": sha12(c), "step": st, "why": why} for c, (st, why) in refused.items()), key=lambda x: x["sha12"])
    by_step = {}
    for x in rows:
        by_step[x["step"]] = by_step.get(x["step"], 0) + 1
    rec = {"prereg": "PREREG_FRESH2.md s1.3 (no replacement)", "n_pool_fetched": n_pool, "n_refused": len(rows), "refused_frac": round(frac, 4),
           "by_step": by_step, "rows": rows, "n_usable": n_pool - len(rows)}
    (PKG / "select").mkdir(exist_ok=True)
    (PKG / "select" / "chain_failures_sha12.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if frac > MAX_FRAC:
        (PKG / "result").mkdir(exist_ok=True)
        (PKG / "result" / "REFUSAL_STOP.json").write_text(json.dumps({**rec, "stop": f"refused {frac:.3f} > {MAX_FRAC} of the pool: STOP per the brief"}, indent=1), encoding="utf-8")
        print(f"ZZF2REFUSALSTOPZZ {len(rows)}/{n_pool}", flush=True)
        return 5
    DISC.mkdir(parents=True, exist_ok=True)
    moved = 0
    for c in refused:
        p = EPS / f"{c}.v2ep.pt"
        if p.exists():
            shutil.move(str(p), str(DISC / p.name))
            moved += 1
    if moved and (EPS / "_v2manifest.pt").exists():
        os.remove(EPS / "_v2manifest.pt")
    usable = [r for r in pool if r["clip_id"] not in refused]
    json.dump(usable, open(IDS, "w", encoding="utf-8"))
    json.dump([r["clip_id"] for r in usable], open(IDS.parent / "usable_clips.json", "w", encoding="utf-8"))
    print(f"ZZF2REFUSED{len(rows)}/{n_pool}ZZ by_step {by_step} moved {moved} usable {len(usable)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
