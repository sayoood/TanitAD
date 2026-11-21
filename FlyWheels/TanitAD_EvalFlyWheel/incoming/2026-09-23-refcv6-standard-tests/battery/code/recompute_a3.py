"""SPEC AMENDMENT A3, applied to a tag whose panels were computed BEFORE A3 existed.

usage: python recompute_a3.py <tag_dir>          e.g. C:/Users/Admin/ev6_battery/raw/step5000

For every `panel_s<k>` / `s6_s<k>` in the tag it recomputes the paired cells with the A3 cell
(`refcv6_panel.cross_paired`, the same function the runner calls) and, for two seeds, the
inference-seed replicate. The pre-A3 JSONs are KEPT as `*.preA3.json`.

⛔ Refuses to write unless EVERY non-A3 cell of the recomputation is bit-identical to the banked
pre-A3 cell (same delta / lo / hi / separated / n): A3 adds one cell and must move nothing else.
The comparison count and the difference count are written to `A3_recompute_record.json`.
CPU only; no forward pass.
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_panel as P  # noqa: E402
import run_battery as RB  # noqa: E402

KEYS = ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")


def _cells(cp: dict):
    """(pair, family, metric) -> the comparable fields, skipping the A3 cell itself."""
    out = {}
    for nm, blk in (cp.get("pairs") or {"_replicate": cp}).items():
        for fam, ms in (blk.get("families") or {}).items():
            for mk, c in ms.items():
                if mk == P.YAW_VALID or not isinstance(c, dict):
                    continue
                out[(nm, fam, mk)] = tuple(c.get(k) for k in KEYS)
    return out


def _compare(old: dict, new: dict) -> dict:
    o, n = _cells(old), _cells(new)
    diffs = [f"{k}: {o.get(k)} -> {n.get(k)}" for k in sorted(set(o) | set(n), key=str) if o.get(k) != n.get(k)]
    return {"n_cells_compared": len(set(o) | set(n)), "n_differences": len(diffs), "differences": diffs[:20]}


def _has_a3(cp: dict) -> bool:
    blocks = (cp.get("pairs") or {"_r": cp}).values()
    return any(P.YAW_VALID in ((b.get("families") or {}).get("lateral") or {}) for b in blocks)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1])
    spec = HERE.parent / "SPEC.md"
    rec = {"tool": "recompute_a3.py", "tag_dir": str(root), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "spec_sha256": _sha(spec), "amendment": "A3", "files": {}}
    jobs = []
    for pd in sorted(root.glob("panel_s*")):
        s = pd.name.split("_s")[-1]
        jobs.append((pd, root / f"cross_paired_s{s}.json"))
    for sd in sorted(root.glob("s6_s*")):
        s = sd.name.split("_s")[-1]
        jobs.append((sd, root / f"cross_paired_s6_s{s}.json"))
    for panel, out in jobs:
        if not out.exists():
            rec["files"][out.name] = {"status": "ABSENT (nothing to amend)"}
            continue
        old = json.load(open(out, encoding="utf-8"))
        if _has_a3(old):
            rec["files"][out.name] = {"status": "ALREADY A3 (computed after the amendment)"}
            continue
        new = P.cross_paired(str(panel), RB.PAIRS, n_boot=2000)
        cmp_ = _compare(old, new)
        if cmp_["n_differences"]:
            rec["files"][out.name] = {"status": "REFUSED: a non-A3 cell moved", **cmp_}
            continue
        pre = out.with_name(out.stem + ".preA3.json")
        if not pre.exists():
            shutil.copy2(out, pre)
        json.dump(new, open(out, "w", encoding="utf-8"), indent=1)
        rec["files"][out.name] = {"status": "AMENDED", "pre_a3_kept_as": pre.name, **cmp_}
    seeds = sorted(p.name.split("_s")[-1] for p in root.glob("panel_s*"))
    rj = root / "inference_seed_replicate.json"
    if len(seeds) >= 2 and rj.exists():
        old = json.load(open(rj, encoding="utf-8"))
        if _has_a3(old):
            rec["files"][rj.name] = {"status": "ALREADY A3"}
        else:
            new = P.seed_replicate(str(root / f"panel_s{seeds[0]}"), str(root / f"panel_s{seeds[1]}"),
                                   n_boot=2000)
            cmp_ = _compare(old, new)
            if cmp_["n_differences"] or new["max_abs_path_diff_m"] != old["max_abs_path_diff_m"]:
                rec["files"][rj.name] = {"status": "REFUSED: a non-A3 cell moved", **cmp_}
            else:
                pre = rj.with_name(rj.stem + ".preA3.json")
                if not pre.exists():
                    shutil.copy2(rj, pre)
                json.dump(new, open(rj, "w", encoding="utf-8"), indent=1)
                rec["files"][rj.name] = {"status": "AMENDED", "pre_a3_kept_as": pre.name, **cmp_}
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    refused = [k for k, v in rec["files"].items() if str(v.get("status", "")).startswith("REFUSED")]
    rec["verdict"] = "REFUSED on " + ", ".join(refused) if refused else "OK"
    json.dump(rec, open(root / "A3_recompute_record.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v.get("status") for k, v in rec["files"].items()} | {"verdict": rec["verdict"]},
                     indent=1))
    if refused:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
