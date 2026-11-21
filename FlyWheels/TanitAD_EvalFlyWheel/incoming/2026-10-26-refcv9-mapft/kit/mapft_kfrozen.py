#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mapft_kfrozen.py -- PREREG_MAPFT s2 control K-FROZEN, as a script that reads ARTIFACTS (never a status code) and carries its own controls.

  K-FROZEN (sealed text): `assert_map_only_finetune.py --src ckpt_ema_48852.pt --dst <final>` exits 0 on the EMA view AND the raw view;
  it must name changed tensors ONLY under the three map modules, AND AT LEAST ONE IN EACH.

The landed tool checks "nothing outside the map modules changed" (a PASS even when NOTHING changed at all).  The sealed text also demands
"at least one changed tensor in each of the three modules" -- that half is computed HERE, from the tool's own JSON, so a run that
trained nothing (or trained one module) cannot read as K-FROZEN PASS.

  python mapft_kfrozen.py run --tree <launch_tree> --src <ckpt_ema_48852.pt> --raw <run>/ckpt.pt --ema <run>/model_ema.pt --out-dir <dir>
  python mapft_kfrozen.py self-test --tree <any tree carrying stack/scripts/assert_map_only_finetune.py> [--work <dir>]

What `run` does (CPU only; ~400 MB per checkpoint read, mmap):
  V1  assert tool on --raw  (ckpt.pt: TWO views, raw + ema shadow)                -> out_dir/kfrozen_raw.json
  V2  assert tool on --ema  (model_ema.pt: ONE view, ema)                         -> out_dir/kfrozen_ema.json
  C1  CONTROL  src vs src          : the tool says ok, and my per-module criterion MUST read FAIL (nothing changed)
  C2  CONTROL  ckpt.pt ema view vs model_ema.pt : zero changed tensors (the file MAPSEL scores IS the checkpoint's EMA view)
  Verdict = V1 and V2 pass the per-view criterion AND C1 reads FAIL-as-expected AND C2 reads identical.   Markers ZZMAPFT-KFROZEN-{PASS,FAIL}ZZ.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

MODULES = ("_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")


def under(key: str, mod: str) -> bool:
    return key == mod or key.startswith(mod + ".")


def tool_path(tree: Path) -> Path:
    p = tree / "stack" / "scripts" / "assert_map_only_finetune.py"
    if not p.is_file():
        raise SystemExit(f"REFUSED: no {p}")
    return p


def run_tool(tool: Path, src: str, dst: str, out_json: Path, extra=()) -> dict:
    """Run the LANDED tool as a subprocess; the verdict is the JSON it WROTE.  A missing/unparsable JSON is an ERROR, never a pass."""
    if out_json.exists():
        out_json.unlink()
    r = subprocess.run([sys.executable, "-I", str(tool), "--src", src, "--dst", dst, "--json", str(out_json), *extra],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out_json.with_suffix(".txt").write_text(r.stdout + "\n--stderr--\n" + r.stderr, encoding="utf-8")
    if not out_json.is_file():
        return {"error": f"the tool wrote no JSON (rc={r.returncode}): {(r.stderr or r.stdout)[-300:]}", "rc": r.returncode}
    try:
        d = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as e:                                  # noqa: BLE001
        return {"error": f"unparsable JSON: {e}", "rc": r.returncode}
    d["_rc"] = r.returncode
    return d


def per_view_criterion(res: dict) -> dict:
    """-> {view: {"tool_ok", "non_map_changed", "per_module_changed", "each_module_changed", "pass"}}.  Computed from the JSON only."""
    out = {}
    for name, v in (res.get("views") or {}).items():
        if "error" in v:
            out[name] = {"pass": False, "error": v["error"]}
            continue
        keys = [r["key"] for r in v.get("map_changed", [])]
        per = {m: sum(1 for k in keys if under(k, m)) for m in MODULES}
        stray = [k for k in keys if not any(under(k, m) for m in MODULES)]          # a "map_changed" row outside the 3 modules cannot happen; assert it
        each = all(n >= 1 for n in per.values())
        out[name] = {"tool_ok": bool(v.get("ok")), "n_non_map_changed": v.get("n_non_map_changed"), "n_missing": v.get("n_missing"),
                     "n_extra": v.get("n_extra"), "per_module_changed": per, "each_module_changed": each, "stray_rows": stray,
                     "pass": bool(v.get("ok")) and each and not stray}
    return out


def run(a) -> int:
    tree, od = Path(a.tree), Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    tool = tool_path(tree)
    rec: dict = {"tool": str(tool), "src": a.src, "raw": a.raw, "ema": a.ema}
    ok = True
    for tag, dst in (("raw", a.raw), ("ema", a.ema)):
        if not Path(dst).is_file():
            rec[tag] = {"error": f"{dst} missing"}
            ok = False
            continue
        res = run_tool(tool, a.src, dst, od / f"kfrozen_{tag}.json")
        if "error" in res:
            rec[tag] = res
            ok = False
            continue
        crit = per_view_criterion(res)
        rec[tag] = {"dst": res.get("dst"), "views": crit}
        if a.expect_step is not None:                       # a mid-run ckpt (a stale rolling file) must not be graded as the final one
            got_step = (res.get("dst") or {}).get("step")
            rec[tag]["step_ok"] = (got_step == a.expect_step)
            rec[tag]["step"] = got_step
            if got_step != a.expect_step:
                ok = False
        want = ("raw", "ema") if tag == "raw" else ("ema",)
        if sorted(crit) != sorted(want):                      # ckpt.pt must expose BOTH views, model_ema.pt exactly the ema one
            rec[tag]["error"] = f"expected views {want}, got {sorted(crit)}"
            ok = False
        ok = ok and all(c.get("pass") for c in crit.values())
    # C1 -- the criterion must read FAIL when nothing changed (src vs itself): a pass here would mean the "at least one in each" half is not computed
    c1 = run_tool(tool, a.src, a.src, od / "kfrozen_control_selfsrc.json")
    if "error" in c1:
        rec["C1_control_src_vs_src"] = c1
        ok = False
    else:
        crit = per_view_criterion(c1)
        tool_ok = all(v.get("tool_ok") for v in crit.values()) and bool(crit)
        criterion_fail = all(not v.get("pass") for v in crit.values()) and bool(crit)
        rec["C1_control_src_vs_src"] = {"tool_says_ok": tool_ok, "my_criterion_reads_FAIL": criterion_fail, "views": crit,
                                        "control_ok": bool(tool_ok and criterion_fail)}
        ok = ok and rec["C1_control_src_vs_src"]["control_ok"]
    # C2 -- the EMA file MAPSEL will score must BE the checkpoint's EMA view
    if Path(a.raw).is_file() and Path(a.ema).is_file():
        c2 = run_tool(tool, a.raw, a.ema, od / "kfrozen_control_ema_file_vs_ckpt_ema_view.json", extra=("--src-view", "ema"))
        if "error" in c2:
            rec["C2_control_ema_file_is_ckpt_ema_view"] = c2
            ok = False
        else:
            v = (c2.get("views") or {}).get("ema", {})
            same = bool(c2.get("ok")) and v.get("n_map_changed") == 0 and v.get("n_non_map_changed") == 0
            rec["C2_control_ema_file_is_ckpt_ema_view"] = {"identical": same, "n_map_changed": v.get("n_map_changed")}
            ok = ok and same
    rec["K_FROZEN"] = "PASS" if ok else "FAIL"
    (od / "KFROZEN.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rec, indent=1, default=str)[:3500])
    print("ZZMAPFT-KFROZEN-PASSZZ" if ok else "ZZMAPFT-KFROZEN-FAILZZ")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------------------------
def self_test(a) -> int:
    """Literal expectations on synthetic checkpoints; each MUTANT must be refused."""
    import torch
    tree = Path(a.tree)
    tool = tool_path(tree)
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="kfrozen_selftest_"))
    work.mkdir(parents=True, exist_ok=True)

    def sd(change=()):
        base = {"_map_hires.refine.w": torch.zeros(4), "_map_hires.near.w": torch.zeros(4), "_map_hires.near_refine.w": torch.zeros(4),
                "trunk.w": torch.arange(4.0), "bn.running_mean": torch.ones(3), "bn.num_batches_tracked": torch.tensor(7)}
        for k in change:
            base[k] = base[k] + 1 if base[k].is_floating_point() else base[k] + 1
        return base

    def save(name, state):
        p = work / name
        torch.save({"model": state, "step": 5, "kind": "ema_model_only"}, p)
        return str(p)

    src = save("src.pt", sd())
    maps = ["_map_hires.refine.w", "_map_hires.near.w", "_map_hires.near_refine.w"]
    cases = [                                  # (name, changed keys, expected pass)
        ("good_all_three_modules", maps, True),
        ("MUT_nothing_changed", [], False),                              # the tool says ok; the sealed 'at least one in each' must refuse
        ("MUT_only_two_modules", maps[:2], False),
        ("MUT_trunk_weight_changed", maps + ["trunk.w"], False),
        ("MUT_bn_stat_changed", maps + ["bn.running_mean"], False),
        ("MUT_bn_counter_changed", maps + ["bn.num_batches_tracked"], False),
    ]
    bad = []
    for name, ch, want in cases:
        dst = save(name + ".pt", sd(ch))
        res = run_tool(tool, src, dst, work / (name + ".json"))
        crit = per_view_criterion(res) if "error" not in res else {}
        got = bool(crit) and all(c.get("pass") for c in crit.values())
        print(f"{'OK ' if got == want else 'BAD'} {name}: expected {'PASS' if want else 'FAIL'}, read {'PASS' if got else 'FAIL'} "
              f"{ {v: c.get('per_module_changed') for v, c in crit.items()} }")
        if got != want:
            bad.append(name)
    print("ZZMAPFT-KFROZEN-SELFTEST-OKZZ" if not bad else f"ZZMAPFT-KFROZEN-SELFTEST-FAILZZ {bad}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tree", required=True)
    r.add_argument("--src", required=True)
    r.add_argument("--raw", required=True)
    r.add_argument("--ema", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--expect-step", type=int, default=None, help="the run's final step N: both dst files must carry exactly this step")
    s = sub.add_parser("self-test")
    s.add_argument("--tree", required=True)
    s.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else self_test(a)


if __name__ == "__main__":
    sys.exit(main())
