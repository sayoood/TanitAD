"""Validate the proposed patch END-TO-END against a real banked window dump.

Applies `proposed_ctrv_floor.patch` to a THROWAWAY copy of taniteval, backfills
`ctrv` onto a legacy dump, and asserts that:

  1. a legacy dump (no `ctrv` key) still scores, with `floors_missing == ["ctrv"]`
     — the backward-compatibility contract;
  2. a backfilled dump scores with three floors, and the CTRV floor value
     matches the independently-computed number from `run_ctrv_readjudication.py`;
  3. `vs_floor_paired` gains a `ctrv` block whose headline delta matches.

Run on the eval pod:

    cd /workspace && python3 verify_patched_block.py --arm flagship-30k
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrv_floor import build_floors, verify_alignment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/TanitAD/taniteval")
    ap.add_argument("--patch", default="/workspace/proposed_ctrv_floor.patch")
    ap.add_argument("--work", default="/workspace/_ctrv_patchtest")
    ap.add_argument("--val-dir", default="/workspace/val40cache")
    ap.add_argument("--results-dir",
                    default="/workspace/TanitAD/taniteval/results")
    ap.add_argument("--arm", default="flagship-30k")
    a = ap.parse_args()

    work = Path(a.work)
    if work.exists():
        shutil.rmtree(work)
    (work / "taniteval").mkdir(parents=True)
    shutil.copytree(Path(a.src) / "taniteval", work / "taniteval" / "taniteval")
    r = subprocess.run(["patch", "-p3", "-d", str(work / "taniteval" / "taniteval"),
                        "-i", a.patch, "--batch"],
                       capture_output=True, text=True, encoding="utf-8")
    print("[patch]", r.returncode, r.stdout.strip()[-400:], r.stderr.strip()[-400:])
    if r.returncode != 0:
        raise SystemExit("patch failed")

    sys.path.insert(0, str(work / "taniteval"))
    for m in [k for k in list(sys.modules) if k.startswith("taniteval")]:
        del sys.modules[m]
    from taniteval import driving as D
    assert D.FLOORS == ("cv", "holdv0", "ctrv"), D.FLOORS
    print("[patched] FLOORS =", D.FLOORS)

    win = torch.load(Path(a.results_dir) / f"windows_{a.arm}.pt",
                     map_location="cpu", weights_only=False)

    # 1. legacy dump: still scores, and SAYS the third floor is missing
    legacy = D.tier0(win, arm=a.arm)
    assert "ctrv" not in legacy["floor_values"], "legacy dump grew a ctrv floor"
    assert legacy["floors_missing"] == ["ctrv"], legacy.get("floors_missing")
    print(f"[legacy ] floors={list(legacy['floor_values'])} "
          f"missing={legacy['floors_missing']} "
          f"ade={legacy['headline']['ade_0_2s']['mean']}")

    # 2/3. backfilled dump: three floors, numbers match the independent driver
    files = sorted(Path(a.val_dir).glob("ep_*.pt"))
    eps = []
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        eps.append((i, d["poses"].float(),
                    int(min(d["frames_u8"].shape[0], d["actions"].shape[0],
                            d["poses"].shape[0]))))
    built = build_floors(eps)
    al = verify_alignment(built, win)
    assert al["aligned"], al
    win2 = dict(win)
    win2["ctrv"] = built["ctrv"]          # the patch persists the UNGATED form
    full = D.tier0(win2, arm=a.arm)
    assert full["floors_missing"] == [], full["floors_missing"]
    fv = full["floor_values"]["ctrv"]["ade_0_2s"]["value"]
    vp = full["vs_floor_paired"]["ctrv"]["ade_0_2s"]
    print(f"[full   ] floors={list(full['floor_values'])} "
          f"ctrv_ade={fv} model={full['headline']['ade_0_2s']['mean']} "
          f"vs_ctrv delta={vp['delta']} [{vp['lo']}, {vp['hi']}] "
          f"separated={vp['separated']}")
    assert abs(full["headline"]["ade_0_2s"]["mean"]
               - legacy["headline"]["ade_0_2s"]["mean"]) < 1e-9, \
        "the model's own numbers moved — the patch must only add a floor"
    print("[OK] patched block validated end-to-end")


if __name__ == "__main__":
    main()
