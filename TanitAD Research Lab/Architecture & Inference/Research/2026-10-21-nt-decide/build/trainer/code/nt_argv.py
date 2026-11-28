#!/usr/bin/env python3
"""nt_argv -- the exact trainer argv of H-NT-DECIDE-1's arms (A1, A1r) and of the dev-box CPU smoke, DERIVED from
the refcv8 launch argv (``stack/ops/runs.d/refcv8-r101-s0.argv.json`` at fb47a58) by a declared, closed list of
edits -- so the departures are exactly the ones listed in ``EDITS`` and nothing else can drift.

``python nt_argv.py a1 --paths paths_thor.json --out argv/``  -> ``nt-a1-s0.argv.json`` + ``nt-a1r-s1.argv.json``
``python nt_argv.py smoke --paths paths_devbox.json --out argv/`` -> ``nt-smoke.argv.json``

Each output carries ``argv`` (the list), ``argv_sha256`` (sha256 of ``json.dumps(argv)``), ``base`` (path + sha256 of
the refcv8 argv file) and ``edits`` (every flag replaced / removed / added, with the reason).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: A1 recipe facts (PREREG_NT_DECIDE / DESIGN_AND_COSTING sec. 4): 2 epochs of 99,824 tokens at batch 16
STEPS_A1 = 12478
SNAPSHOTS_A1 = "0,1560,3120,6239,9359,12478"     # step 0 = K0's input; then 0.25 / 0.5 / 1 / 1.5 / 2 epochs
GRAD_SHARE_EVERY_A1 = 50                         # K7 P-GRAD: 1,000 / 6,239 / 12,478 are not all multiples of 250;
GRAD_SHARE_AT_A1 = "1000,6250,12450"             # the nearest LOGGED multiples of 50 (declared, +11 / -28 steps)

#: (flag, kind, why) -- kind: "value" (the path/value comes from --paths), "drop", "add"
EDITS = (
    ("--v2-cache", "value", "the NavSim token-episode cache (E1)"),
    ("--v7-labels", "value", "E2: the minimal s2-geom-v7 records of the NavSim tokens"),
    ("--clip-clock-sidecar", "value", "E2: NavSim clock rows (grid 0.0 / dt 0.1) MERGED with the PhysicalAI eval139 "
                                      "rows -- NOT droppable: G3 refuses v7 labels on a real corpus without it"),
    ("--speed-max-sidecar-v6", "value", "E2: the all-UNKNOWN road-type rows (the LEGAL inference feed)"),
    ("--agent-join", "value", "E3: OpenScene anns (class map) MERGED with the eval139 PhysicalAI records"),
    ("--agent-rig-extrinsics", "value", "E3: per-token stitch camera (frames416 M + F0 mount - road plane) + eval139"),
    ("--join3d", "value", "E3: the same records + cz (road z = 0) / h, MERGED with eval139"),
    ("--vis1-sidecar", "value", "E3: VIS-1 z-buffer on the stitch's observed mask, MERGED with eval139"),
    ("--r8-v9-labels", "value", "E2: the v9 release, one row per token at k = 9"),
    ("--r8-v9-md5", "value", "E2: the NEW v9 release md5"),
    ("--join-defect-masks", "value", "the refcv8 file itself, from THIS launch tree (NavSim adds no entry: ego-"
                                     "footprint boxes are dropped at E3 build time, 0 found; eval139 unchanged)"),
    ("--init-from", "value", "theta_DEC (ckpt_wiseft_mod2_DEC.pt, md5 dad4cb5b...)"),
    ("--out", "value", "the arm's run directory"),
    ("--steps", "value", f"2 epochs = {STEPS_A1} steps at batch 16 (PREREG)"),
    ("--seed", "value", "A1 seed 0 / A1r seed 1 (H-ESTIM-SEED-1 replicate)"),
    ("--grad-share-every", "value", f"K7 P-GRAD cadence {GRAD_SHARE_EVERY_A1} (a multiple of --log-every 50)"),
    ("--grad-share-at", "value", f"K7 P-GRAD readings at {GRAD_SHARE_AT_A1} (nearest logged steps to 1,000 / "
                                 f"6,239 / 12,478)"),
    ("--pose-sync-sidecar", "drop", "a PhysicalAI timing artifact (DESIGN sec. 3 'dropped'); NavSim rows are on the "
                                    "nominal 0.1 s grid by construction"),
    ("--window-starts", "add", "R2: ONE window per token-episode (t = 0)"),
    ("--nt-heldout", "add", "K3: the 38 selection-scan + 136 navtest logs"),
    ("--nt-heldout-sha256", "add", "K3: the guard file's pinned digest"),
    ("--nt-token-table", "add", "K3: clip -> (token, log) for every train clip"),
    ("--r8-speed-unknown-always", "add", "the train speed sidecar is all-unknown by declaration"),
    ("--map-min-coverage", "add", "0.0: no NavSim map GT exists; the 10 cm map loss then reads zero labelled windows "
                                  "and its classifier gets EXACTLY zero gradient (K4) -- no architecture change"),
    ("--snapshot-steps", "add", f"{SNAPSHOTS_A1}: K0's step-0 save + the 5 selection snapshots (EMA + RAW)"),
)
FLAGS_WITH_VALUE_ADDED = {"--window-starts": "first", "--map-min-coverage": "0.0"}
BOOL_ADDED = {"--r8-speed-unknown-always"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _flag_spans(argv: list) -> dict:
    """flag -> (index, n_values) for every ``--flag`` token (values = the following non-flag tokens)."""
    out = {}
    i = 0
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            j = i + 1
            while j < len(argv) and not (argv[j].startswith("--") and not _is_num(argv[j])):
                j += 1
            if t in out:
                raise SystemExit(f"[nt-argv] flag {t} twice in the base argv")
            out[t] = (i, j - i - 1)
            i = j
        else:
            raise SystemExit(f"[nt-argv] stray token {t!r} at {i}")
    return out


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def derive(base: list, values: dict, *, steps: int, seed: int, snapshots: str, gs_every: int, gs_at: str | None,
           extra_drop: tuple = (), extra_set: dict | None = None) -> tuple[list, list]:
    """Apply EDITS to ``base``. ``values`` = path values for the 'value' flags + the 'add' flags' values."""
    spans = _flag_spans(base)
    vals = dict(values, **{"--steps": str(steps), "--seed": str(seed), "--grad-share-every": str(gs_every)})
    if gs_at is not None:
        vals["--grad-share-at"] = gs_at
    vals["--snapshot-steps"] = snapshots
    vals.update(FLAGS_WITH_VALUE_ADDED)
    vals.update(extra_set or {})
    drop = {f for f, k, _ in EDITS if k == "drop"} | set(extra_drop)
    if gs_at is None:
        drop.add("--grad-share-at")
    out, log = [], []
    for f, (i, n) in sorted(spans.items(), key=lambda kv: kv[1][0]):
        if f in drop:
            log.append({"flag": f, "edit": "drop", "was": base[i + 1:i + 1 + n]})
            continue
        if f in vals:
            new = vals[f] if isinstance(vals[f], list) else [vals[f]]
            out += [f] + [str(x) for x in new]
            if base[i + 1:i + 1 + n] != [str(x) for x in new]:
                log.append({"flag": f, "edit": "value", "was": base[i + 1:i + 1 + n], "now": [str(x) for x in new]})
            continue
        out += base[i:i + 1 + n]
    for f, k, _ in EDITS:
        if k == "add":
            if f in BOOL_ADDED:
                out.append(f)
                log.append({"flag": f, "edit": "add"})
            else:
                if f not in vals:
                    raise SystemExit(f"[nt-argv] no value for added flag {f}")
                out += [f, str(vals[f])]
                log.append({"flag": f, "edit": "add", "now": [str(vals[f])]})
    for f, k, _ in EDITS:
        if k == "value" and f not in vals and f not in drop:
            raise SystemExit(f"[nt-argv] no value given for {f}")
    _flag_spans(out)                         # re-parse: no duplicate flag, no stray token
    return out, log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("a1", "smoke"))
    ap.add_argument("--base", default=str(HERE.parent / "tree_fb47a58/stack/ops/runs.d/refcv8-r101-s0.argv.json"))
    ap.add_argument("--paths", required=True, help="JSON {flag: value} for every 'value' / added flag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--b1", choices=("keep", "drop"), default="keep",
                    help="NT-B1-GUARD (an OPEN gate item until the Master Mind sets launch_gate._NT_B1_GUARD): keep "
                         "refcv8's B1 guard literals (PREREG sec. 2, the default) or drop both flags (PREREG sec. 8)")
    a = ap.parse_args(argv)
    braw = Path(a.base).read_bytes()
    bd = json.loads(braw)
    base = bd["argv"] if isinstance(bd, dict) else bd
    if base and not base[0].startswith("--"):          # tolerate a leading script path
        base = [t for t in base if True]
        while base and not base[0].startswith("--"):
            base = base[1:]
    paths = json.loads(Path(a.paths).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    if a.mode == "a1":
        for name, seed in (("nt-a1-s0", 0), ("nt-a1r-s1", 1)):
            p = dict(paths)
            p["--out"] = paths["--out"].replace("{arm}", name)
            argl, log = derive(base, p, steps=STEPS_A1, seed=seed, snapshots=SNAPSHOTS_A1,
                               gs_every=GRAD_SHARE_EVERY_A1, gs_at=GRAD_SHARE_AT_A1,
                               extra_drop=(("--b1-guard-ref-ap2m", "--b1-guard-step") if a.b1 == "drop" else ()))
            runs.append((name, argl, log))
    else:
        cpu_drop = ("--trunk-compile", "--trunk-bf16", "--trunk-channels-last",   # CPU: no compile / bf16 / NHWC
                    "--b1-guard-ref-ap2m", "--b1-guard-step")                    # a 2-step smoke: guard step > steps
        argl, log = derive(base, paths, steps=2, seed=0, snapshots="0,2", gs_every=1, gs_at=None,
                           extra_drop=cpu_drop,
                           extra_set={"--batch": "2", "--workers": "0", "--eval-batches": "1", "--log-every": "1",
                                      "--save-every": "1"})
        runs.append(("nt-smoke", argl, log))
    for name, argl, log in runs:
        rec = {"run": name, "argv": argl, "b1_guard": (a.b1 if a.mode == "a1" else "dropped (smoke)"), "argv_sha256": _sha(json.dumps(argl).encode("utf-8")),
               "base": {"path": a.base, "sha256": _sha(braw)}, "edits": log,
               "edit_reasons": {f: why for f, _k, why in EDITS}, "tool": "nt_argv.py",
               "tool_sha256": _sha(Path(__file__).read_bytes())}
        (out / f"{name}.argv.json").write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")
        print(f"{name}: {len(argl)} tokens, {len(log)} edits, argv_sha256 {rec['argv_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
