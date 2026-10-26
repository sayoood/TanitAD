#!/usr/bin/env python3
"""orct1_manifest.py -- builds the reader manifest runs.json (and the --m2b-sha values) for orc_read.py from the pulled O-RECOVER runs and the dev-box T1-kin rollouts.

    manifest   --art ART --out OUT --pfx PFX [--runs orc:0,orc0:0,orc:1,orc0:1] --write PATH     verify every file exists and is the right kind, write runs.json
    manifest   ... --dry-run                                                                      print the JSON it would write (paths only, nothing verified, nothing written)
    m2bsha     --art ART [--runs ...] [--b2-json PATH]                                            print "0=<sha256> 1=<sha256>" (the M2b init checkpoint, from the runs' own records)

Schema = what orc_read.load_manifest / read() consume (raw/2026-10-26-orcread/orc_read.py, test_orc_read.Fx):
    {"seeds": [0, 1], "arms": {"ORC": {"0": E, "1": E}, "ORC0": {"0": E, "1": E}}}
    E = {"ckpt_tail": "M2b_s0_orc/ckpt_last.pt",
         "train_dir": <ART>/M2b_s0_orc          (argv.json, summary.json, selfcheck.json)
         "token":     <ART>/M2b_s0_orc/launch_gate_token.json,
         "t1":        {"tag", "ticks", "dump_dir", "json"}        the decisive  --limit causal  rollout
         "t1_logged": {"tag", "ticks", "dump_dir", "json"}        the K4-only   --limit logged  rollout (PREREG Amendment 2)
         "t0":        {"json": <ART>/<run>/eval_M2b-orc_s0.json, "npz": ...}}
No "M2b" arm: the plain-M2b T1-kin rollout with heads does not exist (the T1-HEADS runs are M2b + HACC post-map: another checkpoint, another ckpt_tail), and PREREG
Amendment 2 reads ORC - M2b at T0 only ("no M2b T1-kin run is needed").   The reader treats the M2b arm as optional.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orct1_pullcheck as PC  # noqa: E402

ARM_READER = {"orc": "ORC", "orc0": "ORC0"}
DEFAULT_RUNS = "orc:0,orc0:0,orc:1,orc0:1"
LIMITS = ("causal", "logged")


def parse_runs(s: str) -> list[tuple[str, int]]:
    out = []
    for it in s.split(","):
        arm, _, seed = it.partition(":")
        if arm not in ARM_READER or not seed.isdigit():
            raise SystemExit(f"--runs: expected arm:seed with arm in {sorted(ARM_READER)}, got {it!r}")
        out.append((arm, int(seed)))
    return out


def tag_of(pfx: str, arm: str, seed: int, limit: str) -> str:
    """the rollout's --tag: <PFX><ORC|ORC0>_s<S>_m10_<limit>_heads  (T1-HEADS naming: HACC_s0_m10_causal_heads)"""
    return f"{pfx}{ARM_READER[arm]}_s{seed}_m10_{limit}_heads"


def fwd(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def t1_entry(out: str, pfx: str, arm: str, seed: int, limit: str) -> dict:
    t = tag_of(pfx, arm, seed, limit)
    return {"tag": t, "ticks": fwd(f"{out}/ticks/ticks_{t}.npz"), "dump_dir": fwd(f"{out}/dump_{t}"), "json": fwd(f"{out}/t1v8f_{t}.json")}


def build(art: str, out: str, pfx: str, runs: list[tuple[str, int]], replicate: tuple[str, int] | None = None, label: str | None = None,
          t0_fresh_dir: str | None = None, t0_use: str = "standard") -> dict:
    """The manifest dict.  ``replicate`` (SMOKE only): every one of the four slots points at the files of that ONE stand-in run."""
    arms: dict[str, dict] = {"ORC": {}, "ORC0": {}}
    slots = runs if replicate is None else [(a, s) for a in ("orc", "orc0") for s in (0, 1)]
    for arm, seed in slots:
        src_arm, src_seed = replicate if replicate is not None else (arm, seed)
        run = PC.run_name(src_arm, src_seed)
        d = fwd(f"{art}/{run}")
        tg = PC.eval_tag(src_arm, src_seed)
        e = {"ckpt_tail": f"{run}/ckpt_last.pt", "train_dir": d, "token": f"{d}/launch_gate_token.json",
             "t1": t1_entry(out, pfx, src_arm, src_seed, "causal"), "t1_logged": t1_entry(out, pfx, src_arm, src_seed, "logged"),
             "t0": {"json": f"{d}/eval_{tg}.json", "npz": f"{d}/eval_{tg}.npz"}}
        if t0_fresh_dir:                                  # the optional CPU T0 eval at --ages fresh (orc_t1_chain.sh ORCT_T0_FRESH=1), the identity K4 compares tick 0 with
            f = {"json": fwd(f"{t0_fresh_dir}/eval_{pfx}{tg}-fresh.json"), "npz": fwd(f"{t0_fresh_dir}/eval_{pfx}{tg}-fresh.npz")}      # the chain's t0tag: <PFX>M2b-<arm>_s<S>-fresh
            e["t0_fresh"] = f
            if t0_use == "fresh":
                e["t0_standard"], e["t0"] = e["t0"], f
        arms[ARM_READER[arm]][str(seed)] = e
    m = {"seeds": [0, 1], "arms": arms,
         "provenance": {"prereg": "Project Steering/PREREG_V8F_ORECOVER.md + Amendments 1-3", "built_by": "orct1_manifest.py (the dev-box T1 chain, orc_t1_chain.sh)",
                        "t1": "v8f_t1_eval.py run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --dump-ticks --dump-heads, plain (no post-map)",
                        "M2b_arm": "absent on purpose: no plain-M2b T1-kin run with heads exists; ORC - M2b is a T0 beside row (Amendment 2)"}}
    if label:
        m["LABEL"] = label
    return m


def verify(m: dict, n_dump_min: int, limits=LIMITS) -> list[str]:
    """Every file the manifest points at exists and is the right kind (the reader's controls do the science; this is the chain's own 'no dangling path' guard)."""
    bad = []

    def need_file(p, what):
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            bad.append(f"{what}: {p} missing or empty")

    for arm, per in m["arms"].items():
        for seed, e in per.items():
            lab = f"{arm} s{seed}"
            for d in ("argv.json", "summary.json", "selfcheck.json"):
                need_file(os.path.join(e["train_dir"], d), f"{lab} {d}")
            need_file(e["token"], f"{lab} token")
            need_file(e["t0"]["json"], f"{lab} T0 json")
            need_file(e["t0"]["npz"], f"{lab} T0 npz")
            for k in ("t0_fresh", "t0_standard"):
                if k in e:
                    need_file(e[k]["json"], f"{lab} {k} json")
                    need_file(e[k]["npz"], f"{lab} {k} npz")
            for key, lim in (("t1", "causal"), ("t1_logged", "logged")):
                t = e[key]
                need_file(t["ticks"], f"{lab} {key} ticks")
                need_file(t["json"], f"{lab} {key} json")
                n = len(glob.glob(os.path.join(t["dump_dir"], "ep*.npz")))
                if n < n_dump_min:
                    bad.append(f"{lab} {key}: {n} ep*.npz in {t['dump_dir']} (< {n_dump_min})")
                try:
                    with open(t["json"], encoding="utf-8") as f:
                        h = json.load(f).get("t1v8f") or {}
                except (OSError, ValueError) as ex:
                    bad.append(f"{lab} {key} json does not parse: {ex}")
                    continue
                if h.get("limit") != lim or h.get("ticks_heads") is not True or h.get("tag") != t["tag"]:
                    bad.append(f"{lab} {key}: json header limit {h.get('limit')!r} (want {lim}), ticks_heads {h.get('ticks_heads')!r}, tag {h.get('tag')!r} (want {t['tag']})")
                tail = "/".join(str(h.get("ckpt", "")).replace("\\", "/").split("/")[-2:])
                if tail != e["ckpt_tail"]:
                    bad.append(f"{lab} {key}: the rollout scored .../{tail}, the manifest expects .../{e['ckpt_tail']}")
    return bad


def init_sha(rec: dict) -> list[str]:
    """every place the trainer recorded the init checkpoint's sha256 (levers.init_ckpt.sha256, bare or under totals / levers / built_levers)"""
    out = []
    for pre in ("", "totals.", "levers.", "built_levers."):
        cur = rec
        ok = True
        for part in (pre + "init_ckpt.sha256").split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            out.append(str(cur))
    return out


def m2b_sha(art: str, runs: list[tuple[str, int]], b2_json: str | None) -> dict[int, str]:
    got: dict[int, set] = {}
    for arm, seed in runs:
        d = os.path.join(art, PC.run_name(arm, seed))
        for f in ("summary.json", "selfcheck.json"):
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                vals = init_sha(json.load(fh))
            if not vals:
                raise SystemExit(f"{d}/{f}: no init_ckpt.sha256 recorded")
            got.setdefault(seed, set()).update(vals)
    res = {}
    for seed, vals in sorted(got.items()):
        if len(vals) != 1:
            raise SystemExit(f"seed {seed}: the runs' own records disagree about the M2b init checkpoint: {sorted(vals)}")
        res[seed] = next(iter(vals))
    if b2_json:
        with open(b2_json, encoding="utf-8") as fh:
            b2 = json.load(fh)
        for seed in res:
            v = b2.get(f"M2b_s{seed}")
            if v is None:
                raise SystemExit(f"{b2_json} has no M2b_s{seed}")
            if v != res[seed]:
                raise SystemExit(f"seed {seed}: the chain's measured M2b sha256 {v[:16]} != the runs' recorded init {res[seed][:16]}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("manifest", "m2bsha"):
        p = sub.add_parser(name)
        p.add_argument("--art", required=True)
        p.add_argument("--runs", default=DEFAULT_RUNS)
        if name == "manifest":
            p.add_argument("--out", required=True)
            p.add_argument("--pfx", default="")
            p.add_argument("--write")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--n-dump-min", type=int, default=141, help="ep*.npz files a rollout's dump dir must hold (141 episodes; SMOKE: 1)")
            p.add_argument("--smoke-replicate", help="arm:seed -- SMOKE only: all four slots point at this one stand-in run")
            p.add_argument("--label")
            p.add_argument("--t0-fresh-dir", help="dir holding eval_M2b-<arm>_s<S>-fresh.{json,npz} (the CPU --ages fresh T0 evals); adds a t0_fresh entry per run")
            p.add_argument("--t0-use", choices=("standard", "fresh"), default="standard", help="which T0 eval the reader's `t0` points at: standard = box 2's (the PREREG as written), fresh = the CPU --ages fresh one")
        else:
            p.add_argument("--b2-json")
            p.add_argument("--smoke-replicate", help="arm:seed -- SMOKE only: read the init sha from this one stand-in run for both seeds")
    a = ap.parse_args(argv)
    runs = parse_runs(a.runs)
    if a.cmd == "m2bsha":
        if a.smoke_replicate:
            arm, seed = parse_runs(a.smoke_replicate)[0]
            r = m2b_sha(a.art, [(arm, seed)], a.b2_json)
            print(" ".join(f"{s}={r[seed]}" for s in (0, 1)))
            return 0
        r = m2b_sha(a.art, runs, a.b2_json)
        if sorted(r) != [0, 1]:
            raise SystemExit(f"the runs give an init sha for seeds {sorted(r)}, the read needs both 0 and 1")
        print(" ".join(f"{s}={r[s]}" for s in (0, 1)))
        return 0
    rep = parse_runs(a.smoke_replicate)[0] if a.smoke_replicate else None
    if a.t0_use == "fresh" and not a.t0_fresh_dir:
        raise SystemExit("--t0-use fresh needs --t0-fresh-dir")
    m = build(a.art, a.out, a.pfx, runs, rep, a.label, a.t0_fresh_dir, a.t0_use)
    if a.dry_run:
        print(json.dumps(m, indent=1))
        return 0
    bad = verify(m, a.n_dump_min)
    if bad:
        print("MANIFEST REFUSED:")
        for b in bad:
            print("  -", b)
        return 1
    if not a.write:
        raise SystemExit("--write PATH is required unless --dry-run")
    os.makedirs(os.path.dirname(os.path.abspath(a.write)), exist_ok=True)
    with open(a.write, "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, indent=1)
    print(f"wrote {a.write} ({sum(len(v) for v in m['arms'].values())} arm-seed entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
