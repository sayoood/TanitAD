#!/usr/bin/env python3
"""dag2t1_manifest.py -- the dev-box CPU T1 chain of PREREG_V8F_DAGGER2: builds the reader manifest runs.json for dag2_read.py, and the small guards the chain needs.   CPU only.
Compact sibling of dagt1_manifest.py (which ran PREREG_V8F_DAGGER's chain): the DAG arm names, no ORC-0 (PREREG_V8F_DAGGER2 change 3 fixes RF), DAG's banked outputs checked instead.
Utilities that did not change (the eval / TRAIN id lists, the tool-tree import-closure parity) are imported from dagt1_manifest.py, sha256-pinned below, not copied.

    manifest   --art ART --out OUT --ids-dir DIR --dag-art DAG_ART [--runs dag2:0,dag20:0,dag2:1,dag20:1] [--pfx P] --t0-fresh-dir D --n-dump-min N (--write PATH | --dry-run)
    initsha    --art ART --dag-art DAG_ART [--runs ..] [--b2-json PATH]     prints "0=<sha256> 1=<sha256>": the checkpoint DAG2 / DAG2-0 started from = DAG_s<S> (the runs' own records; must equal DAG's SUMS)
    dagbase    --dag-art DAG_ART --dag-t1 DAG_T1                            DAG's BANKED outputs (read, never re-run) are present and are the pinned files
    treetip    --tree TREE --git-dir G --tip T                              every .py of the tool tree that the tip also has is byte-identical to the tip's blob (git hash-object --no-filters)
    ids / parity                                                           delegated to dagt1_manifest.py unchanged (see its --help)

Manifest (dag_read.load_manifest's schema): {"seeds": [0, 1], "arms": {"DAG": {"0": E, "1": E}, "DAG0": {...}}, "bank": {...}, "eval141_clip_ids", "train_clip_ids"}.
   reader slot DAG := DAG2 (run M2b_s<S>_dag2, tag DAG2_s<S>_...), DAG0 := DAG2-0 (run M2b_s<S>_dag20, tag DAG20_s<S>_...).  dag2_read.py fills its ORC0 slot with DAG0's own entry (fixed-RF mode).
   E = {"ckpt_tail", "train_dir", "token", "t1", "t1_logged", "t0" (box 2's standard eval, R-T0), "t0_fresh" (dev-box CPU --ages fresh, K4's comparator)}.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DAGT1_DIR = os.environ.get("DAG2T1_DAGT1_DIR", "E:/Projects/TanitAD-artifacts/dagt1")
D1M_SHA256 = "10c28ae0cd322f15da34a27ce4e739f57188e493755b8adf3ed1553acbb1e90a"          # dagt1_manifest.py as it ran PREREG_V8F_DAGGER's chain
D1P_SHA256 = "00bf2049c96d032154f0e48375a5c3650aa6fbe8930caedccc50eda24e701381"          # dagt1_pullcheck.py (its dependency)
sys.path.insert(0, HERE)
sys.path.insert(1, DAGT1_DIR)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for _n, _w in (("dagt1_manifest.py", D1M_SHA256), ("dagt1_pullcheck.py", D1P_SHA256)):
    _p = os.path.join(DAGT1_DIR, _n)
    if not os.path.isfile(_p) or _sha(_p) != _w:
        raise SystemExit(f"REFUSED: {_p} is not the pinned file (sha256 {_w[:12]}...): dag2t1_manifest.py imports it unchanged")
import dag2t1_pullcheck as PC  # noqa: E402
import dagt1_manifest as D1M  # noqa: E402

ARM_READER = {"dag2": "DAG", "dag20": "DAG0"}                      # the DAGGER reader's slot names
ARM_TAG = {"dag2": "DAG2", "dag20": "DAG20"}                        # the rollouts' --tag stems
DEFAULT_RUNS = "dag2:0,dag20:0,dag2:1,dag20:1"                      # the PREREG's order: DAG2 s0, DAG2-0 s0, DAG2 s1, DAG2-0 s1
DAG_CKPT_SHA = {0: "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", 1: "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"}   # v8f-dagger/M2b_s{S}_dag/SHA256SUMS.txt
DAG_READ_SHA = "3393fdd56bd561fd5181a8219d2c75d700036b9373da6ec12a8bcba51decde3d"
fwd, tail2, sums_of = D1M.fwd, D1M.tail2, D1M.sums_of


def tag_of(pfx: str, arm: str, seed: int, limit: str) -> str:
    return f"{pfx}{ARM_TAG[arm]}_s{seed}_m10_{limit}_heads"


def t0_tag(pfx: str, arm: str, seed: int) -> str:
    return f"{pfx}{PC.eval_tag(arm, seed)}-fresh"


def parse_runs(s: str) -> list[tuple[str, int]]:
    out = []
    for it in s.split(","):
        arm, _, seed = it.partition(":")
        if arm not in ARM_READER or not seed.isdigit():
            raise SystemExit(f"--runs: expected arm:seed with arm in {sorted(ARM_READER)}, got {it!r}")
        out.append((arm, int(seed)))
    return out


def t1_entry(out: str, pfx: str, arm: str, seed: int, limit: str) -> dict:
    t = tag_of(pfx, arm, seed, limit)
    return {"tag": t, "ticks": fwd(f"{out}/ticks/ticks_{t}.npz"), "dump_dir": fwd(f"{out}/dump_{t}"), "json": fwd(f"{out}/t1v8f_{t}.json")}


def build(art: str, out: str, pfx: str, runs: list[tuple[str, int]], ids_dir: str, t0_fresh_dir: str | None) -> dict:
    arms: dict[str, dict] = {"DAG": {}, "DAG0": {}}
    for arm, seed in runs:
        run = PC.run_name(arm, seed)
        d = fwd(f"{art}/{run}")
        tg = PC.eval_tag(arm, seed)
        e = {"ckpt_tail": f"{run}/ckpt_last.pt", "train_dir": d, "token": f"{d}/launch_gate_token.json",
             "t1": t1_entry(out, pfx, arm, seed, "causal"), "t1_logged": t1_entry(out, pfx, arm, seed, "logged"),
             "t0": {"json": f"{d}/eval_{tg}.json", "npz": f"{d}/eval_{tg}.npz"}}
        if t0_fresh_dir:
            e["t0_fresh"] = {"json": fwd(f"{t0_fresh_dir}/eval_{t0_tag(pfx, arm, seed)}.json"), "npz": fwd(f"{t0_fresh_dir}/eval_{t0_tag(pfx, arm, seed)}.npz")}
        arms[ARM_READER[arm]][str(seed)] = e
    return {"seeds": [0, 1], "arms": arms,
            "bank": {str(s): {"report": fwd(f"{art}/bank_s{s}/PROVENANCE.json"), "bank_file": fwd(f"{art}/bank_s{s}/bank.json")} for s in (0, 1)},
            "eval141_clip_ids": fwd(f"{ids_dir}/eval141_clip_ids.json"), "train_clip_ids": fwd(f"{ids_dir}/train_clip_ids.json"),
            "provenance": {"prereg": "Project Steering/PREREG_V8F_DAGGER2.md @ 1446333d (Amendment 1, A1-2026-10-27; inherits PREREG_V8F_DAGGER + Amendments 1, 2), claim row H-V8F-DAGGER2-1",
                           "built_by": "dag2t1_manifest.py (the dev-box T1 chain, dag2_t1_chain.sh)",
                           "slots": "DAG := DAG2 (run M2b_s<S>_dag2), DAG0 := DAG2-0 (run M2b_s<S>_dag20); the reader's ORC0 slot is filled by dag2_read.py with DAG0's own entry (fixed RF)",
                           "t1": "v8f_t1_eval.py run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --dump-ticks --dump-heads, plain (no post-map)",
                           "t0": "box 2's standard eval (--ages cycle): R-T0, T0-CKPT", "t0_fresh": "dev-box CPU --ages fresh eval: K4's comparator"}}


def verify(m: dict, n_dump_min: int, dag_art: str | None = None) -> list[str]:
    """Every file the manifest points at exists, every entry is THIS arm's (no path or checkpoint shared between DAG2 and DAG2-0, none equal to DAG's), the rollouts' and fresh-T0 done records name the
    pulled checkpoint's sha256, and both iteration-2 banks are COMPLETE."""
    bad: list[str] = []

    def need_file(p, what):
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            bad.append(f"{what}: {p} missing or empty")

    if sorted(m["arms"]) != ["DAG", "DAG0"]:
        bad.append(f"arms {sorted(m['arms'])}: exactly DAG (:= DAG2) and DAG0 (:= DAG2-0) are read")
    seen: dict[str, str] = {}
    for arm_r in ("DAG", "DAG0"):
        per = m["arms"].get(arm_r, {})
        arm = "dag2" if arm_r == "DAG" else "dag20"
        for seed in ("0", "1"):
            e = per.get(seed)
            lab = f"{arm_r} s{seed}"
            if e is None:
                bad.append(f"{lab}: no entry")
                continue
            d = e["train_dir"]
            want_run = PC.run_name(arm, int(seed))
            if os.path.basename(d.rstrip("/")) != want_run or e["ckpt_tail"] != f"{want_run}/ckpt_last.pt":
                bad.append(f"{lab}: train_dir / ckpt_tail {d!r} / {e['ckpt_tail']!r} is not {want_run}")
            for p in (e["token"], e["t0"]["json"], e["t0"]["npz"]):
                need_file(p, f"{lab} file")
            for f in ("argv.json", "summary.json", "selfcheck.json", "ckpt_last.pt"):
                need_file(f"{d}/{f}", f"{lab} {f}")
            sums = sums_of(d)
            ck = f"{d}/ckpt_last.pt"
            sha = _sha(ck) if os.path.isfile(ck) else None
            if sha is None or sums.get("ckpt_last.pt") != sha:
                bad.append(f"{lab}: ckpt_last.pt hashes to {str(sha)[:12]}, SHA256SUMS.txt lists {str(sums.get('ckpt_last.pt'))[:12]}")
            if sha:
                if sha in seen:
                    bad.append(f"{lab}: its checkpoint sha256 {sha[:12]} is also {seen[sha]}'s: two arms cannot share a checkpoint")
                seen[sha] = lab
                if sha == DAG_CKPT_SHA[int(seed)]:
                    bad.append(f"{lab}: its checkpoint sha256 {sha[:12]} IS DAG_s{seed}: nothing was fine-tuned")
            for kind in ("t1", "t1_logged"):
                t1 = e[kind]
                for k in ("ticks", "json"):
                    need_file(t1[k], f"{lab} {kind} {k}")
                nd = len(glob.glob(os.path.join(t1["dump_dir"], "ep*.npz")))
                if nd < n_dump_min:
                    bad.append(f"{lab} {kind}: {nd} ep*.npz in {t1['dump_dir']} (< {n_dump_min})")
                if (tail2(json.load(open(t1["json"], encoding="utf-8")).get("t1v8f", {}).get("ckpt", "")) if os.path.isfile(t1["json"]) else None) != e["ckpt_tail"]:
                    bad.append(f"{lab} {kind}: the rollout json says it scored another checkpoint than {e['ckpt_tail']}")
                dn = os.path.join(os.path.dirname(t1["json"]), f"done_{t1['tag']}.txt")
                try:
                    dr = json.load(open(dn, encoding="utf-8"))
                    if dr.get("ckpt_sha256") != sha:
                        bad.append(f"{lab} {kind}: done record {os.path.basename(dn)} names checkpoint {str(dr.get('ckpt_sha256'))[:12]}, the pulled one is {str(sha)[:12]}")
                except (OSError, ValueError) as ex:
                    bad.append(f"{lab} {kind}: done record {dn} unreadable: {ex}")
            tf = e.get("t0_fresh")
            if not tf:
                bad.append(f"{lab}: no t0_fresh entry (K4's comparator)")
            else:
                for k in ("json", "npz"):
                    need_file(tf[k], f"{lab} t0_fresh {k}")
                for k, w in (("json", "fresh"), ("t0", "cycle")):
                    pj = tf["json"] if k == "json" else e["t0"]["json"]
                    if os.path.isfile(pj):
                        ages = json.load(open(pj, encoding="utf-8")).get("ages")
                        if ages != w:
                            bad.append(f"{lab}: {'fresh T0' if k == 'json' else 'standard T0'} eval json says ages {ages!r}, expected {w!r}")
                        ck_t = tail2(json.load(open(pj, encoding="utf-8")).get("ckpt", ""))
                        if ck_t != e["ckpt_tail"]:
                            bad.append(f"{lab}: the {'fresh' if k == 'json' else 'standard'} T0 eval scored {ck_t!r}, not {e['ckpt_tail']}")
            for k, v in (("train_dir", d), ("t1.dump_dir", e["t1"]["dump_dir"]), ("t0.json", e["t0"]["json"])):
                key = f"{k}={v}"
                if key in seen:
                    bad.append(f"{lab}: {k} {v} is shared with {seen[key]}")
                seen[key] = lab
    for s in ("0", "1"):
        be = m.get("bank", {}).get(s)
        if not be:
            bad.append(f"bank s{s}: no entry")
            continue
        art = os.path.dirname(os.path.dirname(be["report"]))
        st, why, _ = PC.check_bank_status(art, int(s), expect_shards=4)
        if st != "COMPLETE":
            bad.append(f"bank s{s}: {st}: {why}")
        if os.path.dirname(be["bank_file"]) != os.path.dirname(be["report"]):
            bad.append(f"bank s{s}: bank_file is not the bank.json beside the report")
    for k in ("eval141_clip_ids", "train_clip_ids"):
        need_file(m[k], k)
    return bad


def initsha(art: str, dag_art: str, runs: list[tuple[str, int]], b2_json: str | None) -> dict[int, str]:
    """The checkpoint each seed's DAG2 and DAG2-0 started from, from the runs' own records; all four must agree per seed and equal DAG_s<S> (DAG's SUMS and the pinned value)."""
    res: dict[int, str] = {}
    for seed in (0, 1):
        vals = set()
        for arm in ("dag2", "dag20"):
            if (arm, seed) not in runs:
                continue
            d = os.path.join(art, PC.run_name(arm, seed))
            for f in ("selfcheck.json", "summary.json"):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    vals |= set(D1M.init_sha(D1M.read_json(p)))
        if len(vals) != 1:
            raise SystemExit(f"seed {seed}: the runs' own records disagree about (or do not carry) the init checkpoint: {sorted(vals)}")
        res[seed] = next(iter(vals))
        listed = sums_of(os.path.join(dag_art, f"M2b_s{seed}_dag")).get("ckpt_last.pt")
        if res[seed] != listed or res[seed] != DAG_CKPT_SHA[seed]:
            raise SystemExit(f"seed {seed}: the runs' recorded init {res[seed][:16]} is not DAG_s{seed} (DAG's SUMS {str(listed)[:16]}, pinned {DAG_CKPT_SHA[seed][:16]})")
        if b2_json:
            b2 = D1M.read_json(b2_json)
            v = b2.get(f"M2b_s{seed}_dag")
            if v is None:
                raise SystemExit(f"{b2_json} has no M2b_s{seed}_dag")
            if v != res[seed]:
                raise SystemExit(f"seed {seed}: the chain's measured DAG sha256 {v[:16]} != the runs' recorded init {res[seed][:16]}")
    return res


def dagbase(dag_art: str, dag_t1: str, n_dump_min: int = 141) -> list[str]:
    bad = []
    rj = os.path.join(dag_t1, "read", "DAG_READ.json")
    if not os.path.isfile(rj) or _sha(rj) != DAG_READ_SHA:
        bad.append(f"{rj} is not the DAG read the DAG2 chain is pinned to (sha256 {DAG_READ_SHA[:12]}...)")
    else:
        oc = json.load(open(rj, encoding="utf-8")).get("OUTCOME", {}).get("OUTCOME")
        if oc != "ROBUST":
            bad.append(f"{rj}: OUTCOME {oc!r}, not ROBUST")
    for s in (0, 1):
        d = os.path.join(dag_art, f"M2b_s{s}_dag")
        ck = os.path.join(d, "ckpt_last.pt")
        if not os.path.isfile(ck):
            bad.append(f"DAG s{s}: {ck} absent")
            continue
        got = _sha(ck)
        if got != DAG_CKPT_SHA[s] or sums_of(d).get("ckpt_last.pt") != got:
            bad.append(f"DAG s{s}: ckpt_last.pt hashes to {got[:12]}, pinned {DAG_CKPT_SHA[s][:12]}, SUMS {str(sums_of(d).get('ckpt_last.pt'))[:12]}")
        t = f"DAG_s{s}_m10_causal_heads"
        try:
            dr = json.load(open(os.path.join(dag_t1, f"done_{t}.txt"), encoding="utf-8"))
            if dr.get("ckpt_sha256") != got:
                bad.append(f"DAG s{s}: done_{t}.txt names {str(dr.get('ckpt_sha256'))[:12]}, DAG's checkpoint is {got[:12]}")
        except (OSError, ValueError) as ex:
            bad.append(f"DAG s{s}: done_{t}.txt unreadable: {ex}")
        n = len(glob.glob(os.path.join(dag_t1, f"dump_{t}", "ep*.npz")))
        if n < n_dump_min:
            bad.append(f"DAG s{s}: {n} ep*.npz in dump_{t} (< {n_dump_min})")
    return bad


def treetip(tree: str, git_dir: str, tip: str) -> dict:
    """tree .py files vs the tip's blobs.  A file the tip also has must be byte-identical (git hash-object --no-filters == the tip's blob id).  A tree file the tip lacks is reported."""
    g = ["git", "-C", git_dir, "-c", "safe.directory=*"]
    ls = subprocess.run(g + ["ls-tree", "-r", tip, "--", "stack", "taniteval"], capture_output=True, text=True, check=True).stdout.splitlines()
    tip_blob = {}
    for ln in ls:
        meta, _, path = ln.partition("\t")
        tip_blob[path] = meta.split()[2]
    files = []
    for root in ("stack", "taniteval"):
        for dp, _dn, fn in os.walk(os.path.join(tree, root)):
            for f in fn:
                if f.endswith(".py"):
                    files.append(os.path.relpath(os.path.join(dp, f), tree).replace("\\", "/"))
    files.sort()
    hs = subprocess.run(["git", "hash-object", "--no-filters", "--stdin-paths"], input="\n".join(os.path.join(tree, f) for f in files) + "\n", capture_output=True, text=True, cwd=tree, check=True).stdout.split()
    ident, differ, absent = [], [], []
    for f, h in zip(files, hs):
        if f not in tip_blob:
            absent.append(f)
        elif tip_blob[f] == h:
            ident.append(f)
        else:
            differ.append(f)
    return {"tip": tip, "n_tree_py": len(files), "identical": len(ident), "differ": differ, "not_in_tip": absent}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--art", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids-dir", required=True)
    p.add_argument("--dag-art", required=True)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--pfx", default="")
    p.add_argument("--t0-fresh-dir")
    p.add_argument("--write")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n-dump-min", type=int, default=141)
    p = sub.add_parser("initsha")
    p.add_argument("--art", required=True)
    p.add_argument("--dag-art", required=True)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--b2-json")
    p = sub.add_parser("dagbase")
    p.add_argument("--dag-art", required=True)
    p.add_argument("--dag-t1", required=True)
    p = sub.add_parser("treetip")
    p.add_argument("--tree", required=True)
    p.add_argument("--git-dir", required=True)
    p.add_argument("--tip", required=True)
    for nm in ("ids", "parity"):
        sub.add_parser(nm, add_help=False)
    a, rest = ap.parse_known_args(argv)
    if a.cmd in ("ids", "parity"):
        return D1M.main([a.cmd] + rest)
    if a.cmd == "initsha":
        r = initsha(a.art, a.dag_art, parse_runs(a.runs), a.b2_json)
        print(" ".join(f"{s}={r[s]}" for s in (0, 1)))
        return 0
    if a.cmd == "dagbase":
        bad = dagbase(a.dag_art, a.dag_t1)
        if bad:
            print("DAG's banked outputs REFUSED:")
            for b in bad:
                print("  -", b)
            return 1
        print("DAG's banked outputs present and pinned: DAG_READ.json (ROBUST, sha256 3393fdd5...), both DAG checkpoints, both causal dump sets (done records name the checkpoints): read, not re-run")
        return 0
    if a.cmd == "treetip":
        r = treetip(a.tree, a.git_dir, a.tip)
        print(f"tree vs tip {a.tip}: {r['identical']} of {r['n_tree_py']} .py identical to the tip's blobs, {len(r['differ'])} differ, {len(r['not_in_tip'])} not in the tip")
        for f in r["differ"]:
            print("  DIFFERS:", f)
        return 1 if r["differ"] else 0
    runs = parse_runs(a.runs)
    m = build(a.art, a.out, a.pfx, runs, a.ids_dir, a.t0_fresh_dir)
    if a.dry_run:
        print(json.dumps(m, indent=1))
        return 0
    bad = verify(m, a.n_dump_min)
    if bad:
        print("MANIFEST REFUSED:")
        for b in bad:
            print("  -", b)
        return 1
    os.makedirs(os.path.dirname(a.write) or ".", exist_ok=True)
    with open(a.write, "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, indent=1)
    print(f"wrote {a.write} ({sum(len(v) for v in m['arms'].values())} arm-seed entries)")
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
