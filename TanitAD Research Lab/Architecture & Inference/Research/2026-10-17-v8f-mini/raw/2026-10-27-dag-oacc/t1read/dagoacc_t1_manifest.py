#!/usr/bin/env python3
"""dagoacc_t1_manifest.py -- the dev-box CPU T1 chain of PREREG_V8F_DAG_OACC: builds the reader manifest runs.json for dagoacc_read.py, and the small guards the chain needs.   CPU only; read-only on every input.
Sibling of flatdag_t1_manifest.py.  Utilities that did not change (the tool-tree import-closure `parity`, the tree-vs-tip check `treetip`, DAG's banked outputs `dagbase`) are DELEGATED to the sha-pinned
dag2t1_manifest.py, not copied.  There is NO `ids` command and NO bank: DAG-OACC trains on DAG's BANKED iteration-1 bank (read-only; its sha256 identity is C-O1), so the id lists the bank controls need are not built.

    manifest   --art ART --out OUT --dag-art DIR --dag-t1 DIR --inputs DIR (--write PATH | --dry-run) [--runs dagoacc:0,dagoacc:1] [--n-dump-min N]
    initsha    --art ART [--runs ..]      prints "0=<sha256> 1=<sha256>": the checkpoint DAG-OACC started from = M2b_s<S> (the runs' own records == the pinned value, DAG's init)
    dagbase / treetip / parity            delegated unchanged to dag2t1_manifest.py

Manifest (read by dagoacc_read.py; the paths are absolute):
   {"seeds": [0, 1], "arms": {"DAGOACC": {"0": E, "1": E}}, "control": {"DAG": {"0": CE, "1": CE}}, "wacc": {...}, "provenance": {...}}
   E  = {"ckpt_tail", "train_dir", "token", "gate_dir", "t1" (the decisive --limit causal rollout), "t0" (box 2's standard eval: R-T0 and the T0 families, beside)}      NEW arm, M2b_s<S>_dagoacc
   CE = {"ckpt_tail", "train_dir", "t1" (DAG's BANKED rollout), "t0" (DAG's banked standard eval)}                                                                   the CONTROL, BANKED, read never re-run
   There is no t1_logged, no fresh-age T0 and no oracle eval: PREREG_V8F_DAG_OACC section 5 names ONE dev-box step, the 2 new T1-kin rollouts and the reader.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D2_DIR = os.environ.get("DOT1_DAG2T1_DIR", "E:/Projects/TanitAD-artifacts/dag2t1")
D2M_SHA256 = "0bfb74c423a4018745b50c832aea83b13f3045a612d43047075577bc999fd897"          # dag2t1_manifest.py as it ran PREREG_V8F_DAGGER2's chain
D2P_SHA256 = "2fc1b042d3a2a3b78431370c6bcd10d9ab62112e51fea1c106b713124b3ffd44"          # dag2t1_pullcheck.py (its dependency)
ART = "E:/Projects/TanitAD-artifacts"
sys.path.insert(0, HERE)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for _n, _w in (("dag2t1_manifest.py", D2M_SHA256), ("dag2t1_pullcheck.py", D2P_SHA256)):
    _p = os.path.join(D2_DIR, _n)
    if not os.path.isfile(_p) or _sha(_p) != _w:
        raise SystemExit(f"REFUSED: {_p} is not the pinned file (sha256 {_w[:12]}...): dagoacc_t1_manifest.py delegates to it unchanged")
sys.path.insert(1, D2_DIR)
_spec = importlib.util.spec_from_file_location("dag2t1_manifest", os.path.join(D2_DIR, "dag2t1_manifest.py"))
D2M = importlib.util.module_from_spec(_spec)
sys.modules["dag2t1_manifest"] = D2M
_spec.loader.exec_module(D2M)
import dagoacc_t1_pullcheck as PC  # noqa: E402

D1M = D2M.D1M
fwd, tail2, sums_of = D2M.fwd, D2M.tail2, D2M.sums_of
DEFAULT_RUNS = "dagoacc:0,dagoacc:1"                                # the PREREG's order (section 5): s0 then s1


def tag_of(seed: int) -> str:
    return f"DAGOACC_s{seed}_m10_causal_heads"


def dag_tag(seed: int) -> str:
    return f"DAG_s{seed}_m10_causal_heads"


def parse_runs(s: str) -> list[tuple[str, int]]:
    out = []
    for it in s.split(","):
        arm, _, seed = it.partition(":")
        if arm not in PC.ARMS or not seed.isdigit() or int(seed) not in (0, 1):
            raise SystemExit(f"--runs: expected dagoacc:<0|1>, got {it!r}")
        out.append((arm, int(seed)))
    return out


def t1_entry(out: str, seed: int) -> dict:
    t = tag_of(seed)
    return {"tag": t, "ticks": fwd(f"{out}/ticks/ticks_{t}.npz"), "dump_dir": fwd(f"{out}/dump_{t}"), "json": fwd(f"{out}/t1v8f_{t}.json"), "argv": fwd(f"{out}/argv/{t}.argv")}


def build(art: str, out: str, runs: list[tuple[str, int]], dag_art: str, dag_t1: str, inputs: str) -> dict:
    new, ctl = {}, {}
    for arm, seed in runs:
        run = PC.run_name(arm, seed)
        d = fwd(f"{art}/{run}")
        tg = PC.eval_tag(arm, seed)
        new[str(seed)] = {"ckpt_tail": f"{run}/ckpt_last.pt", "train_dir": d, "token": f"{d}/launch_gate_token.json", "gate_dir": f"{d}/gate", "t1": t1_entry(out, seed),
                          "t0": {"json": f"{d}/eval_{tg}.json", "npz": f"{d}/eval_{tg}.npz"}}
    for seed in (0, 1):
        dd = fwd(f"{dag_art}/M2b_s{seed}_dag")
        t = dag_tag(seed)
        ctl[str(seed)] = {"ckpt_tail": f"M2b_s{seed}_dag/ckpt_last.pt", "train_dir": dd,
                          "t1": {"tag": t, "ticks": fwd(f"{dag_t1}/ticks/ticks_{t}.npz"), "dump_dir": fwd(f"{dag_t1}/dump_{t}"), "json": fwd(f"{dag_t1}/t1v8f_{t}.json"),
                                 "done": fwd(f"{dag_t1}/done_{t}.txt")},
                          "t0": {"json": f"{dd}/eval_M2b-dag_s{seed}.json", "npz": f"{dd}/eval_M2b-dag_s{seed}.npz"}}
    return {"seeds": [0, 1], "arms": {"DAGOACC": new}, "control": {"DAG": ctl},
            "wacc": {"box": fwd(f"{art}/wacc/w_acc.json"), "txt": fwd(f"{art}/wacc/w_acc.txt"), "offline": fwd(f"{inputs}/wacc_offline_DAG_s0.json")},
            "provenance": {"prereg": "Project Steering/PREREG_V8F_DAG_OACC.md @ 95063711, claim row H-V8F-DAG-OACC-1",
                           "built_by": "dagoacc_t1_manifest.py (the dev-box T1 chain, dagoacc_t1_chain.sh)",
                           "slots": "DAGOACC := DAG-OACC (run M2b_s<S>_dagoacc); the CONTROL DAG is the banked run (v8f-dagger / v8f-dagger-t1), read never re-run",
                           "t1": "v8f_t1_eval.py run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks --dump-heads, plain (no post-map); FLAT's banked rollout argv except --ckpt/--tag/--out-dir/--dump-ticks",
                           "t0": "box 2's standard eval (--ages cycle): R-T0 and the four families at T0 (beside)"}}


def verify(m: dict, n_dump_min: int, dag_art: str = ART + "/v8f-dagger") -> list[str]:
    """Every file the manifest points at exists, every entry is THIS arm's (no path or checkpoint shared between DAG-OACC and DAG, none equal to DAG's or M2b's), the rollouts' done records name the pulled
    checkpoint's sha256, the rollout argv was written, the w_acc record and both of DAG's banked iteration-1 banks are COMPLETE, and the control's banked rollout is DAG's."""
    bad: list[str] = []

    def need_file(p, what):
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            bad.append(f"{what}: {p} missing or empty")

    if sorted(m["arms"]) != ["DAGOACC"] or sorted(m.get("control", {})) != ["DAG"]:
        bad.append(f"arms {sorted(m['arms'])} / control {sorted(m.get('control', {}))}: exactly DAGOACC (new) and DAG (banked control) are read")
    seen: dict[str, str] = {}
    for seed in ("0", "1"):
        e = m["arms"].get("DAGOACC", {}).get(seed)
        lab = f"DAGOACC s{seed}"
        if e is None:
            bad.append(f"{lab}: no entry")
            continue
        d = e["train_dir"]
        want_run = PC.run_name("dagoacc", int(seed))
        if os.path.basename(d.rstrip("/")) != want_run or e["ckpt_tail"] != f"{want_run}/ckpt_last.pt":
            bad.append(f"{lab}: train_dir / ckpt_tail {d!r} / {e['ckpt_tail']!r} is not {want_run}")
        for p in (e["token"], e["t0"]["json"], e["t0"]["npz"]):
            need_file(p, f"{lab} file")
        for f in ("argv.json", "summary.json", "selfcheck.json", "ckpt_last.pt"):
            need_file(f"{d}/{f}", f"{lab} {f}")
        need_file(f"{e['gate_dir']}/selfcheck.json", f"{lab} gate record (C-O2)")
        sums = sums_of(d)
        ck = f"{d}/ckpt_last.pt"
        sha = _sha(ck) if os.path.isfile(ck) else None
        if sha is None or sums.get("ckpt_last.pt") != sha:
            bad.append(f"{lab}: ckpt_last.pt hashes to {str(sha)[:12]}, SHA256SUMS.txt lists {str(sums.get('ckpt_last.pt'))[:12]}")
        if sha:
            if sha in seen:
                bad.append(f"{lab}: its checkpoint sha256 {sha[:12]} is also {seen[sha]}'s: two arms cannot share a checkpoint")
            seen[sha] = lab
            if sha == PC.DAG_CKPT_SHA[int(seed)]:
                bad.append(f"{lab}: its checkpoint sha256 {sha[:12]} IS DAG_s{seed}: nothing was fine-tuned with the lever")
            if sha == PC.M2B_CKPT_SHA[int(seed)]:
                bad.append(f"{lab}: its checkpoint sha256 {sha[:12]} IS M2b_s{seed}")
        t1 = e["t1"]
        for k in ("ticks", "json", "argv"):
            need_file(t1[k], f"{lab} t1 {k}")
        nd = len(glob.glob(os.path.join(t1["dump_dir"], "ep*.npz")))
        if nd < n_dump_min:
            bad.append(f"{lab} t1: {nd} ep*.npz in {t1['dump_dir']} (< {n_dump_min})")
        if (tail2(json.load(open(t1["json"], encoding="utf-8")).get("t1v8f", {}).get("ckpt", "")) if os.path.isfile(t1["json"]) else None) != e["ckpt_tail"]:
            bad.append(f"{lab} t1: the rollout json says it scored another checkpoint than {e['ckpt_tail']}")
        dn = os.path.join(os.path.dirname(t1["json"]), f"done_{t1['tag']}.txt")
        try:
            dr = json.load(open(dn, encoding="utf-8"))
            if dr.get("ckpt_sha256") != sha:
                bad.append(f"{lab} t1: done record {os.path.basename(dn)} names checkpoint {str(dr.get('ckpt_sha256'))[:12]}, the pulled one is {str(sha)[:12]}")
        except (OSError, ValueError) as ex:
            bad.append(f"{lab} t1: done record {dn} unreadable: {ex}")
        for k, v in (("train_dir", d), ("t1.dump_dir", t1["dump_dir"]), ("t0.json", e["t0"]["json"])):
            key = f"{k}={v}"
            if key in seen:
                bad.append(f"{lab}: {k} {v} is shared with {seen[key]}")
            seen[key] = lab
        c = m["control"].get("DAG", {}).get(seed)
        if c is None:
            bad.append(f"DAG s{seed}: no control entry")
            continue
        cck = f"{c['train_dir']}/ckpt_last.pt"
        if not os.path.isfile(cck) or _sha(cck) != PC.DAG_CKPT_SHA[int(seed)]:
            bad.append(f"DAG s{seed} (control): {cck} is absent or is not the banked DAG_s{seed} checkpoint")
        for p in (c["t1"]["ticks"], c["t1"]["json"], c["t1"]["done"], c["t0"]["json"], c["t0"]["npz"]):
            need_file(p, f"DAG s{seed} (control) file")
        if len(glob.glob(os.path.join(c["t1"]["dump_dir"], "ep*.npz"))) < n_dump_min:
            bad.append(f"DAG s{seed} (control): fewer than {n_dump_min} ep*.npz in {c['t1']['dump_dir']}")
        st, why = PC.dagbank_status(dag_art, int(seed))
        if st != "COMPLETE":
            bad.append(f"DAG's iteration-1 bank s{seed}: {st}: {why}")
    st, w = PC.wacc_status(os.path.dirname(os.path.dirname(m["wacc"]["box"])))
    if st != "OK":
        bad.append(f"the w_acc record: {st}: {w}")
    need_file(m["wacc"]["offline"], "C-O3's offline reference")
    return bad


def initsha(art: str, runs: list[tuple[str, int]]) -> dict[int, str]:
    """The checkpoint each seed's DAG-OACC started from, from the runs' own records; it must equal M2b_s<S> (the pinned value, DAG's init too)."""
    res: dict[int, str] = {}
    for seed in (0, 1):
        vals = set()
        if ("dagoacc", seed) in runs:
            d = os.path.join(art, PC.run_name("dagoacc", seed))
            for f in ("selfcheck.json", "summary.json"):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    vals |= set(D1M.init_sha(D1M.read_json(p)))
        if len(vals) != 1:
            raise SystemExit(f"seed {seed}: the run's own records disagree about (or do not carry) the init checkpoint: {sorted(vals)}")
        res[seed] = next(iter(vals))
        if res[seed] != PC.M2B_CKPT_SHA[seed]:
            raise SystemExit(f"seed {seed}: the run's recorded init {res[seed][:16]} is not M2b_s{seed} (pinned {PC.M2B_CKPT_SHA[seed][:16]})")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--art", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dag-art", default=ART + "/v8f-dagger")
    p.add_argument("--dag-t1", default=ART + "/v8f-dagger-t1")
    p.add_argument("--inputs", default=HERE.replace("\\", "/") + "/inputs")
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--write")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n-dump-min", type=int, default=141)
    p = sub.add_parser("initsha")
    p.add_argument("--art", required=True)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    for nm in ("dagbase", "treetip", "parity"):
        sub.add_parser(nm, add_help=False)
    a, rest = ap.parse_known_args(argv)
    if a.cmd in ("dagbase", "treetip", "parity"):
        return D2M.main([a.cmd] + rest)
    if a.cmd == "initsha":
        r = initsha(a.art, parse_runs(a.runs))
        print(" ".join(f"{s}={r[s]}" for s in (0, 1)))
        return 0
    if a.cmd == "manifest":
        m = build(a.art, a.out, parse_runs(a.runs), a.dag_art, a.dag_t1, a.inputs)
        if a.dry_run:
            print(json.dumps({"runs": [tag_of(s) for _, s in parse_runs(a.runs)], "slots": {"DAGOACC": "DAG-OACC (new)", "control DAG": "banked"}, "wacc": m["wacc"]}, indent=1))
            return 0
        bad = verify(m, a.n_dump_min, a.dag_art)
        if bad:
            print("MANIFEST REFUSED:")
            for b in bad:
                print("  -", b)
            return 1
        os.makedirs(os.path.dirname(os.path.abspath(a.write)), exist_ok=True)
        with open(a.write, "w", encoding="utf-8", newline="\n") as f:
            json.dump(m, f, indent=1)
        print(f"manifest written {a.write} sha256 {_sha(a.write)} (2 DAG-OACC runs + 2 banked DAG controls, w_acc record and 2 iteration-1 banks verified COMPLETE)")
        return 0
    return 2


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
