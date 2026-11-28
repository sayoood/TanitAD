#!/usr/bin/env python3
"""flatdag_t1_manifest.py -- the dev-box CPU T1 chain of PREREG_V8F_FLATDAG: builds the reader manifest runs.json for flatdag_read.py, and the small guards the chain needs.   CPU only; read-only on every input.
Sibling of dag2t1_manifest.py (which ran PREREG_V8F_DAGGER2's chain).  Utilities that did not change (the eval / TRAIN id lists `ids`, the tool-tree import-closure `parity`, the tree-vs-tip check `treetip`,
DAG's banked outputs `dagbase`) are DELEGATED to the sha-pinned dag2t1_manifest.py, not copied.

    manifest   --art ART --out OUT --ids-dir DIR (--write PATH | --dry-run) [--runs flatdag:0,flatdag0:0,flatdag:1,flatdag0:1] [--n-dump-min N]
    initsha    --art ART [--runs ..] [--flat-dir DIR]      prints "0=<sha256> 1=<sha256>": the checkpoint FLAT-DAG / FLAT-DAG-0 started from = FLAT_s<S> (the runs' own records == the pinned value == the dev box's
                                                           v8f-mini-flat/M2b_s<S>_flat.pt, re-hashed)
    flatbase   --dagreread DIR                              FLAT's BANKED B-2 material (packed DAG and FLAT dumps, the B-2 table, FLAT's rollout argv) is present and is the pinned files
    dagbase / treetip / ids / parity                        delegated unchanged to dag2t1_manifest.py

Manifest (dag_read.load_manifest's schema): {"seeds": [0, 1], "arms": {"DAG": {"0": E, "1": E}, "DAG0": {...}}, "bank": {...}, "eval141_clip_ids", "train_clip_ids"}.
   reader slot DAG := FLAT-DAG (run M2b_s<S>_flatdag, tag FLATDAG_s<S>_...), DAG0 := FLAT-DAG-0 (run M2b_s<S>_flatdag0, tag FLATDAG0_s<S>_...).
   E = {"ckpt_tail", "train_dir", "token", "t1" (the decisive --limit causal rollout), "t0" (box 2's standard eval, R-T0), "t0_oracle" (box 2's GT-code oracle eval, C-F3)}.  There is no t1_logged and no t0_fresh:
   PREREG_V8F_FLATDAG section 5 names ONE dev-box step, the 4 new T1-kin rollouts and the reader.
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
D2_DIR = os.environ.get("FDT1_DAG2T1_DIR", "E:/Projects/TanitAD-artifacts/dag2t1")
D2M_SHA256 = "0bfb74c423a4018745b50c832aea83b13f3045a612d43047075577bc999fd897"          # dag2t1_manifest.py as it ran PREREG_V8F_DAGGER2's chain
D2P_SHA256 = "2fc1b042d3a2a3b78431370c6bcd10d9ab62112e51fea1c106b713124b3ffd44"          # dag2t1_pullcheck.py (its dependency)
ART = "E:/Projects/TanitAD-artifacts"
FLAT_CKPT_SHA = {0: "8496934a2ce262c128a3e975da156de2f40e422925893c9979169a4c114493cc", 1: "6fdfc9b02ce87248c5315dd7604d3642a3b39f7ae0f593ec6b6ef5ef8df64c95"}   # v8f-mini-flat/M2b_s{S}_flat.pt
DAG_CKPT_SHA = {0: "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", 1: "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"}
FLAT_BANKED = {   # dagreread/ (MANIFEST_SHA256.txt of that package)
    "packed/dumps_DAG_s0_m10_causal.npz": "ed03adb499010b20aba3d96ab05d15dacaad00bb613987d941efbf402db42ac2",
    "packed/dumps_DAG_s1_m10_causal.npz": "706e15cc971eab25fc31c227dde0ecc3942f43deb01d4cc7ac4e1a343db4f32e",
    "packed/dumps_FLAT_s0_m10_causal.npz": "1e34d77f0e0d35c8a5608eab94ee23ad5648d012f45ac8a4589af67d63227d8e",
    "packed/dumps_FLAT_s1_m10_causal.npz": "ed89af3883d157d4e02e6ddc67c45e9e1425272c1a9a85a88e0d93cd304366e7",
    "raw/B2_T1kin_DAG_vs_FLAT.json": "dd2e0f36800cef09abbd4efb7a71c9788505172ba90a24d80ffe616bb97eade3",
    "argv/FLAT_s0_m10_causal_heads.argv": "aa74c65c5b5229d29050518cdd636d1ab1c7650be20989e2740d95e6801c2979",
    "argv/FLAT_s1_m10_causal_heads.argv": "206c777f26293e3263cb54b50b4d4d08ff5a514e7bd32286a4e2d8c1f9448f17",
}
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
        raise SystemExit(f"REFUSED: {_p} is not the pinned file (sha256 {_w[:12]}...): flatdag_t1_manifest.py delegates to it unchanged")
sys.path.insert(1, D2_DIR)
_spec = importlib.util.spec_from_file_location("dag2t1_manifest", os.path.join(D2_DIR, "dag2t1_manifest.py"))
D2M = importlib.util.module_from_spec(_spec)
sys.modules["dag2t1_manifest"] = D2M
_spec.loader.exec_module(D2M)
import flatdag_t1_pullcheck as PC  # noqa: E402

D1M = D2M.D1M
fwd, tail2, sums_of = D2M.fwd, D2M.tail2, D2M.sums_of
ARM_READER = {"flatdag": "DAG", "flatdag0": "DAG0"}                 # the DAGGER reader's slot names
ARM_TAG = {"flatdag": "FLATDAG", "flatdag0": "FLATDAG0"}            # the rollouts' --tag stems
DEFAULT_RUNS = "flatdag:0,flatdag0:0,flatdag:1,flatdag0:1"          # the PREREG's order: FLAT-DAG s0, FLAT-DAG-0 s0, FLAT-DAG s1, FLAT-DAG-0 s1


def tag_of(arm: str, seed: int) -> str:
    return f"{ARM_TAG[arm]}_s{seed}_m10_causal_heads"


def parse_runs(s: str) -> list[tuple[str, int]]:
    out = []
    for it in s.split(","):
        arm, _, seed = it.partition(":")
        if arm not in ARM_READER or not seed.isdigit():
            raise SystemExit(f"--runs: expected arm:seed with arm in {sorted(ARM_READER)}, got {it!r}")
        out.append((arm, int(seed)))
    return out


def t1_entry(out: str, arm: str, seed: int) -> dict:
    t = tag_of(arm, seed)
    return {"tag": t, "ticks": fwd(f"{out}/ticks/ticks_{t}.npz"), "dump_dir": fwd(f"{out}/dump_{t}"), "json": fwd(f"{out}/t1v8f_{t}.json"), "argv": fwd(f"{out}/argv/{t}.argv")}


def build(art: str, out: str, runs: list[tuple[str, int]], ids_dir: str) -> dict:
    arms: dict[str, dict] = {"DAG": {}, "DAG0": {}}
    for arm, seed in runs:
        run = PC.run_name(arm, seed)
        d = fwd(f"{art}/{run}")
        tg = PC.eval_tag(arm, seed)
        arms[ARM_READER[arm]][str(seed)] = {
            "ckpt_tail": f"{run}/ckpt_last.pt", "train_dir": d, "token": f"{d}/launch_gate_token.json", "t1": t1_entry(out, arm, seed),
            "t0": {"json": f"{d}/eval_{tg}.json", "npz": f"{d}/eval_{tg}.npz"}, "t0_oracle": {"json": f"{d}/eval_{tg}-tacorc.json", "npz": f"{d}/eval_{tg}-tacorc.npz"}}
    return {"seeds": [0, 1], "arms": arms,
            "bank": {str(s): {"report": fwd(f"{art}/bank_s{s}/PROVENANCE.json"), "bank_file": fwd(f"{art}/bank_s{s}/bank.json")} for s in (0, 1)},
            "eval141_clip_ids": fwd(f"{ids_dir}/eval141_clip_ids.json"), "train_clip_ids": fwd(f"{ids_dir}/train_clip_ids.json"),
            "provenance": {"prereg": "Project Steering/PREREG_V8F_FLATDAG.md @ 9b91b654 (sha256 8713d0a3...), claim row H-V8F-FLATDAG-1",
                           "built_by": "flatdag_t1_manifest.py (the dev-box T1 chain, flatdag_t1_chain.sh)",
                           "slots": "DAG := FLAT-DAG (run M2b_s<S>_flatdag), DAG0 := FLAT-DAG-0 (run M2b_s<S>_flatdag0); the banked DAG arm is read by the reader from v8f-dagger-t1/",
                           "t1": "v8f_t1_eval.py run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks --dump-heads, plain (no post-map); FLAT's B-2 rollout argv except --ckpt/--tag/--out-dir/--dump-ticks",
                           "t0": "box 2's standard eval (--ages cycle): R-T0 (beside)", "t0_oracle": "box 2's GT-code oracle eval: C-F3's identity row"}}


def verify(m: dict, n_dump_min: int) -> list[str]:
    """Every file the manifest points at exists, every entry is THIS arm's (no path or checkpoint shared between FLAT-DAG and FLAT-DAG-0, none equal to FLAT's or DAG's), the rollouts' done records name the pulled
    checkpoint's sha256, the rollout argv was written, and both FLAT banks are COMPLETE."""
    bad: list[str] = []

    def need_file(p, what):
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            bad.append(f"{what}: {p} missing or empty")

    if sorted(m["arms"]) != ["DAG", "DAG0"]:
        bad.append(f"arms {sorted(m['arms'])}: exactly DAG (:= FLAT-DAG) and DAG0 (:= FLAT-DAG-0) are read")
    seen: dict[str, str] = {}
    for arm_r in ("DAG", "DAG0"):
        per = m["arms"].get(arm_r, {})
        arm = "flatdag" if arm_r == "DAG" else "flatdag0"
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
            for p in (e["token"], e["t0"]["json"], e["t0"]["npz"], e["t0_oracle"]["npz"]):
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
                if sha == FLAT_CKPT_SHA[int(seed)]:
                    bad.append(f"{lab}: its checkpoint sha256 {sha[:12]} IS FLAT_s{seed}: nothing was fine-tuned")
                if sha == DAG_CKPT_SHA[int(seed)]:
                    bad.append(f"{lab}: its checkpoint sha256 {sha[:12]} IS DAG_s{seed}")
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


def initsha(art: str, runs: list[tuple[str, int]], flat_dir: str | None) -> dict[int, str]:
    """The checkpoint each seed's FLAT-DAG and FLAT-DAG-0 started from, from the runs' own records; all must agree per seed and equal FLAT_s<S> (the pinned value and, when flat_dir is given, the file re-hashed)."""
    res: dict[int, str] = {}
    for seed in (0, 1):
        vals = set()
        for arm in ("flatdag", "flatdag0"):
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
        if res[seed] != FLAT_CKPT_SHA[seed]:
            raise SystemExit(f"seed {seed}: the runs' recorded init {res[seed][:16]} is not FLAT_s{seed} (pinned {FLAT_CKPT_SHA[seed][:16]})")
        if flat_dir:
            p = os.path.join(flat_dir, f"M2b_s{seed}_flat.pt")
            got = _sha(p) if os.path.isfile(p) else None
            if got != res[seed]:
                raise SystemExit(f"seed {seed}: {p} hashes to {str(got)[:16]}, the runs' recorded init is {res[seed][:16]}")
    return res


def flatbase(dagreread: str) -> list[str]:
    bad = []
    for rel, want in FLAT_BANKED.items():
        p = os.path.join(dagreread, rel)
        if not os.path.isfile(p):
            bad.append(f"{p} absent")
        elif _sha(p) != want:
            bad.append(f"{p} hashes to {_sha(p)[:12]}, pinned {want[:12]}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--art", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids-dir", required=True)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--write")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n-dump-min", type=int, default=141)
    p = sub.add_parser("initsha")
    p.add_argument("--art", required=True)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--flat-dir")
    p = sub.add_parser("flatbase")
    p.add_argument("--dagreread", required=True)
    for nm in ("dagbase", "treetip", "ids", "parity"):
        sub.add_parser(nm, add_help=False)
    a, rest = ap.parse_known_args(argv)
    if a.cmd in ("dagbase", "treetip", "ids", "parity"):
        return D2M.main([a.cmd] + rest)
    if a.cmd == "initsha":
        r = initsha(a.art, parse_runs(a.runs), a.flat_dir)
        print(" ".join(f"{s}={r[s]}" for s in (0, 1)))
        return 0
    if a.cmd == "flatbase":
        bad = flatbase(a.dagreread)
        if bad:
            print("FLAT's banked B-2 material REFUSED:")
            for b in bad:
                print("  -", b)
            return 1
        print(f"FLAT's banked B-2 material present and pinned: {len(FLAT_BANKED)} files (packed DAG and FLAT causal dumps, the B-2 table, FLAT's two rollout argv): read, not re-run")
        return 0
    if a.cmd == "manifest":
        m = build(a.art, a.out, parse_runs(a.runs), a.ids_dir)
        if a.dry_run:
            print(json.dumps({"runs": [tag_of(arm, s) for arm, s in parse_runs(a.runs)], "slots": {"DAG": "FLAT-DAG", "DAG0": "FLAT-DAG-0"},
                              "bank": m["bank"], "ids": [m["eval141_clip_ids"], m["train_clip_ids"]]}, indent=1))
            return 0
        bad = verify(m, a.n_dump_min)
        if bad:
            print("MANIFEST REFUSED:")
            for b in bad:
                print("  -", b)
            return 1
        os.makedirs(os.path.dirname(os.path.abspath(a.write)), exist_ok=True)
        with open(a.write, "w", encoding="utf-8", newline="\n") as f:
            json.dump(m, f, indent=1)
        print(f"manifest written {a.write} sha256 {_sha(a.write)} ({len(m['arms']['DAG'])} + {len(m['arms']['DAG0'])} runs, 2 banks verified COMPLETE)")
        return 0
    return 2


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
