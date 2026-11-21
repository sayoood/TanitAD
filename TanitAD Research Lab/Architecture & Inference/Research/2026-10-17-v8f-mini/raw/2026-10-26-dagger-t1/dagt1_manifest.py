#!/usr/bin/env python3
"""dagt1_manifest.py -- the dev-box CPU T1 chain of PREREG_V8F_DAGGER: builds the reader manifest runs.json for dag_read.py, and the small guards the chain needs.

    manifest   --art ART --out OUT --orc-art ORC_ART --orc-t1 ORC_T1 --ids-dir DIR [--pfx P] [--runs dag:0,dag0:0,dag:1,dag0:1] --t0-fresh-dir D --write PATH | --dry-run
    m2bsha     --art ART --orc-art ORC_ART [--runs ..] [--b2-json PATH]         prints "0=<sha256> 1=<sha256>"  (the M2b init checkpoint the runs' own records name; must equal the banked ones)
    ids        --grid GRID --release RELEASE.jsonl.gz --out-dir DIR [--want-train-sha HEX]   writes eval141_clip_ids.json / train_clip_ids.json, REFUSES unless the TRAIN list hashes to the banked train_sha256
    orc0       --orc-art ORC_ART --orc-t1 ORC_T1                                 checks the REUSED O-RECOVER ORC-0 outputs (never re-run): checkpoint sha256 = every done record's, files present
    parity     --tree TREE --ref REF_TREE                                        every module the T1 / T0 tools can import is identical (CR-normalised) to the ORC-0 tool tree's, or additive-only, or new

Schema = what dag_read.load_manifest / read() consume (dag_read.py, test_dag_read.Fx is the fixture):
    {"seeds": [0, 1], "arms": {"DAG": {"0": E, "1": E}, "DAG0": {...}, "ORC0": {...}},
     "bank": {"0": {"report": <ART>/bank_s0/PROVENANCE.json, "bank_file": <ART>/bank_s0/bank.json}, "1": {...}}, "eval141_clip_ids": <json list>, "train_clip_ids": <json list>}
    bank.{s}: the PULLED copies box 2's chain publishes (the 11-name SHA256SUMS.txt of <ART>/bank_s<S>/: PROVENANCE.json, bank.json, the shard provenance files, the shard npz, bank_stage.log); the reader
    reads the shard provenance and the shard npz BESIDE `report` (load_bank), so all of them must sit in that one directory; `bank_file` is the pulled bank.json (its sha256 is the bank's identity)
    E = {"ckpt_tail": "M2b_s0_dag/ckpt_last.pt", "train_dir": <ART>/M2b_s0_dag (argv.json summary.json selfcheck.json), "token": <ART>/M2b_s0_dag/launch_gate_token.json   (DAG / DAG0 only),
         "t1": {"tag", "ticks", "dump_dir", "json"}        the decisive --limit causal rollout            "t1_logged": same, --limit logged (K4 only)
         "t0": {"json", "npz"}                             box 2's standard T0 eval (--ages cycle): R-T0, T0-CKPT
         "t0_fresh": {"json", "npz"}                       the dev-box CPU --ages fresh T0 eval: K4's comparator (ORC Amendment 4, PREREG_V8F_DAGGER section 5)}
ORC0 is NOT re-run: its entries point at the banked O-RECOVER outputs (box-2 pull in ORC_ART, rollouts and fresh evals in ORC_T1); it is read only for the replicate floor RF.
"""
from __future__ import annotations

import argparse
import ast
import glob
import gzip
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dagt1_pullcheck as PC  # noqa: E402

ARM_READER = {"dag": "DAG", "dag0": "DAG0", "orc0": "ORC0"}
DEFAULT_RUNS = "dag:0,dag0:0,dag:1,dag0:1"                       # the PREREG's order (section 8, queue item 2)
LIMITS = ("causal", "logged")
# the banked M2b checkpoints (PREREG section 1; the box-2 chain asserts the same two values)
BANKED_M2B_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
# M2b's TRAIN list as the trainer recorded it (MEASURED 2026-10-26 in C:/Users/Admin/box2_ship/logs/v8f_mini/runs/M2b_s{0,1}/summary.json: n_clips 4512, identical for both seeds)
M2B_TRAIN_SHA256 = "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1"
M2B_EXCLUDED_SHA256 = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
N_TRAIN_CLIPS, N_EVAL_CLIPS, EXCLUDE = 4512, 141, (540, 600)


def fwd(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def tail2(p: str) -> str:
    return "/".join(str(p).replace("\\", "/").split("/")[-2:])


def tag_of(pfx: str, arm: str, seed: int, limit: str) -> str:
    """the rollout's --tag: <PFX><DAG|DAG0|ORC0>_s<S>_m10_<limit>_heads  (T1-HEADS naming)"""
    return f"{pfx}{ARM_READER[arm]}_s{seed}_m10_{limit}_heads"


def parse_runs(s: str) -> list[tuple[str, int]]:
    out = []
    for it in s.split(","):
        arm, _, seed = it.partition(":")
        if arm not in ("dag", "dag0") or not seed.isdigit():
            raise SystemExit(f"--runs: expected arm:seed with arm in ['dag', 'dag0'], got {it!r}")
        out.append((arm, int(seed)))
    return out


def t1_entry(out: str, pfx: str, arm: str, seed: int, limit: str) -> dict:
    t = tag_of(pfx, arm, seed, limit)
    return {"tag": t, "ticks": fwd(f"{out}/ticks/ticks_{t}.npz"), "dump_dir": fwd(f"{out}/dump_{t}"), "json": fwd(f"{out}/t1v8f_{t}.json")}


def t0_tag(pfx: str, arm: str, seed: int) -> str:
    return f"{pfx}{PC.eval_tag(arm, seed)}-fresh"                   # the chain's t0tag: <PFX>M2b-<arm>_s<S>-fresh


def build(art: str, out: str, pfx: str, runs: list[tuple[str, int]], orc_art: str, orc_t1: str, ids_dir: str, t0_fresh_dir: str | None,
          replicate: tuple[str, int] | None = None, label: str | None = None) -> dict:
    """The manifest dict.  ``replicate`` (SMOKE only): every one of the slots (DAG, DAG0, ORC0 x seeds 0/1) points at the files of that ONE stand-in run."""
    arms: dict[str, dict] = {"DAG": {}, "DAG0": {}, "ORC0": {}}
    slots = runs if replicate is None else [(a, s) for a in ("dag", "dag0") for s in (0, 1)]
    for arm, seed in slots:
        src_arm, src_seed = replicate if replicate is not None else (arm, seed)
        run = PC.run_name(src_arm, src_seed)
        d = fwd(f"{art}/{run}")
        tg = PC.eval_tag(src_arm, src_seed)
        e = {"ckpt_tail": f"{run}/ckpt_last.pt", "train_dir": d, "token": f"{d}/launch_gate_token.json",
             "t1": t1_entry(out, pfx, src_arm, src_seed, "causal"), "t1_logged": t1_entry(out, pfx, src_arm, src_seed, "logged"),
             "t0": {"json": f"{d}/eval_{tg}.json", "npz": f"{d}/eval_{tg}.npz"}}
        if t0_fresh_dir:
            e["t0_fresh"] = {"json": fwd(f"{t0_fresh_dir}/eval_{t0_tag(pfx, src_arm, src_seed)}.json"), "npz": fwd(f"{t0_fresh_dir}/eval_{t0_tag(pfx, src_arm, src_seed)}.npz")}
        arms[ARM_READER[arm]][str(seed)] = e
    for seed in (0, 1):                                              # ORC-0: the REUSED banked outputs (the replicate floor); never re-run
        if replicate is not None:
            src_arm, src_seed = replicate
            run, d, o1 = PC.run_name(src_arm, src_seed), fwd(f"{art}/{PC.run_name(src_arm, src_seed)}"), out
            tg, pf, tarm, tseed = PC.eval_tag(src_arm, src_seed), pfx, src_arm, src_seed
            fresh = fwd(f"{t0_fresh_dir}/eval_{t0_tag(pfx, src_arm, src_seed)}") if t0_fresh_dir else None
            e = {"ckpt_tail": f"{run}/ckpt_last.pt", "train_dir": d,
                 "t1": t1_entry(o1, pf, tarm, tseed, "causal"), "t1_logged": t1_entry(o1, pf, tarm, tseed, "logged"), "t0": {"json": f"{d}/eval_{tg}.json", "npz": f"{d}/eval_{tg}.npz"}}
            if fresh:
                e["t0_fresh"] = {"json": fresh + ".json", "npz": fresh + ".npz"}
        else:
            run = f"M2b_s{seed}_orc0"
            d = fwd(f"{orc_art}/{run}")
            e = {"ckpt_tail": f"{run}/ckpt_last.pt", "train_dir": d,
                 "t1": {"tag": f"ORC0_s{seed}_m10_causal_heads", "ticks": fwd(f"{orc_t1}/ticks/ticks_ORC0_s{seed}_m10_causal_heads.npz"),
                        "dump_dir": fwd(f"{orc_t1}/dump_ORC0_s{seed}_m10_causal_heads"), "json": fwd(f"{orc_t1}/t1v8f_ORC0_s{seed}_m10_causal_heads.json")},
                 "t1_logged": {"tag": f"ORC0_s{seed}_m10_logged_heads", "ticks": fwd(f"{orc_t1}/ticks/ticks_ORC0_s{seed}_m10_logged_heads.npz"),
                               "dump_dir": fwd(f"{orc_t1}/dump_ORC0_s{seed}_m10_logged_heads"), "json": fwd(f"{orc_t1}/t1v8f_ORC0_s{seed}_m10_logged_heads.json")},
                 "t0": {"json": f"{d}/eval_M2b-orc0_s{seed}.json", "npz": f"{d}/eval_M2b-orc0_s{seed}.npz"},
                 "t0_fresh": {"json": fwd(f"{orc_t1}/t0fresh/eval_M2b-orc0_s{seed}-fresh.json"), "npz": fwd(f"{orc_t1}/t0fresh/eval_M2b-orc0_s{seed}-fresh.npz")}}
        arms["ORC0"][str(seed)] = e
    m = {"seeds": [0, 1], "arms": arms,
         "bank": {str(s): {"report": fwd(f"{art}/bank_s{(replicate[1] if replicate else s)}/PROVENANCE.json"),
                           "bank_file": fwd(f"{art}/bank_s{(replicate[1] if replicate else s)}/bank.json")} for s in (0, 1)},      # the PULLED copies; the shard provenance and the shard npz are read beside the report
         "eval141_clip_ids": fwd(f"{ids_dir}/eval141_clip_ids.json"), "train_clip_ids": fwd(f"{ids_dir}/train_clip_ids.json"),
         "provenance": {"prereg": "Project Steering/PREREG_V8F_DAGGER.md + Amendment 1 (A1-2026-10-26), claim row H-V8F-DAGGER-1",
                        "built_by": "dagt1_manifest.py (the dev-box T1 chain, dag_t1_chain.sh)",
                        "t1": "v8f_t1_eval.py run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --dump-ticks --dump-heads, plain (no post-map)",
                        "ORC0": "the banked O-RECOVER ORC-0 outputs, REUSED (replicate floor RF); not re-run",
                        "t0": "box 2's standard eval (--ages cycle): R-T0, T0-CKPT", "t0_fresh": "dev-box CPU --ages fresh eval: K4's comparator"}}
    if label:
        m["LABEL"] = label
    return m


def read_json(p: str):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sums_of(d: str) -> dict[str, str]:
    out = {}
    p = os.path.join(d, "SHA256SUMS.txt")
    if os.path.isfile(p):
        with open(p, encoding="utf-8", newline="") as f:
            for raw in f.read().splitlines():
                m = PC.LINE.match(raw.rstrip("\r"))
                if m:
                    out[m.group(2).strip()] = m.group(1)
    return out


def verify(m: dict, n_dump_min: int, smoke: bool = False, check_ckpt_sha: bool = True) -> list[str]:
    """Every file the manifest points at exists and is the right kind, and every entry is THIS arm's: no path or checkpoint is shared between arms (a DAG-0 slot pointing at DAG's run), the
    rollouts and the T0 evals scored the entry's own checkpoint, the rollouts' done records name the pulled checkpoint's sha256, and the four DAG / DAG-0 checkpoints are pairwise different."""
    bad: list[str] = []

    def need_file(p, what):
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            bad.append(f"{what}: {p} missing or empty")

    seen: dict[str, str] = {}
    for arm_r, per in m["arms"].items():
        arm = arm_r.lower()
        for seed, e in per.items():
            lab = f"{arm_r} s{seed}"
            if not smoke:
                want_run = f"M2b_s{seed}_{arm}"
                if e["ckpt_tail"] != f"{want_run}/ckpt_last.pt":
                    bad.append(f"{lab}: ckpt_tail {e['ckpt_tail']!r} is not {want_run}/ckpt_last.pt (another arm's / seed's run)")
                if os.path.basename(fwd(e["train_dir"])) != want_run:
                    bad.append(f"{lab}: train_dir {e['train_dir']} is not .../{want_run} (another arm's / seed's run)")
                if arm_r in ("DAG", "DAG0"):
                    want_tag = {"t1": tag_of("", arm, int(seed), "causal"), "t1_logged": tag_of("", arm, int(seed), "logged")}
                    for key, wt in want_tag.items():
                        if e[key]["tag"] != wt:
                            bad.append(f"{lab} {key}: tag {e[key]['tag']!r} is not {wt!r}")
            for what, p in (("train_dir", e["train_dir"]), ("t1.json", e["t1"]["json"]), ("t1_logged.json", e["t1_logged"]["json"]), ("t1.dump_dir", e["t1"]["dump_dir"]),
                            ("t1_logged.dump_dir", e["t1_logged"]["dump_dir"]), ("t0.json", e["t0"]["json"]), ("t0_fresh.json", (e.get("t0_fresh") or {}).get("json", "")),
                            ("ckpt_tail", e["ckpt_tail"])):
                if not smoke and p:
                    if p in seen and seen[p] != lab:
                        bad.append(f"{lab}: {what} {p} is also the path of {seen[p]}: two arms share one run")
                    seen[p] = lab
            for d in ("argv.json", "summary.json", "selfcheck.json"):
                need_file(os.path.join(e["train_dir"], d), f"{lab} {d}")
            if arm_r in ("DAG", "DAG0"):
                need_file(e.get("token", ""), f"{lab} token")
            need_file(e["t0"]["json"], f"{lab} T0 json")
            need_file(e["t0"]["npz"], f"{lab} T0 npz")
            if "t0_fresh" not in e:
                bad.append(f"{lab} t0_fresh: absent (K4's comparator, the --ages fresh T0 eval)")
            else:
                need_file(e["t0_fresh"]["json"], f"{lab} t0_fresh json")
                need_file(e["t0_fresh"]["npz"], f"{lab} t0_fresh npz")
            for key, ev, want_ages in (("t0", e["t0"], "cycle"), ("t0_fresh", e.get("t0_fresh"), "fresh")):
                if ev and os.path.isfile(ev["json"]):
                    try:
                        j = read_json(ev["json"])
                    except (OSError, ValueError) as ex:
                        bad.append(f"{lab} {key} json does not parse: {ex}")
                        continue
                    if tail2(j.get("ckpt", "")) != e["ckpt_tail"]:
                        bad.append(f"{lab} {key}: the eval scored .../{tail2(j.get('ckpt', ''))}, the manifest expects .../{e['ckpt_tail']}")
                    if j.get("ages") != want_ages:
                        bad.append(f"{lab} {key}: ages {j.get('ages')!r}, must be {want_ages!r}")
            for key, lim in (("t1", "causal"), ("t1_logged", "logged")):
                t = e[key]
                need_file(t["ticks"], f"{lab} {key} ticks")
                need_file(t["json"], f"{lab} {key} json")
                n = len(glob.glob(os.path.join(t["dump_dir"], "ep*.npz")))
                if n < n_dump_min:
                    bad.append(f"{lab} {key}: {n} ep*.npz in {t['dump_dir']} (< {n_dump_min})")
                try:
                    h = read_json(t["json"]).get("t1v8f") or {}
                except (OSError, ValueError) as ex:
                    bad.append(f"{lab} {key} json does not parse: {ex}")
                    continue
                if h.get("limit") != lim or h.get("ticks_heads") is not True or h.get("tag") != t["tag"]:
                    bad.append(f"{lab} {key}: json header limit {h.get('limit')!r} (want {lim}), ticks_heads {h.get('ticks_heads')!r}, tag {h.get('tag')!r} (want {t['tag']})")
                if tail2(str(h.get("ckpt", ""))) != e["ckpt_tail"]:
                    bad.append(f"{lab} {key}: the rollout scored .../{tail2(str(h.get('ckpt', '')))}, the manifest expects .../{e['ckpt_tail']}")
    if check_ckpt_sha and not smoke:                                  # the pulled checkpoints: distinct, and the rollouts' done records name them
        shas: dict[str, str] = {}
        for arm_r in ("DAG", "DAG0"):
            for seed, e in m["arms"].get(arm_r, {}).items():
                lab = f"{arm_r} s{seed}"
                sums = sums_of(e["train_dir"])
                sha = sums.get("ckpt_last.pt")
                if not sha:
                    bad.append(f"{lab}: no ckpt_last.pt entry in {e['train_dir']}/SHA256SUMS.txt")
                    continue
                if sha in shas:
                    bad.append(f"{lab}: its checkpoint sha256 {sha[:12]} is the same as {shas[sha]}'s: one run's checkpoint sits in two arms' directories")
                shas[sha] = lab
                if sha in BANKED_M2B_SHA.values():
                    bad.append(f"{lab}: its checkpoint sha256 {sha[:12]} IS the banked M2b init: nothing was fine-tuned")
                for key in ("t1", "t1_logged"):
                    dp = os.path.join(os.path.dirname(e[key]["json"]), f"done_{e[key]['tag']}.txt")
                    try:
                        dr = read_json(dp)
                    except (OSError, ValueError) as ex:
                        bad.append(f"{lab} {key}: done record {dp} unreadable: {ex}")
                        continue
                    if dr.get("ckpt_sha256") != sha:
                        bad.append(f"{lab} {key}: the done record names checkpoint {str(dr.get('ckpt_sha256'))[:12]}, the pulled one is {sha[:12]}")
    for s, be in (m.get("bank") or {}).items():
        if not smoke:
            bdir = os.path.dirname(be["report"])
            bseed = int(os.path.basename(bdir).rsplit("_s", 1)[1])
            if bseed != int(s) or os.path.basename(be["report"]) != "PROVENANCE.json":
                bad.append(f"bank s{s}: the report {be['report']} is not .../bank_s{s}/PROVENANCE.json (another seed's bank)")
            ok, why, _ = PC.check_bank(os.path.dirname(bdir), bseed)                  # the queue's 4-shard count is the chain's bank gate (--expect-shards 4); here: the bank on disk is complete and self-consistent
            if not ok:
                bad.append(f"bank s{s}: {why}")
            bf = be.get("bank_file")
            if not bf:
                bad.append(f"bank s{s}: no bank_file (the pulled bank.json the reader's bank identity is hashed from)")
            elif fwd(bf) != fwd(os.path.join(bdir, "bank.json")):
                bad.append(f"bank s{s}: bank_file {bf} is not the pulled {fwd(os.path.join(bdir, 'bank.json'))} beside the report")
    for k in ("eval141_clip_ids", "train_clip_ids"):
        need_file(m[k], k)
    return bad


# ---------------------------------------------------------------------------------------------------------------- the M2b init checkpoint
def init_sha(rec: dict) -> list[str]:
    out = []
    for pre in ("", "totals.", "levers.", "built_levers."):
        cur, ok = rec, True
        for part in (pre + "init_ckpt.sha256").split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            out.append(str(cur))
    return out


def m2b_sha(art: str, runs: list[tuple[str, int]], orc_art: str | None, b2_json: str | None, require_banked: bool = True) -> dict[int, str]:
    got: dict[int, set] = {}
    dirs = [(os.path.join(art, PC.run_name(a, s)), s) for a, s in runs]
    if orc_art:
        dirs += [(os.path.join(orc_art, f"M2b_s{s}_orc0"), s) for s in (0, 1)]
    for d, seed in dirs:
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
        if require_banked and res[seed] != BANKED_M2B_SHA[seed]:
            raise SystemExit(f"seed {seed}: the runs' recorded init {res[seed][:16]} is not the banked M2b_s{seed} {BANKED_M2B_SHA[seed][:16]}")
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


# ---------------------------------------------------------------------------------------------------------------- the id lists the reader needs
def sha12(c: str) -> str:
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def train_list_sha256(ids) -> str:
    """the trainer's own TRAIN-list hash (v8f_mini_train.build_data: sha256 of the clip ids sorted by sha12, newline-joined, recorded as `train_sha256`)"""
    return hashlib.sha256(chr(10).join(sorted(ids, key=sha12)).encode("utf-8")).hexdigest()


def make_ids(grid: str, release: str, out_dir: str, want_train: str = M2B_TRAIN_SHA256, want_excl: str = M2B_EXCLUDED_SHA256, dry: bool = False) -> dict:
    """eval141_clip_ids = the eval grid's `order`; train_clip_ids = the v7.2 label release's train clip ids (4,572, one record per clip) minus the clips the trainer excluded
    (`sorted(clips, key=sha12)[540:600]`, `--exclude 540:600`).  REFUSED unless the result hashes to the banked train_sha256 AND the excluded set to the banked excluded_sha256."""
    g = read_json(grid)
    ev = [str(x) for x in (g["order"] if isinstance(g, dict) else g)]
    if len(ev) != N_EVAL_CLIPS or len(set(ev)) != N_EVAL_CLIPS:
        raise SystemExit(f"REFUSED: the eval grid's order holds {len(ev)} clips ({len(set(ev))} distinct), expected {N_EVAL_CLIPS}")
    allc = []
    with gzip.open(release, "rt", encoding="utf-8") as f:
        for line in f:
            allc.append(str(json.loads(line)["clip_id"]))
    if len(set(allc)) != len(allc):
        raise SystemExit("REFUSED: the release lists a clip twice")
    lo, hi = EXCLUDE
    excl = sorted(allc, key=sha12)[lo:hi]
    ex = set(excl)
    tr = [c for c in allc if c not in ex]
    got_t = train_list_sha256(tr)
    got_e = hashlib.sha256(chr(10).join(excl).encode("utf-8")).hexdigest()
    bad = []
    if len(tr) != N_TRAIN_CLIPS:
        bad.append(f"{len(tr)} TRAIN clips, the trainer recorded {N_TRAIN_CLIPS}")
    if got_t != want_train:
        bad.append(f"TRAIN list sha256 {got_t[:12]} != the banked train_sha256 {want_train[:12]}")
    if got_e != want_excl:
        bad.append(f"excluded list sha256 {got_e[:12]} != the banked excluded_sha256 {want_excl[:12]}")
    if set(tr) & set(ev):
        bad.append("the derived TRAIN list and the eval-141 list overlap")
    if bad:
        raise SystemExit("REFUSED: the TRAIN clip list derived from the release is not the one M2b trained on: " + "; ".join(bad))
    if not dry:
        os.makedirs(out_dir, exist_ok=True)
        for nm, v in (("eval141_clip_ids.json", ev), ("train_clip_ids.json", tr)):
            with open(os.path.join(out_dir, nm), "w", encoding="utf-8", newline="\n") as f:
                json.dump(v, f)
    return {"n_eval": len(ev), "n_train": len(tr), "n_release": len(allc), "train_sha256": got_t, "excluded_sha256": got_e}


# ---------------------------------------------------------------------------------------------------------------- ORC-0 (reused, never re-run)
def orc0_check(orc_art: str, orc_t1: str, n_dump_min: int = 141) -> list[str]:
    bad = []
    for s in (0, 1):
        d = os.path.join(orc_art, f"M2b_s{s}_orc0")
        sums = sums_of(d)
        ck = os.path.join(d, "ckpt_last.pt")
        if not os.path.isfile(ck):
            bad.append(f"ORC-0 s{s}: {ck} absent")
            continue
        listed = sums.get("ckpt_last.pt")
        got = PC.sha256_file(ck)
        if got != listed:
            bad.append(f"ORC-0 s{s}: ckpt_last.pt hashes to {got[:12]}, SHA256SUMS.txt lists {str(listed)[:12]}")
        recs = [f"done_ORC0_s{s}_m10_causal_heads.txt", f"done_ORC0_s{s}_m10_logged_heads.txt", f"done_t0f_M2b-orc0_s{s}-fresh.txt"]
        for r in recs:
            p = os.path.join(orc_t1, r)
            try:
                dr = read_json(p)
            except (OSError, ValueError) as ex:
                bad.append(f"ORC-0 s{s}: done record {r} unreadable: {ex}")
                continue
            if dr.get("ckpt_sha256") != got:
                bad.append(f"ORC-0 s{s}: {r} names checkpoint {str(dr.get('ckpt_sha256'))[:12]}, the pulled one is {got[:12]}")
            if dr.get("smoke") is not False:
                bad.append(f"ORC-0 s{s}: {r} is a SMOKE record")
        for lim in LIMITS:
            t = f"ORC0_s{s}_m10_{lim}_heads"
            for p in (f"{orc_t1}/t1v8f_{t}.json", f"{orc_t1}/ticks/ticks_{t}.npz"):
                if not os.path.isfile(p) or os.path.getsize(p) == 0:
                    bad.append(f"ORC-0 s{s}: {p} missing or empty")
            n = len(glob.glob(f"{orc_t1}/dump_{t}/ep*.npz"))
            if n < n_dump_min:
                bad.append(f"ORC-0 s{s}: {n} ep*.npz in dump_{t} (< {n_dump_min})")
        for ext in ("json", "npz"):
            p = f"{orc_t1}/t0fresh/eval_M2b-orc0_s{s}-fresh.{ext}"
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                bad.append(f"ORC-0 s{s}: {p} missing or empty")
    return bad


# ---------------------------------------------------------------------------------------------------------------- tool-tree parity with the tree that rolled ORC-0 out
ROOTS = ("stack", "taniteval", "stack/scripts")
ENTRIES = ("stack/scripts/v8f_t1_eval.py", "stack/scripts/v8f_mini_eval.py", "taniteval/tools/t1_eval.py")


def _resolve(tree: str, mod: str) -> list[str]:
    parts = mod.split(".")
    out = []
    for r in ROOTS:
        base = os.path.join(tree, r)
        for k in range(len(parts), 0, -1):
            p = os.path.join(base, *parts[:k])
            for cand in (p + ".py", os.path.join(p, "__init__.py")):
                if os.path.isfile(cand):
                    out.append(cand)
            if out:
                break
    return out


def import_closure(tree: str, entries=ENTRIES) -> set[str]:
    """Every .py under ``tree`` that the entry scripts can import, statically (ast; function-level imports included; relative to the tree's three sys.path roots).  A superset of the run's imports."""
    todo = [os.path.join(tree, e) for e in entries if os.path.isfile(os.path.join(tree, e))]
    seen: set[str] = set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        try:
            with open(f, "rb") as fh:
                node = ast.parse(fh.read())
        except (OSError, SyntaxError):
            continue
        for n in ast.walk(node):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            for m_ in mods:
                for c in _resolve(tree, m_):
                    if c not in seen:
                        todo.append(c)
                    d = os.path.dirname(c)
                    init = os.path.join(d, "__init__.py")        # the package __init__ chain also runs
                    while os.path.normpath(d) not in [os.path.normpath(os.path.join(tree, r)) for r in ROOTS] and os.path.isfile(init):
                        if init not in seen:
                            todo.append(init)
                        d = os.path.dirname(d)
                        init = os.path.join(d, "__init__.py")
    return seen


def _norm_bytes(p: str) -> bytes:
    with open(p, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def parity(tree: str, ref: str) -> dict:
    """Every module the T1 / T0 tools of ``tree`` can import is, against ``ref`` (the tool tree that rolled the ORC-0 outputs out): IDENTICAL (CR-normalised), NEW (absent from ref), or ADDITIVE-ONLY
    (only inserted lines).  Anything else (a modified or deleted line, or a file ref has and tree lost) is a VIOLATION: the replicate floor would then compare two different tools."""
    import difflib
    res = {"identical": [], "new": [], "additive_only": [], "violations": [], "closure_files": 0}
    cl = import_closure(tree)
    res["closure_files"] = len(cl)
    for f in sorted(cl):
        rel = os.path.relpath(f, tree).replace("\\", "/")
        rf = os.path.join(ref, rel)
        if not os.path.isfile(rf):
            res["new"].append(rel)
            continue
        a, b = _norm_bytes(rf), _norm_bytes(f)
        if a == b:
            res["identical"].append(rel)
            continue
        ops = difflib.SequenceMatcher(None, a.split(b"\n"), b.split(b"\n"), autojunk=False).get_opcodes()
        if all(op[0] in ("equal", "insert") for op in ops):
            res["additive_only"].append((rel, sum(op[4] - op[3] for op in ops if op[0] == "insert")))
        else:
            res["violations"].append(f"{rel}: lines of the reference tree were modified or deleted")
    lost = sorted(os.path.relpath(f, ref).replace("\\", "/") for f in import_closure(ref) if not os.path.isfile(os.path.join(tree, os.path.relpath(f, ref))))
    for rel in lost:
        res["violations"].append(f"{rel}: in the reference tree's import closure, absent from the tree")
    return res


# ---------------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--art", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--orc-art", required=True)
    p.add_argument("--orc-t1", required=True)
    p.add_argument("--ids-dir", required=True)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--pfx", default="")
    p.add_argument("--t0-fresh-dir")
    p.add_argument("--write")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n-dump-min", type=int, default=141)
    p.add_argument("--smoke-replicate", help="arm:seed -- SMOKE only: every slot points at this one stand-in run")
    p.add_argument("--label")
    p = sub.add_parser("m2bsha")
    p.add_argument("--art", required=True)
    p.add_argument("--orc-art")
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--b2-json")
    p.add_argument("--smoke-replicate")
    p = sub.add_parser("ids")
    p.add_argument("--grid", required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--want-train-sha", default=M2B_TRAIN_SHA256)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("orc0")
    p.add_argument("--orc-art", required=True)
    p.add_argument("--orc-t1", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--tree", required=True)
    p.add_argument("--ref", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "m2bsha":
        if a.smoke_replicate:
            arm, seed = parse_runs(a.smoke_replicate)[0]
            r = m2b_sha(a.art, [(arm, seed)], None, a.b2_json, require_banked=False)
            print(" ".join(f"{s}={r[seed]}" for s in (0, 1)))
            return 0
        r = m2b_sha(a.art, parse_runs(a.runs), a.orc_art, a.b2_json)
        if sorted(r) != [0, 1]:
            raise SystemExit(f"the runs give an init sha for seeds {sorted(r)}, the read needs both 0 and 1")
        print(" ".join(f"{s}={r[s]}" for s in (0, 1)))
        return 0
    if a.cmd == "ids":
        r = make_ids(a.grid, a.release, a.out_dir, a.want_train_sha, dry=a.dry_run)
        print(("would write" if a.dry_run else "wrote") + f" {a.out_dir}/eval141_clip_ids.json ({r['n_eval']}) and train_clip_ids.json ({r['n_train']} of {r['n_release']} release clips): "
              f"train_sha256 {r['train_sha256']} (= the banked one), excluded_sha256 {r['excluded_sha256'][:16]} (= the banked one)")
        return 0
    if a.cmd == "orc0":
        bad = orc0_check(a.orc_art, a.orc_t1)
        if bad:
            print("ORC-0 REFUSED:")
            for b in bad:
                print("  -", b)
            return 1
        print("ORC-0 outputs present and bound to the banked checkpoints (done records, SHA256SUMS): reused, not re-run")
        return 0
    if a.cmd == "parity":
        r = parity(a.tree, a.ref)
        print(f"closure {r['closure_files']} files: {len(r['identical'])} identical to the reference, {len(r['new'])} new, {len(r['additive_only'])} additive-only "
              f"({'; '.join(f'{f} +{n} lines' for f, n in r['additive_only'])}), {len(r['violations'])} violations")
        for f in r["new"]:
            print("  new:", f)
        for v in r["violations"]:
            print("  VIOLATION:", v)
        return 1 if r["violations"] else 0
    runs = parse_runs(a.runs)
    rep = parse_runs(a.smoke_replicate)[0] if a.smoke_replicate else None
    m = build(a.art, a.out, a.pfx, runs, a.orc_art, a.orc_t1, a.ids_dir, a.t0_fresh_dir, rep, a.label)
    if a.dry_run:
        print(json.dumps(m, indent=1))
        return 0
    bad = verify(m, a.n_dump_min, smoke=rep is not None)
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
