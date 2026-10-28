#!/usr/bin/env python3
"""lit1_tree.py -- the tool-tree, K-BUILD and manifest half of lit1_lib.py (CPU only, read-only on every input; git only through `cat-file` / `diff` / `rev-parse` on the mirror).

WHY A SECOND TREE (MEASURED by `cfgprobe` / `treeplan`, and the reason this chain is not a copy of the DAG-ORC one):  the pinned tool tree (dagt1_tree_b43c7a9a, digest 6a4b47c8...) loads a checkpoint with
`M.V8FMiniConfig(**ck["cfg"])` (v8f_t1_eval.py:506, v8f_mini_eval.py:237) and the launch trainer saves `"cfg": cfg.__dict__` (v8f_mini_train.py:1917), which at the launch commit carries a NEW key `launch_intent_w`.  The pinned
tree's V8FMiniConfig has no such field => every LAUNCH-INTENT checkpoint raises TypeError in BOTH tools.  The LAUNCH-INTENT tree is therefore the pinned tree's code with EXACTLY ONE file replaced -- stack/tanitad/models/v8f_mini.py by the
launch commit's blob -- and `modelproof` shows by AST that every changed or added definition mentions the launch lever (the inference path is the pinned tree's, node for node).

    treeplan   [--base B]                              what would be copied / replaced, and the digests (writes nothing)
    treebuild  --dest D [--mirror M]                   build the LAUNCH-INTENT tree at D (partial dir, then renamed); an existing D is verified, or MOVED ASIDE, never deleted
    treecheck  --tree D [--base B] [--mirror M]        digest + one-file-difference proof + modelproof + cfg reach; rc 0 / 3
    modelproof --pinned P --launch L                   the AST proof alone
    cfgprobe   --tree T [--ckpt C]                     builds V8FMiniConfig(**cfg + launch_intent_w) in a child python through T's own model module
    rolloutcode --mirror M --launch C [--base B]       the runtime import set vs the launch commit's blobs
    kbuild     --mirror M --launch C [--reader R]      the K-BUILD record (and the reader's own verdict on it)
    manifest   --art A --out O --ctrl-runs R --rep-runs R2 --mirror M --launch C [--write P | --dry-run]
    verify     --manifest P [--n-dump-min N]
    parsecheck --tool T KIND ARGV...                    parse one rendered argv with T's own argparse (no run)
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lit1_lib as L  # noqa: E402

RUNTIME_SET = (
    'stack/scripts/v8f_mini_eval.py', 'stack/scripts/v8f_t1_eval.py', 'stack/tanitad/__init__.py', 'stack/tanitad/channel_admissibility.py', 'stack/tanitad/config.py', 'stack/tanitad/data/__init__.py',
    'stack/tanitad/data/_contract.py', 'stack/tanitad/data/comma2k19.py', 'stack/tanitad/data/metadrive_env.py', 'stack/tanitad/data/stats.py', 'stack/tanitad/data/toy_driving.py', 'stack/tanitad/data/v8f_mini_data.py',
    'stack/tanitad/models/__init__.py', 'stack/tanitad/models/_validate.py', 'stack/tanitad/models/ego_history.py', 'stack/tanitad/models/encoder.py', 'stack/tanitad/models/fourbrain.py', 'stack/tanitad/models/hjepa.py',
    'stack/tanitad/models/imagination.py', 'stack/tanitad/models/inverse_dynamics.py', 'stack/tanitad/models/kinematic.py', 'stack/tanitad/models/kinematic_prior.py', 'stack/tanitad/models/metric_dynamics.py',
    'stack/tanitad/models/predictor.py', 'stack/tanitad/models/readout.py', 'stack/tanitad/models/sigreg.py', 'stack/tanitad/models/v8f_mini.py', 'stack/tanitad/refs/__init__.py', 'stack/tanitad/refs/refa.py',
    'stack/tanitad/refs/refb.py', 'stack/tanitad/refs/refc_tactical.py', 'stack/tanitad/refs/refcv6_max_speed.py', 'stack/tanitad/train/__init__.py', 'stack/tanitad/train/config_hygiene.py',
    'taniteval/taniteval/__init__.py', 'taniteval/taniteval/ci.py', 'taniteval/taniteval/floors_kin.py', 'taniteval/taniteval/four_families.py', 'taniteval/taniteval/progress.py', 'taniteval/taniteval/selgap.py',
    'taniteval/taniteval/stack_guard.py', 'taniteval/taniteval/v0_antiecho.py', 'taniteval/taniteval/v8f_beta_eval.py', 'taniteval/taniteval/v8f_consistency.py', 'taniteval/tools/t1_eval.py',
)      # the 45 files the T1 harness (causal + logged lane) and the fresh-age T0 eval LOADED when they ran (dagorc_t1/smoke/trace_*.json, the union): MEASURED at DAG-ORC's smoke
EXTRA_MAX_BYTES = 2 << 20
COMPOSE_RECORD = f"{L.AR}/dagorc_t1/kbuild_r2_launch_8c24823f.json"      # DAG-ORC r2's accepted K-BUILD record: the base -> 8c24823f (compose-flag) step
BASE_COMMIT = "aeb544683b1c6b70e29f241fe7133d4290a1b4a5"      # the commit DAG-OACC trained at (the reader's K-BUILD base)
LI_PARENT = "37d78ed16d84f632487e93be30aa894b8dbe13ea"        # the landed flag commit's first parent: its three build blobs are the ones the flag's tests were written against
RAWPK = f"{L.PK}/raw/2026-10-28-launch-intent-flag"
BUILD_FILES = (L.TRAINER_PATH, L.DATA_PATH, L.MODEL_PATH)


def git(mirror: str, *a, text: bool = False, check: bool = True):
    r = subprocess.run(["git", "-c", "safe.directory=*", "-c", "core.autocrlf=false", "-C", mirror, *a], capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(a[:3])}: rc {r.returncode}: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    if not check:
        return r.returncode, r.stdout.decode("utf-8", "replace") if text else r.stdout
    return r.stdout.decode("utf-8", "replace") if text else r.stdout


def blob(mirror: str, commit: str, path: str):
    rc, out = git(mirror, "cat-file", "blob", f"{commit}:{path}", check=False)
    return out if rc == 0 else None


def sha_b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ the tree
def digest_set(root: str) -> list:
    """The files the pinned digest covers: every *.py under stack/tanitad and taniteval, plus the T1 tool (608 files in the pinned tree)."""
    paths = set()
    for r in ("stack/tanitad", "taniteval"):
        for dp, _dn, fn in os.walk(os.path.join(root, r)):
            for f in fn:
                if f.endswith(".py"):
                    paths.add(os.path.relpath(os.path.join(dp, f), root).replace("\\", "/"))
    paths.add("stack/scripts/v8f_t1_eval.py")
    return sorted(paths, key=lambda s: s.encode())


def tree_digest(root: str, paths: list | None = None) -> str:
    """sha256 of the `sha256sum -b` lines of the digest set (reproduces the pinned 6a4b47c8... on dagt1_tree_b43c7a9a)."""
    lines = "".join(f"{L.sha256_file(os.path.join(root, p))} *{p}\n" for p in (paths or digest_set(root)))
    return hashlib.sha256(lines.encode()).hexdigest()


def extras(base: str) -> list:
    """Non-.py files below EXTRA_MAX_BYTES under the two code packages (configs etc.), plus the second tool script."""
    out = ["stack/scripts/v8f_mini_eval.py"]
    for r in ("stack/tanitad", "taniteval/taniteval"):
        for dp, dn, fn in os.walk(os.path.join(base, r)):
            dn[:] = [d for d in dn if d != "__pycache__"]
            for f in fn:
                p = os.path.join(dp, f)
                if not f.endswith((".py", ".pyc")) and os.path.getsize(p) <= EXTRA_MAX_BYTES:
                    out.append(os.path.relpath(p, base).replace("\\", "/"))
    return sorted(set(out), key=lambda s: s.encode())


def treeplan(base: str) -> dict:
    ds = digest_set(base)
    ex = extras(base)
    return {"base": base, "base_digest": tree_digest(base, ds), "n_py": len(ds), "n_extra": len(ex), "extra_bytes": sum(os.path.getsize(os.path.join(base, p)) for p in ex),
            "overlay": L.MODEL_PATH, "overlay_sha256": L.LAUNCH_MODEL_SHA, "pinned_model_sha256": L.sha256_file(os.path.join(base, L.MODEL_PATH))}


def treebuild(dest: str, base: str, mirror: str) -> str:
    """Build `dest` = base's code + the launch model blob.  Returns "built" | "reused".  An existing dest that fails treecheck is MOVED ASIDE (never deleted)."""
    dest = dest.rstrip("/\\")
    if os.path.isdir(dest):
        ok, _info = treecheck(dest, base, mirror)
        if ok:
            return "reused"
        aside = f"{dest}.bad_{time.strftime('%m%d%H%M%S')}"
        os.rename(dest, aside)
        print(f"moved the unverifiable tree {dest} aside to {aside}")
    mb = blob(mirror, L.LAUNCH_COMMIT, L.MODEL_PATH)
    if mb is None or sha_b(mb) != L.LAUNCH_MODEL_SHA:
        raise RuntimeError(f"the launch commit's model blob is {None if mb is None else sha_b(mb)[:16]}, pinned {L.LAUNCH_MODEL_SHA[:16]}")
    part = f"{dest}.partial_{time.strftime('%m%d%H%M%S')}"
    os.makedirs(part)
    for p in sorted(set(digest_set(base)) | set(extras(base))):
        t = os.path.join(part, p)
        os.makedirs(os.path.dirname(t), exist_ok=True)
        shutil.copyfile(os.path.join(base, p), t)
    with open(os.path.join(part, L.MODEL_PATH), "wb") as f:
        f.write(mb)
    ok, info = treecheck(part, base, mirror)
    if not ok:
        raise RuntimeError(f"the freshly built tree does not verify: {info}")
    os.rename(part, dest)
    return "built"


def treecheck(tree: str, base: str, mirror: str | None = None, model_sha: str = L.LAUNCH_MODEL_SHA, base_digest: str = L.BASE_DIGEST) -> tuple:
    """-> (ok, info).  The tree's digest set equals the base's; every file but the model overlay is byte-identical to the base; the overlay IS the launch commit's blob (pinned sha256); modelproof and cfg reach hold."""
    info: dict = {"problems": []}
    pr = info["problems"]
    if not os.path.isdir(tree):
        return False, {"problems": [f"{tree} is not a directory"]}
    ds_b, ds_t = digest_set(base), digest_set(tree)
    if ds_b != ds_t and L.on("tree-set"):
        pr.append(f"digest set differs: only in base {sorted(set(ds_b) - set(ds_t))[:3]}, only in tree {sorted(set(ds_t) - set(ds_b))[:3]}")
    diff = [p for p in sorted(set(ds_b) & set(ds_t)) if L.sha256_file(os.path.join(base, p)) != L.sha256_file(os.path.join(tree, p))]
    if diff != [L.MODEL_PATH] and L.on("tree-onefile"):
        pr.append(f"the tree differs from the pinned base in {diff[:6]}, must be exactly [{L.MODEL_PATH!r}]")
    ex_bad = [p for p in extras(base) if not os.path.isfile(os.path.join(tree, p)) or L.sha256_file(os.path.join(tree, p)) != L.sha256_file(os.path.join(base, p))]
    if ex_bad and L.on("tree-extras"):
        pr.append(f"{len(ex_bad)} non-.py extras missing or different, e.g. {ex_bad[:3]}")
    mp = os.path.join(tree, L.MODEL_PATH)
    if not os.path.isfile(mp) or L.sha256_file(mp) != model_sha:
        if L.on("tree-overlay"):
            pr.append(f"the overlay model is not the launch commit's blob {model_sha[:16]}")
    else:
        proof = modelproof(open(os.path.join(base, L.MODEL_PATH), encoding="utf-8").read(), open(mp, encoding="utf-8").read())
        info["modelproof"] = {k: proof[k] for k in ("ok", "changed", "added", "removed")}
        if not proof["ok"]:
            pr.append("modelproof: " + "; ".join(proof["problems"])[:500])
        reach = cfg_reach(open(mp, encoding="utf-8").read(), open(mp, encoding="utf-8").read())
        if reach["missing"]:
            pr.append(f"the tree's own V8FMiniConfig lacks {reach['missing']}")
    info["digest"] = tree_digest(tree, ds_t)
    info["base_digest"] = tree_digest(base, ds_b)
    if info["base_digest"] != base_digest and L.on("tree-base-digest"):
        pr.append(f"the base tree's digest {info['base_digest'][:12]} is not the one CTRL / REP were rolled out with ({base_digest[:12]})")
    info["differs_from_base"] = diff
    return (not pr), info


# ------------------------------------------------------------------------------------------------------------------------------ the AST proof that the overlay changes nothing on the inference path
def _qual(src: str) -> dict:
    tree = ast.parse(src)
    out: dict = {}

    def seg(n):
        return ast.get_source_segment(src, n) or ""

    def visit(body, prefix):
        other = []
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + n.name] = (ast.dump(n), seg(n))
            elif isinstance(n, ast.ClassDef):
                head = ast.ClassDef(name=n.name, bases=n.bases, keywords=n.keywords, body=[], decorator_list=n.decorator_list, type_params=getattr(n, "type_params", []))
                out[prefix + n.name + ".<header>"] = (ast.dump(head), n.name)
                visit(n.body, prefix + n.name + ".")
            else:
                other.append(n)
        out[prefix + "<body>"] = ("\n".join(ast.dump(o) for o in other), "\n".join(seg(o) for o in other))

    visit(tree.body, "")
    return out


def _diff_lines(a: str, b: str) -> list:
    return [ln for ln in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0) if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]


def src_diff(a_src: str, b_src: str, keyword: str, name_ok: str | None = None) -> dict:
    """Qualified-name diff of two sources.  ok iff nothing is removed, every ADDED definition has `keyword` in its name, and every +/- line of every CHANGED segment mentions `keyword` (comments and blanks aside).
    A qualified name matching ``name_ok`` (a whole class that is the lever's own) is accepted without the line test."""
    qa, qb = _qual(a_src), _qual(b_src)
    changed = sorted(k for k in qa if k in qb and qa[k][0] != qb[k][0])
    added = sorted(k for k in qb if k not in qa)
    removed = sorted(k for k in qa if k not in qb)
    pr = []
    rx = re.compile(keyword, re.I)
    if removed and L.on("proof-removed"):
        pr.append(f"definitions REMOVED: {removed[:4]}")
    nok = re.compile(name_ok) if name_ok else None
    for k in added:
        if not rx.search(k) and not (nok and nok.match(k)) and not k.endswith("<body>") and L.on("proof-added"):
            pr.append(f"definition ADDED that does not mention the lever: {k}")
    for k in changed:
        if nok and nok.match(k):
            continue
        for ln in _diff_lines(qa[k][1], qb[k][1]):
            body = ln[1:].strip()
            if body and not body.startswith("#") and not rx.search(body) and L.on("proof-lines"):
                pr.append(f"{k}: a changed line does not mention the lever: {ln[:120]}")
                break
    return {"ok": not pr, "changed": changed, "added": added, "removed": removed, "problems": pr}


def modelproof(pinned_src: str, launch_src: str) -> dict:
    return src_diff(pinned_src, launch_src, r"launch")


def config_fields(src: str, cls: str = "V8FMiniConfig") -> list:
    for n in ast.parse(src).body:
        if isinstance(n, ast.ClassDef) and n.name == cls:
            return [m.target.id for m in n.body if isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name)]
    return []


def cfg_reach(tree_model_src: str, launch_model_src: str) -> dict:
    """The trainer saves `cfg.__dict__` = every dataclass field of the LAUNCH model's V8FMiniConfig; the tool builds V8FMiniConfig(**cfg) from the TREE's model.  missing = fields the tree's config would reject."""
    w, a = config_fields(launch_model_src), config_fields(tree_model_src)
    return {"written_by_trainer": len(w), "accepted_by_tree": len(a), "missing": sorted(set(w) - set(a))}


# ------------------------------------------------------------------------------------------------------------------------------ child-python probes (no rollout)
def cfgprobe(tree: str, ckpt: str, extra: dict) -> tuple:
    """Build the config of a banked checkpoint (+ the keys the launch trainer adds) through `tree`'s own model module, exactly as v8f_t1_eval.py:506 does.  -> (accepted, text)."""
    code = ("import sys, json, torch\nsys.path.insert(0, sys.argv[1] + '/stack')\nfrom tanitad.models import v8f_mini as M\n"
            "ck = torch.load(sys.argv[2], map_location='cpu', weights_only=False)\ncfg = dict(ck['cfg']); cfg.update(json.loads(sys.argv[3]))\n"
            "try:\n    c = M.V8FMiniConfig(**cfg); print('CFG-ACCEPTED launch_intent_w=%r' % getattr(c, 'launch_intent_w', 'ABSENT'))\n"
            "except TypeError as e:\n    print('CFG-REJECTED ' + str(e)[:200])\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([L.PY, "-I", "-B", "-c", code, tree, ckpt, json.dumps(extra)], capture_output=True, env=env, stdin=subprocess.DEVNULL)
    out = (r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")).strip().splitlines()
    line = next((x for x in out if x.startswith("CFG-")), out[-1] if out else "no output")
    return line.startswith("CFG-ACCEPTED"), line


_CAP = ("import argparse, importlib.util, json, sys\nclass Cap(Exception): pass\norig = argparse.ArgumentParser.parse_args\n"
        "def pa(self, args=None, namespace=None):\n    ns = orig(self, args, namespace)\n    print('ZZPARSED ' + json.dumps(vars(ns), default=str))\n    raise Cap()\n"
        "argparse.ArgumentParser.parse_args = pa\nspec = importlib.util.spec_from_file_location('lit1_tool', sys.argv[1])\nm = importlib.util.module_from_spec(spec)\nsys.modules['lit1_tool'] = m\n"
        "spec.loader.exec_module(m)\ntry:\n    m.main(sys.argv[2:])\nexcept Cap:\n    pass\n")


def parsecheck(tool: str, args: list) -> tuple:
    """Parse `args` (the tokens after the script path) with the tool's own argparse; the run is cut off by a patched parse_args BEFORE anything executes.  -> (ok, parsed namespace or the error)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env["TANITEVAL_STACK_OVERRIDE"] = os.path.dirname(os.path.dirname(os.path.abspath(tool)))      # <tree>/stack: the tool's OWN tree, never another tree's
    r = subprocess.run([L.PY, "-I", "-B", "-c", _CAP, tool, *args], capture_output=True, env=env, stdin=subprocess.DEVNULL)
    out = r.stdout.decode("utf-8", "replace")
    for ln in out.splitlines():
        if ln.startswith("ZZPARSED "):
            return True, json.loads(ln[9:])
    return False, (r.stderr.decode("utf-8", "replace") or out)[-400:]


# ------------------------------------------------------------------------------------------------------------------------------ the runtime import set vs the launch commit
def rolloutcode(mirror: str, launch: str, base: str) -> dict:
    """Every file the harness LOADED (RUNTIME_SET) in the pinned tree vs the launch commit's blob.  Allowed differences: the model (the overlay, proven by modelproof) and the data module IF every changed definition
    mentions the ego perturbation and no other runtime file does.  Anything else differing is a finding that STOPs the chain."""
    res = {"n": len(RUNTIME_SET), "identical": [], "differs": [], "absent_at_launch": [], "problems": []}
    for p in RUNTIME_SET:
        a = open(os.path.join(base, p), "rb").read()
        b = blob(mirror, launch, p)
        if b is None:
            if a:
                res["problems"].append(f"{p} is absent at the launch commit and not empty in the pinned tree")
            res["absent_at_launch"].append(p)
        elif a == b:
            res["identical"].append(p)
        else:
            res["differs"].append(p)
    for p in res["differs"]:
        a, b = open(os.path.join(base, p), encoding="utf-8").read(), blob(mirror, launch, p).decode("utf-8")
        if p == L.MODEL_PATH:
            pf = modelproof(a, b)
            res["model"] = {k: pf[k] for k in ("ok", "changed", "added")}
            if not pf["ok"]:
                res["problems"].append("model: " + "; ".join(pf["problems"])[:300])
        elif p == L.DATA_PATH:
            pf = src_diff(a, b, r"EGO_PERTURB", name_ok=r"EgoPerturb(\.|$)")
            res["data"] = {k: pf[k] for k in ("ok", "changed", "added")}
            if not pf["ok"]:
                res["problems"].append("data: " + "; ".join(pf["problems"])[:300])
            users = [q for q in RUNTIME_SET if q != p and re.search(r"EgoPerturb|ego_perturb", open(os.path.join(base, q), encoding="utf-8", errors="replace").read())]
            res["data"]["runtime_files_naming_the_perturbation"] = users
            if users:
                res["problems"].append(f"runtime files name the perturbation the data module changed: {users[:3]}")
        else:
            res["problems"].append(f"{p} differs between the pinned tree and the launch commit")
    res["ok"] = not res["problems"]
    return res


# ------------------------------------------------------------------------------------------------------------------------------ K-BUILD
def _pm_lines(diff_text: str) -> list:
    return [ln for ln in diff_text.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]


def kbuild(mirror: str, launch: str, compose_record: str = COMPOSE_RECORD, base_commit: str = BASE_COMMIT, li_parent: str = LI_PARENT, flag_commit: str = L.LAUNCH_COMMIT) -> dict:
    """The K-BUILD record for the reader's k_build(): {"launch_commit", "base_commit", "files", "changed"}.  sha256 and hunks are MEASURED from git objects.  The record is TWO-STEP because the landed commit sits on top of DAG-ORC r2's
    compose commit:  step 1 = base (aeb54468) -> LI_PARENT (the compose flag): its hunks and PASS come from DAG-ORC r2's accepted K-BUILD record (its launch files must equal LI_PARENT's blobs);  step 2 = LI_PARENT -> launch (the
    launch-intent flag): hunks computed here; PASS only if the evidence at the landed commit holds (diff.patch == the landed diff, green log, mutation summary, test, golden, default-off declarations, launch bytes == tested bytes).
    Anything unverified is written 'UNVERIFIED: <why>', which the reader refuses.  NOT hidden: v8f_mini_data.py differs from the base (the compose step) -- the reader decides whether that is admissible."""
    for c in (launch, base_commit, li_parent, flag_commit):
        git(mirror, "cat-file", "-e", f"{c}^{{commit}}")
    files, ev, prob = {}, {}, {}
    for f in BUILD_FILES:
        bl, bb = blob(mirror, launch, f), blob(mirror, base_commit, f)
        files[f] = {"sha256_launch": sha_b(bl), "sha256_base": sha_b(bb)}
    mid = {f: sha_b(blob(mirror, li_parent, f)) for f in BUILD_FILES}
    changed: list = []
    # step 1: compose (base -> LI_PARENT), from DAG-ORC r2's accepted record
    comp = json.load(open(compose_record, encoding="utf-8"))
    ev["compose_record"] = {"path": compose_record, "sha256": L.sha256_file(compose_record), "launch_commit": comp.get("launch_commit")}
    for f in BUILD_FILES:
        cf = (comp.get("files") or {}).get(f) or {}
        if (cf.get("sha256_launch") != mid[f] or cf.get("sha256_base") != files[f]["sha256_base"]) and L.on("kb-compose-bytes"):
            prob[f"compose:{f}"] = f"the DAG-ORC r2 record's launch/base blobs for {f} are not LI_PARENT's / the base's"
    for h in comp.get("changed") or []:
        ok = not prob and h.get("bit_identity_test") == "PASS" and h.get("default_off") is True
        changed.append({"file": h["file"], "hunk": h["hunk"], "behind_flag": h.get("behind_flag"), "default_off": bool(h.get("default_off")) and not prob,
                        "bit_identity_test": "PASS" if ok else "UNVERIFIED: " + ("; ".join(f"{k}: {v}" for k, v in prob.items()) or "the DAG-ORC record's hunk is not PASS"),
                        "step": "base->LI_PARENT (compose flag --ego-perturb-with-dagger), INHERITED from the DAG-ORC r2 K-BUILD record"})
    # step 2: launch intent (LI_PARENT -> launch)
    ev_prob: dict = {}
    base_pkg = f"{RAWPK}"
    flag_sha = {f: (blob(mirror, flag_commit, f) and sha_b(blob(mirror, flag_commit, f))) for f in BUILD_FILES}
    for f in (L.TRAINER_PATH, L.MODEL_PATH):
        if files[f]["sha256_launch"] != flag_sha[f] and L.on("kb-launch-bytes"):
            ev_prob[f"bytes:{f}"] = f"the launch commit's {f} is not the landed flag commit's (the tests ran against the landed bytes)"
    dp = blob(mirror, flag_commit, f"{base_pkg}/diff.patch")
    landed = git(mirror, "diff", "--no-color", "-U0", li_parent, flag_commit, "--", L.TRAINER_PATH, L.MODEL_PATH, text=True)
    dp_ok = dp is not None and sha_b("\n".join(_pm_lines(dp.decode("utf-8"))).encode()) == sha_b("\n".join(_pm_lines(landed)).encode())
    if not dp_ok and L.on("kb-diffpatch"):
        ev_prob["diff.patch"] = "the package's diff.patch (the diff the tests ran against) is not the landed diff of the trainer and the model"
    elif dp is not None:
        ev["diff_patch"] = {"path": f"{base_pkg}/diff.patch", "sha256": sha_b(dp), "pm_lines": len(_pm_lines(landed))}
    gl = blob(mirror, flag_commit, f"{base_pkg}/logs/final_green.log")
    gtxt = gl.decode("utf-8", "replace") if gl else ""
    m = re.findall(r"\b(\d+) passed\b", gtxt)
    green_ok = bool(m) and int(m[-1]) >= 36 and not re.search(r"\b\d+ (failed|error)", gtxt)
    if not green_ok and L.on("kb-green"):
        ev_prob["green"] = "logs/final_green.log does not end in '>= 36 passed' with no failure"
    elif gl is not None:
        ev["green_log"] = {"passed": int(m[-1]) if m else None, "sha256": sha_b(gl)}
    mt = blob(mirror, flag_commit, f"{base_pkg}/mutation.txt")
    mm = re.search(r"SUMMARY: (\d+)/(\d+) mutants killed; survivors: none", mt.decode("utf-8", "replace")) if mt else None
    if not (mm and mm.group(1) == mm.group(2) and int(mm.group(2)) >= 30) and L.on("kb-mutation"):
        ev_prob["mutation"] = "mutation.txt does not show 'SUMMARY: n/n mutants killed; survivors: none' with n >= 30"
    elif mt is not None:
        ev["mutation"] = {"killed": int(mm.group(1)) if mm else None, "sha256": sha_b(mt)}
    tf = blob(mirror, flag_commit, "stack/tests/test_v8f_launch_intent.py")
    if (tf is None or b"def test_flag_off_reproduces_the_unmodified_tip_bit_for_bit" not in tf) and L.on("kb-flagoff-test"):
        ev_prob["flagoff_test"] = "stack/tests/test_v8f_launch_intent.py lacks test_flag_off_reproduces_the_unmodified_tip_bit_for_bit"
    gd = blob(mirror, flag_commit, "stack/tests/launch_intent_golden.json")
    if not gd and L.on("kb-golden"):
        ev_prob["golden"] = "stack/tests/launch_intent_golden.json is absent at the landed commit"
    else:
        ev["flag_off_test"] = {"test_sha256": sha_b(tf or b""), "golden_sha256": sha_b(gd)}
    tsrc = (blob(mirror, launch, L.TRAINER_PATH) or b"").decode("utf-8", "replace")
    i = tsrc.find('add_argument("--launch-intent-w"')
    snip = tsrc[i:i + 300] if i >= 0 else ""
    msrc = (blob(mirror, launch, L.MODEL_PATH) or b"").decode("utf-8", "replace")
    default_off = bool(i >= 0 and re.search(r"default\s*=\s*0\.0", snip) and re.search(r"launch_intent_w\s*:\s*float\s*=\s*0\.0", msrc))
    if not default_off and L.on("kb-default-off"):
        ev_prob["default_off"] = "the launch trainer / model do not declare --launch-intent-w / launch_intent_w with default 0.0"
    for f in (L.TRAINER_PATH, L.MODEL_PATH):
        if files[f]["sha256_launch"] != mid[f]:
            d = git(mirror, "diff", "-U0", "--no-color", li_parent, launch, "--", f, text=True)
            hunks = re.findall(r"^(@@ [^@]+ @@)", d, re.M)
            for h in hunks or ["(no hunk header parsed)"]:
                changed.append({"file": f, "hunk": h, "behind_flag": "--launch-intent-w", "default_off": default_off and not ev_prob,
                                "bit_identity_test": "PASS" if (not ev_prob and hunks) else "UNVERIFIED: " + "; ".join(f"{k}: {v}" for k, v in ev_prob.items())[:400],
                                "step": "LI_PARENT->launch (launch-intent flag)"})
    if files[L.DATA_PATH]["sha256_launch"] != mid[L.DATA_PATH]:
        changed.append({"file": L.DATA_PATH, "hunk": "(launch differs from LI_PARENT)", "behind_flag": None, "default_off": False, "bit_identity_test": "UNVERIFIED: the launch commit changes the data module after the tested commit"})
    return {"launch_commit": launch, "base_commit": base_commit, "files": files, "changed": changed,
            "identical_files": [f for f in BUILD_FILES if files[f]["sha256_launch"] == files[f]["sha256_base"]], "evidence": ev, "li_parent": li_parent, "landed_flag_commit": flag_commit,
            "attested_by": "lit1_tree.py kbuild: sha256 and hunks MEASURED from git objects; the compose step INHERITED from DAG-ORC r2's accepted record (bound by blob sha256); the launch-intent step bound to the tested bytes by diff.patch == the landed diff"}


def reader_kbuild(reader: str, rec: dict) -> tuple:
    """The reader's OWN k_build on a record (import by path; needs numpy)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("lit1_reader_probe", reader)
    m = importlib.util.module_from_spec(spec)
    sys.modules["lit1_reader_probe"] = m
    spec.loader.exec_module(m)
    return m.k_build(rec)


# ------------------------------------------------------------------------------------------------------------------------------ the manifest
def t1_entry(out: str, arm: str, seed: int, limit: str) -> dict:
    t = L.t1_tag(arm, seed, limit)
    return {"tag": t, "ticks": f"{out}/ticks/ticks_{t}.npz", "dump_dir": f"{out}/dump_{t}", "json": f"{out}/t1v8f_{t}.json", "argv": f"{out}/argv/{t}.argv", "done": f"{out}/done_{t}.txt"}


def build_manifest(art: str, out: str, ctrl_runs: str, rep_runs: str, build: dict | None, tool: dict | None = None) -> dict:
    arms: dict = {"A": {}, "Arep": {}}
    for arm, s in L.RUNS:
        run, tg = L.run_name(arm, s), L.eval_tag(arm, s)
        d = f"{art}/{run}"
        arms[L.ROLE[arm]][str(s)] = {"ckpt_tail": f"{run}/ckpt_last.pt", "train_dir": d, "token": f"{d}/launch_gate_token.json", "t1": t1_entry(out, arm, s, "causal"), "t1_logged": t1_entry(out, arm, s, "logged"),
                                     "t0": {"json": f"{d}/eval_{tg}.json", "npz": f"{d}/eval_{tg}.npz"},
                                     "t0_fresh": {"json": f"{out}/t0fresh/eval_{L.t0f_tag(arm, s)}.json", "npz": f"{out}/t0fresh/eval_{L.t0f_tag(arm, s)}.npz"}}
    arms["CTRL"] = L.jload(ctrl_runs)["arms"]["DAGOACC"]
    arms["REP"] = L.jload(rep_runs)["arms"]["REP"]
    m = {"seeds": [0, 1], "arms": arms, "provenance": {
        "prereg": "Project Steering/PREREG_V8F_LAUNCH_INTENT.md (draft v2, MM_REDESIGN_2026-10-28), claim row H-V8F-LAUNCH-INTENT-1",
        "built_by": "lit1_tree.py manifest (the dev-box T1-kin chain, lit1_chain.sh)",
        "slots": "A := launchintent (A_s), Arep := launchintentrep (A_rep,s), CTRL := DAG-OACC (banked: v8f-dagoacc + v8f-dagoacc-t1), REP := DAG-ORC r2's dagoaccorcrep (banked: v8f-dag-orc-oacc + dagorc_t1_out). B / Brep are not queued.",
        "t1": "v8f_t1_eval.py run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --dump-ticks --dump-heads, plain (no post-map); lanes: --limit causal (decisive) and --limit logged (K4 only)",
        "t0": "box 2's standard eval (--ages cycle): R-T0", "t0_fresh": "dev-box CPU --ages fresh eval: K4's comparator",
        "tool": tool or {}}}
    if build is not None:
        m["build"] = build
    return m


def verify_manifest(m: dict, n_dump_min: int, tool_digest: str | None = None) -> list:
    """Every file the manifest points at exists; every new-arm entry is THIS arm's; the done records name the pulled checkpoint and the tool digest; no smoke output is in it; CTRL / REP are the banked checkpoints."""
    bad: list = []

    def need(p, what):
        if not p or not os.path.isfile(p) or os.path.getsize(p) == 0:
            bad.append(f"{what}: {p} missing or empty")

    seen: dict = {}
    for arm, seed in L.RUNS:
        role, lab = L.ROLE[arm], L.LABEL[(arm, seed)]
        e = (m["arms"].get(role) or {}).get(str(seed))
        if not e:
            bad.append(f"{lab}: no entry")
            continue
        d, run = e["train_dir"], L.run_name(arm, seed)
        if not L.npath(d).endswith("/" + run) or e.get("ckpt_tail") != f"{run}/ckpt_last.pt":
            bad.append(f"{lab}: train_dir / ckpt_tail {d} / {e.get('ckpt_tail')} is not {run}")
        ck = f"{d}/ckpt_last.pt"
        need(ck, f"{lab} ckpt")
        sha = L.sha256_file(ck) if os.path.isfile(ck) else None
        if sha in seen:
            bad.append(f"{lab}: its checkpoint {str(sha)[:12]} is also {seen[sha]}'s")
        if sha:
            seen[sha] = lab
            if sha in (L.OACC_CKPT_SHA[seed], L.REP_CKPT_SHA[seed], L.M2B_SHA[seed]):
                bad.append(f"{lab}: its checkpoint IS a control's (nothing was trained)")
        for k in (e["token"], e["t0"]["json"], e["t0"]["npz"]):
            need(k, f"{lab} file")
        for f in ("argv.json", "summary.json", "selfcheck.json"):
            need(f"{d}/{f}", f"{lab} {f}")
        for kind, want_limit in (("t1", "causal"), ("t1_logged", "logged")):
            t = e[kind]
            for k in ("ticks", "json", "done", "argv"):
                need(t[k], f"{lab} {kind} {k}")
            nd = len([x for x in os.listdir(t["dump_dir"]) if x.startswith("ep") and x.endswith(".npz")]) if os.path.isdir(t["dump_dir"]) else 0
            if nd < n_dump_min:
                bad.append(f"{lab} {kind}: {nd} ep*.npz in {t['dump_dir']} (< {n_dump_min})")
            if os.path.isfile(t["json"]):
                j = L.jload(t["json"]).get("t1v8f", {})
                if L.tail2(j.get("ckpt", "")) != e["ckpt_tail"]:
                    bad.append(f"{lab} {kind}: the rollout json scored {L.tail2(j.get('ckpt', ''))}, not {e['ckpt_tail']}")
                if j.get("max_windows") is not None or j.get("limit") != want_limit or j.get("n_windows") != L.N_WINDOWS:
                    bad.append(f"{lab} {kind}: rollout json max_windows {j.get('max_windows')!r} limit {j.get('limit')!r} n_windows {j.get('n_windows')!r} (a full {want_limit} run is required)")
            if os.path.isfile(t["done"]):
                dr = L.jload(t["done"])
                if dr.get("ckpt_sha256") != sha or dr.get("ckpt_sha256_before") != sha or dr.get("ckpt_sha256_after") != sha or dr.get("smoke") is not False:
                    bad.append(f"{lab} {kind}: done record names {str(dr.get('ckpt_sha256'))[:12]} (before {str(dr.get('ckpt_sha256_before'))[:8]}, after {str(dr.get('ckpt_sha256_after'))[:8]}) smoke={dr.get('smoke')!r}; the pulled checkpoint is {str(sha)[:12]}")
                if tool_digest and dr.get("tool_digest") != tool_digest:
                    bad.append(f"{lab} {kind}: done record tool digest {str(dr.get('tool_digest'))[:12]} is not the LAUNCH-INTENT tree's {tool_digest[:12]}")
        tf = e["t0_fresh"]
        for k in ("json", "npz"):
            need(tf[k], f"{lab} t0_fresh {k}")
        dn = os.path.join(os.path.dirname(os.path.dirname(tf["json"])), f"done_t0f_{L.t0f_tag(arm, seed)}.txt")
        need(dn, f"{lab} fresh-T0 done record")
        if os.path.isfile(dn):
            dr = L.jload(dn)
            if dr.get("ckpt_sha256") != sha or (tool_digest and dr.get("tool_digest") != tool_digest) or dr.get("smoke") is not False:
                bad.append(f"{lab}: the fresh-T0 done record names another checkpoint / tool than the pulled one")
        for pj, want in ((tf["json"], "fresh"), (e["t0"]["json"], "cycle")):
            if os.path.isfile(pj):
                jj = L.jload(pj)
                if jj.get("ages") != want or L.tail2(str(jj.get("ckpt", ""))) != e["ckpt_tail"]:
                    bad.append(f"{lab}: {os.path.basename(pj)} ages {jj.get('ages')!r} ckpt {L.tail2(str(jj.get('ckpt', '')))!r} (expected {want}, {e['ckpt_tail']})")
    for role, want in (("CTRL", L.OACC_CKPT_SHA), ("REP", L.REP_CKPT_SHA)):
        for seed in (0, 1):
            e = (m["arms"].get(role) or {}).get(str(seed))
            if not e:
                bad.append(f"{role} s{seed}: no entry")
                continue
            ck = f"{e['train_dir']}/ckpt_last.pt"
            if not os.path.isfile(ck) or L.sha256_file(ck) != want[seed]:
                bad.append(f"{role} s{seed}: {ck} is not the banked checkpoint {want[seed][:12]}")
            need(e["t1"]["json"], f"{role} s{seed} t1 json")
            need(e["token"], f"{role} s{seed} token")
    return bad


# ------------------------------------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(newline="\n", encoding="utf-8")
        except Exception:      # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(prog="lit1_tree.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("treeplan", "treedigest", "treebuild", "treecheck", "modelproof", "cfgprobe", "rolloutcode", "kbuild", "manifest", "verify", "parsecheck"):
        p = sub.add_parser(nm)
        p.add_argument("--base", default=L.BASE_TREE)
        p.add_argument("--mirror", default="C:/Users/Admin/tanitad-push")
        p.add_argument("--dest")
        p.add_argument("--tree")
        p.add_argument("--pinned")
        p.add_argument("--launch", default=L.LAUNCH_COMMIT)
        p.add_argument("--reader")
        p.add_argument("--ckpt", default=f"{L.AR}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt")
        p.add_argument("--art")
        p.add_argument("--out")
        p.add_argument("--ctrl-runs", default=f"{L.AR}/v8f-dagoacc-t1/runs.json")
        p.add_argument("--rep-runs", default="F:/TanitAD-scratch/dagorc_t1_out/runs.json")
        p.add_argument("--write")
        p.add_argument("--manifest")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--n-dump-min", type=int, default=L.N_EPISODES)
        p.add_argument("--tool-digest")
        p.add_argument("--tool")
        p.add_argument("--build-json")
        p.add_argument("rest", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "treeplan":
        print(json.dumps(treeplan(a.base), indent=1))
        return 0
    if a.cmd == "treedigest":
        print(tree_digest(a.tree))
        return 0
    if a.cmd == "treebuild":
        print(treebuild(a.dest, a.base, a.mirror))
        return 0
    if a.cmd == "treecheck":
        ok, info = treecheck(a.tree, a.base, a.mirror)
        print(json.dumps(info, indent=1))
        print("TREE-OK" if ok else "TREE-NOT-OK")
        return 0 if ok else 3
    if a.cmd == "modelproof":
        pf = modelproof(open(a.pinned, encoding="utf-8").read(), open(a.tree, encoding="utf-8").read())
        print(json.dumps(pf, indent=1))
        return 0 if pf["ok"] else 3
    if a.cmd == "cfgprobe":
        ok, line = cfgprobe(a.tree, a.ckpt, {"launch_intent_w": L.LI_W})
        print(line)
        return 0 if ok else 3
    if a.cmd == "rolloutcode":
        r = rolloutcode(a.mirror, a.launch, a.base)
        print(json.dumps({k: v for k, v in r.items() if k != "identical"}, indent=1))
        print("ROLLOUT-CODE-OK" if r["ok"] else "ROLLOUT-CODE-NOT-OK")
        return 0 if r["ok"] else 3
    if a.cmd == "kbuild":
        rec = kbuild(a.mirror, a.launch)
        print(json.dumps(rec, indent=1))
        if a.reader:
            ok, det = reader_kbuild(a.reader, rec)
            print("READER-K-BUILD", "PASS" if ok else "REFUSED", json.dumps(det, default=str)[:800] if not ok else "")
            return 0 if ok else 3
        return 0
    if a.cmd == "manifest":
        build = json.load(open(a.build_json, encoding="utf-8")) if a.build_json else kbuild(a.mirror, a.launch)
        tool = {"base_tree": a.base, "base_digest": L.BASE_DIGEST, "launch_tree": a.tree, "launch_tree_digest": a.tool_digest}
        m = build_manifest(a.art, a.out, a.ctrl_runs, a.rep_runs, build, tool)
        if a.dry_run:
            print(json.dumps(m, indent=1)[:6000])
            return 0
        bad = verify_manifest(m, a.n_dump_min, a.tool_digest)
        if bad:
            print("MANIFEST-REFUSED", *bad, sep="\n  ")
            return 3
        with open(a.write, "w", encoding="utf-8", newline="\n") as f:
            json.dump(m, f, indent=1)
        print(f"MANIFEST-WRITTEN {a.write} (A/Arep x 2 seeds + banked CTRL/REP; build {'present' if build else 'absent'})")
        return 0
    if a.cmd == "verify":
        bad = verify_manifest(L.jload(a.manifest), a.n_dump_min, a.tool_digest)
        print("MANIFEST-OK" if not bad else "MANIFEST-BAD\n  " + "\n  ".join(bad))
        return 0 if not bad else 3
    if a.cmd == "parsecheck":
        ok, ns = parsecheck(a.tool, a.rest)
        print("PARSE-OK" if ok else "PARSE-FAIL", json.dumps(ns, default=str)[:300] if ok else ns)
        return 0 if ok else 3
    return 2


if __name__ == "__main__":
    sys.exit(main())
