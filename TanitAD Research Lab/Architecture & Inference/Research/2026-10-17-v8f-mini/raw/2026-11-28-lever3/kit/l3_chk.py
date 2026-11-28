#!/usr/bin/env python3
"""l3_chk.py -- the dev-box port of the pod's chk_arm_s3.sh for the LEVER3 arms (D3K_s0, D3K_s1, D9K_s0, D9K_s1): `prep | mint | run ARM`.
The gate is the study-side checker kit/dscale10k_check_l3.py (= the pod's dscale10k_check_s3.py + the four arms; patch_l3.py), minting `dscale10k-gate-token-v1` into <root>/runs/ARM/gate/gate_gatetok.json.
`run` REFUSES (rc 2, nothing started) without a token that binds this exact argv, commit, data, tree, trainer, checker, template and pair, then starts the trainer under the same argv.
Inputs (every one named in REPORT_GATE.md): templates and records from kit/inputs/records (the pod's records tar), the arm's argv from l3_argv.py, the pod's G-SUITE result of 97fe0a56 (the tree is blob-identical,
kit/tree_ok.json), the stage-A evidence of UE_sK (INHERITED, as the pod's rh_evidence.py), the arm's own bank and selfcheck, the cache manifest and the join from the full build.
usage: l3_chk.py prep ARM | mint ARM | run ARM   [--root C:/Users/Admin/lever3] [--selfcheck PATH] [--dry-gate-dir DIR]   (env L3_ROOT also sets the root; DRY: --dry-gate-dir must contain /dry_, mint only)"""
import argparse, copy, glob, json, os, subprocess, sys

KIT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, KIT)
import l3_argv as A  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
COMMIT = "97fe0a56767320dbc5f4b823d67fb07a3209a745"
REC = KIT + "/inputs/records"
CHECKER = KIT + "/dscale10k_check_l3.py"
TENK_V9 = "F:/TanitAD-scratch/rollhist/lever3/labels/in/tenk/labels/v9_labels_datascale10k.npz"
CORPUS_MANIFEST = "F:/TanitAD-scratch/speedread/cache/train/manifest.jsonl"       # the holdout manifest: md5 71b0f65c... = the merge record's corpus_manifest
CORPUS_JOIN = KIT + "/inputs/join_train_rebuilt.npz"       # re-created by l3_corpus_join.py (trainer build_data --fit-join on the corpus inputs); md5 9ebc7a48cbca45cc634eab2aab5a3960 = the required prefix 9ebc7a48 (MEASURED)
EVAL_IDS = KIT + "/inputs/clips141.txt"
DATA_PATH_FLAGS = ("--cache-dir", "--v9-train", "--lead-train", "--speed-train", "--join-train", "--p10", "--tracks")


def load(p):
    o = json.load(open(p, encoding="utf-8"))
    return o["argv"] if isinstance(o, dict) else o


def mapped_u_template(seed, root):
    """The pod's stage-B template U_s<seed>.B.tmpl.json with ONLY the data / p10 path flags mapped to the dev box (the arm's own flags --out --init-ckpt --dagger-bank* keep the template's text)."""
    t = load(f"{REC}/dscale10k/argv/U_s{seed}.B.tmpl.json")
    p = A.paths(f"D3K_s{seed}", root)
    for f in DATA_PATH_FLAGS:
        t[t.index(f) + 1] = p[f]
    return t


def mapped_base(root):
    """The B_s0 stage-B template with ONLY --p10 mapped (the data flags of the B set stay: a union arm may differ from them, argv_diff_problems)."""
    t = load(f"{REC}/dscale10k/argv/B_s0.B.tmpl.json")
    t[t.index("--p10") + 1] = A.P10
    return t


def pair_of(arm):
    kind, seed = arm.split("_s")
    return f"{'D9K' if kind == 'D3K' else 'D3K'}_s{seed}"


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw).returncode


def prep(arm, root):
    seed = arm.split("_s")[1]
    D = f"{root}/runs/{arm}"
    GD = f"{D}/gate"
    os.makedirs(GD, exist_ok=True)
    json.dump({"argv": mapped_base(root)}, open(f"{GD}/base_B_s0.B.mapped.json", "w"), indent=1)
    json.dump({"argv": mapped_u_template(seed, root)}, open(f"{GD}/tmpl_U_s{seed}.B.mapped.json", "w"), indent=1)
    return sh([PY, "-I", KIT + "/rh_evidence.py", "--ue-evidence", f"{REC}/dscale10k/gate/evidence_UE_s{seed}.json", "--gsuite", f"{REC}/rollhist/gsuite_result_rh.json",
               "--commit", COMMIT, "--out", f"{GD}/evidence_{arm}.json"])


def files(arm, root, selfcheck=None, merge_record=True):
    seed = arm.split("_s")[1]
    D = f"{root}/runs/{arm}"
    GD = f"{D}/gate"
    return [
        "--evidence", f"{GD}/evidence_{arm}.json", "--dry", f"{REC}/dscale10k/dry/UE_s{seed}.A.dry.json", "--selfcheck", selfcheck or f"{D}/selfcheck/selfcheck.json",
        ] + (["--merge-record", f"{root}/merged/merge_record.json"] if merge_record else []) + ["--union-release", f"{root}/merged/v9_labels_union.npz", "--tenk-v9", TENK_V9,
        "--template", f"{GD}/tmpl_U_s{seed}.B.mapped.json", "--pair-argv", f"{root}/runs/{pair_of(arm)}/argv.json", "--readout-token", "causal_g1x"]


def common(arm, root, gate_dir):
    D = f"{root}/runs/{arm}"
    return ["--arm", arm, "--stage", "B", "--argv", f"{D}/argv.json", "--root", f"{root}/tree_cdb", "--commit", COMMIT, "--gate-dir", gate_dir, "--py", PY, "--gpu", "0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "mint", "run"])
    ap.add_argument("arm", choices=A.ARMS)
    ap.add_argument("--root", default=os.environ.get("L3_ROOT", "C:/Users/Admin/lever3"))
    ap.add_argument("--selfcheck", default=None)
    ap.add_argument("--dry-gate-dir", default=None)
    ap.add_argument("--no-merge-record", action="store_true", help="dry mint only: skip C-MERGE (its rebuild of the 2 GB v9 union needs a few GB of free commit)")
    ap.add_argument("--corpus-join", default=CORPUS_JOIN)
    ap.add_argument("--eval-ids", default=EVAL_IDS)
    a = ap.parse_args()
    root, arm = a.root.replace("\\", "/"), a.arm
    seed = arm.split("_s")[1]
    D = f"{root}/runs/{arm}"
    gd = f"{D}/gate"
    if a.dry_gate_dir:
        if a.cmd != "mint" or "/dry_" not in a.dry_gate_dir.replace("\\", "/"):
            print("REFUSED: --dry-gate-dir is mint only and must contain /dry_")
            return 2
        gd = a.dry_gate_dir.replace("\\", "/")
    if a.cmd == "prep":
        return prep(arm, root)
    if a.cmd == "run" and a.selfcheck:
        print("REFUSED: --selfcheck is a dry-mint option; `run` binds the arm's own selfcheck")
        return 2
    os.makedirs(gd, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=";".join([f"{root}/tree_cdb/stack", f"{root}/tree_cdb/stack/scripts", f"{root}/tree_cdb", KIT]), PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_XET="1",
               DS10K_TMP=os.environ.get("DS10K_TMP", "F:/TanitAD-scratch/rollhist/lever3/kit/tmp_cmerge"))
    if a.cmd == "mint":
        os.makedirs(env["DS10K_TMP"], exist_ok=True)
        rc = prep(arm, root)
        if rc:
            print("REFUSED: evidence:", rc)
            return 1
        dig = sorted(glob.glob(f"{root}/tree_cdb/stack/tanitad/data/*_clip_digests.json"))
        cmd = [PY, "-I", CHECKER, "mint"] + common(arm, root, gd) + ["--base-argv", f"{D}/gate/base_B_s0.B.mapped.json"] + files(arm, root, a.selfcheck, not a.no_merge_record) + [
            "--corpus-join", a.corpus_join, "--union-join", f"{root}/merged/join_union.npz", "--eval-ids", a.eval_ids, "--holdout-manifest", CORPUS_MANIFEST, "--digest-json"] + dig
        # -I would drop PYTHONPATH: the checker puts its own folder on sys.path (dscale10k_merge.py sits beside it) and needs the tree for dry/selfcheck only
        cmd.remove("-I")
        return subprocess.run([str(c) for c in cmd], env=env, cwd=KIT).returncode
    cmd = [PY, CHECKER, "run"] + common(arm, root, gd) + files(arm, root)
    return subprocess.run([str(c) for c in cmd], env=env, cwd=KIT).returncode        # REFUSES (rc 2) without a binding token


if __name__ == "__main__":
    sys.exit(main())
