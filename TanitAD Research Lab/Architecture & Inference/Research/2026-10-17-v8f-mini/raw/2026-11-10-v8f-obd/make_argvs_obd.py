r"""Builds the OBD retrain argvs: each arm = DAG-OACC's argv of its init (v8f-dagoacc/M2b_s{k}_dagoacc/argv.json) VERBATIM, with ONLY
  --out D:\tanitad\runs\v8f_obd\<tag>   and the per-arm additions of PREREG_V8F_OBD_DRAFT (NULL nothing; CTL --detach-upper; BAND --detach-upper --w-traj 0 --w-op .. --w-tac-traj ..;
  *_rep = the s0 arm at --seed 2, replaced IN PLACE). Checks every argv parses with the trainer at the commit of record (export dir OBD_TREE) and that
  each diff against the DAG-OACC argv is exactly the declared one. Writes argvs_obd.json, argvs_obd.txt, argvs_obd.meta.json. Nothing is launched.
Usage: python make_argvs_obd.py   (env OBD_TREE = export of commit 202849da, default F:/TanitAD-scratch/obd_build/tree)"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "E:/Projects/TanitAD-artifacts/v8f-dagoacc"
TREE = os.environ.get("OBD_TREE", "F:/TanitAD-scratch/obd_build/tree")
COMMIT = "202849da013b5d74c0b0086dd9865fcda6836018"
TRAINER_SHA256_AT_COMMIT = "1ddf491d1aaede1cce3e1fecd73206632cdf6805c1b5f129a1d9ff85dde276f0"      # sha256 of `git show 202849da:stack/scripts/v8f_mini_train.py` (MEASURED 2026-11-10)
OUT_ROOT = r"D:\tanitad\runs\v8f_obd" + "\\"
BANDS = ["--w-traj", "0", "--w-op", "0.3333333", "--w-tac-traj", "0.6666667"]
DET = ["--detach-upper"]
# arm -> (init seed k, extra flags appended, training seed override or None, what)
ARMS = {
    "OBD-NULL_s0": (0, [], None, "DAG-OACC s0's argv, only --out (code-parity control at the commit of record)"),
    "OBD-NULL_rep": (0, [], 2, "OBD-NULL s0's argv at --seed 2 (the run-to-run floor of the base recipe at this commit)"),
    "OBD-CTL_s0": (0, DET, None, "DAG-OACC s0 + --detach-upper"),
    "OBD-CTL_s1": (1, DET, None, "DAG-OACC s1 + --detach-upper"),
    "OBD-BAND_s0": (0, [*DET, *BANDS], None, "DAG-OACC s0 + --detach-upper + the band loss"),
    "OBD-BAND_s1": (1, [*DET, *BANDS], None, "DAG-OACC s1 + --detach-upper + the band loss"),
    "OBD-BAND_rep": (0, [*DET, *BANDS], 2, "OBD-BAND s0's flags at --seed 2 (the run-to-run floor of the band arm, H-ESTIM-SEED-1)"),
    "OBD-NULL_s1": (1, [], None, "CONDITIONAL (only if NULL parity fails): DAG-OACC s1's argv, only --out"),
}


def load(k):
    return json.load(open(f"{SRC}/M2b_s{k}_dagoacc/argv.json", encoding="utf-8"))


def main():
    sys.path[:0] = [TREE + "/stack", TREE + "/stack/scripts"]
    import v8f_mini_train as TR  # noqa: E402
    trainer_sha = hashlib.sha256(open(TREE + "/stack/scripts/v8f_mini_train.py", "rb").read()).hexdigest()
    assert trainer_sha == TRAINER_SHA256_AT_COMMIT, f"the export at OBD_TREE is not the unmodified trainer of {COMMIT[:8]} (sha256 {trainer_sha})"
    base = {k: load(k) for k in (0, 1)}
    for k in (0, 1):
        assert TR.argv_sha(base[k]["argv"]) == base[k]["sha"], f"DAG-OACC s{k}: argv_sha at the commit {TR.argv_sha(base[k]['argv'])} != recorded {base[k]['sha']}"
    out, lines = {}, []
    for tag, (k, extra, seed, what) in ARMS.items():
        av = list(base[k]["argv"])
        i = av.index("--out")
        av[i + 1] = OUT_ROOT + tag
        if seed is not None:
            j = av.index("--seed")
            av[j + 1] = str(seed)
        av += extra
        # the diff against the base argv is exactly: --out value, [--seed value], the appended flags
        b = base[k]["argv"]
        diffs = [n for n in range(len(b)) if av[n] != b[n]]
        want = [i + 1] + ([av.index("--seed") + 1] if seed is not None else [])
        assert diffs == sorted(want), (tag, diffs, want)
        assert av[len(b):] == extra, tag
        TR.build_parser().parse_args(av)
        out[tag] = {"what": what, "init_seed": k, "argv": av}
        lines.append(f"{tag}: " + " ".join(av))
    json.dump(out, open(HERE + "/argvs_obd.json", "w", encoding="utf-8", newline="\n"), indent=1)
    open(HERE + "/argvs_obd.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    meta = {"commit": COMMIT, "trainer_sha256_at_commit": trainer_sha, "dagoacc_argv_sha": {str(k): base[k]["sha"] for k in base},
            "argvs_json_sha256": hashlib.sha256(open(HERE + "/argvs_obd.json", "rb").read()).hexdigest()}
    json.dump(meta, open(HERE + "/argvs_obd.meta.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(len(out), "argvs parse; trainer sha256", trainer_sha, "; argvs sha256", meta["argvs_json_sha256"])


if __name__ == "__main__":
    main()
