"""AMEND_HWMPLAN_5 (RA-L1) stage table: the ONE place that spells chassis, inputs, base folders and the argv of every RA arm, so the controls, the launcher and the readout read the same table.
Nothing here plans or scores. Paths:
  AMEND5_ROOT   (default: the amend5 folder)         train/ score/ controls/ readout/ live here; the build's smoke sets it elsewhere so the real folders stay new
  AMEND4_ROOT   (default: the amend4 folder)         P / P23 base folders (AMEND_4's banked plans, with l2_prim_raw)
  LON-PLAN root                                      A0 base folders
The base solver is read from BASE_SOLVER_AMEND5.md (written by the MM from AMEND_HWMPLAN_4's result): machine lines `BASE_SOLVER: A0|P|P23` and `BRANCH: B1|B2|B3`. SCORE is refused without it."""
import hashlib, json, os, re, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
L0 = "F:/TanitAD-scratch/actrsp/lonplan"
ROOT = os.environ.get("AMEND5_ROOT", L0 + "/amend5")
A4ROOT = os.environ.get("AMEND4_ROOT", L0 + "/amend4")
CODE_ROOT = os.environ.get("LONPLAN_CODE_ROOT", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory")
LONPLAN_BUILD = os.environ.get("AMEND5_LONPLAN_BUILD", L0 + "/build")
AMEND4_BUILD = os.environ.get("AMEND5_AMEND4_BUILD", L0 + "/amend4/build")
SPEC = L0 + "/amend5/AMEND_HWMPLAN_5.md"
BASE_SOLVER_FILE = os.environ.get("AMEND5_BASE_SOLVER", L0 + "/amend5/BASE_SOLVER_AMEND5.md")
READOUT_DIR = os.environ.get("AMEND5_READOUT_DIR", L0 + "/amend5/readout_in")      # blobs 7fcafdfb / 95a43146 of the mirror, exported with autocrlf off, sha256-equal to section 2.1
READOUT_FALLBACKS = ["E:/Projects/TanitAD/raw/2026-11-16-v8f-actrsp-prereg/amend1/out"]
RA_SHA = {"s0": "edab7dce7b57f9ae21d90e5cc317c914aed432751fde7a7d4e6cf6af0748a6cf", "s1": "a6d252aa405f624a7cea4129a7f7d65ba5c047045d88e89b340a8abb935bb495"}
SPEC_SHA = "de6f74773e110c49fcd37e1d9d851b15b5744fc6c32392b3f3c6a6ff8a124781"
CH = {"s0": dict(tag="R35_s0", win=L0 + "/score/A0_s0/windows.npz", wsha="371c753bcc3f28f1052c0e5e7e4898c0bbbed6140477b3338c8133a3632bdaee",
                 ck="C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/ckpt_last.pt", cksha="d920d62c6c391a61291b966a6af1783156b4351d25c7408913e4191daa0215cf",
                 lat="F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt", latsha="76a2d4f5482bd98670bab40ebbf4b3c85fde8ebf222596947fa16f0c7e82754c"),
      "s1": dict(tag="R35_s1", win=L0 + "/score/A0_s1/windows.npz", wsha="8833e2923c1781d7678f8483775487db476117e3ddd563a444d2d122334061f0",
                 ck="C:/Users/Admin/t2v8f_pulls/ladder_R35_s1/ckpt_last.pt", cksha="a97e6017cffb13830dacd04e1ea108fa8534bc3ab3007b8f536f01ac0846f881",
                 lat="F:/TanitAD-scratch/hwm_plan/latents_train_R35_s1.pt", latsha="692eb002")}      # latsha: 8-hex prefix, as the record spells it (checked as a prefix)
BASES = ("A0", "P", "P23")
ARMS = ("RAL1", "RAL2", "MEANL1", "MEANL2", "NEG", "SHUF", "ORACLE", "OFF")
# arm -> (--ra-l1 value, --ra-src, kind). MEAN arms are a different assembly (no world model, no RA): kind 'mean'.
ARM_SPEC = {"RAL1": ("l1", "l1", "ra"), "RAL2": ("l2", "l2", "ra"), "NEG": ("neg", None, "ra"), "SHUF": ("shuf", None, "ra"), "ORACLE": ("oracle", "none", "ra"), "OFF": ("off", "none", "off"),
            "MEANL1": ("off", "none", "mean"), "MEANL2": ("off", "none", "mean")}
MEAN_SRC = {"MEANL1": "l1", "MEANL2": "l2"}
STAGES = ("train", "score")


class NoBaseSolver(Exception):
    """SCORE is refused: BASE_SOLVER_AMEND5.md is missing, unreadable or inconsistent."""


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_base_solver(path=None):
    """-> dict(base, branch). The file names the branch and the base on machine lines `BASE_SOLVER: <A0|P|P23>` and `BRANCH: <B1|B2|B3>`; B3 <-> A0, B1 / B2 <-> P or P23. Raises NoBaseSolver."""
    p = path or BASE_SOLVER_FILE
    if not os.path.exists(p):
        raise NoBaseSolver(f"{p} does not exist: the MM writes it from AMEND_HWMPLAN_4's result before any RA plan exists on SCORE")
    txt = open(p, encoding="utf-8", errors="replace").read()
    mb = re.findall(r"^[\s>*\-`]*BASE_SOLVER\s*[:=]\s*[*`]*\s*(A0|P23|P)\b", txt, flags=re.M)
    mr = re.findall(r"^[\s>*\-`]*BRANCH\s*[:=]\s*[*`]*\s*(B[123])\b", txt, flags=re.M)
    if len(set(mb)) != 1 or len(set(mr)) != 1:
        raise NoBaseSolver(f"{p} must carry exactly one `BASE_SOLVER: <A0|P|P23>` and one `BRANCH: <B1|B2|B3>` line (found base {mb}, branch {mr})")
    base, br = mb[0], mr[0]
    if (br == "B3") != (base == "A0"):
        raise NoBaseSolver(f"{p}: branch {br} and base {base} disagree (B3 <-> A0; B1 / B2 <-> P or P23)")
    return dict(base=base, branch=br, path=p, sha256=sha256_file(p))


def base_dir(stage, base, k):
    if base == "A0":
        return f"{L0}/score/A0_{k}" if stage == "score" else f"{L0}/sel_{k}/lam_0"
    return f"{A4ROOT}/{stage}/{base}_{k}"


def base_plan(stage, base, k, seed=0):
    return f"{base_dir(stage, base, k)}/plans_hwm_oracle_s{seed}.npz"


def windows_of(stage, k):
    return CH[k]["win"] if stage == "score" else f"{L0}/sel_{k}/selection_windows.npz"


def out_dir(stage, arm, base, k, root=None):
    return f"{root or ROOT}/{stage}/{arm}_{base}_{k}"


def seeds_of(stage, arm, committed=True):
    """SCORE: lever arms and MEAN and ORACLE seeds 0 1; OFF, NEG, SHUF seed 0. TRAIN: seed 0 only (the base's TRAIN plans have one seed)."""
    if stage == "train":
        return [0]
    return [0, 1] if arm in ("RAL1", "RAL2", "MEANL1", "MEANL2", "ORACLE") else [0]


def readout_path(k):
    for d in [READOUT_DIR] + READOUT_FALLBACKS:
        p = f"{d}/readout_K10_R35_{k}.pt"
        if os.path.exists(p):
            return p
    return f"{READOUT_DIR}/readout_K10_R35_{k}.pt"


def build(stage, arm, base, k, out=None, src_for_nl=None):
    """-> dict(argv, declared, ...). argv names --ra-l1 (default off disables the pass) and --ra-src for every arm. NEG / SHUF take the committed arm's source (src_for_nl: 'l1' or 'l2')."""
    if stage not in STAGES or arm not in ARMS or base not in BASES or k not in CH:
        raise ValueError((stage, arm, base, k))
    mode, src, kind = ARM_SPEC[arm]
    if arm in ("NEG", "SHUF"):
        if src_for_nl not in ("l1", "l2"):
            raise ValueError(f"{arm} needs the committed arm's source (l1 or l2)")
        src = src_for_nl
    msrc = MEAN_SRC.get(arm, "none")
    if (src == "l2" or msrc == "l2") and base == "A0":
        raise ValueError("RA-L2 is not computable on A0: A0 has no level-2 primitives for knots 1-10 (D-HWM-UPPER-OFFIMAGE; MM ruling M3)")
    o = out or out_dir(stage, arm, base, k)
    argv = ["--stage", stage, "--arm", arm, "--base", base, "--chassis", k, "--ra-l1", mode, "--ra-src", src if src else "none", "--mean-src", msrc, "--seeds", *map(str, seeds_of(stage, arm)), "--out", o]
    declared = dict(arm=arm, mode=mode, src=(src if src else "none"), kind=kind, mean_src=msrc, mirror=(mode == "neg"), perm=(mode == "shuf"), readout_sha256=RA_SHA[k], base=base, stage=stage, chassis=k)
    return dict(argv=argv, declared=declared, windows_src=windows_of(stage, k), ckpt_sha=CH[k]["cksha"], out=o)


# The frozen card spells hwm_plan.py as 5a072272...; the file and LON-PLAN's score/A0_s0/SOLVER.json read 5a0722719f7c... (the card's 8th hex digit is a slip). C-IN pins the FULL 64-hex values.
CODE_SHA_FULL = {"hwm_plan.py": "5a0722719f7ce06c94c91213db6ef542b2fe16d346622ba506ce8808bd6d714b", "run_hwm.py": "935b34226158ae50be6edc7d282aa8a8e90b855728e5624a800c6082e26e095e",
                 "hwm_eval.py": "7939207520159b5be82c7c7430239819842dfa1a3f320789a499d5a114607e8d", "hwm_data.py": "a2d3922e9cb96304a08eba6e7826ab6bf4a5ae64a2cc2724a4a140d0f959fad7"}
