"""rh_common.py -- shared constants and argv rendering of the ROLLHIST read package (card PREREG_V8F_ROLLHIST sha256 781eea87, Amendment 1 sha256 87928487; code of record 43319227).
Dev box, CPU, zero training.  Tier of every rollout: T1-kin/hb0.2 = `v8f_t1_eval_hb.py --kappa-max 0.2 --inner v8f_t1_eval_reader_rule_rzg <argv>` (the executor of record).
A rollout of ARM under FORM lands in  <W>/t1/I10<KIND>_<FORM>_s<k>/  with  dump_<tag>/ep*.npz  and  ticks/ticks_<name>_m10_causal_heads.npz  (the folder layout of the pod queues,
so the frozen readers headway_read_r2.py / q3_read_r3.py / strat_roll_read.py find them unchanged).  KIND = RH | B0 | UEA (stage-A reference); k = model seed."""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PY = os.environ.get("RH_PY", "C:/Users/Admin/venvs/tanitad/Scripts/python.exe")
TREE = os.environ.get("RH_TREE", HERE + "/tree")
CODE = HERE + "/code"
OUT = os.environ.get("RH_OUT", HERE + "/out")
W = OUT                                                      # t1 root = OUT/t1
CK = os.environ.get("RH_CKPTS", HERE + "/ckpts")
ART = "E:/Projects/TanitAD-artifacts"
SF = ART + "/sfresh_inputs"
SR = "F:/TanitAD-scratch/strat1/read"
INPUTS = {"--tracks": SF + "/inputs/native_tracks.pt", "--eval-cache": SF + "/cache",
          "--eval-grid": SR + "/inputs/eval_grid_windows.json", "--eval-join": SR + "/inputs/join_eval141.npz"}
WRAPPER = "v8f_t1_eval_reader_rule_rzgi"                     # adapter over the unedited rzg (the card's `--inner v8f_t1_eval_reader_rule_rzg` cannot compose with hb as written; see the adapter header)
STUDY = "v8f_t1_eval_reader_rule_rzgfh"                      # the ADD67/ADD68 study wrapper (sha256 373ffe72f2655c35...), used only by control c-EXEC
KAPPA = "0.2"
# form -> extra argv of the readout wrapper (names as card ADD68; suffix Hb20 = heading bound 0.2)
FORMS = {
    "R0Hb20": ["--readout", "pinned"],
    "R0C10Hb20": ["--readout", "causal"],
    "R0G50Hb20": ["--readout", "causal_g", "--acc-gain", "0.5"],
    "R0G0Hb20": ["--readout", "causal_g", "--acc-gain", "0"],
    "R0KHb20": ["--readout", "causal_g", "--acc-gain", "0", "--path-exec"],
    "R0C10XHb20": ["--readout", "causal_g", "--acc-gain", "1", "--path-exec"],
}
FORM_ORDER = tuple(FORMS)
# ROLLHIST P03 (card 3.4 PARTIAL row, Amendment 7 DRAFT): the same read for the next arm RHp3 (share p = 0.3 on the SAME bank).  RH_AKIND selects the TREATED arm of every pair: RH (default: the PARTIAL read, unchanged
# byte for byte) or RHp3.  With RHp3 the comparator is still the EXISTING B0, the RH dumps are reused (the beside pairs RHp3 minus RH), and only RHp3 is pulled, rolled (ROLL_KINDS) and T0-evaluated.
A_KIND = os.environ.get("RH_AKIND", "RH")
assert A_KIND in ("RH", "RHp3"), A_KIND
KINDS = ("RH", "B0") if A_KIND == "RH" else ("RHp3", "B0")
ROLL_KINDS = ("RH", "B0") if A_KIND == "RH" else ("RHp3",)          # the kinds the read chain pulls and rolls (B0 and RH dumps are the finished read's)
SEEDS = (0, 1)
CKPT_SHA_UEA = {0: "2fd4b747532f55b24523c578948cb713fb80fa0e698586c329dd23ba6e6df914", 1: "f13df8ce026eed31e0d9d1df72de3e7b7977fefa6da3def2b411b7d74dbd4ebc"}   # UE_s0.A full; UE_s1.A prefix (card FACTS A5)
POD_CKPT = "/root/rollhist/runs/{arm}/train/ckpt_last.pt"   # arm = RH_s0 | RH_s1 | B0_s0 | B0_s1 (pod path; the host is read by the pod helper from its own file, never written here)


def sha256_file(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def run_name(kind, form):
    return f"{kind}_{form}"                                  # folder I10<run_name>_s<k>


def folder(kind, form, k, w=None):
    return f"{w or W}/t1/I10{run_name(kind, form)}_s{k}"


def tag(kind, form, k):
    return f"I10{run_name(kind, form)}_s{k}_m10_causal_heads"


def ckpt_of(kind, k):
    return f"{CK}/UE_s{k}.A/ckpt_last.pt" if kind == "UEA" else f"{CK}/{kind}_s{k}/ckpt_last.pt"


def env(tree=None):
    t = tree or TREE
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
             DISTRULE_DIR=CODE + "/distrule", RH_TREE=t, TMP=HERE + "/tmp", TEMP=HERE + "/tmp")
    e["PYTHONPATH"] = ";".join(os.path.normpath(p) for p in (CODE + "/integ10k", t + "/stack", t + "/taniteval"))
    return e


def common_tail(ckpt, tg, out, nwin=0):
    a = ["run", "--ckpt", ckpt, "--tag", tg]
    for k, v in INPUTS.items():
        a += [k, v]
    a += ["--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", out + "/ticks", "--dump-heads", "--device", "cpu"]
    return a + (["--max-windows", str(nwin)] if nwin else [])


def argv_record(kind, form, k, ckpt=None, out=None, tg=None, nwin=0):
    """the tier-of-record argv: v8f_t1_eval_hb.py --kappa-max 0.2 --inner <wrapper> --hb-log ... <wrapper argv>"""
    out = out or folder(kind, form, k)
    tg = tg or tag(kind, form, k)
    name = f"I10{run_name(kind, form)}_s{k}"
    return ([PY, TREE + "/stack/scripts/v8f_t1_eval_hb.py", "--kappa-max", KAPPA, "--inner", WRAPPER, "--hb-log", f"{out}/ticks/hb_{name}.json",
             "--tree", TREE, "--form", "R0"] + FORMS[form] + ["--readout-log", f"{out}/ticks/readout_log_{name}.json"]
            + common_tail(ckpt or ckpt_of(kind, k), tg, out, nwin))


def argv_study(kind, form, k, ckpt=None, out=None, tg=None, nwin=0):
    """the ADD67/ADD68 study form (control c-EXEC only): the wrapper rzgfh carries the bound itself"""
    out = out or folder(kind, form, k)
    tg = tg or tag(kind, form, k)
    name = f"I10{run_name(kind, form)}_s{k}"
    return ([PY, CODE + f"/integ10k/{STUDY}.py", "--tree", TREE, "--form", "R0"] + FORMS[form] + ["--kappa-max", KAPPA, "--readout-log", f"{out}/ticks/readout_log_{name}.json"]
            + common_tail(ckpt or ckpt_of(kind, k), tg, out, nwin))


# ---- pairs of the card (section 3.3).  A run is "<KIND>_<FORM>"; every pair is A minus B, per model seed (RH_sk with B0_sk), pooled by the paired episode-cluster bootstrap B 2,000 seed 0.
def _r(kind, form):
    return run_name(kind, form)


PAIRS = {
    "P1": (_r(A_KIND, "R0C10XHb20"), _r("B0", "R0C10XHb20")),                       # the lever, readout C*
    "P2_ACC_RH": (_r(A_KIND, "R0C10XHb20"), _r(A_KIND, "R0KHb20")),                  # ADD68's ACC pair on RH
    "P2_D1_RH": (_r(A_KIND, "R0G50Hb20"), _r(A_KIND, "R0G0Hb20")),                   # ADD63 D1 on the bounded executor
    "P2_D2_RH": (_r(A_KIND, "R0C10Hb20"), _r(A_KIND, "R0G50Hb20")),                  # ADD63 D2
    "P2_ACC_B0": (_r("B0", "R0C10XHb20"), _r("B0", "R0KHb20")),                  # attribution control: must still read WORSE on at least two of three families
    "P2_D1_B0": (_r("B0", "R0G50Hb20"), _r("B0", "R0G0Hb20")),                   # beside
    "P2_D2_B0": (_r("B0", "R0C10Hb20"), _r("B0", "R0G50Hb20")),                  # beside
    "P4": (_r(A_KIND, "R0C10XHb20"), _r("B0", "R0KHb20")),                         # K unneeded: RH under C* minus B0 under K
}
for _f in FORM_ORDER:
    if _f != "R0C10XHb20":
        PAIRS["BESIDE_RH_minus_B0_" + _f] = (_r(A_KIND, _f), _r("B0", _f))      # reported, no bar  (the name is the PARTIAL read's; its first run is the treated arm A_KIND)
if A_KIND == "RHp3":                                                           # RHp3 minus RH (the dose p = 0.3 against p = 0.1, same bank): beside, no bar, under the convention and the readout C*
    for _f in ("R0Hb20", "R0C10XHb20"):
        PAIRS["BESIDE_RHp3_minus_RH_" + _f] = (_r("RHp3", _f), _r("RH", _f))
ARM_FORMS_ALL = tuple(sorted({x for p in PAIRS.values() for x in p}))
RUN_KINDS = KINDS + tuple(k for k in sorted({r.split("_", 1)[0] for r in ARM_FORMS_ALL}) if k not in KINDS)      # every kind that has a run in a pair (RHp3 reads also RH's two beside runs); = KINDS for the default read
FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
NAMED = {"LONGITUDINAL": ["LONGITUDINAL:speed_mae_mps", "LONGITUDINAL:along_mae_m"], "LATERAL": ["LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps"], "TACTICAL": ["TACTICAL:ade_tac_m"]}
LONG_U = "U"                          # headway reader U (unsafe rate), a named LONGITUDINAL cell (card 3.2, distance-keeping)
ARCS = (30.0, 50.0, 80.0)
F_T0 = 0.0705                         # G-T0: abs(2.0376 - 1.9671), FACTS E3, INHERITED
