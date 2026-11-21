"""one-shot: fork ../../build/box2/actrsp_check.py -> optb_check.py (names, tag table, argv rule, gate assertions, C5). Re-runnable."""
import re, sys
src = open("F:/TanitAD-scratch/actrsp/build/box2/actrsp_check.py", encoding="utf-8").read()
s = src
def sub(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:80])
    s = s.replace(old, new)
# --- names
s = s.replace("ACTRSP", "OPTB").replace("actrsp", "optb")
s = s.replace("argvs_optb.json", "argvs_optb.json")
# --- header
s = s.replace('"""OPTB launch-gate checker.', '"""OPTB (ACTRSP Option B, AMEND_OPTB sha256 7cb0245c) launch-gate checker.', 1)
# --- frozen identities
s = re.sub(r'ARGVS_SHA256 = "[0-9a-f]{64}"[^\n]*', 'ARGVS_SHA256 = "2f9acec3e28ad5baa1e775e9af0eb4fc13ab55c83a6d2754b29287bc7cd5f817"        # argvs_optb.json, built by make_argvs_optb.py (re-pin if any token moves)', s, 1)
s = re.sub(r'INIT_SHA = \{[^\n]*\}', 'INIT_SHA = {0: "d920d62c6c391a61291b966a6af1783156b4351d25c7408913e4191daa0215cf", 1: "a97e6017cffb13830dacd04e1ea108fa8534bc3ab3007b8f536f01ac0846f881"}   # the R35_s{k} ckpt_last.pt (45,266,310 B each) that every arm of chassis k starts from (MEASURED sha256 of the dev-box bank)\nPROBE_SHA = {0: "edab7dce7b57f9ae21d90e5cc317c914aed432751fde7a7d4e6cf6af0748a6cf", 1: "a6d252aa405f624a7cea4129a7f7d65ba5c047045d88e89b340a8abb935bb495"}\nS_CAL = {0: 0.20076, 1: 0.20323}      # AMEND_OPTB section 1 (5 digits): the BUILT slope is read back to 1e-5\nN_WM_TENSORS, N_PROJ_TENSORS = 145, 4   # AMEND_OPTB C5: 141 of 145 wm.* Adam tensors, wm.proj bit-equal', s, 1)
sub("RUN_STEPS, PILOT_STEPS, LIMIT_H = 19164, 200, 72.0", "RUN_STEPS, PILOT_STEPS, LIMIT_H = 4000, 200, 72.0")
sub('M2B_DECL = DECL_D + r"\runs\v8f_mini_full"', 'M2B_DECL = DECL_D + r"\runs\v8f_mini_full"') if False else None
s = s.replace(r'HOME_DECL, REPO_DECL, M2B_DECL = DECL_D + r"\optb", DECL_D + r"\optb\repo", DECL_D + r"\runs\v8f_mini_full"', r'HOME_DECL, REPO_DECL, INPUTS_DECL = DECL_D + r"\optb", DECL_D + r"\optb\repo", DECL_D + r"\optb\inputs"')
assert "INPUTS_DECL" in s
# --- tag table
i0, i1 = s.index("TABLE = {"), s.index("PILOT_TAG, PILOT_BASE")
rows = [("FTRSP_s0", "rsp", 0, 0, 1), ("FTNULL_s0", "null", 0, 0, 2), ("FTRSP_s2", "rsp", 2, 0, 3), ("FTNULL_s2", "null", 2, 0, 4), ("FTNULL_s4", "null", 4, 0, 5),
        ("FTRSP_s1", "rsp", 1, 1, 6), ("FTNULL_s1", "null", 1, 1, 7), ("FTRSP_s3", "rsp", 3, 1, 8), ("FTNULL_s3", "null", 3, 1, 9), ("FTNULL_s5", "null", 5, 1, 10)]
tbl = "TABLE = {\n" + "".join(f'    "{t}": {{"lever": "{l}", "seed": {sd}, "init": {k}, "order": {o}, "blocked": None}},\n' for t, l, sd, k, o in rows) + "}\n"
s = s[:i0] + tbl + s[i1:]
s = s.replace('PILOT_TAG, PILOT_BASE = "PILOT_RSP_s0", "RSP_s0"', 'PILOT_TAG, PILOT_BASE = "PILOT_FTRSP_s0", "FTRSP_s0"')
s = s.replace('GATE_AT_PILOT = ["RSP_s0", "RSP_s2"]', 'GATE_AT_PILOT = ["FTRSP_s0", "FTRSP_s2", "FTNULL_s0"]')
s = s.replace(r'PILOT_OUT_DECL + r"\RSP_s0"', r'PILOT_OUT_DECL + r"\FTRSP_s0"')
# --- NEEDLES
s = re.sub(r"NEEDLES = \{TRAINER_REL: \([^\n]*\n[^\n]*\n", 'NEEDLES = {TRAINER_REL: (\'"--wm-rsp-w"\', "def bad_wm_rsp", "def wm_rsp_readback", "lever:wm_rsp_w", \'"--wm-rsp-readout"\', \'"--wm-freeze-proj"\', "def bad_wm_freeze", "def load_wm_only_init", \'"wm/rsp/L1"\', "--wm-only"),\n           MODEL_REL: ("wm_rsp_w", "def built_rsp", "RSP_DELTA", "def load_rsp_probe"), HJEPA_REL: ("def response_loss", "def _response_loss_fixed_k", "def built_rsp", "RSP_K, RSP_STRIDE, RSP_CTX, RSP_S_MIN", "rsp_w: float = 0.0", "rsp_readout"), WINDOWS_REL: ("class V8FMiniWMWindows",)}\n', s, 1, flags=re.M)
# --- argv rule
i0, i1 = s.index("def argv_rsp_problems"), s.index("def c3_problems")
new_rule = '''RSP_FLAGS = ("--wm-rsp-w", "--wm-rsp-readout", "--wm-rsp-probe", "--wm-rsp-probe-sha256")
FT_FLAGS = (("--wm-init-from-ckpt-wm", "on"), ("--wm-freeze-proj", "on"))


def argv_rsp_problems(tag, toks):
    """AMEND_OPTB section 1. ALL ten arms name `--wm-init-from-ckpt-wm on --wm-freeze-proj on` exactly once (a flag whose default disables it must be named). An RSP arm names the four RSP flags exactly once with the values `1.0`, `fixed_k`, the chassis' probe
    file under the inputs dir and its sha256; a NULL arm names none of the four. Refused before anything is built."""
    spec = TABLE[base_tag(tag)]
    p = []
    for f, v in FT_FLAGS:
        idx = [i for i, t in enumerate(toks) if t == f]
        if len(idx) != 1 or idx[0] + 1 >= len(toks) or toks[idx[0] + 1] != v:
            p.append(f"{tag}: {f} must be named exactly once as `{f} {v}` (named {len(idx)} times)")
    idx = {f: [i for i, t in enumerate(toks) if t == f] for f in RSP_FLAGS}
    if spec["lever"] == "null":
        return p + [f"{tag}: a NULL arm names {f} (the four response flags must be absent)" for f, v in idx.items() if v]
    val = lambda f: toks[idx[f][0] + 1] if len(idx[f]) == 1 and idx[f][0] + 1 < len(toks) else None
    want = {"--wm-rsp-w": "1.0", "--wm-rsp-readout": "fixed_k", "--wm-rsp-probe-sha256": PROBE_SHA[spec["init"]]}
    for f in RSP_FLAGS:
        if len(idx[f]) != 1:
            p.append(f"{tag}: an RSP arm must name {f} exactly once (named {len(idx[f])} times)")
    for f, w in want.items():
        if len(idx[f]) == 1 and val(f) != w:
            p.append(f"{tag}: {f} is {val(f)!r}, wanted {w!r}")
    pr = val("--wm-rsp-probe")
    if len(idx["--wm-rsp-probe"]) == 1 and ntpath.normpath(str(pr)).lower() != ntpath.normpath(INPUTS_DECL + rf"\readout_K10_R35_s{spec['init']}.pt").lower():
        p.append(f"{tag}: --wm-rsp-probe is {pr!r}, wanted the chassis-{spec['init']} probe under {INPUTS_DECL}")
    return p


'''
s = s[:i0] + new_rule + s[i1:]
open("optb_check.py", "w", encoding="utf-8", newline="\n").write(s)
print("forked", len(s))
