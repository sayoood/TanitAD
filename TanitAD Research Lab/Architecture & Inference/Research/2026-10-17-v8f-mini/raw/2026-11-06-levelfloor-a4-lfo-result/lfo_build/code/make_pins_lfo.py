"""make_pins_lfo.py -- PINS.txt of the LFO build: sha256 of the LF tool (the patch's source) and the LFO tool (with a regeneration check: apply_lfo_patch.py on the pinned LF tool must give the same bytes), of level_floor.py (the LF
operator, unchanged: the code copy of lf_build and the tree copy), of the build's code / tests, of the tree's model / data / trainer files, the A4 card, the reader that defines the oracle (lf_read.py) and the four checkpoints (from their
SHA256SUMS.txt lines). The key lines (key=value) are what the read package's chain reads: patched_tool, level_floor.
usage: python -I make_pins_lfo.py"""
import hashlib
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
TREE = "F:/TanitAD-scratch/motion_c/lfo_read/tree"
ART = "E:/Projects/TanitAD-artifacts"
MC = f"{ART}/dagland/motion_c"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
LF_TOOL = f"{MC}/lf_build/tool/v8f_t1_eval_ri_lf.py"
LF_SHA = "6166178a604edc5343faa4fc6f139ec83dc65d00b91155a77b3f33dda65f4d70"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert sha(LF_TOOL) == LF_SHA, "the LF tool is not the pinned 6166178a"
    regen = "F:/TanitAD-scratch/motion_c/lfo_build/regen_tool_lfo.py"
    subprocess.run([PY, "-I", f"{PK}/code/apply_lfo_patch.py", LF_TOOL, regen], check=True, capture_output=True)
    lines = ["# PINS of the LFO build (sha256, full)", ""]
    pins = [("tool LF (the patch's source; lf_build/tool)", LF_TOOL), ("tool LFO (lfo_build/tool)", f"{PK}/tool/v8f_t1_eval_ri_lfo.py"), ("tool LFO (the read tree's stack/scripts/v8f_t1_eval_ri.py)", f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"),
            ("tool LFO regenerated just now by apply_lfo_patch.py", regen), ("level_floor.py (lf_build/code; unchanged by the LFO patch)", f"{MC}/lf_build/code/level_floor.py"),
            ("level_floor.py (the read tree's stack/scripts/)", f"{TREE}/stack/scripts/level_floor.py"), ("apply_lfo_patch.py", f"{PK}/code/apply_lfo_patch.py"), ("tests/test_lfo.py", f"{PK}/tests/test_lfo.py"),
            ("run_mutations_lfo.py", f"{PK}/code/run_mutations_lfo.py"), ("ka_lfo.py", f"{PK}/code/ka_lfo.py"), ("smoke_lfo.py", f"{PK}/code/smoke_lfo.py"), ("make_pins_lfo.py", f"{PK}/code/make_pins_lfo.py"),
            ("raw/KA_LFO.json (KA-O1, KA-O2)", f"{PK}/raw/KA_LFO.json"), ("raw/SMOKE_LFO.json", f"{PK}/raw/SMOKE_LFO.json"), ("logs/MUTATIONS_LFO.txt", f"{PK}/logs/MUTATIONS_LFO.txt"),
            ("tree model   stack/tanitad/models/v8f_mini.py", f"{TREE}/stack/tanitad/models/v8f_mini.py"), ("tree data    stack/tanitad/data/v8f_mini_data.py", f"{TREE}/stack/tanitad/data/v8f_mini_data.py"),
            ("tree trainer stack/scripts/v8f_mini_train.py", f"{TREE}/stack/scripts/v8f_mini_train.py"), ("tree stack/tanitad/data/v8f_recover.py", f"{TREE}/stack/tanitad/data/v8f_recover.py"),
            ("tree stack/tanitad/refs/feasible_decode.py", f"{TREE}/stack/tanitad/refs/feasible_decode.py"), ("tree taniteval/taniteval/floors_kin.py", f"{TREE}/taniteval/taniteval/floors_kin.py"),
            ("card PREREG_LEVELFLOOR_A4.md", f"{MC}/a4_prereg/PREREG_LEVELFLOOR_A4.md"), ("lf_read.py (defines logged_speed, the oracle's definition)", f"{MC}/lf_read/code/lf_read.py")]
    for s in (0, 1):
        for arm in ("C0", "LLR"):
            pins.append((f"checkpoint M2b_s{s}_recft_{arm}/ckpt_last.pt (the SHA256SUMS.txt line)", None))
    for name, p in pins:
        if p is None:
            s, arm = int(name.split("_s")[1][0]), name.split("_recft_")[1].split("/")[0]
            d = f"{ART}/v8f-recover/M2b_s{s}_recft_{arm}"
            line = [l.split()[0] for l in open(f"{d}/SHA256SUMS.txt") if l.strip().endswith("ckpt_last.pt")][0]
            lines.append(f"{line}  {name}")
        else:
            lines.append(f"{sha(p) if os.path.isfile(p) else 'MISSING'}  {name}")
    same = sha(regen) == sha(f"{PK}/tool/v8f_t1_eval_ri_lfo.py") == sha(f"{TREE}/stack/scripts/v8f_t1_eval_ri.py")
    op_same = sha(f"{MC}/lf_build/code/level_floor.py") == sha(f"{TREE}/stack/scripts/level_floor.py")
    lines += ["", f"regeneration check: apply_lfo_patch.py on the pinned LF tool reproduces the LFO tool byte for byte: {'YES' if same else 'NO'}",
              f"level_floor.py lf_build copy == tree copy: {'YES' if op_same else 'NO'}"]
    keys = [f"patched_tool={sha(f'{PK}/tool/v8f_t1_eval_ri_lfo.py')}", f"patched_tool_path={TREE}/stack/scripts/v8f_t1_eval_ri.py", f"lf_tool={sha(LF_TOOL)}",
            f"level_floor={sha(f'{MC}/lf_build/code/level_floor.py')}", f"level_floor_path={TREE}/stack/scripts/level_floor.py", f"tree={TREE}",
            f"model={sha(f'{TREE}/stack/tanitad/models/v8f_mini.py')}", f"data={sha(f'{TREE}/stack/tanitad/data/v8f_mini_data.py')}", f"trainer={sha(f'{TREE}/stack/scripts/v8f_mini_train.py')}",
            f"a4_card={sha(f'{MC}/a4_prereg/PREREG_LEVELFLOOR_A4.md')}", f"regeneration_check={'YES' if same and op_same else 'NO'}", ""]
    lines[2:2] = ["# keys (key=value; the read package reads patched_tool and level_floor)"] + keys
    open(f"{PK}/PINS.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if same and op_same else 2)


if __name__ == "__main__":
    main()
