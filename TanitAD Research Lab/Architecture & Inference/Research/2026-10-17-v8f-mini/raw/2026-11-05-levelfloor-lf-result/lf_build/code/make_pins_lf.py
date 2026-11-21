"""make_pins_lf.py -- PINS.txt of the LEVEL-FLOOR build: sha256 of the original and patched tool (with a regeneration check: the patch is re-run on the pinned source and must give the same bytes), of level_floor.py
(the code copy and the tree copy), of the build's code / tests, of the tree's model / data / trainer files, the card, and the four checkpoints (from their SHA256SUMS.txt lines, which ka_lf.py re-hashes for C0 s0).
usage: python -I make_pins_lf.py"""
import hashlib
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
TREE = "F:/TanitAD-scratch/motion_c/lf_read/tree"
ART = "E:/Projects/TanitAD-artifacts"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    regen = "F:/TanitAD-scratch/motion_c/lf_read/ka/regen_tool_lf.py"
    subprocess.run([PY, "-I", f"{PK}/code/apply_lf_patch.py", f"{PK}/tool/orig/v8f_t1_eval_ri_0417c838.py", regen], check=True, capture_output=True)
    lines = ["# PINS of the LEVEL-FLOOR build (sha256, full)", ""]
    pins = [("tool ORIGINAL (RI tool 0417c838)", f"{PK}/tool/orig/v8f_t1_eval_ri_0417c838.py"), ("tool PATCHED (lf_build/tool)", f"{PK}/tool/v8f_t1_eval_ri_lf.py"),
            ("tool PATCHED (the read tree's stack/scripts/v8f_t1_eval_ri.py)", f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"), ("tool PATCHED regenerated just now by apply_lf_patch.py", regen),
            ("level_floor.py (lf_build/code)", f"{PK}/code/level_floor.py"), ("level_floor.py (the read tree's stack/scripts/)", f"{TREE}/stack/scripts/level_floor.py"),
            ("apply_lf_patch.py", f"{PK}/code/apply_lf_patch.py"), ("tests/test_level_floor.py", f"{PK}/tests/test_level_floor.py"), ("run_mutations_lf.py", f"{PK}/code/run_mutations_lf.py"),
            ("ka_lf.py", f"{PK}/code/ka_lf.py"), ("smoke_lf.py", f"{PK}/code/smoke_lf.py"), ("make_pins_lf.py", f"{PK}/code/make_pins_lf.py"),
            ("tree model   stack/tanitad/models/v8f_mini.py", f"{TREE}/stack/tanitad/models/v8f_mini.py"), ("tree data    stack/tanitad/data/v8f_mini_data.py", f"{TREE}/stack/tanitad/data/v8f_mini_data.py"),
            ("tree trainer stack/scripts/v8f_mini_train.py", f"{TREE}/stack/scripts/v8f_mini_train.py"), ("tree stack/tanitad/data/v8f_recover.py", f"{TREE}/stack/tanitad/data/v8f_recover.py"),
            ("tree stack/tanitad/refs/feasible_decode.py", f"{TREE}/stack/tanitad/refs/feasible_decode.py"), ("tree taniteval/taniteval/floors_kin.py", f"{TREE}/taniteval/taniteval/floors_kin.py"),
            ("card PREREG_LEVELFLOOR_A3.md", f"{ART}/dagland/motion_c/a3_prereg/PREREG_LEVELFLOOR_A3.md"), ("ka_new.py (loaded by ka_lf.py, pinned there)", f"{ART}/dagland/motion_c/cl_read/code/ka_new.py"),
            ("run_tool_subset.py (cl_read/code)", f"{ART}/dagland/motion_c/cl_read/code/run_tool_subset.py")]
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
    same = sha(regen) == sha(f"{PK}/tool/v8f_t1_eval_ri_lf.py") == sha(f"{TREE}/stack/scripts/v8f_t1_eval_ri.py")
    lines += ["", f"regeneration check: apply_lf_patch.py on the pinned source reproduces the patched tool byte for byte: {'YES' if same else 'NO'}",
              f"level_floor.py code copy == tree copy: {'YES' if sha(f'{PK}/code/level_floor.py') == sha(f'{TREE}/stack/scripts/level_floor.py') else 'NO'}"]
    keys = [f"patched_tool={sha(f'{PK}/tool/v8f_t1_eval_ri_lf.py')}", f"patched_tool_path={TREE}/stack/scripts/v8f_t1_eval_ri.py", f"original_tool={sha(f'{PK}/tool/orig/v8f_t1_eval_ri_0417c838.py')}",
            f"level_floor={sha(f'{PK}/code/level_floor.py')}", f"level_floor_path={TREE}/stack/scripts/level_floor.py", f"tree={TREE}",
            f"model={sha(f'{TREE}/stack/tanitad/models/v8f_mini.py')}", f"data={sha(f'{TREE}/stack/tanitad/data/v8f_mini_data.py')}", f"trainer={sha(f'{TREE}/stack/scripts/v8f_mini_train.py')}",
            f"regeneration_check={'YES' if same else 'NO'}", ""]
    lines[2:2] = ["# keys (key=value; the read package reads patched_tool)"] + keys
    open(f"{PK}/PINS.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if same else 2)


if __name__ == "__main__":
    main()
