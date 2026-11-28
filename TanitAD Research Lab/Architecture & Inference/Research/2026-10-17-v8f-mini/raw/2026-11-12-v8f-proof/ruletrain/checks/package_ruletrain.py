"""Builds RULETRAIN.patch (tree_base = LEADSUP tree -> tree_patch), patched/ (the changed and new files), PINS.txt and ARMS_RULETRAIN.txt into DEST. LF, UTF-8. Reads the trees only. Usage: package_ruletrain.py"""
import difflib
import hashlib
import os

S = "C:/Users/Admin/v8proof_c/ruletrain"
DEST = "F:/TanitAD-scratch/v8proof/ruletrain"
CHANGED = ["stack/scripts/v8f_mini_train.py"]
NEW = ["stack/tanitad/data/rule_relabel.py", "stack/tanitad/data/dist_rule.py", "stack/tests/test_v8f_rule_relabel.py"]
UNCHANGED_PINNED = ["stack/tanitad/models/v8f_mini.py", "stack/tanitad/models/hjepa.py", "stack/tanitad/models/v8f_beta.py", "stack/scripts/v8f_t1_eval.py", "stack/tanitad/data/v8f_mini_data.py",
                    "stack/tests/test_v8f_lead_sup.py", "stack/scripts/lead_dump.py", "stack/tests/test_v8f_target_levers.py", "stack/tests/test_v8f_cell_predictor.py", "stack/tests/test_v8f_motion_levers.py"]
CHAIN_PINS = ["stack/tanitad/models/v8f_mini.py", "stack/tanitad/models/v8f_beta.py", "stack/tanitad/models/hjepa.py", "stack/scripts/v8f_mini_train.py", "stack/scripts/v8f_t1_eval.py",
              "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/data/rule_relabel.py", "stack/tanitad/data/dist_rule.py"]
TREE = "/workspace/v8proof/tree_ruletrain"


def rd(p):
    with open(p, "rb") as fh:
        return fh.read()


sha = lambda b: hashlib.sha256(b).hexdigest()
os.makedirs(DEST, exist_ok=True)
patch = []
for rel in CHANGED + NEW:
    new = rd(f"{S}/tree_patch/{rel}")
    assert b"\r" not in new, rel
    if os.path.exists(f"{S}/tree_base/{rel}"):
        old = rd(f"{S}/tree_base/{rel}")
        assert b"\r" not in old
        d = difflib.unified_diff(old.decode("utf-8").splitlines(keepends=True), new.decode("utf-8").splitlines(keepends=True), fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3)
        hdr = f"diff --git a/{rel} b/{rel}\n"
    else:
        d = difflib.unified_diff([], new.decode("utf-8").splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/{rel}", n=3)
        hdr = f"diff --git a/{rel} b/{rel}\nnew file mode 100644\n"
    body = "".join(d)
    if body and not body.endswith("\n"):
        body += "\n\\ No newline at end of file\n"
    patch.append(hdr + body)
with open(f"{DEST}/RULETRAIN.patch", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("".join(patch))
pins, base_pins = {}, {}
for rel in CHANGED + NEW + UNCHANGED_PINNED:
    pins[rel] = sha(rd(f"{S}/tree_patch/{rel}"))
    if rel in CHANGED + NEW:
        dst = f"{DEST}/patched/{rel[len('stack/'):]}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(rd(f"{S}/tree_patch/{rel}"))
    if os.path.exists(f"{S}/tree_base/{rel}"):
        base_pins[rel] = sha(rd(f"{S}/tree_base/{rel}"))
with open(f"{DEST}/PINS.txt", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("# PINS: sha256 of the files of tree_patch (paths relative to the repo root; patched/ holds the changed and new ones without the leading stack/).\n")
    fh.write("# base = spatial/leadsup/tree_patch (the LEADSUP result, its sha256 of spatial/leadsup/PINS.txt verified on the copy). changed = base sha256 -> patched sha256\n")
    for rel in CHANGED:
        fh.write(f"changed  {rel}  {base_pins[rel]} -> {pins[rel]}\n")
    for rel in NEW:
        fh.write(f"new      {rel}  {pins[rel]}\n")
    fh.write("# dist_rule.py is byte-identical to F:/TanitAD-scratch/v8proof/distrule/dist_rule.py (card ADD26)\n")
    fh.write("# unchanged files the chain pins (same sha256 in base and patched)\n")
    for rel in UNCHANGED_PINNED:
        assert base_pins[rel] == pins[rel], rel
        fh.write(f"same     {rel}  {pins[rel]}\n")
assert pins["stack/tanitad/data/dist_rule.py"] == sha(rd("F:/TanitAD-scratch/v8proof/distrule/dist_rule.py"))
G = "--wm-state grid --wm-cell-dim 16"
EXP = "levers.wm_state=grid levers.wm_z1_width=640 levers.wm_cell_dim=16 levers.wm_l1_pred=flat levers.wm_l1_pred_width=256 levers.wm_l1_act=acc_yaw levers.wm_l1_futtok_k=0 levers.w_wm_futtok=0 levers.lead_sup=off"
FT = "--steps 2000 --lr 1e-4 --warmup 100"
BASE = {0: "/workspace/v8proof/runs/spatial/G_s0/ckpt_last.pt", 1: "/workspace/v8proof/runs/spatial/G_s1/ckpt_last.pt"}
# name, base seed (the checkpoint), fine-tune seed (--seed), rule flag
ARMS = [("GFT_s0", 0, 0, "off"), ("GFT_s1", 1, 1, "off"), ("GFTb_s0", 0, 100, "off"), ("GFTb_s1", 1, 101, "off"), ("GRT_s0", 0, 0, "on"), ("GRT_s1", 1, 1, "on")]
with open(f"{DEST}/ARMS_RULETRAIN.txt", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("# RULETRAIN (card PREREG_V8F_PROOF_ADD32): six fine-tunes of 2,000 steps from the banked arm-G checkpoints (--init-ckpt, strict load, fresh optimiser), batch 64, peak lr 1e-4, warm-up 100, cosine to a tenth.\n")
    fh.write("# GFT = control (targets unchanged, fine-tune seed = base seed); GFTb = replicate of the control (fine-tune seed = base seed + 100); GRT = --rule-relabel on (targets relabelled by the label-fed distance rule). Baselines: G_s0, G_s1.\n")
    fh.write("# The ARM field's seed column is the fine-tune --seed. Labels come from the label join (--join-train): the gate refuses a launch whose probe batch has no label lead (count 0). info.rule_relabel_n is read from the first real-data selfcheck, not predeclared.\n")
    fh.write("# A flag-off arm has no rule_relabel key in the selfcheck; its expectation line equals arm G's. The G ckpt has no lead head (levers.lead_sup=off).\n")
    for rel in CHAIN_PINS:
        fh.write(f"PIN|{TREE}|{rel}|{pins.get(rel) or sha(rd(f'{S}/tree_patch/{rel}'))}\n")
    for name, bs, seed, rr in ARMS:
        extra = f"{G} --init-ckpt {BASE[bs]} {FT}" + (" --rule-relabel on" if rr == "on" else "")
        exp = EXP + (" levers.rule_relabel=on" if rr == "on" else "") + " info.n_clips=4512"
        fh.write(f"ARM|{name}|{TREE}|HIER|{seed}|540:600|/workspace/v8proof/data/cache_pod|{extra}|{exp}\n")
print("packaged", {r: pins[r][:8] for r in pins})
