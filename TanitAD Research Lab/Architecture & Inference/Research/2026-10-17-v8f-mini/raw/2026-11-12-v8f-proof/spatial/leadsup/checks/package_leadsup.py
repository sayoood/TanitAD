"""Builds LEADSUP.patch (tree_base -> tree_patch), patched/ (the changed and new files), PINS.txt and ARMS_LEADSUP.txt into DEST. LF, UTF-8. Reads the trees only. Usage: package_leadsup.py"""
import difflib
import hashlib
import os

S = "C:/Users/Admin/v8proof_c/leadsup"
DEST = "F:/TanitAD-scratch/v8proof/spatial/leadsup"
CHANGED = ["stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py"]
NEW = ["stack/tests/test_v8f_lead_sup.py", "stack/scripts/lead_dump.py"]
UNCHANGED_PINNED = ["stack/tanitad/models/hjepa.py", "stack/tanitad/models/v8f_beta.py", "stack/scripts/v8f_t1_eval.py", "stack/tests/test_v8f_target_levers.py",
                    "stack/tests/test_v8f_cell_predictor.py", "stack/tests/test_v8f_motion_levers.py"]
CHAIN_PINS = ["stack/tanitad/models/v8f_mini.py", "stack/tanitad/models/v8f_beta.py", "stack/tanitad/models/hjepa.py", "stack/scripts/v8f_mini_train.py", "stack/scripts/v8f_t1_eval.py",
              "stack/tanitad/data/v8f_mini_data.py"]
TREE = "/workspace/v8proof/tree_leadsup"


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
with open(f"{DEST}/LEADSUP.patch", "w", encoding="utf-8", newline="\n") as fh:
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
    fh.write("# base = spatial/target/tree_patch (the TARGET result, nine sha256 of spatial/target/PINS.txt verified on the copy). changed = base sha256 -> patched sha256\n")
    for rel in CHANGED:
        fh.write(f"changed  {rel}  {base_pins[rel]} -> {pins[rel]}\n")
    for rel in NEW:
        fh.write(f"new      {rel}  {pins[rel]}\n")
    fh.write("# unchanged files the chain pins (same sha256 in base and patched)\n")
    for rel in UNCHANGED_PINNED:
        assert base_pins[rel] == pins[rel], rel
        fh.write(f"same     {rel}  {pins[rel]}\n")
G = "--wm-state grid --wm-cell-dim 16"
EXP = "levers.wm_state=grid levers.wm_z1_width=640 levers.wm_cell_dim=16 levers.wm_l1_pred=flat levers.wm_l1_pred_width=256 levers.wm_l1_act=acc_yaw levers.wm_l1_futtok_k=0 levers.w_wm_futtok=0"
ARMS = [("LS_s0", 0, "--lead-sup on --w-lead-sup 1.0", "levers.lead_sup=on levers.w_lead_sup=1 levers.lead_hist=5 levers.lead_sup_detach=off"),
        ("LS_s1", 1, "--lead-sup on --w-lead-sup 1.0", "levers.lead_sup=on levers.w_lead_sup=1 levers.lead_hist=5 levers.lead_sup_detach=off"),
        ("LSD_s0", 0, "--lead-sup on --w-lead-sup 1.0 --lead-sup-detach on", "levers.lead_sup=on levers.w_lead_sup=1 levers.lead_hist=5 levers.lead_sup_detach=on"),
        ("LSD_s1", 1, "--lead-sup on --w-lead-sup 1.0 --lead-sup-detach on", "levers.lead_sup=on levers.w_lead_sup=1 levers.lead_hist=5 levers.lead_sup_detach=on")]
with open(f"{DEST}/ARMS_LEADSUP.txt", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("# LEADSUP: arms LS_s0 / LS_s1 = arm G (grid state 640, flat level-1 predictor, card ADD12) plus --lead-sup on --w-lead-sup 1.0 (a supervised lead-vehicle head on the level-1 grid states of the last 5 frames, gradient into the\n")
    fh.write("# state encoder); LSD_s0 / LSD_s1 = the same with --lead-sup-detach on (the head reads state.detach(): the term reaches no parameter outside the head). Baseline = G_s0, G_s1.\n")
    fh.write("# Labels come from the label join (data/inputs/join_train.npz: lead_valid, lead_gap): the gate refuses a join with no lead rows. info.lead_sup_n_labelled is read from the first selfcheck (real data), not predeclared.\n")
    for rel in CHAIN_PINS:
        fh.write(f"PIN|{TREE}|{rel}|{pins.get(rel) or sha(rd(f'{S}/tree_patch/{rel}'))}\n")
    for name, seed, extra, exp in ARMS:
        fh.write(f"ARM|{name}|{TREE}|HIER|{seed}|540:600|/workspace/v8proof/data/cache_pod|{G} {extra}|{EXP} {exp} info.n_clips=4512\n")
print("packaged", {r: pins[r][:8] for r in pins})
