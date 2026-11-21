"""Mutation check of amend1_concordance.py: each mutation is applied to a COPY of the code dir and test_amend1.py must go RED.  Planted data only."""
import os, shutil, subprocess, sys
SRC = "C:/Users/Admin/par18/l2stepb/code"
DST = "C:/Users/Admin/par18/l2stepb/test_amend1/mut/code"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MUT = {
 "M1_swap_anchor_candidate": ('"a_logit": np.asarray(so["pred_logit_pick"], np.float64)[ii], "p_logit": np.asarray(out_logit, np.float64)[ii, ss]',
                              '"p_logit": np.asarray(so["pred_logit_pick"], np.float64)[ii], "a_logit": np.asarray(out_logit, np.float64)[ii, ss]'),
 "M2_include_identical_outputs": ("keep = (~identical) & anchor_ok[:, None]", "keep = anchor_ok[:, None] & np.ones_like(identical)"),
 "M3_no_cross_arm_identity": ("ok = bool(np.array_equal(lp[:, others], p2[:, others]))", "ok = True"),
 "M4_wrong_bootstrap_seed": ("B: int = A.N_BOOT, seed: int = A.SEED,", "B: int = A.N_BOOT, seed: int = A.SEED + 1,"),
 "M5_seal_ignored": ("if rec[0].lower() not in ok:", "if False:"),
 "M6_per_seed_mask_inverted": ("mask=(np.asarray(seed_idx) == s)", "mask=(np.asarray(seed_idx) != s)"),
 "M7_unmoved_logit_identity_removed": ("bad = np.argwhere(um & ~np.all(lo == lp[:, None, :], axis=2))", "bad = np.argwhere(um & False)"),
 "M8_p2head_difference_sign": ("return {\"point\": pa - pb,", "return {\"point\": pb - pa,"),
 "M9_pins_not_checked": ("if m.get(\"head_sha256\") != want_head:", "if False:"),
 "M10_label_binding_dropped": ("if str(lab[\"search_sha256\"]) != sha256_file(arm[\"search_path\"]):", "if False:"),
 "M11_done_marker_unanchored": ('DONE_RE = re.compile(r"^ZZSTEPB_DONE(?:\s|$)", re.M)', 'DONE_RE = re.compile(r"ZZSTEPB_DONE")'),
 "M12_identity_compares_with_tolerance": ("if not np.array_equal(pick_re, bank):", "if not np.allclose(pick_re, bank, atol=1e-6):"),
}
res = {}
for name, (old, new) in MUT.items():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__"))
    p = os.path.join(DST, "amend1_concordance.py")
    s = open(p, encoding="utf-8", newline="").read()
    if s.count(old) != 1:
        res[name] = f"MUTATION NOT APPLIED (count {s.count(old)})"
        continue
    open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([PY, "-m", "pytest", "test_amend1.py", "-q", "-p", "no:cacheprovider", "-x", "--basetemp=C:/Users/Admin/par18/l2stepb/test_amend1/mut/tmp"], cwd=DST, env=env, capture_output=True, text=True)
    last = [l for l in r.stdout.splitlines() if l.strip()][-1]
    failed = [l for l in r.stdout.splitlines() if l.startswith("FAILED")]
    res[name] = ("RED  " if r.returncode != 0 else "GREEN (NOT CAUGHT)  ") + (failed[0][:150] if failed else last)
for k, v in res.items():
    print(f"{k:42s} {v}")
