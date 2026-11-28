"""run_mutations: deliberate mutations of wsp_edit.py; every one must turn tests/test_wsp_edit.py RED (the clean copy must be GREEN first). usage: run_mutations.py -> logs/mutations.log"""
import os, sys, shutil, subprocess, tempfile, json
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
src = open(HERE + "/wsp_edit.py", encoding="utf-8").read()
MUT = {
 "M1_lateral_rows_also_edited": ("    W[:nb] = torch.where(dWt == 0, Wx, Wx + dWt).to(W.dtype)", "    W[:nb] = torch.where(dWt == 0, Wx, Wx + dWt).to(W.dtype)\n    W[nb:2 * nb] = (W[nb:2 * nb].double() + dWt).to(W.dtype)"),
 "M2_bias_lateral_edited": ("    B[:nb] = torch.where(dbt == 0, Bx, Bx + dbt).to(B.dtype)", "    B[:nb] = torch.where(dbt == 0, Bx, Bx + dbt).to(B.dtype)\n    B[nb:2 * nb] = B[nb:2 * nb] + 1e-3"),
 "M3_u_sign_flipped": ("    u = unit_direction(r_res) * float(u_sign)", "    u = -unit_direction(r_res) * float(u_sign)"),
 "M4_alpha_not_applied_to_dW": ("    dW = float(alpha) * np.outer(u, d)", "    dW = np.outer(u, d)"),
 "M5_kappa_rule_dropped": ("    kappa = 1.0 - m / shift", "    kappa = 1.0"),
 "M6_u_not_unit_readout": ("    return r / float(r @ r)", "    return r / float(np.linalg.norm(r))"),
 "M7_ridge_lambda_zero_scale": ("    lam = lam_frac * float(np.trace(C))", "    lam = lam_frac * float(np.trace(C)) * 1e6"),
 "M8_zero_delta_not_skipped_sign": ("    return {k: v.clone() for k, v in sd.items()}", "    return {k: v.clone() for k, v in sd.items()}"),
 "M9_base_gain_used_as_model_gain": ("    dg_v, dg_a = g_v_base - g_v_model, g_a_base - g_a_model", "    dg_v, dg_a = g_v_base - g_v_base, g_a_base - g_a_base"),
 "M10_state_modified_in_place": ("    out = {k: v.clone() for k, v in sd.items()}\n    W = out[KEY_W]", "    out = sd\n    W = out[KEY_W]"),
}
log = []
def run(tmp):
    env = dict(os.environ, WSP_EDIT_SRC=tmp, CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", HERE + "/tests/test_wsp_edit.py"], capture_output=True, text=True, env=env)
    return r.returncode, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
def mk(mut):
    tmp = tempfile.mkdtemp(prefix="wspmut_")
    for f in ("wsp_common.py", "wsp_edit.py"):
        shutil.copy(HERE + "/" + f, tmp + "/" + f)
    if mut:
        a, b = mut
        s = open(tmp + "/wsp_edit.py", encoding="utf-8").read()
        assert a in s, a
        open(tmp + "/wsp_edit.py", "w", encoding="utf-8", newline="\n").write(s.replace(a, b, 1))
    return tmp
rc, last = run(mk(None)); log.append(f"CLEAN rc={rc} :: {last}")
assert rc == 0, "clean copy is not green"
ok_all = True
for k, mut in MUT.items():
    if k == "M8_zero_delta_not_skipped_sign": continue    # a no-op placeholder, not a mutation
    rc, last = run(mk(mut)); red = rc != 0; ok_all &= red
    log.append(f"{k}: {'RED (detected)' if red else 'GREEN (NOT DETECTED)'} :: {last}")
log.append("MUTATIONS ALL RED" if ok_all else "MUTATIONS NOT ALL RED")
open(HERE + "/logs/mutations.log", "w", encoding="utf-8", newline="\n").write("\n".join(log) + "\n")
print("\n".join(log))
