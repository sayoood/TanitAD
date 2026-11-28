"""Mutation proof for the scoped ORC queue tests: each mutant of the TIP queue must turn the two scoped tests RED; the unmutated queue GREEN.
python mutate_orc_scope.py <tree root>"""
import os
import subprocess
import sys

ROOT = sys.argv[1]
Q = os.path.join(ROOT, "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_mini_box2_queue.cmd")
q = open(Q, "rb").read()
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
K = "test_queue_is_crlf or test_queue_orc_stages_are_one_flat"


def mut(old, new, where=None):
    assert q.count(old) >= 1, old
    if where == "last":
        i = q.rfind(old)
        return q[:i] + new + q[i + len(old):]
    return q.replace(old, new, 1)


M = {
    "none": q,
    "undeclared_label_at_tail": q + b":dagfoo\r\nexit /b 0\r\n",
    "undeclared_label_between_orc_and_dag": mut(b"\r\n:dagname\r\n", b"\r\n:zzz\r\nexit /b 0\r\n:dagname\r\n"),
    "orc_ego_perturb_p_edited": mut(b"--ego-perturb-p %ORCP%%ORCL%", b"--ego-perturb-p 0.2%ORCL%"),
    "extra_ego_perturb_p_in_orc_region": mut(b"\r\n:evalorc\r\n", b"\r\n:evalorc\r\nrem x\r\nset \"ZZ=--ego-perturb-p 0\"\r\n"),
    "later_usage_line_moved_into_env": mut(b"rem   v8f_mini_box2_queue.cmd bankdag2 SEED", b"rem   v8f_mini_box2_queue.cmd bankdagX SEED"),
    "older_dispatch_line_dropped": mut(b'if /i "%~1"=="evalorc" goto :evalorc\r\n', b""),
    "orc_dispatch_doubled": mut(b'if /i "%~1"=="gateorc" goto :gateorc\r\n', b'if /i "%~1"=="gateorc" goto :gateorc\r\nif /i "%~1"=="gateorc" goto :gateorc\r\n'),
    "evalorc_tail_changed": mut(b"echo [ok] eval %TG%\r\nexit /b 0\r\n\r\n:dagname", b"echo [ok] eval %TG% done\r\nexit /b 0\r\n\r\n:dagname"),
}
res = {}
for name, b in M.items():
    p = os.path.join(ROOT, f"_mut_{name}.cmd")
    open(p, "wb").write(b)
    env = dict(os.environ, ORC_CHAIN_QUEUE=p, CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "stack/tests/test_v8f_orc_chain.py", "-k", K], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    res[name] = (r.returncode, last)
    os.remove(p)
    print(f"{name:42s} rc {r.returncode}  {last}")
ok = res["none"][0] == 0 and all(rc != 0 for n, (rc, _) in res.items() if n != "none")
print("MUTATION PROOF:", "PASS" if ok else "FAIL", f"({sum(rc != 0 for n, (rc, _) in res.items() if n != 'none')}/{len(res) - 1} killed)")
