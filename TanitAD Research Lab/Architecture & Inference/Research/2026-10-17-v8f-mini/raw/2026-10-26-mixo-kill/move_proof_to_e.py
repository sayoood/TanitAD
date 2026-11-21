"""Point the MIXO mutation proof's scratch at E: so it no longer needs 3 GB free on C: (2026-10-26 07:20).
The tree stays on C: (mutate_mixo.py edits it in place); only pytest basetemps and the mutant scratch move. Originals kept as *.pre_e."""
import os
import shutil

K = "C:/Users/Admin/mixo_kill/"
MM = K + "tree/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-mixo/mutate_mixo.py"
E = "E:\\Projects\\TanitAD-artifacts\\mixo_kill_e"


def rep(path, pairs, newline):
    shutil.copyfile(path, path + ".pre_e")
    s = open(path, encoding="utf-8", newline="").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a)
        s = s.replace(a, b)
    assert "\x08" not in s and "\t" not in s
    open(path, "w", encoding="utf-8", newline="").write(s)


rep(MM, [('BASE = Path("C:/Users/Admin/mixo_kill/tmp") / "mut"\nBASE.parent.mkdir(exist_ok=True)\n',
          'BASE = Path(os.environ.get("MIXO_MUT_BASE", "C:/Users/Admin/mixo_kill/tmp")) / "mut"   # default unchanged; MIXO_MUT_BASE moves the scratch off C:\n'
          'BASE.parent.mkdir(parents=True, exist_ok=True)\n')], "")
rp = K + "run_proof.cmd"
crlf = b"\r\n" in open(rp, "rb").read()
nl = "\r\n" if crlf else "\n"
rep(rp, [('set "PYTHONDONTWRITEBYTECODE=1"' + nl,
          'set "PYTHONDONTWRITEBYTECODE=1"' + nl + 'set "MIXO_MUT_BASE=' + E + '\\tmp"' + nl + 'set "TMP=' + E + '\\systmp"' + nl
          + 'set "TEMP=' + E + '\\systmp"' + nl + 'if not exist "' + E + '\\systmp" mkdir "' + E + '\\systmp"' + nl),
         ('--basetemp "%KILL%\\tmp\\coupled"', '--basetemp "' + E + '\\tmp\\coupled"'),
         ('--basetemp "%KILL%\\tmp\\detail"', '--basetemp "' + E + '\\tmp\\detail"')], nl)
wp = K + "wait_no_pytest.ps1"
rep(wp, [("$deadline = (Get-Date).AddMinutes(90)", "$deadline = (Get-Date).AddMinutes(240)"),
         ("$free -ge 3GB", "$free -ge 1GB"), ("Free -lt 3GB", "Free -lt 1GB")], "")
os.makedirs(E.replace("\\", "/") + "/tmp", exist_ok=True)
print("ok; run_proof CRLF =", crlf)
