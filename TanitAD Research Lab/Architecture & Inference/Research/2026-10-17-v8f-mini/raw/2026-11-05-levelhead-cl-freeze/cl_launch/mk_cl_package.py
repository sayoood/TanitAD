"""Generator of the arm-CL launch package for box 2 (PREREG_LEVELHEAD_CL.md section 2). Offline except `stage` (reads F: and the git mirror) and `verify-tip` (git ls-remote on GitHub; never box 2).

  python -I mk_cl_package.py stage        copy the four shipped stack files into stack_ship/ (the two new ones from F:/TanitAD-scratch/motion_c/build/new, the two unchanged ones raw from the git mirror at the remote tip),
                                          sha256-check them against build/SHA_CHANGES.txt (new) and the tip blobs (unchanged), copy C0's two argv.json into c0_argv/
  python -I mk_cl_package.py              (gen) write PINS.json and every box-2 file from the staged sources, C0's argv, cl_prereg/FROZEN_W.txt (absent -> the placeholder __W__) and the card's sha256
  python -I mk_cl_package.py check        gen into a temp dir and compare byte for byte with what is on disk (exit 1 on any difference)
  python -I mk_cl_package.py verify-tip   git ls-remote origin <branch>: the unchanged files' blobs at that tip must still equal their pins (network: GitHub only)

The argv files are DERIVED from C0's argv.json (a copy under c0_argv/, equal to v8f-recover/M2b_s{s}_recft_C0/argv.json, sha256 in PINS.json) by two mechanical edits; cl_check.template.py holds an independent literal encoding of the same tokens, and the tests cross-check both against C0's own file.
.cmd files: CRLF, real backslashes (this file is written without a shell, so the raw strings below reach disk unchanged)."""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

H = Path(__file__).resolve().parent
MIRROR = "C:/Users/Admin/tanitad-push"
BRANCH = "agent/arch-inf-20260803"
BUILD_NEW = Path("F:/TanitAD-scratch/motion_c/build/new")
SHA_CHANGES = H.parent / "build" / "SHA_CHANGES.txt"
C0_DIR = Path("E:/Projects/TanitAD-artifacts/v8f-recover")
PREREG_DIR = Path(os.environ.get("CL_PREREG_DIR", str(H.parent / "cl_prereg")))
CARD = "PREREG_LEVELHEAD_CL.md"
PLACEHOLDER = "__W__"
#: rel path -> (staged origin, where the bytes come from). "new": the build's file (sha256 = SHA_CHANGES.txt `new`); "tip": unchanged from the remote tip (sha256 = the tip blob = the RPT launcher's pin).
STACK = {"stack/scripts/v8f_mini_train.py": "new", "stack/tanitad/models/v8f_mini.py": "new",
         "stack/tanitad/data/v8f_recover.py": "tip", "stack/tanitad/data/v8f_mini_data.py": "tip"}
# the RPT launcher's pins for the two files that stay as the tip has them (recover_next/box2_files/rnext_check.py SHIPPED); `stage` refuses a tip blob that differs
RPT_PINS = {"stack/tanitad/data/v8f_recover.py": "0257ccb93251827fe5f1b3d7195999b5836145e8615cfd0d0a599262b8834b4c",
            "stack/tanitad/data/v8f_mini_data.py": "209da7a60765ba81268ee794aba187e21b0a6c73a49b9e98c86e2d876581ece6"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*a):
    r = subprocess.run(["git", "-C", MIRROR, *a], capture_output=True)
    if r.returncode:
        raise SystemExit(f"git {a}: {r.stderr.decode()[:300]}")
    return r.stdout


def changes():
    """{rel: new sha256} from build/SHA_CHANGES.txt."""
    out, cur = {}, None
    for l in SHA_CHANGES.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(stack/\S+)\s+\[", l)
        if m:
            cur = m.group(1)
        m = re.match(r"^\s+new ([0-9a-f]{64})$", l)
        if m and cur:
            out[cur] = m.group(1)
    return out


def stage():
    ch = changes()
    tip = git("rev-parse", BRANCH).decode().strip()
    for rel, kind in STACK.items():
        data = (BUILD_NEW / rel).read_bytes() if kind == "new" else git("cat-file", "blob", f"{tip}:{rel}")
        want = ch.get(rel) if kind == "new" else RPT_PINS[rel]
        assert want and sha(data) == want, f"{rel}: sha256 {sha(data)} != {want} ({'SHA_CHANGES.txt' if kind == 'new' else 'the RPT launcher pin'})"
        dst = H / "stack_ship" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        print(f"staged {rel} {sha(data)[:12]} ({kind})")
    for s in (0, 1):
        src = C0_DIR / f"M2b_s{s}_recft_C0" / "argv.json"
        (H / "c0_argv" / f"C0_s{s}.argv.json").write_bytes(src.read_bytes())
        print(f"copied {src}")
    (H / "tip_commit.txt").write_text(tip + "\n", encoding="utf-8", newline="\n")
    print("tip", tip)


def frozen_w():
    p = PREREG_DIR / "FROZEN_W.txt"
    if not p.exists():
        return PLACEHOLDER, None
    m = None
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip() and not l.lstrip().startswith("#"):
            m = re.fullmatch(r"\s*(?:W\s*=\s*)?([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*", l)
            break
    if not m or not float(m.group(1)) > 0:
        raise SystemExit(f"FROZEN_W.txt: the first non-comment line must be a positive number (optionally `W=<number>`); got {l!r}")
    return m.group(1), p.read_bytes()


def c0_tokens(s):
    d = json.loads((H / "c0_argv" / f"C0_s{s}.argv.json").read_text(encoding="utf-8"))
    return d["argv"]


def cl_tokens(s, w, out_name="recft_CL", tail=None):
    t = list(c0_tokens(s))
    i = t.index("--out") + 1
    assert t[i].endswith(f"M2b_s{s}_recft_C0"), t[i]
    t[i] = t[i][:-len(f"M2b_s{s}_recft_C0")] + (f"M2b_s{s}_{out_name}")
    return t + (tail if tail is not None else ["--level-head", "on", "--w-level", w])


def build():
    """{relative path under the package: bytes} of every generated file."""
    out = {}
    w, wbytes = frozen_w()
    stack = [{"rel": rel, "src": f"stack_ship/{rel}", "sha256": sha((H / "stack_ship" / rel).read_bytes()), "origin": kind} for rel, kind in STACK.items()]
    card = PREREG_DIR / CARD
    pins = {"tip_commit": (H / "tip_commit.txt").read_text(encoding="utf-8").strip(), "stack": stack, "w": w,
            "card_sha256": sha(card.read_bytes()), "card_file": CARD, "frozen_w_sha256": sha(wbytes) if wbytes else None,
            "c0_argv_sha256": {str(s): sha((H / "c0_argv" / f"C0_s{s}.argv.json").read_bytes()) for s in (0, 1)}}
    pj = json.dumps(pins, indent=1, sort_keys=True)
    out["PINS.json"] = (pj + "\n").encode()
    for s in (0, 1):
        out[f"box2_files/argv/CL_{s}.train.txt"] = (" ".join(cl_tokens(s, w)) + "\n").encode()
    out["box2_files/argv/WPROBE_0.gate.txt"] = (" ".join(cl_tokens(0, "1", "wprobe_CL", ["--level-head", "on", "--w-level", "1", "--selfcheck-only"])) + "\n").encode()
    if wbytes is not None:
        out["box2_files/FROZEN_W.txt"] = wbytes
    out["box2_files/cl_check.py"] = (H / "cl_check.template.py").read_text(encoding="utf-8").replace("__PINS_JSON__", pj).encode()
    for name, lines in CMD.items():
        out[f"box2_files/{name}"] = ("\r\n".join(lines) + "\r\n").encode()
    for name, ends in (("cl_guard_wprobe.ps1", ["WPROBE.done", "WPROBE.FAILED", "WPROBE.STOPPED"]), ("cl_guard_chain.ps1", ["CHAIN_CL.done", "CHAIN_CL.STOPPED"])):
        out[f"box2_files/{name}"] = GUARD.replace("__ENDS__", ", ".join(f"'{e}'" for e in ends)).replace("__WHAT__", name[len("cl_guard_"):-len(".ps1")]).encode()
    return out


def ship_files():
    return sorted(k[len("box2_files/"):] for k in build() if k.startswith("box2_files/"))


RUNSAFE_NOTE = "rem %RUNSAFE% (env.cmd) = start /b /wait /belownormal /affinity FF; env.cmd also sets the 4 threads, AVX2 and TEMP (SAFE MODE, box-2 memory note)."
_CHAIN_HEAD = [
    "@echo off",
    r"rem LEVELHEAD arm CL chain (PREREG_LEVELHEAD_CL.md section 2): CL_0 then CL_1 = C0's argv + --level-head on --w-level W (the frozen W, FROZEN_W.txt), 3,000 steps each. Needs the PI's box-2 go (the launcher refuses without CL_PI_GO).",
    r"rem Per run: argv audit + pin -> in-process launch gate (the trainer's --selfcheck-only, asserted by cl_check.py) -> training -> post checks -> marker RUN_<tag>.done / RUN_<tag>.FAILED, all in D:\tanitad\runs\v8f_mini_full\cl.",
    "rem FAIL CLOSED, NEVER STALL THE GPU: a refusal / failed gate / failed run is marked FAILED and the chain proceeds to the NEXT run. Only a SAFETY stop (guard temperature stop, disk floor) halts the chain (CHAIN_CL.STOPPED).",
    r"rem Launch: powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\cl\cl_chain.cmd -Log D:\tanitad\logs\cl_chain.log",
    RUNSAFE_NOTE,
    "setlocal enabledelayedexpansion",
    r"call D:\tanitad\env.cmd",
    r'set "RUNSM=D:\tanitad\runs\v8f_mini_full"',
    r'set "RCD=%RUNSM%\cl"',
    r'set "CHK=D:\tanitad\cl\cl_check.py"',
    r'set "RUNCMD=D:\tanitad\cl\cl_run.cmd"',
]
_PY_GUARD = [
    "rem ONE job at a time (box 2 is the PI's machine and has blue-screened under load): refuse to start while ANY python process exists",
    'set "NPY=0"',
    r"""for /f %%N in ('tasklist /fi "imagename eq python.exe" /fo csv /nh ^| find /c "python.exe"') do set "NPY=%%N" """.rstrip(),
]
CMD = {
    "cl_chain.cmd": _CHAIN_HEAD + [
        "echo ===== CL CHAIN START %DATE% %TIME%",
        'if not exist "%RCD%" mkdir "%RCD%"',
        'if not exist "%RCD%\\logs" mkdir "%RCD%\\logs"',
    ] + _PY_GUARD + [
        r'if not "!NPY!"=="0" (echo ZZCL-STOP a python process is already running on box 2 ^(!NPY!^) & "%PY%" "%CHK%" stopped python process present & exit /b 3)',
        '"%PY%" "%CHK%" preflight chain',
        'set "RC=!ERRORLEVEL!"',
        r'if !RC! NEQ 0 (echo ZZCL-STOP preflight rc !RC! & "%PY%" "%CHK%" stopped preflight failed rc !RC! & exit /b 3)',
        "for %%T in (CL_0 CL_1) do (",
        "  call :one %%T",
        "  if !ERRORLEVEL! EQU 9 goto :halt",
        ")",
        '"%PY%" "%CHK%" finish',
        "echo ===== CL CHAIN END %DATE% %TIME%",
        r"echo stop> D:\tanitad\logs\telemetry.stop",
        "exit /b 0",
        "",
        ":halt",
        "echo ===== CL CHAIN HALTED %DATE% %TIME%",
        '"%PY%" "%CHK%" stopped safety stop, see the chain log',
        r"echo stop> D:\tanitad\logs\telemetry.stop",
        "exit /b 9",
        "",
        ":one",
        'set "TAG=%~1"',
        r'if exist "%RCD%\RUN_%TAG%.done" (echo [skip] %TAG% already done & exit /b 0)',
        r'if exist "%RCD%\GUARD_STOP.txt" (echo ZZCL-STOP guard stop marker present & exit /b 9)',
        '"%PY%" "%CHK%" preflight run %TAG%',
        'set "RC=!ERRORLEVEL!"',
        "if !RC! NEQ 0 exit /b 9",
        '"%PY%" "%CHK%" pin %TAG%',
        'set "RC=!ERRORLEVEL!"',
        r'if !RC! NEQ 0 ( "%PY%" "%CHK%" fail %TAG% pin refused rc !RC! & exit /b 0 )',
        '"%PY%" "%CHK%" gate-cached %TAG%',
        'set "RC=!ERRORLEVEL!"',
        r"if !RC! EQU 0 ( echo [%DATE% %TIME%] GATE %TAG% PASS ^(cached: the gate token is bound to this commit, argv, selfcheck and shipped bytes^) & goto :traintime )",
        "echo [%DATE% %TIME%] GATE %TAG%",
        "%RUNSAFE% cmd /c %RUNCMD% gate %TAG%",
        'set "RC=!ERRORLEVEL!"',
        '"%PY%" "%CHK%" gate %TAG% !RC!',
        'set "GRC=!ERRORLEVEL!"',
        r"if !GRC! NEQ 0 ( echo [FAIL] gate %TAG% - see %RCD%\logs\%TAG%.gate.log & exit /b 0 )",
        r'if exist "%RCD%\GUARD_STOP.txt" exit /b 9',
        ":traintime",
        "echo [%DATE% %TIME%] TRAIN %TAG%",
        "%RUNSAFE% cmd /c %RUNCMD% train %TAG%",
        'set "RC=!ERRORLEVEL!"',
        '"%PY%" "%CHK%" post %TAG% !RC!',
        "echo [%DATE% %TIME%] END %TAG% trainer rc !RC!",
        "exit /b 0",
    ],
    "cl_wprobe.cmd": [
        "@echo off",
        r"rem LEVELHEAD W PROBE (step 0 of arm CL): CL_0's argv with --selfcheck-only --level-head on --w-level 1 and its own --out (M2b_s0_wprobe_CL). It trains nothing; it writes selfcheck.json, from which the pull reads",
        r"rem level_head_probe.w_level_for_10pct_share. Markers WPROBE.done / WPROBE.FAILED / WPROBE.STOPPED in D:\tanitad\runs\v8f_mini_full\cl.",
        r"rem Launch: powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\cl\cl_wprobe.cmd -Log D:\tanitad\logs\cl_wprobe.log",
        RUNSAFE_NOTE,
        "setlocal enabledelayedexpansion",
        r"call D:\tanitad\env.cmd",
        r'set "RUNSM=D:\tanitad\runs\v8f_mini_full"',
        r'set "RCD=%RUNSM%\cl"',
        r'set "CHK=D:\tanitad\cl\cl_check.py"',
        r'set "RUNCMD=D:\tanitad\cl\cl_run.cmd"',
        "echo ===== CL WPROBE START %DATE% %TIME%",
        'if not exist "%RCD%" mkdir "%RCD%"',
        'if not exist "%RCD%\\logs" mkdir "%RCD%\\logs"',
    ] + _PY_GUARD + [
        r'if not "!NPY!"=="0" (echo ZZCL-STOP a python process is already running on box 2 ^(!NPY!^) & "%PY%" "%CHK%" wprobe-stopped python process present & echo stop> D:\tanitad\logs\telemetry.stop & exit /b 3)',
        '"%PY%" "%CHK%" preflight wprobe',
        'set "RC=!ERRORLEVEL!"',
        r'if !RC! NEQ 0 (echo ZZCL-STOP preflight rc !RC! & "%PY%" "%CHK%" wprobe-stopped preflight failed rc !RC! & echo stop> D:\tanitad\logs\telemetry.stop & exit /b 3)',
        '"%PY%" "%CHK%" pin WPROBE_0',
        'set "RC=!ERRORLEVEL!"',
        r'if !RC! NEQ 0 ( "%PY%" "%CHK%" fail WPROBE_0 pin refused rc !RC! & echo stop> D:\tanitad\logs\telemetry.stop & exit /b 3 )',
        "echo [%DATE% %TIME%] GATE WPROBE_0",
        "%RUNSAFE% cmd /c %RUNCMD% gate WPROBE_0",
        'set "RC=!ERRORLEVEL!"',
        '"%PY%" "%CHK%" gate WPROBE_0 !RC!',
        'set "GRC=!ERRORLEVEL!"',
        "echo ===== CL WPROBE END gate-verdict-rc=!GRC! trainer-rc=!RC! %DATE% %TIME%",
        r"echo stop> D:\tanitad\logs\telemetry.stop",
        "exit /b !GRC!",
    ],
    "cl_run.cmd": [
        "@echo off",
        r"rem LEVELHEAD arm CL: ONE trainer invocation (gate = the argv's --selfcheck-only twin, or the training run; the W probe is a gate-mode call) from its PINNED line. Called as: cmd /c D:\tanitad\cl\cl_run.cmd <gate|train> <tag> under %RUNSAFE%.",
        r"rem The exit code is the trainer's: the caller (cl_check.py gate / post) judges it AND the artifacts. Resumes from ckpt_last.pt when the same line is run again (the trainer's own rule).",
        "setlocal enabledelayedexpansion",
        r"call D:\tanitad\env.cmd",
        'set "MODE=%~1"',
        'set "TAG=%~2"',
        r'set "RUNSM=D:\tanitad\runs\v8f_mini_full"',
        r'set "TRAIN=D:\tanitad\repo\stack\scripts\v8f_mini_train.py"',
        r'set "PINF=%RUNSM%\cl\pin\%TAG%.%MODE%.txt"',
        r'set "LOG=%RUNSM%\cl\logs\%TAG%.%MODE%.log"',
        'set "X="',
        'for /f "usebackq delims=" %%L in ("%PINF%") do set "X=%%L"',
        "if not defined X (echo REFUSED: no pinned line at %PINF% & exit /b 2)",
        'echo ===== RUN-START %MODE% %TAG% %DATE% %TIME% >> "%LOG%"',
        '"%PY%" "%TRAIN%" !X! >> "%LOG%" 2>&1',
        'set "RC=!ERRORLEVEL!"',
        'echo ===== RUN-END %MODE% %TAG% rc=!RC! %DATE% %TIME% >> "%LOG%"',
        "exit /b !RC!",
    ],
    "cl_run_guard_wprobe.cmd": ["@echo off", r"rem LEVELHEAD CL: the box-2 temperature guard for the W probe (ends on WPROBE.done / FAILED / STOPPED). Same rule as rnext_guard.ps1.",
                                r"powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\cl\cl_guard_wprobe.ps1"],
    "cl_run_guard_chain.cmd": ["@echo off", r"rem LEVELHEAD CL: the box-2 temperature guard for the chain (ends on CHAIN_CL.done / STOPPED only: a stale WPROBE marker cannot end it). Same rule as rnext_guard.ps1.",
                               r"powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\cl\cl_guard_chain.ps1"],
    "cl_run_telemetry.cmd": [
        "@echo off",
        r"rem LEVELHEAD CL: the long telemetry (read-only; D:\tanitad\telemetry_long.ps1, UNCHANGED and reused by path: one row per 15 s into D:\tanitad\logs\telemetry_full.csv; it stops when D:\tanitad\logs\telemetry.stop exists, which the job writes at its end).",
        r"rem A previous job's telemetry.stop would end the new telemetry at once and leave the guard blind, so it is MOVED ASIDE (never deleted) first.",
        r"""if exist D:\tanitad\logs\telemetry.stop powershell -NoProfile -Command "Move-Item -LiteralPath 'D:\tanitad\logs\telemetry.stop' -Destination ('D:\tanitad\logs\telemetry.stop.prev_' + (Get-Date -Format yyyyMMdd_HHmmss))" """.rstrip(),
        r"powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\telemetry_long.ps1",
    ],
}
GUARD = r"""# LEVELHEAD arm CL guard (__WHAT__), runs ON box 2 (a dev-box guard would die with the launching session). v8F rule, unchanged (rnext_guard.ps1 / v8f_guard_*.sh): the bus-powered USB HDD (D:) holds cache_full;
#   STOP if >= 4 of the last 20 telemetry samples (15 s apart) read >= 61 C, or any one reads >= 63 C. Stop = write GUARD_STOP.txt (the chain halts, never proceeds to the next run) and Stop-Process OUR trainer (cmdline v8f_mini_train) + its children.
# A stopped run resumes from ckpt_last.pt on the same pinned line. Exits when one of its own end markers appears, or after 9 h. Reads only; touches nothing outside D:\tanitad.
# A telemetry csv that has not been written for 10 min is LOGGED as stale (the guard is blind then); it never stops anything.
$rc  = 'D:\tanitad\runs\v8f_mini_full\cl'
$csv = 'D:\tanitad\logs\telemetry_full.csv'
$log = 'D:\tanitad\logs\cl_guard.log'
$ends = @(__ENDS__)
$t0 = Get-Date; $hi = 0; $lastStale = $null
Add-Content $log ((Get-Date).ToString('s') + ' guard armed (USB >= 4/20 samples >= 61 C or one >= 63 C)')
while (((Get-Date) - $t0).TotalHours -lt 9) {
  if (@($ends | Where-Object { Test-Path "$rc\$_" }).Count -gt 0) { Add-Content $log ((Get-Date).ToString('s') + " job ended; guard off (max USB $hi C)"); break }
  $fi = Get-Item $csv -ErrorAction SilentlyContinue
  if ((-not $fi) -or (((Get-Date) - $fi.LastWriteTime).TotalMinutes -gt 10)) {
    if ((-not $lastStale) -or (((Get-Date) - $lastStale).TotalMinutes -gt 10)) { Add-Content $log ((Get-Date).ToString('s') + ' WARNING telemetry csv missing or not written for 10 min: the guard is blind'); $lastStale = Get-Date }
  }
  $rows = @(Get-Content $csv -Tail 20 -ErrorAction SilentlyContinue | Where-Object { $_ -match '^20' })
  $u = @($rows | ForEach-Object { $f = $_.Split(','); if ($f.Count -ge 11 -and $f[10] -match '^\d+$') { [int]$f[10] } })
  if ($u.Count -gt 0) {
    $n61 = @($u | Where-Object { $_ -ge 61 }).Count
    $m = ($u | Measure-Object -Maximum).Maximum
    if ($m -gt $hi) { $hi = $m; Add-Content $log ((Get-Date).ToString('s') + " new max USB temp $m C") }
    if ($n61 -ge 4 -or $m -ge 63) {
      New-Item -ItemType Directory -Force $rc | Out-Null
      Set-Content "$rc\GUARD_STOP.txt" ("ZZGUARD-STOP TEMP " + (Get-Date).ToString('s') + " USB HDD: $n61 of the last 20 samples >= 61 C, max $m C")
      $p = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'")
      $ours = @($p | Where-Object { $_.CommandLine -match 'v8f_mini_train' })
      $ids = @($ours | ForEach-Object { $_.ProcessId })
      $kids = @($p | Where-Object { $ids -contains $_.ParentProcessId })
      foreach ($x in @($ours) + @($kids)) { Stop-Process -Id $x.ProcessId -Force -ErrorAction SilentlyContinue }
      Add-Content $log ((Get-Date).ToString('s') + " ZZGUARD-STOP TEMP: $n61 of 20 >= 61 C, max $m C; stopped $(@($ours).Count + @($kids).Count) process(es)")
      break
    }
  }
  Start-Sleep -Seconds 60
}
"""


def write_all(files, root):
    for rel, data in files.items():
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def main(a):
    cmd = a[0] if a else "gen"
    if cmd == "stage":
        return stage()
    if cmd == "gen":
        files = build()
        write_all(files, H)
        w = json.loads(files["PINS.json"])["w"]
        print(f"wrote {len(files)} files; W = {w}" + ("  (PLACEHOLDER: the launcher refuses the chain until FROZEN_W.txt exists and this is regenerated)" if w == PLACEHOLDER else ""))
        return 0
    if cmd == "check":
        files, bad = build(), []
        for rel, data in files.items():
            p = H / rel
            if not p.exists() or p.read_bytes() != data:
                bad.append(rel)
        stale = [str(p.relative_to(H)).replace("\\", "/") for p in (H / "box2_files").rglob("*") if p.is_file() and "__pycache__" not in p.parts and str(p.relative_to(H)).replace("\\", "/") not in files]
        print("check: " + ("OK, every generated file is what the generator writes" if not bad and not stale else f"DIFFERS {bad} STALE {stale}"))
        return 1 if bad or stale else 0
    if cmd == "verify-tip":
        pins = json.loads((H / "PINS.json").read_text(encoding="utf-8"))
        r = subprocess.run(["git", "-C", MIRROR, "ls-remote", "origin", BRANCH], capture_output=True, text=True)
        tip = r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None
        if not tip:
            print("verify-tip: ls-remote failed", r.stderr[:200]); return 2
        git("cat-file", "-t", tip)
        bad = []
        for x in pins["stack"]:
            if x["origin"] == "tip":
                got = sha(git("cat-file", "blob", f"{tip}:{x['rel']}"))
                if got != x["sha256"]:
                    bad.append(f"{x['rel']}: tip {tip[:8]} blob {got[:12]} != pin {x['sha256'][:12]}")
            else:
                got = sha(git("cat-file", "blob", f"{tip}:{x['rel']}"))
                print(f"info: {x['rel']} at tip {tip[:8]} is {got[:12]}; the build replaced it by {x['sha256'][:12]}")
        print(f"verify-tip: remote tip {tip[:8]} (staged against {pins['tip_commit'][:8]}): " + ("the two unchanged files still equal their pins" if not bad else "; ".join(bad)))
        return 1 if bad else 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
