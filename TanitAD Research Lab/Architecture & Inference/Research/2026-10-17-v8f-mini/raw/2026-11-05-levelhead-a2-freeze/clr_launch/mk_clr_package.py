"""Generator of the arm-CLR launch package for box 2 (PREREG_LEVELHEAD_A2.md, section "Arms": CLR_s{0,1} = RAPR-P01's argv + `--level-head on --w-level 0.2027`). Offline except `stage` (reads the git mirror) and `verify-tip` (git ls-remote on GitHub; never box 2).

  python -I mk_clr_package.py stage        copy the four shipped stack files raw from the git mirror at the REMOTE tip (git ls-remote), sha256-checking each against its pin below, and copy RAPR-P01's two argv.json into rapr_argv/
  python -I mk_clr_package.py              (gen) write PINS.json and every box-2 file from the staged sources, RAPR-P01's argv, CL's frozen W (cl_prereg/FROZEN_W.txt: REQUIRED, there is no placeholder state) and the A2 card's sha256
  python -I mk_clr_package.py check        gen into memory and compare byte for byte with what is on disk (exit 1 on any difference)
  python -I mk_clr_package.py verify-tip   git ls-remote origin <branch>: all four shipped files' blobs at that tip must equal their pins (network: GitHub only)

The argv files are DERIVED from RAPR-P01's argv.json (a copy under rapr_argv/, equal to v8f-recover/M2b_s{s}_recft_RAPRP01/argv.json, sha256 in PINS.json) by two mechanical edits (`--out` -> ...M2b_s{s}_recft_CLR, and `--level-head on --w-level W` appended);
clr_check.template.py holds an independent literal encoding of the same tokens, and the tests cross-check both against RAPR-P01's own file.
.cmd files: CRLF, real backslashes (this file is written without a shell, so the raw strings below reach disk unchanged)."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
MIRROR = "C:/Users/Admin/tanitad-push"
BRANCH = "agent/arch-inf-20260803"
RAPR_DIR = Path("E:/Projects/TanitAD-artifacts/v8f-recover")
PREREG_DIR = Path(os.environ.get("CLR_PREREG_DIR", str(H.parent / "a2_prereg")))      # the frozen card A2 and its FROZEN_SHA.txt
W_DIR = Path(os.environ.get("CLR_W_DIR", str(H.parent / "cl_prereg")))                  # CL's frozen W (FROZEN_W.txt) and its probe (W_PROBE.json): arm CLR runs NO probe of its own
CARD = "PREREG_LEVELHEAD_A2.md"
#: rel path -> pinned sha256. All four equal the blobs at remote tip 219f6c76 (checked 2026-11-05): trainer and model are the CL build (6ece77f), the two data files are the RPT pins.
STACK = {"stack/scripts/v8f_mini_train.py": "9ffdf1b4788358d3eb2698f11ca4008bffb042c5ee3887995bac4f01adb390ae",
         "stack/tanitad/models/v8f_mini.py": "a363427677c881cb7c82cf20fcfad317e864a9cde58cf58306e9fed24ef3658e",
         "stack/tanitad/data/v8f_recover.py": "0257ccb93251827fe5f1b3d7195999b5836145e8615cfd0d0a599262b8834b4c",
         "stack/tanitad/data/v8f_mini_data.py": "209da7a60765ba81268ee794aba187e21b0a6c73a49b9e98c86e2d876581ece6"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*a):
    r = subprocess.run(["git", "-C", MIRROR, *a], capture_output=True)
    if r.returncode:
        raise SystemExit(f"git {a}: {r.stderr.decode()[:300]}")
    return r.stdout


def remote_tip():
    r = subprocess.run(["git", "-C", MIRROR, "ls-remote", "origin", BRANCH], capture_output=True, text=True)
    tip = r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None
    if not tip:
        raise SystemExit(f"git ls-remote origin {BRANCH} failed: {r.stderr[:200]}")
    return tip


def stage():
    tip = remote_tip()
    git("cat-file", "-t", tip)                              # the remote tip must be an object of the mirror (else the mirror is behind: fetch it first)
    for rel, want in STACK.items():
        data = git("cat-file", "blob", f"{tip}:{rel}")
        assert sha(data) == want, f"{rel}: the blob at remote tip {tip[:8]} hashes {sha(data)} != the pin {want}"
        dst = H / "stack_ship" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        print(f"staged {rel} {want[:12]} (tip {tip[:8]})")
    for s in (0, 1):
        src = RAPR_DIR / f"M2b_s{s}_recft_RAPRP01" / "argv.json"
        (H / "rapr_argv").mkdir(exist_ok=True)
        (H / "rapr_argv" / f"RAPRP01_s{s}.argv.json").write_bytes(src.read_bytes())
        print(f"copied {src}")
    (H / "tip_commit.txt").write_text(tip + "\n", encoding="utf-8", newline="\n")
    print("tip", tip)


def frozen_w():
    p = W_DIR / "FROZEN_W.txt"
    if not p.exists():
        raise SystemExit(f"{p} is absent: arm CLR uses CL's frozen W, there is no placeholder state")
    m = None
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip() and not l.lstrip().startswith("#"):
            m = re.fullmatch(r"\s*(?:W\s*=\s*)?([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*", l)
            break
    if not m or not float(m.group(1)) > 0:
        raise SystemExit(f"FROZEN_W.txt: the first non-comment line must be a positive number (optionally `W=<number>`); got {l!r}")
    return m.group(1), p.read_bytes()


def rapr_tokens(s):
    d = json.loads((H / "rapr_argv" / f"RAPRP01_s{s}.argv.json").read_text(encoding="utf-8"))
    return d["argv"]


def clr_tokens(s, w):
    t = list(rapr_tokens(s))
    i = t.index("--out") + 1
    assert t[i].endswith(f"M2b_s{s}_recft_RAPRP01"), t[i]
    t[i] = t[i][:-len(f"M2b_s{s}_recft_RAPRP01")] + f"M2b_s{s}_recft_CLR"
    return t + ["--level-head", "on", "--w-level", w]


def build():
    """{relative path under the package: bytes} of every generated file."""
    out = {}
    w, wbytes = frozen_w()
    stack = []
    for rel, want in STACK.items():
        got = sha((H / "stack_ship" / rel).read_bytes())
        if got != want:
            raise SystemExit(f"stack_ship/{rel} hashes {got}, the pin is {want} (run: python -I mk_clr_package.py stage)")
        stack.append({"rel": rel, "src": f"stack_ship/{rel}", "sha256": got, "origin": "tip"})
    card = PREREG_DIR / CARD
    pins = {"tip_commit": (H / "tip_commit.txt").read_text(encoding="utf-8").strip(), "stack": stack, "w": w, "w_source": "cl_prereg/FROZEN_W.txt (CL's frozen W; no probe of arm CLR's own)",
            "card_sha256": sha(card.read_bytes()), "card_file": CARD, "card_frozen_sha_file": "FROZEN_SHA.txt", "frozen_w_sha256": sha(wbytes),
            "rapr_argv_sha256": {str(s): sha((H / "rapr_argv" / f"RAPRP01_s{s}.argv.json").read_bytes()) for s in (0, 1)}}
    pj = json.dumps(pins, indent=1, sort_keys=True)
    out["PINS.json"] = (pj + "\n").encode()
    for s in (0, 1):
        out[f"box2_files/argv/CLR_{s}.train.txt"] = (" ".join(clr_tokens(s, w)) + "\n").encode()
    out["box2_files/FROZEN_W.txt"] = wbytes
    out["box2_files/clr_check.py"] = (H / "clr_check.template.py").read_text(encoding="utf-8").replace("__PINS_JSON__", pj).encode()
    for name, lines in CMD.items():
        out[f"box2_files/{name}"] = ("\r\n".join(lines) + "\r\n").encode()
    out["box2_files/clr_guard_chain.ps1"] = GUARD.replace("__ENDS__", ", ".join(f"'{e}'" for e in ["CHAIN_CLR.done", "CHAIN_CLR.STOPPED"])).replace("__WHAT__", "chain").encode()
    return out


def ship_files():
    return sorted(k[len("box2_files/"):] for k in build() if k.startswith("box2_files/"))


RUNSAFE_NOTE = "rem %RUNSAFE% (env.cmd) = start /b /wait /belownormal /affinity FF; env.cmd also sets the 4 threads, AVX2 and TEMP (SAFE MODE, box-2 memory note)."
_CHAIN_HEAD = [
    "@echo off",
    r"rem LEVELHEAD arm CLR chain (PREREG_LEVELHEAD_A2.md, Arms): CLR_0 then CLR_1 = RAPR-P01's argv (recovery p 0.1, a-priori law, rejoin target) + --level-head on --w-level W (CL's frozen W, FROZEN_W.txt), 3,000 steps each. Needs the PI's box-2 go (the launcher refuses without CLR_PI_GO).",
    r"rem Per run: argv audit + pin -> in-process launch gate (the trainer's --selfcheck-only, asserted by clr_check.py) -> training -> post checks -> marker RUN_<tag>.done / RUN_<tag>.FAILED, all in D:\tanitad\runs\v8f_mini_full\clr.",
    "rem FAIL CLOSED, NEVER STALL THE GPU: a refusal / failed gate / failed run is marked FAILED and the chain proceeds to the NEXT run. Only a SAFETY stop (guard temperature stop, disk floor) halts the chain (CHAIN_CLR.STOPPED).",
    r"rem Launch: powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\clr\clr_chain.cmd -Log D:\tanitad\logs\clr_chain.log",
    RUNSAFE_NOTE,
    "setlocal enabledelayedexpansion",
    r"call D:\tanitad\env.cmd",
    r'set "RUNSM=D:\tanitad\runs\v8f_mini_full"',
    r'set "RCD=%RUNSM%\clr"',
    r'set "CHK=D:\tanitad\clr\clr_check.py"',
    r'set "RUNCMD=D:\tanitad\clr\clr_run.cmd"',
]
_PY_GUARD = [
    "rem ONE job at a time (box 2 is the PI's machine and has blue-screened under load): refuse to start while ANY python process exists",
    'set "NPY=0"',
    r"""for /f %%N in ('tasklist /fi "imagename eq python.exe" /fo csv /nh ^| find /c "python.exe"') do set "NPY=%%N" """.rstrip(),
]
CMD = {
    "clr_chain.cmd": _CHAIN_HEAD + [
        "echo ===== CLR CHAIN START %DATE% %TIME%",
        'if not exist "%RCD%" mkdir "%RCD%"',
        'if not exist "%RCD%\\logs" mkdir "%RCD%\\logs"',
    ] + _PY_GUARD + [
        r'if not "!NPY!"=="0" (echo ZZCLR-STOP a python process is already running on box 2 ^(!NPY!^) & "%PY%" "%CHK%" stopped python process present & exit /b 3)',
        '"%PY%" "%CHK%" preflight chain',
        'set "RC=!ERRORLEVEL!"',
        r'if !RC! NEQ 0 (echo ZZCLR-STOP preflight rc !RC! & "%PY%" "%CHK%" stopped preflight failed rc !RC! & exit /b 3)',
        "for %%T in (CLR_0 CLR_1) do (",
        "  call :one %%T",
        "  if !ERRORLEVEL! EQU 9 goto :halt",
        ")",
        '"%PY%" "%CHK%" finish',
        "echo ===== CLR CHAIN END %DATE% %TIME%",
        r"echo stop> D:\tanitad\logs\telemetry.stop",
        "exit /b 0",
        "",
        ":halt",
        "echo ===== CLR CHAIN HALTED %DATE% %TIME%",
        '"%PY%" "%CHK%" stopped safety stop, see the chain log',
        r"echo stop> D:\tanitad\logs\telemetry.stop",
        "exit /b 9",
        "",
        ":one",
        'set "TAG=%~1"',
        r'if exist "%RCD%\RUN_%TAG%.done" (echo [skip] %TAG% already done & exit /b 0)',
        r'if exist "%RCD%\GUARD_STOP.txt" (echo ZZCLR-STOP guard stop marker present & exit /b 9)',
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
    "clr_run.cmd": [
        "@echo off",
        r"rem LEVELHEAD arm CLR: ONE trainer invocation (gate = the argv's --selfcheck-only twin, or the training run) from its PINNED line. Called as: cmd /c D:\tanitad\clr\clr_run.cmd <gate|train> <tag> under %RUNSAFE%.",
        r"rem The exit code is the trainer's: the caller (clr_check.py gate / post) judges it AND the artifacts. Resumes from ckpt_last.pt when the same line is run again (the trainer's own rule).",
        "setlocal enabledelayedexpansion",
        r"call D:\tanitad\env.cmd",
        'set "MODE=%~1"',
        'set "TAG=%~2"',
        r'set "RUNSM=D:\tanitad\runs\v8f_mini_full"',
        r'set "TRAIN=D:\tanitad\repo\stack\scripts\v8f_mini_train.py"',
        r'set "PINF=%RUNSM%\clr\pin\%TAG%.%MODE%.txt"',
        r'set "LOG=%RUNSM%\clr\logs\%TAG%.%MODE%.log"',
        'set "X="',
        'for /f "usebackq delims=" %%L in ("%PINF%") do set "X=%%L"',
        "if not defined X (echo REFUSED: no pinned line at %PINF% & exit /b 2)",
        'echo ===== RUN-START %MODE% %TAG% %DATE% %TIME% >> "%LOG%"',
        '"%PY%" "%TRAIN%" !X! >> "%LOG%" 2>&1',
        'set "RC=!ERRORLEVEL!"',
        'echo ===== RUN-END %MODE% %TAG% rc=!RC! %DATE% %TIME% >> "%LOG%"',
        "exit /b !RC!",
    ],
    "clr_run_guard_chain.cmd": ["@echo off", r"rem LEVELHEAD CLR: the box-2 temperature guard for the chain (ends on CHAIN_CLR.done / STOPPED only: a stale marker of another job cannot end it). Same rule as rnext_guard.ps1.",
                               r"powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\clr\clr_guard_chain.ps1"],
    "clr_run_telemetry.cmd": [
        "@echo off",
        r"rem LEVELHEAD CLR: the long telemetry (read-only; D:\tanitad\telemetry_long.ps1, UNCHANGED and reused by path: one row per 15 s into D:\tanitad\logs\telemetry_full.csv; it stops when D:\tanitad\logs\telemetry.stop exists, which the job writes at its end).",
        r"rem A previous job's telemetry.stop would end the new telemetry at once and leave the guard blind, so it is MOVED ASIDE (never deleted) first.",
        r"""if exist D:\tanitad\logs\telemetry.stop powershell -NoProfile -Command "Move-Item -LiteralPath 'D:\tanitad\logs\telemetry.stop' -Destination ('D:\tanitad\logs\telemetry.stop.prev_' + (Get-Date -Format yyyyMMdd_HHmmss))" """.rstrip(),
        r"powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\telemetry_long.ps1",
    ],
}
GUARD = r"""# LEVELHEAD arm CLR guard (__WHAT__), runs ON box 2 (a dev-box guard would die with the launching session). v8F rule, unchanged (rnext_guard.ps1 / v8f_guard_*.sh): the bus-powered USB HDD (D:) holds cache_full;
#   STOP if >= 4 of the last 20 telemetry samples (15 s apart) read >= 61 C, or any one reads >= 63 C. Stop = write GUARD_STOP.txt (the chain halts, never proceeds to the next run) and Stop-Process OUR trainer (cmdline v8f_mini_train) + its children.
# A stopped run resumes from ckpt_last.pt on the same pinned line. Exits when one of its own end markers appears, or after 9 h. Reads only; touches nothing outside D:\tanitad.
# A telemetry csv that has not been written for 10 min is LOGGED as stale (the guard is blind then); it never stops anything.
$rc  = 'D:\tanitad\runs\v8f_mini_full\clr'
$csv = 'D:\tanitad\logs\telemetry_full.csv'
$log = 'D:\tanitad\logs\clr_guard.log'
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
        print(f"wrote {len(files)} files; W = {json.loads(files['PINS.json'])['w']}")
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
        tip = remote_tip()
        git("cat-file", "-t", tip)
        bad = []
        for x in pins["stack"]:
            got = sha(git("cat-file", "blob", f"{tip}:{x['rel']}"))
            if got != x["sha256"]:
                bad.append(f"{x['rel']}: tip {tip[:8]} blob {got[:12]} != pin {x['sha256'][:12]}")
        print(f"verify-tip: remote tip {tip[:8]} (staged against {pins['tip_commit'][:8]}): " + ("all four shipped files equal their pins" if not bad else "; ".join(bad)))
        return 1 if bad else 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
