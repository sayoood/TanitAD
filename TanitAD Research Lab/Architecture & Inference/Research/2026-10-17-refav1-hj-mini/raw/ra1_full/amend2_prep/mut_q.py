import sys

src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
BS = bytes([92])
CRLF = bytes([13, 10])
b = open(src, 'rb').read()
i = b.index(CRLF + b':full5' + CRLF)
j = b.index(b'exit /b 0', i)
blk = b[i:j]
if mode == 'dir':
    old = b'set "RUNS=%FROOT%' + BS + b'st%~2"'
    assert blk.count(old) == 1
    blk = blk.replace(old, b'set "RUNS=%FROOT%' + BS + b'st%~2_g"')
elif mode == 'verdict':
    blk = blk + b'set "VN=full5_verdict.json"' + CRLF + b'call :verdict --run "A1@0=%RUNS%' + BS + b'A1_x' + BS + b'eval"' + CRLF
elif mode == 'beta':
    old = b'A1 %%S %~3 %~4 ""'
    assert blk.count(old) == 1
    blk = blk.replace(old, b'A1 %%S 0.04 %~4 ""')
elif mode == 'oneseed':
    old = b'for %%S in (0 1) do'
    assert blk.count(old) == 1
    blk = blk.replace(old, b'for %%S in (0) do')
open(dst, 'wb').write(b[:i] + blk + b[j:])
