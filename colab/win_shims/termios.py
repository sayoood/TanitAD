# UNSUPPORTED import-shim for running google-colab-cli on native Windows.
# Google supports the Colab CLI on Linux/macOS only; colab_cli imports the
# Unix-only `termios` module unconditionally (console.py:20), so even
# `colab --help` dies on Windows. Putting this directory on PYTHONPATH
# satisfies the import; any REAL terminal-mode call raises loudly instead of
# pretending to work (so `repl`/`console` fail honestly — use exec/run).
# Provenance: colab/COLAB_CLI_MCP.md §3.1 (MEASURED 2026-08-16).
TCSADRAIN = 1
TCSANOW = 0
TCSAFLUSH = 2


def _unsupported(*a, **k):
    raise OSError("termios shim: real terminal control is not available on Windows")


tcgetattr = _unsupported
tcsetattr = _unsupported
tcdrain = _unsupported
tcflush = _unsupported
tcflow = _unsupported
tcsendbreak = _unsupported


class error(Exception):
    pass
