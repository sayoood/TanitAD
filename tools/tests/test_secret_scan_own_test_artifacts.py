"""The scanner's own test artifacts are not credential files (MEASURED 2026-09-27: `*secret*` blocked three
banked pytest logs of test_secret_scan.py and failed the refcv7 launch gate's G-SUITE-PINNED). The exemption
is by NAME and narrow; every expectation below is a literal."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import secret_scan as ss  # noqa: E402


def _blocking(path):
    return [f for f in ss.scan_path_shape(path) if f.blocking]


def test_the_three_banked_logs_of_the_scanner_test_do_not_block():
    for p in ("TanitAD Research Lab/x/raw/pinned_rehearsal_3cca805/logs/pinned/stack__tests__test_secret_scan.py.log",
              "TanitAD Research Lab/x/raw/pinned_rehearsal_3cca805_r2/logs/pinned/stack__tests__test_secret_scan.py.log",
              "TanitAD Research Lab/x/raw/rerun_secret_scan_3cca805/tests__test_secret_scan.py.log"):
        assert _blocking(p) == [], p


def test_RED_any_other_secret_named_log_still_blocks():
    for p in ("raw/my_secret_keys.log", "raw/secrets.log", "raw/aws_credentials.log", "raw/test_secret.log"):
        assert len(_blocking(p)) == 1, p


def test_RED_exact_credential_names_still_block_even_with_the_stem():
    # the exact-name tier runs BEFORE the substring tier and is untouched by the exemption
    assert len(_blocking("raw/.env")) == 1
    assert len(_blocking("raw/id_ed25519")) == 1


def test_RED_mutation_without_the_exemption_the_logs_block_again(monkeypatch):
    monkeypatch.setattr(ss, "_is_own_test_artifact", lambda base: False)
    p = "TanitAD Research Lab/x/raw/rerun_secret_scan_3cca805/tests__test_secret_scan.py.log"
    assert len(_blocking(p)) == 1
