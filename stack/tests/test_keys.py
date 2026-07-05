"""Secret loader: parsing + env-var mapping, using a synthetic keys file.

Never touches the real Keys.txt — uses fabricated tokens in a temp file, so the
test is deterministic and secret-free.
"""

import os

from tanitad import keys as K


_FAKE = """\
Openrouter key
sk-or-v1-deadbeef00
huggingface key:
hf_FAKEtoken123
Google AI Studio API key
AIzaFAKE_key-123
"""


def test_load_keys_maps_all_three(tmp_path, monkeypatch):
    for name in ("OPENROUTER_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                 "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    f = tmp_path / "Keys.txt"
    f.write_text(_FAKE, encoding="utf-8")

    set_vars = K.load_keys(f)

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-v1-deadbeef00"
    assert os.environ["HF_TOKEN"] == "hf_FAKEtoken123"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_FAKEtoken123"
    assert os.environ["GOOGLE_API_KEY"] == "AIzaFAKE_key-123"
    assert os.environ["GEMINI_API_KEY"] == "AIzaFAKE_key-123"
    assert set(set_vars) == {"OPENROUTER_API_KEY", "HF_TOKEN",
                             "HUGGING_FACE_HUB_TOKEN", "GOOGLE_API_KEY",
                             "GEMINI_API_KEY"}


def test_load_keys_missing_file_is_noop(tmp_path):
    assert K.load_keys(tmp_path / "nope.txt") == []


def test_load_keys_does_not_overwrite_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_preexisting")
    f = tmp_path / "Keys.txt"
    f.write_text(_FAKE, encoding="utf-8")
    K.load_keys(f)                       # default: don't clobber
    assert os.environ["HF_TOKEN"] == "hf_preexisting"
    K.load_keys(f, overwrite=True)       # explicit override
    assert os.environ["HF_TOKEN"] == "hf_FAKEtoken123"


def test_keys_path_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TANITAD_KEYS_FILE", str(tmp_path / "custom.txt"))
    assert K.keys_path() == tmp_path / "custom.txt"


def test_enable_tls_returns_bool():
    assert isinstance(K.enable_tls(), bool)
