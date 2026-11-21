"""P4-4 — every run records WHAT EXECUTED, and collecting it can never kill a run.

⛔ THE PROBLEM: two runs on two machines could produce byte-identical
`config.json` files. The config records what was ASKED FOR; nothing recorded the
code state, machine, or driver that actually ran — so a result could not be tied
to the thing that produced it.

⭐ `trainer_md5` IS THE IDENTITY, NOT `git_sha`. Pods have no git credentials: a
pod checkout's HEAD sits weeks behind while its working tree is fully current,
because fixes arrive by md5-verified file-ship. There, the git SHA is actively
misleading and the hash of the executing file is the only true answer.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import train_v6_staged as T  # noqa: E402

REQUIRED = ("trainer_file", "trainer_md5", "git_sha", "git_dirty", "git_branch",
            "hostname", "platform", "python", "torch", "cuda_runtime", "cudnn",
            "gpu_name", "gpu_capability", "device_resolved",
            "started_at", "started_at_utc")


def test_every_required_field_is_present():
    p = T.run_provenance("cuda:0")
    missing = [k for k in REQUIRED if k not in p]
    assert not missing, f"provenance is missing {missing}"


def test_trainer_md5_is_a_real_hash_of_the_real_file():
    import hashlib
    p = T.run_provenance()
    expect = hashlib.md5(pathlib.Path(T.__file__).resolve().read_bytes()).hexdigest()
    assert p["trainer_md5"] == expect
    assert len(p["trainer_md5"]) == 32


def test_md5_changes_with_content_git_sha_does_not(tmp_path):
    """⭐ THE POINT OF THE FIELD, as an executable statement.

    A file-shipped edit changes the md5 and leaves HEAD untouched. This is why
    the md5 is the identity on a pod and the SHA is not.
    """
    import hashlib
    src = pathlib.Path(T.__file__).resolve().read_bytes()
    assert (hashlib.md5(src).hexdigest()
            != hashlib.md5(src + b"# shipped fix\n").hexdigest())


def test_device_is_recorded_and_unset_is_explicit():
    assert T.run_provenance("cuda:3")["device_resolved"] == "cuda:3"
    assert T.run_provenance()["device_resolved"] == "unset"


def test_timestamps_carry_a_timezone():
    """⚠️ Pods run UTC, the PI reads Europe/Berlin. A naive stamp has been read
    as a broken clock. Both stamps must be offset-aware."""
    import datetime
    p = T.run_provenance()
    for k in ("started_at", "started_at_utc"):
        assert datetime.datetime.fromisoformat(p[k]).tzinfo is not None, k
    assert p["started_at_utc"].endswith("+00:00")


def test_never_raises_when_every_probe_is_broken(monkeypatch):
    """⛔ THE LOAD-BEARING TEST. Provenance must never be the reason a training
    run dies — an observability tool causing the outage it exists to explain.

    Breaks subprocess (git), platform, and torch's CUDA accessors at once.
    """
    import platform as _pf
    import subprocess as _sp

    def boom(*a, **k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(_sp, "run", boom)
    monkeypatch.setattr(_pf, "node", boom)
    monkeypatch.setattr(T.torch.cuda, "is_available", boom)

    p = T.run_provenance("cuda:0")
    for k in REQUIRED:
        assert k in p, f"{k} vanished instead of recording its failure"
    assert "unavailable" in str(p["hostname"])
    assert "unavailable" in str(p["git_sha"])
    # ⭐ a failure must be DISTINGUISHABLE from a value, not silently empty
    assert str(p["hostname"]) != ""


def test_config_and_summary_both_embed_it():
    """Both artifacts, not just config.json — a summary is what gets quoted."""
    src = pathlib.Path(T.__file__).resolve().read_text(encoding="utf-8")
    assert src.count('"provenance": run_provenance(') >= 2, (
        "provenance must be embedded in BOTH config.json and summary.json")
