"""Tests of dscale_banker.py with a fake hub: a mismatch must never write bank_ok, and nothing is ever deleted. Run: python -m unittest test_dscale_banker -v"""
import hashlib, os, tempfile, unittest
from pathlib import Path
import dscale_banker as B


def mk(root, arm="B_s0", stage="A", payload=b"ckpt-bytes" * 1000, announce=True, sha=None):
    r = Path(root) / "runs" / arm / stage
    r.mkdir(parents=True, exist_ok=True)
    (r / "ckpt_last.pt").write_bytes(payload)
    (r / "argv.json").write_text("{}")
    s = hashlib.sha256(payload).hexdigest()
    (r / f"ARM_{stage}_DONE").write_text(s + " t\n")
    rd = Path(root) / "bank_ready"
    rd.mkdir(exist_ok=True)
    (rd / f"{arm}.{stage}.txt").write_text(f"{r / 'ckpt_last.pt'} {sha or s} 2026-11-17T00:00:00Z\n")
    return s


class T(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root, self.hubdir = Path(self.t.name) / "d", Path(self.t.name) / "hub"
        self.root.mkdir()
        self.hub = B.LocalHub(self.hubdir)
        self.b = B.Banker(self.root, self.hub, log=lambda *_: None, sleep=lambda *_: None)

    def tearDown(self):
        self.t.cleanup()

    def marker(self, tag="B_s0.A"):
        return self.root / "bank_ok" / f"{tag}.sha256"

    def test_banks_and_is_idempotent_and_deletes_nothing(self):
        s = mk(self.root)
        self.assertEqual(self.b.run(), {"B_s0.A": "BANKED"})
        self.assertEqual(self.marker().read_text().split()[0], s)
        self.assertTrue((self.hubdir / "B_s0/A/ckpt_last.pt").is_file())
        n = self.hub.n_commits
        self.assertEqual(self.b.run(), {"B_s0.A": "ALREADY"})
        self.assertEqual(self.hub.n_commits, n)                              # no second upload
        self.assertTrue((self.root / "runs/B_s0/A/ckpt_last.pt").is_file())  # nothing deleted

    def test_corrupt_upload_never_writes_marker(self):
        mk(self.root)
        self.hub.corrupt_next = 99
        r = self.b.run()
        self.assertTrue(r["B_s0.A"].startswith("REFUSED"), r)
        self.assertFalse(self.marker().exists())

    def test_one_corrupt_attempt_then_retry_banks(self):
        mk(self.root)
        self.hub.corrupt_next = 1
        self.assertEqual(self.b.run(), {"B_s0.A": "BANKED"})

    def test_lfs_missing_or_wrong_refuses(self):
        for mode in ("none", "wrong"):
            self.hub.lfs_mode = mode
            mk(self.root)
            self.assertTrue(self.b.run()["B_s0.A"].startswith("REFUSED"), mode)
            self.assertFalse(self.marker().exists(), mode)

    def test_upload_failure_refuses(self):
        mk(self.root)
        self.hub.fail_next = 99
        self.assertTrue(self.b.run()["B_s0.A"].startswith("REFUSED"))
        self.assertFalse(self.marker().exists())

    def test_record_sha_mismatch_refuses_without_upload(self):
        mk(self.root, sha="0" * 64)
        self.assertTrue(self.b.run()["B_s0.A"].startswith("REFUSED"))
        self.assertEqual(self.hub.n_commits, 0)
        self.assertFalse(self.marker().exists())

    def test_wrong_path_refuses(self):
        mk(self.root)
        (self.root / "bank_ready/B_s0.A.txt").write_text(f"{self.root}/runs/B_s0/B/ckpt_last.pt {'a' * 64} t\n")
        self.assertTrue(self.b.run()["B_s0.A"].startswith("REFUSED"))

    def test_done_marker_disagreement_refuses(self):
        mk(self.root)
        (self.root / "runs/B_s0/A/ARM_A_DONE").write_text("b" * 64 + " t\n")
        self.assertTrue(self.b.run()["B_s0.A"].startswith("REFUSED"))

    def test_existing_marker_for_other_bytes_is_a_conflict_not_overwritten(self):
        mk(self.root)
        self.marker().parent.mkdir(exist_ok=True)
        self.marker().write_text("c" * 64 + " old\n")
        self.assertIn("CONFLICT", self.b.run()["B_s0.A"])
        self.assertEqual(self.marker().read_text().split()[0], "c" * 64)

    def test_public_repo_refused(self):
        mk(self.root)
        self.hub.private = False
        with self.assertRaises(B.HubError):
            self.b.run()
        self.assertFalse(self.marker().exists())

    def test_marker_is_what_chainlib_reads_and_sidecars_ride_along(self):
        mk(self.root, "U_s0", "B")
        self.b.run()
        self.assertTrue((self.hubdir / "U_s0/B/argv.json").is_file())
        self.assertEqual(len(self.marker("U_s0.B").read_text().split()), 6)


if __name__ == "__main__":
    unittest.main()
