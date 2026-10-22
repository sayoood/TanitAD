"""P7-WIRE-INTEGRATE-2 / Gap 2 probe (needs torch: run with the world-modeling venv).
Is the digest the GATE computes (gate_checks.bank_targets_sha256) the SAME value the TRAINER records in its identity (train.run_identity(...)["targets_sha256"]) and compares on every --resume
(train._identity_diff)?  Calls the real trainer functions, not a copy.
usage: python probe_digest_vs_trainer.py <refe dir>"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REFE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REFE))
import gate_checks as C  # noqa: E402
import session_reserve as SR  # noqa: E402
import train as T  # noqa: E402


class DS:
    def __init__(self, rows):
        self.rows, self.scene_groups = rows, {(r["log_name"], r["token"], r["step"]): [r] for r in rows}

    def __len__(self):
        return len(self.rows)


with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "train_grow_cmd_3c_p7"
    bank.mkdir()
    rows0 = [{"log_name": f"L{i}", "token": f"t{i}", "step": 0, "rank": 0} for i in range(5)]
    rows1 = [{"log_name": "L1", "token": "t1", "step": 0, "rank": 1}]
    (bank / "targets_rank0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows0), encoding="utf-8", newline="\n")
    (bank / "targets_rank1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows1), encoding="utf-8", newline="\n")
    (bank / "reserve_filter_report.json").write_text("{}", encoding="utf-8")                     # a non-rank file: neither definition may hash it
    a = SimpleNamespace(targets=str(bank), grow=False, calib=None, field_labels=None, aux_drv_labels=None)
    ds = DS(rows0 + rows1)
    ident = T.run_identity(a, ds, SimpleNamespace(files=[]), "frozen")
    files = sorted(bank.glob("targets_rank*.jsonl"))
    vals = {"train.run_identity()['targets_sha256']": ident["targets_sha256"], "train._digest_files(sorted(glob))": T._digest_files(files),
            "gate_checks.bank_targets_sha256(dir)": C.bank_targets_sha256(bank), "gate_checks._digest_files(files)": C._digest_files(files), "session_reserve._digest_files(files)": SR._digest_files(files)}
    for k, v in vals.items():
        print(f"{k:46s} {v}")
    print("ALL EQUAL:", len(set(vals.values())) == 1)
    print("identity n_tuples / n_scenes:", ident["n_tuples"], ident["n_scenes"])
    # the same call under --grow records None (so a --grow launch could not be pinned: it is FORBIDDEN in refe_p7)
    a.grow = True
    print("grow identity targets_sha256:", T.run_identity(a, ds, SimpleNamespace(files=[]), "frozen")["targets_sha256"])
    # a one-byte change at the same size changes both
    p = bank / "targets_rank0.jsonl"
    b = p.read_bytes()
    p.write_bytes(b.replace(b'"L1"', b'"L9"', 1))
    a.grow = False
    print("after a same-size edit: trainer", T.run_identity(a, ds, SimpleNamespace(files=[]), "frozen")["targets_sha256"][:16], "gate", C.bank_targets_sha256(bank)[:16], "changed:", C.bank_targets_sha256(bank) != vals["gate_checks.bank_targets_sha256(dir)"])
    # and the trainer's own resume check refuses it
    old = T.run_identity(SimpleNamespace(targets=str(bank), grow=False, calib=None, field_labels=None, aux_drv_labels=None), ds, SimpleNamespace(files=[]), "frozen")
    old["targets_sha256"] = vals["train.run_identity()['targets_sha256']"]
    new = T.run_identity(SimpleNamespace(targets=str(bank), grow=False, calib=None, field_labels=None, aux_drv_labels=None), ds, SimpleNamespace(files=[]), "frozen")
    print("trainer _identity_diff on the edited bank:", [d for d in T._identity_diff(old, new) if "targets_sha256" in d])
