"""OBD seed-3 launch-gate checker (AMEND_OBD_SEED3.md). Runs on box 2 (venv python) and on the dev box.

It is the OBD checker of record (theory/obd/box2/obd_check.py, sha256-pinned below, byte-identical on both sides) with FOUR things changed from outside, none by editing it:
  * the tag table is the four seed-3 tags (OBD-CTL_s10, OBD-BAND_s10, OBD-CTL_s11, OBD-BAND_s11); the frozen argv file is frozen/argvs_obd_seed3.json (sha256 pinned below), derived by make_argvs_seed3.py
    from the banked argvs token for token, only --seed and --out changed;
  * run folders and gate outputs live under D:\\tanitad\\obd3\\ (own state); the code tree D:\\tanitad\\obd\\repo and its tree_manifest.json are the OBD chain's and are READ ONLY here;
  * no pilot, no NULL arms, no 12 h rule, no NULL parity (the commands project / permit / parity are refused);
  * a new command `shipcheck`: proves on box 2 that the tree at D:\\tanitad\\obd\\repo is the code of record (TIP_OBD.txt, every manifest member re-hashed, the trainer's needle strings).
Everything else (pin, selfcheck gate with BUILT-lever assertions, per-run token, post check, cleanup, render, verify-token) is the base checker's code, unchanged.

  argv --tag T [--out-txt F]     dev box: the frozen argv of tag T, one line
  render --template F --out F --set K=V
  preflight | stage --tag T --gid G | cleanup --tag T --gid G | verify-token --token F --argv-file F --tag T --commit C   (base checker)
  shipcheck --gid G              box 2: read-only proof of the code tree -> obd3\\ship\\SHIPCHECK_RESULT.json and SHIPCHECK_<G>.done / .fail
  base-dir                       the folder the pinned base files were found in
Test hooks (dev box only, with OBD_ROOT = a fake box folder): the base checker's (OBD_ROOT, OBD_FAKE_*, OBD_POLL_S, OBD_FLOOR_BYTES, OBD_CKPT_LOADER, OBD_NVSMI) and OBD3_CODE_COMMIT (replaces the pinned commit)."""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- pins ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CODE_COMMIT = "cdbff971fa6dad4d20b937fe8abda6cb9f622adb"          # the code of record; the tree the OBD chain shipped to box 2
ARGVS_SHA256 = "379c32def43e960139078f7406c5646983b853a5828fd4ba1f7ac80f98c634a5"        # frozen/argvs_obd_seed3.json (make_argvs_seed3.py)
BASE_SHA256 = {                                                   # the OBD chain's files, read from theory/obd/box2/ (never edited); also shipped to box 2 unchanged
    "obd_check.py": "dd53b73460f68d0742bc8d1c1f072b6db2a243cfaf7335182db7b29aeb80a83d",
    "obd_refresh.py": "e8cd05d6b2de62e0c2b357d081124f35ab3aa7b29e8dd84ebd46fcc81c34252c",
    "obd_ship_tree.py": "c8a9e539425ff0c5a005ca38830d648fa3108e50b64a45b4f055a3b484af22a4",
}
TABLE = {
    "OBD-CTL_s10":  {"kind": "ctl",  "seed": 10, "init": 0, "order": 1, "blocked": None},
    "OBD-BAND_s10": {"kind": "band", "seed": 10, "init": 0, "order": 2, "blocked": None},
    "OBD-CTL_s11":  {"kind": "ctl",  "seed": 11, "init": 1, "order": 3, "blocked": None},
    "OBD-BAND_s11": {"kind": "band", "seed": 11, "init": 1, "order": 4, "blocked": None},
}
RUN_ORDER = sorted(TABLE, key=lambda t: TABLE[t]["order"])
HOME3_DECL = r"D:\tanitad\obd3"
REFUSED_CMDS = ("project", "permit", "parity")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def find_base():
    """-> the folder holding the pinned base files: OBD3_BASE, else next to this file (box 2), else theory/obd/box2 (dev box). Every base file found there must carry its pinned sha256."""
    cands = [os.environ.get("OBD3_BASE") or "", HERE, os.path.normpath(os.path.join(HERE, "..", "..", "box2"))]
    for d in cands:
        if d and os.path.isfile(os.path.join(d, "obd_check.py")):
            for name, want in BASE_SHA256.items():
                p = os.path.join(d, name)
                if os.path.exists(p) and sha_file(p) != want:
                    raise SystemExit(f"REFUSED: {p} sha256 {sha_file(p)} != the pinned {want} (the base file changed)")
            return d
    raise SystemExit("REFUSED: the pinned base obd_check.py is not next to the checker, not in OBD3_BASE and not in theory/obd/box2")


BASE_DIR = find_base()
sys.path.insert(0, BASE_DIR)
import obd_check as B  # noqa: E402

if os.environ.get("OBD_TEST_ARGVS") and not os.environ.get("OBD_ROOT"):
    raise SystemExit("REFUSED: OBD_TEST_ARGVS (a test hook) is set outside a fake box")

# ---- the changes, applied to the imported module's globals (the base functions read them at call time) ---------------------------------------------------------------------------------
B.ARGVS_FILE = next((p for p in (os.path.join(HERE, "argvs_obd_seed3.json"), os.path.join(HERE, "frozen", "argvs_obd_seed3.json")) if os.path.exists(p)), os.path.join(HERE, "frozen", "argvs_obd_seed3.json"))
B.ARGVS_SHA256 = ARGVS_SHA256
B.TABLE = TABLE
B.RUN_ORDER = list(RUN_ORDER)
B.GATE_AT_PILOT = []
B.PILOT_TAG, B.PILOT_BASE = "PILOT-NOT-IN-THIS-PACKAGE", RUN_ORDER[0]


def run_dir(tag):
    return os.path.join(B.loc(HOME3_DECL), "runs", tag)


def gate_out_decl(tag):
    return HOME3_DECL + "\\gate_out\\" + tag


def data_paths():
    t = B.tokens(RUN_ORDER[1])
    g = lambda f: t[t.index(f) + 1]                                  # noqa: E731
    return {"v9_labels_train": g("--v9-train"), "lead_train": g("--lead-train"), "speed_train": g("--speed-train"), "join_train": g("--join-train"), "p10_prior": g("--p10"), "native_tracks": g("--tracks")}


B.run_dir, B.gate_out_decl, B.data_paths = run_dir, gate_out_decl, data_paths


def code_commit():
    return os.environ.get("OBD3_CODE_COMMIT") if B.fake() and os.environ.get("OBD3_CODE_COMMIT") else CODE_COMMIT


def cmd_shipcheck(a):
    """READ-ONLY proof that the OBD tree on this box is the code of record. Writes only under obd3\\ship."""
    gid, c = a.get("gid"), B.commit()
    sd = B.loc(HOME3_DECL + "\\ship")
    os.makedirs(sd, exist_ok=True)
    p = []
    if c != code_commit():
        p.append(f"OBD_COMMIT {c!r} is not the code of record {code_commit()}")
    p += B.tree_problems()
    members, man_sha = 0, None
    try:
        mp = os.path.join(B.home(), "tree_manifest.json")
        man_sha, members = sha_file(mp), len(json.load(open(mp, encoding="utf-8")).get("members") or {})
    except (OSError, ValueError):
        pass
    res = {"schema": "obd3-shipcheck-v1", "at": B.now(), "gid": gid, "commit": c, "code_of_record": code_commit(), "ok": not p, "problems": p, "n_members": members, "tree_manifest_sha256": man_sha,
           "repo": B.repo()}
    B.write_text(os.path.join(sd, "SHIPCHECK_RESULT.json"), json.dumps(res, indent=1))
    if p:
        B.write_text(os.path.join(sd, f"SHIPCHECK_{gid}.fail"), " | ".join(p)[:800] + "\n")
        print("OBD3-SHIPCHECK-NOT-PROVEN " + " | ".join(p)[:600])
        return 3
    B.write_text(os.path.join(sd, f"SHIPCHECK_{gid}.done"), "done\n")
    print(f"OBD3-SHIPCHECK-OK {c[:12]} members {members}")
    return 0


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, a = argv[0], B.parse(argv[1:])
    if cmd in REFUSED_CMDS:
        print(f"refused: `{cmd}` does not exist in the seed-3 package (no pilot, no 12 h rule, no NULL parity)")
        return 2
    if cmd == "base-dir":
        print(BASE_DIR)
        return 0
    if cmd == "shipcheck":
        return cmd_shipcheck(a)
    if a.get("pilot"):
        print("refused: no pilot in the seed-3 package")
        return 2
    if cmd in ("argv", "stage", "cleanup", "verify-token") and a.get("tag") not in TABLE:
        print(f"unknown tag {a.get('tag')!r} (this package knows {', '.join(RUN_ORDER)})")
        return 2
    if cmd in ("stage", "preflight") and B.commit() != code_commit():
        print(f"refused: OBD_COMMIT {B.commit()!r} is not the code of record {code_commit()}")
        return 3
    return B.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
