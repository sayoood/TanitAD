"""mutate_navrbd_chain.py -- mutation arms for test_navrbd_chain.py: each arm edits ONE line of a source file in place, runs the whole test file, requires it to go RED with the named test failing, and restores the file
(byte-exact, in a finally block).  The unmutated run must be green first.  Usage: python mutate_navrbd_chain.py   (env TMP / NAVRBD_TREE as for the tests)"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ARMS = [  # (file, old, new, test that must fail)
    ("navrbd_t0.py", '"donor": ["--nav-shuffle", "--nav-rb-variant", "zero"]', '"donor": ["--nav-shuffle", "--nav-rb-variant", "follow"]', "test_t0_donor_holds_the_channel_at_zero_beside_the_donor_nav"),
    ("navrbd_t0.py", '"--nav-rb-feat", "exitdist"', '"--nav-rb-feat", "state"', "test_t0_real_variant_flags"),
    ("navrbd_t0.py", '"zero": ["--nav-withheld"]', '"zero": ["--nav-rb-variant", "zero"]', "test_t0_zero_is_nav_withheld"),
    ("navrbd_view.py", 'f"t0/eval_STRAT1-N1_s{s}_{v}.npz"', 'f"t0/eval_STRAT1-S1_s{s}_{v}.npz"', "test_view_presents_n1d_as_n1_and_keeps_c0"),
    ("navrbd_strategic.py", 'return "N1D-N1R"', 'return "N1D-N1"', "test_contrast_key_is_the_one_the_read_expects"),
    ("navrbd_strategic.py", 'f"NAVRB2-N1R_s{s}"', 'f"NAVRB2-S1R_s{s}"', "test_strategic_compares_n1d_with_the_navrb2_n1r_pass"),
    ("navrbd_pod.py", 'm = ALL_END.match(last_line(text))', 'm = next(filter(None, map(ALL_END.match, text.splitlines())), None)', "test_all_end_is_the_only_end"),
    ("navrbd_pod.py", 'bad.append(f"sha256 differs {rel}")', 'pass', "test_sums_and_verify"),
    ("navrbd_export.py", 'elif blob_sha1(q) != sha:', 'elif False:', "test_export_blob_check_catches_crlf_and_long_paths"),
    ("navrbd_chain.sh", 'navrbd_pod.py wait-end', 'navrbd_pod.py wait-done', "test_chain_waits_for_the_final_pod_line_and_the_um_sentinel"),
    ("navrbd_chain.sh", 'step pull_ckpts "pull N1D_s0', 'step pull_ckpt "pull N1D_s0', "test_chain_order_gate_before_any_pull_and_read_last"),
]


def run_tests():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", os.environ.get("TMP", ".") + "/bt_mut", "test_navrbd_chain.py"], cwd=HERE, capture_output=True, encoding="utf-8")
    return r.returncode, r.stdout


def main():
    rc, out = run_tests()
    print("baseline:", "GREEN" if rc == 0 else "RED", out.strip().splitlines()[-1])
    if rc:
        sys.exit(2)
    res = []
    for f, old, new, test in ARMS:
        p = f"{HERE}/{f}"
        raw = open(p, "rb").read()
        txt = raw.decode("utf-8")
        assert txt.count(old) >= 1, ("mutation anchor not found", f, old)
        try:
            open(p, "wb").write(txt.replace(old, new, 1).encode("utf-8"))
            rc, out = run_tests()
        finally:
            open(p, "wb").write(raw)
        red = rc != 0 and (f"FAILED test_navrbd_chain.py::{test}" in out)
        res.append({"file": f, "mutation": old[:60], "test": test, "red": red})
        print("RED " if red else "NOT-RED", f, test, flush=True)
    assert run_tests()[0] == 0, "files not restored"
    json.dump(res, open(HERE + "/out/MUTATIONS_NAVRBD_CHAIN.json", "w"), indent=1)
    sys.exit(0 if all(x["red"] for x in res) else 1)


if __name__ == "__main__":
    main()
