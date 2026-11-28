"""A4: the reader takes K4's comparator from its OWN manifest key (`t0_fresh`, the arm's T0 eval at --ages fresh) and R-T0 / T0-CKPT / the
families from `t0` (box 2's standard eval, --ages cycle), with two new controls (T0-AGES, T0F-CKPT), 4 tests and 5 mutants; banks the landed
PREREG text after Amendment 4. python apply_a4.py <tip>   (run in C:/Users/Admin/orcread; originals kept as *.pre_a4)"""
import io
import shutil
import subprocess
import sys

H = "C:/Users/Admin/orcread/"
TIP = sys.argv[1]
pre = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "show", f"{TIP}:Project Steering/PREREG_V8F_ORECOVER.md"],
                     capture_output=True, check=True).stdout
a3 = open(H + "PREREG_V8F_ORECOVER.A3.md", "rb").read()
assert pre.startswith(a3) and b"## Amendment 4" in pre[len(a3):] and b"## Amendment 4" not in a3, "the landed PREREG is not A3 + Amendment 4"
open(H + "PREREG_V8F_ORECOVER.A4.md", "wb").write(pre)
for f in ("orc_read.py", "test_orc_read.py", "mutate_orc.py"):
    b = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "show",
                        f"{TIP}:TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-orcread/{f}"],
                       capture_output=True, check=True).stdout
    assert b == open(H + f, "rb").read(), f"{f}: the working copy is not the tip's"


def rep(name, pairs):
    p = H + name
    shutil.copyfile(p, p + ".pre_a4")
    s = io.open(p, encoding="utf-8", newline="").read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:90])
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8", newline="").write(s)


rep("orc_read.py", [
    ('+ Amendment 3 (A3-2026-10-26)"', '+ Amendment 3 (A3-2026-10-26) + Amendment 4 (A4-2026-10-26)"'),
    ('''            ctl.add(f"T0-CKPT:{tag}", t0_ok, f"T0 eval json ckpt {rec['t0json'].get('ckpt')!r} vs .../{ckpt_tail}")
''',
     '''            ctl.add(f"T0-CKPT:{tag}", t0_ok, f"T0 eval json ckpt {rec['t0json'].get('ckpt')!r} vs .../{ckpt_tail}")
            t0f = e.get("t0_fresh")                                 # A4: K4's comparator is the arm's own T0 eval at --ages fresh, from its OWN key
            if not t0f:
                raise Refusal(f"{tag}: manifest has no 't0_fresh' (A4: K4 compares tick 0 with the arm's own T0 eval at --ages fresh)")
            rec["t0fjson"] = jload(need(t0f["json"], f"{tag} fresh T0 eval json"))
            ctl.add(f"T0F-CKPT:{tag}", tail2(str(rec["t0fjson"].get("ckpt", ""))) == ckpt_tail,
                    f"fresh T0 eval json ckpt {rec['t0fjson'].get('ckpt')!r} vs .../{ckpt_tail}")
            ages = (rec["t0json"].get("ages"), rec["t0fjson"].get("ages"))     # A4: R-T0 can never read the fresh eval, K4 never the standard one
            ctl.add(f"T0-AGES:{tag}", ages == ("cycle", "fresh"),
                    {"t0": ages[0], "t0_fresh": ages[1], "committed": ["cycle: box 2's standard eval (R-T0, T0-CKPT, the families)", "fresh: K4 only (A4)"]})
'''),
    ('ok, det = backend.k4(lg, pk, rec["entry"]["t0"], s)', 'ok, det = backend.k4(lg, pk, rec["entry"]["t0_fresh"], s)'),
    ('[t1_logged], t0{json,npz}")', '[t1_logged], t0{json,npz} (box 2\'s standard eval), t0_fresh{json,npz} (A4: the --ages fresh eval K4 reads)")'),
])
rep("test_orc_read.py", [
    ('PREREG_A3 = HERE / "PREREG_V8F_ORECOVER.A3.md"           # the landed text after Amendment 3: the A2 text plus the amendment, appended',
     'PREREG_A3 = HERE / "PREREG_V8F_ORECOVER.A3.md"           # the landed text after Amendment 3: the A2 text plus the amendment, appended\n'
     'PREREG_A4 = HERE / "PREREG_V8F_ORECOVER.A4.md"           # the landed text after Amendment 4: the A3 text plus the amendment, appended'),
    ('''        e["t0"] = self.write_t0(arm, s, W, tag)
        return e''',
     '''        e["t0"] = self.write_t0(arm, s, W, tag)
        e["t0_fresh"] = self.write_t0(arm, s, W, tag, ages="fresh")
        return e'''),
    ('''    def write_t0(self, arm, s, W, tag):
        root = self.root
        ade = (W["t0_1"][W["order"]] if arm == "ORC" else W["t0_0"])
        jdump(root / "t0" / f"eval_{tag}.json", {"ckpt": f"{root}/runs/{tag}/ckpt_last.pt", "raw": {}})
        np.savez(root / "t0" / f"eval_{tag}.npz", ade6=ade, eid=W["ci"], ckpt=np.zeros((len(ade), 3, 3)), lat_logits=np.zeros((len(ade), 5)), lon_logits=np.zeros((len(ade), 7)))
        return {"json": str(root / "t0" / f"eval_{tag}.json"), "npz": str(root / "t0" / f"eval_{tag}.npz")}''',
     '''    def write_t0(self, arm, s, W, tag, ages="cycle"):
        root = self.root
        ade = (W["t0_1"][W["order"]] if arm == "ORC" else W["t0_0"])
        nm = f"eval_{tag}" + ("-fresh" if ages == "fresh" else "")
        if ages == "fresh":                     # A4: a different eval. ORC's is 7 m worse, so a reader that took R-T0 from it would read a +7 m point
            ade = ade + (7.0 if arm == "ORC" else 0.0)
        jdump(root / "t0" / f"{nm}.json", {"ckpt": f"{root}/runs/{tag}/ckpt_last.pt", "ages": ages, "raw": {}})
        np.savez(root / "t0" / f"{nm}.npz", ade6=ade, eid=W["ci"], ckpt=np.zeros((len(ade), 3, 3)), lat_logits=np.zeros((len(ade), 5)), lon_logits=np.zeros((len(ade), 7)))
        return {"json": str(root / "t0" / f"{nm}.json"), "npz": str(root / "t0" / f"{nm}.npz")}'''),
    ('''    def k4(self, entry_logged, pack, t0_entry, seed):
        tag = pack.meta["tag"].replace("_causal", "")
        return (not ({("K4", tag)} & self.world.ctl_fail)), {"source": "synthetic"}''',
     '''    def k4(self, entry_logged, pack, t0_entry, seed):
        tag = pack.meta["tag"].replace("_causal", "")
        fresh = str(t0_entry["json"]).endswith("-fresh.json")          # A4: the synthetic K4 holds only against the arm's fresh-age eval
        return (fresh and not ({("K4", tag)} & self.world.ctl_fail)), {"source": "synthetic", "t0_json": str(t0_entry["json"])}'''),
    ('''def test_a2_table_is_complete_no_reachable_UNPLACED():''',
     '''def test_a4_text_extends_a3_and_carries_the_k4_rule():
    a3, a4 = PREREG_A3.read_bytes(), PREREG_A4.read_bytes()
    assert a4.startswith(a3) and b"## Amendment 4" in a4[len(a3):] and b"## Amendment 4" not in a3
    t = a4[len(a3):].decode("utf-8").replace("\\n", " ")
    assert "`v8f_mini_eval.py run --ages fresh`" in t and "R-T0, T0-CKPT and the four families stay on box 2's standard eval" in t
    assert "separate manifest keys" in t and "Amendment 4 (A4-2026-10-26)" in orc.PREREG_ID


def test_a4_k4_reads_t0_fresh_and_r_t0_reads_the_standard_eval(tmp_path, capsys):
    """A4: K4's comparator is the fresh-age eval, from its own key; R-T0 stays on box 2's standard eval. The fixture's fresh eval is 7 m worse for ORC, so a
    reader that took R-T0 from it would read a +7 m point; the synthetic K4 fails unless it is handed the -fresh file."""
    rc, res, *_ = run(tmp_path, **STRONG)
    assert rc == 0, capsys.readouterr().err
    for tag in ("ORC_s0", "ORC0_s0", "ORC_s1", "ORC0_s1"):
        assert all(res["controls_PASS"][f"{k}:{tag}"] for k in ("K4", "T0-AGES", "T0F-CKPT")), tag
        assert res["controls"][f"K4:{tag}"]["detail"]["t0_json"].endswith(f"eval_{tag}-fresh.json")
    for s in ("s0", "s1"):
        assert abs(res["rows"][s]["R-T0"]["point"]) < 1.0, res["rows"][s]["R-T0"]
    assert res["OUTCOME"]["OUTCOME"] == "ROBUST"


def test_a4_refuses_a_manifest_without_t0_fresh(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path, edit=lambda fx: fx.manifest["arms"]["ORC0"]["1"].pop("t0_fresh"), **STRONG)
    refused(capsys, rc, out, "ORC0_s1: manifest has no 't0_fresh'")


def test_a4_t0_ages_and_the_fresh_ckpt_are_controls(tmp_path, capsys):
    edits = chain(
        lambda fx: jedit(fx.root / "t0" / "eval_ORC_s0.json", lambda o: o.update(ages="fresh")),          # R-T0 would read a fresh-age eval
        lambda fx: jedit(fx.root / "t0" / "eval_ORC0_s1-fresh.json", lambda o: o.update(ages="cycle")),   # K4 would read a cycle-age eval
        lambda fx: jedit(fx.root / "t0" / "eval_ORC_s1-fresh.json", lambda o: o.update(ckpt="/x/runs/ORC0_s1/ckpt_last.pt")),
    )
    rc, _, out, _ = run(tmp_path, edit=edits, **STRONG)
    refused(capsys, rc, out, "T0-AGES:ORC_s0", "T0-AGES:ORC0_s1", "T0F-CKPT:ORC_s1",
            absent=("T0-AGES:ORC0_s0", "T0-AGES:ORC_s1", "T0F-CKPT:ORC_s0", "T0F-CKPT:ORC0_s1", "T0-CKPT:ORC_s1"))


def test_a2_table_is_complete_no_reachable_UNPLACED():'''),
])
rep("mutate_orc.py", [
    ('''MUTANTS = [
''',
     '''MUTANTS = [
    ("M26", "A4: K4 reads the standard T0 eval", sub('backend.k4(lg, pk, rec["entry"]["t0_fresh"], s)', 'backend.k4(lg, pk, rec["entry"]["t0"], s)'), "a4"),
    ("M27", "A4: the T0-AGES control is blind", sub('ages == ("cycle", "fresh"),', 'True,'), "a4"),
    ("M28", "A4: the fresh eval's checkpoint is not checked", sub('tail2(str(rec["t0fjson"].get("ckpt", ""))) == ckpt_tail,', 'True,'), "a4"),
    ("M29", "A4: a manifest without t0_fresh falls back to t0", sub("""            if not t0f:
                raise Refusal(f"{tag}: manifest has no 't0_fresh' (A4""", """            if not t0f:
                e["t0_fresh"] = t0f = e["t0"]
            if False:
                raise Refusal(f"{tag}: manifest has no 't0_fresh' (A4"""), "a4"),
    ("M30", "A4: R-T0 reads the fresh eval", sub('            t0e = e.get("t0")\\n', '            t0e = e.get("t0_fresh")\\n'), "a4 or world_ROBUST"),
'''),
])
print("A4 applied")
