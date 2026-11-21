#!/bin/bash
# Pod: re-run, on the fixed tree /root/strat1/new3, every test FILE that holds a G-SUITE regression, plus the three STRAT1 test files.
# One file at a time (serial), junit per file, then count pass/fail per regression id.
set -u
R=/root/strat1; T=$R/new3; J=$R/junit/new3_aff; TMP=$R/tmp/new3_aff; PY=/root/venv/bin/python
rm -rf "$J" "$TMP"; mkdir -p "$J" "$TMP"
cd "$T" || exit 2
export PYTHONPATH="$T/stack:$T" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
FILES=$($PY - <<'EOF'
import json
reg = json.load(open("/root/strat1/gsuite_result.json"))["regressions"]
mods = sorted({r.split("::")[0].split(".")[-1] for r in reg} | {"test_v8f_strat1", "test_v8f_strat1_navbuilder", "test_strat1_token"})
print(" ".join(mods))
EOF
)
echo "files: $FILES"
one() {
  m=$1; f=$T/stack/tests/$m.py
  [ -f "$f" ] || { echo "MISSING $f"; return; }
  mkdir -p "$TMP/$m"
  TMPDIR="$TMP/$m" TMP="$TMP/$m" TEMP="$TMP/$m" timeout 2h $PY -m pytest -q -p no:cacheprovider --basetemp "$TMP/$m/bt"       --junitxml "$J/$m.xml" "$f" > "$J/$m.log" 2>&1
  echo "$(date -u +%T) $m rc=$? $(tail -1 $J/$m.log)"
}
export -f one; export T J TMP PY
printf '%s
' $FILES | xargs -P 4 -I{} bash -c 'one {}'
$PY - <<'EOF'
import glob, json, xml.etree.ElementTree as ET
reg = set(json.load(open("/root/strat1/gsuite_result.json"))["regressions"])
st = {}
for f in glob.glob("/root/strat1/junit/new3_aff/*.xml"):
    for tc in ET.parse(f).getroot().iter("testcase"):
        tid = (tc.get("classname") or "") + "::" + (tc.get("name") or "")
        bad = any(ch.tag in ("failure", "error") for ch in tc)
        skip = any(ch.tag == "skipped" for ch in tc)
        st[tid] = "fail" if bad else ("skip" if skip else "pass")
fixed = sum(1 for r in reg if st.get(r) == "pass")
still = sorted(r for r in reg if st.get(r) != "pass")
allfail = sorted(t for t, s in st.items() if s == "fail")
print(json.dumps({"n_regressions": len(reg), "now_pass": fixed, "not_pass": still[:40], "n_tests_run": len(st), "n_fail_any": len(allfail), "fail_any": allfail[:40]}, indent=1))
EOF
echo "ZZ RERUN AFFECTED DONE ZZ"
