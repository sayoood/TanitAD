"""Tests of the pilot audit page: frame selection, the self-contained HTML, and its JS logic run in node (reveal hidden until finished, export complete).
Run: CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_pilot_audit.py"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import build_pilot_audit as B

NODE = shutil.which("node")
TINY_JPG = base64.b64encode(bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9")).decode()


def synth_df(n_clips=200, seed=0):
    rng = np.random.default_rng(seed)
    countries = [f"C{i:02d}" for i in range(25)]
    bands = ["night", "twilight", "low_sun", "day"]
    rows = []
    for c in range(n_clips):
        for k in range(0, 101):
            d = lambda n: rng.dirichlet(np.ones(n) * 0.6).astype(np.float32)
            rows.append({"clip_id": f"clip{c:03d}", "frame_idx": k, "video_frame": 6 * k, "country": countries[c % 25], "hour_of_day": int(rng.integers(0, 24)),
                         "sun_band_est": bands[c % 4], "vlm_frame": k % 5 == 0, "weather_p": d(4), "surface_p": d(3), "visibility_p": d(3), "z_weather_p": d(4),
                         "z_surface_p": d(3), "illum_img_p": d(5), "luma_mean": float(rng.uniform(5, 150)), "luma_p95": float(rng.uniform(40, 255)),
                         "sky_luma": float(rng.uniform(5, 220)), "headlight_frac": float(rng.uniform(0, .2)), "lit_frac": float(rng.uniform(0, .4))})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def items():
    return B.select(synth_df())


def test_selection_counts_and_one_frame_per_clip(items):
    assert len(items) == 200
    assert sum(h["stratum"] == "random" for h in items) == 100
    assert len({h["clip"] for h in items}) == 200
    assert all(h["k"] % 5 == 0 for h in items)


def test_selection_is_deterministic(items):
    again = B.select(synth_df())
    assert [(h["clip"], h["k"]) for h in again] == [(h["clip"], h["k"]) for h in items]


def test_html_is_self_contained_and_hides_model_labels(items, tmp_path):
    out = tmp_path / "p.html"
    B.build_html(items, [TINY_JPG] * len(items), str(out))
    html = out.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html and "<script src" not in html and "<link" not in html
    visible = re.sub(r'const HIDDEN_B64 = "[^"]*";', "", html)
    for h in items[:20]:
        assert h["clip"] not in visible                       # clip ids are only inside the encoded block
    blob = re.search(r'const HIDDEN_B64 = "([^"]*)";', html).group(1)
    assert json.loads(base64.b64decode(blob))[0]["src"]["weather"]["qwen"] in B.WEATHER
    assert os.path.getsize(out) < 40 * 2 ** 20


@pytest.mark.skipif(NODE is None, reason="node missing")
def test_js_logic_reveal_gated_and_export_complete(items, tmp_path):
    out = tmp_path / "p.html"
    B.build_html(items, [TINY_JPG] * len(items), str(out))
    html = out.read_text(encoding="utf-8")
    logic = html.split("// ===== LOGIC (pure; tested in node) =====")[1].split("// ===== END LOGIC =====")[0]
    hidden = [{"clip": h["clip"], "k": h["k"], "stratum": h["stratum"], "src": h["src"]} for h in items]
    js = logic + """
const hidden = %s;
const out = {};
let st = makeState(200);
out.before_hidden = hiddenFor(st, hidden);
out.reveal_empty = reveal(st);
for (let i = 0; i < 199; i++) for (const f of Object.keys(FIELDS)) answer(st, i, f, FIELDS[f][0]);
out.done199 = nDone(st); out.can199 = canReveal(st); out.reveal199 = reveal(st); out.agree199 = agreement(st, hidden);
for (const f of Object.keys(FIELDS)) answer(st, 199, f, f === "weather" ? "cant_tell" : FIELDS[f][1]);
out.done200 = nDone(st); out.can200 = canReveal(st);
out.hidden_before_reveal = hiddenFor(st, hidden);
out.agree_before_reveal = agreement(st, hidden);
out.reveal200 = reveal(st);
out.frozen = answer(st, 0, "weather", "rain");
out.hidden_after = hiddenFor(st, hidden) === hidden;
const ag = agreement(st, hidden);
out.ag_weather_siglip = ag.weather.siglip; out.ag_vis_sources = Object.keys(ag.visibility);
const ex = exportObj(st, hidden);
out.export_n = ex.frames.length; out.export_fields_ok = ex.frames.every(r => ["weather","surface","visibility","illumination","tod"].every(f => f in r));
out.export_weather_199 = ex.frames[199].weather; out.export_answered = ex.n_answered;
let bad = false; try { answer(makeState(2), 0, "weather", "sunny"); } catch (e) { bad = true; }
out.bad_value_rejected = bad;
console.log(JSON.stringify(out));
""" % json.dumps(hidden)
    (tmp_path / "t.js").write_text(js, encoding="utf-8")
    r = subprocess.run([NODE, str(tmp_path / "t.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout)
    assert o["before_hidden"] is None and o["reveal_empty"] is False
    assert o["done199"] == 199 and o["can199"] is False and o["reveal199"] is False and o["agree199"] is None
    assert o["done200"] == 200 and o["can200"] is True
    assert o["hidden_before_reveal"] is None and o["agree_before_reveal"] is None
    assert o["reveal200"] is True and o["frozen"] is False and o["hidden_after"] is True
    assert o["export_n"] == 200 and o["export_fields_ok"] and o["export_weather_199"] == "cant_tell" and o["export_answered"] == 200
    assert o["bad_value_rejected"] is True
    s = o["ag_weather_siglip"]
    assert s["cant"] == 1 and s["n"] == 199 and s["n"] == sum(v["n"] for v in s["per"].values())
    assert o["ag_vis_sources"] == ["qwen"]


@pytest.mark.skipif(NODE is None, reason="node missing")
def test_regression_page_that_reveals_early_goes_red(items, tmp_path):
    """Deliberate regression: canReveal that is always true must break the 'not before finished' assertion."""
    out = tmp_path / "p.html"
    B.build_html(items, [TINY_JPG] * len(items), str(out))
    html = out.read_text(encoding="utf-8").replace("function canReveal(st){return finished(st);}", "function canReveal(st){return true;}")
    logic = html.split("// ===== LOGIC (pure; tested in node) =====")[1].split("// ===== END LOGIC =====")[0]
    r = subprocess.run([NODE, "-e", logic + "let st=makeState(200); console.log(JSON.stringify(reveal(st)));"], capture_output=True, text=True)
    assert json.loads(r.stdout) is True            # the mutated page reveals an empty session: the real test above requires False


def test_illum_rule_known_cases():
    import pilot_common as P
    assert P.illum_rule(120, 200, 200, 0.01, 0.4) == "day"
    assert P.illum_rule(10, 30, 8, 0.0, 0.0) == "dark"
    assert P.illum_rule(60, 220, 30, 0.02, 0.2) == "lit_night"
    assert P.illum_rule(150, 255, 200, 0.2, 0.6) == "glare"
