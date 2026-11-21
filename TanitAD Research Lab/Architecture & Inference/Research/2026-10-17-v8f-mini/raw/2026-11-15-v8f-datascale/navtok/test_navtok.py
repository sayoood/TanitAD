import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import navtok as N

HERE = os.path.dirname(os.path.abspath(__file__))
TEXTS = os.path.join(HERE, "raw", "texts.json")


@pytest.mark.parametrize("text,want", [
    ("Turn right in 249m", ("TURN_R", 249)),
    ("Turn left", ("TURN_L", None)),
    ("Turn sharp left in 12m", ("TURN_SHARP_L", 12)),
    ("Turn slight right in 0m", ("TURN_SLIGHT_R", 0)),
    ("Continue straight", ("CONTINUE_STRAIGHT", None)),
    ("Continue in 800m", ("CONTINUE", 800)),
    ("Continue slight left in 5m", ("CONTINUE_SLIGHT_L", 5)),
    ("Keep left in 30m", ("KEEP_L", 30)),
    ("Keep slight right", ("KEEP_SLIGHT_R", None)),
    ("Merge slight left in 100m", ("MERGE_SLIGHT_L", 100)),
    ("Merge left in 100m", ("MERGE_L", 100)),
    ("Take the ramp slight right in 70m", ("RAMP_SLIGHT_R", 70)),
    ("Make a U-turn in 40m", ("UTURN", 40)),
    ("Exit rotary in 214m", ("ROTARY_EXIT", 214)),
    ("Exit roundabout in 3m", ("RB_EXIT", 3)),
    ("Enter roundabout", ("RB_ENTER", None)),
    ("At roundabout, take exit 2 in 60m", ("RB_TAKE_EXIT", 60)),
    ("At roundabout, take exit 3", ("RB_TAKE_EXIT", None)),
    ("Continue around the roundabout and take the next exit", ("RB_NEXT_EXIT", None)),
])
def test_known_texts(text, want):
    c, d, raw = N.parse(text)
    assert (c, d) == want and raw == text


def test_exit_number_is_kept():
    assert N.parse_full("At roundabout, take exit 4 in 9m").exit_n == 4


@pytest.mark.parametrize("bad", ["", "Turn around", "turn left in 5m", "Turn left in 5 m", "Turn left in 5m extra", "Turn right in -5m",
                                 "Bear left", "Turn slight sharp left", "Keep sharp left"])
def test_unknown_is_UNPARSED_never_a_default(bad):
    assert N.parse(bad)[0] == "UNPARSED"
    assert N.coarse("UNPARSED") == -2


def test_vocab_is_frozen():
    assert len(N.VOCAB) == len(set(N.VOCAB)) == 44
    assert N.VOCAB[0] == "UNPARSED" and N.VOCAB[1] == "CONTINUE"
    assert N.VOCAB[-1] == "RB_NEXT_EXIT"


def test_coarse_map():
    assert [N.coarse(c) for c in ("TURN_L", "TURN_SLIGHT_R", "KEEP_L", "RAMP_R", "RB_ENTER", "CONTINUE_STRAIGHT")] == [1, 2, 0, 0, 0, 0]
    assert N.coarse("UTURN") == -1


@pytest.mark.skipif(not os.path.exists(TEXTS), reason="raw/texts.json not in the package")
def test_all_9864_distinct_texts_parse():
    texts = json.load(open(TEXTS, encoding="utf-8"))
    assert len(texts) == 9864
    bad = [t for t in texts if N.parse(t)[0] == "UNPARSED"]
    assert bad == []
    # every class in the vocabulary other than UNPARSED that the corpus uses is in VOCAB, and the distance is the digits in the text
    for t in texts:
        c, d, _ = N.parse(t)
        m = re.search(r" in (\d+)m$", t)
        assert (d is None) == (m is None) and (d is None or d == int(m.group(1)))


# ---- deliberate-regression arm: a broken regex MUST make the corpus test fail --------------------------------------------------------
def test_DELIBERATE_REGRESSION_broken_regex_is_caught(monkeypatch):
    broken = [(v, re.compile(rx.pattern.replace("sharp|slight", "sharp"))) for v, rx in N._COMPILED]
    monkeypatch.setattr(N, "_COMPILED", broken)
    assert N.parse("Turn slight right in 5m")[0] == "UNPARSED"          # the broken rule no longer matches ...
    if os.path.exists(TEXTS):
        texts = json.load(open(TEXTS, encoding="utf-8"))
        assert any(N.parse(t)[0] == "UNPARSED" for t in texts)          # ... and the corpus test WOULD be red
    # and the distance group: dropping it must change the parsed distance
    broken2 = [(v, re.compile(rx.pattern.replace(r"(?: in (?P<d>\d+)m)?", ""))) for v, rx in N._COMPILED]
    monkeypatch.setattr(N, "_COMPILED", broken2)
    assert N.parse("Turn right in 249m")[0] == "UNPARSED"
