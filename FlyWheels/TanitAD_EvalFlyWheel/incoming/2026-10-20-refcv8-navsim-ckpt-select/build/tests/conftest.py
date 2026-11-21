"""Pytest bootstrap for the navtrain-selection build: puts ``code/`` and ``tests/`` on sys.path and records every deliberate-regression
outcome into ``raw/mutation_record.json`` (written at session end when ``R8SEL_BANK=1``)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.dirname(HERE)
for p in (os.path.join(BUILD, "code"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

MUTATION_RECORD: dict = {}


def pytest_sessionfinish(session, exitstatus):                       # noqa: ARG001
    if os.environ.get("R8SEL_BANK") == "1" and MUTATION_RECORD:
        path = os.path.join(BUILD, "raw", os.environ.get("R8SEL_BANK_NAME", "mutation_record.json"))
        try:
            old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        except (OSError, ValueError):
            old = {}
        old.update(MUTATION_RECORD)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(old, fh, indent=1, sort_keys=True)
