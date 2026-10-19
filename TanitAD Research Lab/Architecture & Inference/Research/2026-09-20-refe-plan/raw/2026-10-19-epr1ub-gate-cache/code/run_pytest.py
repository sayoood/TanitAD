"""Run pytest from the gate's interpreter without installing it into that venv: python -I run_pytest.py <pytest args>.
The driverl-eval venv (the interpreter the gates are registered for) has no pytest.  Copy pytest, _pytest, pluggy, iniconfig, packaging, colorama and pygments from any venv that has them into a plain directory
(PYTEST_SITE, default D:/eu1/pytest_site) -- it is APPENDED to sys.path, so torch / numpy still resolve from the gate's venv.  No random-order plugin is installed, so pytest runs the order it is given."""
import os
import sys
sys.path.append(os.environ.get("PYTEST_SITE", "D:/eu1/pytest_site"))
import pytest
sys.exit(pytest.main(sys.argv[1:]))
