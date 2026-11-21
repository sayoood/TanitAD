#!/usr/bin/env bash
# STOPBAL-STUB-TRANSPORT (marker the chain looks for: a file carrying it is never the real client). Dry-run sftp: forwards to fake_box.py, which touches only the fake box folder. Needs FAKE_BOX and FAKE_PY.
exec "${FAKE_PY:?FAKE_PY}" "$(dirname "${BASH_SOURCE[0]}")/fake_box.py" sftp "$@"
