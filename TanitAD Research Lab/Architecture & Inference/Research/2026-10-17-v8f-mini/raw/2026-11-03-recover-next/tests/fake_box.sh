#!/usr/bin/env bash
# Stand-in for `ssh tanitad-box2 <cmd>` in the waker test: prints the marker-directory listing the test staged in $RNEXT_FAKE_LISTING (one name per line). Reaches nothing.
cat "$RNEXT_FAKE_LISTING" 2>/dev/null
