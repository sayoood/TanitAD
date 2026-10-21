#!/bin/bash
# md5 the navtest metric cache + the 147 test log pickles on the dev box and on Thor; diff.  Markers QQCACHEMD5_* / QQLOGSMD5_*.
set -u
L=/c/Users/Admin/refcv8_pf/navtest_read/raw/ship_verify; mkdir -p $L
cd /d/Archive/devbox-C/navsim/data/openscene/navsim_logs/test && find . -type f | sort | xargs md5sum | sed 's/ \*/  /' > $L/logs_md5_devbox.txt
ssh -n tanitad-thor 'cd /home/nvidia/r8ns/navtest/nsdata/openscene/navsim_logs/test && find . -type f | sort | xargs md5sum' > $L/logs_md5_thor.txt
diff $L/logs_md5_devbox.txt $L/logs_md5_thor.txt > $L/logs_md5_diff.txt && echo "QQLOGSMD5_MATCH n=$(wc -l < $L/logs_md5_thor.txt)" || echo QQLOGSMD5_DIFF
cd /d/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest && find . -type f | sort | xargs md5sum | sed 's/ \*/  /' > $L/cache_md5_devbox.txt
ssh -n tanitad-thor 'cd /home/nvidia/r8ns/navtest/nsexp/w3_navtest_v1/metric_cache_navtest && find . -type f | sort | xargs md5sum' > $L/cache_md5_thor.txt
diff $L/cache_md5_devbox.txt $L/cache_md5_thor.txt > $L/cache_md5_diff.txt && echo "QQCACHEMD5_MATCH n=$(wc -l < $L/cache_md5_thor.txt)" || echo QQCACHEMD5_DIFF
