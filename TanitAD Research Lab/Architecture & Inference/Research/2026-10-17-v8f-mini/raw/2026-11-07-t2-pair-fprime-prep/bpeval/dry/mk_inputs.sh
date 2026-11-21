set -e
D=$1; mkdir -p $D/ckpt/M2b_s0 $D/ckpt/M2b_s1 $D/cache/clips $D/work $D/out
for s in 0 1; do head -c 2048 /dev/urandom > $D/ckpt/M2b_s$s/ckpt_last.pt; echo "{\"argv\": [\"--arm\", \"M2b\", \"--seed\", \"$s\"]}" > $D/ckpt/M2b_s$s/argv.json; echo '{"done": true}' > $D/ckpt/M2b_s$s/summary.json; done
for i in $(seq 1 141); do head -c 4096 /dev/urandom > $D/cache/clips/clip$i.pt; echo "{\"clip\": \"clip$i\", \"n_frames\": 10}" >> $D/cache/manifest.jsonl; done
(cd $D/cache && for f in clips/*.pt manifest.jsonl; do echo "$(sha256sum $f | cut -c1-64)  $f"; done) > $D/cache_pod_eval.sha256
