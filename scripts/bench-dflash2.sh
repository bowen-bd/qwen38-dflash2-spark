#!/usr/bin/env bash
# Benchmark one SGLang arm with the identical suite used for the DSpark baseline.
# ARM label comes in as $1 so DSpark-on-upstream and DFlash2 can share this script.
ARM="${1:-dflash2}"
OUT=/home/bdeng/llm/ab-results
LOG=$OUT/${ARM}-bench.log
exec >> "$LOG" 2>&1
URL=http://127.0.0.1:30000/v1
MODEL=qwen3.8-27b

gpu () { nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ' '; }
mem () { free -m | awk 'NR==2{printf "MemAvailable=%.1f GiB used=%.1f GiB",$7/1024,$3/1024}'; }

echo; echo "===== ARM: $ARM ====="; date '+%F %T'
s=0
while [ $s -lt 2400 ]; do
  [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health 2>/dev/null)" = "200" ] && break
  # The scheduler can die while the container stays Up, so watch the log too.
  if grep -q "Scheduler hit an exception" "$OUT/${ARM}-serve.log" 2>/dev/null; then
    echo "SCHEDULER CRASHED at ${s}s:"
    grep -A3 "Scheduler hit an exception" "$OUT/${ARM}-serve.log" | head -4
    grep -E "^[A-Za-z]*(Error|Exception):" "$OUT/${ARM}-serve.log" | tail -3
    exit 1
  fi
  if [ $s -ge 60 ] && ! docker ps --format '{{.Names}}' | grep -q qwen38-sglang-run; then
    echo "CONTAINER DIED at ${s}s. Last lines:"; tail -30 "$OUT/${ARM}-serve.log"; exit 1
  fi
  sleep 10; s=$((s+10))
done
[ $s -ge 2400 ] && { echo "TIMEOUT after ${s}s"; tail -30 "$OUT/${ARM}-serve.log"; exit 1; }
echo "healthy after ${s}s"
echo "--- footprint: $(gpu)"
echo "--- $(mem)"
grep -iE "KV Cache is allocated|max_total_num_tokens|Load weight end" "$OUT/${ARM}-serve.log" | tail -5

echo; echo "--- workloads (batch-1 decode, thinking off)"
timeout 2400 python3 /home/bdeng/llm/bench-workloads.py "$URL" "$MODEL"

echo; echo "--- concurrency + prefill"
timeout 2400 python3 /home/bdeng/llm/bench-qwen38.py throughput --url "$URL" --model "$MODEL" \
    --concurrency 1 2 4 8 --prefill 1000 4000 16000

echo; echo "--- vision (1 chart)"
timeout 900 python3 /home/bdeng/llm/bench-qwen38.py vision --url "$URL" --model "$MODEL" \
    --image /home/bdeng/llm/test-chart.png --effort none --max-tokens 400 \
    --question "Read this chart. List every bar with its tok/s and GB value."

echo; echo "--- accuracy GSM8K 150 (identical settings to the DSpark baseline)"
timeout 3600 python3 /home/bdeng/llm/bench-qwen38.py accuracy --url "$URL" --model "$MODEL" \
    --n 150 --acc-conc 4 --effort none --max-tokens 768

echo; echo "--- acceptance length from server log (spec decoding efficiency)"
grep -oE "accept len: [0-9.]+" "$OUT/${ARM}-serve.log" | tail -40 | awk '{s+=$3;n++} END{if(n)printf "mean accept len over last %d decode batches: %.2f\n",n,s/n; else print "not logged"}'

echo; echo "--- footprint after: $(gpu)"
echo "--- $(mem)"
echo "===== ARM $ARM DONE ====="; date '+%F %T'
