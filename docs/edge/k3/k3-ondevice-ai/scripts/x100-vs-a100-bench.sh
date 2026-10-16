set +e
M=/home/pingji/models/Qwen3.5-2B-Q4_0.gguf
cd /home/pingji/k3-ai
echo "############ 观察：默认运行时线程实际落在哪些核 ############"
llama-bench -m $M -p 64 -n 8 -t 8 -r 1 >/dev/null 2>&1 &
P=$!
sleep 4
echo "  各线程亲和性分布（默认，不干预）:"
cat /proc/$P/task/*/status 2>/dev/null | grep Cpus_allowed_list | sort | uniq -c | sed 's/^/    /'
wait $P
echo
echo "############ A) 默认（不干预）############"
llama-bench -m $M -p 128 -n 32 -t 8 -r 2 2>&1 | tail -6 | sed 's/^/  /'
echo
echo "############ B) 强制 X100（taskset 0-7）############"
taskset -c 0-7 llama-bench -m $M -p 128 -n 32 -t 8 -r 2 2>&1 | tail -6 | sed 's/^/  /'
echo
echo "############ C) A100（echo 0 > /proc/set_ai_thread + taskset 8-15）############"
bash -c "echo 0 > /proc/set_ai_thread; exec taskset -c 8-15 llama-bench -m \$0 -p 128 -n 32 -t 8 -r 2" $M 2>&1 | tail -6 | sed 's/^/  /'
echo
echo "############ 确认 A100 组确实落在 8-15 ############"
bash -c "echo 0 > /proc/set_ai_thread; exec taskset -c 8-15 llama-bench -m \$0 -p 64 -n 8 -t 8 -r 1" $M >/dev/null 2>&1 &
P=$!
sleep 4
cat /proc/$P/task/*/status 2>/dev/null | grep Cpus_allowed_list | sort | uniq -c | sed 's/^/    /'
wait $P
