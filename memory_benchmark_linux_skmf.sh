#!/bin/bash

# Function to monitor memory usage of a process and its subprocesses on macOS/BSD
monitor_memory() {
    local pid=$1
    local logfile=$2

    # Log memory usage every second until the process ends
    while ps -p "$pid" > /dev/null; do
        # Get memory usage for the main process and all subprocesses
        mem_usage=$(ps --no-headers -o pid,rss --ppid "$pid" -p "$pid" | awk '{sum += $2} END {print sum}')
        mem_usage_mb=$(echo "scale=2; $mem_usage/1024" | bc) # Convert to MB

        # Log timestamp and memory usage
        echo "$(date '+%Y-%m-%d %H:%M:%S') - PID: $pid - Memory: $mem_usage_mb MB" >> "$logfile"

        # Wait for 1 second before next check
        sleep 0.5
    done
}

# skmf benchmark
echo "Running run_skmf_benchmark.py"
# Run the Python script in the background
python3 run_skmf_benchmark.py &
PID=$!

# Output log file for memory usage
log_file="./memory_log/memory_usage_skmf.log"

# Monitor memory usage of the Python script and its subprocesses
monitor_memory $PID $log_file &

# Wait for the Python script to finish
wait $PID

echo "Memory usage logging finished for PID $PID. Check $log_file for details."







