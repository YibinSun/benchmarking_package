#!/bin/bash

# Function to monitor memory usage of a process and its subprocesses on macOS/BSD
monitor_memory() {
    local pid=$1
    local logfile=$2

    # Log memory usage every second until the process ends
    while ps -p "$pid" > /dev/null; do
        # Get memory usage for the main process and all subprocesses
        # Adjust the ps command for macOS/BSD
        mem_usage=$(ps -o pid,rss -p "$pid" $(pgrep -P "$pid") | awk '{if(NR>1) sum += $2} END {print sum}')
        mem_usage_mb=$(echo "scale=2; $mem_usage/1024" | bc) # Convert to MB

        # Log timestamp and memory usage
        echo "$(date '+%Y-%m-%d %H:%M:%S') - PID: $pid - Memory: $mem_usage_mb MB" >> "$logfile"

        # Wait for 1 second before next check
        sleep 0.5
    done
}




# Run the moa benchmark
#echo "Running run_moa_benchmark.py"
#
## Run the Python script in the background
#python3 run_moa_benchmark.py &
#PID=$!
#
## Output log file for memory usage
#log_file="./memory_log/memory_usage_moa.log"
#
## Monitor memory usage of the Python script and its subprocesses
#monitor_memory $PID $log_file &
## Wait for the Python script to finish
#wait $PID
#
#echo "Memory usage logging finished for PID $PID. Check $log_file for details."
#
#
#
#
# Run the capymoa benchmark
#echo "Running run_capymoa_benchmark.py"
#
## Run the Python script in the background
#python3 run_capymoa_benchmark.py &
#PID=$!
#
## Output log file for memory usage
#log_file="./memory_log/memory_usage_capymoa.log"
#
## Monitor memory usage of the Python script and its subprocesses
#monitor_memory $PID $log_file &
## Wait for the Python script to finish
#wait $PID
#
#echo "Memory usage logging finished for PID $PID. Check $log_file for details."
#
#
#
# Run the river benchmark
echo "Running run_river_benchmark.py"

# Run the Python script in the background
python3 run_river_benchmark.py &
PID=$!

# Output log file for memory usage
log_file="./memory_log/memory_usage_river.log"

# Monitor memory usage of the Python script and its subprocesses
monitor_memory $PID $log_file &
# Wait for the Python script to finish
wait $PID

echo "Memory usage logging finished for PID $PID. Check $log_file for details."




# skmf benchmark
#echo "Running run_skmf_benchmark.py"
## Run the Python script in the background
#python3 run_skmf_benchmark.py &
#PID=$!
#
## Output log file for memory usage
#log_file="./memory_log/memory_usage_skmf.log"
#
## Monitor memory usage of the Python script and its subprocesses
#monitor_memory $PID $log_file &
#
## Wait for the Python script to finish
#wait $PID
#
#echo "Memory usage logging finished for PID $PID. Check $log_file for details."







