import subprocess
import time
import json

memory_usage = []
max_memory_used = 0
log_path = "result_of_memory.json"

try:
    while True:
        usage = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
        ).decode().split('\n')[0].strip())/2**10
        
        memory_usage.append(usage)
        if usage > max_memory_used:
            max_memory_used = usage

        time.sleep(1)

except KeyboardInterrupt:
    with open(log_path, "w") as f:
        json.dump({
            "max_memory": max_memory_used,
            "all_readings": memory_usage
        }, f)
    print(f"Max used memory: {max_memory_used} GB")
