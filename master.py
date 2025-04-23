import subprocess
import time
import json
import os
import signal

monitor_proc = subprocess.Popen(["python", "monitor_gpu.py"])

print("run main.py (model training)...")
main_proc = subprocess.run(["python", "main.py"])

print("Monitoring completed...")
monitor_proc.send_signal(signal.SIGINT)
monitor_proc.wait()

with open("result_of_training.json") as f:
    result = json.load(f)

with open("result_of_memory.json") as f:
    memory_stats = json.load(f)

result['nvidia_smi_peak'] = memory_stats["max_memory"]

with open(f"result_{result['loss_type']}_{result['dataset']}_{result['model']}.json", "w") as f:
    json.dump(result, f)
    
with open(f"memory_{result['loss_type']}_{result['dataset']}_{result['model']}.json", "w") as f:
    json.dump(memory_stats, f)

print(result)

os.remove("result_of_training.json")
os.remove("result_of_memory.json")