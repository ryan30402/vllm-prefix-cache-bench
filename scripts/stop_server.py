"""Stop the vLLM server: try the recorded PID, then any 'vllm serve' process."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

pid_file = Path(sys.argv[1]) / "server.pid"
if pid_file.exists():
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, signal.SIGINT)
        print(f"Sent SIGINT to recorded PID {pid}")
    except ProcessLookupError:
        print(f"Recorded PID {pid} already gone, falling back to pkill")
subprocess.run(["pkill", "-INT", "-f", "vllm serve"])
time.sleep(10)
subprocess.run(["pkill", "-9", "-f", "vllm serve"])
print("Stopped")
