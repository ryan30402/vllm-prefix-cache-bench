"""Stand-in for `nvidia-smi`: answers the queries the v3 harness makes.

FAKE_SMI_UUID changes the reported GPU UUID (a "different machine").
FAKE_SMI_BAD_FIELD names one query field this fake driver does not know.
"""

import os
import sys
import time
from datetime import datetime

VALUES = {
    "name": "Fake GPU",
    "uuid": os.environ.get("FAKE_SMI_UUID", "GPU-fake-0000"),
    "memory.total": "46068 MiB",
    "driver_version": "0.0",
    "temperature.gpu": "55",
    "power.draw": "210.5",
    "utilization.gpu": "97",
    "utilization.memory": "40",
    "memory.used": "41000",
    "clocks.current.sm": "[N/A]",
    "clocks.current.memory": "[N/A]",
}


def row(fields: list[str]) -> str:
    values = dict(
        VALUES, timestamp=datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    )
    return ", ".join(values[f] for f in fields)


def main() -> None:
    args = sys.argv[1:]
    if args == ["-L"]:
        print("GPU 0: Fake GPU (UUID: GPU-fake-0000)")
        return
    query = next(
        (a.split("=", 1)[1] for a in args if a.startswith("--query-gpu=")), None
    )
    if query is None:
        sys.exit(1)
    fields = query.split(",")
    bad = os.environ.get("FAKE_SMI_BAD_FIELD", "")
    if any(f not in VALUES and f != "timestamp" or f == bad for f in fields):
        print(f"Field is not a valid field to query.", file=sys.stderr)
        sys.exit(2)
    if "-lms" in args:
        interval = int(args[args.index("-lms") + 1]) / 1000
        print(", ".join(fields), flush=True)
        while True:
            print(row(fields), flush=True)
            time.sleep(interval)
    print(row(fields))


if __name__ == "__main__":
    main()
