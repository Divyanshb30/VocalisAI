"""Round-trip time from this machine to the speech and LLM endpoints, for reading the latency numbers.

uv run python -m evals.netcheck   # -> evals/results/network.json
"""

from __future__ import annotations

import json
import socket
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

HOSTS = {"deepgram": "api.deepgram.com", "cerebras": "api.cerebras.ai"}
OUT = Path("evals/results/network.json")


def tcp_rtt_ms(host: str, n: int = 15) -> dict[str, float]:
    ip = socket.gethostbyname(host)
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        socket.create_connection((ip, 443), timeout=5).close()
        xs.append(1000 * (time.perf_counter() - t))
        time.sleep(0.2)
    return {"median_ms": round(statistics.median(xs)), "min_ms": round(min(xs)), "max_ms": round(max(xs))}


def main() -> None:
    out = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "TCP connect time to port 443 (one network round trip), 15 samples",
        "rtt": {name: {"host": host, **tcp_rtt_ms(host)} for name, host in HOSTS.items()},
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
