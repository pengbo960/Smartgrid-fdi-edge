from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_time_seconds: float
    memory_mb: float


class ResourceMonitor:
    """Measure process CPU time and peak resident memory without extras."""

    def snapshot(self) -> ResourceSnapshot:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        maximum_rss = float(usage.ru_maxrss)
        if sys.platform != "darwin":
            maximum_rss *= 1024.0
        return ResourceSnapshot(
            cpu_time_seconds=time.process_time(),
            memory_mb=maximum_rss / (1024 ** 2),
        )
