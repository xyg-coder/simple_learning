from __future__ import annotations

from typing import List

import enum
import time


class TimeStatKey(enum.IntEnum):
    DATA_LOAD = 1
    FORWARD = 2
    BACKWARD = 3


class TimeStat:
    """
    Utility class for recording the duration of some arbitrary logic.
    """

    def __init__(self):
        self.start_time = None
        self.time_history = []

    def start(self) -> None:
        self.start_time = time.time()

    def end(self) -> None:
        assert self.start_time is not None, "start must be called before end"
        time_delta = time.time() - self.start_time
        self.time_history.append(time_delta)
        self.start_time = None

    def get(self) -> List[float]:
        return self.time_history

    def get_avg(self) -> float:
        assert len(self.time_history) > 0, "No time recorded"
        return sum(self.time_history) / len(self.time_history)


class ThroughputStat:
    """
    Utility class for recording the throughput of some arbitrary logic.
    """

    def __init__(self):
        self.start_time = None
        self.time_delta = None
        self.start_count = None
        self.count_delta = None

    def start(self, value: int) -> None:
        self.start_time = time.time()
        self.start_count = value

    def end(self, value: int) -> None:
        assert self.start_time is not None and self.start_count is not None, "start must be called before end"
        self.time_delta = time.time() - self.start_time
        self.count_delta = value - self.start_count
        self.start_time = self.start_count = None

    def get(self) -> float:
        assert self.time_delta is not None and self.count_delta is not None, "end must be called before get"
        return self.count_delta / self.time_delta


class PerfTracker:
    """
    Utility class for tracking the performance metrics of mlenv solvers.

    It tracks the time spent per batch in data loading, the forward and backward passes.
    It also tracks the solver throughput over several batches.
    """

    def __init__(self):
        self.time_stats = {key: TimeStat() for key in TimeStatKey}
        self.throughput = ThroughputStat()

        self.stat_name_template = "mlenv.trainer.{key}_time_ms"

    def time_start(self, key: TimeStatKey) -> None:
        self.time_stats[key].start()

    def time_end(self, key: TimeStatKey) -> None:
        self.time_stats[key].end()

    def get_duration(self, key: TimeStatKey) -> List[float]:
        return self.time_stats[key].get()

    def throughput_start(self, value: int) -> None:
        self.throughput.start(value)

    def throughput_end(self, value: int) -> None:
        self.throughput.end(value)

    def get_throughput(self) -> float:
        return self.throughput.get()

    def clock_throughput(self, value: int) -> float:
        # Equivalent to calling end, getting and restarting
        self.throughput.end(value)
        throughput = self.throughput.get()
        self.throughput.start(value)
        return throughput

    def print_stats(self) -> None:
        for key, stat in self.time_stats.items():
            print(f"average for {self.stat_name_template.format(key=key.name)}: {stat.get_avg()} ms")
        print(f"Throughput: {self.get_throughput()} samples/s")
