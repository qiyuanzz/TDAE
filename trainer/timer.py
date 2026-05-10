from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def wall_timer() -> Iterator[dict[str, float]]:
    record: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = time.perf_counter() - start


class CudaTimer:
    def __init__(self) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        self.end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        self.start_wall = 0.0

    def __enter__(self) -> "CudaTimer":
        if self.start_event is not None:
            torch.cuda.synchronize()
            self.start_event.record()
        self.start_wall = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.end_event is not None:
            self.end_event.record()
            torch.cuda.synchronize()
            self.seconds = self.start_event.elapsed_time(self.end_event) / 1000.0
        else:
            self.seconds = time.perf_counter() - self.start_wall

