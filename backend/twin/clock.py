"""Simulation clock: maps campus 'now' onto the historical datasets.

The twin runs on simulated time that starts at the real current time and can
be sped up (or jumped) for demos. Each dataset is read at the same ISO week
and weekday in its own year, so weekday/weekend and term-time patterns line up.
"""
import threading
from datetime import datetime, timedelta


class SimClock:
    def __init__(self, speed=60.0):
        self._lock = threading.Lock()
        self._anchor_real = datetime.now()
        self._anchor_sim = datetime.now().replace(second=0, microsecond=0)
        self.speed = speed

    def now(self):
        with self._lock:
            elapsed = (datetime.now() - self._anchor_real).total_seconds()
            return self._anchor_sim + timedelta(seconds=elapsed * self.speed)

    def set_speed(self, speed):
        with self._lock:
            cur = self._now_unlocked()
            self._anchor_real, self._anchor_sim, self.speed = datetime.now(), cur, float(speed)

    def jump_to(self, when):
        with self._lock:
            self._anchor_real, self._anchor_sim = datetime.now(), when

    def _now_unlocked(self):
        elapsed = (datetime.now() - self._anchor_real).total_seconds()
        return self._anchor_sim + timedelta(seconds=elapsed * self.speed)


def to_dataset_time(sim, year):
    """Same ISO week + weekday + time of day, in the dataset's year."""
    _, week, weekday = sim.isocalendar()
    week = min(week, 52)
    day = datetime.fromisocalendar(year, week, weekday)
    if day.year != year:  # ISO week 1 may start in the previous December
        day = datetime.fromisocalendar(year, week + 1, weekday)
    return day.replace(hour=sim.hour, minute=sim.minute)
