"""ML models behind the twin's anomaly detection.

- Robust hour-of-week baselines (median / MAD) for every building meter.
- Isolation Forest per building as a second opinion on energy anomalies.
- Random Forest occupancy model trained on UCI Occupancy Detection.
- Leak detector for the campus water inlet (time-of-week residual + EWMA),
  evaluated against BattLeDIM's labelled 2018 leaks.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import CACHE

OCC_FEATURES = ["Temperature", "Humidity", "Light"]


def hour_of_week(index):
    return index.dayofweek * 24 + index.hour


class Baselines:
    """Median and MAD of a series for each of the 168 hours of the week."""

    def __init__(self, series):
        how = hour_of_week(series.index)
        g = series.groupby(how)
        self.median = g.median().reindex(range(168)).interpolate().bfill().values
        mad = g.apply(lambda s: (s - s.median()).abs().median())
        self.mad = np.maximum(mad.reindex(range(168)).interpolate().bfill().values, 1e-6)
        self.night_floor = float(np.percentile(series.dropna(), 5))

    def expected(self, ts):
        return float(self.median[ts.dayofweek * 24 + ts.hour])

    def zscore(self, ts, value):
        i = ts.dayofweek * 24 + ts.hour
        return float((value - self.median[i]) / (1.4826 * self.mad[i]))


class EnergyIForest:
    def __init__(self, series):
        self.mean = float(series.mean()) or 1.0
        X = self._features(series.index, series.values)
        self.model = IsolationForest(n_estimators=150, contamination=0.02, random_state=7).fit(X)

    def _features(self, index, values):
        h = index.hour + index.minute / 60
        return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                                (index.dayofweek >= 5).astype(float), np.asarray(values) / self.mean])

    def is_anomaly(self, ts, value):
        idx = pd.DatetimeIndex([ts])
        return bool(self.model.predict(self._features(idx, [value]))[0] == -1)


class OccupancyModel:
    def __init__(self):
        occ = pd.read_pickle(CACHE / "occupancy.pkl")
        train = occ[occ.split == "datatraining"]
        test = occ[occ.split != "datatraining"]
        self.model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=7)
        self.model.fit(train[OCC_FEATURES], train.Occupancy)
        pred = self.model.predict(test[OCC_FEATURES])
        self.metrics = {
            "dataset": "UCI Occupancy Detection (held-out test files)",
            "features": OCC_FEATURES,
            "accuracy": round(accuracy_score(test.Occupancy, pred), 3),
            "f1": round(f1_score(test.Occupancy, pred), 3),
            "test_rows": int(len(test)),
        }

    def probs(self, rows):
        """Occupancy probability for many (temp, rh, lux) rows in one call."""
        if not rows:
            return []
        X = pd.DataFrame(rows, columns=OCC_FEATURES)
        return [float(p) for p in self.model.predict_proba(X)[:, 1]]

    def prob(self, temp, rh, lux):
        X = pd.DataFrame([[temp, rh, lux]], columns=OCC_FEATURES)
        return float(self.model.predict_proba(X)[0, 1])


class LeakDetector:
    """Minimum-night-flow (MNF) leak detection, the standard water-utility method.

    Each night's minimum inflow (01:00-05:00) is compared with the median of the
    previous 7 nights. A new leak raises the floor that demand never drops below.
    A night is flagged when the excess beats max(3 x robust sigma, 1% of baseline).
    """

    def __init__(self, water, onsets, window=7, k=3.0):
        self.water = water
        tow = water.index.dayofweek * 288 + water.index.hour * 12 + water.index.minute // 5
        self.baseline = water.inlet_lph.groupby(tow).median().reindex(range(2016)).interpolate().bfill().values
        night = water.inlet_lph.between_time("01:00", "05:00")
        self.mnf = night.resample("D").min()
        base = self.mnf.rolling(window, min_periods=4).median().shift(1)
        mad = self.mnf.rolling(window, min_periods=4).apply(lambda s: np.median(np.abs(s - np.median(s))), raw=True).shift(1)
        self.threshold = np.maximum(k * 1.4826 * mad, 0.01 * base)
        self.excess = (self.mnf - base)
        self.score_day = (self.excess / self.threshold).fillna(0)
        self.flag_day = self.score_day > 1
        self.onsets = sorted(pd.Timestamp(t).normalize() for t in onsets)
        self.metrics = self._evaluate()

    def _evaluate(self, catch_days=3, effect_days=10):
        caught, delays = 0, []
        for o in self.onsets:
            win = self.flag_day[o: o + pd.Timedelta(days=catch_days)]
            hit = win[win]
            if len(hit):
                caught += 1
                delays.append((hit.index[0] - o).days)
        flagged = self.flag_day[self.flag_day].index
        true_pos = [d for d in flagged if any(o <= d <= o + pd.Timedelta(days=effect_days) for o in self.onsets)]
        return {
            "dataset": "BattLeDIM 2018 L-Town SCADA (14 labelled pipe bursts)",
            "method": "minimum night flow vs 7-night rolling median",
            "leak_events": len(self.onsets),
            "events_caught": caught,
            "catch_window_days": catch_days,
            "median_delay_days": float(np.median(delays)) if delays else None,
            "flagged_nights": int(len(flagged)),
            "precision": round(len(true_pos) / len(flagged), 2) if len(flagged) else None,
        }

    def _day(self, ts):
        d = ts.normalize()
        return d if ts.hour >= 4 else d - pd.Timedelta(days=1)

    def score_at(self, ts):
        return float(self.score_day.get(self._day(ts), 0.0))

    def excess_at(self, ts):
        v = self.excess.get(self._day(ts), 0.0)
        return float(0 if pd.isna(v) else v)

    def threshold_at(self, ts):
        v = self.threshold.get(self._day(ts), np.nan)
        return float(np.nanmedian(self.threshold)) if pd.isna(v) else float(v)

    def expected(self, ts):
        return float(self.baseline[ts.dayofweek * 288 + ts.hour * 12 + ts.minute // 5])


class LoadForecaster:
    """Day-ahead electricity forecast for one building (gradient-boosted trees).

    Every feature is at least 24 h old (same hour yesterday, same hour last week,
    the 24 h mean ending yesterday), so the next 24 hours are predicted directly,
    without feeding predictions back in. Trained on Jan-Oct, tested on Nov-Dec,
    and compared with the seasonal-naive forecast (same hour last week).
    """

    def __init__(self, series):
        from sklearn.ensemble import HistGradientBoostingRegressor
        s = series.astype(float)
        self.s = s
        X, y = self._features(s), s
        ok = X.notna().all(axis=1)
        X, y = X[ok], y[ok]
        split = pd.Timestamp(f"{s.index[0].year}-11-01")
        tr, te = X.index < split, X.index >= split
        self.model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=7)
        self.model.fit(X[tr], y[tr])
        pred = self.model.predict(X[te])
        naive = X[te]["lag168"].values
        denom = np.maximum(np.abs(y[te].values), 0.05 * float(np.abs(y).mean()) + 1e-6)
        self.mape = float(np.mean(np.abs(pred - y[te].values) / denom) * 100)
        self.mape_naive = float(np.mean(np.abs(naive - y[te].values) / denom) * 100)
        self.model.fit(X, y)  # refit on the full year for live forecasts

    @staticmethod
    def _features(s, index=None):
        idx = s.index if index is None else index
        h = idx.hour
        lag24 = s.shift(24).reindex(idx) if index is None else s.reindex(idx - pd.Timedelta(hours=24)).set_axis(idx)
        lag168 = s.shift(168).reindex(idx) if index is None else s.reindex(idx - pd.Timedelta(hours=168)).set_axis(idx)
        roll = s.rolling(24).mean()
        roll24 = roll.shift(24).reindex(idx) if index is None else roll.reindex(idx - pd.Timedelta(hours=24)).set_axis(idx)
        return pd.DataFrame({"sin_h": np.sin(2 * np.pi * h / 24), "cos_h": np.cos(2 * np.pi * h / 24),
                             "dow": idx.dayofweek, "weekend": (idx.dayofweek >= 5).astype(int),
                             "lag24": lag24.values, "lag168": lag168.values, "roll24": roll24.values}, index=idx)

    def forecast(self, t, hours=24):
        """Forecast for the `hours` after dataset timestamp t (exclusive)."""
        idx = pd.date_range(t + pd.Timedelta(hours=1), periods=hours, freq="h")
        X = self._features(self.s, idx).ffill().bfill().fillna(float(self.s.mean()))
        return np.maximum(self.model.predict(X), 0)


def save_metrics(metrics):
    (CACHE / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
