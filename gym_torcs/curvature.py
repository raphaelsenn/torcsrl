"""Curvature of a racing line recorded over several laps.

Method adapted from https://github.com/dergestaler/estimate_curvatures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

MIN_LAP_SAMPLES = 50


@dataclass(slots=True)
class RacingLineCurvatureMap:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    curvature: np.ndarray
    track_length: float

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        x_col: str = "x",
        y_col: str = "y",
        dist_col: str = "distFromStart",
        spacing: float = 1.0,
        smoothing_window: int = 21,
        max_step_xy: float = 25.0,
        min_lap_coverage: float = 0.90,
        outlier_sigma: float = 3.5,
        max_abs_curvature: float = 0.5,
    ) -> RacingLineCurvatureMap:
        """Build the map from a CSV with one row per time step of a driven line.

        Each lap is resampled every ``spacing`` meters. Incomplete laps and laps
        with position jumps are dropped, and the remaining laps are merged into
        one line by an outlier robust median before the curvature is computed.
        """
        df = pd.read_csv(csv_path, usecols=[x_col, y_col, dist_col])
        df = df[np.isfinite(df.to_numpy()).all(axis=1)]
        x, y, dist = (df[col].to_numpy(dtype=np.float64) for col in (x_col, y_col, dist_col))

        laps = [lap for lap in _split_laps(dist) if lap.stop - lap.start >= MIN_LAP_SAMPLES]
        if not laps:
            raise ValueError(f"No usable laps found in {csv_path}")

        track_length = float(np.median([dist[lap].max() for lap in laps]))
        if track_length <= 1.0:
            raise ValueError(f"Invalid track length {track_length} in {csv_path}")

        s = np.arange(0.0, track_length, spacing)
        resampled = [
            _resample_lap(dist[lap], x[lap], y[lap], s, track_length, min_lap_coverage, max_step_xy)
            for lap in laps
        ]
        resampled = [lap for lap in resampled if lap is not None]
        if len(resampled) < 2:
            raise ValueError(f"Only {len(resampled)} usable laps found in {csv_path}")

        xs = np.stack([lap[0] for lap in resampled])
        ys = np.stack([lap[1] for lap in resampled])
        x_line, y_line = _robust_median_line(xs, ys, outlier_sigma)

        x_line = _circular_moving_average(x_line, smoothing_window)
        y_line = _circular_moving_average(y_line, smoothing_window)

        curvature = _curvature(x_line, y_line, spacing)
        curvature = np.clip(np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0),
                            -max_abs_curvature, max_abs_curvature)

        return cls(
            s=s.astype(np.float32),
            x=x_line.astype(np.float32),
            y=y_line.astype(np.float32),
            curvature=curvature.astype(np.float32),
            track_length=track_length,
        )

    def lookahead(self, dist_from_start: float, distances: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Curvature at the given distances ahead of ``dist_from_start``."""
        query = (dist_from_start + np.asarray(distances, dtype=np.float64)) % self.track_length
        return np.interp(query, self.s, self.curvature, period=self.track_length).astype(np.float32)


def _split_laps(dist: np.ndarray) -> list[slice]:
    """Split where the distance from the start line jumps back to zero."""
    approx_length = np.percentile(dist, 99.9)
    wraps = np.flatnonzero(np.diff(dist) < -0.5 * approx_length) + 1
    bounds = [0, *wraps, len(dist)]
    return [slice(start, stop) for start, stop in zip(bounds[:-1], bounds[1:])]


def _resample_lap(
    dist: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    track_length: float,
    min_coverage: float,
    max_step: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    order = np.argsort(dist)
    dist, x, y = dist[order], x[order], y[order]

    increasing = np.concatenate([[True], np.diff(dist) > 1e-6])
    dist, x, y = dist[increasing], x[increasing], y[increasing]

    if len(dist) < MIN_LAP_SAMPLES:
        return None
    if (dist[-1] - dist[0]) / track_length < min_coverage:
        return None
    if np.max(np.hypot(np.diff(x), np.diff(y))) > max_step:
        return None

    # Pad with the neighbouring samples across the start line so interpolation wraps.
    dist = np.concatenate([[dist[-1] - track_length], dist, [dist[0] + track_length]])
    x = np.concatenate([[x[-1]], x, [x[0]]])
    y = np.concatenate([[y[-1]], y, [y[0]]])
    return np.interp(s, dist, x), np.interp(s, dist, y)


def _robust_median_line(xs: np.ndarray, ys: np.ndarray, outlier_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-point median over laps, ignoring laps that stray far from the plain median."""
    x_median = np.median(xs, axis=0)
    y_median = np.median(ys, axis=0)

    deviation = np.hypot(xs - x_median, ys - y_median)
    typical = np.median(deviation, axis=0)
    robust_std = 1.4826 * np.median(np.abs(deviation - typical), axis=0) + 1e-6
    inlier = deviation <= typical + outlier_sigma * robust_std

    x_line = np.nanmedian(np.where(inlier, xs, np.nan), axis=0)
    y_line = np.nanmedian(np.where(inlier, ys, np.nan), axis=0)
    return x_line, y_line


def _circular_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad = window // 2
    window = 2 * pad + 1
    padded = np.concatenate([values[-pad:], values, values[:pad]])
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _curvature(x: np.ndarray, y: np.ndarray, spacing: float) -> np.ndarray:
    """Signed curvature of a closed curve from periodic central differences."""
    dx = (np.roll(x, -1) - np.roll(x, 1)) / (2.0 * spacing)
    dy = (np.roll(y, -1) - np.roll(y, 1)) / (2.0 * spacing)
    ddx = (np.roll(x, -1) - 2.0 * x + np.roll(x, 1)) / spacing**2
    ddy = (np.roll(y, -1) - 2.0 * y + np.roll(y, 1)) / spacing**2

    speed_cubed = (dx**2 + dy**2) ** 1.5
    curvature = np.zeros_like(x)
    moving = speed_cubed > 1e-6
    curvature[moving] = (dx * ddy - dy * ddx)[moving] / speed_cubed[moving]
    return curvature
