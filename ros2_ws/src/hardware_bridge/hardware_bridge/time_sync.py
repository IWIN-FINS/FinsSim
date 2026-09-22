from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import median


UINT32_MODULUS = 1 << 32
UINT32_HALF_RANGE = 1 << 31


@dataclass(frozen=True)
class McuClockEstimate:
    """Mapped time for one packet plus diagnostics for that mapping."""

    mcu_time_ms_unwrapped: int
    mapped_ros_time_sec: float
    offset_sec: float
    clock_scale: float
    receive_delay_sec: float


class McuClockMapper:
    """Map an MCU millisecond clock to ROS time using one shared packet clock.

    The MCU timestamp is unwrapped before fitting an affine clock model. A
    median intercept over the recent window makes the estimate less sensitive
    to serial/UDP receive jitter, while the slope captures slow clock drift.
    """

    def __init__(self, window_size: int = 128) -> None:
        self._window: deque[tuple[float, float]] = deque(maxlen=max(4, int(window_size)))
        self._last_raw: int | None = None
        self._last_unwrapped_ms: int | None = None
        self._wrap_count = 0
        self._out_of_order_count = 0
        self._last_estimate: McuClockEstimate | None = None

    def _unwrap(self, raw_value: int) -> int:
        raw = int(raw_value) & 0xFFFFFFFF
        if self._last_raw is not None:
            forward_delta = (raw - self._last_raw) & 0xFFFFFFFF
            if forward_delta >= UINT32_HALF_RANGE:
                self._out_of_order_count += 1
                assert self._last_unwrapped_ms is not None
                return self._last_unwrapped_ms
            if forward_delta < UINT32_HALF_RANGE and raw < self._last_raw:
                self._wrap_count += 1
        self._last_raw = raw
        self._last_unwrapped_ms = self._wrap_count * UINT32_MODULUS + raw
        return self._last_unwrapped_ms

    def update(self, mcu_time_ms: int, receive_ros_time_sec: float) -> McuClockEstimate:
        receive_sec = float(receive_ros_time_sec)
        if not math.isfinite(receive_sec):
            raise ValueError("receive_ros_time_sec must be finite")
        previous_unwrapped_ms = self._last_unwrapped_ms
        unwrapped_ms = self._unwrap(mcu_time_ms)
        mcu_sec = unwrapped_ms * 1e-3
        if previous_unwrapped_ms is not None and unwrapped_ms <= previous_unwrapped_ms:
            # Duplicate/out-of-order packets diagnose latency but must not
            # move the fitted MCU clock backward or bias its offset.
            pass
        else:
            self._window.append((mcu_sec, receive_sec))

        scale = 1.0
        if len(self._window) >= 3:
            x_mean = sum(x for x, _ in self._window) / len(self._window)
            y_mean = sum(y for _, y in self._window) / len(self._window)
            denominator = sum((x - x_mean) ** 2 for x, _ in self._window)
            if denominator > 1e-12:
                covariance = sum((x - x_mean) * (y - y_mean) for x, y in self._window)
                scale = max(0.99, min(1.01, covariance / denominator))

        # The median intercept rejects occasional receive-time spikes.
        intercepts = [receive - scale * mcu for mcu, receive in self._window]
        offset = float(median(intercepts))
        # Never publish a sample timestamp in the future relative to packet
        # receipt. This keeps host_receive_time_ns a valid upper bound even
        # when the current packet arrives faster than the recent median delay.
        mapped = min(scale * mcu_sec + offset, receive_sec)
        estimate = McuClockEstimate(
            mcu_time_ms_unwrapped=unwrapped_ms,
            mapped_ros_time_sec=mapped,
            offset_sec=offset,
            clock_scale=scale,
            receive_delay_sec=max(0.0, receive_sec - mapped),
        )
        self._last_estimate = estimate
        return estimate

    def diagnostics(self) -> dict[str, float | int | None]:
        estimate = self._last_estimate
        if estimate is None:
            return {
                "sample_count": 0,
                "wrap_count": self._wrap_count,
                "out_of_order_count": self._out_of_order_count,
                "offset_sec": None,
                "clock_scale": None,
                "receive_delay_sec": None,
            }
        return {
            "sample_count": len(self._window),
            "wrap_count": self._wrap_count,
            "out_of_order_count": self._out_of_order_count,
            "offset_sec": estimate.offset_sec,
            "clock_scale": estimate.clock_scale,
            "receive_delay_sec": estimate.receive_delay_sec,
            "mcu_time_ms_unwrapped": estimate.mcu_time_ms_unwrapped,
            "mapped_ros_time_sec": estimate.mapped_ros_time_sec,
        }
