"""UDP client for the SCR server robot of TORCS.

Protocol adapted from https://github.com/ugo-nama-kun/gym_torcs
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass

from gym_torcs.constants import TRACK_SENSOR_ANGLES

BUFFER_SIZE = 2**17

logger = logging.getLogger(__name__)

Sensors = dict[str, float | list[float]]


@dataclass(slots=True)
class Action:
    steer: float = 0.0
    accel: float = 0.0
    brake: float = 0.0
    gear: int = 1

    def encode(self) -> bytes:
        return (
            f"(accel {self.accel:.3f})(brake {self.brake:.3f})(clutch 0.000)"
            f"(gear {self.gear})(steer {self.steer:.3f})"
            f"(focus -90 -45 0 45 90)(meta 0)"
        ).encode()


def parse_sensors(message: str) -> Sensors:
    """Parse a message like ``(angle 0.01)(track 7.3 8.1 ...)``."""
    sensors: Sensors = {}
    for item in message.strip("\x00 ").strip("()").split(")("):
        parts = item.split()
        if len(parts) < 2:
            continue
        name, *values = parts
        numbers = [float(v) for v in values]
        sensors[name] = numbers[0] if len(numbers) == 1 else numbers
    return sensors


class TorcsClient:
    def __init__(self, host: str, port: int, socket_timeout: float = 2.0) -> None:
        self.address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(socket_timeout)
        self.socket_timeout = socket_timeout
        self.stale_messages = 0

    def connect(self, timeout: float) -> None:
        angles = " ".join(str(a) for a in TRACK_SENSOR_ANGLES)
        handshake = f"SCR(init {angles})".encode()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.sock.sendto(handshake, self.address)
            try:
                data, _ = self.sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                continue
            if b"***identified***" in data:
                return

        raise TimeoutError(f"No SCR server answered on {self.address[0]}:{self.address[1]}.")

    def receive(self, timeout: float, keepalive: Action | None = None) -> Sensors:
        """Wait for the next sensor message.

        While the race is still loading, TORCS gives up on clients that stay
        silent. Passing ``keepalive`` resends that action on every socket timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                data, _ = self.sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                if keepalive is not None:
                    self.send(keepalive)
                continue

            message = self._newest(data).decode(errors="replace")
            if "***shutdown***" in message or "***restart***" in message:
                raise ConnectionError(message.strip("\x00 "))
            if message and "***identified***" not in message:
                return parse_sensors(message)

        raise TimeoutError(f"No sensor data from {self.address[0]}:{self.address[1]}.")

    def _newest(self, data: bytes) -> bytes:
        """Return the newest queued message and drop older ones.

        Older messages pile up when TORCS stops waiting for the client. They are out of date.
        """
        dropped = 0
        self.sock.setblocking(False)
        try:
            while True:
                data, _ = self.sock.recvfrom(BUFFER_SIZE)
                dropped += 1
        except (BlockingIOError, socket.timeout):
            pass
        finally:
            self.sock.settimeout(self.socket_timeout)

        if dropped:
            self.stale_messages += dropped
            logger.warning(
                "Dropped %d old sensor message(s) from %s:%d. TORCS is not waiting for the "
                "client. Check the -t flag.", dropped, *self.address,
            )
        return data

    def send(self, action: Action) -> None:
        self.sock.sendto(action.encode(), self.address)

    def close(self) -> None:
        self.sock.close()
