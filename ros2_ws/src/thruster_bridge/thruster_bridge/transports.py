import socket
import time
from abc import ABC, abstractmethod
from typing import Optional


class TransportError(RuntimeError):
    pass


class Transport(ABC):
    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send(self, data: bytes) -> None:
        raise NotImplementedError


class DryRunTransport(Transport):
    def open(self) -> None:
        return

    def close(self) -> None:
        return

    def send(self, data: bytes) -> None:
        return


class UdpTransport(Transport):
    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, int(port))
        self._sock: Optional[socket.socket] = None

    def open(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send(self, data: bytes) -> None:
        if self._sock is None:
            self.open()
        assert self._sock is not None
        self._sock.sendto(data, self._addr)


class TcpTransport(Transport):
    def __init__(self, host: str, port: int, timeout_sec: float, reconnect_interval_sec: float) -> None:
        self._addr = (host, int(port))
        self._timeout_sec = float(timeout_sec)
        self._reconnect_interval_sec = float(reconnect_interval_sec)
        self._sock: Optional[socket.socket] = None
        self._last_connect_attempt = 0.0

    def open(self) -> None:
        now = time.monotonic()
        if now - self._last_connect_attempt < self._reconnect_interval_sec:
            raise TransportError("TCP reconnect interval has not elapsed")
        self._last_connect_attempt = now
        sock = socket.create_connection(self._addr, timeout=self._timeout_sec)
        sock.settimeout(self._timeout_sec)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send(self, data: bytes) -> None:
        if self._sock is None:
            self.open()
        assert self._sock is not None
        try:
            self._sock.sendall(data)
        except OSError as exc:
            self.close()
            raise TransportError(str(exc)) from exc


class SerialTransport(Transport):
    def __init__(self, port: str, baudrate: int, timeout_sec: float) -> None:
        self._port = port
        self._baudrate = int(baudrate)
        self._timeout_sec = float(timeout_sec)
        self._serial = None

    def open(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise TransportError("pyserial is not installed; install python3-serial") from exc
        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            timeout=self._timeout_sec,
            write_timeout=self._timeout_sec,
        )

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def send(self, data: bytes) -> None:
        if self._serial is None:
            self.open()
        try:
            written = self._serial.write(data)
            if written != len(data):
                raise TransportError(f"short serial write: {written}/{len(data)} bytes")
        except OSError as exc:
            self.close()
            raise TransportError(str(exc)) from exc


def make_transport(
    transport_type: str,
    *,
    serial_port: str,
    baudrate: int,
    host: str,
    port: int,
    timeout_sec: float,
    reconnect_interval_sec: float,
) -> Transport:
    normalized = transport_type.strip().lower()
    if normalized == "dry_run":
        return DryRunTransport()
    if normalized == "udp":
        return UdpTransport(host, port)
    if normalized == "tcp":
        return TcpTransport(host, port, timeout_sec, reconnect_interval_sec)
    if normalized == "serial":
        return SerialTransport(serial_port, baudrate, timeout_sec)
    raise ValueError(f"unsupported transport_type {transport_type!r}; use serial, tcp, udp, or dry_run")
