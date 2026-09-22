from __future__ import annotations

import socket
import time
from abc import ABC, abstractmethod
from typing import Optional


class TransportError(RuntimeError):
    pass


class FullDuplexTransport(ABC):
    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def write(self, data: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, max_bytes: int) -> bytes:
        raise NotImplementedError


class DryRunTransport(FullDuplexTransport):
    def open(self) -> None:
        return

    def close(self) -> None:
        return

    def write(self, data: bytes) -> None:
        return

    def read(self, max_bytes: int) -> bytes:
        return b""


class SerialTransport(FullDuplexTransport):
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
            timeout=0.0,
            write_timeout=self._timeout_sec,
        )

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def write(self, data: bytes) -> None:
        if self._serial is None:
            self.open()
        try:
            written = self._serial.write(data)
            if written != len(data):
                raise TransportError(f"short serial write: {written}/{len(data)}")
        except OSError as exc:
            self.close()
            raise TransportError(str(exc)) from exc

    def read(self, max_bytes: int) -> bytes:
        if self._serial is None:
            self.open()
        waiting = int(getattr(self._serial, "in_waiting", 0))
        if waiting <= 0:
            return b""
        return bytes(self._serial.read(min(max_bytes, waiting)))


class TcpTransport(FullDuplexTransport):
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
        sock.setblocking(False)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def write(self, data: bytes) -> None:
        if self._sock is None:
            self.open()
        assert self._sock is not None
        try:
            self._sock.sendall(data)
        except OSError as exc:
            self.close()
            raise TransportError(str(exc)) from exc

    def read(self, max_bytes: int) -> bytes:
        if self._sock is None:
            self.open()
        assert self._sock is not None
        try:
            return self._sock.recv(max_bytes)
        except BlockingIOError:
            return b""
        except OSError as exc:
            self.close()
            raise TransportError(str(exc)) from exc


class UdpTransport(FullDuplexTransport):
    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        remote_host: str,
        remote_port: int,
        timeout_sec: float,
    ) -> None:
        self._bind_addr = (bind_host, int(bind_port))
        self._remote_addr = (remote_host, int(remote_port))
        self._timeout_sec = float(timeout_sec)
        self._sock: Optional[socket.socket] = None

    def open(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(self._bind_addr)
        sock.setblocking(False)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def write(self, data: bytes) -> None:
        if self._sock is None:
            self.open()
        assert self._sock is not None
        try:
            sent = self._sock.sendto(data, self._remote_addr)
            if sent != len(data):
                raise TransportError(f"short UDP write: {sent}/{len(data)}")
        except OSError as exc:
            self.close()
            raise TransportError(str(exc)) from exc

    def read(self, max_bytes: int) -> bytes:
        if self._sock is None:
            self.open()
        assert self._sock is not None
        try:
            return self._sock.recv(max_bytes)
        except BlockingIOError:
            return b""
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
    udp_bind_host: str,
    udp_bind_port: int,
    udp_remote_host: str,
    udp_remote_port: int,
    timeout_sec: float,
    reconnect_interval_sec: float,
) -> FullDuplexTransport:
    normalized = transport_type.strip().lower()
    if normalized == "dry_run":
        return DryRunTransport()
    if normalized == "serial":
        return SerialTransport(serial_port, baudrate, timeout_sec)
    if normalized == "tcp":
        return TcpTransport(host, port, timeout_sec, reconnect_interval_sec)
    if normalized == "udp":
        return UdpTransport(udp_bind_host, udp_bind_port, udp_remote_host, udp_remote_port, timeout_sec)
    raise ValueError(f"unsupported transport {transport_type!r}; use serial, tcp, udp, or dry_run")
