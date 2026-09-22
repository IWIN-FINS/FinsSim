from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForceReading:
    raw_value: int | None
    signed_value: int | None
    force_n: float | None
    ok: bool
    error: str = ""


def decode_signed_u16(raw_value: int) -> int:
    raw = int(raw_value) & 0xFFFF
    return raw - 65536 if raw > 32767 else raw


def convert_signed_to_force(
    signed_value: int,
    *,
    divisor: float = 100.0,
    scale: float = 0.98,
    offset_n: float = 0.0,
) -> float:
    return float(signed_value) / float(divisor) * float(scale) + float(offset_n)


def decode_force_register(
    raw_value: int,
    *,
    divisor: float = 100.0,
    scale: float = 0.98,
    offset_n: float = 0.0,
) -> ForceReading:
    signed = decode_signed_u16(raw_value)
    force_n = convert_signed_to_force(signed, divisor=divisor, scale=scale, offset_n=offset_n)
    return ForceReading(raw_value=int(raw_value), signed_value=signed, force_n=force_n, ok=True)


class ModbusForceSensor:
    def __init__(
        self,
        *,
        port: str,
        baudrate: int = 115200,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: int = 1,
        timeout: float = 0.2,
        address: int = 205,
        count: int = 1,
        device_id: int = 1,
        divisor: float = 100.0,
        scale: float = 0.98,
        offset_n: float = 0.0,
    ) -> None:
        self._port = port
        self._baudrate = int(baudrate)
        self._bytesize = int(bytesize)
        self._parity = parity
        self._stopbits = int(stopbits)
        self._timeout = float(timeout)
        self._address = int(address)
        self._count = int(count)
        self._device_id = int(device_id)
        self._divisor = float(divisor)
        self._scale = float(scale)
        self._offset_n = float(offset_n)
        self._client = None

    def connect(self) -> bool:
        from pymodbus.client import ModbusSerialClient

        self._client = ModbusSerialClient(
            port=self._port,
            baudrate=self._baudrate,
            bytesize=self._bytesize,
            parity=self._parity,
            stopbits=self._stopbits,
            timeout=self._timeout,
        )
        return bool(self._client.connect())

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def read(self) -> ForceReading:
        if self._client is None:
            return ForceReading(None, None, None, ok=False, error="sensor is not connected")
        try:
            response = self._read_holding_registers()
        except Exception as exc:
            return ForceReading(None, None, None, ok=False, error=str(exc))

        if response.isError():
            return ForceReading(None, None, None, ok=False, error=str(response))
        if not getattr(response, "registers", None):
            return ForceReading(None, None, None, ok=False, error="empty Modbus response")
        return decode_force_register(
            response.registers[0],
            divisor=self._divisor,
            scale=self._scale,
            offset_n=self._offset_n,
        )

    def _read_holding_registers(self):
        # pymodbus has renamed the unit/slave argument across releases. Try the
        # modern name first, then fall back for older environments.
        assert self._client is not None
        try:
            return self._client.read_holding_registers(
                address=self._address,
                count=self._count,
                device_id=self._device_id,
            )
        except TypeError:
            try:
                return self._client.read_holding_registers(
                    address=self._address,
                    count=self._count,
                    slave=self._device_id,
                )
            except TypeError:
                return self._client.read_holding_registers(
                    address=self._address,
                    count=self._count,
                    unit=self._device_id,
                )
