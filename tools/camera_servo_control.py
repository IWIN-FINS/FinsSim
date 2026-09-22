#!/usr/bin/env python3
"""Send camera-servo commands to the ROV control bridge."""

from __future__ import annotations

import argparse
import socket
import sys


SERVO_FRAME_HEAD = 0x03
CMD_ENABLE = 0x00
CMD_DISABLE = 0x01
CMD_CENTER = 0x02
CMD_PITCH_UP = 0x03
CMD_PITCH_DOWN = 0x04
CMD_YAW_LEFT = 0x05
CMD_YAW_RIGHT = 0x06

COMMAND_MAP = {
    "enable": CMD_ENABLE,
    "disable": CMD_DISABLE,
    "center": CMD_CENTER,
    "pitch-up": CMD_PITCH_UP,
    "pitch-down": CMD_PITCH_DOWN,
    "yaw-left": CMD_YAW_LEFT,
    "yaw-right": CMD_YAW_RIGHT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=sorted(COMMAND_MAP.keys()),
        help="camera servo command to send",
    )
    parser.add_argument("--host", default="192.168.138.2", help="ROV bridge host")
    parser.add_argument("--port", type=int, default=5000, help="ROV bridge TCP port")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = bytes((SERVO_FRAME_HEAD, COMMAND_MAP[args.command]))
    with socket.create_connection((args.host, args.port), timeout=3.0) as sock:
        sock.sendall(frame)
    print(f"sent {args.command}: {frame.hex(' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
