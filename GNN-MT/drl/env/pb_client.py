import json
import socket
import threading
from typing import Any, Optional

from procbridge.const import Keys, StatusCode, Versions
from procbridge.errors import ProtocolError, ServerError
from procbridge.protocol import read_response


def build_request_frame(method: Optional[str], payload: Any = None) -> bytes:
    body = {}
    if method is not None:
        body[Keys.METHOD.value] = method
    if payload is not None:
        body[Keys.PAYLOAD.value] = payload
    json_bytes = json.dumps(body, separators=(",", ":")).encode("utf-8")
    header = bytearray()
    header += b"pb"
    header += Versions.current().value
    header += bytes([StatusCode.REQUEST.value])
    header += b"\x00\x00"
    header += len(json_bytes).to_bytes(4, byteorder="little")
    return bytes(header) + json_bytes


class PersistentClient:
    """Procbridge client that keeps a single TCP connection with TCP_NODELAY."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connection: Optional[socket.socket] = None
        self.connection_lock = threading.Lock()

    def reconnect(self) -> socket.socket:
        self.close_connection()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect((self.host, self.port))
        self.connection = sock
        return sock

    def ensure_connected(self) -> socket.socket:
        if self.connection is None:
            return self.reconnect()
        return self.connection

    def close_connection(self) -> None:
        if self.connection is None:
            return
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.connection.close()
        except OSError:
            pass
        self.connection = None

    def send(self, method: str, payload: Any = None) -> None:
        frame = build_request_frame(method, payload)
        try:
            self.ensure_connected().sendall(frame)
        except (OSError, ConnectionError):
            self.reconnect().sendall(frame)

    def recv(self) -> Any:
        sock = self.ensure_connected()
        code, result = read_response(sock)
        if code == StatusCode.GOOD_RESPONSE:
            return result
        raise ServerError(result)

    def request(self, method: str, payload: Any = None) -> Any:
        with self.connection_lock:
            try:
                self.send(method, payload)
                return self.recv()
            except (OSError, ConnectionError, ProtocolError):
                self.reconnect()
                self.send(method, payload)
                return self.recv()

    def close(self) -> None:
        with self.connection_lock:
            self.close_connection()
