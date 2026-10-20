"""Suite-wide test configuration.

With BLOCK_NETWORK=1 (set in CI), any attempt to open a network connection
fails the test, so unit tests can't silently depend on live services."""

import os
import socket
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("BLOCK_NETWORK") != "1":
        return

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"Network access in a unit test: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
