"""Resource-id negotiation wire messages (run mode 'online').

The server announces the ids of the resource sets available on disk; the
client picks one (an explicit ``--res-index`` or a random available id) and
announces its choice back, so both sides restore the same set. This exchange
happens after both parties have built their layers, outside the model-info
message; run modes 'offline' and 'standard' never negotiate a set id.
"""

from __future__ import annotations

import json
import socket

from src.comm.basic import recv_chunk, send_chunk


def _send_payload(sock: socket.socket, payload: dict) -> int:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return send_chunk(sock, data)


def _recv_payload(sock: socket.socket) -> dict:
    data, _ = recv_chunk(sock)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("invalid resource-id message: not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid resource-id message: not a JSON object")
    return payload


def send_resource_ids(sock: socket.socket, ids) -> int:
    """Announce the available resource-set ids."""
    return _send_payload(sock, {"resource_ids": [int(i) for i in ids]})


def recv_resource_ids(sock: socket.socket) -> list:
    """Receive the announced resource-set ids."""
    payload = _recv_payload(sock)
    try:
        return [int(i) for i in payload["resource_ids"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid resource-id list payload") from exc


def send_resource_id(sock: socket.socket, resource_id: int) -> int:
    """Announce the chosen resource-set id."""
    return _send_payload(sock, {"resource_id": int(resource_id)})


def recv_resource_id(sock: socket.socket) -> int:
    """Receive the chosen resource-set id."""
    payload = _recv_payload(sock)
    try:
        return int(payload["resource_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid resource-id payload") from exc