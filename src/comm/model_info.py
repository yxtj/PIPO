"""Model-info wire message (server -> client).

The server parses the model and sends its structural IR right after the client
connects, so the client can build its protocol layers without loading a model.
Uses the same canonical JSON encoding as ``model_info_io`` over the existing
length-prefixed chunk framing.

Resource-set selection is a separate exchange (``src.comm.resource``) used only
in run mode 'online'; it is intentionally not part of this message.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass

from src.comm.basic import recv_chunk, send_chunk
from src.common.model_info_io import model_info_from_dict, model_info_to_dict
from src.common.types import ModelInfo


@dataclass
class ModelInfoMessage:
    model_name: str
    info: ModelInfo


def send_model_info(sock: socket.socket, model_name: str, info: ModelInfo) -> int:
    """Serialize ``(model_name, info)`` and send it as one framed chunk."""
    payload = {
        "model_name": model_name,
        "info": model_info_to_dict(info),
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return send_chunk(sock, data)


def recv_model_info(sock: socket.socket) -> ModelInfoMessage:
    """Receive and validate the server's model-info message."""
    data, _ = recv_chunk(sock)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("invalid model-info message: not valid JSON") from exc
    try:
        model_name = str(payload["model_name"])
        info = model_info_from_dict(payload["info"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid model-info payload") from exc
    return ModelInfoMessage(model_name=model_name, info=info)