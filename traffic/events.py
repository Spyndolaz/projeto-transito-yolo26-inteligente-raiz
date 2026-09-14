"""Eventos estruturados persistidos em JSON Lines."""
import json
from datetime import datetime, timezone
from pathlib import Path


def _json_default(value):
    """Converte escalares NumPy/PyTorch para tipos JSON sem perder o evento."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Tipo não serializável: {type(value).__name__}")


class EventLogger:
    def __init__(self, path=None):
        root = Path(__file__).resolve().parents[1]
        self.path = Path(path or root / "logs" / "events.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type, **data):
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": event_type, **data}
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
        return event
