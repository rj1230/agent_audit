"""Process-lifetime SQLite checkpoint storage for AgentAudit."""

import atexit
import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


CHECKPOINT_DB_PATH = Path(__file__).resolve().parent / "checkpoints.sqlite3"

# Keep LangGraph's built-in safe types and explicitly allow our structured
# response model. Other unregistered Python types remain blocked.
serializer = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("agent.objects", "Answer"),
    ]
)

_connection = sqlite3.connect(
    str(CHECKPOINT_DB_PATH),
    check_same_thread=False,
)

checkpointer = SqliteSaver(
    _connection,
    serde=serializer,
)

try:
    checkpointer.setup()
except BaseException:
    _connection.close()
    raise

# Keep the SQLite connection alive for the application process.
atexit.register(_connection.close)
