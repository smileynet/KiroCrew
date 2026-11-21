"""Kimi Code CLI's ``fs/*`` requests, answered by Kiro Crew.

Crew declares ``fs.readTextFile`` and ``fs.writeTextFile`` for Kimi (not ``terminal``). Kimi then
sends every text ``Write`` and ``Edit`` as ``fs/write_text_file`` with the whole content, and every
text ``Read`` as ``fs/read_text_file``, and does not touch the disk itself (measured on 2.7.2,
``reports/kimi-levers-acp.md`` rows 3 and 4). That makes this module the second veto point on a
write, after the permission request:

* A write is carried out only if a ``session/request_permission`` for the SAME resolved path was
  APPROVED earlier in the SAME turn, and each approval pays for one write. Anything else is
  refused with ``-32000``. It is never ``-32002``: Kimi reads that as "not found", creates the
  parent directory itself and retries.
* Kimi probes ``<cwd>/.kimi-code/AGENTS.md``, ``<cwd>/AGENTS.md`` and ``<cwd>/agents.md`` through
  ``fs/read_text_file`` many times a turn. Those answer not-found (``-32002``) and nothing is read:
  a project's instruction files are not loaded into a harness Crew did not approve them for.
* Every other read goes through Crew's sensitive-path floor, then a bounded read.

The state lives on :class:`KimiFsGate`, one per ``AcpClient``, and is created on first use.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kiro_crew.security import is_sensitive_path, is_sensitive_write_path

if TYPE_CHECKING:
    from kiro_crew.acp.client import AcpClient
    from kiro_crew.acp.types import JsonRpcMessage

__all__ = [
    "FS_READ_METHOD",
    "FS_WRITE_METHOD",
    "KIMI_AGENTS_PROBES",
    "KIMI_FS_METHODS",
    "KIMI_OPTION_APPROVE_ALWAYS",
    "KIMI_OPTION_APPROVE_ONCE",
    "KimiFsGate",
    "answer_fs_request",
    "approved_write_paths",
    "kimi_fs_gate",
]

FS_READ_METHOD = "fs/read_text_file"
FS_WRITE_METHOD = "fs/write_text_file"
KIMI_FS_METHODS = frozenset({FS_READ_METHOD, FS_WRITE_METHOD})

#: Kimi's permission option ids. "Approve for this session" saves a rule that skips its ask
#: rules, survives ``session/load`` and a new process, and completes later calls with no
#: request, so Crew never sends it and never offers it (``approve_tool``, the event builder).
KIMI_OPTION_APPROVE_ONCE = "approve_once"
KIMI_OPTION_APPROVE_ALWAYS = "approve_always"

#: The files Kimi probes under the session cwd for project instructions, as cwd-relative paths.
KIMI_AGENTS_PROBES = (".kimi-code/AGENTS.md", "AGENTS.md", "agents.md")

#: ACP's "resource not found". Kimi's ``appendText`` treats it as an empty file, and its
#: ``writeText`` retries after creating a missing parent, so a REFUSED write must not use it.
_RESOURCE_NOT_FOUND = -32002
_REFUSED = -32000

#: shortcut: a read answers at most this many bytes, then is refused; Kimi's own Read tool has no
#: smaller limit on the wire. Raise only with a capture of a larger legitimate read.
_MAX_READ_BYTES = 8 * 1024 * 1024

#: The two tools whose approval pays for one ``fs/write_text_file``.
_WRITE_TITLES = frozenset({"Write", "Edit"})
_TEXT_PATH = re.compile(r"Requesting approval to (?:Writing|Editing) (.+)\Z", re.DOTALL)


def _resolve(path: str, cwd: str) -> str:
    """*path* made absolute against *cwd* and canonical (symlinks followed)."""
    return os.path.realpath(os.path.join(cwd, os.path.expanduser(path)))


def approved_write_paths(params: dict[str, Any], cwd: str) -> tuple[str, ...]:
    """The resolved paths a ``Write``/``Edit`` permission request asks to change.

    Kimi names them in the request's own content: an ``Edit`` carries a ``diff`` item with the
    path, and both carry the text ``Requesting approval to Writing|Editing <path>``. The main
    agent's text path may be relative, so each is resolved against the session *cwd*. A request
    for any other tool, or one that names no path, answers ``()``: its approval pays for no write.
    So does one whose paths resolve to different files: the person saw one path and a write
    would land on another, so it fails closed.
    """
    tool_call = params.get("toolCall")
    if not isinstance(tool_call, dict) or tool_call.get("title") not in _WRITE_TITLES:
        return ()
    found: list[str] = []
    for item in tool_call.get("content") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "diff" and isinstance(item.get("path"), str) and item["path"]:
            found.append(item["path"])
        inner = item.get("content")
        text = inner.get("text") if isinstance(inner, dict) else None
        match = _TEXT_PATH.match(text) if isinstance(text, str) else None
        if match and match.group(1):
            found.append(match.group(1))
    resolved = tuple(dict.fromkeys(_resolve(path, cwd) for path in found))
    return resolved if len(resolved) == 1 else ()


class KimiFsGate:
    """What the permission handler learned about writes, for one client.

    ``pending`` maps an open request id to the paths it would pay for; ``approved`` counts, per
    path, the approvals not yet spent on a write. ``new_turn`` empties both.
    """

    def __init__(self) -> None:
        self.pending: dict[str | int, tuple[str, ...]] = {}
        self.approved: dict[str, int] = {}

    def note_request(self, request_id: str | int, params: dict[str, Any], cwd: str) -> None:
        paths = approved_write_paths(params, cwd)
        if paths:
            self.pending[request_id] = paths

    def approve(self, request_id: str | int) -> tuple[str, ...]:
        """Credit the request's paths; the paths credited, for :meth:`undo`."""
        paths = self.pending.pop(request_id, ())
        for path in paths:
            self.approved[path] = self.approved.get(path, 0) + 1
        return paths

    def undo(self, request_id: str | int, paths: tuple[str, ...]) -> None:
        """Take back what :meth:`approve` credited (the answer was never sent)."""
        for path in paths:
            self.spend(path)
        if paths:
            self.pending[request_id] = paths

    def drop(self, request_id: str | int) -> None:
        self.pending.pop(request_id, None)

    def new_turn(self) -> None:
        self.pending.clear()
        self.approved.clear()

    def spend(self, path: str) -> bool:
        """Use one approval for *path*; False when it has none."""
        left = self.approved.get(path, 0)
        if left <= 0:
            return False
        if left == 1:
            del self.approved[path]
        else:
            self.approved[path] = left - 1
        return True


def kimi_fs_gate(client: AcpClient) -> KimiFsGate:
    gate = getattr(client, "_kimi_fs", None)
    if gate is None:
        gate = client._kimi_fs = KimiFsGate()
    return gate


def _read_text(path: str) -> str:
    size = os.stat(path).st_size
    if size > _MAX_READ_BYTES:
        raise OSError(f"file is larger than {_MAX_READ_BYTES} bytes")
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


async def answer_fs_request(client: AcpClient, msg: JsonRpcMessage) -> None:
    """Answer one ``fs/read_text_file`` or ``fs/write_text_file`` from Kimi."""
    params = msg.params if isinstance(msg.params, dict) else {}
    raw_path = params.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        await client._send_error(msg.id, -32602, "path is required")
        return
    cwd = os.path.realpath(str(client._work_dir))
    path = _resolve(raw_path, cwd)
    if msg.method == FS_READ_METHOD:
        await _answer_read(client, msg, path, cwd)
    else:
        await _answer_write(client, msg, path, params.get("content"))


async def _answer_read(client: AcpClient, msg: JsonRpcMessage, path: str, cwd: str) -> None:
    if any(path == _resolve(probe, cwd) for probe in KIMI_AGENTS_PROBES):
        await client._send_error(msg.id, _RESOURCE_NOT_FOUND, "Resource not found")
        return
    if await asyncio.to_thread(is_sensitive_path, path):
        await client._send_error(msg.id, _REFUSED, "Kiro Crew does not serve this path")
        return
    try:
        content = await asyncio.to_thread(_read_text, path)
    except (FileNotFoundError, NotADirectoryError):
        await client._send_error(msg.id, _RESOURCE_NOT_FOUND, "Resource not found")
    except OSError as exc:
        await client._send_error(msg.id, _REFUSED, f"read failed: {exc.strerror or 'error'}")
    else:
        await client._send_response(msg.id, {"content": content})


async def _answer_write(client: AcpClient, msg: JsonRpcMessage, path: str, content: object) -> None:
    if not isinstance(content, str):
        await client._send_error(msg.id, -32602, "content must be text")
        return
    if await asyncio.to_thread(is_sensitive_write_path, path):
        await client._send_error(msg.id, _REFUSED, "write blocked by Kiro Crew policy")
        return
    if not kimi_fs_gate(client).spend(path):
        await client._send_error(
            msg.id, _REFUSED, "write blocked: no approved permission request for this path"
        )
        return
    try:
        await asyncio.to_thread(_write_text, path, content)
    except OSError as exc:
        await client._send_error(msg.id, _REFUSED, f"write failed: {exc.strerror or 'error'}")
    else:
        await client._send_response(msg.id, {})
