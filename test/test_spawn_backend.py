"""``spawn_run(backend=...)``: a sub-agent starts on the harness its caller names.

One path, pinned end to end: the tool's schema and wire body, the spawn
endpoint's refusal of a backend this gateway cannot serve, the manager carrying
the value through the queue, the run handing it to the provider factory (and
off the shared runtime, which is the parent's process), the factory's selection
gate, the warm pool, and the follow-ups -- a continuation and a retry run where
the original ran.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro_crew.execution_context import execution_for_store
from kiro_crew.validation import SPAWN_RUN_SCHEMA, ValidationError, validate_tool_args

pytestmark = pytest.mark.usefixtures("healthy_host_memory")


@pytest.fixture(autouse=True)
def _close_subagent_managers(close_subagent_managers):
    """Every manager built here is closed at teardown; the body is in ``conftest``."""


@pytest.fixture
def selectable(monkeypatch):
    """Pin the selectable set to Kiro, Claude, Codex and Pi, and this machine's
    installs: Pi is allowed but missing, Codex's probe could not decide."""
    from kiro_crew import acp_backends
    from kiro_crew.agent_sdk import backend_install
    from kiro_crew.agent_sdk.backend_install import (
        INSTALLED,
        MISSING,
        UNKNOWN,
        BackendInstallState,
    )

    values = ["", "claude", "codex", "pi", "kas"]
    installs = {
        "": INSTALLED,
        "claude": INSTALLED,
        "codex": UNKNOWN,
        "pi": MISSING,
        "kas": INSTALLED,
    }
    monkeypatch.setattr(acp_backends, "selectable_backend_values", lambda: sorted(values))
    monkeypatch.setattr(
        backend_install,
        "probe_backend",
        lambda b: BackendInstallState(
            b,
            b or "kiro",
            installs[b],
            missing_components=("pi-acp",) if installs[b] == MISSING else (),
            install_command="npm i -g pi-acp" if installs[b] == MISSING else "",
            # KAS is on disk but was installed after the gateway started.
            restart_required=b == "kas",
        ),
    )
    return values


# ── Schema ────────────────────────────────────────────────────────────────────


class TestSchema:
    @pytest.mark.parametrize("name", ["kiro", "claude", "codex", "open-code_2"])
    def test_a_backend_name_is_accepted(self, name):
        cleaned = validate_tool_args({"task": "x", "backend": name}, SPAWN_RUN_SCHEMA)
        assert cleaned["backend"] == name

    @pytest.mark.parametrize("bad", ["Claude", "co dex", "../kiro", "-x", 3, True])
    def test_a_malformed_name_is_refused(self, bad):
        with pytest.raises(ValidationError):
            validate_tool_args({"task": "x", "backend": bad}, SPAWN_RUN_SCHEMA)

    def test_a_task_entry_may_name_its_own_backend(self):
        cleaned = validate_tool_args(
            {"tasks": [{"task": "x", "backend": "codex"}]}, SPAWN_RUN_SCHEMA
        )
        assert cleaned["tasks"][0]["backend"] == "codex"


# ── The tool's wire body ──────────────────────────────────────────────────────


def _drive(result: Any) -> str:
    """``spawn_run`` returns a :class:`DeferredTool` the stdio loop steps;
    drive it to its text here, as ``_call_tool`` does."""
    from kiro_crew import mcp_core, mcp_shared

    if isinstance(result, mcp_shared.DeferredTool):
        return mcp_shared.drive_deferred(result, clock=mcp_core.time)
    return result


def _run_tool(args: dict[str, Any]) -> list[dict]:
    from kiro_crew import mcp_core

    bodies: list[dict] = []

    def _fake_post(path: str, body: dict) -> dict:
        if path == "/api/spawn":
            bodies.append(body)
        return {"id": f"a{len(bodies)}"}

    with (
        patch.object(mcp_core, "_post", side_effect=_fake_post),
        patch.object(mcp_core, "_resolve_session_key", return_value="dashboard:chat-1"),
        patch.object(mcp_core, "sel", MagicMock()),
    ):
        _drive(mcp_core._call_tool_inner("spawn_run", args))
    return bodies


class TestToolForwarding:
    def test_the_backend_is_sent_batch_wide(self):
        bodies = _run_tool({"tasks": ["t1", "t2"], "backend": "claude"})
        assert [b["backend"] for b in bodies] == ["claude", "claude"]

    def test_a_task_entry_overrides_the_batch_backend(self):
        bodies = _run_tool(
            {"tasks": [{"task": "t1", "backend": "codex"}, "t2"], "backend": "claude"}
        )
        assert [b["backend"] for b in bodies] == ["codex", "claude"]

    def test_no_backend_sends_none(self):
        bodies = _run_tool({"tasks": ["t1", "t2"]})
        assert all("backend" not in b for b in bodies)

    def test_the_tool_advertises_the_field(self):
        from kiro_crew.mcp_tools.spawn import schemas

        spec = next(t for t in schemas() if t["name"] == "spawn_run")
        props = spec["inputSchema"]["properties"]
        assert "backend" in props
        assert "backend" in props["tasks"]["items"]["properties"]

    def test_a_not_installed_refusal_shows_its_install_command(self):
        from kiro_crew import mcp_core

        refusal = {
            "error": "backend 'codex' is not installed on this machine",
            "code": "backend_not_installed",
            "install_command": "npm i -g @agentclientprotocol/codex-acp",
        }
        with (
            patch.object(mcp_core, "_post", return_value=refusal),
            patch.object(mcp_core, "_resolve_session_key", return_value="dashboard:chat-1"),
            patch.object(mcp_core, "sel", MagicMock()),
        ):
            out = _drive(mcp_core._call_tool_inner("spawn_run", {"task": "t1", "backend": "codex"}))
        assert "npm i -g @agentclientprotocol/codex-acp" in str(out)


# ── The spawn endpoint ────────────────────────────────────────────────────────


def _request(body: dict) -> tuple[Any, MagicMock]:
    mgr = MagicMock()
    mgr.spawn.return_value = SimpleNamespace(id="a1", done=False, error="")
    mgr.max_concurrent = 4
    state = SimpleNamespace(subagents=mgr, conversation_log=MagicMock())
    request = MagicMock()
    request.app = {"state": state}
    request.headers = {}

    async def _json() -> dict:
        return body

    request.json = _json
    return request, mgr


def _real_request(body: dict) -> tuple[Any, Any]:
    """Like ``_request``, but on a real manager whose runs never start."""
    mgr = _mgr()
    mgr._run = AsyncMock()  # type: ignore[method-assign]
    state = SimpleNamespace(subagents=mgr, conversation_log=MagicMock())
    request = MagicMock()
    request.app = {"state": state}
    request.headers = {}

    async def _json() -> dict:
        return body

    request.json = _json
    return request, mgr


class TestSpawnEndpoint:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name, reaches",
        [
            pytest.param("claude", "claude", id="test_a_selectable_backend_reaches_spawn"),
            pytest.param("kiro", "", id="test_kiro_is_spelled_by_name_and_reaches_spawn_as_its_id"),
            # The fixture's install probe cannot decide for codex; that must not block.
            pytest.param("codex", "codex", id="test_a_probe_that_could_not_decide_does_not_block"),
        ],
    )
    async def test_an_admitted_backend_reaches_spawn(self, selectable, name, reaches):
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        request, mgr = _real_request({"task": "x", "backend": name})
        resp = await api_spawn(request)
        assert mgr.get(json.loads(resp.text)["id"]).backend == reaches

    @pytest.mark.asyncio
    async def test_the_parent_scope_is_read_before_the_backend_probe(
        self, selectable, monkeypatch, tmp_path
    ):
        """The parent's session ends while the route awaits its backend probe:
        the spawn still carries the project scope the parent had when it asked."""
        from kiro_crew import subagent
        from kiro_crew.dashboard.handlers import messaging

        live = [str(tmp_path)]
        monkeypatch.setattr(
            subagent,
            "_parent_project_resolver",
            lambda key: list(live) if key == "dashboard:p" else [],
        )
        real_probe = messaging._requested_backend

        async def _probe_then_end_parent(raw):
            live.clear()
            return await real_probe(raw)

        monkeypatch.setattr(messaging, "_requested_backend", _probe_then_end_parent)

        async def _persistent(*_a):
            return "persistent"

        monkeypatch.setattr(messaging, "_spawn_request_memory_mode", _persistent)
        request, mgr = _request({"task": "x", "backend": "claude", "parent_session": "dashboard:p"})
        request.app["state"]._slots = {}
        # The parent has no execution record, so the route reads its live selection.
        request.app["state"].sessions = SimpleNamespace(
            get_agent_selection=lambda key: ("template", "")
        )
        response = await messaging.api_spawn(request)
        assert response.status == 200, response.text
        assert not live, "the backend probe did not run"
        assert mgr.spawn.call_args.kwargs["_parent_project_dirs"] == [str(tmp_path)]

    @pytest.mark.asyncio
    async def test_no_backend_reaches_spawn_as_none(self, selectable):
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        request, mgr = _real_request({"task": "x"})
        resp = await api_spawn(request)
        assert mgr.get(json.loads(resp.text)["id"]).backend is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["goose", "nonesuch"])
    async def test_an_unavailable_backend_is_refused_with_the_list(self, selectable, name):
        # "goose" is a backend this build knows but this gateway does not allow;
        # "nonesuch" is not a backend at all. Neither may degrade to the default.
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        request, mgr = _request({"task": "x", "backend": name})
        resp = await api_spawn(request)
        assert resp.status == 400
        mgr.spawn.assert_not_called()
        import json

        payload = json.loads(resp.body)
        assert payload["code"] == "unknown_backend"
        assert payload["backends"] == ["claude", "codex", "kiro"]

    @pytest.mark.asyncio
    async def test_an_allowed_backend_that_is_not_installed_is_refused_with_its_install(
        self, selectable
    ):
        # Allowed by governance is not startable: refused up front, with the
        # command that installs it, instead of failing at cold start.
        import json

        from kiro_crew.dashboard.handlers.messaging import api_spawn

        request, mgr = _request({"task": "x", "backend": "pi"})
        resp = await api_spawn(request)
        assert resp.status == 400
        mgr.spawn.assert_not_called()
        payload = json.loads(resp.body)
        assert payload["code"] == "backend_not_installed"
        assert payload["install_command"] == "npm i -g pi-acp"
        assert "pi" not in payload["backends"]

    @pytest.mark.asyncio
    async def test_a_backend_installed_after_the_gateway_started_is_refused(self, selectable):
        # The gateway still holds the cached absence, so the start would fail.
        import json

        from kiro_crew.dashboard.handlers.messaging import api_spawn

        request, mgr = _request({"task": "x", "backend": "kas"})
        resp = await api_spawn(request)
        assert resp.status == 400
        mgr.spawn.assert_not_called()
        payload = json.loads(resp.body)
        assert payload["code"] == "backend_restart_required"
        assert "kas" not in payload["backends"]


class TestTheParentsCreatedHarnessIsInherited:
    """A session created with ``session_create(backend=...)`` hands its harness to a
    child that names none, resolved at admission, so the run's own ``backend``
    carries it through the durable queue, a restart replay and a continuation."""

    async def _spawn(self, monkeypatch, body: dict) -> MagicMock:
        from kiro_crew.dashboard.handlers import messaging

        async def _persistent(*_a):
            return "persistent"

        monkeypatch.setattr(messaging, "_spawn_request_memory_mode", _persistent)
        request, mgr = _request({"task": "x", "parent_session": "dashboard:w", **body})
        request.app["state"]._slots = {}
        request.app["state"].sessions = SimpleNamespace(
            get_agent_selection=lambda key: ("template", "")
        )
        response = await messaging.api_spawn(request)
        assert response.status == 200, response.text
        return mgr

    @pytest.mark.asyncio
    async def test_a_child_with_no_backend_starts_on_the_parents_pick(
        self, selectable, monkeypatch
    ):
        from kiro_crew import members

        members.note_session_backend_pick("w", "claude")
        mgr = await self._spawn(monkeypatch, {})
        assert mgr.spawn.call_args.kwargs["backend"] == "claude"

    @pytest.mark.asyncio
    async def test_the_childs_own_backend_wins(self, selectable, monkeypatch):
        from kiro_crew import members

        members.note_session_backend_pick("w", "claude")
        mgr = await self._spawn(monkeypatch, {"backend": "kiro"})
        assert mgr.spawn.call_args.kwargs["backend"] == ""

    @pytest.mark.asyncio
    async def test_a_parent_with_no_pick_hands_down_nothing(self, selectable, monkeypatch):
        mgr = await self._spawn(monkeypatch, {})
        assert mgr.spawn.call_args.kwargs["backend"] is None

    def test_a_child_selected_as_another_bot_inherits_nothing(self):
        from kiro_crew import members
        from kiro_crew.dashboard.handlers.messaging import _inherited_backend_name

        members.note_session_backend_pick("w", "codex")
        assert _inherited_backend_name("dashboard:w", member=False) == "codex"
        assert _inherited_backend_name("dashboard:w", member=True) == ""
        assert _inherited_backend_name(None, member=False) == ""


# ── Spawn-time failover (RFC Phase 1) ─────────────────────────────────────────


class TestSpawnFailover:
    """An unavailable chosen backend starts on a later member of the operator's
    ``agent.acp_failover`` chain, and the receipt names both. A disabled, invalid
    or exhausted chain falls through to the usual typed refusal, never Kiro."""

    def _failover(self, monkeypatch, block: dict) -> None:
        """Make the endpoint read *block* as ``agent.acp_failover``.

        A copy of the loaded config carries it, so the cached instance other
        tests read is never mutated.
        """
        import copy
        import dataclasses

        from kiro_crew.config import KiroCrewConfig

        real_load = KiroCrewConfig.load.__func__  # type: ignore[attr-defined]

        def _load(c):
            loaded = copy.copy(real_load(c))
            loaded.agent = dataclasses.replace(loaded.agent, acp_failover=dict(block))
            return loaded

        monkeypatch.setattr(KiroCrewConfig, "load", classmethod(_load))

    async def _spawn(self, body: dict):
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        request, mgr = _real_request(body)
        resp = await api_spawn(request)
        return resp, mgr

    @pytest.mark.asyncio
    async def test_an_uninstalled_backend_starts_on_the_next_and_says_so(
        self, selectable, monkeypatch
    ):
        self._failover(monkeypatch, {"enabled": True, "chain": ["pi", "claude"]})
        resp, mgr = await self._spawn({"task": "x", "backend": "pi"})
        assert resp.status == 200, resp.text
        payload = json.loads(resp.text)
        assert payload["backend"] == "claude"
        assert payload["failover_from"] == "pi"
        info = mgr.get(payload["id"])
        assert (info.backend, info.failover_from) == ("claude", "pi")

    @pytest.mark.asyncio
    async def test_a_restart_required_backend_fails_over(self, selectable, monkeypatch):
        self._failover(monkeypatch, {"enabled": True, "chain": ["kas", "claude"]})
        resp, mgr = await self._spawn({"task": "x", "backend": "kas"})
        assert resp.status == 200, resp.text
        payload = json.loads(resp.text)
        assert (payload["backend"], payload["failover_from"]) == ("claude", "kas")

    @pytest.mark.asyncio
    async def test_kiro_is_a_chain_member_by_its_config_spelling(self, selectable, monkeypatch):
        self._failover(monkeypatch, {"enabled": True, "chain": ["pi", "kiro"]})
        resp, mgr = await self._spawn({"task": "x", "backend": "pi"})
        assert resp.status == 200, resp.text
        payload = json.loads(resp.text)
        assert payload["backend"] == "kiro"
        assert mgr.get(payload["id"]).backend == ""

    @pytest.mark.asyncio
    async def test_a_member_that_cannot_be_selected_is_skipped(self, selectable, monkeypatch):
        # goose is a backend this gateway does not allow: skipped, not fatal.
        self._failover(monkeypatch, {"enabled": True, "chain": ["pi", "goose", "codex"]})
        resp, mgr = await self._spawn({"task": "x", "backend": "pi"})
        assert resp.status == 200, resp.text
        assert json.loads(resp.text)["backend"] == "codex"

    @pytest.mark.asyncio
    async def test_a_member_a_ceiling_denies_later_is_never_tried(self, selectable, monkeypatch):
        # codex was selectable when the chain was written; a ceiling then denies
        # it. The spawn skips it without probing it and starts on claude.
        from kiro_crew.agent_sdk import backend_install

        self._failover(monkeypatch, {"enabled": True, "chain": ["pi", "codex", "claude"]})
        selectable.remove("codex")
        real_probe = backend_install.probe_backend
        probed: list[str] = []

        def _recording_probe(b):
            probed.append(b)
            return real_probe(b)

        monkeypatch.setattr(backend_install, "probe_backend", _recording_probe)
        resp, mgr = await self._spawn({"task": "x", "backend": "pi"})
        assert resp.status == 200, resp.text
        payload = json.loads(resp.text)
        assert (payload["backend"], payload["failover_from"]) == ("claude", "pi")
        assert "codex" not in probed
        assert mgr.get(payload["id"]).backend == "claude"

    @pytest.mark.asyncio
    async def test_a_spawn_that_did_not_hop_names_its_backend_only(self, selectable, monkeypatch):
        self._failover(monkeypatch, {"enabled": True, "chain": ["claude", "codex"]})
        resp, _mgr = await self._spawn({"task": "x", "backend": "claude"})
        payload = json.loads(resp.text)
        assert payload["backend"] == "claude"
        assert "failover_from" not in payload

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "block",
        [
            {"enabled": False, "chain": ["pi", "claude"]},
            # A repeated name makes the block invalid, so it reads as off.
            {"enabled": True, "chain": ["pi", "kas", "pi", "claude"]},
            # Phase 2 triggers are refused, so the block reads as off.
            {"enabled": True, "chain": ["pi", "claude"], "triggers": ["usage_limit"]},
            # The refusal is backend_not_installed; only restart is a trigger.
            {"enabled": True, "chain": ["pi", "claude"], "triggers": ["backend_restart_required"]},
            # Every later member is unavailable: exhausted, never Kiro.
            {"enabled": True, "chain": ["pi", "kas"]},
        ],
        ids=["disabled", "repeated-name", "phase2-trigger", "trigger-omitted", "exhausted"],
    )
    async def test_no_hop_keeps_the_typed_refusal(self, selectable, monkeypatch, block):
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        self._failover(monkeypatch, block)
        request, mgr = _request({"task": "x", "backend": "pi"})
        resp = await api_spawn(request)
        assert resp.status == 400
        mgr.spawn.assert_not_called()
        payload = json.loads(resp.body)
        assert payload["code"] == "backend_not_installed"
        # Only a chain that was actually walked is named.
        walked = (
            block["enabled"] is True
            and block.get("triggers") is None
            and len(set(block["chain"])) == len(block["chain"])
        )
        assert bool(payload["failover_chain"]) is walked

    @pytest.mark.asyncio
    async def test_an_exhausted_chain_is_named_in_the_refusal(self, selectable, monkeypatch):
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        self._failover(monkeypatch, {"enabled": True, "chain": ["pi", "kas"]})
        request, _mgr = _request({"task": "x", "backend": "pi"})
        payload = json.loads((await api_spawn(request)).body)
        # Only the members the walk tried, never the requested backend itself.
        assert payload["failover_chain"] == ["kas"]
        assert "agent.acp_failover also tried ['kas'], and none can start" in payload["error"]

    @pytest.mark.asyncio
    async def test_the_last_chain_member_names_no_tried_members(self, selectable, monkeypatch):
        # claude can start, but it comes BEFORE pi: nothing after pi is tried, so the
        # refusal must not claim the chain has no member that can start.
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        self._failover(monkeypatch, {"enabled": True, "chain": ["claude", "pi"]})
        request, mgr = _request({"task": "x", "backend": "pi"})
        payload = json.loads((await api_spawn(request)).body)
        assert payload["code"] == "backend_not_installed"
        assert payload["failover_chain"] == []
        assert "acp_failover" not in payload["error"]
        mgr.spawn.assert_not_called()

    def test_status_and_list_name_a_hop(self):
        from kiro_crew.dashboard.messaging_api.run_views import _failover_fields
        from kiro_crew.subagent import SubagentInfo

        moved = SubagentInfo(id="a", task="t", backend="", failover_from="pi")
        assert _failover_fields(moved) == {"failover_from": "pi", "backend": "kiro"}
        assert _failover_fields(SubagentInfo(id="b", task="t", backend="claude")) == {}

    def test_spawn_status_and_spawn_list_render_a_hop(self):
        from unittest.mock import patch

        from kiro_crew import mcp_core
        from kiro_crew.mcp_tools.spawn_parts import listing

        status = {
            "id": "a1",
            "done": True,
            "result": "ok",
            "backend": "claude",
            "failover_from": "pi",
        }
        with patch.object(mcp_core, "_get", return_value=status):
            out = listing.spawn_status("spawn_status", {"agent_id": "a1"})
        assert "(ran on 'claude': 'pi' could not start, agent.acp_failover)" in out
        row = {"id": "a1", "task": "t", "done": True, "backend": "claude", "failover_from": "pi"}
        with (
            patch.object(mcp_core, "_get", return_value={"agents": [row]}),
            patch.object(mcp_core, "list_agents", return_value=[]),
        ):
            listed = listing.spawn_list("spawn_list", {})
        assert "failover: pi→claude" in listed

    @pytest.mark.asyncio
    async def test_an_unknown_backend_never_fails_over(self, selectable, monkeypatch):
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        self._failover(monkeypatch, {"enabled": True, "chain": ["nonesuch", "claude"]})
        request, mgr = _request({"task": "x", "backend": "nonesuch"})
        resp = await api_spawn(request)
        assert resp.status == 400
        assert json.loads(resp.body)["code"] == "unknown_backend"

    @pytest.mark.asyncio
    async def test_a_continuation_after_a_hop_stays_on_the_backend_that_ran(
        self, selectable, monkeypatch
    ):
        self._failover(monkeypatch, {"enabled": True, "chain": ["pi", "claude"]})
        resp, mgr = await self._spawn({"task": "x", "backend": "pi"})
        run_id = json.loads(resp.text)["id"]
        assert TestFollowUps()._continue(mgr, monkeypatch, run_id)["backend"] == "claude"


# ── The manager ───────────────────────────────────────────────────────────────


def _mgr():
    from kiro_crew.subagent import SubagentManager

    sessions = MagicMock()
    sessions.get_pid = MagicMock(return_value=None)
    sessions.get_approval_policy = MagicMock(return_value="auto")
    sessions.get_agent = MagicMock(return_value="")
    sessions.get_agent_selection = MagicMock(return_value=("template", ""))
    sessions.has_session = MagicMock(return_value=True)
    sessions.release = MagicMock()
    sessions.reset = AsyncMock()
    ctx = MagicMock()
    ctx.hooks.auto_approve_subagent_spawn = True
    return SubagentManager(sessions=sessions, ctx_builder=ctx)


class TestManager:
    @pytest.mark.asyncio
    async def test_spawn_records_the_backend_on_the_run(self):
        mgr = _mgr()
        mgr._run = AsyncMock()  # type: ignore[method-assign]
        info = mgr.spawn("do the thing", backend="codex")
        assert info is not None and info.backend == "codex"

    @pytest.mark.asyncio
    async def test_no_backend_is_none(self):
        mgr = _mgr()
        mgr._run = AsyncMock()  # type: ignore[method-assign]
        info = mgr.spawn("do the thing")
        assert info is not None and info.backend is None

    def test_a_queued_spawn_starts_on_its_backend(self):
        mgr = _mgr()
        mgr._should_stagger_queue = MagicMock(return_value=(True, False))  # type: ignore[method-assign]
        mgr.spawn("validate this finding", backend="claude")
        assert mgr._queue[0]["backend"] == "claude"
        captured: dict[str, object] = {}
        mgr.spawn = lambda **kw: captured.update(kw)  # type: ignore[method-assign]
        mgr._max_concurrent = 4
        mgr._running_count = 0
        mgr._spawn_stagger_secs = 0.0
        mgr._drain_queue()
        assert captured["backend"] == "claude"


# ── The run hands it to the factory ───────────────────────────────────────────


def _run(backend: str | None) -> tuple[dict, AsyncMock]:
    from kiro_crew.config.loader import AgentConfig, KiroCrewConfig
    from kiro_crew.providers.base import EVENT_COMPLETE, LLMEvent
    from kiro_crew.subagent import SubagentInfo, SubagentManager

    sessions = MagicMock()
    sessions.get_pid = MagicMock(return_value=None)
    sessions.get_approval_policy = MagicMock(return_value="")
    sessions.get_agent = MagicMock(return_value="")
    sessions.get_agent_selection = MagicMock(return_value=("template", ""))
    ctx_builder = MagicMock()
    ctx_builder.build_message = MagicMock(return_value=("msg", None))
    ctx_builder.hooks.auto_approve_subagent_tools = False
    captured: dict = {}
    client = MagicMock()

    async def fake_get_or_create(key, agent=None, approval_policy="", **kwargs):
        captured.update(kwargs)
        return client, True, False

    async def fake_stream(msg):
        yield LLMEvent(kind=EVENT_COMPLETE)

    sessions.get_or_create = fake_get_or_create
    client.stream = fake_stream
    cfg = KiroCrewConfig(agent=AgentConfig(session_sharing=True))
    runner = SubagentManager(sessions=sessions, ctx_builder=ctx_builder)
    shared = AsyncMock(side_effect=AssertionError("shared runtime used for a per-spawn backend"))
    info = SubagentInfo(
        execution_context=execution_for_store(""),
        id="sub1",
        task="test",
        parent_session_key="parent-key",
        backend=backend,
    )
    runner._log_spawned(info)
    sessions.is_session_sharing_eligible = MagicMock(return_value=True)
    with (
        patch.object(runner, "_create_shared_session", shared),
        patch("kiro_crew.config.loader.KiroCrewConfig.load", classmethod(lambda c: cfg)),
    ):
        asyncio.run(runner._run_inner(info, "subagent:sub1"))
    return captured, shared


class TestRun:
    @pytest.mark.parametrize("backend", ["claude", ""])
    def test_the_backend_reaches_the_factory_on_a_process_of_its_own(self, backend):
        # ``""`` (Kiro) is a request too: under a non-Kiro default it must reach
        # the factory, or the run starts on the default harness.
        captured, shared = _run(backend)
        shared.assert_not_called()
        assert captured["backend_override"] == backend

    def test_no_backend_passes_nothing(self):
        captured, _ = _run(None)
        assert "backend_override" not in captured

    def test_sharing_is_ruled_out_by_a_backend(self):
        from kiro_crew.config.loader import AgentConfig, KiroCrewConfig
        from kiro_crew.subagent import SubagentInfo

        mgr = _mgr()
        mgr._sessions.is_session_sharing_eligible = MagicMock(return_value=True)
        cfg = KiroCrewConfig(agent=AgentConfig(session_sharing=True))
        with patch("kiro_crew.config.loader.KiroCrewConfig.load", classmethod(lambda c: cfg)):
            plain = SubagentInfo(id="s1", task="t", parent_session_key="p")
            picked = SubagentInfo(id="s2", task="t", parent_session_key="p", backend="codex")
            assert mgr._should_use_session_sharing(plain) is True
            assert mgr._should_use_session_sharing(picked) is False


# ── The factory's selection gate ──────────────────────────────────────────────


class TestSelectionGate:
    def test_the_override_beats_the_member_route_and_the_default(self, monkeypatch):
        from kiro_crew.agent_sdk import backends
        from kiro_crew.members import select_provider_backend

        monkeypatch.setattr(backends, "selectable_backends", lambda: {"", "claude", "kas"})
        assert select_provider_backend("dashboard:dm-x", "kas", "", backend_override="claude") == (
            "claude"
        )

    def test_kiro_is_an_override_not_unset(self, monkeypatch):
        from kiro_crew.agent_sdk import backends
        from kiro_crew.members import select_provider_backend

        monkeypatch.setattr(backends, "selectable_backends", lambda: {"", "claude"})
        assert select_provider_backend("subagent:a", "", "claude", backend_override="") == ""

    def test_no_override_leaves_selection_as_it_was(self):
        from kiro_crew.members import select_provider_backend

        assert select_provider_backend("subagent:a", "", "claude") == "claude"

    def test_an_override_that_stopped_being_selectable_degrades_to_kiro(self, monkeypatch):
        from kiro_crew.agent_sdk import backends
        from kiro_crew.members import select_provider_backend

        monkeypatch.setattr(backends, "selectable_backends", lambda: {""})
        assert select_provider_backend("subagent:a", "", "", backend_override="codex") == ""

    def test_the_factory_names_the_kwarg(self):
        # Named, not left to ``**_kwargs``: swallowed there, the session would
        # start on the default backend with no error.
        from kiro_crew.config.loader import KiroCrewConfig

        src = inspect.getsource(KiroCrewConfig.create_provider_factory)
        assert "backend_override: str | None = None" in src
        assert "backend_override=backend_override" in src


class TestWarmPool:
    def test_a_backend_override_skips_the_warm_pool(self):
        from kiro_crew import session_allocation

        src = "".join(
            inspect.getsource(sys.modules[name])
            for name in (session_allocation.__name__, *session_allocation._ALLOCATION_OWNERS)
        )
        assert 'pool_decision = "bypass_backend_override"' in src
        assert 'extra_factory_kwargs.get("backend_override") is not None' in src


class TestSuccessorsKeepTheBackend:
    """A reset successor and a compaction restart rebuild from ``allocation_identity``:
    a session started on a per-spawn backend must come back on the same one."""

    def _session(self, backend):
        from kiro_crew.session import _Session

        session = _Session.__new__(_Session)
        session.agent = "kirocrew"
        session.approval_policy = ""
        session.provider = SimpleNamespace(cwd="/w")
        session.requested_model = ""
        session.capability_member = ""
        session.backend_override = backend
        return session

    @pytest.mark.parametrize("backend", ["claude", ""])
    def test_the_identity_carries_the_backend(self, backend):
        from kiro_crew.session_lifecycle import allocation_identity

        owner = SimpleNamespace(get_channel=lambda key: "")
        ident = allocation_identity(owner, "subagent:a", self._session(backend))
        assert ident["backend_override"] == backend

    def test_a_session_without_one_rebuilds_with_the_old_kwargs(self):
        from kiro_crew.session_lifecycle import allocation_identity

        owner = SimpleNamespace(get_channel=lambda key: "")
        ident = allocation_identity(owner, "subagent:a", self._session(None))
        assert "backend_override" not in ident

    def test_registration_stamps_the_backend_the_session_started_on(self):
        from kiro_crew import session_allocation

        src = "".join(
            inspect.getsource(sys.modules[name])
            for name in (session_allocation.__name__, *session_allocation._ALLOCATION_OWNERS)
        )
        assert 'session.backend_override = extra_factory_kwargs.get("backend_override")' in src


# ── The effort verdict is judged on the requested backend ────────────────────


class TestEffortVerdict:
    @pytest.mark.parametrize("backend", ["pi", "goose", "opencode"])
    def test_a_backend_that_advertises_its_effort_gets_no_verdict(self, backend):
        # These report their levels at session/new (goose per model); the registry
        # cannot judge them, so neither line may be printed -- a "dropped" there
        # would be a false report.
        from kiro_crew.subagent import effort_applied_note, effort_drop_reason

        assert effort_drop_reason("some-model", "high", backend=backend) == ""
        assert effort_applied_note("some-model", "high", backend=backend) == ""

    def test_the_applied_note_names_the_channel_the_backend_uses(self, monkeypatch):
        # kiro-cli reads the level from cli.json; an adapter harness gets it as a
        # live config option -- the receipt must name the one that delivers it.
        from kiro_crew import subagent
        from kiro_crew.acp_backends import effort_config_option_id

        monkeypatch.setattr(subagent, "_spawn_effective_model", lambda *a, **kw: "gpt-5")
        monkeypatch.setattr(subagent, "model_supports_effort", lambda m: True)
        codex = subagent.effort_applied_note("gpt-5", "high", backend="codex")
        assert codex == f"gpt-5 → config option {effort_config_option_id('codex')}"
        # KAS is kiro-family but reads no cli.json; its level rides
        # ``effortLevel``, so the receipt must not name a file it never opens.
        kas = subagent.effort_applied_note("gpt-5", "high", backend="kas")
        assert kas == f"gpt-5 → config option {effort_config_option_id('kas')}"
        kiro = subagent.effort_applied_note("gpt-5", "high", backend="")
        assert kiro.endswith(".effort") and "config option" not in kiro
        assert subagent.effort_applied_note("gpt-5", "high") == kiro

    @pytest.mark.parametrize("backend", ["some-harness-nobody-registered"])
    def test_a_backend_that_takes_no_effort_is_reported_dropped(self, monkeypatch, backend):
        # Neither cli.json nor a config option reaches such a harness: the level
        # is never sent, so the receipt must say dropped, not applied.
        from kiro_crew import subagent

        monkeypatch.setattr(subagent, "_spawn_effective_model", lambda *a, **kw: "claude-sonnet")
        monkeypatch.setattr(subagent, "model_supports_effort", lambda m: True)
        assert subagent.effort_applied_note("claude-sonnet", "high", backend=backend) == ""
        assert backend in subagent.effort_drop_reason("claude-sonnet", "high", backend=backend)

    def test_the_model_is_translated_into_the_requested_backends_namespace(self, monkeypatch):
        from kiro_crew.config.loader import KiroCrewConfig
        from kiro_crew.subagent import _spawn_effective_model

        seen: dict = {}

        def _effective(self, agent, override, global_model=None, namespace=None):
            seen["namespace"] = namespace
            return "m"

        monkeypatch.setattr(KiroCrewConfig, "acp_effective_model", _effective)
        monkeypatch.setattr(KiroCrewConfig, "load", classmethod(lambda c: KiroCrewConfig()))
        from kiro_crew.agent_sdk.capabilities import capabilities_for

        assert _spawn_effective_model("m", "", crew_agent="", backend="claude") == "m"
        assert seen["namespace"] == capabilities_for("claude").model_id_namespace
        _spawn_effective_model("m", "", crew_agent="")
        assert seen["namespace"] is None

    @pytest.mark.asyncio
    async def test_the_endpoint_hands_the_backend_to_the_verdict(self, selectable, monkeypatch):
        from kiro_crew.dashboard.handlers import messaging

        calls: list = []
        monkeypatch.setattr(
            messaging, "effort_drop_reason", lambda *a, **kw: calls.append(kw) or ""
        )
        monkeypatch.setattr(messaging, "effort_applied_note", lambda *a, **kw: "")
        request, _mgr = _request({"task": "x", "backend": "claude", "reasoning_effort": "high"})
        request.app["state"].sessions = MagicMock()
        await messaging.api_spawn(request)
        assert calls and calls[0]["backend"] == "claude"


# ── Follow-ups run where the original ran ─────────────────────────────────────


class TestFollowUps:
    def _continue(self, mgr, monkeypatch, conv_id: str) -> dict:
        monkeypatch.setattr(mgr, "_conversation_busy", lambda _k: None)
        monkeypatch.setattr(mgr._sessions, "resumable_sid", lambda _k: "sid-1")
        monkeypatch.setattr(mgr, "_promote_conversation", lambda *_a: None)
        captured: dict[str, object] = {}
        monkeypatch.setattr(mgr, "spawn", lambda *_a, **kw: captured.update(kw))
        mgr.continue_conversation(conv_id, "follow up")
        return captured

    def test_a_continuation_inherits_the_live_runs_backend(self, monkeypatch):
        from kiro_crew.subagent import SubagentInfo

        mgr = _mgr()
        mgr._agents["conv1"] = SubagentInfo(id="conv1", task="t", backend="codex")
        assert self._continue(mgr, monkeypatch, "conv1")["backend"] == "codex"

    def test_a_continuation_after_a_restart_reads_the_recorded_backend(self, monkeypatch):
        mgr = _mgr()
        monkeypatch.setattr("kiro_crew.subagent.read_state", lambda _id: {"backend": "claude"})
        monkeypatch.setattr(mgr, "_inherited_memory_store", lambda _id: "")
        monkeypatch.setattr(
            mgr._continuation._persistence, "read_run_app", lambda _id: "", raising=True
        )
        assert self._continue(mgr, monkeypatch, "gone")["backend"] == "claude"

    def test_a_run_predating_the_field_continues_on_the_default(self, monkeypatch):
        from kiro_crew.subagent import SubagentInfo

        mgr = _mgr()
        mgr._agents["conv2"] = SubagentInfo(id="conv2", task="t")
        assert self._continue(mgr, monkeypatch, "conv2")["backend"] is None

    def test_the_spawn_writes_the_backend_into_the_run_record(self, tmp_path, monkeypatch):
        from kiro_crew import subagent_persistence
        from kiro_crew.subagent import SubagentInfo

        monkeypatch.setattr(subagent_persistence, "_subagents_dir", lambda: tmp_path)
        mgr = _mgr()
        info = SubagentInfo(
            execution_context=execution_for_store(""), id="r3", task="t", backend="codex"
        )
        mgr._log_spawned(info)
        assert subagent_persistence.read_state("r3")["backend"] == "codex"

    def test_the_run_record_carries_the_backend(self, tmp_path, monkeypatch):
        from kiro_crew import subagent_persistence

        monkeypatch.setattr(subagent_persistence, "_subagents_dir", lambda: tmp_path)
        subagent_persistence.create_agent_folder("r1", task="t", backend="codex")
        assert subagent_persistence.read_state("r1")["backend"] == "codex"
        subagent_persistence.create_agent_folder("r2", task="t")
        assert "backend" not in subagent_persistence.read_state("r2")

    @pytest.mark.asyncio
    async def test_a_retry_runs_on_the_original_backend(self):
        from kiro_crew.dashboard.handlers.messaging import api_spawn_retry
        from kiro_crew.execution_context import ExecutionContext, MemoryStoreRef
        from kiro_crew.subagent import SubagentInfo

        mgr = _mgr()
        mgr._run = AsyncMock()  # type: ignore[method-assign]
        mgr._agents["a1"] = SubagentInfo(
            id="a1",
            task="t",
            _raw_task="t",
            parent_session_key="dash:1",
            backend="claude",
            done=True,
            error="boom",
            execution_context=ExecutionContext(None, MemoryStoreRef("default"), "template", "k"),
        )
        request = MagicMock()
        request.app = {"state": SimpleNamespace(subagents=mgr)}
        request.headers = {}
        request.match_info = {"agent_id": "a1"}
        resp = await api_spawn_retry(request)
        retried = mgr.get(json.loads(resp.text)["id"])
        assert retried.backend == "claude"


# ── A Kiro CLI child on a host whose kiro-cli offers v3 ───────────────────────


class TestAKiroCliChildRunsOnV3:
    """``backend: "kiro"`` moves onto KAS under the rule a stored Kiro CLI pick
    follows (``kiro_on_kas_notice.kiro_cli_runs_on_kas``), and the receipt says so.
    Without that evidence, on the bundle route, or for a failover hop it keeps
    the Kiro CLI launch."""

    @pytest.fixture
    def kas_ready(self, selectable, monkeypatch, tmp_path):
        """The ``selectable`` fixture, with KAS installed before the gateway started,
        and a child agent ``plain`` whose file sets no ``deniedCommands``."""
        from kiro_crew import agent_discovery
        from kiro_crew.agent_sdk import backend_install
        from kiro_crew.agent_sdk.backend_install import INSTALLED, BackendInstallState

        agents = tmp_path / "user-agents"
        agents.mkdir()
        (agents / "plain.json").write_text(
            json.dumps({"name": "plain", "tools": ["*"]}), encoding="utf-8"
        )
        monkeypatch.setattr(agent_discovery, "_KIRO_AGENTS_DIR", agents)
        agent_discovery.clear_list_agents_cache()
        real_probe = backend_install.probe_backend

        def _probe(b):
            if b == "kas":
                return BackendInstallState(b, b, INSTALLED)
            return real_probe(b)

        monkeypatch.setattr(backend_install, "probe_backend", _probe)
        return selectable

    @staticmethod
    def _named(body: dict) -> tuple[Any, Any]:
        """``_real_request`` with a session manager, which a named agent's warm-up reads."""
        request, mgr = _real_request(body)
        request.app["state"].sessions = SimpleNamespace(_pool_cwd="")
        return request, mgr

    def _v3(self, monkeypatch, offered: bool) -> None:
        from kiro_crew import kiro_on_kas_notice

        monkeypatch.setattr(kiro_on_kas_notice, "kiro_cli_runs_on_kas", lambda: offered)

    @pytest.mark.asyncio
    async def test_a_kiro_child_starts_on_kas_and_the_receipt_says_so(self, kas_ready, monkeypatch):
        from kiro_crew.dashboard.handlers.messaging import KIRO_CLI_ON_V3_NOTE, api_spawn

        self._v3(monkeypatch, True)
        request, mgr = self._named({"task": "x", "backend": "kiro", "agent": "plain"})
        resp = await api_spawn(request)
        assert resp.status == 200, resp.text
        payload = json.loads(resp.text)
        assert mgr.get(payload["id"]).backend == "kas"
        assert payload["backend"] == "kas"
        assert payload["runtime_note"] == KIRO_CLI_ON_V3_NOTE
        assert "failover_from" not in payload

    @pytest.mark.asyncio
    async def test_without_v3_a_kiro_child_keeps_the_kiro_cli_launch(self, kas_ready, monkeypatch):
        """No evidence, the bundle route or KAS denied all answer False there."""
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        self._v3(monkeypatch, False)
        request, mgr = self._named({"task": "x", "backend": "kiro", "agent": "plain"})
        resp = await api_spawn(request)
        payload = json.loads(resp.text)
        assert mgr.get(payload["id"]).backend == ""
        assert "runtime_note" not in payload

    @pytest.mark.asyncio
    async def test_a_kas_that_needs_a_restart_keeps_the_kiro_cli_launch(
        self, kas_ready, monkeypatch
    ):
        from kiro_crew.agent_sdk import backend_install
        from kiro_crew.agent_sdk.backend_install import INSTALLED, BackendInstallState
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        ready_probe = backend_install.probe_backend
        monkeypatch.setattr(
            backend_install,
            "probe_backend",
            lambda b: (
                BackendInstallState(b, b, INSTALLED, restart_required=True)
                if b == "kas"
                else ready_probe(b)
            ),
        )
        self._v3(monkeypatch, True)
        request, mgr = self._named({"task": "x", "backend": "kiro", "agent": "plain"})
        resp = await api_spawn(request)
        assert mgr.get(json.loads(resp.text)["id"]).backend == ""

    @pytest.mark.asyncio
    async def test_a_policy_that_denies_kas_keeps_the_kiro_cli_launch(self, kas_ready, monkeypatch):
        """``kas`` not selectable: the child is not moved onto a backend whose own
        spawn would be refused."""
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        self._v3(monkeypatch, True)
        kas_ready.remove("kas")
        request, mgr = self._named({"task": "x", "backend": "kiro", "agent": "plain"})
        resp = await api_spawn(request)
        assert mgr.get(json.loads(resp.text)["id"]).backend == ""

    @pytest.mark.asyncio
    async def test_a_child_inheriting_a_kiro_pick_starts_on_kas(self, kas_ready, monkeypatch):
        from kiro_crew import members
        from kiro_crew.dashboard.handlers import messaging

        self._v3(monkeypatch, True)

        async def _persistent(*_a):
            return "persistent"

        monkeypatch.setattr(messaging, "_spawn_request_memory_mode", _persistent)
        members.note_session_backend_pick("w", "kiro")
        request, mgr = _request({"task": "x", "parent_session": "dashboard:w", "agent": "plain"})
        request.app["state"]._slots = {}
        request.app["state"].sessions = SimpleNamespace(
            get_agent_selection=lambda key: ("template", "")
        )
        response = await messaging.api_spawn(request)
        assert response.status == 200, response.text
        assert mgr.spawn.call_args.kwargs["backend"] == "kas"

    @pytest.mark.asyncio
    async def test_a_failover_hop_onto_kiro_is_not_moved(self, kas_ready, monkeypatch):
        """The hop was taken because a backend could not start; moving it could send
        it back there."""
        self._v3(monkeypatch, True)
        TestSpawnFailover()._failover(monkeypatch, {"enabled": True, "chain": ["pi", "kiro"]})
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        request, mgr = _real_request({"task": "x", "backend": "pi"})
        resp = await api_spawn(request)
        payload = json.loads(resp.text)
        assert (payload["backend"], payload["failover_from"]) == ("kiro", "pi")
        assert mgr.get(payload["id"]).backend == ""

    @pytest.mark.asyncio
    async def test_a_named_kas_child_is_untouched(self, kas_ready, monkeypatch):
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        self._v3(monkeypatch, True)
        request, mgr = _real_request({"task": "x", "backend": "kas"})
        resp = await api_spawn(request)
        payload = json.loads(resp.text)
        assert mgr.get(payload["id"]).backend == "kas"
        assert "runtime_note" not in payload

    def test_the_tool_receipt_shows_the_note(self):
        """``spawn_run`` renders the gateway's ``runtime_note`` under the run."""
        from kiro_crew import mcp_core
        from kiro_crew.dashboard.handlers.messaging import KIRO_CLI_ON_V3_NOTE

        posted = {"id": "a1", "status": "spawned", "backend": "kas"}
        posted["runtime_note"] = KIRO_CLI_ON_V3_NOTE
        with patch.object(mcp_core, "_post", return_value=posted):
            out = _drive(mcp_core._call_tool_inner("spawn_run", {"task": "t", "backend": "kiro"}))
        assert KIRO_CLI_ON_V3_NOTE in str(out)


class TestAChildWhoseFileDeniesCommandsStaysOnKiroCli:
    """kiro-cli refuses a ``deniedCommands`` regex itself, while on KAS an entry that
    neither travels as a deny rule nor is read by Kiro Crew's gate refuses nothing.
    So a Kiro CLI child whose own file sets the key is not moved."""

    @pytest.fixture
    def agents(self, tmp_path, monkeypatch):
        from kiro_crew import agent_discovery
        from kiro_crew.config import paths as paths_mod

        user = tmp_path / "user-agents"
        user.mkdir()
        mates = tmp_path / "mates-agents"
        mates.mkdir()
        monkeypatch.setattr(agent_discovery, "_KIRO_AGENTS_DIR", user)
        monkeypatch.setattr(paths_mod, "mate_agents_dir", lambda: mates)
        agent_discovery.clear_list_agents_cache()
        yield user
        agent_discovery.clear_list_agents_cache()

    @staticmethod
    def _write(folder, name: str, spec: dict) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.json").write_text(json.dumps({"name": name, **spec}), encoding="utf-8")

    @pytest.mark.parametrize(
        "tools_settings, stays",
        [
            pytest.param({"execute_bash": {"deniedCommands": ["(rm|shred) .*"]}}, True, id="bash"),
            pytest.param({"shell": {"deniedCommands": ["touch .*"]}}, True, id="shell-alias"),
            pytest.param({"executeBash": {"deniedCommands": []}}, True, id="empty-list"),
            pytest.param({"execute_bash": {"allowedCommands": ["ls .*"]}}, False, id="allow-only"),
            pytest.param({"fs_write": {"deniedPaths": ["/etc"]}}, True, id="write-denied-paths"),
            pytest.param({"fs_read": {"deniedPaths": ["~/.ssh"]}}, True, id="read-denied-paths"),
            pytest.param({"grep": {"deniedPaths": ["/secret"]}}, True, id="grep-denied-paths"),
            pytest.param({"fs_write": {"allowedPaths": ["/tmp/**"]}}, False, id="write-allow"),
            pytest.param({"web_fetch": {"blocked": ["evil\\.com"]}}, True, id="fetch-blocked"),
            pytest.param({"web_fetch": {"trusted": ["https://x"]}}, False, id="fetch-trusted"),
            pytest.param({"subagent": {"availableAgents": ["a*"]}}, False, id="sub-available"),
            pytest.param({"crew": {"availableAgents": []}}, False, id="crew-alias"),
            pytest.param({"subagent": {"trustedAgents": ["a"]}}, False, id="sub-trusted"),
            pytest.param({"use_aws": {"deniedServices": ["iam"]}}, True, id="aws-denied"),
            pytest.param({"aws": {"allowedServices": ["s3"]}}, False, id="aws-allow"),
            pytest.param({"some_new_tool": {"deny": ["x"]}}, True, id="unknown-tool"),
            pytest.param({"some_new_tool": {}}, False, id="unknown-tool-empty"),
            pytest.param({"shell": "deny"}, True, id="not-an-object"),
            pytest.param(None, False, id="no-tools-settings"),
        ],
    )
    def test_the_table(self, agents, tools_settings, stays):
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        spec = {"tools": ["*"]}
        if tools_settings is not None:
            spec["toolsSettings"] = tools_settings
        self._write(agents, "child", spec)
        assert _child_file_needs_kiro_cli("child", ()) is stays

    def test_a_missing_file_does_not_hold_the_child(self, agents):
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        assert _child_file_needs_kiro_cli("ghost", ()) is False

    def test_the_checkout_file_kiro_cli_resolves_first_decides(self, agents, tmp_path):
        """kiro-cli resolves ``--agent`` in the child's ``.kiro/agents`` first."""
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        self._write(agents, "child", {"tools": ["*"]})
        project = tmp_path / "proj"
        denied = {"toolsSettings": {"execute_bash": {"deniedCommands": ["rm .*"]}}}
        self._write(project / ".kiro" / "agents", "child", {"tools": ["*"], **denied})
        assert _child_file_needs_kiro_cli("child", (str(project),)) is True
        assert _child_file_needs_kiro_cli("child", ()) is False

    def test_any_spec_that_could_be_the_child_holds_it(self, agents, tmp_path):
        """Which file a launch runs depends on scope order, declared names and pins,
        and the engines differ, so every candidate counts: a second user-level file
        declaring the name, a checkout file, or one in the pool's work dir."""
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        denied = {"toolsSettings": {"execute_bash": {"deniedCommands": ["touch \\p{L}+"]}}}
        self._write(agents, "child", {"tools": ["*"]})
        (agents / "zz-other.json").write_text(
            json.dumps({"name": "child", "tools": ["*"], **denied}), encoding="utf-8"
        )
        assert _child_file_needs_kiro_cli("child", ()) is True
        (agents / "zz-other.json").unlink()
        pool = tmp_path / "pool"
        self._write(pool / ".kiro" / "agents", "child", {"tools": ["*"], **denied})
        assert _child_file_needs_kiro_cli("child", ("", str(pool))) is True
        assert _child_file_needs_kiro_cli("child", ()) is False

    def test_two_candidates_hold_even_with_no_deny(self, agents, tmp_path):
        """Two specs that could carry the name: which one a launch runs is not this
        check's to decide, so the child stays even when neither sets a deny."""
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        self._write(agents, "child", {"tools": ["*"]})
        assert _child_file_needs_kiro_cli("child", ()) is False
        self._write(tmp_path / "mates-agents", "child", {"tools": ["*"]})
        assert _child_file_needs_kiro_cli("child", ()) is True

    def test_one_folder_named_twice_is_one_candidate(self, agents, monkeypatch):
        from kiro_crew.config import paths as paths_mod
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        monkeypatch.setattr(paths_mod, "mate_agents_dir", lambda: agents)
        self._write(agents, "child", {"tools": ["*"]})
        assert _child_file_needs_kiro_cli("child", ()) is False

    @pytest.mark.asyncio
    async def test_the_parent_s_project_dirs_reach_the_check(
        self, agents, selectable, monkeypatch, tmp_path
    ):
        """Discovery reads the parent session's project dirs too, so a copy there that
        denies commands holds the child (a user file of the same name sets none)."""
        from kiro_crew import kiro_on_kas_notice
        from kiro_crew.dashboard.handlers import messaging
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        parent_slot = tmp_path / "slot"
        denied = {"toolsSettings": {"shell": {"deniedCommands": ["rm .*"]}}}
        self._write(parent_slot / ".kiro" / "agents", "dup", {"tools": ["*"], **denied})
        self._write(agents, "dup", {"tools": ["*"]})
        monkeypatch.setattr(messaging, "parent_project_dirs", lambda _key: (str(parent_slot),))
        monkeypatch.setattr(
            messaging, "_spawn_request_memory_mode", AsyncMock(return_value="persistent")
        )
        monkeypatch.setattr(kiro_on_kas_notice, "kas_backend_offered", lambda: True)
        monkeypatch.setattr(kiro_on_kas_notice, "kiro_cli_runs_on_kas", lambda: True)
        request, mgr = _real_request(
            {"task": "x", "backend": "kiro", "agent": "dup", "parent_session": "p:1"}
        )
        request.app["state"].sessions = SimpleNamespace(_pool_cwd="")
        resp = await api_spawn(request)
        assert resp.status == 200, resp.text
        assert mgr.get(json.loads(resp.text)["id"]).backend == ""

    def test_a_same_mtime_rewrite_is_read_fresh(self, agents):
        """A deny added by a timestamp-preserving copy is seen: no catalog snapshot."""
        import os

        from kiro_crew.agent_discovery import parsed_agent_specs
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        self._write(agents, "child", {"tools": ["*"]})
        target = agents / "child.json"
        stamp = target.stat()
        parsed_agent_specs(agents, operation="test", source="test")  # warm the snapshot
        assert _child_file_needs_kiro_cli("child", ()) is False
        denied = {"toolsSettings": {"shell": {"deniedCommands": ["touch \\p{L}+"]}}}
        self._write(agents, "child", {"tools": ["*"], **denied})
        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert _child_file_needs_kiro_cli("child", ()) is True

    def test_an_unreadable_own_file_holds(self, agents):
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        (agents / "child.json").write_text("{not json", encoding="utf-8")
        assert _child_file_needs_kiro_cli("child", ()) is True

    def test_any_checkout_candidate_holds(self, agents, tmp_path):
        """KAS takes only a few keys of a checkout's spec (no ``toolsSettings``, no
        ``allowedTools``, no hooks unless consented), so a checkout candidate holds
        even when it sets no deny: ``denyByDefault`` and ``fallbackAction`` included."""
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        project = tmp_path / "proj"
        narrow = {"toolsSettings": {"shell": {"denyByDefault": True, "allowedCommands": []}}}
        self._write(project / ".kiro" / "agents", "child", {"tools": ["*"], **narrow})
        assert _child_file_needs_kiro_cli("child", (str(project),)) is True
        self._write(project / ".kiro" / "agents", "child", {"tools": ["*"]})
        assert _child_file_needs_kiro_cli("child", (str(project),)) is True
        assert _child_file_needs_kiro_cli("other", (str(project),)) is False

    def test_a_bot_s_own_file_counts(self, agents, tmp_path):
        """KAS resolves a live Bot's file before a same-named checkout agent."""
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        denied = {"toolsSettings": {"shell": {"deniedCommands": ["touch \\p{L}+"]}}}
        self._write(tmp_path / "mates-agents", "scout", {"tools": ["*"], **denied})
        assert _child_file_needs_kiro_cli("scout", ()) is True

    def test_a_home_relative_work_dir_is_expanded(self, agents, tmp_path, monkeypatch):
        """The spawn gate expands ``~`` before the child runs there, so the check does."""
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        monkeypatch.setenv("HOME", str(tmp_path))
        denied = {"toolsSettings": {"shell": {"deniedCommands": ["rm .*"]}}}
        self._write(tmp_path / "repo" / ".kiro" / "agents", "child", {"tools": ["*"], **denied})
        assert _child_file_needs_kiro_cli("child", ("~/repo",)) is True

    def test_a_relative_work_dir_holds_the_child(self, agents):
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        assert _child_file_needs_kiro_cli("child", ("repo",)) is True

    def test_an_unnamed_child_is_held(self, agents):
        """A child that names no agent runs its parent's or a Bot's template, which
        admission resolves later, so its spec cannot be read here: it stays."""
        from kiro_crew.dashboard.handlers.messaging import _child_file_needs_kiro_cli

        self._write(agents, "kirocrew", {"tools": ["*"]})
        assert _child_file_needs_kiro_cli("", ()) is True

    @pytest.mark.asyncio
    async def test_an_unnamed_or_bot_kiro_child_keeps_the_kiro_cli_launch(
        self, agents, selectable, monkeypatch
    ):
        """No ``agent``, or a ``crew`` / ``target_member`` child: the endpoint hands the
        check no agent, so the child stays (the parent's template may deny commands)."""
        from kiro_crew import kiro_on_kas_notice
        from kiro_crew.dashboard.handlers.messaging import api_spawn

        monkeypatch.setattr(kiro_on_kas_notice, "kas_backend_offered", lambda: True)
        monkeypatch.setattr(kiro_on_kas_notice, "kiro_cli_runs_on_kas", lambda: True)
        request, mgr = _real_request({"task": "x", "backend": "kiro"})
        resp = await api_spawn(request)
        assert resp.status == 200, resp.text
        assert mgr.get(json.loads(resp.text)["id"]).backend == ""

    @pytest.mark.asyncio
    async def test_such_a_kiro_child_keeps_the_kiro_cli_launch(
        self, agents, selectable, monkeypatch
    ):
        from kiro_crew import kiro_on_kas_notice
        from kiro_crew.agent_sdk import backend_install
        from kiro_crew.agent_sdk.backend_install import INSTALLED, BackendInstallState
        from kiro_crew.dashboard.handlers.messaging import _kiro_cli_child_on_v3

        real_probe = backend_install.probe_backend
        monkeypatch.setattr(
            backend_install,
            "probe_backend",
            lambda b: BackendInstallState(b, b, INSTALLED) if b == "kas" else real_probe(b),
        )
        monkeypatch.setattr(kiro_on_kas_notice, "kiro_cli_runs_on_kas", lambda: True)
        denied = {"toolsSettings": {"execute_bash": {"deniedCommands": ["(rm|shred) .*"]}}}
        self._write(agents, "narrow", {"tools": ["*"], **denied})
        self._write(agents, "plain", {"tools": ["*"]})
        assert await _kiro_cli_child_on_v3("", None, "narrow", ()) == ("", "")
        assert (await _kiro_cli_child_on_v3("", None, "plain", ()))[0] == "kas"
