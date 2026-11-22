"""Contract tests for the dormant Kimi Code CLI harness.

The wire facts asserted here were observed on Kimi Code CLI 2.7.2. The handshake and
``mode`` ones are rebuilt below as small synthetic frames; the permission request, the
elicitation, ``session/load``, the signed-out state and the unasked Write are pinned
against the live corpus in ``test/fixtures/acp_frames/kimi/``. Every assertion about
behaviour the harness was not observed to have is a ``TODO(capture)`` in the code and
an entry in the Kimi worked example of harness-onboarding.md.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp import client_kimi_session
from kiro_crew.acp import launch as launch_mod
from kiro_crew.acp.client import PROTOCOL_VERSION_KIMI, AcpClient, KimiAcpClient
from kiro_crew.acp.harness import kimi as kimi_mod
from kiro_crew.acp.harness import kimi_fs as kimi_fs_mod
from kiro_crew.acp.harness.kimi import KIMI_AGENT_NAME
from kiro_crew.acp.harness import process_adapter_for
from kiro_crew.acp.harness.base import SpawnContext
from kiro_crew.acp.no_model_provider import AcpNoModelProvider
from kiro_crew.acp.transport_errors import AcpError, AcpToolGateUnroutable
from kiro_crew.acp.types import PROVIDER_LABEL_BY_BACKEND, JsonRpcMessage
from kiro_crew.agent_sdk import backend_install, host_auth
from kiro_crew.agent_sdk import tool_gate as gate
from kiro_crew.agent_sdk.backends import (
    ACP_BACKEND_FORBIDDEN_MODES,
    ACP_BACKEND_KIMI,
    ACP_BACKEND_LAUNCH,
    ACP_BACKEND_PROMPT_COMMAND_ALLOWLIST,
    ACP_BACKENDS_KNOWN,
    ACP_BACKENDS_SELF_SERVED_ACP,
    BASELINE_SELECTABLE_BACKENDS,
    KIMI_VERIFIED_VERSION,
    permission_config_for,
    prompt_command_issue,
    session_config_drift_issue,
)
from kiro_crew.agent_sdk.drivers import acp as acp_driver
from kiro_crew.security.paths import is_sensitive_path

KIMI = ACP_BACKEND_KIMI


def _initialize(name: str = "Kimi Code CLI", version: str = KIMI_VERIFIED_VERSION) -> dict:
    """The ``initialize`` result Kimi Code CLI 2.7.2 answered, reduced to what is read."""
    return {
        "protocolVersion": 1,
        "agentCapabilities": {"loadSession": True},
        "authMethods": [{"id": "login", "type": "terminal"}],
        "agentInfo": {"name": name, "version": version},
    }


def _mode_option(value: str = "default") -> dict:
    return {
        "type": "select",
        "id": "mode",
        "name": "Mode",
        "category": "mode",
        "currentValue": value,
        "options": [{"value": v} for v in ("default", "plan", "auto", "yolo")],
    }


def _handshake_client(init: dict) -> AcpClient:
    client = AcpClient(acp_backend=KIMI)
    client._initialize_result = init
    client._agent_version = init["agentInfo"]["version"] if "agentInfo" in init else ""
    client._agent_version_read = True
    return client


def _mode_update(mode: str) -> JsonRpcMessage:
    return JsonRpcMessage(
        method="session/update",
        params={
            "sessionId": "s",
            "update": {"sessionUpdate": "current_mode_update", "currentModeId": mode},
        },
    )


# ── dormancy and vocabulary ──────────────────────────────────────────────────


def test_kimi_is_known_and_selectable() -> None:
    assert KIMI in ACP_BACKENDS_KNOWN
    assert KIMI in BASELINE_SELECTABLE_BACKENDS
    assert PROVIDER_LABEL_BY_BACKEND[KIMI] == "kimi"


def test_kimi_joined_no_capability_set_without_a_capture() -> None:
    """Membership beyond these needs a live capture, so none is held."""
    from kiro_crew.agent_sdk import backends

    decided = {
        "ACP_BACKENDS_KNOWN",
        "ACP_BACKENDS_SELF_SERVED_ACP",
        "ACP_BACKENDS_ACP_CLIENT_SPAWNABLE",
        # Its read-only tools run with no permission request (``HARNESS_UNASKED_TOOL_IDS``).
        "ACP_BACKENDS_UNASKED_NATIVE_TOOLS",
        # Its mcpServers array carries Crew's servers (``providers.mirrors.kimi``).
        "ACP_BACKENDS_SESSION_MCP_ARRAY",
    }
    joined = {
        name
        for name, value in vars(backends).items()
        if name.startswith("ACP_BACKENDS_") and isinstance(value, frozenset) and KIMI in value
    }
    assert joined == decided


# ── launch ───────────────────────────────────────────────────────────────────


def test_the_launch_row_is_the_pinned_kimi_code_cli() -> None:
    row = ACP_BACKEND_LAUNCH[KIMI]
    assert (row.binary, row.acp_args, row.bin_env_var) == ("kimi", ("acp",), "KIMI_BIN")
    assert KIMI_VERIFIED_VERSION == "2.7.2"
    assert row.install_command == "npm i -g @moonshot-ai/kimi-code@2.7.2"
    assert KIMI in ACP_BACKENDS_SELF_SERVED_ACP
    assert row.spawn_label == "kimi acp"


def test_the_handshake_dialect_is_its_own_constant() -> None:
    from kiro_crew.acp.client import _PROTOCOL_VERSION_BY_BACKEND

    assert PROTOCOL_VERSION_KIMI == 1
    assert _PROTOCOL_VERSION_BY_BACKEND[KIMI] == PROTOCOL_VERSION_KIMI


def test_the_client_class_and_adapter_are_registered() -> None:
    assert type(AcpClient(acp_backend=KIMI)) is KimiAcpClient
    assert isinstance(process_adapter_for(KIMI), kimi_mod.KimiLaunch)


def test_the_child_env_is_crews_home_no_telemetry_no_self_update_and_no_watch() -> None:
    launch = kimi_mod.KimiLaunch()
    launch._home = "/crew/kimi-home/abc"
    # An operator's own KIMI_CODE_HOME names where the sign-in was copied FROM; it must not survive.
    # An inherited KIMI_CODE_TRUST_WORKSPACE would trust every repo; it must not survive either.
    env: dict[str, str] = {
        "KIMI_CODE_HOME": "/home/operator/.kimi-code",
        "KIMI_CODE_TRUST_WORKSPACE": "1",
    }
    launch.apply_spawn_env(env)
    assert env == {
        "KIMI_CODE_HOME": "/crew/kimi-home/abc",
        "KIMI_DISABLE_TELEMETRY": "1",
        "KIMI_CODE_NO_AUTO_UPDATE": "1",
        "KIMI_CODE_WATCH": "0",
        "KIMI_CODE_EXPERIMENTAL_TOOL_SELECT": "1",
    }


def test_a_child_without_crews_home_is_refused() -> None:
    with pytest.raises(AcpToolGateUnroutable):
        kimi_mod.KimiLaunch().apply_spawn_env({})


@pytest.mark.asyncio
async def test_the_spawn_resolves_the_binary_and_arms_the_unasked_tripwire(
    tmp_path, monkeypatch
) -> None:
    resolved = AsyncMock(
        return_value=("/opt/bin/kimi", ["/opt/bin/kimi", "acp"], "kimi acp", "kimi")
    )
    monkeypatch.setattr(launch_mod, "resolve_self_served_launch", resolved)
    monkeypatch.setattr(launch_mod, "_run_preflight_bounded", AsyncMock(return_value=()))
    monkeypatch.setattr(kimi_mod, "kimi_node_found", lambda _bin: "")
    monkeypatch.setattr(kimi_mod, "config_dir", lambda: tmp_path / "crew")
    # A home that ``KIMI_CODE_HOME=<home> kimi login`` signed in.
    (tmp_path / "crew" / "kimi-home" / "credentials").mkdir(parents=True)
    (tmp_path / "crew" / "kimi-home" / "credentials" / "kimi-code.json").write_text("{}")
    client = AcpClient(work_dir=tmp_path, acp_backend=KIMI)
    ctx = SpawnContext.__new__(SpawnContext)
    object.__setattr__(ctx, "session", client)
    object.__setattr__(ctx, "work_dir", tmp_path)
    object.__setattr__(ctx, "sandbox_mode", "auto")
    object.__setattr__(ctx, "environ", {})
    object.__setattr__(ctx, "home", tmp_path / "userhome")

    warmed = AsyncMock()
    monkeypatch.setattr(client, "_prepare_session_mcp", warmed)
    launch = kimi_mod.KimiLaunch()
    plan = await launch.resolve_spawn(ctx)

    # The mirror reads the agent spec into the session's MCP cache before ``session/new``.
    warmed.assert_awaited_once()
    assert plan.argv == ["/opt/bin/kimi", "acp"]
    assert (plan.spawn_label, plan.stderr_label) == ("kimi acp", "kimi")
    resolved.assert_awaited_once_with(KIMI)
    # Fail-closed outside ``HARNESS_UNASKED_TOOL_IDS[kimi]``: every other call that completes
    # with no permission request ends the session.
    assert client._unasked_shell_tripwire is True
    # The child runs out of the Crew-owned home the spawn just prepared.
    env: dict[str, str] = {}
    launch.apply_spawn_env(env)
    assert Path(env["KIMI_CODE_HOME"]) == tmp_path / "crew" / "kimi-home"
    assert (Path(env["KIMI_CODE_HOME"]) / "config.toml").is_file()


def test_the_session_array_is_the_kimi_mirrors_and_reaches_the_launch_seam() -> None:
    from kiro_crew.agent_sdk.backends import ACP_BACKENDS_SESSION_MCP_ARRAY
    from kiro_crew.providers.mirrors.registry import MIRRORS, PROJECTIONS, ProjectionKind

    assert KIMI in ACP_BACKENDS_SESSION_MCP_ARRAY
    assert PROJECTIONS[KIMI].kind is ProjectionKind.MIRROR and KIMI in MIRRORS
    client = AcpClient(acp_backend=KIMI)
    client._session_mcp_cache = [{"name": "kirocrew-core", "command": "x", "args": [], "env": []}]
    assert kimi_mod.KimiLaunch().session_mcp_servers(client) == client._session_mcp_cache


# ── handshake: identity and release ──────────────────────────────────────────


def test_the_verified_release_starts() -> None:
    _handshake_client(_initialize())._note_harness_adapter_version()


@pytest.mark.parametrize(
    "name,version",
    [
        ("kimi-cli", KIMI_VERIFIED_VERSION),  # the archived Python build, right number
        ("Kimi Code CLI", "2.7.3"),
        ("Kimi Code CLI", "2.6.0"),
        ("Kimi Code CLI", ""),
        ("", KIMI_VERIFIED_VERSION),
    ],
)
def test_another_name_or_release_is_refused_at_the_handshake(name: str, version: str) -> None:
    init = _initialize(name, version)
    with pytest.raises(AcpToolGateUnroutable) as raised:
        _handshake_client(init)._note_harness_adapter_version()
    # No respawn, and every surface shows the remedy.
    assert raised.value.transient is False and raised.value.user_worded is True
    message = str(raised.value)
    assert "npm i -g @moonshot-ai/kimi-code@2.7.2" in message
    assert "Kimi Code CLI 2.7.2" in message


def test_an_initialize_with_no_agent_info_is_refused() -> None:
    client = _handshake_client({"protocolVersion": 1})
    with pytest.raises(AcpToolGateUnroutable):
        client._note_harness_adapter_version()


def test_the_identity_check_is_kimis_alone() -> None:
    client = AcpClient(acp_backend="opencode")
    client._initialize_result = _initialize("whatever", "0")
    client._note_harness_adapter_version()


def test_the_identity_check_runs_at_the_handshake() -> None:
    from kiro_crew.acp import client_start

    source = inspect.getsource(client_start)
    assert "self._note_harness_adapter_version()" in source


# ── routing, the mode pin and the forbidden modes ────────────────────────────


def test_kimi_routes_by_session_config_with_the_mode_pinned() -> None:
    assert gate.routing_for(KIMI) is gate.Routing.SESSION_CONFIG
    assert permission_config_for(KIMI) == ("mode", "default")
    assert gate.session_config_issue(KIMI, [_mode_option()]) == ""
    assert KIMI in ACP_BACKEND_FORBIDDEN_MODES
    assert ACP_BACKEND_FORBIDDEN_MODES[KIMI] == frozenset({"yolo", "auto"})


def test_a_session_that_does_not_advertise_the_mode_is_not_routable() -> None:
    assert gate.session_config_issue(KIMI, []) != ""
    other = {"id": "model", "type": "select", "currentValue": "x", "options": [{"value": "x"}]}
    assert gate.session_config_issue(KIMI, [other]) != ""


@pytest.mark.asyncio
async def test_crew_writes_mode_default_through_a_config_option_and_never_set_mode() -> None:
    client = AcpClient(acp_backend=KIMI)
    client._acp_config_options = [_mode_option("yolo")]
    written: list[tuple[str, str]] = []

    async def _write(config_id: str, value: str) -> None:
        written.append((config_id, value))
        client._acp_config_options = [_mode_option(value)]

    client.set_config_option = _write  # type: ignore[method-assign]
    client._send_request = AsyncMock()

    await client._apply_session_permission_routing()

    assert written == [("mode", "default")]
    methods = [call.args[0] for call in client._send_request.await_args_list]
    assert "session/set_mode" not in methods


def test_no_kimi_module_calls_set_mode() -> None:
    for module in (kimi_mod, kimi_fs_mod, client_kimi_session):
        names = {
            node.attr if isinstance(node, ast.Attribute) else node.id
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, (ast.Attribute, ast.Name))
        }
        assert names.isdisjoint({"set_mode", "METHOD_SET_MODE"})


@pytest.mark.asyncio
async def test_a_write_the_harness_answers_with_yolo_still_on_is_refused() -> None:
    client = AcpClient(acp_backend=KIMI)
    client._acp_config_options = [_mode_option("yolo")]

    async def _write(config_id: str, value: str) -> None:
        client._acp_config_options = [_mode_option("yolo")]

    client.set_config_option = _write  # type: ignore[method-assign]
    with pytest.raises(AcpToolGateUnroutable):
        await client._apply_session_permission_routing()


@pytest.mark.parametrize("mode", ["yolo", "auto"])
def test_a_forbidden_mode_is_drift_in_either_frame_shape(mode: str) -> None:
    assert session_config_drift_issue(
        KIMI, {"sessionUpdate": "current_mode_update", "currentModeId": mode}
    )
    assert session_config_drift_issue(
        KIMI, {"sessionUpdate": "config_option_update", "configOptions": [_mode_option(mode)]}
    )


def test_the_pinned_mode_is_not_drift() -> None:
    assert not session_config_drift_issue(
        KIMI, {"sessionUpdate": "current_mode_update", "currentModeId": "default"}
    )
    assert not session_config_drift_issue(
        KIMI, {"sessionUpdate": "config_option_update", "configOptions": [_mode_option()]}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["yolo", "auto"])
async def test_the_harness_moving_itself_to_a_forbidden_mode_stops_the_session(mode: str) -> None:
    client = AcpClient(acp_backend=KIMI)
    client._session_id = "s"
    client._session_key = "dashboard:test"
    client._kill_process = AsyncMock()

    with pytest.raises(AcpToolGateUnroutable):
        await client._tripwire_session_config(_mode_update(mode))
    client._kill_process.assert_awaited_once_with(force=True)


@pytest.mark.asyncio
async def test_the_pinned_mode_reported_again_stops_nothing() -> None:
    client = AcpClient(acp_backend=KIMI)
    client._session_id = "s"
    client._kill_process = AsyncMock()
    await client._tripwire_session_config(_mode_update("default"))
    client._kill_process.assert_not_awaited()


# ── the prompt-command allowlist ─────────────────────────────────────────────

# Mode-changing or credential-changing commands, spelled the ways a parser may fold.
_REFUSED = (
    "/yolo",
    "/auto",
    "/YOLO",
    " /yolo",
    "\n/auto",
    "\u00a0/yolo",
    "/yolo on",
    "/login",
    "/logout",
    "/goal ship it",
    "/plan",
    "/mcp-config",
    "/sub-skill",
    "/import-from-cc-codex",
    "/some-command-this-build-has-never-seen",
)
_PASSED = ("/compact", "/status", "/usage", "/mcp", "/tasks", "/help", "/COMPACT now")
# "Reply ok.\n/status" was sent live: Kimi 2.7.2 answered it as model text, not as a
# command, so a slash after other text needs no refusal.
_NOT_COMMANDS = (
    "hello",
    "",
    "/yolo.",
    "/yolo/x",
    "//yolo",
    "/ yolo",
    "read /etc/hosts",
    "Reply ok.\n/yolo",
)


@pytest.mark.parametrize("text", _REFUSED)
def test_a_mode_changing_command_is_refused_before_the_wire(text: str) -> None:
    assert prompt_command_issue(KIMI, [text]) != ""


@pytest.mark.parametrize("text", _PASSED + _NOT_COMMANDS)
def test_a_read_only_command_or_plain_text_passes(text: str) -> None:
    assert prompt_command_issue(KIMI, [text]) == ""


def test_every_text_block_is_checked() -> None:
    assert prompt_command_issue(KIMI, ["hello", "/yolo"]) != ""


def test_the_allowlist_admits_no_permission_command() -> None:
    allowed = ACP_BACKEND_PROMPT_COMMAND_ALLOWLIST[KIMI]
    assert allowed.isdisjoint({"yolo", "auto", "plan", "login", "logout", "goal", "mcp-config"})


@pytest.mark.asyncio
async def test_a_refused_prompt_never_reaches_the_wire(tmp_path: Path) -> None:
    client = AcpClient(work_dir=tmp_path, acp_backend=KIMI)
    client._process = MagicMock(returncode=None)
    client._session_id = "kimi-session-1"
    sent: list = []

    async def _send(method: str, params: dict) -> int:
        sent.append(method)
        return len(sent)

    client._send_request = AsyncMock(side_effect=_send)
    client._settle_orphan_steers = AsyncMock()
    client._turn_done.clear()

    with pytest.raises(AcpError) as raised:
        await client._send_prompt("/yolo")
    assert sent == [] and "/yolo" in str(raised.value)
    assert client._turn_done.is_set()

    await client._send_prompt("/status")
    assert sent == ["session/prompt"]


# ── auth declaration ─────────────────────────────────────────────────────────


def test_the_auth_declaration() -> None:
    declaration = host_auth.declaration_for(KIMI)
    assert declaration.entitlement_source == host_auth.ENTITLEMENT_OWN_CREDENTIAL_FILE
    assert declaration.credential_leaves == (
        ".kimi-code/credentials",
        ".kimi-code/config.toml",
        ".kimi-code/logs",
        ".kimi-code/sessions",
        ".kimi-code/oauth",
        ".kimi-code/device_id",
        ".kimi-code/user-history",
    )
    assert declaration.home_override_env_vars == ("KIMI_CODE_HOME",)
    # The child runs out of Crew's own home (``prepare_kimi_home``), so none of the person's
    # own leaves is spared from its mask.
    assert declaration.adapter_own_leaves == ()
    assert declaration.host_logout_retires_children is False
    # ``sign_in_remedy`` is plain text (no ``--``), so it names the region option in words.
    assert "kimi login with its region option set to global" in declaration.sign_in_remedy
    text = declaration.signed_out_message
    assert "KIMI_CODE_HOME=~/.kiro/crew/kimi-home kimi login --region global" in text
    assert "$env:KIMI_CODE_HOME=" in text and 'set "KIMI_CODE_HOME=' in text
    # Signed out on Windows, Kimi 2.7.2's ``session/new`` fails with -32000
    # "Authentication required"; on Linux ``kimi_signed_out`` reads the handshake.
    assert declaration.signed_out_signature == "Authentication required"


@pytest.mark.parametrize(
    "leaf",
    [
        "credentials/kimi-code.json",
        "credentials/mcp/server.json",
        "config.toml",
        "logs/kimi.log",
        "sessions/key/session_1/agents/main/wire.jsonl",
        "oauth/kimi-code",
        "device_id",
        "user-history/prompts.jsonl",
    ],
)
def test_the_agents_file_tools_cannot_read_a_kimi_leaf(leaf: str, tmp_path, monkeypatch) -> None:
    import os

    assert is_sensitive_path(os.path.join(os.path.expanduser("~"), ".kimi-code", leaf))
    # Relocated by the override, the same leaf stays fenced.
    relocated = tmp_path / "kimi-home"
    monkeypatch.setenv("KIMI_CODE_HOME", str(relocated))
    assert is_sensitive_path(str(relocated / leaf))


def test_the_child_may_use_crews_home_and_never_the_persons_own_or_a_foreign_childs() -> None:
    import os

    from kiro_crew.security import crew_home_prefixes

    home = os.path.expanduser("~")
    masked = set(gate.adapter_hidden_credential_dirs(KIMI))
    for leaf in host_auth.declaration_for(KIMI).credential_leaves:
        assert os.path.join(home, *leaf.split("/")) in masked
    for floor_leaf in (".aws", ".ssh"):
        assert os.path.join(home, floor_leaf) in masked
    crew_homes = {os.path.join(home, p, gate.KIMI_HOME_LEAF) for p in crew_home_prefixes()}
    assert crew_homes.isdisjoint(masked)
    # Another harness's child keeps Kimi's home masked, and the agent's file tools cannot open it.
    other = set(gate.adapter_hidden_credential_dirs("codex"))
    assert crew_homes <= other
    for prefix in crew_home_prefixes():
        assert is_sensitive_path(
            os.path.join(home, prefix, gate.KIMI_HOME_LEAF, "abc", "credentials", "kimi-code.json")
        )


# ── a project's own configuration ────────────────────────────────────────────


def test_a_project_with_its_own_kimi_directory_is_refused(tmp_path) -> None:
    (tmp_path / ".kimi-code").mkdir()
    (tmp_path / ".kimi-code" / "mcp.json").write_text("{}")
    issue = kimi_mod.kimi_repo_config_issue(tmp_path)
    assert str(tmp_path / ".kimi-code") in issue and "Kimi Code" in issue


def test_a_project_without_it_starts(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    assert kimi_mod.kimi_repo_config_issue(tmp_path) == ""
    assert kimi_mod.kimi_repo_config_issue("") == ""


def test_the_walk_finds_a_parents_directory_and_stops_at_the_git_root(tmp_path) -> None:
    root = tmp_path / "repo"
    work = root / "sub" / "deep"
    work.mkdir(parents=True)
    (root / ".git").mkdir()
    (tmp_path / ".kimi-code").mkdir()  # above the git root: not this project's
    assert kimi_mod.kimi_repo_config_issue(work) == ""
    (root / ".kimi-code").write_text("")  # a file counts as present
    assert str(root / ".kimi-code") in kimi_mod.kimi_repo_config_issue(work)


def _fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work = home / "work" / "notes"
    work.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("KIMI_CODE_HOME", raising=False)
    return home, work


def test_kimis_own_user_home_is_not_a_project_config(tmp_path, monkeypatch) -> None:
    home, work = _fake_home(tmp_path, monkeypatch)
    (home / ".kimi-code").mkdir()
    assert kimi_mod.kimi_repo_config_issue(work) == ""


def test_a_relocated_kimi_home_is_not_a_project_config_either(tmp_path, monkeypatch) -> None:
    home, work = _fake_home(tmp_path, monkeypatch)
    (home / ".kimi-code").mkdir()
    elsewhere = home / "work" / ".kimi-code"
    elsewhere.mkdir()
    monkeypatch.setenv("KIMI_CODE_HOME", str(elsewhere))
    assert kimi_mod.kimi_repo_config_issue(work) == ""


def test_a_project_config_below_home_is_still_refused(tmp_path, monkeypatch) -> None:
    home, work = _fake_home(tmp_path, monkeypatch)
    (home / ".kimi-code").mkdir()
    (home / "work" / ".kimi-code").mkdir()
    assert str(home / "work" / ".kimi-code") in kimi_mod.kimi_repo_config_issue(work)


@pytest.mark.asyncio
async def test_a_refused_project_resolves_nothing(tmp_path, monkeypatch) -> None:
    (tmp_path / ".kimi-code").mkdir()
    resolved = AsyncMock()
    monkeypatch.setattr(launch_mod, "resolve_self_served_launch", resolved)
    client = AcpClient(work_dir=tmp_path, acp_backend=KIMI)
    ctx = SpawnContext.__new__(SpawnContext)
    object.__setattr__(ctx, "session", client)
    object.__setattr__(ctx, "work_dir", tmp_path)
    object.__setattr__(ctx, "sandbox_mode", "auto")

    with pytest.raises(AcpToolGateUnroutable) as raised:
        await kimi_mod.KimiLaunch().resolve_spawn(ctx)
    assert raised.value.transient is False and raised.value.user_worded is True
    resolved.assert_not_awaited()


# ── install probe ────────────────────────────────────────────────────────────


def _npm_install(root: Path, version: str) -> Path:
    package = root / "lib" / "node_modules" / "@moonshot-ai" / "kimi-code"
    (package / "dist").mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps({"name": "@moonshot-ai/kimi-code", "version": version})
    )
    (package / "dist" / "main.mjs").write_text("")
    (root / "bin").mkdir()
    link = root / "bin" / "kimi"
    link.symlink_to(package / "dist" / "main.mjs")
    return link


def test_the_installed_release_is_read_from_the_npm_manifest(tmp_path) -> None:
    assert kimi_mod.kimi_installed_version(str(_npm_install(tmp_path, "2.7.2"))) == "2.7.2"


def test_the_release_is_read_below_a_windows_npm_shim(tmp_path) -> None:
    package = tmp_path / "kimi2" / "node_modules" / "@moonshot-ai" / "kimi-code"
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps({"name": "@moonshot-ai/kimi-code", "version": "2.7.2"})
    )
    shim = tmp_path / "kimi2" / "kimi.cmd"
    shim.write_text("")
    assert kimi_mod.kimi_installed_version(str(shim)) == "2.7.2"


def test_a_build_with_no_manifest_reads_as_unknown(tmp_path) -> None:
    legacy = tmp_path / "kimi"
    legacy.write_text("")
    assert kimi_mod.kimi_installed_version(str(legacy)) == ""
    assert kimi_mod.kimi_installed_version(str(tmp_path / "absent")) == ""


def test_a_manifest_of_another_package_is_not_read(tmp_path) -> None:
    link = _npm_install(tmp_path, "9.9.9")
    manifest = link.resolve().parent.parent / "package.json"
    manifest.write_text(json.dumps({"name": "something-else", "version": "9.9.9"}))
    assert kimi_mod.kimi_installed_version(str(link)) == ""


def test_the_driver_names_a_wrong_release(tmp_path, monkeypatch) -> None:
    from kiro_crew.acp import client as acp_client

    link = _npm_install(tmp_path, "2.7.1")
    monkeypatch.setattr(acp_client, "_resolve_self_served_bin", lambda backend: (str(link), ""))
    assert acp_driver.self_served_wrong_version(KIMI) == "2.7.1"
    link.resolve().parent.parent.joinpath("package.json").write_text(
        json.dumps({"name": "@moonshot-ai/kimi-code", "version": KIMI_VERIFIED_VERSION})
    )
    assert acp_driver.self_served_wrong_version(KIMI) == ""


def test_the_probe_reports_a_wrong_release_as_missing_with_the_pinned_command(
    monkeypatch,
) -> None:
    monkeypatch.setattr(acp_driver, "self_served_resolves", lambda backend: True)
    monkeypatch.setattr(acp_driver, "self_served_wrong_version", lambda backend: "2.7.1")
    state = backend_install._probe_self_served(KIMI)
    assert state.installed == backend_install.MISSING
    assert state.missing_components == ("kimi 2.7.2",)
    assert state.found_version == "2.7.1"
    assert state.install_command == "npm i -g @moonshot-ai/kimi-code@2.7.2"


def test_the_probe_reports_a_missing_binary_with_the_install_command(monkeypatch) -> None:
    monkeypatch.setattr(acp_driver, "self_served_resolves", lambda backend: False)
    state = backend_install._probe_self_served(KIMI)
    assert state.installed == backend_install.MISSING
    assert state.missing_components == ("kimi",)
    assert state.install_command == "npm i -g @moonshot-ai/kimi-code@2.7.2"


# ── live corpus (test/fixtures/acp_frames/kimi) ──────────────────────────────

_CORPUS = Path(__file__).parent / "fixtures" / "acp_frames" / "kimi"


def _corpus(name: str) -> list[dict]:
    lines = (_CORPUS / f"{name}.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines[1:]]


def _updates(frames: list[dict], kind: str) -> list[dict]:
    return [
        f["params"]["update"]
        for f in frames
        if f.get("method") == "session/update" and f["params"]["update"]["sessionUpdate"] == kind
    ]


def _session_new(name: str) -> dict:
    return next(f["result"] for f in _corpus(name) if "configOptions" in f.get("result", {}))


def test_the_handshake_corpus_pins_the_terminal_auth_method_and_the_mode_spellings() -> None:
    frames = _corpus("handshake-live")
    init = frames[0]["result"]
    assert init["agentInfo"] == {"name": KIMI_AGENT_NAME, "version": KIMI_VERIFIED_VERSION}
    assert [(m["id"], m["type"]) for m in init["authMethods"]] == [("login", "terminal")]
    new = _session_new("handshake-live")
    assert [m["id"] for m in new["modes"]["availableModes"]] == ["default", "plan", "auto", "yolo"]
    assert new["modes"]["currentModeId"] == "default"
    # A mode change is announced twice, with these spellings, and answered with ``{}``.
    assert _updates(frames, "current_mode_update")[0]["currentModeId"] == "plan"
    assert _updates(frames, "config_option_update")
    assert any(f.get("result") == {} for f in frames)
    assert any(f.get("error", {}).get("code") == -32602 for f in frames)


def test_the_permission_request_has_three_options_and_a_bare_tool_call() -> None:
    for name in ("turn-live-permission-bash", "turn-live-permission-mcp"):
        request = next(f for f in _corpus(name) if f.get("method") == "session/request_permission")
        assert {o["optionId"]: o["kind"] for o in request["params"]["options"]} == {
            "approve_once": "allow_once",
            "approve_always": "allow_always",
            "reject": "reject_once",
        }
        # No ``kind``, ``rawInput`` or ``status`` here: those ride the earlier updates.
        assert set(request["params"]["toolCall"]) == {"toolCallId", "title", "content"}


def test_a_question_arrives_as_a_form_elicitation() -> None:
    request = next(
        f for f in _corpus("turn-live-elicitation") if f.get("method") == "elicitation/create"
    )
    assert request["params"]["mode"] == "form"
    assert request["params"]["requestedSchema"]["required"] == ["q0"]


def test_session_load_answers_without_a_session_id_after_replaying_history() -> None:
    frames = _corpus("session-load-live")
    result = frames[-1]["result"]
    assert "sessionId" not in result and set(result) == {"configOptions", "modes"}
    assert _updates(frames[:-1], "user_message_chunk") and _updates(frames[:-1], "tool_call")


def test_the_unasked_write_is_pinned_so_a_change_in_it_is_seen() -> None:
    frames = _corpus("turn-live-write-unasked")
    assert not any(f.get("method") == "session/request_permission" for f in frames)
    assert _updates(frames, "tool_call")[0]["title"] == "Write"
    assert _updates(frames, "tool_call_update")[-1]["status"] == "completed"


def test_signed_out_is_read_from_the_thinking_select_not_from_an_error() -> None:
    assert kimi_mod.kimi_signed_out(_session_new("signed-out-live")["configOptions"]) is True
    assert kimi_mod.kimi_signed_out(_session_new("handshake-live")["configOptions"]) is False
    assert kimi_mod.kimi_signed_out(None) is False
    assert kimi_mod.kimi_signed_out([]) is False


def test_a_signed_out_session_is_refused_before_its_first_prompt() -> None:
    client = AcpClient(acp_backend=KIMI)
    client._acp_config_options = _session_new("handshake-live")["configOptions"]
    client._refuse_harness_default_model()
    client._acp_config_options = _session_new("signed-out-live")["configOptions"]
    with pytest.raises(AcpNoModelProvider) as raised:
        client._refuse_harness_default_model()
    assert "kimi login" in str(raised.value)


def test_the_signed_out_refusal_uses_the_sign_in_command_kimi_reports(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    client = AcpClient(acp_backend=KIMI)
    client._acp_config_options = _session_new("signed-out-live")["configOptions"]
    client._initialize_result = {
        "authMethods": [{"_meta": {"terminal-auth": {"command": "/k/main.mjs", "args": ["login"]}}}]
    }
    with pytest.raises(AcpNoModelProvider) as raised:
        client._refuse_harness_default_model()
    assert "node /k/main.mjs login --region global" in str(raised.value)


def test_a_bad_kimi_model_is_a_value_refusal_by_its_text_not_by_code_32602() -> None:
    from kiro_crew.acp.client import _is_config_value_rejection

    refused = AcpError('Model "nope" is not configured in config.toml.', code=-32603)
    assert _is_config_value_rejection(refused, "model", KIMI) is True
    # A bare -32603 stays a fault, and the phrase is Kimi's and the model option's only.
    assert (
        _is_config_value_rejection(AcpError("Internal error", code=-32603), "model", KIMI) is False
    )
    assert _is_config_value_rejection(refused, "thinking", KIMI) is False
    assert _is_config_value_rejection(refused, "model", "claude") is False


def _kimi_bash_ask(
    command: str | None = "echo hi", *, extra_updates=()
) -> tuple[AcpClient, object]:
    """The recorded Bash frames replayed through a real client up to the ask.

    *command* replaces the recorded command in the pre-ask update; ``None`` drops that
    update (no earlier frame states a command). Returns the client and the built event.
    """
    frames = _corpus("turn-live-permission-bash")
    client = AcpClient(acp_backend=KIMI)
    for frame in frames:
        if frame.get("method") == "session/request_permission":
            break
        update = frame["params"]["update"]
        if update["sessionUpdate"] == "tool_call_update":
            if command is None:
                continue
            text = json.dumps({"command": command})
            update["content"][0]["content"]["text"] = text
        msg = JsonRpcMessage.from_dict(frame)
        client._extract_tool_event(msg)
        client._extract_tool_call_update(msg)
    for update in extra_updates:
        client._extract_tool_call_update(
            JsonRpcMessage.from_dict(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {"sessionId": "kimi-session-1", "update": update},
                }
            )
        )
    ask = next(f for f in frames if f.get("method") == "session/request_permission")
    return client, client._build_permission_event(JsonRpcMessage.from_dict(ask))


def _gate(event):
    from kiro_crew.hook_runtime.tool_identity import hook_gate_kwargs
    from kiro_crew.hooks import HookManager, HooksConfig

    return HookManager(HooksConfig()).on_tool_call(event.title, **hook_gate_kwargs(event))


def test_the_bash_ask_takes_its_command_from_the_earlier_update() -> None:
    _client, event = _kimi_bash_ask("echo hi")
    assert event.shell_command == "echo hi"
    assert "could not be verified" not in (_gate(event).reason or "")


def test_a_multi_line_command_is_recovered_verbatim() -> None:
    _client, event = _kimi_bash_ask("echo a\necho 'b'")
    assert event.shell_command == "echo a\necho 'b'"


def test_a_bash_ask_with_no_earlier_command_is_refused() -> None:
    _client, event = _kimi_bash_ask(None)
    assert event.shell_command is None
    decision = _gate(event)
    assert decision.action == "deny" and "could not be verified" in (decision.reason or "")


def test_two_different_commands_for_one_call_are_refused() -> None:
    other = {
        "toolCallId": "3:tool-1",
        "sessionUpdate": "tool_call_update",
        "status": "in_progress",
        "content": [{"type": "content", "content": {"type": "text", "text": '{"command":"id"}'}}],
    }
    _client, event = _kimi_bash_ask("echo hi", extra_updates=[other])
    assert event.shell_command is None
    assert _gate(event).action == "deny"


def _command_update(tool_call_id: str, command: str, **extra) -> dict:
    text = json.dumps({"command": command})
    return {
        "toolCallId": tool_call_id,
        "sessionUpdate": "tool_call_update",
        "status": "in_progress",
        "content": [{"type": "content", "content": {"type": "text", "text": text}}],
        **extra,
    }


def _feed(client, update: dict) -> None:
    msg = JsonRpcMessage.from_dict(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": "kimi-session-1", "update": update},
        }
    )
    client._extract_tool_event(msg)
    client._extract_tool_call_update(msg)


def test_an_update_for_another_tool_call_does_not_touch_this_commands() -> None:
    other = _command_update("3:tool-OTHER", "id")
    client, event = _kimi_bash_ask("echo hi", extra_updates=[other])
    assert event.shell_command == "echo hi"
    assert "3:tool-OTHER" not in client._tool_call_params


def test_a_late_raw_input_does_not_change_the_event_already_gated() -> None:
    client, event = _kimi_bash_ask("echo hi")
    _feed(
        client,
        {
            "toolCallId": "3:tool-1",
            "sessionUpdate": "tool_call_update",
            "rawInput": {"command": "id"},
        },
    )
    assert event.shell_command == "echo hi"
    assert _gate(event).action != "deny" or "could not be verified" not in (
        _gate(event).reason or ""
    )


def test_a_blank_command_stays_blank_when_the_first_command_comes_again() -> None:
    client, event = _kimi_bash_ask("echo hi", extra_updates=[_command_update("3:tool-1", "id")])
    assert event.shell_command is None
    _feed(client, _command_update("3:tool-1", "echo hi"))
    assert client._tool_call_params["3:tool-1"]["command"] == ""


@pytest.mark.parametrize("text", ["not json", "[1]", '{"cmd":"x"}', '{"command":3}', ""])
def test_an_unreadable_command_update_recovers_nothing(text: str) -> None:
    update = {"content": [{"type": "content", "content": {"type": "text", "text": text}}]}
    assert kimi_mod.kimi_command_from_update(update) == ""
    assert kimi_mod.kimi_command_from_update({"content": [update["content"][0]] * 2}) == ""


# ── Node.js floor ────────────────────────────────────────────────────────────


def _fake_node(monkeypatch, version: str | None) -> None:
    monkeypatch.setattr(
        kimi_mod.shutil, "which", lambda _name: None if version is None else "/n/node"
    )
    monkeypatch.setattr(
        kimi_mod.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, stdout=f"{version}\n"),
    )


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("v20.20.2", "v20.20.2"),
        ("v22.18.9", "v22.18.9"),
        ("v22.19.0", ""),
        ("v24.1.0", ""),
        ("odd", ""),
    ],
)
def test_a_node_below_the_floor_is_named(tmp_path, monkeypatch, version, expected) -> None:
    _fake_node(monkeypatch, version)
    assert kimi_mod.kimi_node_found(str(_npm_install(tmp_path, "2.7.2"))) == expected


def test_no_node_on_path_is_named_and_a_build_without_a_manifest_is_not_judged(
    tmp_path, monkeypatch
) -> None:
    _fake_node(monkeypatch, None)
    assert kimi_mod.kimi_node_found(str(_npm_install(tmp_path, "2.7.2"))) == "none"
    assert kimi_mod.kimi_node_found(str(tmp_path / "standalone-kimi")) == ""


def test_the_node_remedy_names_the_version_found_and_the_floor() -> None:
    assert "22.19.0" in kimi_mod.kimi_node_message("v20.20.2")
    assert "v20.20.2" in kimi_mod.kimi_node_message("v20.20.2")
    assert "no node on PATH" in kimi_mod.kimi_node_message("none")


@pytest.mark.asyncio
async def test_an_old_node_refuses_the_spawn_with_the_node_remedy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        launch_mod,
        "resolve_self_served_launch",
        AsyncMock(return_value=("/opt/bin/kimi", ["/opt/bin/kimi", "acp"], "kimi acp", "kimi")),
    )
    monkeypatch.setattr(kimi_mod, "kimi_node_found", lambda _bin: "v20.20.2")
    client = AcpClient(work_dir=tmp_path, acp_backend=KIMI)
    ctx = SpawnContext.__new__(SpawnContext)
    object.__setattr__(ctx, "session", client)
    object.__setattr__(ctx, "work_dir", tmp_path)
    object.__setattr__(ctx, "sandbox_mode", "auto")
    with pytest.raises(AcpToolGateUnroutable, match="v20.20.2") as raised:
        await kimi_mod.KimiLaunch().resolve_spawn(ctx)
    assert raised.value.transient is False and raised.value.user_worded is True


def test_the_probe_reports_an_old_node_as_missing_with_the_version_found(monkeypatch) -> None:
    monkeypatch.setattr(acp_driver, "self_served_resolves", lambda backend: True)
    monkeypatch.setattr(acp_driver, "self_served_wrong_version", lambda backend: "")
    monkeypatch.setattr(acp_driver, "self_served_old_node", lambda backend: "v20.20.2")
    state = backend_install._probe_self_served(KIMI)
    assert state.installed == backend_install.MISSING
    assert state.missing_components == ("node 22.19",)
    assert state.found_version == "v20.20.2"


def test_a_project_mcp_json_in_the_cwd_or_a_parent_is_refused(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".mcp.json").write_text("{}")
    work = tmp_path / "sub" / "deep"
    work.mkdir(parents=True)
    issue = kimi_mod.kimi_repo_config_issue(work)
    assert str(tmp_path / ".mcp.json") in issue and "Kimi Code" in issue
