"""Option B for the dormant Kimi Code CLI harness: a Crew-owned ``KIMI_CODE_HOME``.

Covers the generated config, the home's own sign-in, the approval mapping, the re-pin of the
mode, the ``fs/*`` write gate, the AGENTS.md probe, the tripwire rows and the child's
environment. The wire shapes come from ``test/fixtures/acp_frames/kimi/`` (live, Kimi Code
CLI 2.7.2, ``reports/kimi-levers-acp.md``).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kiro_crew import sel as sel_module
from kiro_crew.acp.client import AcpClient
from kiro_crew.acp.harness import kimi as kimi_mod
from kiro_crew.acp.harness import kimi_fs
from kiro_crew.acp.harness_tool_names import HARNESS_UNASKED_TOOL_IDS
from kiro_crew.acp.transport_errors import AcpToolGateUnroutable
from kiro_crew.acp.types import JsonRpcMessage
from kiro_crew.agent_sdk.backends import (
    ACP_BACKEND_KIMI,
    ACP_BACKENDS_HONOR_ZERO_TOOL_BAN,
    ACP_BACKENDS_UNASKED_NATIVE_TOOLS,
    permission_value_for,
    session_config_drift_issue,
)

KIMI = ACP_BACKEND_KIMI
_CORPUS = Path(__file__).parent / "fixtures" / "acp_frames" / "kimi"


def _frames(name: str) -> list[dict]:
    lines = (_CORPUS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[1:]]


def _message(frame: dict) -> JsonRpcMessage:
    return JsonRpcMessage(
        id=frame.get("id"), method=frame.get("method"), params=frame.get("params")
    )


def _request(name: str, method: str, nth: int = 0) -> JsonRpcMessage:
    found = [f for f in _frames(name) if f.get("method") == method]
    return _message(found[nth])


def _permission(
    title: str, text: str, request_id: int = 0, content: list | None = None
) -> JsonRpcMessage:
    """A Kimi permission request in the live shape (three options, a bare toolCall)."""
    return _message(
        {
            "id": request_id,
            "method": "session/request_permission",
            "params": {
                "sessionId": "s",
                "options": [
                    {"optionId": "approve_once", "name": "Approve once", "kind": "allow_once"},
                    {
                        "optionId": "approve_always",
                        "name": "Approve for this session",
                        "kind": "allow_always",
                    },
                    {"optionId": "reject", "name": "Reject", "kind": "reject_once"},
                ],
                "toolCall": {
                    "toolCallId": f"{request_id}:tool-{request_id}",
                    "title": title,
                    "content": content
                    or [{"type": "content", "content": {"type": "text", "text": text}}],
                },
            },
        }
    )


def _client(tmp_path: Path) -> AcpClient:
    client = AcpClient(work_dir=tmp_path, acp_backend=KIMI)
    client._unasked_shell_tripwire = True
    client._send_response = AsyncMock()
    client._send_error = AsyncMock()
    return client


def _errors(client: AcpClient) -> list[tuple[int, str]]:
    return [(c.args[1], c.args[2]) for c in client._send_error.await_args_list]


# ── 1. the config and the home ───────────────────────────────────────────────

#: What ``kimi login`` leaves in a fresh home's config.toml (the shape of a signed-in 2.7.2
#: home, values replaced), plus keys Crew must drop.
_LOGIN_CONFIG = """
default_model = "kimi-code/kimi-for-coding"
default_permission_mode = "yolo"

[providers."managed:kimi-code"]
type = "kimi"
base_url = "https://api.kimi.com/coding/v1"
api_key = ""

[providers."managed:kimi-code".oauth]
storage = "file"
key = "oauth/kimi-code"

[models."kimi-code/kimi-for-coding"]
provider = "managed:kimi-code"
model = "kimi-for-coding"
max_context_size = 1048576
capabilities = ["thinking", "image_in"]

[thinking]
enabled = true

[[hooks]]
event = "PreToolUse"
command = "true"

[[permission.rules]]
decision = "allow"
pattern = "*"

[services.moonshot_search]
base_url = "https://api.example.invalid/search"
"""


def test_the_config_is_the_live_tested_one() -> None:
    config = tomllib.loads(kimi_mod.render_kimi_config("/crew/kimi-home"))
    assert config["default_permission_mode"] == "manual"
    assert config["watch"] == {"enabled": False}
    assert [(r["decision"], r["pattern"]) for r in config["permission"]["rules"]] == [
        ("ask", "Write"),
        ("ask", "Edit"),
        ("ask", "Read"),
        ("deny", "Write(/crew/kimi-home/**)"),
        ("deny", "Edit(/crew/kimi-home/**)"),
    ]
    assert config["tools"]["disabled"] == list(kimi_mod.KIMI_DISABLED_TOOLS)
    # Crew's own MCP servers are offered as deferred tools (spike M1), so ``mcp__*`` is not disabled.
    assert config["mcp"] == {"default_deferred": True}
    assert not any(tool.startswith("mcp__") for tool in kimi_mod.KIMI_DISABLED_TOOLS)
    # The tools that read or reach the network without asking stay off; a later addition is fine.
    assert {
        "WebSearch",
        "FetchURL",
        "ReadMediaFile",
        "Grep",
        "Glob",
        "Skill",
    } <= set(kimi_mod.KIMI_DISABLED_TOOLS)


def test_a_home_mcp_json_refuses_the_spawn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    home = kimi_mod.kimi_home_dir()
    (home / "credentials").mkdir(parents=True)
    (home / "credentials" / "a.json").write_text("{}")
    (home / "mcp.json").write_text("{}")
    with pytest.raises(AcpToolGateUnroutable) as raised:
        kimi_mod.prepare_kimi_home()
    assert "mcp.json" in str(raised.value) and raised.value.user_worded is True


def test_the_login_part_is_kept_and_everything_else_is_crews() -> None:
    config = tomllib.loads(
        kimi_mod.render_kimi_config("/crew/kimi-home", tomllib.loads(_LOGIN_CONFIG))
    )
    login = tomllib.loads(_LOGIN_CONFIG)
    assert config["default_model"] == login["default_model"]
    assert config["providers"] == login["providers"]
    # Each kept model also declares the capability that lets Kimi defer MCP tools for it.
    kept_model = login["models"]["kimi-code/kimi-for-coding"]
    assert config["models"] == {
        "kimi-code/kimi-for-coding": {
            **kept_model,
            "capabilities": [*kept_model["capabilities"], "dynamically_loaded_tools"],
        }
    }
    assert config["providers"]["managed:kimi-code"]["type"] == "kimi"
    assert config["thinking"] == {"enabled": True}
    # The login's own mode, rules, hooks and services never survive the rewrite.
    assert config["default_permission_mode"] == "manual"
    assert "hooks" not in config and "services" not in config
    assert ("allow", "*") not in [
        (r["decision"], r["pattern"]) for r in config["permission"]["rules"]
    ]


@pytest.mark.parametrize("bad", ["/crew/a*b", "/crew/[x]", "/crew/{a,b}", "/crew/a(b)", "/crew/!x"])
def test_a_home_path_that_would_widen_a_rule_is_refused(bad: str) -> None:
    with pytest.raises(AcpToolGateUnroutable) as raised:
        kimi_mod.render_kimi_config(bad)
    assert raised.value.transient is False and raised.value.user_worded is True


def test_a_path_with_quotes_and_unicode_survives_the_toml_round_trip() -> None:
    home = '/crew/ho"me é'
    config = tomllib.loads(kimi_mod.render_kimi_config(home))
    assert config["permission"]["rules"][3]["pattern"] == f"Write({home}/**)"


def _crew(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(kimi_mod, "config_dir", lambda: tmp_path / "crew")
    return tmp_path / "crew" / "kimi-home"


def _sign_in(home: Path) -> None:
    """What ``KIMI_CODE_HOME=<home> kimi login`` leaves: a credential file and its config."""
    (home / "credentials").mkdir(parents=True, mode=0o700)
    (home / "credentials" / "kimi-code.json").write_text('{"placeholder": true}')
    (home / "config.toml").write_text(_LOGIN_CONFIG)


def test_a_home_with_no_sign_in_is_refused_with_the_one_login_command(
    tmp_path: Path, monkeypatch
) -> None:
    home = _crew(tmp_path, monkeypatch)
    with pytest.raises(AcpToolGateUnroutable) as raised:
        kimi_mod.prepare_kimi_home()
    assert raised.value.transient is False and raised.value.user_worded is True
    assert f"KIMI_CODE_HOME={home} kimi login --region global" in str(raised.value)
    # The home exists, private and empty, ready for that login; no config was written yet.
    assert home.is_dir() and (home.stat().st_mode & 0o777) == 0o700
    assert list(home.iterdir()) == []


def test_the_login_command_quotes_the_home() -> None:
    assert (
        kimi_mod.kimi_login_command("/a b/kimi-home")
        == "KIMI_CODE_HOME='/a b/kimi-home' kimi login --region global"
    )


_WIN_HOME = r"C:\Users\a b\.kiro\crew\kimi-home"


def test_the_login_command_is_spelled_per_shell() -> None:
    assert kimi_mod.kimi_login_command(_WIN_HOME, "powershell") == (
        r"$env:KIMI_CODE_HOME='C:\Users\a b\.kiro\crew\kimi-home'; kimi login --region global"
    )
    assert kimi_mod.kimi_login_command("C:\\it's", "powershell").startswith(
        "$env:KIMI_CODE_HOME='C:\\it''s'; "
    )
    assert kimi_mod.kimi_login_command(_WIN_HOME, "cmd") == (
        r'set "KIMI_CODE_HOME=C:\Users\a b\.kiro\crew\kimi-home" && kimi login --region global'
    )


_WIN_AUTH = {
    "authMethods": [
        {
            "_meta": {
                "terminal-auth": {
                    "command": r"C:\Users\a b\kimi2\main.mjs",
                    "args": ["login"],
                    "env": {"KIMI_CODE_HOME": "ignored"},
                }
            }
        }
    ]
}


def test_the_login_command_uses_the_invocation_kimi_reports() -> None:
    auth = kimi_mod.kimi_terminal_auth(_WIN_AUTH)
    assert auth == ("node", r"C:\Users\a b\kimi2\main.mjs", "login")
    assert kimi_mod.kimi_login_command(_WIN_HOME, "powershell", auth) == (
        r"$env:KIMI_CODE_HOME='C:\Users\a b\.kiro\crew\kimi-home'; "
        r"& 'node' 'C:\Users\a b\kimi2\main.mjs' 'login' --region global"
    )
    assert kimi_mod.kimi_login_command(_WIN_HOME, "cmd", auth) == (
        r'set "KIMI_CODE_HOME=C:\Users\a b\.kiro\crew\kimi-home" && '
        r'"node" "C:\Users\a b\kimi2\main.mjs" "login" --region global'
    )
    assert kimi_mod.kimi_login_command("/h", "posix", ("/opt/k i/kimi", "login")) == (
        "KIMI_CODE_HOME=/h '/opt/k i/kimi' login --region global"
    )


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"authMethods": []},
        {"authMethods": [{"_meta": {"terminal-auth": {"command": 1}}}]},
    ],
)
def test_a_missing_or_malformed_terminal_auth_falls_back_to_the_bare_command(bad) -> None:
    assert kimi_mod.kimi_terminal_auth(bad) == ()
    assert (
        kimi_mod.kimi_login_command("/h", "posix") == "KIMI_CODE_HOME=/h kimi login --region global"
    )


def test_the_signed_out_message_names_the_windows_shells_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(kimi_mod.os, "name", "nt")
    message = kimi_mod.kimi_signed_out_message(_WIN_HOME)
    assert "$env:KIMI_CODE_HOME=" in message and 'set "KIMI_CODE_HOME=' in message
    assert message.count("--region global") == 2


@pytest.mark.parametrize("home", [_WIN_HOME, "C:/Users/uosmi/kimi-home", r"\\host\share\kimi-home"])
def test_a_windows_home_is_spelled_with_forward_slashes_in_the_deny_pattern(home: str) -> None:
    config = tomllib.loads(kimi_mod.render_kimi_config(home))
    patterns = [r["pattern"] for r in config["permission"]["rules"] if r["decision"] == "deny"]
    assert patterns and all("\\" not in p for p in patterns)
    assert patterns[0] == "Write(" + home.replace("\\", "/") + "/**)"


@pytest.mark.parametrize(
    "home",
    [r"C:\Users\a(b)\kimi-home", r"C:\Users\a*\kimi-home", "/home/a\\b/kimi-home", "/a[b]/h"],
)
def test_other_glob_characters_and_a_posix_backslash_are_still_refused(home: str) -> None:
    with pytest.raises(AcpToolGateUnroutable):
        kimi_mod.render_kimi_config(home)


def test_a_linked_credential_is_not_a_sign_in(tmp_path: Path, monkeypatch) -> None:
    home = _crew(tmp_path, monkeypatch)
    (home / "credentials").mkdir(parents=True)
    elsewhere = tmp_path / "persons-own.json"
    elsewhere.write_text("{}")
    (home / "credentials" / "kimi-code.json").symlink_to(elsewhere)
    assert not kimi_mod.kimi_signed_in(home)


def test_a_signed_in_home_gets_crews_config_and_nothing_from_the_persons_home(
    tmp_path: Path, monkeypatch
) -> None:
    home = _crew(tmp_path, monkeypatch)
    persons = tmp_path / "user" / ".kimi-code"
    (persons / "credentials").mkdir(parents=True)
    (persons / "credentials" / "kimi-code.json").write_text('{"persons": true}')
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("KIMI_CODE_HOME", str(persons))
    _sign_in(home)
    before = sorted(p.name for p in home.rglob("*"))

    assert kimi_mod.prepare_kimi_home() == str(home)

    # Same entries (plus none): nothing was copied or linked in from the person's home.
    assert sorted(p.name for p in home.rglob("*")) == before
    assert (home / "credentials" / "kimi-code.json").read_text() == '{"placeholder": true}'
    assert not any(p.is_symlink() for p in home.rglob("*"))
    assert ((home / "config.toml").stat().st_mode & 0o777) == 0o600
    config = tomllib.loads((home / "config.toml").read_text())
    assert config["default_permission_mode"] == "manual" and "hooks" not in config
    assert config["providers"] == tomllib.loads(_LOGIN_CONFIG)["providers"]
    # A second spawn rewrites it the same way.
    kimi_mod.prepare_kimi_home()
    assert tomllib.loads((home / "config.toml").read_text()) == config


def test_one_home_per_install_whatever_the_session(tmp_path: Path, monkeypatch) -> None:
    home = _crew(tmp_path, monkeypatch)
    _sign_in(home)
    assert kimi_mod.prepare_kimi_home() == kimi_mod.prepare_kimi_home() == str(home)


def test_the_signed_out_remedy_names_crews_home_and_not_the_persons_login(
    tmp_path: Path, monkeypatch
) -> None:
    home = _crew(tmp_path, monkeypatch)
    message = kimi_mod.kimi_signed_out_message()
    assert f"KIMI_CODE_HOME={home} kimi login --region global" in message
    assert "your own kimi login is not used" in message


def test_a_provider_pointing_at_another_host_is_refused() -> None:
    kept = tomllib.loads(_LOGIN_CONFIG)
    kept["providers"]["managed:kimi-code"]["base_url"] = "https://evil.example/v1"
    with pytest.raises(AcpToolGateUnroutable) as raised:
        kimi_mod.render_kimi_config("/crew/kimi-home", kept)
    assert "evil.example" not in str(raised.value) and "base_url" in str(raised.value)
    for bad in (
        "http://api.kimi.com/v1",
        "https://kimi.com.evil.example/v1",
        "https://notkimi.com",
    ):
        kept["providers"]["managed:kimi-code"]["base_url"] = bad
        with pytest.raises(AcpToolGateUnroutable):
            kimi_mod.render_kimi_config("/crew/kimi-home", kept)
    kept["providers"]["managed:kimi-code"]["base_url"] = "https://api.moonshot.cn/v1"
    kept["providers"]["managed:kimi-code"]["oauth"]["oauth_host"] = "https://evil.example"
    with pytest.raises(AcpToolGateUnroutable) as raised:
        kimi_mod.render_kimi_config("/crew/kimi-home", kept)
    assert "oauth.oauth_host" in str(raised.value)


def test_unknown_provider_and_model_sub_keys_are_dropped() -> None:
    kept = tomllib.loads(_LOGIN_CONFIG)
    kept["providers"]["managed:kimi-code"]["command"] = "run-something"
    kept["providers"]["managed:kimi-code"]["oauth"]["helper"] = "x"
    kept["models"]["kimi-code/kimi-for-coding"]["on_start"] = "x"
    kept["thinking"]["hook"] = "x"
    config = tomllib.loads(kimi_mod.render_kimi_config("/crew/kimi-home", kept))
    assert config["providers"] == tomllib.loads(_LOGIN_CONFIG)["providers"]
    assert "on_start" not in config["models"]["kimi-code/kimi-for-coding"]
    assert config["thinking"] == {"enabled": True}


def test_a_plain_top_level_thinking_is_dropped_without_raising() -> None:
    # Kimi 2.7.2's `thinking` is a table; the plain form (`default_thinking`) is not a setting.
    config = tomllib.loads(kimi_mod.render_kimi_config("/crew/kimi-home", {"thinking": True}))
    assert "thinking" not in config


def test_the_config_is_written_through_a_unique_temp_file(tmp_path: Path, monkeypatch) -> None:
    home = _crew(tmp_path, monkeypatch)
    _sign_in(home)
    kimi_mod.prepare_kimi_home()
    assert sorted(p.name for p in home.iterdir()) == ["config.toml", "credentials"]
    monkeypatch.setattr(kimi_mod.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        kimi_mod.prepare_kimi_home()
    assert sorted(p.name for p in home.iterdir()) == ["config.toml", "credentials"]


# ── 2. never a session approval ──────────────────────────────────────────────


def _sent_option(client: AcpClient) -> str:
    return client._send_response.await_args.args[1]["outcome"]["optionId"]


@pytest.mark.asyncio
@pytest.mark.parametrize("always", [False, True])
async def test_approve_always_is_answered_approve_once(tmp_path: Path, always: bool) -> None:
    client = _client(tmp_path)
    msg = _request("turn-live-write-fs", "session/request_permission")
    event = client._build_permission_event(msg)
    assert event is not None
    assert await client.approve_tool(event.request_id, always=always)
    assert _sent_option(client) == "approve_once"


@pytest.mark.asyncio
async def test_an_explicit_approve_always_option_id_is_still_answered_once(tmp_path: Path) -> None:
    client = _client(tmp_path)
    event = client._build_permission_event(
        _request("turn-live-write-fs", "session/request_permission")
    )
    assert await client.approve_tool(event.request_id, "approve_always")
    assert _sent_option(client) == "approve_once"


def test_approve_always_is_never_offered_to_the_person(tmp_path: Path) -> None:
    client = _client(tmp_path)
    event = client._build_permission_event(
        _request("turn-live-write-fs", "session/request_permission")
    )
    assert [o["id"] for o in event.options] == ["approve_once", "reject"]
    assert "always" not in client._permission_options[event.request_id].get("once", "")
    assert client._permission_options[event.request_id]["always"] == "approve_once"


@pytest.mark.asyncio
async def test_a_denial_answers_the_reject_option(tmp_path: Path) -> None:
    client = _client(tmp_path)
    event = client._build_permission_event(
        _request("turn-live-write-fs", "session/request_permission")
    )
    await client.reject_tool(event.request_id)
    assert _sent_option(client) == "reject"


# ── 3. the mode pin ──────────────────────────────────────────────────────────


def _mode_option(value: str) -> dict:
    return {
        "type": "select",
        "id": "mode",
        "name": "Mode",
        "category": "mode",
        "currentValue": value,
        "options": [{"value": v} for v in ("default", "plan", "auto", "yolo")],
    }


def test_the_yolo_config_capture_reports_default_so_the_report_cannot_be_trusted() -> None:
    frames = _frames("turn-live-yolo-config")
    new = next(f for f in frames if "modes" in (f.get("result") or {}))
    assert new["result"]["modes"]["currentModeId"] == "default"
    # ...while the engine ran in yolo: the first Bash ran with no request, and Write (an ask
    # rule beats yolo) still asked.
    updates = [
        f["params"]["update"]
        for f in frames
        if f.get("method") == "session/update"
        and f["params"]["update"].get("sessionUpdate") in ("tool_call", "tool_call_update")
    ]
    titles = {u["toolCallId"]: u["title"] for u in updates if u.get("sessionUpdate") == "tool_call"}
    requested = [
        f["params"]["toolCall"]["toolCallId"]
        for f in frames
        if f.get("method") == "session/request_permission"
    ]
    bash = [call for call, title in titles.items() if title == "Bash"]
    write = next(call for call, title in titles.items() if title == "Write")
    assert bash[0] not in requested and write in requested
    assert any(u.get("status") == "completed" and u["toolCallId"] == bash[0] for u in updates)
    # After the write of ``default`` the same tool asks.
    assert bash[-1] in requested


@pytest.mark.asyncio
async def test_the_pin_is_a_config_option_write_that_reads_back(tmp_path: Path) -> None:
    """``set_config_option(mode, default)`` is Kimi's own ``setMode`` with no early return
    (``main.mjs`` ``setSessionConfigOption`` -> ``setMode`` -> ``agent.setPermission``), so it
    corrects a yolo config the handshake reports as ``default``; Crew's read-back then judges
    the answer."""
    client = _client(tmp_path)
    client._acp_config_options = [_mode_option("default")]
    written: list[tuple[str, str]] = []

    async def _write(config_id: str, value: str) -> None:
        written.append((config_id, value))
        client._acp_config_options = [_mode_option(value)]

    client.set_config_option = _write  # type: ignore[method-assign]
    await client._apply_session_permission_routing()
    assert written == [("mode", permission_value_for(KIMI, client._acp_config_options))]
    assert written == [("mode", "default")]


def test_the_set_mode_answer_frames_of_the_capture_are_not_drift() -> None:
    frames = _frames("turn-live-yolo-config")
    updates = [
        f["params"]["update"]
        for f in frames
        if f.get("method") == "session/update"
        and f["params"]["update"].get("sessionUpdate")
        in ("current_mode_update", "config_option_update")
    ]
    assert {u["sessionUpdate"] for u in updates} == {"current_mode_update", "config_option_update"}
    for update in updates:
        assert session_config_drift_issue(KIMI, update) == ""
    assert session_config_drift_issue(
        KIMI, {"sessionUpdate": "current_mode_update", "currentModeId": "yolo"}
    )


def test_every_path_that_makes_a_session_re_pins_it() -> None:
    """``new`` and ``load`` both end in the routing write; Crew uses neither ``resume`` nor
    ``fork`` for this harness, so no session is made that skips it."""
    root = Path(__file__).resolve().parents[1] / "src" / "kiro_crew" / "acp"
    start = (root / "client_start.py").read_text()
    # One call after the load-or-new branch of ``_initialize_session``.
    assert start.count("await self._apply_session_permission_routing()") == 1
    load_at = start.index("METHOD_SESSION_LOAD")
    new_at = start.index("METHOD_SESSION_NEW", load_at)
    assert load_at < new_at < start.index("await self._apply_session_permission_routing()")
    from kiro_crew.agent_sdk.backends import ACP_BACKENDS_RESUME_WITHOUT_LOAD

    assert KIMI not in ACP_BACKENDS_RESUME_WITHOUT_LOAD
    forks = {p.name for p in root.rglob("*.py") if "session/fork" in p.read_text()}
    assert forks <= {"client_claude_session.py", "types.py"}, forks
    for path in root.rglob("*kimi*.py"):
        assert "session/fork" not in path.read_text() and "session/resume" not in path.read_text()


# ── 4. the fs write gate ─────────────────────────────────────────────────────


async def _approve(client: AcpClient, msg: JsonRpcMessage) -> None:
    client._build_permission_event(msg)
    client._note_pi_gate_asked(msg)
    assert await client.approve_tool(msg.id)


@pytest.mark.asyncio
async def test_a_write_after_its_approval_is_carried_out(tmp_path: Path) -> None:
    client = _client(tmp_path)
    target = tmp_path / "a.txt"
    await _approve(client, _permission("Write", "Requesting approval to Writing a.txt"))
    write = _message(
        {
            "id": 7,
            "method": "fs/write_text_file",
            "params": {"path": str(target), "content": "hi\n"},
        }
    )
    await client._reject_unknown_server_request(write)
    assert target.read_text() == "hi\n"
    client._send_response.assert_awaited_with(7, {})
    client._send_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_live_write_capture_is_gated_end_to_end(tmp_path: Path) -> None:
    """Replay of ``turn-live-write-fs``: the permission request, then the write it paid for."""
    client = _client(tmp_path)
    frames = _frames("turn-live-write-fs")
    request = next(f for f in frames if f.get("method") == "session/request_permission")
    write = next(f for f in frames if f.get("method") == "fs/write_text_file")
    # The capture recorded a path of the recording host; the gate resolves both the same way.
    path = write["params"]["path"].replace("~", str(tmp_path))
    text = request["params"]["toolCall"]["content"][0]["content"]["text"]
    request["params"]["toolCall"]["content"][0]["content"]["text"] = text.replace(
        "~", str(tmp_path)
    )
    await _approve(client, _message(request))
    await client._reject_unknown_server_request(
        _message({**write, "params": {**write["params"], "path": path}})
    )
    assert Path(path).read_text() == write["params"]["content"]


@pytest.mark.asyncio
async def test_a_write_with_no_approval_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    client = _client(tmp_path)
    target = tmp_path / "b.txt"
    await client._reject_unknown_server_request(
        _message(
            {
                "id": 3,
                "method": "fs/write_text_file",
                "params": {"path": str(target), "content": "x"},
            }
        )
    )
    assert not target.exists()
    ((code, text),) = _errors(client)
    # Never "not found": Kimi would create the parent and retry.
    assert code == -32000 and code != -32002
    assert "no approved permission request" in text


@pytest.mark.asyncio
async def test_an_approval_for_one_path_does_not_pay_for_another(tmp_path: Path) -> None:
    client = _client(tmp_path)
    await _approve(client, _permission("Write", "Requesting approval to Writing a.txt"))
    other = tmp_path / "other.txt"
    await client._reject_unknown_server_request(
        _message(
            {
                "id": 4,
                "method": "fs/write_text_file",
                "params": {"path": str(other), "content": "x"},
            }
        )
    )
    assert not other.exists() and _errors(client)[0][0] == -32000


@pytest.mark.asyncio
async def test_a_relative_path_in_the_permission_text_is_resolved_against_the_cwd(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    sub = tmp_path / "sub"
    sub.mkdir()
    await _approve(client, _permission("Write", "Requesting approval to Writing sub/../sub/c.txt"))
    await client._reject_unknown_server_request(
        _message(
            {
                "id": 5,
                "method": "fs/write_text_file",
                "params": {"path": str(sub / "c.txt"), "content": "c"},
            }
        )
    )
    assert (sub / "c.txt").read_text() == "c"


@pytest.mark.asyncio
async def test_an_approval_pays_for_one_write_and_not_for_a_later_turn(tmp_path: Path) -> None:
    client = _client(tmp_path)
    target = tmp_path / "once.txt"
    frame = {
        "id": 8,
        "method": "fs/write_text_file",
        "params": {"path": str(target), "content": "1"},
    }
    await _approve(client, _permission("Write", "Requesting approval to Writing once.txt"))
    await client._reject_unknown_server_request(_message(frame))
    assert target.read_text() == "1"
    await client._reject_unknown_server_request(_message({**frame, "id": 9}))
    assert _errors(client) and _errors(client)[-1][0] == -32000

    await _approve(client, _permission("Write", "Requesting approval to Writing once.txt", 1))
    client._kimi_fs.new_turn()
    await client._reject_unknown_server_request(_message({**frame, "id": 10}))
    assert _errors(client)[-1][0] == -32000


@pytest.mark.asyncio
async def test_a_rejected_request_pays_for_nothing(tmp_path: Path) -> None:
    client = _client(tmp_path)
    msg = _permission("Write", "Requesting approval to Writing d.txt")
    client._build_permission_event(msg)
    client._note_pi_gate_asked(msg)
    await client.reject_tool(msg.id)
    await client._reject_unknown_server_request(
        _message(
            {
                "id": 6,
                "method": "fs/write_text_file",
                "params": {"path": str(tmp_path / "d.txt"), "content": "x"},
            }
        )
    )
    assert not (tmp_path / "d.txt").exists()


@pytest.mark.asyncio
async def test_an_edit_approval_names_its_path_in_the_diff_item(tmp_path: Path) -> None:
    client = _client(tmp_path)
    target = tmp_path / "e.txt"
    target.write_text("hi")
    content = [
        {"type": "diff", "path": str(target), "oldText": "hi", "newText": "yo"},
        {
            "type": "content",
            "content": {"type": "text", "text": f"Requesting approval to Editing {target}"},
        },
    ]
    await _approve(client, _permission("Edit", "", content=content))
    await client._reject_unknown_server_request(
        _message(
            {
                "id": 11,
                "method": "fs/write_text_file",
                "params": {"path": str(target), "content": "yo\n"},
            }
        )
    )
    assert target.read_text() == "yo\n"


@pytest.mark.asyncio
async def test_an_approval_whose_diff_and_text_paths_differ_pays_for_nothing(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    content = [
        {"type": "diff", "path": str(tmp_path / "b.txt"), "oldText": "", "newText": "x"},
        {
            "type": "content",
            "content": {"type": "text", "text": f"Requesting approval to Editing {tmp_path}/a.txt"},
        },
    ]
    params = {"toolCall": {"title": "Edit", "content": content}}
    assert kimi_fs.approved_write_paths(params, str(tmp_path)) == ()
    await _approve(client, _permission("Edit", "", content=content))
    for n, name in enumerate(("a.txt", "b.txt")):
        await client._reject_unknown_server_request(
            _message(
                {
                    "id": 20 + n,
                    "method": "fs/write_text_file",
                    "params": {"path": str(tmp_path / name), "content": "x"},
                }
            )
        )
        assert not (tmp_path / name).exists()


@pytest.mark.asyncio
async def test_the_approval_is_recorded_before_the_answer_and_undone_when_it_is_not_sent(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    target = tmp_path / "n2.txt"
    gate = kimi_fs.kimi_fs_gate(client)
    seen: list[dict] = []

    async def _send(request_id, result, **_):
        seen.append(dict(gate.approved))

    client._send_response = _send
    msg = _permission("Write", f"Requesting approval to Writing {target}")
    await _approve(client, msg)
    assert seen == [{str(target): 1}]

    async def _fail(request_id, result, **_):
        raise OSError("pipe closed")

    client._send_response = _fail
    msg = _permission("Write", f"Requesting approval to Writing {target}")
    gate.new_turn()
    client._build_permission_event(msg)
    client._note_pi_gate_asked(msg)
    with pytest.raises(OSError):
        await client.approve_tool(msg.id)
    assert gate.approved == {} and gate.pending == {msg.id: (str(target),)}


@pytest.mark.asyncio
async def test_only_write_and_edit_approvals_pay_for_a_write(tmp_path: Path) -> None:
    client = _client(tmp_path)
    await _approve(
        client, _permission("Bash", f"Requesting approval to Running: echo > {tmp_path}/x")
    )
    await client._reject_unknown_server_request(
        _message(
            {
                "id": 12,
                "method": "fs/write_text_file",
                "params": {"path": str(tmp_path / "x"), "content": "x"},
            }
        )
    )
    assert not (tmp_path / "x").exists()


@pytest.mark.asyncio
async def test_an_approved_write_to_a_sensitive_path_is_still_refused(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path)
    monkeypatch.setattr(kimi_fs, "is_sensitive_write_path", lambda path: True)
    await _approve(client, _permission("Write", "Requesting approval to Writing s.txt"))
    await client._reject_unknown_server_request(
        _message(
            {
                "id": 13,
                "method": "fs/write_text_file",
                "params": {"path": str(tmp_path / "s.txt"), "content": "x"},
            }
        )
    )
    assert not (tmp_path / "s.txt").exists() and _errors(client)[0][0] == -32000


# ── 5. reads and the AGENTS.md probe ─────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", [".kimi-code/AGENTS.md", "AGENTS.md", "agents.md"])
async def test_the_agents_probes_answer_not_found_even_when_the_file_exists(
    tmp_path: Path, probe: str
) -> None:
    client = _client(tmp_path)
    (tmp_path / ".kimi-code").mkdir()
    (tmp_path / probe).write_text("instructions Crew did not approve")
    await client._reject_unknown_server_request(
        _message(
            {"id": 20, "method": "fs/read_text_file", "params": {"path": str(tmp_path / probe)}}
        )
    )
    assert _errors(client) == [(-32002, "Resource not found")]
    client._send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_read_in_the_cwd_is_answered_with_its_text(tmp_path: Path) -> None:
    client = _client(tmp_path)
    (tmp_path / "hello.txt").write_text("hi\n")
    await client._reject_unknown_server_request(
        _message({"id": 21, "method": "fs/read_text_file", "params": {"path": "hello.txt"}})
    )
    client._send_response.assert_awaited_with(21, {"content": "hi\n"})


@pytest.mark.asyncio
async def test_a_missing_file_is_not_found_so_an_append_starts_empty(tmp_path: Path) -> None:
    client = _client(tmp_path)
    await client._reject_unknown_server_request(
        _message(
            {"id": 22, "method": "fs/read_text_file", "params": {"path": str(tmp_path / "nope")}}
        )
    )
    assert _errors(client) == [(-32002, "Resource not found")]


@pytest.mark.asyncio
async def test_a_sensitive_read_goes_through_the_floor(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path)
    (tmp_path / "secret").write_text("S")
    monkeypatch.setattr(kimi_fs, "is_sensitive_path", lambda path: path.endswith("secret"))
    await client._reject_unknown_server_request(
        _message(
            {"id": 23, "method": "fs/read_text_file", "params": {"path": str(tmp_path / "secret")}}
        )
    )
    assert _errors(client)[0][0] == -32000
    client._send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_another_harness_still_gets_method_not_found(tmp_path: Path) -> None:
    client = AcpClient(work_dir=tmp_path, acp_backend="codex")
    client._send_error = AsyncMock()
    await client._reject_unknown_server_request(
        _message({"id": 24, "method": "fs/read_text_file", "params": {"path": "x"}})
    )
    assert client._send_error.await_args.args[1] == -32601


def test_the_declared_capabilities_are_fs_and_not_terminal() -> None:
    from kiro_crew.acp.client_native_children import client_capabilities_for

    caps = client_capabilities_for(KIMI)
    assert caps["fs"] == {"readTextFile": True, "writeTextFile": True}
    assert caps["terminal"] is False
    assert client_capabilities_for("codex")["fs"] == {"readTextFile": False, "writeTextFile": False}


# ── 6. the tripwire rows ─────────────────────────────────────────────────────

_KIMI_ROW = {
    "TodoList",
    "SetTodoList",
    "TaskList",
    "TaskOutput",
    "WaitFor",
    "CronList",
    "AskUserQuestion",
    "ExitPlanMode",
    "select_tools",
    "GetGoal",
}


def test_kimi_stays_off_the_shared_acp_runtime() -> None:
    # Not reachable: Kimi's `approve_always` is only downgraded in `client_permissions`. If Kimi
    # ever joins the set, add the `approve_always` -> `approve_once` downgrade in
    # `session_handle_permissions` first (Kimi rejects an unknown option id).
    from kiro_crew.agent_sdk.backends_sessions import ACP_BACKENDS_ACP_RUNTIME

    assert KIMI not in ACP_BACKENDS_ACP_RUNTIME


def test_the_row_holds_read_only_tools_and_nothing_that_writes_runs_or_reaches_out() -> None:
    assert HARNESS_UNASKED_TOOL_IDS[KIMI] == _KIMI_ROW
    for tool in (
        "Read", "Write", "Edit", "Bash", "TaskStop", "CronCreate", "CronDelete", "WebSearch", "FetchURL",
        "ReadMediaFile", "Grep", "Glob", "Skill", "Agent", "AgentSwarm", "NotifyUser", "CreateGoal",
        "SetGoalBudget", "UpdateGoal",
    ):  # fmt: skip
        assert tool not in HARNESS_UNASKED_TOOL_IDS[KIMI]
    assert KIMI in ACP_BACKENDS_UNASKED_NATIVE_TOOLS
    assert set(HARNESS_UNASKED_TOOL_IDS) == set(ACP_BACKENDS_UNASKED_NATIVE_TOOLS)
    assert KIMI not in ACP_BACKENDS_HONOR_ZERO_TOOL_BAN


def _tripwire_client(tmp_path: Path) -> AcpClient:
    client = AcpClient.__new__(AcpClient)
    client._acp_backend = KIMI
    client._unasked_shell_tripwire = True
    client._pi_gate_asked_ids = set()
    client._pi_gate_denied_ids = set()
    client._pi_gate_request_tool = {}
    client._tool_call_is_shell = {}
    client._tool_call_mcp_server = {}
    client._tool_call_harness_tool_name = {}
    client._tool_call_params = {}
    client._work_dir = tmp_path
    client._session_id = "s1"
    client._session_key = "k1"
    client._kill_process = AsyncMock(return_value=None)
    return client


def _completed(tool_call_id: str, **extra) -> JsonRpcMessage:
    update = {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tool_call_id,
        "status": "completed",
    }
    update.update(extra)
    return JsonRpcMessage(method="session/update", params={"sessionId": "s1", "update": update})


@pytest.fixture
def _audit(monkeypatch) -> list[dict]:
    written: list[dict] = []

    def _log(**fields):
        if fields.get("session_key") != "sandbox":
            written.append(fields)

    monkeypatch.setattr(sel_module, "sel", lambda: SimpleNamespace(log_tool_invocation=_log))
    return written


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(_KIMI_ROW))
async def test_a_row_tool_that_runs_unasked_does_not_end_the_session(
    tmp_path: Path, name: str
) -> None:
    client = _tripwire_client(tmp_path)
    client._tool_call_harness_tool_name["t1"] = name
    await client._tripwire_pi_gate(_completed("t1"))
    client._kill_process.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["Read", "Write", "Edit", "Bash", "Agent", "AgentSwarm", "Grep", "WebSearch", ""]
)
async def test_any_other_tool_that_runs_unasked_ends_the_session(
    tmp_path: Path, _audit, name: str
) -> None:
    client = _tripwire_client(tmp_path)
    client._tool_call_harness_tool_name["t1"] = name
    with pytest.raises(AcpToolGateUnroutable, match="without asking"):
        await client._tripwire_pi_gate(_completed("t1"))
    assert _audit[0]["outcome"] == "ran_without_gate"


@pytest.mark.asyncio
async def test_the_live_unasked_write_capture_still_trips_and_the_asked_one_does_not(
    tmp_path: Path, _audit
) -> None:
    # Before: the capture where Write ran with no request (default config, no Crew home).
    client = _tripwire_client(tmp_path)
    for frame in _frames("turn-live-write-unasked"):
        msg = _message(frame)
        update = (frame.get("params") or {}).get("update") or {}
        if update.get("sessionUpdate") == "tool_call":
            client._tool_call_harness_tool_name[update["toolCallId"]] = update["title"]
        if update.get("status") == "completed":
            with pytest.raises(AcpToolGateUnroutable):
                await client._tripwire_pi_gate(msg)
            break
    # After: the Crew-home capture, where the request came first.
    client = _tripwire_client(tmp_path)
    for frame in _frames("turn-live-write-asked"):
        msg = _message(frame)
        update = (frame.get("params") or {}).get("update") or {}
        if frame.get("method") == "session/request_permission":
            client._note_pi_gate_asked(msg)
            client._pi_gate_request_tool.clear()  # Crew answered it
        if update.get("sessionUpdate") == "tool_call":
            client._tool_call_harness_tool_name[update["toolCallId"]] = update["title"]
        if update.get("status") == "completed":
            await client._tripwire_pi_gate(msg)
    client._kill_process.assert_not_awaited()


# ── 7. one place for the list ────────────────────────────────────────────────


def test_the_disabled_tools_are_one_named_constant() -> None:
    assert isinstance(kimi_mod.KIMI_DISABLED_TOOLS, tuple)
    assert not hasattr(kimi_mod, "KIMI_SEED_LEAVES")


# ── 8. edge cases from the live edge round (reports/kimi-edge-tests.md) ─────


@pytest.mark.asyncio
async def test_a_load_of_a_yolo_session_reports_default_and_crew_still_writes_the_pin(
    tmp_path: Path,
) -> None:
    """Case 7: ``session/load`` of a session saved in yolo reports ``default`` while yolo is
    live, and Bash ran with no request. The report cannot be trusted, so the pin is written
    whatever the load said. Kimi 2.7.2's ``setSessionConfigOption`` for ``mode`` calls the same
    ``AcpSession.setMode`` -> ``agent.setPermission`` as ``session/set_mode``, with no early
    return on an unchanged value, so the write restores asking the way ``set_mode`` did in the
    capture (its ``current_mode_update`` and ``config_option_update`` are replayed below)."""
    frames = _frames("session-load-yolo-live")
    load = next(f for f in frames if "modes" in (f.get("result") or {}))
    assert load["result"]["modes"]["currentModeId"] == "default"
    reported = [o for o in load["result"]["configOptions"] if o["id"] == "mode"]
    assert reported[0]["currentValue"] == "default"
    updates = [
        f["params"]["update"]
        for f in frames
        if f.get("method") == "session/update"
        and f["params"]["update"].get("sessionUpdate") in ("tool_call", "tool_call_update")
    ]
    bash = next(u["toolCallId"] for u in updates if u.get("title") == "Bash")
    assert not any(f.get("method") == "session/request_permission" for f in frames)
    assert any(u.get("status") == "completed" and u["toolCallId"] == bash for u in updates)

    client = _client(tmp_path)
    client._acp_config_options = load["result"]["configOptions"]
    after = [
        f["params"]["update"]
        for f in frames
        if f.get("method") == "session/update"
        and f["params"]["update"].get("sessionUpdate") == "config_option_update"
    ]

    sent: list[tuple[str, dict]] = []

    async def _send_request(method: str, params: dict) -> int:
        sent.append((method, params))
        return 99

    async def _wait(request_id: int, timeout: float) -> dict:
        assert request_id == 99
        return {"configOptions": after[-1]["configOptions"]}

    client._session_id = "sess"
    client._send_request = _send_request  # type: ignore[method-assign]
    client._wait_for_response = _wait  # type: ignore[method-assign]
    await client._apply_session_permission_routing()
    # Written although the load already said ``default``: the real ``set_config_option`` sent
    # the request, stored the reply's options, and the read-back finds no drift.
    assert sent == [
        (
            "session/set_config_option",
            {"sessionId": "sess", "configId": "mode", "value": "default"},
        )
    ]
    assert client._acp_config_options == after[-1]["configOptions"]
    assert session_config_drift_issue(KIMI, after[-1]) == ""


def test_a_load_is_followed_by_the_pin_on_the_one_start_path() -> None:
    """The load branch and the new branch of ``_initialize_session`` both fall through to the
    single routing write, so a resumed Kimi session is never prompted before the pin."""
    root = Path(__file__).resolve().parents[1] / "src" / "kiro_crew" / "acp"
    start = (root / "client_start.py").read_text()
    body = start[start.index("async def _initialize_session") :]
    body = body[: body.index("\nasync def ", 10)]
    assert body.count("await self._apply_session_permission_routing()") == 1
    assert body.index("METHOD_SESSION_LOAD") < body.index(
        "await self._apply_session_permission_routing()"
    )


def test_auto_stays_forbidden_because_it_ignores_ask_rules() -> None:
    """Case 2: in ``auto`` Kimi ignores every ask rule, so it is refused like ``yolo``."""
    from kiro_crew.agent_sdk.backends import ACP_BACKEND_FORBIDDEN_MODES

    assert ACP_BACKEND_FORBIDDEN_MODES[KIMI] == frozenset({"yolo", "auto"})
    assert session_config_drift_issue(
        KIMI, {"sessionUpdate": "current_mode_update", "currentModeId": "auto"}
    )


def test_crew_writes_no_hook_so_no_guarantee_rests_on_one() -> None:
    """Case 3: Kimi's PreToolUse hooks fail open on a timeout and on exit 1."""
    config = tomllib.loads(kimi_mod.render_kimi_config("/crew/kimi-home"))
    assert "hooks" not in config


def test_signed_out_at_session_new_is_the_signed_out_remedy() -> None:
    """Windows (flamingdragon, Node 24.17): signed out, ``session/new`` fails with -32000
    "Authentication required" instead of narrowing the thinking select."""
    from kiro_crew.acp.transport_errors import AcpError, _raise_acp_error
    from kiro_crew.agent_sdk import host_auth

    frame = {"code": -32000, "message": "Authentication required"}
    # What the start path sees, as the live log printed it.
    assert host_auth.reports_signed_out(KIMI, f"JSON-RPC error: {frame}")
    assert not host_auth.reports_signed_out("codex", f"JSON-RPC error: {frame}")
    with pytest.raises(AcpError) as raised:
        _raise_acp_error(frame, backend=KIMI)
    assert "kimi-home kimi login" in str(raised.value)


def test_a_home_plugins_manifest_refuses_the_spawn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    home = kimi_mod.kimi_home_dir()
    (home / "credentials").mkdir(parents=True)
    (home / "credentials" / "a.json").write_text("{}")
    (home / "plugins").mkdir()
    (home / "plugins" / "installed.json").write_text("{}")
    with pytest.raises(AcpToolGateUnroutable) as raised:
        kimi_mod.prepare_kimi_home()
    assert "plugins" in str(raised.value) and raised.value.user_worded is True


def _auth(command: str, *args: str) -> dict:
    return {
        "authMethods": [
            {"_meta": {"terminal-auth": {"command": command, "args": list(args or ["login"])}}}
        ]
    }


@pytest.mark.parametrize(
    "hostile",
    [
        _auth('C:\\x" & calc & "'),
        _auth(r"C:\%COMSPEC%\kimi.exe"),
        _auth("C:\\kimi.exe", "login", "a\u2019; calc; \u2019b"),
        _auth("C:\\kimi.exe", "login", "`calc`"),
        _auth("C:\\kimi.exe", "login", "$(calc)"),
        _auth(r"C:\tools\powershell.exe", "login"),
    ],
    ids=["cmd-quote", "cmd-percent", "ps-smart-quote", "ps-backtick", "ps-dollar", "not-kimi"],
)
def test_a_hostile_terminal_auth_falls_back_to_the_bare_command(hostile) -> None:
    assert kimi_mod.kimi_terminal_auth(hostile) == ()
    for shell in ("powershell", "cmd", "posix"):
        assert "calc" not in kimi_mod.kimi_login_command(
            _WIN_HOME, shell, kimi_mod.kimi_terminal_auth(hostile)
        )
    assert "kimi login --region global" in kimi_mod.kimi_login_command(_WIN_HOME, "cmd", ())


@pytest.mark.parametrize("command", ["kimi", "C:\\n\\kimi.cmd", "/opt/k i/kimi", "C:\\a\\kimi.exe"])
def test_a_plain_kimi_launcher_is_kept(command) -> None:
    assert kimi_mod.kimi_terminal_auth(_auth(command)) == (command, "login")
