"""Moonshot's Kimi Code CLI, launched one process per session by ``AcpClient``.

Kimi Code CLI serves ACP from its own binary (``kimi acp``), so its executable comes
off the shared self-served ladder (``ACP_BACKEND_LAUNCH``) and there is no adapter
package. DORMANT: the id can be spelled and launched, but it is not selectable
(it is in the ``NOT_SHIPPED_SELECTABLE`` set of ``test/test_agent_backend_editable.py``, not in ``BASELINE_SELECTABLE_BACKENDS``; the Kimi worked example in harness-onboarding.md lists
each decision still waiting on a live capture). Seven things are its own:

* **Its permission route is a session config option**, ``mode``, which Crew writes
  ``default`` after ``session/new`` and ``session/load`` and before the first prompt
  (``Routing.SESSION_CONFIG``). ``yolo`` and ``auto`` approve tools on their own and
  are refused mid-session (``ACP_BACKEND_FORBIDDEN_MODES``). Crew never calls
  ``session/set_mode``: Kimi's ``set_config_option(mode)`` runs the same ``setMode`` with no
  early return, so it also corrects a ``yolo`` config that ``session/new`` reports as ``default``.
* **Kimi runs out of a Crew-owned home** (:func:`prepare_kimi_home`). ``KIMI_CODE_HOME`` is the
  only way to give Kimi a config, and its config is what makes ``Write`` and ``Edit`` ask
  (otherwise they run unasked in a git cwd) and what removes the tools that read or reach the
  network without asking (:data:`KIMI_DISABLED_TOOLS`). It holds its own sign-in
  (:func:`kimi_login_command`); nothing is copied from the person's own Kimi home.
* **Crew never grants Kimi a session approval.** ``approve_always`` is answered ``approve_once``
  (``AcpClient.approve_tool``, because ``_unasked_shell_tripwire`` is set), since a saved
  approval skips Kimi's ask rules, survives a reload, and is invisible to the tripwire.
* **Crew serves Kimi's ``fs/*`` requests** (:mod:`kiro_crew.acp.harness.kimi_fs`).
* **The name ``kimi`` is shared with the archived Python ``kimi-cli``**, which also
  answers ``kimi acp``. The handshake therefore checks ``agentInfo.name`` and the
  verified release (``acp.client_kimi_session``), and the install probe reads the
  release off the npm manifest.
* **The child never reports usage to Moonshot and never updates itself mid-run**:
  ``KIMI_DISABLE_TELEMETRY=1`` and ``KIMI_CODE_NO_AUTO_UPDATE=1`` (:data:`_CHILD_ENV`); its
  config watcher is off by ``KIMI_CODE_WATCH=0`` (which does NOT stop a rewritten
  ``config.toml`` reaching the next ``session/new``: Crew's per-spawn re-render and the write deny
  on the home are the guard), and an inherited
  ``KIMI_CODE_TRUST_WORKSPACE`` is stripped.
* **A project's own ``.kimi-code/`` directory is refused.** It can hold a local
  config, ``mcp.json`` and permission ``allow`` rules, none of which Crew
  approved (:func:`kimi_repo_config_issue`).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping
from typing import Any
from pathlib import Path
from urllib.parse import urlsplit

from kiro_crew import acp_tool_gate
from kiro_crew.acp import launch as launch_mod
from kiro_crew.acp.harness.base import ProcessAdapter, ProcessSession, SpawnContext, SpawnPlan
from kiro_crew.acp.harness.copilot import _ancestors as _project_ancestors
from kiro_crew.acp.transport_errors import AcpToolGateUnroutable
from kiro_crew.agent_sdk.backends import ACP_BACKEND_KIMI
from kiro_crew.agent_sdk.tool_gate import KIMI_HOME_LEAF
from kiro_crew.config.paths import config_dir
from kiro_crew.sandbox import crew_home_visible_spellings

__all__ = [
    "KIMI_AGENT_NAME",
    "KIMI_DISABLED_TOOLS",
    "KIMI_MIN_NODE",
    "KIMI_NPM_PACKAGE",
    "KIMI_PROJECT_DIR",
    "KimiLaunch",
    "kimi_command_from_update",
    "kimi_installed_version",
    "kimi_node_found",
    "kimi_node_message",
    "kimi_home_dir",
    "kimi_login_command",
    "kimi_repo_config_issue",
    "kimi_signed_in",
    "kimi_terminal_auth",
    "kimi_signed_out",
    "kimi_home_unreadable_message",
    "kimi_signed_out_refusal",
    "kimi_signed_out_message",
    "prepare_kimi_home",
    "render_kimi_config",
]

#: ``agentInfo.name`` Kimi Code CLI reports on ``initialize`` (observed on 2.7.2).
#: The archived Python kimi-cli reports another name, which is how the two are told
#: apart: both install a command called ``kimi`` and both serve ``kimi acp``.
KIMI_AGENT_NAME = "Kimi Code CLI"

#: The npm package Kimi Code CLI ships as, whose manifest names the installed release.
KIMI_NPM_PACKAGE = "@moonshot-ai/kimi-code"

#: The project directory Kimi Code reads local configuration from.
KIMI_PROJECT_DIR = ".kimi-code"

#: The oldest Node.js an npm-installed Kimi Code CLI 2.7.2 runs on (its manifest says
#: ``engines.node >=22.19.0``; on Node 20 the binary dies at import with a
#: ``SyntaxError`` about ``node:zlib`` before it can answer ``initialize``).
KIMI_MIN_NODE = (22, 19, 0)

#: How many directories above the resolved binary a manifest is looked for in. The
#: npm bin link resolves to ``<package>/dist/main.mjs``, so two levels reach the
#: package root; the margin covers a loader moved one level.
_MANIFEST_SEARCH_DEPTH = 3

#: The tools Kimi's config removes for every agent (``[tools] disabled``), so the model never
#: sees them and Kimi refuses them at execution. Each either reaches the network
#: (``WebSearch``, ``FetchURL``), reads media or any path without an ask rule that can fence it
#: (``ReadMediaFile``, ``Grep``, ``Glob``: their rule subject is the search pattern, not the
#: path, and they run unasked), or puts text from a skill folder in the
#: model's context with no ask (``Skill``; its builtin ``update-config`` and ``mcp-config`` skills
#: also coach edits Crew denies). ``Bash`` stays
#: and asks, so a search still goes through the person. Source: ``reports/kimi-levers-source.md``
#: section 10, read fencing choice A; live proof for the first four in ``kimi-levers-acp.md``.
KIMI_DISABLED_TOOLS = ("WebSearch", "FetchURL", "ReadMediaFile", "Grep", "Glob", "Skill")

#: The top-level keys of the home's previous ``config.toml`` that survive Crew's rewrite: what
#: ``kimi login`` writes for its managed provider and models (seen in a signed-in 2.7.2 home).
_KEPT_CONFIG_KEYS = frozenset({"default_model", "providers", "models", "thinking"})

#: Set in the child so the session reports nothing to the vendor, a native install does not
#: swap itself for a new build between runs, and the config file watcher stays off even if the
#: config seed fails. Neither this nor ``[watch]`` stops a rewrite reaching the next
#: ``session/new`` (spike X1 S2); the guard is the per-spawn re-render and the write deny.
#: ``KIMI_CODE_HOME`` is added per spawn.
_CHILD_ENV = {
    "KIMI_DISABLE_TELEMETRY": "1",
    "KIMI_CODE_NO_AUTO_UPDATE": "1",
    "KIMI_CODE_WATCH": "0",
    # Without it a late-connecting MCP server's tools never reach the model (Stage 7). With
    # ``[mcp] default_deferred`` the tools are announced and loaded through ``select_tools``
    # (spike M1, ``reports/w1-spikes-mcp.md``). Experimental in Kimi 2.7.2.
    "KIMI_CODE_EXPERIMENTAL_TOOL_SELECT": "1",
}

#: Removed from the child: set, it marks every repo trusted, which would let a project's own
#: MCP servers and permission rules run without Crew's ask.
_STRIPPED_ENV = ("KIMI_CODE_TRUST_WORKSPACE",)

#: Characters that would change what a permission pattern built from a path matches.
_GLOB_CHARS = frozenset("*?[]{}()!\\")

#: A Windows drive or UNC path. Kimi's globs want ``/``, so only these have ``\\`` spelled as ``/``;
#: on POSIX a backslash is a glob escape and stays refused.
_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")

#: The hosts ``kimi login`` talks to. A kept provider whose ``base_url`` or ``oauth_host`` points
#: anywhere else would send the sign-in there, so it is refused (a tampered home config).
_KIMI_HOST_SUFFIXES = ("kimi.com", "kimi.ai", "moonshot.cn", "moonshot.ai")

#: Sub-keys of the kept tables that survive a rewrite, named from a signed-in 2.7.2 home. A key
#: not listed (a command-style or unknown one) is dropped.
_KEPT_PROVIDER_KEYS = frozenset({"type", "base_url", "api_key", "oauth"})
_KEPT_OAUTH_KEYS = frozenset({"storage", "key", "oauth_host"})
_KEPT_MODEL_KEYS = frozenset(
    {
        "provider",
        "model",
        "max_context_size",
        "capabilities",
        "display_name",
        "support_efforts",
        "default_effort",
    }
)
_KEPT_THINKING_KEYS = frozenset({"enabled"})
_DEFERRED_TOOLS = "dynamically_loaded_tools"


def kimi_installed_version(kimi_bin: str) -> str:
    """The release the npm install *kimi_bin* runs, or ``""`` when unreadable.

    Read from ``@moonshot-ai/kimi-code``'s own ``package.json``, never by running the
    binary: the install probe is polled by the dashboard. A standalone build or the
    archived Python ``kimi-cli`` carries no such manifest and answers ``""``: an
    unread version is not a wrong one, and the handshake still checks the name and
    the release. Blocking (reads files); the probe runs off the loop.
    """
    try:
        here = Path(os.path.realpath(kimi_bin)).parent
    except (OSError, ValueError):
        return ""
    # A POSIX bin link resolves to ``<package>/dist/main.mjs``, so the manifest is above it. A
    # Windows npm shim (``kimi.cmd``) sits beside ``node_modules``, so it is BELOW it.
    manifests = [here / "node_modules" / "@moonshot-ai" / "kimi-code" / "package.json"]
    manifests += [d / "package.json" for d in [here, *here.parents][:_MANIFEST_SEARCH_DEPTH]]
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("name") == KIMI_NPM_PACKAGE:
            version = data.get("version")
            return version.strip() if isinstance(version, str) else ""
    return ""


def kimi_command_from_update(update: Mapping[str, object]) -> str:
    """The shell command Kimi states in a ``tool_call_update``'s content, or ``""``.

    Kimi renders a call's arguments as ONE text block holding a JSON object
    (``{"command":"echo hi"}``, ``test/fixtures/acp_frames/kimi/turn-live-permission-bash``).
    Anything else (several blocks, text that is not an object, a missing or non-string
    ``command``) yields ``""``, so the caller recovers nothing and the call is refused.
    """
    content = update.get("content")
    if not isinstance(content, list) or len(content) != 1:
        return ""
    block = content[0]
    inner = block.get("content") if isinstance(block, dict) else None
    text = inner.get("text") if isinstance(inner, dict) else None
    if not isinstance(text, str):
        return ""
    try:
        args = json.loads(text)
    except ValueError:
        return ""
    command = args.get("command") if isinstance(args, dict) else None
    return command if isinstance(command, str) else ""


def kimi_node_found(kimi_bin: str) -> str:
    """The ``node`` version that is too old for *kimi_bin*, ``"none"`` when there is no
    ``node`` on PATH, or ``""`` when it is fine or cannot be judged. Blocking.

    Only an npm install (one with the package manifest) runs on the PATH's ``node``; a
    standalone build or the archived Python kimi-cli answers ``""``. An unreadable
    version is not a wrong one, so it also answers ``""``.
    """
    if not kimi_installed_version(kimi_bin):
        return ""
    node = shutil.which("node")
    if not node:
        return "none"
    try:
        out = subprocess.run(  # noqa: S603 -- the node on PATH, fixed argument
            [node, "--version"], capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)\S*", out)
    if not match or tuple(int(part) for part in match.groups()) >= KIMI_MIN_NODE:
        return ""
    return out


def kimi_node_message(found: str) -> str:
    """The remedy for a Node.js too old (or missing) for Kimi Code CLI."""
    needed = ".".join(str(part) for part in KIMI_MIN_NODE)
    have = "no node on PATH" if found == "none" else f"node {found} on PATH"
    return (
        f"Kimi Code CLI needs Node.js {needed} or newer, and this host has {have}. "
        f"On Node 20 it crashes at start. Install Node.js {needed} or newer, put it "
        "first on the gateway's PATH, restart Kiro Crew, then start a new chat."
    )


def kimi_signed_out(config_options: object) -> bool:
    """True when ``session/new``'s ``thinking`` select offers only ``on``.

    Observed on 2.7.2: signed out, Kimi answers ``session/prompt`` with nothing at all
    (no error, no -32000), so there is no error text to match. The same ``session/new``
    advertises a ``thinking`` select with the single value ``on``; signed in it offers
    ``low``, ``high`` and ``max``. The check reads that on the handshake, before any
    prompt, so a signed-out session ends with the sign-in remedy rather than a hang.
    shortcut: this is Kimi's own UI state read as a sign-in fact; if a release stops
    narrowing the list, the check goes quiet (a signed-out prompt then hangs, as before).
    """
    if not isinstance(config_options, list):
        return False
    for option in config_options:
        if isinstance(option, dict) and option.get("id") == "thinking":
            values = [o.get("value") for o in option.get("options", []) if isinstance(o, dict)]
            return values == ["on"]
    return False


def _toml_str(value: str) -> str:
    """*value* as a TOML basic string (JSON's escapes are TOML's for the characters it escapes)."""
    return json.dumps(value, ensure_ascii=False)


def _toml_value(value: object) -> str | None:
    """A scalar or a list of scalars as TOML, or ``None`` for anything else (it is dropped)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, list) and not any(isinstance(item, (list, dict)) for item in value):
        items = [_toml_value(item) for item in value]
        if None not in items:
            return "[" + ", ".join(items) + "]"  # type: ignore[arg-type]
    return None


def _toml_tables(table: dict, path: tuple[str, ...]) -> list[str]:
    """*table* as ``[a."b"]`` sections: its scalars under its own header, then its subtables."""
    lines: list[str] = []
    scalars = [(k, _toml_value(v)) for k, v in table.items() if not isinstance(v, dict)]
    scalars = [(k, v) for k, v in scalars if v is not None]
    if scalars and path:
        lines += ["", "[" + ".".join(_toml_str(part) for part in path) + "]"]
        lines += [f"{_toml_str(k)} = {v}" for k, v in scalars]
    for key, value in table.items():
        if isinstance(value, dict):
            lines += _toml_tables(value, (*path, key))
    return lines


def _pattern_path(path: str, what: str) -> str:
    """*path* for use inside a permission pattern, or a refusal when it could widen the pattern."""
    if _WINDOWS_PATH.match(path):
        path = path.replace("\\", "/")
    if _GLOB_CHARS.intersection(path):
        raise AcpToolGateUnroutable(
            f"Kimi Code's permission rules match paths as globs, and {what} ({path}) holds a "
            "character that would change what a rule matches (one of * ? [ ] { } ( ) ! or a "
            "backslash), so Kiro Crew does not start a Kimi Code session. Move the Kiro Crew "
            "data folder or pick another harness in Settings › Agent Harness.",
            transient=False,
            user_worded=True,
        )
    return path


def _kimi_host(value: object) -> bool:
    """Whether *value* (a URL or a bare host) names a Moonshot/Kimi host over https."""
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value if "://" in value else f"https://{value}")
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(
        host == suffix or host.endswith("." + suffix) for suffix in _KIMI_HOST_SUFFIXES
    )


def _keep(table: object, allowed: frozenset[str]) -> dict:
    return {k: v for k, v in table.items() if k in allowed} if isinstance(table, dict) else {}


def _with_deferred_tools(model: dict) -> dict:
    """*model* with ``dynamically_loaded_tools`` among its capabilities.

    Without it Kimi ignores ``[mcp] default_deferred`` for that model and a deferred MCP tool is
    never offered (spike M1). The login writes none of its own, so it is added to every kept model.
    """
    capabilities = model.get("capabilities")
    capabilities = list(capabilities) if isinstance(capabilities, list) else []
    if _DEFERRED_TOOLS not in capabilities:
        capabilities.append(_DEFERRED_TOOLS)
    return {**model, "capabilities": capabilities}


def _kept_tables(kept: Mapping[str, object]) -> dict[str, dict]:
    """The ``providers``, ``models`` and ``thinking`` of *kept*, cut to the allowed sub-keys.

    Refused (never silently repaired) when a provider's ``base_url`` or ``oauth_host`` is not a
    Kimi host: the sign-in's token goes to that host.
    """
    providers: dict[str, dict] = {}
    raw = kept.get("providers")
    for name, provider in (raw.items() if isinstance(raw, dict) else ()):
        entry = _keep(provider, _KEPT_PROVIDER_KEYS)
        if "oauth" in entry:
            entry["oauth"] = _keep(entry["oauth"], _KEPT_OAUTH_KEYS)
        for where, value in (
            ("base_url", entry.get("base_url")),
            ("oauth.oauth_host", entry.get("oauth", {}).get("oauth_host")),
        ):
            if value is not None and not _kimi_host(value):
                raise AcpToolGateUnroutable(
                    f"Kiro Crew's Kimi home config.toml names a provider ({name}) whose {where} "
                    "is not a Kimi or Moonshot host (kimi.com, kimi.ai, moonshot.cn, "
                    "moonshot.ai), so the sign-in would be sent elsewhere and Kiro Crew does not "
                    "start a Kimi Code session. Delete Kiro Crew's kimi-home/config.toml and "
                    f"sign in again: {kimi_login_command()}",
                    transient=False,
                    user_worded=True,
                )
        providers[name] = entry
    models = kept.get("models")
    return {
        "providers": providers,
        "models": {
            name: _with_deferred_tools(_keep(model, _KEPT_MODEL_KEYS))
            for name, model in (models.items() if isinstance(models, dict) else ())
        },
        "thinking": _keep(kept.get("thinking"), _KEPT_THINKING_KEYS),
    }


def render_kimi_config(home: str, kept: Mapping[str, object] | None = None) -> str:
    """The ``config.toml`` of Crew's Kimi home.

    Crew's part is the rules live-tested on 2.7.2 (``reports/kimi-levers-acp.md``,
    "Recommended config"): ``manual`` mode, no config hot-reload, ``Write``, ``Edit`` and ``Read``
    always ask (so they reach Crew's gate), the agent cannot write the home, and
    :data:`KIMI_DISABLED_TOOLS`. ``Read`` asks on EVERY path rather than only outside the cwd:
    the home is shared by every Crew session, so its config cannot name one session's cwd, and a
    relative ``Read(!./**)`` is no fence (picomatch matches ``**`` against any absolute path, so
    the negated rule never asks; checked against Kimi 2.7.2's ``pathGlobMatch``).

    *kept* is the part ``kimi login`` wrote and Kimi needs to reach its model, filtered to
    :data:`_KEPT_CONFIG_KEYS` and then to the sub-keys listed above; everything else a previous file held (hooks, rules, MCP, tools,
    services) is dropped, so a rewrite always restores Crew's posture.
    """
    home = _pattern_path(home, "Kiro Crew's Kimi home")
    kept = {k: v for k, v in (kept or {}).items() if k in _KEPT_CONFIG_KEYS}
    lines = ['default_permission_mode = "manual"']
    if isinstance(kept.get("default_model"), str):
        lines.append(f"default_model = {_toml_str(kept['default_model'])}")
    lines += ["", "[watch]", "enabled = false"]
    # Deferred, so a server that connects after the profile bind is still offered through
    # ``select_tools`` (spike M1). Needs the child env above and a ``kimi``-protocol provider.
    lines += ["", "[mcp]", "default_deferred = true"]
    rules = (
        ("ask", "Write"),
        ("ask", "Edit"),
        ("ask", "Read"),
        ("deny", f"Write({home}/**)"),
        ("deny", f"Edit({home}/**)"),
    )
    for decision, pattern in rules:
        lines += [
            "",
            "[[permission.rules]]",
            f'decision = "{decision}"',
            f"pattern = {_toml_str(pattern)}",
        ]
    disabled = ", ".join(_toml_str(tool) for tool in KIMI_DISABLED_TOOLS)
    lines += ["", "[tools]", f"disabled = [{disabled}]"]
    for key, table in _kept_tables(kept).items():
        lines += _toml_tables(table, (key,))
    return "\n".join(lines) + "\n"


def kimi_home_dir() -> Path:
    """Crew's own Kimi home: ``<crew data>/kimi-home``."""
    return Path(os.path.realpath(str(config_dir()))) / KIMI_HOME_LEAF


def kimi_terminal_auth(initialize_result: object) -> tuple[str, ...]:
    """Kimi's own sign-in invocation, ``(command, *args)``, from its ``initialize`` answer, or ``()``.

    Read from ``authMethods[0]._meta["terminal-auth"]`` (observed on 2.7.2: ``command`` is the
    binary Kimi runs as, ``args`` is ``["login"]``). It is the one spelling that works where a
    bare ``kimi`` is not on PATH (a Windows npm prefix). Anything not plain text is ignored, so
    the caller falls back to the bare command.
    """
    methods = initialize_result.get("authMethods") if isinstance(initialize_result, dict) else None
    meta = methods[0].get("_meta") if isinstance(methods, list) and methods else None
    auth = meta.get("terminal-auth") if isinstance(meta, dict) else None
    if not isinstance(auth, dict):
        return ()
    command, args = auth.get("command"), auth.get("args")
    if not isinstance(command, str) or not isinstance(args, list):
        return ()
    parts = (command, *args)
    if not all(isinstance(p, str) and p and p.isprintable() for p in parts):
        return ()
    # The text is printed to be pasted into a terminal, and it comes from the binary's own
    # answer. Anything a cmd or PowerShell line would read as syntax is refused, and so is a
    # command that is not Kimi's launcher, so the caller prints the bare form instead.
    if any(_TERMINAL_AUTH_HOSTILE.search(p) for p in parts):
        return ()
    if re.split(r"[\\/]", command)[-1].lower() not in _TERMINAL_AUTH_COMMANDS:
        return ()
    if command.lower().endswith((".mjs", ".js")):
        return ("node", *parts)
    return parts


#: cmd metacharacters and ``%``; PowerShell's typographic quotes (U+2018 to U+201B, which it
#: reads as quote marks), backtick and ``$``; and the plain quote.
_TERMINAL_AUTH_HOSTILE = re.compile("[\"%&|<>^`$\u2018-\u201b]")
_TERMINAL_AUTH_COMMANDS = frozenset({"kimi", "kimi.cmd", "kimi.exe", "main.mjs"})


def kimi_login_command(
    home: Path | str | None = None,
    shell: str | None = None,
    terminal_auth: tuple[str, ...] = (),
) -> str:
    """The one command that signs Crew's Kimi home in: a device login of its own, for *shell*.

    *shell* is ``posix``, ``powershell`` or ``cmd``; by default the one this host's terminal
    most likely is. ``--region global`` is always passed: ``kimi login`` defaults to the
    mainland-China site and a global account cannot sign in there. One fixed region is the
    simplest; a mainland account can run the command with ``--region cn`` instead.

    *terminal_auth* (:func:`kimi_terminal_auth`) replaces the bare ``kimi login`` with the
    invocation Kimi itself reported, for a host where ``kimi`` is not on PATH.
    """
    path = str(home or kimi_home_dir())
    shell = shell or ("powershell" if os.name == "nt" else "posix")
    if shell == "powershell":
        login = _LOGIN
        if terminal_auth:
            first, *rest = map(_ps_quote, terminal_auth)
            login = " ".join(["&", first, *rest, _REGION])
        return f"$env:KIMI_CODE_HOME={_ps_quote(path)}; {login}"
    if shell == "cmd":
        login = " ".join([*(f'"{part}"' for part in terminal_auth), _REGION])
        # Quoted whole, so a space in the path is not part of the command and none trails it.
        return f'set "KIMI_CODE_HOME={path}" && {login if terminal_auth else _LOGIN}'
    login = " ".join([*map(shlex.quote, terminal_auth), _REGION])
    return f"KIMI_CODE_HOME={shlex.quote(path)} {login if terminal_auth else _LOGIN}"


def _ps_quote(part: str) -> str:
    return "'" + part.replace("'", "''") + "'"


_REGION = "--region global"
_LOGIN = f"kimi login {_REGION}"


def kimi_signed_in(home: Path) -> bool:
    """Whether *home* holds a sign-in: a regular token file under ``credentials/``. Blocking."""
    try:
        return any(
            entry.is_file() and not entry.is_symlink() and entry.suffix == ".json"
            for entry in (home / "credentials").iterdir()
        )
    except PermissionError:
        # An unreadable home is not a signed-out one; the caller names which it is.
        raise
    except OSError:
        return False


def kimi_signed_out_message(
    home: Path | str | None = None, terminal_auth: tuple[str, ...] = ()
) -> str:
    """The remedy for a Crew Kimi home with no (or an expired) sign-in."""
    shells = ("powershell", "cmd") if os.name == "nt" else ("posix",)
    commands = " or ".join(
        f"`{kimi_login_command(home, shell, terminal_auth)}`" for shell in shells
    )
    return (
        "Kimi Code CLI is not signed in for Kiro Crew. Kiro Crew runs Kimi out of its own "
        "home with its own sign-in, so your own kimi login is not used (sharing one would "
        "make the two copies sign each other out). In a terminal on the gateway host run "
        f"{commands}, complete the device-code sign-in, then start a new chat."
    )


def kimi_signed_out_refusal(home: Path | None = None, terminal_auth: tuple[str, ...] = ()) -> str:
    """The refusal for a handshake that reports Kimi signed out. Blocking (one directory read).

    Kimi shows the signed-out UI both when its home holds no sign-in and when it cannot
    read the home it was given. Crew reads the same home from outside the sandbox before
    the spawn, so a sign-in the gateway can see that the child cannot means the home is
    hidden from the child, and telling the person to sign in again would not help.
    """
    home = home or kimi_home_dir()
    try:
        if not kimi_signed_in(home):
            return kimi_signed_out_message(home, terminal_auth)
    except OSError:
        return kimi_home_unreadable_message(home)
    return (
        f"Kimi Code CLI reports it is not signed in, but Kiro Crew's Kimi home {home} holds "
        "a sign-in. The Kimi process most likely cannot read that folder, because Kiro Crew's "
        "sandbox hides the path from it (a data folder moved under a hidden parent). Move the "
        "Kiro Crew data folder out from under it, then start a new chat."
    )


def kimi_home_unreadable_message(home: Path | str) -> str:
    """The refusal for a Kimi home this process may not read: not a sign-in problem."""
    return (
        f"Kiro Crew cannot read its own Kimi home {home}, so it cannot tell whether Kimi is "
        "signed in. Check that this folder and the Kiro Crew data folder above it are readable "
        "by the gateway (and not hidden by a sandbox it runs in), then start a new chat."
    )


def prepare_kimi_home() -> str:
    """Create Crew's Kimi home if needed, rewrite its ``config.toml``, and return its path.

    Blocking (file work); the caller runs it off the loop. One home per Kiro Crew install,
    ``<crew data>/kimi-home`` at mode 0700, and ONE sign-in of its own, made by
    :func:`kimi_login_command`. Nothing is copied or linked from the person's own Kimi home:
    Kimi rotates its refresh token on each refresh, so two homes holding the same token sign
    each other out (observed: a copied token invalidated the person's own login). One home per
    install rather than per session or per cwd, because each home would need its own device
    login. Refused, with that command, while the home holds no sign-in.

    The config is rewritten at every spawn, last and atomically, keeping only what
    :func:`render_kimi_config` keeps of the previous file (what ``kimi login`` wrote).
    """
    home = kimi_home_dir()
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    try:
        signed_in = kimi_signed_in(home)
    except PermissionError:
        raise AcpToolGateUnroutable(
            kimi_home_unreadable_message(home), transient=False, user_worded=True
        ) from None
    if not signed_in:
        raise AcpToolGateUnroutable(
            kimi_signed_out_message(home), transient=False, user_worded=True
        )
    if os.path.lexists(home / "mcp.json"):
        # Crew's MCP servers arrive on the ``session/new`` array. Kimi also reads this file, and
        # ``mcp__*`` is no longer in ``[tools] disabled``, so a server named here would be offered
        # with none of Crew's say.
        raise AcpToolGateUnroutable(
            f"Kiro Crew's Kimi home holds an mcp.json ({home / 'mcp.json'}), which would mount "
            "MCP servers Kiro Crew did not approve, so Kiro Crew does not start a Kimi Code "
            "session. Delete that file, then start a new chat.",
            transient=False,
            user_worded=True,
        )
    if os.path.lexists(home / "plugins" / "installed.json"):
        # Kimi loads a plugin's MCP servers from here whatever its workspace trust says.
        raise AcpToolGateUnroutable(
            f"Kiro Crew's Kimi home holds installed plugins ({home / 'plugins' / 'installed.json'}), "
            "which would mount MCP servers Kiro Crew did not approve, so Kiro Crew does not "
            "start a Kimi Code session. Delete the plugins folder, then start a new chat.",
            transient=False,
            user_worded=True,
        )
    try:
        with open(home / "config.toml", "rb") as handle:
            previous = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        previous = {}
    config = render_kimi_config(str(home), previous)
    fd, temp = tempfile.mkstemp(prefix=".config.toml.", suffix=".tmp", dir=home)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(config)
        os.replace(temp, home / "config.toml")
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return str(home)


def kimi_repo_config_issue(work_dir: object) -> str:
    """Why a Kimi Code session must not start in *work_dir*, or ``""``. Blocking.

    A project ``.kimi-code`` can hold ``mcp.json``, ``local.toml`` and ``[[permission.rules]]``
    allow rules; any of them could run or pre-approve a tool without Kiro Crew's approval.
    Hooks are NOT among them: Kimi 2.7.2 reads hooks from user-level config only
    (Kimi's hooks guide: "no project-level hook file"). Refused on
    presence, in the working directory and every parent up to its git root or the user's home
    (never ``~/.kimi-code`` or ``KIMI_CODE_HOME``, which are Kimi's own home), along the path
    as given AND as resolved, with Copilot's walk. A link or an unreadable entry counts as
    present (``lexists``).
    Kimi's workspace trust gates project MCP and project permission rules, but a headless
    ACP session cannot answer its prompt and that was not measured, so the refusal stays.
    """
    if not work_dir:
        return ""
    given = os.path.abspath(str(work_dir))
    seen: set[str] = set()
    home = os.path.realpath(os.path.expanduser("~"))
    user_homes = {os.path.join(home, KIMI_PROJECT_DIR)}
    if override := os.environ.get("KIMI_CODE_HOME"):
        user_homes.add(os.path.realpath(os.path.expanduser(override)))
    for directory in [*_project_ancestors(given), *_project_ancestors(os.path.realpath(given))]:
        if directory in seen:
            continue
        seen.add(directory)
        if os.path.realpath(directory) == home:
            break
        found = os.path.join(directory, KIMI_PROJECT_DIR)
        if os.path.realpath(found) in user_homes:
            continue
        project_mcp = os.path.join(directory, ".mcp.json")
        if os.path.lexists(project_mcp):
            return (
                f"This project carries an MCP server file ({project_mcp}). Kimi Code CLI "
                "reads it from the project and its parents, so it could mount servers "
                "Kiro Crew did not approve. Kiro Crew therefore does not start a Kimi Code "
                "session here; this applies only to the Kimi Code harness. Remove or "
                "rename that file, or pick another harness in Settings › Agent Harness, "
                "then start a new chat."
            )
        if os.path.lexists(found):
            return (
                f"This project carries Kimi Code CLI's own configuration ({found}). "
                "Kimi Code can load a project's hooks, MCP servers and permission rules "
                "from it without asking, so they would bypass Kiro Crew's tool approval. "
                "Kiro Crew therefore does not start a Kimi Code session here; this "
                "applies only to the Kimi Code harness. Remove or rename that directory, "
                "or pick another harness in Settings › Agent Harness, then start a new chat."
            )
    return ""


class KimiLaunch(ProcessAdapter):
    """Kimi Code CLI: a self-served binary whose asking is a session config option.

    Its MCP array is the session's own (``providers.mirrors.kimi``, warmed by the spawn):
    Crew's servers reach the model as deferred tools, announced and loaded through Kimi's
    ``select_tools`` (the home config's ``[mcp] default_deferred`` and the child env's
    ``KIMI_CODE_EXPERIMENTAL_TOOL_SELECT``; spike M1). A call to one asks like any tool.
    """

    backend = ACP_BACKEND_KIMI

    #: The Crew-owned ``KIMI_CODE_HOME`` this spawn's child runs out of, set by
    #: :meth:`resolve_spawn` and put on the child's environment by :meth:`apply_spawn_env`.
    _home: str = ""

    async def resolve_spawn(self, ctx: SpawnContext) -> SpawnPlan:
        """Refuse a project with its own config, resolve the binary, then mask."""
        session = ctx.session
        assert session is not None, "kimi is launched for one session"
        # The mirror reads the agent spec into this session's MCP cache, off the loop.
        await session._prepare_session_mcp()
        # Before anything is resolved: a fact about the folder, so a retry refuses the
        # same way and every surface shows the remedy.
        repo_issue = await asyncio.to_thread(
            kimi_repo_config_issue, session._spawn_work_dir or ctx.work_dir
        )
        if repo_issue:
            raise AcpToolGateUnroutable(repo_issue, transient=False, user_worded=True)
        (
            kimi_bin,
            argv,
            spawn_label,
            stderr_label,
        ) = await launch_mod.resolve_self_served_launch(self.backend)
        # Kimi 2.7.2 dies at import on Node 20, before ``initialize``: name that, not a
        # generic crash.
        node_found = await asyncio.to_thread(kimi_node_found, kimi_bin)
        if node_found:
            raise AcpToolGateUnroutable(
                kimi_node_message(node_found), transient=False, user_worded=True
            )
        # Crew's own home: refused while it has no sign-in, config rewritten last. After the checks that
        # refuse the session for a fact about the folder or the install, before the mask.
        try:
            self._home = await asyncio.to_thread(prepare_kimi_home)
        except OSError as exc:
            raise AcpToolGateUnroutable(
                f"Kiro Crew could not prepare its own Kimi Code home: {exc.strerror or exc}. "
                "Check that the Kiro Crew data folder is writable, then start a new chat.",
                transient=False,
                user_worded=True,
            ) from exc
        # The refuse-then-mask preflight every enforced host runs. ACP cannot make a
        # harness ask about a passive read, so the OS credential mask is the
        # compensating control, as for Codex and Copilot.
        hidden = await launch_mod._run_preflight_bounded(
            launch_mod._sandbox_preflight, self.backend, ctx.sandbox_mode
        )
        # The mode write proves the asking state; nothing proves the harness then asks
        # before every call. Arm the completed-unasked tripwire
        # (``AcpClient._tripwire_pi_gate``): a call that completes with no permission
        # request ends the session unless the harness's ``HARNESS_UNASKED_TOOL_IDS``
        # row names it. Kimi's row is its read-only tools that still run unasked (todo and
        # task lists, goal reads; ``Read`` asks by rule); a file write, a shell command,
        # a subagent or anything on the network trips. Captured on 2.7.2: in ``default`` a
        # Write inside a git cwd ran with NO permission request
        # (``test/fixtures/acp_frames/kimi/turn-live-write-unasked``); the Crew-owned home
        # above makes Write and Edit ask, so that capture is the case it closes.
        session._unasked_shell_tripwire = True
        return SpawnPlan(
            argv=argv,
            spawn_label=spawn_label,
            stderr_label=stderr_label,
            extra_hidden_dirs=hidden,
            extra_expose_files=acp_tool_gate.adapter_expose_files(self.backend, hidden),
            extra_private_dirs=await asyncio.to_thread(crew_home_visible_spellings, self._home),
            refuse_masked_workdir=True,
        )

    def session_mcp_servers(self, session: ProcessSession) -> list[dict[str, Any]]:
        """The Kimi session's own ``mcpServers`` entries, warmed by the spawn."""
        return session._kimi_session_mcp_servers()

    def apply_spawn_env(self, env: dict[str, str], *, spawned_binary: str | None = None) -> None:
        """Point the child at Crew's home, and keep it from reporting usage or updating itself.

        An operator's own ``KIMI_CODE_HOME`` is overwritten: a child running out of it would use
        that home's config, rules and hooks, none of which Crew wrote.
        """
        if not self._home:
            raise AcpToolGateUnroutable(
                "Kimi Code was about to start without Kiro Crew's own home, so it was refused.",
                transient=False,
                user_worded=True,
            )
        for name in _STRIPPED_ENV:
            env.pop(name, None)
        env.update(_CHILD_ENV)
        env["KIMI_CODE_HOME"] = self._home
