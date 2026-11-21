"""Moonshot's Kimi Code CLI, launched one process per session by ``AcpClient``.

Kimi Code CLI serves ACP from its own binary (``kimi acp``), so its executable comes
off the shared self-served ladder (``ACP_BACKEND_LAUNCH``) and there is no adapter
package. DORMANT: the id can be spelled and launched, but it is not selectable
(``NOT_SHIPPED_SELECTABLE``; the Kimi worked example in harness-onboarding.md lists
each decision still waiting on a live capture). Four things are its own:

* **Its permission route is a session config option**, ``mode``, which Crew writes
  ``default`` after ``session/new`` and ``session/load`` and before the first prompt
  (``Routing.SESSION_CONFIG``). ``yolo`` and ``auto`` approve tools on their own and
  are refused mid-session (``ACP_BACKEND_FORBIDDEN_MODES``). Crew never calls
  ``session/set_mode``.
* **The name ``kimi`` is shared with the archived Python ``kimi-cli``**, which also
  answers ``kimi acp``. The handshake therefore checks ``agentInfo.name`` and the
  verified release (``acp.client_kimi_session``), and the install probe reads the
  release off the npm manifest.
* **The child never reports usage to Moonshot**: ``KIMI_DISABLE_TELEMETRY=1``.
* **A project's own ``.kimi-code/`` directory is refused.** It can hold a local
  config, ``mcp.json``, hooks and permission ``allow`` rules, none of which Crew
  approved (:func:`kimi_repo_config_issue`).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from kiro_crew import acp_tool_gate
from kiro_crew.acp import launch as launch_mod
from kiro_crew.acp.harness.base import ProcessAdapter, SpawnContext, SpawnPlan
from kiro_crew.acp.harness.copilot import _ancestors as _project_ancestors
from kiro_crew.acp.transport_errors import AcpToolGateUnroutable
from kiro_crew.agent_sdk.backends import ACP_BACKEND_KIMI

__all__ = [
    "KIMI_AGENT_NAME",
    "KIMI_NPM_PACKAGE",
    "KIMI_PROJECT_DIR",
    "KimiLaunch",
    "kimi_installed_version",
    "kimi_repo_config_issue",
]

#: ``agentInfo.name`` Kimi Code CLI reports on ``initialize`` (observed on 2.7.2).
#: The archived Python kimi-cli reports another name, which is how the two are told
#: apart: both install a command called ``kimi`` and both serve ``kimi acp``.
KIMI_AGENT_NAME = "Kimi Code CLI"

#: The npm package Kimi Code CLI ships as, whose manifest names the installed release.
KIMI_NPM_PACKAGE = "@moonshot-ai/kimi-code"

#: The project directory Kimi Code reads local configuration from.
KIMI_PROJECT_DIR = ".kimi-code"

#: How many directories above the resolved binary a manifest is looked for in. The
#: npm bin link resolves to ``<package>/dist/main.mjs``, so two levels reach the
#: package root; the margin covers a loader moved one level.
_MANIFEST_SEARCH_DEPTH = 3

#: Set in the child so the session reports nothing to the vendor (``KIMI_DISABLE_TELEMETRY``
#: is the documented switch).
_CHILD_ENV = {"KIMI_DISABLE_TELEMETRY": "1"}


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
    for directory in [here, *here.parents][:_MANIFEST_SEARCH_DEPTH]:
        try:
            data = json.loads((directory / "package.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("name") == KIMI_NPM_PACKAGE:
            version = data.get("version")
            return version.strip() if isinstance(version, str) else ""
    return ""


def kimi_repo_config_issue(work_dir: object) -> str:
    """Why a Kimi Code session must not start in *work_dir*, or ``""``. Blocking.

    A project ``.kimi-code`` can hold a local config, ``mcp.json``, ``[[hooks]]`` and
    ``[[permission.rules]]`` allow rules; any of them could run or pre-approve a tool
    without Kiro Crew's approval. Refused on presence, in the working directory and
    every parent up to its git root, along the path as given AND as resolved, with
    Copilot's walk. A link or an unreadable entry counts as present (``lexists``).
    TODO(capture): whether Kimi's workspace-trust prompt, which a headless ACP session
    cannot answer, already stops these files; until measured, refuse.
    """
    if not work_dir:
        return ""
    given = os.path.abspath(str(work_dir))
    seen: set[str] = set()
    for directory in [*_project_ancestors(given), *_project_ancestors(os.path.realpath(given))]:
        if directory in seen:
            continue
        seen.add(directory)
        found = os.path.join(directory, KIMI_PROJECT_DIR)
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

    It adds nothing to the session's MCP array (the ``ProcessAdapter`` default): Crew's
    MCP projection for Kimi is a Stage 7 decision not taken yet.
    """

    backend = ACP_BACKEND_KIMI

    async def resolve_spawn(self, ctx: SpawnContext) -> SpawnPlan:
        """Refuse a project with its own config, resolve the binary, then mask."""
        session = ctx.session
        assert session is not None, "kimi is launched for one session"
        # Before anything is resolved: a fact about the folder, so a retry refuses the
        # same way and every surface shows the remedy.
        repo_issue = await asyncio.to_thread(
            kimi_repo_config_issue, session._spawn_work_dir or ctx.work_dir
        )
        if repo_issue:
            raise AcpToolGateUnroutable(repo_issue, transient=False, user_worded=True)
        (
            _bin,
            argv,
            spawn_label,
            stderr_label,
        ) = await launch_mod.resolve_self_served_launch(self.backend)
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
        # row names it. Kimi has no row yet, so every unasked completion trips.
        # TODO(capture): the docs say read-only calls run unasked in ``default``; add
        # the row (and ``ACP_BACKENDS_UNASKED_NATIVE_TOOLS`` membership) from a capture.
        session._unasked_shell_tripwire = True
        return SpawnPlan(
            argv=argv,
            spawn_label=spawn_label,
            stderr_label=stderr_label,
            extra_hidden_dirs=hidden,
            extra_expose_files=acp_tool_gate.adapter_expose_files(self.backend, hidden),
        )

    def apply_spawn_env(self, env: dict[str, str], *, spawned_binary: str | None = None) -> None:
        """Keep the child from reporting usage to the vendor."""
        env.update(_CHILD_ENV)
