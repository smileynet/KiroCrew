"""Project the agent spec onto Kimi Code CLI's ACP session.

Kimi reads the ACP ``mcpServers`` array and no Kiro agent file. Its array mechanics
match goose, Copilot and Antigravity, so this mirror delegates to the same placement
function and keeps Kimi's own rulings beside it.

Driven on 2.7.2 (``reports/kimi-captures.md``, "MCP stdio"): a stdio element on
``session/new`` was started, its tool listed as ``mcp__<server>__<tool>``, and a call to
it asked ``session/request_permission`` first. Crew's config keeps no MCP of its own
(``acp.harness.kimi.render_kimi_config``), so the array is the whole MCP channel.

The tools reach the model as DEFERRED tools (spike M1, ``reports/w1-spikes-mcp.md``): the
home config sets ``[mcp] default_deferred``, the child env sets
``KIMI_CODE_EXPERIMENTAL_TOOL_SELECT=1``, and the model loads a tool through Kimi's
``select_tools``. That needs a ``kimi``-protocol provider, which the sign-in writes.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from kiro_crew.acp.session_mcp import session_mcp_projection
from kiro_crew.agent_sdk.backends import ACP_BACKEND_KIMI
from kiro_crew.providers.mirrors.base import TOOLS_SETTINGS_GATE_CHANNEL, AgentConfigMirror, Concern
from kiro_crew.providers.mirrors.base import Disposition as _D
from kiro_crew.providers.mirrors.base import Ruling, SessionProjection
from kiro_crew.providers.mirrors.opencode import place_single_binary_array

__all__ = ["KimiMirror", "kimi_projection"]


def kimi_projection(
    agent: str | None,
    *,
    stub_server_names: Collection[str] = (),
    stub_elements: Collection[Mapping[str, Any]] = (),
    work_dir: object = None,
    session_key: str = "",
    channel_id: str = "",
    session_token: str = "",
) -> SessionProjection:
    """Build the complete MCP array for one Kimi session.

    A server narrowed per tool is withheld whole. Kimi names an MCP tool ``mcp__<server>__<tool>`` but the permission request was not
    captured with a split server and tool, so this projection leaves no switched-off
    tool reachable on the strength of a title parse.

    Blocking (parses the agent spec once), so callers use it off the event loop.
    """
    projection = session_mcp_projection(
        agent,
        stub_server_names=stub_server_names,
        work_dir=work_dir,  # type: ignore[arg-type]
    )
    narrowed = frozenset(server for server, _tool in projection.disabled_tools)
    out = place_single_binary_array(
        projection,
        label="kimi",
        unhonoured=narrowed,
        stub_elements=stub_elements,
        session_key=session_key,
        channel_id=channel_id,
        session_token=session_token,
    )
    return SessionProjection(
        params={"mcpServers": out},
        disabled_servers=projection.disabled_servers,
        restricted_servers=narrowed,
        unhonoured_servers=narrowed,
        zero_tools=projection.zero_tools,
        derived_spec_snapshot=projection.derived_spec_snapshot,
        agent_spec=projection.agent_spec,
    )


class KimiMirror(AgentConfigMirror):
    """Projects the agent spec onto ``kimi acp``."""

    backend = ACP_BACKEND_KIMI

    def rulings(self) -> Mapping[Concern, Ruling]:
        return {
            Concern.MCP_SERVERS: Ruling(
                _D.DELIVERED,
                "the session/new and session/load mcpServers array, translated by the "
                "shared session-MCP projection and placed by the same function used "
                "for goose, Copilot and Antigravity. Driven on Kimi Code CLI 2.7.2: a "
                "stdio element was started, listed as mcp__<server>__<tool> and asked "
                "session/request_permission before its call. Crew's config.toml carries "
                "no MCP of its own, so the array is the whole channel",
            ),
            Concern.TOOL_ALLOWLIST: Ruling(
                _D.TRANSLATED,
                "into the shared server allowlist that decides which MCP servers enter "
                "the array; native-tool restrictions continue through Crew's permission "
                "gate and the [tools] disabled list Crew writes, not a second Kimi "
                "setting. Kimi's todo, task and cron lists, Skill and GetGoal run "
                "without a permission request, so the gate never sees them",
            ),
            Concern.DENIED_TOOLS: Ruling(
                _D.TRANSLATED,
                "by withholding the whole server named by a disabledTools entry, Crew's "
                "control plane included, and declaring per_tool_deny=whole-server. "
                "Kimi's permission request for an MCP tool was captured without a "
                "separate server and tool, so a narrower per-call refusal would rest on "
                "a title parse",
            ),
            Concern.MODEL: Ruling(
                _D.WITHHELD,
                "Kimi takes its model from default_model in the config.toml Crew keeps "
                "from the sign-in, not from this array",
            ),
            Concern.MODEL_ALLOWLIST: Ruling(
                _D.WITHHELD,
                "Kimi advertises its own model vocabulary on the session; a Crew-side "
                "list would be a second and potentially stale answer",
            ),
            Concern.AUTO_APPROVE: Ruling(
                _D.WITHHELD,
                "no per-tool auto-approval is projected. Crew never answers "
                "approve_always (it is stripped from the offered options and sent as "
                "approve_once), so a spec's autoApprove cannot widen what Kimi asks",
            ),
            Concern.PERMISSION_MODE: Ruling(
                _D.WITHHELD,
                "not projected as spec data. Crew owns the mode through the required "
                "session config write mode=default after every session/new and "
                "session/load, and refuses yolo and auto",
            ),
            Concern.PROMPT: Ruling(
                _D.WITHHELD,
                "no agent definition is written for Kimi; the session's instructions "
                "arrive in the turn's own text",
            ),
            Concern.RESOURCES: Ruling(
                _D.WITHHELD,
                "the session MCP array carries servers rather than resources, and no "
                "resource channel is projected by this adapter",
            ),
            Concern.HOOKS: Ruling(
                _D.NO_CHANNEL,
                "the ACP session array has no hooks field, and Crew writes none into the "
                "config.toml it owns (Kimi's own hooks are dropped on purpose). Crew's "
                "turn loop does not fire the spec's hooks for Kimi either, so a spec's "
                "hooks run nowhere on this harness. Crew's OWN hooks (hooks.py, fired on "
                "ACP tool events) are unaffected",
                channel="a spec-hook channel for Kimi: either Crew firing the spec's "
                "events on its turn loop (ACP_BACKENDS_CREW_FIRES_SPEC_OBSERVER_HOOKS) "
                "or a Crew-written [[hooks]] block in the home's config.toml",
            ),
            Concern.TOOLS_SETTINGS: Ruling(
                _D.NO_CHANNEL,
                "kiro-cli's per-tool settings (shell denyByDefault and allowedCommands, the "
                "write allowlist, deniedPaths, web_fetch.blocked) have no field on the ACP "
                "session/new element set, and Crew feeds this session's permission requests no "
                "spec allowlist, so the field is applied nowhere. Its subagent.availableAgents "
                "is the exception: Crew's spawn gate reads it from the parent's spec on every "
                "backend (subagent_parts/agent_policy.py), so no notice names it",
                channel=TOOLS_SETTINGS_GATE_CHANNEL,
            ),
        }

    def session_params(
        self,
        agent: str | None,
        *,
        stub_server_names: Collection[str] = (),
        work_dir: object = None,
        session_key: str = "",
        channel_id: str = "",
        session_token: str = "",
        **kwargs: object,
    ) -> dict[str, object]:
        stubs = kwargs.get("stub_elements") or ()
        return self.session_projection(
            agent,
            stub_server_names=stub_server_names,
            stub_elements=stubs if isinstance(stubs, (list, tuple)) else (),
            work_dir=work_dir,
            session_key=session_key,
            channel_id=channel_id,
            session_token=session_token,
        ).params

    def session_projection(
        self,
        agent: str | None,
        *,
        stub_server_names: Collection[str] = (),
        stub_elements: Collection[Mapping[str, Any]] = (),
        work_dir: object = None,
        session_key: str = "",
        channel_id: str = "",
        session_token: str = "",
        **kwargs: object,
    ) -> SessionProjection:
        del kwargs
        return kimi_projection(
            agent,
            stub_server_names=stub_server_names,
            stub_elements=stub_elements,
            work_dir=work_dir,
            session_key=session_key,
            channel_id=channel_id,
            session_token=session_token,
        )
