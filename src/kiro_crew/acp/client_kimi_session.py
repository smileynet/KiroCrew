"""The Kimi Code CLI handshake contract ``KimiAcpClient`` keeps.

Two commands called ``kimi`` exist: Kimi Code CLI (npm ``@moonshot-ai/kimi-code``) and
the archived Python kimi-cli, and both answer ``kimi acp``. Crew's routing was observed
on one release of the first, so the session is refused at ``initialize`` unless
``agentInfo`` names it and the verified release. The refusal is raised before any
session exists, so no tool call can run on a harness Crew did not observe.

Each function here whose first parameter is ``self`` is a ``KimiAcpClient`` method: the
class binds it by name, so it reads the session's own state on the instance, and a
patch of the method on the class or on an instance reaches every caller.

Instance fields written: none.

Instance fields read: none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kiro_crew.acp.harness.kimi import KIMI_AGENT_NAME
from kiro_crew.acp.transport_errors import AcpToolGateUnroutable
from kiro_crew.agent_sdk.backends import ACP_BACKEND_KIMI, ACP_BACKEND_LAUNCH, KIMI_VERIFIED_VERSION

if TYPE_CHECKING:
    from kiro_crew.acp.client import KimiAcpClient


# The ``AcpClient`` hook ``KimiAcpClient`` fills.


def _note_harness_adapter_version(self: KimiAcpClient) -> None:
    """Refuse a ``kimi`` that is not the verified Kimi Code CLI release, at the handshake.

    Reads ``agentInfo`` off the ``initialize`` answer: its ``name`` must be
    :data:`KIMI_AGENT_NAME` (the archived kimi-cli reports another) and its ``version``
    the pinned one. A missing name or version is refused too. Raised as
    ``AcpToolGateUnroutable``, so the start path ends the child WITHOUT the respawn a
    transport fault earns (a fresh process reports the same identity), user-worded and
    non-transient so every surface shows the install command.
    """
    info = (self._initialize_result or {}).get("agentInfo")
    name = info.get("name") if isinstance(info, dict) else None
    version = self.agent_version
    if name == KIMI_AGENT_NAME and version == KIMI_VERIFIED_VERSION:
        return
    reported = f"{name or 'no name'} {version or 'no version'}"
    raise AcpToolGateUnroutable(
        f"The kimi command on this host reported {reported!r}, not {KIMI_AGENT_NAME} "
        f"{KIMI_VERIFIED_VERSION}, so this chat was refused. Kiro Crew verifies "
        f"{KIMI_AGENT_NAME} {KIMI_VERIFIED_VERSION} only, and the archived Python "
        f"kimi-cli uses the same command name. Run "
        f"{ACP_BACKEND_LAUNCH[ACP_BACKEND_KIMI].install_command} (or set KIMI_BIN to "
        "that release's kimi), then start a new chat.",
        transient=False,
        user_worded=True,
    )
