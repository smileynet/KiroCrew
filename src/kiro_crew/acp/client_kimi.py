"""``class KimiAcpClient``: the per-session client for Kimi Code CLI.

``AcpClient`` is the client every per-session backend shares. What only a Kimi Code
session does lives here instead: refusing, at the handshake, a ``kimi`` that is the
archived Python kimi-cli or a release other than the verified one. ``AcpClient``
reaches it through a hook it declares with the behaviour every other backend keeps,
and this class fills the hook. Building ``AcpClient`` for the kimi backend builds this
class (``AcpClient.__new__``), so no caller names it.

The member is bound from :mod:`kiro_crew.acp.client_kimi_session` the way ``AcpClient``
binds its own owners, and the class keeps the facade as its module for reprs, pickles
and type hints.
"""

from __future__ import annotations

from kiro_crew.acp import client_kimi_session as _client_kimi_session
from kiro_crew.acp.client_class import AcpClient
from kiro_crew.acp.types import ACP_BACKEND_KIMI


class KimiAcpClient(AcpClient, serves=ACP_BACKEND_KIMI):
    """The ``AcpClient`` of a Kimi Code CLI session."""

    # The facade is the class's import path, so pickles and reprs keep naming it.
    __module__ = "kiro_crew.acp.client"

    # The ``AcpClient`` hooks this class fills, bound from the Kimi Code owner.
    _note_harness_adapter_version = _client_kimi_session._note_harness_adapter_version
    _refuse_harness_default_model = _client_kimi_session._refuse_harness_default_model
