"""Kimi under a masked path: the hidden working folder and the relocated data home.

Both cases are planned, not spawned: the confinement plan and the Seatbelt profile are pure
functions of the spawn's arguments, so they run on any OS without a sandbox backend.
"""

from __future__ import annotations

import os
import re

import pytest

from kiro_crew import sandbox, sandbox_plan
from kiro_crew.acp import launch as launch_mod
from kiro_crew.acp.harness import kimi as kimi_mod
from kiro_crew.acp.transport_errors import AcpToolGateUnroutable
from kiro_crew.agent_sdk import tool_gate
from kiro_crew.agent_sdk.backends import ACP_BACKEND_KIMI
from kiro_crew.sandbox_seatbelt import render_seatbelt_profile

BACKENDS = [sandbox_plan.BACKEND_NAMESPACE, sandbox_plan.BACKEND_SEATBELT]
DETECTED = {
    sandbox_plan.BACKEND_NAMESPACE: "namespace",
    sandbox_plan.BACKEND_SEATBELT: "sandbox-exec",
}


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A HOME with a data home relocated under the masked ``<home>/.kiro/crew/scratch``."""
    home = tmp_path.resolve() / "home"
    scratch = home / ".kiro" / "crew" / "scratch" / "abc"
    data = scratch / "crewhome"
    (data / "kimi-home" / "credentials").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KIROCREW_HOME", str(data))
    monkeypatch.delenv("KIROCREW_POD", raising=False)
    return home, scratch, data


def _issue(monkeypatch, backend, workspace, windows=()):
    monkeypatch.setattr(sandbox, "detect_backend", lambda *_a, **_k: DETECTED[backend])
    hidden = tool_gate.adapter_hidden_credential_dirs(ACP_BACKEND_KIMI)
    return launch_mod.workspace_mask_issue(str(workspace), "standard", hidden, tuple(windows), ())


@pytest.mark.parametrize("backend", BACKENDS)
def test_cwd_under_masked_path_is_refused_and_names_the_mask(layout, monkeypatch, backend):
    home, scratch, _data = layout
    cwd = scratch / "elsewhere"
    cwd.mkdir()
    masked_by = _issue(monkeypatch, backend, cwd)
    assert masked_by.endswith(os.path.join(".kiro", "crew", "scratch"))
    message = launch_mod.hidden_workdir_message(str(cwd), masked_by)
    assert str(cwd) in message and masked_by in message


@pytest.mark.parametrize("backend", BACKENDS)
def test_cwd_in_the_spawns_own_window_is_allowed(layout, monkeypatch, backend):
    _home, scratch, _data = layout
    window = scratch / "own"
    window.mkdir()
    assert _issue(monkeypatch, backend, window / "work", windows=(str(window),)) == ""


@pytest.mark.parametrize("backend", BACKENDS)
def test_cwd_outside_every_mask_is_allowed(layout, monkeypatch, backend):
    home, _scratch, _data = layout
    for cwd in (home / "project", home / ".kiro" / "crew" / "workspace"):
        cwd.mkdir(parents=True, exist_ok=True)
        assert _issue(monkeypatch, backend, cwd) == "", cwd


@pytest.mark.parametrize("backend", BACKENDS)
def test_relocated_kimi_home_under_masked_parent_needs_its_window(layout, monkeypatch, backend):
    _home, _scratch, data = layout
    kimi_home = kimi_mod.kimi_home_dir()
    assert kimi_home == data.resolve() / "kimi-home"
    assert _issue(monkeypatch, backend, kimi_home) != ""
    window = kimi_mod.crew_home_visible_spellings(str(kimi_home))
    assert _issue(monkeypatch, backend, kimi_home, windows=window) == ""
    # Everything else under the masked parent, the data home's own secrets included, stays hidden.
    assert _issue(monkeypatch, backend, data / "run", windows=window) != ""


def test_seatbelt_profile_carves_only_the_relocated_kimi_home(layout, monkeypatch):
    kimi_home = str(kimi_mod.kimi_home_dir())
    hidden = tool_gate.adapter_hidden_credential_dirs(ACP_BACKEND_KIMI)
    plan = sandbox._spawn_plan(
        sandbox_plan.BACKEND_SEATBELT,
        "standard",
        extra_hidden_dirs=hidden,
        extra_private_dirs=kimi_mod.crew_home_visible_spellings(kimi_home),
    )
    profile = render_seatbelt_profile(plan)
    carve = f'(require-not (subpath "{kimi_home}"))'
    assert carve in profile
    assert re.search(r'\(deny file-read\* \(require-all \(subpath "[^"]*/scratch"\) ', profile)


def test_signed_out_refusal_says_unreadable_when_the_gateway_sees_a_sign_in(tmp_path):
    empty = tmp_path / "empty"
    (empty / "credentials").mkdir(parents=True)
    assert "not signed in for Kiro Crew" in kimi_mod.kimi_signed_out_refusal(empty)
    signed = tmp_path / "signed"
    (signed / "credentials").mkdir(parents=True)
    (signed / "credentials" / "kimi.json").write_text("{}")
    refusal = kimi_mod.kimi_signed_out_refusal(signed)
    assert "cannot read" in refusal and str(signed) in refusal
    assert "kimi login" not in refusal


def test_unreadable_kimi_home_is_not_reported_as_signed_out(tmp_path):
    home = tmp_path / "locked"
    (home / "credentials").mkdir(parents=True)
    os.chmod(home / "credentials", 0)
    try:
        if os.access(home / "credentials", os.R_OK):
            pytest.skip("running as a user that ignores directory modes")
        with pytest.raises(PermissionError):
            kimi_mod.kimi_signed_in(home)
        refusal = kimi_mod.kimi_signed_out_refusal(home)
        assert "cannot read" in refusal and "not signed in" not in refusal
    finally:
        os.chmod(home / "credentials", 0o700)


def test_hidden_cwd_stderr_pattern():
    from kiro_crew.acp.client_transport import _HIDDEN_CWD_STDERR

    assert _HIDDEN_CWD_STDERR.search("Error: EPERM: process.cwd failed with error uv_cwd")
    assert not _HIDDEN_CWD_STDERR.search("EPERM: operation not permitted, open 'x'")
    assert issubclass(AcpToolGateUnroutable, Exception)
