"""CLI tests — worker HTTP mocked; select works fully offline."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from conftest import VOICE_LAB_ROOT


@pytest.fixture
def cli():
    """Load voice_lab/scripts/voice_lab.py under a unique module name (the
    voice_lab/ directory itself would shadow a plain `import voice_lab`)."""
    path = VOICE_LAB_ROOT / "scripts" / "voice_lab.py"
    spec = importlib.util.spec_from_file_location("voice_lab_cli", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["voice_lab_cli"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("voice_lab_cli", None)


def test_select_is_offline_and_atomic(cli, manager, monkeypatch, capsys):
    """select never needs the worker — it writes active.json directly."""
    monkeypatch.setattr(cli, "ProfileManager", lambda: manager)
    monkeypatch.setattr(
        cli.httpx, "get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("select must not call the worker")),
    )
    monkeypatch.setattr(
        cli.httpx, "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("select must not call the worker")),
    )
    assert cli.main(["select", "fifi_calm"]) == 0
    out = capsys.readouterr().out
    assert "fifi_calm" in out and "atomically" in out
    on_disk = json.loads(manager.active_path.read_text(encoding="utf-8"))
    assert on_disk["profile"] == "fifi_calm"


def test_select_unknown_profile(cli, manager, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ProfileManager", lambda: manager)
    assert cli.main(["select", "ghost"]) == 1
    assert "Unknown profile" in capsys.readouterr().out


def test_list_marks_active(cli, manager, monkeypatch, capsys):
    manager.select("fifi_warm")
    monkeypatch.setattr(cli, "ProfileManager", lambda: manager)
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "fifi_warm" in out and "*ACTIVE*" in out
    assert "fifi_calm" in out


def test_status_with_worker_down(cli, manager, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ProfileManager", lambda: manager)
    monkeypatch.setattr(cli, "_get", lambda path: None)
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "not running" in out
    assert "(none selected)" in out


def test_preview_requires_worker(cli, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_get", lambda path: None)
    assert cli.main(["preview", "fifi_warm"]) == 1
    assert "voice-start" in capsys.readouterr().out


def test_compare_previews_every_profile(cli, manager, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ProfileManager", lambda: manager)
    monkeypatch.setattr(cli, "_get", lambda path: {"status": "ok"})
    requested = []

    def fake_post(path, payload):
        requested.append(payload["profile"])
        return {"status": "ok", "engine": "kokoro", "duration_seconds": 1.0, "file": "x.wav"}

    monkeypatch.setattr(cli, "_post", fake_post)
    assert cli.main(["compare"]) == 0
    assert requested == ["fifi_calm", "fifi_warm"]


def test_new_subcommands_exist(cli):
    parser = cli.build_parser()
    assert parser.parse_args(["speakers"]).func is cli.cmd_speakers
    assert parser.parse_args(["design", "fifi_aurora"]).func is cli.cmd_design
    assert parser.parse_args(["compare-speakers"]).func is cli.cmd_compare_speakers
    assert parser.parse_args(["compare-identities"]).func is cli.cmd_compare_identities


def test_speakers_distinguishes_variants_from_base_voices(cli, monkeypatch, capsys):
    from app.profiles.manager import ProfileManager as RealManager

    monkeypatch.setattr(cli, "ProfileManager", RealManager)  # real repo profiles
    assert cli.main(["speakers"]) == 0
    out = capsys.readouterr().out
    assert "ef_dora" in out and "em_alex" in out and "em_santa" in out
    assert "VARIANT" in out  # explains variants vs voices
    assert "fifi_aurora" in out and "designed" in out
    # Style variants appear as profiles under a voice, not as catalog voices.
    dora_line = next(line for line in out.splitlines() if line.strip().startswith("ef_dora"))
    assert "fifi_es_calm" not in dora_line


def test_design_shows_the_instruction(cli, monkeypatch, capsys):
    from app.profiles.manager import ProfileManager as RealManager

    monkeypatch.setattr(cli, "ProfileManager", RealManager)
    monkeypatch.setattr(cli, "_need_worker", lambda: True)
    monkeypatch.setattr(cli, "_preview_one", lambda name: True)
    assert cli.main(["design", "fifi_serena"]) == 0
    out = capsys.readouterr().out
    assert "Design          :" in out and "serena" in out.lower()


def test_design_rejects_non_designed_profiles(cli, monkeypatch, capsys):
    from app.profiles.manager import ProfileManager as RealManager

    monkeypatch.setattr(cli, "ProfileManager", RealManager)
    assert cli.main(["design", "fifi_es_warm"]) == 1
    assert "not a designed identity" in capsys.readouterr().out


def test_compare_identities_previews_all_four(cli, monkeypatch):
    from app.profiles.manager import ProfileManager as RealManager

    monkeypatch.setattr(cli, "ProfileManager", RealManager)
    monkeypatch.setattr(cli, "_need_worker", lambda: True)
    previewed = []
    monkeypatch.setattr(cli, "_preview_one", lambda name: previewed.append(name) or True)
    assert cli.main(["compare-identities"]) == 0
    assert previewed == ["fifi_aurora", "fifi_ejecutiva", "fifi_nova", "fifi_serena"]


def test_compare_speakers_one_per_base_voice(cli, monkeypatch):
    from app.profiles.manager import ProfileManager as RealManager

    monkeypatch.setattr(cli, "ProfileManager", RealManager)
    monkeypatch.setattr(cli, "_need_worker", lambda: True)
    previewed = []
    monkeypatch.setattr(cli, "_preview_one", lambda name: previewed.append(name) or True)
    assert cli.main(["compare-speakers", "--language", "es"]) == 0
    assert sorted(previewed) == ["fifi_es_alex", "fifi_es_santa", "fifi_es_warm"]


def test_unload_via_worker(cli, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_get", lambda path: {"status": "ok"})
    monkeypatch.setattr(cli, "_post", lambda path, payload: {"status": "ok", "unloaded": ["kokoro"]})
    assert cli.main(["unload"]) == 0
    assert "kokoro" in capsys.readouterr().out
