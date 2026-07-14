"""Containerization: offline model seeding, missing-only download, HF cache
resolution, and the host-only SAPI switch (Phase 3D.x).

Runs in the voice_lab suite (its own `app` package). Everything is filesystem
and pure-python — no Docker, no network, no real models.
"""

import json
import sys
from pathlib import Path

import pytest

VOICE_LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VOICE_LAB_ROOT / "docker"))

import download_models  # noqa: E402
import seed_models  # noqa: E402

MODELS = [
    "hexgrad/Kokoro-82M",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]


def make_repo(hf_root: Path, model_id: str, *, size: int = 16) -> None:
    """Create a fake non-empty snapshot for a repo under <hf_root>/hub."""
    slug = "models--" + model_id.replace("/", "--")
    snap = hf_root / "hub" / slug / "snapshots" / "deadbeef"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_bytes(b"x" * size)
    blobs = hf_root / "hub" / slug / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / "deadbeef").write_bytes(b"x" * size)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": MODELS}), encoding="utf-8")
    return path


# --- the shipped config is the single source of truth -------------------------------


def test_models_json_is_the_single_source_of_truth():
    shipped = json.loads((VOICE_LAB_ROOT / "models.json").read_text(encoding="utf-8"))
    assert shipped["models"] == MODELS
    # Both services read ids from configuration, never a hard-coded list in code.
    assert callable(seed_models.load_models)
    assert callable(download_models.load_models)
    seed_src = (VOICE_LAB_ROOT / "docker" / "seed_models.py").read_text(encoding="utf-8")
    dl_src = (VOICE_LAB_ROOT / "docker" / "download_models.py").read_text(encoding="utf-8")
    for model_id in MODELS:
        assert model_id not in seed_src, f"{model_id} should live in models.json, not seed code"
        assert model_id not in dl_src, f"{model_id} should live in models.json, not download code"


# --- seed: offline, atomic, idempotent, verifying ------------------------------------


def test_seed_makes_no_network_calls():
    """Static guarantee: the seed script imports nothing that hits the network."""
    source = (VOICE_LAB_ROOT / "docker" / "seed_models.py").read_text(encoding="utf-8")
    for forbidden in ("import httpx", "import requests", "import urllib", "import socket",
                      "huggingface_hub", "snapshot_download"):
        assert forbidden not in source, f"seed must stay offline, found {forbidden!r}"


def test_seed_copies_all_three_repos_offline(tmp_path, config_file):
    src = tmp_path / "src" / "hf"
    dst = tmp_path / "vol" / "hf"
    for model in MODELS:
        make_repo(src, model)

    rc = seed_models.seed(src=src, dst=dst, config=config_file)
    assert rc == 0
    # All three repositories exist after seeding (requirement 8).
    for model in MODELS:
        assert seed_models.repo_complete(dst, model)
    # Structure preserved: hub/<slug>/snapshots/... mirrored.
    assert (dst / "hub").is_dir()


def test_seed_marker_is_atomic_and_leaves_no_temp(tmp_path, config_file):
    src = tmp_path / "src" / "hf"
    dst = tmp_path / "vol" / "hf"
    for model in MODELS:
        make_repo(src, model)
    seed_models.seed(src=src, dst=dst, config=config_file)
    assert (dst / ".seeded").exists()
    assert not (dst / ".seeded.tmp").exists()  # atomic os.replace left no temp


def test_seed_skips_when_already_seeded(tmp_path, config_file):
    src = tmp_path / "src" / "hf"
    dst = tmp_path / "vol" / "hf"
    for model in MODELS:
        make_repo(src, model)
    assert seed_models.seed(src=src, dst=dst, config=config_file) == 0
    # Remove the source entirely; a second run must still succeed via the marker
    # (proving it did NOT try to copy again).
    import shutil

    shutil.rmtree(src)
    assert seed_models.seed(src=src, dst=dst, config=config_file) == 0


def test_seed_fails_when_a_repo_is_missing(tmp_path, config_file):
    src = tmp_path / "src" / "hf"
    dst = tmp_path / "vol" / "hf"
    # Only two of the three repos present in the source.
    make_repo(src, MODELS[0])
    make_repo(src, MODELS[1])
    assert seed_models.seed(src=src, dst=dst, config=config_file) == 1
    assert not (dst / ".seeded").exists()  # never marks an incomplete seed


def test_seed_does_not_rewrite_existing_files(tmp_path, config_file):
    src = tmp_path / "src" / "hf"
    dst = tmp_path / "vol" / "hf"
    for model in MODELS:
        make_repo(src, model)
    # Pre-populate the destination with one identical-size file.
    make_repo(dst, MODELS[0])
    copied, skipped = seed_models.copy_tree_skip_existing(src, dst)
    assert skipped >= 1  # same-size files are not rewritten


# --- download: missing only, official API, no needless refetch -----------------------


def test_download_fetches_only_missing_models(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    # Two already cached, one missing.
    make_repo(tmp_path, MODELS[0])
    make_repo(tmp_path, MODELS[1])

    fetched_calls = []

    def fake_downloader(repo_id):
        fetched_calls.append(repo_id)
        make_repo(tmp_path, repo_id)  # simulate the fetch landing in cache

    got = download_models.download(MODELS, downloader=fake_downloader)
    assert got == [MODELS[2]]           # only the missing one
    assert fetched_calls == [MODELS[2]]  # official API called once


def test_download_skips_when_all_cached(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    for model in MODELS:
        make_repo(tmp_path, model)

    def fake_downloader(repo_id):
        raise AssertionError("must not re-fetch a complete cached model")

    assert download_models.download(MODELS, downloader=fake_downloader) == []
    assert download_models.missing_models(MODELS, hub=hub) == []


def test_download_reads_ids_from_config(config_file):
    assert download_models.load_models(config_file) == MODELS


# --- worker sees the configured HF cache (HF_HOME) -----------------------------------


def test_worker_sees_configured_hf_cache_via_hf_home(tmp_path, monkeypatch):
    from app import gpu

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    # Nothing there yet.
    assert gpu.model_cache_status("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign") == "not_installed"
    # Seed the volume-style cache the container would mount.
    make_repo(tmp_path / "hf", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    assert gpu.model_cache_status("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign") == "cached"
    assert gpu._hf_cache_root() == tmp_path / "hf"


# --- SAPI is host-only ---------------------------------------------------------------


def test_windows_sapi_disabled_by_flag(monkeypatch):
    from app.engines.windows_sapi import WindowsSapiEngine

    monkeypatch.setenv("VOICE_LAB_DISABLE_SAPI", "true")
    ok, reason = WindowsSapiEngine().available()
    assert ok is False
    assert "host-only" in reason
    assert ":\\" not in reason  # no absolute paths in the reason


def test_windows_sapi_disabled_off_windows(monkeypatch):
    from app.engines import windows_sapi

    monkeypatch.delenv("VOICE_LAB_DISABLE_SAPI", raising=False)
    monkeypatch.setattr(windows_sapi.sys, "platform", "linux")
    ok, reason = windows_sapi.WindowsSapiEngine().available()
    assert ok is False
    assert "host-only" in reason
