"""Download ONLY the missing configured models into the cache volume.

Run by the `voice-model-download` compose service (profile `voice-download`),
which mounts the `voice-lab-models` volume at /models/hf. Unlike the seed
service, this one is allowed network access — it is the escape hatch for a
fresh machine that has no local cache to migrate.

Contract:
- model identifiers come from the shared config (voice_lab/models.json), never
  duplicated in code,
- download ONLY models that are not already complete in the cache,
- use the official Hugging Face cache API (huggingface_hub.snapshot_download),
  which itself resumes/skips files already present — a complete cached model is
  never re-fetched unnecessarily,
- honor HF_HOME / HF_HUB_CACHE so everything lands in the mounted volume.
"""

import json
import os
from pathlib import Path


def cache_root() -> Path:
    """The HF hub cache dir, from HF_HUB_CACHE or HF_HOME/hub."""
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("HF_HOME", "/models/hf")) / "hub"


def load_models(config: Path | None = None) -> list[str]:
    path = config or Path(os.environ.get("VOICE_LAB_MODELS_CONFIG", "/config/models.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["models"])


def repo_slug(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def is_cached(model_id: str, hub: Path | None = None) -> bool:
    """True when the repo already has a non-empty snapshot in the cache."""
    root = hub or cache_root()
    snapshots = root / repo_slug(model_id) / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(
        child.is_dir() and any(child.iterdir()) for child in snapshots.iterdir()
    )


def missing_models(models: list[str], hub: Path | None = None) -> list[str]:
    return [m for m in models if not is_cached(m, hub)]


def download(models: list[str], downloader=None) -> list[str]:
    """Fetch each missing model via snapshot_download. Returns what was fetched."""
    if downloader is None:
        from huggingface_hub import snapshot_download as downloader  # official cache API

    fetched: list[str] = []
    for model_id in models:
        if is_cached(model_id):
            print(f"[download] {model_id}: already cached — skipping (no re-fetch).")
            continue
        print(f"[download] {model_id}: fetching via official HF cache API…")
        downloader(repo_id=model_id)
        fetched.append(model_id)
    return fetched


def main() -> int:
    models = load_models()
    pending = missing_models(models)
    if not pending:
        print(f"[download] all {len(models)} configured models already cached — nothing to do.")
        return 0
    print(f"[download] missing models to fetch: {pending}")
    fetched = download(models)
    print(f"[download] done — fetched: {fetched or 'nothing (all cached)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
