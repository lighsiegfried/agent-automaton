"""Seed the Hugging Face cache volume from a read-only host copy — OFFLINE.

Run by the `voice-model-seed` compose service (profile `voice-seed`), which
mounts the existing host cache read-only at /seed/hf and the destination
`voice-lab-models` volume at /models/hf. The container runs with
`network_mode: none`, and this script imports NOTHING that touches the network:
the migration is a pure filesystem copy, so the ~14 GB of weights are never
re-downloaded.

Contract:
- copy the complete HF cache preserving directory structure,
- skip files that already exist with the same size (HF blobs are immutable /
  content-addressed, so a complete model is never rewritten unnecessarily),
- verify the three configured model repositories are present afterwards,
- write an ATOMIC completion marker (.seeded via os.replace),
- skip everything when already seeded.

No argparse, no imports beyond the stdlib — keep it dependency-free so it runs
in a bare python:3.12-slim with no network.
"""

import json
import os
import shutil
import sys
from pathlib import Path

SRC = Path(os.environ.get("VOICE_LAB_SEED_SRC", "/seed/hf"))
DST = Path(os.environ.get("VOICE_LAB_SEED_DST", "/models/hf"))
CONFIG = Path(os.environ.get("VOICE_LAB_MODELS_CONFIG", "/config/models.json"))


def load_models(config: Path = CONFIG) -> list[str]:
    """Model repo ids from the shared config (never hard-coded here)."""
    data = json.loads(config.read_text(encoding="utf-8"))
    return list(data["models"])


def repo_slug(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def repo_complete(root: Path, model_id: str) -> bool:
    """True when `root` holds at least one non-empty snapshot for the repo."""
    snapshots = root / "hub" / repo_slug(model_id) / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(
        child.is_dir() and any(child.iterdir()) for child in snapshots.iterdir()
    )


def copy_tree_skip_existing(src: Path, dst: Path) -> tuple[int, int]:
    """Copy src -> dst preserving structure; skip same-size files already there.

    Returns (copied, skipped) file counts. Content-addressed blobs mean a
    same-size file is the same file, so complete models are not rewritten.
    """
    copied = skipped = 0
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            source_file = Path(root) / name
            dest_file = target_dir / name
            try:
                if dest_file.exists() and dest_file.stat().st_size == source_file.stat().st_size:
                    skipped += 1
                    continue
            except OSError:
                pass
            shutil.copy2(source_file, dest_file)
            copied += 1
    return copied, skipped


def marker_path(dst: Path) -> Path:
    return dst / ".seeded"


def already_seeded(models: list[str], dst: Path = DST) -> bool:
    return marker_path(dst).exists() and all(repo_complete(dst, m) for m in models)


def write_marker(models: list[str], dst: Path = DST) -> None:
    """Atomic completion marker — a partial copy never looks seeded."""
    dst.mkdir(parents=True, exist_ok=True)
    tmp = dst / ".seeded.tmp"
    tmp.write_text(json.dumps({"models": models}), encoding="utf-8")
    os.replace(tmp, marker_path(dst))  # atomic on the same filesystem


def seed(src: Path = SRC, dst: Path = DST, config: Path = CONFIG) -> int:
    models = load_models(config)

    if already_seeded(models, dst):
        print("[seed] already seeded — nothing to do.")
        return 0

    if not src.is_dir():
        print(f"[seed] ERROR: source cache not found at {src}.")
        return 1

    print(f"[seed] copying HF cache {src} -> {dst} (offline, structure preserved)…")
    copied, skipped = copy_tree_skip_existing(src, dst)
    print(f"[seed] files copied: {copied}, skipped (already present): {skipped}")

    missing = [m for m in models if not repo_complete(dst, m)]
    if missing:
        print(f"[seed] ERROR: model repositories missing after copy: {missing}")
        return 1

    write_marker(models, dst)
    print(f"[seed] verified {len(models)} model repositories: {models}")
    print("[seed] complete — marker written atomically.")
    return 0


def main() -> int:
    return seed()


if __name__ == "__main__":
    raise SystemExit(main())
