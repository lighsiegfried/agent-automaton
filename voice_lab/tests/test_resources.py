"""Canonical resource inspection (Phase 3D.0.4, Part A).

Pins the rules that ended the false "insufficient space" era:
- disk is not RAM, RAM is not VRAM, shared GPU memory is not dedicated VRAM;
- every comparison happens in BYTES (MiB from nvidia-smi converted exactly);
- global nvidia-smi values are used (never per-process WDDM figures);
- enough free VRAM/disk NEVER produces a rejection; genuine shortage produces
  the resource-SPECIFIC Spanish message;
- loaded models are never double-counted on top of measured GPU usage.
"""

import pytest

from app import resources

GiB = resources.GiB
MiB = resources.MiB


# --- unit conversions ---------------------------------------------------------------


def test_byte_conversions_are_exact():
    assert resources.mib_to_bytes(1) == 1024 * 1024
    assert resources.mib_to_bytes(16311) == 16311 * 1024 * 1024
    assert resources.gb_to_bytes(2.5) == int(2.5 * GiB)
    assert resources.bytes_to_gb(16 * GiB) == 16.0
    assert resources.bytes_to_gb(None) is None
    # MiB values are NEVER comparable to GB thresholds without conversion:
    # 7600 MiB is ~7.42 GiB — far above a 2.5 GB threshold once converted.
    assert resources.mib_to_bytes(7600) > resources.gb_to_bytes(2.5)
    # ...but numerically 7600 (MiB) vs 2.5 (GB) compared raw would be nonsense;
    # the module never exposes such a comparison path (bytes only).
    assert resources.mib_to_bytes(2000) < resources.gb_to_bytes(2.5)


def test_no_integer_truncation_for_fractional_gb():
    assert resources.gb_to_bytes(0.5) == 536870912
    assert resources.bytes_to_gb(536870912) == 0.5


# --- nvidia-smi parsing ----------------------------------------------------------------


def test_nvidia_smi_global_values_parse_to_bytes(monkeypatch):
    class Result:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 5060 Ti, 610.62, 16311, 8100, 8211\n"

    monkeypatch.setattr(resources.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: Result())
    vram = resources._vram_via_nvidia_smi()
    assert vram["source"] == "nvidia-smi"
    assert vram["name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert vram["driver"] == "610.62"
    assert vram["total_bytes"] == 16311 * MiB
    assert vram["used_bytes"] == 8100 * MiB
    assert vram["free_bytes"] == 8211 * MiB  # GLOBAL free — the source of truth


def test_nvidia_smi_garbage_is_safe(monkeypatch):
    class Result:
        returncode = 0
        stdout = "N/A, N/A, [N/A], nope\n"

    monkeypatch.setattr(resources.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: Result())
    assert resources._vram_via_nvidia_smi() == {}  # parse errors never raise


# --- separation of resource types ---------------------------------------------------------


@pytest.fixture
def fake_probes(monkeypatch):
    values = {
        "ram": {"total_bytes": 16 * GiB, "available_bytes": 5 * GiB, "used_bytes": 11 * GiB},
        "disk": {"total_bytes": 223 * GiB, "free_bytes": 18 * GiB, "used_bytes": 205 * GiB},
        "vram": {"total_bytes": 16 * GiB, "used_bytes": int(8.1 * GiB),
                 "free_bytes": int(7.6 * GiB), "name": "RTX 5060 Ti",
                 "driver": "610.62", "source": "test"},
    }
    monkeypatch.setattr(resources, "system_ram", lambda: dict(values["ram"]))
    monkeypatch.setattr(resources, "disk", lambda: dict(values["disk"]))
    monkeypatch.setattr(resources, "dedicated_vram", lambda: dict(values["vram"]))
    return values


def test_snapshot_separates_ram_vram_disk(fake_probes, monkeypatch):
    from app import gpu as gpu_module

    monkeypatch.setattr(gpu_module, "ollama_loaded_models", lambda: [
        {"name": "qwen2.5:7b", "vram_gb": 4.4},
    ])
    monkeypatch.setattr(gpu_module, "whisper_state", lambda: {"reachable": True, "loaded": True})
    snap = resources.snapshot()
    assert snap["system_ram_available_bytes"] == 5 * GiB
    assert snap["disk_free_bytes"] == 18 * GiB
    assert snap["dedicated_vram_free_bytes"] == int(7.6 * GiB)
    # Shared GPU memory is deliberately separate — and never mixed in.
    assert snap["shared_gpu_memory"] is None
    assert snap["gpu_name"] == "RTX 5060 Ti"
    assert snap["gpu_driver"] == "610.62"
    # NO double counting: Ollama's 4.4 GB is already inside used_bytes; the
    # snapshot reports the model list separately, not added to VRAM figures.
    assert snap["dedicated_vram_used_bytes"] == int(8.1 * GiB)
    assert snap["ollama"]["loaded"][0]["name"] == "qwen2.5:7b"


def test_enough_vram_never_blocks(fake_probes):
    """The live false-rejection scenario: 8.1 GB used / 7.6 GB free must PASS
    the 2.5 GB threshold — usage alone is never a reason to refuse."""
    result = resources.check_resources()
    assert result["checks"]["vram"]["ok"] is True
    assert result["checks"]["disk"]["ok"] is True
    assert result["checks"]["system_ram"]["ok"] is True
    assert result["blocking"] is None
    assert result["message"] == ""


def test_genuinely_low_vram_blocks_with_vram_message(fake_probes, monkeypatch):
    monkeypatch.setattr(resources, "dedicated_vram", lambda: {
        "total_bytes": 16 * GiB, "used_bytes": 15 * GiB, "free_bytes": 1 * GiB,
        "name": "X", "source": "test",
    })
    result = resources.check_resources()
    assert result["blocking"] == "vram"
    assert result["message"] == "No hay suficiente memoria VRAM disponible."


def test_genuinely_low_disk_blocks_with_storage_message(fake_probes, monkeypatch):
    monkeypatch.setattr(resources, "disk", lambda: {
        "total_bytes": 223 * GiB, "free_bytes": 1 * GiB, "used_bytes": 222 * GiB,
    })
    result = resources.check_resources()
    assert result["blocking"] == "disk"
    assert result["message"] == "No hay suficiente espacio de almacenamiento."
    assert "RAM" not in result["message"]  # storage is not memory


def test_low_ram_blocks_with_ram_message(fake_probes, monkeypatch):
    monkeypatch.setattr(resources, "system_ram", lambda: {
        "total_bytes": 16 * GiB, "available_bytes": 1 * GiB, "used_bytes": 15 * GiB,
    })
    result = resources.check_resources()
    assert result["blocking"] == "system_ram"
    assert result["message"] == "No hay suficiente memoria RAM disponible."


def test_thresholds_compare_in_bytes(fake_probes, monkeypatch):
    """2.6 GB free vs a 2.5 GB threshold: passes ONLY with byte-exact math
    (truncating to integer GB would produce a false rejection)."""
    monkeypatch.setattr(resources, "dedicated_vram", lambda: {
        "total_bytes": 16 * GiB, "used_bytes": 16 * GiB - int(2.6 * GiB),
        "free_bytes": int(2.6 * GiB), "name": "X", "source": "test",
    })
    result = resources.check_resources()
    assert result["checks"]["vram"]["ok"] is True
    assert result["checks"]["vram"]["free_bytes"] == int(2.6 * GiB)
    assert result["checks"]["vram"]["required_bytes"] == resources.gb_to_bytes(2.5)


def test_unprobeable_resource_is_permitted_not_refused(fake_probes, monkeypatch):
    monkeypatch.setattr(resources, "dedicated_vram", lambda: {})
    result = resources.check_resources()
    assert result["checks"]["vram"]["ok"] is None  # unknown, NOT a refusal
    assert result["blocking"] is None


def test_refusal_messages_are_resource_specific_never_generic(fake_probes, monkeypatch):
    """Spec: never a generic 'no hay espacio' — each shortage names ITS
    resource, and the three messages are all distinct."""
    messages = {}
    low = {"total_bytes": 16 * GiB, "available_bytes": 1 * GiB, "used_bytes": 15 * GiB,
           "free_bytes": 1 * GiB, "name": "X", "source": "test"}
    for resource_name, probe in (
        ("system_ram", "system_ram"), ("vram", "dedicated_vram"), ("disk", "disk"),
    ):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(resources, probe, lambda low=low: dict(low))
            result = resources.check_resources()
        messages[resource_name] = result["message"]
        assert result["blocking"] == resource_name
        assert result["message"] != "No hay espacio."
        assert result["message"] != "no hay espacio"
    assert len(set(messages.values())) == 3  # RAM, VRAM, and disk all differ
    assert "RAM" in messages["system_ram"]
    assert "VRAM" in messages["vram"]
    assert "almacenamiento" in messages["disk"]


def test_disk_resolves_voice_lab_volume(monkeypatch, tmp_path):
    calls = []

    def fake_usage(path):
        calls.append(path)

        class Usage:
            total, free = 100 * GiB, 42 * GiB

        return Usage()

    monkeypatch.setattr(resources.shutil, "disk_usage", fake_usage)
    result = resources.disk()
    assert result["free_bytes"] == 42 * GiB
    # Measured against the models/storage location, not the CWD.
    assert str(calls[0]).endswith("models") or "voice_lab" in str(calls[0]).replace("\\", "/")
