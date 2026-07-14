# Fifi Voice Lab worker — ISOLATED Linux image (Phase 3D.x).
#
# This is NOT the root Dockerfile (that one is the simulated main API and stays
# untouched). This image mirrors voice_lab/scripts/setup.py exactly:
#   1. the cu128 PyTorch wheel FIRST (the RTX 50xx / Blackwell sm_120 index),
#   2. then voice_lab/requirements.txt,
#   3. then the optional Qwen3-TTS deps (requirements-qwen.txt) so VoiceDesign
#      and the 0.6B clone base run in the container.
#
# Model weights NEVER enter this image: the 14 GB Hugging Face cache is kept out
# of the build context by .dockerignore and reaches the container only through
# the mounted `voice-lab-models` volume (seeded by the voice-model-seed service).
#
# Build context is the repository root (see compose.yml `voice-lab.build`), so
# COPY paths are repo-relative.
FROM python:3.12-slim

# System packages proven necessary by the runtime deps:
#   libsndfile1   — soundfile (WAV validation / IO)
#   libportaudio2 — sounddevice import path (playback is disabled in-container)
#   espeak-ng     — phonemizer-fork backend for non-English G2P
#   sox, ffmpeg   — torchaudio / qwen-tts audio backends (resampling, IO)
# espeakng-loader still bundles its own library; espeak-ng here is belt-and-braces.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libportaudio2 \
        espeak-ng \
        sox \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/agent-automaton/voice_lab

# cu128 torch + torchaudio TOGETHER — the exact index setup.py uses (Blackwell /
# sm_120). Installing them from the SAME index guarantees one consistent CUDA
# build; a PyPI-default torchaudio links libcudart.so.13 (CUDA 13) and cannot
# load against the cu128 (CUDA 12.8) torch. The wheels bundle the CUDA runtime,
# so no CUDA base image is needed — the GPU reaches the container through the
# NVIDIA runtime (compose `deploy` + NVIDIA_* env), identical to Ollama.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Dependencies next (their own layer so torch caches independently of app churn).
COPY voice_lab/requirements.txt voice_lab/requirements-qwen.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-qwen.txt \
    # qwen-tts may pull a PyPI torchaudio built for a different CUDA; re-pin it to
    # the cu128 build so torchaudio's native lib always matches this torch.
    && pip install --no-cache-dir --force-reinstall --no-deps torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Worker source only — no models, no venv, no storage (all excluded / mounted).
COPY voice_lab/app ./app
COPY voice_lab/run_worker.py ./run_worker.py

# Isolation + cache pinning. HF_HOME / HF_HUB_CACHE point at the mounted volume
# so every model load reads the seeded cache and downloads nothing. Playback and
# Windows SAPI are host-only: the container must never touch audio devices, and
# the Windows fallback lives in the main HOST API, not here.
ENV HF_HOME=/models/hf \
    HF_HUB_CACHE=/models/hf/hub \
    HUGGINGFACE_HUB_CACHE=/models/hf/hub \
    VOICE_LAB_HOST=0.0.0.0 \
    VOICE_LAB_PORT=8766 \
    VOICE_LAB_PLAYBACK_ENABLED=false \
    VOICE_LAB_DISABLE_SAPI=true \
    PIP_NO_CACHE_DIR=1

EXPOSE 8766

# uvicorn on 0.0.0.0:8766 (compose binds it to 127.0.0.1 on the host only).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8766"]
