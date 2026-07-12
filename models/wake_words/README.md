# Custom wake-word models

Place the trained "Fifi" openWakeWord model here as `fifi.onnx`
(configurable via `WAKE_WORD_MODEL_PATH` in `.env`).

Rules (enforced by `app/voice/wake_word.py`):

- The model is **never downloaded or replaced automatically**. If this file is
  missing, wake listening reports `unavailable` and refuses to start.
- Only this explicit file is loaded — no pretrained openWakeWord models are
  substituted.
- Train a custom model with openWakeWord's training notebook / tools:
  https://github.com/dscripka/openWakeWord (target phrase: "fifi").
- openWakeWord's shared feature-extraction models (melspectrogram/embedding)
  are a one-time, deliberate install if your openwakeword version does not
  bundle them:
  `python -c "import openwakeword.utils; openwakeword.utils.download_models()"`

Model files are local artifacts and are not committed (see `.gitignore`).
