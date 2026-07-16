# Custom wake-word models

Place the trained "Fifi" openWakeWord model here as `fifi.onnx`
(configurable via `WAKE_WORD_MODEL_PATH` in `.env`).

Rules (enforced by `app/voice/wake_word.py`):

- The model is **never downloaded or replaced automatically**. If this file is
  missing, wake listening reports `unavailable` and refuses to start.
- Only this explicit file is loaded — no pretrained openWakeWord models are
  substituted.
- Train a custom model with the isolated workspace in `wake_training/`
  (Phase 3D.2), which drives the official openWakeWord pipeline and installs the
  result here for you:

  ```
  python wake_training/scripts/setup.py                       # isolated .venv
  py=wake_training/.venv/Scripts/python.exe
  $py wake_training/scripts/wake_trainer.py train fifi        # or: train oye_fifi
  $py wake_training/scripts/wake_trainer.py evaluate fifi --manifest <clips.json>
  $py wake_training/scripts/wake_trainer.py compare           # rank by evaluation
  $py wake_training/scripts/wake_trainer.py install fifi      # validate + install here
  ```

- `install` validates the ONNX input/output, atomically writes `fifi.onnx`, keeps
  a versioned backup under `backups/`, and writes **`fifi.metadata.json`**
  (training version, phrase, threshold, metrics, sha256 hash, provenance). The
  runtime reads that metadata in `wake-doctor` / `wake-status` to prove the model
  on disk is the one it describes. Replacing an installed model always requires
  `--force` and always keeps a backup — a model is never silently replaced.

- openWakeWord's shared feature-extraction models (melspectrogram/embedding)
  are a one-time, deliberate install if your openwakeword version does not
  bundle them:
  `python -c "import openwakeword.utils; openwakeword.utils.download_models()"`

Model files, `fifi.metadata.json`, and `backups/` are local artifacts and are not
committed (see `.gitignore`).
