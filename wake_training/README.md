# Fifi wake-word training workspace (Phase 3D.2)

An **isolated** workspace that trains, evaluates, calibrates, and installs the
custom local wake-word model the runtime needs at `models/wake_words/fifi.onnx`.
Like `voice_lab/`, it has its **own virtualenv** (`wake_training/.venv`) — the
main project `.venv` is never touched. Models, datasets, generated audio, and
caches are all local and gitignored.

Wake detection runs entirely on the Windows host. Nothing here downloads or
replaces the installed model behind your back: `install` is explicit, validates
ONNX compatibility, backs up the previous model, and refuses to overwrite the
active model unless validation passes.

## Two candidates

| Candidate  | Phrase        | Config                |
|------------|---------------|-----------------------|
| `fifi`     | "Fifi"        | `config/fifi.yaml`    |
| `oye_fifi` | "Oye Fifi"    | `config/oye_fifi.yaml`|

Both use the official openWakeWord training pipeline and export an ONNX artifact
that the existing `app/voice/wake_word.py` loads unchanged
(`Model(wakeword_models=[...], inference_framework="onnx")`).

## One-time setup (isolated)

```bash
python wake_training/scripts/setup.py            # creates wake_training/.venv + installs deps
python wake_training/scripts/setup.py --download-data   # also fetch negatives/RIRs/piper voices
```

`setup.py` installs the CUDA build of torch (cu128 for RTX 50 / Blackwell) when a
GPU is present, clones `piper-sample-generator` (Spanish voices) into the
workspace, and downloads its generator model. Training deps never enter the main
`.venv`.

## CLI

```bash
py=wake_training/.venv/Scripts/python.exe            # the ISOLATED interpreter
$py wake_training/scripts/wake_trainer.py status            # workspace + candidates + install
$py wake_training/scripts/wake_trainer.py train fifi        # train the "fifi" candidate
$py wake_training/scripts/wake_trainer.py train oye_fifi    # train the "oye fifi" candidate
$py wake_training/scripts/wake_trainer.py evaluate fifi     # metrics + threshold sweep
$py wake_training/scripts/wake_trainer.py compare           # rank candidates by evaluation
$py wake_training/scripts/wake_trainer.py calibrate         # live scores + threshold recommend
$py wake_training/scripts/wake_trainer.py install fifi      # validate + atomically install
```

Optional authorized real-user recordings (kept local, gitignored — training works
without them, they just improve robustness and enable the optional verifier):

```bash
$py wake_training/scripts/record_samples.py     # record at several distances/volumes/rooms
```

## What each stage produces

- **train** → `outputs/<candidate>/<candidate>.onnx` + a training report in
  `reports/<candidate>-train.json`.
- **evaluate** → `reports/<candidate>-eval.json`: recall, false-rejection rate,
  false-positives/hour, precision, a threshold sweep, inference latency, CPU use,
  model size, and a breakdown by quiet / noisy / far-field / music / conversation
  / keyboard conditions. **A model is never selected from training accuracy
  alone** — `compare` ranks on these evaluation metrics.
- **calibrate** → recommends a threshold from live observations or an eval sweep,
  recording only score/timestamp and optional labels — never idle audio.
- **install** → validates the ONNX I/O, atomically copies the selected model to
  `models/wake_words/fifi.onnx`, keeps a versioned backup under
  `models/wake_words/backups/`, and writes `models/wake_words/fifi.metadata.json`
  (training version, phrase, threshold, metrics, sha256 hash, provenance).

## Runtime integration

Once installed, the runtime picks it up with no code changes:

```bash
python scripts/local_runtime.py wake-doctor      # validates model + metadata
python scripts/local_runtime.py wake-status       # phrase, model version, threshold, hash
python scripts/local_runtime.py wake-calibrate    # live calibration (never executes commands)
python scripts/local_runtime.py wake-start        # hands-free wake mode
```

Push-to-talk remains available as a fallback either way.

## Datasets layout

```
datasets/
  positive/           # synthetic positives (generated) + optional user/ recordings
  negative/           # speech/ noise/ music/ rir/ — general + hard negatives
  validation/         # held-out positives + negative feature files for FP/hr
```

Provide negatives by pointing `config/*.yaml` `negatives.*` at local 16 kHz mono
audio folders, or run `setup.py --download-data`. See `config/defaults.yaml` for
every knob (seed, counts, augmentation ranges, hard-negative phrases, targets).

Python (not PowerShell) on purpose: this machine's AllSigned Group Policy blocks
unsigned `.ps1` files.
