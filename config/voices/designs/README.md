# Voice designs

Metadata for identities created in the Voice Designer
(`python scripts/local_runtime.py voice-ui` → http://127.0.0.1:8766/ui).

Each `<name>.json` records HOW a voice was made: the structured design fields,
the composed VoiceDesign instruction, the chosen variant, the exact reference
transcript, and the cloning authorization record. The matching profile lives
in `../profiles/<name>.json`.

What is and isn't committed:

- design metadata (this folder) — committed; contains no audio and no
  absolute paths (references are relative to `voice_lab/`).
- reference audio (`voice_lab/storage/voice-identities/<name>/reference.wav`)
  — generated locally, gitignored, NEVER committed. If it is missing on a new
  machine, regenerate and re-freeze the identity in the Designer.

Cloning safety: synthesis with a cloned voice is authorization-gated.
Designed voices are synthetic (`authorization.type: "synthetic"`). Cloning an
external reference requires an explicit authorization statement, or the
engine refuses to synthesize.
