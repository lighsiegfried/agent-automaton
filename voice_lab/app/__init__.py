"""Fifi Voice Lab (Phase 3D.0) — embedded but isolated TTS laboratory.

Lives inside the agent-automaton repo but runs in its OWN virtualenv with its
own .env, dependencies, model cache, and logs (see voice_lab/README notes in
each module). The main API never imports this package; it only talks to the
local worker over http://127.0.0.1:8766.
"""
