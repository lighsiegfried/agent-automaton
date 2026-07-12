# Graph Report - .  (2026-07-10)

## Corpus Check
- Corpus is ~18,703 words - fits in a single context window. You may not need a graph.

## Summary
- 415 nodes · 877 edges · 26 communities (23 shown, 3 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.57)
- Token cost: 45,000 input · 4,048 output

## Community Hubs (Navigation)
- [[_COMMUNITY_App Core & Voice Services|App Core & Voice Services]]
- [[_COMMUNITY_LLM Planning & Safety|LLM Planning & Safety]]
- [[_COMMUNITY_Voice & Response Tests|Voice & Response Tests]]
- [[_COMMUNITY_Router & Response Generation|Router & Response Generation]]
- [[_COMMUNITY_Docs & Config|Docs & Config]]
- [[_COMMUNITY_Docker LLM Orchestration|Docker LLM Orchestration]]
- [[_COMMUNITY_Command Handling & Planner Tests|Command Handling & Planner Tests]]
- [[_COMMUNITY_Windows Tool Automation|Windows Tool Automation]]
- [[_COMMUNITY_LLM Smoke Script|LLM Smoke Script]]
- [[_COMMUNITY_Docker Compose Tests|Docker Compose Tests]]
- [[_COMMUNITY_Voice Command Client|Voice Command Client]]
- [[_COMMUNITY_Test Fixtures (Launch Recorder)|Test Fixtures (Launch Recorder)]]
- [[_COMMUNITY_Fifi Persona|Fifi Persona]]
- [[_COMMUNITY_Modular Architecture (doc)|Modular Architecture (doc)]]
- [[_COMMUNITY_Wake Word Concept (doc)|Wake Word Concept (doc)]]

## God Nodes (most connected - your core abstractions)
1. `get_settings()` - 43 edges
2. `CommandRequest` - 39 edges
3. `handle_command()` - 36 edges
4. `SafetyLevel` - 20 edges
5. `CommandResponse` - 19 edges
6. `Intent` - 16 edges
7. `ExecutionStatus` - 16 edges
8. `upload()` - 16 edges
9. `real_windows_tools_enabled()` - 13 edges
10. `evaluate()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `test_default_agent_name_is_fifi()` --calls--> `get_settings()`  [EXTRACTED]
  tests/test_identity.py → app/config.py
- `test_project_folder_and_naming_unchanged()` --calls--> `get_settings()`  [EXTRACTED]
  tests/test_identity.py → app/config.py
- `test_docker_mode_never_reports_real_tools()` --calls--> `real_windows_tools_enabled()`  [EXTRACTED]
  tests/test_docker.py → app/config.py
- `FakeInfo` --uses--> `ExecutionStatus`  [INFERRED]
  tests/test_voice.py → app/schemas/commands.py
- `FakeSegment` --uses--> `ExecutionStatus`  [INFERRED]
  tests/test_voice.py → app/schemas/commands.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Command execution pipeline (router to safety to tools to memory)** — docs_architecture_router, docs_architecture_safety_layer, docs_architecture_tool_registry, docs_architecture_memory [EXTRACTED 1.00]
- **Voice pipeline (STT to shared entry point to TTS)** — docs_architecture_voice_api, docs_architecture_stt_service, docs_architecture_handle_command, docs_architecture_tts_service [EXTRACTED 1.00]
- **Safety-related design principles** — docs_architecture_safety_by_construction, docs_architecture_simulate_first, docs_safety_rules_destructive_blocked, docs_safety_rules_llm_untrusted, docs_safety_rules_persona_inert [INFERRED 0.75]

## Communities (26 total, 3 thin omitted)

### Community 0 - "App Core & Voice Services"
Cohesion: 0.05
Nodes (58): get_settings(), Application settings, loaded from environment variables / .env., Real execution requires both the opt-in flag and a Windows host.      Inside Doc, real_windows_tools_enabled(), Settings, get_logger(), Central logging setup. Uses rich if available, plain stdlib otherwise., _connect() (+50 more)

### Community 1 - "LLM Planning & Safety"
Cohesion: 0.06
Nodes (47): evaluate(), Safety layer: decides whether a tool may run.  Policy (see docs/SAFETY_RULES.md), Return whether an action at the given safety level may proceed., SafetyDecision, _call_llm(), CommandPlan, _extract_json(), plan_command() (+39 more)

### Community 2 - "Voice & Response Tests"
Cohesion: 0.07
Nodes (28): Any, Path, Transcribe a .wav file. Returns {"text", "language", ...} or {"error"}., SpeechToTextService, _install_fake_tts(), _install_fake_whisper(), Response generator + spoken response tests. Ollama and TTS are mocked., test_needs_confirmation_spoken_response() (+20 more)

### Community 3 - "Router & Response Generation"
Cohesion: 0.08
Nodes (34): detect_intent(), _dispatch_plan(), _dispatch_rules(), _execute(), Command router: intent detection + dispatch through the safety layer.  Two plann, Safety-check and run one tool. `tighten` may only raise the bar:     a safe tool, _unknown_response(), build_response_prompt() (+26 more)

### Community 4 - "Docs & Config"
Cohesion: 0.07
Nodes (35): Docker api service (simulated mode), GPU-first NVIDIA reservation, Docker ollama service (llm profile, GPU-first), Settings (app/config.py), handle_command() shared entry point, LLM command planner (app/llm/command_planner.py), Local-first design principle, Memory (app/core/memory.py, SQLite) (+27 more)

### Community 5 - "Docker LLM Orchestration"
Cohesion: 0.07
Nodes (17): Response, cpu_allowed(), detect_gpu(), docker_available(), get_ollama_container_id(), main(), Start Dockerized Ollama (GPU-first) and run the unattended LLM smoke test.  This, CPU inference is opt-in only (ALLOW_CPU_OLLAMA=true in env or .env). (+9 more)

### Community 6 - "Command Handling & Planner Tests"
Cohesion: 0.19
Nodes (28): handle_command(), command(), Accept a text command; return intent, safety status and simulated result., CommandRequest, test_docker_and_shell_not_launchable_via_commands(), test_agent_name_does_not_alter_safety(), test_saying_fifi_does_not_bypass_confirmation(), mock_llm() (+20 more)

### Community 7 - "Windows Tool Automation"
Cohesion: 0.14
Nodes (24): Any, search_web(), open_app(), open_folder(), Any, Path, Resolve a folder request to (path, None) or (None, rejection reason)., shutdown_pc() (+16 more)

### Community 8 - "LLM Smoke Script"
Cohesion: 0.22
Nodes (16): Popen, config_value(), ensure_model(), evaluate_command(), get_json(), main(), model_present(), ollama_available() (+8 more)

### Community 9 - "Docker Compose Tests"
Cohesion: 0.29
Nodes (7): _active_compose_lines(), compose_text(), Docker/compose conventions and containment: support services only, never desktop, test_api_service_stays_simulated_in_docker(), test_docker_mode_never_reports_real_tools(), test_ollama_gpu_reservation_is_active_not_commented(), test_ollama_service_is_profile_gated()

### Community 10 - "Voice Command Client"
Cohesion: 0.43
Nodes (7): agent_name(), main(), Path, Record a short clip (or take a .wav file) and send it to /voice/command.  Usage, The assistant's name from /identity; falls back to 'Fifi' quietly., record_clip(), send()

### Community 12 - "Test Fixtures (Launch Recorder)"
Cohesion: 0.40
Nodes (4): launches(), LaunchRecorder, Records what would have been launched instead of launching it., Stub every launch primitive so no test can ever start a real app.

## Knowledge Gaps
- **7 isolated node(s):** `Fifi (assistant persona)`, `Command pipeline (router to safety to tools to memory)`, `Memory (app/core/memory.py, SQLite)`, `Wake word placeholder (app/voice/wake_word.py)`, `Settings (app/config.py)` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_settings()` connect `App Core & Voice Services` to `LLM Planning & Safety`, `Voice & Response Tests`, `Router & Response Generation`, `Command Handling & Planner Tests`?**
  _High betweenness centrality (0.134) - this node is a cross-community bridge._
- **Why does `handle_command()` connect `Command Handling & Planner Tests` to `App Core & Voice Services`, `LLM Planning & Safety`, `Voice & Response Tests`, `Router & Response Generation`, `Docker Compose Tests`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `CommandRequest` connect `Command Handling & Planner Tests` to `App Core & Voice Services`, `LLM Planning & Safety`, `Voice & Response Tests`, `Router & Response Generation`, `Docker Compose Tests`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `CommandRequest` (e.g. with `SpeakRequest` and `FakeInfo`) actually correct?**
  _`CommandRequest` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `SafetyLevel` (e.g. with `SafetyDecision` and `CommandPlan`) actually correct?**
  _`SafetyLevel` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Application settings, loaded from environment variables / .env.`, `Real execution requires both the opt-in flag and a Windows host.      Inside Doc`, `Central logging setup. Uses rich if available, plain stdlib otherwise.` to the rest of the system?**
  _85 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `App Core & Voice Services` be split into smaller, more focused modules?**
  _Cohesion score 0.052943354313217325 - nodes in this community are weakly interconnected._