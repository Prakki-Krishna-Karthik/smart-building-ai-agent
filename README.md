# Smart Building AI Agent

An AI-powered autonomous building optimization system for the Honeywell
hackathon. The closed loop runs EnergyPlus simulations, interprets
their outputs with local Ollama/Qwen2.5 inference, applies validated HVAC
control decisions, and visualizes outcomes in Streamlit.

## Current status

The core proof-of-concept is implemented: EnergyPlus execution and parsing,
Ollama decisioning, deterministic safety validation, closed-loop optimization,
safe IDF copy-and-mutate control application, JSON/CSV reporting, and a
Streamlit dashboard are available. The system remains a hackathon PoC; real
building deployment still requires hardware/BMS integration and operational
approval.

## Project structure

```text
src/
├── agent/          # Closed-loop application orchestration
├── simulator/      # EnergyPlus adapter boundary
├── llm/            # Ollama/Qwen decision boundary
├── dashboard/      # Streamlit presentation layer
├── controllers/    # HVAC policy and actuator boundary
├── models/         # Shared domain contracts
├── utils/          # Cross-cutting utilities such as logging
└── config/         # Central environment-driven settings
data/
├── input/          # EnergyPlus models and weather inputs
├── output/         # Generated simulation artifacts (ignored by Git)
└── logs/           # Runtime logs (ignored by Git)
docs/               # Architecture and design notes
tests/              # Automated tests
main.py             # Runtime bootstrap
```

## Quick start

Requires Python 3.12+ and Git. EnergyPlus and Ollama are external local
dependencies and can be configured through `.env`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python main.py
pytest
```

Start the dashboard after producing a controller report with:

```powershell
streamlit run src/dashboard/app.py
```

## Hackathon submission

- GitHub repository: https://github.com/Prakki-Krishna-Karthik/smart-building-ai-agent
- Demonstration video: https://drive.google.com/file/d/1gAWvdqjXP3XGLajKUaHfdbsI5GxqxSBc/view?usp=drive_link


The verified MediumOffice demonstration achieved 2.24% measured energy savings
with two recommendations applied. Comfort and prediction validation are shown
separately in the generated report and dashboard.

The dashboard is view-only. It reads the controller's
`optimization_report.json` and `optimization_summary.csv`, and provides the
requested overview, building, energy, comfort, AI decision, history, and log
pages with downloadable reports.

Run the complete integration demo with one command:

```powershell
python scripts/demo_run.py
```

The demo uses `DEMO_IDF_PATH` and `DEMO_WEATHER_FILE` when configured, then
falls back to files under `data/input`. It validates EnergyPlus and Ollama,
runs baseline and optimized simulations, prints BuildingState and performance
summaries, and stores artifacts under `data/output`.

## Design principles

- Keep domain contracts independent of infrastructure frameworks.
- Isolate EnergyPlus and Ollama behind replaceable adapters.
- Parse every simulation's standard EnergyPlus output into a typed building
  state while tolerating missing optional variables.
- Validate model-generated actions before they reach controls.
- Expose simulation, parsing, control, comparison, and reporting operations
  through typed custom agentic tools with an audited dispatcher.
- Keep Streamlit focused on presentation and user interaction.
- Make every control-loop iteration observable through structured logs and
  persisted output artifacts.



## EnergyPlus output parsing

After each simulation, `EnergyPlusRunner` automatically invokes
`EnergyPlusOutputParser`. The parser reads `eplusout.csv`, `eplusout.err`,
`eplusout.end`, and optional `eplusout.eso` files into a strongly typed
`BuildingState` containing simulation health, energy, thermal, comfort, zone,
occupancy, and HVAC-state information. Missing files and columns are logged
and represented with safe optional/default values.

## AI decision engine

`DecisionEngine` is the safety gate around `OllamaClient`. It skips failed or
empty building states, sends usable state to the LLM, rejects unknown zones,
unsupported controls, impossible temperatures, unsafe modes, out-of-range fan
values, and conflicting recommendations, then returns typed actions enriched
with deterministic confidence, priority, energy-savings, and comfort-impact
estimates.

## Custom agentic tools

The project uses an in-process custom tool framework rather than pretending to
be an MCP server. `src/agent/tools.py` defines the `Tool` interface,
`ToolRegistry`, `ToolDispatcher`, typed `ToolResult`, and the built-in tools:
`validate_energyplus`, `run_energyplus`, `parse_outputs`,
`inspect_runtime_errors`, `get_building_state`, `apply_recommendations`,
`compare_results`, and `generate_report`.

`AgenticDecisionEngine` asks Ollama to select only the next registered tool and
a high-level intent. It never accepts model-supplied paths or execution
arguments. Python injects trusted project resources from configuration and the
current controller run, validates them before execution, returns structured
resource errors when unavailable, and sends redacted results back to the model.
Every call is logged and retained in the dispatcher audit history. The
deterministic `DecisionEngine` still validates the final recommendation before
it can reach the controller.

## Ollama setup

Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env`. The client supports Qwen2.5,
Llama3, and Mistral model tags, verifies the model through `/api/tags`, and can
pull a missing model when `OLLAMA_AUTO_PULL=true`. Optimization prompts live in
`src/llm/prompts/`, and validated responses are returned as typed
`BuildingOptimization` and `OptimizationAction` dataclasses.

## EnergyPlus setup

Install EnergyPlus separately and either place its executable on `PATH` or set
the optional `ENERGYPLUS_INSTALLATION_PATH` in `.env` to an installation
directory or direct executable path. `ENERGYPLUS_EXECUTABLE` can override the
default executable name (`energyplus`). The runner validates the installation
before each simulation and stores captured process output in its result.

## Team workflow

Use focused branches and small commits. Add tests alongside each adapter and
controller implementation, and keep generated EnergyPlus outputs out of Git.
