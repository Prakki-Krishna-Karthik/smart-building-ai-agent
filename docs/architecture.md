# Smart Building AI Agent — System Architecture

## Overview

The Smart Building AI Agent is an autonomous HVAC optimization system that combines EnergyPlus simulation with a local Ollama/Qwen2.5 language model.

EnergyPlus provides physics-based building feedback. The LLM analyzes the structured building state and proposes safe HVAC actions. Every action is validated before being applied to a copied IDF model.

## Architecture Flow

User  
→ Streamlit Dashboard  
→ OptimizationController  
→ EnergyPlus Validation  
→ Baseline Simulation  
→ Output Parser  
→ BuildingState  
→ Ollama/Qwen2.5  
→ ToolRegistry  
→ ToolDispatcher  
→ DecisionEngine Validation  
→ Copied IDF Modification  
→ Optimized Simulation  
→ Energy and Comfort Comparison  
→ JSON/CSV Reports  
→ Dashboard

## Custom Agentic Tools

The project uses a custom agentic tool framework instead of a real MCP server.

The framework contains:

- `Tool` interface
- `ToolRegistry`
- `ToolDispatcher`
- Typed `ToolCall`
- Typed `ToolResult`

Available tools include:

- `validate_energyplus`
- `run_energyplus`
- `parse_outputs`
- `inspect_runtime_errors`
- `get_building_state`
- `apply_recommendations`
- `compare_results`
- `generate_report`

The LLM can select only registered tools. Python injects trusted file paths and execution arguments.

## Safety and Validation

The LLM proposes high-level HVAC recommendations, but Python remains responsible for safety.

The DecisionEngine validates:

- Required fields
- Zone names
- HVAC parameters
- Temperature limits
- Numeric values
- Conflicting recommendations
- Unsafe or unrealistic actions

Unsafe recommendations are rejected and never applied.

The original IDF file is never overwritten. All modifications are made to a copied IDF inside the run output directory.

## Prompt Engineering

The LLM receives:

- Structured BuildingState data
- Registered tool definitions
- Current workflow context
- Results from previous tool calls

Tool selection uses strict JSON mode. The model is instructed to return only valid JSON and only registered tool names.

The recommendation prompt requires a fixed JSON schema containing the reasoning and HVAC actions.

## Prompt Latency and Context Management

Large EnergyPlus models can produce large BuildingState objects. To reduce prompt size, the system prioritizes optimization-relevant fields such as:

- Zone names
- Temperatures
- Humidity
- Occupancy
- PMV and PPD
- HVAC energy
- Current setpoints

Prompt size, estimated token count, response length, retries, and validation failures are logged.

## Handling Simulation Logs

EnergyPlus outputs are parsed from:

- `eplusout.csv`
- `eplusout.eso`
- `eplusout.err`
- `eplusout.end`

Missing optional files are treated as warnings. Genuine fatal errors or non-zero EnergyPlus exit codes cause the simulation to fail.

The parser converts the outputs into a typed `BuildingState`.

## Closed-Loop Optimization

The controller performs the following steps:

1. Run the baseline simulation.
2. Parse the baseline outputs.
3. Send the BuildingState to the AI agent.
4. Select tools or receive a recommendation.
5. Validate the recommendation.
6. Modify a copied IDF.
7. Run the optimized simulation.
8. Parse optimized outputs.
9. Compare energy and comfort.
10. Generate JSON and CSV reports.

The system separates AI predictions from measured EnergyPlus results.

## Measured Demonstration Result

A verified office-building run produced:

- Baseline energy: 129.19B reported units
- Optimized energy: 126.30B reported units
- Measured savings: 2.24%
- Recommendations applied: 2

The dashboard also reports PMV, PPD, prediction validation, comfort validation, execution time, and agent activity.
