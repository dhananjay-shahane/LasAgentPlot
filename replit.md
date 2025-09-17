# LAS File Processing and Plotting Agent

## Overview

This project is a Python-based command-line interface (CLI) application for processing and visualizing LAS (Log ASCII Standard) files, which are commonly used in the oil and gas industry for well log data. The application leverages LangChain agents integrated with Ollama LLM to provide intelligent processing capabilities, allowing users to analyze well log curves, generate plots, and perform data quality checks through natural language interactions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Agent-Based Architecture
The system is built around a LangChain ReAct agent that can interpret natural language commands and execute appropriate tools for LAS file processing. The agent uses a local Ollama LLM (llama3.2:1b) for natural language understanding and reasoning.

### Core Components

**Configuration Management**: Centralized configuration through `agent_config.py` that defines LLM settings, agent behavior, file paths, plotting parameters, and MCP (Model Context Protocol) operations. This approach allows for easy customization without code changes.

**Tool-Based Architecture**: The system implements LangChain tools with Pydantic input validation for structured interactions. Each tool handles specific LAS file operations like plotting curves and extracting file information.

**Modular Plot Generation**: A dedicated `LASPlotGenerator` class in the scripts folder provides reusable plotting functionality that can be used both by the agent tools and as standalone utilities.

**Error Handling and Logging**: Comprehensive error handling throughout the system with configurable logging levels for debugging and monitoring agent operations.

### Data Processing Pipeline

**File Management**: Structured directory organization with separate folders for input data, output files, and utility scripts. The system automatically creates required directories on startup.

**LAS File Processing**: Uses the `lasio` library for robust parsing of LAS files, with support for multiple encodings and error recovery mechanisms.

**Visualization**: matplotlib-based plotting with headless backend support for server environments, configurable plot styling, and automatic output file generation.

### Design Patterns

**Configuration-Driven Design**: All system behavior is controlled through centralized configuration dictionaries, making the application highly customizable without code modifications.

**Tool Registry Pattern**: LangChain tools are defined with clear input schemas and descriptions, allowing the agent to understand and select appropriate tools based on user requests.

**Separation of Concerns**: Clear separation between agent logic, tool implementations, and utility functions for maintainability and testability.

## External Dependencies

### Core Libraries
- **LangChain**: Provides the agent framework and tool system for natural language interaction
- **Ollama**: Local LLM runtime for agent reasoning (llama3.2:1b model)
- **lasio**: Specialized library for reading and parsing LAS well log files
- **matplotlib**: Plotting and visualization library for generating curve plots
- **Pydantic**: Data validation and serialization for tool inputs

### System Requirements
- **Local Ollama Server**: Must be running on localhost:11434 for LLM inference
- **Python Environment**: Requires Python 3.x with scientific computing packages
- **File System**: Requires read/write access for data processing and plot generation

### Optional Integrations
- **MCP (Model Context Protocol)**: Framework for extending agent capabilities with additional tools and operations
- **Multiple Encoding Support**: Handles various text encodings commonly found in LAS files (utf-8, latin-1, cp1252)

The architecture is designed to be extensible, allowing for easy addition of new LAS processing tools and integration with additional external services or databases as needed.

## Replit Environment Setup

### Recent Changes (September 17, 2025)

**Import Setup Completed**: Successfully configured the LAS File Processing and Plotting Agent to run in the Replit environment.

**Key Configuration Changes**:
- **Dependencies**: Installed all required Python dependencies using `uv` package manager from `pyproject.toml`
- **Ollama Configuration**: Updated `agent_config.py` to use localhost:11434 for local Ollama server
- **Error Handling**: Enhanced error handling to gracefully fallback when Ollama is unavailable
- **Plotting Fixes**: Fixed numpy array comparison issues in plot generation scripts
- **Directory Structure**: Created required output folder for generated plots
- **Workflow Setup**: Configured "LAS CLI Agent" workflow to run the main application

**Working Features**:
- Standalone plot generation script works correctly (`python scripts/plot_generator.py`)
- All plot types functional: depth plots, gamma ray plots, multi-curve plots
- Sample data processing with `data/sample_well.las`
- Generated plots saved to `output/` directory

**Usage**:
1. **Interactive Mode**: Run the workflow "LAS CLI Agent" for full agent interaction (requires Ollama)
2. **Standalone Mode**: Use `python scripts/plot_generator.py sample_well.las <plot_type> [curve_name]` for direct plotting

**Dependencies Installed**: langchain, langchain-community, langchain-core, langchain-ollama, lasio, matplotlib, numpy, pydantic, and all supporting libraries.