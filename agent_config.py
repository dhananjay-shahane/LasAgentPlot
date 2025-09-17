#!/usr/bin/env python3
"""
Agent Configuration File for LAS File Processing CLI
"""

# LLM Configuration
LLM_CONFIG = {
    "model": "llama3.2:1b",
    "temperature": 0.3,
    "base_url": "http://localhost:11434",
    "timeout": 120
}

# Agent Configuration
AGENT_CONFIG = {
    "handle_parsing_errors": True,
    "verbose": True,
    "max_iterations": 5,
    "react_prompt": "hwchase17/react"
}

# File Paths
PATHS = {
    "data_folder": "data",
    "output_folder": "output",
    "scripts_folder": "scripts"
}

# Plot Configuration
PLOT_CONFIG = {
    "figure_size": (6, 8),
    "dpi": 100,
    "style": "default",
    "grid": True,
    "invert_y_axis": True
}

# MCP Configuration
MCP_CONFIG = {
    "operations": {
        "basic_stats": "Calculate basic statistics for LAS curves",
        "quality_check": "Check data quality and missing values",
        "normalize": "Normalize curve values to [0,1] range",
        "rescue": "Attempt to fix common LAS file issues"
    },
    "supported_encodings": ["utf-8", "latin-1", "cp1252"]
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}