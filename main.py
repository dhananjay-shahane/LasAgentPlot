#!/usr/bin/env python3
"""
LAS File Processing and Plotting Agent with MCP Tools
"""

import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import lasio
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from agent_config import LLM_CONFIG, AGENT_CONFIG, PATHS, PLOT_CONFIG, MCP_CONFIG, LOGGING_CONFIG

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
os.makedirs(PATHS["data_folder"], exist_ok=True)
os.makedirs(PATHS["output_folder"], exist_ok=True)
os.makedirs(PATHS["scripts_folder"], exist_ok=True)

# Set matplotlib backend for headless environment
plt.switch_backend('Agg')

# -----------------------------
# Pydantic input models
# -----------------------------
class LASPlotInput(BaseModel):
    filename: str = Field(description="LAS file name in data folder")
    curve_name: str = Field(description="Name of curve to plot from LAS file")

class LASInfoInput(BaseModel):
    filename: str = Field(description="LAS file name in data folder")


# -----------------------------
# Tool: Create LAS curve plot
# -----------------------------
@tool(args_schema=LASPlotInput)
def las_create_plot(filename: str, curve_name: str) -> str:
    """Create a plot image for a specific curve from a LAS file in the data folder.
    
    Use this tool to generate depth plots for well log curves.
    
    Args:
        filename: Name of the LAS file in data folder (e.g., "sample_well.las")
        curve_name: Name of the curve to plot (e.g., "GR", "RT", "NPHI")
    
    Example usage:
        Action: las_create_plot
        Action Input: {"filename": "sample_well.las", "curve_name": "GR"}
    """
    file_path = os.path.join(PATHS["data_folder"], filename)
    if not os.path.isfile(file_path):
        return f"Error: File '{filename}' not found in data folder."

    try:
        las = lasio.read(file_path)
        available_curves = [curve.mnemonic for curve in las.curves]
        if curve_name not in available_curves:
            return f"Error: Curve '{curve_name}' not found in '{filename}'. Available curves: {', '.join(available_curves)}"

        curve_data = las[curve_name]
        depth = las['DEPT'] if 'DEPT' in las else las.index

        # Handle NaN values safely
        mask = ~np.isnan(curve_data)
        plt.figure(figsize=PLOT_CONFIG["figure_size"], dpi=PLOT_CONFIG["dpi"])
        plt.plot(curve_data[mask], depth[mask])
        
        if PLOT_CONFIG["invert_y_axis"]:
            plt.gca().invert_yaxis()
        
        plt.xlabel(curve_name)
        plt.ylabel("Depth")
        plt.title(f"{curve_name} vs Depth from {filename}")
        
        if PLOT_CONFIG["grid"]:
            plt.grid(True)

        output_path = os.path.join(PATHS["output_folder"], f"{filename}_{curve_name}.png")
        plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"])
        plt.close()

        return f"Success: Plot saved to {output_path}"
    except Exception as e:
        logger.error(f"Failed to create plot: {str(e)}")
        return f"Error: Failed to create plot: {str(e)}"


# -----------------------------
# Tool: List LAS files
# -----------------------------
@tool
def list_las_files() -> str:
    """List all LAS files available in the data folder.
    
    Use this tool to see what LAS files are available for processing.
    
    Example usage:
        Action: list_las_files
        Action Input: {}
    """
    try:
        files = [f for f in os.listdir(PATHS["data_folder"]) if f.endswith(".las")]
        if not files:
            return "No LAS files found in data folder."
        return f"Available LAS files: {', '.join(files)}"
    except Exception as e:
        return f"Error reading data folder: {str(e)}"


# -----------------------------
# Tool: List curves from a LAS file
# -----------------------------
@tool(args_schema=LASInfoInput)
def list_curves(filename: str) -> str:
    """List all curves (data channels) available in a specific LAS file.
    
    Use this tool to see what well log curves are in a LAS file.
    
    Args:
        filename: Name of the LAS file in data folder (e.g., "sample_well.las")
    
    Example usage:
        Action: list_curves
        Action Input: {"filename": "sample_well.las"}
    """
    file_path = os.path.join(PATHS["data_folder"], filename)
    if not os.path.isfile(file_path):
        return f"Error: File '{filename}' not found in data folder."
    try:
        las = lasio.read(file_path)
        curves = [curve.mnemonic for curve in las.curves]
        return f"Available curves in {filename}: {', '.join(curves)}"
    except Exception as e:
        return f"Error reading curves: {str(e)}"


# -----------------------------
# Tool: Get LAS file info
# -----------------------------
@tool(args_schema=LASInfoInput)
def get_las_info(filename: str) -> str:
    """Get detailed information about a LAS file including version, curves, and well data.
    
    Use this tool to get comprehensive information about a LAS file.
    
    Args:
        filename: Name of the LAS file in data folder (e.g., "sample_well.las")
    
    Example usage:
        Action: get_las_info
        Action Input: {"filename": "sample_well.las"}
    """
    file_path = os.path.join(PATHS["data_folder"], filename)
    if not os.path.isfile(file_path):
        return f"Error: File '{filename}' not found in data folder."
    
    try:
        las = lasio.read(file_path)
        info = f"Information for {filename}:\n"
        info += f"Version: {getattr(las, 'version', 'N/A')}\n"
        info += f"Wrap: {getattr(las, 'wrap', 'N/A')}\n"
        info += f"Number of curves: {len(las.curves)}\n"
        info += f"Data shape: {las.data.shape}\n"
        info += f"Start depth: {las.index[0]}\n"
        info += f"End depth: {las.index[-1]}\n"
        
        # Add well information if available
        if hasattr(las, 'well') and las.well:
            info += "\nWell Information:\n"
            for item in las.well:
                info += f"{item.mnemonic}: {item.value} {item.unit}\n"
                
        return info
    except Exception as e:
        return f"Error reading LAS file: {str(e)}"


# -----------------------------
# MCP Tool: Process LAS with MCP server
# -----------------------------
@tool
def mcp_process_las(filename: str, operation: str) -> str:
    """Process a LAS file using MCP operations for statistics, quality checks, or normalization.
    
    Use this tool to analyze LAS file data or perform data processing operations.
    
    Args:
        filename: Name of LAS file in data folder (e.g., "sample_well.las")
        operation: Operation to perform - must be "basic_stats", "quality_check", or "normalize"
    
    Example usage:
        Action: mcp_process_las
        Action Input: {"filename": "sample_well.las", "operation": "basic_stats"}
    """
    file_path = os.path.join(PATHS["data_folder"], filename)
    if not os.path.isfile(file_path):
        return f"Error: File '{filename}' not found in data folder."
    
    if operation not in MCP_CONFIG["operations"]:
        available_ops = list(MCP_CONFIG["operations"].keys())
        return f"Error: Unknown operation '{operation}'. Available operations: {', '.join(available_ops)}"
    
    try:
        if operation == "basic_stats":
            las = lasio.read(file_path)
            stats_info = f"Basic statistics for {filename}:\n"
            for curve in las.curves:
                if curve.mnemonic != 'DEPT':
                    curve_data = las[curve.mnemonic]
                    valid_data = curve_data[~np.isnan(curve_data)]
                    if len(valid_data) > 0:
                        stats_info += f"{curve.mnemonic}: min={np.min(valid_data):.3f}, max={np.max(valid_data):.3f}, mean={np.mean(valid_data):.3f}\n"
            return stats_info
            
        elif operation == "quality_check":
            las = lasio.read(file_path)
            quality_info = f"Quality check for {filename}:\n"
            for curve in las.curves:
                curve_data = las[curve.mnemonic]
                nan_count = np.sum(np.isnan(curve_data))
                percent_nan = (nan_count / len(curve_data)) * 100
                quality_info += f"{curve.mnemonic}: {percent_nan:.1f}% missing data\n"
            return quality_info
            
        elif operation == "normalize":
            las = lasio.read(file_path)
            norm_info = f"Normalization applied to {filename}:\n"
            for curve in las.curves:
                if curve.mnemonic != 'DEPT':
                    curve_data = las[curve.mnemonic]
                    valid_data = curve_data[~np.isnan(curve_data)]
                    if len(valid_data) > 0:
                        min_val = np.min(valid_data)
                        max_val = np.max(valid_data)
                        if max_val > min_val:
                            las[curve.mnemonic] = (curve_data - min_val) / (max_val - min_val)
                            norm_info += f"{curve.mnemonic}: normalized to [0,1] range\n"
            
            output_filename = f"normalized_{filename}"
            output_path = os.path.join(PATHS["data_folder"], output_filename)
            las.write(output_path)
            norm_info += f"Normalized file saved as {output_filename}"
            return norm_info
        
        return f"Error: Unhandled operation '{operation}'"
    
    except Exception as e:
        return f"Error processing LAS file: {str(e)}"


# -----------------------------
# MCP Tool: LAS file rescue (fix common issues)
# -----------------------------
@tool(args_schema=LASInfoInput)
def mcp_rescue_las(filename: str) -> str:
    """Attempt to fix common issues and errors in a LAS file.
    
    Use this tool to repair problematic LAS files with encoding or format issues.
    
    Args:
        filename: Name of the LAS file in data folder (e.g., "sample_well.las")
    
    Example usage:
        Action: mcp_rescue_las
        Action Input: {"filename": "sample_well.las"}
    """
    file_path = os.path.join(PATHS["data_folder"], filename)
    if not os.path.isfile(file_path):
        return f"Error: File '{filename}' not found in data folder."
    
    try:
        encodings = MCP_CONFIG["supported_encodings"]
        las = None
        
        for encoding in encodings:
            try:
                las = lasio.read(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if las is None:
            return "Error: Could not read LAS file with any known encoding."
        
        issues_fixed = []
        
        if not hasattr(las, 'version') or las.version is None:
            las.version = [1.2, 'CWLS']
            issues_fixed.append("Added missing version section")
        
        if not hasattr(las, 'well') or las.well is None:
            las.well = []
            issues_fixed.append("Added missing well section")
        
        if not hasattr(las, 'curves') or las.curves is None:
            return "Error: No curve information found in LAS file."
        
        rescue_filename = f"rescued_{filename}"
        rescue_path = os.path.join(PATHS["data_folder"], rescue_filename)
        las.write(rescue_path)
        
        if issues_fixed:
            return f"Rescued file saved as {rescue_filename}. Issues fixed: {', '.join(issues_fixed)}"
        else:
            return f"No issues found in {filename}. A copy was saved as {rescue_filename} for verification."
            
    except Exception as e:
        return f"Error rescuing LAS file: {str(e)}"

def initialize_agent():
    """Initialize the LangChain agent with tools."""
    llm = ChatOllama(
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
        base_url=LLM_CONFIG["base_url"],
        timeout=LLM_CONFIG["timeout"]
    )

    tools = [
        las_create_plot, 
        list_las_files, 
        list_curves, 
        get_las_info,
        mcp_process_las,
        mcp_rescue_las
    ]

    # Get the ReAct prompt
    prompt = hub.pull(AGENT_CONFIG["react_prompt"])

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=AGENT_CONFIG["handle_parsing_errors"],
        verbose=AGENT_CONFIG["verbose"],
        max_iterations=AGENT_CONFIG["max_iterations"]
    )
    
    return agent_executor

def main():
    """Main CLI loop."""
    print("LAS File Agent with MCP Tools is starting...")
    print("Initializing agent...")
    
    # Check if we have any LAS files
    las_files = [f for f in os.listdir(PATHS["data_folder"]) if f.endswith(".las")]
    if not las_files:
        print(f"No LAS files found in {PATHS['data_folder']} folder.")
        print("Please add some sample LAS files to test the functionality.")
    
    # Initialize the agent - this will raise an exception if Ollama is not available
    agent_executor = initialize_agent()
    
    print("\n" + "="*60)
    print("LAS File Agent with MCP Tools is ready!")
    print("="*60)
    print("Ask me anything about LAS files or use natural language queries like:")
    print("- 'What LAS files are available?'")
    print("- 'Show me information about sample_well.las'")
    print("- 'Create a gamma ray plot for sample_well.las'")
    print("- 'What curves are in the file?'")
    print("- 'Calculate basic statistics'")
    print("- 'Check data quality'")
    print("- Type 'exit' or 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting LAS File Agent. Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                response = agent_executor.invoke({"input": user_input})
                print(f"Agent: {response['output']}")
            except Exception as e:
                print(f"Agent error: {str(e)}")
                print("Please ensure Ollama is running and try again.")
                    
        except KeyboardInterrupt:
            print("\nExiting LAS File Agent. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()