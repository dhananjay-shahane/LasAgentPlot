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
    """Create a plot image for the given curve from a LAS file in data folder."""
    file_path = os.path.join(PATHS["data_folder"], filename)
    if not os.path.isfile(file_path):
        return f"Error: File '{filename}' not found in data folder."

    try:
        las = lasio.read(file_path)
        available_curves = [curve.mnemonic for curve in las.curves]
        if curve_name not in available_curves:
            return f"Error: Curve '{curve_name}' not found in '{filename}'. Available curves: {', '.join(available_curves)}"

        curve_data = las[curve_name]
        depth = las['DEPT'] if 'DEPT' in las else las.index  # Use DEPT curve or index as depth

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
    """List LAS files available in the data folder."""
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
    """List curves available in a given LAS file inside the data folder."""
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
    """Get basic information about a LAS file."""
    file_path = os.path.join(PATHS["data_folder"], filename)
    if not os.path.isfile(file_path):
        return f"Error: File '{filename}' not found in data folder."
    
    try:
        las = lasio.read(file_path)
        info = f"Information for {filename}:\n"
        info += f"Version: {las.version[0]}.{las.version[1]}\n"
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
    """
    Process a LAS file using MCP server tools.
    Available operations: 'basic_stats', 'quality_check', 'normalize'
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
        
        # This should never be reached due to the operation check above
        return f"Error: Unhandled operation '{operation}'"
    
    except Exception as e:
        return f"Error processing LAS file: {str(e)}"

# -----------------------------
# MCP Tool: LAS file rescue (fix common issues)
# -----------------------------
@tool(args_schema=LASInfoInput)
def mcp_rescue_las(filename: str) -> str:
    """Attempt to fix common issues in a LAS file."""
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
    try:
        # Note: In a real environment, you would need Ollama running
        # For now, we'll create a mock LLM or handle the connection gracefully
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
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        return None

def main():
    """Main CLI loop."""
    print("LAS File Agent with MCP Tools is starting...")
    print("Initializing agent...")
    
    # Check if we have any LAS files
    las_files = [f for f in os.listdir(PATHS["data_folder"]) if f.endswith(".las")]
    if not las_files:
        print(f"No LAS files found in {PATHS['data_folder']} folder.")
        print("Please add some sample LAS files to test the functionality.")
    
    try:
        agent_executor = initialize_agent()
        if agent_executor is None:
            print("Warning: Could not connect to Ollama. Running in mock mode.")
            print("To use the full agent functionality, please ensure Ollama is running.")
            agent_executor = None
    except Exception as e:
        print(f"Warning: Agent initialization failed: {str(e)}")
        print("Running in basic mode without LLM agent.")
        agent_executor = None
    
    print("\n" + "="*60)
    print("LAS File Agent with MCP Tools is ready!")
    print("="*60)
    print("Available commands:")
    print("- 'list files' or 'show files' - List LAS files in data folder")
    print("- 'info <filename>' - Get information about a LAS file")
    print("- 'curves <filename>' - List curves in a LAS file")
    print("- 'plot <filename> <curve>' - Create a plot for a curve")
    print("- 'stats <filename>' - Get basic statistics")
    print("- 'quality <filename>' - Check data quality")
    print("- 'normalize <filename>' - Normalize curve data")
    print("- 'rescue <filename>' - Fix problematic LAS file")
    print("- 'help' - Show this help message")
    print("- 'exit' or 'quit' - Exit the program")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting LAS File Agent. Goodbye!")
                break
            
            if user_input.lower() in ['help', '?']:
                print("\nAvailable commands:")
                print("- list files - List LAS files in data folder")
                print("- info <filename> - Get information about a LAS file")
                print("- curves <filename> - List curves in a LAS file")
                print("- plot <filename> <curve> - Create a plot for a curve")
                print("- stats <filename> - Get basic statistics")
                print("- quality <filename> - Check data quality")
                print("- normalize <filename> - Normalize curve data")
                print("- rescue <filename> - Fix problematic LAS file")
                continue
            
            # Handle basic commands without agent
            if user_input.lower() in ['list files', 'show files', 'list']:
                result = list_las_files()
                print(f"Agent: {result}")
                continue
            
            # For complex queries, use the agent if available
            if agent_executor:
                try:
                    response = agent_executor.invoke({"input": user_input})
                    print(f"Agent: {response['output']}")
                except Exception as e:
                    print(f"Agent error: {str(e)}")
                    print("Try a simpler command or check if Ollama is running.")
            else:
                # Basic command parsing without agent
                parts = user_input.split()
                if len(parts) >= 2:
                    command = parts[0].lower()
                    filename = parts[1]
                    
                    if command == 'info':
                        result = get_las_info(filename)
                        print(f"Agent: {result}")
                    elif command == 'curves':
                        result = list_curves(filename)
                        print(f"Agent: {result}")
                    elif command == 'plot' and len(parts) >= 3:
                        curve_name = parts[2]
                        result = las_create_plot(filename, curve_name)
                        print(f"Agent: {result}")
                    elif command == 'stats':
                        result = mcp_process_las(filename, 'basic_stats')
                        print(f"Agent: {result}")
                    elif command == 'quality':
                        result = mcp_process_las(filename, 'quality_check')
                        print(f"Agent: {result}")
                    elif command == 'normalize':
                        result = mcp_process_las(filename, 'normalize')
                        print(f"Agent: {result}")
                    elif command == 'rescue':
                        result = mcp_rescue_las(filename)
                        print(f"Agent: {result}")
                    else:
                        print("Agent: Command not recognized. Type 'help' for available commands.")
                else:
                    print("Agent: Please provide a valid command. Type 'help' for available commands.")
                    
        except KeyboardInterrupt:
            print("\nExiting LAS File Agent. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()