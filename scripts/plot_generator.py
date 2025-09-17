#!/usr/bin/env python3
"""
Modular Plot Generation Scripts for LAS File Processing
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import lasio
from typing import List, Optional, Tuple

# Add parent directory to path to import agent_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_config import PATHS, PLOT_CONFIG

# Set matplotlib backend for headless environment
plt.switch_backend('Agg')

class LASPlotGenerator:
    """Class for generating various types of LAS file plots."""
    
    def __init__(self, data_folder: Optional[str] = None, output_folder: Optional[str] = None):
        self.data_folder = data_folder or PATHS["data_folder"]
        self.output_folder = output_folder or PATHS["output_folder"]
        
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)
    
    def load_las_file(self, filename: str) -> Optional[lasio.LASFile]:
        """Load a LAS file and return the LAS object."""
        file_path = os.path.join(self.data_folder, filename)
        if not os.path.isfile(file_path):
            print(f"Error: File '{filename}' not found in {self.data_folder}")
            return None
        
        try:
            las = lasio.read(file_path)
            return las
        except Exception as e:
            print(f"Error reading LAS file: {str(e)}")
            return None
    
    def create_depth_plot(self, filename: str, curve_name: str, 
                         title: Optional[str] = None, xlabel: Optional[str] = None) -> str:
        """Create a depth vs curve plot."""
        las = self.load_las_file(filename)
        if las is None:
            return f"Failed to load LAS file: {filename}"
        
        available_curves = [curve.mnemonic for curve in las.curves]
        if curve_name not in available_curves:
            return f"Curve '{curve_name}' not found. Available: {', '.join(available_curves)}"
        
        try:
            curve_data = las[curve_name]
            available_curves = [c.mnemonic for c in las.curves]
            depth = np.asarray(las['DEPT'] if 'DEPT' in available_curves else las.index, dtype=float)
            
            # Handle NaN values safely
            curve_data = np.asarray(curve_data, dtype=float)
            depth = np.asarray(depth, dtype=float)
            mask = ~np.isnan(curve_data)
            
            plt.figure(figsize=PLOT_CONFIG["figure_size"], dpi=PLOT_CONFIG["dpi"])
            plt.plot(curve_data[mask], depth[mask], linewidth=1.5)
            
            if PLOT_CONFIG["invert_y_axis"]:
                plt.gca().invert_yaxis()
            
            plt.xlabel(xlabel or curve_name)
            plt.ylabel("Depth (ft)")
            plt.title(title or f"{curve_name} vs Depth - {filename}")
            
            if PLOT_CONFIG["grid"]:
                plt.grid(True, alpha=0.3)
            
            # Add some styling
            plt.tight_layout()
            
            output_path = os.path.join(self.output_folder, f"{filename}_{curve_name}_depth.png")
            plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
            plt.close()
            
            return f"Depth plot saved: {output_path}"
        except Exception as e:
            return f"Error creating depth plot: {str(e)}"
    
    def create_gamma_ray_plot(self, filename: str, gamma_curve: str = "GR") -> str:
        """Create a specialized gamma ray log plot."""
        las = self.load_las_file(filename)
        if las is None:
            return f"Failed to load LAS file: {filename}"
        
        # Try common gamma ray curve names
        gamma_names = [gamma_curve, "GR", "GAMMA", "GAMMA_RAY", "GRD"]
        curve_found = None
        
        for name in gamma_names:
            if name in [curve.mnemonic for curve in las.curves]:
                curve_found = name
                break
        
        if curve_found is None:
            available_curves = [curve.mnemonic for curve in las.curves]
            return f"No gamma ray curve found. Tried: {', '.join(gamma_names)}. Available: {', '.join(available_curves)}"
        
        try:
            gamma_data = las[curve_found]
            available_curves = [c.mnemonic for c in las.curves]
            depth = np.asarray(las['DEPT'] if 'DEPT' in available_curves else las.index, dtype=float)
            
            # Handle NaN values safely
            gamma_data = np.asarray(gamma_data, dtype=float)
            depth = np.asarray(depth, dtype=float)
            mask = ~np.isnan(gamma_data)
            
            plt.figure(figsize=(8, 10), dpi=PLOT_CONFIG["dpi"])
            plt.plot(gamma_data[mask], depth[mask], 'g-', linewidth=1.5, label='Gamma Ray')
            
            # Fill the gamma ray curve for better visualization
            plt.fill_betweenx(depth[mask], 0, gamma_data[mask], alpha=0.3, color='green')
            
            if PLOT_CONFIG["invert_y_axis"]:
                plt.gca().invert_yaxis()
            
            plt.xlabel(f"{curve_found} (API Units)")
            plt.ylabel("Depth (ft)")
            plt.title(f"Gamma Ray Log - {filename}")
            plt.legend()
            
            if PLOT_CONFIG["grid"]:
                plt.grid(True, alpha=0.3)
            
            # Add reference lines for typical gamma ray values
            plt.axvline(x=75, color='red', linestyle='--', alpha=0.5, label='Typical Sand/Shale Boundary')
            plt.legend()
            
            plt.tight_layout()
            
            output_path = os.path.join(self.output_folder, f"{filename}_gamma_ray_log.png")
            plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
            plt.close()
            
            return f"Gamma ray plot saved: {output_path}"
        except Exception as e:
            return f"Error creating gamma ray plot: {str(e)}"
    
    def create_multi_curve_plot(self, filename: str, curve_names: List[str], 
                               subplot_layout: bool = True) -> str:
        """Create a multi-curve plot with multiple tracks."""
        las = self.load_las_file(filename)
        if las is None:
            return f"Failed to load LAS file: {filename}"
        
        available_curves = [curve.mnemonic for curve in las.curves]
        valid_curves = [curve for curve in curve_names if curve in available_curves]
        
        if not valid_curves:
            return f"No valid curves found. Requested: {', '.join(curve_names)}. Available: {', '.join(available_curves)}"
        
        try:
            available_curves = [c.mnemonic for c in las.curves]
            depth = np.asarray(las['DEPT'] if 'DEPT' in available_curves else las.index, dtype=float)
            
            if subplot_layout:
                # Create subplots for each curve
                fig, axes = plt.subplots(1, len(valid_curves), figsize=(4*len(valid_curves), 10), dpi=PLOT_CONFIG["dpi"])
                if len(valid_curves) == 1:
                    axes = [axes]
                
                for i, curve_name in enumerate(valid_curves):
                    curve_data = np.asarray(las[curve_name], dtype=float)
                    mask = ~np.isnan(curve_data)
                    
                    axes[i].plot(curve_data[mask], depth[mask], linewidth=1.5)
                    if PLOT_CONFIG["invert_y_axis"]:
                        axes[i].invert_yaxis()
                    axes[i].set_xlabel(curve_name)
                    if i == 0:
                        axes[i].set_ylabel("Depth (ft)")
                    axes[i].set_title(f"{curve_name}")
                    if PLOT_CONFIG["grid"]:
                        axes[i].grid(True, alpha=0.3)
                
                plt.suptitle(f"Multi-Curve Log - {filename}")
                plt.tight_layout()
            else:
                # Single plot with multiple curves
                plt.figure(figsize=(12, 10), dpi=PLOT_CONFIG["dpi"])
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                
                for i, curve_name in enumerate(valid_curves):
                    curve_data = np.asarray(las[curve_name], dtype=float)
                    mask = ~np.isnan(curve_data)
                    color = colors[i % len(colors)]
                    
                    plt.plot(curve_data[mask], depth[mask], color=color, 
                            linewidth=1.5, label=curve_name)
                
                if PLOT_CONFIG["invert_y_axis"]:
                    plt.gca().invert_yaxis()
                
                plt.xlabel("Curve Values")
                plt.ylabel("Depth (ft)")
                plt.title(f"Multi-Curve Log - {filename}")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                if PLOT_CONFIG["grid"]:
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
            
            plot_type = "subplots" if subplot_layout else "overlay"
            output_path = os.path.join(self.output_folder, f"{filename}_multi_curve_{plot_type}.png")
            plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
            plt.close()
            
            return f"Multi-curve plot saved: {output_path}"
        except Exception as e:
            return f"Error creating multi-curve plot: {str(e)}"
    
    def create_resistivity_plot(self, filename: str) -> str:
        """Create a specialized resistivity log plot."""
        las = self.load_las_file(filename)
        if las is None:
            return f"Failed to load LAS file: {filename}"
        
        # Try common resistivity curve names
        resistivity_names = ["RT", "RES", "RESISTIVITY", "ILD", "LLD", "LLS", "MSFL"]
        found_curves = []
        
        available_curves = [curve.mnemonic for curve in las.curves]
        for name in resistivity_names:
            if name in available_curves:
                found_curves.append(name)
        
        if not found_curves:
            return f"No resistivity curves found. Tried: {', '.join(resistivity_names)}. Available: {', '.join(available_curves)}"
        
        try:
            available_curves = [c.mnemonic for c in las.curves]
            depth = np.asarray(las['DEPT'] if 'DEPT' in available_curves else las.index, dtype=float)
            
            plt.figure(figsize=(10, 10), dpi=PLOT_CONFIG["dpi"])
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, curve_name in enumerate(found_curves):
                curve_data = np.asarray(las[curve_name], dtype=float)
                mask = ~np.isnan(curve_data) & (curve_data > 0)  # Remove negative values for log scale
                
                if np.any(mask):
                    color = colors[i % len(colors)]
                    plt.semilogx(curve_data[mask], depth[mask], color=color, 
                               linewidth=1.5, label=curve_name)
            
            if PLOT_CONFIG["invert_y_axis"]:
                plt.gca().invert_yaxis()
            
            plt.xlabel("Resistivity (ohm-m)")
            plt.ylabel("Depth (ft)")
            plt.title(f"Resistivity Log - {filename}")
            plt.legend()
            
            if PLOT_CONFIG["grid"]:
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_path = os.path.join(self.output_folder, f"{filename}_resistivity_log.png")
            plt.savefig(output_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
            plt.close()
            
            return f"Resistivity plot saved: {output_path}"
        except Exception as e:
            return f"Error creating resistivity plot: {str(e)}"

def main():
    """CLI interface for the plot generator."""
    if len(sys.argv) < 3:
        print("Usage: python plot_generator.py <filename> <plot_type> [curve_name(s)]")
        print("Plot types: depth, gamma, multi, resistivity")
        print("Examples:")
        print("  python plot_generator.py sample.las depth GR")
        print("  python plot_generator.py sample.las gamma")
        print("  python plot_generator.py sample.las multi GR RT NPHI")
        return
    
    filename = sys.argv[1]
    plot_type = sys.argv[2].lower()
    
    generator = LASPlotGenerator()
    
    if plot_type == "depth":
        if len(sys.argv) < 4:
            print("Error: depth plot requires curve name")
            return
        curve_name = sys.argv[3]
        result = generator.create_depth_plot(filename, curve_name)
        print(result)
    
    elif plot_type == "gamma":
        gamma_curve = sys.argv[3] if len(sys.argv) > 3 else "GR"
        result = generator.create_gamma_ray_plot(filename, gamma_curve)
        print(result)
    
    elif plot_type == "multi":
        if len(sys.argv) < 4:
            print("Error: multi plot requires at least one curve name")
            return
        curve_names = sys.argv[3:]
        result = generator.create_multi_curve_plot(filename, curve_names, subplot_layout=True)
        print(result)
    
    elif plot_type == "resistivity":
        result = generator.create_resistivity_plot(filename)
        print(result)
    
    else:
        print(f"Unknown plot type: {plot_type}")
        print("Available plot types: depth, gamma, multi, resistivity")

if __name__ == "__main__":
    main()