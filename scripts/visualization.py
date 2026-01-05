import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define paths and model
model = "openai"
results_dir = f"results/metrics/{model}/self_improvement_iterative"
techniques = ["self_improvement_iterative"]
operations = ["addition", "convolution", "dot_product","matrix_multiplication", "multiplication"]

# Create empty dataframes to store results
comp_data = []    # For compilability
func_data = []    # For functionality
crystal_data = [] # For CrystalBLEU

# Helper function to safely load JSON
def safe_load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Could not load file: {file_path}")
        return None

# Loop through techniques and operations to collect data
for technique in techniques:
    for operation in operations:
        # Read compilability results
        comp_file = os.path.join(results_dir, f"compilability_{operation}.json")
        comp_json = safe_load_json(comp_file)
        if comp_json:
            comp_data.append({
                'technique': technique,
                'operation': operation,
                'pass_at_1': comp_json.get("pass_at_1", 0),
                'success_rate': comp_json.get("compilable_count", 0) / comp_json.get("total_samples", 1)
            })
        else:
            # Add zero values if file doesn't exist
            comp_data.append({
                'technique': technique,
                'operation': operation,
                'pass_at_1': 0,
                'success_rate': 0
            })
        
        # Read functionality results
        func_file = os.path.join(results_dir, f"functionality_{operation}.json")
        func_json = safe_load_json(func_file)
        # If functionality file doesn't exist (no .exe generated), use zero values
        if func_json:
            func_data.append({
                'technique': technique,
                'operation': operation,
                'pass_at_1': func_json.get("pass_at_1", 0),
                'success_rate': func_json.get("functional_count", 0) / func_json.get("total_samples", 1)
            })
        else:
            # Handle missing functionality file (when no .exe was generated)
            func_data.append({
                'technique': technique,
                'operation': operation,
                'pass_at_1': 0,
                'success_rate': 0
            })
            print(f"No functionality data for {operation} - using zero values")
        
        # Read CrystalBLEU results
        crystal_file = os.path.join(results_dir, f"crystalbleu_{operation}.json")
        crystal_json = safe_load_json(crystal_file)
        if crystal_json:
            crystal_data.append({
                'technique': technique,
                'operation': operation,
                'avg_bleu': crystal_json.get("avg_bleu", 0),
                'total_samples': crystal_json.get("total_samples", 0)
            })
        else:
            # Add zero values if file doesn't exist
            crystal_data.append({
                'technique': technique,
                'operation': operation,
                'avg_bleu': 0,
                'total_samples': 0
            })

# Convert to dataframes - no need for empty checks now since we always populate data
comp_df = pd.DataFrame(comp_data)
func_df = pd.DataFrame(func_data)
crystal_df = pd.DataFrame(crystal_data)

# Visualization with error checking
plt.figure(figsize=(18, 6))

# Compilability Heatmap
plt.subplot(1, 3, 1)
comp_pivot = comp_df.pivot_table(values='success_rate', index='technique', columns='operation', fill_value=0)
sns.heatmap(comp_pivot, annot=True, fmt=".2f", cmap="YlGnBu", 
            cbar_kws={'label': 'Compilability Success Rate'})
plt.title(f'{model.capitalize()}: Compilability by Technique')

# Functionality Heatmap
plt.subplot(1, 3, 2)
func_pivot = func_df.pivot_table(values='success_rate', index='technique', columns='operation', fill_value=0)
sns.heatmap(func_pivot, annot=True, fmt=".2f", cmap="YlOrRd", 
            cbar_kws={'label': 'Functionality Success Rate'})
plt.title(f'{model.capitalize()}: Functionality by Technique')

# CrystalBLEU Heatmap
plt.subplot(1, 3, 3)
crystal_pivot = crystal_df.pivot_table(values='avg_bleu', index='technique', columns='operation', fill_value=0)
sns.heatmap(crystal_pivot, annot=True, fmt=".4f", cmap="PuRd", 
            cbar_kws={'label': 'CrystalBLEU Score'})
plt.title(f'{model.capitalize()}: CrystalBLEU by Technique')

plt.tight_layout()
plt.savefig(f'{model}_performance_heatmap.png', dpi=300)

# Bar chart comparing metrics
plt.figure(figsize=(12, 6))
operations_formatted = [op.replace('_', ' ').capitalize() for op in operations]

# Prepare data for the operations comparison
# No need for safe checks now since we always have data
comp_by_op = comp_df.groupby('operation')['success_rate'].mean()
func_by_op = func_df.groupby('operation')['success_rate'].mean()
crystal_by_op = crystal_df.groupby('operation')['avg_bleu'].mean()

x = range(len(operations))
width = 0.25

plt.bar([i - width for i in x], comp_by_op, width, label='Compilability')
plt.bar(x, func_by_op, width, label='Functionality')
plt.bar([i + width for i in x], crystal_by_op, width, label='CrystalBLEU')

plt.xlabel('Operation')
plt.ylabel('Average Score')
plt.title(f'{model.capitalize()}: Compilability vs Functionality vs CrystalBLEU')
plt.xticks(x, operations_formatted)
plt.legend()
plt.tight_layout()
plt.savefig(f'{model}_comp_func_crystal.png', dpi=300)

# Print summary statistics
print("Compilability Summary:")
print(comp_by_op)
print("\nFunctionality Summary:")
print(func_by_op)
print("\nCrystalBLEU Summary:")
print(crystal_by_op)