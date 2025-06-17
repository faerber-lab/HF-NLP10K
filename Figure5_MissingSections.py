import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load dataset
file_path = "CMA-PROJ_Dataset.csv"
df = pd.read_csv(file_path)

# Rename column for better label on plot
df = df.rename(columns={"Evaluation Metrics via Model Card": "Evaluation Metrics (MC)"})

# Define the sections to check for missing data (with updated column name)
sections = ["Limitations", "Use Cases", "Specialization", "Training Data", "Comparison", 
            "Evaluation Metrics (MC)", "Base Model", "Fine-Tuning"]

# Function to count missing and non-missing values
def count_values(column):
    missing = (column.isna() | column.str.strip().isin(["None explicitly mentioned.", "Not explicitly mentioned."])).sum()
    present = len(column) - missing
    return pd.Series({'Present': present, 'Missing': missing})

# Apply the function column-wise
missing_data = df[sections].apply(count_values)

# Prepare data for plotting
present = missing_data.loc['Present']
missing = missing_data.loc['Missing']
total = present + missing

# Use matplotlib directly for full control
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(14, 8))

indices = np.arange(len(sections))
bar_width = 0.7  # keep bars thick

# Plot stacked bars
ax.bar(indices, present, bar_width, color="#2CA02C", label='Present')
ax.bar(indices, missing, bar_width, bottom=present, color="#D62728", label='Missing')

# Add percentage labels inside bars
for i in range(len(sections)):
    ax.text(indices[i], present[i] / 2, f"{(present[i] / total[i])*100:.1f}%", 
            ha='center', va='center', fontsize=18, fontweight='bold', color='white')
    ax.text(indices[i], present[i] + (missing[i] / 2), f"{(missing[i] / total[i])*100:.1f}%", 
            ha='center', va='center', fontsize=18, fontweight='bold', color='white')

# Styling for paper
ax.set_ylabel("Number of Entries", fontsize=22, fontweight='bold')
ax.set_title("Breakdown of Missing Sections in Dataset Cards (Stacked)", fontsize=24, fontweight='bold')
ax.set_xticks(indices)
ax.set_xticklabels(sections, fontsize=20, fontweight='bold', rotation=30, ha="right")
ax.legend(title="Status", title_fontsize=20, fontsize=18)
ax.tick_params(axis='y', labelsize=18)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()

# High-resolution export for CIKM paper
plt.savefig("Breakdown_Missing_Sections_CIKM_Final.png", dpi=600, bbox_inches='tight')
plt.show()
