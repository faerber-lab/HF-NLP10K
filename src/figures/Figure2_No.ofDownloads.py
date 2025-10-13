import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "S1-All_NLP.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Convert Downloads column to numeric
df['Downloads'] = pd.to_numeric(df['Downloads'], errors='coerce')

# Define thresholds
thresholds = [1000, 3000, 10000, 50000, 100000, 1000000]

# Calculate number of models above each threshold
counts = [ (df['Downloads'] >= th).sum() for th in thresholds ]

# Convert thresholds for x-axis labels
labels = ['1k', '3k', '10k', '50k', '100k', '1000k']

# Set global font sizes
plt.rcParams.update({'font.size': 22})

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, counts, color='skyblue', edgecolor='black')

# Add count labels on top of bars
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts)*0.01, 
             str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add total models box inside the plot (using axes coordinates)
total_models = len(df)
models_over_1k = (df['Downloads'] >= 1000).sum()
textstr = f"Total Models: {total_models}\nModels ≥ 1k: {models_over_1k}"

# Place text box inside plot
plt.gca().text(0.65, 0.85, textstr, transform=plt.gca().transAxes, 
               fontsize=20,  # <--- Increased font size here
               bbox=dict(facecolor='white', edgecolor='black'))

# Labels and title
plt.xlabel("Minimum Number of Downloads", fontsize=22, fontweight='bold')
plt.ylabel("Number of Models", fontsize=22, fontweight='bold')


plt.tight_layout()
plt.show()
