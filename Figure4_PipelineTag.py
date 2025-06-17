import pandas as pd
import matplotlib.pyplot as plt

def plot_pipeline_tag_distribution(csv_file='CMA-PROJ_Dataset.csv'):
    # Load dataset
    df = pd.read_csv(csv_file)

    # Count occurrences of each pipeline tag
    tag_counts = df['Pipeline_tag'].value_counts().sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(tag_counts.index, tag_counts.values, color='skyblue', edgecolor='black')

    # Add count labels on top of bars
    for bar, count in zip(bars, tag_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tag_counts.values)*0.01,
                 str(count), ha='center', va='bottom', fontsize=16, fontweight='bold', 
                 bbox=dict(facecolor='white', edgecolor='blue'))

    # Labels and title (bigger + bold)
    plt.xlabel("Pipeline Tag", fontsize=24, fontweight='bold')
    plt.ylabel("Number of Models", fontsize=22, fontweight='bold')

    # Bold & bigger x-tick labels
    plt.xticks(rotation=45, ha='right', fontsize=22, fontweight='bold')
    plt.yticks(fontsize=22)

    # Black box boundary
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_pipeline_tag_distribution()
