import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_nlp_model_download_distribution(csv_file='HLT-NLP10K.csv.gz'):
    # Load data with correct column names
    df = pd.read_csv(csv_file, usecols=['Model ID', 'Downloads', 'Likes'])

    # Drop missing values
    df = df.dropna()

    # Sort by downloads in descending order
    df = df.sort_values(by='Downloads', ascending=False).reset_index(drop=True)

    # Create rank
    df['rank'] = df.index + 1

    # Compute cumulative downloads
    total_downloads = df['Downloads'].sum()
    df['cumulative_percentage'] = df['Downloads'].cumsum() / total_downloads * 100

    # Find the number of NLP models accounting for 80% of downloads
    top_80_percent = df[df['cumulative_percentage'] <= 80]
    num_top_80 = len(top_80_percent)

    # Seaborn style
    sns.set_style("whitegrid")

    # Plot
    plt.figure(figsize=(12, 8))
    plt.loglog(df['rank'], df['Downloads'], 'bo', markersize=4, label='NLP Models', alpha=0.7)

    # Get top models to annotate
    top_models = df.head(4)

    # Custom annotation positions (adjusted to move boxes lower)
    annotation_positions = [
        (2.0, 1.13),  # FacebookAI/xlm-roberta-large
        (5.0, 0.5),   # sentence-transformers/all-MiniLM-L6-v2
        (3.5, 0.3),   # google-bert/bert-base-uncased
        (1.5, 0.2),   # sentence-transformers/all-mpnet-base-v2
    ]

    arrowprops = dict(arrowstyle="->", color='black', lw=1.5)

    for i, (_, row) in enumerate(top_models.iterrows()):
        dx, dy = annotation_positions[i]
        # Position of annotation text
        xy_text = (row['rank'] * dx, row['Downloads'] * dy)
        plt.annotate(
            row['Model ID'],
            xy=(row['rank'], row['Downloads']),
            xytext=xy_text,
            textcoords='data',
            arrowprops=arrowprops,
            fontsize=16,
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
            ha='left', va='top'   # <-- This makes arrow start at top-left corner of box
        )

    # Axis labels
    plt.xlabel('Model Downloads Rank', fontsize=28, fontweight='bold')
    plt.ylabel('Downloads', fontsize=28, fontweight='bold')

    # Tick sizes
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.legend(fontsize=20, loc='lower left')
    plt.grid(True, which='major', linestyle='--', alpha=0.4)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)

    # Thicker black box
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color('black')

    plt.tight_layout()
    plt.show()

# Example usage:
plot_nlp_model_download_distribution()
