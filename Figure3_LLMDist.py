import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_nlp_model_download_distribution(csv_file='CMA-PROJ_Dataset.csv'):
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

    # Set seaborn style for cleaner background
    sns.set_style("whitegrid")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(df['rank'], df['Downloads'], 'bo', markersize=4, label='NLP Models', alpha=0.7)

    # Annotate 80% threshold
    plt.text(df['rank'].iloc[int(len(df) * 0.25)],
             max(df['Downloads']) * 0.3,
             f'Top {num_top_80} models account for 80% of downloads.',
             bbox=dict(facecolor='white', alpha=0.7),
             fontsize=16, ha='center')

    # Annotate top few models
    top_models = df.head(4)  
    y_offsets = np.linspace(10, -10, len(top_models))  
    x_offsets = np.linspace(-25, 45, len(top_models))  

    for i, (_, row) in enumerate(top_models.iterrows()):
        plt.annotate(row['Model ID'], (row['rank'], row['Downloads']),
                     textcoords="offset points", xytext=(x_offsets[i], y_offsets[i]),
                     ha='left', fontsize=14, color='blue')

    # Labels and title
    plt.xlabel('Model Downloads Rank', fontsize=26, fontweight='bold')
    plt.ylabel('Downloads', fontsize=26, fontweight='bold')

    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, which='major', linestyle='--', alpha=0.4)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)

    # Make box boundary black and thicker
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.show()

# Example usage:
plot_nlp_model_download_distribution()
