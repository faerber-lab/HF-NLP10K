import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def plot_llm_growth(csv_file='Data-Stats.csv.gz.csv', save_file='Picture1.pdf'):
    # Load data
    df = pd.read_csv(csv_file, usecols=['created_at'])
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna()
    df = df.sort_values('created_at')
    
    date_range = pd.date_range(start=df['created_at'].min(), end=df['created_at'].max(), freq='D')
    cumulative_counts = df.groupby('created_at').size().cumsum()
    cumulative_counts = cumulative_counts.reindex(date_range, method='ffill')
    
    x = np.arange(len(cumulative_counts))
    y = cumulative_counts.values
    log_y = np.log(y + 1)
    b, log_a = np.polyfit(x, log_y, 1)
    a = np.exp(log_a)
    weekly_growth_rate = (np.exp(b * 7) - 1) * 100
    y_fit = a * np.exp(b * x)
    
    plt.figure(figsize=(10, 6))
    plt.bar(date_range, cumulative_counts, color='lightblue', label='Total NLP models')
    plt.plot(date_range, y_fit, 'b--', label='Exponential Fit', linewidth=2)
    
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Total Number of NLP models', fontsize=18)
    plt.title('Growth of NLP models Over Time', fontsize=18)
    plt.legend(fontsize=18)
    
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.text(date_range[int(len(date_range) * 0.6)], max(y_fit) * 0.5,
             f'Weekly growth rate: {weekly_growth_rate:.2f}%',
             bbox=dict(facecolor='white', alpha=0.7),
             fontsize=18)
    
    plt.subplots_adjust(bottom=0.25)  # <-- KEY: avoid cutting off x-labels
    plt.savefig(save_file, bbox_inches='tight', format='pdf')
    plt.close()

# Usage:
plot_llm_growth()
