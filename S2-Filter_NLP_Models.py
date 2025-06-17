import pandas as pd

# Load the original CSV file
df = pd.read_csv("S1-All_NLP.csv")

# Filter models with at least 5000 downloads
df_filtered = df[df["Downloads"] >= 1000]

# Save to a new CSV file
filtered_csv_file_path = "S2-Filtered_NLP_Models.csv"
df_filtered.to_csv(filtered_csv_file_path, index=False)

print(f"Filtered CSV file saved at: {filtered_csv_file_path}")
