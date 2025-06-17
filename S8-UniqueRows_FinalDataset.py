import pandas as pd

# Define file paths
input_file = "S7-CombineCols-CompleteDataset.csv"  # Update if needed
output_file = "S8-FinalDataset.csv"  # File to save unique models

try:
    # Load the dataset
    df = pd.read_csv(input_file)

    # Check if "Model ID" exists
    if "Model ID" in df.columns:
        # Drop duplicate Model IDs, keeping the first occurrence
        unique_df = df.drop_duplicates(subset=["Model ID"], keep="first")

        # Save to a new CSV file
        unique_df.to_csv(output_file, index=False)

        print(f"Saved {len(unique_df)} unique rows to {output_file}")
    else:
        print("Error: 'Model ID' column not found in the dataset.")

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
