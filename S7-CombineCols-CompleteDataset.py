import pandas as pd

# Load the two datasets
fixed_details = pd.read_csv("S6-DeepSeek-Dataprocessing.csv", dtype=str)
api_data = pd.read_csv("step4again.csv", dtype=str)

# Select only the necessary columns from fixed_model_details
cols_to_add = ["Limitations", "Use Cases", "Specialization", "Training Data", 
               "Comparison", "Evaluation Metrics", "Base Model", "Fine-Tuning"]

# Ensure Model ID columns are stripped of whitespace and normalized
fixed_details["Model ID"] = fixed_details["Model ID"].str.strip()
api_data["modelId"] = api_data["modelId"].str.strip()

# Keep fixed_model_details.csv as the standard, ensuring unique Model IDs
fixed_details = fixed_details.drop_duplicates(subset="Model ID", keep="first")
fixed_details = fixed_details[["Model ID"] + cols_to_add]

# Merge using Model ID from fixed_model_details as the standard reference
merged_data = fixed_details.merge(api_data, left_on="Model ID", right_on="modelId", how="left")

# Drop the redundant "modelId" column after merging
merged_data.drop(columns=["modelId"], inplace=True)

# Save the merged dataset
merged_data.to_csv("S7-CombineCols-CompleteDataset.csv", index=False)

print("Merging complete. The file has been saved as 'S7-CombineCols-CompleteDataset'.")
