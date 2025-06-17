import pandas as pd
import requests
import time  # Ensure time is imported for sleep function

# Load the list of models from the new dataset
input_file = "S2-Filtered_NLP_Models.csv"  # Updated dataset name
df_models = pd.read_csv(input_file)

# Ensure column name is correct
if "Model ID" not in df_models.columns:
    print("Error: The input file must have a column named 'Model ID'.")
    exit()

# Extract model names from CSV
model_list = df_models["Model ID"].dropna().tolist()

# List to store extracted data
data = []
skipped_models = []

# Fetch metadata for each model
for index, model_id in enumerate(model_list):
    print(f"Fetching metadata for model: {model_id} ({index+1}/{len(model_list)})")
    
    # API endpoint for model metadata
    metadata_url = f"https://huggingface.co/api/models/{model_id}"
    meta_response = requests.get(metadata_url)

    model_data = {"Model ID": model_id}  # Add model name to metadata
    if meta_response.status_code == 200:
        model_data.update(meta_response.json())
    else:
        print(f"⚠️ Failed to fetch metadata for model: {model_id}")
        skipped_models.append(model_id)
    
    # Fetch the full model card (README.md)
    readme_content = "Failed to fetch README"
    readme_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    readme_response = requests.get(readme_url)
    
    if readme_response.status_code == 200:
        readme_content = readme_response.text
    
    model_data["readme_content"] = readme_content  # Store README content
    data.append(model_data)
    
    # Avoid rate limits by adding a short delay
    time.sleep(0.2)

# Convert to DataFrame
df_output = pd.DataFrame(data)

# Move 'readme_content' column to the last position
columns = [col for col in df_output.columns if col != "readme_content"] + ["readme_content"]
df_output = df_output[columns]

# Save to CSV
output_file = "huggingface_models_metadata.csv"  # Updated output filename
df_output.to_csv(output_file, index=False)

# Save skipped models for debugging
skipped_csv_filename = "S3-Filtered_NLP_DataAPI.csv"
pd.DataFrame({"Skipped Models": skipped_models}).to_csv(skipped_csv_filename, index=False)

print(f"\n✅ Saved metadata (including README content) of {len(df_output)} models to {output_file}")
print(f"⚠️ Skipped models count: {len(skipped_models)} (Saved to {skipped_csv_filename})")
