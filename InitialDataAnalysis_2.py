import json
import csv

# Define the input JSON files and output file paths
input_json_files = [
    "Data-Stats_1.json",
    "Data-Stats_2.json",
    "Data-Stats_3.json",
    "Data-Stats_4.json"
]
output_csv_file = "Data-Stats.csv"

# Define the fields to extract
fields_to_extract = [
    "id", "author", "last_modified", "created_at", "private",
    "gated", "downloads", "likes", "library_name", "pipeline_tag",
    "trendingScore"
]

def process_json_files(input_files, output_file):
    extracted_data = []
    
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as file:
            data = json.load(file)  # Load JSON data
        
        # Extract relevant fields
        for model in data:
            extracted_entry = {field: model.get(field, None) for field in fields_to_extract}
            extracted_entry["Model card"] = "Yes" if model.get("model_card") else "No"
            extracted_data.append(extracted_entry)
    
    # Write data to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields_to_extract + ["Model card"])
        writer.writeheader()
        writer.writerows(extracted_data)
    
    print(f"Extracted data saved to {output_file}")

# Run the function
process_json_files(input_json_files, output_csv_file)
