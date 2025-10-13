import pandas as pd
import re
import json

def extract_json_from_text(text):
    """Extracts JSON content from text enclosed in triple backticks."""
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1))  # Extract and parse JSON
        except json.JSONDecodeError:
            return None  # Invalid JSON, fallback needed
    return None  # No JSON found

def process_csv(input_file, output_file):
    """Processes CSV to extract JSON from Raw Output and update missing fields."""
    df = pd.read_csv(input_file)
    
    for index, row in df.iterrows():
        if isinstance(row.get("Error"), str) and "Invalid response format" in row["Error"]:
            raw_output = str(row.get("Raw Output", ""))  # Ensure it's a string

            # Extract JSON if available
            extracted_json = extract_json_from_text(raw_output)

            if extracted_json:
                # Override only fields that are "None explicitly mentioned."
                for key in extracted_json.keys():
                    if key in df.columns and row.get(key) == "None explicitly mentioned.":  
                        df.at[index, key] = extracted_json[key]
    
    df.to_csv(output_file, index=False)
    print(f"Processed file saved as {output_file}")

# Example usage
process_csv("S5-DeepSeek-UnprocessedData.csv", "S6-DeepSeek-Dataprocessing.csv")
