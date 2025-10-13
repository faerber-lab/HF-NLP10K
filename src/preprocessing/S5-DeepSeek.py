import pandas as pd
import requests
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(filename="scadsai_api_errors.log", level=logging.ERROR, format="%(asctime)s - %(message)s")

# Load ScaDS.AI API key
with open("my_key", "r") as f:
    API_KEY = f.read().strip()

# API endpoint & headers
url = "https://llm.scads.ai/v1/completions"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Load the CSV file
csv_file_path = 'S3-Filtered_NLP_DataAPI.csv'
df = pd.read_csv(csv_file_path)

description_column = "Cleaned Model Card"
id_column = "id"

if description_column not in df.columns or id_column not in df.columns:
    raise ValueError(f"Columns '{description_column}' and '{id_column}' not found in the CSV file.")

# Function to extract details using ScaDS.AI API with retries and backoff
def query_scadsai_llm(model_id, prompt, max_retries=3, base_delay=5):
    """ Queries the ScaDS.AI API with retries and exponential backoff. """
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.05,
        "top_p": 0.7
    }

    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)  # Increased timeout
            
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    text_output = choices[0].get("text", "").strip()
                    
                    # **Clean response: Remove unwanted tags (e.g., </think>)**
                    cleaned_text = re.sub(r"</?\w+>", "", text_output).strip()
                    
                    # **Ensure response is valid JSON**
                    try:
                        json_output = json.loads(cleaned_text)
                        return model_id, json_output
                    except json.JSONDecodeError:
                        logging.error(f"Invalid JSON response: {cleaned_text}")
                        return model_id, {"Error": "Invalid response format", "Raw Output": cleaned_text}

                return model_id, {"Error": "No response text"}
            else:
                logging.error(f"API Error: {response.status_code} - {response.text}")
                return model_id, {"Error": f"API Error: {response.status_code}", "Raw Output": response.text}

        except (RequestException, requests.exceptions.ReadTimeout) as e:
            logging.error(f"Request Exception (Attempt {attempt+1}): {str(e)}")
            time.sleep(base_delay * (2 ** attempt))  # Exponential backoff
            attempt += 1

    return model_id, {"Error": "Max retries exceeded", "Raw Output": "Request timed out after multiple attempts"}

# Define the structured prompt
def generate_prompt(description):
    return f"""Extract the following details from the provided model description. Provide **ONLY factual and explicit** answers. Do **NOT** infer missing details.

1. **Limitations/Challenges/Shortcomings/Disadvantages**: List **only** the explicitly stated weaknesses of the model.
2. **Intended / Primary Use Cases**: Provide the **main applications** the model is designed for.
3. **Model Specialization**: Explain **what makes this model unique** and what it is optimized for.
4. **Training Data**: Mention **the type and source of training data** used for this model.
5. **Comparison with Other Models**: If comparisons are explicitly mentioned, list them.
6. **Evaluation Metrics (Accuracy/Precision/Performance/Correctness Results)**: List any explicitly mentioned evaluation results.
7. **Base Model**: If this model is based on another pre-trained model, specify its name.
8. **Fine-Tuning Information**: Provide details about the fine-tuning process if explicitly mentioned.

**Response Format (Strict JSON Only)**:
{{
  "Limitations": "...",
  "Use Cases": "...",
  "Specialization": "...",
  "Training Data": "...",
  "Comparison": "...",
  "Evaluation Metrics": "...",
  "Base Model": "...",
  "Fine-Tuning": "..."
}}

If a detail is **not explicitly mentioned**, respond with `"None explicitly mentioned."`

Model description:
{description}
"""

# ✅ Process **ALL** rows in parallel with controlled concurrency
df_subset = df[[id_column, description_column]].dropna()

# List to store extracted results
details_list = []

# Use ThreadPoolExecutor for parallel API calls
MAX_WORKERS = 5  # Reduced to avoid overwhelming API

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_model = {
        executor.submit(query_scadsai_llm, row[id_column], generate_prompt(row[description_column])): row[id_column]
        for _, row in df_subset.iterrows()
    }
    
    for future in as_completed(future_to_model):
        model_id, response_data = future.result()
        
        complete_data = {
            "Model ID": model_id,
            "Limitations": response_data.get("Limitations", "None explicitly mentioned."),
            "Use Cases": response_data.get("Use Cases", "None explicitly mentioned."),
            "Specialization": response_data.get("Specialization", "None explicitly mentioned."),
            "Training Data": response_data.get("Training Data", "None explicitly mentioned."),
            "Comparison": response_data.get("Comparison", "None explicitly mentioned."),
            "Evaluation Metrics": response_data.get("Evaluation Metrics", "None explicitly mentioned."),
            "Base Model": response_data.get("Base Model", "None explicitly mentioned."),
            "Fine-Tuning": response_data.get("Fine-Tuning", "None explicitly mentioned."),
            "Error": response_data.get("Error", "None"),
            "Raw Output": response_data.get("Raw Output", "None")
        }
        
        details_list.append(complete_data)

# Convert extracted details to DataFrame
extracted_df = pd.DataFrame(details_list)

# Save results to a CSV file
output_file = "S5-DeepSeek-UnprocessedData.csv"
extracted_df.to_csv(output_file, index=False)

# Display results
print(extracted_df.head())
print(f"Extracted details saved to {output_file}")
