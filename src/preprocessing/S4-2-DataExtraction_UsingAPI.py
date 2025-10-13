import pandas as pd
import json
import ast  # Safe literal evaluation
import re  # Regex for extracting two-letter language codes

# Function to extract 'metrics' details from 'model-index' column
def extract_metrics(model_index):
    if isinstance(model_index, str):
        try:
            model_index_data = ast.literal_eval(model_index) if "[" in model_index else []
            for entry in model_index_data:
                if "results" in entry and isinstance(entry["results"], list):
                    for metric in entry["results"]:
                        if "metrics" in metric:
                            return json.dumps(metric["metrics"])  # Convert to JSON string
            return "none"
        except (json.JSONDecodeError, ValueError, SyntaxError):
            return "error"
    return "none"

# Function to extract tensor type and model size from safetensors or gguf
def extract_model_size(row):
    safetensors = row.get("safetensors", "")
    gguf = row.get("gguf", "")

    if isinstance(safetensors, str) and "parameters" in safetensors:
        try:
            safetensor_data = ast.literal_eval(safetensors)
            parameters = safetensor_data.get("parameters", {})
            tensor_types = ", ".join(parameters.keys())  # Extract tensor types like BF16, F32
            model_size = safetensor_data.get("total", 0) / 1e9  # Convert to Billions (B)
            return tensor_types, round(model_size, 2)
        except (ValueError, SyntaxError):
            return "not applicable", "unknown"

    elif isinstance(gguf, str) and "total" in gguf:
        try:
            gguf_data = ast.literal_eval(gguf)
            model_size = gguf_data.get("total", 0) / 1e9  # Convert to Billions (B)
            return "not applicable", round(model_size, 2)
        except (ValueError, SyntaxError):
            return "not applicable", "unknown"

    return "not applicable", "unknown"

# Function to check if a tag is present
def check_tag(tags, keyword):
    if isinstance(tags, str):
        return "yes" if keyword in tags else "no"
    return "no"

non_language_tags = {'tf', 'pt'}  # 'tf' for TensorFlow, 'pt' for PyTorch

def extract_languages(tags):
    if isinstance(tags, str):
        try:
            tag_list = eval(tags)  # Convert string to list safely
            languages = [
                t for t in tag_list
                if re.fullmatch(r"[a-z]{2}", t) and t not in non_language_tags
            ]  # Match exactly two lowercase letters and exclude non-language tags
            return ", ".join(languages) if languages else "unknown"
        except (SyntaxError, ValueError):
            return "unknown"
    return "unknown"

# Function to count number of detected two-letter language codes
def count_languages(tags):
    if isinstance(tags, str):
        try:
            tag_list = eval(tags)  # Convert string to list safely
            languages = [t for t in tag_list if re.fullmatch(r"[a-z]{2}", t)]
            return len(languages)  # Count number of two-letter codes
        except (SyntaxError, ValueError):
            return 0
    return 0

# Function to extract arXiv ID from tags
def extract_arxiv(tags):
    if isinstance(tags, str):
        try:
            tag_list = eval(tags)  # Convert string to list safely
            arxiv_id = next((t.split(":")[1] for t in tag_list if t.startswith("arxiv:")), "none")
            return arxiv_id
        except (SyntaxError, ValueError):
            return "none"
    return "none"

# Load dataset
input_file = "S3-Filtered_NLP_DataAPI.csv"
df = pd.read_csv(input_file, dtype=str)

# Extract relevant columns and process
filtered_df = pd.DataFrame()

filtered_df["modelId"] = df["id"]
filtered_df["private"] = df["private"]
filtered_df["pipeline_tag"] = df["pipeline_tag"]
filtered_df["library_name"] = df["library_name"]
filtered_df["license"] = df["tags"].apply(lambda x: next((t.split(":")[1] for t in eval(x) if t.startswith("license:")), "unknown"))
filtered_df["region"] = df["tags"].apply(lambda x: next((t.split(":")[1] for t in eval(x) if t.startswith("region:")), "unknown"))
filtered_df["auto_train_compatible"] = df["tags"].apply(lambda x: check_tag(x, "autotrain_compatible"))
filtered_df["language"] = df["tags"].apply(extract_languages)
filtered_df["language_count"] = df["tags"].apply(count_languages)
filtered_df["endpoints_deployed"] = df["tags"].apply(lambda x: check_tag(x, "endpoints_compatible"))
filtered_df["arxiv"] = df["tags"].apply(extract_arxiv)
filtered_df["metrics"] = df["model-index"].apply(extract_metrics)
filtered_df["downloads"] = df["downloads"]
filtered_df["likes"] = df["likes"]
filtered_df["author"] = df["author"]
filtered_df["lastModified"] = df["lastModified"]
filtered_df["gated"] = df["gated"]
filtered_df["disabled"] = df["disabled"]
filtered_df["spaces"] = df["spaces"]
filtered_df["spaces_count"] = df["spaces"].apply(lambda x: len(eval(x)) if isinstance(x, str) and x.startswith("[") else 0)
filtered_df["createdAt"] = df["createdAt"]

# Extract tensor type and model size from safetensors or gguf
filtered_df["tensor_type"], filtered_df["model_size (B)"] = zip(*df.apply(extract_model_size, axis=1))

filtered_df["inference"] = df["inference"]
filtered_df["usedStorage"] = df["usedStorage"]

# Save to CSV
output_file = "Step4-2.csv"
filtered_df.to_csv(output_file, index=False)

print(f"✅ Process completed! Filtered data saved to {output_file}")
