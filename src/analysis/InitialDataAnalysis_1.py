import json
import requests
from huggingface_hub import list_models
from datetime import datetime

def fetch_and_store_model_metadata(total_models, chunk_size, output_prefix="Data-Stats"):
    # Define the categories (pipeline tags) to filter models
    categories = [
        "text-classification",
        "token-classification",
        "table-question-answering",
        "question-answering",
        "zero-shot-classification",
        "translation",
        "summarization",
        "feature-extraction",
        "text-generation",
        "text2text-generation",
        "fill-mask",
        "sentence-similarity"
    ]
    
    # Fetch metadata for all models at once.
    models_info = list(list_models(limit=total_models, full=True))
    
    # Filter models based on the specified categories
    filtered_models = [model for model in models_info if model.pipeline_tag in categories]
    
    # We will split 'filtered_models' into chunks of size 'chunk_size'
    num_chunks = len(filtered_models) // chunk_size + (1 if len(filtered_models) % chunk_size != 0 else 0)

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min(start_index + chunk_size, len(filtered_models))
        chunk_data = filtered_models[start_index:end_index]

        # Prepare a list of dictionaries to be serialized to JSON
        chunk_dicts = []
        for model in chunk_data:
            # Convert ModelInfo object to dictionary
            model_dict = {attr: getattr(model, attr) for attr in model.__dict__.keys()}
            
            # Convert datetime objects to strings
            for key, value in model_dict.items():
                if isinstance(value, datetime):
                    model_dict[key] = value.isoformat()
                elif isinstance(value, list):
                    # Convert list of RepoSibling objects to list of dicts
                    model_dict[key] = [
                        sibling.__dict__ for sibling in value 
                        if hasattr(sibling, '__dict__')
                    ]
            
            # Fetch the full model card (README.md)
            model_name = model_dict.get("modelId", None) or model_dict.get("id", None)
            if model_name:
                url = f"https://huggingface.co/{model_name}/raw/main/README.md"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        model_card_text = response.text
                        model_dict["model_card"] = model_card_text
                    else:
                        model_dict["model_card"] = None
                except requests.RequestException:
                    model_dict["model_card"] = None
            else:
                model_dict["model_card"] = None
            
            chunk_dicts.append(model_dict)

        output_filename = f"{output_prefix}_{i+1}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)

        print(f"Saved models {start_index+1} to {end_index} in '{output_filename}'")

# Example usage
fetch_and_store_model_metadata(total_models=1374394, chunk_size=100000)
