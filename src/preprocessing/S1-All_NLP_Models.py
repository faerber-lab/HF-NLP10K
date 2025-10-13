from huggingface_hub import HfApi
import pandas as pd

# Initialize API client
hf_api = HfApi()

# Define the desired categories
categories = [
    "text-classification", "token-classification", "table-question-answering",
    "question-answering", "zero-shot-classification", "translation",
    "summarization", "feature-extraction", "text-generation",
    "text2text-generation", "fill-mask", "sentence-similarity"
]

# Fetch models matching the tags
all_models = []
for category in categories:
    models = hf_api.list_models(filter=category)
    for model in models:
        all_models.append({
            "Model ID": model.modelId,
            "Category": category,
            "Author": model.author,
            "Downloads": model.downloads,
            "Likes": model.likes,
            "Pipeline": model.pipeline_tag,
            "Library": model.library_name,
        })

# Convert to DataFrame
df = pd.DataFrame(all_models)

# Save to CSV
csv_file_path = "S1-All_NLP.csv"
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved at: {csv_file_path}")
