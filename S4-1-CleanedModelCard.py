import re
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# Function to clean model card text properly
def clean_model_card(text):
    """Cleans the model card text by removing code blocks, markdown, HTML elements, and unnecessary details."""
    if pd.isna(text):  # Handle missing values
        return ""

    # Remove <details> sections (including multi-line content)
    text = re.sub(r"<details>[\s\S]*?</details>", "", text, flags=re.DOTALL)

    # Remove code blocks (triple backticks)
    text = re.sub(r"```[\s\S]*?```", "", text, flags=re.MULTILINE)

    # Remove inline code (single backticks)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove markdown images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove markdown links
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)

    # Remove markdown headings
    text = re.sub(r"#+\s+", "", text)

    # Remove front matter (YAML metadata)
    text = re.sub(r"^---[\s\S]*?---", "", text, flags=re.DOTALL)

    # Remove HTML tags while keeping text
    text = BeautifulSoup(text, "html.parser").get_text()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Load CSV file
csv_file = "S3-Filtered_NLP_DataAPI.csv"

if Path(csv_file).exists():
    df = pd.read_csv(csv_file)

    # Ensure 'model_card' column exists
    if "readme_content" in df.columns:
        # Apply cleaning function to 'model_card' column
        df["Cleaned Model Card"] = df["readme_content"].apply(clean_model_card)

        # Save updated DataFrame back to CSV
        df.to_csv(csv_file, index=False)

        print("Cleaned Model Card column added successfully!")
    else:
        print("Error: 'model_card' column not found in CSV.")
else:
    print(f"Error: File '{csv_file}' not found!")
