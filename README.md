# HF-NLP10K: A Dataset and Metadata Analysis of 10,000+ Hugging Face NLP Models

This repository contains the data, scripts, and figures accompanying the paper:

*HF-NLP10K: A Dataset and Metadata Analysis of 10,000+ Hugging Face NLP Models*  
(Under submission)

The first version (v1) of the dataset and the pipeline scripts is archived on Zenodo: https://zenodo.org/records/15682522  
The dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence. The underlying model cards remain under their authors' licences.

---

## 🌍 Overview

Hugging Face has become a central platform for sharing NLP models. However, inconsistent and incomplete metadata make it difficult to find suitable models.  
**HF-NLP10K** introduces a structured dataset of over **10,000 NLP models** enriched with **33 metadata fields**, combining Hugging Face API data with **LLM-based parsing** of free-text model cards.

---

## 📦 Repository Structure

```
data/
├── raw/                 # Raw extracted data from Hugging Face API
│   ├── Data-Stats.csv.gz
│   └── S1-All_NLP.csv.gz
└── final/               # Final structured dataset
    └── HLT-NLP10K.csv

src/
├── preprocessing/       # Scripts for dataset construction (S1–S8)
├── analysis/            # Exploratory data analysis scripts
└── figures/             # Figure generation scripts (for paper visuals)

outputs/
└── figures/             # Generated figures and plots

requirements.txt          # Python dependencies
README.md                 # Overview and usage instructions
```

---

## 🧪 Dataset Highlights

- Covers **10,022 NLP models** with at least 1,000 downloads
- Includes **33 metadata fields**, such as:
  - Pipeline tag, license, training data, evaluation metrics, base model
  - Download counts, fine-tuning compatibility, model card sections (limitations, use cases, etc.)
- Combines **API enrichment** with **LLM-based model card parsing**

---

## 🔎 Example Query

Find sentence-similarity models with the apache-2.0 license whose use cases mention semantic search:

```python
import pandas as pd

df = pd.read_csv("data/final/HLT-NLP10K.csv")

sim = df[df["Pipeline_tag"] == "sentence-similarity"]        # 425 models
apache = sim[sim["License"] == "apache-2.0"]                 # 207 models
hits = apache[apache["Use Cases"].str.contains("semantic search", case=False, na=False)]  # 112 models
print(hits[["Model ID", "Downloads"]].head())
```

---

## 🔧 Reproducing the Dataset

Run the scripts sequentially from `S1` to `S8`:

```bash
cd src/preprocessing

python S1-All_NLP_Models.py
python S2-Filter_NLP_Models.py
...
python S8-UniqueRows_FinalDataset.py
```

The released dataset (`HLT-NLP10K.csv`, available under `data/final/`) contains the 10,022 models of the S8 output that carry an NLP pipeline tag, with the column names of the released schema.

The dataset can be updated by re-running the pipeline on a newer Hub snapshot. Updates can be released as new versions on Zenodo.

---

## 📊 Figures

The figures presented in the paper can be reproduced using:

```bash
cd src/figures
python Figure1_NLPGrowth.py
python Figure2_No.ofDownloads.py
...
python Figure5_MissingSections.py
```

Resulting images will be stored under `outputs/figures/`.

---

## ⚙️ Requirements

Install dependencies before running the scripts:

```bash
pip install -r requirements.txt
```

---

## 🧩 Citation

If you use this dataset or scripts, please cite:

> *HF-NLP10K: A Dataset and Metadata Analysis of 10,000+ Hugging Face NLP Models. Under submission.*

---
