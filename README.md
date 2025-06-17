# HF-NLP10K: A Dataset and Metadata Analysis of 10,000+ Hugging Face NLP Models

This repository contains the data, scripts, and figures accompanying the paper:

**Batool, S. K., Jatowt, A., & Färber, M.**  
*HF-NLP10K: A Dataset and Metadata Analysis of 10,000+ Hugging Face NLP Models*  
(Under submission)

---

## 🌍 Overview

Hugging Face has become a central platform for sharing NLP models. However, inconsistent and incomplete metadata makes it difficult to find suitable models. In this project, we introduce **HF-NLP10K**, a structured dataset of over **10,000 NLP models** enriched with **33 metadata fields**, combining Hugging Face API data with LLM-based parsing of free-text model cards.

This repository includes:

- 📦 The HF-NLP10K dataset (`CMA-PROJ_Dataset.csv.gz`)
- 🛠 Scripts for generating the initial dataset (`InitialDataAnalysis_1.py`, `InitialDataAnalysis_2.py`)
- 📂 Initial extracted dataset (`Data-Stats.csv.gz`)
- 📜 Scripts for dataset creation (steps S1-S8)
- 📊 Python scripts for generating figures used in the paper

---

## 📁 Contents

| File | Description |
|------|-------------|
| `CMA-PROJ_Dataset.csv.gz` | Final structured dataset of 10,022 NLP models |
| `S1-All_NLP.csv.gz` | Raw model metadata from Hugging Face API |
| `InitialDataAnalysis_1.py`, `InitialDataAnalysis_2.py`, `Data-Stats.csv.gz` | Initial extracted data files used for reproducibility and exploratory analysis |
| `S1-All_NLP_Models.py` to `S8-UniqueRows_FinalDataset.py` | Scripts for dataset construction pipeline |
| `Figure1_NLPGrowth.py` to `Figure5_MissingSections.py` | Scripts to generate figures from the paper |
| `README.md` | Overview and usage instructions |


---

## 🧪 Dataset Highlights

- Covers **10,022** NLP models with at least 1,000 downloads
- Extracted **33 metadata fields**, including:
  - Pipeline tag, license, training data, evaluation metrics, base model
  - Download counts, fine-tuning compatibility, model card sections (limitations, use cases, etc.)
- Built using API enrichment and **LLM-based model card parsing**

---

## 🔧 Reproducing the Dataset

Run the scripts in order from `S1` to `S8`:

```bash
python S1-All_NLP_Models.py
python S2-Filter_NLP_Models.py
...
python S8-UniqueRows_FinalDataset.py
