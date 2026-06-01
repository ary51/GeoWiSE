# GeoWiSE: Geo-Localization Without Segmentation Explicitly

**Authors:** Yash Desai & Aarya Dalal (Rutgers University)  
**Course:** CS 439 Final Project

## Overview
Continental visual geolocalization attempts to determine the exact coordinates of a location given only an image. Traditional SOTA models rely on explicit semantic segmentation masks, creating a massive computational bottleneck. 

**GeoWiSE** challenges this by utilizing a dual-backbone architecture that fuses foundation models (**SigLIP** and **DINOv3**) without any intermediate segmentation step. By leaning on the structural self-attention maps of DINOv3 for depth/boundaries and the text-aligned embeddings of SigLIP for macro-environmental clues, GeoWiSE predicts locations across a 50,000-class Level-12 S2 topological grid.

## Repository Structure
Our codebase is modularized into data handling, model architecture, execution scripts, and generated outputs:

* **`data/`**: Contains the raw metric logs and outputs from our evaluation runs (`metrics.csv`, `guess_log.txt`, etc.).
* **`figures/`**: Contains the compiled visualizations used in our final paper (e.g., `Attention_Heatmap.png`, `Fig1_Loss_Curves.png`).
* **`models/`** (or `src/models/`): Contains the PyTorch architecture definitions, including our MLP Fusion Head (`head.py`).
* **`scripts/`**: Executable scripts for training and evaluation.
  * `train.py`: Main training loop for the dual-backbone model.
  * `test.py` / `best_guesses.py`: Scripts to evaluate the model on the holdout test set and calculate Haversine distances.
  * `generate_heatmap.py` / `generate_results.py`: Scripts used to dynamically generate the attention maps and loss curves from our paper.
* **`src/data/`**: Core data processing utilities.
  * `geometry.py`: Maps raw continuous GPS coordinates to Level-12 S2 cell IDs.
  * `transform.py`: Handles the Gaussian blur and localized crop generation for the 3 distinct image views.
  * `vocab.json` / `vocab.py`: Tokenization and vocabulary handling for the synthetic text captions.
* **`requirements.txt`**: Standardized list of Python dependencies.

## Requirements & Installation
All models were trained and evaluated utilizing a singular shared NVIDIA A4500 GPU. 

1. Clone this repository:
   ```bash
   git clone https://https://github.com/ary51/GeoWiSE
   cd GeoWiSE
2. Create a virtual environment (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Requirements:
   ```bash
   pip install -r requirements.txt

## Reproducing the Pipeline

4. Training the model
   ```bash
   python scripts/train.py

  Note: model weights will save to checkpoint automatically every 50 steps (run cron job to ensure rerun after crashes)
  
5. Evaluate
   ```bash
   python scripts/test.py   
