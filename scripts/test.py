import sys
import os
import csv
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from datasets import load_dataset
import json
import s2sphere

# Fix path to find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.transform import Transform
from scripts.train import GeoLightningModel, OSV5MCollator, NUM_CLASSES 

def main():
    # 1. Load Vocab
    vocab_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'vocab.json')
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    s2_to_class = {int(k): v for k, v in vocab["s2_to_class"].items()}
    class_to_s2 = {int(k): int(v) for k, v in vocab["class_to_s2"].items()}

    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    fast_transform = Transform()
    collator = OSV5MCollator(processor, fast_transform, s2_to_class)

    dataset = load_dataset(
        "osv5m/osv5m", 
        split="test", 
        streaming=True,
        trust_remote_code=True
    ).take(5000)
    
    
    
    # num_workers=4 speeds up local loading. Change to 0 if your laptop crashes.
    test_loader = DataLoader(dataset, batch_size=48, collate_fn=collator, num_workers=4, pin_memory=True)
    
    centroids = torch.zeros((NUM_CLASSES, 2))
    for class_id, s2_str in class_to_s2.items():
        class_id = int(class_id)
        cell = s2sphere.CellId(int(s2_str))
        lat_lng = cell.to_lat_lng()
        centroids[class_id, 0] = lat_lng.lat().degrees
        centroids[class_id, 1] = lat_lng.lng().degrees

    # "auto" finds your local GPU, Apple Silicon, or CPU
    trainer = pl.Trainer(accelerator="auto", devices=1, precision="16-mixed", logger=False)

    csv_file = os.path.join(os.path.dirname(__file__), '..', 'final_paper_results.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model_Version", "Median_Dist_km", "Acc_25km", "Acc_200km", "Acc_750km", "Acc_2500km"])

    stage1_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'stage1_baseline.ckpt')
    if os.path.exists(stage1_path):
        print("\n---> Grading Stage 1 (Before Bias Ablation)...")
        model_s1 = GeoLightningModel.load_from_checkpoint(stage1_path, class_centroids=centroids, strict=False)
        metrics_s1 = trainer.test(model_s1, test_loader)[0]
        
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Stage 1 (Baseline)", 
                round(metrics_s1.get("test_median_dist", 0), 2),
                round(metrics_s1.get("acc_25km", 0) * 100, 2),
                round(metrics_s1.get("acc_200km", 0) * 100, 2),
                round(metrics_s1.get("acc_750km", 0) * 100, 2),
                round(metrics_s1.get("acc_2500km", 0) * 100, 2)
            ])
    else:
        print(f"Skipping Stage 1: Could not find {stage1_path}")

if __name__ == "__main__":
    main()