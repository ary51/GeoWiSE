import sys
import os
import glob
import re
import csv
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from datasets import load_dataset
import json
import s2sphere

# Tell Python to look in the main folder for 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your custom modules
from src.data.transform import Transform
from scripts.train import GeoLightningModel, OSV5MCollator, NUM_CLASSES 

def main():
    print("==================================================")
    print(" Initializing Time-Lapse Test Pipeline...")
    print("==================================================")
    
    # 1. Load Vocab
    vocab_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'vocab.json')
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    s2_to_class = {int(k): v for k, v in vocab["s2_to_class"].items()}
    class_to_s2 = {int(k): int(v) for k, v in vocab["class_to_s2"].items()}

    # 2. Setup Data Pipeline
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    fast_transform = Transform()
    collator = OSV5MCollator(processor, fast_transform, s2_to_class)
    
    print("Connecting to the OFFICIAL OSV-5M Test Stream...")
    # NOTE: Set split="test" for your official final paper numbers. 
    # (Or split="train" if you are still doing the capacity sanity check)
    dataset = load_dataset(
        "osv5m/osv5m", 
        split="test", 
        streaming=True, 
        trust_remote_code=True
    ).take(1000) 
    
    test_loader = DataLoader(
        dataset, 
        batch_size=48, 
        collate_fn=collator, 
        num_workers=1, 
        pin_memory=True
    )
    
    # 3. Rebuild Centroids
    centroids = torch.zeros((NUM_CLASSES, 2))
    for class_id, s2_str in class_to_s2.items():
        class_id = int(class_id)
        cell = s2sphere.CellId(int(s2_str))
        lat_lng = cell.to_lat_lng()
        centroids[class_id, 0] = lat_lng.lat().degrees
        centroids[class_id, 1] = lat_lng.lng().degrees

    # 4. Find and Sort All Checkpoints
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_stage2')
    # Grab all checkpoints EXCEPT last.ckpt (to avoid duplicates)
    all_ckpts = glob.glob(os.path.join(ckpt_dir, "geoguessr-stage2-*.ckpt"))
    
    if not all_ckpts:
        print("ERROR: No checkpoints found in checkpoints_stage2/")
        return

    # Sort them mathematically by the Step Number in the filename
    def extract_step(filepath):
        match = re.search(r'step=(\d+)', filepath)
        return int(match.group(1)) if match else 0
        
    all_ckpts.sort(key=extract_step)
    print(f"Found {len(all_ckpts)} checkpoints. Beginning chronological evaluation...")

    # 5. Prepare the CSV File
    csv_file = os.path.join(os.path.dirname(__file__), '..', 'test_evolution_metrics.csv')
    
    # Write the headers
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Step", 
            "Median_Dist_km", 
            "Mean_Dist_km", 
            "Acc_25km_City", 
            "Acc_200km_Region", 
            "Acc_750km_Country", 
            "Acc_2500km_Continent"
        ])

    # 6. The Time-Lapse Loop
    for ckpt_path in all_ckpts:
        step_num = extract_step(ckpt_path)
        print(f"\n---> Evaluating Checkpoint at Step {step_num}...")
        
        # Load the specific checkpoint
        model = GeoLightningModel.load_from_checkpoint(ckpt_path, class_centroids=centroids, strict=False)
        
        # logger=False prevents PyTorch from creating a million messy logs during evaluation
        trainer = pl.Trainer(accelerator="gpu", devices=1, precision="16-mixed", logger=False)
        
        # Run the test
        metrics_list = trainer.test(model, test_loader)
        metrics = metrics_list[0] # Grab the dictionary of results
        
        # Extract the exact numbers needed for your paper
        median_dist = metrics.get("test_median_dist", 0.0)
        mean_dist = metrics.get("test_mean_dist", 0.0)
        acc_25 = metrics.get("acc_25km", 0.0) * 100    # Convert to %
        acc_200 = metrics.get("acc_200km", 0.0) * 100  # Convert to %
        acc_750 = metrics.get("acc_750km", 0.0) * 100  # Convert to %
        acc_2500 = metrics.get("acc_2500km", 0.0) * 100 # Convert to %
        
        print(f"[Results Step {step_num}] Median: {median_dist:.1f}km | Country Acc: {acc_750:.1f}% | Continent Acc: {acc_2500:.1f}%")

        # Append to the CSV in real-time
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                step_num, 
                round(median_dist, 2), 
                round(mean_dist, 2), 
                round(acc_25, 2), 
                round(acc_200, 2), 
                round(acc_750, 2), 
                round(acc_2500, 2)
            ])
            
    print("\n==================================================")
    print(f" Evaluation Complete! Results saved to: {csv_file}")
    print(" Drop this CSV into Excel/Python to generate your paper's graphs!")
    print("==================================================")

if __name__ == "__main__":
    main()