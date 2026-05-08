import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# FILE PATHS - CHANGE THESE IF NEEDED
# ==========================================
STAGE_1_LOG = "guess_log.txt"
STAGE_2_LOG = "guess_log2.txt"        # Add .txt if your file has it!
STAGE_1_CSV = "metrics.csv"
STAGE_2_CSV = "metrics2.csv"          # Add .csv if your file has it!

def analyze_guess_log(filepath, stage_name):
    distances = []
    if not os.path.exists(filepath):
        print(f"Warning: Could not find {filepath}")
        return None

    # Parse the text file
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # We only want the top guess (1.)
            if line.strip().startswith('1.') and 'km' in line:
                try:
                    # Split by '|', grab the middle part, remove 'km' and whitespace
                    dist_str = line.split('|')[1].replace('km', '').strip()
                    distances.append(float(dist_str))
                except:
                    continue
                    
    if not distances:
        print(f"No valid distances found in {filepath}")
        return None
        
    distances = np.array(distances)
    
    # Calculate all the stats you asked for
    stats = {
        "count": len(distances),
        "closest": distances.min(),
        "average": distances.mean(),
        "median": np.median(distances),
        "under_1k_count": np.sum(distances < 1000),
        "acc_750": np.mean(distances <= 750) * 100,
        "acc_2500": np.mean(distances <= 2500) * 100
    }
    
    # Print the Terminal Report
    print(f"\n{'='*40}")
    print(f" {stage_name} TERMINAL REPORT")
    print(f"{'='*40}")
    print(f"Total Images Graded : {stats['count']}")
    print(f"Closest Guess       : {stats['closest']:.2f} km")
    print(f"Average Distance    : {stats['average']:.2f} km")
    print(f"Median Distance     : {stats['median']:.2f} km")
    print(f"Guesses < 1,000 km  : {stats['under_1k_count']} images")
    print(f"Country Acc (750km) : {stats['acc_750']:.2f}%")
    print(f"Continent Acc(2.5k) : {stats['acc_2500']:.2f}%")
    
    return stats

def plot_loss_curves():
    if not os.path.exists(STAGE_1_CSV) or not os.path.exists(STAGE_2_CSV):
        print("\nMissing CSV files. Skipping loss curves.")
        return

    print("\nGenerating Loss Curve Graph...")
    df1 = pd.read_csv(STAGE_1_CSV)[['step', 's2_loss']].ffill().dropna()
    df2 = pd.read_csv(STAGE_2_CSV)[['step', 's2_loss']].ffill().dropna()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df1['step'], df1['s2_loss'], color='tab:red', linewidth=2, label='Stage 1 (Biased)')
    ax.plot(df2['step'], df2['s2_loss'], color='tab:blue', linewidth=2.5, label='Stage 2 (ImplicitGeo)')

    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Entropy (S2 Loss)', fontsize=12, fontweight='bold')
    ax.set_title('Training Dynamics: Bias Removal & Convergence', fontsize=14, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    
    plt.savefig("Fig1_Loss_Curves.png", dpi=300)
    print(" -> Saved 'Fig1_Loss_Curves.png'")

def plot_accuracy_bar(stats1, stats2):
    if not stats1 or not stats2:
        return
        
    print("Generating Accuracy Bar Chart...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6))

    labels = ['Median Error (km)', 'Country Acc (750km) %', 'Continent Acc (2500km) %']
    s1_data = [stats1['median'], stats1['acc_750'], stats1['acc_2500']]
    s2_data = [stats2['median'], stats2['acc_750'], stats2['acc_2500']]

    x = np.arange(len(labels)) 
    width = 0.35  

    ax.set_yscale('log') # Log scale because km are huge and percentages are small
    ax.bar(x - width/2, s1_data, width, label='Stage 1 (Baseline)', color='tab:red')
    ax.bar(x + width/2, s2_data, width, label='Stage 2 (ImplicitGeo)', color='tab:blue')

    ax.set_ylabel('Value (Log Scale)', fontweight='bold', fontsize=12)
    ax.set_title('Geographical Accuracy: Baseline vs. Optimized', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.legend()
    fig.tight_layout()

    plt.savefig("Fig2_Accuracy_Comparison.png", dpi=300)
    print(" -> Saved 'Fig2_Accuracy_Comparison.png'")

if __name__ == "__main__":
    # 1. Parse text logs and print reports
    s1_stats = analyze_guess_log(STAGE_1_LOG, "STAGE 1 (BASELINE)")
    s2_stats = analyze_guess_log(STAGE_2_LOG, "STAGE 2 (IMPLICITGEO)")
    
    # 2. Generate the visual graphs for the paper
    plot_loss_curves()
    if s1_stats and s2_stats:
        plot_accuracy_bar(s1_stats, s2_stats)
        
    print("\nAll analysis complete! Check your folder for the new PNG graphs.")