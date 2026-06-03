import numpy as np
import json
from pathlib import Path

def build_luna_stats(data_dir="data/processed/dataset"):
    stats = {}
    # We search for all .npy files in train and test
    files = list(Path(data_dir).glob("**/*.npy"))
    
    # Extract NAC-ID (the part after the last underscore)
    for p in files:
        nac_id = p.stem.split('_')[-1]
        if nac_id not in stats:
            stats[nac_id] = []
        stats[nac_id].append(np.load(p))

    final_map = {}
    for nac_id, data_list in stats.items():
        # flatten every array to 1D before concatenation
        all_data = np.concatenate([arr.ravel() for arr in data_list])

        p2, p98 = np.percentile(all_data, [2, 98])

        final_map[nac_id] = {
            "min": float(p2),
            "max": float(p98)
        }

        print(f"NAC {nac_id}: P2={p2:.1f}, P98={p98:.1f}")

    with open("data/nac_stats.json", "w") as f:
        json.dump(final_map, f, indent=4)

if __name__ == "__main__":
    build_luna_stats()