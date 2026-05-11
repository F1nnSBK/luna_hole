import os
import shutil
from sklearn.model_selection import GroupShuffleSplit

# Pfade definieren
base_path = "data/processed/dataset"
pits_path = os.path.join(base_path, "pits")
negs_path = os.path.join(base_path, "negatives")

# Zielstruktur erstellen
for s in ['train', 'test']:
    for c in ['pits', 'negatives']:
        os.makedirs(os.path.join(base_path, s, c), exist_ok=True)

# 1. Pit-Dateien für den Group-Split analysieren
pit_files = [f for f in os.listdir(pits_path) if f.endswith('.png')]
# Extrahiert die NAC-ID (den Teil nach dem letzten Unterstrich)
pit_strips = [f.rsplit('_', 1)[-1].replace('.png', '') for f in pit_files]

# 2. Split basierend auf NAC-IDs berechnen (verhindert Data Leakage)
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(pit_files, groups=pit_strips))

train_strips = set([pit_strips[i] for i in train_idx])
test_strips = set([pit_strips[i] for i in test_idx])

def move_and_count(source_folder, category_name):
    all_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]
    c_train, c_test = 0, 0

    for f in all_files:
        strip_id = f.rsplit('_', 1)[-1].replace('.png', '')
        name_without_ext = os.path.splitext(f)[0]
        
        # Zuordnung basierend auf der Strip-ID
        if strip_id in train_strips:
            target_sub = "train"
            c_train += 1
        elif strip_id in test_strips:
            target_sub = "test"
            c_test += 1
        else:
            # Falls ein Strip nur Negatives und kein Pit hat -> ab ins Training
            target_sub = "train"
            c_train += 1

        target_dir = os.path.join(base_path, target_sub, category_name)
        
        # PNG und NPY verschieben
        for ext in ['.png', '.npy']:
            src = os.path.join(source_folder, name_without_ext + ext)
            dst = os.path.join(target_dir, name_without_ext + ext)
            if os.path.exists(src):
                shutil.move(src, dst)
    
    return c_train, c_test

# 3. Dateien verschieben und Ergebnisse sammeln
p_train, p_test = move_and_count(pits_path, "pits")
n_train, n_test = move_and_count(negs_path, "negatives")

# 4. Statistik ausgeben
def print_stats(p, n, label):
    total = p + n
    ratio = n / p if p > 0 else 0
    print(f"{label}:")
    print(f"  Pits:      {p}")
    print(f"  Negatives: {n}")
    print(f"  Verhältnis: 1 : {ratio:.1f}")
    print(f"  Gesamt:    {total}")

print("\n--- Split Statistik ---")
print_stats(p_train, n_train, "TRAIN")
print("-" * 25)
print_stats(p_test, n_test, "TEST")

# TRAIN:
#   Pits:      228
#   Negatives: 1150
#   Verhältnis: 1 : 5.0
#   Gesamt:    1378
# -------------------------
# TEST:
#   Pits:      50
#   Negatives: 290
#   Verhältnis: 1 : 5.8
#   Gesamt:    340