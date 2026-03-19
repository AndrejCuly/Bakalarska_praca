import os
import shutil
import random

# =========================
# PATHS
# =========================

dataset_root = r"C:\Users\culya\Desktop\data_bakalarka\data\patches"
output_root = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

classes = ["cancer", "cancer_free"]

# =========================
# CREATE OUTPUT STRUCTURE
# =========================

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

# =========================
# SPLIT PATIENTS
# =========================

for cls in classes:

    class_path = os.path.join(dataset_root, cls)

    patients = [
        p for p in os.listdir(class_path)
        if os.path.isdir(os.path.join(class_path, p))
    ]

    random.shuffle(patients)

    n = len(patients)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_patients = patients[:train_end]
    val_patients = patients[train_end:val_end]
    test_patients = patients[val_end:]

    print(f"\n{cls}")
    print("Train:", len(train_patients))
    print("Val:", len(val_patients))
    print("Test:", len(test_patients))

    # =========================
    # COPY DATA
    # =========================

    for p in train_patients:

        src = os.path.join(class_path, p)
        dst = os.path.join(output_root, "train", cls, p)

        shutil.copytree(src, dst)

    for p in val_patients:

        src = os.path.join(class_path, p)
        dst = os.path.join(output_root, "val", cls, p)

        shutil.copytree(src, dst)

    for p in test_patients:

        src = os.path.join(class_path, p)
        dst = os.path.join(output_root, "test", cls, p)

        shutil.copytree(src, dst)

print("\nDataset split finished.")
