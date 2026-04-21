import os
import shutil

PATCHES_ROOT = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split"
MAMMS_ROOT   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed_v2"

# Build sets of all patients in whole mamms split
mamms_all        = set()  # every patient regardless of class
mamms_cancerous  = set()  # only cancerous patients

for split in ["train", "val", "test"]:
    for cls in ["cancerous", "cancer_free"]:
        folder = os.path.join(MAMMS_ROOT, split, cls)
        if not os.path.exists(folder):
            continue
        for patient in os.listdir(folder):
            mamms_all.add(patient)
            if cls == "cancerous":
                mamms_cancerous.add(patient)

print(f"Total patients in whole mamms: {len(mamms_all)}")
print(f"Cancerous patients in whole mamms: {len(mamms_cancerous)}")

# Dry run first — set to True to actually delete
DRY_RUN = False

removed_orphan    = 0
removed_mislabeled = 0

for split in ["train", "val", "test"]:
    for cls in ["cancer", "cancer_free"]:
        folder = os.path.join(PATCHES_ROOT, split, cls)
        if not os.path.exists(folder):
            continue
        for patient in os.listdir(folder):
            patient_path = os.path.join(folder, patient)

            # Rule 1 — patient doesn't exist in whole mamms at all
            if patient not in mamms_all:
                print(f"[ORPHAN]     {split}/{cls}/{patient}")
                if not DRY_RUN:
                    shutil.rmtree(patient_path)
                removed_orphan += 1
                continue

            # Rule 2 — patient is cancerous in whole mamms but in cancer_free in patches
            if cls == "cancer_free" and patient in mamms_cancerous:
                print(f"[MISLABELED] {split}/{cls}/{patient}")
                if not DRY_RUN:
                    shutil.rmtree(patient_path)
                removed_mislabeled += 1

print(f"\nOrphaned patients to remove:   {removed_orphan}")
print(f"Mislabeled patients to remove: {removed_mislabeled}")
if DRY_RUN:
    print("\nDRY RUN — no files deleted. Set DRY_RUN = False to apply.")