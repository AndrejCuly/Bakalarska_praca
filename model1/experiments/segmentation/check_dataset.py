from experiments.segmentation.dataset3d import Mammogram3DDataset

CANCER_PATH      = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancerous"
CANCER_FREE_PATH = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancer_free"

dataset = Mammogram3DDataset(
    cancer_dir=CANCER_PATH,
    cancer_free_dir=CANCER_FREE_PATH,
)

print("Dataset size:", len(dataset))

n_cancer      = sum(1 for s in dataset.samples if s['label'] == 1)
n_cancer_free = sum(1 for s in dataset.samples if s['label'] == 0)
print(f"  Cancerous sequences:   {n_cancer}")
print(f"  Cancer-free sequences: {n_cancer_free}")
print(f"  Imbalance ratio:       {n_cancer_free / max(n_cancer, 1):.1f}:1")

print("\nChecking first 50 samples...")
for i in range(min(50, len(dataset))):
    imgs, masks, pad_mask, label = dataset[i]

    assert imgs.shape[0] == 1,           f"[{i}] Expected 1 channel, got {imgs.shape[0]}"
    assert imgs.shape == masks.shape,    f"[{i}] imgs/masks shape mismatch: {imgs.shape} vs {masks.shape}"
    assert pad_mask.shape[0] == imgs.shape[1], f"[{i}] pad_mask length mismatch"

    s = dataset.samples[i]
    if i < 5:
        print(f"\nSample {i}:")
        print(f"  files:    {s['filenames']}")
        print(f"  imgs:     {imgs.shape}  dtype: {imgs.dtype}")
        print(f"  masks:    {masks.shape}  sum: {masks.sum().item():.0f}")
        print(f"  pad_mask: {pad_mask.tolist()}")
        print(f"  label:    {int(label.item())}  ({'cancer' if label.item() == 1 else 'cancer_free'})")

print("\nAll checks passed.")