from experiments.segmentation.dataset3d import Mammogram3DDataset

dataset = Mammogram3DDataset(
    r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train"
)

count_positive = 0

for i in range(200):

    img, mask = dataset[i]

    if mask.sum() > 0:
        count_positive += 1

print("positive samples:", count_positive)

img, mask = dataset[0]

print(img.shape)
print(mask.shape)
print("mask sum:", mask.sum())