import torch
import matplotlib.pyplot as plt
from experiments.segmentation.dataset3d import Mammogram3DDataset
from unet3d_model.unet3d import UNet3D


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = Mammogram3DDataset(
    r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\val"
)


model = UNet3D(in_channels=1, num_classes=1)

model.load_state_dict(torch.load("best_model.pth", map_location=device))

model.to(device)

model.eval()


img, mask = dataset[0]

img_input = img.unsqueeze(0).to(device)


with torch.no_grad():

    pred = model(img_input)

    pred = torch.sigmoid(pred)


pred = pred.cpu()[0,0]
img = img[0]
mask = mask[0]


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(img[0], cmap="gray")

plt.subplot(1,3,2)
plt.title("Mask")
plt.imshow(mask[0], cmap="gray")

plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(pred[0], cmap="gray")

plt.show()