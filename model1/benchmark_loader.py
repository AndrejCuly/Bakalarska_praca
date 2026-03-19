import time
from torch.utils.data import DataLoader
from experiments.segmentation.dataset3d import Mammogram3DDataset


def main():

    dataset = Mammogram3DDataset(
        r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train"
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    print("Dataset size:", len(dataset))
    print("Testing loader speed...")

    start = time.time()

    for i, (imgs, masks) in enumerate(loader):

        if i == 50:
            break

    end = time.time()

    print("Time for 50 batches:", end - start)
    print("Average batch load time:", (end - start) / 50)


if __name__ == "__main__":
    main()