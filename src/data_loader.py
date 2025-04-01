import os
import random
from typing import List, Tuple, Any
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
def load_dataset(image_dir: str, labels_dir: str) -> List[Tuple[List[Image.Image], Any]]:
    logger.info("Loading dataset using image file names directly...")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    df_images = pd.DataFrame({'path': image_files})
    df_images["patient_name"] = df_images["path"].apply(lambda x: x.split('_')[0]).astype(str)
    patient_label_csv = os.path.join(labels_dir, "patient_label.csv")
    df_labels = pd.read_csv(patient_label_csv)
    df_labels["patient_name"] = df_labels["patient_name"].astype(str)
    logger.info(f"Found {len(df_images)} image files and {len(df_labels)} label records.")
    merged = pd.merge(df_images, df_labels, on="patient_name")
    logger.info(f"Merged dataset has {len(merged)} records across {merged['patient_name'].nunique()} patients.")
    dataset = []
    skipped_images = 0
    for patient, group in merged.groupby("patient_name"):
        images_list = []
        label = group["histo_label"].iloc[0]
        for _, row in group.iterrows():
            img_path = os.path.join(image_dir, row["path"])
            try:
                img = Image.open(img_path).convert("RGB")
                images_list.append(img)
            except Exception:
                skipped_images += 1
                logger.warning(f"Skipping image: {img_path}")
        dataset.append((images_list, label))
    logger.info(f"Finished loading dataset. Total patients: {len(dataset)}. Skipped images: {skipped_images}")
    return dataset


def create_train_test_split(dataset: List[Tuple[List[Image.Image], Any]], split_ratio: float) -> Tuple[List[Tuple[List[Image.Image], Any]], List[Tuple[List[Image.Image], Any]]]:
    logger.info("Shuffling and splitting dataset...")
    random.shuffle(dataset)
    split_index = int(len(dataset) * split_ratio)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    logger.info(f"Split dataset into {len(train_data)} training and {len(test_data)} testing samples.")
    return train_data, test_data

class CustomDataset(Dataset):
    def __init__(self, data: List[Tuple[List[Image.Image], Any]]):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def create_dataloaders(train_dataset: List[Tuple[List[Image.Image], Any]], test_dataset: List[Tuple[List[Image.Image], Any]], batch_size: int) -> Tuple[DataLoader, DataLoader]:
    logger.info(f"Creating DataLoaders with batch size = {batch_size}")
    train_loader = DataLoader(CustomDataset(train_dataset), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CustomDataset(test_dataset), batch_size=batch_size, shuffle=False)
    logger.info("DataLoaders ready.")
    return train_loader, test_loader
