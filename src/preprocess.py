import os
import cv2
import numpy as np
import pandas as pd

raw_path = "../data/raw"
processed_path = "../data/processed"
label_path = "../data/labels"

def get_raw_names(raw_path):
    return os.listdir(raw_path)

def get_processed_names(processed_path):
    return os.listdir(processed_path)

def get_unprocessed_names(raw_path, processed_path):
    return list(set(get_raw_names(raw_path)) - set(get_processed_names(processed_path)))

def process_image(image_name, raw_path, processed_path, dimension=224):
    image_path = os.path.join(raw_path, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return
    h, w = image.shape[:2]
    scale = min(dimension / w, dimension / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)  
    padded_image = np.zeros((dimension, dimension, image.shape[2]), dtype=resized_image.dtype)
    pad_x = (dimension - new_w) // 2
    pad_y = (dimension - new_h) // 2
    padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image
    processed_image_path = os.path.join(processed_path, image_name)
    cv2.imwrite(processed_image_path, padded_image)
    print(f"Processed and saved image: {processed_image_path}")

def process_all_unprocessed_images(raw_path, processed_path):
    for image_name in get_unprocessed_names(raw_path, processed_path):
        process_image(image_name, raw_path, processed_path)

def clean_csv(csv_file, processed_folder, output_file=None):
    df = pd.read_csv(csv_file)
    mask = df["path"].apply(lambda x: os.path.exists(os.path.join(processed_folder, x)))
    df_clean = df[mask]
    if output_file is None:
        output_file = csv_file
    df_clean.to_csv(output_file, index=False)
    return df_clean

def main():
    process_all_unprocessed_images(raw_path, processed_path)
    clean_csv("../data/labels/patient_image.csv",processed_path,"../data/labels/new_patient_image.csv")

if __name__ == "__main__":
    main()
