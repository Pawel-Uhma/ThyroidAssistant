<<<<<<< HEAD
🧠 Thyroid Nodule Classification with Deep Learning
This project implements a deep learning system for classifying thyroid nodules as benign or malignant using multiple ultrasound images per patient. It uses a ResNet18-based convolutional neural network and applies patient-level aggregation to perform diagnosis from a bag of images, making it suitable for multi-instance medical imaging tasks.

📂 Project Structure

├── data/
│ ├── raw/ # Raw input images
│ ├── processed/ # Preprocessed images (resized & padded)
│ └── labels/ # CSVs: patient_label.csv
├── src/
│ ├── model.py # Model architecture (ResNet + aggregation)
│ ├── data_loader.py # Dataset loading and DataLoader creation
│ ├── train.py # Training and evaluation logic
│ ├── evaluate.py # Inference and confusion matrix visualization
│ └── utils.py # Plotting utilities (loss, confusion matrix, etc.)
├── config.yaml # Config for paths, hyperparameters, device
└── README.md # Project description (this file)
🛠 Features
✅ Patient-level classification using multiple ultrasound images

✅ ResNet18 backbone with mean aggregation

✅ Custom DataLoader for variable-length image bags

✅ Evaluation with accuracy and confusion matrix

✅ Training visualization (loss curves, class balance)

✅ Supports GPU training (Colab / local CUDA)

📊 Dataset
Each patient has 2–10 ultrasound images

Labels (0 = benign, 1 = malignant) are assigned per patient

Filenames are structured as: patientID_imagenumber.jpg, e.g., 354_002.jpg

Labels are stored in patient_label.csv as:

patient_name,histo_label
354,1
355,0
🚀 How to Train
Mount or organize your dataset in the data/ directory.

Adjust config.yaml:

paths:
image_dir: ../data/processed
labels_dir: ../data/labels
model_save_path: ../models/best_model.pth
training:
batch_size: 1
learning_rate: 0.001
epochs: 60
split_ratio: 0.8
log:
save_model: true
plot_loss: true
device: cuda
Run training:

python src/train.py
📈 Evaluation
After training, train.py will output:

Training/Validation Loss

Test Accuracy

Confusion Matrix

Loss and Accuracy plots

You can also run evaluate.py independently to test a saved model.

🧪 Example Use Case
You have a folder of thyroid ultrasound images per patient and want to build a system that predicts malignancy at the patient level, rather than per image. This system processes each image with ResNet, aggregates them, and returns a robust prediction.

🤝 Acknowledgments
Based on real-world diagnostic challenges

Inspired by medical MIL papers and multi-view CNN architectures
=======
# ThyroidAssistant
>>>>>>> a106adca2917adfcaf833720c3c2622304985af2
