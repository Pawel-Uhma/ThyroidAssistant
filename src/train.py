import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import yaml
import logging

from data_loader import load_dataset, create_train_test_split
from utils import plot_loss, plot_confusion_matrix, plot_accuracy
from model import PatientResNet, create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_test_predictions(model, loader):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for bags, labels in loader:
            patient_bag = bags[0]
            transformed = [transform(img).to(device) for img in patient_bag]
            output = model(transformed)
            prediction = (torch.sigmoid(output) >= 0.5).item()
            pred_labels.append(int(prediction))
            true_labels.append(int(labels[0]))
    return true_labels, pred_labels

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for bags, labels in loader:
        patient_bag = bags[0]
        transformed = [transform(img).to(device) for img in patient_bag]
        optimizer.zero_grad()
        output = model(transformed)
        target = torch.tensor([[labels[0]]], dtype=torch.float32).to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for bags, labels in loader:
            patient_bag = bags[0]
            transformed = [transform(img).to(device) for img in patient_bag]
            output = model(transformed)
            target = torch.tensor([[labels[0]]], dtype=torch.float32).to(device)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = torch.sigmoid(output) >= 0.5
            correct += (pred.float() == target).sum().item()
    return total_loss / len(loader), correct / len(loader)

def main():
    logger.info("Starting training process...")
    image_dir = config["paths"]["image_dir"]
    labels_dir = config["paths"]["labels_dir"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    split_ratio = config["training"]["split_ratio"]
    epochs = config["training"]["epochs"]
    save_path = config["paths"]["model_save_path"]

    logger.info("Loading dataset...")
    dataset = load_dataset(image_dir, labels_dir)
    train_data, test_data = create_train_test_split(dataset, split_ratio)
    train_loader, test_loader = create_dataloaders(train_data, test_data, batch_size=batch_size)
    
    logger.info("Initializing model, loss, and optimizer...")
    model = PatientResNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    accuracy = []

    logger.info(f"Beginning training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(test_loss)
        accuracy.append(test_acc)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}")
        logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}")

    logger.info("Training complete.")
    if config["log"]["save_model"]:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    logger.info("Plotting...")
    true_labels, pred_labels = get_test_predictions(model, test_loader)
    plot_confusion_matrix(true_labels, pred_labels)
    plot_loss(train_losses, val_losses)
    plot_accuracy(accuracy)

if __name__ == "__main__":
    main()
