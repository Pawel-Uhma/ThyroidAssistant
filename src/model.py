import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from torchvision.models import resnet18

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
image_size = config["model"].get("image_size", 224)

class PatientResNet(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(PatientResNet, self).__init__()
        resnet = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.flattened_size = 512
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 1)
        )

    def forward(self, bag):
        device = next(self.parameters()).device
        features = []
        for img in bag:
            x = img.unsqueeze(0)
            feat = self.feature_extractor(x)
            feat = feat.view(feat.size(0), -1)
            features.append(feat)
        if features:
            features = torch.cat(features, dim=0)
            agg = torch.mean(features, dim=0, keepdim=True)
        else:
            agg = torch.zeros(1, self.flattened_size, device=device)
        out = self.classifier(agg)
        return out

class CustomDataset(Dataset):
    def __init__(self, data: list):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    bags, labels = zip(*batch)
    return list(bags), list(labels)

def create_dataloaders(train_dataset: list, test_dataset: list, batch_size: int):
    train_loader = DataLoader(
        CustomDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        CustomDataset(test_dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    return train_loader, test_loader