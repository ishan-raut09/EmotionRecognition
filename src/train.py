import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.model import VGG  # Using the new VGG19 architecture
import pandas as pd
import numpy as np
import os

class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pixels = self.df.iloc[idx]['pixels'].split(' ')
        # Standardize to 48x48
        image = np.array(pixels, dtype='uint8').reshape(48, 48)
        label = int(self.df.iloc[idx]['emotion'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train():
    # --- GitHub Optimized Config ---
    csv_path = "data/fer2013.csv"
    save_path = "models/emotion_model.pth"
    batch_size = 128  # Increased for stability
    epochs = 50       # More focused than the GitHub 250
    lr = 0.01         # GitHub's starting LR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Initializing GitHub-Optimized Training on: {device}")

    # 1. Data Prep with 44x44 Random Cropping (The GitHub "Secret")
    df = pd.read_csv(csv_path)
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'] == 'PublicTest']
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(44),          # GitHub strategy
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(44),           # Matching size for validation
        transforms.ToTensor(),
    ])

    train_dataset = FERDataset(train_df, transform=train_transform)
    val_dataset = FERDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. VGG19 + SGD Optimizer
    model = VGG('VGG19').to(device)
    criterion = nn.CrossEntropyLoss()
    # SGD generalizes better than Adam for FER
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

    # 3. Training Loop
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1} [{i}/{len(train_loader)}] Loss: {loss.item():.3f} | Acc: {100.*correct/total:.1f}%")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"📊 Epoch {epoch+1} Summary: Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc) # Decay LR if accuracy plateaus

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"🌟 New best accuracy: {val_acc:.2f}%! Saving...")
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train()
