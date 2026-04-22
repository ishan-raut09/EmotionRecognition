import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.model import EmotionResNet
import pandas as pd
import numpy as np
import os

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss
        return loss.mean()

class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pixels = self.df.iloc[idx]['pixels'].split(' ')
        image = np.array(pixels, dtype='uint8').reshape(48, 48)
        image = np.stack([image, image, image], axis=2)
        label = int(self.df.iloc[idx]['emotion'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train():
    csv_path = "data/fer2013.csv"
    save_path = "models/emotion_model.pth"
    batch_size = 128  
    epochs = 25       
    lr = 0.0003       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(csv_path)
    train_df = df[df['Usage'].isin(['Training', 'PublicTest'])]
    val_df = df[df['Usage'] == 'PrivateTest']
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FERDataset(train_df, transform=train_transform)
    val_dataset = FERDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EmotionResNet(pretrained=True).to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.fc.parameters():
        param.requires_grad = True
    
    class_counts = train_df['emotion'].value_counts().sort_index().values
    total_samples = sum(class_counts)
    class_weights = total_samples / (len(class_counts) * class_counts)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # 4. Focal Loss implementation (Handles Hard Classes like Fear/Disgust)
    criterion = FocalLoss(weight=weights, gamma=2.0, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 1. Cosine Annealing LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        if epoch == 5:
            print("🔓 Unfreezing ResNet Backbone for deep fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            # 2. Lower LR on unfreeze to protect pretrained features
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

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

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 3. Test-Time Augmentation (TTA)
                pred1 = model(images)
                pred2 = model(torch.flip(images, dims=[3]))
                outputs = (pred1 + pred2) / 2.0
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"📊 Epoch {epoch+1} Summary: Val Acc: {val_acc:.2f}%")
        
        scheduler.step() 

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            print(f"🌟 New best accuracy: {val_acc:.2f}%! Saving...")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered. Model converged at {best_acc:.2f}% validation accuracy.")
                break

if __name__ == "__main__":
    train()
