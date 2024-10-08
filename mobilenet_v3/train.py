from loss import LossCombine
from mobilenet_v3 import MobileDepth
from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.notebook as tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_image_path, train_depth_path, test_image_path, test_depth_path = getPath().values()
dataloader_train = getData(16, True ,  train_image_path, train_depth_path)
dataloader_val = getData(16, False, test_image_path, test_depth_path)

model = MobileDepth()
model.to(device)
criterion = LossCombine(ld = 0.1)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.01, betas = (0.9, 0.999))
EPOCHS = 100


import torch
from tqdm import tqdm

def train(model, dataloader, val_dataloader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for sample in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            image, depth = sample['image'].to(device), sample['depth'].to(device)

            pred = model(image)

            l = criterion(pred, depth)
            l.backward()
            optimizer.step()

            epoch_loss += l

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}')

        evaluate(model, val_dataloader, criterion, device)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating"):
            image, depth = sample['image'].to(device), sample['depth'].to(device)
            pred = model(image)

            val_loss = criterion(pred, depth)
            total_val_loss += val_loss

    avg_val_loss = total_val_loss / len(dataloader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
