from model.unet import UNet 
from model.recnet import RecNetWithST as RecNet
from utils.fundus_dataloader import FundusSeg_Loader
from torch import optim
import torch.nn as nn
import random
import torch
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


dataset_name = "drive" # 

train_data_path = "./dataset/drive/train/"
valid_data_path = "./dataset/drive/test/"
N_epochs = 1200
lr_decay_step = [1000]
lr_init = 0.0001
batch_size = 1
test_epoch = 10
dataset_mean=[0.4969, 0.2702, 0.1620]
dataset_std=[0.3479,0.1896,0.1075]

def train_net(net, recnet, device, epochs=N_epochs, batch_size=batch_size, lr=lr_init):
    train_dataset = FundusSeg_Loader(train_data_path, 1, dataset_name, dataset_mean, dataset_std)
    valid_dataset = FundusSeg_Loader(valid_data_path, 0, dataset_name, dataset_mean, dataset_std)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=1, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    print('Train images: %s' % len(train_loader.dataset))
    print('Valid images: %s' % len(valid_loader.dataset))

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=lr_decay_step,gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        net.train()
        train_loss = 0
        for i, (image, label, filename, raw_height, raw_width) in enumerate(train_loader):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                img_aug = recnet(image)
            if torch.rand(1).item() <= 0.5:
                img_fuse = image
            else:
                alpha = torch.rand(1).item()
                img_fuse = alpha*image + (1-alpha)*img_aug
            pred = net(img_fuse)
            loss = criterion(pred, label)
            train_loss = train_loss + loss.item()

            loss.backward()
            optimizer.step()

        # Validation
        # epoch != test_epoch
        if ((epoch+1) % test_epoch == 0):
            net.eval()
            val_loss = 0
            for i, (image, label, filename, raw_height, raw_width) in enumerate(valid_loader):
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                val_loss = val_loss + loss.item()

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(), './snapshot/drive_seg.pth')
                print('saving model............................................')
        
            print('Loss/valid', val_loss / i)
            sys.stdout.flush()

        scheduler.step()

if __name__ == "__main__":
    run_num=3
    random.seed(run_num) 
    np.random.seed(run_num)
    torch.manual_seed(run_num)
    torch.cuda.manual_seed(run_num)
    torch.cuda.manual_seed_all(run_num)
    device = torch.device('cuda')
    recnet = RecNet()
    seg_net = UNet(n_channels=3, n_classes=1)
    recnet.load_state_dict(torch.load('./snapshot/drive_rec.pth'))
    recnet.to(device=device)
    seg_net.to(device=device)
    train_net(seg_net, recnet, device)
