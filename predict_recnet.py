import numpy as np
from PIL import Image
import torch
import random
import cv2
from tqdm import tqdm
import torch.nn as nn
from model.recnet import RecNetWithST as RecNet
from utils.dataset import FundusSeg_Loader
from utils.eval_metrics import perform_metrics
import copy
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

target_domain="drive"
source_domain=target_domain
model_path='./snapshot/'+source_domain+'_rec.pth'

test_data_path = "./dataset/drive/test/"
dataset_mean=[0.3258]
dataset_std=[0.2234]

save_path='./results/'

if __name__ == "__main__":
    run_num=3
    random.seed(run_num) 
    np.random.seed(run_num)
    torch.manual_seed(run_num)
    torch.cuda.manual_seed(run_num)
    torch.cuda.manual_seed_all(run_num)

    with torch.no_grad():
        test_dataset = FundusSeg_Loader(test_data_path,0, target_domain, dataset_mean, dataset_std)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        print('Testing images: %s' %len(test_loader.dataset))
        device = torch.device('cuda')
        net = RecNet()
        net.to(device=device)
        print(f'Loading model {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=device))

        for image, label, filename, raw_height, raw_width in test_loader:
            image = image.cuda().float()
            image = image.to(device=device, dtype=torch.float32)
            pred = net(image)
            pred  = pred[:,:,:raw_height,:raw_width]  
            pred = pred.cpu().numpy().astype(np.double)[0]  
            image  = image.cpu().numpy().astype(np.double)[0]  
            image = image.transpose(1,2,0)
            pred = pred.transpose(1,2,0)
            pred = pred * 255
            print(pred.min())
            print(pred.max())
            #save_filename = save_path + filename[0] + '.png'
            #pred = np.uint8(pred)
            #pred = 0.5*image+0.5*pred
            #Image.fromarray(np.uint8(pred)).save(save_filename,"png")
            #cv2.imwrite(save_filename, pred)
