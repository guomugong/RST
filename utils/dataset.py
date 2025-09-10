import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import copy

class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train, dataset_name, data_mean, data_std):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.data_mean = data_mean
        self.data_std = data_std

        if self.dataset_name == "drive" or self.dataset_name == "chase" or self.dataset_name == "rc-slo":
            self.imgs_path = sorted(glob.glob(os.path.join(data_path, 'img/*.tif')))
            self.labels_path = sorted(glob.glob(os.path.join(data_path, 'label/*.tif')))
        if self.dataset_name == "stare":
            self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        if self.dataset_name == "rimone" or self.dataset_name == "hrf" or self.dataset_name == "refuge" or self.dataset_name == "idrid" or self.dataset_name == "refuge2":
            self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.jpg'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))

        self.is_train = is_train

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        if self.dataset_name == "drive":
            label_path = image_path.replace('img', 'label')
            if self.is_train == 1:
                label_path = label_path.replace('_training.tif', '_manual1.tif') 
            else:
                label_path = label_path.replace('_test.tif', '_manual1.tif') 

        if self.dataset_name == "chase":
            label_path = image_path.replace('img', 'label')
            label_path = label_path.replace('.tif', '_1stHO.tif') 

        if self.dataset_name == "stare":
            label_path = image_path.replace('image', 'label')

        if self.dataset_name == "rc-slo":
            label_path = image_path.replace('img', 'label')

        if self.dataset_name == "rimone" or self.dataset_name == "hrf" or self.dataset_name == "refuge" or self.dataset_name == "idrid" or self.dataset_name == "refuge2":
            label_path = image_path.replace('img', 'label')

        img_rgb = Image.open(image_path)
        #img_rgb = img_rgb.resize((256,256))
        if self.dataset_name == "rimone" or self.dataset_name == "hrf" or self.dataset_name == "refuge" or self.dataset_name == "idrid" or self.dataset_name == "refuge2":
            img_rgb = img_rgb.resize((256,256))

        img_gray = img_rgb.convert('L')
        raw_height = img_rgb.size[1]
        raw_width = img_rgb.size[0]


        # Online augmentation
        if self.is_train == 1:
            if torch.rand(1).item() <= 0.9:
                img_rgb, img_gray= self.randomRotation(img_rgb, img_gray)

            if torch.rand(1).item() <= 0.25:
                img_rgb  = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                img_gray = img_gray.transpose(Image.FLIP_LEFT_RIGHT)

            if torch.rand(1).item() <= 0.25:
                img_rgb  = img_rgb.transpose(Image.FLIP_TOP_BOTTOM)
                img_gray = img_gray.transpose(Image.FLIP_TOP_BOTTOM)


        img_gray = np.asarray(img_gray)
        #img_gray = img_gray / 255
        img_gray = img_gray.reshape(1, img_gray.shape[0], img_gray.shape[1]) # CHW
        img_rgb  = np.asarray(img_rgb)
        img_rgb = img_rgb / 255
        img_rgb = img_rgb.transpose(2, 0, 1) #3HW

        sp = image_path.split('/')
        filename = sp[len(sp)-1]
        filename = filename[0:len(filename)-4] # del .tif

        return img_rgb, img_rgb, filename, raw_height, raw_width

    def __len__(self):
        return len(self.imgs_path)

    def randomRotation(self, image, label, mode=Image.BICUBIC):
        random_angle = torch.randint(low=0,high=360,size=(1,1)).long().item()
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    def padding_image(self,image, label, pad_to_h, pad_to_w):
        new_image = Image.new('RGB', (pad_to_w, pad_to_h), (0, 0, 0))
        new_label = Image.new('P', (pad_to_w, pad_to_h), (0, 0, 0))
        new_image.paste(image, (0, 0))
        new_label.paste(label, (0, 0))
        return new_image, new_label
