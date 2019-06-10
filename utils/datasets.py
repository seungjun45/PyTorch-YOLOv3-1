import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from address_prefix import *
import _pickle as cPickle

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pdb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad1=pad1.item()
    pad2=pad2.item()
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)



class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [addr_label_prefix+path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files]
        self.img_files = [addr_prefix + path.replace("images","Images") for path in self.img_files]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

            # Apply augmentations
            if self.augment:
                if np.random.random() < 0.5:
                    # try:
                    img, targets = horisontal_flip(img, targets)
                    # except:
                    #     print('==============')
                    #     print(label_path)
                    #     print('==============')
                    #     pdb.set_trace()
        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        new_targets=[]
        new_imgs=[]
        new_paths=[]
        for (path,img,boxes) in zip(paths,imgs,targets):
            if boxes is not None:
                new_targets.append(boxes)
                new_imgs.append(img)
                new_paths.append(path)
        imgs=new_imgs
        paths=new_paths
        targets=new_targets
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape

        # Considering Gray-Scale Images... I changed codes a little bit
        tmp_list=[]
        for img in imgs:
            img=resize(img, self.img_size)
            tmp_list.append(img)
        imgs=torch.stack(tmp_list)
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

class GTADataSet(Dataset):
    def __init__(self, root_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        self.root_path = root_path
        json_path=root_path+'instances_GTA_train_small_2.json'
        json_=json.load(open(json_path))

        self.img_path_root=root_path+'images/'
        self.img_files=json_['images']
        self.anno=json_['annotations']
        self.img2anno=cPickle.load(open('img2anno.pkl','rb'))
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.root_path+'images/'+self.img_files[index]['file_name']

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        img=resize(img,(416,416))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        targets = None
        if self.img_files[index]['id'] in self.img2anno:
            anno_ids=self.img2anno[self.img_files[index]['id']]
            boxes=[]
            for anno_id in anno_ids:
                anno=self.anno[anno_id-1]
                tmp_box=anno['bbox'].copy()
                width=anno['width']
                height=anno['height']
                tmp_box=[tmp_box[0]/width, tmp_box[1]/height, tmp_box[2]/width, tmp_box[3]/height]
                tmp_box.insert(0,anno['category_id'])
                boxes.append(torch.Tensor(tmp_box))

            #boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            boxes=torch.stack(boxes)
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

            # Apply augmentations
            if self.augment:
                if np.random.random() < 0.5:
                    # try:
                    img, targets = horisontal_flip(img, targets)
                    # except:
                    #     print('==============')
                    #     print(label_path)
                    #     print('==============')
                    #     pdb.set_trace()
        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        new_targets=[]
        new_imgs=[]
        new_paths=[]
        for (path,img,boxes) in zip(paths,imgs,targets):
            if boxes is not None:
                new_targets.append(boxes)
                new_imgs.append(img)
                new_paths.append(path)
        imgs=new_imgs
        paths=new_paths
        targets=new_targets
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape

        # Considering Gray-Scale Images... I changed codes a little bit
        tmp_list=[]
        for img in imgs:
            img=resize(img, self.img_size)
            tmp_list.append(img)
        imgs=torch.stack(tmp_list)
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)