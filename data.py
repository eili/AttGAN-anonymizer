# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Custom datasets for CelebA and CelebA-HQ."""

import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFilter


class YolodetectedWithlabel(data.Dataset):
    def __init__(self, image_path, box_path, attr_path, image_size, selected_attrs):
        self.image_path = image_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        self.labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        
        #If only one box (line in file), shape of array is (5,). If more lines, shape is (nlines, 5)
        boxlist = np.loadtxt(box_path, dtype=np.float32)
        if len(boxlist.shape) == 1:
            boxlist = boxlist.reshape(1, 5)
        boxlist = boxlist[:,1:] #remove the first column, which is the class (Yolo. Always face here)
        #self.images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        
        img = Image.open(image_path)
        width, height = img.size   
        
        # try to make it square
        '''
        if width > height:
            height = width
        if height > width:
            width = height
        '''
        
        #Convert from Yolo format. Values are normalized, and uses centerX, centerY, width and height of box. 
        #Calculates rectangle box in absolute image coordinates x1x2, x2y2.

        #Absolute Width and Height of box:
        W = boxlist[:,2] * width
        H = boxlist[:,3] * height

        # Add margin by scaling up the rectangle
        #W *= 1.4
        #H *= 1.4
        #Absolute center:
        X = boxlist[:,0] * width
        Y = boxlist[:,1] * height   
        
        boxlist[:,0] = X - W/2
        boxlist[:,1] = Y - H/2
        boxlist[:,2] = X + W/2
        boxlist[:,3] = Y + H/2
        
        self.boxes = boxlist
        
        self.regions = []
        for i, box in enumerate(boxlist):
            
            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            
            if box[2] > width:
                box[2] = width
            if box[3] > height:
                box[3] = height
            
                
            region = img.crop(box)
            self.regions.append(region)
        
        #ensure the image size is square
        sizeTuple = (image_size, image_size)
        
        self.tf = transforms.Compose([
            transforms.Resize(sizeTuple),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Remove to load into PIL and show
        ])
        
    
    def __getitem__(self, index):
        #return None
        img = self.tf(self.regions[index])
        box = torch.tensor(self.boxes[index])
        # Convert values from -1, 1 to 0, 1
        tmp = self.labels[0] + 1
        tmp = tmp // 2
        att = torch.tensor(tmp)
       
        return img, att, box
    
    def __len__(self):
        return len(self.regions)
    
class Nolabel(data.Dataset):
    def __init__(self, image_path, box_path, image_size):
        self.image_path = image_path
       
        #If only one box (line in file), shape of array is (5,). If more lines, shape is (nlines, 5)
        boxlist = np.loadtxt(box_path, dtype=np.float32)
        boxlist = boxlist.reshape(1, 5)
        boxlist = boxlist[:,1:] #remove the first column, which is the class (Yolo. Always face here)
        #self.images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        
        img = Image.open(image_path)
        width, height = img.size   
        
        # try to make it square
        '''
        if width > height:
            height = width
        if height > width:
            width = height
        '''
        
        #Convert from Yolo format. Values are normalized, and uses centerX, centerY, width and height of box. 
        #Calculates rectangle box in absolute image coordinates x1x2, x2y2.

        #Absolute Width and Height of box:
        W = boxlist[:,2] * width
        H = boxlist[:,3] * height

        # Add margin by scaling up the rectangle
        W *= 1.5
     #   H *= 1.5
        #Absolute center:
        X = boxlist[:,0] * width
        Y = boxlist[:,1] * height   
        
        boxlist[:,0] = X - W/2
        boxlist[:,1] = Y - H/2
        boxlist[:,2] = X + W/2
        boxlist[:,3] = Y + H/2
        
        self.boxes = boxlist
        
        self.regions = []
        for i, box in enumerate(boxlist):
            
            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            
            if box[2] > width:
                box[2] = width
            if box[3] > height:
                box[3] = height
            
                
            region = img.crop(box)
            self.regions.append(region)
        
        #ensure the image size is square
        sizeTuple = (image_size, image_size)
        
        self.tf = transforms.Compose([
            transforms.Resize(sizeTuple),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Remove to load into PIL and show
        ])
        
    
    def __getitem__(self, index):
        #return None
        img = self.tf(self.regions[index])
        box = torch.tensor(self.boxes[index])
        return img, box
    
    def __len__(self):
        return len(self.regions)
    
    
class Custom(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, selected_attrs):
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        self.images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        self.labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        self.image_size = image_size

        self.tf = transforms.Compose([
            transforms.CenterCrop(178), 
            transforms.Resize(image_size),
            transforms.ToTensor(),
            #image = (image - mean) / std. This will normalize the image in the range [-1,1]
            #https://discuss.pytorch.org/t/understanding-transform-normalize/21730
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #[0-1]
        ])
    
    def __getitem__(self, index):        
        #Original:
        #img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        #return img, att
        pilimg = Image.open(os.path.join(self.data_path, self.images[index]))
        
        #pixelation
        #pilimgsmall = pilimg.resize((16, 20))
        #pilimg = pilimgsmall.resize((178,218))
        
        img = self.tf(pilimg)
        return img, att
    
    
    def __len__(self):
        return len(self.images)
   
    def getImage(self, index):
        return Image.open(os.path.join(self.data_path, self.images[index]))  
    
    
    
class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs):
        super(CelebA, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        
        if mode == 'train':
            self.images = images[:182000]
            self.labels = labels[:182000]
        if mode == 'valid':
            self.images = images[182000:182637]
            self.labels = labels[182000:182637]
        if mode == 'test':
            self.images = images[182637:]
            self.labels = labels[182637:]
            #self.images = images[183236:]
            #self.labels = labels[183236:]
        
        self.tf = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                                       
        self.length = len(self.images)
    def __getitem__(self, index):
        pilimg = Image.open(os.path.join(self.data_path, self.images[index]))
        #pilimg = pilimg.filter(ImageFilter.BLUR)
        #pilimg = pilimg.filter(ImageFilter.GaussianBlur(1))
        #pilimgsmall = pilimg.resize((16,16))
        #pilimg = pilimgsmall.resize((128,128))
        
        img = self.tf(pilimg)
         
        #img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        
        
        
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    def __len__(self):
        return self.length
    
    def getImage(self, index):
        return Image.open(os.path.join(self.data_path, self.images[index]))  
    

class CelebA_HQ(data.Dataset):
    def __init__(self, data_path, attr_path, image_list_path, image_size, mode, selected_attrs):
        super(CelebA_HQ, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        orig_images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        orig_labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        indices = np.loadtxt(image_list_path, skiprows=1, usecols=[1], dtype=np.int)
        
        images = ['{:d}.jpg'.format(i) for i in range(30000)]
        labels = orig_labels[indices]
        
        if mode == 'train':
            self.images = images[:28000]
            self.labels = labels[:28000]
        if mode == 'valid':
            self.images = images[28000:28500]
            self.labels = labels[28000:28500]
        if mode == 'test':
            self.images = images[28500:]
            self.labels = labels[28500:]
        
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                                       
        self.length = len(self.images)
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    def __len__(self):
        return self.length

def check_attribute_conflict(att_batch, att_name, att_names):
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value
    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0:
                _set(att, 1-att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
    return att_batch


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    attrs_default = [
        'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
        'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to test')
    parser.add_argument('--data_path', dest='data_path', type=str, required=True)
    parser.add_argument('--attr_path', dest='attr_path', type=str, required=True)
    args = parser.parse_args()
    
    dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
    dataloader = data.DataLoader(
        dataset, batch_size=64, shuffle=False, drop_last=False
    )

    print('Attributes:')
    print(args.attrs)
    for x, y in dataloader:
        vutils.save_image(x, 'test.png', nrow=8, normalize=True, range=(-1., 1.))
        print(y)
        break
    del x, y
    
    dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
    dataloader = data.DataLoader(
        dataset, batch_size=16, shuffle=False, drop_last=False
    )