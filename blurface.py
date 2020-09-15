# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:16:13 2020
Blur or pixelate images located in folder
@author: 103920eili
"""
import argparse
import json
import os
from os.path import join

import torch
import torch.utils.data as data
import torchvision.utils as vutils
from facenet_pytorch import MTCNN, InceptionResnetV1

from attgan import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar
from utils import find_model
import numpy as np
from PIL import Image
from torchvision import transforms
from PIL import ImageFilter
import random
import csv
from boundingboxutils import expandBox


device = torch.device('cpu')

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=10,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def collate_fn(x):
    return x[0]

imgfolder = '5941'
iteration = 1


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', default='128_shortcut1_inject0_none')
     
    parser.add_argument('--num_test', dest='num_test', type=int, default='1')
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    
     
  # Denne er det vanlige...  
    parser.add_argument('--custom_img', action='store_true', default='True')
    parser.add_argument('--custom_data', type=str, default='./data/' + imgfolder)
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_celeb_grouped_' + imgfolder + '.txt')
    
    
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    return parser.parse_args(args)

def createArgs():
    args_ = parse()
    
    with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
        args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
        

    args.num_test = args_.num_test
    args.load_epoch = args_.load_epoch
    args.custom_img = args_.custom_img
    args.custom_data = args_.custom_data
    args.custom_attr = args_.custom_attr
    args.gpu = args_.gpu
    args.multi_gpu = args_.multi_gpu
    
    print(args)
    return args


#  Converts normalized tensor image to regular image
#  https://discuss.pytorch.org/t/conversion-from-a-tensor-to-a-pil-image-not-working-well-what-is-going-wrong/26121/2
#  mean and std to 0.5, as in dataloader, giving a range [-1, 1]
def convertTensorToImage(tr_im):
    z = tr_im * torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    z = z + torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    faceImg = transforms.ToPILImage(mode='RGB')(z)
    return faceImg

def normDistance(alignedImages):
    daligned = torch.stack(alignedImages).to(device)
    embeddings = resnet(daligned).detach().cpu()
    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    p = dists[0][1]
    return p


def createArray(img_a, img_b):
    arr = []
    arr.append(img_a)
    arr.append(img_b)
    return arr

def saveDistanceToFile(path):
    file_name = join(path, 'dist_att.csv')
    with open(file_name, mode='w', newline='') as result_file:
        # result_writer = csv.writer(result_file, delimiter=';',  quoting=csv.QUOTE_MINIMAL)
        for i in range(len(distances)):        
            line = names[i]
            line += ';'
            line += str(distances[i])
            line += ';'
                        
            result_file.write(line +  os.linesep)
            

args = createArgs()

output_path = join('output', args.experiment_name, 'pixelate_64_' + imgfolder + '_' + str(iteration))


names = []
distances = []
iterations = []

from data import Custom
test_dataset = Custom(args.custom_data, args.custom_attr, args.img_size, args.attrs)


#from data import CelebA
#test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'test', args.attrs)

os.makedirs(output_path, exist_ok=True)
test_dataloader = data.DataLoader(
    test_dataset, batch_size=1, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
if args.num_test is None:
    print('Testing images:', len(test_dataset))
else:
    print('Testing images:', min(len(test_dataset), args.num_test))


for idx, (img_a, att_a) in enumerate(test_dataloader):    
    
    img_a = img_a.cuda() if args.gpu else img_a
    faceImg = convertTensorToImage(img_a[0])
    #faceImg.show()
    #pilimg = img_a.filter(ImageFilter.GaussianBlur(1))
    currentfile = '{:06d}.jpg'.format(idx + 182638)
    
            
    x_aligned, prob = mtcnn(faceImg, return_prob=True) 
    if x_aligned is None:
        print('No face')
        continue
    
    boxes, probs, landmarks = mtcnn.detect(faceImg, landmarks=True)    
    
    
    tf = transforms.Compose([    
        transforms.ToTensor(),     
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    samples = []
    for i in range(1):
        transformed = img_a.clone()
        transformedImg = convertTensorToImage(transformed[0])
        
        #Blur image:
        finalImg = transformedImg.filter(ImageFilter.GaussianBlur(i+1))
        
        #Pixelate image:
        #sx = int(128/2)        
        #print(sx)
        #pilimgsmall = transformedImg.resize((sx, sx))
        #finalImg = pilimgsmall.resize((128,128))
        
        names.append(currentfile)        
        
        #finalImg.show()    
        #boxes, probs = mtcnn.detect(faceImg)    
        b0 = boxes[0]
        expandedBox = expandBox(finalImg.size, b0, landmarks, 1.5)
        subImg = finalImg.crop(expandedBox)
        faceImg.paste(subImg, (int(expandedBox[0]), int(expandedBox[1])))
                               
            
        #faceImg.show()  
        #print(i)
        file_name = join(output_path, currentfile)
        #faceImg.save(file_name)
        tfimage = tf(faceImg)
        samples.append(tfimage)
        
    #Changes dimension, so that slider is horizontal instead of vertical.
    samples = torch.cat(samples, dim=2)        
    vutils.save_image(
        samples, join(output_path, currentfile),
        nrow=1, normalize=True, range=(-1., 1.)
    )
    
        
    
saveDistanceToFile(output_path)    