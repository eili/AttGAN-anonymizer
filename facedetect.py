# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:05:13 2019

@author: 103920eili
"""

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageDraw
from numpy import asarray
from matplotlib import pyplot
from os import listdir
import csv

workers = 0 if os.name == 'nt' else 4

device = torch.device('cpu')
#device = torch.device('cuda:0')

attr = 'attrib_all_point5'
#attr = 'gender_matrix'

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def collate_fn(x):
    return x[0]

#dataset = datasets.ImageFolder('./data/Cartoonset')
#dataset = datasets.ImageFolder('./data/Blur_test/test2')
#dataset = datasets.ImageFolder('./data/sample_testing_multi_3')
#dataset = datasets.ImageFolder('../AttGAN-PyTorch-master/data/CelebA')
dataset = datasets.ImageFolder('./output/128_shortcut1_inject0_none/attrib_all_point5')
#
#dataset = datasets.ImageFolder('./data/reference/Gender/')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


aligned = []
names = []
probs = []

def alignImages():
     for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            aligned.append(x_aligned)   
            filename = dataset.idx_to_class[y]
            names.append(filename)
            

def faceDetect():
    i=0
    for x, y in loader:
        imgfilename = dataset.samples[i][0]
        #print(i, imgfilename)
        i += 1
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            
            
            filename = dataset.idx_to_class[y]
            #print('Face detected with probability: {:8f}'.format(prob))
           # print(imgfilename + ': Face detected with probability: {:4f}'.format(prob))
            aligned.append(x_aligned)            
            names.append(imgfilename)
            probs.append(prob)
        else:
            print(imgfilename, ': mtcnn returned null')
    
def saveNameProbsToFile():
    with open('distance_' + attr + '.csv', mode='w', newline='') as result_file:
        result_writer = csv.writer(result_file, delimiter=';',  quoting=csv.QUOTE_MINIMAL)
        for i in range(len(names)):
            result_writer.writerow([names[i], probs[i]])
        
        

def printDistanceMatrix(aligned):        
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    print(pd.DataFrame(dists, columns=names, index=names))
    pd.DataFrame(dists, columns=names, index=names).to_csv('Distance_custom1.csv', sep=';', decimal=',')
    
    

'''
This function expects a folder structure of 
 -Folder (image name)
    - original file
    - anonymized file
 -Folder etc
This function will then read the two files in the directory and calculate the distance
between the two files, original and anonymized. 
This value is printed. Clear internal variables and move to next folder to repeat.    
'''    
def printDistanceTable():      
    k=0
    daligned = []
    prevFilename=''
    for x, y in loader:
        filename = dataset.idx_to_class[y]       
        x_aligned, prob = mtcnn(x, return_prob=True)
            
        if x_aligned is not None:
            if prevFilename != filename:
                k=0
            k=k+1
                
            prevFilename = filename
            
            daligned.append(x_aligned)             
            if k==2:  
                                
                daligned = torch.stack(daligned).to(device)
                embeddings = resnet(daligned).detach().cpu()
                dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
                p = dists[0][1]                
                               
                names.append(filename)          
                probs.append(p)
                print(filename + ': {:4f}'.format(p))
                k=0
                daligned = []
        else:
            k=0
            print(filename + ': mtcnn returned null...')
    print("Finished")
    print('k=' + str(k))
    
    
def datasetLoader():
    prevFilename=''
    for x, y in loader:
        filename = dataset.idx_to_class[y]       
    

  
    
def testLoadImage(filename):
    image = Image.open(filename)
    #pixels = asarray(image)
    #results = mtcnn.detect_faces(pixels)
    face_tensor, prob = mtcnn(image, return_prob=True)


#Extracts faces from image, and saves them to individual files    
def multifaceTest(filename):
    mtcnn = MTCNN(keep_all=True)
    img = Image.open(filename)
    boxes, probs = mtcnn.detect(img)
    #mtcnn(img, save_path='./data/tmp.png')
    

    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        draw.rectangle(box.tolist())
    img.save('multifaceTest-boxes.jpg');


def alignImage(filename):
    basename = os.path.basename(filename)
    fname = os.path.splitext(basename)[0]
    fext = os.path.splitext(basename)[1]
    filename_mod = fname + '_aligned' + fext
    print(filename_mod)
    img = Image.open(filename)
    img_align = mtcnn(img, save_path='data/test_images_aligned/{}/1.png'.format(fname))
    
    


#faceDetect()    
#saveNameProbsToFile()

#alignImages()
printDistanceTable()
#saveNameProbsToFile()
#printDistanceMatrix(aligned)
#plotFacesInFolder()
#multifaceTest('./data/multiface.jpg')
#alignImage('./data/wider/3_Riot_Riot_3_689_1.jpg')