# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:06:34 2020

@author: 103920eili
"""

import argparse
import json
import os
from os.path import join, isfile
from os import listdir
import torch
import torch.utils.data as data
import torchvision.utils as vutils
from facenet_pytorch import MTCNN, InceptionResnetV1
from matplotlib import pyplot as plt

from attgan import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar
from utils import find_model
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from PIL import ImageFilter
import random
import csv
from boundingboxutils import expandBox
from bb_intersection_over_union import bb_intersection_over_union
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier



device = torch.device('cpu')

mtcnn = MTCNN(
    image_size=160,     
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], 
    #thresholds=[0.6, 0.6, 0.6], 
    factor=0.709, 
    post_process=False,
    device=device,
    select_largest=True,
    keep_all=False,    
)

mtcnn_margin = MTCNN(
    image_size=160,     
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],     
    factor=0.709, 
    post_process=False,
    device=device,
    select_largest=True,
    keep_all=False,
    margin=70
)


mtcnn_embedding = MTCNN(
    image_size=160, 
    margin=0, 
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], 
    factor=0.709, 
    post_process=True,
    device=device
)



# Ground truth bounding box CelebA
annotDf = pd.read_csv('D:/AI/Trainingdata/CelebA/annotations/list_bbox_celeba_3k.csv', sep=';')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
imgfolder = '5941'
isMale = False


# scikit-learn MLPClassifier classifier
#clf = pickle.load(open('female_model2.sav', 'rb'))


'''
0:  'Bald'
1:  'Bangs'
2:  'Black_Hair'
3:  'Blond_Hair'
4:  'Brown_Hair'
5:  'Bushy_Eyebrows'
6:  'Eyeglasses'
7:  'Male'
8:  'Mouth_Slightly_Open'
9:  'Mustache'
10: 'No_Beard'
11: 'Pale_Skin'
12: 'Young'
'''

#When using only 10
'''
0:  'Bangs'
1:  'Brown_Hair'
2:  'Bushy_Eyebrows'
3:  'Eyeglasses'
4:  'Male'
5:  'Mouth_Slightly_Open'
6:  'Mustache'
7: 'No_Beard'
8: 'Pale_Skin'
9: 'Young'
'''


def collate_fn(x):
    return x[0]

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', default='128_shortcut1_inject0_none')
    parser.add_argument('--test_atts', dest='test_atts', nargs='+', default=['Young'])
    parser.add_argument('--test_ints', dest='test_ints', type=float, nargs='+', help='test_ints', default=[-1.0] )
    parser.add_argument('--num_test', dest='num_test', type=int, default='30')
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    
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
        
    args.test_atts = args_.test_atts
    args.test_ints = args_.test_ints
    args.num_test = args_.num_test
    args.load_epoch = args_.load_epoch
    args.custom_img = args_.custom_img
    args.custom_data = args_.custom_data
    args.custom_attr = args_.custom_attr
    args.gpu = args_.gpu
    args.multi_gpu = args_.multi_gpu
    
    print(args)
    return args

#AttGAN stuff
args = createArgs()
attgan = AttGAN(args)
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
attgan.eval()



# Read bounding box CelebA format
# NAME_ID,X,Y,W,H
# "n000001/0001_01",60,60,79,109
def getGetBoxCelebA(nameId):
    # image_id;x_1;y_1;width;height
    row = annotDf.loc[annotDf['image_id'] == nameId]
    x0 = int(row['x_1'])
    y0 = int(row['y_1'])
    
    w = int(row['width'])
    h = int(row['height'])
    
    if y0 < 0:
        y0 = 0
    return [x0,y0, x0+w, y0+h]




def loadImage(path, filename=None):
    if filename is None:
        return Image.open(path)
    else:
        return Image.open(os.path.join(path,filename))    

# Returns a PyTorch transform object
def imageToTensorTransform():
    tf = transforms.Compose([
        #    transforms.CenterCrop(170), 
            transforms.Resize((128,128)),
            transforms.ToTensor(),            
            #https://discuss.pytorch.org/t/understanding-transform-normalize/21730
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))         
        ])
    return tf


def randomBool():
    val = random.uniform(0, 1.0)
    return val > 0.5


def optimumRandom(isMale):
    att_a = np.full((1, 13), 0.0)
    att_a = torch.tensor(att_a)
    att_a = att_a.type(torch.float)
    maxVal = 1.0
    minVal = -1.0
  
    #bangs
    att_a[:, 1] = random.uniform(minVal, maxVal)
    for i in range(4, 13, 1):
        #i=12: young
        val = random.uniform(minVal, maxVal)
        if i == 7: #male
            if isMale==True:
                val = random.uniform(0, minVal)
            else:
                val = random.uniform(0, maxVal)
        if i== 9: #Mustache
            if isMale==True:
                val = random.uniform(0, minVal)
            else:
                val = random.uniform(0, maxVal)
        if i== 10: #no beard
            if isMale==True:
                val = random.uniform(0, maxVal)
            else:
                val = random.uniform(0, minVal)
        if i == 11: #pale
            val = random.uniform(minVal, maxVal)
        att_a[:, i] = val
       
            
    att_b = att_a.clone()
    return att_b

    


def applyAttGan(image, attributes):
    transformed = [attgan.G(image, attributes)]
    transformed = torch.cat(transformed, dim=3)
    return transformed


def applyAttGanWithReverse(image, attributes, reverseAtts):
    transformed = applyAttGan(image, attributes)    
    return applyAttGan(transformed, reverseAtts)



#  Converts normalized tensor image to regular image
#  https://discuss.pytorch.org/t/conversion-from-a-tensor-to-a-pil-image-not-working-well-what-is-going-wrong/26121/2
#  mean and std to 0.5, as in dataloader, giving a range [-1, 1]
def convertTensorToImage(tr_im):
    z = tr_im * torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    z = z + torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    faceImg = transforms.ToPILImage(mode='RGB')(z)
    return faceImg


def getImageFilenamesInFolder(basePath, foldername):
    imgPath = join(basePath, foldername)
    files = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]
    return files
 
# alignedImages: array of aligned images, as torch tensors
def normDistance(alignedImages):
    daligned = torch.stack(alignedImages).to(device)
    embeddings = resnet(daligned).detach().cpu()
    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    p = dists[0][1]
    return p



# x_id, status, iou
# status for completed successfully is 3.
# -1: No faces detected. 0: more than one. 1: Not detected after 1st attgan transf. 2: not detected after 2nd attgan transform.
def saveDataToFile(imgFolder, dataArr):
    filename = join('tmp', imgFolder)
    filename = filename + '.csv'
    
    with open(filename, mode='w', newline='') as result_file:
        result_writer = csv.writer(result_file, delimiter=';',  quoting=csv.QUOTE_MINIMAL)
        for i in range(len(dataArr)):
            row = dataArr[i]            
            result_writer.writerow([i, row[0], row[1], row[2], row[3], row[4], row[5]]  )
            
            
# Saves the attribute sets to file on one line.
# AttrCollection: Array of attributes 13
# First column is filename, then attr1 1-13           
def saveAttrToFile(imgFolder):
    filename = join('tmp', imgFolder)
    filename = filename + '_attr.csv'
    with open(filename, mode='w', newline='') as result_file:        
        for i in range(len(attValues)):                                
            attr1 = attValues[i]
            line = str(i)
            line += ';'            
            d = attr1[0].numpy()            
            for k in d:
                line += str(k)
                line += ';'
            #print(line)
            result_file.write(line)
            
            
           
def appendResult(filename, attr, result):
    with open(filename, mode='a+', newline='') as result_file:  
        line = ''
        attr = attr[0]
        for i in range(len(attr)):            
            line += str(attr[i])
            line += ';'
        
        for j in range(len(result)):
            line += str(result[j])
            line += ';'
        line += '\n'    
        result_file.write(line)
        
        
        

def singelfaceAnon(origImg, tf, faceBox, landmarks, attr):
    expandedBox = expandBox(origImg.size, faceBox, landmarks, 1.5)    
    doPixelate = False
    w = int(expandedBox[2] -  expandedBox[0])
    h = int(expandedBox[3] -  expandedBox[1])
    #print(faceBox)
    #print(expandedBox)
    if w > 9964:
        doPixelate = True
    
    origSize = origImg.size
    pixelationFactor = origSize[0] / 64
    sw = int(origSize[0] / pixelationFactor)
    sh = int(origSize[1] / pixelationFactor)
    
    subImg = origImg.crop(expandedBox)
    #print(subImg.size)
    
    if doPixelate:
        subImg = subImg.resize((sw, sh))   
        subImg = subImg.resize((origSize[0],origSize[1]))
    #subImg.show()
    
    
    #subImg.show()
    
    subImgTensor = tf(subImg)    
    toTransform = subImgTensor.clone()
    toTransform = toTransform.unsqueeze(0)
    

    with torch.no_grad():
        
        transformed = applyAttGan(toTransform, attr)       
        transformedImg = convertTensorToImage(transformed[0, :])
        transformedImg = transformedImg.resize((w,h))
        #transformedImg.show() #!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return transformedImg, expandedBox
    

'''
basePath: 'D:/AI/Trainingdata/vggface2/Curated' 
srcpath: 'n000001' The person folder
srcFile: a specific image in the folder
'''
def faceAnonProcess(basePath, srcPath, srcFile, destPath, test, index):
    
    imagePath = join(basePath, srcPath)
    
    origImg = loadImage(imagePath, srcFile)    
    
    
   # origImg.show()
   
    # Note: Currently IOU calculation only one image per file.
    # Detect count: increases by 1 for eact MTCNN detection
    # Data: Filename; DetectCount; IoU
    # Need a different way to handle ground truth for multi face images.
    # Store results in array. For each image:
    # For IOU: The first MTCNN box is defined as ground thruth.    
    
    # contains the detected face as Tensor. Array shall contain original image in index 0, and the anonymized image in index 1.
    x_aligned = []
    
    
    finalImg = origImg.copy()
    origImg = origImg.copy()
    tmpImg = origImg.copy()
    tf = imageToTensorTransform()
    #origImgTensor = tf(origImg)    
    #toTransform = origImgTensor.clone()
    
    # Detect all faces in image    
    boxesOrig, probs, landmarks = mtcnn.detect(tmpImg, landmarks=True)
    if boxesOrig is None:
        #print('No face in file ' + srcFile)
        print(srcFile, str(0), str(0), 'Step 1')        
        attarr = att_b_.cpu().detach().numpy()
        appendResult('./tmp/learningattr.csv', attarr, (srcFile, 1, 0, 0, 0))
        return
    
    
    aligned1, prob = mtcnn_embedding(tmpImg, return_prob=True)
    if aligned1 is None: 
        print(srcFile, str(0), str(0), 'Step 1 aligned1') 
        attarr = att_b_.cpu().detach().numpy()
        appendResult('./tmp/learningattr.csv', attarr, (srcFile, 1, 0, 0, 0))
        return
    
     
    
    #Calculate IoU for the original un-anonymized image
    boxOrig = boxesOrig[0]    
    
       
    nfaces=len(boxesOrig)
    if nfaces>1:        
        print('nfaces: ' + str(nfaces))
        return
        
    # Anonymize all detected faces.
    # Returned array contains for each detection an anonymied image and the box coordinates 
    resultArr = []
    i=0
    for box in boxesOrig:
        result = singelfaceAnon(tmpImg, tf, box, landmarks[i], att_b_)     
        if test == True:
            result[0].show()
        resultArr.append(result)
        i = i + 1
    
        

    # Modify the original image. Paste the anonymized face in the position from the box        
    for anonResult in resultArr:
        transformedImg = anonResult[0]
        box = anonResult[1]
        tmpImg.paste(transformedImg, (int(box[0]), int(box[1])))
        
   # if test == True:
   #     tmpImg.show()  
        
    resultArr = []
    i=0
    boxes, probs, landmarks = mtcnn.detect(tmpImg, landmarks=True)
    if boxes is None:
        #print('No face in file ' + srcFile)
        attValues.append(att_b_)
        results.append((srcFile, 1, 0, 0, 0))
        
        return
    
        
    for box in boxes:
        result = singelfaceAnon(tmpImg, tf, box, landmarks[i], att_b_rev)
        #result[0].show()
        resultArr.append(result)
        i = i + 1
    
    for anonResult in resultArr:
        transformedImg = anonResult[0]
        box = anonResult[1]
        finalImg.paste(transformedImg, (int(box[0]), int(box[1])))
        
        
    anonfile = os.path.join(destPath, str(index) + '_' + srcFile)
    
    if test == True:
        finalImg.show()  
    else:
        finalImg.save(anonfile)    
    
    
    boxes_b, probs = mtcnn.detect(finalImg)
    if boxes_b is None:
        #print('Found no face in anonymized image: ' + srcFile)
        print(srcFile, str(0), str(nfaces), 'Step 3' )         
        attarr = att_b_.cpu().detach().numpy()
        appendResult('./tmp/learningattr.csv', attarr, (srcFile, 2, 0, 0, 0))
        return
    
    aligned2, prob = mtcnn_embedding(finalImg, return_prob=True)
    if aligned2 is None:
        print(srcFile, str(0), str(nfaces), 'Step 3 aligned2' )         
        attarr = att_b_.cpu().detach().numpy()
        appendResult('./tmp/learningattr.csv', attarr, (srcFile, 2, 0, 0, 0))
        return
    
    
    x_aligned.append(aligned1)
    x_aligned.append(aligned2)
    
    distance = normDistance(x_aligned)
                         
    nfaces_b = len(boxes_b)
    
    boxAn = boxes_b[0]
            
    iou = bb_intersection_over_union(boxOrig, boxAn)
    accepted = 0
    if iou > 0.5 and nfaces_b==1:
        accepted = 1
        
        
    attarr = att_b_.cpu().detach().numpy()
    appendResult('./tmp/learningattr.csv', attarr, (srcFile, 3, iou, distance, prob))
    
#    nfaces_b = len(boxes_b)
        
   
     

        


#isMale = randomBool()
isMale = False
print('Male: ', str(isMale))
attValues = []
results = []



'''
files = getImageFilenamesInFolder(basePath, srcpath)
print('File count: ' + str(len(files)))
i=0
for file in files:
    print(file)
    faceAnonProcess(basePath, srcpath, file, dstPath, False, i)    
    i += 1
    
saveDataToFile(srcpath, results)
'''

# CelebA 
basePath = 'data'
srcpath = 'CelebA_female'
dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/Learningfemales/'



i=0
files = getImageFilenamesInFolder(basePath, srcpath)
for i in range(500):
    att_b_ = optimumRandom(isMale)    
    att_b_rev = att_b_ * -1  
    print('Batch ' + str(i))
    for file in files:         
        print(file)
        faceAnonProcess(basePath, srcpath, file, dstPath, False, i)    
    



    

