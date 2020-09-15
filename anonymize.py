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
    #thresholds=[0.6, 0.7, 0.7], 
    thresholds=[0.6, 0.6, 0.6], 
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


# Ground truth bounding box VGGFaces
annotDf = pd.read_csv('D:/AI/Trainingdata/vggface2/bb_landmark/loose_bb_test.csv')

# Ground truth bounding box CelebA
#annotDf = pd.read_csv('D:/AI/Trainingdata/CelebA/annotations/list_bbox_celeba_.csv', sep=';')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
imgfolder = '5941'
isMale = False

attribDf = pd.read_csv('tmp/n000009_0016_01_2000_attriblist.csv', sep=';', header=None)
attribDf.drop(attribDf.columns[[0]], axis=1, inplace=True)
attrmatrix = pd.DataFrame.from_records(attribDf.values)


# scikit-learn MLPClassifier classifier
clf = pickle.load(open('female_model1.sav', 'rb'))


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


# Read bounding box VGGFace format
# NAME_ID,X,Y,W,H
# "n000001/0001_01",60,60,79,109
def getGtBox(nameId):
    # NAME_ID,X,Y,W,H
    row = annotDf.loc[annotDf['NAME_ID'] == nameId]
    x0 = int(row['X'])
    y0 = int(row['Y'])
    
    w = int(row['W'])
    h = int(row['H'])
    
    if y0 < 0:
        y0 = 0
    return [x0,y0, x0+w, y0+h]


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
            #transforms.CenterCrop(170), 
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
    maxVal = 0.8
    minVal = -0.8
    
    #maxVal = 0.4
    #minVal = -0.4
    
    maxPaleVal = 0.7
    minPaleVal = -0.8
    
    #bangs
    att_a[:, 1] = random.uniform(minVal, maxVal)
    for i in range(4, 13, 1): # First tests started from 5. when expandbox=1.5 brown hair was included and start from 4.
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
            val = random.uniform(minPaleVal, maxPaleVal)
        att_a[:, i] = val
       
            
    att_b = att_a.clone()
    return att_b



def makeAttribs10(isMale):
    arr = np.zeros(10)
    maxVal = 1.1
    minVal = -1.1
    maxValPale = 0.8
    minValPale = -0.8
    
    for i in range(0,10, 1):        
        arr[i] = np.random.uniform(minVal, maxVal)
        if i == 4 or i == 6: #male, Mustache
             arr[i] = np.random.uniform(-0.1, minVal) if isMale==True else np.random.uniform(0.1, maxVal)
        if i == 7: #no beard
             arr[i] = np.random.uniform(0.1, maxVal) if isMale==True else np.random.uniform(-0.1, minVal)
        if i == 9: #pale
            arr[i] = np.random.uniform(minValPale, maxValPale)
    return arr
        
    
    
def evaluateAttr(arr):
    X = []
    X.append(arr)
    y_pred = clf.predict(X)
    return y_pred


def makeGoodAttr(isMale):
    isok = 0
    k = 0
    while isok==0:
        arr = makeAttribs10(False)
        isok = evaluateAttr(arr)[0]
        k += 1
    return arr,k

def convertToAttGanAttr(arr):
    arr = arr[0]
    arr = np.insert(arr,0,0, axis=0)
    arr = np.insert(arr,2,0, axis=0)
    arr = np.insert(arr,2,0, axis=0)
    att_a = torch.tensor(arr)
    att_a = att_a.type(torch.float)
    att_a = att_a.unsqueeze(0) # inc dimension to [1,13]
    return att_a.clone()    
    

def singleAttArray(index, value):
    if index > 12:
        index = 12
    
    att_a =  np.full((1, 13), 0.0)
    att_a[:, index] = value
   
    att_a = torch.tensor(att_a)
    att_a = att_a.type(torch.float)
    att_b = att_a.clone()
    return att_b

def binaryAttArray():
    x = np.unpackbits(np.arange(4).astype(np.uint8)[:,None], axis=1)
    return x


def attribFromFile(index):
    attrRow = attrmatrix.loc[index,:]        
    
    att_a = torch.tensor(attrRow)
    att_a = att_a.type(torch.float)
    att_a = att_a.unsqueeze(0) # inc dimension to [1,13]
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
            
            
# Saves the two attribute sets to file on one line.
# AttrCollection: Array of attributes 13
# First column is filename, then attr1 1-13           
def saveAttrToFile(imgFolder, imgfilename, attrCollection):
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
            result_file.write(line +  os.linesep)
            
            
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


def singelfaceAnon(origImg, tf, faceBox, landmarks, attr, doPixelate):
    expandedBox = expandBox(origImg.size, faceBox, landmarks, 1.5)    
    
    w = int(expandedBox[2] -  expandedBox[0])
    h = int(expandedBox[3] -  expandedBox[1])
    #print(faceBox)
    #print(expandedBox)
    #if w > 9964:
    #    doPixelate = True
    boxfactor = h / w
    origSize = origImg.size
    pw = pixelate_x
    ph = int(pw * boxfactor)
    
    
    subImg = origImg.crop(expandedBox)
    #print(subImg.size)
    
    
    pixelateSize = (pw, ph)
    upsize =(w, h)
    if doPixelate:
        subImg = subImg.resize((pixelateSize))   
        subImg = subImg.resize((upsize))
    #subImg.show()
    
    k=1;
    if doPixelate == True:
        k = 0
    
    subImg.save('C:/temp/pixelate/6011_29_' + str(k) + '.jpg')
    
    subImgTensor = tf(subImg)    
    toTransform = subImgTensor.clone()
    toTransform = toTransform.unsqueeze(0)
    
    # Make gender random
    #isMale = randomBool()
    
    #att_b_ = optimumRandom(isMale)
    #att_b_rev = optimumRandom(not isMale)
    #att_b_rev = att_b_ * -1    
    #att_b_ = singleAttArray(12, -1)
    #print(att_b_.data)
    with torch.no_grad():
        #transformed = applyAttGanWithReverse(toTransform, att_b_, att_b_rev)                         
        transformed = applyAttGan(toTransform, attr)                         
   
        transformedImg = convertTensorToImage(transformed[0, :])
        transformedImg = transformedImg.resize((w,h))
    #    transformedImg.show()
        return transformedImg, expandedBox
    

'''
basePath: 'D:/AI/Trainingdata/vggface2/Curated' 
srcpath: 'n000001' The person folder
srcFile: a specific image in the folder
'''
def faceAnonProcess(basePath, srcPath, srcFile, destPath, test, index):
    #srcPath = './data/test_images/kate_siegel'
    #srcPath = './data/test_images/shea_whigham'
    #srcPath = './data/test_images/multiface'
    #srcFile = '1.jpg'
        
    imagePath = join(basePath, srcPath)
    
    origImg = loadImage(imagePath, srcFile)
    
    #Filename without extension .jpg
    filenameWoExt = os.path.splitext(srcFile)[0]
    
    #Annotation file ground thruth identity
#    gtId = srcPath + '/' + filenameWoExt
    
    # CelebA
    #boxGt = getGetBoxCelebA(srcFile)
    # VGGFace:
#    boxGt = getGtBox(gtId)
    
    origImg.show()
   
    # Note: Currently IOU calculation only one image per file.
    # Detect count: increases by 1 for eact MTCNN detection
    # Data: Filename; DetectCount; IoU
    # Need a different way to handle ground thruth for multi face images.
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
        results.append((srcFile, -1, 0, 0, 0, 0))
        return
    aligned1, prob = mtcnn_embedding(tmpImg, return_prob=True)
    #Calculate IoU for the original un-anonymized image
#    boxOrig = boxesOrig[0]    
#    iouOrig = bb_intersection_over_union(boxGt, boxOrig)
    

    #x_aligned1 = mtcnn(origImg, return_prob=False).squeeze()    
    #x_aligned.append(x_aligned1)
    #tmp = convertTensorToImage(x_aligned1[0,:])      
    #tmp.show()

    
    #subImg.show()

    
    # Draw rectangles around detections
    # Make a deep copy of the image
    '''
    if test == True:
        tmpImg = origImg.copy()
        draw = ImageDraw.Draw(tmpImg)
        for i, box in enumerate(boxesOrig):
            draw.rectangle(box.tolist())
        arrbox = np.asarray(boxGt).tolist()
        draw.rectangle(arrbox, outline='red')
        tmpImg.save('C:/temp/orig_'+ srcFile);        
    
    '''
    nfaces=len(boxesOrig)
    if nfaces>1:
#        results.append((srcFile, 0, 0, 0, 0, iouOrig))
        print('nfaces: ' + str(nfaces))
        return
        
    # Anonymize all detected faces.
    # Returned array contains for each detection an anonymied image and the box coordinates 
    resultArr = []
    i=0
    for box in boxesOrig:
        result = singelfaceAnon(tmpImg, tf, box, landmarks[i], att_b_, True)     
        if test == True:
            result[0].show()
        resultArr.append(result)
        i = i + 1
    
        

    # Modify the original image. Paste the anonymized face in the position from the box        
    for anonResult in resultArr:
        transformedImg = anonResult[0]
        box = anonResult[1]
        tmpImg.paste(transformedImg, (int(box[0]), int(box[1])))
    
    transformedImg.save('C:/temp/pixelate/6011_29_transf1' + '.jpg')
    
   
   # if test == True:
   #     tmpImg.show()  
        
    resultArr = []
    i=0
    boxes, probs, landmarks = mtcnn.detect(tmpImg, landmarks=True)
    if boxes is None:
        #print('No face in file ' + srcFile)
        print(srcFile, str(0), str(0), 'Step 2')
#        results.append((srcFile, 1, 0, 0, 0, iouOrig))
        
        return
    
    # Reverse transform        
    for box in boxes:
        result = singelfaceAnon(tmpImg, tf, box, landmarks[i], att_b_rev, False)        
        
        #result[0].show()
        resultArr.append(result)
        i = i + 1
    
    for anonResult in resultArr:
        transformedImg = anonResult[0]
        box = anonResult[1]
        finalImg.paste(transformedImg, (int(box[0]), int(box[1])))
        
       
    transformedImg.save('C:/temp/pixelate/6011_29_transf2' + '.jpg')
    anonfile = os.path.join(destPath, str(index) + '_' + srcFile)
    
    if test == True:
        finalImg.show()  
    else:
        finalImg.save(anonfile)    
    
    
    boxes_b, probs = mtcnn.detect(finalImg)
    if boxes_b is None:
        #print('Found no face in anonymized image: ' + srcFile)
        print(srcFile, str(0), str(nfaces), 'Step 3' ) 
#        results.append((srcFile, 2, 0, 0, 0, iouOrig))
        return
    aligned2, prob = mtcnn_embedding(finalImg, return_prob=True)
                 
    nfaces_b = len(boxes_b)
    
    boxAn = boxes_b[0]
        
    
#    iou = bb_intersection_over_union(boxGt, boxAn)
    accepted = 0
#    if iou > 0.5 and nfaces_b==1:
#        accepted = 1
        
#   results.append((srcFile, 3, iou, accepted, nfaces_b, iouOrig))
    results.append((srcFile, 3, 0, accepted, nfaces_b, 0))
    
    nfaces_b = len(boxes_b)
    #print(srcFile + ': ' +  ': ' + str(nfaces_b)  + '/' + str(nfaces) + '= ' + str(round(100*nfaces_b / nfaces, 2)) + '%') 
#    print(srcFile, str(nfaces_b), str(nfaces) ) 
#    print(srcFile, str(round(iou, 2)), str(nfaces_b) ) 
    x_aligned.append(aligned1)
    x_aligned.append(aligned2)
    if aligned2 is None or aligned1 is None:
        print('aligned is None')
        distance = -1
    else:
        distance = normDistance(x_aligned)
   
    attarr = att_b_.cpu().detach().numpy()
    appendResult(join(dstPath, 'result.csv'), attarr, (srcFile, distance, prob))
    
    '''
    if test == False:
        #This is for saving original and transformed images in named folder
        #Is used for calculating distance between the two images.
        #path_for_distance = 'D:/AI/Results/AttGAN/Anonymized/vggface2/Distance'
        path_for_distance = 'D:/AI/Results/AttGAN/Anonymized/CelebA/prepixelate/5941/batch1/Distance'
        #imgFolder = os.path.splitext(srcFile)[0]
        #For running a whole directory as batch
        #os.mkdir(join(path_for_distance, filenameWoExt ))        
        #origImg.save(join(path_for_distance, filenameWoExt, srcFile))    
        #finalImg.save(join(path_for_distance, filenameWoExt, 'a_'+srcFile))    

        filenameWoExt = str(index)
        #For running single image in a loop:
        os.mkdir(join(path_for_distance, filenameWoExt))
        origImg.save(join(path_for_distance, filenameWoExt, srcFile))    
        finalImg.save(join(path_for_distance, filenameWoExt, 'a_'+srcFile))    
     '''

        
    '''
    tmpImg = origImg.copy()
    draw = ImageDraw.Draw(tmpImg)
    for i, box in enumerate(boxes_b):
        draw.rectangle(box.tolist())
    tmpImg.save('C:/temp/3_Riot_Riot_3_26_8-bluranon_boxes.jpg');
    '''

    

def facedetect(srcPath, srcFile):
    origImg = loadImage(srcPath, srcFile)
    #origImg.show()
    faces = mtcnn(origImg)
    if faces is None:
        print('No face in file ' + srcFile)
        return
    nfaces=len(faces)
    print(srcFile + ': ' + str(nfaces) )
    # Visualize
    for face in faces:        
        #change from: torch.Size([3, 160, 160]) to 160,160,3
        plt.imshow(face.permute(1, 2, 0).int().numpy())
        plt.axis('off');
    

# Save the detected face(s) for save destination    
def facedetectSave(srcPath, srcFile, savePath):
    origImg = loadImage(srcPath, srcFile)
    savePath = os.path.join(savePath,srcFile)
    #origImg.show()
    mtcnn_margin(origImg, save_path=savePath)
    
    
    
srcpath = 'D:/AI/Trainingdata/WIDER\WIDER_val/images/3--Riot'   
dstPath = 'D:/AI/Results/AttGAN/Anonymized/WIDER/3--Riot_pxfilter64'

#VGG
basePath = 'D:/AI/Trainingdata/vggface2/Curated'
srcpath = 'n000009'


#female
dstPath = 'D:/AI/Results/AttGAN/Anonymized/vggface2/n000009_016_femalemodel1'
srcImg = '0016_01.jpg'

# CelebA 
#basePath = 'data'
#srcpath = '10154'
#dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/LearnedAttribs/10154/batch1'

basePath = 'C:/Users/103920eili/source/AttGAN-PyTorch-master/data'
srcpath = '6011'
dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/\prepixelate/5941/30x30/b5'
dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/\prepixelate/5941/straight/b7'
	
#os.mkdir(dstPath)


#male
#dstPath = 'D:/AI/Results/AttGAN/Anonymized/vggface2/n000689_0020_01'
#srcImg = '0020_01.jpg'

#srcpath = 'D:/AI/Results/AttGAN/Anonymized/WIDER/2--Demonstration_pxfilter64'
#dstPath = 'D:/AI/Results/AttGAN/Anonymized/WIDER/2--Demonstration_pxfilter64_faces'

#srcpath = 'D:/AI/Trainingdata/WIDER\WIDER_val/images/2--Demonstration'   
#dstPath = 'D:/AI/Results/AttGAN/Anonymized/WIDER/2--Demonstration_faces'
#srcpath = 'D:/AI/Trainingdata/Video_images/climatechange_orig'
#dstPath = 'D:/AI/Results/AttGAN/Anonymized/video/climatechange'

#isMale = randomBool()
isMale = True
print('Male: ', str(isMale))
attValues = []
att_b_ = optimumRandom(isMale)
att_b_rev = att_b_ * -1    

#att_b_rev = optimumRandom(not isMale)
#goodArr = makeGoodAttr(False)
#att_b_ = convertToAttGanAttr(goodArr)
#att_b_rev = att_b_ * -1   

results = []
ioulist = []



pixelate_x = 30

#saveAttrToFile(srcpath,srcImg, attValues)
#q = binaryAttArray()

#print(att_b_)
#print(att_b_rev)
batchno=20
for batch in range(10):
    att_b_ = optimumRandom(isMale)
    att_b_rev = att_b_ * -1  
        
    files = getImageFilenamesInFolder(basePath, srcpath)
    print('File count: ' + str(len(files)))
    
    
    pixelate_x = 29
    dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/prepixelate/6011/' + str(pixelate_x) 	
   # if batchno == 1:
   #     os.mkdir(dstPath)        
    dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/prepixelate/6011/' + str(pixelate_x) + '/a' + str(batchno)	
    os.mkdir(dstPath)        
    i = 0
    for file in files:
        print(file)
        faceAnonProcess(basePath, srcpath, file, dstPath, False, i)    
        i += 1    
        
    #Pixelate 25   
    pixelate_x = 25
    dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/prepixelate/6011/' + str(pixelate_x) 	
    if batchno == 1:
        os.mkdir(dstPath)        
    dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/prepixelate/6011/' + str(pixelate_x) + '/a' + str(batchno)	
    os.mkdir(dstPath)        
    i = 0
    for file in files:
        print(file)
        faceAnonProcess(basePath, srcpath, file, dstPath, False, i)    
        i += 1    
        
    #Pixelate 20  
    pixelate_x = 20
    dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/prepixelate/6011/' + str(pixelate_x) 	
    if batchno == 1:
        os.mkdir(dstPath)        
    dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/prepixelate/6011/' + str(pixelate_x) + '/a' + str(batchno)	
    os.mkdir(dstPath)        
    i = 0
    for file in files:
        print(file)
        faceAnonProcess(basePath, srcpath, file, dstPath, False, i)    
        i += 1  
        
    pixelate_x = 16
    dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/prepixelate/6011/' + str(pixelate_x) 	
    if batchno == 1:
        os.mkdir(dstPath)        
    dstPath = 'D:/AI/Results/AttGAN/Anonymized/CelebA/prepixelate/6011/' + str(pixelate_x) + '/a' + str(batchno)	
    os.mkdir(dstPath)        
    i = 0
    for file in files:
        print(file)
        faceAnonProcess(basePath, srcpath, file, dstPath, False, i)    
        i += 1  
        
    batchno += 1
    
#saveDataToFile(srcpath, results)

#print(att_b_)
#print(att_b_rev)




'''
#srcpath = 'D:/AI/Trainingdata/cartoonset10k'
#srcImg = '0375_01.jpg' #Dalai Lama tatoo with extra face
#srcImg = '0003_01.jpg'
#faceAnonProcess(basePath, srcpath, srcImg, '', True, 1)    
#facedetect(srcpath, '0375_01.jpg' )
#srcpath = 'D:/AI/Results/AttGAN/Anonymized/vggface2/'
#folder = 'n000009_016_predict2_attr'
#folder = 'tmp'
#dstPath = 'D:/AI/Results/AttGAN/Anonymized/vggface2/n000009_016_predict2_attr_cropped'
files = getImageFilenamesInFolder(basePath, srcpath)
for file in files:
   print(file)
   facedetectSave(dstPath, file, dstPath)
   break
'''


'''
tot = 0
for i in range(1000):    
    goodArr,k = makeGoodAttr(False)
    print(k)
    tot += k
mean = tot / 1000
print(mean)
'''    
    
'''    

srcImg = '0016_01.jpg'


for i in range(1000):
    
    #att_b_ = attribFromFile(i)    

    #att_b_ = optimumRandom(isMale)
    goodArr = makeGoodAttr(False)
    att_b_ = convertToAttGanAttr(goodArr)
    att_b_rev = att_b_ * -1   
    attValues.append(att_b_)
    faceAnonProcess(basePath, srcpath, srcImg, dstPath, False, i)    

saveDataToFile(srcpath, results)   
saveAttrToFile(srcpath, srcImg, attValues)
'''    

               
#os.system("ffmpeg -f image2 -r 30 -i D:/AI/Results/AttGAN/Anonymized/video/climatechange/*.jpg -vcodec mpeg4 -y c:/Temp/climatechante_demo_anon.mp4")
