# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:39:14 2020

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
from boundingboxutils import expandBox, inflate


from attrgenerator import makeGoodAttr, convertToAttGanAttr
device = torch.device('cpu')

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def collate_fn(x):
    return x[0]

imgfolder = '5941'
imgfolder = 'CelebA_female'
iteration = 1
isMale = False

#print(int(178/12),int(218/12))
 

#Attributes:
#'Bald','Bangs','Black_Hair','Blond_Hair','Brown_Hair','Bushy_Eyebrows','Eyeglasses','Male','Mouth_Slightly_Open','Mustache','No_Beard','Pale_Skin','Young'
def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', default='128_shortcut1_inject0_none')
    
    parser.add_argument('--test_atts', dest='test_atts', nargs='+', default=['Young'])
    parser.add_argument('--test_ints', dest='test_ints', type=float, nargs='+', help='test_ints', default=[-1.0] )
    
    #parser.add_argument('--test_atts', dest='test_atts', nargs='+', default=['Bushy_Eyebrows', 'Black_Hair', "Mouth_Slightly_Open", 'Young', 'Male'])
    #parser.add_argument('--test_ints', dest='test_ints', type=float, nargs='+', help='test_ints', default=[1.0, 1.0, 1.0, 1.0, 1.0] )
        
    #parser.add_argument('--test_atts', dest='test_atts', nargs='+', default=['Bangs', 'Blond_Hair', 'Male', 'Pale_Skin', 'Young'])
    #parser.add_argument('--test_ints', dest='test_ints', type=float, nargs='+', help='test_ints', default=[0.8, 0.8, 0.8, 0.8, 0.8] )
    
    parser.add_argument('--num_test', dest='num_test', type=int, default='30')
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    
    
    #parser.add_argument('--custom_img', action='store_true')
    #parser.add_argument('--custom_data', type=str, default='./data/testset3k')
    #parser.add_argument('--custom_attr', type=str, default='./data/list_attr_celeba_testset3k.txt')
    
    #parser.add_argument('--custom_img', action='store_true', default='True')
    #parser.add_argument('--custom_data', type=str, default='./data/wider_face')
    #parser.add_argument('--custom_attr', type=str, default='./data/list_attr_wider_face.txt')
  
  # Denne er det vanlige...  
    parser.add_argument('--custom_img', action='store_true', default='True')
    parser.add_argument('--custom_data', type=str, default='./data/' + imgfolder)
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_celeb_grouped_' + imgfolder + '.txt')
    
  #  parser.add_argument('--custom_img', action='store_true', default='True')
  #  parser.add_argument('--custom_data', type=str, default='./data/custom')
  #  parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')
    
    
    #parser.add_argument('--custom_img', action='store_true')
    #parser.add_argument('--custom_data', type=str, default='./data/young')
    #parser.add_argument('--custom_attr', type=str, default='./data/list_attr_celeba_young.txt')
    
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


def customAttArray(index):
    if index > 12:
        index = 12
    
    att_a =  np.full((1, 13), 0)
    #att_a[:, index] = 0
    att_a[:, 0] = 0 #not bald
    att_a[:, 3] = 0 #not blond hair
    att_a[:, 4] = 0 #not brown hair
    att_a[:, 6] = 0 #glasses
    att_a[:, 7] = 1 #Gender 
    att_a[:, 8] = 0 #mouth opens a bit
    
    att_a = torch.tensor(att_a)
    att_a = att_a.type(torch.float)
    att_b = att_a.clone()
    return att_b


def singleAttArray(index, value):
    if index > 12:
        index = 12
    
    att_a =  np.full((1, 13), 0)
    att_a[:, index] = value
   
    att_a = torch.tensor(att_a)
    att_a = att_a.type(torch.float)
    att_b = att_a.clone()
    return att_b


def allRandom():
    att_a = np.full((1, 13), 0)
    att_a = torch.tensor(att_a)
    att_a = att_a.type(torch.float)
    for i in range(5, 13, 1):
        val = random.uniform(-1.0, 1.0)
        att_a[:, i] = val
    att_b = att_a.clone()
    return att_b


def optimumRandom(isMale):
    att_a = np.full((1, 13), 0)
    att_a = torch.tensor(att_a)
    att_a = att_a.type(torch.float)
    #bangs
    att_a[:, 1] = random.uniform(-1.1, 1.1)
    for i in range(5, 13, 1):
        #i=12: young
        val = random.uniform(-1.0, 1.1)
        if i == 7: #male
            if isMale==True:
                val = random.uniform(0, -1.1)
            else:
                val = random.uniform(0, 1.1)
        if i== 9: #Mustache
            if isMale==True:
                val = random.uniform(0, -1.1)
            else:
                val = random.uniform(0, 1.1)
        if i== 10: #no beard
            if isMale==True:
                val = random.uniform(0, 1.1)
            else:
                val = random.uniform(0, -1.1)
        if i == 11: #pale
            val = random.uniform(-0.8, 0.8)
        att_a[:, i] = val
       
            
    att_b = att_a.clone()
    return att_b


def spesificParams():
    att_a = np.full((1, 13), 0)
    att_a = torch.tensor(att_a)
    att_a = att_a.type(torch.float)
    att_a[:, 0] = 0
    att_a[:, 1] = 0.27609786
    att_a[:, 2] = 0
    att_a[:, 3] = 0
    att_a[:, 4] = 0
    att_a[:, 5] = -0.5671934
    att_a[:, 6] = 0.26872316
    att_a[:, 7] = -1 #-0.2952891
    att_a[:, 8] = -0.04619908
    att_a[:, 9] = -0.56060475
    att_a[:, 10] = 0.94840837
    att_a[:, 11] = 0.23503868
    att_a[:, 12] = -1
    
    att_b = att_a.clone()
    return att_b



def reversAttributes(att_):
    att_c_ = (att_ * 2 - 1) * args.thres_int
    for a, i in zip(args.test_atts, [-1.0]):
        att_c_[..., args.attrs.index(a)] = att_c_[..., args.attrs.index(a)] * i / args.thres_int
    return att_c_


def applyAttGan(image, attributes):
    transformed = [attgan.G(image, attributes)]
    transformed = torch.cat(transformed, dim=3)
    return transformed


def applyAttGanWithReverse(image, attributes, reverseAtts):
    transformed = applyAttGan(image, attributes)
    #reverseAtts = attributes * -1    
    return applyAttGan(transformed, reverseAtts)


#  Converts normalized tensor image to regular image
#  https://discuss.pytorch.org/t/conversion-from-a-tensor-to-a-pil-image-not-working-well-what-is-going-wrong/26121/2
#  mean and std to 0.5, as in dataloader, giving a range [-1, 1]
def convertTensorToImage(tr_im):
    z = tr_im * torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    z = z + torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    faceImg = transforms.ToPILImage(mode='RGB')(z)
    return faceImg


# alignedImages: array of aligned images, as torch tensors
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


def saveNameProbsToFile():
    with open('distance2_.csv', mode='w', newline='') as result_file:
        result_writer = csv.writer(result_file, delimiter=';',  quoting=csv.QUOTE_MINIMAL)
        for i in range(len(names)):
            result_writer.writerow([names[i], iterations[i], distances[i]])


def saveAttsDistanceToFile(path):
    file_name = join(path, 'dist_att.csv')
    with open(file_name, mode='w', newline='') as result_file:
        # result_writer = csv.writer(result_file, delimiter=';',  quoting=csv.QUOTE_MINIMAL)
        for i in range(len(attValues)):
            d = attValues[i][0].numpy()
            line = names[i]
            line += ';'
            line += str(distanceDelta[i])
            line += ';'
            for k in d:
                line += str(k)
                line += ';'
            #print(line)
            result_file.write(line +  os.linesep)

# Saving the parameters for attributes first, 
# then iterating over the image names and writing out distance
def saveAttsDistanceToFile2(path):
    file_name = join(path, 'dist_att.csv')
    with open(file_name, mode='w', newline='') as result_file:
        # result_writer = csv.writer(result_file, delimiter=';',  quoting=csv.QUOTE_MINIMAL)
        for i in range(len(attValues)):
            d = attValues[i][0].numpy()
            line = ''
            for k in d:
                line += str(k)
                line += ';'
            result_file.write(line +  os.linesep)
                         
        for i in range(len(names)):            
            line = names[i]
            line += ';'
            line += str(distances[i])            
            result_file.write(line +  os.linesep)
            

args = createArgs()

#output_path = join('output', args.experiment_name, 'anongroup_reshape_10_13_' + imgfolder + '_' + str(iteration))
output_path = join('output', args.experiment_name, 'testanonymizer_' + imgfolder + '_' + str(iteration))

names = []
distances = []
iterations = []

attValues = []
distanceDelta = []

#output_path = join('output', args.experiment_name, 'sample_testing_single_tmp' + str(args.test_atts))
#output_path = join('output', args.experiment_name, 'attrib_revert_customdata_wider_handshake_2' + str(args.test_atts))
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


attgan = AttGAN(args)
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
progressbar = Progressbar()

att_b_ = optimumRandom(isMale)
att_b_reverse = optimumRandom(not isMale)

att_b_ = allRandom()
att_b_reverse = att_b_ * -1    

#att_b_ = spesificParams()
#att_b_ = singleAttArray(12, -1)


goodArr = makeGoodAttr(isMale)
#att_b_ = convertToAttGanAttr(goodArr)
#att_b_reverse = att_b_ * -1 


attValues.append(att_b_)
attValues.append(att_b_reverse)


print(att_b_[0])

attgan.eval()
for idx, (img_a, att_a) in enumerate(test_dataloader):
    if args.num_test is not None and idx == args.num_test:
        break
    
    img_a = img_a.cuda() if args.gpu else img_a
    att_a = att_a.cuda() if args.gpu else att_a
    att_a = att_a.type(torch.float)
    att_b = att_a.clone()
    currentfile = '{:06d}.jpg'.format(idx + 182638)
    filename = '{:06d}'.format(idx + 0)
    #print(currentfile)
    origImg = test_dataset.getImage(idx)
    #origImg.show()

    with torch.no_grad():
        samples = [img_a]
        
        orig = [img_a]
        orig = torch.cat(orig, dim=3)
          
        faceImg = convertTensorToImage(img_a[0])        
        
        #faceImg.save(join('c:/temp', filename+'_0.jpg'))
        #Show original
        faceImg.show()

        boxesAligned, probs, landmarks = mtcnn.detect(faceImg, landmarks=True)
        if boxesAligned is None:
            print('No face boxesAligned')
            continue
        boxAligned = boxesAligned[0]
        
        
        alignedImgBox = faceImg.crop(boxAligned)
        #print(alignedImgBox)
        alignedImgBox.show()
        
        transformed = img_a.clone()
        q = transformed.data.numpy()
        #picks one color
        #qq = q[0][2]
        #transformed = torch.zeros(1, 3, 128, 128)
        #transformed.data[0][0] = torch.from_numpy(qq)        
        transformed = applyAttGanWithReverse(transformed, att_b_, att_b_reverse)                 
        
        finalImg = convertTensorToImage(transformed[0, :])
        #finalImg.save(join('c:/temp', filename+'_b.jpg'))        
        #finalImg.show()
        '''
        boxes, probs, landmarks = mtcnn.detect(finalImg, landmarks=True)
        if boxes is None:
            print('No face')
            continue
        
        b0 = boxes[0]        
        '''
        
        boxesOrig, probs, landmarks = mtcnn.detect(origImg, landmarks=True)
        boxOrig = boxesOrig[0]    
        #print(boxOrig)
        #boxOrig = inflate(origImg.size, boxOrig, 1.2)
        print(boxOrig)
        print(boxAligned)        
        
              
        rw = int(boxOrig[2] - boxOrig[0])
        rh = int(boxOrig[3] - boxOrig[1])
        anonymized = finalImg.crop(boxAligned)
        anonymized = anonymized.resize((rw, rh))
        
        
        file_name = join(output_path, currentfile)                
        #expandedBox = expandBox(finalImg.size, b0, landmarks, 1.2)
                
        
        finalImg.show()
        #subImg = finalImg.crop(boxAligned)
        
        origImg.paste(anonymized, (int(boxOrig[0]), int(boxOrig[1])))
        origImg.save(file_name)        
        #origImg.save(join('c:/temp', filename+'_c.jpg'))        
        origImg.show()
    break
        
        
#saveNameProbsToFile()        
# use number2 to write first the attributes, then the list of filenames and their distance
#saveAttsDistanceToFile2(output_path)

# after applyAttGanWithReverse        
'''
        x_aligned, prob = mtcnn(faceImg, return_prob=True) 
        if x_aligned is None:
            print('No face')
            continue
  
        distance = 0
        n=0 #number of iterations
        transformed = img_a.clone()
        prevDistance = 0
        transformed = applyAttGanWithReverse(transformed, att_b_, att_b_reverse)        
        
        #Measure distance
        transfImage = convertTensorToImage(transformed[0,:])            
        tr_aligned = mtcnn(transfImage, return_prob=False)
        if tr_aligned is not None:
            aligned = createArray(x_aligned, tr_aligned) 
            distance = normDistance(aligned)
            names.append(currentfile)
            distances.append(distance)
        else:
            print(currentfile, ': MTCNN returned null')
            
'''         