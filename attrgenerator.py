# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:44:28 2020

@author: 103920eili
"""
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
import torch


# scikit-learn MLPClassifier classifier
clf = pickle.load(open('female_model2.sav', 'rb'))


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
